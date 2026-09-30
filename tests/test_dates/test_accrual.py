"""Tests for accrual periods, coupon sizing, and clean/dirty prices.

Guards the defects these functions were written to fix:

* every fixed-coupon pricer paid ``coupon_rate / frequency`` per period,
  ignoring both the period's actual length and the instrument's own
  ``day_count``;
* there was no accrued-interest function anywhere, so every price was dirty
  with no way to obtain the quoted clean price;
* the analytic and Monte Carlo FRN pricers disagreed about what the accrual
  period even was.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from valax.dates.accrual import (
    accrual_periods,
    accrued_interest,
    accrued_interest_by_time,
    bond_accrued_interest,
    clean_from_dirty,
    dirty_from_clean,
    fixed_coupon_amounts,
    floating_accrual_periods,
)
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.dates.schedule import generate_schedule
from valax.instruments.bonds import FixedRateBond, FloatingRateBond


def make_bond(
    day_count: str = "act_365",
    settlement: tuple[int, int, int] = (2025, 1, 31),
    explicit_starts: bool = True,
    coupon: float = 0.05,
    frequency: int = 2,
) -> FixedRateBond:
    starts, _, pays = generate_schedule(
        2025, 1, 31, 2030, 1, 31, frequency=frequency
    )
    return FixedRateBond(
        payment_dates=pays,
        accrual_start_dates=starts if explicit_starts else None,
        settlement_date=ymd_to_ordinal(*settlement),
        coupon_rate=jnp.asarray(coupon),
        face_value=jnp.asarray(100.0),
        frequency=frequency,
        day_count=day_count,
    )


class TestAccrualPeriods:
    def test_explicit_starts_are_authoritative(self):
        starts, _, pays = generate_schedule(2025, 4, 20, 2026, 10, 15, frequency=4)
        got_starts, got_ends = accrual_periods(pays, starts)
        assert jnp.all(got_starts == starts)
        assert jnp.all(got_ends == pays)

    def test_derived_starts_are_the_previous_payment_dates(self):
        _, _, pays = generate_schedule(2025, 1, 31, 2030, 1, 31)
        starts, ends = accrual_periods(pays, None)
        assert jnp.all(starts[1:] == pays[:-1])
        assert jnp.all(ends == pays)

    def test_derived_first_period_precedes_the_first_payment(self):
        """A bond pays its whole next coupon however late you buy it.

        Anchoring the first period at settlement instead would shrink that
        coupon to the remaining stub and under-price every seasoned bond.
        """
        _, _, pays = generate_schedule(2025, 1, 31, 2030, 1, 31)
        starts, _ = accrual_periods(pays, None)
        assert int(starts[0]) < int(pays[0])
        assert int(pays[0]) - int(starts[0]) == pytest.approx(184, abs=4)

    def test_a_stub_is_not_recoverable_without_explicit_starts(self):
        """Why the field exists: extrapolation cannot see a stub."""
        starts, _, pays = generate_schedule(2025, 4, 20, 2026, 10, 15, frequency=4)
        derived, _ = accrual_periods(pays, None)
        assert int(derived[0]) != int(starts[0])

    def test_single_payment_schedule_uses_the_frequency(self):
        pays = jnp.asarray([ymd_to_ordinal(2026, 1, 15)])
        starts, _ = accrual_periods(pays, None, frequency=2)
        assert int(pays[0]) - int(starts[0]) == 182


class TestFixedCouponAmounts:
    def test_coupons_are_unequal_under_an_actual_convention(self):
        """The defect: ``c/f`` paid the same amount for 181 and 184 days."""
        coupons = fixed_coupon_amounts(make_bond("act_365"))
        assert len(set(float(c) for c in coupons)) > 1
        # 181-day and 184-day periods alternate.
        assert float(coupons[0]) == pytest.approx(100 * 0.05 * 181 / 365, abs=1e-12)
        assert float(coupons[1]) == pytest.approx(100 * 0.05 * 184 / 365, abs=1e-12)

    def test_coupons_are_equal_under_thirty_360(self):
        """30/360 is the convention under which ``c/f`` was ever correct."""
        coupons = fixed_coupon_amounts(make_bond("30_360"))
        assert float(jnp.max(coupons) - jnp.min(coupons)) == pytest.approx(
            0.0, abs=1e-12
        )
        assert float(coupons[0]) == pytest.approx(2.5, abs=1e-12)

    def test_five_years_of_coupons_sum_to_the_stated_rate(self):
        """Act/365 over exactly 5 calendar years of a 5% bond pays ~25."""
        total = float(jnp.sum(fixed_coupon_amounts(make_bond("act_365"))))
        assert total == pytest.approx(25.0, abs=0.05)

    def test_act_360_pays_more_than_act_365(self):
        c360 = float(jnp.sum(fixed_coupon_amounts(make_bond("act_360"))))
        c365 = float(jnp.sum(fixed_coupon_amounts(make_bond("act_365"))))
        assert c360 > c365
        assert c360 / c365 == pytest.approx(365.0 / 360.0, abs=1e-9)

    def test_day_count_is_actually_honoured(self):
        """``day_count`` used to be stored and ignored for coupon sizing.

        The three actual/N conventions must all disagree. ACT/ACT and 30/360
        genuinely coincide here — over five *whole* calendar years both are
        exactly 5.0, ACT/ACT because each year contributes 1.0 by definition
        and 30/360 because each is ten 30-day half-years. That agreement is a
        property worth pinning, not a bug.
        """
        amounts = {
            dc: float(jnp.sum(fixed_coupon_amounts(make_bond(dc))))
            for dc in ("act_360", "act_365", "act_act", "30_360")
        }
        assert amounts["act_360"] != pytest.approx(amounts["act_365"], abs=1e-9)
        assert amounts["act_365"] != pytest.approx(amounts["act_act"], abs=1e-9)
        assert amounts["act_360"] != pytest.approx(amounts["act_act"], abs=1e-9)
        assert amounts["act_act"] == pytest.approx(25.0, abs=1e-12)
        assert amounts["30_360"] == pytest.approx(25.0, abs=1e-12)

    def test_act_act_and_thirty_360_diverge_on_a_stubbed_schedule(self):
        """The coincidence above is specific to whole calendar years."""
        starts, _, pays = generate_schedule(2025, 4, 20, 2026, 10, 15, frequency=4)
        bonds = {
            dc: FixedRateBond(
                payment_dates=pays,
                accrual_start_dates=starts,
                settlement_date=ymd_to_ordinal(2025, 4, 20),
                coupon_rate=jnp.asarray(0.05),
                face_value=jnp.asarray(100.0),
                frequency=4,
                day_count=dc,
            )
            for dc in ("act_act", "30_360")
        }
        totals = {
            dc: float(jnp.sum(fixed_coupon_amounts(b))) for dc, b in bonds.items()
        }
        assert totals["act_act"] != pytest.approx(totals["30_360"], abs=1e-6)

    def test_differentiable_in_coupon_rate(self):
        bond = make_bond()
        g = eqx.filter_grad(
            lambda r: jnp.sum(
                fixed_coupon_amounts(
                    eqx.tree_at(lambda b: b.coupon_rate, bond, r)
                )
            )
        )(jnp.asarray(0.05))
        assert float(g) > 0.0

    def test_jit_safe(self):
        assert float(
            jnp.sum(eqx.filter_jit(fixed_coupon_amounts)(make_bond()))
        ) == pytest.approx(
            float(jnp.sum(fixed_coupon_amounts(make_bond()))), abs=1e-12
        )


class TestAccruedInterest:
    def test_zero_on_every_coupon_date(self):
        """A coupon date ends one period and starts the next."""
        bond = make_bond()
        for date in bond.payment_dates:
            assert float(bond_accrued_interest(bond, date)) == pytest.approx(
                0.0, abs=1e-12
            )

    def test_zero_at_the_dated_date(self):
        bond = make_bond()
        assert float(
            bond_accrued_interest(bond, bond.accrual_start_dates[0])
        ) == pytest.approx(0.0, abs=1e-12)

    def test_positive_mid_period(self):
        bond = make_bond()
        assert float(bond_accrued_interest(bond, ymd_to_ordinal(2025, 4, 30))) > 0.0

    def test_matches_the_elapsed_day_count_fraction(self):
        bond = make_bond("act_365")
        settle = ymd_to_ordinal(2025, 4, 30)
        expected = 100.0 * 0.05 * float(
            year_fraction(ymd_to_ordinal(2025, 1, 31), settle, "act_365")
        )
        assert float(bond_accrued_interest(bond, settle)) == pytest.approx(
            expected, abs=1e-12
        )

    def test_never_exceeds_the_coupon_it_belongs_to(self):
        bond = make_bond()
        coupons = fixed_coupon_amounts(bond)
        largest = float(jnp.max(coupons))
        for day in range(1, 181, 7):
            settle = int(bond.accrual_start_dates[0]) + day
            assert 0.0 <= float(bond_accrued_interest(bond, jnp.asarray(settle))) <= largest

    def test_grows_monotonically_within_a_period(self):
        bond = make_bond()
        base = int(bond.accrual_start_dates[0])
        series = [
            float(bond_accrued_interest(bond, jnp.asarray(base + d)))
            for d in range(0, 175, 15)
        ]
        assert all(b > a for a, b in zip(series, series[1:]))

    def test_resets_across_a_coupon_date(self):
        bond = make_bond()
        coupon_date = int(bond.payment_dates[0])
        before = float(bond_accrued_interest(bond, jnp.asarray(coupon_date - 1)))
        after = float(bond_accrued_interest(bond, jnp.asarray(coupon_date + 1)))
        assert before > after

    def test_zero_outside_the_instruments_life(self):
        bond = make_bond()
        assert float(bond_accrued_interest(bond, ymd_to_ordinal(2010, 6, 1))) == 0.0
        assert float(bond_accrued_interest(bond, ymd_to_ordinal(2040, 6, 1))) == 0.0

    def test_honours_the_day_count(self):
        settle = ymd_to_ordinal(2025, 4, 30)
        ai360 = float(bond_accrued_interest(make_bond("act_360"), settle))
        ai365 = float(bond_accrued_interest(make_bond("act_365"), settle))
        assert ai360 > ai365

    def test_jit_and_grad_safe(self):
        bond = make_bond(settlement=(2025, 4, 30))
        assert float(eqx.filter_jit(bond_accrued_interest)(bond)) == pytest.approx(
            float(bond_accrued_interest(bond)), abs=1e-12
        )
        g = eqx.filter_grad(
            lambda r: bond_accrued_interest(
                eqx.tree_at(lambda b: b.coupon_rate, bond, r)
            )
        )(jnp.asarray(0.05))
        assert float(g) > 0.0

    def test_standalone_function_matches_the_bond_wrapper(self):
        bond = make_bond(settlement=(2025, 4, 30))
        assert float(
            accrued_interest(
                settlement_date=bond.settlement_date,
                payment_dates=bond.payment_dates,
                coupon_rate=bond.coupon_rate,
                face_value=bond.face_value,
                accrual_start_dates=bond.accrual_start_dates,
                day_count=bond.day_count,
            )
        ) == pytest.approx(float(bond_accrued_interest(bond)), abs=1e-12)


class TestAccruedInterestByTime:
    """The year-fraction-space variant used by the lattice and PDE engines."""

    def test_agrees_with_the_ordinal_version(self):
        bond = make_bond("act_365")
        ref = bond.accrual_start_dates[0]
        starts, ends = accrual_periods(
            bond.payment_dates, bond.accrual_start_dates
        )
        offsets = jnp.asarray([0, 40, 90, 181, 200, 400])
        by_time = accrued_interest_by_time(
            year_fraction(ref, ref + offsets, "act_365"),
            year_fraction(ref, starts, "act_365"),
            year_fraction(ref, ends, "act_365"),
            bond.coupon_rate,
            bond.face_value,
        )
        for k, offset in enumerate(offsets):
            expected = float(bond_accrued_interest(bond, ref + offset))
            assert float(by_time[k]) == pytest.approx(expected, abs=1e-12)

    def test_zero_at_period_boundaries(self):
        starts = jnp.asarray([0.0, 0.5, 1.0])
        ends = jnp.asarray([0.5, 1.0, 1.5])
        got = accrued_interest_by_time(
            jnp.asarray([0.0, 0.5, 1.0, 1.5]),
            starts,
            ends,
            jnp.asarray(0.05),
            jnp.asarray(100.0),
        )
        assert jnp.allclose(got, 0.0)


class TestCleanDirty:
    def test_dirty_minus_clean_is_exactly_accrued(self):
        bond = make_bond(settlement=(2025, 4, 30))
        ai = bond_accrued_interest(bond)
        dirty = jnp.asarray(104.25)
        assert float(dirty - clean_from_dirty(dirty, ai)) == pytest.approx(
            float(ai), abs=1e-14
        )

    def test_round_trip_is_exact(self):
        bond = make_bond(settlement=(2025, 4, 30))
        ai = bond_accrued_interest(bond)
        dirty = jnp.asarray(104.25)
        assert float(dirty_from_clean(clean_from_dirty(dirty, ai), ai)) == (
            pytest.approx(float(dirty), abs=1e-14)
        )

    def test_clean_equals_dirty_only_on_a_coupon_date(self):
        on_coupon = make_bond(settlement=(2025, 7, 31))
        mid = make_bond(settlement=(2025, 4, 30))
        assert float(bond_accrued_interest(on_coupon)) == pytest.approx(0.0, abs=1e-12)
        assert float(bond_accrued_interest(mid)) > 0.5

    @settings(deadline=None, max_examples=60)
    @given(
        offset=st.integers(min_value=0, max_value=1800),
        coupon=st.floats(min_value=0.0, max_value=0.15),
    )
    def test_accrual_is_always_a_fraction_of_a_coupon(self, offset, coupon):
        bond = make_bond(coupon=coupon)
        settle = bond.accrual_start_dates[0] + offset
        ai = float(bond_accrued_interest(bond, settle))
        largest = float(jnp.max(fixed_coupon_amounts(bond)))
        assert -1e-12 <= ai <= largest + 1e-12


class TestFloatingAccrualConvention:
    """One accrual convention for every FRN engine.

    The analytic pricer accrued from ``fixing_dates`` while the Monte Carlo
    recipe accrued from the previous payment date (and the first period from
    settlement), so a note with a fixing lag priced differently depending on
    the engine.
    """

    def _frn(self, accrual_starts=None, fixing_lag: int = 0) -> FloatingRateBond:
        _, _, pays = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=1)
        period_starts = jnp.concatenate(
            [jnp.asarray([ymd_to_ordinal(2025, 1, 1)], dtype=pays.dtype), pays[:-1]]
        )
        return FloatingRateBond(
            payment_dates=pays,
            fixing_dates=period_starts - fixing_lag,
            settlement_date=ymd_to_ordinal(2025, 1, 1),
            spread=jnp.asarray(0.002),
            face_value=jnp.asarray(100.0),
            accrual_start_dates=accrual_starts,
            day_count="act_365",
        )

    def test_fixing_dates_stand_in_when_starts_are_absent(self):
        frn = self._frn()
        starts, ends = floating_accrual_periods(frn)
        assert jnp.all(starts == frn.fixing_dates)
        assert jnp.all(ends == frn.payment_dates)

    def test_explicit_starts_win_over_fixing_dates(self):
        _, _, pays = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=1)
        period_starts = jnp.concatenate(
            [jnp.asarray([ymd_to_ordinal(2025, 1, 1)], dtype=pays.dtype), pays[:-1]]
        )
        frn = self._frn(accrual_starts=period_starts, fixing_lag=2)
        starts, _ = floating_accrual_periods(frn)
        assert jnp.all(starts == period_starts)
        assert not jnp.all(starts == frn.fixing_dates)

    def test_a_fixing_lag_does_not_lengthen_the_accrual_period(self):
        """Observation and accrual are separate concepts.

        With explicit accrual starts a 2-day fixing lag leaves every accrual
        period exactly as long as it was; without them the lag would silently
        stretch each period by two days and over-pay every coupon.
        """
        _, _, pays = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=1)
        period_starts = jnp.concatenate(
            [jnp.asarray([ymd_to_ordinal(2025, 1, 1)], dtype=pays.dtype), pays[:-1]]
        )
        pinned = self._frn(accrual_starts=period_starts, fixing_lag=2)
        unpinned = self._frn(fixing_lag=2)
        pinned_tau = year_fraction(*floating_accrual_periods(pinned), "act_365")
        unpinned_tau = year_fraction(*floating_accrual_periods(unpinned), "act_365")
        assert jnp.all(unpinned_tau > pinned_tau)
        assert float(jnp.sum(unpinned_tau - pinned_tau)) == pytest.approx(
            5 * 2 / 365.0, abs=1e-12
        )
