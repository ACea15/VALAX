"""Tests for day count conventions and date utilities."""

import jax.numpy as jnp
import pytest

from valax.dates.daycounts import (
    DAY_COUNT_FNS,
    DAY_COUNT_INVERSE_FNS,
    act_360,
    act_365,
    act_act,
    thirty_360,
    ordinal_from_year_fraction,
    year_fraction,
    ymd_to_ordinal,
)


class TestYmdToOrdinal:
    def test_epoch(self):
        assert int(ymd_to_ordinal(1970, 1, 1)) == 0

    def test_known_date(self):
        # 2025-01-01 = ordinal 20089 (days since 1970-01-01)
        import datetime
        expected = (datetime.date(2025, 1, 1) - datetime.date(1970, 1, 1)).days
        assert int(ymd_to_ordinal(2025, 1, 1)) == expected

    def test_leap_year(self):
        import datetime
        d = datetime.date(2024, 2, 29)
        expected = (d - datetime.date(1970, 1, 1)).days
        assert int(ymd_to_ordinal(2024, 2, 29)) == expected

    def test_roundtrip_multiple_dates(self):
        import datetime
        test_dates = [
            (2000, 3, 15), (2010, 7, 4), (2023, 12, 31),
            (1999, 1, 1), (2030, 6, 30),
        ]
        for y, m, d in test_dates:
            expected = (datetime.date(y, m, d) - datetime.date(1970, 1, 1)).days
            assert int(ymd_to_ordinal(y, m, d)) == expected, f"Failed for {y}-{m}-{d}"


class TestAct365:
    def test_one_year(self):
        start = ymd_to_ordinal(2024, 1, 1)
        end = ymd_to_ordinal(2025, 1, 1)
        frac = act_365(start, end)
        # 2024 is a leap year: 366 days
        assert abs(float(frac) - 366.0 / 365.0) < 1e-10

    def test_half_year(self):
        start = ymd_to_ordinal(2025, 1, 1)
        end = ymd_to_ordinal(2025, 7, 2)
        frac = act_365(start, end)
        # 31+28+31+30+31+1 = 182 days
        assert abs(float(frac) - 182.0 / 365.0) < 1e-10


class TestAct360:
    def test_90_days(self):
        start = ymd_to_ordinal(2025, 1, 1)
        end = ymd_to_ordinal(2025, 4, 1)
        frac = act_360(start, end)
        assert abs(float(frac) - 90.0 / 360.0) < 1e-10


class TestActAct:
    """ACT/ACT ISDA: each day weighted by the length of *its own* year.

    Previously this convention was a flat ``/365.25``, which is wrong for
    every period — and was never tested, which is why it survived.
    """

    def test_isda_reference_example(self):
        """ISDA 2006 §4.16(b) worked example: 2003-11-01 → 2004-05-01.

        61 days fall in 2003 (365-day year) and 121 in 2004 (366-day),
        giving 61/365 + 121/366. A flat 365.25 denominator returns
        0.498289 — wrong in the fourth decimal place.
        """
        start = ymd_to_ordinal(2003, 11, 1)
        end = ymd_to_ordinal(2004, 5, 1)
        expected = 61.0 / 365.0 + 121.0 / 366.0
        assert float(act_act(start, end)) == pytest.approx(expected, abs=1e-12)
        assert float(act_act(start, end)) != pytest.approx(
            182.0 / 365.25, abs=1e-6
        )

    @pytest.mark.parametrize(
        "year,length", [(2019, 365), (2020, 366), (2100, 365), (2000, 366)]
    )
    def test_whole_calendar_year_is_exactly_one(self, year, length):
        """Every whole calendar year is 1.0, leap or not."""
        start = ymd_to_ordinal(year, 1, 1)
        end = ymd_to_ordinal(year + 1, 1, 1)
        assert int(end) - int(start) == length
        assert float(act_act(start, end)) == pytest.approx(1.0, abs=1e-12)

    def test_straddling_a_leap_boundary_exceeds_one(self):
        """Jul→Jul across a leap year is slightly more than a year."""
        start = ymd_to_ordinal(2019, 7, 1)
        end = ymd_to_ordinal(2020, 7, 1)
        expected = 184.0 / 365.0 + 182.0 / 366.0
        assert float(act_act(start, end)) == pytest.approx(expected, abs=1e-12)
        assert float(act_act(start, end)) > 1.0

    def test_additive_across_a_split(self):
        """Additivity over days is the property that makes it closed-form."""
        a = ymd_to_ordinal(2019, 3, 10)
        mid = ymd_to_ordinal(2020, 8, 21)
        b = ymd_to_ordinal(2022, 1, 5)
        assert float(act_act(a, b)) == pytest.approx(
            float(act_act(a, mid)) + float(act_act(mid, b)), abs=1e-12
        )


class TestThirty360:
    def test_standard(self):
        start = ymd_to_ordinal(2025, 1, 15)
        end = ymd_to_ordinal(2025, 7, 15)
        frac = thirty_360(start, end)
        # 30/360: 6 months = 180/360 = 0.5
        assert abs(float(frac) - 0.5) < 1e-10

    def test_full_year(self):
        start = ymd_to_ordinal(2025, 3, 1)
        end = ymd_to_ordinal(2026, 3, 1)
        frac = thirty_360(start, end)
        assert abs(float(frac) - 1.0) < 1e-10

    def test_both_endpoints_on_the_31st(self):
        """Both days truncate to 30, so this is exactly six 30-day months."""
        start = ymd_to_ordinal(2025, 1, 31)
        end = ymd_to_ordinal(2025, 7, 31)
        assert float(thirty_360(start, end)) == pytest.approx(0.5, abs=1e-12)

    def test_d1_below_30_leaves_d2_untruncated(self):
        """The ``d1 >= 30`` branch: 15 Jan → 31 Mar keeps d2 = 31."""
        start = ymd_to_ordinal(2025, 1, 15)
        end = ymd_to_ordinal(2025, 3, 31)
        expected = (30 * 2 + (31 - 15)) / 360.0
        assert float(thirty_360(start, end)) == pytest.approx(
            expected, abs=1e-12
        )

    def test_february_end_is_not_treated_as_month_end(self):
        """US bond basis does not roll 28/29 Feb up to 30."""
        start = ymd_to_ordinal(2025, 2, 28)
        end = ymd_to_ordinal(2025, 8, 28)
        assert float(thirty_360(start, end)) == pytest.approx(0.5, abs=1e-12)
        leap = ymd_to_ordinal(2024, 2, 29)
        leap_end = ymd_to_ordinal(2024, 8, 29)
        assert float(thirty_360(leap, leap_end)) == pytest.approx(
            0.5, abs=1e-12
        )

    def test_zero_and_negative_intervals(self):
        a = ymd_to_ordinal(2025, 5, 10)
        b = ymd_to_ordinal(2025, 11, 10)
        assert float(thirty_360(a, a)) == pytest.approx(0.0, abs=1e-12)
        assert float(thirty_360(b, a)) == pytest.approx(
            -float(thirty_360(a, b)), abs=1e-12
        )


class TestCrossConventionSpread:
    """Measure the disagreement between conventions instead of avoiding it.

    Every QuantLib comparison in this repo pins both sides to Act/365 and
    every bond test used a single convention, so the cost of picking the
    wrong day count was never observed. These tests assert *bounds on the
    measured error*, which is the honest form of the claim: conventions
    genuinely differ, by a known and bounded amount.
    """

    @pytest.mark.parametrize("convention", sorted(DAY_COUNT_FNS))
    def test_a_year_is_within_two_percent_of_one(self, convention):
        start = ymd_to_ordinal(2025, 6, 15)
        end = ymd_to_ordinal(2026, 6, 15)
        assert float(year_fraction(start, end, convention)) == pytest.approx(
            1.0, abs=0.02
        )

    def test_act_360_exceeds_act_365_by_the_expected_ratio(self):
        start = ymd_to_ordinal(2025, 1, 1)
        end = ymd_to_ordinal(2026, 1, 1)
        ratio = float(year_fraction(start, end, "act_360")) / float(
            year_fraction(start, end, "act_365")
        )
        assert ratio == pytest.approx(365.0 / 360.0, abs=1e-12)

    def test_semiannual_coupon_error_from_the_wrong_convention(self):
        """A 5% semiannual coupon mis-sized by choosing the wrong basis.

        Act/360 over a 181-day period pays 2.5139 per 100 of face while
        Act/365 pays 2.4795 — 3.4 cents per coupon, ~1.4% of the coupon.
        Small per cashflow, systematic across every cashflow, which is how
        it reaches basis points of yield.
        """
        start = ymd_to_ordinal(2025, 1, 31)
        end = ymd_to_ordinal(2025, 7, 31)
        c360 = 100.0 * 0.05 * float(year_fraction(start, end, "act_360"))
        c365 = 100.0 * 0.05 * float(year_fraction(start, end, "act_365"))
        assert c360 > c365
        assert (c360 - c365) == pytest.approx(0.0344, abs=5e-4)
        assert (c360 - c365) / c365 == pytest.approx(0.0139, abs=1e-3)

    def test_thirty_360_equalises_periods_that_actuals_do_not(self):
        """The structural difference: 30/360 makes both half-years 0.5."""
        h1 = (ymd_to_ordinal(2025, 1, 31), ymd_to_ordinal(2025, 7, 31))
        h2 = (ymd_to_ordinal(2025, 7, 31), ymd_to_ordinal(2026, 1, 31))
        assert float(thirty_360(*h1)) == pytest.approx(
            float(thirty_360(*h2)), abs=1e-12
        )
        assert float(act_365(*h1)) != pytest.approx(
            float(act_365(*h2)), abs=1e-6
        )
        # 181 vs 184 days: the actual/N alternation, ~0.8% of a period.
        drift = abs(float(act_365(*h1)) - float(act_365(*h2)))
        assert drift == pytest.approx(3.0 / 365.0, abs=1e-12)


class TestYearFraction:
    def test_dispatch(self):
        start = ymd_to_ordinal(2025, 1, 1)
        end = ymd_to_ordinal(2025, 7, 1)
        f365 = year_fraction(start, end, "act_365")
        f360 = year_fraction(start, end, "act_360")
        # act_360 should give a larger fraction for the same period
        assert float(f360) > float(f365)


class TestOrdinalFromYearFraction:
    """The inverse: year fraction → (fractional) ordinal.

    The bridge between an equity instrument's year-fraction ``expiry`` and
    a curve's date-ordinal index. Returns a *fractional* ordinal so the
    round trip is exact and the path stays differentiable.
    """

    @pytest.mark.parametrize("convention", sorted(DAY_COUNT_INVERSE_FNS))
    @pytest.mark.parametrize("years", [0.25, 0.5, 1.0, 2.0, 7.5])
    def test_round_trip_is_exact(self, convention, years):
        start = ymd_to_ordinal(2026, 1, 1)
        target = jnp.asarray(years)
        ordinal = ordinal_from_year_fraction(start, target, convention)
        assert float(year_fraction(start, ordinal, convention)) == pytest.approx(
            years, abs=1e-12,
        )

    def test_uses_the_conventions_denominator(self):
        """Not a hard-coded 365: act_360 lands 360 days out, act_365 365."""
        start = ymd_to_ordinal(2026, 1, 1)
        one_year = jnp.asarray(1.0)
        assert float(
            ordinal_from_year_fraction(start, one_year, "act_360") - start
        ) == pytest.approx(360.0)
        assert float(
            ordinal_from_year_fraction(start, one_year, "act_365") - start
        ) == pytest.approx(365.0)

    @pytest.mark.parametrize(
        "year,length", [(2026, 365), (2028, 366)]
    )
    def test_act_act_inverse_lands_on_the_calendar_year(self, year, length):
        """One ACT/ACT year from 1 Jan is that year's actual length.

        The old flat-365.25 inverse returned a fractional 365.25 days for
        every year, which could not round-trip against a correct forward
        calculation.
        """
        start = ymd_to_ordinal(year, 1, 1)
        ordinal = ordinal_from_year_fraction(start, jnp.asarray(1.0), "act_act")
        assert float(ordinal - start) == pytest.approx(float(length), abs=1e-9)

    def test_is_differentiable_in_the_year_fraction(self):
        """A rate projected at this horizon must keep its gradient."""
        import jax

        start = ymd_to_ordinal(2026, 1, 1)
        g = jax.grad(
            lambda t: ordinal_from_year_fraction(start, t, "act_365")
        )(jnp.asarray(1.5))
        assert float(g) == pytest.approx(365.0)

    def test_thirty_360_has_no_closed_form_inverse(self):
        start = ymd_to_ordinal(2026, 1, 1)
        with pytest.raises(ValueError, match="no closed-form inverse"):
            ordinal_from_year_fraction(start, jnp.asarray(1.0), "30_360")

    def test_unknown_convention_raises(self):
        start = ymd_to_ordinal(2026, 1, 1)
        with pytest.raises(ValueError, match="Unknown day count"):
            ordinal_from_year_fraction(start, jnp.asarray(1.0), "act_366")
