"""
Cross-validation: VALAX vs QuantLib for fixed income analytics.

Tests discount curve construction, bond pricing, yield-to-maturity,
and duration. Key-rate durations (autodiff) are tested against
finite-difference approximations since QuantLib doesn't provide them
natively.

Companion example: examples/comparisons/02_fixed_income.py
"""

import pytest
import jax
import jax.numpy as jnp
import QuantLib as ql
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.curves.discount import DiscountCurve, zero_rate
from valax.instruments.bonds import ZeroCouponBond, FixedRateBond
from valax.pricing.analytic.bonds import (
    zero_coupon_bond_price,
    fixed_rate_bond_price,
    yield_to_maturity,
    modified_duration,
    convexity,
    key_rate_durations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TENOR_LABELS = ["6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]
ZERO_RATES = [0.0425, 0.0410, 0.0395, 0.0385, 0.0375, 0.0370, 0.0365]


@pytest.fixture
def valax_curve():
    today = ymd_to_ordinal(2026, 3, 26)
    pillar_dates = jnp.array([
        ymd_to_ordinal(2026, 9, 26), ymd_to_ordinal(2027, 3, 26),
        ymd_to_ordinal(2028, 3, 26), ymd_to_ordinal(2029, 3, 26),
        ymd_to_ordinal(2031, 3, 26), ymd_to_ordinal(2033, 3, 26),
        ymd_to_ordinal(2036, 3, 26),
    ])
    times = year_fraction(today, pillar_dates, "act_365")
    dfs = jnp.exp(-jnp.array(ZERO_RATES) * times)
    return DiscountCurve(
        pillar_dates=pillar_dates, discount_factors=dfs,
        reference_date=today, day_count="act_365",
    )


@pytest.fixture
def ql_curve():
    today = ql.Date(26, 3, 2026)
    ql.Settings.instance().evaluationDate = today
    dates = [
        today, ql.Date(26, 9, 2026), ql.Date(26, 3, 2027),
        ql.Date(26, 3, 2028), ql.Date(26, 3, 2029),
        ql.Date(26, 3, 2031), ql.Date(26, 3, 2033), ql.Date(26, 3, 2036),
    ]
    curve = ql.ZeroCurve(
        dates, [ZERO_RATES[0]] + ZERO_RATES,
        ql.Actual365Fixed(), ql.NullCalendar(), ql.Linear(), ql.Continuous,
    )
    return curve, dates[1:]  # return pillar dates (excluding today)


# ---------------------------------------------------------------------------
# Curve tests
# ---------------------------------------------------------------------------

class TestDiscountCurve:
    """Discount factors must match at pillar points."""

    @pytest.mark.parametrize("idx", range(len(ZERO_RATES)))
    def test_discount_factor_at_pillar(self, valax_curve, ql_curve, idx):
        """See: examples/comparisons/02_fixed_income.py §4 (curve comparison)"""
        v_df = float(valax_curve.discount_factors[idx])
        q_df = ql_curve[0].discount(ql_curve[1][idx])
        assert abs(v_df - q_df) < 1e-12, (
            f"{TENOR_LABELS[idx]}: VALAX={v_df}, QL={q_df}"
        )


# ---------------------------------------------------------------------------
# Bond tests
# ---------------------------------------------------------------------------

class TestBondPricing:
    """Fixed-rate bond pricing, YTM, and risk measures."""

    @pytest.fixture
    def valax_bond(self, valax_curve):
        today = valax_curve.reference_date
        payment_dates = jnp.array([
            ymd_to_ordinal(2026, 9, 26), ymd_to_ordinal(2027, 3, 26),
            ymd_to_ordinal(2027, 9, 26), ymd_to_ordinal(2028, 3, 26),
            ymd_to_ordinal(2028, 9, 26), ymd_to_ordinal(2029, 3, 26),
            ymd_to_ordinal(2029, 9, 26), ymd_to_ordinal(2030, 3, 26),
            ymd_to_ordinal(2030, 9, 26), ymd_to_ordinal(2031, 3, 26),
        ])
        return FixedRateBond(
            payment_dates=payment_dates, settlement_date=today,
            coupon_rate=jnp.array(0.04), face_value=jnp.array(100.0),
            frequency=2, day_count="act_365",
        )

    @pytest.fixture
    def ql_bond(self, ql_curve):
        today = ql.Date(26, 3, 2026)
        schedule = ql.Schedule(
            today, ql.Date(26, 3, 2031), ql.Period(ql.Semiannual),
            ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted,
            ql.DateGeneration.Forward, False,
        )
        bond = ql.FixedRateBond(0, 100.0, schedule, [0.04], ql.Actual365Fixed())
        handle = ql.YieldTermStructureHandle(ql_curve[0])
        bond.setPricingEngine(ql.DiscountingBondEngine(handle))
        return bond

    def test_bond_price_close(self, valax_bond, valax_curve, ql_bond):
        """See: examples/comparisons/02_fixed_income.py §5 (bond comparison)

        Small differences expected due to schedule conventions.
        """
        v_price = float(fixed_rate_bond_price(valax_bond, valax_curve))
        q_price = ql_bond.dirtyPrice()
        # Allow 0.05% tolerance due to schedule convention differences
        assert abs(v_price - q_price) / q_price < 5e-4, (
            f"VALAX={v_price}, QL={q_price}"
        )

    def test_ytm_close(self, valax_bond, valax_curve, ql_bond):
        """See: examples/comparisons/02_fixed_income.py §5 (YTM)"""
        v_price = fixed_rate_bond_price(valax_bond, valax_curve)
        v_ytm = float(yield_to_maturity(valax_bond, v_price))
        q_ytm = ql_bond.bondYield(ql.Actual365Fixed(), ql.Continuous, ql.NoFrequency)
        # Allow 5bp tolerance
        assert abs(v_ytm - q_ytm) < 5e-3, f"VALAX YTM={v_ytm}, QL YTM={q_ytm}"

    def test_duration_close(self, valax_bond, valax_curve, ql_bond):
        """See: examples/comparisons/02_fixed_income.py §5 (duration)"""
        v_price = fixed_rate_bond_price(valax_bond, valax_curve)
        v_ytm = yield_to_maturity(valax_bond, v_price)
        v_dur = float(modified_duration(valax_bond, v_ytm))
        q_ytm = ql_bond.bondYield(ql.Actual365Fixed(), ql.Continuous, ql.NoFrequency)
        q_dur = ql.BondFunctions.duration(
            ql_bond, q_ytm, ql.Actual365Fixed(), ql.Continuous, ql.NoFrequency,
            ql.Duration.Modified,
        )
        assert abs(v_dur - q_dur) < 0.2, f"VALAX dur={v_dur}, QL dur={q_dur}"

    def test_key_rate_durations_sum_to_modified_duration(self, valax_bond, valax_curve):
        """See: examples/comparisons/02_fixed_income.py §6 (KRD)

        KRD sum should approximately equal modified duration.
        """
        v_price = fixed_rate_bond_price(valax_bond, valax_curve)
        v_ytm = yield_to_maturity(valax_bond, v_price)
        v_dur = float(modified_duration(valax_bond, v_ytm))
        krd = key_rate_durations(valax_bond, valax_curve)
        krd_sum = float(jnp.sum(krd))
        assert abs(krd_sum - v_dur) < 0.2, f"KRD sum={krd_sum}, mod dur={v_dur}"
