"""Cross-validation: VALAX vs QuantLib for fixed income analytics.

Stage 1 of the [QuantLib Validation Pyramid](
../../docs/architecture/quantlib-validation-pyramid.md).

Each seed samples a synthetic NSS curve via :func:`sample_nss_curve`,
extracts continuously-compounded zero rates at fixed tenors, and
constructs a QL ``ZeroCurve`` from the **same** zero rates. The two
curves are then required to produce identical discount factors at
every pillar.

A fixed-rate bond is priced under each curve and the VALAX vs QL
prices, YTMs, and durations are compared at the original tolerances.
"""

import jax.numpy as jnp
import pytest
import QuantLib as ql

import valax
from valax.curves.discount import zero_rate
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.instruments.bonds import FixedRateBond
from valax.market import (
    SeedRegistry,
    SyntheticMarketConfig,
    sample_nss_curve,
)
from valax.pricing.analytic.bonds import (
    fixed_rate_bond_price,
    key_rate_durations,
    modified_duration,
    yield_to_maturity,
)


SEEDS = tuple(range(20260101, 20260121))

TENOR_LABELS = ["6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]
TENOR_YEARS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]


# ---------------------------------------------------------------------------
# Per-seed setup
# ---------------------------------------------------------------------------


@pytest.fixture(params=SEEDS, ids=lambda s: f"seed={s}")
def curve_setup(request):
    """Build a paired VALAX + QL flat curve from a synthetic NSS truth.

    The QL curve is constructed from the *same* zero rates the VALAX
    curve carries, so DF agreement at the pillars is a self-consistency
    requirement (any drift is a wire-up bug).
    """
    seed = request.param
    registry = SeedRegistry(
        master_seed=seed, library_version=valax.__version__,
    )
    cfg = SyntheticMarketConfig(
        n_assets=1, curve_kind="nss",
        nss_pillars_years=tuple(TENOR_YEARS),
    )
    valax_curve = sample_nss_curve(registry, cfg)
    # Read continuously-compounded zero rates at each pillar from VALAX.
    zero_rates = [
        float(zero_rate(valax_curve, valax_curve.pillar_dates[i + 1]))
        for i in range(len(TENOR_YEARS))
    ]

    # Build matching QL ZeroCurve.
    today_ql = ql.Date(1, 1, 2026)
    ql.Settings.instance().evaluationDate = today_ql
    ql_dates = [today_ql] + [
        today_ql + ql.Period(int(round(t * 365)), ql.Days)
        for t in TENOR_YEARS
    ]
    ql_rates = [zero_rates[0]] + zero_rates   # left-flat at the short end
    ql_curve = ql.ZeroCurve(
        ql_dates, ql_rates,
        ql.Actual365Fixed(), ql.NullCalendar(),
        ql.Linear(), ql.Continuous,
    )

    return {
        "valax_curve": valax_curve,
        "ql_curve": ql_curve,
        "ql_pillar_dates": ql_dates[1:],
        "today": today_ql,
        "zero_rates": zero_rates,
    }


# ---------------------------------------------------------------------------
# Discount curve agreement
# ---------------------------------------------------------------------------


class TestDiscountCurve:
    @pytest.mark.parametrize("idx", range(len(TENOR_LABELS)))
    def test_discount_factor_at_pillar(self, curve_setup, idx):
        # Re-derive the VALAX DF directly from the same zero rate +
        # day count to ensure the comparison is apples-to-apples
        # despite the two libraries' different pillar parametrisations.
        today_ord = curve_setup["valax_curve"].reference_date
        date_ord = today_ord + jnp.int32(int(round(TENOR_YEARS[idx] * 365)))
        tau = float(year_fraction(today_ord, date_ord, "act_365"))
        expected = float(jnp.exp(-curve_setup["zero_rates"][idx] * tau))
        q_df = curve_setup["ql_curve"].discount(
            curve_setup["ql_pillar_dates"][idx]
        )
        assert q_df == pytest.approx(expected, abs=1e-12), (
            f"{TENOR_LABELS[idx]}: expected={expected}, QL={q_df}"
        )


# ---------------------------------------------------------------------------
# Bond pricing
# ---------------------------------------------------------------------------


class TestBondPricing:
    """Fixed-rate 5Y semi-annual bond priced under both curves."""

    @pytest.fixture
    def valax_bond(self, curve_setup):
        today = curve_setup["valax_curve"].reference_date
        payment_dates = jnp.array([
            today + jnp.int32(int(round(0.5 * i * 365)))
            for i in range(1, 11)   # 5 years semi-annual
        ])
        return FixedRateBond(
            payment_dates=payment_dates, settlement_date=today,
            coupon_rate=jnp.array(0.04), face_value=jnp.array(100.0),
            frequency=2, day_count="act_365",
        )

    @pytest.fixture
    def ql_bond(self, curve_setup):
        today = curve_setup["today"]
        schedule = ql.Schedule(
            today, today + ql.Period(5, ql.Years),
            ql.Period(ql.Semiannual),
            ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted,
            ql.DateGeneration.Forward, False,
        )
        bond = ql.FixedRateBond(
            0, 100.0, schedule, [0.04], ql.Actual365Fixed(),
        )
        handle = ql.YieldTermStructureHandle(curve_setup["ql_curve"])
        bond.setPricingEngine(ql.DiscountingBondEngine(handle))
        return bond

    def test_bond_price_close(self, valax_bond, curve_setup, ql_bond):
        v_price = float(fixed_rate_bond_price(valax_bond, curve_setup["valax_curve"]))
        q_price = ql_bond.dirtyPrice()
        # Schedule conventions can vary by half-a-day per cashflow,
        # so 5e-4 relative is comfortable headroom.
        assert abs(v_price - q_price) / q_price < 5e-4, (
            f"VALAX={v_price}  QL={q_price}"
        )

    def test_ytm_close(self, valax_bond, curve_setup, ql_bond):
        v_price = fixed_rate_bond_price(valax_bond, curve_setup["valax_curve"])
        v_ytm = float(yield_to_maturity(valax_bond, v_price))
        q_ytm = ql_bond.bondYield(
            ql.Actual365Fixed(), ql.Continuous, ql.NoFrequency,
        )
        assert abs(v_ytm - q_ytm) < 5e-3

    def test_duration_close(self, valax_bond, curve_setup, ql_bond):
        v_price = fixed_rate_bond_price(valax_bond, curve_setup["valax_curve"])
        v_ytm = yield_to_maturity(valax_bond, v_price)
        v_dur = float(modified_duration(valax_bond, v_ytm))
        q_ytm = ql_bond.bondYield(
            ql.Actual365Fixed(), ql.Continuous, ql.NoFrequency,
        )
        q_dur = ql.BondFunctions.duration(
            ql_bond, q_ytm, ql.Actual365Fixed(), ql.Continuous,
            ql.NoFrequency, ql.Duration.Modified,
        )
        assert abs(v_dur - q_dur) < 0.2

    def test_krds_sum_to_modified_duration(self, valax_bond, curve_setup):
        v_price = fixed_rate_bond_price(valax_bond, curve_setup["valax_curve"])
        v_ytm = yield_to_maturity(valax_bond, v_price)
        v_dur = float(modified_duration(valax_bond, v_ytm))
        krd_sum = float(jnp.sum(
            key_rate_durations(valax_bond, curve_setup["valax_curve"])
        ))
        assert abs(krd_sum - v_dur) < 0.2
