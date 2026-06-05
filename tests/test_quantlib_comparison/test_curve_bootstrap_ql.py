"""Cross-validation: VALAX vs QuantLib curve bootstrapping.

Stage 2 of the [QuantLib Validation Pyramid](
../../docs/architecture/quantlib-validation-pyramid.md).

Per seed:
  1. Synthesise an NSS *truth* curve via :func:`sample_nss_curve`.
  2. Read simply-compounded par-deposit rates at four custom tenors
     **deliberately offset from the NSS pillar grid** (4M, 18M, 4Y,
     8Y).
  3. Bootstrap a new curve with both
     :func:`valax.curves.bootstrap_sequential` (using
     :class:`DepositRate` quotes) and
     ``ql.PiecewiseLogLinearDiscount`` (using
     ``ql.DepositRateHelper`` quotes).
  4. Assert the two fitted curves produce identical discount factors
     at every checked off-pillar date, and that both reproduce the
     input par quotes exactly.

Three disjoint date grids — NSS truth pillars, bootstrap quote
tenors, comparison dates — guarantee that DF agreement at the
comparison dates is real interpolation evidence, not a coincidence.

Tolerance: ``abs < 1e-10`` because both libraries use log-linear
interpolation on discount factors and the same Act/365 day count.
Any drift is a wiring bug, not a numerical artefact.
"""

import jax.numpy as jnp
import pytest
import QuantLib as ql

import valax
from valax.curves.bootstrap import bootstrap_sequential
from valax.curves.instruments import DepositRate
from valax.dates.daycounts import year_fraction
from valax.market import (
    SeedRegistry,
    SyntheticMarketConfig,
    sample_nss_curve,
)

from tests.test_quantlib_comparison._ql_adapters import DEFAULT_QL_DATE


SEEDS = tuple(range(20260101, 20260121))

# Bootstrap quote tenors: deliberately *not* on the NSS pillar grid
# (which sits at 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y).
BOOTSTRAP_TENORS_DAYS = (120, 540, 1460, 2920)   # 4M, 18M, 4Y, 8Y

# Comparison dates: a third disjoint grid for the interpolation test.
COMPARISON_TENORS_DAYS = (240, 900, 1825, 2400)   # 8M, 30M, 5Y, 6.6Y


def _par_deposit_rate(truth_curve, end_ordinal) -> float:
    """Simply-compounded par-deposit rate implied by ``truth_curve``."""
    ref = truth_curve.reference_date
    tau = float(year_fraction(ref, end_ordinal, truth_curve.day_count))
    df_end = float(truth_curve(end_ordinal))
    # DF(start) = 1, so rate = (1 / DF(end) - 1) / tau.
    return (1.0 / df_end - 1.0) / tau


@pytest.fixture(params=SEEDS, ids=lambda s: f"seed={s}")
def truth_curve(request):
    """Per-seed synthetic NSS truth curve."""
    registry = SeedRegistry(
        master_seed=request.param, library_version=valax.__version__,
    )
    cfg = SyntheticMarketConfig(n_assets=1, curve_kind="nss")
    return sample_nss_curve(registry, cfg)


@pytest.fixture
def bootstrap_setup(truth_curve):
    """Build par-deposit quotes from truth at non-NSS tenors and
    bootstrap with both libraries."""
    ref_ord = truth_curve.reference_date

    # 1. Read par-deposit rates from truth at the bootstrap tenors.
    rates = []
    for d in BOOTSTRAP_TENORS_DAYS:
        end_ord = ref_ord + jnp.int32(d)
        rates.append(_par_deposit_rate(truth_curve, end_ord))

    # 2. VALAX bootstrap from DepositRate quotes.
    valax_quotes = [
        DepositRate(
            start_date=ref_ord,
            end_date=ref_ord + jnp.int32(d),
            rate=jnp.array(r),
            day_count="act_365",
        )
        for d, r in zip(BOOTSTRAP_TENORS_DAYS, rates, strict=True)
    ]
    valax_curve = bootstrap_sequential(
        ref_ord, valax_quotes, day_count="act_365",
    )

    # 3. QL bootstrap from DepositRateHelper quotes.
    #
    # ql.DepositRateHelper(rate, period, fixingDays, calendar,
    #                       convention, endOfMonth, dayCount)
    # Settings.evaluationDate must be set to the reference.
    today_ql = DEFAULT_QL_DATE
    ql.Settings.instance().evaluationDate = today_ql
    dc = ql.Actual365Fixed()
    cal = ql.NullCalendar()

    helpers = [
        ql.DepositRateHelper(
            ql.QuoteHandle(ql.SimpleQuote(r)),
            ql.Period(d, ql.Days),
            0,                          # fixing days
            cal,
            ql.Unadjusted,
            False,                      # end of month
            dc,
        )
        for d, r in zip(BOOTSTRAP_TENORS_DAYS, rates, strict=True)
    ]
    ql_curve = ql.PiecewiseLogLinearDiscount(today_ql, helpers, dc)

    return {
        "rates": rates,
        "valax_curve": valax_curve,
        "ql_curve": ql_curve,
        "today_ql": today_ql,
    }


# ---------------------------------------------------------------------------
# Per-pillar agreement (the bootstrap must reproduce its own quotes)
# ---------------------------------------------------------------------------


class TestBootstrapAtPillars:
    """At each bootstrap pillar both curves must hit the par-quote DF
    to machine precision."""

    @pytest.mark.parametrize("idx", range(len(BOOTSTRAP_TENORS_DAYS)))
    def test_pillar_df_matches(self, bootstrap_setup, idx):
        days = BOOTSTRAP_TENORS_DAYS[idx]
        rate = bootstrap_setup["rates"][idx]
        tau = days / 365.0

        # The DF that *should* live at the pillar.
        expected = 1.0 / (1.0 + rate * tau)

        v_df = float(bootstrap_setup["valax_curve"](
            bootstrap_setup["valax_curve"].reference_date + jnp.int32(days)
        ))
        q_df = bootstrap_setup["ql_curve"].discount(
            bootstrap_setup["today_ql"] + ql.Period(days, ql.Days)
        )

        assert v_df == pytest.approx(expected, abs=1e-12), (
            f"VALAX off-quote at d={days}: {v_df} vs expected {expected}"
        )
        assert q_df == pytest.approx(expected, abs=1e-12), (
            f"QL off-quote at d={days}: {q_df} vs expected {expected}"
        )


# ---------------------------------------------------------------------------
# Off-pillar agreement (the interpolation test)
# ---------------------------------------------------------------------------


class TestBootstrapOffPillar:
    """At dates *between* the bootstrap pillars, the two curves must
    interpolate identically (log-linear in DF on both sides)."""

    @pytest.mark.parametrize("days", COMPARISON_TENORS_DAYS)
    def test_offpillar_df_matches(self, bootstrap_setup, days):
        ref_v = bootstrap_setup["valax_curve"].reference_date
        v_df = float(
            bootstrap_setup["valax_curve"](ref_v + jnp.int32(days))
        )
        q_df = bootstrap_setup["ql_curve"].discount(
            bootstrap_setup["today_ql"] + ql.Period(days, ql.Days)
        )
        assert v_df == pytest.approx(q_df, abs=1e-10), (
            f"Off-pillar mismatch at d={days}: VALAX={v_df:.15f}  "
            f"QL={q_df:.15f}  diff={abs(v_df - q_df):.2e}"
        )
