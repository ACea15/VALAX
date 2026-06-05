"""Tests for HPL / RTPL P&L vectors over scenario sets."""

import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.options import EuropeanOption
from valax.market.data import MarketData
from valax.market.scenario import MarketScenario
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.risk.ladders import compute_ladder, waterfall_pnl
from valax.risk.pnl_vectors import (
    explained_unexplained_vector,
    hypothetical_pnl_vector,
    risk_theoretical_pnl_vector,
)
from valax.risk.scenarios import stack_scenarios, stress_scenario
from valax.risk.var import portfolio_pnl


def _bs_market_fn(option, market: MarketData) -> jnp.ndarray:
    from valax.risk.var import _extract_short_rate
    rate = _extract_short_rate(market.discount_curve)
    return black_scholes_price(
        option, market.spots, market.vols, rate, market.dividends,
    )


@pytest.fixture
def simple_portfolio():
    instruments = EuropeanOption(
        strike=jnp.array([100.0, 110.0]),
        expiry=jnp.array([0.5, 1.0]),
        is_call=True,
    )
    ref = ymd_to_ordinal(2026, 1, 1)
    pillars = jnp.array([
        ymd_to_ordinal(2026, 1, 1),
        ymd_to_ordinal(2026, 7, 1),
        ymd_to_ordinal(2027, 1, 1),
    ])
    rate = 0.05
    times = (pillars - ref).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-rate * times)
    curve = DiscountCurve(
        pillar_dates=pillars, discount_factors=dfs, reference_date=ref,
    )
    base = MarketData(
        spots=jnp.array([100.0, 100.0]),
        vols=jnp.array([0.2, 0.25]),
        dividends=jnp.array([0.0, 0.0]),
        discount_curve=curve,
    )
    return instruments, base


@pytest.fixture
def scenarios_small():
    return stack_scenarios([
        stress_scenario(2, 3, spot_shock=2.0, vol_shock=0.01),
        stress_scenario(2, 3, spot_shock=-3.0),
        stress_scenario(2, 3, parallel_rate_shift=0.005),
        stress_scenario(2, 3, vol_shock=-0.02),
        stress_scenario(2, 3),  # zero scenario
    ])


class TestRTPL:
    def test_shape(self, simple_portfolio, scenarios_small):
        instruments, base = simple_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)
        rtpl = risk_theoretical_pnl_vector(ladder, scenarios_small, base)
        assert rtpl.shape == (5,)

    def test_zero_scenario_zero_pnl(self, simple_portfolio, scenarios_small):
        instruments, base = simple_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)
        rtpl = risk_theoretical_pnl_vector(ladder, scenarios_small, base)
        # Last scenario is the all-zero one
        assert jnp.abs(rtpl[-1]) < 1e-10

    def test_matches_single_waterfall(self, simple_portfolio, scenarios_small):
        """Each entry of the RTPL vector must equal waterfall_pnl(...).predicted
        for the corresponding scenario."""
        instruments, base = simple_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)
        rtpl = risk_theoretical_pnl_vector(ladder, scenarios_small, base)

        # Reconstruct each scenario and call waterfall_pnl directly
        for i in range(scenarios_small.spot_shocks.shape[0]):
            scenario = MarketScenario(
                spot_shocks=scenarios_small.spot_shocks[i],
                vol_shocks=scenarios_small.vol_shocks[i],
                rate_shocks=scenarios_small.rate_shocks[i],
                dividend_shocks=scenarios_small.dividend_shocks[i],
                multiplicative=scenarios_small.multiplicative,
            )
            wf = waterfall_pnl(ladder, scenario, base)
            assert jnp.isclose(rtpl[i], wf.predicted, atol=1e-8), (
                f"scenario {i}: rtpl={float(rtpl[i])}, predicted={float(wf.predicted)}"
            )


class TestHPL:
    def test_matches_portfolio_pnl(self, simple_portfolio, scenarios_small):
        instruments, base = simple_portfolio
        hpl = hypothetical_pnl_vector(
            _bs_market_fn, instruments, base, scenarios_small,
        )
        ref = portfolio_pnl(_bs_market_fn, instruments, base, scenarios_small)
        assert jnp.allclose(hpl, ref, atol=1e-10)


class TestExplainedUnexplained:
    def test_unexplained_equals_hpl_minus_rtpl(
        self, simple_portfolio, scenarios_small,
    ):
        instruments, base = simple_portfolio
        report = explained_unexplained_vector(
            _bs_market_fn, instruments, base, scenarios_small,
        )
        assert jnp.allclose(
            report["unexplained"], report["hpl"] - report["rtpl"], atol=1e-10,
        )

    def test_small_moves_small_unexplained(
        self, simple_portfolio, scenarios_small,
    ):
        """For small scenarios, the second-order ladder should explain almost all P&L."""
        instruments, base = simple_portfolio
        report = explained_unexplained_vector(
            _bs_market_fn, instruments, base, scenarios_small,
        )
        # Drop the zero scenario (avoid divide-by-zero)
        mask = jnp.abs(report["hpl"]) > 1e-6
        rel = jnp.abs(report["unexplained"][mask]) / jnp.abs(report["hpl"][mask])
        # Each non-trivial scenario should be explained to better than 5%
        assert jnp.all(rel < 0.05), f"max relative unexplained {float(jnp.max(rel)):.4f}"

    def test_precomputed_ladder_matches_on_the_fly(
        self, simple_portfolio, scenarios_small,
    ):
        instruments, base = simple_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)
        r1 = explained_unexplained_vector(
            _bs_market_fn, instruments, base, scenarios_small, ladder=ladder,
        )
        r2 = explained_unexplained_vector(
            _bs_market_fn, instruments, base, scenarios_small,
        )
        assert jnp.allclose(r1["rtpl"], r2["rtpl"], atol=1e-10)
        assert jnp.allclose(r1["hpl"], r2["hpl"], atol=1e-10)
