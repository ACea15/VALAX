"""Tests for sensitivity ladders and waterfall P&L decomposition."""

import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.options import EuropeanOption
from valax.market.data import MarketData
from valax.market.scenario import MarketScenario
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.risk.ladders import (
    SensitivityLadder,
    WaterfallPnL,
    compute_ladder,
    waterfall_pnl,
    waterfall_pnl_report,
)
from valax.risk.var import pnl_attribution


# ── Fixtures ─────────────────────────────────────────────────────────


def _bs_market_fn(option, market: MarketData) -> jnp.ndarray:
    """Black-Scholes pricing function with MarketData signature."""
    from valax.risk.var import _extract_short_rate

    rate = _extract_short_rate(market.discount_curve)
    return black_scholes_price(
        option, market.spots, market.vols, rate, market.dividends,
    )


@pytest.fixture
def simple_portfolio():
    """Two call options and a base market state."""
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
        pillar_dates=pillars,
        discount_factors=dfs,
        reference_date=ref,
    )
    base = MarketData(
        spots=jnp.array([100.0, 100.0]),
        vols=jnp.array([0.2, 0.25]),
        dividends=jnp.array([0.0, 0.0]),
        discount_curve=curve,
    )
    return instruments, base


@pytest.fixture
def small_scenario():
    """A moderate scenario with moves in all risk factors."""
    return MarketScenario(
        spot_shocks=jnp.array([2.0, -3.0]),
        vol_shocks=jnp.array([0.01, -0.02]),
        rate_shocks=jnp.array([0.005, 0.005, 0.005]),
        dividend_shocks=jnp.array([0.001, 0.0]),
    )


@pytest.fixture
def large_scenario():
    """A large scenario where nonlinear terms matter."""
    return MarketScenario(
        spot_shocks=jnp.array([15.0, -20.0]),
        vol_shocks=jnp.array([0.05, -0.08]),
        rate_shocks=jnp.array([0.02, 0.02, 0.02]),
        dividend_shocks=jnp.array([0.005, 0.005]),
    )


# ── Test: Ladder shape and structure ─────────────────────────────────


class TestComputeLadder:
    def test_shapes(self, simple_portfolio):
        """All ladder fields should have correct shapes."""
        instruments, base = simple_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)

        n_assets = 2
        n_pillars = 3

        # First order
        assert ladder.delta_spot.shape == (n_assets,)
        assert ladder.delta_vol.shape == (n_assets,)
        assert ladder.delta_rate.shape == (n_pillars,)
        assert ladder.delta_div.shape == (n_assets,)

        # Second order diagonals
        assert ladder.gamma_spot.shape == (n_assets,)
        assert ladder.gamma_rate.shape == (n_pillars,)
        assert ladder.volga.shape == (n_assets,)
        assert ladder.vanna.shape == (n_assets,)

        # Cross blocks
        assert ladder.cross_spot_rate.shape == (n_assets, n_pillars)
        assert ladder.cross_vol_rate.shape == (n_assets, n_pillars)

    def test_delta_spot_positive_for_calls(self, simple_portfolio):
        """Call option deltas should be positive."""
        instruments, base = simple_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)
        assert jnp.all(ladder.delta_spot > 0)

    def test_gamma_spot_positive_for_calls(self, simple_portfolio):
        """Call option gammas should be positive."""
        instruments, base = simple_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)
        assert jnp.all(ladder.gamma_spot > 0)

    def test_vega_positive_for_long_options(self, simple_portfolio):
        """Long option vegas should be positive."""
        instruments, base = simple_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)
        assert jnp.all(ladder.delta_vol > 0)

    def test_volga_positive_for_options(self, simple_portfolio):
        """Long option volgas should be positive (vega convexity)."""
        instruments, base = simple_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)
        assert jnp.all(ladder.volga > 0)


# ── Test: Ladder first order matches existing pnl_attribution ────────


class TestLadderConsistency:
    def test_first_order_matches_pnl_attribution(
        self, simple_portfolio, small_scenario,
    ):
        """Ladder delta terms should match pnl_attribution delta terms."""
        instruments, base = simple_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)

        # Existing pnl_attribution
        attrib = pnl_attribution(
            _bs_market_fn, instruments, base, small_scenario,
        )

        # Ladder first-order P&L
        d_spots = small_scenario.spot_shocks
        d_vols = small_scenario.vol_shocks
        d_rates = small_scenario.rate_shocks

        ladder_delta_spot = jnp.sum(ladder.delta_spot * d_spots)
        ladder_delta_vol = jnp.sum(ladder.delta_vol * d_vols)
        ladder_delta_rate = jnp.sum(ladder.delta_rate * d_rates)

        assert jnp.isclose(ladder_delta_spot, attrib["delta_spot"], rtol=1e-4)
        assert jnp.isclose(ladder_delta_vol, attrib["delta_vol"], rtol=1e-4)
        assert jnp.isclose(ladder_delta_rate, attrib["delta_rate"], rtol=1e-4)

    def test_spot_gamma_matches_pnl_attribution(
        self, simple_portfolio, small_scenario,
    ):
        """Ladder gamma_spot P&L should match pnl_attribution gamma_spot."""
        instruments, base = simple_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)
        attrib = pnl_attribution(
            _bs_market_fn, instruments, base, small_scenario,
        )

        d_spots = small_scenario.spot_shocks
        ladder_gamma = 0.5 * jnp.sum(ladder.gamma_spot * d_spots**2)

        assert jnp.isclose(ladder_gamma, attrib["gamma_spot"], rtol=1e-3)


# ── Test: Waterfall P&L ──────────────────────────────────────────────


class TestWaterfallPnL:
    def test_zero_scenario_zero_pnl(self, simple_portfolio):
        """Zero scenario should produce zero waterfall P&L."""
        instruments, base = simple_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)

        zero = MarketScenario(
            spot_shocks=jnp.zeros(2),
            vol_shocks=jnp.zeros(2),
            rate_shocks=jnp.zeros(3),
            dividend_shocks=jnp.zeros(2),
        )
        wf = waterfall_pnl(ladder, zero, base)

        assert jnp.isclose(wf.delta_spot, 0.0, atol=1e-10)
        assert jnp.isclose(wf.delta_vol, 0.0, atol=1e-10)
        assert jnp.isclose(wf.predicted, 0.0, atol=1e-10)

    def test_rungs_sum_to_predicted(self, simple_portfolio, small_scenario):
        """Sum of all rungs should equal predicted."""
        instruments, base = simple_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)
        wf = waterfall_pnl(ladder, small_scenario, base)

        rung_sum = (
            wf.delta_spot
            + wf.delta_vol
            + wf.delta_rate
            + wf.delta_div
            + wf.gamma_spot
            + wf.gamma_rate
            + wf.vanna_pnl
            + wf.volga_pnl
            + wf.cross_spot_rate_pnl
            + wf.cross_vol_rate_pnl
        )
        assert jnp.isclose(rung_sum, wf.predicted, atol=1e-10)

    def test_first_plus_second_equals_predicted(
        self, simple_portfolio, small_scenario,
    ):
        """total_first_order + total_second_order should equal predicted."""
        instruments, base = simple_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)
        wf = waterfall_pnl(ladder, small_scenario, base)

        assert jnp.isclose(
            wf.total_first_order + wf.total_second_order,
            wf.predicted,
            atol=1e-10,
        )


# ── Test: Waterfall report with full repricing ───────────────────────


class TestWaterfallReport:
    def test_actual_matches_full_repricing(
        self, simple_portfolio, small_scenario,
    ):
        """Waterfall actual should match pnl_attribution actual."""
        instruments, base = simple_portfolio
        wf = waterfall_pnl_report(
            _bs_market_fn, instruments, base, small_scenario,
        )
        attrib = pnl_attribution(
            _bs_market_fn, instruments, base, small_scenario,
        )
        assert jnp.isclose(wf.actual, attrib["actual"], atol=1e-8)

    def test_unexplained_populated(self, simple_portfolio, small_scenario):
        """Unexplained should be actual - predicted (not NaN)."""
        instruments, base = simple_portfolio
        wf = waterfall_pnl_report(
            _bs_market_fn, instruments, base, small_scenario,
        )
        assert not jnp.isnan(wf.unexplained)
        assert jnp.isclose(
            wf.unexplained, wf.actual - wf.predicted, atol=1e-10,
        )

    def test_waterfall_more_accurate_than_first_order(
        self, simple_portfolio, large_scenario,
    ):
        """Full waterfall (with 2nd-order) should have smaller unexplained
        than a first-order-only approximation."""
        instruments, base = simple_portfolio
        wf = waterfall_pnl_report(
            _bs_market_fn, instruments, base, large_scenario,
        )

        # First-order-only unexplained
        first_order_unexplained = jnp.abs(wf.actual - wf.total_first_order)
        # Full waterfall unexplained
        full_unexplained = jnp.abs(wf.unexplained)

        assert full_unexplained < first_order_unexplained

    def test_small_scenario_small_unexplained(
        self, simple_portfolio, small_scenario,
    ):
        """For small moves, the Taylor approximation should be very accurate."""
        instruments, base = simple_portfolio
        wf = waterfall_pnl_report(
            _bs_market_fn, instruments, base, small_scenario,
        )
        # Unexplained should be <1% of actual for small moves
        rel_error = jnp.abs(wf.unexplained) / jnp.maximum(
            jnp.abs(wf.actual), 1e-10,
        )
        assert rel_error < 0.01


# ── Test: Precomputed ladder reuse ───────────────────────────────────


class TestLadderReuse:
    def test_precomputed_matches_on_the_fly(
        self, simple_portfolio, small_scenario,
    ):
        """waterfall_pnl_report with precomputed ladder should match
        on-the-fly computation."""
        instruments, base = simple_portfolio
        ladder = compute_ladder(_bs_market_fn, instruments, base)

        wf_precomputed = waterfall_pnl_report(
            _bs_market_fn, instruments, base, small_scenario, ladder=ladder,
        )
        wf_fresh = waterfall_pnl_report(
            _bs_market_fn, instruments, base, small_scenario,
        )

        assert jnp.isclose(wf_precomputed.predicted, wf_fresh.predicted, atol=1e-10)
        assert jnp.isclose(wf_precomputed.actual, wf_fresh.actual, atol=1e-10)
