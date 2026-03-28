"""End-to-end tests for VaR, parametric VaR, and P&L attribution."""

import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.options import EuropeanOption
from valax.instruments.rates import InterestRateSwap
from valax.market.data import MarketData
from valax.market.scenario import MarketScenario, ScenarioSet
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.analytic.swaptions import swap_price
from valax.risk.scenarios import parametric_scenarios, stress_scenario, stack_scenarios
from valax.risk.var import (
    expected_shortfall,
    parametric_var,
    pnl_attribution,
    portfolio_pnl,
    reprice_under_scenario,
    value_at_risk,
    wrap_equity_pricing_fn,
)


# ── Fixtures ─────────────────────────────────────────────────────────


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


def _bs_market_fn(option, market: MarketData) -> jnp.ndarray:
    """Black-Scholes pricing function with MarketData signature.

    Receives per-instrument MarketData (scalar spot, vol, dividend)
    plus the shared discount curve. Extracts a scalar rate from the curve.
    """
    from valax.risk.var import _extract_short_rate
    rate = _extract_short_rate(market.discount_curve)
    return black_scholes_price(option, market.spots, market.vols, rate, market.dividends)


# ── Test: wrap_equity_pricing_fn ─────────────────────────────────────


class TestWrapEquityPricingFn:
    def test_wrapped_matches_direct(self, simple_portfolio):
        """Wrapped equity fn should produce same results as direct call."""
        instruments, base = simple_portfolio
        wrapped = wrap_equity_pricing_fn(black_scholes_price)

        # Direct pricing
        p1 = black_scholes_price(
            EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(0.5), is_call=True),
            jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.0),
        )
        # Via wrapper
        single_inst = EuropeanOption(
            strike=jnp.array(100.0), expiry=jnp.array(0.5), is_call=True,
        )
        # Wrapper expects scalar market data for a single instrument
        single_market = MarketData(
            spots=jnp.array(100.0),
            vols=jnp.array(0.2),
            dividends=jnp.array(0.0),
            discount_curve=base.discount_curve,
        )
        p2 = wrapped(single_inst, single_market)
        assert jnp.isclose(p1, p2, atol=1e-6)


# ── Test: Full-revaluation repricing ─────────────────────────────────


class TestRepriceUnderScenario:
    def test_zero_scenario_matches_base(self, simple_portfolio):
        instruments, base = simple_portfolio
        zero = MarketScenario(
            spot_shocks=jnp.zeros(2),
            vol_shocks=jnp.zeros(2),
            rate_shocks=jnp.zeros(3),
            dividend_shocks=jnp.zeros(2),
        )
        val = reprice_under_scenario(_bs_market_fn, instruments, base, zero)
        # Should match direct pricing
        p1 = black_scholes_price(
            EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(0.5), is_call=True),
            jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.0),
        )
        p2 = black_scholes_price(
            EuropeanOption(strike=jnp.array(110.0), expiry=jnp.array(1.0), is_call=True),
            jnp.array(100.0), jnp.array(0.25), jnp.array(0.05), jnp.array(0.0),
        )
        assert jnp.isclose(val, p1 + p2, atol=1e-4)

    def test_spot_shock_changes_value(self, simple_portfolio):
        instruments, base = simple_portfolio
        up = MarketScenario(
            spot_shocks=jnp.array([10.0, 10.0]),
            vol_shocks=jnp.zeros(2),
            rate_shocks=jnp.zeros(3),
            dividend_shocks=jnp.zeros(2),
        )
        down = MarketScenario(
            spot_shocks=jnp.array([-10.0, -10.0]),
            vol_shocks=jnp.zeros(2),
            rate_shocks=jnp.zeros(3),
            dividend_shocks=jnp.zeros(2),
        )
        val_up = reprice_under_scenario(_bs_market_fn, instruments, base, up)
        val_down = reprice_under_scenario(_bs_market_fn, instruments, base, down)
        assert val_up > val_down

    def test_rate_shock_affects_value(self, simple_portfolio):
        """Rate shocks should change option values via the full curve."""
        instruments, base = simple_portfolio
        rate_up = MarketScenario(
            spot_shocks=jnp.zeros(2),
            vol_shocks=jnp.zeros(2),
            rate_shocks=jnp.full(3, 0.02),  # +200bp
            dividend_shocks=jnp.zeros(2),
        )
        zero = MarketScenario(
            spot_shocks=jnp.zeros(2),
            vol_shocks=jnp.zeros(2),
            rate_shocks=jnp.zeros(3),
            dividend_shocks=jnp.zeros(2),
        )
        val_base = reprice_under_scenario(_bs_market_fn, instruments, base, zero)
        val_up = reprice_under_scenario(_bs_market_fn, instruments, base, rate_up)
        # Rate change should affect the value
        assert not jnp.isclose(val_base, val_up, atol=1e-4)


# ── Test: Rates product repricing ────────────────────────────────────


class TestRatesRepricing:
    def test_swap_repricing_under_parallel_shift(self):
        """A payer swap should gain value when rates go up."""
        ref = ymd_to_ordinal(2026, 1, 1)
        pillars = jnp.array([
            ymd_to_ordinal(2026, 1, 1),
            ymd_to_ordinal(2027, 1, 1),
            ymd_to_ordinal(2028, 1, 1),
            ymd_to_ordinal(2029, 1, 1),
        ])
        rate = 0.04
        times = (pillars - ref).astype(jnp.float64) / 365.0
        dfs = jnp.exp(-rate * times)
        curve = DiscountCurve(
            pillar_dates=pillars, discount_factors=dfs, reference_date=ref,
        )
        base = MarketData(
            spots=jnp.array([0.0]),  # unused for rates
            vols=jnp.array([0.0]),
            dividends=jnp.array([0.0]),
            discount_curve=curve,
        )

        # Compute the actual par swap rate on this curve
        from valax.pricing.analytic.swaptions import swap_rate as compute_swap_rate
        par_swap = InterestRateSwap(
            start_date=pillars[0],
            fixed_dates=pillars[1:],
            fixed_rate=jnp.array(0.04),  # placeholder
            notional=jnp.array(1e6),
            pay_fixed=True,
            day_count="act_365",
        )
        par_rate = compute_swap_rate(par_swap, curve)

        # Batch of one swap at the par rate
        swaps = InterestRateSwap(
            start_date=pillars[0:1],
            fixed_dates=pillars[1:].reshape(1, -1),
            fixed_rate=par_rate[None],
            notional=jnp.array([1e6]),
            pay_fixed=True,
            day_count="act_365",
        )

        def swap_market_fn(swap_inst, market: MarketData):
            return swap_price(swap_inst, market.discount_curve)

        zero = MarketScenario(
            spot_shocks=jnp.zeros(1),
            vol_shocks=jnp.zeros(1),
            rate_shocks=jnp.zeros(4),
            dividend_shocks=jnp.zeros(1),
        )
        rate_up = MarketScenario(
            spot_shocks=jnp.zeros(1),
            vol_shocks=jnp.zeros(1),
            rate_shocks=jnp.full(4, 0.01),  # +100bp
            dividend_shocks=jnp.zeros(1),
        )

        val_base = reprice_under_scenario(swap_market_fn, swaps, base, zero)
        val_up = reprice_under_scenario(swap_market_fn, swaps, base, rate_up)

        # ATM payer swap at par rate should be ~0
        assert abs(float(val_base)) < 10.0, f"ATM swap NPV should be ~0, got {val_base}"
        # Payer swap gains when rates go up (receive float worth more)
        assert float(val_up) > float(val_base) + 1000.0


# ── Test: Portfolio P&L ──────────────────────────────────────────────


class TestPortfolioPnl:
    def test_shape(self, simple_portfolio):
        instruments, base = simple_portfolio
        scenarios = stack_scenarios([
            stress_scenario(2, 3, spot_shock=5.0),
            stress_scenario(2, 3, spot_shock=-5.0),
            stress_scenario(2, 3, vol_shock=0.05),
        ])
        pnl = portfolio_pnl(_bs_market_fn, instruments, base, scenarios)
        assert pnl.shape == (3,)

    def test_spot_up_positive_pnl(self, simple_portfolio):
        instruments, base = simple_portfolio
        scenarios = stack_scenarios([
            stress_scenario(2, 3, spot_shock=10.0),
        ])
        pnl = portfolio_pnl(_bs_market_fn, instruments, base, scenarios)
        assert pnl[0] > 0


# ── Test: VaR and ES ─────────────────────────────────────────────────


class TestVaRAndES:
    def test_var_positive_for_risky_portfolio(self, simple_portfolio):
        instruments, base = simple_portfolio
        key = jax.random.PRNGKey(0)
        n_factors = 2 + 2 + 3 + 2
        cov = jnp.eye(n_factors) * 0.01
        cov = cov.at[0, 0].set(100.0)
        cov = cov.at[1, 1].set(100.0)
        scenarios = parametric_scenarios(key, cov, 5000, 2, 3)
        pnl = portfolio_pnl(_bs_market_fn, instruments, base, scenarios)
        var_99 = value_at_risk(pnl, confidence=0.99)
        es_99 = expected_shortfall(pnl, confidence=0.99)
        assert var_99 > 0
        assert es_99 >= var_99 - 1e-6

    def test_var_increases_with_confidence(self, simple_portfolio):
        instruments, base = simple_portfolio
        key = jax.random.PRNGKey(1)
        n_factors = 9
        cov = jnp.eye(n_factors) * 1.0
        scenarios = parametric_scenarios(key, cov, 10_000, 2, 3)
        pnl = portfolio_pnl(_bs_market_fn, instruments, base, scenarios)
        var_95 = value_at_risk(pnl, confidence=0.95)
        var_99 = value_at_risk(pnl, confidence=0.99)
        assert var_99 >= var_95 - 1e-6


# ── Test: Parametric VaR ─────────────────────────────────────────────


class TestParametricVaR:
    def test_positive_for_risky_portfolio(self, simple_portfolio):
        instruments, base = simple_portfolio
        n_factors = 2 + 2 + 3 + 2
        cov = jnp.eye(n_factors) * 0.01
        cov = cov.at[0, 0].set(100.0)
        cov = cov.at[1, 1].set(100.0)
        pvar = parametric_var(_bs_market_fn, instruments, base, cov, confidence=0.99)
        assert pvar > 0

    def test_increases_with_confidence(self, simple_portfolio):
        instruments, base = simple_portfolio
        n_factors = 9
        cov = jnp.eye(n_factors) * 1.0
        pvar_95 = parametric_var(_bs_market_fn, instruments, base, cov, confidence=0.95)
        pvar_99 = parametric_var(_bs_market_fn, instruments, base, cov, confidence=0.99)
        assert pvar_99 >= pvar_95 - 1e-6

    def test_zero_cov_gives_zero_var(self, simple_portfolio):
        """No risk factor variance → zero VaR."""
        instruments, base = simple_portfolio
        n_factors = 9
        cov = jnp.zeros((n_factors, n_factors))
        pvar = parametric_var(_bs_market_fn, instruments, base, cov, confidence=0.99)
        assert abs(float(pvar)) < 1e-6


# ── Test: P&L Attribution ────────────────────────────────────────────


class TestPnLAttribution:
    def test_spot_shock_attribution(self, simple_portfolio):
        """Spot shock P&L should be mostly explained by delta + gamma."""
        instruments, base = simple_portfolio
        scenario = stress_scenario(2, 3, spot_shock=5.0)
        attr = pnl_attribution(_bs_market_fn, instruments, base, scenario)

        # Delta should be the dominant term for a small shock
        assert abs(float(attr["delta_spot"])) > 0
        # Second-order approximation should be close to actual
        assert abs(float(attr["unexplained"])) < abs(float(attr["actual"])) * 0.1 + 0.01

    def test_zero_scenario_zero_attribution(self, simple_portfolio):
        """Zero scenario should give zero P&L everywhere."""
        instruments, base = simple_portfolio
        zero = stress_scenario(2, 3)
        attr = pnl_attribution(_bs_market_fn, instruments, base, zero)

        assert abs(float(attr["actual"])) < 1e-6
        assert abs(float(attr["total_first_order"])) < 1e-6

    def test_all_components_present(self, simple_portfolio):
        instruments, base = simple_portfolio
        scenario = stress_scenario(2, 3, spot_shock=3.0, vol_shock=0.01)
        attr = pnl_attribution(_bs_market_fn, instruments, base, scenario)

        expected_keys = {
            "delta_spot", "delta_vol", "delta_rate", "delta_div",
            "gamma_spot", "total_first_order", "total_second_order",
            "actual", "unexplained",
        }
        assert set(attr.keys()) == expected_keys
