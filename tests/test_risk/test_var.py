"""End-to-end tests for VaR and scenario repricing."""

import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.options import EuropeanOption
from valax.market.data import MarketData
from valax.market.scenario import MarketScenario, ScenarioSet
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.risk.scenarios import parametric_scenarios, stress_scenario, stack_scenarios
from valax.risk.var import (
    expected_shortfall,
    portfolio_pnl,
    reprice_under_scenario,
    value_at_risk,
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


class TestRepriceUnderScenario:
    def test_zero_scenario_matches_base(self, simple_portfolio):
        instruments, base = simple_portfolio
        zero = MarketScenario(
            spot_shocks=jnp.zeros(2),
            vol_shocks=jnp.zeros(2),
            rate_shocks=jnp.zeros(3),
            dividend_shocks=jnp.zeros(2),
        )
        val = reprice_under_scenario(black_scholes_price, instruments, base, zero)
        # Should match direct pricing
        p1 = black_scholes_price(
            EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(0.5), is_call=True),
            jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.0),
        )
        p2 = black_scholes_price(
            EuropeanOption(strike=jnp.array(110.0), expiry=jnp.array(1.0), is_call=True),
            jnp.array(100.0), jnp.array(0.25), jnp.array(0.05), jnp.array(0.0),
        )
        assert jnp.isclose(val, p1 + p2, atol=1e-6)

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
        val_up = reprice_under_scenario(black_scholes_price, instruments, base, up)
        val_down = reprice_under_scenario(black_scholes_price, instruments, base, down)
        # Calls should be worth more when spot goes up
        assert val_up > val_down


class TestPortfolioPnl:
    def test_shape(self, simple_portfolio):
        instruments, base = simple_portfolio
        scenarios = stack_scenarios([
            stress_scenario(2, 3, spot_shock=5.0),
            stress_scenario(2, 3, spot_shock=-5.0),
            stress_scenario(2, 3, vol_shock=0.05),
        ])
        pnl = portfolio_pnl(black_scholes_price, instruments, base, scenarios)
        assert pnl.shape == (3,)

    def test_spot_up_positive_pnl(self, simple_portfolio):
        instruments, base = simple_portfolio
        scenarios = stack_scenarios([
            stress_scenario(2, 3, spot_shock=10.0),
        ])
        pnl = portfolio_pnl(black_scholes_price, instruments, base, scenarios)
        # Long calls gain when spot goes up
        assert pnl[0] > 0


class TestVaRAndES:
    def test_var_positive_for_risky_portfolio(self, simple_portfolio):
        instruments, base = simple_portfolio
        key = jax.random.PRNGKey(0)
        n_factors = 2 + 2 + 3 + 2  # spots + vols + pillars + dividends
        cov = jnp.eye(n_factors) * 0.01
        # Make spot variance larger
        cov = cov.at[0, 0].set(100.0)
        cov = cov.at[1, 1].set(100.0)
        scenarios = parametric_scenarios(key, cov, 5000, 2, 3)
        pnl = portfolio_pnl(black_scholes_price, instruments, base, scenarios)
        var_99 = value_at_risk(pnl, confidence=0.99)
        es_99 = expected_shortfall(pnl, confidence=0.99)
        # VaR should be positive (there is risk)
        assert var_99 > 0
        # ES >= VaR by definition
        assert es_99 >= var_99 - 1e-6

    def test_var_increases_with_confidence(self, simple_portfolio):
        instruments, base = simple_portfolio
        key = jax.random.PRNGKey(1)
        n_factors = 9
        cov = jnp.eye(n_factors) * 1.0
        scenarios = parametric_scenarios(key, cov, 10_000, 2, 3)
        pnl = portfolio_pnl(black_scholes_price, instruments, base, scenarios)
        var_95 = value_at_risk(pnl, confidence=0.95)
        var_99 = value_at_risk(pnl, confidence=0.99)
        assert var_99 >= var_95 - 1e-6
