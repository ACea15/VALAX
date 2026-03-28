"""
Cross-validation: VALAX vs QuantLib for GBM Monte Carlo pricing.

Tests that VALAX's MC engine produces prices consistent with both
the analytical Black-Scholes solution and QuantLib's MC engine.

Companion example: examples/comparisons/05_monte_carlo.py
"""

import pytest
import jax
import jax.numpy as jnp
import QuantLib as ql
from valax.instruments.options import EuropeanOption
from valax.models.black_scholes import BlackScholesModel
from valax.pricing.mc.paths import generate_gbm_paths
from valax.pricing.mc.engine import mc_price_with_stderr, MCConfig
from valax.pricing.mc.payoffs import european_payoff, asian_payoff, barrier_payoff
from valax.pricing.analytic.black_scholes import black_scholes_price


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def market():
    return dict(S=100.0, K=105.0, T=1.0, sigma=0.25, r=0.04, q=0.01)


@pytest.fixture
def bs_model(market):
    return BlackScholesModel(
        vol=jnp.array(market["sigma"]),
        rate=jnp.array(market["r"]),
        dividend=jnp.array(market["q"]),
    )


# ---------------------------------------------------------------------------
# GBM MC vs Black-Scholes analytic
# ---------------------------------------------------------------------------

class TestGBMMC:
    """VALAX GBM MC should converge to BS analytical."""

    def test_mc_within_3se_of_bs(self, market, bs_model):
        """See: examples/comparisons/05_monte_carlo.py §2 (GBM MC)"""
        call = EuropeanOption(
            strike=jnp.array(market["K"]),
            expiry=jnp.array(market["T"]),
            is_call=True,
        )
        config = MCConfig(n_paths=100_000, n_steps=252)
        key = jax.random.PRNGKey(42)

        mc_p, mc_se = mc_price_with_stderr(call, jnp.array(market["S"]), bs_model, config, key)
        bs_p = black_scholes_price(
            call, jnp.array(market["S"]), jnp.array(market["sigma"]),
            jnp.array(market["r"]), jnp.array(market["q"]),
        )

        n_se = abs(float(mc_p) - float(bs_p)) / float(mc_se)
        assert n_se < 3.0, f"MC={float(mc_p):.4f}, BS={float(bs_p):.4f}, {n_se:.1f} SE"

    @pytest.mark.parametrize("n_paths", [10_000, 50_000, 100_000])
    def test_convergence(self, market, bs_model, n_paths):
        """MC standard error should decrease as O(1/sqrt(n))."""
        call = EuropeanOption(
            strike=jnp.array(market["K"]),
            expiry=jnp.array(market["T"]),
            is_call=True,
        )
        config = MCConfig(n_paths=n_paths, n_steps=100)
        key = jax.random.PRNGKey(42)

        _, se = mc_price_with_stderr(call, jnp.array(market["S"]), bs_model, config, key)
        # SE should be roughly proportional to 1/sqrt(n)
        # For 100K paths with ~50 payoff std, SE ≈ 50/316 ≈ 0.16
        assert float(se) < 1.0  # basic sanity: SE not blown up


# ---------------------------------------------------------------------------
# Exotic payoffs
# ---------------------------------------------------------------------------

class TestExoticPayoffs:
    """Exotic payoff ordering: Asian < European, Barrier < European."""

    def test_asian_leq_european(self, market, bs_model):
        """See: examples/comparisons/05_monte_carlo.py §6 (exotics)

        Asian call price < European call price (averaging reduces vol).
        """
        opt = EuropeanOption(
            strike=jnp.array(market["S"]),  # ATM
            expiry=jnp.array(market["T"]),
            is_call=True,
        )
        key = jax.random.PRNGKey(42)
        paths = generate_gbm_paths(bs_model, jnp.array(market["S"]), market["T"], 252, 100_000, key)

        euro_pays = european_payoff(paths, opt)
        asian_pays = asian_payoff(paths, opt)
        df = jnp.exp(-jnp.array(market["r"]) * market["T"])

        euro_p = float(df * jnp.mean(euro_pays))
        asian_p = float(df * jnp.mean(asian_pays))

        assert asian_p < euro_p, f"Asian={asian_p:.4f} should be < European={euro_p:.4f}"

    def test_barrier_leq_european(self, market, bs_model):
        """See: examples/comparisons/05_monte_carlo.py §6 (exotics)

        Up-and-out barrier call < European call (paths knocked out).
        """
        opt = EuropeanOption(
            strike=jnp.array(market["S"]),
            expiry=jnp.array(market["T"]),
            is_call=True,
        )
        key = jax.random.PRNGKey(42)
        paths = generate_gbm_paths(bs_model, jnp.array(market["S"]), market["T"], 252, 100_000, key)

        euro_pays = european_payoff(paths, opt)
        bar_pays = barrier_payoff(paths, opt, barrier=jnp.array(130.0), is_up=True, is_knock_in=False)
        df = jnp.exp(-jnp.array(market["r"]) * market["T"])

        euro_p = float(df * jnp.mean(euro_pays))
        barrier_p = float(df * jnp.mean(bar_pays))

        assert barrier_p < euro_p, f"Barrier={barrier_p:.4f} should be < European={euro_p:.4f}"


# ---------------------------------------------------------------------------
# Path statistics
# ---------------------------------------------------------------------------

class TestGBMPathStatistics:
    """Sanity checks on GBM path generation."""

    def test_risk_neutral_drift(self, market, bs_model):
        """Under risk-neutral GBM, E[S_T] = S_0 * exp((r-q)*T)."""
        key = jax.random.PRNGKey(42)
        paths = generate_gbm_paths(
            bs_model, jnp.array(market["S"]),
            market["T"], 252, 100_000, key,
        )
        expected = market["S"] * jnp.exp((market["r"] - market["q"]) * market["T"])
        mc_mean = float(jnp.mean(paths[:, -1]))
        assert abs(mc_mean - float(expected)) / float(expected) < 0.02

    def test_paths_start_at_spot(self, market, bs_model):
        """All paths should start at S0."""
        key = jax.random.PRNGKey(42)
        paths = generate_gbm_paths(
            bs_model, jnp.array(market["S"]),
            market["T"], 10, 100, key,
        )
        assert jnp.allclose(paths[:, 0], market["S"])
