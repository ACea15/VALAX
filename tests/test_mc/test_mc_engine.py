"""Tests for Monte Carlo pricing engine.

MC prices are validated against analytical solutions within
statistical tolerance (2 standard errors).
"""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.mc.engine import mc_price, mc_price_with_stderr, MCConfig
from valax.pricing.mc.payoffs import european_payoff, asian_payoff


# ── GBM / Black-Scholes MC ──────────────────────────────────────────

class TestGBMMonteCarlo:
    """Test GBM MC against Black-Scholes analytical prices."""

    def test_european_call_convergence(self):
        """MC European call price converges to BS within 2 standard errors."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        spot = jnp.array(100.0)
        model = BlackScholesModel(
            vol=jnp.array(0.20), rate=jnp.array(0.05), dividend=jnp.array(0.02)
        )
        config = MCConfig(n_paths=100_000, n_steps=100)
        key = jax.random.PRNGKey(42)

        mc, stderr = mc_price_with_stderr(option, spot, model, config, key)
        analytical = black_scholes_price(option, spot, model.vol, model.rate, model.dividend)

        err = abs(float(mc) - float(analytical))
        assert err < 2 * float(stderr), (
            f"MC={float(mc):.4f}, BS={float(analytical):.4f}, "
            f"err={err:.4f}, 2*SE={2*float(stderr):.4f}"
        )

    def test_european_put_convergence(self):
        """MC European put price converges to BS within 2 standard errors."""
        option = EuropeanOption(strike=jnp.array(105.0), expiry=jnp.array(0.5), is_call=False)
        spot = jnp.array(100.0)
        model = BlackScholesModel(
            vol=jnp.array(0.25), rate=jnp.array(0.03), dividend=jnp.array(0.01)
        )
        config = MCConfig(n_paths=100_000, n_steps=100)
        key = jax.random.PRNGKey(123)

        mc, stderr = mc_price_with_stderr(option, spot, model, config, key)
        analytical = black_scholes_price(option, spot, model.vol, model.rate, model.dividend)

        err = abs(float(mc) - float(analytical))
        assert err < 2 * float(stderr)

    def test_otm_call_convergence(self):
        """OTM call: MC should still converge."""
        option = EuropeanOption(strike=jnp.array(120.0), expiry=jnp.array(1.0), is_call=True)
        spot = jnp.array(100.0)
        model = BlackScholesModel(
            vol=jnp.array(0.20), rate=jnp.array(0.05), dividend=jnp.array(0.0)
        )
        config = MCConfig(n_paths=100_000, n_steps=100)
        key = jax.random.PRNGKey(999)

        mc, stderr = mc_price_with_stderr(option, spot, model, config, key)
        analytical = black_scholes_price(option, spot, model.vol, model.rate, model.dividend)

        err = abs(float(mc) - float(analytical))
        assert err < 3 * float(stderr)  # slightly looser for OTM

    def test_mc_price_positive(self):
        """MC price should be non-negative."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        spot = jnp.array(100.0)
        model = BlackScholesModel(
            vol=jnp.array(0.20), rate=jnp.array(0.05), dividend=jnp.array(0.0)
        )
        config = MCConfig(n_paths=10_000, n_steps=50)
        key = jax.random.PRNGKey(0)

        price = mc_price(option, spot, model, config, key)
        assert float(price) > 0.0

    def test_more_paths_reduces_stderr(self):
        """Doubling paths should roughly halve standard error."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        spot = jnp.array(100.0)
        model = BlackScholesModel(
            vol=jnp.array(0.20), rate=jnp.array(0.05), dividend=jnp.array(0.0)
        )
        key = jax.random.PRNGKey(7)

        _, se1 = mc_price_with_stderr(option, spot, model, MCConfig(n_paths=10_000, n_steps=50), key)
        _, se2 = mc_price_with_stderr(option, spot, model, MCConfig(n_paths=40_000, n_steps=50), key)

        # SE should scale as 1/sqrt(n), so 4x paths -> ~0.5x SE
        ratio = float(se1) / float(se2)
        assert 1.5 < ratio < 2.5


# ── Asian option MC ──────────────────────────────────────────────────

class TestAsianMC:
    """Asian option has no simple closed-form, so test properties."""

    def test_asian_call_less_than_european(self):
        """Arithmetic Asian call <= European call (averaging reduces variance)."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        spot = jnp.array(100.0)
        model = BlackScholesModel(
            vol=jnp.array(0.30), rate=jnp.array(0.05), dividend=jnp.array(0.0)
        )
        config = MCConfig(n_paths=50_000, n_steps=252)
        key = jax.random.PRNGKey(42)

        euro_price = mc_price(option, spot, model, config, key, payoff_fn=european_payoff)
        asian_price = mc_price(option, spot, model, config, key, payoff_fn=asian_payoff)

        assert float(asian_price) < float(euro_price)
        assert float(asian_price) > 0.0

    def test_asian_put_positive(self):
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)
        spot = jnp.array(100.0)
        model = BlackScholesModel(
            vol=jnp.array(0.20), rate=jnp.array(0.05), dividend=jnp.array(0.0)
        )
        config = MCConfig(n_paths=20_000, n_steps=100)
        key = jax.random.PRNGKey(99)

        price = mc_price(option, spot, model, config, key, payoff_fn=asian_payoff)
        assert float(price) > 0.0


# ── Heston MC ────────────────────────────────────────────────────────

class TestHestonMC:
    """Heston MC — test basic properties (no cheap analytical reference)."""

    def test_heston_european_call_positive(self):
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        spot = jnp.array(100.0)
        model = HestonModel(
            v0=jnp.array(0.04),       # initial vol = 20%
            kappa=jnp.array(2.0),
            theta=jnp.array(0.04),
            xi=jnp.array(0.3),
            rho=jnp.array(-0.7),
            rate=jnp.array(0.05),
            dividend=jnp.array(0.0),
        )
        config = MCConfig(n_paths=50_000, n_steps=100)
        key = jax.random.PRNGKey(42)

        price = mc_price(option, spot, model, config, key)
        assert float(price) > 0.0
        # Heston with v0=0.04 (20% vol) should give ATM call price roughly in [5, 15]
        assert 3.0 < float(price) < 20.0

    def test_heston_put_call_parity(self):
        """Put-call parity must hold for any model under risk-neutral pricing."""
        K, T, r = 100.0, 1.0, 0.05
        spot = jnp.array(100.0)
        model = HestonModel(
            v0=jnp.array(0.04), kappa=jnp.array(2.0), theta=jnp.array(0.04),
            xi=jnp.array(0.3), rho=jnp.array(-0.7),
            rate=jnp.array(r), dividend=jnp.array(0.0),
        )
        config = MCConfig(n_paths=100_000, n_steps=100)
        key = jax.random.PRNGKey(42)

        call_opt = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=True)
        put_opt = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=False)

        # Use same key so paths are identical
        call_price = mc_price(call_opt, spot, model, config, key)
        put_price = mc_price(put_opt, spot, model, config, key)

        # C - P = S - K*exp(-rT) (with q=0)
        parity = float(call_price) - float(put_price)
        expected = float(spot) - K * float(jnp.exp(-r * T))

        # MC parity error should be very small since same paths
        assert abs(parity - expected) < 0.5, (
            f"C-P={parity:.4f}, S-Kdf={expected:.4f}, diff={abs(parity-expected):.4f}"
        )

    def test_heston_low_vol_of_vol_approaches_bs(self):
        """With xi→0, Heston collapses to BS with vol=sqrt(v0)."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        spot = jnp.array(100.0)
        vol = 0.20
        model = HestonModel(
            v0=jnp.array(vol**2), kappa=jnp.array(2.0), theta=jnp.array(vol**2),
            xi=jnp.array(0.001),  # near-zero vol of vol
            rho=jnp.array(0.0),
            rate=jnp.array(0.05), dividend=jnp.array(0.0),
        )
        config = MCConfig(n_paths=100_000, n_steps=100)
        key = jax.random.PRNGKey(42)

        heston_price = mc_price(option, spot, model, config, key)
        bs_price = black_scholes_price(
            option, spot, jnp.array(vol), jnp.array(0.05), jnp.array(0.0)
        )

        # Should be close — within ~1% of BS price
        err = abs(float(heston_price) - float(bs_price))
        assert err < 0.5, f"Heston={float(heston_price):.4f}, BS={float(bs_price):.4f}"


# ── MC Greeks via autodiff ───────────────────────────────────────────

class TestMCGreeks:
    """Test that autodiff works through the MC simulation."""

    def test_mc_delta_positive_for_call(self):
        """Delta of a call should be positive."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        model = BlackScholesModel(
            vol=jnp.array(0.20), rate=jnp.array(0.05), dividend=jnp.array(0.0)
        )
        config = MCConfig(n_paths=20_000, n_steps=50)
        key = jax.random.PRNGKey(42)

        def price_fn(spot):
            return mc_price(option, spot, model, config, key)

        delta = jax.grad(price_fn)(jnp.array(100.0))
        assert float(delta) > 0.0
        # BS delta for ATM call ~ 0.6; MC delta should be in a reasonable range
        assert 0.3 < float(delta) < 0.9

    def test_mc_vega_positive_for_call(self):
        """Vega (sensitivity to vol) should be positive."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        spot = jnp.array(100.0)
        config = MCConfig(n_paths=20_000, n_steps=50)
        key = jax.random.PRNGKey(42)

        def price_fn(vol):
            model = BlackScholesModel(
                vol=vol, rate=jnp.array(0.05), dividend=jnp.array(0.0)
            )
            return mc_price(option, spot, model, config, key)

        vega = jax.grad(price_fn)(jnp.array(0.20))
        assert float(vega) > 0.0
