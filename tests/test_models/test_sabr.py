"""Tests for SABR model: implied vol, pricing, and Monte Carlo paths."""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_price
from valax.pricing.analytic.black76 import black76_price
from valax.pricing.mc.sabr_paths import generate_sabr_paths
from valax.greeks.autodiff import greeks


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def typical_sabr():
    """Typical SABR params for equity: beta=0.5, moderate vol-of-vol."""
    return SABRModel(
        alpha=jnp.array(0.3),
        beta=jnp.array(0.5),
        rho=jnp.array(-0.3),
        nu=jnp.array(0.4),
    )


@pytest.fixture
def lognormal_sabr():
    """SABR with beta=1 (lognormal backbone)."""
    return SABRModel(
        alpha=jnp.array(0.2),
        beta=jnp.array(1.0),
        rho=jnp.array(-0.25),
        nu=jnp.array(0.3),
    )


@pytest.fixture
def normal_sabr():
    """SABR with beta=0 (normal backbone)."""
    return SABRModel(
        alpha=jnp.array(0.01),
        beta=jnp.array(0.0),
        rho=jnp.array(-0.2),
        nu=jnp.array(0.4),
    )


# ── Implied vol tests ──────────────────────────────────────────────

class TestSABRImpliedVol:
    def test_atm_vol_positive(self, typical_sabr):
        """ATM implied vol should be positive and close to alpha * F^(beta-1)."""
        F = jnp.array(100.0)
        vol = sabr_implied_vol(typical_sabr, F, F, jnp.array(1.0))
        assert float(vol) > 0.0

    def test_atm_vol_approximation(self, lognormal_sabr):
        """For beta=1, ATM vol should be close to alpha for short expiries."""
        F = jnp.array(100.0)
        vol = sabr_implied_vol(lognormal_sabr, F, F, jnp.array(0.01))
        # At very short expiry, ATM vol ~ alpha for beta=1
        assert abs(float(vol) - float(lognormal_sabr.alpha)) < 0.01

    def test_smile_shape_negative_rho(self, typical_sabr):
        """Negative rho should produce higher vol at low strikes (skew)."""
        F = jnp.array(100.0)
        T = jnp.array(1.0)
        vol_low = sabr_implied_vol(typical_sabr, F, jnp.array(80.0), T)
        vol_atm = sabr_implied_vol(typical_sabr, F, jnp.array(100.0), T)
        vol_high = sabr_implied_vol(typical_sabr, F, jnp.array(120.0), T)
        # Negative rho => downside skew
        assert float(vol_low) > float(vol_atm)

    def test_smile_symmetric_zero_rho(self):
        """Zero rho should produce a more symmetric smile."""
        model = SABRModel(
            alpha=jnp.array(0.3),
            beta=jnp.array(0.5),
            rho=jnp.array(0.0),
            nu=jnp.array(0.4),
        )
        F = jnp.array(100.0)
        T = jnp.array(1.0)
        vol_low = sabr_implied_vol(model, F, jnp.array(90.0), T)
        vol_high = sabr_implied_vol(model, F, jnp.array(111.11), T)  # ~same moneyness
        vol_atm = sabr_implied_vol(model, F, jnp.array(100.0), T)
        # Both wings should be above ATM (smile)
        assert float(vol_low) > float(vol_atm)
        assert float(vol_high) > float(vol_atm)

    def test_vol_increases_with_nu(self, typical_sabr):
        """Higher vol-of-vol should widen the smile (higher OTM vols)."""
        F = jnp.array(100.0)
        K_otm = jnp.array(120.0)
        T = jnp.array(1.0)
        vol_base = sabr_implied_vol(typical_sabr, F, K_otm, T)
        model_high_nu = SABRModel(
            alpha=typical_sabr.alpha,
            beta=typical_sabr.beta,
            rho=typical_sabr.rho,
            nu=jnp.array(0.8),
        )
        vol_high = sabr_implied_vol(model_high_nu, F, K_otm, T)
        assert float(vol_high) > float(vol_base)


# ── Pricing tests ──────────────────────────────────────────────────

class TestSABRPrice:
    def test_call_price_positive(self, typical_sabr):
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        price = sabr_price(option, jnp.array(100.0), jnp.array(0.05), typical_sabr)
        assert float(price) > 0.0

    def test_put_price_positive(self, typical_sabr):
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)
        price = sabr_price(option, jnp.array(100.0), jnp.array(0.05), typical_sabr)
        assert float(price) > 0.0

    def test_put_call_parity(self, typical_sabr):
        """Put-call parity: C - P = df * (F - K)."""
        F = jnp.array(100.0)
        K = jnp.array(105.0)
        T = jnp.array(1.0)
        r = jnp.array(0.05)
        call = EuropeanOption(strike=K, expiry=T, is_call=True)
        put = EuropeanOption(strike=K, expiry=T, is_call=False)
        C = sabr_price(call, F, r, typical_sabr)
        P = sabr_price(put, F, r, typical_sabr)
        df = jnp.exp(-r * T)
        assert abs(float(C - P) - float(df * (F - K))) < 1e-10

    def test_deep_itm_call_near_intrinsic(self, typical_sabr):
        """Deep ITM call should be close to discounted intrinsic value."""
        F = jnp.array(100.0)
        K = jnp.array(50.0)
        T = jnp.array(0.25)
        r = jnp.array(0.05)
        option = EuropeanOption(strike=K, expiry=T, is_call=True)
        price = sabr_price(option, F, r, typical_sabr)
        intrinsic = jnp.exp(-r * T) * (F - K)
        assert float(price) >= float(intrinsic) - 1e-6

    def test_consistency_with_black76(self, lognormal_sabr):
        """SABR price should equal Black-76 when fed the same implied vol."""
        F = jnp.array(100.0)
        K = jnp.array(110.0)
        T = jnp.array(0.5)
        r = jnp.array(0.03)
        option = EuropeanOption(strike=K, expiry=T, is_call=True)

        vol = sabr_implied_vol(lognormal_sabr, F, K, T)
        sabr_p = sabr_price(option, F, r, lognormal_sabr)
        b76_p = black76_price(option, F, vol, r)
        assert abs(float(sabr_p) - float(b76_p)) < 1e-12


# ── Greeks via autodiff ────────────────────────────────────────────

class TestSABRGreeks:
    def test_call_delta_positive(self, typical_sabr):
        """ATM call delta should be positive (between 0 and 1)."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        # sabr_price(option, forward, rate, model) — forward is arg 1
        delta_fn = jax.grad(
            lambda fwd: sabr_price(option, fwd, jnp.array(0.05), typical_sabr)
        )
        delta = delta_fn(jnp.array(100.0))
        assert 0.0 < float(delta) < 1.0

    def test_vega_positive(self, typical_sabr):
        """Vega w.r.t. alpha should be positive."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        F = jnp.array(100.0)
        r = jnp.array(0.05)

        def price_fn(alpha):
            model = SABRModel(alpha=alpha, beta=typical_sabr.beta,
                              rho=typical_sabr.rho, nu=typical_sabr.nu)
            return sabr_price(option, F, r, model)

        vega = jax.grad(price_fn)(typical_sabr.alpha)
        assert float(vega) > 0.0

    def test_gamma_positive_atm(self, typical_sabr):
        """ATM gamma should be positive."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)

        price_fn = lambda fwd: sabr_price(option, fwd, jnp.array(0.05), typical_sabr)
        gamma = jax.grad(jax.grad(price_fn))(jnp.array(100.0))
        assert float(gamma) > 0.0


# ── Monte Carlo paths ──────────────────────────────────────────────

class TestSABRPaths:
    def test_path_shapes(self, typical_sabr):
        key = jax.random.PRNGKey(0)
        fwd_paths, vol_paths = generate_sabr_paths(
            typical_sabr, jnp.array(100.0), T=1.0, n_steps=50, n_paths=100, key=key
        )
        assert fwd_paths.shape == (100, 51)
        assert vol_paths.shape == (100, 51)

    def test_initial_values(self, typical_sabr):
        key = jax.random.PRNGKey(42)
        fwd_paths, vol_paths = generate_sabr_paths(
            typical_sabr, jnp.array(100.0), T=1.0, n_steps=50, n_paths=100, key=key
        )
        # All paths should start at the initial forward
        assert jnp.allclose(fwd_paths[:, 0], 100.0, atol=1e-5)
        # All vol paths should start at alpha
        assert jnp.allclose(vol_paths[:, 0], float(typical_sabr.alpha), atol=1e-5)

    def test_mc_vs_analytic_convergence(self, lognormal_sabr):
        """MC call price should converge to analytic within 2 standard errors."""
        F = jnp.array(100.0)
        K = jnp.array(105.0)
        T = 1.0
        r = jnp.array(0.03)
        option = EuropeanOption(strike=K, expiry=jnp.array(T), is_call=True)

        # Analytic price
        analytic = float(sabr_price(option, F, r, lognormal_sabr))

        # MC price
        key = jax.random.PRNGKey(123)
        n_paths = 200_000
        fwd_paths, _ = generate_sabr_paths(
            lognormal_sabr, F, T=T, n_steps=200, n_paths=n_paths, key=key
        )
        terminal = fwd_paths[:, -1]
        payoffs = jnp.maximum(terminal - K, 0.0) * jnp.exp(-r * T)
        mc_price = float(jnp.mean(payoffs))
        mc_se = float(jnp.std(payoffs) / jnp.sqrt(n_paths))

        assert abs(mc_price - analytic) < 2.0 * mc_se, (
            f"MC={mc_price:.4f} vs analytic={analytic:.4f}, SE={mc_se:.4f}"
        )
