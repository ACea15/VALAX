"""Tests for SVI volatility surface and calibration."""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from valax.surfaces.svi import (
    SVISlice,
    SVIVolSurface,
    svi_total_variance,
    svi_implied_vol,
    calibrate_svi_slice,
)


@pytest.fixture
def svi_params():
    """Typical equity SVI parameters."""
    return SVISlice(
        a=jnp.array(0.04),   # ATM total variance ~ 0.04 (20% vol at T=1)
        b=jnp.array(0.1),
        rho=jnp.array(-0.3),
        m=jnp.array(0.0),
        sigma=jnp.array(0.1),
    )


@pytest.fixture
def svi_surface():
    """Two-expiry SVI surface."""
    return SVIVolSurface(
        expiries=jnp.array([0.5, 1.0]),
        forwards=jnp.array([100.0, 100.0]),
        a_vec=jnp.array([0.02, 0.04]),
        b_vec=jnp.array([0.1, 0.1]),
        rho_vec=jnp.array([-0.3, -0.3]),
        m_vec=jnp.array([0.0, 0.0]),
        sigma_vec=jnp.array([0.1, 0.1]),
    )


class TestSVIFormula:
    def test_atm_total_variance(self, svi_params):
        """At k=0 (ATM), w = a + b * sigma (when m=0)."""
        w = svi_total_variance(svi_params, jnp.array(0.0))
        expected = 0.04 + 0.1 * 0.1  # a + b * sigma
        assert jnp.isclose(w, expected, atol=1e-10)

    def test_symmetric_when_rho_zero(self):
        """With rho=0, smile should be symmetric."""
        params = SVISlice(
            a=jnp.array(0.04), b=jnp.array(0.1),
            rho=jnp.array(0.0), m=jnp.array(0.0), sigma=jnp.array(0.1),
        )
        w_pos = svi_total_variance(params, jnp.array(0.1))
        w_neg = svi_total_variance(params, jnp.array(-0.1))
        assert jnp.isclose(w_pos, w_neg, atol=1e-10)

    def test_skew_with_negative_rho(self, svi_params):
        """Negative rho produces downward skew (higher vol for lower strikes)."""
        w_low = svi_total_variance(svi_params, jnp.array(-0.2))
        w_high = svi_total_variance(svi_params, jnp.array(0.2))
        assert w_low > w_high

    def test_wings_increase(self, svi_params):
        """Total variance increases in both wings."""
        w_atm = svi_total_variance(svi_params, jnp.array(0.0))
        w_far_left = svi_total_variance(svi_params, jnp.array(-1.0))
        w_far_right = svi_total_variance(svi_params, jnp.array(1.0))
        assert w_far_left > w_atm
        assert w_far_right > w_atm

    def test_implied_vol_positive(self, svi_params):
        vol = svi_implied_vol(svi_params, jnp.array(100.0), jnp.array(100.0), jnp.array(1.0))
        assert vol > 0


class TestSVIVolSurface:
    def test_atm_vol_reasonable(self, svi_surface):
        vol = svi_surface(jnp.array(100.0), jnp.array(1.0))
        # With a=0.04, ATM total var ~0.05, vol ~sqrt(0.05/1) ~0.224
        assert 0.15 < float(vol) < 0.30

    def test_total_variance_increases_with_expiry(self, svi_surface):
        """Calendar spread condition: total var should increase with T."""
        vol_05 = svi_surface(jnp.array(100.0), jnp.array(0.5))
        vol_10 = svi_surface(jnp.array(100.0), jnp.array(1.0))
        # Total variance = vol^2 * T
        w_05 = vol_05**2 * 0.5
        w_10 = vol_10**2 * 1.0
        assert w_10 > w_05

    def test_jit_compatible(self, svi_surface):
        f = eqx.filter_jit(svi_surface)
        vol = f(jnp.array(100.0), jnp.array(0.5))
        assert jnp.isfinite(vol)

    def test_differentiable(self, svi_surface):
        grad = jax.grad(lambda k: svi_surface(k, jnp.array(0.5)))(jnp.array(100.0))
        assert jnp.isfinite(grad)

    def test_vmap_over_strikes(self, svi_surface):
        strikes = jnp.linspace(80.0, 120.0, 10)
        vols = jax.vmap(lambda k: svi_surface(k, jnp.array(0.5)))(strikes)
        assert vols.shape == (10,)
        assert jnp.all(jnp.isfinite(vols))


class TestSVICalibration:
    def test_round_trip(self):
        """Calibrate SVI to its own output — should recover parameters."""
        true_params = SVISlice(
            a=jnp.array(0.04), b=jnp.array(0.1),
            rho=jnp.array(-0.3), m=jnp.array(0.0), sigma=jnp.array(0.1),
        )
        forward = jnp.array(100.0)
        expiry = jnp.array(1.0)
        strikes = jnp.linspace(80.0, 120.0, 15)

        # Generate "market" vols from true params
        market_vols = jax.vmap(
            lambda K: svi_implied_vol(true_params, forward, K, expiry)
        )(strikes)

        # Calibrate
        fitted, sol = calibrate_svi_slice(strikes, market_vols, forward, expiry)

        # Check fitted vols match
        fitted_vols = jax.vmap(
            lambda K: svi_implied_vol(fitted, forward, K, expiry)
        )(strikes)
        assert jnp.allclose(fitted_vols, market_vols, atol=1e-4)
