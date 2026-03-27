"""Tests for GridVolSurface."""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from valax.surfaces.grid import GridVolSurface


@pytest.fixture
def flat_surface():
    """20% flat vol surface."""
    strikes = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0])
    expiries = jnp.array([0.25, 0.5, 1.0, 2.0])
    vols = jnp.full((4, 5), 0.20)
    return GridVolSurface(strikes=strikes, expiries=expiries, vols=vols)


@pytest.fixture
def smile_surface():
    """Surface with a vol smile (higher vol at wings)."""
    strikes = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0])
    expiries = jnp.array([0.5, 1.0])
    # Smile: higher vol at wings
    smile_05 = jnp.array([0.30, 0.24, 0.20, 0.22, 0.28])
    smile_10 = jnp.array([0.28, 0.23, 0.19, 0.21, 0.26])
    vols = jnp.stack([smile_05, smile_10])
    return GridVolSurface(strikes=strikes, expiries=expiries, vols=vols)


class TestGridVolSurface:
    def test_flat_surface_returns_constant(self, flat_surface):
        vol = flat_surface(jnp.array(100.0), jnp.array(0.5))
        assert jnp.isclose(vol, 0.20, atol=1e-10)

    def test_exact_at_grid_point(self, smile_surface):
        # ATM at T=0.5 should be exactly 0.20
        vol = smile_surface(jnp.array(100.0), jnp.array(0.5))
        assert jnp.isclose(vol, 0.20, atol=1e-10)

    def test_exact_at_grid_point_wing(self, smile_surface):
        # 80-strike at T=1.0 should be exactly 0.28
        vol = smile_surface(jnp.array(80.0), jnp.array(1.0))
        assert jnp.isclose(vol, 0.28, atol=1e-10)

    def test_interpolates_between_strikes(self, smile_surface):
        # K=95 at T=0.5: between 0.24 (K=90) and 0.20 (K=100) -> ~0.22
        vol = smile_surface(jnp.array(95.0), jnp.array(0.5))
        assert 0.19 < float(vol) < 0.25

    def test_interpolates_between_expiries(self, smile_surface):
        # ATM at T=0.75: between 0.20 (T=0.5) and 0.19 (T=1.0) -> ~0.195
        vol = smile_surface(jnp.array(100.0), jnp.array(0.75))
        assert 0.18 < float(vol) < 0.21

    def test_flat_extrapolation_strikes(self, smile_surface):
        # Below min strike: should flat extrapolate
        vol_below = smile_surface(jnp.array(60.0), jnp.array(0.5))
        vol_at_min = smile_surface(jnp.array(80.0), jnp.array(0.5))
        assert jnp.isclose(vol_below, vol_at_min, atol=1e-10)

    def test_jit_compatible(self, smile_surface):
        f = eqx.filter_jit(smile_surface)
        vol = f(jnp.array(100.0), jnp.array(0.5))
        assert jnp.isclose(vol, 0.20, atol=1e-10)

    def test_differentiable_wrt_strike(self, smile_surface):
        grad_fn = jax.grad(lambda k: smile_surface(k, jnp.array(0.5)))
        dvol_dk = grad_fn(jnp.array(100.0))
        # Should be finite and non-zero (smile has slope at ATM)
        assert jnp.isfinite(dvol_dk)

    def test_differentiable_wrt_vol_grid(self, smile_surface):
        """Gradient of output vol w.r.t. the grid vols (for vega bucketing)."""
        def f(vols_flat):
            surface = GridVolSurface(
                strikes=smile_surface.strikes,
                expiries=smile_surface.expiries,
                vols=vols_flat.reshape(smile_surface.vols.shape),
            )
            return surface(jnp.array(100.0), jnp.array(0.5))

        grad = jax.grad(f)(smile_surface.vols.flatten())
        # At least one grid point should have non-zero sensitivity
        assert jnp.any(grad != 0)

    def test_vmap_over_strikes(self, smile_surface):
        strikes = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = jax.vmap(lambda k: smile_surface(k, jnp.array(0.5)))(strikes)
        assert vols.shape == (5,)
        # ATM should be the minimum (smile shape)
        assert jnp.argmin(vols) == 2
