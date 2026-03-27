"""Tests for SABRVolSurface."""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol
from valax.surfaces.sabr_surface import SABRVolSurface


@pytest.fixture
def sabr_surface():
    """Two-expiry SABR surface with known parameters."""
    return SABRVolSurface(
        expiries=jnp.array([0.5, 1.0]),
        forwards=jnp.array([100.0, 100.0]),
        alphas=jnp.array([0.3, 0.25]),
        betas=jnp.array([0.5, 0.5]),
        rhos=jnp.array([-0.2, -0.3]),
        nus=jnp.array([0.4, 0.35]),
    )


class TestSABRVolSurface:
    def test_matches_direct_sabr_at_slice(self, sabr_surface):
        """At an exact expiry, should match direct sabr_implied_vol."""
        model = SABRModel(
            alpha=jnp.array(0.3),
            beta=jnp.array(0.5),
            rho=jnp.array(-0.2),
            nu=jnp.array(0.4),
        )
        expected = sabr_implied_vol(model, jnp.array(100.0), jnp.array(95.0), jnp.array(0.5))
        actual = sabr_surface(jnp.array(95.0), jnp.array(0.5))
        assert jnp.isclose(actual, expected, atol=1e-8)

    def test_matches_second_slice(self, sabr_surface):
        model = SABRModel(
            alpha=jnp.array(0.25),
            beta=jnp.array(0.5),
            rho=jnp.array(-0.3),
            nu=jnp.array(0.35),
        )
        expected = sabr_implied_vol(model, jnp.array(100.0), jnp.array(110.0), jnp.array(1.0))
        actual = sabr_surface(jnp.array(110.0), jnp.array(1.0))
        assert jnp.isclose(actual, expected, atol=1e-8)

    def test_interpolation_between_expiries(self, sabr_surface):
        """At T=0.75, should return something between the two slices."""
        vol_05 = sabr_surface(jnp.array(100.0), jnp.array(0.5))
        vol_10 = sabr_surface(jnp.array(100.0), jnp.array(1.0))
        vol_075 = sabr_surface(jnp.array(100.0), jnp.array(0.75))
        lo = jnp.minimum(vol_05, vol_10)
        hi = jnp.maximum(vol_05, vol_10)
        assert lo <= vol_075 + 1e-6
        assert vol_075 <= hi + 1e-6

    def test_jit_compatible(self, sabr_surface):
        f = eqx.filter_jit(sabr_surface)
        vol = f(jnp.array(100.0), jnp.array(0.5))
        assert jnp.isfinite(vol)

    def test_differentiable_wrt_strike(self, sabr_surface):
        grad = jax.grad(lambda k: sabr_surface(k, jnp.array(0.5)))(jnp.array(100.0))
        assert jnp.isfinite(grad)

    def test_differentiable_wrt_alphas(self, sabr_surface):
        """Gradient of vol w.r.t. SABR alpha parameters."""
        def f(alphas):
            surface = SABRVolSurface(
                expiries=sabr_surface.expiries,
                forwards=sabr_surface.forwards,
                alphas=alphas,
                betas=sabr_surface.betas,
                rhos=sabr_surface.rhos,
                nus=sabr_surface.nus,
            )
            return surface(jnp.array(100.0), jnp.array(0.5))

        grad = jax.grad(f)(sabr_surface.alphas)
        assert jnp.all(jnp.isfinite(grad))
        # Alpha at the queried expiry should have highest sensitivity
        assert jnp.abs(grad[0]) > 0

    def test_vmap_over_strikes(self, sabr_surface):
        strikes = jnp.linspace(80.0, 120.0, 10)
        vols = jax.vmap(lambda k: sabr_surface(k, jnp.array(0.5)))(strikes)
        assert vols.shape == (10,)
        assert jnp.all(jnp.isfinite(vols))
        # Negative rho should produce downward-sloping skew (lower strikes -> higher vol)
        assert vols[0] > vols[-1]
