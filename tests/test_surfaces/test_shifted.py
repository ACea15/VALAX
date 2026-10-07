"""Tests for the parallel-shift vol-surface decorator."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.surfaces.constant import ConstantVol
from valax.surfaces.grid import GridVolSurface
from valax.surfaces.sabr_surface import SABRVolSurface
from valax.surfaces.shifted import ShiftedVolSurface
from valax.surfaces.svi import SVIVolSurface


@pytest.fixture
def grid():
    """A mildly skewed 2x3 grid surface."""
    return GridVolSurface(
        strikes=jnp.array([90.0, 100.0, 110.0]),
        expiries=jnp.array([0.5, 1.0]),
        vols=jnp.array([[0.25, 0.20, 0.18], [0.24, 0.21, 0.19]]),
    )


@pytest.fixture
def svi():
    return SVIVolSurface(
        expiries=jnp.array([0.5, 1.0]),
        forwards=jnp.array([100.0, 100.0]),
        a_vec=jnp.array([0.01, 0.02]),
        b_vec=jnp.array([0.10, 0.10]),
        rho_vec=jnp.array([-0.3, -0.3]),
        m_vec=jnp.array([0.0, 0.0]),
        sigma_vec=jnp.array([0.2, 0.2]),
    )


@pytest.fixture
def sabr():
    return SABRVolSurface(
        expiries=jnp.array([0.5, 1.0]),
        forwards=jnp.array([100.0, 100.0]),
        alphas=jnp.array([0.2, 0.22]),
        betas=jnp.array([0.5, 0.5]),
        rhos=jnp.array([-0.3, -0.3]),
        nus=jnp.array([0.4, 0.4]),
    )


class TestParallelShift:
    @pytest.mark.parametrize("surface_name", ["grid", "svi", "sabr"])
    def test_call_is_base_plus_shift(self, surface_name, request):
        """Uniform across surface types: one number moves every quote."""
        surface = request.getfixturevalue(surface_name)
        shifted = ShiftedVolSurface(base=surface, shift=jnp.array(0.01))
        for strike in (90.0, 100.0, 110.0):
            for expiry in (0.5, 0.75, 1.0):
                k, t = jnp.array(strike), jnp.array(expiry)
                assert jnp.isclose(
                    shifted(k, t), surface(k, t) + 0.01, atol=1e-12,
                )

    def test_zero_shift_is_identity(self, grid):
        shifted = ShiftedVolSurface(base=grid, shift=jnp.array(0.0))
        k, t = jnp.array(100.0), jnp.array(1.0)
        assert shifted(k, t) == grid(k, t)

    def test_shape_of_the_smile_is_preserved(self, grid):
        """A parallel shift moves the level, never the skew."""
        shifted = ShiftedVolSurface(base=grid, shift=jnp.array(0.03))
        t = jnp.array(1.0)
        base_skew = grid(jnp.array(90.0), t) - grid(jnp.array(110.0), t)
        new_skew = shifted(jnp.array(90.0), t) - shifted(jnp.array(110.0), t)
        assert jnp.isclose(base_skew, new_skew, atol=1e-12)

    def test_constant_vol_can_be_wrapped(self):
        flat = ConstantVol(vol=jnp.array(0.2))
        shifted = ShiftedVolSurface(base=flat, shift=jnp.array(0.05))
        assert jnp.isclose(shifted(jnp.array(100.0), jnp.array(1.0)), 0.25)

    def test_is_normal_is_inherited(self):
        normal = ConstantVol(vol=jnp.array(0.01), is_normal=True)
        assert ShiftedVolSurface(base=normal, shift=jnp.array(0.0)).is_normal
        lognormal = ConstantVol(vol=jnp.array(0.2))
        assert not ShiftedVolSurface(
            base=lognormal, shift=jnp.array(0.0),
        ).is_normal

    def test_is_normal_defaults_false(self, grid):
        """A surface that declares no convention is lognormal."""
        assert not ShiftedVolSurface(base=grid, shift=jnp.array(0.0)).is_normal


class TestTotalVariance:
    @pytest.mark.parametrize("surface_name", ["svi", "sabr", "grid"])
    def test_consistent_with_call(self, surface_name, request):
        """``total_variance`` and ``__call__`` describe the same surface."""
        surface = request.getfixturevalue(surface_name)
        shifted = ShiftedVolSurface(base=surface, shift=jnp.array(0.02))
        expiry = jnp.array(1.0)
        # The base's own consistency convention fixes the coordinate map:
        # compare the shifted vol implied by each entry point.
        k = jnp.array(0.0) if surface_name != "grid" else jnp.array(100.0)
        w_base = surface.total_variance(k, expiry)
        w_shifted = shifted.total_variance(k, expiry)
        sigma_base = jnp.sqrt(w_base / expiry)
        sigma_shifted = jnp.sqrt(w_shifted / expiry)
        assert jnp.isclose(sigma_shifted, sigma_base + 0.02, atol=1e-10)

    def test_zero_shift_is_identity(self, svi):
        shifted = ShiftedVolSurface(base=svi, shift=jnp.array(0.0))
        k, t = jnp.array(0.1), jnp.array(0.75)
        assert jnp.isclose(
            shifted.total_variance(k, t), svi.total_variance(k, t), atol=1e-12,
        )

    def test_zero_expiry_does_not_divide_by_zero(self, svi):
        shifted = ShiftedVolSurface(base=svi, shift=jnp.array(0.01))
        w = shifted.total_variance(jnp.array(0.0), jnp.array(0.0))
        assert jnp.isfinite(w)

    def test_raises_when_base_has_no_total_variance(self):
        shifted = ShiftedVolSurface(
            base=ConstantVol(vol=jnp.array(0.2)), shift=jnp.array(0.01),
        )
        with pytest.raises(AttributeError):
            shifted.total_variance(jnp.array(0.0), jnp.array(1.0))


class TestDifferentiability:
    def test_grad_wrt_shift(self, grid):
        """d(sigma)/d(shift) == 1 — the shift is a first-class risk factor."""
        def f(shift):
            return ShiftedVolSurface(base=grid, shift=shift)(
                jnp.array(100.0), jnp.array(1.0),
            )

        assert jnp.isclose(jax.grad(f)(jnp.array(0.0)), 1.0, atol=1e-12)

    def test_grad_flows_to_base_nodes(self, grid):
        """The wrapper does not block node-level vega on the base."""
        def f(vols):
            surface = eqx.tree_at(lambda s: s.vols, grid, vols)
            return ShiftedVolSurface(base=surface, shift=jnp.array(0.01))(
                jnp.array(100.0), jnp.array(1.0),
            )

        g = jax.grad(f)(grid.vols)
        assert jnp.any(g != 0.0)

    def test_filter_jit_smoke(self, sabr):
        @eqx.filter_jit
        def f(surface, shift):
            return ShiftedVolSurface(base=surface, shift=shift)(
                jnp.array(100.0), jnp.array(1.0),
            )

        assert jnp.isfinite(f(sabr, jnp.array(0.01)))
