"""Tests for the CMS convexity adjustment (Hagan analytic + replication)."""

import jax
import jax.numpy as jnp
import pytest

from valax.surfaces.constant import ConstantVol
from valax.surfaces.swaption_cube import SwaptionCube
from valax.pricing.analytic.cms_convexity import (
    cms_convexity_adjustment,
    cms_convexity_adjusted_rates,
    _standard_annuity,
    _standard_g,
)


F0 = jnp.array(0.03)
T0 = jnp.array(5.0)
TENOR = 10


def _flat_cube(vol, is_normal=False):
    """A 2x2 SwaptionCube whose SABR smile is (near) flat at ``vol``."""
    exp = jnp.array([1.0, 10.0])
    ten = jnp.array([2.0, 30.0])
    g = lambda v: jnp.full((2, 2), v)
    # beta=1, nu~0 => essentially flat lognormal smile at alpha≈vol.
    return SwaptionCube(
        expiries=exp, tenors=ten, forwards=g(0.03),
        alphas=g(vol), betas=g(1.0), rhos=g(0.0), nus=g(1e-6),
        is_normal=is_normal,
    )


class TestStandardModel:
    """Sanity of the street-standard annuity / G-function."""

    def test_annuity_matches_flat_yield_bond_math(self):
        # A(S) * S = 1 - (1+S)^{-n} for an annual swap (freq=1).
        S = 0.04
        a = float(_standard_annuity(jnp.array(S), TENOR, 1))
        assert a * S == pytest.approx(1.0 - (1.0 + S) ** (-TENOR), rel=1e-10)

    def test_g_is_reciprocal_annuity(self):
        S = jnp.array(0.035)
        g = float(_standard_g(S, TENOR, 1))
        a = float(_standard_annuity(S, TENOR, 1))
        assert g == pytest.approx(float(S) / a, rel=1e-12)


class TestConvexityBasics:
    """Sign, monotonicity, and JIT/vmap behaviour."""

    @pytest.mark.parametrize("method", ["analytic", "replication"])
    def test_adjustment_is_positive(self, method):
        ca = float(cms_convexity_adjustment(F0, T0, TENOR, jnp.array(0.25), method=method))
        assert ca > 0.0

    @pytest.mark.parametrize("method", ["analytic", "replication"])
    def test_monotone_increasing_in_vol(self, method):
        vols = jnp.array([0.10, 0.20, 0.30, 0.40])
        cas = jnp.array([
            cms_convexity_adjustment(F0, T0, TENOR, v, method=method) for v in vols
        ])
        assert jnp.all(jnp.diff(cas) > 0.0)

    @pytest.mark.parametrize("method", ["analytic", "replication"])
    def test_grows_with_expiry(self, method):
        short = float(cms_convexity_adjustment(F0, jnp.array(1.0), TENOR, jnp.array(0.25), method=method))
        long = float(cms_convexity_adjustment(F0, jnp.array(10.0), TENOR, jnp.array(0.25), method=method))
        assert long > short > 0.0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            cms_convexity_adjustment(F0, T0, TENOR, jnp.array(0.2), method="bogus")

    def test_jit_smoke(self):
        f = jax.jit(cms_convexity_adjustment, static_argnums=(2, 4))
        eager = float(cms_convexity_adjustment(F0, T0, TENOR, jnp.array(0.2), "replication"))
        jitted = float(f(F0, T0, TENOR, jnp.array(0.2), "replication"))
        assert jitted == pytest.approx(eager, rel=1e-10)

    def test_vmap_across_forwards(self):
        forwards = jnp.array([0.02, 0.03, 0.04, 0.05])
        expiries = jnp.array([1.0, 2.0, 5.0, 10.0])
        rates = cms_convexity_adjusted_rates(forwards, expiries, TENOR, jnp.array(0.25))
        assert rates.shape == (4,)
        # Adjusted CMS rate exceeds the forward (positive convexity).
        assert jnp.all(rates > forwards)

    def test_differentiable_in_vol(self):
        g = jax.grad(lambda v: cms_convexity_adjustment(F0, T0, TENOR, v, method="analytic"))
        d = float(g(jnp.array(0.25)))
        assert d > 0.0 and jnp.isfinite(d)


class TestAnalyticVsReplication:
    """The two routes must converge as the smile flattens / vol shrinks."""

    def test_converge_in_low_vol_limit(self):
        # As vol -> 0, g'' is locally constant over the (shrinking) support,
        # so replication collapses onto the analytic quadratic formula.
        for vol, tol in [(0.20, 0.15), (0.05, 6e-3), (0.01, 5e-4)]:
            a = float(cms_convexity_adjustment(F0, T0, TENOR, jnp.array(vol), method="analytic"))
            r = float(cms_convexity_adjustment(F0, T0, TENOR, jnp.array(vol), method="replication"))
            assert abs(r / a - 1.0) < tol, f"vol={vol}: analytic={a} repl={r}"

    def test_flat_cube_matches_scalar(self):
        # A (near-)flat SwaptionCube reproduces the scalar-vol adjustment.
        vol = 0.20
        scalar = float(cms_convexity_adjustment(F0, T0, TENOR, jnp.array(vol), method="replication"))
        cube = float(cms_convexity_adjustment(F0, T0, TENOR, _flat_cube(vol), method="replication"))
        assert cube == pytest.approx(scalar, rel=2e-3)


class TestVolSourceEquivalence:
    """Scalar and ConstantVol sources are interchangeable."""

    @pytest.mark.parametrize("method", ["analytic", "replication"])
    def test_scalar_equals_constantvol(self, method):
        s = float(cms_convexity_adjustment(F0, T0, TENOR, jnp.array(0.25), method=method))
        c = float(cms_convexity_adjustment(F0, T0, TENOR, ConstantVol(jnp.array(0.25)), method=method))
        assert s == pytest.approx(c, rel=1e-10)

    def test_skewed_cube_differs_between_methods(self):
        # With a real skew the replication route (integrating the whole smile)
        # departs from the single-vol analytic route — the smile matters.
        skewed = SwaptionCube(
            expiries=jnp.array([1.0, 10.0]), tenors=jnp.array([2.0, 30.0]),
            forwards=jnp.full((2, 2), 0.03), alphas=jnp.full((2, 2), 0.02),
            betas=jnp.full((2, 2), 0.5), rhos=jnp.full((2, 2), -0.4),
            nus=jnp.full((2, 2), 0.5),
        )
        a = float(cms_convexity_adjustment(F0, T0, TENOR, skewed, method="analytic"))
        r = float(cms_convexity_adjustment(F0, T0, TENOR, skewed, method="replication"))
        assert a > 0.0 and r > 0.0
        assert abs(r - a) / a > 0.02
