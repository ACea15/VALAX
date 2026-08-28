"""Tests for the SABR OptionletVolSurface (caplet/floorlet, expiry x strike).

Covers parameter interpolation, lognormal and normal calibration round-trips,
the normal-vs-lognormal quoting flag, negative-strike capability, and a jit
smoke test.
"""

import functools

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_normal_implied_vol
from valax.surfaces import OptionletVolSurface, build_sabr_caplet_surface


EXPIRIES = jnp.array([0.5, 1.0, 2.0, 5.0])


def _surface(alphas, nus, forward=0.03, beta=0.5, rho=-0.3):
    return OptionletVolSurface(
        expiries=EXPIRIES,
        forwards=jnp.full(EXPIRIES.shape, forward),
        alphas=alphas,
        betas=jnp.full(EXPIRIES.shape, beta),
        rhos=jnp.full(EXPIRIES.shape, rho),
        nus=nus,
    )


class TestInterpolation:
    def test_node_values_exact(self):
        alphas = jnp.array([0.30, 0.32, 0.34, 0.36])
        nus = jnp.array([0.40, 0.42, 0.44, 0.46])
        surf = _surface(alphas, nus)
        m = surf.model_at(EXPIRIES[2])
        assert abs(float(m.alpha) - 0.34) < 1e-12
        assert abs(float(m.nu) - 0.44) < 1e-12
        assert abs(float(surf.forward_at(EXPIRIES[1])) - 0.03) < 1e-12

    def test_midpoint_is_linear_average(self):
        alphas = jnp.array([0.30, 0.50, 0.70, 0.90])
        nus = jnp.full(4, 0.4)
        surf = _surface(alphas, nus)
        # Halfway between expiries[1]=1.0 and [2]=2.0 => 1.5 => (0.50+0.70)/2.
        m = surf.model_at(jnp.array(1.5))
        assert abs(float(m.alpha) - 0.60) < 1e-12

    def test_matches_direct_hagan_at_node(self):
        alphas = jnp.array([0.30, 0.32, 0.34, 0.36])
        surf = _surface(alphas, jnp.full(4, 0.4))
        K = jnp.array(0.028)
        got = surf(K, EXPIRIES[2])
        model = surf.model_at(EXPIRIES[2])
        want = sabr_implied_vol(model, jnp.array(0.03), K, EXPIRIES[2])
        assert abs(float(got - want)) < 1e-12


class TestLognormalStripping:
    def test_round_trip(self):
        forward = 0.03
        strikes = jnp.array([0.02, 0.025, 0.03, 0.035, 0.04])

        def truth(i):
            return SABRModel(alpha=jnp.array(0.20 + 0.01 * i), beta=jnp.array(0.5),
                             rho=jnp.array(-0.25), nu=jnp.array(0.35))

        strikes_per_expiry = [strikes for _ in range(4)]
        market = [
            jax.vmap(lambda K: sabr_implied_vol(truth(i), jnp.array(forward), K, EXPIRIES[i]))(strikes)
            for i in range(4)
        ]
        forwards = jnp.full(4, forward)

        surf = build_sabr_caplet_surface(
            strikes_per_expiry, market, forwards, EXPIRIES, fixed_beta=jnp.array(0.5),
        )
        max_err = max(
            abs(float(surf(strikes[k], EXPIRIES[i]) - market[i][k]))
            for i in range(4) for k in range(strikes.shape[0])
        )
        assert max_err < 1e-3, f"max vol error {max_err}"


class TestNormalStripping:
    def test_round_trip(self):
        forward = 0.02
        shift = 0.03
        strikes = jnp.array([0.01, 0.015, 0.02, 0.025, 0.03])
        vol_fn = functools.partial(sabr_normal_implied_vol, shift=jnp.asarray(shift))

        def truth(i):
            return SABRModel(alpha=jnp.array(0.009 + 0.001 * i), beta=jnp.array(0.0),
                             rho=jnp.array(-0.2), nu=jnp.array(0.3))

        strikes_per_expiry = [strikes for _ in range(4)]
        market = [
            jax.vmap(lambda K: vol_fn(truth(i), jnp.array(forward), K, EXPIRIES[i]))(strikes)
            for i in range(4)
        ]
        forwards = jnp.full(4, forward)

        surf = build_sabr_caplet_surface(
            strikes_per_expiry, market, forwards, EXPIRIES,
            is_normal=True, shift=shift, fixed_beta=jnp.array(0.0),
        )
        assert surf.is_normal is True
        max_err = max(
            abs(float(surf(strikes[k], EXPIRIES[i]) - market[i][k]))
            for i in range(4) for k in range(strikes.shape[0])
        )
        assert max_err < 1e-6, f"max vol error {max_err}"


class TestQuotingConvention:
    def test_lognormal_nan_at_negative_strike(self):
        surf = _surface(jnp.full(4, 0.3), jnp.full(4, 0.4))
        assert surf.is_normal is False
        assert jnp.isnan(surf(jnp.array(-0.01), EXPIRIES[1]))

    def test_normal_finite_at_negative_strike(self):
        surf = OptionletVolSurface(
            expiries=EXPIRIES,
            forwards=jnp.full(4, 0.01),
            alphas=jnp.full(4, 0.01),
            betas=jnp.full(4, 0.0),
            rhos=jnp.full(4, -0.2),
            nus=jnp.full(4, 0.3),
            shift=jnp.array(0.03),
            is_normal=True,
        )
        vol = surf(jnp.array(-0.005), EXPIRIES[2])
        assert jnp.isfinite(vol) and float(vol) > 0.0


class TestJit:
    def test_filter_jit_smoke(self):
        surf = _surface(jnp.full(4, 0.3), jnp.full(4, 0.4))
        vol = eqx.filter_jit(surf.__call__)(jnp.array(0.03), EXPIRIES[1])
        assert jnp.isfinite(vol)
