"""Tests for the SABR SwaptionCube (expiry x tenor x strike).

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
from valax.surfaces import SwaptionCube, calibrate_swaption_cube


EXPIRIES = jnp.array([1.0, 2.0, 5.0])
TENORS = jnp.array([2.0, 10.0])


def _const_grids(alpha_grid, nu_grid, forward=0.03, beta=0.5, rho=-0.3):
    """Build a SwaptionCube with given alpha/nu grids and constant beta/rho/fwd."""
    shape = alpha_grid.shape
    return SwaptionCube(
        expiries=EXPIRIES,
        tenors=TENORS,
        forwards=jnp.full(shape, forward),
        alphas=alpha_grid,
        betas=jnp.full(shape, beta),
        rhos=jnp.full(shape, rho),
        nus=nu_grid,
    )


class TestInterpolation:
    def test_node_values_exact(self):
        alphas = jnp.array([[0.30, 0.31], [0.32, 0.33], [0.35, 0.36]])
        nus = jnp.array([[0.40, 0.41], [0.42, 0.43], [0.45, 0.46]])
        cube = _const_grids(alphas, nus)
        # At an exact grid node the interpolation returns the node parameters.
        m = cube.model_at(EXPIRIES[1], TENORS[1])
        assert abs(float(m.alpha) - 0.33) < 1e-12
        assert abs(float(m.nu) - 0.43) < 1e-12
        assert abs(float(cube.forward_at(EXPIRIES[2], TENORS[0])) - 0.03) < 1e-12

    def test_midpoint_is_bilinear_average(self):
        alphas = jnp.array([[0.30, 0.40], [0.50, 0.60], [0.90, 1.00]])
        nus = jnp.full((3, 2), 0.4)
        cube = _const_grids(alphas, nus)
        # Halfway in expiry (1 <-> 2 => 1.5) and tenor (2 <-> 10 => 6.0):
        # average of the four surrounding nodes 0.30,0.40,0.50,0.60 = 0.45.
        m = cube.model_at(jnp.array(1.5), jnp.array(6.0))
        assert abs(float(m.alpha) - 0.45) < 1e-12

    def test_flat_extrapolation(self):
        alphas = jnp.array([[0.30, 0.31], [0.32, 0.33], [0.35, 0.36]])
        cube = _const_grids(alphas, jnp.full((3, 2), 0.4))
        # Query below the grid clamps to the first node.
        m = cube.model_at(jnp.array(0.1), jnp.array(0.5))
        assert abs(float(m.alpha) - 0.30) < 1e-12


class TestLognormalCalibration:
    def test_node_round_trip(self):
        """Fit per-node SABR smiles generated from known models; recover vols."""
        forward = 100.0
        strikes = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0])

        def truth(i, j):
            return SABRModel(
                alpha=jnp.array(0.28 + 0.01 * i + 0.02 * j),
                beta=jnp.array(0.5),
                rho=jnp.array(-0.3),
                nu=jnp.array(0.35 + 0.05 * j),
            )

        strikes_per_node = [[strikes for _ in range(2)] for _ in range(3)]
        market = [
            [
                jax.vmap(lambda K: sabr_implied_vol(truth(i, j), jnp.array(forward), K, EXPIRIES[i]))(strikes)
                for j in range(2)
            ]
            for i in range(3)
        ]
        forwards = jnp.full((3, 2), forward)

        cube = calibrate_swaption_cube(
            strikes_per_node, market, forwards, EXPIRIES, TENORS,
            fixed_beta=jnp.array(0.5),
        )

        max_err = 0.0
        for i in range(3):
            for j in range(2):
                for k in range(strikes.shape[0]):
                    got = cube(strikes[k], EXPIRIES[i], TENORS[j])
                    max_err = max(max_err, abs(float(got - market[i][j][k])))
        assert max_err < 1e-3, f"max node vol error {max_err}"


class TestNormalCalibration:
    def test_node_round_trip(self):
        """Fit normal-quoted (beta=0, shifted) smiles and recover them.

        Guards against the historical vol_fn plumbing bug where normal quotes
        were silently fit with the lognormal formula (see
        examples/pitfalls/01_normal_sabr_calibration_divergence.py).
        """
        forward = 0.025
        shift = 0.03
        strikes = jnp.array([0.015, 0.02, 0.025, 0.03, 0.035])
        vol_fn = functools.partial(sabr_normal_implied_vol, shift=jnp.asarray(shift))

        def truth(i, j):
            return SABRModel(
                alpha=jnp.array(0.009 + 0.001 * i + 0.001 * j),
                beta=jnp.array(0.0),
                rho=jnp.array(-0.2),
                nu=jnp.array(0.3 + 0.05 * j),
            )

        strikes_per_node = [[strikes for _ in range(2)] for _ in range(3)]
        market = [
            [
                jax.vmap(lambda K: vol_fn(truth(i, j), jnp.array(forward), K, EXPIRIES[i]))(strikes)
                for j in range(2)
            ]
            for i in range(3)
        ]
        forwards = jnp.full((3, 2), forward)

        cube = calibrate_swaption_cube(
            strikes_per_node, market, forwards, EXPIRIES, TENORS,
            is_normal=True, shift=shift, fixed_beta=jnp.array(0.0),
        )
        assert cube.is_normal is True

        max_err = 0.0
        for i in range(3):
            for j in range(2):
                for k in range(strikes.shape[0]):
                    got = cube(strikes[k], EXPIRIES[i], TENORS[j])
                    max_err = max(max_err, abs(float(got - market[i][j][k])))
        assert max_err < 1e-6, f"max node vol error {max_err}"


class TestQuotingConvention:
    def test_lognormal_cube_nan_at_negative_strike(self):
        cube = _const_grids(jnp.full((3, 2), 0.3), jnp.full((3, 2), 0.4))
        assert cube.is_normal is False
        vol = cube(jnp.array(-5.0), EXPIRIES[1], TENORS[0])
        assert jnp.isnan(vol)

    def test_normal_cube_matches_direct_hagan_at_node(self):
        model = SABRModel(alpha=jnp.array(0.01), beta=jnp.array(0.0),
                          rho=jnp.array(-0.2), nu=jnp.array(0.3))
        cube = SwaptionCube(
            expiries=EXPIRIES, tenors=TENORS,
            forwards=jnp.full((3, 2), 0.025),
            alphas=jnp.full((3, 2), 0.01),
            betas=jnp.full((3, 2), 0.0),
            rhos=jnp.full((3, 2), -0.2),
            nus=jnp.full((3, 2), 0.3),
            shift=jnp.array(0.03),
            is_normal=True,
        )
        K = jnp.array(0.02)
        got = cube(K, EXPIRIES[1], TENORS[0])
        want = sabr_normal_implied_vol(model, jnp.array(0.025), K, EXPIRIES[1], jnp.array(0.03))
        assert abs(float(got - want)) < 1e-12

    def test_normal_cube_finite_at_negative_strike(self):
        cube = SwaptionCube(
            expiries=EXPIRIES, tenors=TENORS,
            forwards=jnp.full((3, 2), 0.01),
            alphas=jnp.full((3, 2), 0.01),
            betas=jnp.full((3, 2), 0.0),
            rhos=jnp.full((3, 2), -0.2),
            nus=jnp.full((3, 2), 0.3),
            shift=jnp.array(0.03),
            is_normal=True,
        )
        vol = cube(jnp.array(-0.005), EXPIRIES[1], TENORS[1])
        assert jnp.isfinite(vol)
        assert float(vol) > 0.0


class TestJit:
    def test_filter_jit_smoke(self):
        cube = _const_grids(jnp.full((3, 2), 0.3), jnp.full((3, 2), 0.4))
        vol = eqx.filter_jit(cube.__call__)(jnp.array(100.0), EXPIRIES[1], TENORS[0])
        assert jnp.isfinite(vol)
