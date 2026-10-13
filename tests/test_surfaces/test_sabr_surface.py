"""SABR expiry-slice surfaces (and SVI) as batched calibration levels.

The batched level must reproduce the per-slice fits it replaced exactly (the
per-slice :func:`calibrate_sabr` is itself bit-identical to the pre-migration
solver ladder), padding a ragged strike grid must not move the fit, and a
slice that cannot be fitted is named by its expiry.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.calibration import CalibrationError, calibrate_sabr
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol
from valax.surfaces import (
    SVISlice,
    calibrate_sabr_surface,
    calibrate_svi_slice,
    calibrate_svi_surface,
    svi_implied_vol,
)

jax.config.update("jax_enable_x64", True)

EXPIRIES = jnp.array([0.25, 1.0, 2.0])
FORWARDS = jnp.array([100.0, 101.0, 102.0])


def _smiles(n_k=(9, 9, 9)):
    strikes, vols = [], []
    for i, n in enumerate(n_k):
        truth = SABRModel(
            alpha=jnp.array(2.0 + 0.2 * i), beta=jnp.array(0.5),
            rho=jnp.array(-0.3 + 0.05 * i), nu=jnp.array(0.5 - 0.05 * i),
        )
        k = jnp.linspace(80.0, 120.0, n)
        v = jax.vmap(
            lambda kk: sabr_implied_vol(truth, FORWARDS[i], kk, EXPIRIES[i])
        )(k)
        strikes.append(k)
        vols.append(v + 2e-4 * jnp.sin(k))  # not exactly SABR: residual != 0
    return strikes, vols


class TestSABRSurface:
    def test_batched_equals_per_slice_fits(self):
        strikes, vols = _smiles()
        res = calibrate_sabr_surface(
            strikes, vols, FORWARDS, EXPIRIES, fixed_beta=jnp.array(0.5)
        ).check()
        assert res.labels == ("3M", "1Y", "2Y")
        for i in range(3):
            one = calibrate_sabr(
                strikes[i], vols[i], FORWARDS[i], EXPIRIES[i],
                fixed_beta=jnp.array(0.5),
            ).value
            np.testing.assert_allclose(res.value.alphas[i], one.alpha, atol=1e-12)
            np.testing.assert_allclose(res.value.rhos[i], one.rho, atol=1e-12)
            np.testing.assert_allclose(res.value.nus[i], one.nu, atol=1e-12)

    def test_ragged_grid_padding_is_neutral(self):
        strikes, vols = _smiles((9, 5, 7))
        res = calibrate_sabr_surface(
            strikes, vols, FORWARDS, EXPIRIES, fixed_beta=jnp.array(0.5)
        ).check()
        one = calibrate_sabr(
            strikes[1], vols[1], FORWARDS[1], EXPIRIES[1],
            fixed_beta=jnp.array(0.5),
        ).value
        np.testing.assert_allclose(res.value.alphas[1], one.alpha, atol=1e-12)
        np.testing.assert_allclose(res.value.nus[1], one.nu, atol=1e-12)

    def test_bad_slice_is_named(self):
        strikes, vols = _smiles()
        vols[2] = vols[2].at[0].set(jnp.nan)
        res = calibrate_sabr_surface(
            strikes, vols, FORWARDS, EXPIRIES, fixed_beta=jnp.array(0.5)
        )
        assert res.failed_labels() == ("2Y",)
        with pytest.raises(CalibrationError, match="2Y"):
            res.check()


def _svi_smiles():
    strikes, vols = [], []
    for i in range(3):
        truth = SVISlice(
            a=jnp.array(0.01 + 0.01 * i), b=jnp.array(0.1),
            rho=jnp.array(-0.3), m=jnp.array(0.02), sigma=jnp.array(0.15),
        )
        k = jnp.linspace(80.0, 120.0, 9)
        v = jax.vmap(
            lambda kk: svi_implied_vol(truth, FORWARDS[i], kk, EXPIRIES[i])
        )(k)
        strikes.append(k)
        vols.append(v + 1e-4 * jnp.cos(k))
    return strikes, vols


class TestSVISurface:
    def test_batched_equals_per_slice_fits(self):
        strikes, vols = _svi_smiles()
        res = calibrate_svi_surface(strikes, vols, FORWARDS, EXPIRIES).check()
        for i in range(3):
            one = calibrate_svi_slice(
                strikes[i], vols[i], FORWARDS[i], EXPIRIES[i]
            ).value
            np.testing.assert_allclose(res.value.a_vec[i], one.a, atol=1e-10)
            np.testing.assert_allclose(res.value.b_vec[i], one.b, atol=1e-10)
            np.testing.assert_allclose(res.value.m_vec[i], one.m, atol=1e-10)

    def test_bad_slice_is_named(self):
        strikes, vols = _svi_smiles()
        vols[0] = vols[0].at[3].set(jnp.nan)
        res = calibrate_svi_surface(strikes, vols, FORWARDS, EXPIRIES)
        assert res.failed_labels() == ("3M",)
        with pytest.raises(CalibrationError, match="3M"):
            res.check()
