"""Tests for SABR calibration."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.calibration import CalibrationError, CalibrationResult, SolverSettings

from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol
from valax.calibration.sabr import calibrate_sabr


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def true_sabr():
    """Known SABR model to generate synthetic market data."""
    return SABRModel(
        alpha=jnp.array(0.25),
        beta=jnp.array(0.5),
        rho=jnp.array(-0.35),
        nu=jnp.array(0.45),
    )


@pytest.fixture
def synthetic_smile(true_sabr):
    """Generate synthetic vol smile from the true model."""
    forward = jnp.array(100.0)
    expiry = jnp.array(1.0)
    strikes = jnp.linspace(70.0, 130.0, 13)
    market_vols = jax.vmap(
        lambda K: sabr_implied_vol(true_sabr, forward, K, expiry)
    )(strikes)
    return strikes, market_vols, forward, expiry


# ── Round-trip recovery ─────────────────────────────────────────────

class TestSABRCalibration:
    def test_roundtrip_fixed_beta_lm(self, true_sabr, synthetic_smile):
        """LM recovers true params when beta is fixed."""
        strikes, market_vols, forward, expiry = synthetic_smile

        # Start from a different initial guess
        guess = SABRModel(
            alpha=jnp.array(0.15),
            beta=jnp.array(0.5),
            rho=jnp.array(0.0),
            nu=jnp.array(0.2),
        )
        fitted = calibrate_sabr(
            strikes, market_vols, forward, expiry,
            initial_guess=guess,
            fixed_beta=jnp.array(0.5),
        ).value

        assert jnp.allclose(fitted.alpha, true_sabr.alpha, atol=1e-4)
        assert jnp.allclose(fitted.rho, true_sabr.rho, atol=1e-3)
        assert jnp.allclose(fitted.nu, true_sabr.nu, atol=1e-3)
        assert float(fitted.beta) == 0.5

    def test_roundtrip_free_beta_lm(self, true_sabr, synthetic_smile):
        """LM recovers params when all 4 are free (harder problem)."""
        strikes, market_vols, forward, expiry = synthetic_smile

        guess = SABRModel(
            alpha=jnp.array(0.20),
            beta=jnp.array(0.6),
            rho=jnp.array(-0.1),
            nu=jnp.array(0.3),
        )
        fitted = calibrate_sabr(
            strikes, market_vols, forward, expiry,
            initial_guess=guess,
        ).value

        # Looser tolerances — beta is poorly identified
        fitted_vols = jax.vmap(
            lambda K: sabr_implied_vol(fitted, forward, K, expiry)
        )(strikes)
        assert jnp.allclose(fitted_vols, market_vols, atol=1e-5)

    def test_fixed_beta_value_preserved(self, synthetic_smile):
        """Fitted model has exactly the fixed beta value."""
        strikes, market_vols, forward, expiry = synthetic_smile
        fitted = calibrate_sabr(
            strikes, market_vols, forward, expiry,
            fixed_beta=jnp.array(0.7),
        ).value
        assert float(fitted.beta) == pytest.approx(0.7, abs=1e-10)

    def test_default_initial_guess(self, synthetic_smile):
        """Calibration works without providing an initial guess."""
        strikes, market_vols, forward, expiry = synthetic_smile
        fitted = calibrate_sabr(
            strikes, market_vols, forward, expiry,
            fixed_beta=jnp.array(0.5),
        ).value
        # Just verify it converges to a reasonable fit
        fitted_vols = jax.vmap(
            lambda K: sabr_implied_vol(fitted, forward, K, expiry)
        )(strikes)
        assert jnp.allclose(fitted_vols, market_vols, atol=1e-4)

    def test_bfgs_solver(self, true_sabr, synthetic_smile):
        """BFGS also converges on the same problem."""
        strikes, market_vols, forward, expiry = synthetic_smile
        fitted = calibrate_sabr(
            strikes, market_vols, forward, expiry,
            fixed_beta=jnp.array(0.5),
            settings=SolverSettings(method="bfgs", rtol=1e-8, atol=1e-8),
        ).value
        fitted_vols = jax.vmap(
            lambda K: sabr_implied_vol(fitted, forward, K, expiry)
        )(strikes)
        assert jnp.allclose(fitted_vols, market_vols, atol=1e-4)

    def test_weighted_calibration(self, synthetic_smile):
        """Weights emphasize ATM strikes."""
        strikes, market_vols, forward, expiry = synthetic_smile
        # Weight ATM more heavily
        weights = jnp.exp(-0.5 * ((strikes - forward) / 10.0) ** 2)
        fitted = calibrate_sabr(
            strikes, market_vols, forward, expiry,
            fixed_beta=jnp.array(0.5),
            weights=weights,
        ).value
        # ATM fit should be very tight
        atm_idx = jnp.argmin(jnp.abs(strikes - forward))
        fitted_atm_vol = sabr_implied_vol(fitted, forward, strikes[atm_idx], expiry)
        assert abs(float(fitted_atm_vol - market_vols[atm_idx])) < 1e-5

    def test_invalid_solver_raises(self, synthetic_smile):
        strikes, market_vols, forward, expiry = synthetic_smile
        with pytest.raises(ValueError, match="not available"):
            calibrate_sabr(
                strikes, market_vols, forward, expiry,
                settings=SolverSettings(method="nonexistent"),
            )

    def test_returns_calibration_result(self, synthetic_smile):
        """The result carries the level, its inputs and diagnostics."""
        strikes, market_vols, forward, expiry = synthetic_smile
        res = calibrate_sabr(
            strikes, market_vols, forward, expiry, fixed_beta=jnp.array(0.5)
        )
        assert isinstance(res, CalibrationResult)
        assert res.problem == "least_squares"
        assert bool(res.diagnostics.converged)
        assert float(res.diagnostics.max_abs_residual) < 1e-6
        np.testing.assert_allclose(res.inputs.quotes, market_vols)
        assert float(res.inputs.fixed["beta"]) == 0.5
        res.check()

    def test_unconvergeable_fit_is_reported(self, synthetic_smile):
        """Anti-vacuity: a starved solve reports failure instead of raising."""
        strikes, market_vols, forward, expiry = synthetic_smile
        res = calibrate_sabr(
            strikes, market_vols, forward, expiry, fixed_beta=jnp.array(0.5),
            settings=SolverSettings(
                method="levenberg_marquardt", rtol=1e-8, atol=1e-8, max_steps=2
            ),
        )
        assert not bool(res.diagnostics.converged)
        with pytest.raises(CalibrationError, match="did not converge"):
            res.check()

    def test_traceable(self, synthetic_smile):
        """The calibration runs under jit and vmap over smiles."""
        strikes, market_vols, forward, expiry = synthetic_smile
        fit = eqx.filter_jit(
            lambda v: calibrate_sabr(
                strikes, v, forward, expiry, fixed_beta=jnp.array(0.5)
            ).value.alpha
        )
        np.testing.assert_allclose(
            fit(market_vols),
            calibrate_sabr(
                strikes, market_vols, forward, expiry, fixed_beta=jnp.array(0.5)
            ).value.alpha,
            atol=1e-12,
        )
        alphas = jax.vmap(fit)(jnp.stack([market_vols, market_vols * 1.01]))
        assert alphas.shape == (2,)
