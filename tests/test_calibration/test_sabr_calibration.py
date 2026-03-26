"""Tests for SABR calibration."""

import jax
import jax.numpy as jnp
import pytest

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
        fitted, sol = calibrate_sabr(
            strikes, market_vols, forward, expiry,
            initial_guess=guess,
            fixed_beta=jnp.array(0.5),
            solver="levenberg_marquardt",
        )

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
        fitted, sol = calibrate_sabr(
            strikes, market_vols, forward, expiry,
            initial_guess=guess,
            solver="levenberg_marquardt",
        )

        # Looser tolerances — beta is poorly identified
        fitted_vols = jax.vmap(
            lambda K: sabr_implied_vol(fitted, forward, K, expiry)
        )(strikes)
        assert jnp.allclose(fitted_vols, market_vols, atol=1e-5)

    def test_fixed_beta_value_preserved(self, synthetic_smile):
        """Fitted model has exactly the fixed beta value."""
        strikes, market_vols, forward, expiry = synthetic_smile
        fitted, _ = calibrate_sabr(
            strikes, market_vols, forward, expiry,
            fixed_beta=jnp.array(0.7),
        )
        assert float(fitted.beta) == pytest.approx(0.7, abs=1e-10)

    def test_default_initial_guess(self, synthetic_smile):
        """Calibration works without providing an initial guess."""
        strikes, market_vols, forward, expiry = synthetic_smile
        fitted, sol = calibrate_sabr(
            strikes, market_vols, forward, expiry,
            fixed_beta=jnp.array(0.5),
        )
        # Just verify it converges to a reasonable fit
        fitted_vols = jax.vmap(
            lambda K: sabr_implied_vol(fitted, forward, K, expiry)
        )(strikes)
        assert jnp.allclose(fitted_vols, market_vols, atol=1e-4)

    def test_bfgs_solver(self, true_sabr, synthetic_smile):
        """BFGS also converges on the same problem."""
        strikes, market_vols, forward, expiry = synthetic_smile
        fitted, sol = calibrate_sabr(
            strikes, market_vols, forward, expiry,
            fixed_beta=jnp.array(0.5),
            solver="bfgs",
        )
        fitted_vols = jax.vmap(
            lambda K: sabr_implied_vol(fitted, forward, K, expiry)
        )(strikes)
        assert jnp.allclose(fitted_vols, market_vols, atol=1e-4)

    def test_weighted_calibration(self, synthetic_smile):
        """Weights emphasize ATM strikes."""
        strikes, market_vols, forward, expiry = synthetic_smile
        # Weight ATM more heavily
        weights = jnp.exp(-0.5 * ((strikes - forward) / 10.0) ** 2)
        fitted, sol = calibrate_sabr(
            strikes, market_vols, forward, expiry,
            fixed_beta=jnp.array(0.5),
            weights=weights,
        )
        # ATM fit should be very tight
        atm_idx = jnp.argmin(jnp.abs(strikes - forward))
        fitted_atm_vol = sabr_implied_vol(fitted, forward, strikes[atm_idx], expiry)
        assert abs(float(fitted_atm_vol - market_vols[atm_idx])) < 1e-5

    def test_invalid_solver_raises(self, synthetic_smile):
        strikes, market_vols, forward, expiry = synthetic_smile
        with pytest.raises(ValueError, match="Unknown solver"):
            calibrate_sabr(
                strikes, market_vols, forward, expiry,
                solver="nonexistent",
            )
