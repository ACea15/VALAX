"""Tests for Heston calibration with a mock pricing function."""

import jax
import jax.numpy as jnp
import pytest

from valax.models.heston import HestonModel
from valax.calibration.heston import calibrate_heston


# ── Mock pricer ─────────────────────────────────────────────────────

def _mock_heston_pricer(model, strike, spot, rate, dividend, expiry):
    """Simplified mock: BS-like price using sqrt(v0) as vol.

    Not a real Heston pricer — just tests that the calibration plumbing
    works end-to-end.
    """
    vol = jnp.sqrt(model.v0)
    d1 = (jnp.log(spot / strike) + (rate - dividend + 0.5 * vol**2) * expiry) / (
        vol * jnp.sqrt(expiry)
    )
    d2 = d1 - vol * jnp.sqrt(expiry)
    call = (
        spot * jnp.exp(-dividend * expiry) * jax.scipy.stats.norm.cdf(d1)
        - strike * jnp.exp(-rate * expiry) * jax.scipy.stats.norm.cdf(d2)
    )
    return call


class TestHestonCalibration:
    def test_roundtrip_mock_pricer(self):
        """Calibration plumbing works: recover v0 from mock prices."""
        true_model = HestonModel(
            v0=jnp.array(0.04),
            kappa=jnp.array(2.0),
            theta=jnp.array(0.04),
            xi=jnp.array(0.5),
            rho=jnp.array(-0.7),
            rate=jnp.array(0.05),
            dividend=jnp.array(0.02),
        )

        spot = jnp.array(100.0)
        rate = jnp.array(0.05)
        dividend = jnp.array(0.02)
        expiry = jnp.array(1.0)
        strikes = jnp.linspace(80.0, 120.0, 9)

        # Generate synthetic prices from the mock pricer
        market_prices = jax.vmap(
            lambda K: _mock_heston_pricer(true_model, K, spot, rate, dividend, expiry)
        )(strikes)

        # Calibrate from a different initial guess
        guess = HestonModel(
            v0=jnp.array(0.09),
            kappa=jnp.array(1.0),
            theta=jnp.array(0.09),
            xi=jnp.array(0.3),
            rho=jnp.array(-0.3),
            rate=rate,
            dividend=dividend,
        )

        fitted, sol = calibrate_heston(
            strikes, market_prices, spot, rate, dividend, expiry,
            pricing_fn=_mock_heston_pricer,
            initial_guess=guess,
            solver="levenberg_marquardt",
        )

        # The mock pricer only depends on v0, so v0 should be well recovered.
        # Other params may not converge to true values since the mock
        # doesn't actually use them for pricing.
        assert jnp.allclose(fitted.v0, true_model.v0, atol=1e-2)

    def test_invalid_solver_raises(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            calibrate_heston(
                jnp.array([100.0]), jnp.array([5.0]),
                jnp.array(100.0), jnp.array(0.05), jnp.array(0.0),
                jnp.array(1.0),
                pricing_fn=_mock_heston_pricer,
                solver="nonexistent",
            )
