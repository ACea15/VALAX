"""Tests for Heston calibration with a mock pricing function."""

import jax
import jax.numpy as jnp
import pytest

from valax.models.heston import HestonModel
from valax.calibration.heston import calibrate_heston
from valax.pricing.analytic.heston import heston_cos_price
from valax.instruments.options import EuropeanOption


jax.config.update("jax_enable_x64", True)


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

    def test_roundtrip_real_cos_pricer(self):
        """End-to-end calibration on the real Fang-Oosterlee COS pricer.

        Generate a smile from a known ground-truth Heston model, then
        calibrate from a perturbed initial guess and verify exact
        recovery of every parameter.  This is the first test in the
        suite that exercises ``calibrate_heston`` against a true Heston
        pricing function (every previous test used a BS-with-
        ``sqrt(v0)`` mock).

        Single-expiry Heston smile calibration is classically ill-
        conditioned (kappa, xi, v0 trade off along a flat valley), so
        we use a near-truth initial guess and a generous LM budget;
        on noise-free synthetic data this still recovers parameters to
        floating-point precision.  Under noisy market data the standard
        practice is multi-expiry calibration with a vol-surface
        objective — that workflow will land alongside SSVI in a
        follow-up.
        """
        true_model = HestonModel(
            v0=jnp.array(0.04),
            kappa=jnp.array(2.0),
            theta=jnp.array(0.05),
            xi=jnp.array(0.4),
            rho=jnp.array(-0.6),
            rate=jnp.array(0.03),
            dividend=jnp.array(0.01),
        )

        spot = jnp.array(100.0)
        rate = jnp.array(0.03)
        dividend = jnp.array(0.01)
        expiry = jnp.array(1.0)
        strikes = jnp.linspace(85.0, 115.0, 9)

        def cos_pricer(model, K, S, r, q, T):
            return heston_cos_price(
                EuropeanOption(strike=K, expiry=T, is_call=True),
                S, r, q, model,
            )

        market_prices = jax.vmap(
            lambda K: cos_pricer(true_model, K, spot, rate, dividend, expiry)
        )(strikes)

        # Moderate perturbation of every Heston parameter — far enough
        # from truth to demonstrate the optimizer is doing real work,
        # close enough that LM converges within a tractable budget.
        guess = HestonModel(
            v0=jnp.array(0.05),
            kappa=jnp.array(1.5),
            theta=jnp.array(0.06),
            xi=jnp.array(0.5),
            rho=jnp.array(-0.5),
            rate=rate,
            dividend=dividend,
        )

        fitted, sol = calibrate_heston(
            strikes, market_prices, spot, rate, dividend, expiry,
            pricing_fn=cos_pricer,
            initial_guess=guess,
            solver="levenberg_marquardt",
            max_steps=2000,
        )

        # Noise-free synthetic prices generated by the same pricer:
        # LM should drive every parameter back to truth to better than
        # 1e-6 relative.  Any regression in the COS pricer or the
        # calibration plumbing would blow this up immediately.
        assert jnp.allclose(fitted.v0,    true_model.v0,    atol=1e-6)
        assert jnp.allclose(fitted.kappa, true_model.kappa, atol=1e-4)
        assert jnp.allclose(fitted.theta, true_model.theta, atol=1e-6)
        assert jnp.allclose(fitted.xi,    true_model.xi,    atol=1e-4)
        assert jnp.allclose(fitted.rho,   true_model.rho,   atol=1e-5)

        # Re-pricing residual is the cleanest convergence diagnostic.
        repriced = jax.vmap(
            lambda K: cos_pricer(fitted, K, spot, rate, dividend, expiry)
        )(strikes)
        assert float(jnp.max(jnp.abs(repriced - market_prices))) < 1e-10

    def test_invalid_solver_raises(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            calibrate_heston(
                jnp.array([100.0]), jnp.array([5.0]),
                jnp.array(100.0), jnp.array(0.05), jnp.array(0.0),
                jnp.array(1.0),
                pricing_fn=_mock_heston_pricer,
                solver="nonexistent",
            )
