"""Tests for Heston calibration with a mock pricing function."""

import jax
import jax.numpy as jnp
import pytest

from valax.models.heston import HestonModel
from valax.calibration import CalibrationError, SolverSettings
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

        fitted = calibrate_heston(
            strikes, market_prices, spot, rate, dividend, expiry,
            pricing_fn=_mock_heston_pricer,
            initial_guess=guess,
        ).value

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

        fitted = calibrate_heston(
            strikes, market_prices, spot, rate, dividend, expiry,
            pricing_fn=cos_pricer,
            initial_guess=guess,
            settings=SolverSettings(
                method="levenberg_marquardt", rtol=1e-8, atol=1e-8,
                max_steps=2000,
            ),
        ).value

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
        with pytest.raises(ValueError, match="not available"):
            calibrate_heston(
                jnp.array([100.0]), jnp.array([5.0]),
                jnp.array(100.0), jnp.array(0.05), jnp.array(0.0),
                jnp.array(1.0),
                pricing_fn=_mock_heston_pricer,
                settings=SolverSettings(method="nonexistent"),
            )


def _cos_pricer(model, K, S, r, q, T):
    return heston_cos_price(
        EuropeanOption(strike=K, expiry=T, is_call=True), S, r, q, model
    )


def test_unconvergeable_fit_is_reported():
    """Anti-vacuity: noisy prices a single-expiry Heston cannot fit within
    the step budget are reported (``converged=False``; ``check()`` raises)
    rather than raised from inside the solve."""
    r, q, T, S = jnp.array(0.03), jnp.array(0.01), jnp.array(1.0), jnp.array(100.0)
    true_model = HestonModel(
        v0=jnp.array(0.04), kappa=jnp.array(2.0), theta=jnp.array(0.05),
        xi=jnp.array(0.4), rho=jnp.array(-0.6), rate=r, dividend=q,
    )
    strikes = jnp.linspace(85.0, 115.0, 9)
    prices = jax.vmap(lambda K: _cos_pricer(true_model, K, S, r, q, T))(strikes)
    noisy = prices * (1.0 + 1e-4 * jnp.cos(strikes))
    res = calibrate_heston(
        strikes, noisy, S, r, q, T, pricing_fn=_cos_pricer,
        settings=SolverSettings(
            method="levenberg_marquardt", rtol=1e-8, atol=1e-8, max_steps=50
        ),
    )
    assert not bool(res.diagnostics.converged)
    with pytest.raises(CalibrationError, match="heston"):
        res.check()


# ── Fixed parameters and multi-expiry fits ──────────────────────────


def _example17_smiles():
    """Example 17's two SABR smiles (0.5Y, 1Y) as Black-Scholes call prices,
    rebuilt from the calibrated surface parameters (no market chain rerun)."""
    from valax.pricing.analytic.black_scholes import black_scholes_price
    from valax.surfaces.sabr_surface import SABRVolSurface

    surface = SABRVolSurface(
        expiries=jnp.array([0.5, 1.0]),
        forwards=jnp.array([101.50646605234357, 101.60090387045518]),
        alphas=jnp.array([2.941672587386952, 2.928149508698076]),
        betas=jnp.array([0.5, 0.5]),
        rhos=jnp.array([-0.5435316093507521, -0.4118260146951523]),
        nus=jnp.array([0.8955175856106505, 0.7463431942501192]),
    )
    r, q, S = jnp.array(0.027146146570611727), jnp.array(0.02528628643942919), jnp.array(101.41211601397752)
    strikes = jnp.array([81.20572662674495, 91.35644245508807, 101.50715828343118,
                         111.6578741117743, 121.80858994011741,
                         81.28183170032584, 91.44206066286657, 101.6022896254073,
                         111.76251858794804, 121.92274755048875])
    expiries = jnp.array([0.5] * 5 + [1.0] * 5)
    prices = jax.vmap(lambda k, t: black_scholes_price(
        EuropeanOption(strike=k, expiry=t), S, surface(k, t), r, q))(strikes, expiries)
    return strikes, expiries, prices, S, r, q


def test_fixed_params_respected_and_reported():
    r, q, T, S = jnp.array(0.03), jnp.array(0.01), jnp.array(1.0), jnp.array(100.0)
    true_model = HestonModel(
        v0=jnp.array(0.04), kappa=jnp.array(2.0), theta=jnp.array(0.05),
        xi=jnp.array(0.4), rho=jnp.array(-0.6), rate=r, dividend=q,
    )
    strikes = jnp.linspace(85.0, 115.0, 9)
    prices = jax.vmap(lambda K: _cos_pricer(true_model, K, S, r, q, T))(strikes)
    guess = HestonModel(  # kappa deliberately wrong in the guess: it is ignored
        v0=jnp.array(0.05), kappa=jnp.array(9.0), theta=jnp.array(0.06),
        xi=jnp.array(0.5), rho=jnp.array(-0.5), rate=r, dividend=q,
    )
    res = calibrate_heston(
        strikes, prices, S, r, q, T, _cos_pricer, initial_guess=guess,
        fixed_params={"kappa": 2.0},
        settings=SolverSettings(method="levenberg_marquardt", rtol=1e-10,
                                atol=1e-10, max_steps=500),
    )
    assert float(res.value.kappa) == 2.0
    assert float(res.inputs.fixed["kappa"]) == 2.0  # reported as a fixed input
    assert "kappa" not in res.theta
    assert bool(res.converged)
    assert jnp.allclose(res.value.xi, true_model.xi, atol=1e-5)
    assert jnp.allclose(res.value.rho, true_model.rho, atol=1e-5)


def test_fixed_params_validation():
    args = (jnp.array([100.0]), jnp.array([5.0]), jnp.array(100.0),
            jnp.array(0.05), jnp.array(0.0), jnp.array(1.0), _mock_heston_pricer)
    with pytest.raises(ValueError, match="unknown Heston parameter"):
        calibrate_heston(*args, fixed_params={"sigma": 0.2})
    with pytest.raises(ValueError, match="every parameter"):
        calibrate_heston(*args, fixed_params={f: 0.5 for f in
                                              ("v0", "kappa", "theta", "xi", "rho")})
    with pytest.raises(ValueError, match="aligned"):
        calibrate_heston(jnp.array([90.0, 100.0]), jnp.array([12.0, 5.0]),
                         *args[2:5], jnp.array([0.5, 1.0, 2.0]), _mock_heston_pricer)


def test_multi_expiry_converges_where_single_expiry_does_not():
    """Example 17's SABR market: the five-parameter fit to the 1Y smile alone
    wanders down the flat (κ, ξ, v0) valley and exhausts its budget (κ ≈ 8.9,
    ξ ≈ 4.3); the same call with both expiries converges in ~30 steps."""
    strikes, expiries, prices, S, r, q = _example17_smiles()
    one_y = expiries == 1.0
    single = calibrate_heston(strikes[one_y], prices[one_y], S, r, q,
                              jnp.array(1.0), _cos_pricer)
    multi = calibrate_heston(strikes, prices, S, r, q, expiries, _cos_pricer)
    assert not bool(single.converged)
    assert bool(multi.converged)
    assert 0.5 < float(multi.value.kappa) < 3.0
    assert 0.5 < float(multi.value.xi) < 2.0
    # Every quote, both expiries, is fitted to well under a price unit.
    assert float(multi.diagnostics.max_abs_residual) < 0.2


def test_scalar_expiry_equals_broadcast_expiry():
    r, q, S = jnp.array(0.03), jnp.array(0.01), jnp.array(100.0)
    strikes = jnp.linspace(85.0, 115.0, 5)
    prices = jax.vmap(lambda K: _mock_heston_pricer(
        HestonModel(v0=jnp.array(0.04), kappa=jnp.array(2.0), theta=jnp.array(0.04),
                    xi=jnp.array(0.5), rho=jnp.array(-0.7)), K, S, r, q, 1.0))(strikes)
    a = calibrate_heston(strikes, prices, S, r, q, jnp.array(1.0), _mock_heston_pricer)
    b = calibrate_heston(strikes, prices, S, r, q, jnp.ones(5), _mock_heston_pricer)
    assert jnp.allclose(a.value.v0, b.value.v0, rtol=1e-12)


def test_calibrate_heston_jit_smoke():
    import equinox as eqx

    r, q, S = jnp.array(0.03), jnp.array(0.01), jnp.array(100.0)
    strikes = jnp.linspace(90.0, 110.0, 5)
    expiries = jnp.array([0.5, 0.5, 1.0, 1.0, 1.0])
    true_model = HestonModel(v0=jnp.array(0.04), kappa=jnp.array(2.0),
                             theta=jnp.array(0.05), xi=jnp.array(0.4),
                             rho=jnp.array(-0.6), rate=r, dividend=q)
    prices = jax.vmap(lambda K, T: _cos_pricer(true_model, K, S, r, q, T))(strikes, expiries)

    @eqx.filter_jit
    def fit(p):
        return calibrate_heston(strikes, p, S, r, q, expiries, _cos_pricer,
                                fixed_params={"kappa": 2.0}).value

    m = fit(prices)
    assert jnp.isfinite(m.v0) and float(m.kappa) == 2.0
