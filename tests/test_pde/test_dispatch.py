"""Tests for the PDE dispatcher: routing, errors, and European accuracy."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption, LookbackOption
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.pde import (
    PDEConfig,
    PDEConfig2D,
    PDEResult,
    pde_price_dispatch,
    registered_recipes,
)

FINE = PDEConfig(n_spot=400, n_time=400)
COARSE = PDEConfig(n_spot=60, n_time=60)


def _model():
    return BlackScholesModel(
        vol=jnp.array(0.2), rate=jnp.array(0.05), dividend=jnp.array(0.02)
    )


def test_registered_recipes_contains_bs_pairs():
    recipes = registered_recipes()
    assert ("EuropeanOption", "BlackScholesModel") in recipes
    assert ("AmericanOption", "BlackScholesModel") in recipes
    assert ("DigitalOption", "BlackScholesModel") in recipes
    assert ("EquityBarrierOption", "BlackScholesModel") in recipes


def test_european_matches_analytic():
    opt = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
    model = _model()
    result = pde_price_dispatch(opt, model, FINE, spot=jnp.array(100.0))
    assert isinstance(result, PDEResult)
    bs = float(
        black_scholes_price(
            opt, jnp.array(100.0), model.vol, model.rate, model.dividend
        )
    )
    rel = abs(float(result) - bs) / bs
    assert rel < 5e-3, f"pde={float(result):.6f} bs={bs:.6f} rel={rel:.2e}"


def test_convergence_ordering():
    opt = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
    model = _model()
    bs = float(
        black_scholes_price(
            opt, jnp.array(100.0), model.vol, model.rate, model.dividend
        )
    )
    coarse = abs(float(pde_price_dispatch(opt, model, COARSE, spot=jnp.array(100.0))) - bs)
    fine = abs(float(pde_price_dispatch(opt, model, FINE, spot=jnp.array(100.0))) - bs)
    assert fine < coarse


def test_registered_recipes_contains_heston():
    assert ("EuropeanOption", "HestonModel") in registered_recipes()


def test_heston_dispatch_returns_pde_result():
    m = HestonModel(
        v0=jnp.array(0.04),
        kappa=jnp.array(1.5),
        theta=jnp.array(0.04),
        xi=jnp.array(0.5),
        rho=jnp.array(-0.6),
        rate=jnp.array(0.03),
        dividend=jnp.array(0.01),
    )
    opt = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
    cfg = PDEConfig2D(n_x=120, n_y=48, n_time=100, x_range=4.0, y_max=0.5, y_scale=0.03)
    result = pde_price_dispatch(opt, m, cfg, spot=jnp.array(100.0))
    assert isinstance(result, PDEResult)
    assert float(result) > 0.0


def test_unregistered_pair_raises():
    lookback = LookbackOption(expiry=jnp.array(1.0), is_call=True)
    with pytest.raises(ValueError, match="No PDE recipe registered"):
        pde_price_dispatch(lookback, _model(), FINE, spot=jnp.array(100.0))


def test_filter_jit_smoke():
    opt = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)

    @eqx.filter_jit
    def priced(spot, vol, rate, dividend):
        model = BlackScholesModel(vol=vol, rate=rate, dividend=dividend)
        return pde_price_dispatch(opt, model, COARSE, spot=spot).price

    val = priced(jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.02))
    assert float(val) > 0.0
