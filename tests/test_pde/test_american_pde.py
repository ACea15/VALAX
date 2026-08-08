"""American option PDE (penalty method) vs the CRR binomial tree."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import AmericanOption, EuropeanOption
from valax.models.black_scholes import BlackScholesModel
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.lattice.binomial import BinomialConfig, binomial_price
from valax.pricing.pde import PDEConfig, pde_price_dispatch

FINE = PDEConfig(n_spot=400, n_time=400, penalty_rho=1.0e6, penalty_iters=6)


def _model(vol=0.2, rate=0.05, dividend=0.0):
    return BlackScholesModel(
        vol=jnp.array(vol), rate=jnp.array(rate), dividend=jnp.array(dividend)
    )


@pytest.mark.parametrize("S,K,vol,r,q", [
    (100.0, 100.0, 0.20, 0.05, 0.0),
    (100.0, 110.0, 0.25, 0.05, 0.0),
    (100.0, 90.0, 0.30, 0.08, 0.02),
])
def test_american_put_matches_binomial(S, K, vol, r, q):
    put = AmericanOption(strike=jnp.array(K), expiry=jnp.array(1.0), is_call=False)
    model = _model(vol, r, q)
    pde = float(pde_price_dispatch(put, model, FINE, spot=jnp.array(S)))

    euro = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(1.0), is_call=False)
    bino = float(
        binomial_price(
            euro, jnp.array(S), jnp.array(vol), jnp.array(r), jnp.array(q),
            BinomialConfig(n_steps=1000, american=True),
        )
    )
    assert abs(pde - bino) < 0.05, f"pde={pde:.5f} binomial={bino:.5f}"


def test_early_exercise_premium_nonnegative():
    put = AmericanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)
    euro = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)
    model = _model(0.2, 0.05, 0.0)
    amer = float(pde_price_dispatch(put, model, FINE, spot=jnp.array(100.0)))
    eur = float(black_scholes_price(euro, jnp.array(100.0), model.vol, model.rate, model.dividend))
    assert amer >= eur - 1e-6


def test_american_call_no_dividend_equals_european():
    # With no dividends, early exercise of a call is never optimal.
    call = AmericanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
    euro = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
    model = _model(0.2, 0.05, 0.0)
    amer = float(pde_price_dispatch(call, model, FINE, spot=jnp.array(100.0)))
    eur = float(black_scholes_price(euro, jnp.array(100.0), model.vol, model.rate, model.dividend))
    assert abs(amer - eur) < 0.05


def test_put_delta_negative():
    put = AmericanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)

    def fn(spot):
        return pde_price_dispatch(put, _model(0.2, 0.05, 0.0), FINE, spot=spot).price

    delta = float(jax.grad(fn)(jnp.array(100.0)))
    assert -1.0 < delta < 0.0


def test_filter_jit_smoke():
    put = AmericanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)
    cfg = PDEConfig(n_spot=120, n_time=120, penalty_rho=1.0e6, penalty_iters=5)

    @eqx.filter_jit
    def priced(spot, vol, rate, dividend):
        model = BlackScholesModel(vol=vol, rate=rate, dividend=dividend)
        return pde_price_dispatch(put, model, cfg, spot=spot).price

    val = priced(jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.0))
    assert float(val) > 0.0
