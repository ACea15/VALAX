"""Cash-or-nothing digital PDE (CN + Rannacher) vs the closed-form price."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from valax.instruments.options import DigitalOption
from valax.models.black_scholes import BlackScholesModel
from valax.pricing.analytic import digital_option_price
from valax.pricing.pde import PDEConfig, pde_price_dispatch

FINE = PDEConfig(n_spot=600, n_time=600)
COARSE = PDEConfig(n_spot=80, n_time=80)


def _model(vol=0.2, rate=0.05, dividend=0.0):
    return BlackScholesModel(
        vol=jnp.array(vol), rate=jnp.array(rate), dividend=jnp.array(dividend)
    )


@pytest.mark.parametrize("S,K,is_call", [
    (100.0, 100.0, True),
    (100.0, 110.0, True),
    (100.0, 90.0, True),
    (100.0, 100.0, False),
    (100.0, 95.0, False),
])
def test_digital_matches_analytic(S, K, is_call):
    dig = DigitalOption(
        strike=jnp.array(K), expiry=jnp.array(1.0),
        payout=jnp.array(1.0), is_call=is_call,
    )
    model = _model()
    pde = float(pde_price_dispatch(dig, model, FINE, spot=jnp.array(S)))
    analytic = float(
        digital_option_price(dig, jnp.array(S), model.vol, model.rate, model.dividend)
    )
    assert abs(pde - analytic) < 0.01, f"pde={pde:.5f} analytic={analytic:.5f}"


def test_convergence_ordering():
    dig = DigitalOption(
        strike=jnp.array(105.0), expiry=jnp.array(1.0),
        payout=jnp.array(1.0), is_call=True,
    )
    model = _model()
    analytic = float(
        digital_option_price(dig, jnp.array(100.0), model.vol, model.rate, model.dividend)
    )
    coarse = abs(float(pde_price_dispatch(dig, model, COARSE, spot=jnp.array(100.0))) - analytic)
    fine = abs(float(pde_price_dispatch(dig, model, FINE, spot=jnp.array(100.0))) - analytic)
    assert fine < coarse


def test_filter_jit_smoke():
    dig = DigitalOption(
        strike=jnp.array(100.0), expiry=jnp.array(1.0),
        payout=jnp.array(1.0), is_call=True,
    )
    cfg = PDEConfig(n_spot=120, n_time=120)

    @eqx.filter_jit
    def priced(spot, vol, rate, dividend):
        model = BlackScholesModel(vol=vol, rate=rate, dividend=dividend)
        return pde_price_dispatch(dig, model, cfg, spot=spot).price

    val = priced(jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.0))
    assert float(val) > 0.0
