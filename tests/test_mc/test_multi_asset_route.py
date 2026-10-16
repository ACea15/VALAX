"""Multi-asset options through ``price()``: keyed spots in model order (D16).

``SpreadOption`` and ``WorstOfBasketOption`` name their assets in
``underlying_ids``; ``price()`` resolves those spots from the market and
hands them, stacked in that order, to the MC recipe as ``spots`` — the same
call a direct ``mc_price_dispatch`` makes, so the price is bit-identical for
the same key and config. The ids are static, so the route works with the
trade traced under ``jit`` and the gradient reaches each named spot.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core.market_state import MarketState
from valax.core.pricing import auto_method, price
from valax.curves.discount import flat_curve
from valax.curves.graph import CurveGraph
from valax.instruments.options import SpreadOption, WorstOfBasketOption
from valax.models.multi_asset import MultiAssetGBMModel
from valax.pricing.mc import MCConfig
from valax.pricing.mc.dispatch import mc_price_dispatch

jax.config.update("jax_enable_x64", True)

MARKET = MarketState(
    curves=CurveGraph(curves={"_default_": flat_curve(0.03)}),
    spots={"SPX": jnp.asarray(100.0), "SX5E": jnp.asarray(50.0),
           "NKY": jnp.asarray(80.0)},
)
CFG = MCConfig(n_paths=4096, n_steps=20)
KEY = jax.random.PRNGKey(7)


def _model(n=2):
    vols = jnp.array([0.2, 0.3, 0.25])[:n]
    corr = 0.5 * jnp.ones((n, n)) + 0.5 * jnp.eye(n)
    return MultiAssetGBMModel(vols=vols, rate=jnp.asarray(0.03),
                              dividends=jnp.zeros(n), correlation=corr)


def _spread(ids=("SPX", "SX5E")):
    return SpreadOption(expiry=jnp.asarray(1.0), strike=jnp.asarray(45.0),
                        notional=jnp.asarray(1.0), underlying_ids=ids)


def _worst(ids=("SPX", "SX5E", "NKY")):
    return WorstOfBasketOption(expiry=jnp.asarray(1.0), strike=jnp.asarray(1.0),
                               notional=jnp.asarray(1.0), n_assets=3,
                               underlying_ids=ids)


def _spots(ids):
    return jnp.stack([MARKET.spots[i] for i in ids])


@pytest.mark.parametrize("case", ["spread", "worst_of"])
def test_route_is_the_direct_recipe(case):
    inst, model = (_spread(), _model(2)) if case == "spread" else (_worst(), _model(3))
    got = price(inst, model, MARKET, method="mc", config=CFG, key=KEY)
    want = mc_price_dispatch(inst, model, CFG, KEY, spots=_spots(inst.underlying_ids))
    np.testing.assert_array_equal(np.asarray(got.price), np.asarray(want.price))
    np.testing.assert_array_equal(np.asarray(got.stderr), np.asarray(want.stderr))
    assert float(got.price) > 0.0


def test_auto_is_mc_and_the_order_matters():
    assert auto_method(_spread(), _model()) == "mc"
    ab = price(_spread(("SPX", "SX5E")), _model(), MARKET, config=CFG, key=KEY).price
    ba = price(_spread(("SX5E", "SPX")), _model(), MARKET, config=CFG, key=KEY).price
    assert float(ab) != float(ba)


def test_without_underlying_ids_price_says_what_is_missing():
    bare = SpreadOption(expiry=jnp.asarray(1.0), strike=jnp.asarray(5.0),
                        notional=jnp.asarray(1.0))
    with pytest.raises(ValueError, match="underlying_ids"):
        price(bare, _model(), MARKET, method="mc", config=CFG, key=KEY)


def test_ids_are_validated_at_construction():
    with pytest.raises(ValueError, match="at least two"):
        _spread(("SPX",))
    with pytest.raises(ValueError, match="n_assets=3"):
        _worst(("SPX", "SX5E"))


def test_traced_trade_and_spot_gradients():
    inst, model = _spread(), _model()

    @eqx.filter_jit
    def value(i, m):
        return price(i, model, m, method="mc", config=CFG, key=KEY).price

    assert np.isfinite(float(value(inst, MARKET)))
    g = eqx.filter_grad(lambda m: value(inst, m))(MARKET).spots
    assert float(g["SPX"]) > 0.0 and float(g["SX5E"]) < 0.0   # long S1, short S2
    assert float(g["NKY"]) == 0.0


def test_missing_spot_is_named():
    with pytest.raises(KeyError, match="DAX"):
        price(_spread(("SPX", "DAX")), _model(), MARKET, method="mc", config=CFG, key=KEY)
