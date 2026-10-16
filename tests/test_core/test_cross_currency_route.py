"""``CrossCurrencySwap`` through ``price()``: two curves and a keyed FX spot.

The domestic curve is the trade's ``discount_id``, the foreign one its
``foreign_discount_id``, and the spot is ``MarketState.fx_spots`` under the
currency pair without the ``/`` — all chosen by static fields, so the route
works with the trade traced under ``jit`` and the gradient reaches exactly
the two curves and the spot.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core.market_state import MarketState
from valax.core.pricing import auto_method, price
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import CrossCurrencySwap
from valax.pricing.analytic.rates_exotics import cross_currency_swap_price

jax.config.update("jax_enable_x64", True)

REF = int(ymd_to_ordinal(2025, 1, 1))
USD, EUR, GBP = "USD.SOFR.OIS", "EUR.ESTR.OIS", "GBP.SONIA.OIS"


def _curve(rate: float) -> DiscountCurve:
    pillars = jnp.array([REF + 365 * k for k in range(8)], dtype=jnp.int32)
    t = (pillars - REF) / 365.0
    return DiscountCurve(pillar_dates=pillars, discount_factors=jnp.exp(-rate * t),
                         reference_date=jnp.int32(REF), day_count="act_365")


MARKET = MarketState(
    curves=CurveGraph(curves={USD: _curve(0.04), EUR: _curve(0.025), GBP: _curve(0.045)}),
    fx_spots={"EURUSD": jnp.asarray(1.08)},
)


def _swap(spread=-0.0015, exchange=True, **kw):
    fields = dict(currency_pair="EUR/USD", discount_id=USD, foreign_discount_id=EUR)
    fields.update(kw)
    return CrossCurrencySwap(
        start_date=jnp.int32(REF + 2),
        payment_dates=jnp.array([REF + 2 + 91 * k for k in range(1, 13)], dtype=jnp.int32),
        maturity_date=jnp.int32(REF + 2 + 91 * 12),
        domestic_notional=jnp.asarray(1.08e6), foreign_notional=jnp.asarray(1e6),
        basis_spread=jnp.asarray(spread), exchange_notional=exchange, **fields,
    )


@pytest.mark.parametrize("exchange", [True, False], ids=["exchange", "no_exchange"])
def test_route_is_the_kernel(exchange):
    swap = _swap(exchange=exchange)
    got = price(swap, None, MARKET)
    assert got.method == "analytic"
    want = cross_currency_swap_price(swap, MARKET.curve(USD), MARKET.curve(EUR),
                                     MARKET.fx_spots["EURUSD"])
    np.testing.assert_array_equal(np.asarray(got.price), np.asarray(want))
    assert float(got.price) != 0.0


def test_auto_is_analytic():
    assert auto_method(_swap(), None) == "analytic"


def test_traced_trade_and_gradients_on_exactly_its_inputs():
    swap = _swap(exchange=False)

    @eqx.filter_jit
    def value(s, m):
        return price(s, None, m).price

    assert np.isfinite(float(value(swap, MARKET)))
    g = eqx.filter_grad(lambda m: value(swap, m))(MARKET)
    assert np.any(np.asarray(g.curves.curves[USD].discount_factors) != 0.0)
    assert np.any(np.asarray(g.curves.curves[EUR].discount_factors) != 0.0)
    assert np.all(np.asarray(g.curves.curves[GBP].discount_factors) == 0.0)
    assert float(g.fx_spots["EURUSD"]) != 0.0


def test_the_selectors_pick_the_curves():
    """Anti-vacuity: a GBP foreign curve gives a different price."""
    gbp = _swap(foreign_discount_id=GBP, exchange=False)
    assert float(price(gbp, None, MARKET).price) != float(
        price(_swap(exchange=False), None, MARKET).price)


def test_batch_of_trades_sharing_ids():
    spreads = jnp.array([-0.002, -0.001, 0.0])
    batch = eqx.filter_vmap(lambda s: price(_swap(spread=s), None, MARKET).price)(spreads)
    assert batch.shape == (3,) and float(batch[2]) > float(batch[0])


def test_missing_fx_spot_and_forward_id_are_refused():
    with pytest.raises(KeyError, match="FX spot 'GBPUSD'"):
        price(_swap(currency_pair="GBP/USD"), None, MARKET)
    with pytest.raises(ValueError, match="forward_id"):
        price(_swap(forward_id=GBP), None, MARKET)
