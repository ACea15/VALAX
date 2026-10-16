"""Inflation through ``price()``: keyed inflation curves (D16).

ZCIS and YYIS (``model=None``) and inflation caps / floors
(``ImpliedVolModel``) read the inflation curve filed under their
``inflation_id`` and discount on their ``discount_id`` — static selectors, so
the routes work with the trade traced under ``jit``, and the gradient reaches
the named inflation curve, the named discount curve and (for the cap) the vol
surface.
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
from valax.curves.inflation import from_zc_rates
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.instruments.inflation import (
    InflationCapFloor,
    YearOnYearInflationSwap,
    ZeroCouponInflationSwap,
)
from valax.models import ImpliedVolModel
from valax.pricing.analytic.inflation import (
    inflation_cap_floor_price_black76,
    yyis_price,
    zcis_price,
)
from valax.surfaces import ConstantVol

jax.config.update("jax_enable_x64", True)

REF = int(ymd_to_ordinal(2025, 1, 1))
USD, EUR = "USD.SOFR.OIS", "EUR.ESTR.OIS"
CPI, HICP = "USD.CPI", "EUR.HICP"
VOL, NORMAL = "USD.CPI.VOL", "USD.CPI.VOL.N"


def _curve(rate: float) -> DiscountCurve:
    pillars = jnp.array([REF + 365 * k for k in range(12)], dtype=jnp.int32)
    t = (pillars - REF) / 365.0
    return DiscountCurve(pillar_dates=pillars, discount_factors=jnp.exp(-rate * t),
                         reference_date=jnp.int32(REF), day_count="act_365")


def _cpi(rate: float, base: float):
    pillars = jnp.array([REF + 365 * k for k in range(1, 12)], dtype=jnp.int32)
    return from_zc_rates(jnp.int32(REF), pillars,
                         jnp.full(pillars.shape, rate), jnp.asarray(base))


MARKET = MarketState(
    curves=CurveGraph(curves={USD: _curve(0.04), EUR: _curve(0.025)}),
    inflation_curves={CPI: _cpi(0.025, 300.0), HICP: _cpi(0.02, 120.0)},
    surfaces={VOL: ConstantVol(jnp.asarray(0.3)),
              NORMAL: ConstantVol(jnp.asarray(0.01), is_normal=True)},
)
PAYS = jnp.array([REF + 365 * k for k in range(1, 6)], dtype=jnp.int32)


def _ids(kw):
    return {"discount_id": USD, "inflation_id": CPI, **kw}


def _zcis(**kw):
    return ZeroCouponInflationSwap(
        effective_date=jnp.int32(REF), maturity_date=jnp.int32(REF + 5 * 365),
        fixed_rate=jnp.asarray(0.022), notional=jnp.asarray(1e6),
        base_cpi=jnp.asarray(300.0), **_ids(kw))


def _yyis(**kw):
    return YearOnYearInflationSwap(
        effective_date=jnp.int32(REF), payment_dates=PAYS, fixed_rate=jnp.asarray(0.022),
        notional=jnp.asarray(1e6), base_cpi=jnp.asarray(300.0), **_ids(kw))


def _cap(**kw):
    return InflationCapFloor(
        effective_date=jnp.int32(REF), payment_dates=PAYS, strike=jnp.asarray(0.024),
        notional=jnp.asarray(1e6), base_cpi=jnp.asarray(300.0), **_ids(kw))


def test_swaps_are_the_kernels():
    infl, disc = MARKET.inflation_curves[CPI], MARKET.curve(USD)
    for inst, kernel in ((_zcis(), zcis_price), (_yyis(), yyis_price)):
        got = price(inst, None, MARKET)
        assert got.method == "analytic"
        np.testing.assert_array_equal(np.asarray(got.price),
                                      np.asarray(kernel(inst, infl, disc)))
        assert float(got.price) != 0.0


@pytest.mark.parametrize("is_cap", [True, False], ids=["cap", "floor"])
def test_cap_floor_is_the_black76_kernel(is_cap):
    cap = _cap(is_cap=is_cap)
    got = price(cap, ImpliedVolModel(VOL), MARKET).price
    want = inflation_cap_floor_price_black76(
        cap, MARKET.inflation_curves[CPI], MARKET.curve(USD), jnp.asarray(0.3))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert float(got) > 0.0


def test_cap_reads_the_vol_at_each_fixing_time():
    """A surface varying with expiry gives each caplet its own vol."""

    class _TermVol(eqx.Module):
        level: jnp.ndarray

        def __call__(self, strike, expiry, **kw):
            return self.level + 0.01 * expiry

    market = eqx.tree_at(lambda m: m.surfaces, MARKET,
                         {**MARKET.surfaces, "USD.CPI.VOL.TERM": _TermVol(jnp.asarray(0.25))})
    cap = _cap()
    got = price(cap, ImpliedVolModel("USD.CPI.VOL.TERM"), market).price
    starts = jnp.concatenate([cap.effective_date[None], cap.payment_dates[:-1]])
    vols = 0.25 + 0.01 * year_fraction(REF, starts, cap.day_count)
    want = inflation_cap_floor_price_black76(
        cap, MARKET.inflation_curves[CPI], MARKET.curve(USD), vols)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_normal_vol_is_refused():
    with pytest.raises(ValueError, match="normal"):
        price(_cap(), ImpliedVolModel(NORMAL), MARKET)


def test_auto_is_analytic():
    assert auto_method(_zcis(), None) == "analytic"
    assert auto_method(_cap(), ImpliedVolModel(VOL)) == "analytic"


def test_selectors_pick_the_curves():
    """Anti-vacuity: the EUR / HICP variant reads the other curves."""
    eur = _yyis(discount_id=EUR, inflation_id=HICP)
    assert float(price(eur, None, MARKET).price) != float(price(_yyis(), None, MARKET).price)


def test_traced_trade_and_gradients_on_exactly_its_inputs():
    cap = _cap()

    @eqx.filter_jit
    def value(c, m):
        return price(c, ImpliedVolModel(VOL), m).price

    assert np.isfinite(float(value(cap, MARKET)))
    g = eqx.filter_grad(lambda m: value(cap, m))(MARKET)
    assert np.any(np.asarray(g.inflation_curves[CPI].forward_cpis) != 0.0)
    assert np.all(np.asarray(g.inflation_curves[HICP].forward_cpis) == 0.0)
    assert np.any(np.asarray(g.curves.curves[USD].discount_factors) != 0.0)
    assert np.all(np.asarray(g.curves.curves[EUR].discount_factors) == 0.0)
    assert float(g.surfaces[VOL].vol) != 0.0


def test_batch_of_trades_sharing_ids():
    rates = jnp.array([0.02, 0.022, 0.024])
    batch = eqx.filter_vmap(
        lambda k: price(eqx.tree_at(lambda s: s.fixed_rate, _zcis(), k), None, MARKET).price
    )(rates)
    assert batch.shape == (3,) and float(batch[0]) > float(batch[2])


def test_missing_inflation_curve_is_named():
    with pytest.raises(KeyError, match="Inflation curve 'GBP.RPI'"):
        price(_zcis(inflation_id="GBP.RPI"), None, MARKET)
