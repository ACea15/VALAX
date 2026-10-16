"""``MultiLegSwap`` through ``price()``: every leg on its own curves (D16).

The route is the leg engine (``swap_pv``) on ``resolved.curves`` — the
sub-graph of the curves the legs name — so it must equal ``swap_pv`` on the
market's graph bit for bit, for a fixed-vs-float swap, a three-curve tenor
basis swap, an amortising swap and a seasoned swap (realised fixings from the
market). The curve choice is static, so it holds with the trade as a traced
argument of ``jit``, gradients reach exactly the named curves, and trades
sharing their ids batch under ``vmap``.
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
from valax.curves.fixings import FixingHistory, FixingSeries
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.legs import FixedLeg, FloatingLeg, MultiLegSwap
from valax.pricing.analytic.legs import swap_pv

jax.config.update("jax_enable_x64", True)

REF = int(ymd_to_ordinal(2025, 1, 1))
OIS, F3M, F6M, EUR = "USD.SOFR.OIS", "USD.SOFR.3M", "USD.SOFR.6M", "EUR.ESTR.OIS"


def _curve(short: float, slope: float) -> DiscountCurve:
    pillars = jnp.array([REF - 400 + 365 * k for k in range(14)], dtype=jnp.int32)
    t = (pillars - REF) / 365.0
    return DiscountCurve(pillar_dates=pillars,
                         discount_factors=jnp.exp(-(short + slope * t) * t),
                         reference_date=jnp.int32(REF), day_count="act_365")


_DAILY = jnp.arange(REF - 200, REF, dtype=jnp.int32)
MARKET = MarketState(
    curves=CurveGraph(curves={OIS: _curve(0.030, 0.001), F3M: _curve(0.033, 0.002),
                              F6M: _curve(0.035, 0.002), EUR: _curve(0.02, 0.0)}),
    fixings=FixingHistory(indices={"USD.SOFR": FixingSeries(
        fixing_dates=_DAILY, fixings=jnp.full(_DAILY.shape, 0.0325))}),
)
REF_DATE = MARKET.curve(OIS).reference_date


def _grid(start: int, n: int = 6, step: int = 182):
    s = jnp.array([start + step * k for k in range(n)], dtype=jnp.int32)
    return s, s + step


def _fixed(pay=True, amortisation=None, start=REF + 2):
    s, e = _grid(start)
    return FixedLeg(accrual_start_dates=s, accrual_end_dates=e, payment_dates=e,
                    fixed_rate=jnp.asarray(0.031), notional=jnp.asarray(1e6), pay=pay,
                    amortisation=amortisation, discount_id=OIS, day_count="act_365")


def _floating(forward_id, pay=False, spread=0.0, start=REF + 2, index_id="USD.SOFR.3M",
              amortisation=None):
    s, e = _grid(start)
    return FloatingLeg(accrual_start_dates=s, accrual_end_dates=e, payment_dates=e,
                       spread=jnp.asarray(spread), notional=jnp.asarray(1e6), pay=pay,
                       amortisation=amortisation, index_id=index_id,
                       discount_id=OIS, forward_id=forward_id, day_count="act_360")


AMORT = jnp.linspace(1.0, 0.5, 6)
CASES = {
    "fixed_float": lambda: MultiLegSwap(legs=(_fixed(), _floating(F3M))),
    "tenor_basis": lambda: MultiLegSwap(
        legs=(_floating(F3M), _floating(F6M, pay=True, spread=0.0012,
                                        index_id="USD.SOFR.6M"))),
    "amortising": lambda: MultiLegSwap(
        legs=(_fixed(amortisation=AMORT), _floating(F3M, amortisation=AMORT))),
    "seasoned": lambda: MultiLegSwap(
        legs=(_fixed(start=REF - 150),
              _floating(OIS, start=REF - 150, index_id="USD.SOFR"))),
}


@pytest.mark.parametrize("name", list(CASES))
def test_route_is_the_leg_engine(name):
    swap = CASES[name]()
    got = price(swap, None, MARKET)
    assert got.method == "analytic" and got.config is None
    want = swap_pv(swap, MARKET.curves, MARKET.fixings, REF_DATE)
    np.testing.assert_array_equal(np.asarray(got.price), np.asarray(want))
    assert float(got.price) != 0.0


def test_seasoning_reads_the_market_fixings():
    """Anti-vacuity: the realised fixings move the seasoned price."""
    swap = CASES["seasoned"]()
    with_fixings = float(price(swap, None, MARKET).price)
    bumped = eqx.tree_at(lambda m: m.fixings.indices["USD.SOFR"].fixings, MARKET,
                         jnp.full(_DAILY.shape, 0.05))
    assert float(price(swap, None, bumped).price) != with_fixings


def test_auto_is_analytic():
    assert auto_method(CASES["tenor_basis"](), None) == "analytic"


def _value(swap, market):
    return price(swap, None, market).price


def test_traced_trade_under_jit_and_gradients_on_exactly_its_curves():
    swap = CASES["tenor_basis"]()
    jitted = eqx.filter_jit(_value)
    assert np.isfinite(float(jitted(swap, MARKET)))
    grads = eqx.filter_grad(lambda m: jitted(swap, m))(MARKET).curves.curves
    for cid in (OIS, F3M, F6M):
        assert np.any(np.asarray(grads[cid].discount_factors) != 0.0), cid
    assert np.all(np.asarray(grads[EUR].discount_factors) == 0.0)


def test_3m_and_6m_variants_read_different_curves():
    on_3m = MultiLegSwap(legs=(_fixed(), _floating(F3M)))
    on_6m = MultiLegSwap(legs=(_fixed(), _floating(F6M)))
    assert float(_value(on_3m, MARKET)) != float(_value(on_6m, MARKET))
    g3 = eqx.filter_grad(lambda m: _value(on_3m, m))(MARKET).curves.curves
    assert np.all(np.asarray(g3[F6M].discount_factors) == 0.0)
    assert np.any(np.asarray(g3[F3M].discount_factors) != 0.0)


def test_trades_sharing_ids_batch_under_vmap():
    spreads = jnp.array([0.0, 0.001, 0.002])

    def basis(spread):
        return MultiLegSwap(legs=(_floating(F3M), _floating(
            F6M, pay=True, spread=spread, index_id="USD.SOFR.6M")))

    batch = eqx.filter_vmap(lambda s: _value(basis(s), MARKET))(spreads)
    for k, s in enumerate(spreads):
        np.testing.assert_allclose(float(batch[k]), float(_value(basis(s), MARKET)),
                                   rtol=1e-13)
    assert float(batch[0]) > float(batch[2])   # paying a larger spread


def test_missing_leg_curve_names_the_curve():
    swap = MultiLegSwap(legs=(_fixed(), _floating("USD.SOFR.1M")))
    with pytest.raises(KeyError, match="USD.SOFR.1M"):
        price(swap, None, MARKET)
