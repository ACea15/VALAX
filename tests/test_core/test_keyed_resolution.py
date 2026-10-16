"""D16: a trade resolves every market entry it reads, keyed by id.

``resolve`` binds the curves (and spots, FX spots, inflation curves) a trade
names into keyed views of ``ResolvedMarket``, decided by static selectors
only; the discount / projection pair is a view of the same objects; recipes
receive the keyed views only by naming them; a multi-index trade can never be
priced on a subset of its curves; and a model may declare a keyed family of
curves for one field.
"""

from __future__ import annotations

from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core import pricing as P
from valax.core.binding import bind_curves, curve_requirements
from valax.core.market_state import MarketState
from valax.core.pricing import PricingResult, price, register_pricer
from valax.core.resolve import (
    curve_ids_of,
    forward_id_of,
    fx_ids_of,
    inflation_ids_of,
    resolve,
    underlying_ids_of,
)
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.legs import FixedLeg, FloatingLeg, MultiLegSwap
from valax.instruments.rates import Swaption
from valax.pricing.mc import MCConfig

jax.config.update("jax_enable_x64", True)

REF = int(ymd_to_ordinal(2025, 1, 1))
OIS, F3M, F6M, EUR = "USD.SOFR.OIS", "USD.SOFR.3M", "USD.SOFR.6M", "EUR.ESTR.OIS"


def _curve(rate: float) -> DiscountCurve:
    pillars = jnp.array([REF + 365 * k for k in range(11)], dtype=jnp.int32)
    t = (pillars - REF) / 365.0
    return DiscountCurve(pillar_dates=pillars, discount_factors=jnp.exp(-rate * t),
                         reference_date=jnp.int32(REF), day_count="act_365")


MARKET = MarketState(
    curves=CurveGraph(curves={OIS: _curve(0.03), F3M: _curve(0.034),
                              F6M: _curve(0.036), EUR: _curve(0.02)}),
    spots={"SPX": jnp.asarray(100.0), "SX5E": jnp.asarray(50.0)},
    fx_spots={"EURUSD": jnp.asarray(1.1)},
    inflation_curves={"USD.CPI": jnp.asarray(300.0)},
)


def _dates(start_years, n=4, step=91):
    s = jnp.array([REF + 365 * start_years + step * k for k in range(n)], dtype=jnp.int32)
    return s, s + step


def _float_leg(forward_id, pay=False, spread=0.0):
    s, e = _dates(1)
    return FloatingLeg(accrual_start_dates=s, accrual_end_dates=e, payment_dates=e,
                       spread=jnp.asarray(spread), notional=jnp.asarray(1.0), pay=pay,
                       discount_id=OIS, forward_id=forward_id, day_count="act_365")


def _fixed_leg(pay=True):
    s, e = _dates(1)
    return FixedLeg(accrual_start_dates=s, accrual_end_dates=e, payment_dates=e,
                    fixed_rate=jnp.asarray(0.03), notional=jnp.asarray(1.0), pay=pay,
                    discount_id=OIS, day_count="act_365")


def _basis_swap():
    return MultiLegSwap(legs=(_float_leg(F3M), _float_leg(F6M, pay=True, spread=0.001)))


def _swaption(forward_id=F3M):
    return Swaption(expiry_date=jnp.int32(REF + 365),
                    fixed_dates=jnp.array([REF + 365 * k for k in (2, 3, 4)], dtype=jnp.int32),
                    strike=jnp.asarray(0.03), notional=jnp.asarray(1.0),
                    day_count="act_365", discount_id=OIS, forward_id=forward_id)


# ── Readers and views ────────────────────────────────────────────────


def test_primary_pair_is_a_view_of_the_same_objects():
    r = resolve(_swaption(), None, MARKET)
    assert tuple(r.curves.keys()) == (OIS, F3M)
    assert r.discount_curve is MARKET.curve(OIS)
    assert r.forward_curve is MARKET.curve(F3M)
    assert r.discount_id == OIS and r.forward_id == F3M
    single = resolve(_swaption(forward_id=None), None, MARKET)
    assert tuple(single.curves.keys()) == (OIS,) and single.forward_curve is None


def test_curves_is_the_sub_graph_the_trade_names():
    """3M and 6M variants read different curves; nothing else is resolved."""
    assert tuple(resolve(_swaption(F3M), None, MARKET).curves.keys()) == (OIS, F3M)
    assert tuple(resolve(_swaption(F6M), None, MARKET).curves.keys()) == (OIS, F6M)
    with pytest.raises(KeyError):
        resolve(_swaption(F3M), None, MARKET).curves[EUR]


def test_three_curve_trade_has_no_single_projection_curve():
    swap = _basis_swap()
    assert swap.curves_touched == (OIS, F3M, F6M)
    assert forward_id_of(swap) is None          # was F3M, silently dropping 6M
    r = resolve(swap, None, MARKET)
    assert tuple(r.curves.keys()) == (OIS, F3M, F6M)
    assert r.forward_curve is None
    two = MultiLegSwap(legs=(_fixed_leg(), _float_leg(F3M)))
    assert forward_id_of(two) == F3M and curve_ids_of(two) == (OIS, F3M)


def test_missing_curve_fails_at_resolve():
    with pytest.raises(KeyError, match="USD.SOFR.1M"):
        resolve(_swaption("USD.SOFR.1M"), None, MARKET)


class _Keyed(eqx.Module):
    """A test instrument naming every keyed slot."""

    notional: jnp.ndarray
    underlying_ids: tuple = eqx.field(static=True, default=("SX5E", "SPX"))
    inflation_id: Optional[str] = eqx.field(static=True, default="USD.CPI")
    discount_id: str = eqx.field(static=True, default=OIS)

    @property
    def fx_ids(self) -> tuple[str, ...]:
        return ("EURUSD",)


def test_other_keyed_slots():
    inst = _Keyed(notional=jnp.asarray(1.0))
    assert underlying_ids_of(inst) == ("SX5E", "SPX")
    assert fx_ids_of(inst) == ("EURUSD",) and inflation_ids_of(inst) == ("USD.CPI",)
    r = resolve(inst, None, MARKET)
    assert tuple(r.spots) == ("SX5E", "SPX")        # instrument order
    assert r.spot is MARKET.spots["SX5E"]           # primary = first
    assert r.fx_spots["EURUSD"] is MARKET.fx_spots["EURUSD"]
    assert r.inflation_curves["USD.CPI"] is MARKET.inflation_curves["USD.CPI"]
    with pytest.raises(KeyError, match="Inflation curve 'EUR.HICP'"):
        resolve(_Keyed(notional=jnp.asarray(1.0), inflation_id="EUR.HICP"), None, MARKET)


def test_rates_trade_on_a_market_with_equity_spots():
    """No underlying named, none under the sentinel: no spot is looked up."""
    r = resolve(_swaption(), None, MARKET)
    assert r.spot is None and r.spots == {} and r.dividend is None


# ── Recipe opt-in and the multi-index guard ──────────────────────────


def test_recipes_get_keyed_views_only_by_name():
    r = resolve(_Keyed(notional=jnp.asarray(1.0)), None, MARKET)

    def named(*, curves, spots, **kw): ...

    def var_kwargs(**kw): ...

    args = P._recipe_market_args(named, r)
    assert args["curves"] is r.curves
    np.testing.assert_array_equal(np.asarray(args["spots"]), [50.0, 100.0])
    assert set(P._recipe_market_args(var_kwargs, r)) == {"spot"}


@pytest.fixture
def basis_mc_recipes():
    """Two MC recipes for MultiLegSwap x a test model, removed afterwards."""

    class _Model(eqx.Module):
        scale: jnp.ndarray

    reg = P._legacy_registry("mc")
    seen = {}

    def subset_recipe(*, instrument, model, config, key, discount_curve,
                      forward_curve, **kw):
        return _Result(discount_curve(jnp.int32(REF + 365)))

    def full_recipe(*, instrument, model, config, key, curves, **kw):
        seen["curves"] = curves
        return _Result(sum(c(jnp.int32(REF + 365)) for c in curves.values()))

    yield _Model, reg, subset_recipe, full_recipe, seen
    reg.pop((MultiLegSwap, _Model), None)


class _Result(eqx.Module):
    price: jnp.ndarray
    stderr: Any = None


def test_multi_index_trade_on_a_pair_only_recipe_raises(basis_mc_recipes):
    Model, reg, subset_recipe, full_recipe, seen = basis_mc_recipes
    reg[(MultiLegSwap, Model)] = subset_recipe
    kw = dict(method="mc", config=MCConfig(n_paths=1, n_steps=1),
              key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="also touches"):
        price(_basis_swap(), Model(jnp.asarray(1.0)), MARKET, **kw)
    reg[(MultiLegSwap, Model)] = full_recipe
    res = price(_basis_swap(), Model(jnp.asarray(1.0)), MARKET, **kw)
    assert tuple(seen["curves"].keys()) == (OIS, F3M, F6M)
    expected = sum(MARKET.curve(c)(jnp.int32(REF + 365)) for c in (OIS, F3M, F6M))
    assert float(res.price) == float(expected)


# ── Keyed curve families on a model field ────────────────────────────


class _MultiCurveModel(eqx.Module):
    """A test model with one initial curve per modelled index."""

    initial_curves: Optional[dict] = None


@curve_requirements.register(_MultiCurveModel)
def _multi_requirements(model, instrument):
    return {"initial_curves": {"3M": F3M, "6M": F6M}}


class _Probe(eqx.Module):
    date: jnp.ndarray
    discount_id: str = eqx.field(static=True, default=OIS)


@pytest.fixture
def probe_pricer():
    key = (_Probe, _MultiCurveModel, "analytic")

    @register_pricer(*key)
    def _adapter(*, instrument, model, market, resolved, config, key):
        c = model.initial_curves
        return PricingResult(price=c["3M"](instrument.date) + 2.0 * c["6M"](instrument.date),
                             method="analytic")

    yield
    P._REGISTRY.pop(key, None)


def test_mapping_requirement_binds_a_dict_of_curves(probe_pricer):
    probe = _Probe(date=jnp.int32(REF + 730))
    model = _MultiCurveModel()
    got = price(probe, model, MARKET, method="analytic").price
    want = MARKET.curve(F3M)(probe.date) + 2.0 * MARKET.curve(F6M)(probe.date)
    assert float(got) == float(want)

    grads = eqx.filter_grad(
        lambda m: price(probe, model, m, method="analytic").price
    )(MARKET).curves.curves
    for cid in (F3M, F6M):
        assert np.any(np.asarray(grads[cid].discount_factors) != 0.0), cid
    assert np.all(np.asarray(grads[OIS].discount_factors) == 0.0)

    with pytest.raises(ValueError, match="pre-bound"):
        price(probe, bind_curves(model, MARKET), MARKET, method="analytic")
    single = bind_curves(model, MARKET.curve(OIS))
    for key in ("3M", "6M"):
        np.testing.assert_array_equal(
            np.asarray(single.initial_curves[key].discount_factors),
            np.asarray(MARKET.curve(OIS).discount_factors),
        )


# ── Trace-time contract ──────────────────────────────────────────────


def test_selection_is_static_under_jit_with_a_traced_instrument(probe_pricer):
    """The instrument is a traced argument; its static ids still pick curves."""
    model = _MultiCurveModel()

    @eqx.filter_jit
    def value(probe, market):
        return price(probe, model, market, method="analytic").price

    probe = _Probe(date=jnp.int32(REF + 730))
    want = MARKET.curve(F3M)(probe.date) + 2.0 * MARKET.curve(F6M)(probe.date)
    np.testing.assert_allclose(float(value(probe, MARKET)), float(want), rtol=1e-14)
    batch = eqx.filter_vmap(lambda d: value(_Probe(date=d), MARKET))(
        jnp.array([REF + 365, REF + 730], dtype=jnp.int32)
    )
    assert batch.shape == (2,) and np.all(np.isfinite(np.asarray(batch)))
