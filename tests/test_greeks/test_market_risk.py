"""``market_risk``: the price's sensitivity to every declared market entry.

The report is keyed by ``declared_entries`` — every curve the instrument and
its model read (not only the discount curve), its spots, dividends, FX spots,
inflation curves and the model's surface — and each value is the matching
leaf of the full market gradient, from one backward pass. Entries the trade
does not declare are not reported, and their gradient is exactly zero.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from valax.core import MarketState, declared_entries
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.greeks import market_gradient, market_risk
from valax.instruments.legs import FloatingLeg, MultiLegSwap
from valax.instruments.options import EuropeanOption
from valax.instruments.rates import CrossCurrencySwap, Swaption
from valax.models.black_scholes import BlackScholesModel
from valax.models.hull_white import HullWhiteModel

jax.config.update("jax_enable_x64", True)

REF = int(ymd_to_ordinal(2025, 1, 1))
OIS, F3M, F6M, EUR = "USD.SOFR.OIS", "USD.SOFR.3M", "USD.SOFR.6M", "EUR.ESTR.OIS"


def _curve(rate: float) -> DiscountCurve:
    pillars = jnp.array([REF + 365 * k for k in range(9)], dtype=jnp.int32)
    t = (pillars - REF) / 365.0
    return DiscountCurve(pillar_dates=pillars, discount_factors=jnp.exp(-rate * t),
                         reference_date=jnp.int32(REF), day_count="act_365")


MARKET = MarketState(
    curves=CurveGraph(curves={OIS: _curve(0.03), F3M: _curve(0.034),
                              F6M: _curve(0.036), EUR: _curve(0.02)}),
    fx_spots={"EURUSD": jnp.asarray(1.1)},
)


def _leg(forward_id, **kw):
    starts = jnp.array([REF + 2 + 182 * k for k in range(8)], dtype=jnp.int32)
    return FloatingLeg(accrual_start_dates=starts, accrual_end_dates=starts + 182,
                       payment_dates=starts + 182, notional=jnp.asarray(1e6),
                       compounding="simple", fixing_dates=starts, index_id=forward_id,
                       discount_id=OIS, forward_id=forward_id, day_count="act_360", **kw)


def _check_against_full_gradient(risk, grad):
    for cid, deltas in risk.curves.items():
        np.testing.assert_array_equal(np.asarray(deltas),
                                      np.asarray(grad.curves.curves[cid].discount_factors))
    for cid, curve_grad in grad.curves.curves.items():
        if cid not in risk.curves:
            assert np.all(np.asarray(curve_grad.discount_factors) == 0.0), cid


def test_every_curve_of_a_tenor_basis_swap_is_reported():
    swap = MultiLegSwap(legs=(_leg(F3M, spread=jnp.asarray(0.0)),
                              _leg(F6M, spread=jnp.asarray(-0.001), pay=True)))
    risk = market_risk(swap, None, MARKET)
    assert tuple(risk.curves) == (OIS, F3M, F6M)
    for cid in (OIS, F3M, F6M):
        assert np.any(np.asarray(risk.curves[cid]) != 0.0), cid
    _check_against_full_gradient(risk, market_gradient(swap, None, MARKET))
    assert risk.fx_spots == {} and risk.spots == {}


def test_a_models_own_curve_is_declared_and_reported():
    """A Hull-White model pinned to another curve reads it: it is reported."""
    swaption = Swaption(expiry_date=jnp.int32(REF + 365),
                        fixed_dates=jnp.array([REF + 365 * k for k in (2, 3, 4)],
                                              dtype=jnp.int32),
                        strike=jnp.asarray(0.03), notional=jnp.asarray(1e6),
                        day_count="act_365", discount_id=OIS)
    model = HullWhiteModel(mean_reversion=jnp.asarray(0.05),
                           volatility=jnp.asarray(0.01), curve_id=F6M)
    assert declared_entries(swaption, model, MARKET).curves == (OIS, F6M)
    risk = market_risk(swaption, model, MARKET)
    assert np.any(np.asarray(risk.curves[F6M]) != 0.0)
    _check_against_full_gradient(risk, market_gradient(swaption, model, MARKET))


def test_cross_currency_reports_both_curves_and_the_fx_spot():
    xccy = CrossCurrencySwap(
        start_date=jnp.int32(REF + 2),
        payment_dates=jnp.array([REF + 2 + 91 * k for k in range(1, 9)], dtype=jnp.int32),
        maturity_date=jnp.int32(REF + 2 + 91 * 8),
        domestic_notional=jnp.asarray(1.1e6), foreign_notional=jnp.asarray(1e6),
        basis_spread=jnp.asarray(-0.001), exchange_notional=False,
        currency_pair="EUR/USD", discount_id=OIS, foreign_discount_id=EUR)
    risk = market_risk(xccy, None, MARKET)
    assert tuple(risk.curves) == (OIS, EUR) and tuple(risk.fx_spots) == ("EURUSD",)
    grad = market_gradient(xccy, None, MARKET)
    assert float(risk.fx_spots["EURUSD"]) == float(grad.fx_spots["EURUSD"]) != 0.0
    _check_against_full_gradient(risk, grad)


def test_equity_reports_spot_dividend_surface_and_curve():
    market = MarketState.single_curve(_curve(0.03), spot=jnp.asarray(100.0),
                                      vol=jnp.asarray(0.2), dividend=jnp.asarray(0.01))
    option = EuropeanOption(strike=jnp.asarray(100.0), expiry=jnp.asarray(1.0),
                            is_call=True)
    model = BlackScholesModel()            # vol, rate, dividend from the market
    risk = market_risk(option, model, market)
    grad = market_gradient(option, model, market)
    assert tuple(risk.spots) == tuple(risk.dividends) == tuple(risk.surfaces) == ("_default_",)
    assert float(risk.spots["_default_"]) == float(grad.spots["_default_"]) > 0.0
    assert float(risk.dividends["_default_"]) < 0.0
    assert float(risk.surfaces["_default_"].vol) > 0.0          # vega
    assert np.any(np.asarray(risk.curves["_default_"]) != 0.0)  # rho via the curve


def test_report_works_under_jit():
    swap = MultiLegSwap(legs=(_leg(F3M, spread=jnp.asarray(0.0)),
                              _leg(F6M, spread=jnp.asarray(-0.001), pay=True)))
    risk = eqx.filter_jit(lambda s, m: market_risk(s, None, m))(swap, MARKET)
    # (dict keys come back sorted through jit's pytree round trip)
    assert set(risk.curves) == {OIS, F3M, F6M} and np.isfinite(float(risk.price))
