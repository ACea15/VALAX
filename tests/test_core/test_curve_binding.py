"""Curve fields of the short-rate models are bound from the market by price().

The de-embedding contract (``valax.core.binding.curve_requirements``):

* HW / G2++ / Cheyette carry ``initial_curve=None`` and are bound at pricing
  time to ``market.curve(model.curve_id or instrument.discount_id)``;
* a pre-bound curve field is refused by ``price`` (single source of truth);
* the model gradient has no curve leaf, and the market gradient equals the
  total curve delta of the old embedded-curve formulation;
* curve-free models stack under ``eqx.filter_vmap``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core.binding import bind_curves, curve_requirements
from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.cheyette import CheyetteModel
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.pde.config import PDEConfig
from valax.pricing.pde.dispatch import pde_price_dispatch

jax.config.update("jax_enable_x64", True)

REF = 45_000
CFG = PDEConfig(n_spot=160, n_time=120)


def _curve(rate: float, ref: int = REF) -> DiscountCurve:
    pillars = jnp.array([ref + 365 * k for k in range(0, 12)], dtype=jnp.int32)
    t = (pillars - ref) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-(rate + 0.002 * t) * t),
        reference_date=jnp.int32(ref),
        day_count="act_365",
    )


def _swaption(**selectors) -> Swaption:
    return Swaption(
        expiry_date=jnp.int32(REF + 2 * 365),
        fixed_dates=jnp.array([REF + 365 * k for k in range(3, 8)], dtype=jnp.int32),
        strike=jnp.asarray(0.035),
        notional=jnp.asarray(1.0),
        day_count="act_365",
        **selectors,
    )


def _hw(**kw) -> HullWhiteModel:
    return HullWhiteModel(
        mean_reversion=jnp.asarray(0.05), volatility=jnp.asarray(0.01), **kw
    )


@pytest.fixture(scope="module")
def market():
    return MarketState(curves=CurveGraph(curves={
        "USD.SOFR.OIS": _curve(0.03), "USD.ALT.OIS": _curve(0.04),
    }))


class TestRequirements:
    def test_curve_id_else_instrument_discount(self):
        sw = _swaption(discount_id="USD.SOFR.OIS")
        assert curve_requirements(_hw(), sw) == {"initial_curve": "USD.SOFR.OIS"}
        pinned = _hw(curve_id="USD.ALT.OIS")
        assert curve_requirements(pinned, sw) == {"initial_curve": "USD.ALT.OIS"}
        for model in (
            G2PPModel(mean_reversion_x=jnp.asarray(0.5), mean_reversion_y=jnp.asarray(0.1),
                      volatility_x=jnp.asarray(0.01), volatility_ratio=jnp.asarray(0.8),
                      correlation=jnp.asarray(-0.5)),
            CheyetteModel(mean_reversion=jnp.asarray(0.05), volatility=jnp.asarray(0.01),
                          skew=jnp.asarray(0.0)),
        ):
            assert curve_requirements(model, sw) == {"initial_curve": "USD.SOFR.OIS"}

    def test_models_without_curves_declare_none(self):
        from valax.models.black_scholes import BlackScholesModel

        assert curve_requirements(BlackScholesModel(vol=0.2), None) == {}


class TestBinding:
    def test_price_binds_the_market_curve(self, market):
        sw = _swaption(discount_id="USD.SOFR.OIS")
        got = price(sw, _hw(), market, method="pde", config=CFG).price
        ref = pde_price_dispatch(
            sw, bind_curves(_hw(), market.curve("USD.SOFR.OIS")), CFG
        ).price
        np.testing.assert_allclose(got, ref, rtol=1e-14)

    def test_curve_id_pins_a_different_curve(self, market):
        sw = _swaption(discount_id="USD.SOFR.OIS")
        pinned = price(sw, _hw(curve_id="USD.ALT.OIS"), market, method="pde", config=CFG)
        default = price(sw, _hw(), market, method="pde", config=CFG)
        ref = pde_price_dispatch(
            sw, bind_curves(_hw(), market.curve("USD.ALT.OIS")), CFG
        ).price
        np.testing.assert_allclose(pinned.price, ref, rtol=1e-14)
        assert abs(float(pinned.price - default.price)) > 1e-5

    def test_pre_bound_curve_is_refused(self, market):
        sw = _swaption(discount_id="USD.SOFR.OIS")
        bound = bind_curves(_hw(), market.curve("USD.SOFR.OIS"))
        with pytest.raises(ValueError, match="pre-bound"):
            price(sw, bound, market, method="pde", config=CFG)

    def test_unbound_kernel_call_raises(self):
        with pytest.raises(ValueError, match="not bound to a curve"):
            hw_swaption_price(_swaption(), _hw())


class TestRisk:
    def test_model_gradient_has_no_curve_leaf(self, market):
        sw = _swaption(discount_id="USD.SOFR.OIS")
        model_bar, market_bar = eqx.filter_grad(
            lambda mm: price(sw, mm[0], mm[1], method="pde", config=CFG).price
        )((_hw(), market))
        assert model_bar.initial_curve is None
        assert jnp.isfinite(model_bar.volatility[0]) and float(model_bar.volatility[0]) != 0.0
        assert float(jnp.max(jnp.abs(
            market_bar.curve("USD.SOFR.OIS").discount_factors
        ))) > 0.0

    def test_market_delta_equals_embedded_curve_delta(self, market):
        """The curve delta moves from the model to the market, unchanged."""
        sw = _swaption(discount_id="USD.SOFR.OIS")
        curve = market.curve("USD.SOFR.OIS")

        def embedded(c):  # the old formulation: the curve inside the model
            return pde_price_dispatch(sw, bind_curves(_hw(), c), CFG).price

        old = eqx.filter_grad(embedded)(curve).discount_factors
        new = eqx.filter_grad(
            lambda m: price(sw, _hw(), m, method="pde", config=CFG).price
        )(market).curve("USD.SOFR.OIS").discount_factors
        np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-14)


class TestBatching:
    def test_curve_free_models_stack_under_filter_vmap(self, market):
        sw = BermudanSwaption(
            exercise_dates=jnp.array([REF + 365 * k for k in (2, 3, 4)], dtype=jnp.int32),
            fixed_dates=jnp.array([REF + 365 * k for k in range(3, 8)], dtype=jnp.int32),
            strike=jnp.asarray(0.035),
            notional=jnp.asarray(1.0),
            day_count="act_365",
        )
        mkt = MarketState.single_curve(market.curve("USD.SOFR.OIS"))
        models = eqx.filter_vmap(
            lambda v: HullWhiteModel(mean_reversion=jnp.asarray(0.05), volatility=v)
        )(jnp.array([0.008, 0.010, 0.012]))
        assert models.initial_curve is None
        prices = eqx.filter_vmap(
            lambda m: price(sw, m, mkt, method="pde", config=CFG).price
        )(models)
        assert prices.shape == (3,)
        assert bool(jnp.all(jnp.diff(prices) > 0.0))  # vega > 0
