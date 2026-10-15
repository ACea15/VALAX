"""Model-market curve binding for short-rate models (pricing-ontology P5.1).

Short-rate models are curve-free pytrees: ``initial_curve`` defaults to
``None`` and a static ``curve_id`` optionally pins the market curve to bind.
:func:`valax.core.pricing.price` binds the curve from the ``MarketState`` at
pricing time; kernel-level callers bind explicitly with
:func:`valax.core.binding.bind_curves`.  The price-time contract itself
(pre-bound refusal, gradients, batching) is covered in
``tests/test_core/test_curve_binding.py``; this file covers the model-side
defaults and explicit (re-)binding.
"""

import jax.numpy as jnp
import pytest

from valax.core.binding import bind_curves, curve_requirements
from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.instruments.bonds import FixedRateBond
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.pde import PDEConfig
from valax.pricing.pde.dispatch import pde_price_dispatch


REF = jnp.array(739252, dtype=jnp.int32)
_T = jnp.arange(0, 11 * 365, 365, dtype=jnp.int32)


def _curve(level=0.03):
    return DiscountCurve(
        pillar_dates=REF + _T,
        discount_factors=jnp.exp(-level * _T / 365.0),
        reference_date=REF,
        day_count="act_365",
    )


def _hw(**kw):
    return HullWhiteModel(
        mean_reversion=jnp.array(0.05), volatility=jnp.array(0.01), **kw
    )


def _g2pp(**kw):
    return G2PPModel(
        mean_reversion_x=jnp.array(0.03), mean_reversion_y=jnp.array(0.10),
        volatility_x=jnp.array(0.010), volatility_ratio=jnp.array(0.8),
        correlation=jnp.array(-0.5), **kw,
    )


def _bond():
    return FixedRateBond(
        payment_dates=REF + jnp.array([365, 730, 1095], jnp.int32),
        settlement_date=REF,
        coupon_rate=jnp.array(0.03),
        face_value=jnp.array(100.0),
        frequency=1,
    )


# ── Defaults: curve-free models ──────────────────────────────────────

class TestDefaults:
    def test_hw_is_unbound_with_no_pinned_curve(self):
        m = _hw()
        assert m.initial_curve is None
        assert m.curve_id is None

    def test_g2pp_is_unbound_with_no_pinned_curve(self):
        m = _g2pp()
        assert m.initial_curve is None
        assert m.curve_id is None

    def test_unpinned_model_requires_instrument_discount_curve(self):
        assert curve_requirements(_hw(), _bond()) == {"initial_curve": "_default_"}


# ── Explicit binding from a MarketState ──────────────────────────────

class TestBindCurves:
    def test_hw_binds_default_curve(self):
        market = MarketState.single_curve(_curve(0.03))
        m = bind_curves(_hw(), market, _bond())
        assert m.curve_id is None
        assert jnp.allclose(
            m.initial_curve.discount_factors, market.curve().discount_factors
        )

    def test_hw_binds_named_curve(self):
        ois = _curve(0.02)
        market = MarketState(curves=CurveGraph(curves={"USD.SOFR.OIS": ois}))
        m = bind_curves(_hw(curve_id="USD.SOFR.OIS"), market)
        assert m.curve_id == "USD.SOFR.OIS"
        assert jnp.allclose(m.initial_curve.discount_factors, ois.discount_factors)

    def test_g2pp_binds_named_curve(self):
        ois = _curve(0.025)
        market = MarketState(curves=CurveGraph(curves={"EUR.ESTR.OIS": ois}))
        m = bind_curves(_g2pp(curve_id="EUR.ESTR.OIS"), market)
        assert m.curve_id == "EUR.ESTR.OIS"
        assert jnp.allclose(m.initial_curve.discount_factors, ois.discount_factors)

    def test_missing_curve_raises(self):
        market = MarketState.single_curve(_curve())
        with pytest.raises(KeyError):
            bind_curves(_hw(curve_id="USD.SOFR.OIS"), market)


# ── Re-binding under a shocked market ────────────────────────────────

class TestRebind:
    def test_hw_bind_curves_overwrites_a_bound_curve(self):
        base = MarketState.single_curve(_curve(0.03))
        model = bind_curves(_hw(), base, _bond())
        shocked = MarketState.single_curve(_curve(0.06))
        rebound = bind_curves(model, shocked, _bond())
        assert rebound.curve_id is None
        assert jnp.allclose(
            rebound.initial_curve.discount_factors,
            _curve(0.06).discount_factors,
        )
        # Intrinsic params are preserved.
        assert float(rebound.mean_reversion) == float(model.mean_reversion)

    def test_g2pp_rebind_preserves_id(self):
        ois = _curve(0.02)
        base = MarketState(curves=CurveGraph(curves={"USD.SOFR.OIS": ois}))
        model = bind_curves(_g2pp(curve_id="USD.SOFR.OIS"), base)
        shocked = MarketState(
            curves=CurveGraph(curves={"USD.SOFR.OIS": _curve(0.04)})
        )
        rebound = bind_curves(model, shocked)
        assert rebound.curve_id == "USD.SOFR.OIS"
        assert jnp.allclose(
            rebound.initial_curve.discount_factors, _curve(0.04).discount_factors
        )


# ── Unified-path integration ─────────────────────────────────────────

class TestUnifiedPricing:
    def test_unbound_model_prices_through_unified_pde(self):
        curve = _curve(0.03)
        market = MarketState.single_curve(curve)
        result = price(_bond(), _hw(), market, method="pde", config=PDEConfig())
        assert jnp.isfinite(result.price)
        # price() binding equals an explicit kernel-level binding.
        direct = pde_price_dispatch(_bond(), bind_curves(_hw(), curve), PDEConfig())
        assert abs(float(result.price) - float(direct.price)) < 1e-10
