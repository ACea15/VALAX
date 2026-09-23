"""Model-market id-binding for short-rate models (pricing-ontology P5).

P5 ships the additive, backward-compatible half of the model-market
migration: short-rate models keep their embedded ``initial_curve`` (so every
existing pricer/test is untouched) but gain a static ``curve_id`` recording
the linkage, a ``from_market`` constructor that binds the model to a
``MarketState`` curve by id, and ``with_market_curve`` to re-bind under a
shocked market.  Full curve de-embedding (resolve-at-price-time) is P5.1.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest

from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.instruments.bonds import FixedRateBond
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.pde import PDEConfig


REF = jnp.array(739252, dtype=jnp.int32)
_T = jnp.arange(0, 11 * 365, 365, dtype=jnp.int32)


def _curve(level=0.03):
    return DiscountCurve(
        pillar_dates=REF + _T,
        discount_factors=jnp.exp(-level * _T / 365.0),
        reference_date=REF,
        day_count="act_365",
    )


def _hw(curve):
    return HullWhiteModel(
        mean_reversion=jnp.array(0.05), volatility=jnp.array(0.01),
        initial_curve=curve,
    )


def _g2pp(curve):
    return G2PPModel(
        mean_reversion_x=jnp.array(0.03), mean_reversion_y=jnp.array(0.10),
        volatility_x=jnp.array(0.010), volatility_y=jnp.array(0.008),
        correlation=jnp.array(-0.5), initial_curve=curve,
    )


# ── Backward compatibility ───────────────────────────────────────────

class TestBackwardCompatible:
    def test_hw_three_field_construction_defaults_sentinel(self):
        assert _hw(_curve()).curve_id == "_default_"

    def test_g2pp_six_field_construction_defaults_sentinel(self):
        assert _g2pp(_curve()).curve_id == "_default_"

    def test_hw_from_market_equals_direct_construction(self):
        curve = _curve()
        market = MarketState.single_curve(curve)
        from_market = HullWhiteModel.from_market(
            market, mean_reversion=jnp.array(0.05), volatility=jnp.array(0.01)
        )
        assert eqx.tree_equal(from_market, _hw(curve))


# ── from_market binding ──────────────────────────────────────────────

class TestFromMarket:
    def test_hw_binds_default_curve(self):
        market = MarketState.single_curve(_curve(0.03))
        m = HullWhiteModel.from_market(
            market, mean_reversion=jnp.array(0.05), volatility=jnp.array(0.01)
        )
        assert m.curve_id == "_default_"
        assert jnp.allclose(
            m.initial_curve.discount_factors, market.curve().discount_factors
        )

    def test_hw_binds_named_curve(self):
        ois = _curve(0.02)
        market = MarketState(curves=CurveGraph(curves={"USD.SOFR.OIS": ois}))
        m = HullWhiteModel.from_market(
            market, mean_reversion=jnp.array(0.05), volatility=jnp.array(0.01),
            curve_id="USD.SOFR.OIS",
        )
        assert m.curve_id == "USD.SOFR.OIS"
        assert jnp.allclose(m.initial_curve.discount_factors, ois.discount_factors)

    def test_g2pp_binds_named_curve(self):
        ois = _curve(0.025)
        market = MarketState(curves=CurveGraph(curves={"EUR.ESTR.OIS": ois}))
        m = G2PPModel.from_market(
            market,
            mean_reversion_x=jnp.array(0.03), mean_reversion_y=jnp.array(0.10),
            volatility_x=jnp.array(0.010), volatility_y=jnp.array(0.008),
            correlation=jnp.array(-0.5), curve_id="EUR.ESTR.OIS",
        )
        assert m.curve_id == "EUR.ESTR.OIS"
        assert jnp.allclose(m.initial_curve.discount_factors, ois.discount_factors)

    def test_from_market_missing_curve_raises(self):
        market = MarketState.single_curve(_curve())
        with pytest.raises(KeyError):
            HullWhiteModel.from_market(
                market, mean_reversion=jnp.array(0.05),
                volatility=jnp.array(0.01), curve_id="USD.SOFR.OIS",
            )


# ── Re-binding under a shocked market ────────────────────────────────

class TestRebind:
    def test_hw_with_market_curve_rebinds(self):
        base = MarketState.single_curve(_curve(0.03))
        model = HullWhiteModel.from_market(
            base, mean_reversion=jnp.array(0.05), volatility=jnp.array(0.01)
        )
        shocked = MarketState.single_curve(_curve(0.06))
        rebound = model.with_market_curve(shocked)
        assert rebound.curve_id == "_default_"
        assert jnp.allclose(
            rebound.initial_curve.discount_factors,
            _curve(0.06).discount_factors,
        )
        # Intrinsic params are preserved.
        assert float(rebound.mean_reversion) == float(model.mean_reversion)

    def test_g2pp_with_market_curve_preserves_id(self):
        ois = _curve(0.02)
        base = MarketState(curves=CurveGraph(curves={"USD.SOFR.OIS": ois}))
        model = G2PPModel.from_market(
            base,
            mean_reversion_x=jnp.array(0.03), mean_reversion_y=jnp.array(0.10),
            volatility_x=jnp.array(0.010), volatility_y=jnp.array(0.008),
            correlation=jnp.array(-0.5), curve_id="USD.SOFR.OIS",
        )
        shocked = MarketState(
            curves=CurveGraph(curves={"USD.SOFR.OIS": _curve(0.04)})
        )
        rebound = model.with_market_curve(shocked)
        assert rebound.curve_id == "USD.SOFR.OIS"
        assert jnp.allclose(
            rebound.initial_curve.discount_factors, _curve(0.04).discount_factors
        )


# ── Unified-path integration ─────────────────────────────────────────

class TestUnifiedPricing:
    def _bond(self):
        return FixedRateBond(
            payment_dates=REF + jnp.array([365, 730, 1095], jnp.int32),
            settlement_date=REF,
            coupon_rate=jnp.array(0.03),
            face_value=jnp.array(100.0),
            frequency=1,
        )

    def test_from_market_model_prices_through_unified_pde(self):
        curve = _curve(0.03)
        market = MarketState.single_curve(curve)
        model = HullWhiteModel.from_market(
            market, mean_reversion=jnp.array(0.05), volatility=jnp.array(0.01)
        )
        result = price(self._bond(), model, market, method="pde", config=PDEConfig())
        assert jnp.isfinite(result.price)
        # A from_market model and the equivalent direct model give the same price.
        direct = price(
            self._bond(), _hw(curve), market, method="pde", config=PDEConfig()
        )
        assert abs(float(result.price) - float(direct.price)) < 1e-10
