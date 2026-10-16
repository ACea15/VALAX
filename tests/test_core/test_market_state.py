"""Tests for the MarketState calibrated-market container (ontology P2)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.core.market_state import DEFAULT_ID, MarketState, validate_surface_id
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.market.data import MarketData
from valax.surfaces.constant import ConstantVol


REF = jnp.array(739252, dtype=jnp.int32)  # 2025-01-01 ordinal-ish


def _curve(level=0.05):
    times = jnp.array([0, 365, 730], dtype=jnp.int32)
    return DiscountCurve(
        pillar_dates=REF + times,
        discount_factors=jnp.exp(-level * times / 365.0),
        reference_date=REF,
        day_count="act_365",
    )


def _single():
    return MarketState.single_curve(
        _curve(),
        spot=jnp.array(100.0),
        vol=jnp.array(0.2),
        dividend=jnp.array(0.01),
    )


# ── Construction and zero-config single-curve use ────────────────────

class TestSingleCurve:
    def test_defaults_resolve_without_ids(self):
        market = _single()
        assert float(market.spot()) == 100.0
        assert float(market.dividend()) == 0.01
        assert float(market.surface().vol) == 0.2
        assert market.discount_curve is market.curves[DEFAULT_ID]

    def test_scalar_vol_is_wrapped_as_constant_surface(self):
        market = _single()
        assert isinstance(market.surface(), ConstantVol)

    def test_duck_typed_surface_passes_through(self):
        surface = ConstantVol(vol=jnp.array(0.3), is_normal=True)
        market = MarketState.single_curve(_curve(), vol=surface)
        assert market.surface() is surface

    def test_empty_optional_fields(self):
        market = MarketState.single_curve(_curve())
        assert market.spots == {}
        assert float(market.dividend()) == 0.0
        assert market.reference_date == REF


# ── Pytree / JAX behaviour ───────────────────────────────────────────

class TestPytree:
    def test_round_trip(self):
        market = _single()
        leaves, treedef = jax.tree_util.tree_flatten(market)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert float(restored.spot()) == 100.0
        assert jnp.array_equal(
            restored.discount_curve.discount_factors,
            market.discount_curve.discount_factors,
        )

    def test_grad_flows_to_every_leaf(self):
        market = _single()

        def value(m):
            return m.spot() * m.discount_curve(REF + 365) * m.surface()()

        grads = eqx.filter_grad(value)(market)
        assert float(grads.spots[DEFAULT_ID]) != 0.0
        assert float(grads.surfaces[DEFAULT_ID].vol) != 0.0
        assert jnp.any(grads.curves[DEFAULT_ID].discount_factors != 0.0)

    def test_jit_smoke(self):
        market = _single()

        @eqx.filter_jit
        def pv(m):
            return m.spot() * m.discount_curve(REF + 730)

        assert jnp.isfinite(pv(market))

    def test_shocking_returns_a_new_state(self):
        market = _single()
        shocked = eqx.tree_at(lambda m: m.spots[DEFAULT_ID], market, jnp.array(110.0))
        assert float(market.spot()) == 100.0
        assert float(shocked.spot()) == 110.0

    def test_non_payer_among_payers_has_zero_dividend(self):
        """An asset with a spot but no dividend entry is a non-payer even when
        other assets carry dividends (found by the route contract test)."""
        m = MarketState(
            curves=CurveGraph(curves={DEFAULT_ID: _curve()}),
            spots={"SPX": jnp.array(100.0), "SX5E": jnp.array(50.0)},
            dividends={"SPX": jnp.array(0.015)},
        )
        assert float(m.dividend("SX5E")) == 0.0
        assert float(m.dividend("SPX")) == pytest.approx(0.015)
        with pytest.raises(KeyError):
            m.dividend("NKY")       # an unknown asset is still an error


# ── MarketData compatibility view (Q4) ───────────────────────────────

class TestMarketDataView:
    def test_round_trip_is_lossless(self):
        original = MarketData(
            spots=jnp.array([100.0, 50.0]),
            vols=jnp.array([0.2, 0.3]),
            dividends=jnp.array([0.01, 0.02]),
            discount_curve=_curve(),
        )
        state = MarketState.from_market_data(original, asset_ids=("SPX", "NDX"))
        back = state.to_market_data()
        assert jnp.allclose(back.spots, original.spots)
        assert jnp.allclose(back.vols, original.vols)
        assert jnp.allclose(back.dividends, original.dividends)
        assert jnp.allclose(
            back.discount_curve.discount_factors,
            original.discount_curve.discount_factors,
        )

    def test_asset_ids_define_vector_order(self):
        original = MarketData(
            spots=jnp.array([100.0, 50.0]),
            vols=jnp.array([0.2, 0.3]),
            dividends=jnp.array([0.01, 0.02]),
            discount_curve=_curve(),
        )
        state = MarketState.from_market_data(original, asset_ids=("SPX", "NDX"))
        assert state.asset_ids == ("SPX", "NDX")
        assert float(state.spot("NDX")) == 50.0
        assert set(state.surfaces) == {"SPX.VOL", "NDX.VOL"}

    def test_default_asset_ids(self):
        original = MarketData(
            spots=jnp.array([100.0]),
            vols=jnp.array([0.2]),
            dividends=jnp.array([0.0]),
            discount_curve=_curve(),
        )
        state = MarketState.from_market_data(original)
        assert state.asset_ids == ("ASSET_0",)

    def test_mismatched_asset_ids_raises(self):
        original = MarketData(
            spots=jnp.array([100.0, 50.0]),
            vols=jnp.array([0.2, 0.3]),
            dividends=jnp.array([0.0, 0.0]),
            discount_curve=_curve(),
        )
        with pytest.raises(ValueError, match="asset_ids has length"):
            MarketState.from_market_data(original, asset_ids=("SPX",))

    def test_curve_graph_is_carried_over(self):
        graph = CurveGraph(
            curves={"USD.SOFR.OIS": _curve(0.04), "USD.SOFR.3M": _curve(0.045)}
        )
        original = MarketData(
            spots=jnp.array([100.0]),
            vols=jnp.array([0.2]),
            dividends=jnp.array([0.0]),
            discount_curve=_curve(),
            curve_graph=graph,
        )
        state = MarketState.from_market_data(original)
        assert set(state.curves.keys()) == {
            "USD.SOFR.OIS", "USD.SOFR.3M", DEFAULT_ID,
        }
        # The legacy single curve still answers the sentinel.
        assert state.discount_curve is original.discount_curve

    def test_rich_surface_cannot_be_flattened_silently(self):
        from valax.surfaces.grid import GridVolSurface

        surface = GridVolSurface(
            strikes=jnp.array([90.0, 100.0, 110.0]),
            expiries=jnp.array([0.5, 1.0]),
            vols=jnp.full((2, 3), 0.2),
        )
        state = MarketState(
            curves=CurveGraph(curves={DEFAULT_ID: _curve()}),
            surfaces={"SPX.VOL": surface},
            spots={"SPX": jnp.array(100.0)},
            asset_ids=("SPX",),
        )
        with pytest.raises(ValueError, match="cannot represent"):
            state.to_market_data()


# ── Identifier alphabets (Q3) ────────────────────────────────────────

class TestSurfaceIds:
    @pytest.mark.parametrize(
        "surface_id",
        ["SPX.VOL", "EURUSD.VOL", "USD.SOFR.CUBE", "SPX.LV", "SPX.VOL.BID",
         "USD.SOFR.OPTIONLET", "SPX.LEVERAGE", DEFAULT_ID],
    )
    def test_accepted(self, surface_id):
        validate_surface_id(surface_id)

    @pytest.mark.parametrize(
        "surface_id", ["spx.vol", "SPX", "SPX.SMILE", "SPX.VOL.", ".VOL"],
    )
    def test_rejected(self, surface_id):
        with pytest.raises(ValueError, match="Invalid surface identifier"):
            validate_surface_id(surface_id)

    def test_validated_at_construction(self):
        with pytest.raises(ValueError, match="Invalid surface identifier"):
            MarketState(
                curves=CurveGraph(curves={DEFAULT_ID: _curve()}),
                surfaces={"spx_vol": ConstantVol(vol=jnp.array(0.2))},
            )


# ── Lookup errors ────────────────────────────────────────────────────

class TestLookups:
    def test_missing_curve_lists_available(self):
        market = _single()
        with pytest.raises(KeyError, match="not in MarketState"):
            market.curve("USD.SOFR.OIS")

    def test_ambiguous_default_curve_raises(self):
        market = MarketState(
            curves=CurveGraph(
                curves={"USD.SOFR.OIS": _curve(), "USD.SOFR.3M": _curve(0.06)}
            ),
        )
        with pytest.raises(ValueError, match="name the curve explicitly"):
            _ = market.discount_curve

    def test_sole_curve_answers_the_sentinel(self):
        market = MarketState(curves=CurveGraph(curves={"USD.SOFR.OIS": _curve()}))
        assert market.discount_curve is market.curves["USD.SOFR.OIS"]

    def test_sole_asset_answers_the_sentinel(self):
        market = MarketState(
            curves=CurveGraph(curves={DEFAULT_ID: _curve()}),
            spots={"SPX": jnp.array(100.0)},
        )
        assert float(market.spot()) == 100.0

    def test_missing_surface_lists_available(self):
        market = _single()
        with pytest.raises(KeyError, match="not in MarketState"):
            market.surface("SPX.VOL")
