"""Tests for selector resolution against a MarketState (ontology P2)."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from valax.core.market_state import DEFAULT_ID, MarketState
from valax.core.resolve import (
    ResolvedMarket,
    discount_id_of,
    forward_id_of,
    resolve,
    resolve_curves,
    resolve_spot,
    resolve_surface,
    surface_id_of,
    underlying_id_of,
)
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.curves.instruments import DepositRate, IborSwapRate
from valax.instruments.options import EuropeanOption
from valax.models.black_scholes import BlackScholesModel
from valax.surfaces.constant import ConstantVol


REF = jnp.array(739252, dtype=jnp.int32)


def _curve(level=0.05):
    times = jnp.array([0, 365, 730], dtype=jnp.int32)
    return DiscountCurve(
        pillar_dates=REF + times,
        discount_factors=jnp.exp(-level * times / 365.0),
        reference_date=REF,
        day_count="act_365",
    )


def _option():
    return EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0))


class _SelectorOption(eqx.Module):
    """A P3-style instrument carrying explicit static selectors."""

    strike: jnp.ndarray
    discount_id: str = eqx.field(static=True, default="USD.SOFR.OIS")
    forward_id: str | None = eqx.field(static=True, default="USD.SOFR.3M")
    underlying_id: str = eqx.field(static=True, default="SPX")


class _SurfaceModel(eqx.Module):
    """A model that references its surface by id rather than embedding it."""

    surface_id: str = eqx.field(static=True, default="SPX.VOL")


# ── Selector defaults: single-curve needs zero configuration ─────────

class TestSelectorDefaults:
    def test_bare_instrument_resolves_to_the_sentinel(self):
        option = _option()
        assert discount_id_of(option) == DEFAULT_ID
        assert forward_id_of(option) is None
        assert underlying_id_of(option) == DEFAULT_ID

    def test_model_without_surface_reference(self):
        model = BlackScholesModel(
            vol=jnp.array(0.2), rate=jnp.array(0.05), dividend=jnp.array(0.0)
        )
        assert surface_id_of(model) is None

    def test_model_none(self):
        assert surface_id_of(None) is None

    def test_single_curve_market_binds_with_no_ids(self):
        market = MarketState.single_curve(_curve(), spot=jnp.array(100.0))
        discount, forward = resolve_curves(_option(), market)
        assert discount is market.discount_curve
        assert forward is None  # project off the discount curve


# ── Explicit selectors on a multi-curve graph ────────────────────────

class TestExplicitSelectors:
    def _market(self):
        return MarketState(
            curves=CurveGraph(
                curves={
                    "USD.SOFR.OIS": _curve(0.04),
                    "USD.SOFR.3M": _curve(0.045),
                }
            ),
            surfaces={"SPX.VOL": ConstantVol(vol=jnp.array(0.2))},
            spots={"SPX": jnp.array(100.0)},
            asset_ids=("SPX",),
        )

    def test_curves_are_bound_by_id(self):
        instrument = _SelectorOption(strike=jnp.array(100.0))
        discount, forward = resolve_curves(instrument, self._market())
        market = self._market()
        assert jnp.allclose(
            discount.discount_factors,
            market.curves["USD.SOFR.OIS"].discount_factors,
        )
        assert jnp.allclose(
            forward.discount_factors,
            market.curves["USD.SOFR.3M"].discount_factors,
        )

    def test_spot_and_surface_are_bound_by_id(self):
        market = self._market()
        instrument = _SelectorOption(strike=jnp.array(100.0))
        assert float(resolve_spot(instrument, market)) == 100.0
        assert float(resolve_surface(_SurfaceModel(), market).vol) == 0.2

    def test_curves_touched_is_honoured(self):
        """Bootstrap instruments already carry the selector tuple."""
        swap = IborSwapRate(
            start_date=REF,
            fixed_dates=jnp.array([REF + 365], jnp.int32),
            float_dates=jnp.array([REF + 365], jnp.int32),
            fixing_dates=jnp.array([REF], jnp.int32),
            rate=jnp.array(0.04),
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
        )
        assert discount_id_of(swap) == "USD.SOFR.OIS"
        assert forward_id_of(swap) == "USD.SOFR.3M"

    def test_single_curve_bootstrap_quote_has_no_forward(self):
        deposit = DepositRate(
            start_date=REF, end_date=REF + 90, rate=jnp.array(0.04),
        )
        assert discount_id_of(deposit) == DEFAULT_ID
        assert forward_id_of(deposit) is None

    def test_degenerate_curves_touched_pair_has_no_forward(self):
        """``(x, x)`` means single-curve projection, not a second curve."""
        swap = IborSwapRate(
            start_date=REF,
            fixed_dates=jnp.array([REF + 365], jnp.int32),
            float_dates=jnp.array([REF + 365], jnp.int32),
            fixing_dates=jnp.array([REF], jnp.int32),
            rate=jnp.array(0.04),
        )
        assert forward_id_of(swap) is None


# ── resolve() end to end ─────────────────────────────────────────────

class TestResolve:
    def test_returns_every_bound_object(self):
        market = MarketState.single_curve(
            _curve(), spot=jnp.array(100.0), vol=jnp.array(0.2),
        )
        model = BlackScholesModel(
            vol=jnp.array(0.2), rate=jnp.array(0.05), dividend=jnp.array(0.0)
        )
        resolved = resolve(_option(), model, market)
        assert isinstance(resolved, ResolvedMarket)
        assert resolved.forward_curve is None
        assert float(resolved.spot) == 100.0
        assert float(resolved.surface()) == 0.2
        assert resolved.reference_date == REF

    def test_rates_market_without_spots(self):
        market = MarketState.single_curve(_curve())
        resolved = resolve(_option(), None, market)
        assert resolved.spot is None
        assert resolved.surface is None

    def test_is_a_pytree(self):
        market = MarketState.single_curve(_curve(), spot=jnp.array(100.0))
        resolved = resolve(_option(), None, market)
        leaves, treedef = eqx.tree_flatten_one_level(resolved)
        assert leaves  # flattens without error


# ── Error messages ───────────────────────────────────────────────────

class TestErrors:
    def test_missing_curve_lists_available(self):
        market = MarketState(
            curves=CurveGraph(curves={"USD.SOFR.OIS": _curve()}),
        )
        instrument = _SelectorOption(strike=jnp.array(100.0))
        with pytest.raises(KeyError, match="USD.SOFR.3M"):
            resolve_curves(instrument, market)

    def test_missing_spot_lists_available(self):
        market = MarketState(
            curves=CurveGraph(curves={DEFAULT_ID: _curve()}),
            spots={"NDX": jnp.array(50.0), "RTY": jnp.array(20.0)},
        )
        instrument = _SelectorOption(strike=jnp.array(100.0))
        with pytest.raises(KeyError, match="No spot for asset 'SPX'"):
            resolve_spot(instrument, market)

    def test_missing_surface_raises(self):
        market = MarketState(curves=CurveGraph(curves={DEFAULT_ID: _curve()}))
        with pytest.raises(KeyError, match="SPX.VOL"):
            resolve_surface(_SurfaceModel(), market)
