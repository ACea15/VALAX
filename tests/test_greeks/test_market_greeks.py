"""Tests for the unified sensitivities layer (pricing-ontology P4).

Covers the three sensitivity families over ``price(instrument, model,
market)``: market greeks (spot delta, curve key-rate deltas), model greeks
(vega/rho), and contract greeks (dual-delta, theta with its autodiff-vs-bump
policy), plus the bucketing bridge.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.core.market_state import MarketState
from valax.curves.discount import DiscountCurve
from valax.greeks import greek, greeks as scalar_greeks
from valax.greeks.market_greeks import (
    bucketed_curve_deltas,
    curve_pillar_deltas,
    dual_delta,
    market_greeks,
    market_gradient,
    model_gradient,
    spot_delta,
    theta,
)
from valax.instruments.options import EuropeanOption
from valax.instruments.rates import InterestRateSwap
from valax.models.black_scholes import BlackScholesModel
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.pde import PDEConfig
from valax.risk.bucketing import tenor_bucket_map


REF = jnp.array(739252, dtype=jnp.int32)
SPOT = jnp.array(100.0)


def _curve(level=0.05):
    times = jnp.arange(0, 11 * 365, 365, dtype=jnp.int32)
    return DiscountCurve(
        pillar_dates=REF + times,
        discount_factors=jnp.exp(-level * times / 365.0),
        reference_date=REF,
        day_count="act_365",
    )


def _equity_market():
    return MarketState.single_curve(_curve(), spot=SPOT, vol=jnp.array(0.2))


def _bs_model():
    return BlackScholesModel(
        vol=jnp.array(0.2), rate=jnp.array(0.05), dividend=jnp.array(0.0)
    )


def _option(is_call=True):
    return EuropeanOption(
        strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=is_call
    )


def _swap():
    return InterestRateSwap(
        start_date=REF,
        fixed_dates=REF + jnp.array([365, 730, 1095], jnp.int32),
        fixed_rate=jnp.array(0.04),
        notional=jnp.array(1_000_000.0),
    )


# ── Market greeks ────────────────────────────────────────────────────

class TestMarketGreeks:
    def test_spot_delta_matches_scalar_autodiff(self):
        model, option = _bs_model(), _option()
        unified = spot_delta(option, model, _equity_market())
        legacy = greek(
            black_scholes_price, "delta", option,
            SPOT, model.vol, model.rate, model.dividend,
        )
        assert abs(float(unified) - float(legacy)) < 1e-6

    def test_spot_delta_in_unit_interval_for_call(self):
        assert 0.0 < float(spot_delta(_option(), _bs_model(), _equity_market())) < 1.0

    def test_market_gradient_is_marketstate_shaped(self):
        grad = market_gradient(_option(), _bs_model(), _equity_market())
        assert isinstance(grad, MarketState)
        # Spot leaf carries the delta.
        assert float(grad.spots["_default_"]) > 0.0

    def test_curve_pillar_deltas_finite_for_swap(self):
        market = MarketState.single_curve(_curve())
        deltas = curve_pillar_deltas(_swap(), None, market)
        assert deltas.shape == _curve().discount_factors.shape
        assert bool(jnp.all(jnp.isfinite(deltas)))
        assert float(jnp.sum(jnp.abs(deltas))) > 0.0


# ── Model greeks ─────────────────────────────────────────────────────

class TestModelGreeks:
    def test_vega_matches_scalar_autodiff(self):
        model, option = _bs_model(), _option()
        vega = model_gradient(option, model, _equity_market()).vol
        legacy = greek(
            black_scholes_price, "vega", option,
            SPOT, model.vol, model.rate, model.dividend,
        )
        assert abs(float(vega) - float(legacy)) < 1e-6

    def test_model_gradient_requires_a_model(self):
        with pytest.raises(ValueError, match="needs a model"):
            model_gradient(_swap(), None, MarketState.single_curve(_curve()))


# ── Contract greeks: dual-delta and theta ────────────────────────────

class TestContractGreeks:
    def test_dual_delta_sign(self):
        # For a call, price decreases in strike ⇒ dual-delta < 0.
        dd = dual_delta(_option(is_call=True), _bs_model(), _equity_market())
        assert float(dd) < 0.0

    def test_theta_autodiff_matches_fd(self):
        model, option, market = _bs_model(), _option(), _equity_market()
        autodiff = theta(option, model, market)  # analytic ⇒ exact
        fd = scalar_greeks(
            black_scholes_price, option,
            SPOT, model.vol, model.rate, model.dividend,
        )["theta"]
        assert abs(float(autodiff) - float(fd)) < 2e-2

    def test_theta_pde_path_is_finite(self):
        val = theta(
            _option(), _bs_model(), _equity_market(),
            method="pde", config=PDEConfig(),
        )
        assert jnp.isfinite(val)

    def test_theta_rates_instrument_raises(self):
        with pytest.raises(NotImplementedError, match="valuation-date roll"):
            theta(_swap(), None, MarketState.single_curve(_curve()))


# ── Bucketing bridge (J_bucket) ──────────────────────────────────────

class TestBucketing:
    def test_indicator_bucketing_preserves_total(self):
        market = MarketState.single_curve(_curve())
        pillar = curve_pillar_deltas(_swap(), None, market)
        pillar_times = jnp.arange(0, 11 * 365, 365) / 365.0
        bmap = tenor_bucket_map(
            pillar_times, jnp.array([1.0, 2.0, 3.0, 5.0, 10.0]),
            weight="indicator",
        )
        bucketed = bucketed_curve_deltas(pillar, bmap)
        assert bucketed.shape == (5,)
        # Indicator assignment moves each pillar's sensitivity into exactly
        # one bucket, so the total DV01 is conserved.
        assert abs(float(jnp.sum(bucketed)) - float(jnp.sum(pillar))) < 1e-8


# ── Aggregate convenience + JAX behaviour ────────────────────────────

class TestAggregateAndJax:
    def test_market_greeks_keys_for_equity_option(self):
        g = market_greeks(_option(), _bs_model(), _equity_market())
        assert set(g) == {
            "price", "delta", "vega", "rho", "dividend_rho", "theta",
        }
        assert float(g["vega"]) > 0.0

    def test_market_greeks_skips_delta_without_spots(self):
        g = market_greeks(_swap(), None, MarketState.single_curve(_curve()),
                          include_theta=False)
        assert "delta" not in g
        assert "price" in g

    def test_filter_jit_smoke(self):
        market, model, option = _equity_market(), _bs_model(), _option()

        @eqx.filter_jit
        def delta_of(m):
            return spot_delta(option, model, m)

        assert jnp.isfinite(delta_of(market))
