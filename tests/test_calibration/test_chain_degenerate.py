"""Degenerate chains: constant parameters and the single-curve limit.

The unified pipeline must reduce *exactly* to the simpler worlds it
generalises, with no special-cased code path:

* constant-parameter Black-Scholes — quote risk through an explicit
  ``market_inputs_node`` equals the textbook closed-form Greeks;
* the empty chain — no quotes, risk lives on the market / model directly;
* single curve — the chain's curve quote risk equals the pre-existing
  curve-level adjoint (``curve_quote_deltas``) bit for bit in structure and to
  solver tolerance in value.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.calibration import CalibrationChain, calibrate_market
from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.curves import CurveSpec, bootstrap_curve_graph
from valax.greeks import (
    curve_adjoint,
    curve_quote_deltas,
    market_gradient,
    quote_risk_adjoint,
    quote_risk_autodiff,
)
from valax.instruments.bonds import FixedRateBond
from valax.instruments.options import EuropeanOption
from valax.instruments.rates import InterestRateSwap
from valax.market import QuoteSet
from valax.models.black_scholes import BlackScholesModel
from valax.pricing.analytic.black_scholes import black_scholes_price

from tests.test_calibration._chain_fixtures import (
    ANN_3Y,
    DC_KEYS,
    OIS,
    P3M,
    REF,
    SC_KEYS,
    SC_PILLARS,
    d,
    dc_instruments,
    dc_specs,
    dual_curve_chain,
    flat_chain,
    sc_curve_node,
    sc_instruments,
)


OPTION = EuropeanOption(strike=jnp.asarray(105.0), expiry=jnp.asarray(1.0))


class TestConstantParameterBlackScholes:
    """Tier 0: quote risk == textbook Greeks, on both paths."""

    def _textbook(self):
        g = jax.grad(black_scholes_price, argnums=(1, 2, 3, 4))(
            OPTION, 100.0, 0.2, 0.05, 0.01
        )
        return dict(zip(("SPX.SPOT", "SPX.VOL", "USD.RATE", "SPX.DIV"), g))

    def test_price_matches_closed_form(self):
        chain, quotes = flat_chain()
        market = calibrate_market(chain, quotes).market
        pv = price(OPTION, BlackScholesModel(), market).price
        ref = black_scholes_price(OPTION, 100.0, 0.2, 0.05, 0.01)
        assert float(pv) == pytest.approx(float(ref), abs=1e-10)

    @pytest.mark.parametrize("path", ["adjoint", "autodiff"])
    def test_quote_risk_is_textbook_greeks(self, path):
        chain, quotes = flat_chain()
        if path == "adjoint":
            risk = quote_risk_adjoint(
                OPTION, BlackScholesModel(), calibrate_market(chain, quotes)
            )
        else:
            risk = quote_risk_autodiff(OPTION, BlackScholesModel(), chain, quotes)
        for key, value in self._textbook().items():
            assert float(risk.quotes[key]) == pytest.approx(float(value), abs=1e-10)

    def test_market_flat_constructor_agrees(self):
        chain, quotes = flat_chain()
        chained = calibrate_market(chain, quotes).market
        direct = MarketState.flat(0.05, spot=100.0, vol=0.2, dividend=0.01)
        a = price(OPTION, BlackScholesModel(), chained).price
        b = price(OPTION, BlackScholesModel(), direct).price
        assert float(a) == pytest.approx(float(b), abs=1e-12)


class TestEmptyChain:
    def test_no_quotes_no_risk(self):
        """A fully specified model on a base market: nothing to push back."""
        base = MarketState.flat(0.05, spot=100.0)
        model = BlackScholesModel(
            vol=jnp.asarray(0.2), rate=jnp.asarray(0.05), dividend=jnp.asarray(0.0)
        )
        cm = calibrate_market(CalibrationChain(), QuoteSet(), base=base)
        risk = quote_risk_adjoint(OPTION, model, cm)
        assert len(risk.quotes) == 0 and len(risk.fixed) == 0
        # The market greek is still available directly.
        assert float(market_gradient(OPTION, model, base).spots["_default_"]) > 0.0


def _sc_bond():
    return FixedRateBond(
        payment_dates=jnp.array([d(2026, 1, 1), d(2027, 1, 1), d(2028, 1, 1)], jnp.int32),
        settlement_date=REF,
        coupon_rate=jnp.asarray(0.04),
        face_value=jnp.asarray(100.0),
        frequency=1,
    )


class TestSingleCurveLimit:
    """Tier 1 reduces to the curve-level adjoint that predates the chain."""

    def test_single_curve_chain_matches_curve_quote_deltas(self):
        chain = CalibrationChain(nodes=(sc_curve_node(),))
        quotes = QuoteSet.from_vector(
            SC_KEYS, jnp.array([0.045, 0.047, 0.050])
        )
        cm = calibrate_market(chain, quotes, strict=True)
        risk = quote_risk_adjoint(_sc_bond(), None, cm)

        spec = CurveSpec(curve_id="_default_", currency="USD",
                         pillar_dates=SC_PILLARS, day_count="act_365")
        graph, _ = bootstrap_curve_graph(REF, [spec], sc_instruments())
        adj = curve_adjoint(graph, REF, [spec], sc_instruments())
        legacy = curve_quote_deltas(_sc_bond(), None, MarketState(curves=graph), adjoint=adj)
        np.testing.assert_allclose(
            risk.quotes.vector(SC_KEYS), legacy, rtol=1e-9, atol=1e-10
        )

    def test_dual_curve_chain_matches_curve_quote_deltas(self):
        swap = InterestRateSwap(
            start_date=REF, fixed_dates=ANN_3Y, fixed_rate=jnp.asarray(0.04),
            notional=jnp.asarray(1_000_000.0), discount_id=OIS, forward_id=P3M,
        )
        chain, quotes = dual_curve_chain()
        risk = quote_risk_adjoint(swap, None, calibrate_market(chain, quotes))

        graph, _ = bootstrap_curve_graph(REF, dc_specs(), dc_instruments())
        adj = curve_adjoint(graph, REF, dc_specs(), dc_instruments())
        legacy = curve_quote_deltas(swap, None, MarketState(curves=graph), adjoint=adj)
        np.testing.assert_allclose(
            risk.quotes.vector(DC_KEYS), legacy, rtol=1e-8, atol=1e-6
        )
