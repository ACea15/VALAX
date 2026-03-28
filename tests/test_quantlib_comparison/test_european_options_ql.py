"""
Cross-validation: VALAX vs QuantLib for European option pricing.

These tests ensure VALAX's Black-Scholes implementation matches QuantLib's
analytic engine to machine precision. If these fail, the pricing formula
or its inputs have diverged.

Companion example: examples/comparisons/01_european_options.py
"""

import pytest
import jax
import jax.numpy as jnp
import QuantLib as ql
from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price, black_scholes_implied_vol
from valax.greeks.autodiff import greeks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def market():
    """Common market parameters for European option tests."""
    return dict(S=105.0, K=100.0, T=1.0, sigma=0.25, r=0.04, q=0.01)


@pytest.fixture
def valax_call(market):
    return EuropeanOption(
        strike=jnp.array(market["K"]),
        expiry=jnp.array(market["T"]),
        is_call=True,
    )


@pytest.fixture
def valax_put(market):
    return EuropeanOption(
        strike=jnp.array(market["K"]),
        expiry=jnp.array(market["T"]),
        is_call=False,
    )


@pytest.fixture
def ql_call(market):
    """Build a QuantLib European call with analytic BS engine."""
    today = ql.Date(26, 3, 2026)
    ql.Settings.instance().evaluationDate = today

    spot_h = ql.QuoteHandle(ql.SimpleQuote(market["S"]))
    rate_h = ql.YieldTermStructureHandle(ql.FlatForward(today, market["r"], ql.Actual365Fixed()))
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(today, market["q"], ql.Actual365Fixed()))
    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), market["sigma"], ql.Actual365Fixed())
    )
    process = ql.BlackScholesMertonProcess(spot_h, div_h, rate_h, vol_h)

    maturity = today + ql.Period(1, ql.Years)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, market["K"])
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option, process


# ---------------------------------------------------------------------------
# Price tests
# ---------------------------------------------------------------------------

class TestEuropeanPrices:
    """VALAX and QuantLib must agree on BS call/put prices."""

    def test_call_price_matches(self, market, valax_call, ql_call):
        """See: examples/comparisons/01_european_options.py §2 (pricing)"""
        v = black_scholes_price(
            valax_call, jnp.array(market["S"]), jnp.array(market["sigma"]),
            jnp.array(market["r"]), jnp.array(market["q"]),
        )
        q_price = ql_call[0].NPV()
        assert abs(float(v) - q_price) < 1e-10, f"Call price mismatch: VALAX={float(v)}, QL={q_price}"

    def test_put_price_matches(self, market, valax_put, ql_call):
        """See: examples/comparisons/01_european_options.py §2 (pricing)"""
        v = black_scholes_price(
            valax_put, jnp.array(market["S"]), jnp.array(market["sigma"]),
            jnp.array(market["r"]), jnp.array(market["q"]),
        )
        # Build QL put
        today = ql.Date(26, 3, 2026)
        maturity = today + ql.Period(1, ql.Years)
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, market["K"])
        exercise = ql.EuropeanExercise(maturity)
        put = ql.VanillaOption(payoff, exercise)
        put.setPricingEngine(ql.AnalyticEuropeanEngine(ql_call[1]))
        assert abs(float(v) - put.NPV()) < 1e-10

    @pytest.mark.parametrize("K", [80.0, 90.0, 100.0, 110.0, 120.0])
    def test_call_prices_across_strikes(self, market, K):
        """Validate across a range of strikes, not just ATM."""
        today = ql.Date(26, 3, 2026)
        ql.Settings.instance().evaluationDate = today

        opt_v = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(market["T"]), is_call=True)
        v_price = float(black_scholes_price(
            opt_v, jnp.array(market["S"]), jnp.array(market["sigma"]),
            jnp.array(market["r"]), jnp.array(market["q"]),
        ))

        spot_h = ql.QuoteHandle(ql.SimpleQuote(market["S"]))
        rate_h = ql.YieldTermStructureHandle(ql.FlatForward(today, market["r"], ql.Actual365Fixed()))
        div_h = ql.YieldTermStructureHandle(ql.FlatForward(today, market["q"], ql.Actual365Fixed()))
        vol_h = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), market["sigma"], ql.Actual365Fixed())
        )
        process = ql.BlackScholesMertonProcess(spot_h, div_h, rate_h, vol_h)
        maturity = today + ql.Period(1, ql.Years)
        opt_ql = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(maturity))
        opt_ql.setPricingEngine(ql.AnalyticEuropeanEngine(process))

        assert abs(v_price - opt_ql.NPV()) < 1e-10, f"Mismatch at K={K}"


# ---------------------------------------------------------------------------
# Greeks tests
# ---------------------------------------------------------------------------

class TestEuropeanGreeks:
    """VALAX autodiff Greeks must match QuantLib's analytic Greeks."""

    def test_delta_matches(self, market, valax_call, ql_call):
        """See: examples/comparisons/01_european_options.py §3 (Greeks)"""
        g = greeks(black_scholes_price, valax_call,
                   jnp.array(market["S"]), jnp.array(market["sigma"]),
                   jnp.array(market["r"]), jnp.array(market["q"]))
        assert abs(float(g["delta"]) - ql_call[0].delta()) < 1e-10

    def test_gamma_matches(self, market, valax_call, ql_call):
        """See: examples/comparisons/01_european_options.py §3 (Greeks)"""
        g = greeks(black_scholes_price, valax_call,
                   jnp.array(market["S"]), jnp.array(market["sigma"]),
                   jnp.array(market["r"]), jnp.array(market["q"]))
        assert abs(float(g["gamma"]) - ql_call[0].gamma()) < 1e-10

    def test_vega_matches(self, market, valax_call, ql_call):
        """See: examples/comparisons/01_european_options.py §3 (Greeks)"""
        g = greeks(black_scholes_price, valax_call,
                   jnp.array(market["S"]), jnp.array(market["sigma"]),
                   jnp.array(market["r"]), jnp.array(market["q"]))
        assert abs(float(g["vega"]) - ql_call[0].vega()) < 1e-8


# ---------------------------------------------------------------------------
# Implied vol tests
# ---------------------------------------------------------------------------

class TestImpliedVol:
    """Round-trip: price → implied vol → should recover input vol."""

    def test_implied_vol_round_trip(self, market, valax_call):
        """See: examples/comparisons/01_european_options.py §4 (implied vol)"""
        price = black_scholes_price(
            valax_call, jnp.array(market["S"]), jnp.array(market["sigma"]),
            jnp.array(market["r"]), jnp.array(market["q"]),
        )
        recovered = black_scholes_implied_vol(
            valax_call, jnp.array(market["S"]), jnp.array(market["r"]),
            jnp.array(market["q"]), price,
        )
        assert abs(float(recovered) - market["sigma"]) < 1e-12

    def test_implied_vol_matches_quantlib(self, market, valax_call, ql_call):
        """Both solvers should recover the same vol from the same price."""
        price = float(black_scholes_price(
            valax_call, jnp.array(market["S"]), jnp.array(market["sigma"]),
            jnp.array(market["r"]), jnp.array(market["q"]),
        ))
        v_iv = float(black_scholes_implied_vol(
            valax_call, jnp.array(market["S"]), jnp.array(market["r"]),
            jnp.array(market["q"]), jnp.array(price),
        ))
        q_iv = ql_call[0].impliedVolatility(price, ql_call[1], 1e-8, 1000, 0.001, 4.0)
        assert abs(v_iv - q_iv) < 1e-8
