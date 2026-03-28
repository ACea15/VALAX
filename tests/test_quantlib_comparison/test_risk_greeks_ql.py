"""
Cross-validation: VALAX vs QuantLib for risk building blocks.

Tests that VALAX's autodiff-based risk engine (Greeks, repricing under shocks,
P&L attribution, parametric VaR) matches QuantLib's analytic results to
appropriate tolerances.

Companion example: examples/comparisons/08_risk_greeks_var.py
"""

import pytest
import numpy as np

try:
    import QuantLib as ql
    HAS_QUANTLIB = True
except ImportError:
    HAS_QUANTLIB = False

import jax
import jax.numpy as jnp

from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.greeks.autodiff import greeks
from valax.curves.discount import DiscountCurve
from valax.market.data import MarketData
from valax.market.scenario import MarketScenario
from valax.dates.daycounts import ymd_to_ordinal
from valax.risk.var import (
    reprice_under_scenario,
    pnl_attribution,
    parametric_var,
    _extract_short_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def market():
    """Common market parameters."""
    return dict(S=105.0, K=100.0, T=1.0, sigma=0.25, r=0.04, q=0.01)


@pytest.fixture
def valax_call(market):
    return EuropeanOption(
        strike=jnp.array(market["K"]),
        expiry=jnp.array(market["T"]),
        is_call=True,
    )


@pytest.fixture
def valax_instruments(market):
    """Batched single-instrument for risk engine functions."""
    return EuropeanOption(
        strike=jnp.array([market["K"]]),
        expiry=jnp.array([market["T"]]),
        is_call=True,
    )


@pytest.fixture
def base_market(market):
    """MarketData for the risk engine."""
    ref = ymd_to_ordinal(2026, 3, 26)
    pillars = jnp.array([ref, ymd_to_ordinal(2027, 3, 26)])
    pillar_times = (pillars - ref).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-market["r"] * pillar_times)
    curve = DiscountCurve(
        pillar_dates=pillars, discount_factors=dfs, reference_date=ref,
    )
    return MarketData(
        spots=jnp.array([market["S"]]),
        vols=jnp.array([market["sigma"]]),
        dividends=jnp.array([market["q"]]),
        discount_curve=curve,
    )


@pytest.fixture
def ql_option(market):
    """Build a QuantLib European call with analytic BS engine."""
    today = ql.Date(26, 3, 2026)
    ql.Settings.instance().evaluationDate = today

    spot_h = ql.QuoteHandle(ql.SimpleQuote(market["S"]))
    rate_h = ql.YieldTermStructureHandle(
        ql.FlatForward(today, market["r"], ql.Actual365Fixed())
    )
    div_h = ql.YieldTermStructureHandle(
        ql.FlatForward(today, market["q"], ql.Actual365Fixed())
    )
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


def _bs_market_fn(option, market_data: MarketData):
    """Black-Scholes pricing function with MarketData signature."""
    rate = _extract_short_rate(market_data.discount_curve)
    return black_scholes_price(option, market_data.spots, market_data.vols, rate, market_data.dividends)


def _ql_reprice(S, sigma, r, q, K, T):
    """Reprice a QuantLib European call with given parameters."""
    today = ql.Date(26, 3, 2026)
    ql.Settings.instance().evaluationDate = today
    spot_h = ql.QuoteHandle(ql.SimpleQuote(S))
    rate_h = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(today, q, ql.Actual365Fixed()))
    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed())
    )
    proc = ql.BlackScholesMertonProcess(spot_h, div_h, rate_h, vol_h)
    mat = today + ql.Period(1, ql.Years)
    opt = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, K),
        ql.EuropeanExercise(mat),
    )
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
    return opt.NPV()


# ---------------------------------------------------------------------------
# Section A: Greeks comparison
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QUANTLIB, reason="QuantLib not installed")
class TestGreeksMatchQuantLib:
    """VALAX autodiff Greeks must match QuantLib analytic Greeks.

    See: examples/comparisons/08_risk_greeks_var.py Section A
    """

    def test_delta_matches(self, market, valax_call, ql_option):
        g = greeks(black_scholes_price, valax_call,
                   jnp.array(market["S"]), jnp.array(market["sigma"]),
                   jnp.array(market["r"]), jnp.array(market["q"]))
        assert abs(float(g["delta"]) - ql_option[0].delta()) < 1e-4

    def test_gamma_matches(self, market, valax_call, ql_option):
        g = greeks(black_scholes_price, valax_call,
                   jnp.array(market["S"]), jnp.array(market["sigma"]),
                   jnp.array(market["r"]), jnp.array(market["q"]))
        assert abs(float(g["gamma"]) - ql_option[0].gamma()) < 1e-4

    def test_vega_matches(self, market, valax_call, ql_option):
        g = greeks(black_scholes_price, valax_call,
                   jnp.array(market["S"]), jnp.array(market["sigma"]),
                   jnp.array(market["r"]), jnp.array(market["q"]))
        assert abs(float(g["vega"]) - ql_option[0].vega()) < 1e-4

    def test_rho_matches(self, market, valax_call, ql_option):
        g = greeks(black_scholes_price, valax_call,
                   jnp.array(market["S"]), jnp.array(market["sigma"]),
                   jnp.array(market["r"]), jnp.array(market["q"]))
        assert abs(float(g["rho"]) - ql_option[0].rho()) < 1e-4

    def test_theta_matches(self, market, valax_call, ql_option):
        """Theta per day: VALAX bump (per year) / 365 vs QuantLib thetaPerDay."""
        g = greeks(black_scholes_price, valax_call,
                   jnp.array(market["S"]), jnp.array(market["sigma"]),
                   jnp.array(market["r"]), jnp.array(market["q"]))
        # VALAX theta is d(P)/d(T) per year; convert to per day for comparison
        valax_theta_day = float(g["theta"]) / 365.0
        ql_theta_day = ql_option[0].thetaPerDay()
        # Theta via finite difference is less precise; use wider tolerance
        assert abs(valax_theta_day - ql_theta_day) < 1e-3


# ---------------------------------------------------------------------------
# Section B: Repricing under shocks
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QUANTLIB, reason="QuantLib not installed")
class TestRepricingUnderShocks:
    """VALAX repricing under market shocks must match QuantLib.

    See: examples/comparisons/08_risk_greeks_var.py Section B
    """

    def test_spot_bump_matches(self, market, valax_instruments, base_market):
        """Spot +5% repricing matches QuantLib."""
        S, K, sigma, r, q, T = (
            market["S"], market["K"], market["sigma"],
            market["r"], market["q"], market["T"],
        )
        dS = S * 0.05
        scenario = MarketScenario(
            spot_shocks=jnp.array([dS]),
            vol_shocks=jnp.zeros(1),
            rate_shocks=jnp.zeros(2),
            dividend_shocks=jnp.zeros(1),
        )
        valax_val = float(reprice_under_scenario(
            _bs_market_fn, valax_instruments, base_market, scenario,
        ))
        ql_val = _ql_reprice(S + dS, sigma, r, q, K, T)
        assert abs(valax_val - ql_val) < 1e-4, (
            f"Spot bump mismatch: VALAX={valax_val}, QL={ql_val}"
        )

    def test_vol_bump_matches(self, market, valax_instruments, base_market):
        """Vol +1pp repricing matches QuantLib."""
        S, K, sigma, r, q, T = (
            market["S"], market["K"], market["sigma"],
            market["r"], market["q"], market["T"],
        )
        dvol = 0.01
        scenario = MarketScenario(
            spot_shocks=jnp.zeros(1),
            vol_shocks=jnp.array([dvol]),
            rate_shocks=jnp.zeros(2),
            dividend_shocks=jnp.zeros(1),
        )
        valax_val = float(reprice_under_scenario(
            _bs_market_fn, valax_instruments, base_market, scenario,
        ))
        ql_val = _ql_reprice(S, sigma + dvol, r, q, K, T)
        assert abs(valax_val - ql_val) < 1e-4, (
            f"Vol bump mismatch: VALAX={valax_val}, QL={ql_val}"
        )

    def test_rate_bump_matches(self, market, valax_instruments, base_market):
        """Rate +50bp repricing matches QuantLib."""
        S, K, sigma, r, q, T = (
            market["S"], market["K"], market["sigma"],
            market["r"], market["q"], market["T"],
        )
        dr = 0.005
        scenario = MarketScenario(
            spot_shocks=jnp.zeros(1),
            vol_shocks=jnp.zeros(1),
            rate_shocks=jnp.full(2, dr),
            dividend_shocks=jnp.zeros(1),
        )
        valax_val = float(reprice_under_scenario(
            _bs_market_fn, valax_instruments, base_market, scenario,
        ))
        ql_val = _ql_reprice(S, sigma, r + dr, q, K, T)
        assert abs(valax_val - ql_val) < 1e-4, (
            f"Rate bump mismatch: VALAX={valax_val}, QL={ql_val}"
        )

    def test_combined_bump_matches(self, market, valax_instruments, base_market):
        """Combined spot+vol+rate bump matches QuantLib."""
        S, K, sigma, r, q, T = (
            market["S"], market["K"], market["sigma"],
            market["r"], market["q"], market["T"],
        )
        dS = S * 0.05
        dvol = 0.01
        dr = 0.005
        scenario = MarketScenario(
            spot_shocks=jnp.array([dS]),
            vol_shocks=jnp.array([dvol]),
            rate_shocks=jnp.full(2, dr),
            dividend_shocks=jnp.zeros(1),
        )
        valax_val = float(reprice_under_scenario(
            _bs_market_fn, valax_instruments, base_market, scenario,
        ))
        ql_val = _ql_reprice(S + dS, sigma + dvol, r + dr, q, K, T)
        assert abs(valax_val - ql_val) < 1e-4, (
            f"Combined bump mismatch: VALAX={valax_val}, QL={ql_val}"
        )


# ---------------------------------------------------------------------------
# Section C: P&L Attribution
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QUANTLIB, reason="QuantLib not installed")
class TestPnLAttribution:
    """P&L attribution delta terms must match QuantLib Greeks x shocks.

    See: examples/comparisons/08_risk_greeks_var.py Section C
    """

    def test_delta_spot_matches_ql_greek(self, market, valax_instruments, base_market, ql_option):
        """delta_spot term ~ QL delta * dS."""
        S = market["S"]
        dS = S * 0.05
        scenario = MarketScenario(
            spot_shocks=jnp.array([dS]),
            vol_shocks=jnp.zeros(1),
            rate_shocks=jnp.zeros(2),
            dividend_shocks=jnp.zeros(1),
        )
        attr = pnl_attribution(_bs_market_fn, valax_instruments, base_market, scenario)
        ql_delta_pnl = ql_option[0].delta() * dS
        assert abs(float(attr["delta_spot"]) - ql_delta_pnl) < 1e-3, (
            f"delta_spot mismatch: VALAX={float(attr['delta_spot'])}, QL={ql_delta_pnl}"
        )

    def test_gamma_spot_matches_ql_greek(self, market, valax_instruments, base_market, ql_option):
        """gamma_spot term ~ 0.5 * QL gamma * dS^2."""
        S = market["S"]
        dS = S * 0.05
        scenario = MarketScenario(
            spot_shocks=jnp.array([dS]),
            vol_shocks=jnp.zeros(1),
            rate_shocks=jnp.zeros(2),
            dividend_shocks=jnp.zeros(1),
        )
        attr = pnl_attribution(_bs_market_fn, valax_instruments, base_market, scenario)
        ql_gamma_pnl = 0.5 * ql_option[0].gamma() * dS**2
        assert abs(float(attr["gamma_spot"]) - ql_gamma_pnl) < 1e-3, (
            f"gamma_spot mismatch: VALAX={float(attr['gamma_spot'])}, QL={ql_gamma_pnl}"
        )

    def test_delta_vol_matches_ql_vega(self, market, valax_instruments, base_market, ql_option):
        """delta_vol term ~ QL vega * dvol."""
        dvol = 0.01
        scenario = MarketScenario(
            spot_shocks=jnp.zeros(1),
            vol_shocks=jnp.array([dvol]),
            rate_shocks=jnp.zeros(2),
            dividend_shocks=jnp.zeros(1),
        )
        attr = pnl_attribution(_bs_market_fn, valax_instruments, base_market, scenario)
        ql_vega_pnl = ql_option[0].vega() * dvol
        assert abs(float(attr["delta_vol"]) - ql_vega_pnl) < 1e-3, (
            f"delta_vol mismatch: VALAX={float(attr['delta_vol'])}, QL={ql_vega_pnl}"
        )

    def test_delta_rate_matches_ql_rho(self, market, valax_instruments, base_market, ql_option):
        """delta_rate term ~ QL rho * dr."""
        dr = 0.005
        scenario = MarketScenario(
            spot_shocks=jnp.zeros(1),
            vol_shocks=jnp.zeros(1),
            rate_shocks=jnp.full(2, dr),
            dividend_shocks=jnp.zeros(1),
        )
        attr = pnl_attribution(_bs_market_fn, valax_instruments, base_market, scenario)
        ql_rho_pnl = ql_option[0].rho() * dr
        # Wider tolerance: rate sensitivity through curve pillars introduces
        # small differences vs QL's flat-forward rho
        assert abs(float(attr["delta_rate"]) - ql_rho_pnl) < 5e-3, (
            f"delta_rate mismatch: VALAX={float(attr['delta_rate'])}, QL={ql_rho_pnl}"
        )

    def test_second_order_closer_than_first(self, market, valax_instruments, base_market):
        """Second-order approximation should be closer to actual than first-order."""
        S = market["S"]
        dS = S * 0.05
        scenario = MarketScenario(
            spot_shocks=jnp.array([dS]),
            vol_shocks=jnp.array([0.01]),
            rate_shocks=jnp.full(2, 0.005),
            dividend_shocks=jnp.zeros(1),
        )
        attr = pnl_attribution(_bs_market_fn, valax_instruments, base_market, scenario)
        actual = float(attr["actual"])
        err_1st = abs(actual - float(attr["total_first_order"]))
        err_2nd = abs(actual - float(attr["total_second_order"]))
        assert err_2nd < err_1st, (
            f"2nd-order error ({err_2nd:.6f}) should be smaller than "
            f"1st-order error ({err_1st:.6f})"
        )

    def test_combined_attribution_actual_matches_ql(self, market, valax_instruments, base_market):
        """Actual P&L from attribution must match QL repriced P&L."""
        S, K, sigma, r, q, T = (
            market["S"], market["K"], market["sigma"],
            market["r"], market["q"], market["T"],
        )
        dS = S * 0.05
        dvol = 0.01
        dr = 0.005
        scenario = MarketScenario(
            spot_shocks=jnp.array([dS]),
            vol_shocks=jnp.array([dvol]),
            rate_shocks=jnp.full(2, dr),
            dividend_shocks=jnp.zeros(1),
        )
        attr = pnl_attribution(_bs_market_fn, valax_instruments, base_market, scenario)
        ql_base = _ql_reprice(S, sigma, r, q, K, T)
        ql_shocked = _ql_reprice(S + dS, sigma + dvol, r + dr, q, K, T)
        ql_actual = ql_shocked - ql_base
        assert abs(float(attr["actual"]) - ql_actual) < 1e-4, (
            f"Actual P&L mismatch: VALAX={float(attr['actual'])}, QL={ql_actual}"
        )


# ---------------------------------------------------------------------------
# Section D: Parametric VaR
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QUANTLIB, reason="QuantLib not installed")
class TestParametricVaR:
    """VALAX parametric_var must match manual QL-Greeks-based calculation.

    See: examples/comparisons/08_risk_greeks_var.py Section D
    """

    def test_parametric_var_matches_ql_manual(self, market, valax_instruments, base_market, ql_option):
        """VaR from VALAX autodiff must match VaR from QL Greeks + same formula."""
        import scipy.stats

        confidence = 0.99

        # Covariance: [spot, vol, rate_pillar_0, rate_pillar_1, dividend]
        spot_std = market["S"] * 0.02
        vol_std = 0.002
        rate_std = 0.001
        div_std = 0.0001
        cov = jnp.diag(jnp.array([
            spot_std**2, vol_std**2, rate_std**2, rate_std**2, div_std**2,
        ]))

        # VALAX
        valax_pvar = float(parametric_var(
            _bs_market_fn, valax_instruments, base_market, cov, confidence,
        ))

        # Manual with QL Greeks
        opt = ql_option[0]
        # Delta vector matching covariance column order:
        # [spot, vol, rate_0(t=0, no sensitivity), rate_1(t=1), dividend]
        ql_delta_vec = np.array([
            opt.delta(),
            opt.vega(),
            0.0,          # pillar 0 at t=0 has zero sensitivity
            opt.rho(),    # pillar 1 at t=1
            opt.dividendRho() if hasattr(opt, 'dividendRho') else 0.0,
        ])
        cov_np = np.array(cov)
        z_alpha = scipy.stats.norm.ppf(confidence)
        port_var = ql_delta_vec @ cov_np @ ql_delta_vec
        ql_pvar = z_alpha * np.sqrt(max(port_var, 0.0))

        assert abs(valax_pvar - ql_pvar) < 1e-4, (
            f"Parametric VaR mismatch: VALAX={valax_pvar}, QL={ql_pvar}"
        )

    def test_parametric_var_positive(self, market, valax_instruments, base_market):
        """Parametric VaR should be positive for a risky position."""
        spot_std = market["S"] * 0.02
        cov = jnp.diag(jnp.array([
            spot_std**2, 0.002**2, 0.001**2, 0.001**2, 0.0001**2,
        ]))
        pvar = float(parametric_var(
            _bs_market_fn, valax_instruments, base_market, cov, 0.99,
        ))
        assert pvar > 0, f"Parametric VaR should be positive, got {pvar}"

    def test_parametric_var_scales_with_confidence(self, market, valax_instruments, base_market):
        """Higher confidence => higher VaR."""
        spot_std = market["S"] * 0.02
        cov = jnp.diag(jnp.array([
            spot_std**2, 0.002**2, 0.001**2, 0.001**2, 0.0001**2,
        ]))
        pvar_95 = float(parametric_var(
            _bs_market_fn, valax_instruments, base_market, cov, 0.95,
        ))
        pvar_99 = float(parametric_var(
            _bs_market_fn, valax_instruments, base_market, cov, 0.99,
        ))
        assert pvar_99 > pvar_95, (
            f"VaR(99%)={pvar_99} should exceed VaR(95%)={pvar_95}"
        )
