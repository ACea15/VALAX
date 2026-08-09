"""QuantLib cross-checks for the PR-1 PDE recipes.

Optional: skips cleanly if QuantLib is not installed (unlike the older QL test
files, this one uses ``pytest.importorskip``).
"""

import jax.numpy as jnp
import pytest

ql = pytest.importorskip("QuantLib")

from valax.greeks.autodiff import greek
from valax.instruments.options import (
    AmericanOption,
    EquityBarrierOption,
    EuropeanOption,
)
from valax.models.black_scholes import BlackScholesModel
from valax.pricing.pde import PDEConfig, pde_price_dispatch

TODAY = ql.Date(1, 1, 2026)
DAY_COUNT = ql.Actual365Fixed()
CAL = ql.NullCalendar()


def _ql_bsm_process(spot, vol, rate, dividend):
    ql.Settings.instance().evaluationDate = TODAY
    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(TODAY, rate, DAY_COUNT))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(TODAY, dividend, DAY_COUNT))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(TODAY, CAL, vol, DAY_COUNT)
    )
    return ql.BlackScholesMertonProcess(spot_h, q_ts, r_ts, vol_ts)


def test_american_put_matches_quantlib_crr():
    S, K, vol, r, q, T = 100.0, 100.0, 0.20, 0.05, 0.0, 1.0
    n_days = int(round(T * 365))
    expiry = jnp.array(n_days / 365.0)

    put = AmericanOption(strike=jnp.array(K), expiry=expiry, is_call=False)
    model = BlackScholesModel(vol=jnp.array(vol), rate=jnp.array(r), dividend=jnp.array(q))
    cfg = PDEConfig(n_spot=400, n_time=400, penalty_rho=1.0e6, penalty_iters=6)
    valax_p = float(pde_price_dispatch(put, model, cfg, spot=jnp.array(S)))

    process = _ql_bsm_process(S, vol, r, q)
    expiry_date = TODAY + ql.Period(n_days, ql.Days)
    ql_opt = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Put, K),
        ql.AmericanExercise(TODAY, expiry_date),
    )
    ql_opt.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", 1000))
    ql_p = ql_opt.NPV()

    assert abs(valax_p - ql_p) < 0.05, f"valax={valax_p:.5f} ql={ql_p:.5f}"


def test_european_matches_quantlib_fd():
    S, K, vol, r, q, T = 100.0, 100.0, 0.20, 0.05, 0.02, 1.0
    n_days = int(round(T * 365))
    expiry = jnp.array(n_days / 365.0)

    opt = EuropeanOption(strike=jnp.array(K), expiry=expiry, is_call=True)
    model = BlackScholesModel(vol=jnp.array(vol), rate=jnp.array(r), dividend=jnp.array(q))
    valax_p = float(
        pde_price_dispatch(opt, model, PDEConfig(n_spot=200, n_time=200), spot=jnp.array(S))
    )

    process = _ql_bsm_process(S, vol, r, q)
    expiry_date = TODAY + ql.Period(n_days, ql.Days)
    ql_opt = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, K),
        ql.EuropeanExercise(expiry_date),
    )
    ql_opt.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, 200, 200))
    ql_p = ql_opt.NPV()

    assert abs(valax_p - ql_p) / ql_p < 1e-2, f"valax={valax_p:.5f} ql={ql_p:.5f}"


def test_european_gamma_matches_quantlib():
    """Second-order spot Greek: PDE gamma (autodiff) vs QuantLib analytic gamma.

    Guards the read-off curvature fix end-to-end against an independent engine.
    """
    S, K, vol, r, q, T = 100.0, 100.0, 0.20, 0.05, 0.02, 1.0
    n_days = int(round(T * 365))
    expiry = jnp.array(n_days / 365.0)

    opt = EuropeanOption(strike=jnp.array(K), expiry=expiry, is_call=True)
    cfg = PDEConfig(n_spot=400, n_time=400)

    def price(o, spot, v, rate, div):
        model = BlackScholesModel(vol=v, rate=rate, dividend=div)
        return pde_price_dispatch(o, model, cfg, spot=spot).price

    valax_gamma = float(
        greek(price, "gamma", opt, jnp.array(S), jnp.array(vol), jnp.array(r), jnp.array(q))
    )

    process = _ql_bsm_process(S, vol, r, q)
    expiry_date = TODAY + ql.Period(n_days, ql.Days)
    ql_opt = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, K),
        ql.EuropeanExercise(expiry_date),
    )
    ql_opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    ql_gamma = ql_opt.gamma()

    assert abs(valax_gamma - ql_gamma) < 3.0e-4 + 0.02 * abs(ql_gamma), (
        f"valax={valax_gamma:.6f} ql={ql_gamma:.6f}"
    )


def test_barrier_gamma_matches_quantlib():
    """Barrier gamma (autodiff through the PDE) vs QuantLib.

    ``AnalyticBarrierEngine`` does not expose analytic delta/gamma, so the
    reference is a central finite difference of the QuantLib NPV (an
    independent engine). This guards the recipe-level ``stop_gradient`` grid
    detachment that makes barrier gamma well defined.
    """
    S, K, B, vol, r, q, T = 100.0, 100.0, 130.0, 0.20, 0.05, 0.0, 1.0
    n_days = int(round(T * 365))
    expiry = jnp.array(n_days / 365.0)
    cfg = PDEConfig(n_spot=400, n_time=400)

    ko = EquityBarrierOption(
        strike=jnp.array(K), expiry=expiry, barrier=jnp.array(B),
        is_call=True, is_up=True, is_knock_in=False,
    )

    def price(o, spot, v, rate, div):
        model = BlackScholesModel(vol=v, rate=rate, dividend=div)
        return pde_price_dispatch(o, model, cfg, spot=spot).price

    valax_gamma = float(
        greek(price, "gamma", ko, jnp.array(S), jnp.array(vol), jnp.array(r), jnp.array(q))
    )

    # QuantLib reference: central FD of the continuously-monitored barrier NPV.
    expiry_date = TODAY + ql.Period(n_days, ql.Days)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(expiry_date)

    def ql_npv(spot):
        opt = ql.BarrierOption(ql.Barrier.UpOut, B, 0.0, payoff, exercise)
        opt.setPricingEngine(ql.AnalyticBarrierEngine(_ql_bsm_process(spot, vol, r, q)))
        return opt.NPV()

    h = 0.5
    ql_gamma = (ql_npv(S + h) - 2.0 * ql_npv(S) + ql_npv(S - h)) / (h * h)

    assert abs(valax_gamma - ql_gamma) < 5.0e-4 + 0.03 * abs(ql_gamma), (
        f"valax={valax_gamma:.6f} ql={ql_gamma:.6f}"
    )
