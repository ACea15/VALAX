"""QuantLib cross-checks for the PR-1 PDE recipes.

Optional: skips cleanly if QuantLib is not installed (unlike the older QL test
files, this one uses ``pytest.importorskip``).
"""

import jax.numpy as jnp
import pytest

ql = pytest.importorskip("QuantLib")

from valax.instruments.options import AmericanOption, EuropeanOption
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
