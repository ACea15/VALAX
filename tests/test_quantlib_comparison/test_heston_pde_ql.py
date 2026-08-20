"""Cross-validation: VALAX Heston ADI PDE vs QuantLib.

Compares the VALAX 2-D ADI finite-difference Heston price against two QuantLib
engines on identical inputs (integer-day-aligned expiry, flat Act/365 curves):

* ``AnalyticHestonEngine`` — the semi-analytic reference; VALAX's PDE must match
  it tightly (both approximate the same exact price, VALAX to grid tolerance).
* ``FdHestonVanillaEngine`` — QuantLib's own Hundsdorfer ADI engine; the two
  independent finite-difference implementations must agree to a few mils.
"""

import jax.numpy as jnp
import pytest

ql = pytest.importorskip("QuantLib")

from valax.instruments.options import EuropeanOption
from valax.models.heston import HestonModel
from valax.pricing.pde import PDEConfig2D, pde_price_dispatch
from valax.pricing.pde.config import Scheme

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    market_to_ql_heston_process,
    snap_expiry_to_days,
)

SPOT, RATE, DIV = 100.0, 0.03, 0.0
V0, KAPPA, THETA, XI, RHO = 0.04, 2.0, 0.04, 0.4, -0.5
STRIKES = (90.0, 100.0, 110.0)


@pytest.fixture(scope="module")
def setup():
    market = {
        "spot": jnp.array(SPOT),
        "rate": jnp.array(RATE),
        "dividend": jnp.array(DIV),
        "expiry": jnp.array(1.0),
    }
    process, eff = market_to_ql_heston_process(
        market, V0, KAPPA, THETA, XI, RHO
    )
    t_eff = float(eff["expiry"])
    days = snap_expiry_to_days(1.0)[0]
    maturity = DEFAULT_QL_DATE + ql.Period(days, ql.Days)
    hmodel = ql.HestonModel(process)

    model = HestonModel(
        v0=jnp.array(V0), kappa=jnp.array(KAPPA), theta=jnp.array(THETA),
        xi=jnp.array(XI), rho=jnp.array(RHO),
        rate=jnp.array(RATE), dividend=jnp.array(DIV),
    )
    cfg = PDEConfig2D(
        n_x=300, n_y=120, n_time=150, x_range=5.0, y_max=0.7, y_scale=0.03,
        scheme=Scheme.HV,
    )
    return hmodel, maturity, model, cfg, t_eff


def _ql_price(hmodel, maturity, strike, engine):
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(maturity))
    option.setPricingEngine(engine)
    return option.NPV()


def _valax_price(model, cfg, strike, t_eff):
    opt = EuropeanOption(
        strike=jnp.array(strike), expiry=jnp.array(t_eff), is_call=True
    )
    return float(pde_price_dispatch(opt, model, cfg, spot=jnp.array(SPOT)))


@pytest.mark.parametrize("strike", STRIKES)
def test_matches_ql_analytic_heston(setup, strike):
    hmodel, maturity, model, cfg, t_eff = setup
    engine = ql.AnalyticHestonEngine(hmodel)
    ql_price = _ql_price(hmodel, maturity, strike, engine)
    valax = _valax_price(model, cfg, strike, t_eff)
    rel = abs(valax - ql_price) / ql_price
    assert rel < 1e-3, f"K={strike}: valax={valax:.5f} ql={ql_price:.5f} rel={rel:.2e}"


@pytest.mark.parametrize("strike", STRIKES)
def test_matches_ql_fd_heston(setup, strike):
    hmodel, maturity, model, cfg, t_eff = setup
    # QuantLib's own Hundsdorfer ADI engine (tGrid, xGrid, vGrid, dampingSteps).
    engine = ql.FdHestonVanillaEngine(
        hmodel, 100, 200, 100, 0, ql.FdmSchemeDesc.Hundsdorfer()
    )
    ql_price = _ql_price(hmodel, maturity, strike, engine)
    valax = _valax_price(model, cfg, strike, t_eff)
    assert abs(valax - ql_price) < 0.02, f"K={strike}: valax={valax:.5f} qlfd={ql_price:.5f}"
