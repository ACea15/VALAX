"""Cross-validation: VALAX risk engine vs QuantLib.

Stage 1 of the [QuantLib Validation Pyramid](
../../docs/architecture/quantlib-validation-pyramid.md).

Per-seed sampled market drives:
  * Greek comparison (delta, gamma, vega, rho, theta-per-day).
  * Repricing under spot / vol / rate shocks via the synthetic
    :class:`MarketScenario` + the VALAX risk engine, compared with
    QL's analytic engine on the shocked inputs.
  * P&L attribution: each first-order term must match the matching
    QL Greek × shock magnitude.
  * Parametric VaR: VALAX autodiff result must match a manual QL
    Greeks × covariance × z(α) computation.

The original 1e-4 absolute tolerances are preserved.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats
import QuantLib as ql

import valax
from valax.curves.discount import DiscountCurve
from valax.greeks.autodiff import greeks
from valax.instruments.options import EuropeanOption
from valax.market import SeedRegistry, default_config, sample_scalar_market
from valax.market.data import MarketData
from valax.market.scenario import MarketScenario
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.risk.var import (
    _extract_short_rate,
    parametric_var,
    pnl_attribution,
    reprice_under_scenario,
)

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    market_to_ql_bsm,
    snap_expiry_to_days,
)


SEEDS = tuple(range(20260101, 20260121))


def _bs_market_fn(option, market_data: MarketData):
    """Black-Scholes pricing function with MarketData signature."""
    rate = _extract_short_rate(market_data.discount_curve)
    return black_scholes_price(
        option, market_data.spots, market_data.vols, rate,
        market_data.dividends,
    )


def _ql_reprice(S, sigma, r, q, K, T_eff, days):
    """Reprice a QL European call at the given parameters.

    ``days`` is the integer-day-aligned expiry; ``T_eff`` is the
    matching year fraction.  Both engines must agree on this value
    (the adapter enforces this for the base market; the shocked
    repricing function constructs the QL date from ``days``).
    """
    today = DEFAULT_QL_DATE
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    spot_h = ql.QuoteHandle(ql.SimpleQuote(float(S)))
    rate_h = ql.YieldTermStructureHandle(ql.FlatForward(today, float(r), dc))
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(today, float(q), dc))
    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), float(sigma), dc)
    )
    proc = ql.BlackScholesMertonProcess(spot_h, div_h, rate_h, vol_h)
    mat = today + ql.Period(days, ql.Days)
    opt = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, float(K)),
        ql.EuropeanExercise(mat),
    )
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
    return opt.NPV()


@pytest.fixture(params=SEEDS, ids=lambda s: f"seed={s}")
def setup(request):
    """Per-seed market + matching VALAX MarketData + QL option."""
    seed = request.param
    registry = SeedRegistry(
        master_seed=seed, library_version=valax.__version__,
    )
    cfg = default_config(n_assets=1)
    raw = sample_scalar_market(registry, cfg)
    ql_opt, ql_proc, eff = market_to_ql_bsm(raw, is_call=True)
    days = snap_expiry_to_days(float(eff["expiry"]))[0]

    valax_opt = EuropeanOption(
        strike=eff["strike"], expiry=eff["expiry"], is_call=True,
    )
    valax_instruments = EuropeanOption(
        strike=jnp.array([float(eff["strike"])]),
        expiry=jnp.array([float(eff["expiry"])]),
        is_call=True,
    )

    # MarketData with a 2-pillar flat curve at the option expiry.
    from valax.dates.daycounts import ymd_to_ordinal
    ref = ymd_to_ordinal(2026, 1, 1)
    pillars = jnp.array(
        [int(ref), int(ref) + days], dtype=jnp.int32,
    )
    pillar_times = (pillars - int(ref)).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-float(eff["rate"]) * pillar_times)
    curve = DiscountCurve(
        pillar_dates=pillars, discount_factors=dfs,
        reference_date=jnp.int32(int(ref)),
    )
    base_market = MarketData(
        spots=jnp.array([float(eff["spot"])]),
        vols=jnp.array([float(eff["vol"])]),
        dividends=jnp.array([float(eff["dividend"])]),
        discount_curve=curve,
    )
    return {
        "market": eff, "days": days,
        "valax_opt": valax_opt, "valax_instruments": valax_instruments,
        "base_market": base_market,
        "ql_opt": ql_opt, "ql_proc": ql_proc,
    }


# ---------------------------------------------------------------------------
# Section A: Greeks
# ---------------------------------------------------------------------------


class TestGreeksMatchQuantLib:

    def test_delta_matches(self, setup):
        m = setup["market"]
        g = greeks(
            black_scholes_price, setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
        )
        assert float(g["delta"]) == pytest.approx(
            setup["ql_opt"].delta(), abs=1e-4
        )

    def test_gamma_matches(self, setup):
        m = setup["market"]
        g = greeks(
            black_scholes_price, setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
        )
        assert float(g["gamma"]) == pytest.approx(
            setup["ql_opt"].gamma(), abs=1e-4
        )

    def test_vega_matches(self, setup):
        m = setup["market"]
        g = greeks(
            black_scholes_price, setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
        )
        assert float(g["vega"]) == pytest.approx(
            setup["ql_opt"].vega(), abs=1e-4
        )

    def test_rho_matches(self, setup):
        m = setup["market"]
        g = greeks(
            black_scholes_price, setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
        )
        assert float(g["rho"]) == pytest.approx(
            setup["ql_opt"].rho(), abs=1e-4
        )

    def test_theta_per_day_matches(self, setup):
        m = setup["market"]
        g = greeks(
            black_scholes_price, setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
        )
        # VALAX theta is per year; QL theta is per day.
        valax_per_day = float(g["theta"]) / 365.0
        assert valax_per_day == pytest.approx(
            setup["ql_opt"].thetaPerDay(), abs=1e-3
        )


# ---------------------------------------------------------------------------
# Section B: Repricing under shocks
# ---------------------------------------------------------------------------


class TestRepricingUnderShocks:

    def _scenario(self, *, dS=0.0, dvol=0.0, dr=0.0):
        return MarketScenario(
            spot_shocks=jnp.array([dS]),
            vol_shocks=jnp.array([dvol]),
            rate_shocks=jnp.full(2, dr),
            dividend_shocks=jnp.zeros(1),
        )

    def test_spot_bump_matches(self, setup):
        m = setup["market"]
        dS = float(m["spot"]) * 0.05
        valax_val = float(reprice_under_scenario(
            _bs_market_fn, setup["valax_instruments"],
            setup["base_market"], self._scenario(dS=dS),
        ))
        ql_val = _ql_reprice(
            float(m["spot"]) + dS, m["vol"], m["rate"], m["dividend"],
            m["strike"], float(m["expiry"]), setup["days"],
        )
        assert valax_val == pytest.approx(ql_val, abs=1e-4)

    def test_vol_bump_matches(self, setup):
        m = setup["market"]
        dvol = 0.01
        valax_val = float(reprice_under_scenario(
            _bs_market_fn, setup["valax_instruments"],
            setup["base_market"], self._scenario(dvol=dvol),
        ))
        ql_val = _ql_reprice(
            m["spot"], float(m["vol"]) + dvol, m["rate"], m["dividend"],
            m["strike"], float(m["expiry"]), setup["days"],
        )
        assert valax_val == pytest.approx(ql_val, abs=1e-4)

    def test_rate_bump_matches(self, setup):
        m = setup["market"]
        dr = 0.005
        valax_val = float(reprice_under_scenario(
            _bs_market_fn, setup["valax_instruments"],
            setup["base_market"], self._scenario(dr=dr),
        ))
        ql_val = _ql_reprice(
            m["spot"], m["vol"], float(m["rate"]) + dr, m["dividend"],
            m["strike"], float(m["expiry"]), setup["days"],
        )
        assert valax_val == pytest.approx(ql_val, abs=1e-4)


# ---------------------------------------------------------------------------
# Section C: P&L attribution
# ---------------------------------------------------------------------------


class TestPnLAttribution:

    def test_delta_spot_matches_ql_greek(self, setup):
        m = setup["market"]
        dS = float(m["spot"]) * 0.05
        scenario = MarketScenario(
            spot_shocks=jnp.array([dS]),
            vol_shocks=jnp.zeros(1),
            rate_shocks=jnp.zeros(2),
            dividend_shocks=jnp.zeros(1),
        )
        attr = pnl_attribution(
            _bs_market_fn, setup["valax_instruments"],
            setup["base_market"], scenario,
        )
        ql_delta_pnl = setup["ql_opt"].delta() * dS
        assert float(attr["delta_spot"]) == pytest.approx(
            ql_delta_pnl, abs=1e-3
        )

    def test_delta_vol_matches_ql_vega(self, setup):
        dvol = 0.01
        scenario = MarketScenario(
            spot_shocks=jnp.zeros(1),
            vol_shocks=jnp.array([dvol]),
            rate_shocks=jnp.zeros(2),
            dividend_shocks=jnp.zeros(1),
        )
        attr = pnl_attribution(
            _bs_market_fn, setup["valax_instruments"],
            setup["base_market"], scenario,
        )
        ql_vega_pnl = setup["ql_opt"].vega() * dvol
        assert float(attr["delta_vol"]) == pytest.approx(
            ql_vega_pnl, abs=1e-3
        )


# ---------------------------------------------------------------------------
# Section D: Parametric VaR
# ---------------------------------------------------------------------------


class TestParametricVaR:

    def test_parametric_var_matches_ql_manual(self, setup):
        m = setup["market"]
        confidence = 0.99
        spot_std = float(m["spot"]) * 0.02
        cov = jnp.diag(jnp.array([
            spot_std**2, 0.002**2, 0.001**2, 0.001**2, 0.0001**2,
        ]))

        valax_pvar = float(parametric_var(
            _bs_market_fn, setup["valax_instruments"],
            setup["base_market"], cov, confidence,
        ))

        opt = setup["ql_opt"]
        ql_delta_vec = np.array([
            opt.delta(), opt.vega(), 0.0, opt.rho(),
            opt.dividendRho() if hasattr(opt, "dividendRho") else 0.0,
        ])
        cov_np = np.array(cov)
        z_alpha = scipy.stats.norm.ppf(confidence)
        port_var = ql_delta_vec @ cov_np @ ql_delta_vec
        ql_pvar = z_alpha * np.sqrt(max(port_var, 0.0))

        assert valax_pvar == pytest.approx(ql_pvar, abs=1e-3)

    def test_parametric_var_scales_with_confidence(self, setup):
        m = setup["market"]
        spot_std = float(m["spot"]) * 0.02
        cov = jnp.diag(jnp.array([
            spot_std**2, 0.002**2, 0.001**2, 0.001**2, 0.0001**2,
        ]))
        pvar_95 = float(parametric_var(
            _bs_market_fn, setup["valax_instruments"],
            setup["base_market"], cov, 0.95,
        ))
        pvar_99 = float(parametric_var(
            _bs_market_fn, setup["valax_instruments"],
            setup["base_market"], cov, 0.99,
        ))
        assert pvar_99 > pvar_95
