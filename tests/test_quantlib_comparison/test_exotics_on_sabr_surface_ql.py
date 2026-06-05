"""Stage 3 — chain validation: calibrated SABR surface → European prices.

Stage 3 of the [QuantLib Validation Pyramid](
../../docs/architecture/quantlib-validation-pyramid.md).

Chain tested:

  noisy smile → SABR fit (QL) → smile evaluated at exotic strikes →
  Black-Scholes pricer reading vol from the surface (both libraries) →
  agreement assertion.

**Design rule.** The SABR surface is fit *once* (by QL) and the
fitted parameters are *adopted* into VALAX. Both engines then price
the exotic on this single shared surface, so the test isolates
"pricer reading a surface" from "calibrator producing a surface" —
the latter is the job of Stage 2. Any disagreement here is
unambiguously a wiring bug at the calibrator-to-pricer boundary.
"""

import jax.numpy as jnp
import pytest
import QuantLib as ql

import valax
from valax.instruments.options import EuropeanOption
from valax.market import (
    SeedRegistry,
    default_config,
    sample_sabr_params,
    sample_scalar_market,
    synthesize_sabr_smile,
)
from valax.models.sabr import SABRModel
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.analytic.sabr import sabr_implied_vol

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    snap_expiry_to_days,
)


SEEDS = tuple(range(20260101, 20260111))   # 10 seeds — calibration adds cost


EXOTIC_MONEYNESS = (0.82, 0.88, 0.97, 1.03, 1.12, 1.22)


@pytest.fixture(params=SEEDS, ids=lambda s: f"seed={s}")
def shared_surface(request):
    """Build a SABR surface via QL calibration and adopt the params
    into a VALAX SABRModel.  Both engines see the *same* fit."""
    seed = request.param
    registry = SeedRegistry(
        master_seed=seed, library_version=valax.__version__,
    )
    cfg = default_config(n_assets=1)
    market = sample_scalar_market(registry, cfg)
    truth = sample_sabr_params(registry, cfg, fixed_beta=0.5)

    F = float(market["spot"]) * float(
        jnp.exp((market["rate"] - market["dividend"]) * market["expiry"])
    )
    T = float(market["expiry"])
    quote_strikes = F * jnp.array(
        [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
    )
    noisy_smile = synthesize_sabr_smile(
        registry, truth, jnp.array(F), jnp.array(T),
        quote_strikes, vol_bp_noise=5.0,
    )

    # QL fits.
    qsmile = ql.SABRInterpolation(
        [float(k) for k in quote_strikes],
        [float(v) for v in noisy_smile],
        T, F,
        0.2, float(truth.beta), 0.3, -0.2,
        False, True, False, False, False,    # vegaWeighted=False
    )
    # Triggering one __call__ forces the lazy fit.
    _ = qsmile(F, True)

    # Adopt the fitted parameters into a VALAX SABRModel — this is
    # the "shared surface" the rule requires.
    valax_sabr = SABRModel(
        alpha=jnp.array(qsmile.alpha()),
        beta=jnp.array(qsmile.beta()),
        rho=jnp.array(qsmile.rho()),
        nu=jnp.array(qsmile.nu()),
    )

    return {
        "market": market, "F": F, "T": T,
        "qsmile": qsmile, "valax_sabr": valax_sabr,
    }


class TestEuropeanPricedFromSABRSurface:
    """For each exotic strike, both engines read the smile vol from the
    *same* (calibrator-produced) surface and price a European call via
    Black-Scholes. The prices must agree to floating-point precision
    because the smile is shared and both BS implementations should
    produce the same number when given the same vol."""

    @pytest.mark.parametrize("moneyness", EXOTIC_MONEYNESS)
    def test_call_price_matches(self, shared_surface, moneyness):
        market = shared_surface["market"]
        F = shared_surface["F"]
        T = shared_surface["T"]
        K = F * moneyness

        # Vol from the calibrated surface, read by each library.
        v_vol = float(sabr_implied_vol(
            shared_surface["valax_sabr"], jnp.array(F),
            jnp.array(K), jnp.array(T),
        ))
        q_vol = float(shared_surface["qsmile"](K, True))
        # Pre-flight: both libraries see the same surface, so the
        # vols agree. A failure here would mean the parameter-extraction
        # adapter dropped state.
        assert v_vol == pytest.approx(q_vol, abs=1e-12), (
            f"Shared-surface vol disagreement at K={K:.2f}: "
            f"VALAX={v_vol:.12f}  QL={q_vol:.12f}"
        )

        # Now price.
        days, T_eff = snap_expiry_to_days(T)
        valax_opt = EuropeanOption(
            strike=jnp.array(K), expiry=jnp.array(T_eff), is_call=True,
        )
        v_price = float(black_scholes_price(
            valax_opt, market["spot"], jnp.array(v_vol),
            market["rate"], market["dividend"],
        ))

        today = DEFAULT_QL_DATE
        ql.Settings.instance().evaluationDate = today
        dc = ql.Actual365Fixed()
        spot_h = ql.QuoteHandle(ql.SimpleQuote(float(market["spot"])))
        rate_h = ql.YieldTermStructureHandle(
            ql.FlatForward(today, float(market["rate"]), dc)
        )
        div_h = ql.YieldTermStructureHandle(
            ql.FlatForward(today, float(market["dividend"]), dc)
        )
        vol_h = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), q_vol, dc)
        )
        process = ql.BlackScholesMertonProcess(spot_h, div_h, rate_h, vol_h)
        ql_opt = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, K),
            ql.EuropeanExercise(today + ql.Period(days, ql.Days)),
        )
        ql_opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        q_price = ql_opt.NPV()

        assert v_price == pytest.approx(q_price, abs=1e-10), (
            f"K={K:.2f}  shared vol={v_vol:.6f}  "
            f"VALAX={v_price:.10f}  QL={q_price:.10f}"
        )
