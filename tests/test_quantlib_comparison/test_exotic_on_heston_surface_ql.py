"""Stage 3 — chain validation: calibrated Heston surface → Asian option.

Stage 3 of the [QuantLib Validation Pyramid](
../../docs/architecture/quantlib-validation-pyramid.md).

Chain tested::

    noisy SABR-style smile  →  QL calibrates a HestonModel  →
    VALAX adopts the fitted (v0, κ, θ, ξ, ρ) into its HestonModel  →
    both engines MC-price an arithmetic Asian call on the *same*
    calibrated Heston surface  →  agreement within ``3·combined_stderr``.

**Design rule (shared surface).** One library (QL) calibrates; both
libraries price on the *adopted* surface.  This isolates "pricer
reading a calibrated surface" from "calibrator producing a surface" —
the latter has already been verified in Stage 2.  A Stage-3 failure
when Stages 1+2 are green is necessarily a chain bug at the
calibrator-to-pricer boundary.

**Why this file is enabled.**  Previously gated on roadmap item
**HE-1** (Heston Euler bias under violated Feller).  Sprint 5
replaced ``valax.pricing.mc.paths.generate_heston_paths`` with the
Andersen (2008) QE scheme, which is bias-free at the absorbing
variance boundary that QL's single-expiry calibrator routinely
produces.  With QE in place the chain test passes at the standard
``3·combined_stderr`` band; see the validation-pyramid session log.
"""

import jax
import jax.numpy as jnp
import pytest
import QuantLib as ql

import valax
from valax.instruments.options import AsianOption
from valax.market import (
    SeedRegistry,
    default_config,
    sample_sabr_params,
    sample_scalar_market,
    synthesize_sabr_smile,
)
from valax.models.heston import HestonModel
from valax.pricing.mc.paths import generate_heston_paths
from valax.pricing.mc.payoffs import asian_option_payoff

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    snap_expiry_to_days,
)


# Three calibration seeds (the SABR→Heston fit is slow); the
# moneyness grid is intentionally narrow because Asian deep-OTM is
# noise-dominated under both engines.
SEEDS = tuple(range(20260101, 20260104))
MONEYNESS_GRID = (0.95, 1.00, 1.05)

# Fixing schedule density.  ``N_FIXINGS = 12`` mimics monthly
# observations on a roughly one-year horizon.  Both engines see
# *the same* day-count-aligned fixing dates.
N_FIXINGS = 12


@pytest.fixture(params=SEEDS, ids=lambda s: f"seed={s}")
def heston_surface(request):
    """Build a calibrated Heston surface shared between both engines.

    Steps mirror the Stage-2 `calibrated_heston` fixture, but we also
    materialise the discrete fixing schedule used by the Asian payoff
    and rebuild a QL ``HestonProcess`` from the *fitted* parameters
    (so the QL Asian engine and VALAX MC see bit-identical
    parameters, not just the calibrated `HestonModel`).
    """
    seed = request.param
    registry = SeedRegistry(
        master_seed=seed, library_version=valax.__version__,
    )
    cfg = default_config(n_assets=1)
    raw = sample_scalar_market(registry, cfg)

    F = float(raw["spot"]) * float(
        jnp.exp((raw["rate"] - raw["dividend"]) * raw["expiry"])
    )
    expiry_yr = float(raw["expiry"])
    sabr_truth = sample_sabr_params(registry, cfg, fixed_beta=0.5)
    smile_strikes = jnp.array(
        [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]
    ) * float(raw["spot"])
    smile_vols = synthesize_sabr_smile(
        registry, sabr_truth,
        jnp.array(F), jnp.array(expiry_yr),
        smile_strikes, vol_bp_noise=0.0,
    )

    today = DEFAULT_QL_DATE
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    days = snap_expiry_to_days(expiry_yr)[0]
    maturity_period = ql.Period(days, ql.Days)
    maturity_date = today + maturity_period

    spot_h = ql.QuoteHandle(ql.SimpleQuote(float(raw["spot"])))
    rate_h = ql.YieldTermStructureHandle(
        ql.FlatForward(today, float(raw["rate"]), dc)
    )
    div_h = ql.YieldTermStructureHandle(
        ql.FlatForward(today, float(raw["dividend"]), dc)
    )

    # ── QL calibration to the SABR smile.
    helpers: list = []
    for strike, vol in zip(smile_strikes, smile_vols, strict=True):
        helpers.append(ql.HestonModelHelper(
            maturity_period, ql.NullCalendar(),
            float(raw["spot"]), float(strike),
            ql.QuoteHandle(ql.SimpleQuote(float(vol))),
            rate_h, div_h,
            ql.BlackCalibrationHelper.PriceError,
        ))
    seed_process = ql.HestonProcess(
        rate_h, div_h, spot_h,
        0.04, 2.0, 0.04, 0.5, -0.5,
    )
    ql_model = ql.HestonModel(seed_process)
    cal_engine = ql.AnalyticHestonEngine(ql_model)
    for h in helpers:
        h.setPricingEngine(cal_engine)
    ql_model.calibrate(
        helpers, ql.LevenbergMarquardt(),
        ql.EndCriteria(500, 50, 1e-8, 1e-8, 1e-8),
    )

    v0, kappa, theta, xi, rho = (
        ql_model.v0(), ql_model.kappa(), ql_model.theta(),
        ql_model.sigma(), ql_model.rho(),
    )

    # ── Adopt fitted parameters into VALAX HestonModel.
    valax_model = HestonModel(
        v0=jnp.array(v0), kappa=jnp.array(kappa),
        theta=jnp.array(theta), xi=jnp.array(xi), rho=jnp.array(rho),
        rate=jnp.array(float(raw["rate"])),
        dividend=jnp.array(float(raw["dividend"])),
    )

    # ── Build a *fresh* QL HestonProcess with the fitted parameters
    # so the MC Asian engine sees an identical model — not the
    # initial-guess process the calibrator was constructed from.
    fitted_process = ql.HestonProcess(
        rate_h, div_h, spot_h, v0, kappa, theta, xi, rho,
    )

    # ── Discrete fixing schedule, identical for both engines.
    # Day offsets are rounded so both sides land on the same integer
    # days from the evaluation date; ``snap_expiry_to_days`` already
    # ensured the terminal date is integer-aligned.
    fixing_dates = [
        today + ql.Period(int(round(days * i / N_FIXINGS)), ql.Days)
        for i in range(1, N_FIXINGS + 1)
    ]

    return {
        "market": raw,
        "expiry_yr": expiry_yr,
        "days": days,
        "valax_model": valax_model,
        "ql_process": fitted_process,
        "maturity_date": maturity_date,
        "fixing_dates": fixing_dates,
        "fitted_params": {
            "v0": v0, "kappa": kappa, "theta": theta, "xi": xi, "rho": rho,
        },
    }


class TestAsianOnCalibratedHestonSurface:
    """Stage 3.B — chain validation on a shared Heston surface.

    Both engines price an arithmetic Asian call (12 equally-spaced
    fixings) on the *same* calibrated Heston model. The means must
    agree to ``3 · combined_stderr``.

    Combined SE is ``sqrt(SE_VALAX² + SE_QL²)`` because the two MC
    runs use independent random number streams (JAX PRNG on one side,
    QL's Mersenne Twister on the other), so the difference is normal
    with that variance under the null.
    """

    @pytest.mark.parametrize("moneyness", MONEYNESS_GRID)
    def test_asian_call_within_3se(self, heston_surface, moneyness):
        m = heston_surface["market"]
        T = heston_surface["expiry_yr"]
        rate = float(m["rate"])
        K = float(m["spot"]) * moneyness

        # ── VALAX MC.  n_steps == N_FIXINGS so the simulated grid
        # lands on the fixing dates exactly; the Asian payoff averages
        # over `paths[:, 1:]`, which is precisely the 12 fixing
        # observations.
        n_paths = 20_000
        key = jax.random.PRNGKey(0)
        spot_paths, _ = generate_heston_paths(
            heston_surface["valax_model"], m["spot"], T,
            N_FIXINGS, n_paths, key,
        )
        asian = AsianOption(
            strike=jnp.array(K), expiry=jnp.array(T),
            is_call=True, averaging="arithmetic",
        )
        payoff_v = asian_option_payoff(spot_paths, asian)
        df = float(jnp.exp(-rate * T))
        v_mean = df * float(jnp.mean(payoff_v))
        v_se = df * float(jnp.std(payoff_v)) / float(jnp.sqrt(n_paths))

        # ── QL MC.  `MCDiscreteArithmeticAPHestonEngine` runs an
        # internal Euler discretisation; we let it pick its default
        # time-step density (it observes the explicit fixing dates).
        ql_payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
        ql_exercise = ql.EuropeanExercise(heston_surface["maturity_date"])
        asian_ql = ql.DiscreteAveragingAsianOption(
            ql.Average.Arithmetic,
            0.0, 0,
            heston_surface["fixing_dates"],
            ql_payoff, ql_exercise,
        )
        ql_engine = ql.MCDiscreteArithmeticAPHestonEngine(
            heston_surface["ql_process"],
            "pseudorandom",
            requiredSamples=n_paths,
            seed=42,
        )
        asian_ql.setPricingEngine(ql_engine)
        q_mean = asian_ql.NPV()
        q_se = asian_ql.errorEstimate()

        combined_se = (v_se ** 2 + q_se ** 2) ** 0.5
        n_se = abs(v_mean - q_mean) / max(combined_se, 1e-12)
        assert n_se < 3.0, (
            f"K={K:.2f}  "
            f"VALAX={v_mean:.6f} ± {v_se:.6f}  "
            f"QL={q_mean:.6f} ± {q_se:.6f}  "
            f"{n_se:.2f} SE  "
            f"params={heston_surface['fitted_params']}"
        )
