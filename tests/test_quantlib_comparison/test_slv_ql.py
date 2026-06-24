"""Cross-validation: VALAX SLV (flat-leverage limit) vs QuantLib analytic Heston.

Stage 1 of the [QuantLib Validation Pyramid](
../../docs/architecture/quantlib-validation-pyramid.md).

The SLV MC path generator (``generate_slv_paths``) couples Andersen-QE
on the variance leg with a log-Euler/Milstein log-spot leg. Andersen's
exact K-formulation for the spot/variance correlation only applies in
the pure-Heston limit; under non-trivial leverage we use the
approximate coupling ``z_1 = ρ·z_v + √(1-ρ²)·z_⊥``. At ``L ≡ 1`` this
approximate coupling reduces to the exact Andersen formulation up to a
sub-1 bp bias (verified separately in
``tests/test_pricing/test_slv_paths.py::TestFlatLeverageReducesToHeston``).

This file pins **only the flat-leverage limit** against QL's
``AnalyticHestonEngine``. Calibrated-leverage cross-checks are
deliberately out of scope (mirrors the precedent of
``tests/test_quantlib_comparison/test_dupire_ql.py`` which avoids
non-flat-surface comparisons because of interpolation-convention drift
between QL and VALAX). QL's SLV engine (``HestonSLVProcess``) uses a
Fokker-Planck PDE for leverage calibration that is methodologically
incompatible with our particle method — a calibrated-leverage
comparison would be measuring scheme differences, not implementation
bugs.

Gate: VALAX SLV-MC price at ``L ≡ 1`` must agree with QL analytic
Heston within ``3 * mc_stderr`` at every strike on a 5-moneyness grid,
across 5 random parameter seeds.
"""

import jax
import jax.numpy as jnp
import pytest
import QuantLib as ql

import valax
from valax.market import (
    SeedRegistry,
    default_config,
    sample_heston_params,
    sample_scalar_market,
)
from valax.models import SLVModel
from valax.models.heston import HestonModel
from valax.pricing.mc import generate_slv_paths
from valax.surfaces import LeverageGrid

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    market_to_ql_heston_process,
    snap_expiry_to_days,
)


# SLV-MC is the slowest of the QE-coupled equity MCs because of the
# additional leverage-grid lookup per step; keep the seed count
# modest. The Heston-vs-QL test at ``tests/test_quantlib_comparison/
# test_heston_ql.py`` uses the same 5-seed convention.
SEEDS = tuple(range(20260101, 20260106))


@pytest.fixture(params=SEEDS, ids=lambda s: f"seed={s}")
def slv_flat_setup(request):
    """Sample a market + Heston params, build a flat-leverage SLV +
    matched QL Heston analytic engine."""
    seed = request.param
    registry = SeedRegistry(
        master_seed=seed, library_version=valax.__version__,
    )
    cfg = default_config(n_assets=1)
    raw_market = sample_scalar_market(registry, cfg)
    heston = sample_heston_params(registry, cfg, enforce_feller=True)

    process, eff_market = market_to_ql_heston_process(
        raw_market,
        v0=float(heston.v0), kappa=float(heston.kappa),
        theta=float(heston.theta), xi=float(heston.xi),
        rho=float(heston.rho),
    )
    # Align rate/dividend to the QL process's values (from raw_market)
    # — see the corresponding fixture in ``test_heston_ql.py`` for the
    # rationale.
    aligned = HestonModel(
        v0=heston.v0, kappa=heston.kappa, theta=heston.theta,
        xi=heston.xi, rho=heston.rho,
        rate=eff_market["rate"], dividend=eff_market["dividend"],
    )

    # Build a flat-leverage SLV. The leverage grid is dimensionally
    # static under JAX flatten; the (k, t) extent only needs to
    # comfortably cover the simulation's exploration range.
    T = float(eff_market["expiry"])
    leverage = LeverageGrid.flat(
        log_moneyness_grid=jnp.linspace(-1.0, 1.0, 5),
        time_grid=jnp.linspace(0.01, max(T, 2.0), 5),
        value=1.0,
    )
    # Surface field is required by SLVModel but unused at L ≡ 1; pass
    # a dummy SVI just to fill the pytree slot. (Path-generation never
    # queries the surface — see ``valax/pricing/mc/slv_paths.py``.)
    from valax.surfaces import SVIVolSurface
    dummy_surface = SVIVolSurface(
        expiries=jnp.array([0.05, max(T, 2.0)]),
        forwards=jnp.array([
            float(eff_market["spot"]) * 1.0,
            float(eff_market["spot"]) * 1.05,
        ]),
        a_vec=jnp.array([0.001, 0.04]),
        b_vec=jnp.array([0.05, 0.05]),
        rho_vec=jnp.array([0.0, 0.0]),
        m_vec=jnp.array([0.0, 0.0]),
        sigma_vec=jnp.array([0.1, 0.1]),
    )
    slv = SLVModel.from_heston_and_leverage(aligned, dummy_surface, leverage)

    # QL maturity (integer-day-aligned) and analytic engine.
    days = snap_expiry_to_days(T)[0]
    maturity = DEFAULT_QL_DATE + ql.Period(days, ql.Days)
    ql_engine = ql.AnalyticHestonEngine(ql.HestonModel(process))

    # Generate VALAX SLV paths once per seed (heavy step).
    mc_key = jax.random.PRNGKey(seed)
    spot_paths, _var_paths = generate_slv_paths(
        slv, eff_market["spot"], T,
        n_steps=100, n_paths=50_000, key=mc_key,
    )
    return {
        "market": eff_market,
        "spot_paths": spot_paths,
        "maturity": maturity,
        "ql_engine": ql_engine,
    }


def _mc_call_price(
    spot_paths, strike, rate, T,
) -> tuple[float, float]:
    payoff = jnp.maximum(spot_paths[:, -1] - strike, 0.0)
    df = jnp.exp(-rate * T)
    mean = float(df * jnp.mean(payoff))
    n = spot_paths.shape[0]
    se = float(df * jnp.std(payoff) / jnp.sqrt(n))
    return mean, se


class TestSLVFlatLeverageVsQLHeston:
    """SLV with L ≡ 1 must match QL Heston analytic within 3·stderr."""

    @pytest.mark.parametrize(
        "moneyness", [0.85, 0.95, 1.0, 1.05, 1.15],
    )
    def test_call_price_within_3se(self, slv_flat_setup, moneyness):
        m = slv_flat_setup["market"]
        K = float(m["spot"]) * moneyness
        mc_price, mc_se = _mc_call_price(
            slv_flat_setup["spot_paths"], K,
            float(m["rate"]), float(m["expiry"]),
        )

        opt = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, K),
            ql.EuropeanExercise(slv_flat_setup["maturity"]),
        )
        opt.setPricingEngine(slv_flat_setup["ql_engine"])
        ql_price = opt.NPV()

        n_se = abs(mc_price - ql_price) / max(mc_se, 1e-12)
        assert n_se < 3.0, (
            f"K/F={moneyness:.2f}  SLV-MC(L=1)={mc_price:.4f} ± "
            f"{mc_se:.4f}  QL-Heston={ql_price:.4f}  {n_se:.1f} SE"
        )
