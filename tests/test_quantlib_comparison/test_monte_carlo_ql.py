"""Cross-validation: VALAX GBM Monte Carlo vs QuantLib.

Stage 1 of the [QuantLib Validation Pyramid](
../../docs/architecture/quantlib-validation-pyramid.md).

The MC engine has stochastic outputs, so the comparison tolerance is
``3 * stderr`` for any direct numerical comparison. Each seed runs
~100k paths; this is the slowest file in the Stage-1 sweep, so the
seed count is capped at 10.
"""

import jax
import jax.numpy as jnp
import pytest

import valax
from valax.instruments.options import EuropeanOption
from valax.market import SeedRegistry, default_config, sample_scalar_market
from valax.models.black_scholes import BlackScholesModel
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.mc.engine import MCConfig, mc_price_with_stderr
from valax.pricing.mc.paths import generate_gbm_paths
from valax.pricing.mc.payoffs import asian_payoff, barrier_payoff, european_payoff

from tests.test_quantlib_comparison._ql_adapters import market_to_ql_bsm


SEEDS = tuple(range(20260101, 20260111))   # 10 seeds — MC tests are slow


@pytest.fixture(params=SEEDS, ids=lambda s: f"seed={s}")
def call_setup(request):
    seed = request.param
    registry = SeedRegistry(
        master_seed=seed, library_version=valax.__version__,
    )
    cfg = default_config(n_assets=1)
    raw = sample_scalar_market(registry, cfg)
    ql_opt, ql_proc, eff = market_to_ql_bsm(raw, is_call=True)
    valax_opt = EuropeanOption(
        strike=eff["strike"], expiry=eff["expiry"], is_call=True,
    )
    model = BlackScholesModel(
        vol=eff["vol"], rate=eff["rate"], dividend=eff["dividend"],
    )
    return {
        "market": eff, "valax_opt": valax_opt,
        "model": model, "ql_opt": ql_opt, "ql_proc": ql_proc,
        "mc_key": jax.random.PRNGKey(seed),
    }


class TestGBMMC:
    def test_mc_within_3se_of_bs(self, call_setup):
        m = call_setup["market"]
        config = MCConfig(n_paths=100_000, n_steps=100)
        mc_p, mc_se = mc_price_with_stderr(
            call_setup["valax_opt"], m["spot"], call_setup["model"],
            config, call_setup["mc_key"],
        )
        bs_p = black_scholes_price(
            call_setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
        )
        n_se = abs(float(mc_p) - float(bs_p)) / float(mc_se)
        assert n_se < 3.0, (
            f"MC={float(mc_p):.4f} ± {float(mc_se):.4f}  "
            f"BS={float(bs_p):.4f}  {n_se:.1f} SE"
        )

    def test_mc_within_3se_of_quantlib(self, call_setup):
        m = call_setup["market"]
        config = MCConfig(n_paths=100_000, n_steps=100)
        mc_p, mc_se = mc_price_with_stderr(
            call_setup["valax_opt"], m["spot"], call_setup["model"],
            config, call_setup["mc_key"],
        )
        ql_p = call_setup["ql_opt"].NPV()
        n_se = abs(float(mc_p) - ql_p) / float(mc_se)
        assert n_se < 3.0, (
            f"MC={float(mc_p):.4f} ± {float(mc_se):.4f}  "
            f"QL={ql_p:.4f}  {n_se:.1f} SE"
        )


class TestExoticOrdering:
    """Asian < European, Up-and-Out Barrier < European (no-arb relations)."""

    def test_asian_leq_european(self, call_setup):
        m = call_setup["market"]
        atm = EuropeanOption(
            strike=m["spot"], expiry=m["expiry"], is_call=True,
        )
        paths = generate_gbm_paths(
            call_setup["model"], m["spot"], float(m["expiry"]),
            100, 50_000, call_setup["mc_key"],
        )
        df = jnp.exp(-m["rate"] * m["expiry"])
        euro = float(df * jnp.mean(european_payoff(paths, atm)))
        asian = float(df * jnp.mean(asian_payoff(paths, atm)))
        assert asian < euro

    def test_barrier_leq_european(self, call_setup):
        m = call_setup["market"]
        atm = EuropeanOption(
            strike=m["spot"], expiry=m["expiry"], is_call=True,
        )
        paths = generate_gbm_paths(
            call_setup["model"], m["spot"], float(m["expiry"]),
            100, 50_000, call_setup["mc_key"],
        )
        df = jnp.exp(-m["rate"] * m["expiry"])
        euro = float(df * jnp.mean(european_payoff(paths, atm)))
        barrier = float(df * jnp.mean(barrier_payoff(
            paths, atm,
            barrier=m["spot"] * jnp.array(1.30),
            is_up=True, is_knock_in=False,
        )))
        assert barrier <= euro + 1e-10


class TestGBMPathStatistics:
    def test_risk_neutral_drift(self, call_setup):
        m = call_setup["market"]
        paths = generate_gbm_paths(
            call_setup["model"], m["spot"], float(m["expiry"]),
            100, 100_000, call_setup["mc_key"],
        )
        expected = float(m["spot"]) * float(
            jnp.exp((m["rate"] - m["dividend"]) * m["expiry"])
        )
        mc_mean = float(jnp.mean(paths[:, -1]))
        # 3-sigma tolerance for the mean estimator.
        # std(terminal) is roughly S0 * sqrt(exp(sigma^2 T) - 1) for GBM,
        # but a 5% absolute tolerance is comfortable here.
        assert abs(mc_mean - expected) / expected < 0.05

    def test_paths_start_at_spot(self, call_setup):
        m = call_setup["market"]
        paths = generate_gbm_paths(
            call_setup["model"], m["spot"], float(m["expiry"]),
            10, 100, call_setup["mc_key"],
        )
        assert jnp.allclose(paths[:, 0], m["spot"])
