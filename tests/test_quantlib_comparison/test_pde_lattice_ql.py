"""Cross-validation: VALAX vs QuantLib for PDE and lattice methods.

Stage 1 of the [QuantLib Validation Pyramid](
../../docs/architecture/quantlib-validation-pyramid.md). Parametric
sweep of Crank-Nicolson PDE and CRR binomial tree against the
matching QuantLib engines.

These methods are numerical, so tolerances are looser than the
analytic tests:
  - PDE vs analytic BS: relative error scales with `1 / n_spot²`.
  - Tree vs analytic BS: relative error scales with `1 / sqrt(n_steps)`.
"""

import jax
import jax.numpy as jnp
import pytest
import QuantLib as ql

import valax
from valax.instruments.options import EuropeanOption
from valax.market import SeedRegistry, default_config, sample_scalar_market
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.lattice.binomial import BinomialConfig, binomial_price
from valax.pricing.pde.solvers import PDEConfig, pde_price

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    market_to_ql_bsm,
    snap_expiry_to_days,
)


SEEDS = tuple(range(20260101, 20260121))


@pytest.fixture(params=SEEDS, ids=lambda s: f"seed={s}")
def call_setup(request):
    seed = request.param
    registry = SeedRegistry(
        master_seed=seed, library_version=valax.__version__,
    )
    cfg = default_config(n_assets=3)
    raw = sample_scalar_market(registry, cfg)
    ql_opt, ql_proc, eff = market_to_ql_bsm(raw, is_call=True)
    valax_opt = EuropeanOption(
        strike=eff["strike"], expiry=eff["expiry"], is_call=True,
    )
    return {
        "market": eff, "valax_opt": valax_opt,
        "ql_opt": ql_opt, "ql_proc": ql_proc,
    }


class TestCrankNicolson:
    """Crank-Nicolson PDE convergence vs analytic BS and vs QL FD."""

    def test_pde_converges_to_bs(self, call_setup):
        m = call_setup["market"]
        bs_ref = float(black_scholes_price(
            call_setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
        ))
        pde_p = float(pde_price(
            call_setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
            config=PDEConfig(n_spot=400, n_time=400),
        ))
        # 400x400 grid: ~0.5% rel error is the empirical floor.
        assert abs(pde_p - bs_ref) / abs(bs_ref) < 5e-3

    def test_pde_matches_quantlib_fd(self, call_setup):
        m = call_setup["market"]
        valax_p = float(pde_price(
            call_setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
            config=PDEConfig(n_spot=200, n_time=200),
        ))
        # Use a fresh QL VanillaOption with the FD engine so we
        # don't conflict with the analytic engine on the same option.
        ql_opt = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, float(m["strike"])),
            ql.EuropeanExercise(
                DEFAULT_QL_DATE
                + ql.Period(snap_expiry_to_days(float(m["expiry"]))[0], ql.Days)
            ),
        )
        ql_opt.setPricingEngine(
            ql.FdBlackScholesVanillaEngine(call_setup["ql_proc"], 200, 200)
        )
        ql_p = ql_opt.NPV()
        # Two different FD implementations agree to ~1%.
        assert abs(valax_p - ql_p) / abs(ql_p) < 1e-2


class TestBinomialTree:
    """VALAX CRR binomial vs analytic BS and vs QL CRR."""

    def test_european_converges_to_bs(self, call_setup):
        m = call_setup["market"]
        bs_ref = float(black_scholes_price(
            call_setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
        ))
        tree_p = float(binomial_price(
            call_setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
            config=BinomialConfig(n_steps=500, american=False),
        ))
        assert abs(tree_p - bs_ref) / abs(bs_ref) < 1e-2

    def test_european_matches_quantlib_crr(self, call_setup):
        m = call_setup["market"]
        valax_p = float(binomial_price(
            call_setup["valax_opt"],
            m["spot"], m["vol"], m["rate"], m["dividend"],
            config=BinomialConfig(n_steps=500),
        ))
        ql_opt = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, float(m["strike"])),
            ql.EuropeanExercise(
                DEFAULT_QL_DATE
                + ql.Period(snap_expiry_to_days(float(m["expiry"]))[0], ql.Days)
            ),
        )
        ql_opt.setPricingEngine(
            ql.BinomialVanillaEngine(call_setup["ql_proc"], "crr", 500)
        )
        ql_p = ql_opt.NPV()
        assert abs(valax_p - ql_p) / abs(ql_p) < 5e-3


class TestAmericanOptions:
    """American put: early exercise premium vs QuantLib CRR."""

    def test_american_put_premium_matches_quantlib(self, call_setup):
        m = call_setup["market"]
        put = EuropeanOption(
            strike=m["strike"], expiry=m["expiry"], is_call=False,
        )
        v_euro = float(binomial_price(
            put, m["spot"], m["vol"], m["rate"], m["dividend"],
            config=BinomialConfig(n_steps=500, american=False),
        ))
        v_amer = float(binomial_price(
            put, m["spot"], m["vol"], m["rate"], m["dividend"],
            config=BinomialConfig(n_steps=500, american=True),
        ))
        v_prem = v_amer - v_euro

        days = snap_expiry_to_days(float(m["expiry"]))[0]
        maturity = DEFAULT_QL_DATE + ql.Period(days, ql.Days)
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, float(m["strike"]))

        ql_euro = ql.VanillaOption(payoff, ql.EuropeanExercise(maturity))
        ql_euro.setPricingEngine(
            ql.BinomialVanillaEngine(call_setup["ql_proc"], "crr", 500)
        )
        ql_amer = ql.VanillaOption(
            payoff, ql.AmericanExercise(DEFAULT_QL_DATE, maturity)
        )
        ql_amer.setPricingEngine(
            ql.BinomialVanillaEngine(call_setup["ql_proc"], "crr", 500)
        )
        q_prem = ql_amer.NPV() - ql_euro.NPV()
        # Premiums can be small in absolute terms; relative tol is more
        # informative when the premium itself is substantial.
        assert abs(v_prem - q_prem) < 0.05, (
            f"market={ {k: float(x) for k, x in m.items()} } "
            f"VALAX prem={v_prem:.6f} QL prem={q_prem:.6f}"
        )

    def test_american_put_geq_european(self, call_setup):
        m = call_setup["market"]
        put = EuropeanOption(
            strike=m["strike"], expiry=m["expiry"], is_call=False,
        )
        euro = float(binomial_price(
            put, m["spot"], m["vol"], m["rate"], m["dividend"],
            config=BinomialConfig(n_steps=500, american=False),
        ))
        amer = float(binomial_price(
            put, m["spot"], m["vol"], m["rate"], m["dividend"],
            config=BinomialConfig(n_steps=500, american=True),
        ))
        assert amer >= euro - 1e-10
