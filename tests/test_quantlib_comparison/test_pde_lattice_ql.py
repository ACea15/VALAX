"""
Cross-validation: VALAX vs QuantLib for PDE and lattice methods.

Tests Crank-Nicolson PDE and CRR binomial tree pricing against
QuantLib's equivalent engines. Also validates American option
early exercise premium.

Companion example: examples/comparisons/06_pde_and_lattice.py
"""

import pytest
import jax
import jax.numpy as jnp
import QuantLib as ql
from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.pde.solvers import pde_price, PDEConfig
from valax.pricing.lattice.binomial import binomial_price, BinomialConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def market():
    return dict(S=100.0, K=100.0, T=1.0, sigma=0.25, r=0.05, q=0.02)


@pytest.fixture
def valax_args(market):
    return (
        jnp.array(market["S"]),
        jnp.array(market["sigma"]),
        jnp.array(market["r"]),
        jnp.array(market["q"]),
    )


@pytest.fixture
def ql_process(market):
    today = ql.Date(26, 3, 2026)
    ql.Settings.instance().evaluationDate = today
    spot_h = ql.QuoteHandle(ql.SimpleQuote(market["S"]))
    rate_h = ql.YieldTermStructureHandle(ql.FlatForward(today, market["r"], ql.Actual365Fixed()))
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(today, market["q"], ql.Actual365Fixed()))
    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), market["sigma"], ql.Actual365Fixed())
    )
    return ql.BlackScholesMertonProcess(spot_h, div_h, rate_h, vol_h), today


# ---------------------------------------------------------------------------
# PDE tests
# ---------------------------------------------------------------------------

class TestCrankNicolson:
    """VALAX Crank-Nicolson PDE vs QuantLib FdBlackScholes."""

    def test_european_call_pde_converges_to_bs(self, market, valax_args):
        """See: examples/comparisons/06_pde_and_lattice.py §2 (PDE convergence)"""
        call = EuropeanOption(strike=jnp.array(market["K"]), expiry=jnp.array(market["T"]), is_call=True)
        spot, vol, rate, div = valax_args
        bs_ref = float(black_scholes_price(call, spot, vol, rate, div))

        # 400x400 grid should be within 0.1% of BS
        cfg = PDEConfig(n_spot=400, n_time=400)
        pde_p = float(pde_price(call, spot, vol, rate, div, config=cfg))
        assert abs(pde_p - bs_ref) / bs_ref < 1e-3

    def test_pde_matches_quantlib_fd(self, market, valax_args, ql_process):
        """See: examples/comparisons/06_pde_and_lattice.py §2 (head-to-head)"""
        call = EuropeanOption(strike=jnp.array(market["K"]), expiry=jnp.array(market["T"]), is_call=True)
        spot, vol, rate, div = valax_args
        valax_p = float(pde_price(call, spot, vol, rate, div, config=PDEConfig(n_spot=200, n_time=200)))

        process, today = ql_process
        maturity = today + ql.Period(1, ql.Years)
        opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, market["K"]),
                               ql.EuropeanExercise(maturity))
        opt.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, 200, 200))
        ql_p = opt.NPV()

        # Both should be close — different implementations of same method
        assert abs(valax_p - ql_p) / ql_p < 5e-3

    def test_pde_delta_close_to_bs(self, market, valax_args):
        """See: examples/comparisons/06_pde_and_lattice.py §3 (PDE Greeks)"""
        call = EuropeanOption(strike=jnp.array(market["K"]), expiry=jnp.array(market["T"]), is_call=True)
        spot, vol, rate, div = valax_args

        pde_delta = float(jax.grad(lambda s: pde_price(call, s, vol, rate, div))(spot))
        bs_delta = float(jax.grad(lambda s: black_scholes_price(call, s, vol, rate, div))(spot))

        assert abs(pde_delta - bs_delta) < 0.01


# ---------------------------------------------------------------------------
# Binomial tree tests
# ---------------------------------------------------------------------------

class TestBinomialTree:
    """VALAX CRR binomial vs QuantLib BinomialVanillaEngine."""

    def test_european_call_converges_to_bs(self, market, valax_args):
        """See: examples/comparisons/06_pde_and_lattice.py §4 (tree convergence)"""
        call = EuropeanOption(strike=jnp.array(market["K"]), expiry=jnp.array(market["T"]), is_call=True)
        spot, vol, rate, div = valax_args
        bs_ref = float(black_scholes_price(call, spot, vol, rate, div))

        cfg = BinomialConfig(n_steps=500, american=False)
        tree_p = float(binomial_price(call, spot, vol, rate, div, config=cfg))
        assert abs(tree_p - bs_ref) / bs_ref < 5e-3

    def test_european_matches_quantlib_crr(self, market, valax_args, ql_process):
        """See: examples/comparisons/06_pde_and_lattice.py §4 (head-to-head)"""
        call = EuropeanOption(strike=jnp.array(market["K"]), expiry=jnp.array(market["T"]), is_call=True)
        spot, vol, rate, div = valax_args
        valax_p = float(binomial_price(call, spot, vol, rate, div, config=BinomialConfig(n_steps=500)))

        process, today = ql_process
        maturity = today + ql.Period(1, ql.Years)
        opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, market["K"]),
                               ql.EuropeanExercise(maturity))
        opt.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", 500))
        ql_p = opt.NPV()

        assert abs(valax_p - ql_p) / ql_p < 1e-3


class TestAmericanOptions:
    """American option pricing: early exercise premium."""

    @pytest.mark.parametrize("K", [80.0, 90.0, 100.0, 110.0, 120.0])
    def test_american_put_premium_matches_quantlib(self, market, K, ql_process):
        """See: examples/comparisons/06_pde_and_lattice.py §5 (American puts)

        VALAX and QuantLib should produce nearly identical early exercise
        premiums across strikes.
        """
        put = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(market["T"]), is_call=False)
        spot, vol, rate, div = (
            jnp.array(market["S"]), jnp.array(market["sigma"]),
            jnp.array(market["r"]), jnp.array(market["q"]),
        )

        v_euro = float(binomial_price(put, spot, vol, rate, div, config=BinomialConfig(n_steps=500, american=False)))
        v_amer = float(binomial_price(put, spot, vol, rate, div, config=BinomialConfig(n_steps=500, american=True)))
        v_prem = v_amer - v_euro

        process, today = ql_process
        maturity = today + ql.Period(1, ql.Years)
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)

        ql_euro = ql.VanillaOption(payoff, ql.EuropeanExercise(maturity))
        ql_euro.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", 500))
        ql_amer = ql.VanillaOption(payoff, ql.AmericanExercise(today, maturity))
        ql_amer.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", 500))
        q_prem = ql_amer.NPV() - ql_euro.NPV()

        assert abs(v_prem - q_prem) < 0.01, (
            f"K={K}: VALAX premium={v_prem:.4f}, QL premium={q_prem:.4f}"
        )

    def test_american_put_geq_european(self, market, valax_args):
        """American put must always be worth at least as much as European."""
        put = EuropeanOption(strike=jnp.array(market["K"]), expiry=jnp.array(market["T"]), is_call=False)
        spot, vol, rate, div = valax_args

        euro = float(binomial_price(put, spot, vol, rate, div, config=BinomialConfig(n_steps=500, american=False)))
        amer = float(binomial_price(put, spot, vol, rate, div, config=BinomialConfig(n_steps=500, american=True)))
        assert amer >= euro - 1e-10
