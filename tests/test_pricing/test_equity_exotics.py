"""Tests for equity exotic instruments and pricing."""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import (
    AmericanOption,
    AsianOption,
    EquityBarrierOption,
    EuropeanOption,
    LookbackOption,
    VarianceSwap,
)
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.analytic.variance_swap import (
    variance_swap_fair_strike,
    variance_swap_price,
    variance_swap_price_seasoned,
)
from valax.pricing.lattice.binomial import binomial_price, BinomialConfig
from valax.pricing.mc.payoffs import (
    asian_option_payoff,
    barrier_payoff,
    equity_barrier_payoff,
    european_payoff,
    lookback_payoff,
    variance_swap_payoff,
)


# ── Shared constants ─────────────────────────────────────────────────

SPOT = jnp.array(100.0)
VOL = jnp.array(0.20)
RATE = jnp.array(0.05)
DIV = jnp.array(0.0)
EXPIRY = jnp.array(1.0)
STRIKE = jnp.array(100.0)


def _generate_gbm_paths(key, spot, vol, rate, div, T, n_paths=10_000, n_steps=252):
    """Simple GBM path generator for testing."""
    dt = T / n_steps
    drift = (rate - div - 0.5 * vol**2) * dt
    diffusion = vol * jnp.sqrt(dt)
    z = jax.random.normal(key, shape=(n_paths, n_steps))
    log_returns = drift + diffusion * z
    log_paths = jnp.cumsum(log_returns, axis=1)
    log_paths = jnp.concatenate(
        [jnp.zeros((n_paths, 1)), log_paths], axis=1,
    )
    return spot * jnp.exp(log_paths)


@pytest.fixture
def paths():
    """GBM paths for MC tests."""
    return _generate_gbm_paths(
        jax.random.PRNGKey(42), SPOT, VOL, RATE, DIV, EXPIRY,
    )


# ── American Option ──────────────────────────────────────────────────


class TestAmericanOption:
    def test_pytree_construction(self):
        """AmericanOption should be a valid pytree."""
        opt = AmericanOption(strike=STRIKE, expiry=EXPIRY, is_call=False)
        assert opt.strike is not None
        assert opt.is_call is False

    def test_american_put_geq_european_put(self):
        """American put should be >= European put (early exercise premium)."""
        eu_put = EuropeanOption(strike=STRIKE, expiry=EXPIRY, is_call=False)
        am_put = AmericanOption(strike=STRIKE, expiry=EXPIRY, is_call=False)

        eu_price = black_scholes_price(eu_put, SPOT, VOL, RATE, DIV)
        cfg = BinomialConfig(n_steps=200, american=True)
        am_price = binomial_price(am_put, SPOT, VOL, RATE, DIV, cfg)

        assert am_price >= eu_price - 1e-4  # small tolerance for discretization

    def test_american_call_no_div_equals_european(self):
        """American call with no dividends = European call."""
        eu_call = EuropeanOption(strike=STRIKE, expiry=EXPIRY, is_call=True)
        am_call = AmericanOption(strike=STRIKE, expiry=EXPIRY, is_call=True)

        eu_price = black_scholes_price(eu_call, SPOT, VOL, RATE, DIV)
        cfg = BinomialConfig(n_steps=200, american=True)
        am_price = binomial_price(am_call, SPOT, VOL, RATE, DIV, cfg)

        assert jnp.isclose(am_price, eu_price, rtol=0.01)

    def test_binomial_differentiable(self):
        """jax.grad through binomial pricing of American option should work."""
        am_put = AmericanOption(strike=STRIKE, expiry=EXPIRY, is_call=False)
        cfg = BinomialConfig(n_steps=50, american=True)
        delta = jax.grad(
            lambda s: binomial_price(am_put, s, VOL, RATE, DIV, cfg),
        )(SPOT)
        assert jnp.isfinite(delta)
        assert delta < 0  # Put delta is negative


# ── Equity Barrier Option ────────────────────────────────────────────


class TestEquityBarrierOption:
    def test_pytree_construction(self):
        opt = EquityBarrierOption(
            strike=STRIKE, expiry=EXPIRY, barrier=jnp.array(120.0),
            is_call=True, is_up=True, is_knock_in=True,
        )
        assert jnp.isclose(opt.barrier, 120.0)

    def test_knock_in_plus_knock_out_equals_vanilla(self, paths):
        """Knock-in + knock-out = vanilla (barrier parity)."""
        ki = EquityBarrierOption(
            strike=STRIKE, expiry=EXPIRY, barrier=jnp.array(120.0),
            is_call=True, is_up=True, is_knock_in=True,
        )
        ko = EquityBarrierOption(
            strike=STRIKE, expiry=EXPIRY, barrier=jnp.array(120.0),
            is_call=True, is_up=True, is_knock_in=False,
        )
        vanilla = EuropeanOption(strike=STRIKE, expiry=EXPIRY, is_call=True)

        ki_pv = jnp.mean(equity_barrier_payoff(paths, ki))
        ko_pv = jnp.mean(equity_barrier_payoff(paths, ko))
        van_pv = jnp.mean(european_payoff(paths, vanilla))

        assert jnp.isclose(ki_pv + ko_pv, van_pv, atol=1e-6)

    def test_down_and_in_put(self, paths):
        """Down-and-in put should have non-negative payoff."""
        opt = EquityBarrierOption(
            strike=STRIKE, expiry=EXPIRY, barrier=jnp.array(80.0),
            is_call=False, is_up=False, is_knock_in=True,
        )
        payoff = equity_barrier_payoff(paths, opt)
        assert jnp.all(payoff >= -1e-10)

    def test_smoothed_barrier_differentiable(self, paths):
        """Smoothed barrier payoff should be differentiable w.r.t. barrier."""
        opt = EquityBarrierOption(
            strike=STRIKE, expiry=EXPIRY, barrier=jnp.array(120.0),
            is_call=True, is_up=True, is_knock_in=True, smoothing=2.0,
        )

        def mean_payoff(b):
            opt_b = EquityBarrierOption(
                strike=STRIKE, expiry=EXPIRY, barrier=b,
                is_call=True, is_up=True, is_knock_in=True, smoothing=2.0,
            )
            return jnp.mean(equity_barrier_payoff(paths, opt_b))

        grad = jax.grad(mean_payoff)(jnp.array(120.0))
        assert jnp.isfinite(grad)


# ── Asian Option ─────────────────────────────────────────────────────


class TestAsianOption:
    def test_pytree_construction(self):
        opt = AsianOption(strike=STRIKE, expiry=EXPIRY, is_call=True)
        assert opt.averaging == "arithmetic"

    def test_arithmetic_asian_call_leq_european(self, paths):
        """Asian call should be <= European call (averaging reduces variance)."""
        asian = AsianOption(strike=STRIKE, expiry=EXPIRY, is_call=True)
        vanilla = EuropeanOption(strike=STRIKE, expiry=EXPIRY, is_call=True)

        asian_pv = jnp.mean(asian_option_payoff(paths, asian))
        van_pv = jnp.mean(european_payoff(paths, vanilla))

        assert asian_pv <= van_pv + 0.5  # small MC noise tolerance

    def test_geometric_vs_arithmetic(self, paths):
        """Geometric average <= arithmetic average (AM-GM inequality)."""
        arith = AsianOption(
            strike=STRIKE, expiry=EXPIRY, is_call=True, averaging="arithmetic",
        )
        geom = AsianOption(
            strike=STRIKE, expiry=EXPIRY, is_call=True, averaging="geometric",
        )
        arith_pv = jnp.mean(asian_option_payoff(paths, arith))
        geom_pv = jnp.mean(asian_option_payoff(paths, geom))

        assert geom_pv <= arith_pv + 0.5

    def test_payoff_shape(self, paths):
        """Payoff should have shape (n_paths,)."""
        opt = AsianOption(strike=STRIKE, expiry=EXPIRY, is_call=True)
        payoff = asian_option_payoff(paths, opt)
        assert payoff.shape == (paths.shape[0],)

    def test_payoff_non_negative(self, paths):
        """Call payoff should be non-negative."""
        opt = AsianOption(strike=STRIKE, expiry=EXPIRY, is_call=True)
        payoff = asian_option_payoff(paths, opt)
        assert jnp.all(payoff >= -1e-10)


# ── Lookback Option ──────────────────────────────────────────────────


class TestLookbackOption:
    def test_floating_strike_call_always_positive(self, paths):
        """Floating-strike lookback call payoff = S_T - min(S) >= 0."""
        opt = LookbackOption(expiry=EXPIRY, is_call=True, is_fixed_strike=False)
        payoff = lookback_payoff(paths, opt)
        assert jnp.all(payoff >= -1e-10)

    def test_floating_strike_put_always_positive(self, paths):
        """Floating-strike lookback put payoff = max(S) - S_T >= 0."""
        opt = LookbackOption(expiry=EXPIRY, is_call=False, is_fixed_strike=False)
        payoff = lookback_payoff(paths, opt)
        assert jnp.all(payoff >= -1e-10)

    def test_floating_lookback_geq_european(self, paths):
        """Floating-strike lookback call should be >= European call."""
        lookback = LookbackOption(expiry=EXPIRY, is_call=True, is_fixed_strike=False)
        vanilla = EuropeanOption(strike=STRIKE, expiry=EXPIRY, is_call=True)

        lb_pv = jnp.mean(lookback_payoff(paths, lookback))
        van_pv = jnp.mean(european_payoff(paths, vanilla))

        assert lb_pv >= van_pv - 0.5  # MC tolerance

    def test_fixed_strike_payoff_non_negative(self, paths):
        """Fixed-strike lookback payoff should be non-negative."""
        opt = LookbackOption(
            expiry=EXPIRY, is_call=True, is_fixed_strike=True, strike=STRIKE,
        )
        payoff = lookback_payoff(paths, opt)
        assert jnp.all(payoff >= -1e-10)

    def test_fixed_strike_geq_european(self, paths):
        """Fixed-strike lookback call >= European call (max(S) >= S_T)."""
        lb = LookbackOption(
            expiry=EXPIRY, is_call=True, is_fixed_strike=True, strike=STRIKE,
        )
        vanilla = EuropeanOption(strike=STRIKE, expiry=EXPIRY, is_call=True)

        lb_pv = jnp.mean(lookback_payoff(paths, lb))
        van_pv = jnp.mean(european_payoff(paths, vanilla))

        assert lb_pv >= van_pv - 0.5


# ── Variance Swap ────────────────────────────────────────────────────


class TestVarianceSwapAnalytic:
    def test_fair_strike_equals_vol_squared(self):
        """Under BSM, fair variance strike = σ²."""
        fair = variance_swap_fair_strike(VOL)
        assert jnp.isclose(fair, VOL**2, atol=1e-12)

    def test_zero_pnl_at_fair_strike(self):
        """Variance swap should have zero value when struck at fair strike."""
        fair = variance_swap_fair_strike(VOL)
        swap = VarianceSwap(
            expiry=EXPIRY,
            strike_var=fair,
            notional_var=jnp.array(1e4),
        )
        price = variance_swap_price(swap, VOL, RATE)
        assert jnp.isclose(price, 0.0, atol=1e-8)

    def test_positive_pnl_when_vol_rises(self):
        """Long variance swap profits when realized vol > strike vol."""
        swap = VarianceSwap(
            expiry=EXPIRY,
            strike_var=jnp.array(0.04),   # 20% vol strike
            notional_var=jnp.array(1e4),
        )
        higher_vol = jnp.array(0.25)
        price = variance_swap_price(swap, higher_vol, RATE)
        assert price > 0  # vol up → long variance profits

    def test_negative_pnl_when_vol_falls(self):
        """Long variance swap loses when realized vol < strike vol."""
        swap = VarianceSwap(
            expiry=EXPIRY,
            strike_var=jnp.array(0.04),
            notional_var=jnp.array(1e4),
        )
        lower_vol = jnp.array(0.15)
        price = variance_swap_price(swap, lower_vol, RATE)
        assert price < 0  # vol down → long variance loses

    def test_vega_positive_for_long(self):
        """Long variance swap has positive vega."""
        swap = VarianceSwap(
            expiry=EXPIRY,
            strike_var=jnp.array(0.04),
            notional_var=jnp.array(1e4),
        )
        vega = jax.grad(
            lambda v: variance_swap_price(swap, v, RATE),
        )(VOL)
        assert vega > 0

    def test_seasoned_swap_weights(self):
        """Seasoned swap should blend realized and implied correctly."""
        swap = VarianceSwap(
            expiry=EXPIRY,
            strike_var=jnp.array(0.04),
            notional_var=jnp.array(1e4),
        )
        # Half elapsed, realized 25% vol (= 0.0625 var), implied still 20%
        price = variance_swap_price_seasoned(
            swap,
            realized_var_so_far=jnp.array(0.0625),
            elapsed=jnp.array(0.5),
            vol=VOL,
            rate=RATE,
        )
        # Expected total: 0.5*0.0625 + 0.5*0.04 = 0.05125
        # vs strike 0.04 → positive P&L
        assert price > 0


class TestVarianceSwapMC:
    def test_mc_fair_value_near_zero(self, paths):
        """MC variance swap at fair strike should have mean payoff ≈ 0."""
        fair = variance_swap_fair_strike(VOL)
        swap = VarianceSwap(
            expiry=EXPIRY,
            strike_var=fair,
            notional_var=jnp.array(1.0),
        )
        payoff = variance_swap_payoff(paths, swap)
        # Mean should be near zero (with MC noise)
        assert jnp.abs(jnp.mean(payoff)) < 0.01

    def test_mc_payoff_shape(self, paths):
        """MC payoff should have shape (n_paths,)."""
        swap = VarianceSwap(
            expiry=EXPIRY,
            strike_var=jnp.array(0.04),
            notional_var=jnp.array(1.0),
        )
        payoff = variance_swap_payoff(paths, swap)
        assert payoff.shape == (paths.shape[0],)
