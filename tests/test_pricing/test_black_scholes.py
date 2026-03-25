"""Tests for Black-Scholes pricing and autodiff Greeks."""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price, black_scholes_implied_vol
from valax.greeks.autodiff import greeks, greek
from valax.portfolio.batch import batch_price, batch_greeks


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def atm_call():
    """ATM European call: S=100, K=100, T=1, vol=20%, r=5%, q=2%."""
    return (
        EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True),
        jnp.array(100.0),   # spot
        jnp.array(0.20),    # vol
        jnp.array(0.05),    # rate
        jnp.array(0.02),    # dividend
    )


@pytest.fixture
def atm_put():
    return (
        EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False),
        jnp.array(100.0),
        jnp.array(0.20),
        jnp.array(0.05),
        jnp.array(0.02),
    )


# ── Analytical reference values ──────────────────────────────────────
# Computed independently via standard BS formula.
# S=100, K=100, T=1, sigma=0.2, r=0.05, q=0.02
# d1 = (ln(100/100) + (0.05-0.02+0.02)*1) / (0.2*1) = 0.25
# d2 = 0.25 - 0.2 = 0.05

def _reference_bs_call():
    """Reference BS call price for ATM params."""
    from jax.scipy.stats import norm
    S, K, T, sigma, r, q = 100.0, 100.0, 1.0, 0.2, 0.05, 0.02
    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    call = S * jnp.exp(-q * T) * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)
    return float(call)


# ── Price tests ──────────────────────────────────────────────────────

class TestBlackScholesPrice:
    def test_call_price_positive(self, atm_call):
        option, spot, vol, rate, div = atm_call
        price = black_scholes_price(option, spot, vol, rate, div)
        assert float(price) > 0.0

    def test_call_price_matches_reference(self, atm_call):
        option, spot, vol, rate, div = atm_call
        price = float(black_scholes_price(option, spot, vol, rate, div))
        ref = _reference_bs_call()
        assert abs(price - ref) < 1e-10

    def test_put_call_parity(self, atm_call, atm_put):
        option_c, spot, vol, rate, div = atm_call
        option_p, *_ = atm_put
        call = black_scholes_price(option_c, spot, vol, rate, div)
        put = black_scholes_price(option_p, spot, vol, rate, div)
        # C - P = S*exp(-qT) - K*exp(-rT)
        fwd_diff = spot * jnp.exp(-div * option_c.expiry) - option_c.strike * jnp.exp(-rate * option_c.expiry)
        assert abs(float(call - put - fwd_diff)) < 1e-10

    def test_zero_vol_call_intrinsic(self):
        """With zero vol, call = max(S*exp(-qT) - K*exp(-rT), 0)."""
        option = EuropeanOption(strike=jnp.array(90.0), expiry=jnp.array(1.0), is_call=True)
        price = black_scholes_price(
            option, jnp.array(100.0), jnp.array(1e-8), jnp.array(0.05), jnp.array(0.0)
        )
        # Discounted forward minus discounted strike
        expected = 100.0 * jnp.exp(-0.0) - 90.0 * jnp.exp(-0.05)
        assert abs(float(price) - float(expected)) < 1e-3

    def test_jit_compiles(self, atm_call):
        option, spot, vol, rate, div = atm_call
        jitted = jax.jit(black_scholes_price, static_argnames=())
        price = jitted(option, spot, vol, rate, div)
        assert jnp.isfinite(price)


# ── Greek tests ──────────────────────────────────────────────────────

class TestGreeks:
    def test_delta_call_between_0_and_1(self, atm_call):
        option, spot, vol, rate, div = atm_call
        d = greek(black_scholes_price, "delta", option, spot, vol, rate, div)
        assert 0.0 < float(d) < 1.0

    def test_delta_put_between_neg1_and_0(self, atm_put):
        option, spot, vol, rate, div = atm_put
        d = greek(black_scholes_price, "delta", option, spot, vol, rate, div)
        assert -1.0 < float(d) < 0.0

    def test_gamma_positive(self, atm_call):
        option, spot, vol, rate, div = atm_call
        g = greek(black_scholes_price, "gamma", option, spot, vol, rate, div)
        assert float(g) > 0.0

    def test_vega_positive(self, atm_call):
        option, spot, vol, rate, div = atm_call
        v = greek(black_scholes_price, "vega", option, spot, vol, rate, div)
        assert float(v) > 0.0

    def test_greeks_dict_complete(self, atm_call):
        option, spot, vol, rate, div = atm_call
        result = greeks(black_scholes_price, option, spot, vol, rate, div)
        expected_keys = {"price", "delta", "gamma", "vega", "volga", "vanna", "rho", "dividend_rho", "theta"}
        assert set(result.keys()) == expected_keys
        for k, v in result.items():
            assert jnp.isfinite(v), f"{k} is not finite"

    def test_delta_matches_analytical(self, atm_call):
        """BS delta for a call = exp(-qT) * N(d1)."""
        from jax.scipy.stats import norm
        option, spot, vol, rate, div = atm_call
        d1 = (jnp.log(spot / option.strike) + (rate - div + 0.5 * vol**2) * option.expiry) / (vol * jnp.sqrt(option.expiry))
        analytical_delta = float(jnp.exp(-div * option.expiry) * norm.cdf(d1))

        autodiff_delta = float(greek(black_scholes_price, "delta", option, spot, vol, rate, div))
        assert abs(autodiff_delta - analytical_delta) < 1e-10

    def test_gamma_matches_analytical(self, atm_call):
        """BS gamma = exp(-qT) * n(d1) / (S * sigma * sqrt(T))."""
        from jax.scipy.stats import norm
        option, spot, vol, rate, div = atm_call
        d1 = (jnp.log(spot / option.strike) + (rate - div + 0.5 * vol**2) * option.expiry) / (vol * jnp.sqrt(option.expiry))
        analytical_gamma = float(jnp.exp(-div * option.expiry) * norm.pdf(d1) / (spot * vol * jnp.sqrt(option.expiry)))

        autodiff_gamma = float(greek(black_scholes_price, "gamma", option, spot, vol, rate, div))
        assert abs(autodiff_gamma - analytical_gamma) < 1e-10


# ── Implied vol tests ────────────────────────────────────────────────

class TestImpliedVol:
    def test_roundtrip(self, atm_call):
        option, spot, vol, rate, div = atm_call
        price = black_scholes_price(option, spot, vol, rate, div)
        iv = black_scholes_implied_vol(option, spot, rate, div, price)
        assert abs(float(iv) - float(vol)) < 1e-8


# ── Batch pricing tests ─────────────────────────────────────────────

class TestBatchPricing:
    def test_batch_price_matches_scalar(self):
        n = 100
        strikes = jnp.linspace(80.0, 120.0, n)
        expiries = jnp.ones(n)
        options = EuropeanOption(strike=strikes, expiry=expiries, is_call=True)
        spots = jnp.full(n, 100.0)
        vols = jnp.full(n, 0.20)
        rates = jnp.full(n, 0.05)
        divs = jnp.full(n, 0.02)

        prices = batch_price(black_scholes_price, options, spots, vols, rates, divs)
        assert prices.shape == (n,)

        # Check first and last against scalar
        opt_0 = EuropeanOption(strike=strikes[0], expiry=expiries[0], is_call=True)
        p0 = black_scholes_price(opt_0, spots[0], vols[0], rates[0], divs[0])
        assert abs(float(prices[0]) - float(p0)) < 1e-10

    def test_batch_greeks_shape(self):
        n = 50
        options = EuropeanOption(
            strike=jnp.linspace(90.0, 110.0, n),
            expiry=jnp.ones(n),
            is_call=True,
        )
        spots = jnp.full(n, 100.0)
        vols = jnp.full(n, 0.20)
        rates = jnp.full(n, 0.05)
        divs = jnp.full(n, 0.02)

        result = batch_greeks(black_scholes_price, options, spots, vols, rates, divs)
        for k, v in result.items():
            assert v.shape == (n,), f"{k} has wrong shape {v.shape}"
