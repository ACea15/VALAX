"""Tests for Black-76 pricing and autodiff Greeks."""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black76 import black76_price
from valax.greeks.autodiff import greek


@pytest.fixture
def atm_call():
    """ATM call on a forward: F=100, K=100, T=0.5, vol=25%, r=3%."""
    return (
        EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(0.5), is_call=True),
        jnp.array(100.0),  # forward
        jnp.array(0.25),   # vol
        jnp.array(0.03),   # rate
    )


@pytest.fixture
def atm_put():
    return (
        EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(0.5), is_call=False),
        jnp.array(100.0),
        jnp.array(0.25),
        jnp.array(0.03),
    )


class TestBlack76Price:
    def test_call_price_positive(self, atm_call):
        option, fwd, vol, rate = atm_call
        price = black76_price(option, fwd, vol, rate)
        assert float(price) > 0.0

    def test_put_call_parity(self, atm_call, atm_put):
        """C - P = df * (F - K)."""
        option_c, fwd, vol, rate = atm_call
        option_p, *_ = atm_put
        call = black76_price(option_c, fwd, vol, rate)
        put = black76_price(option_p, fwd, vol, rate)
        df = jnp.exp(-rate * option_c.expiry)
        assert abs(float(call - put - df * (fwd - option_c.strike))) < 1e-12

    def test_atm_call_equals_bs_zero_dividend(self):
        """Black-76 with F = S*exp(rT) should match BS with q=0."""
        from valax.pricing.analytic.black_scholes import black_scholes_price
        S, K, T, vol, r = 100.0, 105.0, 1.0, 0.2, 0.05
        F = S * jnp.exp(jnp.array(r) * T)
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=True)
        bs_price = black_scholes_price(option, jnp.array(S), jnp.array(vol), jnp.array(r), jnp.array(0.0))
        b76_price = black76_price(option, jnp.array(F), jnp.array(vol), jnp.array(r))
        assert abs(float(bs_price) - float(b76_price)) < 1e-10

    def test_jit_compiles(self, atm_call):
        option, fwd, vol, rate = atm_call
        price = jax.jit(black76_price)(option, fwd, vol, rate)
        assert jnp.isfinite(price)


class TestBlack76Greeks:
    def test_delta_atm_near_half(self, atm_call):
        """ATM forward delta ~ 0.5 * df."""
        option, fwd, vol, rate = atm_call
        d = greek(black76_price, "delta", option, fwd, vol, rate)
        df = jnp.exp(-rate * option.expiry)
        assert abs(float(d) - 0.5 * float(df)) < 0.05

    def test_vega_positive(self, atm_call):
        option, fwd, vol, rate = atm_call
        v = greek(black76_price, "vega", option, fwd, vol, rate)
        assert float(v) > 0.0

    def test_gamma_positive(self, atm_call):
        option, fwd, vol, rate = atm_call
        g = greek(black76_price, "gamma", option, fwd, vol, rate)
        assert float(g) > 0.0
