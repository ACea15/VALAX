"""Tests for Bachelier (normal model) pricing and autodiff Greeks."""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.bachelier import bachelier_price, bachelier_implied_vol
from valax.greeks.autodiff import greek


@pytest.fixture
def atm_call():
    """ATM call: F=100, K=100, T=1, normal vol=20, r=2%."""
    return (
        EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True),
        jnp.array(100.0),  # forward
        jnp.array(20.0),   # normal vol (absolute, not percentage)
        jnp.array(0.02),   # rate
    )


@pytest.fixture
def atm_put():
    return (
        EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False),
        jnp.array(100.0),
        jnp.array(20.0),
        jnp.array(0.02),
    )


class TestBachelierPrice:
    def test_call_price_positive(self, atm_call):
        option, fwd, vol, rate = atm_call
        price = bachelier_price(option, fwd, vol, rate)
        assert float(price) > 0.0

    def test_put_call_parity(self, atm_call, atm_put):
        """C - P = df * (F - K)."""
        option_c, fwd, vol, rate = atm_call
        option_p, *_ = atm_put
        call = bachelier_price(option_c, fwd, vol, rate)
        put = bachelier_price(option_p, fwd, vol, rate)
        df = jnp.exp(-rate * option_c.expiry)
        assert abs(float(call - put - df * (fwd - option_c.strike))) < 1e-10

    def test_atm_call_closed_form(self, atm_call):
        """ATM Bachelier call = df * sigma * sqrt(T) / sqrt(2*pi)."""
        option, fwd, vol, rate = atm_call
        price = bachelier_price(option, fwd, vol, rate)
        df = jnp.exp(-rate * option.expiry)
        expected = df * vol * jnp.sqrt(option.expiry) / jnp.sqrt(2 * jnp.pi)
        assert abs(float(price) - float(expected)) < 1e-10

    def test_deep_itm_converges_to_intrinsic(self):
        """Deep ITM call with low vol → discounted intrinsic."""
        option = EuropeanOption(strike=jnp.array(80.0), expiry=jnp.array(1.0), is_call=True)
        price = bachelier_price(option, jnp.array(100.0), jnp.array(0.01), jnp.array(0.0))
        assert abs(float(price) - 20.0) < 0.01

    def test_jit_compiles(self, atm_call):
        option, fwd, vol, rate = atm_call
        price = jax.jit(bachelier_price)(option, fwd, vol, rate)
        assert jnp.isfinite(price)


class TestBachelierGreeks:
    def test_delta_atm_near_half_df(self, atm_call):
        """ATM normal delta = 0.5 * df."""
        option, fwd, vol, rate = atm_call
        d = greek(bachelier_price, "delta", option, fwd, vol, rate)
        df = jnp.exp(-rate * option.expiry)
        assert abs(float(d) - 0.5 * float(df)) < 1e-10

    def test_vega_positive(self, atm_call):
        option, fwd, vol, rate = atm_call
        v = greek(bachelier_price, "vega", option, fwd, vol, rate)
        assert float(v) > 0.0

    def test_gamma_positive(self, atm_call):
        option, fwd, vol, rate = atm_call
        g = greek(bachelier_price, "gamma", option, fwd, vol, rate)
        assert float(g) > 0.0


class TestBachelierImpliedVol:
    """Invert bachelier_price for the normal (absolute) volatility."""

    @pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
    @pytest.mark.parametrize("is_call", [True, False])
    def test_round_trip(self, strike, is_call):
        """price -> vol -> price recovers the input vol to tight tolerance."""
        option = EuropeanOption(
            strike=jnp.array(strike), expiry=jnp.array(1.5), is_call=is_call
        )
        fwd = jnp.array(100.0)
        rate = jnp.array(0.02)
        true_vol = jnp.array(18.0)  # normal (absolute) vol
        price = bachelier_price(option, fwd, true_vol, rate)
        iv = bachelier_implied_vol(option, fwd, price, rate)
        assert abs(float(iv) - float(true_vol)) < 1e-8

    def test_round_trip_negative_forward(self):
        """Normal model admits negative forwards/strikes (rates use case)."""
        option = EuropeanOption(
            strike=jnp.array(-0.005), expiry=jnp.array(2.0), is_call=True
        )
        fwd = jnp.array(-0.002)
        rate = jnp.array(0.01)
        true_vol = jnp.array(0.008)
        price = bachelier_price(option, fwd, true_vol, rate)
        iv = bachelier_implied_vol(option, fwd, price, rate)
        assert abs(float(iv) - float(true_vol)) < 1e-8

    def test_jit_compiles(self):
        option = EuropeanOption(
            strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True
        )
        fwd, rate, true_vol = jnp.array(100.0), jnp.array(0.02), jnp.array(20.0)
        price = bachelier_price(option, fwd, true_vol, rate)
        iv = jax.jit(bachelier_implied_vol)(option, fwd, price, rate)
        assert jnp.isfinite(iv)

    def test_vega_via_implicit_diff(self):
        """d(price)/d(vol) inverted: d(vol)/d(price) = 1/vega > 0, finite."""
        option = EuropeanOption(
            strike=jnp.array(105.0), expiry=jnp.array(1.0), is_call=True
        )
        fwd, rate, true_vol = jnp.array(100.0), jnp.array(0.02), jnp.array(20.0)
        price = bachelier_price(option, fwd, true_vol, rate)
        dvol_dprice = jax.grad(
            lambda p: bachelier_implied_vol(option, fwd, p, rate)
        )(price)
        assert float(dvol_dprice) > 0.0
        assert jnp.isfinite(dvol_dprice)
