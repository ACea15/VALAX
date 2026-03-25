"""Tests for CRR binomial tree pricing."""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.lattice.binomial import binomial_price, BinomialConfig


FINE = BinomialConfig(n_steps=500, american=False)
COARSE = BinomialConfig(n_steps=100, american=False)
AMERICAN = BinomialConfig(n_steps=500, american=True)


# ── European price accuracy ──────────────────────────────────────────

class TestBinomialEuropean:
    @pytest.mark.parametrize("S,K,T,sigma,r,q,is_call", [
        (100.0, 100.0, 1.0, 0.20, 0.05, 0.02, True),    # ATM call
        (100.0, 100.0, 1.0, 0.20, 0.05, 0.02, False),   # ATM put
        (100.0, 110.0, 0.5, 0.30, 0.03, 0.01, True),    # OTM call
        (100.0, 90.0, 0.5, 0.30, 0.03, 0.01, False),    # OTM put
        (100.0, 80.0, 2.0, 0.15, 0.08, 0.0, True),      # deep ITM call
    ])
    def test_matches_analytical(self, S, K, T, sigma, r, q, is_call):
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        tree = binomial_price(option, jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q), FINE)
        bs = black_scholes_price(option, jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))

        rel_err = abs(float(tree) - float(bs)) / max(float(bs), 1e-6)
        assert rel_err < 0.005, f"Tree={float(tree):.6f}, BS={float(bs):.6f}, rel_err={rel_err:.6f}"

    def test_put_call_parity(self):
        S, K, T, sigma, r, q = 100.0, 105.0, 1.0, 0.25, 0.05, 0.02
        call_opt = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=True)
        put_opt = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=False)

        call = binomial_price(call_opt, jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q), FINE)
        put = binomial_price(put_opt, jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q), FINE)

        parity = float(call) - float(put)
        expected = S * float(jnp.exp(-q * T)) - K * float(jnp.exp(-r * T))
        assert abs(parity - expected) < 0.1

    def test_convergence(self):
        """More steps → closer to BS."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        args = (jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.02))
        bs = float(black_scholes_price(option, *args))

        err_coarse = abs(float(binomial_price(option, *args, COARSE)) - bs)
        err_fine = abs(float(binomial_price(option, *args, FINE)) - bs)
        assert err_fine < err_coarse


# ── American options ─────────────────────────────────────────────────

class TestBinomialAmerican:
    def test_american_put_geq_european_put(self):
        """American put >= European put (early exercise premium)."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)
        args = (jnp.array(100.0), jnp.array(0.20), jnp.array(0.05), jnp.array(0.0))

        euro = binomial_price(option, *args, BinomialConfig(n_steps=500, american=False))
        amer = binomial_price(option, *args, BinomialConfig(n_steps=500, american=True))
        assert float(amer) >= float(euro) - 1e-10

    def test_american_call_no_dividend_equals_european(self):
        """American call with no dividends = European call (no early exercise)."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        args = (jnp.array(100.0), jnp.array(0.20), jnp.array(0.05), jnp.array(0.0))

        euro = binomial_price(option, *args, BinomialConfig(n_steps=500, american=False))
        amer = binomial_price(option, *args, BinomialConfig(n_steps=500, american=True))
        assert abs(float(amer) - float(euro)) < 0.01

    def test_american_call_with_dividend_geq_european(self):
        """American call with dividends may have early exercise premium."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        args = (jnp.array(100.0), jnp.array(0.20), jnp.array(0.05), jnp.array(0.08))  # high div

        euro = binomial_price(option, *args, BinomialConfig(n_steps=500, american=False))
        amer = binomial_price(option, *args, BinomialConfig(n_steps=500, american=True))
        assert float(amer) >= float(euro) - 1e-10

    def test_deep_itm_american_put_near_intrinsic(self):
        """Deep ITM American put with high rate → price near intrinsic."""
        option = EuropeanOption(strike=jnp.array(150.0), expiry=jnp.array(1.0), is_call=False)
        price = binomial_price(
            option, jnp.array(50.0), jnp.array(0.20), jnp.array(0.10), jnp.array(0.0), AMERICAN
        )
        intrinsic = 150.0 - 50.0
        # Should be very close to intrinsic (immediate exercise optimal)
        assert float(price) >= intrinsic - 0.1
        assert float(price) < intrinsic + 1.0


# ── Greeks via autodiff ──────────────────────────────────────────────

class TestBinomialGreeks:
    def test_delta_call_positive(self):
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        fn = lambda s: binomial_price(option, s, jnp.array(0.2), jnp.array(0.05), jnp.array(0.0), COARSE)
        delta = jax.grad(fn)(jnp.array(100.0))
        assert 0.3 < float(delta) < 0.9

    def test_vega_positive(self):
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        fn = lambda v: binomial_price(option, jnp.array(100.0), v, jnp.array(0.05), jnp.array(0.0), COARSE)
        vega = jax.grad(fn)(jnp.array(0.2))
        assert float(vega) > 0.0

    def test_american_put_delta_negative(self):
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)
        cfg = BinomialConfig(n_steps=100, american=True)
        fn = lambda s: binomial_price(option, s, jnp.array(0.2), jnp.array(0.05), jnp.array(0.0), cfg)
        delta = jax.grad(fn)(jnp.array(100.0))
        assert -0.9 < float(delta) < -0.1
