"""Tests for Crank-Nicolson PDE solver against BS analytical prices."""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.pde.solvers import pde_price, PDEConfig


# Use a fine grid for tight tolerances
FINE = PDEConfig(n_spot=400, n_time=400, spot_range=4.0)
COARSE = PDEConfig(n_spot=100, n_time=100, spot_range=4.0)


# ── Price accuracy tests ─────────────────────────────────────────────

class TestPDEPrice:
    @pytest.mark.parametrize("S,K,T,sigma,r,q,is_call", [
        (100.0, 100.0, 1.0, 0.20, 0.05, 0.02, True),    # ATM call
        (100.0, 100.0, 1.0, 0.20, 0.05, 0.02, False),   # ATM put
        (100.0, 110.0, 0.5, 0.30, 0.03, 0.01, True),    # OTM call
        (100.0, 90.0, 0.5, 0.30, 0.03, 0.01, False),    # OTM put
        (100.0, 80.0, 2.0, 0.15, 0.08, 0.0, True),      # deep ITM call
        (50.0, 55.0, 0.25, 0.40, 0.01, 0.03, True),     # OTM short-dated
    ])
    def test_matches_analytical(self, S, K, T, sigma, r, q, is_call):
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        pde = pde_price(option, jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q), FINE)
        bs = black_scholes_price(option, jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))

        err = abs(float(pde) - float(bs))
        rel_err = err / max(float(bs), 1e-6)
        assert rel_err < 0.005, f"PDE={float(pde):.6f}, BS={float(bs):.6f}, rel_err={rel_err:.6f}"

    def test_put_call_parity(self):
        S, K, T, sigma, r, q = 100.0, 105.0, 1.0, 0.25, 0.05, 0.02
        call_opt = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=True)
        put_opt = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=False)

        call = pde_price(call_opt, jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q), FINE)
        put = pde_price(put_opt, jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q), FINE)

        parity = float(call) - float(put)
        expected = S * float(jnp.exp(-q * T)) - K * float(jnp.exp(-r * T))
        assert abs(parity - expected) < 0.1

    def test_price_positive(self):
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        price = pde_price(option, jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.0), COARSE)
        assert float(price) > 0.0


# ── Convergence test ─────────────────────────────────────────────────

class TestPDEConvergence:
    def test_finer_grid_more_accurate(self):
        """Finer grid should give closer match to analytical."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        args = (jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.02))
        bs = float(black_scholes_price(option, *args))

        coarse = abs(float(pde_price(option, *args, COARSE)) - bs)
        fine = abs(float(pde_price(option, *args, FINE)) - bs)
        assert fine < coarse


# ── Greeks via autodiff through PDE ──────────────────────────────────

class TestPDEGreeks:
    def test_delta_positive_for_call(self):
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        fn = lambda s: pde_price(option, s, jnp.array(0.2), jnp.array(0.05), jnp.array(0.0), COARSE)
        delta = jax.grad(fn)(jnp.array(100.0))
        assert 0.3 < float(delta) < 0.9

    def test_delta_negative_for_put(self):
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)
        fn = lambda s: pde_price(option, s, jnp.array(0.2), jnp.array(0.05), jnp.array(0.0), COARSE)
        delta = jax.grad(fn)(jnp.array(100.0))
        assert -0.9 < float(delta) < -0.1

    def test_vega_positive(self):
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        fn = lambda v: pde_price(option, jnp.array(100.0), v, jnp.array(0.05), jnp.array(0.0), COARSE)
        vega = jax.grad(fn)(jnp.array(0.2))
        assert float(vega) > 0.0

    def test_delta_matches_analytical(self):
        """PDE delta should be close to BS delta."""
        from valax.greeks.autodiff import greek
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        args = (jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.02))

        bs_delta = float(greek(black_scholes_price, "delta", option, *args))
        pde_delta = float(jax.grad(
            lambda s: pde_price(option, s, jnp.array(0.2), jnp.array(0.05), jnp.array(0.02), FINE)
        )(jnp.array(100.0)))

        assert abs(pde_delta - bs_delta) < 0.01
