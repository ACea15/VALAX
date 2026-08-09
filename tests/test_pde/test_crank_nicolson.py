"""Tests for Crank-Nicolson PDE solver against BS analytical prices."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.greeks.autodiff import greek
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


# ── Second-order spot Greek (gamma) via autodiff ─────────────────────
#
# Regression guard for the read-off curvature fix. With the previous
# piecewise-linear ``jnp.interp`` read-off, gamma collapsed to ~0 because a
# spot-centred grid *co-moves* with spot, freezing the read-off at a fixed
# fractional cell position. The fix is a curvature-carrying (Catmull-Rom)
# read-off plus ``stop_gradient`` grid detachment, so the query ``ln(spot)``
# slides across a static value field and ``jax.grad(jax.grad(...))`` recovers a
# correct gamma through the *unified* autodiff Greek engine.

class TestPDESecondOrderGreeks:
    def _pde(self, o, *a):
        return pde_price(o, *a, FINE)

    def test_gamma_is_not_zero(self):
        """The whole point: gamma must not collapse to ~0 (the old bug)."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        args = (jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.0))
        gamma = float(greek(self._pde, "gamma", option, *args))
        assert gamma > 1e-3

    @pytest.mark.parametrize("S,K,T,sigma,r,q,is_call", [
        (100.0, 100.0, 1.0, 0.20, 0.05, 0.02, True),    # ATM call
        (100.0, 100.0, 1.0, 0.20, 0.05, 0.02, False),   # ATM put
        (100.0, 110.0, 0.5, 0.30, 0.03, 0.01, True),    # OTM call
        (100.0, 90.0, 0.5, 0.30, 0.03, 0.01, False),    # OTM put
        (100.0, 80.0, 2.0, 0.15, 0.08, 0.0, True),      # deep ITM call (tiny gamma)
        (50.0, 55.0, 0.25, 0.40, 0.01, 0.03, True),     # OTM short-dated
    ])
    def test_gamma_matches_analytical(self, S, K, T, sigma, r, q, is_call):
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        args = (jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))
        pde_gamma = float(greek(self._pde, "gamma", option, *args))
        bs_gamma = float(greek(black_scholes_price, "gamma", option, *args))
        # Combined atol/rtol: deep-ITM gamma is tiny so rel error inflates,
        # but the absolute error stays well controlled.
        tol = 3.0e-4 + 0.02 * abs(bs_gamma)
        assert abs(pde_gamma - bs_gamma) < tol, (
            f"PDE gamma={pde_gamma:.6f}, BS gamma={bs_gamma:.6f}"
        )

    def test_call_put_gamma_equal(self):
        """Gamma is identical for calls and puts (put-call parity)."""
        args = (jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.02))
        call = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        put = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)
        gc = float(greek(self._pde, "gamma", call, *args))
        gp = float(greek(self._pde, "gamma", put, *args))
        assert abs(gc - gp) < 1e-4

    def test_gamma_converges_with_grid(self):
        """Finer grids should not degrade gamma accuracy."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        args = (jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.0))
        bs_gamma = float(greek(black_scholes_price, "gamma", option, *args))
        coarse = abs(float(greek(lambda o, *a: pde_price(o, *a, COARSE), "gamma", option, *args)) - bs_gamma)
        fine = abs(float(greek(self._pde, "gamma", option, *args)) - bs_gamma)
        assert fine <= coarse + 1e-4

    def test_gamma_under_filter_jit(self):
        """Smoke test: gamma composes with eqx.filter_jit."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        args = (jnp.array(100.0), jnp.array(0.2), jnp.array(0.05), jnp.array(0.0))

        @eqx.filter_jit
        def gamma(o, s, v, r, q):
            return greek(self._pde, "gamma", o, s, v, r, q)

        g = float(gamma(option, *args))
        bs_gamma = float(greek(black_scholes_price, "gamma", option, *args))
        assert abs(g - bs_gamma) < 3.0e-4 + 0.02 * abs(bs_gamma)
