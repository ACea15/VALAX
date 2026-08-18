"""Local-volatility (Dupire) PDE recipe — cross-checks vs Black-Scholes,
Monte-Carlo, and the Dupire surface.

The recipe (``EuropeanOption`` under ``LocalVolModel``) rebuilds the log-spot
operator at every backward time step from the midpoint-in-time Dupire local
variance (:func:`valax.pricing.pde.coefficients.lv_operator_stack`), exercising
the *time-dependent operator stack* path of
:func:`valax.pricing.pde.schemes.solve_backward_1d`.

Cross-check design (see the recipe module docstring for the theory)
-------------------------------------------------------------------
Feeding the *continuous* Dupire local vol into a *discrete* backward FD scheme
reprices the surface exactly only where the local vol is constant in log-spot:

* ``TestFlatSurfaceReducesToBS``    — flat surface ⇒ LV PDE == Black-Scholes.
* ``TestTermStructureVsMCAndDupire``— **headline gate**: a *no-skew* term
  structure of ATM vol (local vol varies in time but not in spot) is repriced
  by the LV PDE, matching both the Dupire surface and LV Monte-Carlo tightly
  across strikes. This is the direct validation of the per-step operator stack.
* ``TestSkewATMConsistency``        — with a skew, the LV PDE reprices ATM and
  agrees with MC near ATM (the skew-dependent FD-Dupire wing gap, shared with
  QuantLib, is covered in ``tests/test_quantlib_comparison`` — not asserted
  against MC here).
* ``TestConvergence``               — grid refinement does not degrade accuracy.
* ``TestGreeks``                    — autodiff delta/gamma + put-call parity +
  ``eqx.filter_jit`` smoke.
* ``TestDispatch``                  — the recipe is registered.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption
from valax.models.local_vol import LocalVolModel
from valax.pricing.analytic import black_scholes_price
from valax.pricing.mc.local_vol_paths import generate_local_vol_paths
from valax.pricing.pde import PDEConfig, pde_price_dispatch, registered_recipes
from valax.surfaces import SVIVolSurface


RATE, DIV = 0.03, 0.01
MU = RATE - DIV
SPOT = 100.0
T = 1.0

FINE = PDEConfig(n_spot=400, n_time=400, spot_range=5.0, rannacher_steps=2)
COARSE = PDEConfig(n_spot=100, n_time=100, spot_range=5.0, rannacher_steps=2)


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


def _svi(a_vec, b_vec, rho_vec, sigma_vec, expiries):
    return SVIVolSurface(
        expiries=expiries,
        forwards=jnp.array(SPOT) * jnp.exp(MU * expiries),
        a_vec=a_vec,
        b_vec=b_vec,
        rho_vec=rho_vec,
        m_vec=jnp.zeros_like(expiries),
        sigma_vec=sigma_vec,
    )


@pytest.fixture
def flat_model():
    """Flat (sigma = 0.25) surface ⇒ constant local vol ⇒ Black-Scholes."""
    sigma = 0.25
    exp = jnp.array([0.05, 0.5, 1.0, 2.0])
    surf = _svi(
        a_vec=sigma**2 * exp,
        b_vec=jnp.zeros_like(exp),
        rho_vec=jnp.zeros_like(exp),
        sigma_vec=jnp.full_like(exp, 0.1),
        expiries=exp,
    )
    return LocalVolModel.from_flat_rate(surf, rate=RATE, dividend=DIV), sigma


@pytest.fixture
def term_structure_model():
    """No-skew term structure of ATM vol: local vol varies in *time* only.

    Here the continuous Dupire local vol is exactly reproduced by the FD
    scheme (constant in log-spot at each slice), so the LV PDE reprices the
    surface and matches LV Monte-Carlo. This is the clean cross-check of the
    time-dependent operator stack.
    """
    exp = jnp.array([0.1, 0.25, 0.5, 1.0, 2.0])
    atm = jnp.array([0.18, 0.19, 0.20, 0.21, 0.23])
    surf = _svi(
        a_vec=atm**2 * exp,
        b_vec=jnp.full_like(exp, 1e-6),   # ~zero skew/curvature
        rho_vec=jnp.zeros_like(exp),
        sigma_vec=jnp.full_like(exp, 0.1),
        expiries=exp,
    )
    return LocalVolModel.from_flat_rate(surf, rate=RATE, dividend=DIV), surf


@pytest.fixture
def skew_model():
    """Skewed SVI surface (real equity smile)."""
    exp = jnp.array([0.1, 0.25, 0.5, 1.0, 2.0])
    atm = jnp.array([0.18, 0.19, 0.20, 0.21, 0.23])
    surf = _svi(
        a_vec=atm**2 * exp,
        b_vec=jnp.full_like(exp, 0.04),
        rho_vec=jnp.full_like(exp, -0.3),
        sigma_vec=jnp.full_like(exp, 0.15),
        expiries=exp,
    )
    return LocalVolModel.from_flat_rate(surf, rate=RATE, dividend=DIV), surf


def _pde(model, K, is_call, config=FINE, spot=SPOT):
    opt = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
    return float(pde_price_dispatch(opt, model, config, spot=jnp.array(spot)).price)


def _surface_bs(surf, K, is_call):
    iv = surf(jnp.array(K), jnp.array(T))
    opt = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
    return float(
        black_scholes_price(
            opt, jnp.array(SPOT), iv, jnp.array(RATE), jnp.array(DIV)
        )
    )


def _mc_call(model, K, n_paths=120_000, n_steps=400, seed=20260101):
    key = jax.random.PRNGKey(seed)
    paths = generate_local_vol_paths(
        model, jnp.array(SPOT), T, n_steps, n_paths, key
    )
    terminal = paths[:, -1]
    payoff = jnp.maximum(terminal - K, 0.0)
    df = jnp.exp(-RATE * T)
    mc = float(df * jnp.mean(payoff))
    se = float(df * jnp.std(payoff) / jnp.sqrt(n_paths))
    return mc, se


# ─────────────────────────────────────────────────────────────────────
# 1. Flat surface ⇒ Black-Scholes (exact)
# ─────────────────────────────────────────────────────────────────────


class TestFlatSurfaceReducesToBS:
    @pytest.mark.parametrize(
        "K,is_call",
        [(80.0, True), (100.0, True), (100.0, False), (120.0, True), (90.0, False)],
    )
    def test_matches_black_scholes(self, flat_model, K, is_call):
        model, sigma = flat_model
        pde = _pde(model, K, is_call)
        opt = EuropeanOption(
            strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call
        )
        bs = float(
            black_scholes_price(
                opt, jnp.array(SPOT), jnp.array(sigma), jnp.array(RATE), jnp.array(DIV)
            )
        )
        rel = abs(pde - bs) / max(bs, 1e-6)
        assert rel < 5e-3, f"K={K} call={is_call}: PDE={pde:.6f} BS={bs:.6f} rel={rel:.2e}"


# ─────────────────────────────────────────────────────────────────────
# 2. No-skew term structure ⇒ reprices surface AND matches MC (headline)
# ─────────────────────────────────────────────────────────────────────


class TestTermStructureVsMCAndDupire:
    """The per-step operator stack must reproduce a time-varying (but
    spot-flat) local vol: LV PDE == Dupire surface == LV Monte-Carlo."""

    @pytest.mark.parametrize("K", [85.0, 95.0, 100.0, 105.0, 115.0])
    def test_pde_reprices_dupire_surface(self, term_structure_model, K):
        model, surf = term_structure_model
        pde = _pde(model, K, True)
        truth = _surface_bs(surf, K, True)
        rel = abs(pde - truth) / max(truth, 1e-6)
        assert rel < 5e-3, f"K={K}: PDE={pde:.5f} surfaceBS={truth:.5f} rel={rel:.2e}"

    @pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
    def test_pde_matches_mc(self, term_structure_model, K):
        model, _ = term_structure_model
        pde = _pde(model, K, True)
        mc, se = _mc_call(model, K)
        nse = abs(pde - mc) / max(se, 1e-12)
        assert nse < 4.0, f"K={K}: PDE={pde:.4f} MC={mc:.4f}±{se:.4f} nse={nse:.2f}"


# ─────────────────────────────────────────────────────────────────────
# 3. Skew: ATM reprice + near-ATM MC agreement
# ─────────────────────────────────────────────────────────────────────


class TestSkewATMConsistency:
    """With a skew, continuous-Dupire FD reprices ATM (where the local vol is
    locally flat in log-spot); the skew-dependent wing gap vs MC/surface is an
    inherent FD-Dupire property (shared with QuantLib) and is cross-checked
    against QuantLib elsewhere, not against MC here."""

    def test_atm_reprices_surface(self, skew_model):
        model, surf = skew_model
        pde = _pde(model, 100.0, True)
        truth = _surface_bs(surf, 100.0, True)
        rel = abs(pde - truth) / truth
        assert rel < 6e-3, f"ATM PDE={pde:.5f} surfaceBS={truth:.5f} rel={rel:.2e}"

    def test_atm_matches_mc(self, skew_model):
        model, _ = skew_model
        pde = _pde(model, 100.0, True)
        mc, se = _mc_call(model, 100.0)
        nse = abs(pde - mc) / max(se, 1e-12)
        assert nse < 5.0, f"ATM PDE={pde:.4f} MC={mc:.4f}±{se:.4f} nse={nse:.2f}"

    def test_prices_positive_and_monotone_in_strike(self, skew_model):
        model, _ = skew_model
        calls = [_pde(model, K, True) for K in (90.0, 100.0, 110.0)]
        assert all(c > 0 for c in calls)
        # Call price is decreasing in strike.
        assert calls[0] > calls[1] > calls[2]


# ─────────────────────────────────────────────────────────────────────
# 4. Convergence
# ─────────────────────────────────────────────────────────────────────


class TestConvergence:
    def test_finer_grid_not_worse(self, term_structure_model):
        model, surf = term_structure_model
        truth = _surface_bs(surf, 100.0, True)
        coarse = abs(_pde(model, 100.0, True, config=COARSE) - truth)
        fine = abs(_pde(model, 100.0, True, config=FINE) - truth)
        assert fine <= coarse + 1e-3


# ─────────────────────────────────────────────────────────────────────
# 5. Greeks via autodiff + jit smoke
# ─────────────────────────────────────────────────────────────────────


class TestGreeks:
    def _price_of_spot(self, model, K, is_call):
        def f(s):
            opt = EuropeanOption(
                strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call
            )
            return pde_price_dispatch(opt, model, FINE, spot=s).price

        return f

    def test_call_delta_in_range(self, skew_model):
        model, _ = skew_model
        delta = float(jax.grad(self._price_of_spot(model, 100.0, True))(jnp.array(SPOT)))
        assert 0.3 < delta < 0.9, f"delta={delta:.4f}"

    def test_put_delta_negative(self, skew_model):
        model, _ = skew_model
        delta = float(jax.grad(self._price_of_spot(model, 100.0, False))(jnp.array(SPOT)))
        assert -0.9 < delta < -0.1, f"delta={delta:.4f}"

    def test_gamma_positive_and_finite(self, skew_model):
        model, _ = skew_model
        f = self._price_of_spot(model, 100.0, True)
        gamma = float(jax.grad(jax.grad(f))(jnp.array(SPOT)))
        assert jnp.isfinite(gamma) and gamma > 1e-3, f"gamma={gamma}"

    def test_call_put_gamma_equal(self, skew_model):
        model, _ = skew_model
        gc = float(jax.grad(jax.grad(self._price_of_spot(model, 100.0, True)))(jnp.array(SPOT)))
        gp = float(jax.grad(jax.grad(self._price_of_spot(model, 100.0, False)))(jnp.array(SPOT)))
        assert abs(gc - gp) < 1e-3, f"call gamma={gc:.5f} put gamma={gp:.5f}"

    def test_filter_jit_smoke(self, skew_model):
        model, _ = skew_model

        @eqx.filter_jit
        def price(m, s):
            opt = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(T), is_call=True)
            return pde_price_dispatch(opt, m, COARSE, spot=s).price

        val = float(price(model, jnp.array(SPOT)))
        assert jnp.isfinite(val) and val > 0.0


# ─────────────────────────────────────────────────────────────────────
# 6. Dispatch registration
# ─────────────────────────────────────────────────────────────────────


class TestDispatch:
    def test_recipe_registered(self):
        assert ("EuropeanOption", "LocalVolModel") in registered_recipes()
