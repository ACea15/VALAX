"""Unit tests for Dupire local-vol extraction.

Test layout (P2.2 Tier 2.3 acceptance gates):

* ``TestFlatSurfaceLimit``  — Dupire on a flat SVI surface returns the
  constant IV at every (k, T), exact to machine precision.
* ``TestAutodiff``          — ``jax.grad`` w.r.t. SVI parameters matches
  central finite differences.
* ``TestArbitrageDetection``— surfaces with a butterfly violation
  produce a non-positive denominator → NaN (correct diagnostic).
* ``TestGolden``            — canonical Dupire grid pinned for drift
  detection.
* ``TestJitAndVmap``        — jit + double vmap stable.
* ``TestX64Guard``          — RuntimeError when x64 is disabled.
* ``TestSurfaceProtocol``   — SVIVolSurface, SABRVolSurface, and
  GridVolSurface all satisfy the duck-typed ``total_variance`` contract.

The ``TestBSMLimit`` reprice check (flat surface → LV MC → BSM) lives
in ``test_local_vol_paths.py`` to keep MC machinery out of this file.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from valax.pricing.analytic.dupire import (
    dupire_local_vol,
    dupire_local_vol_from_strike,
)
from valax.surfaces import GridVolSurface, SABRVolSurface, SVIVolSurface


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def flat_svi():
    """Flat SVI surface (b=0) with constant IV = 0.25 across (k, T)."""
    sigma = 0.25
    expiries = jnp.array([0.05, 0.5, 1.0, 2.0])
    return SVIVolSurface(
        expiries=expiries,
        forwards=jnp.full_like(expiries, 100.0),
        a_vec=sigma ** 2 * expiries,
        b_vec=jnp.zeros_like(expiries),
        rho_vec=jnp.zeros_like(expiries),
        m_vec=jnp.zeros_like(expiries),
        sigma_vec=jnp.full_like(expiries, 0.1),  # ignored when b=0
    )


@pytest.fixture
def smile_svi():
    """SVI surface with realistic equity smile/skew."""
    expiries = jnp.array([0.05, 0.25, 0.5, 1.0, 2.0])
    return SVIVolSurface(
        expiries=expiries,
        forwards=jnp.full_like(expiries, 100.0),
        a_vec=jnp.array([0.0008, 0.005, 0.012, 0.028, 0.06]),
        b_vec=jnp.array([0.04, 0.06, 0.08, 0.10, 0.12]),
        rho_vec=jnp.array([-0.3, -0.3, -0.3, -0.3, -0.3]),
        m_vec=jnp.zeros_like(expiries),
        sigma_vec=jnp.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    )


# ─────────────────────────────────────────────────────────────────────
# 1. Flat-surface limit
# ─────────────────────────────────────────────────────────────────────


class TestFlatSurfaceLimit:
    """A flat SVI surface must give sigma_loc ≡ sigma_iv exactly."""

    @pytest.mark.parametrize("k", [-0.3, -0.1, 0.0, 0.1, 0.3])
    @pytest.mark.parametrize("T", [0.1, 0.3, 0.6, 1.0, 1.5])
    def test_constant_local_vol(self, flat_svi, k, T):
        sigma = dupire_local_vol(flat_svi, jnp.array(k), jnp.array(T))
        assert jnp.isclose(sigma, 0.25, atol=1e-10), (
            f"k={k}, T={T}: sigma_loc={float(sigma)} expected 0.25"
        )

    def test_from_strike_wrapper_matches(self, flat_svi):
        """The ``_from_strike`` ergonomic wrapper agrees with the core fn."""
        K = jnp.array(95.0)
        T = jnp.array(0.75)
        forward = jnp.array(100.0)
        k = jnp.log(K / forward)

        s_strike = dupire_local_vol_from_strike(flat_svi, K, T, forward)
        s_k = dupire_local_vol(flat_svi, k, T)
        assert jnp.isclose(s_strike, s_k, atol=1e-14)


# ─────────────────────────────────────────────────────────────────────
# 2. Autodiff vs finite differences
# ─────────────────────────────────────────────────────────────────────


class TestAutodiff:
    """``jax.grad`` of Dupire w.r.t. SVI params matches central FD."""

    @staticmethod
    def _dupire_of_param(surface, field_name, new_value, k, T):
        """Build a new surface with a single field replaced and price."""
        import equinox as eqx

        new_surface = eqx.tree_at(
            lambda s: getattr(s, field_name), surface, new_value
        )
        return dupire_local_vol(new_surface, k, T)

    @pytest.mark.parametrize("field,slice_idx", [
        ("a_vec", 2),
        ("b_vec", 2),
        ("rho_vec", 2),
    ])
    def test_grad_vs_fd(self, smile_svi, field, slice_idx):
        k = jnp.array(-0.05)
        T = jnp.array(0.5)
        h = 1e-5

        def f(scalar):
            vec = getattr(smile_svi, field).at[slice_idx].set(scalar)
            return self._dupire_of_param(smile_svi, field, vec, k, T)

        x0 = float(getattr(smile_svi, field)[slice_idx])
        g_ad = float(jax.grad(f)(jnp.array(x0)))
        g_fd = (float(f(jnp.array(x0 + h))) - float(f(jnp.array(x0 - h)))) / (2 * h)

        # Loose-ish: SVI param sensitivities span several orders of
        # magnitude; 1e-4 relative is the relevant gate.
        denom = max(abs(g_fd), 1e-6)
        rel = abs(g_ad - g_fd) / denom
        assert rel < 1e-4, (
            f"field={field}[{slice_idx}]  ad={g_ad:.6g}  fd={g_fd:.6g}  rel={rel:.2e}"
        )


# ─────────────────────────────────────────────────────────────────────
# 3. Arbitrage detection
# ─────────────────────────────────────────────────────────────────────


class TestArbitrageDetection:
    """A butterfly-violating slice gives a non-positive denominator → NaN."""

    def test_butterfly_violation_nans(self):
        """Hand-crafted pathological SVI: extremely large b at a far wing
        creates a butterfly arb that should produce NaN/Inf from Dupire."""
        # b is enormous → ∂²w/∂k² blows up; combined with the small w near
        # the wing the denominator can go non-positive.
        expiries = jnp.array([0.5, 1.0])
        surf = SVIVolSurface(
            expiries=expiries,
            forwards=jnp.array([100.0, 100.0]),
            a_vec=jnp.array([0.001, 0.002]),
            b_vec=jnp.array([5.0, 5.0]),          # absurd wing slope
            rho_vec=jnp.array([-0.99, -0.99]),    # near-degenerate
            m_vec=jnp.array([0.0, 0.0]),
            sigma_vec=jnp.array([0.01, 0.01]),    # near-vertex kink
        )
        # Far OTM: small w, large ∂²w/∂k². Probe a few points; at least
        # one should hit the arbitrage region.
        any_nan = False
        for k in [-1.0, -0.7, 0.7, 1.0]:
            s = dupire_local_vol(surf, jnp.array(k), jnp.array(0.75))
            if not jnp.isfinite(s):
                any_nan = True
                break
        assert any_nan, (
            "Expected at least one NaN from a butterfly-violating SVI; "
            "Dupire never tripped — review the test surface."
        )


# ─────────────────────────────────────────────────────────────────────
# 4. Golden
# ─────────────────────────────────────────────────────────────────────


class TestGolden:
    """Pin a 5×5 canonical Dupire grid for drift detection."""

    def test_canonical_grid(self, smile_svi):
        from tests.golden._helpers import assert_matches_golden

        ks = jnp.linspace(-0.2, 0.2, 5)
        Ts = jnp.linspace(0.15, 1.5, 5)

        grid = jax.vmap(
            jax.vmap(
                lambda k, T: dupire_local_vol(smile_svi, k, T),
                in_axes=(None, 0),
            ),
            in_axes=(0, None),
        )(ks, Ts)

        assert_matches_golden(
            "dupire_canonical_grid",
            grid,
            version=1,
            rtol=1e-10,
            atol=1e-10,
        )


# ─────────────────────────────────────────────────────────────────────
# 5. JIT + vmap
# ─────────────────────────────────────────────────────────────────────


class TestJitAndVmap:
    def test_jit(self, smile_svi):
        f = jax.jit(lambda s, k, T: dupire_local_vol(s, k, T))
        s1 = float(dupire_local_vol(smile_svi, jnp.array(0.0), jnp.array(0.5)))
        s2 = float(f(smile_svi, jnp.array(0.0), jnp.array(0.5)))
        assert abs(s1 - s2) < 1e-12

    def test_double_vmap(self, smile_svi):
        ks = jnp.linspace(-0.2, 0.2, 7)
        Ts = jnp.linspace(0.15, 1.5, 4)
        grid = jax.vmap(
            jax.vmap(
                lambda k, T: dupire_local_vol(smile_svi, k, T),
                in_axes=(None, 0),
            ),
            in_axes=(0, None),
        )(ks, Ts)
        assert grid.shape == (7, 4)
        assert jnp.all(jnp.isfinite(grid))
        assert jnp.all(grid > 0)


# ─────────────────────────────────────────────────────────────────────
# 6. x64 guard
# ─────────────────────────────────────────────────────────────────────


class TestX64Guard:
    """RuntimeError if jax_enable_x64 is disabled."""

    def test_x64_disabled_raises(self, smile_svi):
        # Toggle x64 off temporarily. ``valax/__init__.py`` enables it
        # globally; we explicitly disable to probe the guard.
        jax.config.update("jax_enable_x64", False)
        try:
            with pytest.raises(RuntimeError, match="jax_enable_x64"):
                dupire_local_vol(smile_svi, jnp.array(0.0), jnp.array(0.5))
        finally:
            jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────────────
# 7. Duck-typed surface protocol
# ─────────────────────────────────────────────────────────────────────


class TestSurfaceProtocol:
    """SVI, SABR, and Grid surfaces all satisfy ``total_variance``."""

    def test_all_three_surfaces_consumable(self):
        # SVI
        expiries = jnp.array([0.05, 0.5, 1.0, 2.0])
        sigma_flat = 0.20
        svi = SVIVolSurface(
            expiries=expiries,
            forwards=jnp.full_like(expiries, 100.0),
            a_vec=sigma_flat ** 2 * expiries,
            b_vec=jnp.zeros_like(expiries),
            rho_vec=jnp.zeros_like(expiries),
            m_vec=jnp.zeros_like(expiries),
            sigma_vec=jnp.full_like(expiries, 0.1),
        )

        # SABR
        sabr = SABRVolSurface(
            expiries=jnp.array([0.5, 1.0, 2.0]),
            forwards=jnp.full(3, 100.0),
            alphas=jnp.array([0.20, 0.22, 0.25]),
            betas=jnp.array([0.5, 0.5, 0.5]),
            rhos=jnp.array([-0.3, -0.3, -0.3]),
            nus=jnp.array([0.4, 0.4, 0.4]),
        )

        # Grid (built in log-moneyness so it composes with Dupire's
        # log-moneyness query convention; see grid.total_variance docstring).
        ks = jnp.linspace(-0.3, 0.3, 7)
        Ts = jnp.array([0.5, 1.0, 2.0])
        grid_vols = jnp.full((3, 7), 0.20)
        grid = GridVolSurface(strikes=ks, expiries=Ts, vols=grid_vols)

        k_q = jnp.array(0.0)
        T_q = jnp.array(0.75)

        # All three must return a finite scalar
        for name, surf in [("SVI", svi), ("SABR", sabr), ("Grid", grid)]:
            w = surf.total_variance(k_q, T_q)
            assert jnp.isfinite(w), f"{name} total_variance not finite"
            assert w > 0, f"{name} total_variance not positive"

        # Dupire on flat SVI should give the constant IV exactly
        s_svi = dupire_local_vol(svi, k_q, T_q)
        assert jnp.isclose(s_svi, sigma_flat, atol=1e-10)

        # Dupire on flat Grid (constant 0.20 vol everywhere): also exact
        # — w(k, T) is linear in T (since IV is constant), ∂w/∂k = 0,
        # ∂²w/∂k² = 0, so the formula reduces to sqrt(∂_T w) = IV.
        s_grid = dupire_local_vol(grid, k_q, T_q)
        assert jnp.isclose(s_grid, sigma_flat, atol=1e-6), (
            f"Grid Dupire returned {float(s_grid)}, expected {sigma_flat}"
        )
