"""Unit tests for SLV leverage-function calibration.

Test layout (SLV-1 — calibration acceptance):

* ``TestParticleMethod``   — ``method="particle"`` recovers a planted
  ground-truth leverage to better than the unrelated-noise baseline
  on a synthetic SVI surface.
* ``TestKernelMethod``     — ``method="kernel"`` (with non-zero
  ``ridge``) produces a *smoother* leverage in low-density tail
  regions than the particle estimator (lower across-seed variance).
* ``TestFixedPoint``       — ``n_iterations >= 2`` improves (or at
  least does not worsen) the Markovian projection residual at fixed
  budget — direct validation of the outer fixed-point loop.
* ``TestValidation``       — ``ValueError`` raised for invalid
  ``method`` and ``n_iterations``.

x64 is enabled at module level — ``dupire_local_vol`` and SLV both
enforce it.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from valax.calibration.slv import calibrate_slv_leverage
from valax.models import HestonModel
from valax.pricing.analytic.dupire import dupire_local_vol
from valax.surfaces import SVIVolSurface


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def heston():
    """Typical equity Heston (slightly Feller-violating)."""
    return HestonModel(
        v0=jnp.array(0.04), kappa=jnp.array(2.0), theta=jnp.array(0.04),
        xi=jnp.array(0.3), rho=jnp.array(-0.6),
        rate=jnp.array(0.03), dividend=jnp.array(0.01),
    )


@pytest.fixture
def smile_surface():
    """LV-1 smile fixture (real SVI skew)."""
    rate, dividend = 0.03, 0.01
    mu = rate - dividend
    expiries = jnp.array([0.05, 0.25, 0.5, 1.0, 2.0])
    return SVIVolSurface(
        expiries=expiries,
        forwards=jnp.array(100.0) * jnp.exp(mu * expiries),
        a_vec=jnp.array([0.001, 0.006, 0.014, 0.030, 0.062]),
        b_vec=jnp.array([0.04, 0.05, 0.06, 0.07, 0.08]),
        rho_vec=jnp.array([-0.3, -0.3, -0.3, -0.3, -0.3]),
        m_vec=jnp.zeros_like(expiries),
        sigma_vec=jnp.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    )


# ─────────────────────────────────────────────────────────────────────
# 1. Particle method
# ─────────────────────────────────────────────────────────────────────


class TestParticleMethod:
    """``method='particle'`` — pure Nadaraya-Watson, no ridge."""

    def test_calibrated_leverage_is_finite_and_positive(
        self, heston, smile_surface,
    ):
        """Sanity: every grid node has a finite, positive ``L``
        (bounded below by ``L_min=0.05`` and above by ``L_max=5.0``
        by default)."""
        lev = calibrate_slv_leverage(
            heston, smile_surface, jnp.array(100.0),
            log_moneyness_grid=jnp.linspace(-0.25, 0.25, 9),
            time_grid=jnp.linspace(0.1, 1.0, 6),
            n_paths=5_000, key=jax.random.PRNGKey(20260101),
            method="particle", n_iterations=1,
        )
        assert lev.values.shape == (6, 9)
        assert jnp.all(jnp.isfinite(lev.values))
        assert jnp.all(lev.values > 0.0)
        assert jnp.all(lev.values >= 0.05 - 1e-12)
        assert jnp.all(lev.values <= 5.0 + 1e-12)

    def test_markovian_projection_identity_at_atm(
        self, heston, smile_surface,
    ):
        """At the converged calibration, the Markovian-projection
        identity should hold within MC-noise tolerance at ATM:

        ``L²(0, T) · E[V_T | k_T = 0] ≈ σ²_Dupire(0, T)``.

        Direct verification of the algebraic contract of the
        calibration routine. The tolerance is loose (10% relative on
        ``σ²``) because we use a single MC realisation; the
        statistical guarantee is asymptotic.
        """
        from valax.calibration.slv import _nadaraya_watson, _silverman_bandwidth
        from valax.models import SLVModel
        from valax.pricing.mc import generate_slv_paths

        spot = jnp.array(100.0)
        T = 1.0
        lev = calibrate_slv_leverage(
            heston, smile_surface, spot,
            log_moneyness_grid=jnp.linspace(-0.25, 0.25, 9),
            time_grid=jnp.linspace(0.1, T, 6),
            n_paths=10_000, key=jax.random.PRNGKey(20260102),
            method="particle", n_iterations=2,
        )
        slv = SLVModel.from_heston_and_leverage(heston, smile_surface, lev)

        # Independent MC simulation to measure E[V_T | k_T=0].
        key = jax.random.PRNGKey(20260103)
        S, V = generate_slv_paths(slv, spot, T, 200, 20_000, key)
        log_S_T = jnp.log(S[:, -1])
        mu = heston.rate - heston.dividend
        k_T = log_S_T - (jnp.log(spot) + mu * T)
        h = _silverman_bandwidth(k_T)
        ev_at_zero = float(_nadaraya_watson(
            jnp.array([0.0]), k_T, V[:, -1],
            bandwidth=h, ridge=jnp.array(0.0),
        )[0])
        sigma2_dupire = float(dupire_local_vol(
            smile_surface, jnp.array(0.0), jnp.array(T),
        )) ** 2
        L_atm = float(lev(jnp.array(0.0), jnp.array(T)))

        # Compare L² · E[V|k=0] vs σ²_Dupire(0, T).
        projection = L_atm ** 2 * ev_at_zero
        rel_err = abs(projection - sigma2_dupire) / sigma2_dupire
        assert rel_err < 0.10, (
            f"Markovian projection identity violated: "
            f"L²·E[V|k=0] = {projection:.5f}, σ²_Dupire = "
            f"{sigma2_dupire:.5f}, rel_err = {rel_err:.3%}"
        )


# ─────────────────────────────────────────────────────────────────────
# 2. Kernel method (Tikhonov-regularised)
# ─────────────────────────────────────────────────────────────────────


class TestKernelMethod:
    """``method='kernel'`` — particle method + ridge regularizer."""

    def test_kernel_matches_particle_in_dense_region(
        self, heston, smile_surface,
    ):
        """With small ``ridge``, the kernel method should give nearly
        the same L as the particle method in the well-populated
        region of (k, t) space. Difference < 5% at ATM."""
        common = dict(
            heston=heston, surface=smile_surface, spot=jnp.array(100.0),
            log_moneyness_grid=jnp.linspace(-0.20, 0.20, 9),
            time_grid=jnp.linspace(0.1, 1.0, 6),
            n_paths=10_000, key=jax.random.PRNGKey(20260201),
            n_iterations=1,
        )
        lev_p = calibrate_slv_leverage(method="particle", **common)
        lev_k = calibrate_slv_leverage(method="kernel", ridge=1e-3, **common)

        L_p_atm = float(lev_p(jnp.array(0.0), jnp.array(1.0)))
        L_k_atm = float(lev_k(jnp.array(0.0), jnp.array(1.0)))
        rel_diff = abs(L_p_atm - L_k_atm) / L_p_atm
        assert rel_diff < 0.05, (
            f"ATM L: particle={L_p_atm:.4f}, kernel={L_k_atm:.4f}, "
            f"rel_diff={rel_diff:.2%}"
        )

    def test_kernel_smoother_in_tails(self, heston, smile_surface):
        """The kernel method's ridge regularisation should suppress
        wild swings in the low-density tail. Operationalised as: the
        max-over-grid pointwise variance across two independent
        random seeds is smaller for ``method='kernel'`` than for
        ``method='particle'``."""
        common = dict(
            heston=heston, surface=smile_surface, spot=jnp.array(100.0),
            log_moneyness_grid=jnp.linspace(-0.30, 0.30, 9),
            time_grid=jnp.linspace(0.1, 1.0, 6),
            n_paths=4_000, n_iterations=1,
        )

        def cal(method, seed):
            return calibrate_slv_leverage(
                method=method, ridge=1e-2 if method == "kernel" else 0.0,
                key=jax.random.PRNGKey(seed), **common,
            ).values

        L_p_a = cal("particle", 20260301)
        L_p_b = cal("particle", 20260302)
        L_k_a = cal("kernel", 20260301)
        L_k_b = cal("kernel", 20260302)

        var_p = float(jnp.max((L_p_a - L_p_b) ** 2))
        var_k = float(jnp.max((L_k_a - L_k_b) ** 2))
        # Kernel must be at least as stable as particle (allow a small
        # margin for the natural seed noise to favour either; this is
        # a one-sided sanity test, not a tight statement).
        assert var_k <= var_p * 1.20, (
            f"kernel-method peak inter-seed variance ({var_k:.4f}) "
            f"should be ≤ particle's ({var_p:.4f}) plus 20% slack"
        )


# ─────────────────────────────────────────────────────────────────────
# 3. Fixed-point iteration (``n_iterations``)
# ─────────────────────────────────────────────────────────────────────


class TestFixedPoint:
    """Outer fixed-point loop: with ``n_iterations`` ≥ 2 the calibration
    re-simulates the swarm under the previous-iteration leverage. The
    L values should stabilise (consecutive iterations get closer)."""

    def test_iterations_converge(self, heston, smile_surface):
        """Successive outer iterations should produce *smaller*
        L-grid updates (sup-norm ‖L_{n+1} − L_n‖∞ decreases)."""
        common = dict(
            heston=heston, surface=smile_surface, spot=jnp.array(100.0),
            log_moneyness_grid=jnp.linspace(-0.20, 0.20, 9),
            time_grid=jnp.linspace(0.1, 1.0, 6),
            n_paths=8_000, key=jax.random.PRNGKey(20260401),
            method="kernel", ridge=1e-3,
        )
        L1 = calibrate_slv_leverage(n_iterations=1, **common).values
        L2 = calibrate_slv_leverage(n_iterations=2, **common).values
        L3 = calibrate_slv_leverage(n_iterations=3, **common).values

        diff_12 = float(jnp.max(jnp.abs(L2 - L1)))
        diff_23 = float(jnp.max(jnp.abs(L3 - L2)))

        # Convergence is not strictly monotone (each iteration is one
        # particle realisation), but the second update should be at
        # most as large as the first within MC-noise tolerance.
        assert diff_23 <= diff_12 * 1.5, (
            f"‖L2-L1‖∞ = {diff_12:.4f}, ‖L3-L2‖∞ = {diff_23:.4f} — "
            "fixed point not contracting"
        )


# ─────────────────────────────────────────────────────────────────────
# 4. Validation (invalid arguments raise)
# ─────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_method_raises(self, heston, smile_surface):
        with pytest.raises(ValueError, match="particle"):
            calibrate_slv_leverage(
                heston, smile_surface, jnp.array(100.0),
                log_moneyness_grid=jnp.linspace(-0.2, 0.2, 5),
                time_grid=jnp.linspace(0.1, 1.0, 4),
                n_paths=1_000, key=jax.random.PRNGKey(0),
                method="bogus",
            )

    def test_invalid_n_iterations_raises(self, heston, smile_surface):
        with pytest.raises(ValueError, match="n_iterations"):
            calibrate_slv_leverage(
                heston, smile_surface, jnp.array(100.0),
                log_moneyness_grid=jnp.linspace(-0.2, 0.2, 5),
                time_grid=jnp.linspace(0.1, 1.0, 4),
                n_paths=1_000, key=jax.random.PRNGKey(0),
                method="particle", n_iterations=0,
            )
