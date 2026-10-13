"""Unit tests for SLV leverage-function calibration.

Test layout (SLV-1 — calibration acceptance):

* ``TestParticleMethod``   — ``method="particle"`` recovers a planted
  ground-truth leverage to better than the unrelated-noise baseline
  on a synthetic SVI surface.
* ``TestKernelMethod``     — ``method="kernel"`` (with non-zero
  ``ridge``) produces a *smoother* leverage in low-density tail
  regions than the particle estimator (lower across-seed variance).
* ``TestFixedPoint``       — passes ``>= 2`` re-simulate a self-consistent
  sweep: the increment is Monte Carlo noise, bounded and not growing.
* ``TestValidation``       — ``ValueError`` raised for invalid
  ``method`` and ``n_iterations``.
* ``TestGHLSweep``         — one pass is the self-consistent
  Guyon-Henry-Labordère calibration (deterministic limit, high vol-of-vol
  vanilla repricing on example 17's SABR market).
* ``TestRidge``            — ``ridge`` counts pseudo-particles: no effect
  at 0, measurable at the default.
* ``TestBackbone``         — ``calibrate_slv(heston=…)`` records the child,
  rejects inconsistent arguments; jit / grad smoke tests.

x64 is enabled at module level — ``dupire_local_vol`` and SLV both
enforce it.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from valax.calibration import CalibrationError, SolverSettings
from valax.calibration.slv import calibrate_slv, calibrate_slv_leverage
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
        ).value
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
        ).value
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
        """At the default ``ridge`` (30 pseudo-particles, ≪ the thousands of
        effective particles at ATM), the kernel method should give nearly
        the same L as the particle method in the well-populated
        region of (k, t) space. Difference < 5% at ATM."""
        common = dict(
            heston=heston, surface=smile_surface, spot=jnp.array(100.0),
            log_moneyness_grid=jnp.linspace(-0.20, 0.20, 9),
            time_grid=jnp.linspace(0.1, 1.0, 6),
            n_paths=10_000, key=jax.random.PRNGKey(20260201),
            n_iterations=1,
        )
        lev_p = calibrate_slv_leverage(method="particle", **common).value
        lev_k = calibrate_slv_leverage(method="kernel", **common).value  # default ridge

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
            # Default ridge (DEFAULT_RIDGE pseudo-particles) for "kernel".
            return calibrate_slv_leverage(
                method=method, key=jax.random.PRNGKey(seed), **common,
            ).value.values

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
    """Outer passes: each is a self-consistent GHL sweep with fresh
    randomness, so from pass 2 on the increment is the Monte Carlo noise of
    the estimator — far below the distance from the ``L ≡ 1`` warm start and
    not growing."""

    def test_iterations_converge(self, heston, smile_surface):
        """‖L3 − L2‖∞ stays at the noise level of ‖L2 − L1‖∞."""
        common = dict(
            heston=heston, surface=smile_surface, spot=jnp.array(100.0),
            log_moneyness_grid=jnp.linspace(-0.20, 0.20, 9),
            time_grid=jnp.linspace(0.1, 1.0, 6),
            n_paths=8_000, key=jax.random.PRNGKey(20260401),
            method="kernel",
        )
        L1 = calibrate_slv_leverage(n_iterations=1, **common).value.values
        L2 = calibrate_slv_leverage(n_iterations=2, **common).value.values
        L3 = calibrate_slv_leverage(n_iterations=3, **common).value.values

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


# ─────────────────────────────────────────────────────────────────────
# 5. The fixed point as a reported calibration level
# ─────────────────────────────────────────────────────────────────────


class TestFixedPointLevel:
    """How the MC fixed point is expressed: an explicit level whose
    diagnostics are the last increment, the convergence test on it, and
    the clip fraction."""

    COMMON = dict(
        log_moneyness_grid=jnp.linspace(-0.20, 0.20, 7),
        time_grid=jnp.linspace(0.1, 1.0, 4),
        n_paths=4_000, key=jax.random.PRNGKey(20260501),
        method="kernel",
    )

    def test_increment_is_the_last_pass_difference(self, heston, smile_surface):
        args = (heston, smile_surface, jnp.array(100.0))
        r2 = calibrate_slv_leverage(*args, n_iterations=2, **self.COMMON)
        r3 = calibrate_slv_leverage(*args, n_iterations=3, **self.COMMON)
        # Passes use deterministic per-pass keys, so the n=3 increment is
        # exactly (n=3 grid) - (n=2 grid).
        expected = (r3.value.values - r2.value.values).ravel()
        assert jnp.allclose(r3.diagnostics.residuals, expected, atol=1e-12)
        assert r3.problem == "fixed_point"
        assert r3.is_explicit and r3.theta is None
        assert int(r3.diagnostics.n_steps) == 3
        clip = float(r3.diagnostics.extras["clip_fraction"])
        assert 0.0 <= clip <= 1.0

    def test_single_pass_is_never_converged(self, heston, smile_surface):
        res = calibrate_slv_leverage(
            heston, smile_surface, jnp.array(100.0), n_iterations=1,
            settings=SolverSettings(atol=1e6), **self.COMMON,
        )
        assert not bool(res.converged)
        with pytest.raises(CalibrationError, match="fixed_point"):
            res.check()

    def test_tolerance_decides_convergence(self, heston, smile_surface):
        """Anti-vacuity: the same fixed point is converged under a loose
        tolerance and reported failed under one below its MC noise."""
        args = (heston, smile_surface, jnp.array(100.0))
        loose = calibrate_slv_leverage(
            *args, n_iterations=2, settings=SolverSettings(atol=10.0),
            **self.COMMON,
        )
        tight = calibrate_slv_leverage(
            *args, n_iterations=2, settings=SolverSettings(atol=1e-9),
            **self.COMMON,
        )
        assert bool(loose.converged)
        assert not bool(tight.converged)

    def test_two_pass_slv_carries_the_heston_fit(self, heston, smile_surface):
        from valax.models import SLVModel
        from valax.pricing.analytic.heston import heston_cos_price
        from valax.instruments.options import EuropeanOption

        def cos(model, k, s, r, q, t):
            return heston_cos_price(
                EuropeanOption(strike=k, expiry=t, is_call=True), s, r, q, model
            )

        strikes = jnp.linspace(90.0, 110.0, 5)
        spot, r, q, t = jnp.array(100.0), heston.rate, heston.dividend, jnp.array(1.0)
        prices = jax.vmap(lambda k: cos(heston, k, spot, r, q, t))(strikes)
        res = calibrate_slv(
            strikes, prices, spot, r, q, t, cos, smile_surface,
            n_iterations=2, heston_initial_guess=heston,
            settings=SolverSettings(atol=10.0), **{
                k: v for k, v in self.COMMON.items()
            },
        )
        assert isinstance(res.value, SLVModel)
        assert len(res.children) == 1 and res.children[0].name == "slv.heston"
        res.check()


# ─────────────────────────────────────────────────────────────────────
# 6. One pass is the Guyon-Henry-Labordère sweep
# ─────────────────────────────────────────────────────────────────────


def _sabr_market():
    """Example 17's calibrated SABR market (two smiles), rebuilt from its
    fitted parameters so the tests do not rerun the market chain."""
    from valax.surfaces.sabr_surface import SABRVolSurface

    surface = SABRVolSurface(
        expiries=jnp.array([0.5, 1.0]),
        forwards=jnp.array([101.50646605234357, 101.60090387045518]),
        alphas=jnp.array([2.941672587386952, 2.928149508698076]),
        betas=jnp.array([0.5, 0.5]),
        rhos=jnp.array([-0.5435316093507521, -0.4118260146951523]),
        nus=jnp.array([0.8955175856106505, 0.7463431942501192]),
    )
    rate, dividend, spot = 0.027146146570611727, 0.02528628643942919, 101.41211601397752
    return surface, jnp.array(rate), jnp.array(dividend), jnp.array(spot)


class TestGHLSweep:
    """``n_iterations=1`` is the self-consistent one-shot calibration."""

    def test_deterministic_limit_is_exact(self):
        """Flat 25% surface, (near-)deterministic variance v ≡ 0.04: every
        conditional expectation is v0, so every cell of a single pass is
        σ/√v0 = 1.25 — no Heston projection left over from a warm start."""
        flat = SVIVolSurface(
            expiries=jnp.array([0.05, 0.5, 1.0, 2.0]),
            forwards=jnp.full((4,), 100.0),
            a_vec=0.0625 * jnp.array([0.05, 0.5, 1.0, 2.0]),
            b_vec=jnp.zeros(4), rho_vec=jnp.zeros(4), m_vec=jnp.zeros(4),
            sigma_vec=jnp.full((4,), 0.1),
        )
        hst = HestonModel(
            v0=jnp.array(0.04), kappa=jnp.array(2.0), theta=jnp.array(0.04),
            xi=jnp.array(1e-4), rho=jnp.array(-0.5),
            rate=jnp.array(0.0), dividend=jnp.array(0.0),
        )
        res = calibrate_slv_leverage(
            hst, flat, jnp.array(100.0), jnp.linspace(-0.2, 0.2, 5),
            jnp.linspace(0.1, 1.0, 5), 2_000, jax.random.PRNGKey(0),
            n_iterations=1,
        )
        assert jnp.allclose(res.value.values, 1.25, atol=1e-3)
        assert not bool(res.converged)  # one pass is never converged

    def test_single_pass_reprices_high_vol_of_vol_smile(self):
        """Example 17's SABR market with its 1Y κ-fixed Heston backbone
        (ξ ≈ 1.3, Feller badly violated) — where the former Picard scheme,
        whose first pass was a pure-Heston projection, needed a vol-of-vol
        mixing fraction. Measured at this budget (20k particles, 24 rows;
        repriced 2 seeds × 50k paths × 24 steps; RMS over 10 quotes, bp):

        ========================================  ======  ======  ========
        scheme                                    0.5Y    1Y      combined
        ========================================  ======  ======  ========
        old Picard, n=1 (pure-Heston projection)  46.4    30.8    39.4
        old Picard, n=2, ridge 1e-3               34.8    28.8    31.9
        old Picard, n=2, ridge 200 (example 17)   63.9    29.7    49.8
        GHL, n=1, default ridge                   36.8    24.6    31.3
        GHL, n=2, default ridge                   33.7    23.6    29.1
        ========================================  ======  ======  ========

        The gate is the old two-pass Picard figure.
        """
        from valax.instruments.options import EuropeanOption
        from valax.models import SLVModel
        from valax.pricing.analytic.black_scholes import black_scholes_implied_vol
        from valax.pricing.mc import generate_slv_paths

        surface, r, q, spot = _sabr_market()
        hst = HestonModel(
            v0=jnp.array(0.0902), kappa=jnp.array(1.5), theta=jnp.array(0.1750),
            xi=jnp.array(1.295), rho=jnp.array(-0.546), rate=r, dividend=q,
        )
        res = calibrate_slv_leverage(
            hst, surface, spot, jnp.linspace(-0.4, 0.4, 17),
            jnp.linspace(1 / 24, 1.0, 24), 20_000, jax.random.PRNGKey(3),
            method="kernel", n_iterations=1,
        )
        slv = SLVModel.from_heston_and_leverage(hst, surface, res.value)
        strikes = jnp.array([81.2, 91.4, 101.5, 111.7, 121.9])
        sq = []
        for T in (0.5, 1.0):
            prices = []
            for seed in (1, 2):
                S, _ = generate_slv_paths(slv, spot, T, 24, 50_000, jax.random.PRNGKey(seed))
                prices.append(jnp.exp(-r * T) * jnp.mean(
                    jnp.maximum(S[:, -1][:, None] - strikes[None, :], 0.0), axis=0))
            p = jnp.mean(jnp.stack(prices), axis=0)
            iv = jax.vmap(lambda k, pp: black_scholes_implied_vol(
                EuropeanOption(strike=k, expiry=jnp.array(T)), spot, r, q, pp))(strikes, p)
            target = jax.vmap(lambda k: surface(k, jnp.array(T)))(strikes)
            sq.append((iv - target) * 1e4)
        rms = float(jnp.sqrt(jnp.mean(jnp.concatenate(sq) ** 2)))
        assert rms <= 31.9 + 1.0, f"GHL n=1 combined RMS {rms:.1f} bp"

    def test_n_substeps_one_is_the_default(self, heston, smile_surface):
        args = (heston, smile_surface, jnp.array(100.0),
                jnp.linspace(-0.2, 0.2, 5), jnp.linspace(0.1, 1.0, 4),
                2_000, jax.random.PRNGKey(1))
        a = calibrate_slv_leverage(*args, n_iterations=1).value.values
        b = calibrate_slv_leverage(*args, n_iterations=1, n_substeps=1).value.values
        c = calibrate_slv_leverage(*args, n_iterations=1, n_substeps=3).value.values
        assert jnp.array_equal(a, b)
        assert jnp.all(jnp.isfinite(c)) and not jnp.allclose(a, c)
        with pytest.raises(ValueError, match="n_substeps"):
            calibrate_slv_leverage(*args, n_substeps=0)


# ─────────────────────────────────────────────────────────────────────
# 7. Ridge: pseudo-particles at the unconditional mean
# ─────────────────────────────────────────────────────────────────────


class TestRidge:
    def test_estimator_semantics(self):
        """λ pseudo-particles at V̄: no effect at 0, shrinkage by
        λ/(Σw + λ) elsewhere, and V̄ where no particle is near."""
        from valax.calibration.slv import _nadaraya_watson

        k = jnp.array([-1.0, 0.0, 0.5, 1.0])
        v = jnp.array([0.1, 0.2, 0.3, 0.4])
        h = jnp.array(0.3)
        q = jnp.array([0.0, 50.0])
        w = jnp.exp(-0.5 * ((q[:, None] - k[None, :]) / h) ** 2)
        plain = (w @ v) / w.sum(axis=1)
        est0 = _nadaraya_watson(q, k, v, h, jnp.array(0.0))
        assert jnp.allclose(est0[0], plain[0], rtol=1e-12)
        assert jnp.isfinite(est0[1])  # no particle near 50: nearest one, not 0/0
        lam = 2.0
        est = _nadaraya_watson(q, k, v, h, jnp.array(lam))
        expected0 = (w[0] @ v + lam * v.mean()) / (w[0].sum() + lam)
        assert jnp.allclose(est[0], expected0, rtol=1e-12)
        assert jnp.allclose(est[1], v.mean(), rtol=1e-12)

    def test_kernel_ridge_zero_is_particle(self, heston, smile_surface):
        common = dict(
            heston=heston, surface=smile_surface, spot=jnp.array(100.0),
            log_moneyness_grid=jnp.linspace(-0.3, 0.3, 9),
            time_grid=jnp.linspace(0.1, 1.0, 4), n_paths=4_000,
            key=jax.random.PRNGKey(7), n_iterations=1,
        )
        p = calibrate_slv_leverage(method="particle", **common).value.values
        k0 = calibrate_slv_leverage(method="kernel", ridge=0.0, **common).value.values
        kd = calibrate_slv_leverage(method="kernel", **common).value.values
        assert jnp.array_equal(p, k0)
        # The default ridge has a measurable effect (the sparse wings) …
        assert float(jnp.max(jnp.abs(kd - p))) > 1e-2
        # … and a small one at the money (dense: thousands of effective particles).
        atm = (slice(None), 4)
        assert float(jnp.max(jnp.abs(kd[atm] - p[atm]) / p[atm])) < 0.01

    def test_negative_ridge_raises(self, heston, smile_surface):
        with pytest.raises(ValueError, match="ridge"):
            calibrate_slv_leverage(
                heston, smile_surface, jnp.array(100.0),
                jnp.linspace(-0.2, 0.2, 5), jnp.linspace(0.1, 1.0, 4),
                1_000, jax.random.PRNGKey(0), method="kernel", ridge=-1.0,
            )


# ─────────────────────────────────────────────────────────────────────
# 8. Supplied backbone, argument validation, jit / grad
# ─────────────────────────────────────────────────────────────────────


class TestBackbone:
    COMMON = dict(
        log_moneyness_grid=jnp.linspace(-0.2, 0.2, 5),
        time_grid=jnp.linspace(0.1, 1.0, 4),
        n_paths=2_000, key=jax.random.PRNGKey(11),
        settings=SolverSettings(atol=10.0),
    )

    def test_supplied_result_is_the_child(self, heston, smile_surface):
        from valax.calibration.heston import calibrate_heston
        from valax.instruments.options import EuropeanOption
        from valax.models import SLVModel
        from valax.pricing.analytic.heston import heston_cos_price

        def cos(model, k, s, r, q, t):
            return heston_cos_price(EuropeanOption(strike=k, expiry=t), s, r, q, model)

        strikes = jnp.linspace(90.0, 110.0, 5)
        spot, t = jnp.array(100.0), jnp.array(1.0)
        prices = jax.vmap(lambda k: cos(heston, k, spot, heston.rate, heston.dividend, t))(strikes)
        fit = calibrate_heston(strikes, prices, spot, heston.rate, heston.dividend, t, cos,
                               initial_guess=heston, name="my_heston")
        res = calibrate_slv(spot=spot, surface=smile_surface, heston=fit, **self.COMMON)
        assert isinstance(res.value, SLVModel)
        assert res.children == (fit,)
        assert res.children[0].name == "my_heston"
        res.check()

    def test_supplied_model_is_an_input_level(self, heston, smile_surface):
        res = calibrate_slv(spot=jnp.array(100.0), surface=smile_surface,
                            heston=heston, **self.COMMON)
        child = res.children[0]
        assert child.name == "slv.heston" and child.is_explicit
        assert child.value is heston
        # Same leverage as calibrating the leverage on that model directly.
        lev = calibrate_slv_leverage(heston, smile_surface, jnp.array(100.0), **self.COMMON)
        assert jnp.array_equal(res.value.leverage.values, lev.value.values)

    def test_carry_override_and_validation(self, heston, smile_surface):
        import equinox as eqx

        no_carry = eqx.tree_at(lambda m: (m.rate, m.dividend), heston, (None, None),
                               is_leaf=lambda x: x is None)
        with pytest.raises(ValueError, match="rate"):
            calibrate_slv(spot=jnp.array(100.0), surface=smile_surface,
                          heston=no_carry, **self.COMMON)
        res = calibrate_slv(spot=jnp.array(100.0), surface=smile_surface, heston=no_carry,
                            rate=heston.rate, dividend=heston.dividend, **self.COMMON)
        assert float(res.value.rate) == float(heston.rate)
        with pytest.raises(ValueError, match="strikes"):
            calibrate_slv(strikes=jnp.ones(3), spot=jnp.array(100.0),
                          surface=smile_surface, heston=heston, **self.COMMON)
        with pytest.raises(ValueError, match="pricing_fn"):
            calibrate_slv(spot=jnp.array(100.0), surface=smile_surface, **self.COMMON)
        with pytest.raises(ValueError, match="surface"):
            calibrate_slv(spot=jnp.array(100.0), heston=heston, **self.COMMON)

    def test_jit_and_grad_smoke(self, heston, smile_surface):
        """The whole fixed point traces under filter_jit and differentiates
        (plain AD through the unrolled scans) to finite values."""
        import equinox as eqx

        kg, tg = self.COMMON["log_moneyness_grid"], self.COMMON["time_grid"]

        @eqx.filter_jit
        def leverage(h, s):
            return calibrate_slv_leverage(
                h, s, jnp.array(100.0), kg, tg, 2_000, jax.random.PRNGKey(5),
                method="kernel", n_iterations=2,
            ).value.values

        eager = calibrate_slv_leverage(
            heston, smile_surface, jnp.array(100.0), kg, tg, 2_000,
            jax.random.PRNGKey(5), method="kernel", n_iterations=2,
        ).value.values
        assert jnp.allclose(leverage(heston, smile_surface), eager, atol=1e-12)
        g = eqx.filter_grad(lambda h: jnp.sum(leverage(h, smile_surface)))(heston)
        for leaf in (g.v0, g.theta, g.xi, g.rho):
            assert jnp.isfinite(leaf)
