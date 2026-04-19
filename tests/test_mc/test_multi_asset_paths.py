"""Tests for correlated multi-asset GBM path generation.

Validates:

1. Shape and initial-condition correctness.
2. Statistical convergence of empirical correlation to the model input.
3. Risk-neutral drift (terminal-log-mean).
4. Agreement with the single-asset path generator when correlation is
   the identity matrix.
5. JAX transformability (jit / grad).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.models.black_scholes import BlackScholesModel
from valax.models.multi_asset import (
    MultiAssetGBMModel,
    validate_correlation,
)
from valax.pricing.mc.multi_asset_paths import generate_correlated_gbm_paths
from valax.pricing.mc.paths import generate_gbm_paths


# ─────────────────────────────────────────────────────────────────────
# Shape and initial condition
# ─────────────────────────────────────────────────────────────────────


class TestShape:
    def test_output_shape(self):
        """Paths have shape (n_paths, n_steps + 1, n_assets)."""
        n_assets = 3
        model = MultiAssetGBMModel(
            vols=jnp.full(n_assets, 0.2),
            rate=jnp.array(0.03),
            dividends=jnp.zeros(n_assets),
            correlation=jnp.eye(n_assets),
        )
        spots = jnp.full(n_assets, 100.0)
        paths = generate_correlated_gbm_paths(
            model, spots, T=1.0, n_steps=50, n_paths=1_000,
            key=jax.random.PRNGKey(0),
        )
        assert paths.shape == (1_000, 51, n_assets)

    def test_initial_condition(self):
        """First time step of every path equals the input spots."""
        n_assets = 2
        model = MultiAssetGBMModel(
            vols=jnp.array([0.2, 0.3]),
            rate=jnp.array(0.04),
            dividends=jnp.array([0.01, 0.02]),
            correlation=jnp.array([[1.0, 0.5], [0.5, 1.0]]),
        )
        spots = jnp.array([100.0, 80.0])
        paths = generate_correlated_gbm_paths(
            model, spots, T=1.0, n_steps=10, n_paths=500,
            key=jax.random.PRNGKey(1),
        )
        assert jnp.allclose(paths[:, 0, :], spots)


# ─────────────────────────────────────────────────────────────────────
# Statistical properties
# ─────────────────────────────────────────────────────────────────────


class TestStatistics:
    def test_empirical_correlation_matches_input(self):
        """Empirical correlation of log-returns converges to model input."""
        n_assets = 3
        # Distinct off-diagonals so we detect mistakes in the Cholesky.
        C = jnp.array([
            [1.0, 0.7, -0.3],
            [0.7, 1.0,  0.2],
            [-0.3, 0.2, 1.0],
        ])
        model = MultiAssetGBMModel(
            vols=jnp.array([0.2, 0.25, 0.3]),
            rate=jnp.array(0.0),  # zero drift for cleaner correlation estimate
            dividends=jnp.zeros(n_assets),
            correlation=C,
        )
        spots = jnp.full(n_assets, 100.0)
        paths = generate_correlated_gbm_paths(
            model, spots, T=1.0, n_steps=50, n_paths=50_000,
            key=jax.random.PRNGKey(42),
        )
        # Log-returns across all steps, flattened per asset.
        log_returns = jnp.diff(jnp.log(paths), axis=1)  # (n_paths, n_steps, n_assets)
        flat = log_returns.reshape(-1, n_assets)  # (n_paths * n_steps, n_assets)
        emp_corr = np.corrcoef(flat.T)

        # Element-wise tolerance: 3 / sqrt(N) is the rule of thumb for
        # correlation-estimate standard error; use 0.03 for slack.
        assert np.max(np.abs(emp_corr - np.array(C))) < 0.03

    def test_risk_neutral_drift(self):
        """Terminal empirical log-mean matches (r - q - 0.5 sigma^2) T."""
        model = MultiAssetGBMModel(
            vols=jnp.array([0.2, 0.3]),
            rate=jnp.array(0.05),
            dividends=jnp.array([0.01, 0.02]),
            correlation=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
        )
        spots = jnp.array([100.0, 100.0])
        T = 1.0
        paths = generate_correlated_gbm_paths(
            model, spots, T=T, n_steps=100, n_paths=50_000,
            key=jax.random.PRNGKey(7),
        )
        terminal_log = jnp.log(paths[:, -1, :])  # (n_paths, n_assets)
        emp_mean = jnp.mean(terminal_log, axis=0) - jnp.log(spots)
        expected = (model.rate - model.dividends - 0.5 * model.vols**2) * T

        # Std of log-terminal: sigma * sqrt(T); SE of mean: sigma * sqrt(T) / sqrt(N).
        se = model.vols * jnp.sqrt(T) / jnp.sqrt(50_000.0)
        assert jnp.all(jnp.abs(emp_mean - expected) < 3 * se)

    def test_identity_correlation_matches_single_asset(self):
        """With identity correlation, each asset should match generate_gbm_paths
        when called with independent keys of the right shape."""
        model = MultiAssetGBMModel(
            vols=jnp.array([0.2, 0.2]),
            rate=jnp.array(0.05),
            dividends=jnp.array([0.02, 0.02]),
            correlation=jnp.eye(2),
        )
        spots = jnp.array([100.0, 100.0])

        multi_paths = generate_correlated_gbm_paths(
            model, spots, T=1.0, n_steps=50, n_paths=5_000,
            key=jax.random.PRNGKey(100),
        )

        # Under identity correlation, per-asset terminal distributions are
        # independent GBM with identical parameters. Their means and stds
        # should agree between the multi-asset simulator and the
        # single-asset reference, within MC noise.
        single_model = BlackScholesModel(
            vol=model.vols[0], rate=model.rate, dividend=model.dividends[0],
        )
        single_paths = generate_gbm_paths(
            single_model, spots[0], T=1.0, n_steps=50, n_paths=5_000,
            key=jax.random.PRNGKey(200),
        )
        # Compare mean and std of log-terminal.
        single_stats = (
            float(jnp.mean(jnp.log(single_paths[:, -1]))),
            float(jnp.std(jnp.log(single_paths[:, -1]))),
        )
        multi_stats_asset0 = (
            float(jnp.mean(jnp.log(multi_paths[:, -1, 0]))),
            float(jnp.std(jnp.log(multi_paths[:, -1, 0]))),
        )
        multi_stats_asset1 = (
            float(jnp.mean(jnp.log(multi_paths[:, -1, 1]))),
            float(jnp.std(jnp.log(multi_paths[:, -1, 1]))),
        )
        # Generous tolerances — different seeds, same distribution.
        assert abs(multi_stats_asset0[0] - single_stats[0]) < 0.02
        assert abs(multi_stats_asset0[1] - single_stats[1]) < 0.02
        assert abs(multi_stats_asset1[0] - single_stats[0]) < 0.02
        assert abs(multi_stats_asset1[1] - single_stats[1]) < 0.02


# ─────────────────────────────────────────────────────────────────────
# JAX transformability
# ─────────────────────────────────────────────────────────────────────


class TestJaxTransforms:
    def test_jit_compiles(self):
        """Path generation JIT-compiles without errors."""
        model = MultiAssetGBMModel(
            vols=jnp.array([0.2, 0.2]),
            rate=jnp.array(0.05),
            dividends=jnp.zeros(2),
            correlation=jnp.array([[1.0, 0.5], [0.5, 1.0]]),
        )
        spots = jnp.array([100.0, 100.0])
        jit_fn = jax.jit(
            generate_correlated_gbm_paths, static_argnames=["n_steps", "n_paths"],
        )
        paths = jit_fn(model, spots, 1.0, 20, 100, jax.random.PRNGKey(0))
        assert paths.shape == (100, 21, 2)

    def test_grad_through_path_mean(self):
        """jax.grad of a scalar summary of the paths gives a finite gradient."""
        model = MultiAssetGBMModel(
            vols=jnp.array([0.2, 0.25]),
            rate=jnp.array(0.05),
            dividends=jnp.zeros(2),
            correlation=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
        )

        def summary(spots):
            paths = generate_correlated_gbm_paths(
                model, spots, T=1.0, n_steps=20, n_paths=500,
                key=jax.random.PRNGKey(11),
            )
            # Simple terminal mean across paths, summed across assets.
            return jnp.mean(paths[:, -1, :])

        grad = jax.grad(summary)(jnp.array([100.0, 100.0]))
        # Under GBM with zero dividend, d(mean S_T)/d(S_0) ≈ exp((r - q) T) > 0.
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.all(grad > 0.0)


# ─────────────────────────────────────────────────────────────────────
# Correlation validator
# ─────────────────────────────────────────────────────────────────────


class TestCorrelationValidator:
    def test_valid_matrix_passes(self):
        C = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        assert float(validate_correlation(C)) > 0.0

    def test_identity_passes(self):
        C = jnp.eye(5)
        assert float(validate_correlation(C)) >= 1.0 - 1e-6

    def test_non_symmetric_flagged(self):
        """Asymmetric matrix returns -inf."""
        C = jnp.array([[1.0, 0.5], [0.4, 1.0]])
        assert not jnp.isfinite(validate_correlation(C))

    def test_non_unit_diagonal_flagged(self):
        """Non-unit diagonal returns -inf."""
        C = jnp.array([[1.0, 0.3], [0.3, 0.9]])
        assert not jnp.isfinite(validate_correlation(C))

    def test_non_psd_has_negative_eigenvalue(self):
        """An invalid correlation matrix (off-diagonal > 1) has a negative
        minimum eigenvalue."""
        C = jnp.array([
            [1.0, 0.99, 0.99],
            [0.99, 1.0, -0.99],
            [0.99, -0.99, 1.0],
        ])
        # This matrix is symmetric, unit-diagonal, but not PSD.
        assert float(validate_correlation(C)) < 0.0
