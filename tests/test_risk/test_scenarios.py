"""Tests for scenario generation."""

import jax
import jax.numpy as jnp
import pytest

from valax.market.scenario import MarketScenario, ScenarioSet, stack_scenarios
from valax.risk.scenarios import (
    butterfly,
    flattener,
    historical_scenarios,
    parametric_scenarios,
    steepener,
    stress_scenario,
)


N_ASSETS = 2
N_PILLARS = 3


class TestParametricScenarios:
    def test_shape(self):
        key = jax.random.PRNGKey(0)
        n_factors = 2 * N_ASSETS + N_PILLARS + N_ASSETS  # spots + vols + rates + divs
        cov = jnp.eye(n_factors) * 0.01
        ss = parametric_scenarios(key, cov, 1000, N_ASSETS, N_PILLARS)
        assert ss.spot_shocks.shape == (1000, N_ASSETS)
        assert ss.vol_shocks.shape == (1000, N_ASSETS)
        assert ss.rate_shocks.shape == (1000, N_PILLARS)
        assert ss.dividend_shocks.shape == (1000, N_ASSETS)

    def test_normal_covariance_structure(self):
        """Sample covariance should approximate input covariance."""
        key = jax.random.PRNGKey(42)
        n_factors = 2 * N_ASSETS + N_PILLARS + N_ASSETS
        # Diagonal covariance with known variances
        variances = jnp.arange(1, n_factors + 1, dtype=jnp.float64) * 0.001
        cov = jnp.diag(variances)
        ss = parametric_scenarios(key, cov, 50_000, N_ASSETS, N_PILLARS)
        # Reconstruct full sample matrix
        all_samples = jnp.concatenate([
            ss.spot_shocks, ss.vol_shocks, ss.rate_shocks, ss.dividend_shocks
        ], axis=1)
        sample_var = jnp.var(all_samples, axis=0)
        # Within 10% relative error for 50k samples
        assert jnp.allclose(sample_var, variances, rtol=0.1)

    def test_t_distribution_heavier_tails(self):
        """t-distributed scenarios should have heavier tails than normal."""
        key = jax.random.PRNGKey(7)
        n_factors = 2 * N_ASSETS + N_PILLARS + N_ASSETS
        cov = jnp.eye(n_factors) * 0.01
        ss_normal = parametric_scenarios(key, cov, 50_000, N_ASSETS, N_PILLARS)
        ss_t = parametric_scenarios(
            key, cov, 50_000, N_ASSETS, N_PILLARS, distribution="t", df=4.0
        )
        # Kurtosis of t(4) = 3 + 6/(4-4) -> infinite, but in practice
        # sample kurtosis should be much higher than normal's ~3
        normal_kurt = _sample_kurtosis(ss_normal.spot_shocks[:, 0])
        t_kurt = _sample_kurtosis(ss_t.spot_shocks[:, 0])
        assert t_kurt > normal_kurt


class TestHistoricalScenarios:
    def test_round_trip(self):
        """Historical scenarios should exactly reproduce input data."""
        n_factors = 2 * N_ASSETS + N_PILLARS + N_ASSETS
        n_obs = 3
        data = jnp.arange(n_obs * n_factors, dtype=jnp.float64).reshape(n_obs, n_factors)
        ss = historical_scenarios(data, N_ASSETS, N_PILLARS)
        # Reconstruct and compare
        reconstructed = jnp.concatenate([
            ss.spot_shocks, ss.vol_shocks, ss.rate_shocks, ss.dividend_shocks
        ], axis=1)
        assert jnp.allclose(reconstructed, data)


class TestStressScenarios:
    def test_stress_parallel_shift(self):
        s = stress_scenario(N_ASSETS, N_PILLARS, parallel_rate_shift=0.01)
        assert jnp.allclose(s.rate_shocks, jnp.full(N_PILLARS, 0.01))
        assert jnp.allclose(s.spot_shocks, jnp.zeros(N_ASSETS))

    def test_steepener_monotone(self):
        s = steepener(N_ASSETS, N_PILLARS, short_bump=-0.005, long_bump=0.01)
        # Rate shocks should be monotonically increasing
        diffs = jnp.diff(s.rate_shocks)
        assert jnp.all(diffs >= 0)

    def test_flattener_profile(self):
        s = flattener(N_ASSETS, N_PILLARS, short_bump=0.01, long_bump=-0.005)
        assert jnp.isclose(s.rate_shocks[0], 0.01, atol=1e-8)
        assert jnp.isclose(s.rate_shocks[-1], -0.005, atol=1e-8)

    def test_butterfly_endpoints_and_midpoint(self):
        s = butterfly(N_ASSETS, 5, wing_bump=0.01, belly_bump=-0.005)
        # Endpoints should be wing_bump
        assert jnp.isclose(s.rate_shocks[0], 0.01, atol=1e-8)
        assert jnp.isclose(s.rate_shocks[-1], 0.01, atol=1e-8)
        # Midpoint should be belly_bump
        assert jnp.isclose(s.rate_shocks[2], -0.005, atol=1e-8)


class TestStackScenarios:
    def test_stack(self):
        s1 = stress_scenario(N_ASSETS, N_PILLARS, spot_shock=1.0)
        s2 = stress_scenario(N_ASSETS, N_PILLARS, spot_shock=2.0)
        ss = stack_scenarios([s1, s2])
        assert ss.spot_shocks.shape == (2, N_ASSETS)
        assert jnp.isclose(ss.spot_shocks[0, 0], 1.0)
        assert jnp.isclose(ss.spot_shocks[1, 0], 2.0)


def _sample_kurtosis(x):
    """Excess kurtosis of a 1-d array."""
    m = jnp.mean(x)
    s = jnp.std(x)
    return jnp.mean(((x - m) / s) ** 4) - 3.0
