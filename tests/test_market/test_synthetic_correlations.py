"""Structural sanity tests for correlation-matrix generators."""

import jax.numpy as jnp
import pytest

from valax.market.synthetic import (
    block_correlation,
    sample_correlation,
    sample_correlation_from_config,
)
from valax.market.synthetic.config import SyntheticMarketConfig
from valax.models.multi_asset import validate_correlation


def _is_valid_correlation(m, tol: float = 1e-8) -> bool:
    n = m.shape[0]
    if not jnp.allclose(m, m.T, atol=tol):
        return False
    if not jnp.allclose(jnp.diag(m), 1.0, atol=tol):
        return False
    min_eig = float(jnp.min(jnp.linalg.eigvalsh(m)))
    return min_eig > -tol


class TestSampleCorrelation:
    @pytest.mark.parametrize("n", [2, 3, 5, 10])
    def test_is_valid_correlation_matrix(self, seed_registry, n):
        c = sample_correlation(seed_registry, n)
        assert _is_valid_correlation(c)

    def test_off_diagonals_in_band(self, seed_registry):
        c = sample_correlation(
            seed_registry, n=8, min_corr=-0.2, max_corr=0.6
        )
        n = c.shape[0]
        off = c[~jnp.eye(n, dtype=bool)]
        # After PSD reprojection some entries may drift slightly,
        # but they should remain in a reasonable neighbourhood of the band.
        assert float(off.max()) <= 0.9
        assert float(off.min()) >= -0.4

    def test_identity_kind(self, seed_registry):
        c = sample_correlation(seed_registry, n=4, kind="identity")
        assert jnp.allclose(c, jnp.eye(4))

    def test_validate_correlation_passes(self, seed_registry):
        c = sample_correlation(seed_registry, n=6)
        min_eig = float(validate_correlation(c))
        assert min_eig >= -1e-8


class TestBlockCorrelation:
    def test_is_valid(self, seed_registry):
        c = block_correlation(
            seed_registry, block_sizes=(2, 3, 2), intra=0.6, inter=0.1
        )
        assert _is_valid_correlation(c)
        assert c.shape == (7, 7)


class TestSampleFromConfig:
    def test_dispatch_identity(self, seed_registry):
        cfg = SyntheticMarketConfig(n_assets=4, correlation_kind="identity")
        c = sample_correlation_from_config(seed_registry, cfg)
        assert jnp.allclose(c, jnp.eye(4))

    def test_dispatch_random(self, seed_registry):
        cfg = SyntheticMarketConfig(n_assets=4, correlation_kind="random")
        c = sample_correlation_from_config(seed_registry, cfg)
        assert _is_valid_correlation(c)

    def test_block_requires_sizes(self, seed_registry):
        cfg = SyntheticMarketConfig(n_assets=4, correlation_kind="block")
        with pytest.raises(ValueError, match="block_sizes"):
            sample_correlation_from_config(seed_registry, cfg)

    def test_block_sizes_must_sum(self, seed_registry):
        cfg = SyntheticMarketConfig(n_assets=4, correlation_kind="block")
        with pytest.raises(ValueError, match="sum"):
            sample_correlation_from_config(
                seed_registry, cfg, block_sizes=(2, 3)
            )
