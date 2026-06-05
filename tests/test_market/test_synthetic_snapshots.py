"""Structural sanity tests for MarketData snapshot generators."""

import jax.numpy as jnp

from valax.market.data import MarketData
from valax.market.synthetic import (
    sample_market_data,
    sample_market_with_correlation,
    sample_scalar_market,
)


class TestSampleScalarMarket:
    def test_keys_present(self, seed_registry, default_synth_cfg):
        out = sample_scalar_market(seed_registry, default_synth_cfg)
        assert set(out) == {
            "spot", "vol", "rate", "dividend", "expiry", "strike",
        }

    def test_all_float64_scalars(self, seed_registry, default_synth_cfg):
        out = sample_scalar_market(seed_registry, default_synth_cfg)
        for k, v in out.items():
            assert v.shape == (), f"{k} is not scalar"
            assert v.dtype == jnp.float64, f"{k} is not float64"

    def test_positivity(self, seed_registry, default_synth_cfg):
        out = sample_scalar_market(seed_registry, default_synth_cfg)
        assert float(out["spot"]) > 0
        assert float(out["vol"]) > 0
        assert float(out["expiry"]) > 0
        assert float(out["strike"]) > 0


class TestSampleMarketData:
    def test_returns_market_data(self, seed_registry, default_synth_cfg):
        md = sample_market_data(seed_registry, default_synth_cfg)
        assert isinstance(md, MarketData)

    def test_shapes(self, seed_registry, default_synth_cfg):
        md = sample_market_data(seed_registry, default_synth_cfg)
        n = default_synth_cfg.n_assets
        assert md.spots.shape == (n,)
        assert md.vols.shape == (n,)
        assert md.dividends.shape == (n,)

    def test_positivity(self, seed_registry, default_synth_cfg):
        md = sample_market_data(seed_registry, default_synth_cfg)
        assert bool(jnp.all(md.spots > 0))
        assert bool(jnp.all(md.vols > 0))
        assert bool(jnp.all(md.dividends >= 0))

    def test_float64(self, seed_registry, default_synth_cfg):
        md = sample_market_data(seed_registry, default_synth_cfg)
        assert md.spots.dtype == jnp.float64
        assert md.vols.dtype == jnp.float64
        assert md.dividends.dtype == jnp.float64


class TestSampleMarketWithCorrelation:
    def test_correlation_shape_matches_n_assets(
        self, seed_registry, default_synth_cfg
    ):
        md, corr = sample_market_with_correlation(
            seed_registry, default_synth_cfg
        )
        n = default_synth_cfg.n_assets
        assert corr.shape == (n, n)

    def test_correlation_valid(self, seed_registry, default_synth_cfg):
        _, corr = sample_market_with_correlation(
            seed_registry, default_synth_cfg
        )
        assert bool(jnp.allclose(jnp.diag(corr), 1.0, atol=1e-8))
        min_eig = float(jnp.min(jnp.linalg.eigvalsh(corr)))
        assert min_eig > -1e-8
