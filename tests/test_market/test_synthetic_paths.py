"""Tests for synthetic market evolution (Stage 5/6 tape generator)."""

import jax.numpy as jnp

from valax.market.data import MarketData
from valax.market.synthetic import (
    evolve_market,
    sample_market_with_correlation,
)


class TestEvolveMarket:
    def test_tape_shape_matches_dates(
        self, seed_registry, default_synth_cfg
    ):
        md, corr = sample_market_with_correlation(
            seed_registry, default_synth_cfg
        )
        dates = md.discount_curve.reference_date + jnp.array(
            [0, 30, 90, 180, 365], dtype=jnp.int32
        )
        tape = evolve_market(seed_registry, md, dates, corr)
        n_dates = int(dates.shape[0])
        n_assets = md.spots.shape[0]
        assert tape.spots.shape == (n_dates, n_assets)

    def test_initial_state_preserved(
        self, seed_registry, default_synth_cfg
    ):
        md, corr = sample_market_with_correlation(
            seed_registry, default_synth_cfg
        )
        dates = md.discount_curve.reference_date + jnp.array(
            [0, 90, 180], dtype=jnp.int32
        )
        tape = evolve_market(seed_registry, md, dates, corr)
        assert jnp.allclose(tape.spots[0], md.spots)

    def test_spots_strictly_positive(
        self, seed_registry, default_synth_cfg
    ):
        md, corr = sample_market_with_correlation(
            seed_registry, default_synth_cfg
        )
        dates = md.discount_curve.reference_date + jnp.array(
            [0, 30, 60, 120, 180, 365, 730], dtype=jnp.int32
        )
        tape = evolve_market(seed_registry, md, dates, corr)
        assert bool(jnp.all(tape.spots > 0))

    def test_returns_market_data(self, seed_registry, default_synth_cfg):
        md, corr = sample_market_with_correlation(
            seed_registry, default_synth_cfg
        )
        dates = md.discount_curve.reference_date + jnp.array(
            [0, 30, 90], dtype=jnp.int32
        )
        tape = evolve_market(seed_registry, md, dates, corr)
        assert isinstance(tape, MarketData)

    def test_n_paths_axis(self, seed_registry, default_synth_cfg):
        md, corr = sample_market_with_correlation(
            seed_registry, default_synth_cfg
        )
        dates = md.discount_curve.reference_date + jnp.array(
            [0, 90, 180], dtype=jnp.int32
        )
        tape = evolve_market(seed_registry, md, dates, corr, n_paths=4)
        # (n_paths, n_dates, n_assets)
        assert tape.spots.shape == (4, 3, default_synth_cfg.n_assets)
