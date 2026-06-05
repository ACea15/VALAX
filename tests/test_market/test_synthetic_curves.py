"""Structural sanity tests for synthetic discount curve generators."""

import jax.numpy as jnp
import pytest

from valax.dates.daycounts import ymd_to_ordinal
from valax.market.synthetic import (
    SyntheticMarketConfig,
    flat_discount_curve,
    sample_flat_curve,
    sample_nss_curve,
)


class TestFlatDiscountCurve:
    def test_df_at_reference_is_one(self):
        ref = ymd_to_ordinal(2026, 1, 1)
        curve = flat_discount_curve(rate=0.04, reference_date=ref)
        assert float(curve(curve.reference_date)) == pytest.approx(1.0)

    def test_dfs_monotone_decreasing_for_positive_rate(self):
        ref = ymd_to_ordinal(2026, 1, 1)
        curve = flat_discount_curve(rate=0.05, reference_date=ref)
        dates = ref + jnp.array([0, 30, 180, 365, 365 * 5], dtype=jnp.int32)
        dfs = curve(dates)
        assert bool(jnp.all(dfs[1:] <= dfs[:-1]))

    def test_dfs_in_unit_interval(self):
        ref = ymd_to_ordinal(2026, 1, 1)
        curve = flat_discount_curve(rate=0.05, reference_date=ref)
        assert bool(jnp.all(curve.discount_factors > 0.0))
        assert bool(jnp.all(curve.discount_factors <= 1.0))


class TestSampleFlatCurve:
    def test_reproducible(self, seed_registry, default_synth_cfg):
        cfg = SyntheticMarketConfig(curve_kind="flat", n_assets=3)
        c1 = sample_flat_curve(seed_registry, cfg)
        # A fresh registry with identical settings produces identical bytes.
        from valax.market.synthetic import SeedRegistry

        r2 = SeedRegistry(
            master_seed=seed_registry.master_seed,
            library_version=seed_registry.library_version,
        )
        c2 = sample_flat_curve(r2, cfg)
        assert jnp.allclose(c1.discount_factors, c2.discount_factors)
        assert jnp.array_equal(c1.pillar_dates, c2.pillar_dates)

    def test_float64(self, seed_registry):
        cfg = SyntheticMarketConfig(curve_kind="flat")
        c = sample_flat_curve(seed_registry, cfg)
        assert c.discount_factors.dtype == jnp.float64


class TestSampleNSSCurve:
    def test_df_at_reference_is_one(self, seed_registry, default_synth_cfg):
        c = sample_nss_curve(seed_registry, default_synth_cfg)
        assert float(c(c.reference_date)) == pytest.approx(1.0)

    def test_dfs_in_unit_interval(self, seed_registry, default_synth_cfg):
        c = sample_nss_curve(seed_registry, default_synth_cfg)
        assert bool(jnp.all(c.discount_factors > 0.0))
        assert bool(jnp.all(c.discount_factors <= 1.0))

    def test_n_pillars_matches_config(self, seed_registry, default_synth_cfg):
        c = sample_nss_curve(seed_registry, default_synth_cfg)
        # Reference date + body pillars.
        expected = 1 + len(default_synth_cfg.nss_pillars_years)
        assert c.discount_factors.shape == (expected,)

    def test_float64(self, seed_registry, default_synth_cfg):
        c = sample_nss_curve(seed_registry, default_synth_cfg)
        assert c.discount_factors.dtype == jnp.float64
