"""Tests for SeedRegistry reproducibility and independence."""

import jax.numpy as jnp

from valax.market.synthetic import SeedRegistry


class TestSeedRegistry:
    def test_same_inputs_same_key(self):
        """Identical (master_seed, library_version, name, version) ⇒ identical key."""
        r1 = SeedRegistry(master_seed=42, library_version="0.1.0")
        r2 = SeedRegistry(master_seed=42, library_version="0.1.0")
        assert jnp.array_equal(
            r1.key("synthetic.test"), r2.key("synthetic.test")
        )

    def test_different_master_seed_different_key(self):
        r1 = SeedRegistry(master_seed=42, library_version="0.1.0")
        r2 = SeedRegistry(master_seed=43, library_version="0.1.0")
        assert not jnp.array_equal(
            r1.key("synthetic.test"), r2.key("synthetic.test")
        )

    def test_different_name_different_key(self, seed_registry):
        k1 = seed_registry.key("synthetic.stream_a")
        k2 = seed_registry.key("synthetic.stream_b")
        assert not jnp.array_equal(k1, k2)

    def test_different_version_different_key(self, seed_registry):
        k1 = seed_registry.key("synthetic.x", version=1)
        k2 = seed_registry.key("synthetic.x", version=2)
        assert not jnp.array_equal(k1, k2)

    def test_different_library_version_different_key(self):
        r1 = SeedRegistry(master_seed=42, library_version="0.1.0")
        r2 = SeedRegistry(master_seed=42, library_version="0.2.0")
        assert not jnp.array_equal(
            r1.key("synthetic.test"), r2.key("synthetic.test")
        )

    def test_snapshot_records_consumed(self, seed_registry):
        seed_registry.key("synthetic.a")
        seed_registry.key("synthetic.b", version=3)
        snap = seed_registry.snapshot()
        assert snap["consumed"] == {
            "synthetic.a": 1,
            "synthetic.b": 3,
        }
        assert snap["library_version"]
        assert snap["master_seed"]

    def test_version_zero_rejected(self, seed_registry):
        import pytest

        with pytest.raises(ValueError):
            seed_registry.key("synthetic.x", version=0)

    def test_split_helper_matches_manual(self, seed_registry):
        import jax

        manual = jax.random.split(seed_registry.key("synthetic.x"), 5)
        helper = seed_registry.split("synthetic.x", 5)
        assert jnp.array_equal(manual, helper)
