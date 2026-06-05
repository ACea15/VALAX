"""Shared pytest fixtures for the VALAX test suite.

Designed around the synthetic-market generator: every test that needs
random data should request a :class:`SeedRegistry`, *not* call
``jax.random.PRNGKey`` directly.  This makes every test reproducible
with a single environment variable (``VALAX_MASTER_SEED``) and the
golden-dataset harness compatible across pytest sessions.
"""

from __future__ import annotations

import os

import pytest

import valax
from valax.market.synthetic import (
    SeedRegistry,
    SyntheticMarketConfig,
    default_config,
)


@pytest.fixture(scope="session")
def master_seed() -> int:
    """Master seed for the whole test session.

    Override at the command line with ``VALAX_MASTER_SEED=...``.  The
    default ``20260101`` is the same numeric date used as the synthetic
    reference date and is recorded in every golden artifact manifest.
    """
    return int(os.environ.get("VALAX_MASTER_SEED", "20260101"))


@pytest.fixture(scope="session")
def library_version() -> str:
    """Library version string baked into every derived seed."""
    return valax.__version__


@pytest.fixture
def seed_registry(master_seed: int, library_version: str) -> SeedRegistry:
    """Per-test :class:`SeedRegistry`.

    Function-scoped so a test that consumes streams does not affect
    sibling tests.  Use ``registry.key("synthetic.<dotted>")`` to draw
    keys; rename or version-bump the stream when you intentionally
    change a generator's bytes.
    """
    return SeedRegistry(
        master_seed=master_seed, library_version=library_version
    )


@pytest.fixture
def default_synth_cfg() -> SyntheticMarketConfig:
    """Three-asset default configuration used by structural tests."""
    return default_config(n_assets=3)
