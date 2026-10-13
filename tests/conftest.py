"""Shared pytest fixtures for the VALAX test suite.

Designed around the synthetic-market generator: every test that needs
random data should request a :class:`SeedRegistry`, *not* call
``jax.random.PRNGKey`` directly.  This makes every test reproducible
with a single environment variable (``VALAX_MASTER_SEED``) and the
golden-dataset harness compatible across pytest sessions.

It also owns the ``slow`` marker policy: tests marked ``@pytest.mark.slow``
are **deselected by default** so a normal session stays fast. Include them
with ``pytest --runslow`` (or ``VALAX_RUN_SLOW=1``) — the full run required
before a release and done by the scheduled / manual CI job — or select only
them with ``pytest -m slow``. Any explicit ``-m`` expression is applied as
written (so ``pytest -m audit`` collects every audit gate, slow or not).
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


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--runslow``."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="also run tests marked @pytest.mark.slow (the full pre-release run)",
    )


def _run_slow(config: pytest.Config) -> bool:
    """Whether this session includes the ``slow`` tests."""
    if config.getoption("--runslow") or os.environ.get("VALAX_RUN_SLOW") == "1":
        return True
    # An explicit marker expression (``-m slow``, ``-m audit``, ``-m "not
    # slow"``) is the user choosing the selection; respect it as written.
    return bool(config.getoption("markexpr"))


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Deselect ``slow`` tests unless the session asked for them."""
    if _run_slow(config):
        return
    keep, drop = [], []
    for item in items:
        (drop if item.get_closest_marker("slow") else keep).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


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
