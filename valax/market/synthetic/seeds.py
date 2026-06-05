"""Versioned, named PRNG registry for reproducible synthetic data.

Motivation
----------
Tests sprinkled with ``jax.random.PRNGKey(42)`` are reproducible only
within a single test, and silently break when:

- the call order inside a generator changes,
- a new generator is inserted before an existing one,
- the test is moved to a different file.

``SeedRegistry`` derives every stream from a single ``master_seed`` and
a stable hash of ``(name, version)``.  Renaming a stream or bumping its
version is the *only* way to change the bytes a generator sees — and
both are explicit, reviewable acts.

Usage
-----
::

    registry = SeedRegistry(master_seed=20260101, library_version="0.1.0")
    key_spots = registry.key("synthetic.snapshot.spots")
    key_vols  = registry.key("synthetic.snapshot.vols")

    # Bumping the version produces a new, independent stream:
    key_spots_v2 = registry.key("synthetic.snapshot.spots", version=2)

``registry.snapshot()`` returns a manifest describing every key that
has been consumed, suitable for embedding in golden-dataset metadata.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


def _stable_hash_u32(name: str) -> int:
    """Stable 32-bit hash of a string, independent of PYTHONHASHSEED.

    Uses SHA-256 truncated to the leading 4 bytes interpreted as a
    little-endian unsigned 32-bit integer.  Different Python processes
    and architectures produce identical values.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


@dataclass
class SeedRegistry:
    """Reproducible namespace of PRNG keys derived from a master seed.

    Attributes:
        master_seed: Root seed for the entire registry.  Two registries
            with the same ``master_seed`` and ``library_version`` produce
            identical keys for identical ``(name, version)`` pairs.
        library_version: VALAX version string.  Folded into every key
            so a library upgrade that changes numerical contracts can
            be made to produce different random streams (by bumping the
            version in code) without renaming every consumer.
        _consumed: Internal record of every ``(name, version)`` requested
            by :meth:`key`, used by :meth:`snapshot`.

    Notes:
        Key derivation:
            ``k = fold_in(fold_in(PRNGKey(master), hash(version_string)),
            hash(name))``
        where ``version_string == library_version + "::" + str(version)``.
        Both folds use the stable SHA-256-based hash, so neither Python
        nor architecture affects the resulting key.
    """

    master_seed: int
    library_version: str
    _consumed: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Cache the master key as a jax.Array once; further folds are cheap.
        self._master_key: Array = jax.random.PRNGKey(self.master_seed)

    def key(self, name: str, version: int = 1) -> Array:
        """Derive a deterministic ``jax.random`` key for a named stream.

        Args:
            name: Stable, dotted identifier (e.g.
                ``"synthetic.snapshot.spots"``).  Renaming is a
                breaking change for any golden artifact that consumed
                this key.
            version: Integer version of this stream.  Bumping it
                produces an independent key (and therefore independent
                random bytes) without renaming the stream.

        Returns:
            A ``jax.Array`` PRNG key suitable for ``jax.random.*``.
        """
        if version < 1:
            raise ValueError(f"version must be >= 1, got {version}")
        # Record the highest version seen per name (for manifest snapshots).
        prev = self._consumed.get(name, 0)
        if version > prev:
            self._consumed[name] = version

        version_tag = f"{self.library_version}::v{version}"
        version_hash = _stable_hash_u32(version_tag)
        name_hash = _stable_hash_u32(name)

        k = jax.random.fold_in(self._master_key, jnp.uint32(version_hash))
        k = jax.random.fold_in(k, jnp.uint32(name_hash))
        return k

    def split(self, name: str, n: int, version: int = 1) -> Array:
        """Convenience: ``jax.random.split(self.key(name, version), n)``."""
        return jax.random.split(self.key(name, version), n)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the registry state.

        The returned dict embeds in a golden-dataset manifest so a
        test failure can pinpoint which seed produced which artifact.
        """
        return {
            "master_seed": int(self.master_seed),
            "library_version": str(self.library_version),
            "consumed": dict(sorted(self._consumed.items())),
        }


__all__ = ["SeedRegistry", "_stable_hash_u32"]
