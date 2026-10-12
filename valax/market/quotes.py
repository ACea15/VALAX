"""Keyed market quotes: the inputs a calibration chain is differentiated by.

A :class:`QuoteSet` is a flat, identifier-keyed set of scalar quotes — par
rates, basis spreads, implied vols, spots, futures prices. It is the ``q`` of
the calibration chain (and, with the same container, the held-fixed inputs
``p``): :func:`valax.calibration.chain.calibrate_market` consumes one, and the
risk functions in :mod:`valax.greeks.quote_risk` return a ``QuoteSet``-shaped
sensitivity, so risk is reported against the same identifiers the quotes were
loaded under.

Values are dynamic pytree leaves (differentiable, vmappable for scenario
batches); the identifiers are dictionary keys and so part of the tree
structure. Provenance (source, timestamp, staleness) deliberately lives in
:class:`~valax.market.provenance.SnapshotProvenance` instead, so changing a
timestamp never invalidates the JIT cache.

Identifiers are free-form strings; the recommended convention is
``<CCY|UNDERLYING>.<TYPE>.<INDEX>.<TENOR>[.<QUALIFIER>]``, for example
``"USD.OIS.SOFR.5Y"``, ``"USD.IRS.TERM3M.10Y"``, ``"SPX.SPOT"`` or
``"SPX.VOL.1Y.K4000"``.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


class QuoteSet(eqx.Module):
    """Identifier-keyed scalar quotes.

    Attributes:
        values: Mapping ``quote_id -> scalar``. Every value is a dynamic
            leaf, so ``jax.grad`` with respect to a ``QuoteSet`` returns a
            ``QuoteSet`` of sensitivities.
    """

    values: dict[str, Float[Array, ""]] = eqx.field(default_factory=dict)

    @classmethod
    def from_dict(cls, mapping: Mapping[str, float | Array]) -> "QuoteSet":
        """Build a quote set from a mapping of identifiers to numbers.

        Args:
            mapping: ``quote_id -> value``.

        Returns:
            A :class:`QuoteSet` with every value cast to a float array.
        """
        return cls(
            values={
                str(k): jnp.asarray(v, dtype=jnp.float64)
                for k, v in mapping.items()
            }
        )

    @classmethod
    def from_vector(
        cls, keys: Iterable[str], vector: Float[Array, " n"]
    ) -> "QuoteSet":
        """Build a quote set from parallel keys and a vector.

        Args:
            keys: Quote identifiers, one per entry of ``vector``.
            vector: Quote values.

        Returns:
            The :class:`QuoteSet`.

        Raises:
            ValueError: If the lengths differ.
        """
        keys = tuple(keys)
        vector = jnp.asarray(vector)
        if vector.shape != (len(keys),):
            raise ValueError(
                f"QuoteSet.from_vector: {len(keys)} keys but vector has "
                f"shape {vector.shape}."
            )
        return cls(values={k: vector[i] for i, k in enumerate(keys)})

    # ── Views ────────────────────────────────────────────────────────

    def keys(self) -> tuple[str, ...]:
        """Identifiers, in sorted order (the pytree flattening order)."""
        return tuple(sorted(self.values))

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, key: str) -> Float[Array, ""]:
        if key not in self.values:
            raise KeyError(
                f"Quote {key!r} not in QuoteSet (available: {self.keys()})."
            )
        return self.values[key]

    def get(self, key: str, default: float = 0.0) -> Float[Array, ""]:
        """Value of ``key``, or ``default`` if absent.

        Args:
            key: Quote identifier.
            default: Fallback value.

        Returns:
            The quote, or ``default`` as an array.
        """
        return self.values.get(key, jnp.asarray(default))

    def vector(self, keys: Iterable[str] | None = None) -> Float[Array, " n"]:
        """Stack quotes into a vector.

        Args:
            keys: Identifiers in the desired order. Defaults to
                :meth:`keys`.

        Returns:
            The stacked values.

        Raises:
            KeyError: If a key is absent.
        """
        keys = self.keys() if keys is None else tuple(keys)
        if not keys:
            return jnp.zeros((0,))
        return jnp.stack([jnp.asarray(self[k]) for k in keys])

    def select(self, keys: Iterable[str]) -> "QuoteSet":
        """Restrict to a subset of identifiers.

        Args:
            keys: Identifiers to keep.

        Returns:
            A new :class:`QuoteSet`.

        Raises:
            KeyError: If a key is absent.
        """
        return QuoteSet(values={k: self[k] for k in keys})

    # ── Functional updates ───────────────────────────────────────────

    def merge(self, other: "QuoteSet") -> "QuoteSet":
        """Union of two quote sets; ``other`` wins on shared keys.

        Args:
            other: The quote set to merge in.

        Returns:
            A new :class:`QuoteSet`.
        """
        return QuoteSet(values={**self.values, **other.values})

    def with_values(self, mapping: Mapping[str, float | Array]) -> "QuoteSet":
        """Replace (or add) the given quotes.

        Args:
            mapping: ``quote_id -> new value``.

        Returns:
            A new :class:`QuoteSet`.
        """
        new = dict(self.values)
        for k, v in mapping.items():
            new[k] = jnp.asarray(v, dtype=jnp.float64)
        return QuoteSet(values=new)

    def bump(self, key: str, size: float | Array) -> "QuoteSet":
        """Additively bump one quote.

        Args:
            key: Quote identifier.
            size: Additive bump.

        Returns:
            A new :class:`QuoteSet`.

        Raises:
            KeyError: If ``key`` is absent.
        """
        return self.with_values({key: self[key] + size})

    def zeros_like(self) -> "QuoteSet":
        """A quote set with the same identifiers and every value zero."""
        return QuoteSet(values={k: jnp.zeros_like(v) for k, v in self.values.items()})

    def __add__(self, other: "QuoteSet") -> "QuoteSet":
        """Key-wise sum; a key missing on one side counts as zero."""
        keys = set(self.values) | set(other.values)
        return QuoteSet(
            values={
                k: jnp.asarray(self.get(k)) + jnp.asarray(other.get(k))
                for k in keys
            }
        )

    def scale(self, factor: float | Array) -> "QuoteSet":
        """Multiply every value by ``factor``.

        Args:
            factor: Scalar multiplier.

        Returns:
            A new :class:`QuoteSet`.
        """
        return QuoteSet(values=jax.tree.map(lambda v: v * factor, self.values))


__all__ = ["QuoteSet"]
