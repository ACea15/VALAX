"""Realised fixings for partially-seasoned floating legs.

A floating leg is a strip of forward rates :math:`F_i` over accrual periods
:math:`[T_{i-1}, T_i]`. Each rate is *fixed* at :math:`T_{i-1}` (the fixing
date) and *paid* at :math:`T_i`. Once the fixing date has passed, the rate
is no longer forward-looking — it is a known realised value stored in
market data, not a function of any curve.

The bootstrap and pricing layers therefore need access to a history of
realised fixings keyed by index identifier (e.g. ``USD.SOFR``,
``USD.SOFR.3M``, ``EUR.EURIBOR.6M``) and fixing date.

This module provides two pytrees:

* :class:`FixingSeries` — a sorted, immutable history for one index.
* :class:`FixingHistory` — a registry mapping index identifiers to series.

Both are :class:`equinox.Module` subclasses and JAX-pytree-compatible:
``jax.grad`` can flow through a lookup, and ``eqx.tree_at`` can replace a
single series without rebuilding the registry.

See `theory.md` §3.9 for the no-arbitrage motivation: ignoring fixings is
a first-order mis-bootstrap on instruments whose first coupon is a large
fraction of total PV.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Bool, Float, Int
from jax import Array


class FixingSeries(eqx.Module):
    """Sorted history of realised fixings for a single index.

    Attributes:
        fixing_dates: Sorted ordinal fixing dates (shape ``n``).
        fixings: Realised values at each fixing date (shape ``n``).

    The series stores observed values only — there is no implicit
    interpolation. A ``lookup`` for a date not exactly present in
    ``fixing_dates`` returns NaN, signalling to the caller that the
    rate must be projected from a forward curve instead.
    """

    fixing_dates: Int[Array, " n"]
    fixings: Float[Array, " n"]

    def lookup(
        self,
        fixing_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        """Return the realised value at ``fixing_date``, or NaN if absent.

        Uses ``jnp.searchsorted`` (binary search) for JIT-compatibility.
        Returns ``jnp.nan`` rather than raising so that JAX-traced caller
        code can use ``jnp.where(jnp.isnan(...), forward_proj, realised)``
        to fall back to forward projection without Python-level branching.
        """
        n = self.fixing_dates.shape[0]
        idx = jnp.searchsorted(self.fixing_dates, fixing_date)
        # ``idx`` is the leftmost position where ``fixing_date`` could be
        # inserted while keeping the array sorted.  An exact match exists
        # iff ``idx < n`` and ``fixing_dates[idx] == fixing_date``.
        in_range = idx < n
        safe_idx = jnp.where(in_range, idx, 0)
        matches = in_range & (self.fixing_dates[safe_idx] == fixing_date)
        return jnp.where(matches, self.fixings[safe_idx], jnp.nan)

    def has_fixing(
        self,
        fixing_date: Int[Array, ""],
    ) -> Bool[Array, ""]:
        """JIT-friendly presence check: ``True`` iff ``fixing_date`` is in
        ``fixing_dates`` exactly.
        """
        n = self.fixing_dates.shape[0]
        idx = jnp.searchsorted(self.fixing_dates, fixing_date)
        in_range = idx < n
        safe_idx = jnp.where(in_range, idx, 0)
        return in_range & (self.fixing_dates[safe_idx] == fixing_date)


class FixingHistory(eqx.Module):
    """Registry of fixing histories keyed by index identifier.

    Attributes:
        indices: Mapping from index id (e.g. ``"USD.SOFR"``) to its
            :class:`FixingSeries`.

    The dict structure is JAX-pytree native: ``jax.tree_util`` traverses
    the values and treats the string keys as static metadata.  Index
    lookup happens at Python / trace time; the numerical date search
    inside the matched series is JAX-traceable.

    Lookup semantics:

    * Unknown ``index_id`` raises :class:`KeyError` at trace time
      (configuration error — must be caught above the JIT boundary).
    * Known ``index_id`` but ``fixing_date`` not in the series returns
      ``jnp.nan`` from ``lookup`` and ``False`` from ``has_fixing``.
    """

    indices: dict

    def lookup(
        self,
        index_id: str,
        fixing_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        """Return the realised fixing for ``(index_id, fixing_date)`` or NaN."""
        series = self.indices[index_id]   # KeyError at trace time if missing
        return series.lookup(fixing_date)

    def has_fixing(
        self,
        index_id: str,
        fixing_date: Int[Array, ""],
    ) -> Bool[Array, ""]:
        """Return ``True`` iff a realised fixing exists for the (id, date) pair."""
        series = self.indices[index_id]
        return series.has_fixing(fixing_date)


def empty_fixing_history() -> FixingHistory:
    """Construct a :class:`FixingHistory` with no recorded fixings.

    Useful as a default argument for bootstrap routines that may or may
    not need fixings depending on whether any of the input instruments
    are partially seasoned.
    """
    return FixingHistory(indices={})
