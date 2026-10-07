r"""Snapshot provenance: what each quote was, where it came from, and when.

A quote vector on its own is anonymous. ``q̄_c[7] = -41.3`` is not a risk
report until something says that entry 7 was a 5Y OIS par rate in decimal,
from a named source, at a named time, in a named snapshot. Specification Q-1
makes that a MUST, and Q-5 extends it to the fixed inputs ``p``: they are
versioned like quotes and reported separately.

Deliberately *alongside* ``MarketState``, not inside it
-------------------------------------------------------

Every field here is a string or an integer — identifiers, sources,
timestamps. None of it is differentiable, and none of it belongs in a pytree
that :func:`jax.grad` traverses.

It could be carried as ``equinox`` static fields on
:class:`~valax.core.market_state.MarketState` without being differentiated,
but that would be a mistake of a different kind: static fields participate in
JIT cache keys, so every new snapshot id or timestamp would be a fresh
compilation of every jitted pricer. Provenance changes on every snapshot by
definition. Keeping it a sibling object leaves the differentiated pytree and
the JIT cache alike untouched.

Aligned by construction
-----------------------

:meth:`SnapshotProvenance.from_instruments` derives each quote's type and
units from the bootstrap instrument itself, in the same order the solver sees
them. That is what lets :meth:`SnapshotProvenance.label` pair a ``q̄`` vector
with its names and be right: both orders come from the same sequence, so a
reordered quote list cannot silently mislabel a risk report. Hand-maintaining
a parallel list of labels is the failure this avoids.
"""

from __future__ import annotations

from typing import Any, Sequence

import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.curves.bootstrap_graph import _QUOTE_FIELD


#: Units implied by each primary quote field.  Derived from the field name
#: rather than restated per instrument, so a new quote type picks up the
#: right units by naming its field consistently.
_FIELD_UNITS: dict[str, str] = {
    "rate": "decimal rate",
    "spread": "decimal spread",
    "futures_rate": "decimal rate (100 - price)",
    "quoted_forward": "decimal forward rate",
    "far_rate": "decimal outright",
    "jump_size": "decimal rate over the turn window",
}


class QuoteProvenance(eqx.Module):
    """Provenance of one element of ``q``.

    All fields are static: this is metadata, never a differentiated leaf.

    Attributes:
        quote_id: Stable identifier for the quote, unique within the
            snapshot. Used as the report label.
        quote_type: The instrument class the quote belongs to, e.g.
            ``"OISSwapRate"``.
        units: Units of the quote value, e.g. ``"decimal rate"``.
        source: Where the number came from — a feed, a broker, a desk
            override.
        timestamp: Observation time, as an ISO-8601 string. A string
            rather than a ``datetime`` so the object stays hashable and
            free of the Python date types VALAX keeps out of traced code.
        stale: Flag for a quote known to be stale or overridden
            (specification Q-6). Flagging is as far as a generic
            container can go; *excluding or down-weighting* is a
            calibration policy, so it stays with the caller who builds
            the instrument list.
    """

    quote_id: str = eqx.field(static=True)
    quote_type: str = eqx.field(static=True)
    units: str = eqx.field(static=True, default="")
    source: str = eqx.field(static=True, default="")
    timestamp: str = eqx.field(static=True, default="")
    stale: bool = eqx.field(static=True, default=False)


class SnapshotProvenance(eqx.Module):
    """Provenance of a whole market snapshot.

    Attributes:
        snapshot_id: Identifier of the snapshot every quote belongs to.
        as_of: Valuation date as an integer day ordinal, matching
            ``CurveSpec``/``DiscountCurve.reference_date``.
        quotes: Per-quote provenance, **in the order the calibration
            instruments were supplied**, so it aligns with ``q̄``.
    """

    snapshot_id: str = eqx.field(static=True)
    as_of: int = eqx.field(static=True)
    quotes: tuple[QuoteProvenance, ...] = eqx.field(static=True, default=())

    def __check_init__(self) -> None:
        ids = [quote.quote_id for quote in self.quotes]
        duplicates = {qid for qid in ids if ids.count(qid) > 1}
        if duplicates:
            raise ValueError(
                f"SnapshotProvenance quote_ids must be unique within a "
                f"snapshot (Q-1); duplicated: {sorted(duplicates)}."
            )

    def __len__(self) -> int:
        return len(self.quotes)

    @property
    def quote_ids(self) -> tuple[str, ...]:
        """Quote identifiers, in vector order."""
        return tuple(quote.quote_id for quote in self.quotes)

    def for_quote(self, quote_id: str) -> QuoteProvenance:
        """Look up one quote's provenance.

        Args:
            quote_id: The identifier to find.

        Returns:
            The matching :class:`QuoteProvenance`.

        Raises:
            KeyError: If no quote carries that identifier.
        """
        for quote in self.quotes:
            if quote.quote_id == quote_id:
                return quote
        raise KeyError(
            f"No quote {quote_id!r} in snapshot {self.snapshot_id!r}; "
            f"available: {list(self.quote_ids)}."
        )

    def stale_quote_ids(self) -> tuple[str, ...]:
        """Identifiers of every quote flagged stale or overridden (Q-6)."""
        return tuple(
            quote.quote_id for quote in self.quotes if quote.stale
        )

    def label(
        self, quote_adjoints: Float[Array, " n_quotes"]
    ) -> tuple[tuple[str, float], ...]:
        """Pair a ``q̄`` vector with its quote identifiers.

        The point of the whole module: turns an anonymous sensitivity
        vector into a report. The alignment is safe because both orders
        derive from the same instrument sequence — see
        :meth:`from_instruments`.

        Args:
            quote_adjoints: Quote sensitivities from
                :func:`~valax.greeks.adjoint.curve_quote_deltas`, in
                instrument order.

        Returns:
            ``(quote_id, adjoint)`` pairs, in the same order.

        Raises:
            ValueError: If the vector length does not match the number of
                quotes. This is the check that catches a provenance
                record built against a different instrument list — the
                mislabelling it exists to prevent.
        """
        if len(quote_adjoints) != len(self.quotes):
            raise ValueError(
                f"Cannot label {len(quote_adjoints)} quote adjoints with "
                f"{len(self.quotes)} provenance records in snapshot "
                f"{self.snapshot_id!r}. The provenance must be built from "
                f"the same instrument sequence the adjoint was."
            )
        return tuple(
            (quote.quote_id, float(adjoint))
            for quote, adjoint in zip(self.quotes, quote_adjoints)
        )

    @classmethod
    def from_instruments(
        cls,
        snapshot_id: str,
        as_of: int,
        instruments: Sequence[Any],
        *,
        source: str = "",
        timestamp: str = "",
        quote_ids: Sequence[str] | None = None,
        stale: Sequence[bool] | None = None,
    ) -> "SnapshotProvenance":
        """Derive provenance from the calibration instruments themselves.

        Type and units come off each instrument rather than being
        restated, and the order is the instruments' own, so the result
        aligns with ``q̄`` by construction.

        Default identifiers are ``"<index>.<ClassName>"``, which is
        unique and ordered but says nothing about tenor; pass
        ``quote_ids`` for labels a desk would recognise.

        Args:
            snapshot_id: Identifier for this snapshot.
            as_of: Valuation date as an integer day ordinal.
            instruments: The bootstrap quote pytrees, in solver order.
            source: Common source for every quote.
            timestamp: Common ISO-8601 observation time.
            quote_ids: Optional explicit identifiers, one per
                instrument.
            stale: Optional per-quote stale flags (Q-6).

        Returns:
            The assembled :class:`SnapshotProvenance`.

        Raises:
            ValueError: If ``quote_ids`` or ``stale`` is supplied with a
                length other than ``len(instruments)``.
        """
        count = len(instruments)
        if quote_ids is not None and len(quote_ids) != count:
            raise ValueError(
                f"quote_ids has length {len(quote_ids)} but there are "
                f"{count} instruments; supply one per instrument."
            )
        if stale is not None and len(stale) != count:
            raise ValueError(
                f"stale has length {len(stale)} but there are {count} "
                f"instruments; supply one per instrument."
            )

        records = []
        for index, instrument in enumerate(instruments):
            type_name = type(instrument).__name__
            field = _QUOTE_FIELD.get(type_name)
            records.append(
                QuoteProvenance(
                    quote_id=(
                        quote_ids[index]
                        if quote_ids is not None
                        else f"{index}.{type_name}"
                    ),
                    quote_type=type_name,
                    units=_FIELD_UNITS.get(field, "") if field else "",
                    source=source,
                    timestamp=timestamp,
                    stale=bool(stale[index]) if stale is not None else False,
                )
            )
        return cls(
            snapshot_id=snapshot_id,
            as_of=int(as_of),
            quotes=tuple(records),
        )
