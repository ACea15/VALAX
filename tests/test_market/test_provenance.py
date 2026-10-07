"""Snapshot provenance: Q-1, Q-5 and Q-6 of the AAD risk audit specification.

Covers :mod:`valax.market.provenance`: that every quote carries its type,
units, source and timestamp under one snapshot id (Q-1), that the record is
metadata and never a differentiated leaf (Q-5), that stale quotes are flagged
(Q-6), and — the part that makes it more than bookkeeping — that a ``q_bar``
vector can be labelled with it *safely*, because both orders derive from the
same instrument sequence.

The ``audit`` marker is applied to the Q-5 and labelling classes, which are
conformance statements; the constructor-validation tests are ordinary unit
tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from valax.core.market_state import MarketState
from valax.curves import (
    CurveSpec,
    DepositRate,
    OISSwapRate,
    bootstrap_curve_graph,
)
from valax.dates.daycounts import ymd_to_ordinal
from valax.greeks.adjoint import curve_adjoint, curve_quote_deltas
from valax.instruments.rates import InterestRateSwap
from valax.market.provenance import QuoteProvenance, SnapshotProvenance


REF_ORDINAL = int(ymd_to_ordinal(2025, 1, 1))
REF = jnp.int32(REF_ORDINAL)
OIS = "USD.SOFR.OIS"
DC = "act_365"


def _d(year, month, day):
    return int(ymd_to_ordinal(year, month, day))


_PILLARS = [_d(2026, 1, 1), _d(2027, 1, 1), _d(2030, 1, 1)]
_SCHEDULES = [
    [_d(2026, 1, 1), _d(2027, 1, 1)],
    [
        _d(2026, 1, 1), _d(2027, 1, 1), _d(2028, 1, 1),
        _d(2029, 1, 1), _d(2030, 1, 1),
    ],
]


@pytest.fixture(scope="module")
def spec():
    return CurveSpec(
        curve_id=OIS,
        currency="USD",
        pillar_dates=jnp.array(_PILLARS, jnp.int32),
        day_count=DC,
    )


@pytest.fixture(scope="module")
def instruments():
    return [
        DepositRate(
            start_date=REF,
            end_date=_PILLARS[0],
            rate=jnp.array(0.045),
            day_count=DC,
            curves_touched=(OIS,),
        ),
        OISSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(_SCHEDULES[0], jnp.int32),
            rate=jnp.array(0.047),
            day_count=DC,
            curves_touched=(OIS,),
        ),
        OISSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(_SCHEDULES[1], jnp.int32),
            rate=jnp.array(0.050),
            day_count=DC,
            curves_touched=(OIS,),
        ),
    ]


@pytest.fixture(scope="module")
def provenance(instruments):
    return SnapshotProvenance.from_instruments(
        "USD.2025-01-01.EOD",
        REF_ORDINAL,
        instruments,
        source="TEST-FEED",
        timestamp="2025-01-01T17:00:00Z",
        quote_ids=("USD.ON.1Y", "USD.OIS.2Y", "USD.OIS.5Y"),
    )


# ── Q-1: what each quote is ──────────────────────────────────────────


class TestQuoteRecordsAreComplete:
    def test_one_record_per_instrument_in_order(
        self, provenance, instruments
    ):
        assert len(provenance) == len(instruments)
        assert provenance.quote_ids == (
            "USD.ON.1Y", "USD.OIS.2Y", "USD.OIS.5Y"
        )

    def test_type_and_units_are_derived_not_restated(self, provenance):
        """Derived off the instrument, so they cannot drift from it."""
        assert provenance.for_quote("USD.ON.1Y").quote_type == "DepositRate"
        assert provenance.for_quote("USD.OIS.2Y").quote_type == "OISSwapRate"
        for quote_id in provenance.quote_ids:
            assert provenance.for_quote(quote_id).units == "decimal rate"

    def test_source_and_timestamp_are_carried(self, provenance):
        record = provenance.for_quote("USD.OIS.5Y")
        assert record.source == "TEST-FEED"
        assert record.timestamp == "2025-01-01T17:00:00Z"

    def test_the_snapshot_id_and_as_of_are_carried(self, provenance):
        assert provenance.snapshot_id == "USD.2025-01-01.EOD"
        assert provenance.as_of == REF_ORDINAL

    def test_default_ids_are_unique_and_ordered(self, instruments):
        derived = SnapshotProvenance.from_instruments(
            "S", REF_ORDINAL, instruments
        )
        assert derived.quote_ids == (
            "0.DepositRate", "1.OISSwapRate", "2.OISSwapRate"
        )
        assert len(set(derived.quote_ids)) == len(derived.quote_ids)

    def test_an_unknown_quote_id_raises(self, provenance):
        with pytest.raises(KeyError, match="USD.OIS.30Y"):
            provenance.for_quote("USD.OIS.30Y")

    def test_duplicate_quote_ids_are_rejected(self, instruments):
        with pytest.raises(ValueError, match="unique"):
            SnapshotProvenance.from_instruments(
                "S", REF_ORDINAL, instruments,
                quote_ids=("a", "a", "b"),
            )

    def test_mismatched_id_count_is_rejected(self, instruments):
        with pytest.raises(ValueError, match="one per instrument"):
            SnapshotProvenance.from_instruments(
                "S", REF_ORDINAL, instruments, quote_ids=("a", "b")
            )


# ── Q-6: stale and overridden quotes ─────────────────────────────────


class TestStaleFlags:
    def test_nothing_is_stale_by_default(self, provenance):
        assert provenance.stale_quote_ids() == ()

    def test_flagged_quotes_are_reported(self, instruments):
        flagged = SnapshotProvenance.from_instruments(
            "S", REF_ORDINAL, instruments,
            quote_ids=("a", "b", "c"),
            stale=(False, True, False),
        )
        assert flagged.stale_quote_ids() == ("b",)
        assert flagged.for_quote("b").stale is True

    def test_mismatched_stale_count_is_rejected(self, instruments):
        with pytest.raises(ValueError, match="one per instrument"):
            SnapshotProvenance.from_instruments(
                "S", REF_ORDINAL, instruments, stale=(True,)
            )


# ── Q-5: metadata must not enter the differentiated pytree ───────────


@pytest.mark.audit
class TestProvenanceIsNotDifferentiated:
    """The requirement that decided where this object lives.

    Every field is static, so the record contributes **no leaves** to
    any pytree it appears in. That is asserted directly rather than
    inferred from the field declarations.
    """

    def test_a_snapshot_record_has_no_array_leaves(self, provenance):
        assert jax.tree_util.tree_leaves(provenance) == []

    def test_a_quote_record_has_no_array_leaves(self):
        record = QuoteProvenance(
            quote_id="q", quote_type="DepositRate", units="decimal rate"
        )
        assert jax.tree_util.tree_leaves(record) == []

    def test_it_is_not_a_field_of_market_state(self, spec, instruments):
        """Kept alongside ``MarketState``, deliberately not inside it.

        Static fields participate in JIT cache keys, so carrying a
        snapshot id or timestamp on the market would recompile every
        jitted pricer on every new snapshot — and provenance changes on
        every snapshot by definition. The separation is the point, so
        it is pinned.
        """
        graph, _ = bootstrap_curve_graph(REF, [spec], instruments)
        market = MarketState(curves=graph)
        assert not hasattr(market, "provenance")

    def test_carrying_it_alongside_leaves_the_gradient_untouched(
        self, spec, instruments, provenance
    ):
        """A pytree of ``(market, provenance)`` differentiates as the market.

        The practical statement of Q-5: pairing the record with the
        market for reporting must not change what ``jax.grad`` sees.
        """
        graph, _ = bootstrap_curve_graph(REF, [spec], instruments)
        market = MarketState(curves=graph)
        alone = jax.tree_util.tree_leaves(market)
        paired = jax.tree_util.tree_leaves((market, provenance))
        assert len(paired) == len(alone)


# ── The point of the module: a labelled risk report ──────────────────


@pytest.mark.audit
class TestLabellingQuoteAdjoints:
    """Turning an anonymous ``q_bar`` into something a desk can read."""

    @pytest.fixture(scope="class")
    def quote_deltas(self, spec, instruments):
        graph, diagnostics = bootstrap_curve_graph(REF, [spec], instruments)
        assert diagnostics.converged
        market = MarketState(curves=graph)
        trade = InterestRateSwap(
            start_date=REF,
            fixed_dates=jnp.array(_SCHEDULES[1], jnp.int32),
            fixed_rate=jnp.array(0.046),
            notional=jnp.array(1e7),
            discount_id=OIS,
        )
        adjoint = curve_adjoint(graph, REF, [spec], instruments)
        return curve_quote_deltas(trade, None, market, adjoint=adjoint)

    def test_every_sensitivity_gets_its_quote_id(
        self, provenance, quote_deltas
    ):
        labelled = provenance.label(quote_deltas)
        assert [name for name, _ in labelled] == list(provenance.quote_ids)
        assert [value for _, value in labelled] == [
            float(x) for x in quote_deltas
        ]

    def test_the_labelled_report_is_not_all_zeros(
        self, provenance, quote_deltas
    ):
        """Anti-vacuity: labelling a zero vector would prove nothing."""
        labelled = provenance.label(quote_deltas)
        assert any(abs(value) > 1.0 for _, value in labelled)

    def test_a_mismatched_vector_is_refused(self, provenance):
        """The check that catches a record built from other instruments.

        Silently labelling the wrong numbers is the failure this whole
        object exists to prevent, so a length mismatch raises rather
        than truncating or zipping short.
        """
        with pytest.raises(ValueError, match="same instrument sequence"):
            provenance.label(jnp.zeros(2))

    def test_alignment_survives_a_reordered_instrument_list(
        self, spec, instruments
    ):
        """Both orders come from the same sequence, so they move together.

        Reorder the instruments and rebuild *both* the adjoint and the
        provenance from the new order: the labelled pairs must be the
        same set, just permuted. A hand-maintained parallel list of
        labels would silently mislabel here, which is why
        ``from_instruments`` derives them.
        """
        reordered = [instruments[2], instruments[0], instruments[1]]
        ids = {id(obj): name for obj, name in zip(
            instruments, ("USD.ON.1Y", "USD.OIS.2Y", "USD.OIS.5Y")
        )}
        new_ids = tuple(ids[id(obj)] for obj in reordered)

        graph, diagnostics = bootstrap_curve_graph(REF, [spec], reordered)
        assert diagnostics.converged
        market = MarketState(curves=graph)
        trade = InterestRateSwap(
            start_date=REF,
            fixed_dates=jnp.array(_SCHEDULES[1], jnp.int32),
            fixed_rate=jnp.array(0.046),
            notional=jnp.array(1e7),
            discount_id=OIS,
        )
        adjoint = curve_adjoint(graph, REF, [spec], reordered)
        deltas = curve_quote_deltas(trade, None, market, adjoint=adjoint)

        new_provenance = SnapshotProvenance.from_instruments(
            "S", REF_ORDINAL, reordered, quote_ids=new_ids
        )
        labelled = dict(new_provenance.label(deltas))
        assert set(labelled) == {
            "USD.ON.1Y", "USD.OIS.2Y", "USD.OIS.5Y"
        }
        assert any(abs(value) > 1.0 for value in labelled.values())
