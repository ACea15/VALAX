"""Tests for FixingSeries and FixingHistory."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.fixings import (
    FixingHistory,
    FixingSeries,
    empty_fixing_history,
)
from valax.dates.daycounts import ymd_to_ordinal


# ── Helpers ──────────────────────────────────────────────────────────


@pytest.fixture
def sofr_series():
    """Two months of business-daily SOFR fixings (synthetic)."""
    dates = jnp.array(
        [
            int(ymd_to_ordinal(2025, 1, 2)),
            int(ymd_to_ordinal(2025, 1, 6)),
            int(ymd_to_ordinal(2025, 1, 13)),
            int(ymd_to_ordinal(2025, 1, 21)),
            int(ymd_to_ordinal(2025, 2, 3)),
            int(ymd_to_ordinal(2025, 2, 18)),
        ],
        dtype=jnp.int32,
    )
    rates = jnp.array(
        [0.0438, 0.0436, 0.0434, 0.0435, 0.0432, 0.0431], dtype=jnp.float64
    )
    return FixingSeries(fixing_dates=dates, fixings=rates)


@pytest.fixture
def history(sofr_series):
    """A FixingHistory containing one synthetic SOFR series."""
    euribor_dates = jnp.array(
        [
            int(ymd_to_ordinal(2025, 1, 2)),
            int(ymd_to_ordinal(2025, 2, 3)),
        ],
        dtype=jnp.int32,
    )
    euribor_rates = jnp.array([0.0290, 0.0285], dtype=jnp.float64)
    euribor_series = FixingSeries(
        fixing_dates=euribor_dates, fixings=euribor_rates
    )
    return FixingHistory(
        indices={
            "USD.SOFR": sofr_series,
            "EUR.EURIBOR.6M": euribor_series,
        }
    )


# ── FixingSeries: lookup correctness ─────────────────────────────────


class TestFixingSeriesLookup:
    def test_exact_match_first(self, sofr_series):
        d = jnp.asarray(int(ymd_to_ordinal(2025, 1, 2)), dtype=jnp.int32)
        assert float(sofr_series.lookup(d)) == pytest.approx(0.0438, abs=1e-12)

    def test_exact_match_middle(self, sofr_series):
        d = jnp.asarray(int(ymd_to_ordinal(2025, 1, 21)), dtype=jnp.int32)
        assert float(sofr_series.lookup(d)) == pytest.approx(0.0435, abs=1e-12)

    def test_exact_match_last(self, sofr_series):
        d = jnp.asarray(int(ymd_to_ordinal(2025, 2, 18)), dtype=jnp.int32)
        assert float(sofr_series.lookup(d)) == pytest.approx(0.0431, abs=1e-12)

    def test_before_first_returns_nan(self, sofr_series):
        d = jnp.asarray(int(ymd_to_ordinal(2024, 12, 30)), dtype=jnp.int32)
        assert jnp.isnan(sofr_series.lookup(d))

    def test_after_last_returns_nan(self, sofr_series):
        d = jnp.asarray(int(ymd_to_ordinal(2025, 3, 1)), dtype=jnp.int32)
        assert jnp.isnan(sofr_series.lookup(d))

    def test_in_range_no_match_returns_nan(self, sofr_series):
        # 2025-01-15 is between two recorded fixings but not present.
        d = jnp.asarray(int(ymd_to_ordinal(2025, 1, 15)), dtype=jnp.int32)
        assert jnp.isnan(sofr_series.lookup(d))


class TestFixingSeriesLookupMany:
    def test_all_present(self, sofr_series):
        dates = jnp.array(
            [
                int(ymd_to_ordinal(2025, 1, 6)),
                int(ymd_to_ordinal(2025, 1, 21)),
                int(ymd_to_ordinal(2025, 2, 18)),
            ],
            dtype=jnp.int32,
        )
        rates = sofr_series.lookup_many(dates)
        expected = jnp.array([0.0436, 0.0435, 0.0431])
        assert jnp.allclose(rates, expected, atol=1e-12)

    def test_mix_of_present_and_absent_returns_nan_for_absent(
        self, sofr_series
    ):
        dates = jnp.array(
            [
                int(ymd_to_ordinal(2025, 1, 6)),    # present
                int(ymd_to_ordinal(2025, 1, 15)),   # absent in range
                int(ymd_to_ordinal(2025, 2, 18)),   # present
                int(ymd_to_ordinal(2030, 1, 1)),    # absent after
            ],
            dtype=jnp.int32,
        )
        rates = sofr_series.lookup_many(dates)
        assert float(rates[0]) == pytest.approx(0.0436)
        assert jnp.isnan(rates[1])
        assert float(rates[2]) == pytest.approx(0.0431)
        assert jnp.isnan(rates[3])

    def test_jit_compatible(self, sofr_series):
        @jax.jit
        def f(dates):
            return sofr_series.lookup_many(dates)

        dates = jnp.array(
            [
                int(ymd_to_ordinal(2025, 1, 6)),
                int(ymd_to_ordinal(2025, 1, 13)),
            ],
            dtype=jnp.int32,
        )
        rates = f(dates)
        assert float(rates[0]) == pytest.approx(0.0436)
        assert float(rates[1]) == pytest.approx(0.0434)


class TestFixingHistoryLookupMany:
    def test_known_index(self, history):
        dates = jnp.array(
            [
                int(ymd_to_ordinal(2025, 1, 6)),
                int(ymd_to_ordinal(2025, 2, 18)),
            ],
            dtype=jnp.int32,
        )
        rates = history.lookup_many("USD.SOFR", dates)
        expected = jnp.array([0.0436, 0.0431])
        assert jnp.allclose(rates, expected, atol=1e-12)

    def test_unknown_index_raises(self, history):
        dates = jnp.array(
            [int(ymd_to_ordinal(2025, 1, 6))], dtype=jnp.int32
        )
        with pytest.raises(KeyError):
            history.lookup_many("GBP.SONIA", dates)


class TestFixingSeriesHasFixing:
    def test_present(self, sofr_series):
        d = jnp.asarray(int(ymd_to_ordinal(2025, 1, 13)), dtype=jnp.int32)
        assert bool(sofr_series.has_fixing(d))

    def test_absent_in_range(self, sofr_series):
        d = jnp.asarray(int(ymd_to_ordinal(2025, 1, 15)), dtype=jnp.int32)
        assert not bool(sofr_series.has_fixing(d))

    def test_absent_before(self, sofr_series):
        d = jnp.asarray(int(ymd_to_ordinal(2024, 1, 1)), dtype=jnp.int32)
        assert not bool(sofr_series.has_fixing(d))

    def test_absent_after(self, sofr_series):
        d = jnp.asarray(int(ymd_to_ordinal(2030, 1, 1)), dtype=jnp.int32)
        assert not bool(sofr_series.has_fixing(d))


# ── FixingSeries: pytree / JIT properties ────────────────────────────


class TestFixingSeriesPytree:
    def test_is_pytree(self, sofr_series):
        leaves = jax.tree_util.tree_leaves(sofr_series)
        # Two array leaves: fixing_dates and fixings.
        assert len(leaves) == 2

    def test_tree_at_replaces_fixings(self, sofr_series):
        new_fixings = jnp.zeros_like(sofr_series.fixings)
        bumped = eqx.tree_at(lambda s: s.fixings, sofr_series, new_fixings)
        assert jnp.all(bumped.fixings == 0.0)
        # Original is unchanged (frozen module).
        assert float(sofr_series.fixings[0]) == pytest.approx(0.0438)

    def test_jit_lookup(self, sofr_series):
        @jax.jit
        def f(date):
            return sofr_series.lookup(date)

        d = jnp.asarray(int(ymd_to_ordinal(2025, 1, 13)), dtype=jnp.int32)
        assert float(f(d)) == pytest.approx(0.0434, abs=1e-12)

    def test_jit_has_fixing(self, sofr_series):
        @jax.jit
        def f(date):
            return sofr_series.has_fixing(date)

        d_present = jnp.asarray(int(ymd_to_ordinal(2025, 1, 13)), dtype=jnp.int32)
        d_absent = jnp.asarray(int(ymd_to_ordinal(2025, 1, 15)), dtype=jnp.int32)
        assert bool(f(d_present))
        assert not bool(f(d_absent))


# ── FixingHistory: registry semantics ────────────────────────────────


class TestFixingHistoryLookup:
    def test_lookup_known_index(self, history):
        d = jnp.asarray(int(ymd_to_ordinal(2025, 1, 13)), dtype=jnp.int32)
        assert float(history.lookup("USD.SOFR", d)) == pytest.approx(
            0.0434, abs=1e-12
        )

    def test_lookup_other_index(self, history):
        d = jnp.asarray(int(ymd_to_ordinal(2025, 2, 3)), dtype=jnp.int32)
        assert float(history.lookup("EUR.EURIBOR.6M", d)) == pytest.approx(
            0.0285, abs=1e-12
        )

    def test_lookup_unknown_index_raises(self, history):
        d = jnp.asarray(int(ymd_to_ordinal(2025, 1, 13)), dtype=jnp.int32)
        with pytest.raises(KeyError):
            history.lookup("GBP.SONIA", d)

    def test_lookup_known_index_unknown_date_returns_nan(self, history):
        d = jnp.asarray(int(ymd_to_ordinal(2025, 1, 15)), dtype=jnp.int32)
        assert jnp.isnan(history.lookup("USD.SOFR", d))

    def test_has_fixing_true(self, history):
        d = jnp.asarray(int(ymd_to_ordinal(2025, 2, 18)), dtype=jnp.int32)
        assert bool(history.has_fixing("USD.SOFR", d))

    def test_has_fixing_false(self, history):
        d = jnp.asarray(int(ymd_to_ordinal(2030, 1, 1)), dtype=jnp.int32)
        assert not bool(history.has_fixing("USD.SOFR", d))


class TestFixingHistoryPytree:
    def test_is_pytree(self, history):
        leaves = jax.tree_util.tree_leaves(history)
        # 4 array leaves: 2 series × (dates + values).
        assert len(leaves) == 4

    def test_tree_at_replaces_series(self, history):
        # Bump every SOFR fixing by 10 bps.
        old = history.indices["USD.SOFR"]
        new = eqx.tree_at(lambda s: s.fixings, old, old.fixings + 0.0010)
        bumped_history = eqx.tree_at(
            lambda h: h.indices["USD.SOFR"], history, new
        )
        d = jnp.asarray(int(ymd_to_ordinal(2025, 1, 13)), dtype=jnp.int32)
        assert float(bumped_history.lookup("USD.SOFR", d)) == pytest.approx(
            0.0444, abs=1e-12
        )
        # Other index unaffected.
        d_eur = jnp.asarray(int(ymd_to_ordinal(2025, 2, 3)), dtype=jnp.int32)
        assert float(
            bumped_history.lookup("EUR.EURIBOR.6M", d_eur)
        ) == pytest.approx(0.0285, abs=1e-12)

    def test_jit_lookup(self, history):
        # ``index_id`` is a Python str — captured as a closure so the
        # dict lookup happens at trace time.
        @jax.jit
        def f(date):
            return history.lookup("USD.SOFR", date)

        d = jnp.asarray(int(ymd_to_ordinal(2025, 2, 3)), dtype=jnp.int32)
        assert float(f(d)) == pytest.approx(0.0432, abs=1e-12)


# ── Constructor helper ───────────────────────────────────────────────


class TestEmptyFixingHistory:
    def test_construct(self):
        h = empty_fixing_history()
        assert isinstance(h, FixingHistory)
        assert len(h.indices) == 0

    def test_pytree_with_no_leaves(self):
        h = empty_fixing_history()
        assert jax.tree_util.tree_leaves(h) == []

    def test_lookup_any_index_raises(self):
        h = empty_fixing_history()
        d = jnp.asarray(int(ymd_to_ordinal(2025, 1, 1)), dtype=jnp.int32)
        with pytest.raises(KeyError):
            h.lookup("USD.SOFR", d)


# ── Realised compounding (spec C-3) ──────────────────────────────────


class TestCompoundedFactor:
    """``FixingSeries.compounded_factor`` reduces a daily history to the
    realised compounding factor over a window.

    This is the market-data half of a partly-accrued RFR coupon: no
    curve enters it, which is what makes a coupon's realised portion
    carry no curve sensitivity by construction.
    """

    def test_matches_the_closed_form_on_a_constant_history(self):
        """``N`` days at a constant rate ``r`` gives ``(1 + r·δ)^N``."""
        start = int(ymd_to_ordinal(2025, 1, 1))
        n, rate = 90, 0.04
        series = FixingSeries(
            fixing_dates=jnp.arange(start, start + n + 1),
            fixings=jnp.full(n + 1, rate),
        )
        got = float(
            series.compounded_factor(
                jnp.asarray(start), jnp.asarray(start + n), "act_360"
            )
        )
        assert got == pytest.approx(
            (1.0 + rate / 360.0) ** n, rel=1e-13
        )

    def test_empty_window_is_exactly_one(self):
        """Nothing accrued means a factor of exactly 1.0, so the blend
        that consumes it degenerates to pure projection."""
        start = int(ymd_to_ordinal(2025, 1, 1))
        series = FixingSeries(
            fixing_dates=jnp.arange(start, start + 10),
            fixings=jnp.full(10, 0.04),
        )
        assert (
            float(
                series.compounded_factor(
                    jnp.asarray(start), jnp.asarray(start)
                )
            )
            == 1.0
        )

    def test_empty_series_is_exactly_one(self):
        series = FixingSeries(
            fixing_dates=jnp.array([], dtype=jnp.int32),
            fixings=jnp.array([]),
        )
        assert (
            float(
                series.compounded_factor(jnp.asarray(0), jnp.asarray(90))
            )
            == 1.0
        )

    def test_a_fixing_dated_at_the_window_end_is_excluded(self):
        """The overnight rate set today is earned over [today, tomorrow],
        which has not happened yet."""
        start = int(ymd_to_ordinal(2025, 1, 1))
        series = FixingSeries(
            fixing_dates=jnp.arange(start, start + 5),
            fixings=jnp.full(5, 0.04),
        )
        three = float(
            series.compounded_factor(
                jnp.asarray(start), jnp.asarray(start + 3), "act_360"
            )
        )
        assert three == pytest.approx(
            (1.0 + 0.04 / 360.0) ** 3, rel=1e-13
        )

    def test_a_gap_carries_the_previous_rate_across_it(self):
        """Weekend convention, for free: a series with no entry for a
        non-business day leaves the preceding rate applying across the
        gap, weighted by calendar days.  Here one entry covering three
        days must equal three single-day entries at the same rate."""
        start = int(ymd_to_ordinal(2025, 1, 1))
        rate = 0.04
        gapped = FixingSeries(
            fixing_dates=jnp.asarray([start]),
            fixings=jnp.asarray([rate]),
        )
        daily = FixingSeries(
            fixing_dates=jnp.arange(start, start + 3),
            fixings=jnp.full(3, rate),
        )
        end = jnp.asarray(start + 3)
        gapped_factor = float(
            gapped.compounded_factor(jnp.asarray(start), end, "act_360")
        )
        daily_factor = float(
            daily.compounded_factor(jnp.asarray(start), end, "act_360")
        )
        # One 3-day accrual is simple; three 1-day accruals compound.
        assert gapped_factor == pytest.approx(
            1.0 + rate * 3.0 / 360.0, rel=1e-13
        )
        assert daily_factor > gapped_factor
        assert daily_factor == pytest.approx(
            (1.0 + rate / 360.0) ** 3, rel=1e-13
        )

    def test_history_dispatch_tolerates_an_unknown_index(self):
        """Unlike ``lookup``, which raises: a missing *history* means
        nothing has accrued, whereas a missing *fixing* for a coupon
        that needs one is a configuration error."""
        assert (
            float(
                empty_fixing_history().compounded_factor(
                    "NO.SUCH.INDEX", jnp.asarray(0), jnp.asarray(90)
                )
            )
            == 1.0
        )

    def test_history_dispatch_matches_the_series(self, history):
        start = int(ymd_to_ordinal(2025, 1, 1))
        end = int(ymd_to_ordinal(2025, 2, 1))
        assert float(
            history.compounded_factor(
                "USD.SOFR", jnp.asarray(start), jnp.asarray(end)
            )
        ) == float(
            history.indices["USD.SOFR"].compounded_factor(
                jnp.asarray(start), jnp.asarray(end)
            )
        )

    def test_jit_and_grad(self, sofr_series):
        start = int(ymd_to_ordinal(2025, 1, 1))
        end = int(ymd_to_ordinal(2025, 2, 1))

        def f(series):
            return series.compounded_factor(
                jnp.asarray(start), jnp.asarray(end), "act_360"
            )

        assert float(eqx.filter_jit(f)(sofr_series)) == pytest.approx(
            float(f(sofr_series)), rel=1e-14
        )
        grads = eqx.filter_grad(f)(sofr_series)
        # Only the fixings inside the window move the factor.
        inside = (sofr_series.fixing_dates >= start) & (
            sofr_series.fixing_dates < end
        )
        assert jnp.all(grads.fixings[~inside] == 0.0)
        assert jnp.all(jnp.abs(grads.fixings[inside]) > 0.0)
