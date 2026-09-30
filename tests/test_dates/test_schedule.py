"""Tests for coupon schedule generation.

These are the tests that would have caught the defects this module was
rewritten to fix, so each class names the failure it guards against:

* interior dates were clamped to day 28, so no bond paid on its stated day;
* the odd period of a non-integral tenor was silently floor-divided away
  rather than becoming a stub;
* the ``start_*`` arguments influenced only the period *count* and never
  produced a date, so the first accrual period did not exist;
* nothing was validated, so an impossible date or frequency rolled quietly.

The old suite could not see any of this: it used only the 1st and 15th of
the month (both clamped to 28 without changing the assertion), checked only
``shape`` and the final element, and never generated a stub.
"""

import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from valax.dates.conventions import ScheduleConventions
from valax.dates.daycounts import _ordinal_to_ymd, year_fraction, ymd_to_ordinal
from valax.dates.schedule import Schedule, generate_schedule


def ymd(ordinal) -> tuple[int, int, int]:
    """Ordinal back to a ``(year, month, day)`` tuple for assertions."""
    y, m, d = _ordinal_to_ymd(jnp.asarray(ordinal))
    return int(y), int(m), int(d)


def days(ordinals) -> list[int]:
    """Day-of-month of each ordinal."""
    return [ymd(o)[2] for o in ordinals]


class TestReturnContract:
    def test_returns_three_aligned_arrays(self):
        sched = generate_schedule(2025, 1, 15, 2027, 1, 15)
        assert isinstance(sched, Schedule)
        assert sched.accrual_start.shape == (4,)
        assert sched.accrual_end.shape == (4,)
        assert sched.payment_dates.shape == (4,)

    def test_unpacks_as_a_triple(self):
        starts, ends, pays = generate_schedule(2025, 1, 15, 2027, 1, 15)
        assert starts.shape == ends.shape == pays.shape == (4,)

    def test_periods_are_contiguous(self):
        """No gaps and no overlaps: accrual_start[i+1] == accrual_end[i]."""
        starts, ends, _ = generate_schedule(2025, 3, 31, 2031, 3, 31, frequency=4)
        assert jnp.all(starts[1:] == ends[:-1])

    def test_first_accrual_start_is_the_dated_date(self):
        """The start arguments now produce a date rather than only a count."""
        starts, _, _ = generate_schedule(2025, 1, 15, 2027, 1, 15)
        assert ymd(starts[0]) == (2025, 1, 15)

    def test_last_accrual_end_is_maturity(self):
        _, ends, _ = generate_schedule(2025, 1, 15, 2027, 1, 15)
        assert ymd(ends[-1]) == (2027, 1, 15)

    def test_payment_dates_equal_accrual_end_until_tier_1_3(self):
        """The adjustment slot exists but is not yet applied."""
        _, ends, pays = generate_schedule(2025, 1, 15, 2027, 1, 15)
        assert jnp.all(pays == ends)

    def test_ascending_order(self):
        _, _, pays = generate_schedule(2025, 1, 15, 2035, 1, 15)
        assert jnp.all(pays[1:] > pays[:-1])

    @pytest.mark.parametrize(
        "frequency,expected", [(1, 5), (2, 10), (4, 20), (12, 60)]
    )
    def test_period_count_matches_frequency(self, frequency, expected):
        _, _, pays = generate_schedule(
            2025, 1, 15, 2030, 1, 15, frequency=frequency
        )
        assert pays.shape == (expected,)


class TestDayOfMonthPreservation:
    """The ``min(end_day, 28)`` clamp: every interior date was the 28th."""

    @pytest.mark.parametrize("day", [1, 15, 28, 29, 30, 31])
    def test_every_date_keeps_its_day_of_month(self, day):
        starts, ends, _ = generate_schedule(
            2025, 1, day, 2027, 1, day, frequency=2
        )
        # January and July both have 31 days, so no clamping is needed
        # anywhere in this schedule and every date must show `day`.
        assert set(days(starts)) == {day}
        assert set(days(ends)) == {day}

    def test_day_31_is_not_reduced_to_28(self):
        """The precise old failure: a 31st bond paid on the 28th."""
        _, ends, _ = generate_schedule(2025, 1, 31, 2027, 1, 31)
        assert [ymd(o) for o in ends] == [
            (2025, 7, 31),
            (2026, 1, 31),
            (2026, 7, 31),
            (2027, 1, 31),
        ]

    def test_day_29_and_30_survive_in_long_months(self):
        _, ends, _ = generate_schedule(2025, 4, 30, 2026, 4, 30, frequency=2)
        assert [ymd(o) for o in ends] == [(2025, 10, 30), (2026, 4, 30)]

    def test_clamping_does_not_ratchet_down(self):
        """31 Jan → 28 Feb → 31 Mar, not 31 Jan → 28 Feb → 28 Mar.

        The roll day is re-applied from the anchor each period, so a short
        month borrows the clamp for one period only.
        """
        _, ends, _ = generate_schedule(2025, 1, 31, 2025, 7, 31, frequency=12)
        assert [ymd(o) for o in ends] == [
            (2025, 2, 28),
            (2025, 3, 31),
            (2025, 4, 30),
            (2025, 5, 31),
            (2025, 6, 30),
            (2025, 7, 31),
        ]

    def test_leap_february_29_is_reachable(self):
        _, ends, _ = generate_schedule(2023, 2, 28, 2024, 2, 29, frequency=1)
        assert ymd(ends[-1]) == (2024, 2, 29)

    def test_february_29_anchor_clamps_in_non_leap_years(self):
        starts, ends, _ = generate_schedule(2024, 2, 29, 2027, 2, 28, frequency=1)
        assert ymd(starts[0]) == (2024, 2, 29)
        assert [ymd(o) for o in ends] == [
            (2025, 2, 28),
            (2026, 2, 28),
            (2027, 2, 28),
        ]


class TestEndOfMonth:
    def test_end_of_month_tracks_month_end(self):
        """28 Feb → 31 Aug → 28 Feb, each date the last of its month."""
        conv = ScheduleConventions(end_of_month=True)
        _, ends, _ = generate_schedule(
            2024, 2, 29, 2026, 2, 28, conventions=conv
        )
        assert [ymd(o) for o in ends] == [
            (2024, 8, 31),
            (2025, 2, 28),
            (2025, 8, 31),
            (2026, 2, 28),
        ]

    def test_without_end_of_month_the_day_is_preserved_instead(self):
        _, ends, _ = generate_schedule(2024, 2, 29, 2026, 2, 28)
        assert [ymd(o) for o in ends] == [
            (2024, 8, 28),
            (2025, 2, 28),
            (2025, 8, 28),
            (2026, 2, 28),
        ]

    def test_end_of_month_from_a_30_day_month(self):
        conv = ScheduleConventions(end_of_month=True)
        _, ends, _ = generate_schedule(
            2025, 4, 30, 2026, 4, 30, frequency=4, conventions=conv
        )
        assert [ymd(o) for o in ends] == [
            (2025, 7, 31),
            (2025, 10, 31),
            (2026, 1, 31),
            (2026, 4, 30),
        ]


class TestStubs:
    """The odd period used to be floor-divided away and silently lost."""

    # 2025-04-20 → 2026-10-15 quarterly is 5 regular periods plus 86 odd days.
    ODD = (2025, 4, 20, 2026, 10, 15)

    def test_short_front_stub_is_the_first_and_shortest_period(self):
        starts, ends, _ = generate_schedule(
            *self.ODD, frequency=4, conventions=ScheduleConventions(stub="short_front")
        )
        lengths = [int(e) - int(s) for s, e in zip(starts, ends)]
        assert lengths[0] == 86
        assert lengths[0] < min(lengths[1:])
        assert ymd(starts[0]) == (2025, 4, 20)
        assert ymd(ends[0]) == (2025, 7, 15)

    def test_long_front_stub_merges_into_the_first_regular_period(self):
        starts, ends, _ = generate_schedule(
            *self.ODD, frequency=4, conventions=ScheduleConventions(stub="long_front")
        )
        lengths = [int(e) - int(s) for s, e in zip(starts, ends)]
        assert lengths[0] == 178
        assert lengths[0] > max(lengths[1:])
        assert ymd(ends[0]) == (2025, 10, 15)

    def test_short_back_stub_is_the_last_and_shortest_period(self):
        starts, ends, _ = generate_schedule(
            *self.ODD, frequency=4, conventions=ScheduleConventions(stub="short_back")
        )
        lengths = [int(e) - int(s) for s, e in zip(starts, ends)]
        assert lengths[-1] == 87
        assert lengths[-1] < min(lengths[:-1])
        assert ymd(ends[-1]) == (2026, 10, 15)

    def test_long_back_stub_merges_into_the_last_regular_period(self):
        starts, ends, _ = generate_schedule(
            *self.ODD, frequency=4, conventions=ScheduleConventions(stub="long_back")
        )
        lengths = [int(e) - int(s) for s, e in zip(starts, ends)]
        assert lengths[-1] == 178
        assert lengths[-1] > max(lengths[:-1])

    def test_a_long_stub_has_one_fewer_period_than_a_short_stub(self):
        short = generate_schedule(
            *self.ODD, frequency=4, conventions=ScheduleConventions(stub="short_front")
        )
        long = generate_schedule(
            *self.ODD, frequency=4, conventions=ScheduleConventions(stub="long_front")
        )
        assert long.n_periods == short.n_periods - 1

    def test_front_stubs_anchor_on_maturity_back_stubs_on_the_start(self):
        """Roll direction follows stub placement, so the anchor lands exactly."""
        front = generate_schedule(
            *self.ODD, frequency=4, conventions=ScheduleConventions(stub="short_front")
        )
        back = generate_schedule(
            *self.ODD, frequency=4, conventions=ScheduleConventions(stub="short_back")
        )
        assert set(days(front.accrual_end[:-1])) == {15}
        assert set(days(back.accrual_start[1:])) == {20}

    def test_no_stub_is_generated_when_the_tenor_divides_evenly(self):
        for stub in ("short_front", "long_front", "short_back", "long_back"):
            starts, ends, _ = generate_schedule(
                2025, 1, 15, 2028, 1, 15, frequency=2,
                conventions=ScheduleConventions(stub=stub),
            )
            assert starts.shape == (6,)
            assert ymd(starts[0]) == (2025, 1, 15)

    def test_stub_none_rejects_a_non_integral_tenor(self):
        with pytest.raises(ValueError, match="not a whole number"):
            generate_schedule(
                *self.ODD, frequency=4, conventions=ScheduleConventions(stub="none")
            )

    def test_stub_none_accepts_an_integral_tenor(self):
        _, _, pays = generate_schedule(
            2025, 1, 15, 2028, 1, 15, frequency=2,
            conventions=ScheduleConventions(stub="none"),
        )
        assert pays.shape == (6,)

    def test_a_stub_shorter_than_one_period_is_not_dropped(self):
        """An 8-month semiannual tenor is 1 regular period plus a 2-month stub.

        Floor division gave ``8 // 6 == 1`` period, discarding the stub and
        with it two months of accrual.
        """
        starts, ends, _ = generate_schedule(2025, 1, 15, 2025, 9, 15)
        assert starts.shape == (2,)
        assert ymd(starts[0]) == (2025, 1, 15)
        assert ymd(ends[0]) == (2025, 3, 15)


class TestRollDay:
    def test_explicit_roll_day_overrides_the_anchor(self):
        """Interior dates roll on day 20, not on the maturity day 15."""
        conv = ScheduleConventions(roll_day=20)
        _, ends, _ = generate_schedule(
            2025, 1, 5, 2026, 1, 15, frequency=2, conventions=conv
        )
        assert [ymd(o) for o in ends] == [
            (2025, 1, 20),
            (2025, 7, 20),
            (2026, 1, 15),
        ]


class TestValidation:
    """Nothing was validated before; impossible inputs rolled silently."""

    @pytest.mark.parametrize("frequency", [0, 5, 7, 8, 11, 13, -2])
    def test_frequency_must_divide_twelve_months(self, frequency):
        with pytest.raises(ValueError, match="whole number of months"):
            generate_schedule(2025, 1, 15, 2030, 1, 15, frequency=frequency)

    @pytest.mark.parametrize("frequency", [1, 2, 3, 4, 6, 12])
    def test_valid_frequencies_are_accepted(self, frequency):
        generate_schedule(2025, 1, 15, 2030, 1, 15, frequency=frequency)

    def test_impossible_day_of_month_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            generate_schedule(2025, 2, 30, 2027, 1, 1)

    def test_february_29_in_a_non_leap_year_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            generate_schedule(2025, 2, 29, 2027, 1, 1)

    def test_month_out_of_range_raises(self):
        with pytest.raises(ValueError, match="month must be in 1..12"):
            generate_schedule(2025, 13, 1, 2027, 1, 1)

    def test_inverted_dates_raise(self):
        with pytest.raises(ValueError, match="strictly before"):
            generate_schedule(2027, 1, 15, 2025, 1, 15)

    def test_equal_dates_raise(self):
        with pytest.raises(ValueError, match="strictly before"):
            generate_schedule(2025, 1, 15, 2025, 1, 15)


class TestInvariants:
    """Property-based invariants over randomly generated schedules."""

    @settings(deadline=None, max_examples=150)
    @given(
        start_year=st.integers(min_value=1990, max_value=2040),
        start_month=st.integers(min_value=1, max_value=12),
        start_day=st.integers(min_value=1, max_value=28),
        months=st.integers(min_value=1, max_value=240),
        frequency=st.sampled_from([1, 2, 3, 4, 6, 12]),
        stub=st.sampled_from(
            ["short_front", "long_front", "short_back", "long_back"]
        ),
    )
    def test_contiguous_ascending_and_spanning(
        self, start_year, start_month, start_day, months, frequency, stub
    ):
        end_index = (start_year * 12 + start_month - 1) + months
        end_year, end_month = end_index // 12, end_index % 12 + 1
        starts, ends, pays = generate_schedule(
            start_year, start_month, start_day,
            end_year, end_month, start_day,
            frequency=frequency,
            conventions=ScheduleConventions(stub=stub),
        )
        # Contiguous, strictly increasing, and spanning exactly the tenor.
        assert jnp.all(ends > starts)
        assert jnp.all(starts[1:] == ends[:-1])
        assert int(starts[0]) == int(ymd_to_ordinal(start_year, start_month, start_day))
        assert int(ends[-1]) == int(ymd_to_ordinal(end_year, end_month, start_day))
        assert jnp.all(pays == ends)

    @settings(deadline=None, max_examples=100)
    @given(
        months=st.integers(min_value=6, max_value=180),
        frequency=st.sampled_from([1, 2, 4]),
        convention=st.sampled_from(["act_360", "act_365", "act_act", "30_360"]),
    )
    def test_year_fractions_sum_to_the_tenor(
        self, months, frequency, convention
    ):
        """Accrual partitions the tenor: the periods sum to the whole."""
        end_index = (2025 * 12 + 0) + months
        end_year, end_month = end_index // 12, end_index % 12 + 1
        starts, ends, _ = generate_schedule(
            2025, 1, 15, end_year, end_month, 15, frequency=frequency
        )
        total = float(jnp.sum(year_fraction(starts, ends, convention)))
        whole = float(year_fraction(starts[0], ends[-1], convention))
        assert total == pytest.approx(whole, rel=1e-12, abs=1e-12)
