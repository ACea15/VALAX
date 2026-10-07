"""Tests for the schedule convention set and its calendar pairing."""

import jax
import pytest

from valax.dates.calendars import brazil_calendar, target_calendar
from valax.dates.conventions import (
    BUSINESS_DAY_CONVENTIONS,
    STUB_KINDS,
    VALID_FREQUENCIES,
    ScheduleConventions,
)


class TestPytreeShape:
    def test_all_fields_are_static(self):
        """Conventions are configuration, never differentiable leaves."""
        assert jax.tree_util.tree_leaves(ScheduleConventions()) == []

    def test_a_calendar_contributes_exactly_one_leaf(self):
        """The holiday ordinals are a leaf; everything else stays static.

        ``calendar`` is the one non-static field, because a ``jnp`` array
        is unhashable and a static field must be hashable.  The leaf
        carries integer ordinals and enters no gradient path.
        """
        conv = ScheduleConventions(
            business_day_convention="modified_following",
            calendar=target_calendar(),
        )
        leaves = jax.tree_util.tree_leaves(conv)
        assert len(leaves) == 1
        assert leaves[0].shape == target_calendar().holidays.shape

    def test_is_frozen(self):
        conv = ScheduleConventions()
        with pytest.raises(Exception):
            conv.day_count = "act_360"

    def test_defaults(self):
        conv = ScheduleConventions()
        assert conv.day_count == "act_365"
        assert conv.stub == "short_front"
        assert conv.end_of_month is False
        assert conv.roll_day is None
        assert conv.calendar is None
        assert conv.business_day_convention == "unadjusted"


class TestValidation:
    def test_unknown_day_count_raises(self):
        with pytest.raises(ValueError, match="Unknown day count"):
            ScheduleConventions(day_count="act_366")

    @pytest.mark.parametrize("day_count", ["act_360", "act_365", "act_act", "30_360"])
    def test_known_day_counts_accepted(self, day_count):
        assert ScheduleConventions(day_count=day_count).day_count == day_count

    def test_business_day_count_is_a_known_day_count(self):
        """``bus_252`` lives in a second registry but is still valid here."""
        conv = ScheduleConventions(
            day_count="bus_252", calendar=brazil_calendar()
        )
        assert conv.day_count == "bus_252"

    def test_unknown_stub_raises(self):
        with pytest.raises(ValueError, match="Unknown stub kind"):
            ScheduleConventions(stub="middle")

    @pytest.mark.parametrize("stub", STUB_KINDS)
    def test_known_stubs_accepted(self, stub):
        assert ScheduleConventions(stub=stub).stub == stub

    @pytest.mark.parametrize("roll_day", [0, 32, -1])
    def test_roll_day_out_of_range_raises(self, roll_day):
        with pytest.raises(ValueError, match="roll_day must be in 1..31"):
            ScheduleConventions(roll_day=roll_day)

    @pytest.mark.parametrize("roll_day", [1, 15, 31])
    def test_roll_day_in_range_accepted(self, roll_day):
        assert ScheduleConventions(roll_day=roll_day).roll_day == roll_day

    def test_unknown_business_day_convention_raises(self):
        with pytest.raises(ValueError, match="Unknown business day convention"):
            ScheduleConventions(business_day_convention="nearest")


class TestCalendarConventionPairing:
    """A calendar and an adjustment rule are only meaningful together.

    These tests used to assert ``NotImplementedError`` while adjustment was
    unbuilt.  The guards did not go away when it landed — only their reason
    changed, from "not implemented" to "incoherent".  Each half alone is a
    convention that looks configured and is not, which is the failure this
    module exists to prevent.
    """

    @pytest.mark.parametrize(
        "convention",
        [c for c in BUSINESS_DAY_CONVENTIONS if c != "unadjusted"],
    )
    def test_a_convention_without_a_calendar_raises(self, convention):
        with pytest.raises(ValueError, match="needs a calendar"):
            ScheduleConventions(business_day_convention=convention)

    def test_a_calendar_nothing_would_consult_raises(self):
        """Unadjusted plus an elapsed-day count means the calendar is dead."""
        with pytest.raises(ValueError, match="nothing would consult it"):
            ScheduleConventions(calendar=target_calendar())

    def test_a_business_day_count_without_a_calendar_raises(self):
        with pytest.raises(ValueError, match="needs a calendar"):
            ScheduleConventions(day_count="bus_252")

    def test_a_non_calendar_object_raises(self):
        with pytest.raises(ValueError, match="must be a HolidayCalendar"):
            ScheduleConventions(
                calendar=object(), business_day_convention="following"
            )

    @pytest.mark.parametrize(
        "convention",
        [c for c in BUSINESS_DAY_CONVENTIONS if c != "unadjusted"],
    )
    def test_the_pair_together_is_accepted(self, convention):
        conv = ScheduleConventions(
            business_day_convention=convention, calendar=target_calendar()
        )
        assert conv.business_day_convention == convention
        assert conv.calendar.name == "TARGET"

    def test_bus_252_with_its_calendar_is_accepted(self):
        """A calendar is consulted by the day count even when unadjusted."""
        conv = ScheduleConventions(
            day_count="bus_252", calendar=brazil_calendar()
        )
        assert conv.day_count == "bus_252"
        assert conv.business_day_convention == "unadjusted"

    def test_unadjusted_remains_the_default(self):
        assert ScheduleConventions(
            business_day_convention="unadjusted"
        ).business_day_convention == "unadjusted"
        assert ScheduleConventions().calendar is None


def test_valid_frequencies_all_divide_twelve():
    assert all(12 % f == 0 for f in VALID_FREQUENCIES)
