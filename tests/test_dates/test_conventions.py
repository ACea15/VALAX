"""Tests for the schedule convention set and its Tier 1.3 slot."""

import jax
import pytest

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


class TestTier13Slot:
    """The slot must exist and must refuse to be silently ignored.

    Accepting a calendar and not applying it would reintroduce the exact
    failure mode this module was added to prevent — a convention that is
    configured but never honoured.
    """

    def test_supplying_a_calendar_raises_until_tier_1_3(self):
        with pytest.raises(NotImplementedError, match="Tier 1.3"):
            ScheduleConventions(calendar=object())

    @pytest.mark.parametrize(
        "convention",
        [c for c in BUSINESS_DAY_CONVENTIONS if c != "unadjusted"],
    )
    def test_adjustment_conventions_raise_until_tier_1_3(self, convention):
        with pytest.raises(NotImplementedError, match="Tier 1.3"):
            ScheduleConventions(business_day_convention=convention)

    def test_unadjusted_is_honoured_today(self):
        assert ScheduleConventions(
            business_day_convention="unadjusted"
        ).business_day_convention == "unadjusted"


def test_valid_frequencies_all_divide_twelve():
    assert all(12 % f == 0 for f in VALID_FREQUENCIES)
