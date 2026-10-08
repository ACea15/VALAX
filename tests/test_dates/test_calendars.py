"""Tests for business-day calendars, adjustment and Bus/252.

What is and is not an independent check
---------------------------------------
The holiday tables in :mod:`valax.dates._holiday_data` were generated from
``holidays`` and ``pandas_market_calendars``.  Asserting them against those
same packages would prove only that the generator ran, so this file uses
three *different* kinds of reference, in descending order of strength:

1. **Structural rules from the stdlib.**  "Memorial Day is the last Monday
   in May", "a Japanese substitute holiday follows a Sunday holiday", "the
   weekend is Saturday and Sunday" — derived with :mod:`datetime` and
   :mod:`calendar` only.  These cannot agree with a wrong table by
   coincidence.
2. **Hand-transcribed market schedules.**  The full 2025 weekday closure
   list for each market, typed from the published calendars.  A
   transcription error and a data error would have to coincide to pass.
3. **Regeneration.**  With the ``calendars-data`` extra installed, the
   tables are re-derived and compared.  This catches a corrupted artifact
   but, being the same upstream, proves nothing about correctness — which
   is why it is last.

Note what is deliberately *not* here: nothing is asserted against VALAX's
own ordinal algebra.  :func:`~valax.dates.daycounts.ymd_to_ordinal` and
:func:`~valax.dates.daycounts.thirty_360` both route through the same
Hinnant conversion, so checking one with the other shows only that the
algorithm is used twice.  Ordinals here come from
:mod:`datetime` subtraction.

Every adjustment test uses dates that genuinely land on a weekend or
holiday and asserts the adjusted result **differs** from the unadjusted
one.  A fixture where adjustment is a no-op would pass against a function
that does nothing; see ``docs/architecture/numerical-pitfalls.md`` entry 3.
"""

import calendar as pycalendar
import datetime as dt
import importlib.util
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest

from valax.dates import (
    ScheduleConventions,
    brazil_calendar,
    generate_schedule,
    london_calendar,
    market_calendar,
    nyse_calendar,
    sofr_calendar,
    target_calendar,
    tokyo_calendar,
    year_fraction,
)
from valax.dates import _holiday_data
from valax.dates.calendars import HolidayCalendar, adjust, is_weekend, weekday

_EPOCH = dt.date(1970, 1, 1)


def ordinal(year: int, month: int, day: int) -> int:
    """Date to ordinal using the stdlib, never VALAX's own conversion."""
    return (dt.date(year, month, day) - _EPOCH).days


def to_date(value) -> dt.date:
    return _EPOCH + dt.timedelta(days=int(value))


def nth_weekday(year: int, month: int, weekday_index: int, n: int) -> dt.date:
    """The ``n``-th given weekday of a month; ``n = -1`` means the last."""
    days = [
        dt.date(year, month, d)
        for d in range(1, pycalendar.monthrange(year, month)[1] + 1)
        if dt.date(year, month, d).weekday() == weekday_index
    ]
    return days[n if n < 0 else n - 1]


# ──────────────────────────────────────────────────────────────────────
# 1. Structural rules, derived from the standard library alone
# ──────────────────────────────────────────────────────────────────────


class TestWeekendRuleMatchesTheStdlib:
    """The weekend is arithmetic, not data, so it is checkable exactly."""

    def test_weekday_matches_datetime_over_forty_years(self):
        for ord_value in range(ordinal(1990, 1, 1), ordinal(2030, 1, 1), 7919):
            assert int(weekday(jnp.int32(ord_value))) == to_date(
                ord_value
            ).weekday()

    def test_every_weekday_residue_is_covered(self):
        """All seven residues appear, so the test above is not degenerate."""
        start = ordinal(2025, 1, 1)
        seen = {int(weekday(jnp.int32(start + k))) for k in range(7)}
        assert seen == set(range(7))

    def test_is_weekend_matches_datetime(self):
        for ord_value in range(ordinal(2024, 1, 1), ordinal(2026, 1, 1)):
            assert bool(is_weekend(jnp.int32(ord_value))) == (
                to_date(ord_value).weekday() >= 5
            )

    def test_epoch_is_a_thursday(self):
        """Pins the offset the whole weekend rule is built on."""
        assert _EPOCH.weekday() == 3
        assert int(weekday(jnp.int32(0))) == 3


class TestHolidaysFollowTheirPublishedRules:
    """Rule-shaped holidays, checked against the rule via the stdlib.

    These are the strongest assertions in the file: the expected date is
    computed from the statutory rule, so a table that disagrees is wrong
    regardless of where it came from.
    """

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2030])
    def test_mlk_day_is_the_third_monday_in_january(self, year):
        expected = nth_weekday(year, 1, pycalendar.MONDAY, 3)
        cal = sofr_calendar()
        assert bool(cal.is_holiday(jnp.int32((expected - _EPOCH).days)))

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2030])
    def test_memorial_day_is_the_last_monday_in_may(self, year):
        expected = nth_weekday(year, 5, pycalendar.MONDAY, -1)
        assert bool(
            sofr_calendar().is_holiday(jnp.int32((expected - _EPOCH).days))
        )

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2030])
    def test_thanksgiving_is_the_fourth_thursday_in_november(self, year):
        expected = nth_weekday(year, 11, pycalendar.THURSDAY, 4)
        assert bool(
            nyse_calendar().is_holiday(jnp.int32((expected - _EPOCH).days))
        )

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2030])
    def test_uk_summer_bank_holiday_is_the_last_monday_in_august(self, year):
        expected = nth_weekday(year, 8, pycalendar.MONDAY, -1)
        assert bool(
            london_calendar().is_holiday(jnp.int32((expected - _EPOCH).days))
        )

    @pytest.mark.parametrize("year", [2024, 2025, 2026, 2030])
    def test_japanese_coming_of_age_is_the_second_monday_in_january(self, year):
        expected = nth_weekday(year, 1, pycalendar.MONDAY, 2)
        assert bool(
            tokyo_calendar().is_holiday(jnp.int32((expected - _EPOCH).days))
        )

    def test_a_sunday_holiday_is_observed_on_the_monday_in_tokyo(self):
        """Japan's substitute-holiday rule, the fiddly part of observance.

        3 May 2025 (Constitution Day) is a Saturday and 23 November 2025
        (Labour Thanksgiving) is a Sunday.  Japan moves a *Sunday* holiday
        to the next working day and leaves a Saturday one alone, so 24
        November must be closed and 5 May must be closed for its own sake.
        """
        cal = tokyo_calendar()
        assert dt.date(2025, 11, 23).weekday() == 6
        assert bool(cal.is_holiday(jnp.int32(ordinal(2025, 11, 24))))
        assert dt.date(2025, 5, 3).weekday() == 5
        # The Saturday itself is not in the table: weekends never are.
        assert not bool(cal.is_holiday(jnp.int32(ordinal(2025, 5, 3))))


class TestCalendarsAreGenuinelyDifferent:
    """A market calendar that is a copy of another is a bug.

    The NYSE/SOFR pair is the sharp case: they differ in *both*
    directions, so neither contains the other and a single "US calendar"
    would be wrong for one of them.
    """

    def test_nyse_closes_good_friday_but_the_bond_market_opens_columbus_day(
        self,
    ):
        good_friday = jnp.int32(ordinal(2025, 4, 18))
        columbus = jnp.int32(ordinal(2025, 10, 13))
        veterans = jnp.int32(ordinal(2025, 11, 11))

        assert bool(nyse_calendar().is_holiday(good_friday))
        assert bool(sofr_calendar().is_holiday(good_friday))
        # Columbus and Veterans Day: bond market shut, exchange open.
        assert bool(sofr_calendar().is_holiday(columbus))
        assert not bool(nyse_calendar().is_holiday(columbus))
        assert bool(sofr_calendar().is_holiday(veterans))
        assert not bool(nyse_calendar().is_holiday(veterans))

    def test_target_does_not_close_for_us_or_uk_holidays(self):
        assert not bool(
            target_calendar().is_holiday(jnp.int32(ordinal(2025, 7, 4)))
        )
        assert not bool(
            target_calendar().is_holiday(jnp.int32(ordinal(2025, 8, 25)))
        )

    def test_the_six_markets_have_distinct_tables(self):
        tables = {
            name: tuple(factory().holidays.tolist())
            for name, factory in (
                ("TARGET", target_calendar),
                ("NYSE", nyse_calendar),
                ("SOFR", sofr_calendar),
                ("LONDON", london_calendar),
                ("TOKYO", tokyo_calendar),
                ("BRAZIL", brazil_calendar),
            )
        }
        assert len(set(tables.values())) == len(tables)


# ──────────────────────────────────────────────────────────────────────
# 2. Hand-transcribed market schedules
# ──────────────────────────────────────────────────────────────────────

#: Complete **weekday** closure lists for 2025, transcribed from each
#: market's published schedule.  Weekend closures are absent by design, so
#: e.g. Brazil's 7 September (a Sunday in 2025) does not appear.
_2025_CLOSURES = {
    "TARGET": [
        (1, 1),    # New Year's Day
        (4, 18),   # Good Friday
        (4, 21),   # Easter Monday
        (5, 1),    # Labour Day
        (12, 25),  # Christmas Day
        (12, 26),  # Boxing Day
    ],
    "NYSE": [
        (1, 1),    # New Year's Day
        (1, 9),    # National day of mourning, President Carter
        (1, 20),   # Martin Luther King Jr Day
        (2, 17),   # Washington's Birthday
        (4, 18),   # Good Friday
        (5, 26),   # Memorial Day
        (6, 19),   # Juneteenth
        (7, 4),    # Independence Day
        (9, 1),    # Labor Day
        (11, 27),  # Thanksgiving
        (12, 25),  # Christmas Day
    ],
    "SOFR": [
        (1, 1),
        (1, 20),
        (2, 17),
        (4, 18),   # Good Friday: bond market shut, not a federal holiday
        (5, 26),
        (6, 19),
        (7, 4),
        (9, 1),
        (10, 13),  # Columbus Day
        (11, 11),  # Veterans Day
        (11, 27),
        (12, 25),
    ],
    "LONDON": [
        (1, 1),    # New Year's Day
        (4, 18),   # Good Friday
        (4, 21),   # Easter Monday
        (5, 5),    # Early May bank holiday
        (5, 26),   # Spring bank holiday
        (8, 25),   # Summer bank holiday
        (12, 25),  # Christmas Day
        (12, 26),  # Boxing Day
    ],
    "TOKYO": [
        (1, 1),    # New Year's Day
        (1, 2),    # Bank holiday
        (1, 3),    # Bank holiday
        (1, 13),   # Coming of Age Day
        (2, 11),   # National Foundation Day
        (2, 24),   # Emperor's Birthday, observed (23rd is a Sunday)
        (3, 20),   # Vernal Equinox
        (4, 29),   # Showa Day
        (5, 5),    # Children's Day
        (5, 6),    # Observed (Constitution Day, 3 May, is a Saturday)
        (7, 21),   # Marine Day
        (8, 11),   # Mountain Day
        (9, 15),   # Respect for the Aged Day
        (9, 23),   # Autumnal Equinox
        (10, 13),  # Sports Day
        (11, 3),   # Culture Day
        (11, 24),  # Observed (Labour Thanksgiving, 23 Nov, is a Sunday)
        (12, 31),  # Bank holiday
    ],
    "BRAZIL": [
        (1, 1),    # Confraternizacao Universal
        (3, 3),    # Carnival Monday
        (3, 4),    # Carnival Tuesday
        (4, 18),   # Sexta-feira Santa
        (4, 21),   # Tiradentes
        (5, 1),    # Dia do Trabalhador
        (6, 19),   # Corpus Christi
        (11, 20),  # Consciencia Negra
        (12, 24),  # Christmas Eve
        (12, 25),  # Natal
        (12, 31),  # New Year's Eve
    ],
}


@pytest.mark.audit
@pytest.mark.parametrize("market", sorted(_2025_CLOSURES))
def test_2025_closures_match_the_published_schedule(market):
    """N-6 gate: calendars match externally checkable reference data.

    Exact set equality — no missing closures and no invented ones. Marked
    ``audit`` for the same reason the T-10 day-count gate is: it pins
    reference data against a source outside VALAX, which is what the
    conformance row claims.
    """
    cal = market_calendar(market)
    expected = {ordinal(2025, m, d) for m, d in _2025_CLOSURES[market]}
    actual = {
        o
        for o in cal.holidays.tolist()
        if to_date(o).year == 2025
    }
    assert actual == expected


@pytest.mark.parametrize("market", sorted(_2025_CLOSURES))
def test_no_table_contains_a_weekend(market):
    """Weekends are handled by arithmetic, so storing them is a defect."""
    cal = market_calendar(market)
    assert not any(to_date(o).weekday() >= 5 for o in cal.holidays.tolist())


@pytest.mark.parametrize("market", sorted(_2025_CLOSURES))
def test_tables_are_sorted_and_unique(market):
    ordinals = market_calendar(market).holidays.tolist()
    assert ordinals == sorted(set(ordinals))


def test_tse_outage_is_not_a_holiday():
    """1 October 2020 was a trading outage, not a settlement holiday.

    The Tokyo Stock Exchange lost a full day to an Arrowhead hardware
    failure.  Banks settled normally, so a payment must not roll off it.
    Deriving closures from a trading calendar picks such days up, which is
    why the generator carries an explicit exclusion list.
    """
    outage = jnp.int32(ordinal(2020, 10, 1))
    assert dt.date(2020, 10, 1).weekday() == 3
    assert not bool(tokyo_calendar().is_holiday(outage))
    assert bool(tokyo_calendar().is_business_day(outage))


# ──────────────────────────────────────────────────────────────────────
# 3. Adjustment
# ──────────────────────────────────────────────────────────────────────


class TestAdjustmentMovesDatesAndStaysPut:
    def test_a_saturday_rolls_forward_under_following(self):
        cal = target_calendar()
        saturday = ordinal(2025, 5, 31)
        assert dt.date(2025, 5, 31).weekday() == 5
        got = int(cal.adjust(jnp.int32(saturday), "following"))
        assert got != saturday                      # anti-vacuity
        assert to_date(got) == dt.date(2025, 6, 2)

    def test_modified_following_refuses_to_cross_the_month_end(self):
        """The whole point of the 'modified' prefix.

        31 May 2025 is a Saturday, so plain following lands in June.
        Modified following must reverse into 30 May instead.
        """
        cal = target_calendar()
        saturday = jnp.int32(ordinal(2025, 5, 31))
        plain = int(cal.adjust(saturday, "following"))
        modified = int(cal.adjust(saturday, "modified_following"))
        assert to_date(plain).month == 6
        assert to_date(modified) == dt.date(2025, 5, 30)
        assert plain != modified                    # the rule is doing work

    def test_modified_preceding_refuses_to_cross_into_the_previous_month(self):
        """1 March 2025 is a Saturday; preceding would land in February."""
        cal = target_calendar()
        saturday = jnp.int32(ordinal(2025, 3, 1))
        plain = int(cal.adjust(saturday, "preceding"))
        modified = int(cal.adjust(saturday, "modified_preceding"))
        assert to_date(plain).month == 2
        assert to_date(modified) == dt.date(2025, 3, 3)
        assert plain != modified

    def test_a_holiday_run_is_cleared_in_one_call(self):
        """Christmas 2025 on TARGET: 25th, 26th, then a weekend."""
        cal = target_calendar()
        christmas = jnp.int32(ordinal(2025, 12, 25))
        got = int(cal.adjust(christmas, "following"))
        assert to_date(got) == dt.date(2025, 12, 29)
        assert bool(cal.is_business_day(jnp.int32(got)))

    def test_new_year_run_in_tokyo_is_cleared(self):
        """Tokyo's longest run: 31 Dec and 1-3 Jan plus a weekend."""
        cal = tokyo_calendar()
        got = int(cal.adjust(jnp.int32(ordinal(2026, 12, 31)), "following"))
        assert bool(cal.is_business_day(jnp.int32(got)))
        assert to_date(got).year == 2027

    @pytest.mark.parametrize(
        "convention",
        ["following", "modified_following", "preceding", "modified_preceding"],
    )
    @pytest.mark.parametrize(
        "date", [(2025, 5, 31), (2025, 12, 25), (2025, 3, 1), (2025, 4, 18)]
    )
    def test_adjustment_is_idempotent(self, convention, date):
        """Adjusting an adjusted date must be a no-op in every convention."""
        cal = target_calendar()
        once = cal.adjust(jnp.int32(ordinal(*date)), convention)
        twice = cal.adjust(once, convention)
        assert int(once) == int(twice)

    @pytest.mark.parametrize(
        "convention",
        ["following", "modified_following", "preceding", "modified_preceding"],
    )
    def test_adjustment_always_lands_on_a_business_day(self, convention):
        cal = tokyo_calendar()
        for offset in range(400):
            value = jnp.int32(ordinal(2025, 1, 1) + offset)
            assert bool(
                cal.is_business_day(cal.adjust(value, convention))
            )

    def test_a_business_day_is_left_alone(self):
        cal = target_calendar()
        wednesday = ordinal(2025, 6, 11)
        assert dt.date(2025, 6, 11).weekday() == 2
        for convention in (
            "following",
            "modified_following",
            "preceding",
            "modified_preceding",
        ):
            assert int(cal.adjust(jnp.int32(wednesday), convention)) == wednesday

    def test_unadjusted_never_consults_the_calendar(self):
        cal = target_calendar()
        saturday = ordinal(2025, 5, 31)
        assert int(cal.adjust(jnp.int32(saturday), "unadjusted")) == saturday

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError, match="Unknown business day convention"):
            target_calendar().adjust(jnp.int32(0), "nearest")

    def test_vectorises_over_an_array_of_dates(self):
        cal = target_calendar()
        dates = jnp.asarray(
            [ordinal(2025, 5, 31), ordinal(2025, 12, 25), ordinal(2025, 6, 11)],
            dtype=jnp.int32,
        )
        got = [int(x) for x in cal.adjust(dates, "modified_following")]
        assert [to_date(g) for g in got] == [
            dt.date(2025, 5, 30),
            dt.date(2025, 12, 29),
            dt.date(2025, 6, 11),
        ]


class TestEndOfMonthRule:
    def test_last_business_day_of_month_avoids_the_weekend(self):
        cal = target_calendar()
        got = int(cal.last_business_day_of_month(jnp.int32(ordinal(2025, 5, 12))))
        assert to_date(got) == dt.date(2025, 5, 30)
        assert dt.date(2025, 5, 31).weekday() == 5   # the 31st is a Saturday

    def test_handles_february_in_a_leap_year(self):
        cal = target_calendar()
        got = int(cal.last_business_day_of_month(jnp.int32(ordinal(2024, 2, 1))))
        assert to_date(got) == dt.date(2024, 2, 29)

    def test_handles_february_in_a_common_year(self):
        cal = target_calendar()
        got = int(cal.last_business_day_of_month(jnp.int32(ordinal(2025, 2, 1))))
        assert to_date(got) == dt.date(2025, 2, 28)

    def test_steps_back_over_a_month_end_holiday(self):
        """31 December 2025 is a Wednesday but a Brazilian closure."""
        cal = brazil_calendar()
        got = int(cal.last_business_day_of_month(jnp.int32(ordinal(2025, 12, 1))))
        assert to_date(got) == dt.date(2025, 12, 30)


class TestBusinessDayNavigation:
    def test_next_business_day_is_strict(self):
        cal = target_calendar()
        wednesday = ordinal(2025, 6, 11)
        assert to_date(
            int(cal.next_business_day(jnp.int32(wednesday)))
        ) == dt.date(2025, 6, 12)
        assert int(cal.roll_following(jnp.int32(wednesday))) == wednesday

    def test_previous_business_day_is_strict(self):
        cal = target_calendar()
        wednesday = ordinal(2025, 6, 11)
        assert to_date(
            int(cal.previous_business_day(jnp.int32(wednesday)))
        ) == dt.date(2025, 6, 10)

    def test_next_business_day_jumps_a_weekend(self):
        cal = target_calendar()
        friday = ordinal(2025, 6, 13)
        assert dt.date(2025, 6, 13).weekday() == 4
        assert to_date(
            int(cal.next_business_day(jnp.int32(friday)))
        ) == dt.date(2025, 6, 16)

    @pytest.mark.parametrize("lag", [0, 1, 2, 5])
    def test_add_business_days_matches_a_stdlib_walk(self, lag):
        cal = target_calendar()
        start = ordinal(2025, 12, 23)
        holidays = {int(o) for o in cal.holidays.tolist()}

        def closed(value: int) -> bool:
            return to_date(value).weekday() >= 5 or value in holidays

        expected = start
        while closed(expected):
            expected += 1
        for _ in range(lag):
            expected += 1
            while closed(expected):
                expected += 1

        assert int(cal.add_business_days(jnp.int32(start), lag)) == expected


class TestCoverageIsEnforced:
    """An uncovered year is unknown, not holiday-free."""

    def test_adjusting_outside_the_range_raises(self):
        cal = target_calendar()
        with pytest.raises(Exception) as excinfo:
            cal.adjust(jnp.int32(ordinal(2150, 6, 1)), "following")
        assert "covers" in str(excinfo.value) or "EquinoxRuntimeError" in type(
            excinfo.value
        ).__name__

    def test_require_covered_raises_eagerly(self):
        cal = target_calendar()
        with pytest.raises(ValueError, match="covers 1999-2099"):
            cal.require_covered([ordinal(1998, 12, 31)])

    def test_require_covered_accepts_the_boundaries(self):
        cal = target_calendar()
        cal.require_covered([ordinal(1999, 1, 1), ordinal(2099, 12, 31)])

    def test_target_starts_with_the_euro(self):
        """TARGET had no calendar before 1999, so it claims none."""
        assert target_calendar().first_year == 1999

    def test_tokyo_stops_before_the_equinox_horizon(self):
        """Japan's equinoxes are astronomical, fixed a year ahead."""
        assert tokyo_calendar().last_year == 2098


class TestTheTwoRepresentationsAgree:
    """The sparse table defines the calendar; the dense one answers queries.

    Carrying the same information twice is only safe if a divergence is
    caught, so these assert the dense prefix sum against the sparse
    ordinals it was derived from — at *every* covered ordinal, not a
    sample. Without this, a defect in `build` would make every lookup
    quietly wrong while every spot-check still passed.
    """

    @pytest.mark.parametrize(
        "factory", [target_calendar, tokyo_calendar, brazil_calendar]
    )
    def test_dense_membership_matches_the_sparse_table_everywhere(self, factory):
        cal = factory()
        listed = {int(o) for o in cal.holidays.tolist()}
        days = jnp.arange(
            cal.first_ordinal, cal.last_ordinal + 1, dtype=jnp.int32
        )
        got_business = cal.is_business_day(days).tolist()
        got_holiday = cal.is_holiday(days).tolist()
        for offset, (is_bd, is_hol) in enumerate(zip(got_business, got_holiday)):
            value = cal.first_ordinal + offset
            weekend = to_date(value).weekday() >= 5
            assert is_hol == (value in listed)
            assert is_bd == (not weekend and value not in listed)

    @pytest.mark.parametrize(
        "factory", [target_calendar, tokyo_calendar, brazil_calendar]
    )
    def test_prefix_sum_is_the_running_count_of_business_days(self, factory):
        cal = factory()
        days = jnp.arange(
            cal.first_ordinal, cal.last_ordinal + 1, dtype=jnp.int32
        )
        flags = cal.is_business_day(days)
        cumulative = cal.business_days_cumulative
        assert int(cumulative[0]) == 0
        assert int(cumulative[-1]) == int(jnp.sum(flags))
        # Monotone, and advancing by exactly one on each business day.
        steps = jnp.diff(cumulative)
        assert steps.tolist() == flags.astype(cumulative.dtype).tolist()

    def test_counting_agrees_with_summing_the_flags(self):
        """The two things the single array is asked, cross-checked."""
        cal = brazil_calendar()
        lo, hi = ordinal(2025, 1, 1), ordinal(2027, 1, 1)
        days = jnp.arange(lo, hi, dtype=jnp.int32)
        assert int(cal.business_days_between(jnp.int32(lo), jnp.int32(hi))) == int(
            jnp.sum(cal.is_business_day(days))
        )


class TestMaxClosureRunIsMeasured:
    def test_bound_is_large_enough_for_every_market(self):
        """Verify the measured bound against a brute-force scan."""
        for factory in (target_calendar, tokyo_calendar, brazil_calendar):
            cal = factory()
            holidays = {int(o) for o in cal.holidays.tolist()}
            longest = run = 0
            for value in range(cal.first_ordinal, cal.last_ordinal + 1):
                closed = to_date(value).weekday() >= 5 or value in holidays
                run = run + 1 if closed else 0
                longest = max(longest, run)
            assert cal.max_closure_run == longest

    def test_tokyo_has_the_longest_run(self):
        """Tokyo's New Year plus Golden Week exceeds a naive guess of 7."""
        assert tokyo_calendar().max_closure_run > 7


# ──────────────────────────────────────────────────────────────────────
# 4. Set algebra
# ──────────────────────────────────────────────────────────────────────


class TestUnionAndIntersection:
    def test_union_closes_on_either_markets_holiday(self):
        joint = target_calendar().union(sofr_calendar())
        july_four = jnp.int32(ordinal(2025, 7, 4))
        may_one = jnp.int32(ordinal(2025, 5, 1))
        assert bool(joint.is_holiday(july_four))   # US only
        assert bool(joint.is_holiday(may_one))     # TARGET only

    def test_intersection_closes_only_on_shared_holidays(self):
        shared = target_calendar().intersection(sofr_calendar())
        assert bool(shared.is_holiday(jnp.int32(ordinal(2025, 4, 18))))
        assert not bool(shared.is_holiday(jnp.int32(ordinal(2025, 7, 4))))
        assert not bool(shared.is_holiday(jnp.int32(ordinal(2025, 5, 1))))

    def test_union_is_a_superset_and_intersection_a_subset(self):
        a, b = target_calendar(), london_calendar()
        union = {int(o) for o in a.union(b).holidays.tolist()}
        inter = {int(o) for o in a.intersection(b).holidays.tolist()}
        assert inter <= union

    def test_union_narrows_the_covered_range_to_the_overlap(self):
        """Outside the overlap one side has no data, so neither does the union."""
        joint = target_calendar().union(tokyo_calendar())
        assert joint.first_year == 1999   # TARGET's floor
        assert joint.last_year == 2098    # Tokyo's ceiling

    def test_union_is_commutative_in_its_holiday_set(self):
        a, b = target_calendar(), london_calendar()
        assert a.union(b).holidays.tolist() == b.union(a).holidays.tolist()

    def test_union_recomputes_the_closure_run(self):
        """A combined calendar has longer runs, so the bound must grow."""
        a, b = target_calendar(), tokyo_calendar()
        joint = a.union(b)
        assert joint.max_closure_run >= max(a.max_closure_run, b.max_closure_run)

    def test_non_overlapping_ranges_raise(self):
        # Both anchors are weekdays on purpose: a weekend date is now
        # rejected outright, since the weekend rule would double-count it.
        early = HolidayCalendar.build("EARLY", [ordinal(1975, 1, 1)], 1975, 1980)
        late = HolidayCalendar.build("LATE", [ordinal(2050, 1, 3)], 2050, 2060)
        assert dt.date(1975, 1, 1).weekday() < 5
        assert dt.date(2050, 1, 3).weekday() < 5
        with pytest.raises(ValueError, match="do not overlap"):
            early.union(late)


# ──────────────────────────────────────────────────────────────────────
# 5. Bus/252
# ──────────────────────────────────────────────────────────────────────


class TestBus252:
    @pytest.mark.parametrize(
        "start,end",
        [
            ((2025, 1, 1), (2026, 1, 1)),
            ((2025, 2, 1), (2025, 3, 1)),
            ((2024, 12, 20), (2025, 1, 10)),
            ((2025, 6, 11), (2025, 6, 11)),
        ],
    )
    def test_business_day_count_matches_a_stdlib_walk(self, start, end):
        """Independent reference: a day-by-day loop using datetime only."""
        cal = brazil_calendar()
        holidays = {int(o) for o in cal.holidays.tolist()}
        lo, hi = ordinal(*start), ordinal(*end)
        expected = sum(
            1
            for value in range(lo, hi)
            if to_date(value).weekday() < 5 and value not in holidays
        )
        got = int(cal.business_days_between(jnp.int32(lo), jnp.int32(hi)))
        assert got == expected

    def test_a_full_year_is_about_252_days(self):
        cal = brazil_calendar()
        count = int(
            cal.business_days_between(
                jnp.int32(ordinal(2025, 1, 1)), jnp.int32(ordinal(2026, 1, 1))
            )
        )
        assert 245 <= count <= 255

    def test_year_fraction_divides_by_the_market_constant(self):
        cal = brazil_calendar()
        lo, hi = jnp.int32(ordinal(2025, 1, 1)), jnp.int32(ordinal(2026, 1, 1))
        count = int(cal.business_days_between(lo, hi))
        assert float(year_fraction(lo, hi, "bus_252", calendar=cal)) == (
            count / 252.0
        )

    def test_reversed_interval_is_negative(self):
        cal = brazil_calendar()
        lo, hi = jnp.int32(ordinal(2025, 1, 1)), jnp.int32(ordinal(2025, 7, 1))
        assert int(cal.business_days_between(hi, lo)) == -int(
            cal.business_days_between(lo, hi)
        )

    def test_counting_differs_from_elapsed_days(self):
        """Anti-vacuity: Bus/252 must not coincide with ACT/252."""
        cal = brazil_calendar()
        lo, hi = jnp.int32(ordinal(2025, 1, 1)), jnp.int32(ordinal(2026, 1, 1))
        business = float(year_fraction(lo, hi, "bus_252", calendar=cal))
        elapsed = (int(hi) - int(lo)) / 252.0
        assert business != elapsed

    def test_without_a_calendar_it_refuses_to_guess(self):
        with pytest.raises(ValueError, match="needs a calendar"):
            year_fraction(
                jnp.int32(ordinal(2025, 1, 1)),
                jnp.int32(ordinal(2026, 1, 1)),
                "bus_252",
            )

    def test_elapsed_day_conventions_ignore_the_calendar_argument(self):
        """Passing a calendar must not change an elapsed-day convention."""
        lo, hi = jnp.int32(ordinal(2025, 1, 1)), jnp.int32(ordinal(2026, 1, 1))
        assert float(year_fraction(lo, hi, "act_365")) == float(
            year_fraction(lo, hi, "act_365", calendar=brazil_calendar())
        )

    def test_out_of_range_dates_raise_rather_than_miscount(self):
        cal = brazil_calendar()
        with pytest.raises(Exception):
            year_fraction(
                jnp.int32(ordinal(2150, 1, 1)),
                jnp.int32(ordinal(2150, 2, 1)),
                "bus_252",
                calendar=cal,
            )


# ──────────────────────────────────────────────────────────────────────
# 6. JIT and pytree behaviour
# ──────────────────────────────────────────────────────────────────────


class TestJitAndPytree:
    def test_adjust_under_filter_jit(self):
        cal = target_calendar()

        @eqx.filter_jit
        def run(value, calendar):
            return calendar.adjust(value, "modified_following")

        got = int(run(jnp.int32(ordinal(2025, 5, 31)), cal))
        assert to_date(got) == dt.date(2025, 5, 30)

    def test_bus_252_under_filter_jit(self):
        """The business-day *count* is exact under jit; the division is not.

        The count is integer arithmetic, so it must agree bit for bit --
        that is where a date or lookup defect would show.  The subsequent
        division by 252 is entitled to one ulp, because XLA is free to
        rewrite ``n / 252`` as ``n * (1 / 252)`` and does so when the
        expression is compiled.  Asserting exact float equality here would
        pin an XLA code-generation choice rather than anything about day
        counts; the existing T-10 gate makes the same distinction for
        ACT/ACT.
        """
        cal = brazil_calendar()
        lo, hi = jnp.int32(ordinal(2025, 1, 1)), jnp.int32(ordinal(2026, 1, 1))

        @eqx.filter_jit
        def count(start, end, calendar):
            return calendar.business_days_between(start, end)

        @eqx.filter_jit
        def fraction(start, end, calendar):
            return year_fraction(start, end, "bus_252", calendar=calendar)

        assert int(count(lo, hi, cal)) == int(cal.business_days_between(lo, hi))
        eager = float(year_fraction(lo, hi, "bus_252", calendar=cal))
        assert float(fraction(lo, hi, cal)) == pytest.approx(eager, rel=1e-15)

    def test_the_two_tables_are_the_only_leaves(self):
        """Static config stays static; only the two integer tables are leaves.

        The sparse ordinals are the canonical definition, the dense prefix
        sum is what gets queried. Both are leaves rather than static fields
        because a ``jnp`` array is unhashable.
        """
        import jax

        cal = target_calendar()
        leaves = jax.tree_util.tree_leaves(cal)
        assert len(leaves) == 2
        shapes = {leaf.shape for leaf in leaves}
        assert cal.holidays.shape in shapes
        assert cal.business_days_cumulative.shape in shapes

    def test_survives_a_pytree_round_trip(self):
        import jax

        cal = target_calendar()
        flat, treedef = jax.tree_util.tree_flatten(cal)
        rebuilt = jax.tree_util.tree_unflatten(treedef, flat)
        assert rebuilt.name == cal.name
        assert rebuilt.max_closure_run == cal.max_closure_run
        assert int(rebuilt.adjust(jnp.int32(ordinal(2025, 5, 31)), "following")) == int(
            cal.adjust(jnp.int32(ordinal(2025, 5, 31)), "following")
        )


class TestConstruction:
    def test_build_rejects_an_empty_table(self):
        with pytest.raises(ValueError, match="no holidays at all"):
            HolidayCalendar.build("EMPTY", [], 2020, 2030)

    def test_build_sorts_and_deduplicates(self):
        cal = HolidayCalendar.build(
            "X",
            [ordinal(2025, 7, 4), ordinal(2025, 1, 1), ordinal(2025, 7, 4)],
            2025,
            2025,
        )
        assert cal.holidays.tolist() == [ordinal(2025, 1, 1), ordinal(2025, 7, 4)]

    def test_rejects_an_inverted_year_range(self):
        with pytest.raises(ValueError, match="exceeds"):
            HolidayCalendar.build("X", [ordinal(2025, 1, 1)], 2030, 2020)

    def test_rejects_a_float_table(self):
        with pytest.raises(ValueError, match="integer array"):
            HolidayCalendar(
                holidays=jnp.asarray([1.0, 2.0]),
                business_days_cumulative=jnp.zeros(
                    ordinal(2030, 12, 31) - ordinal(2020, 1, 1) + 2,
                    dtype=jnp.int32,
                ),
                name="X",
                first_year=2020,
                last_year=2030,
                first_ordinal=ordinal(2020, 1, 1),
                last_ordinal=ordinal(2030, 12, 31),
                max_closure_run=3,
            )

    def test_rejects_a_mis_sized_dense_table(self):
        """A length mismatch would offset every gather, so it is structural."""
        with pytest.raises(ValueError, match="one entry per covered day"):
            HolidayCalendar(
                holidays=jnp.asarray([ordinal(2025, 1, 1)], dtype=jnp.int32),
                business_days_cumulative=jnp.zeros(5, dtype=jnp.int32),
                name="X",
                first_year=2020,
                last_year=2030,
                first_ordinal=ordinal(2020, 1, 1),
                last_ordinal=ordinal(2030, 12, 31),
                max_closure_run=3,
            )

    def test_rejects_a_weekend_in_the_table(self):
        """Weekends are arithmetic; listing one would double-count it."""
        saturday = ordinal(2025, 5, 31)
        assert dt.date(2025, 5, 31).weekday() == 5
        with pytest.raises(ValueError, match="weekend date"):
            HolidayCalendar.build("X", [saturday], 2025, 2025)

    def test_rejects_holidays_outside_the_declared_range(self):
        with pytest.raises(ValueError, match="not all inside the declared"):
            HolidayCalendar.build("X", [ordinal(2019, 7, 4)], 2025, 2030)

    def test_unknown_market_raises(self):
        with pytest.raises(ValueError, match="Unknown market calendar"):
            market_calendar("MARS")

    def test_factories_are_cached(self):
        assert target_calendar() is target_calendar()


class TestFreeFunctionAdjust:
    def test_unadjusted_needs_no_calendar(self):
        value = ordinal(2025, 5, 31)
        assert int(adjust(jnp.int32(value), "unadjusted")) == value

    def test_other_conventions_require_a_calendar(self):
        with pytest.raises(ValueError, match="needs a calendar"):
            adjust(jnp.int32(ordinal(2025, 5, 31)), "following")

    def test_delegates_to_the_calendar(self):
        cal = target_calendar()
        value = jnp.int32(ordinal(2025, 5, 31))
        assert int(adjust(value, "modified_following", cal)) == int(
            cal.adjust(value, "modified_following")
        )


# ──────────────────────────────────────────────────────────────────────
# 7. Schedule integration
# ──────────────────────────────────────────────────────────────────────


class TestScheduleAdjustment:
    def test_adjusted_schedule_differs_from_the_unadjusted_one(self):
        """Anti-vacuity at the schedule level, on dates that really move."""
        plain = generate_schedule(2025, 1, 31, 2027, 1, 31, frequency=2)
        adjusted = generate_schedule(
            2025,
            1,
            31,
            2027,
            1,
            31,
            frequency=2,
            conventions=ScheduleConventions(
                business_day_convention="modified_following",
                calendar=target_calendar(),
            ),
        )
        assert plain.accrual_end.tolist() != adjusted.accrual_end.tolist()
        assert [to_date(o) for o in adjusted.accrual_end.tolist()] == [
            dt.date(2025, 7, 31),
            dt.date(2026, 1, 30),   # 31 Jan 2026 is a Saturday
            dt.date(2026, 7, 31),
            dt.date(2027, 1, 29),   # 31 Jan 2027 is a Sunday
        ]

    def test_every_adjusted_date_is_a_business_day(self):
        cal = target_calendar()
        schedule = generate_schedule(
            2025,
            1,
            31,
            2035,
            1,
            31,
            frequency=4,
            conventions=ScheduleConventions(
                business_day_convention="modified_following", calendar=cal
            ),
        )
        for value in schedule.accrual_end.tolist():
            assert bool(cal.is_business_day(jnp.int32(value)))

    def test_adjustment_preserves_contiguity(self):
        schedule = generate_schedule(
            2025,
            1,
            31,
            2030,
            1,
            31,
            frequency=4,
            conventions=ScheduleConventions(
                business_day_convention="modified_following",
                calendar=target_calendar(),
            ),
        )
        starts = schedule.accrual_start.tolist()
        ends = schedule.accrual_end.tolist()
        assert starts[1:] == ends[:-1]

    def test_payment_dates_track_the_adjusted_accrual_end(self):
        schedule = generate_schedule(
            2025,
            1,
            31,
            2027,
            1,
            31,
            frequency=2,
            conventions=ScheduleConventions(
                business_day_convention="modified_following",
                calendar=target_calendar(),
            ),
        )
        assert schedule.payment_dates.tolist() == schedule.accrual_end.tolist()

    def test_dates_outside_the_calendar_range_raise(self):
        with pytest.raises(ValueError, match="covers"):
            generate_schedule(
                2096,
                1,
                31,
                2101,
                1,
                31,
                frequency=1,
                conventions=ScheduleConventions(
                    business_day_convention="following",
                    calendar=target_calendar(),
                ),
            )


# ──────────────────────────────────────────────────────────────────────
# 8. Regeneration
# ──────────────────────────────────────────────────────────────────────


def test_tables_match_a_fresh_generation():
    """The artifact is what the generator produces from today's upstreams.

    Weakest of the three reference kinds, and last for that reason: it
    shares an upstream with the data, so it detects a stale or corrupted
    artifact but says nothing about whether the upstream is right.  Skipped
    unless the generator's own extra is installed.
    """
    pytest.importorskip("holidays")
    pytest.importorskip("pandas_market_calendars")

    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "generate_holiday_tables.py"
    )
    spec = importlib.util.spec_from_file_location("_gen_holidays", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for market in module.MARKETS:
        dates = market.loader(market.first_year, market.last_year)
        for excluded in module._EXCLUSIONS.get(market.key, {}):
            dates.discard(excluded)
        expected = sorted((d - module.EPOCH).days for d in dates)
        assert getattr(_holiday_data, market.key) == tuple(expected), market.key


@pytest.mark.audit
def test_provenance_is_recorded():
    """N-6 gate: version-stamped reference data, not anonymous literals."""
    assert _holiday_data.CALENDAR_DATA_VERSION
    assert set(_holiday_data.PROVENANCE) == set(_holiday_data.COVERED_YEARS)
    for market in _holiday_data.COVERED_YEARS:
        assert _holiday_data.PROVENANCE[market]
        assert market in dir(_holiday_data)
