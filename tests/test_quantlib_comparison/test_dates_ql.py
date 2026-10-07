"""QuantLib comparison for schedules and day counts.

This file did not exist, which is why the schedule and day-count defects
survived: every other ``*_ql.py`` in this directory pins both sides to
Act/365, ``NullCalendar``, ``Unadjusted`` and integer-day-aligned dates, and
``test_fixed_income_ql.py`` hand-builds ``today + round(0.5 * i * 365)``
rather than calling :func:`~valax.dates.schedule.generate_schedule` at all.
The one place a schedule difference could show up was neutralised with a
``5e-4`` relative tolerance and a comment naming the cause.

Here the schedule *is* the thing under test, so every convention — roll day,
end-of-month, stub placement, day count, **holiday calendar and business-day
adjustment** — is compared against QuantLib exactly.

This file used to pin both sides to ``NullCalendar`` / ``Unadjusted`` at
module level, because VALAX was holiday-agnostic and leaving QuantLib
adjusting would have failed on a missing feature rather than testing roll
day, end-of-month, stubs and day counts.  Adjustment has since landed
(``valax.dates.calendars``), so the pin is gone.  Rather than simply flip
it, every schedule test now runs under **both** setups in ``_SETUPS``: the
original unadjusted pairing is kept as a regression that the default path
has not moved, and TARGET / modified-following is the new gate.

Dates are compared by exact ``ql.Date`` equality throughout.  A date is a
whole number of days, so a tolerance is never the right assertion — the
same policy ``tests/test_dates/test_audit_conventions.py`` states for T-10.
"""

import datetime
from typing import NamedTuple, Optional

import jax.numpy as jnp
import pytest

import QuantLib as ql

from valax.dates.calendars import HolidayCalendar, target_calendar
from valax.dates.conventions import ScheduleConventions
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.dates.schedule import generate_schedule

_EPOCH = datetime.date(1970, 1, 1)
_NULL = ql.NullCalendar()
_TARGET = ql.TARGET()

_TENORS = {
    1: ql.Period(ql.Annual),
    2: ql.Period(ql.Semiannual),
    4: ql.Period(ql.Quarterly),
    12: ql.Period(ql.Monthly),
}


class Setup(NamedTuple):
    """One calendar/convention pairing, expressed for both libraries."""

    label: str
    ql_calendar: ql.Calendar
    ql_convention: int
    valax_calendar: Optional[HolidayCalendar]
    valax_convention: str


#: QuantLib takes the business-day convention twice, once for the regular
#: dates and once for the termination date; VALAX has a single convention and
#: adjusts the maturity with it, so both QuantLib arguments are set alike.
_SETUPS = [
    Setup("unadjusted", _NULL, ql.Unadjusted, None, "unadjusted"),
    Setup(
        "target_modified_following",
        _TARGET,
        ql.ModifiedFollowing,
        target_calendar(),
        "modified_following",
    ),
]


def conventions(setup: Setup, **kwargs) -> ScheduleConventions:
    """VALAX conventions for a setup, with the calendar pairing attached."""
    return ScheduleConventions(
        calendar=setup.valax_calendar,
        business_day_convention=setup.valax_convention,
        **kwargs,
    )


def to_ql(ordinal) -> ql.Date:
    """VALAX ordinal → ``ql.Date``."""
    d = _EPOCH + datetime.timedelta(days=int(ordinal))
    return ql.Date(d.day, d.month, d.year)


def ql_schedule(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    frequency: int,
    setup: Setup,
    rule=ql.DateGeneration.Backward,
    end_of_month: bool = False,
) -> list[ql.Date]:
    return list(
        ql.Schedule(
            ql.Date(start[2], start[1], start[0]),
            ql.Date(end[2], end[1], end[0]),
            _TENORS[frequency],
            setup.ql_calendar,
            setup.ql_convention,
            setup.ql_convention,
            rule,
            end_of_month,
        )
    )


def valax_boundaries(sched) -> list[ql.Date]:
    """The schedule's period boundaries, in QuantLib's flat form.

    QuantLib returns one flat list of dates, so the comparison is against
    VALAX's accrual leg.  Dropping the payment leg silently would be a hole
    now that adjustment exists, so :func:`assert_payment_leg_tracks_accrual`
    accounts for it explicitly instead.
    """
    starts, ends, _ = sched
    return [to_ql(starts[0])] + [to_ql(o) for o in ends]


def assert_payment_leg_tracks_accrual(sched) -> None:
    """The third array is not dropped, it is asserted equal.

    ``generate_schedule`` takes no payment lag, so adjustment moves an
    accrual end and its payment together.  QuantLib's flat date list has no
    separate payment leg to compare against, so the invariant is pinned here
    rather than left unexamined.
    """
    _, ends, pays = sched
    assert pays.tolist() == ends.tolist()


@pytest.mark.parametrize("setup", _SETUPS, ids=[s.label for s in _SETUPS])
class TestScheduleDatesMatchQuantLib:
    """Every case runs unadjusted *and* TARGET / modified-following."""

    @pytest.mark.parametrize("day", [1, 15, 28, 30, 31])
    @pytest.mark.parametrize("frequency", [1, 2, 4])
    def test_regular_schedule(self, setup, day, frequency):
        """Day-of-month preservation, including the day-31 clamp behaviour.

        The old generator forced every interior date to the 28th, so this
        parametrisation fails on ``day in (30, 31)`` against QuantLib.
        """
        start, end = (2025, 1, day), (2030, 1, day)
        sched = generate_schedule(
            *start, *end, frequency=frequency, conventions=conventions(setup)
        )
        assert valax_boundaries(sched) == ql_schedule(
            start, end, frequency, setup
        )
        assert_payment_leg_tracks_accrual(sched)

    def test_monthly_across_short_months(self, setup):
        """31 Jan → 28 Feb → 31 Mar: the clamp must not ratchet down."""
        start, end = (2025, 1, 31), (2026, 1, 31)
        sched = generate_schedule(
            *start, *end, frequency=12, conventions=conventions(setup)
        )
        assert valax_boundaries(sched) == ql_schedule(start, end, 12, setup)

    def test_leap_year_february(self, setup):
        start, end = (2024, 2, 29), (2028, 2, 29)
        sched = generate_schedule(
            *start, *end, frequency=2, conventions=conventions(setup)
        )
        assert valax_boundaries(sched) == ql_schedule(start, end, 2, setup)

    @pytest.mark.parametrize("frequency", [1, 2, 4])
    def test_end_of_month(self, setup, frequency):
        """The end-of-month rule *and* adjustment, which interact.

        A month-end date that is adjusted must stay month-end: the last
        business day of its own month, not merely a nearby working day.
        """
        start, end = (2024, 2, 29), (2027, 2, 28)
        sched = generate_schedule(
            *start,
            *end,
            frequency=frequency,
            conventions=conventions(setup, end_of_month=True),
        )
        assert valax_boundaries(sched) == ql_schedule(
            start, end, frequency, setup, end_of_month=True
        )

    def test_short_front_stub(self, setup):
        """Backward generation leaves the odd days at the front."""
        start, end = (2025, 4, 20), (2026, 10, 15)
        sched = generate_schedule(
            *start,
            *end,
            frequency=4,
            conventions=conventions(setup, stub="short_front"),
        )
        assert valax_boundaries(sched) == ql_schedule(
            start, end, 4, setup, ql.DateGeneration.Backward
        )

    def test_short_back_stub(self, setup):
        """Forward generation leaves the odd days at maturity."""
        start, end = (2025, 4, 20), (2026, 10, 15)
        sched = generate_schedule(
            *start,
            *end,
            frequency=4,
            conventions=conventions(setup, stub="short_back"),
        )
        assert valax_boundaries(sched) == ql_schedule(
            start, end, 4, setup, ql.DateGeneration.Forward
        )

    def test_sub_period_tenor_keeps_the_stub(self, setup):
        """An 8-month semiannual tenor is one period plus a 2-month stub.

        Floor division used to return a single period and discard the stub.
        """
        start, end = (2025, 1, 15), (2025, 9, 15)
        sched = generate_schedule(
            *start, *end, frequency=2, conventions=conventions(setup)
        )
        assert valax_boundaries(sched) == ql_schedule(start, end, 2, setup)

    def test_a_weekend_anchor_is_adjusted(self, setup):
        """1 March 2025 is a Saturday, so the two setups must disagree.

        Without a case whose dates genuinely land on non-business days,
        every assertion above would also pass against an adjustment
        function that did nothing. The anti-vacuity check is below.
        """
        start, end = (2025, 3, 1), (2027, 3, 1)
        sched = generate_schedule(
            *start, *end, frequency=2, conventions=conventions(setup)
        )
        assert valax_boundaries(sched) == ql_schedule(start, end, 2, setup)


def test_the_two_setups_actually_differ():
    """Anti-vacuity for the parametrisation itself.

    If TARGET/modified-following produced the same dates as unadjusted,
    every test above would pass twice over and prove one thing once. These
    anchors land on a Saturday, a Sunday and Easter Monday respectively.
    """
    cases = [
        ((2025, 3, 1), (2027, 3, 1), 2),      # Saturday anchor
        ((2025, 1, 31), (2030, 1, 31), 2),    # Saturday/Sunday month ends
        ((2025, 4, 21), (2028, 4, 21), 1),    # Easter Monday, a TARGET holiday
    ]
    for start, end, frequency in cases:
        unadjusted, adjusted = (
            valax_boundaries(
                generate_schedule(
                    *start, *end, frequency=frequency, conventions=conventions(s)
                )
            )
            for s in _SETUPS
        )
        assert unadjusted != adjusted, (start, end, frequency)


class TestDayCountsMatchQuantLib:
    _PAIRS = [
        ((2025, 1, 31), (2025, 7, 31)),
        ((2025, 7, 31), (2026, 1, 31)),
        ((2003, 11, 1), (2004, 5, 1)),
        ((2019, 7, 1), (2020, 7, 1)),
        ((2024, 2, 29), (2024, 8, 31)),
        ((2025, 1, 15), (2025, 3, 31)),
        ((2020, 1, 1), (2021, 1, 1)),
    ]

    _CONVENTIONS = {
        "act_360": ql.Actual360(),
        "act_365": ql.Actual365Fixed(),
        "act_act": ql.ActualActual(ql.ActualActual.ISDA),
        "30_360": ql.Thirty360(ql.Thirty360.BondBasis),
    }

    @pytest.mark.parametrize("name,ql_dc", sorted(_CONVENTIONS.items()))
    @pytest.mark.parametrize("start,end", _PAIRS)
    def test_year_fraction(self, name, ql_dc, start, end):
        """Exact agreement, not a slackened tolerance.

        ``act_act`` was a flat ``/365.25`` and untested; it disagrees with
        QuantLib's ISDA Act/Act on every one of these pairs.
        """
        got = float(
            year_fraction(
                ymd_to_ordinal(*start), ymd_to_ordinal(*end), name
            )
        )
        expected = ql_dc.yearFraction(
            ql.Date(start[2], start[1], start[0]),
            ql.Date(end[2], end[1], end[0]),
        )
        assert got == pytest.approx(expected, abs=1e-12)


class TestCouponSizingMatchesQuantLib:
    """Coupon *amounts*, which is where the day count actually bites.

    This class and :class:`TestAccruedInterestMatchesQuantLib` keep the
    unadjusted pairing on purpose, rather than having been overlooked when
    the schedule pins were flipped. What they exercise is day-count sizing
    over *unequal* periods, and the 31 January semiannual grid already
    alternates 181/184 days without any adjustment. Adjusting it would
    perturb the same code path rather than reach a new one, while costing
    the regression value of a fixed, known grid.
    """

    @pytest.mark.parametrize(
        "day_count,ql_dc",
        [
            ("act_365", ql.Actual365Fixed()),
            ("act_360", ql.Actual360()),
            ("30_360", ql.Thirty360(ql.Thirty360.BondBasis)),
        ],
    )
    def test_per_period_amounts(self, day_count, ql_dc):
        """Each period's coupon, compared cashflow by cashflow.

        Under ``c / frequency`` every amount was identical, so this test
        fails on the first 181-day period for any actual/N convention.
        """
        from valax.dates.accrual import fixed_coupon_amounts
        from valax.instruments.bonds import FixedRateBond

        starts, _, pays = generate_schedule(2025, 1, 31, 2030, 1, 31, frequency=2)
        bond = FixedRateBond(
            payment_dates=pays,
            accrual_start_dates=starts,
            settlement_date=ymd_to_ordinal(2025, 1, 31),
            coupon_rate=jnp.asarray(0.05),
            face_value=jnp.asarray(100.0),
            frequency=2,
            day_count=day_count,
        )
        got = [float(c) for c in fixed_coupon_amounts(bond)]

        ql_sched = ql.Schedule(
            ql.Date(31, 1, 2025), ql.Date(31, 1, 2030),
            ql.Period(ql.Semiannual), _NULL,
            ql.Unadjusted, ql.Unadjusted,
            ql.DateGeneration.Backward, False,
        )
        ql_bond = ql.FixedRateBond(0, 100.0, ql_sched, [0.05], ql_dc)
        # `as_coupon` returns None for the redemption, leaving only coupons.
        expected = [
            ql.as_coupon(cf).amount()
            for cf in ql_bond.cashflows()
            if ql.as_coupon(cf) is not None
        ]
        assert len(got) == len(expected)
        for mine, theirs in zip(got, expected):
            assert mine == pytest.approx(theirs, abs=1e-10)


class TestAccruedInterestMatchesQuantLib:
    """Unadjusted on purpose — see :class:`TestCouponSizingMatchesQuantLib`."""

    @pytest.mark.parametrize(
        "settle", [(2025, 1, 31), (2025, 3, 15), (2025, 4, 30), (2025, 7, 31)]
    )
    def test_accrued(self, settle):
        """Including the two coupon dates, where accrual must be exactly zero."""
        from valax.dates.accrual import bond_accrued_interest
        from valax.instruments.bonds import FixedRateBond

        starts, _, pays = generate_schedule(2025, 1, 31, 2030, 1, 31, frequency=2)
        bond = FixedRateBond(
            payment_dates=pays,
            accrual_start_dates=starts,
            settlement_date=ymd_to_ordinal(*settle),
            coupon_rate=jnp.asarray(0.05),
            face_value=jnp.asarray(100.0),
            frequency=2,
            day_count="act_365",
        )
        ql_sched = ql.Schedule(
            ql.Date(31, 1, 2025), ql.Date(31, 1, 2030),
            ql.Period(ql.Semiannual), _NULL,
            ql.Unadjusted, ql.Unadjusted,
            ql.DateGeneration.Backward, False,
        )
        ql_bond = ql.FixedRateBond(
            0, 100.0, ql_sched, [0.05], ql.Actual365Fixed()
        )
        expected = ql_bond.accruedAmount(
            ql.Date(settle[2], settle[1], settle[0])
        )
        assert float(bond_accrued_interest(bond)) == pytest.approx(
            expected, abs=1e-10
        )
