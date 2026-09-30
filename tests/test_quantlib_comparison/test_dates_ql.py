"""QuantLib comparison for schedules and day counts.

This file did not exist, which is why the schedule and day-count defects
survived: every other ``*_ql.py`` in this directory pins both sides to
Act/365, ``NullCalendar``, ``Unadjusted`` and integer-day-aligned dates, and
``test_fixed_income_ql.py`` hand-builds ``today + round(0.5 * i * 365)``
rather than calling :func:`~valax.dates.schedule.generate_schedule` at all.
The one place a schedule difference could show up was neutralised with a
``5e-4`` relative tolerance and a comment naming the cause.

Here the schedule *is* the thing under test, so calendars and adjustment are
still switched off (VALAX is holiday-agnostic until roadmap Tier 1.3) but
every other convention — roll day, end-of-month, stub placement, day count —
is compared against QuantLib exactly.
"""

import datetime

import jax.numpy as jnp
import pytest

import QuantLib as ql

from valax.dates.conventions import ScheduleConventions
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.dates.schedule import generate_schedule

_EPOCH = datetime.date(1970, 1, 1)
_NULL = ql.NullCalendar()

_TENORS = {
    1: ql.Period(ql.Annual),
    2: ql.Period(ql.Semiannual),
    4: ql.Period(ql.Quarterly),
    12: ql.Period(ql.Monthly),
}


def to_ql(ordinal) -> ql.Date:
    """VALAX ordinal → ``ql.Date``."""
    d = _EPOCH + datetime.timedelta(days=int(ordinal))
    return ql.Date(d.day, d.month, d.year)


def ql_schedule(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    frequency: int,
    rule=ql.DateGeneration.Backward,
    end_of_month: bool = False,
) -> list[ql.Date]:
    return list(
        ql.Schedule(
            ql.Date(start[2], start[1], start[0]),
            ql.Date(end[2], end[1], end[0]),
            _TENORS[frequency],
            _NULL,
            ql.Unadjusted,
            ql.Unadjusted,
            rule,
            end_of_month,
        )
    )


def valax_boundaries(sched) -> list[ql.Date]:
    """The schedule's period boundaries, in QuantLib's flat form."""
    starts, ends, _ = sched
    return [to_ql(starts[0])] + [to_ql(o) for o in ends]


class TestScheduleDatesMatchQuantLib:
    @pytest.mark.parametrize("day", [1, 15, 28, 30, 31])
    @pytest.mark.parametrize("frequency", [1, 2, 4])
    def test_regular_schedule(self, day, frequency):
        """Day-of-month preservation, including the day-31 clamp behaviour.

        The old generator forced every interior date to the 28th, so this
        parametrisation fails on ``day in (30, 31)`` against QuantLib.
        """
        start, end = (2025, 1, day), (2030, 1, day)
        got = valax_boundaries(
            generate_schedule(*start, *end, frequency=frequency)
        )
        assert got == ql_schedule(start, end, frequency)

    def test_monthly_across_short_months(self):
        """31 Jan → 28 Feb → 31 Mar: the clamp must not ratchet down."""
        start, end = (2025, 1, 31), (2026, 1, 31)
        got = valax_boundaries(generate_schedule(*start, *end, frequency=12))
        assert got == ql_schedule(start, end, 12)

    def test_leap_year_february(self):
        start, end = (2024, 2, 29), (2028, 2, 29)
        got = valax_boundaries(generate_schedule(*start, *end, frequency=2))
        assert got == ql_schedule(start, end, 2)

    @pytest.mark.parametrize("frequency", [1, 2, 4])
    def test_end_of_month(self, frequency):
        start, end = (2024, 2, 29), (2027, 2, 28)
        got = valax_boundaries(
            generate_schedule(
                *start, *end, frequency=frequency,
                conventions=ScheduleConventions(end_of_month=True),
            )
        )
        assert got == ql_schedule(
            start, end, frequency, end_of_month=True
        )

    def test_short_front_stub(self):
        """Backward generation leaves the odd days at the front."""
        start, end = (2025, 4, 20), (2026, 10, 15)
        got = valax_boundaries(
            generate_schedule(
                *start, *end, frequency=4,
                conventions=ScheduleConventions(stub="short_front"),
            )
        )
        assert got == ql_schedule(start, end, 4, ql.DateGeneration.Backward)

    def test_short_back_stub(self):
        """Forward generation leaves the odd days at maturity."""
        start, end = (2025, 4, 20), (2026, 10, 15)
        got = valax_boundaries(
            generate_schedule(
                *start, *end, frequency=4,
                conventions=ScheduleConventions(stub="short_back"),
            )
        )
        assert got == ql_schedule(start, end, 4, ql.DateGeneration.Forward)

    def test_sub_period_tenor_keeps_the_stub(self):
        """An 8-month semiannual tenor is one period plus a 2-month stub.

        Floor division used to return a single period and discard the stub.
        """
        start, end = (2025, 1, 15), (2025, 9, 15)
        got = valax_boundaries(generate_schedule(*start, *end, frequency=2))
        assert got == ql_schedule(start, end, 2)


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
    """Coupon *amounts*, which is where the day count actually bites."""

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
