"""T-10 conformance gate: day counts and schedules vs reference data.

Verifies :mod:`valax.dates` against the multi-curve calibration / AAD risk
audit specification, requirement **T-10** ("day count and schedule spot checks
on known dates: tau and dates match reference data exactly"), which backs
requirements N-1 (day-count conventions), N-2 (schedule generation) and N-5
(accrual vs payment dates).  Collected with the rest of the T-suite via
``pytest -m audit``.

**What makes this a gate rather than more unit tests.**  The behavioural
coverage of these modules already lives in ``test_daycounts.py`` and
``test_schedule.py`` -- properties, invariants, stub placement, error paths.
This file deliberately does something different and narrower: it pins
*externally checkable reference data*.  Every expected value here is derived
from the published definition of the convention using **Python's standard
library only** (``datetime``, ``calendar``) or from integer arithmetic
transcribed from the convention's own rule.  Nothing is taken from VALAX, so
a self-consistent-but-wrong implementation cannot pass.  That independence is
the whole point: ``ymd_to_ordinal`` and ``thirty_360`` both route through
VALAX's own Hinnant-algorithm ordinal/calendar conversion, so checking one
against the other would prove only that the algorithm is used twice.

Tolerances this implementation commits to:

* ``act_360``, ``act_365``, ``30_360`` -- **exact**, bit-for-bit float
  equality against the independent reference.  Each is integer day (or
  30/360-day) arithmetic followed by a single division, so there is no reason
  to accept anything looser, and a loose tolerance here would hide a
  genuine off-by-one in the day count.
* ``act_act`` -- ``atol = 1e-14``.  **Not** exact, deliberately: see
  :class:`TestActActIsNotBitExact` for the measurement and the reason.
* Dates (ordinals, schedule starts/ends/payments) -- **exact** integer
  equality.  A date is a whole number of days; approximate equality is never
  the right assertion.
"""

from __future__ import annotations

import calendar
import datetime

import jax.numpy as jnp
import pytest

from valax.dates.conventions import ScheduleConventions
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.dates.schedule import generate_schedule


pytestmark = pytest.mark.audit


#: Exact tolerance for the conventions that admit it.
T10_EXACT = 0.0

#: ACT/ACT ISDA tolerance -- justified in :class:`TestActActIsNotBitExact`.
T10_ACT_ACT_ATOL = 1e-14

_EPOCH = datetime.date(1970, 1, 1)


# ── Independent references (standard library only) ───────────────────


def ref_ordinal(year: int, month: int, day: int) -> int:
    """Day ordinal from VALAX's epoch, via the standard library.

    The anchor for every date assertion in this file.  ``date.toordinal``
    is a proleptic Gregorian day count from year 1, so shifting it by the
    epoch's own ordinal gives VALAX's days-since-1970-01-01 convention.
    """
    return datetime.date(year, month, day).toordinal() - _EPOCH.toordinal()


def ref_act_360(start: tuple, end: tuple) -> float:
    """Actual/360 from the published definition: actual days over 360."""
    return (datetime.date(*end) - datetime.date(*start)).days / 360.0


def ref_act_365(start: tuple, end: tuple) -> float:
    """Actual/365 Fixed from the published definition."""
    return (datetime.date(*end) - datetime.date(*start)).days / 365.0


def ref_thirty_360(start: tuple, end: tuple) -> float:
    """30/360 Bond Basis, transcribed from the rule rather than computed.

    ISDA 2006 Section 4.16(f): ``D1 = min(D1, 30)``, and only *then*
    ``D2 = min(D2, 30)`` if ``D1`` was adjusted to 30.  Deliberately
    written as plain integer arithmetic on the calendar triple, with no
    ordinal conversion anywhere, so it shares nothing with the
    implementation under test.
    """
    (y1, m1, d1), (y2, m2, d2) = start, end
    d1_adj = min(d1, 30)
    d2_adj = min(d2, 30) if d1_adj >= 30 else d2
    return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2_adj - d1_adj)) / 360.0


def ref_act_act(start: tuple, end: tuple) -> float:
    """ACT/ACT ISDA from its definition: split at calendar-year boundaries.

    ISDA 2006 Section 4.16(b) defines the fraction as the days falling in
    a leap year over 366 plus the days falling in a non-leap year over
    365.  This walks the year boundaries and accumulates exactly that,
    which is a *structurally different* computation from the cumulative
    closed form in :func:`valax.dates.daycounts.act_act` -- which is why
    it is a real check, and also why the two agree only to rounding.
    """
    cursor, stop = datetime.date(*start), datetime.date(*end)
    total = 0.0
    while cursor < stop:
        boundary = min(datetime.date(cursor.year + 1, 1, 1), stop)
        days = (boundary - cursor).days
        total += days / (366.0 if calendar.isleap(cursor.year) else 365.0)
        cursor = boundary
    return total


def tau(start: tuple, end: tuple, convention: str) -> float:
    """VALAX's year fraction between two calendar triples."""
    return float(
        year_fraction(
            jnp.asarray(ref_ordinal(*start), dtype=jnp.int32),
            jnp.asarray(ref_ordinal(*end), dtype=jnp.int32),
            convention,
        )
    )


def to_ymd(ordinal) -> tuple:
    """A VALAX ordinal back to a calendar triple, via the standard library."""
    date = _EPOCH + datetime.timedelta(days=int(ordinal))
    return (date.year, date.month, date.day)


# ── N-1: ordinal conversion ──────────────────────────────────────────


#: Dates chosen to exercise every branch a Gregorian conversion can get
#: wrong: both century rules (1900 is not a leap year, 2000 is), the
#: 400-year rule at 2400, pre-epoch dates (negative ordinals), the epoch
#: itself, leap day, and the day either side of a leap day.
_ORDINAL_CASES = [
    (1900, 1, 1), (1900, 2, 28), (1900, 3, 1), (1900, 12, 31),
    (1960, 2, 29), (1969, 12, 31),
    (1970, 1, 1), (1970, 1, 2),
    (1999, 12, 31), (2000, 1, 1), (2000, 2, 28), (2000, 2, 29),
    (2000, 3, 1), (2024, 2, 28), (2024, 2, 29), (2024, 3, 1),
    (2100, 2, 28), (2100, 3, 1), (2400, 2, 29), (2401, 1, 1),
]


class TestOrdinalReferenceData:
    """``ymd_to_ordinal`` against the standard library, exactly.

    Everything downstream is indexed by these integers, so an off-by-one
    here would shift every curve pillar and every accrual period by a
    day while leaving the whole suite self-consistent.
    """

    @pytest.mark.parametrize("ymd", _ORDINAL_CASES)
    def test_matches_the_standard_library(self, ymd):
        assert int(ymd_to_ordinal(*ymd)) == ref_ordinal(*ymd)

    def test_the_epoch_is_day_zero(self):
        assert int(ymd_to_ordinal(1970, 1, 1)) == 0

    def test_pre_epoch_dates_are_negative(self):
        assert int(ymd_to_ordinal(1969, 12, 31)) == -1

    def test_every_day_of_a_leap_and_a_non_leap_year_round_trips(self):
        """Exhaustive over two whole years, so no month is untested."""
        for year in (2023, 2024):
            for month in range(1, 13):
                for day in range(1, calendar.monthrange(year, month)[1] + 1):
                    got = int(ymd_to_ordinal(year, month, day))
                    assert got == ref_ordinal(year, month, day)
                    assert to_ymd(got) == (year, month, day)


# ── N-1: day-count reference data ────────────────────────────────────


#: Periods chosen so each case can go wrong differently: a period
#: straddling a leap boundary, a whole leap year, a whole non-leap year,
#: a multi-year span, a two-day span over 29 February, and a regular
#: semi-annual period.
_TAU_CASES = [
    ((2023, 12, 1), (2024, 3, 1)),
    ((2024, 1, 1), (2025, 1, 1)),
    ((2023, 1, 1), (2024, 1, 1)),
    ((2023, 1, 1), (2023, 7, 1)),
    ((2024, 2, 28), (2024, 3, 1)),
    ((2019, 7, 15), (2024, 7, 15)),
    ((2026, 6, 30), (2026, 12, 31)),
    ((2024, 11, 15), (2025, 5, 15)),
]


class TestDayCountReferenceData:
    """tau against independently computed reference values."""

    @pytest.mark.parametrize("start,end", _TAU_CASES)
    def test_act_360_is_exact(self, start, end):
        assert tau(start, end, "act_360") == ref_act_360(start, end)

    @pytest.mark.parametrize("start,end", _TAU_CASES)
    def test_act_365_is_exact(self, start, end):
        assert tau(start, end, "act_365") == ref_act_365(start, end)

    @pytest.mark.parametrize("start,end", _TAU_CASES)
    def test_act_act_matches_the_isda_year_split(self, start, end):
        got = tau(start, end, "act_act")
        assert got == pytest.approx(
            ref_act_act(start, end), abs=T10_ACT_ACT_ATOL
        )

    @pytest.mark.parametrize(
        "start,end",
        [
            # Both endpoints on the 31st: D1 -> 30 forces D2 -> 30, so a
            # semi-annual period is exactly half a year.
            ((2024, 1, 31), (2024, 7, 31)),
            # D1 below 30 leaves D2 untruncated -- the asymmetry of the
            # Bond Basis rule, and the case a symmetric 30E/360
            # implementation would get wrong.
            ((2024, 2, 28), (2024, 3, 31)),
            # Month-end February is *not* treated as a month end: this
            # convention gives 29/360, not 30/360.
            ((2024, 1, 30), (2024, 2, 29)),
            ((2023, 12, 31), (2024, 12, 31)),
            ((2024, 3, 31), (2024, 9, 30)),
            ((2024, 8, 31), (2025, 2, 28)),
            ((2024, 5, 31), (2024, 6, 1)),
        ],
    )
    def test_thirty_360_is_exact(self, start, end):
        assert tau(start, end, "30_360") == ref_thirty_360(start, end)

    def test_a_whole_calendar_year_is_exactly_one_under_act_act(self):
        """The property that fixes ACT/ACT's denominator per year.

        Exact for both year lengths, which is what distinguishes ISDA
        ACT/ACT from the ``/365.25`` approximation it replaced.
        """
        for year in (2023, 2024):
            assert tau((year, 1, 1), (year + 1, 1, 1), "act_act") == 1.0


class TestActActIsNotBitExact:
    """Why ACT/ACT alone carries a tolerance, measured rather than assumed.

    T-10 asks for an exact match against reference data, and three of the
    four conventions deliver it.  ACT/ACT does not, and the reason is
    worth recording rather than hiding behind a tolerance.

    :func:`valax.dates.daycounts.act_act` evaluates a *cumulative*
    function and subtracts: ``F(end) - F(start)``, where ``F`` counts
    whole years plus a within-year fraction.  The ISDA definition instead
    *sums* per-year contributions.  The two are equal in exact
    arithmetic -- algebraically the whole-year terms cancel -- but they
    are different sequences of floating-point operations, so they differ
    in the last bit or two.  Neither is "wrong"; the closed form is in
    fact the better-conditioned of the two, since it has no accumulation
    over years at all.

    The measured disagreement is some two orders of magnitude inside the
    stated tolerance, and nowhere near any economically meaningful scale:
    1e-14 years is under a microsecond.
    """

    def test_the_disagreement_is_real_but_tiny(self):
        worst = max(
            abs(tau(s, e, "act_act") - ref_act_act(s, e))
            for s, e in _TAU_CASES
        )
        # Real: the two formulations are not bit-identical ...
        assert worst > 0.0
        # ... but far inside the stated tolerance.
        assert worst < T10_ACT_ACT_ATOL

    def test_the_other_conventions_need_no_tolerance_at_all(self):
        """Guards the asymmetry: only ACT/ACT gets a tolerance."""
        for start, end in _TAU_CASES:
            assert (
                abs(tau(start, end, "act_360") - ref_act_360(start, end))
                == T10_EXACT
            )
            assert (
                abs(tau(start, end, "act_365") - ref_act_365(start, end))
                == T10_EXACT
            )


# ── N-2 / N-5: schedule reference data ───────────────────────────────


#: ``(label, generate_schedule args, kwargs, expected accrual ends)``.
#: Expected dates are written out as calendar triples -- the form a
#: trade confirmation uses -- so a reader can check them against a
#: counterparty's schedule by eye, which an ordinal table does not allow.
_SCHEDULE_CASES = [
    (
        "semi-annual, five years, on-grid",
        (2024, 1, 15, 2029, 1, 15),
        {"frequency": 2},
        [
            (2024, 7, 15), (2025, 1, 15), (2025, 7, 15), (2026, 1, 15),
            (2026, 7, 15), (2027, 1, 15), (2027, 7, 15), (2028, 1, 15),
            (2028, 7, 15), (2029, 1, 15),
        ],
    ),
    (
        "quarterly, one year, IMM-style day",
        (2024, 3, 20, 2025, 3, 20),
        {"frequency": 4},
        [(2024, 6, 20), (2024, 9, 20), (2024, 12, 20), (2025, 3, 20)],
    ),
    (
        "short front stub, anchored on maturity",
        (2024, 2, 20, 2026, 1, 15),
        {"frequency": 2},
        [(2024, 7, 15), (2025, 1, 15), (2025, 7, 15), (2026, 1, 15)],
    ),
    (
        "end-of-month tracks the month end, not the day number",
        (2024, 1, 31, 2025, 1, 31),
        {
            "frequency": 4,
            "conventions": ScheduleConventions(end_of_month=True),
        },
        [(2024, 4, 30), (2024, 7, 31), (2024, 10, 31), (2025, 1, 31)],
    ),
    (
        "29 February anchor clamps without ratcheting",
        (2024, 2, 29, 2028, 2, 29),
        {"frequency": 1},
        [(2025, 2, 28), (2026, 2, 28), (2027, 2, 28), (2028, 2, 29)],
    ),
]


class TestScheduleReferenceData:
    """Generated schedules against dates written out by hand."""

    @pytest.mark.parametrize(
        "label,args,kwargs,expected_ends",
        _SCHEDULE_CASES,
        ids=[case[0] for case in _SCHEDULE_CASES],
    )
    def test_accrual_dates_match_exactly(
        self, label, args, kwargs, expected_ends
    ):
        schedule = generate_schedule(*args, **kwargs)
        assert schedule.n_periods == len(expected_ends)

        ends = [to_ymd(x) for x in schedule.accrual_end]
        assert ends == expected_ends

        # N-5: starts are the previous end, with the dated date first.
        starts = [to_ymd(x) for x in schedule.accrual_start]
        assert starts == [tuple(args[0:3])] + expected_ends[:-1]

    @pytest.mark.parametrize(
        "label,args,kwargs,expected_ends",
        _SCHEDULE_CASES,
        ids=[case[0] for case in _SCHEDULE_CASES],
    )
    def test_period_year_fractions_match_the_reference_dates(
        self, label, args, kwargs, expected_ends
    ):
        """tau *and* dates together, which is what T-10 actually asks for.

        Correct dates with a mis-wired day count still misprices a
        coupon, so the gate recomputes every period's year fraction from
        the hand-written calendar triples -- not from the schedule's own
        ordinals -- and requires exact agreement.
        """
        schedule = generate_schedule(*args, **kwargs)
        boundaries = [tuple(args[0:3])] + expected_ends

        for i in range(schedule.n_periods):
            start, end = boundaries[i], boundaries[i + 1]
            for convention in ("act_360", "act_365", "30_360"):
                got = float(
                    year_fraction(
                        schedule.accrual_start[i],
                        schedule.accrual_end[i],
                        convention,
                    )
                )
                reference = {
                    "act_360": ref_act_360,
                    "act_365": ref_act_365,
                    "30_360": ref_thirty_360,
                }[convention](start, end)
                assert got == reference

    @pytest.mark.parametrize(
        "label,args,kwargs,expected_ends",
        _SCHEDULE_CASES,
        ids=[case[0] for case in _SCHEDULE_CASES],
    )
    def test_payment_dates_are_the_accrual_ends(
        self, label, args, kwargs, expected_ends
    ):
        """N-5 as the schedule currently stands.

        :class:`~valax.dates.schedule.Schedule` has a distinct
        ``payment_dates`` array but no business-day roll to populate it
        differently, so it equals ``accrual_end`` exactly.  The gate pins
        that equality so the day a calendar lands, this test is what
        tells you the payment leg now needs its own reference data.
        Instrument-level payment lag is a separate axis and is gated by
        T-5 in ``tests/test_curves/test_instrument_residuals.py``.
        """
        schedule = generate_schedule(*args, **kwargs)
        assert [to_ymd(x) for x in schedule.payment_dates] == expected_ends
