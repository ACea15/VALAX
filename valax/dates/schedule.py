"""Coupon schedule generation for fixed income instruments.

A schedule is generated **eagerly, at the user boundary** — before any
tracing — because it depends on calendar structure (month lengths, leap
years, stub placement) that is Python-level configuration rather than
differentiable data.  The arrays it returns are ordinary ordinal arrays
and are safe to close over inside JIT.

The generator returns three aligned arrays rather than one, because a
coupon is not determined by its payment date alone.  Sizing a coupon
needs the period it accrues over, and discounting it needs the date the
cash actually moves; collapsing those into a single array is what forced
every downstream pricer to guess at accrual periods.
"""

from typing import Iterator, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Int
from jax import Array

from valax.dates.conventions import VALID_FREQUENCIES, ScheduleConventions
from valax.dates.daycounts import ymd_to_ordinal

#: Sanity bound on generated periods; guards the roll loops against a
#: pathological tenor silently consuming memory.
_MAX_PERIODS = 12_000


class Schedule(eqx.Module):
    """A generated cashflow grid.

    Three aligned arrays of equal length ``n_periods``. Period ``i``
    accrues over ``[accrual_start[i], accrual_end[i])`` and pays on
    ``payment_dates[i]``.

    ``payment_dates`` equals ``accrual_end`` — both adjusted, when a
    business-day convention is configured.  It is a separate field because
    a payment *lag* moves the payment without moving the accrual it pays
    for, and every consumer that discounts a cashflow therefore reads the
    payment leg.  Generating a lag is Tier 1.4's job, not this function's:
    :class:`~valax.curves.instruments.SwapRate` and friends carry
    ``payment_lag`` and ``payment_dates`` of their own.

    Unpacks as a 3-tuple::

        accrual_start, accrual_end, payment_dates = generate_schedule(...)

    Attributes:
        accrual_start: Ordinals of each period's accrual start.
        accrual_end: Ordinals of each period's accrual end.
        payment_dates: Ordinals on which each period's cashflow settles.
    """

    accrual_start: Int[Array, " n_periods"]
    accrual_end: Int[Array, " n_periods"]
    payment_dates: Int[Array, " n_periods"]

    def __iter__(self) -> Iterator[Int[Array, " n_periods"]]:
        """Yield ``(accrual_start, accrual_end, payment_dates)``."""
        return iter((self.accrual_start, self.accrual_end, self.payment_dates))

    @property
    def n_periods(self) -> int:
        """Number of accrual periods in the schedule."""
        return int(self.payment_dates.shape[0])


# ── Calendar helpers (eager, Python-level) ───────────────────────────

def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _days_in_month(year: int, month: int) -> int:
    if month == 2:
        return 29 if _is_leap(year) else 28
    return 30 if month in (4, 6, 9, 11) else 31


def _add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    """Shift ``(year, month)`` by ``delta`` months, normalising the month."""
    index = (year * 12 + (month - 1)) + delta
    return index // 12, index % 12 + 1


def _rolled_ordinal(
    year: int, month: int, roll_day: int, end_of_month: bool
) -> int:
    """Place the roll day inside ``(year, month)`` and return its ordinal.

    Day-of-month is preserved, not truncated. It is clamped only when the
    target month is too short to hold it, and the clamp does not persist:
    the roll day is re-applied from the original anchor each period, so a
    31st-of-month bond goes 31 Jan → 28 Feb → 31 Mar rather than
    ratcheting down to the 28th for the rest of its life.
    """
    last = _days_in_month(year, month)
    day = last if end_of_month else min(roll_day, last)
    return int(ymd_to_ordinal(year, month, day))


def generate_schedule(
    start_year: int,
    start_month: int,
    start_day: int,
    end_year: int,
    end_month: int,
    end_day: int,
    frequency: int = 2,
    conventions: Optional[ScheduleConventions] = None,
) -> Schedule:
    """Generate an accrual and payment schedule.

    Periods are rolled from the anchor implied by the stub placement: a
    front stub rolls **backward** from maturity (the market convention
    for bonds, and what makes maturity land exactly on its stated day),
    a back stub rolls **forward** from the start date.

    The start date is the **first accrual start** — the dated date — and
    is inclusive; the end date is maturity. Both appear verbatim in the
    output, so a bond maturing on the 31st accrues to the 31st.

    Args:
        start_year: Year of the dated/issue date.
        start_month: Month of the dated/issue date (1-12).
        start_day: Day of the dated/issue date (1-31).
        end_year: Year of maturity.
        end_month: Month of maturity (1-12).
        end_day: Day of maturity (1-31).
        frequency: Coupons per year; one of
            :data:`~valax.dates.conventions.VALID_FREQUENCIES`.
        conventions: Stub placement, end-of-month rule, roll-day anchor,
            holiday calendar and business-day convention. Defaults to a
            short front stub with day-of-month preservation and **no**
            adjustment.

    Returns:
        A :class:`Schedule` of ``n_periods`` accrual starts, accrual ends
        and payment dates, ascending and contiguous
        (``accrual_start[i + 1] == accrual_end[i]``).

        With ``conventions.business_day_convention == "unadjusted"`` (the
        default) the dates are the raw roll-day grid, bit for bit as they
        were before adjustment existed. Otherwise every boundary is moved
        onto a business day of ``conventions.calendar``.

    Raises:
        ValueError: If ``frequency`` is not a whole number of months, if
            the start date is not strictly before maturity, if either
            date is not a valid calendar date, if
            ``conventions.stub == "none"`` but the tenor is not a whole
            number of coupon periods, or if any generated date falls
            outside the calendar's covered year range.

    Example:
        >>> from valax.dates import generate_schedule
        >>> starts, ends, pays = generate_schedule(
        ...     2025, 1, 31, 2027, 1, 31, frequency=2
        ... )
        >>> pays.shape
        (4,)
    """
    conventions = conventions if conventions is not None else ScheduleConventions()

    if frequency not in VALID_FREQUENCIES:
        raise ValueError(
            f"frequency must be a whole number of months per period, i.e. "
            f"one of {list(VALID_FREQUENCIES)}; got {frequency}."
        )
    _validate_calendar_date(start_year, start_month, start_day, "start")
    _validate_calendar_date(end_year, end_month, end_day, "end")

    start_ordinal = int(ymd_to_ordinal(start_year, start_month, start_day))
    end_ordinal = int(ymd_to_ordinal(end_year, end_month, end_day))
    if start_ordinal >= end_ordinal:
        raise ValueError(
            f"Schedule start ({start_year}-{start_month:02d}-{start_day:02d}) "
            f"must be strictly before end "
            f"({end_year}-{end_month:02d}-{end_day:02d}); a schedule with no "
            f"accrual period is never what the caller meant."
        )

    step_months = 12 // frequency
    stub = conventions.stub
    eom = conventions.end_of_month
    roll_backward = stub in ("none", "short_front", "long_front")

    if roll_backward:
        roll_day = conventions.roll_day or end_day
        interior, on_grid = _roll_backward(
            end_year, end_month, start_ordinal, step_months, roll_day, eom
        )
    else:
        roll_day = conventions.roll_day or start_day
        interior, on_grid = _roll_forward(
            start_year, start_month, end_ordinal, step_months, roll_day, eom
        )

    if not on_grid:
        if stub == "none":
            raise ValueError(
                f"Tenor is not a whole number of {step_months}-month periods, "
                f"but conventions.stub == 'none'. Choose a stub placement "
                f"explicitly (one of 'short_front', 'long_front', "
                f"'short_back', 'long_back') rather than leaving the odd "
                f"period to be silently dropped."
            )
        # A long stub is a short stub merged into its neighbouring regular
        # period, i.e. the boundary between them is simply removed.
        if stub == "long_front" and interior:
            interior = interior[1:]
        elif stub == "long_back" and interior:
            interior = interior[:-1]

    boundaries = [start_ordinal, *interior, end_ordinal]

    if conventions.business_day_convention != "unadjusted":
        # Adjustment is applied to the assembled boundary list, never during
        # the roll: the roll-day, end-of-month and stub arithmetic above
        # defines the *unadjusted* grid, and the convention then moves each
        # boundary onto a working day.  Doing it in this order is what makes
        # a 31-day roll day still mean "the 31st" on a calendar where the
        # 31st is a Sunday, and it is also QuantLib's order.
        #
        # Both accrual legs are sliced from one adjusted list, so
        # contiguity (accrual_start[i + 1] == accrual_end[i]) survives by
        # construction rather than by a second pass.
        calendar = conventions.calendar
        calendar.require_covered(boundaries)
        adjusted = calendar.adjust(
            jnp.asarray(boundaries, dtype=jnp.int32),
            conventions.business_day_convention,
        )
        if conventions.end_of_month:
            # A month-end roll must stay month-end after adjusting: the
            # last *business* day of the month, not merely a nearby one.
            # Only the interior and final boundaries follow the roll; the
            # first is the dated date the caller supplied.
            eom_adjusted = calendar.last_business_day_of_month(adjusted)
            adjusted = jnp.concatenate(
                [adjusted[:1], eom_adjusted[1:]]
            )
        boundaries = [int(o) for o in adjusted]

    starts = jnp.asarray(boundaries[:-1], dtype=jnp.int32)
    ends = jnp.asarray(boundaries[1:], dtype=jnp.int32)
    return Schedule(accrual_start=starts, accrual_end=ends, payment_dates=ends)


def _validate_calendar_date(year: int, month: int, day: int, label: str) -> None:
    """Reject impossible dates at the boundary rather than rolling them."""
    if not 1 <= month <= 12:
        raise ValueError(f"{label} month must be in 1..12, got {month}.")
    last = _days_in_month(year, month)
    if not 1 <= day <= last:
        raise ValueError(
            f"{label} date {year}-{month:02d}-{day:02d} does not exist; "
            f"{year}-{month:02d} has {last} days."
        )


def _roll_backward(
    anchor_year: int,
    anchor_month: int,
    start_ordinal: int,
    step_months: int,
    roll_day: int,
    end_of_month: bool,
) -> tuple[list[int], bool]:
    """Roll back from maturity, returning interior boundaries and exactness.

    Returns the strictly-interior period boundaries in ascending order,
    and whether the regular grid landed exactly on ``start_ordinal`` (in
    which case there is no stub).
    """
    interior: list[int] = []
    for k in range(1, _MAX_PERIODS + 1):
        year, month = _add_months(anchor_year, anchor_month, -k * step_months)
        ordinal = _rolled_ordinal(year, month, roll_day, end_of_month)
        if ordinal <= start_ordinal:
            interior.reverse()
            return interior, ordinal == start_ordinal
        interior.append(ordinal)
    raise ValueError(
        f"Schedule exceeds {_MAX_PERIODS} periods; check the dates and "
        f"frequency."
    )


def _roll_forward(
    anchor_year: int,
    anchor_month: int,
    end_ordinal: int,
    step_months: int,
    roll_day: int,
    end_of_month: bool,
) -> tuple[list[int], bool]:
    """Roll forward from the start date; mirror of :func:`_roll_backward`."""
    interior: list[int] = []
    for k in range(1, _MAX_PERIODS + 1):
        year, month = _add_months(anchor_year, anchor_month, k * step_months)
        ordinal = _rolled_ordinal(year, month, roll_day, end_of_month)
        if ordinal >= end_ordinal:
            return interior, ordinal == end_ordinal
        interior.append(ordinal)
    raise ValueError(
        f"Schedule exceeds {_MAX_PERIODS} periods; check the dates and "
        f"frequency."
    )
