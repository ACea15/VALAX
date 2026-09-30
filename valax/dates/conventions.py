"""Schedule conventions: the knobs that turn two dates into a cashflow grid.

A schedule is not determined by its endpoints alone.  Two bonds with the
same issue and maturity dates produce different accrual periods — and so
different coupons — depending on how the day-of-month rolls, whether the
odd period sits at the front or the back, and whether a month-end date
tracks month-end.  Those choices live here, in one frozen pytree, rather
than as positional arguments scattered across call sites.

Tier 1.3 (business calendars) plugs in at :attr:`ScheduleConventions.calendar`
and :attr:`ScheduleConventions.business_day_convention`.  Both fields exist
now and are validated now, but :func:`~valax.dates.schedule.generate_schedule`
does not yet consume them: supplying a calendar raises rather than silently
producing unadjusted dates.
"""

from typing import Any, Optional

import equinox as eqx

from valax.dates.daycounts import DAY_COUNT_FNS

#: How the odd period is placed when the tenor is not a whole number of
#: coupon periods.
#:
#: * ``"short_front"`` — a stub shorter than a regular period, first.
#: * ``"long_front"`` — the odd days are merged into the first regular
#:   period, making it longer than a period.
#: * ``"short_back"`` / ``"long_back"`` — the same two, at maturity.
#: * ``"none"`` — assert the tenor divides evenly; raise if it does not.
STUB_KINDS = ("none", "short_front", "long_front", "short_back", "long_back")

#: Coupon frequencies expressible as a whole number of months.
VALID_FREQUENCIES = (1, 2, 3, 4, 6, 12)

#: Business-day adjustment rules.  Reserved for Tier 1.3; only
#: ``"unadjusted"`` is currently implemented.
BUSINESS_DAY_CONVENTIONS = (
    "unadjusted",
    "following",
    "modified_following",
    "preceding",
    "modified_preceding",
)


class ScheduleConventions(eqx.Module):
    """The convention set governing schedule generation and accrual.

    All fields are static: they are Python-level configuration read at
    trace time, never differentiable leaves.

    Attributes:
        day_count: Day-count name used to convert accrual periods to year
            fractions; a key of :data:`~valax.dates.daycounts.DAY_COUNT_FNS`.
        stub: Placement of the odd period; one of :data:`STUB_KINDS`.
        end_of_month: If ``True``, every generated date rolls to the last
            day of its own month, so a 28 Feb boundary is followed by
            31 Aug rather than 28 Aug. If ``False``, the day-of-month is
            preserved and clamped only where the target month is too
            short (31 Jan → 28 Feb → 31 Mar, anchored on the roll day,
            never ratcheting down).
        roll_day: Day-of-month anchor. ``None`` infers it from the
            generation anchor, which is the market convention.
        calendar: **Tier 1.3 slot.** Holiday calendar to adjust payment
            dates against. Must currently be ``None``.
        business_day_convention: **Tier 1.3 slot.** Adjustment rule; must
            currently be ``"unadjusted"``.

    Raises:
        ValueError: On an unknown day count, stub kind, business-day
            convention, or an out-of-range ``roll_day``.
        NotImplementedError: If a calendar or a non-trivial business-day
            convention is supplied, since Tier 1.3 has not landed.
    """

    day_count: str = eqx.field(static=True, default="act_365")
    stub: str = eqx.field(static=True, default="short_front")
    end_of_month: bool = eqx.field(static=True, default=False)
    roll_day: Optional[int] = eqx.field(static=True, default=None)
    calendar: Optional[Any] = eqx.field(static=True, default=None)
    business_day_convention: str = eqx.field(
        static=True, default="unadjusted"
    )

    def __post_init__(self) -> None:
        if self.day_count not in DAY_COUNT_FNS:
            raise ValueError(
                f"Unknown day count {self.day_count!r}; expected one of "
                f"{sorted(DAY_COUNT_FNS)}."
            )
        if self.stub not in STUB_KINDS:
            raise ValueError(
                f"Unknown stub kind {self.stub!r}; expected one of "
                f"{list(STUB_KINDS)}."
            )
        if self.roll_day is not None and not 1 <= self.roll_day <= 31:
            raise ValueError(
                f"roll_day must be in 1..31, got {self.roll_day}."
            )
        if self.business_day_convention not in BUSINESS_DAY_CONVENTIONS:
            raise ValueError(
                f"Unknown business day convention "
                f"{self.business_day_convention!r}; expected one of "
                f"{list(BUSINESS_DAY_CONVENTIONS)}."
            )
        # ── Tier 1.3 slot ────────────────────────────────────────────
        # Accepting a calendar and ignoring it would reintroduce exactly
        # the failure this module exists to prevent: a convention that is
        # configured but not applied. Fail loudly until 1.3 fills it in.
        if self.calendar is not None:
            raise NotImplementedError(
                "Business-day calendars are not implemented yet (roadmap "
                "Tier 1.3). generate_schedule would ignore this calendar "
                "and return unadjusted dates, so it is rejected rather "
                "than silently dropped."
            )
        if self.business_day_convention != "unadjusted":
            raise NotImplementedError(
                f"Business-day adjustment {self.business_day_convention!r} "
                f"is not implemented yet (roadmap Tier 1.3). Only "
                f"'unadjusted' is currently honoured."
            )
