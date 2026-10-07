"""Schedule conventions: the knobs that turn two dates into a cashflow grid.

A schedule is not determined by its endpoints alone.  Two bonds with the
same issue and maturity dates produce different accrual periods — and so
different coupons — depending on how the day-of-month rolls, whether the
odd period sits at the front or the back, and whether a month-end date
tracks month-end.  Those choices live here, in one frozen pytree, rather
than as positional arguments scattered across call sites.

Business-day adjustment enters at :attr:`ScheduleConventions.calendar` and
:attr:`ScheduleConventions.business_day_convention`, and
:func:`~valax.dates.schedule.generate_schedule` applies both.  The two are
validated *as a pair*: a convention cannot be configured without a calendar
to apply it against, and a calendar cannot be supplied where nothing would
consult it.  Either half alone is a convention that looks configured and is
not, which is the failure this module exists to prevent.
"""

from typing import TYPE_CHECKING, Optional

import equinox as eqx

from valax.dates.daycounts import (
    ALL_DAY_COUNT_NAMES,
    CALENDAR_DAY_COUNT_FNS,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    # ``calendars`` imports this module for BUSINESS_DAY_CONVENTIONS, so the
    # concrete class is imported lazily inside ``__post_init__`` instead.
    from valax.dates.calendars import HolidayCalendar

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

#: Business-day adjustment rules, applied by
#: :meth:`~valax.dates.calendars.HolidayCalendar.adjust`.  The ``modified_*``
#: pair behaves like its plain counterpart unless the roll would change
#: month, in which case it reverses direction.
BUSINESS_DAY_CONVENTIONS = (
    "unadjusted",
    "following",
    "modified_following",
    "preceding",
    "modified_preceding",
)


class ScheduleConventions(eqx.Module):
    """The convention set governing schedule generation and accrual.

    Every field except ``calendar`` is static: Python-level configuration
    read at trace time, never a differentiable leaf.  ``calendar`` is an
    ordinary pytree field because a :class:`jax.Array` is unhashable and a
    static field must be hashable, so marking it static would break the
    moment a convention set crossed a ``filter_jit`` boundary.  Its holiday
    ordinals are a leaf that enters no gradient path.

    Attributes:
        day_count: Day-count name used to convert accrual periods to year
            fractions; one of
            :data:`~valax.dates.daycounts.ALL_DAY_COUNT_NAMES`.  The
            business-day conventions in
            :data:`~valax.dates.daycounts.CALENDAR_DAY_COUNT_FNS`
            (``bus_252``) additionally require ``calendar``.
        stub: Placement of the odd period; one of :data:`STUB_KINDS`.
        end_of_month: If ``True``, every generated date rolls to the last
            day of its own month, so a 28 Feb boundary is followed by
            31 Aug rather than 28 Aug. If ``False``, the day-of-month is
            preserved and clamped only where the target month is too
            short (31 Jan → 28 Feb → 31 Mar, anchored on the roll day,
            never ratcheting down).
        roll_day: Day-of-month anchor. ``None`` infers it from the
            generation anchor, which is the market convention.
        calendar: Holiday calendar that ``business_day_convention`` and a
            ``bus_252`` day count are applied against; one of the market
            factories in :mod:`valax.dates.calendars`. ``None`` means no
            calendar, which is only consistent with an unadjusted,
            elapsed-day convention set.
        business_day_convention: Adjustment rule applied to every generated
            date; one of :data:`BUSINESS_DAY_CONVENTIONS`. The default
            ``"unadjusted"`` leaves dates exactly where the roll-day
            arithmetic put them.

    Raises:
        ValueError: On an unknown day count, stub kind, business-day
            convention, or an out-of-range ``roll_day``; on a ``calendar``
            that is not a
            :class:`~valax.dates.calendars.HolidayCalendar`; or on any
            calendar/convention pairing where one half would be
            configured and never applied.
    """

    day_count: str = eqx.field(static=True, default="act_365")
    stub: str = eqx.field(static=True, default="short_front")
    end_of_month: bool = eqx.field(static=True, default=False)
    roll_day: Optional[int] = eqx.field(static=True, default=None)
    calendar: Optional["HolidayCalendar"] = None
    business_day_convention: str = eqx.field(
        static=True, default="unadjusted"
    )

    def __post_init__(self) -> None:
        if self.day_count not in ALL_DAY_COUNT_NAMES:
            raise ValueError(
                f"Unknown day count {self.day_count!r}; expected one of "
                f"{list(ALL_DAY_COUNT_NAMES)}."
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
        # ── Calendar / convention pairing ────────────────────────────
        # A convention that is configured but not applied is the failure
        # this module exists to prevent, and there are two ways to write
        # one: name an adjustment rule with no calendar to apply it
        # against, or attach a calendar that nothing would consult. Both
        # raise. The guards survived Tier 1.3 landing; only their reason
        # changed, from "not implemented" to "incoherent".
        if self.calendar is not None:
            # Imported here rather than at module scope: ``calendars``
            # imports this module for BUSINESS_DAY_CONVENTIONS.
            from valax.dates.calendars import HolidayCalendar

            if not isinstance(self.calendar, HolidayCalendar):
                raise ValueError(
                    f"calendar must be a HolidayCalendar, got "
                    f"{type(self.calendar).__name__}. Use one of the market "
                    f"factories in valax.dates.calendars (e.g. "
                    f"target_calendar()) or HolidayCalendar.build(...)."
                )

        needs_calendar = self.day_count in CALENDAR_DAY_COUNT_FNS
        adjusts = self.business_day_convention != "unadjusted"

        if adjusts and self.calendar is None:
            raise ValueError(
                f"Business-day convention "
                f"{self.business_day_convention!r} needs a calendar to "
                f"adjust against, but calendar is None. Supplying the rule "
                f"without the data would leave the convention configured "
                f"and never applied. Pass a calendar (e.g. "
                f"valax.dates.target_calendar()) or use 'unadjusted'."
            )
        if needs_calendar and self.calendar is None:
            raise ValueError(
                f"Day count {self.day_count!r} counts business days and so "
                f"needs a calendar, but calendar is None. Pass the market's "
                f"calendar (e.g. valax.dates.brazil_calendar())."
            )
        if self.calendar is not None and not adjusts and not needs_calendar:
            raise ValueError(
                f"calendar={self.calendar.name!r} was supplied but nothing "
                f"would consult it: business_day_convention is "
                f"'unadjusted' and day_count {self.day_count!r} counts "
                f"elapsed days. Choose an adjustment rule, or drop the "
                f"calendar rather than leaving it configured and unused."
            )
