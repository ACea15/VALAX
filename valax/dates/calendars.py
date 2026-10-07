"""Business-day calendars and date adjustment on integer ordinals.

A trade settles on business days.  A period end that lands on a Sunday is
paid on some adjacent working day, and *which* one is a convention, not a
detail: it moves the discount factor, the accrual length and therefore the
coupon.  This module supplies the calendar half of that, and
:func:`~valax.dates.schedule.generate_schedule` applies it.

Design
------
Three properties drive the whole module:

1. **Holidays are integers.**  A calendar is an :class:`equinox.Module`
   carrying one sorted ``jnp.int32`` array of day ordinals (days since
   1970-01-01, the VALAX epoch).  No Python :mod:`datetime` ever crosses
   into traced code; conversion happens at the user boundary with
   :func:`~valax.dates.daycounts.ymd_to_ordinal`.
2. **Lookups are binary searches, not dict hits.**  ``jnp.searchsorted``
   over the sorted array is JIT-compatible and vectorises, which is what
   lets :func:`~valax.dates.daycounts.bus_252` count business days *inside*
   a traced pricer rather than only at the boundary.
3. **Unknown is not the same as open.**  Every calendar states the year
   range its data covers, and a query outside that range raises instead of
   quietly reporting "business day".  Silently treating an uncovered year as
   holiday-free is the same failure mode
   :mod:`valax.dates.conventions` exists to prevent, one layer down.

The weekend rule is arithmetic, not data: ordinal 0 is Thursday
1 January 1970, so ``(ordinal + 3) % 7`` is the ISO weekday with Monday
zero, and 5 or 6 is a weekend.  Storing weekend closures in the tables
would therefore be redundant, and
:mod:`valax.dates._holiday_data` omits them.

Data
----
The per-market tables live in :mod:`valax.dates._holiday_data`, a generated
artifact: see ``scripts/generate_holiday_tables.py`` for the upstream
sources, the cross-checks each table had to pass, and why the data is
checked in rather than fetched at runtime.

Example:
    >>> from valax.dates import nyse_calendar, ymd_to_ordinal
    >>> cal = nyse_calendar()
    >>> independence_day = int(ymd_to_ordinal(2026, 7, 4))   # a Saturday
    >>> int(cal.adjust(independence_day, "modified_following"))
    20639

References:
    ISDA 2006 Definitions §4.12 (business day conventions).
"""

from functools import lru_cache
from typing import Iterable, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Bool, Int
from jax import Array

from valax.dates import _holiday_data
from valax.dates.conventions import BUSINESS_DAY_CONVENTIONS

# ``_ordinal_to_ymd`` is private to ``daycounts`` because it is a vectorised
# internal, but the modified-* conventions need the month of an ordinal and
# re-deriving the Hinnant conversion here would mean maintaining it twice.
from valax.dates.daycounts import _ordinal_to_ymd, ymd_to_ordinal

#: Ordinal 0 is Thursday 1 January 1970.  Adding 3 makes Monday zero, so
#: ``(ordinal + 3) % 7 >= 5`` is the weekend test.
_WEEKDAY_OFFSET = 3

#: Days in each month, January first; February is corrected for leap years.
_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

#: ``ordinal % 7`` of a Saturday and a Sunday.  Ordinal 2 is Saturday
#: 3 January 1970.
_SATURDAY_MOD = 2
_SUNDAY_MOD = 3


def weekday(ordinal: Int[Array, "..."]) -> Int[Array, "..."]:
    """ISO weekday of an ordinal, Monday zero through Sunday six.

    Args:
        ordinal: Day ordinals (days since 1970-01-01).

    Returns:
        The weekday index, matching :meth:`datetime.date.weekday`.
    """
    return (jnp.asarray(ordinal) + _WEEKDAY_OFFSET) % 7


def is_weekend(ordinal: Int[Array, "..."]) -> Bool[Array, "..."]:
    """Whether an ordinal falls on a Saturday or Sunday.

    Args:
        ordinal: Day ordinals.

    Returns:
        Boolean mask, ``True`` on weekends.
    """
    return weekday(ordinal) >= 5


def _count_weekday_matches(
    start: Int[Array, "..."], end: Int[Array, "..."], residue: int
) -> Int[Array, "..."]:
    """Count days in ``[start, end)`` congruent to ``residue`` modulo 7.

    Closed form rather than a loop, so the cost is independent of the span.
    """
    first = start + ((residue - start) % 7)
    return jnp.where(first >= end, 0, (end - first + 6) // 7)


class HolidayCalendar(eqx.Module):
    """Holiday calendar for one market, as a pytree of integer ordinals.

    Construct via :meth:`build` or one of the per-market factories
    (:func:`target_calendar`, :func:`nyse_calendar`, :func:`sofr_calendar`,
    :func:`london_calendar`, :func:`tokyo_calendar`,
    :func:`brazil_calendar`), which cache their result.  The direct
    constructor is available but requires a correct
    ``max_closure_run``; :meth:`build` computes it.

    Attributes:
        holidays: Strictly increasing weekday closures, as day ordinals.
            Weekend closures are absent by construction -- the weekend is
            handled arithmetically -- so this array lists only the days that
            would otherwise have been working days.
        name: Market label, used in error messages.
        first_year: First calendar year the data covers.
        last_year: Last calendar year the data covers, inclusive.
        first_ordinal: 1 January of ``first_year``, as an ordinal.
        last_ordinal: 31 December of ``last_year``, as an ordinal.  Held
            as a static integer rather than derived on demand because
            :meth:`assert_covered` runs inside traced code, and under a
            trace :func:`~valax.dates.daycounts.ymd_to_ordinal` returns a
            tracer that cannot be read back as a Python ``int``.
        max_closure_run: Longest run of consecutive non-business days
            anywhere in the covered range.  Bounds the roll loop in
            :meth:`adjust`, which is what makes adjustment a fixed-cost
            traced operation rather than an unbounded ``while``.

    Note:
        ``holidays`` is a JAX array and therefore a pytree leaf, but it
        carries integer ordinals and enters no gradient path -- nothing
        differentiates with respect to a date.  It is a leaf rather than a
        static field because a ``jnp`` array is unhashable, so marking it
        static would break the moment a calendar crossed a ``filter_jit``
        boundary.
    """

    holidays: Int[Array, " n_holidays"]
    name: str = eqx.field(static=True)
    first_year: int = eqx.field(static=True)
    last_year: int = eqx.field(static=True)
    first_ordinal: int = eqx.field(static=True)
    last_ordinal: int = eqx.field(static=True)
    max_closure_run: int = eqx.field(static=True)

    def __check_init__(self) -> None:
        """Validate structure at construction.

        Runs on every construction *and* on every pytree unflattening, so
        a calendar that survives a ``jit`` round trip is re-checked.  That
        is also why only *structural* properties are inspected here —
        shape, dtype and the static fields.  Inside ``jit`` the holiday
        array is a tracer, and asking whether a tracer is sorted would
        force concretisation and fail.  The value check therefore lives in
        :meth:`build`, which is eager.  Same division as
        :class:`~valax.curves.graph.CurveSpec`, which validates its
        identifier and interpolation mode but never its ``pillar_dates``.
        """
        if self.holidays.ndim != 1:
            raise ValueError(
                f"{self.name}: holidays must be 1-D, got shape "
                f"{self.holidays.shape}."
            )
        if not jnp.issubdtype(self.holidays.dtype, jnp.integer):
            raise ValueError(
                f"{self.name}: holidays must be an integer array, got "
                f"{self.holidays.dtype}."
            )
        if self.first_year > self.last_year:
            raise ValueError(
                f"{self.name}: first_year {self.first_year} exceeds "
                f"last_year {self.last_year}."
            )
        if self.max_closure_run < 1:
            raise ValueError(
                f"{self.name}: max_closure_run must be at least 1 (a "
                f"weekend alone is 2), got {self.max_closure_run}."
            )

    # ── construction ─────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        name: str,
        ordinals: Iterable[int],
        first_year: int,
        last_year: int,
    ) -> "HolidayCalendar":
        """Build a calendar, computing ``max_closure_run`` from the data.

        The roll bound is measured rather than assumed: the covered range is
        scanned once, eagerly, for the longest stretch of consecutive
        non-business days.  A hard-coded constant would be a guess about
        every market's worst case; this is the actual worst case for *this*
        data.

        Args:
            name: Market label.
            ordinals: Weekday closures as day ordinals, in any order; sorted
                and de-duplicated here.
            first_year: First year the data covers.
            last_year: Last year covered, inclusive.

        Returns:
            The constructed calendar.
        """
        unique = sorted(set(int(o) for o in ordinals))
        if not unique:
            raise ValueError(
                f"{name}: a calendar with no holidays at all is almost "
                "certainly a loading error; pass an explicit weekend-only "
                "calendar if that is really what you mean."
            )
        holidays = jnp.asarray(unique, dtype=jnp.int32)
        first_ordinal = int(ymd_to_ordinal(first_year, 1, 1))
        last_ordinal = int(ymd_to_ordinal(last_year, 12, 31))

        holiday_set = set(unique)
        longest = 0
        run = 0
        for ordinal in range(first_ordinal, last_ordinal + 1):
            closed = (ordinal + _WEEKDAY_OFFSET) % 7 >= 5 or ordinal in holiday_set
            run = run + 1 if closed else 0
            longest = max(longest, run)

        return cls(
            holidays=holidays,
            name=name,
            first_year=first_year,
            last_year=last_year,
            first_ordinal=first_ordinal,
            last_ordinal=last_ordinal,
            max_closure_run=max(longest, 1),
        )

    # ── coverage ─────────────────────────────────────────────────────

    @property
    def covered_ordinals(self) -> tuple[int, int]:
        """Inclusive ``(first, last)`` ordinal the data covers."""
        return (self.first_ordinal, self.last_ordinal)

    def require_covered(self, ordinals: Iterable[int] | int) -> None:
        """Raise if any ordinal lies outside the covered range.

        The eager, user-boundary form of the coverage check, for callers
        that already hold concrete Python integers --
        :func:`~valax.dates.schedule.generate_schedule` is one.  Traced
        callers get the same guarantee from :meth:`adjust`, which checks at
        runtime.

        Args:
            ordinals: One ordinal or an iterable of them.

        Raises:
            ValueError: If any ordinal is outside the covered range.
        """
        first, last = self.covered_ordinals
        values = [ordinals] if isinstance(ordinals, int) else [int(o) for o in ordinals]
        outside = [o for o in values if not first <= o <= last]
        if outside:
            raise ValueError(
                f"{self.name} calendar covers {self.first_year}-"
                f"{self.last_year} (ordinals {first}..{last}); "
                f"{len(outside)} date(s) fall outside, e.g. ordinal "
                f"{outside[0]}. Regenerate the tables with a wider range "
                f"(scripts/generate_holiday_tables.py) rather than treating "
                f"an uncovered year as holiday-free."
            )

    def assert_covered(self, ordinal: Int[Array, "..."]) -> Int[Array, "..."]:
        """Attach a runtime coverage check that survives ``jit`` and ``vmap``.

        The traced counterpart to :meth:`require_covered`: it returns the
        ordinal unchanged but makes an out-of-range value raise wherever the
        check runs, including inside ``jit``.

        Args:
            ordinal: Day ordinals.

        Returns:
            The same ordinals, carrying the check.
        """
        first, last = self.covered_ordinals
        outside = (ordinal < first) | (ordinal > last)
        return eqx.error_if(
            ordinal,
            outside,
            f"{self.name} calendar covers {self.first_year}-{self.last_year} "
            "only; a date outside that range cannot be adjusted. An "
            "uncovered year is unknown, not holiday-free.",
        )

    # ── queries ──────────────────────────────────────────────────────

    def is_holiday(self, ordinal: Int[Array, "..."]) -> Bool[Array, "..."]:
        """Whether an ordinal is a listed weekday closure.

        Weekends are *not* holidays by this test -- they are not in the
        table.  Use :meth:`is_business_day` for the combined question.

        Args:
            ordinal: Day ordinals.

        Returns:
            Boolean mask, ``True`` on listed closures.
        """
        ordinal = jnp.asarray(ordinal)
        n = self.holidays.shape[0]
        idx = jnp.searchsorted(self.holidays, ordinal)
        # ``idx`` is the leftmost insertion point keeping the array sorted,
        # so an exact match exists iff ``idx < n`` and the entry there is
        # equal.  Same idiom as ``FixingSeries.lookup``.
        in_range = idx < n
        safe_idx = jnp.where(in_range, idx, 0)
        return in_range & (self.holidays[safe_idx] == ordinal)

    def is_business_day(self, ordinal: Int[Array, "..."]) -> Bool[Array, "..."]:
        """Whether an ordinal is a working day: neither weekend nor holiday.

        Args:
            ordinal: Day ordinals.

        Returns:
            Boolean mask, ``True`` on business days.
        """
        ordinal = jnp.asarray(ordinal)
        return ~is_weekend(ordinal) & ~self.is_holiday(ordinal)

    def business_days_between(
        self, start: Int[Array, "..."], end: Int[Array, "..."]
    ) -> Int[Array, "..."]:
        """Count business days in the half-open interval ``[start, end)``.

        Weekdays are counted in closed form and listed closures are
        subtracted with two binary searches, so the cost does not grow with
        the span.  A reversed interval returns a negative count, matching
        the sign behaviour of the elapsed-day conventions.

        Args:
            start: Interval start ordinals, included.
            end: Interval end ordinals, excluded.

        Returns:
            Signed business-day count.
        """
        start = jnp.asarray(start)
        end = jnp.asarray(end)
        lo = jnp.minimum(start, end)
        hi = jnp.maximum(start, end)
        sign = jnp.where(end >= start, 1, -1)

        span = hi - lo
        weekends = _count_weekday_matches(lo, hi, _SATURDAY_MOD)
        weekends = weekends + _count_weekday_matches(lo, hi, _SUNDAY_MOD)
        closures = jnp.searchsorted(self.holidays, hi) - jnp.searchsorted(
            self.holidays, lo
        )
        return sign * (span - weekends - closures)

    # ── rolling ──────────────────────────────────────────────────────

    def _roll(
        self, ordinal: Int[Array, "..."], step: int
    ) -> Int[Array, "..."]:
        """Step by ``step`` until a business day is reached, or stay put.

        Unrolled ``max_closure_run`` times rather than a ``while_loop``: the
        bound is a measured property of the data (see :meth:`build`), so a
        fixed number of ``where`` operations is provably sufficient and
        keeps the operation a constant-cost, vectorisable primitive.
        """
        current = jnp.asarray(ordinal)
        for _ in range(self.max_closure_run):
            current = jnp.where(
                self.is_business_day(current), current, current + step
            )
        return current

    def roll_following(self, ordinal: Int[Array, "..."]) -> Int[Array, "..."]:
        """First business day on or after ``ordinal``.

        Args:
            ordinal: Day ordinals.

        Returns:
            The adjusted ordinals; unchanged where already a business day.
        """
        return self._roll(ordinal, 1)

    def roll_preceding(self, ordinal: Int[Array, "..."]) -> Int[Array, "..."]:
        """Last business day on or before ``ordinal``.

        Args:
            ordinal: Day ordinals.

        Returns:
            The adjusted ordinals; unchanged where already a business day.
        """
        return self._roll(ordinal, -1)

    def next_business_day(self, ordinal: Int[Array, "..."]) -> Int[Array, "..."]:
        """First business day strictly after ``ordinal``.

        Distinct from :meth:`roll_following`, which is "on or after".  The
        strict form is what a settlement or payment lag counts in.

        Args:
            ordinal: Day ordinals.

        Returns:
            The next business day after each input.
        """
        return self.roll_following(jnp.asarray(ordinal) + 1)

    def previous_business_day(
        self, ordinal: Int[Array, "..."]
    ) -> Int[Array, "..."]:
        """Last business day strictly before ``ordinal``.

        Args:
            ordinal: Day ordinals.

        Returns:
            The previous business day before each input.
        """
        return self.roll_preceding(jnp.asarray(ordinal) - 1)

    def add_business_days(
        self, ordinal: Int[Array, "..."], count: int
    ) -> Int[Array, "..."]:
        """Shift by ``count`` business days.

        Args:
            ordinal: Day ordinals.
            count: Number of business days to move; negative moves back.
                Static, because the number of steps sets the trace length.

        Returns:
            The shifted ordinals.  A zero count rolls to the nearest
            business day on or after the input, which is the market reading
            of a zero-day lag.
        """
        current = self.roll_following(ordinal)
        step = 1 if count >= 0 else -1
        for _ in range(abs(int(count))):
            current = (
                self.next_business_day(current)
                if step == 1
                else self.previous_business_day(current)
            )
        return current

    def last_business_day_of_month(
        self, ordinal: Int[Array, "..."]
    ) -> Int[Array, "..."]:
        """Last business day of the month containing ``ordinal``.

        The end-of-month rule: a date that tracks month-end must still land
        on a working day, and it may not escape into the next month.

        Args:
            ordinal: Day ordinals.

        Returns:
            The last business day of each input's month.
        """
        ordinal = jnp.asarray(ordinal)
        year, month, day = _ordinal_to_ymd(ordinal)
        leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
        length = jnp.asarray(_DAYS_IN_MONTH)[month - 1]
        length = length + jnp.where((month == 2) & leap, 1, 0)
        return self.roll_preceding(ordinal + (length - day))

    def _month_index(self, ordinal: Int[Array, "..."]) -> Int[Array, "..."]:
        """Months since year zero -- a month identity that ignores the day."""
        year, month, _ = _ordinal_to_ymd(jnp.asarray(ordinal))
        return year * 12 + month

    def adjust(
        self, ordinal: Int[Array, "..."], convention: str
    ) -> Int[Array, "..."]:
        """Apply a business-day convention to one or many ordinals.

        Args:
            ordinal: Day ordinals to adjust.
            convention: A member of
                :data:`~valax.dates.conventions.BUSINESS_DAY_CONVENTIONS`.
                A Python string, resolved at trace time.

        Returns:
            The adjusted ordinals, same shape as the input.

        Raises:
            ValueError: On an unknown convention.
            EquinoxRuntimeError: If any ordinal falls outside the covered
                year range -- including under ``jit`` and ``vmap``, since an
                uncovered year is unknown rather than holiday-free.
        """
        if convention not in BUSINESS_DAY_CONVENTIONS:
            raise ValueError(
                f"Unknown business day convention {convention!r}; expected "
                f"one of {list(BUSINESS_DAY_CONVENTIONS)}."
            )
        ordinal = jnp.asarray(ordinal)
        if convention == "unadjusted":
            # No lookup, no coverage check: an unadjusted date is returned
            # untouched, so the calendar is never consulted and a date
            # outside its range is not a question being asked.
            return ordinal

        ordinal = self.assert_covered(ordinal)
        if convention == "following":
            return self.roll_following(ordinal)
        if convention == "preceding":
            return self.roll_preceding(ordinal)

        # The modified conventions may not cross a month boundary: they take
        # the unmodified roll unless it changes month, and otherwise reverse
        # direction.  Reversing cannot itself cross out of the month, since
        # the original date is inside it.
        month = self._month_index(ordinal)
        if convention == "modified_following":
            rolled = self.roll_following(ordinal)
            fallback = self.roll_preceding(ordinal)
        else:  # modified_preceding
            rolled = self.roll_preceding(ordinal)
            fallback = self.roll_following(ordinal)
        return jnp.where(self._month_index(rolled) == month, rolled, fallback)

    # ── set algebra ──────────────────────────────────────────────────

    def union(self, other: "HolidayCalendar") -> "HolidayCalendar":
        """Days closed in *either* calendar -- the cross-currency case.

        A cross-currency payment needs both centres open, so the joint
        calendar is the union of their closures.  The covered range narrows
        to the overlap: outside it one of the two calendars has no data, and
        a union built on a guess is worse than no union.

        Args:
            other: The calendar to combine with.

        Returns:
            A new calendar over the overlapping year range.

        Raises:
            ValueError: If the covered ranges do not overlap.
        """
        return self._combine(other, intersect=False)

    def intersection(self, other: "HolidayCalendar") -> "HolidayCalendar":
        """Days closed in *both* calendars.

        The complement of :meth:`union`: useful for asking when two centres
        are simultaneously shut, rather than when a payment can settle.

        Args:
            other: The calendar to combine with.

        Returns:
            A new calendar over the overlapping year range.

        Raises:
            ValueError: If the covered ranges do not overlap.
        """
        return self._combine(other, intersect=True)

    def _combine(
        self, other: "HolidayCalendar", intersect: bool
    ) -> "HolidayCalendar":
        first_year = max(self.first_year, other.first_year)
        last_year = min(self.last_year, other.last_year)
        if first_year > last_year:
            raise ValueError(
                f"{self.name} covers {self.first_year}-{self.last_year} and "
                f"{other.name} covers {other.first_year}-{other.last_year}; "
                "the ranges do not overlap, so no combined calendar is "
                "defined."
            )
        first_ordinal = int(ymd_to_ordinal(first_year, 1, 1))
        last_ordinal = int(ymd_to_ordinal(last_year, 12, 31))

        mine = {int(o) for o in self.holidays.tolist()}
        theirs = {int(o) for o in other.holidays.tolist()}
        combined = mine & theirs if intersect else mine | theirs
        joiner = "&" if intersect else "+"
        return HolidayCalendar.build(
            f"{self.name}{joiner}{other.name}",
            (o for o in combined if first_ordinal <= o <= last_ordinal),
            first_year,
            last_year,
        )


# ──────────────────────────────────────────────────────────────────────
# Per-market factories
# ──────────────────────────────────────────────────────────────────────


def _from_table(market: str) -> HolidayCalendar:
    first_year, last_year = _holiday_data.COVERED_YEARS[market]
    return HolidayCalendar.build(
        market, getattr(_holiday_data, market), first_year, last_year
    )


@lru_cache(maxsize=None)
def target_calendar() -> HolidayCalendar:
    """EUR settlement calendar: TARGET2 closing days.

    Returns:
        The cached TARGET calendar.
    """
    return _from_table("TARGET")


@lru_cache(maxsize=None)
def nyse_calendar() -> HolidayCalendar:
    """New York Stock Exchange trading calendar.

    Not a USD *settlement* calendar: the exchange closes on Good Friday and
    opens on Columbus Day and Veterans Day, and the bond market does the
    opposite.  Use :func:`sofr_calendar` for USD rates.

    Returns:
        The cached NYSE calendar.
    """
    return _from_table("NYSE")


@lru_cache(maxsize=None)
def sofr_calendar() -> HolidayCalendar:
    """USD rates calendar: the SIFMA US government securities market.

    SOFR is published on the days this market is open, which makes it the
    right calendar for USD OIS schedules.  It is not the Fedwire calendar:
    the bond market closes on Good Friday, which is not a federal holiday,
    and federal agencies observe a Saturday holiday on the preceding Friday
    while the bond market does not.

    Returns:
        The cached SOFR calendar.
    """
    return _from_table("SOFR")


@lru_cache(maxsize=None)
def london_calendar() -> HolidayCalendar:
    """GBP settlement calendar: England and Wales bank holidays.

    Returns:
        The cached London calendar.
    """
    return _from_table("LONDON")


@lru_cache(maxsize=None)
def tokyo_calendar() -> HolidayCalendar:
    """JPY settlement calendar: national holidays plus bank holidays.

    Includes the 2-3 January and 31 December bank holidays, which are not
    national holidays but close the market.  The range stops short of 2100
    because the equinox holidays are astronomical and fixed by cabinet
    order only about a year ahead.

    Returns:
        The cached Tokyo calendar.
    """
    return _from_table("TOKYO")


@lru_cache(maxsize=None)
def brazil_calendar() -> HolidayCalendar:
    """BRL calendar: the B3 / ANBIMA financial calendar.

    The calendar Bus/252 is counted against; see
    :func:`~valax.dates.daycounts.bus_252`.

    Returns:
        The cached Brazil calendar.
    """
    return _from_table("BRAZIL")


#: Market name to factory.  Keys match :mod:`valax.dates._holiday_data`.
MARKET_CALENDARS = {
    "TARGET": target_calendar,
    "NYSE": nyse_calendar,
    "SOFR": sofr_calendar,
    "LONDON": london_calendar,
    "TOKYO": tokyo_calendar,
    "BRAZIL": brazil_calendar,
}


def market_calendar(market: str) -> HolidayCalendar:
    """Look up a per-market calendar by name.

    Args:
        market: A key of :data:`MARKET_CALENDARS`, e.g. ``"TARGET"``.

    Returns:
        The cached calendar for that market.

    Raises:
        ValueError: On an unknown market name.
    """
    try:
        return MARKET_CALENDARS[market]()
    except KeyError:
        raise ValueError(
            f"Unknown market calendar {market!r}; expected one of "
            f"{sorted(MARKET_CALENDARS)}."
        ) from None


def adjust(
    ordinal: Int[Array, "..."],
    convention: str,
    calendar: Optional[HolidayCalendar] = None,
) -> Int[Array, "..."]:
    """Apply a business-day convention, requiring a calendar to do so.

    The free-function form of :meth:`HolidayCalendar.adjust`, for callers
    holding a possibly-``None`` calendar.  A non-trivial convention without
    a calendar raises rather than silently passing the date through: a
    convention that is configured but not applied is the failure
    :mod:`valax.dates.conventions` exists to prevent.

    Args:
        ordinal: Day ordinals to adjust.
        convention: A member of
            :data:`~valax.dates.conventions.BUSINESS_DAY_CONVENTIONS`.
        calendar: The calendar to adjust against. May be ``None`` only when
            ``convention`` is ``"unadjusted"``.

    Returns:
        The adjusted ordinals.

    Raises:
        ValueError: If ``convention`` is unknown, or is not
            ``"unadjusted"`` while ``calendar`` is ``None``.
    """
    if convention not in BUSINESS_DAY_CONVENTIONS:
        raise ValueError(
            f"Unknown business day convention {convention!r}; expected one "
            f"of {list(BUSINESS_DAY_CONVENTIONS)}."
        )
    if convention == "unadjusted":
        return jnp.asarray(ordinal)
    if calendar is None:
        raise ValueError(
            f"Business-day convention {convention!r} needs a calendar to "
            "adjust against, but none was supplied. Pass a HolidayCalendar "
            "(e.g. valax.dates.target_calendar()) or use 'unadjusted'."
        )
    return calendar.adjust(ordinal, convention)
