"""Day count conventions for year fraction computation.

All functions take integer ordinals (days since epoch) and return
year fractions as JAX scalars. Pure functions, fully JIT-compatible.

Most conventions measure *elapsed* days and need nothing but their two
endpoints.  ``bus_252`` measures **business** days and therefore needs a
calendar as well, so it lives in its own registry
(:data:`CALENDAR_DAY_COUNT_FNS`) rather than being forced into the
two-argument contract of :data:`DAY_COUNT_FNS`.
"""

from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    # ``calendars`` imports this module for the ordinal conversions, so the
    # dependency can only run one way at runtime; the calendar arrives as an
    # argument instead.
    from valax.dates.calendars import HolidayCalendar


# ── Ordinal ↔ calendar helpers ───────────────────────────────────────

def _ordinal_to_ymd(ordinal: Int[Array, "..."]) -> tuple:
    """Convert ordinal (days since 1970-01-01) to (year, month, day).

    Uses the algorithm from Howard Hinnant's date library.
    """
    z = ordinal + 719468
    era = jnp.where(z >= 0, z, z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + jnp.where(mp < 10, 3, -9)
    y = y + jnp.where(m <= 2, 1, 0)
    return y, m, d


def ymd_to_ordinal(year: int, month: int, day: int) -> Int[Array, ""]:
    """Convert (year, month, day) to ordinal (days since 1970-01-01).

    Boundary utility — use Python ints at the user-facing boundary,
    then work with ordinals inside JIT-traced code.
    """
    y = year - (month <= 2)
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    m_adj = month + (-3 if month > 2 else 9)
    doy = (153 * m_adj + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return jnp.array(era * 146097 + doe - 719468, dtype=jnp.int32)


def _is_leap_year(year: Int[Array, "..."]) -> Array:
    """Proleptic Gregorian leap-year predicate, elementwise."""
    return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))


def _days_in_year(year: Int[Array, "..."]) -> Int[Array, "..."]:
    """365 or 366, elementwise."""
    return 365 + _is_leap_year(year).astype(jnp.int32)


def _jan1_ordinal(year: Int[Array, "..."]) -> Int[Array, "..."]:
    """Ordinal of 1 January of ``year``, vectorised over JAX arrays.

    The array-valued counterpart of :func:`ymd_to_ordinal` specialised to
    1 January, which is the anchor ACT/ACT needs.  ``month == 1`` forces
    the ``month <= 2`` branch, so the day-of-year term collapses to the
    constant 306.
    """
    y = year - 1
    era = y // 400
    yoe = y - era * 400
    doe = yoe * 365 + yoe // 4 - yoe // 100 + 306
    return era * 146097 + doe - 719468


def _act_act_elapsed_years(ordinal: Int[Array, "..."]) -> Float[Array, "..."]:
    """Years elapsed since 1970-01-01 under ACT/ACT ISDA, elementwise.

    ACT/ACT ISDA weights each calendar day by the reciprocal of the length
    of the year containing it, so it is *additive over days*.  That makes
    it expressible as the difference of a cumulative function, which is
    what turns an otherwise year-by-year summation into closed form:

    .. math::

        F(d) = \\bigl(y(d) - 1970\\bigr) + \\frac{\\mathrm{doy}(d)}{L(y(d))}

    Every whole calendar year contributes exactly 1.0 regardless of its
    length, so the integer part of :math:`F` is the year offset and the
    fractional part is the position within the year.  That structure is
    also what makes the inverse closed-form (see
    :func:`ordinal_from_year_fraction`).
    """
    ordinal = jnp.asarray(ordinal)
    year, _, _ = _ordinal_to_ymd(ordinal)
    day_of_year = ordinal - _jan1_ordinal(year)
    return (year - 1970).astype(jnp.float64) + day_of_year.astype(
        jnp.float64
    ) / _days_in_year(year).astype(jnp.float64)


# ── Day count conventions ────────────────────────────────────────────

def act_360(
    start: Int[Array, "..."],
    end: Int[Array, "..."],
) -> Float[Array, "..."]:
    """Actual/360 day count fraction."""
    return (end - start).astype(jnp.float64) / 360.0


def act_365(
    start: Int[Array, "..."],
    end: Int[Array, "..."],
) -> Float[Array, "..."]:
    """Actual/365 Fixed day count fraction."""
    return (end - start).astype(jnp.float64) / 365.0


def act_act(
    start: Int[Array, "..."],
    end: Int[Array, "..."],
) -> Float[Array, "..."]:
    """Actual/Actual (ISDA) day count fraction.

    Splits the period at calendar-year boundaries and divides the days
    falling in each year by that year's own length:

    .. math::

        \\tau = \\frac{d_{\\text{non-leap}}}{365}
              + \\frac{d_{\\text{leap}}}{366}

    Evaluated in closed form as the difference of the cumulative
    :func:`_act_act_elapsed_years`, so it is a single vectorised
    expression with no loop over years and is safe inside JIT.

    This replaces an earlier fixed ``/365.25`` approximation.  That
    approximation was never exact for any period: it under-measured
    periods inside non-leap years by roughly 68 bp of a year and
    over-measured leap years, which fed straight into coupon sizing.

    Args:
        start: Ordinal of the period start.
        end: Ordinal of the period end.

    Returns:
        Year fraction under ACT/ACT ISDA.

    References:
        ISDA 2006 Definitions, Section 4.16(b) "Actual/Actual (ISDA)".
    """
    return _act_act_elapsed_years(end) - _act_act_elapsed_years(start)


def thirty_360(
    start: Int[Array, "..."],
    end: Int[Array, "..."],
) -> Float[Array, "..."]:
    """30/360 (Bond Basis / US) day count fraction."""
    y1, m1, d1 = _ordinal_to_ymd(start)
    y2, m2, d2 = _ordinal_to_ymd(end)
    d1 = jnp.minimum(d1, 30)
    d2 = jnp.where(d1 >= 30, jnp.minimum(d2, 30), d2)
    return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)).astype(jnp.float64) / 360.0


def bus_252(
    start: Int[Array, "..."],
    end: Int[Array, "..."],
    calendar: "HolidayCalendar",
) -> Float[Array, "..."]:
    """Business/252 day count: Brazilian BRL convention.

    The numerator is the number of **business days** in ``[start, end)``,
    not elapsed days, so a calendar is part of the convention rather than
    an optional refinement — two different holiday calendars give two
    different year fractions for the same pair of dates.  The denominator
    is the fixed market constant 252, the conventional number of business
    days in a Brazilian year; it is not derived from the calendar, so the
    year fraction of a full year is only approximately one.

    JIT-compatible: the count is two binary searches plus closed-form
    weekend arithmetic, which is what allows this to be called from traced
    pricer code through :func:`year_fraction`.

    Args:
        start: Ordinal(s) of the period start, included.
        end: Ordinal(s) of the period end, excluded.
        calendar: Calendar defining the business days; for BRL this is
            :func:`~valax.dates.calendars.brazil_calendar`.

    Returns:
        Year fraction(s) under Bus/252.

    Raises:
        EquinoxRuntimeError: If either endpoint lies outside the
            calendar's covered year range, since an uncovered year is
            unknown rather than holiday-free.

    References:
        ANBIMA, *Caderno de Formulas* (Bus/252 accrual for BRL
        instruments).
    """
    start = calendar.assert_covered(jnp.asarray(start))
    end = calendar.assert_covered(jnp.asarray(end))
    count = calendar.business_days_between(start, end)
    return count.astype(jnp.float64) / 252.0


# ── Registry ─────────────────────────────────────────────────────────

DAY_COUNT_FNS = {
    "act_360": act_360,
    "act_365": act_365,
    "act_act": act_act,
    "30_360": thirty_360,
}

#: Conventions that need a calendar as well as two dates.  Kept separate
#: from :data:`DAY_COUNT_FNS` so that the two-argument contract every
#: elapsed-day convention satisfies stays exactly that — adding a third
#: parameter to all of them, or giving ``bus_252`` a default calendar,
#: would be a worse trade.  A default calendar in particular would be the
#: failure :mod:`valax.dates.conventions` exists to prevent: a convention
#: silently applied against reference data the caller never chose.
CALENDAR_DAY_COUNT_FNS = {
    "bus_252": bus_252,
}

#: Every day-count name :func:`year_fraction` accepts, from both
#: registries.
ALL_DAY_COUNT_NAMES = tuple(
    sorted((*DAY_COUNT_FNS, *CALENDAR_DAY_COUNT_FNS))
)

#: Denominators of the conventions that are *linear* in elapsed days.
#: ``30_360`` is deliberately absent because it depends on the calendar
#: structure of both endpoints, and ``act_act`` is deliberately absent
#: because ISDA ACT/ACT has no fixed denominator — it is still exactly
#: invertible, but via :data:`DAY_COUNT_INVERSE_FNS` rather than a
#: division.
DAY_COUNT_DENOMINATORS = {
    "act_360": 360.0,
    "act_365": 365.0,
}


def _linear_inverse(
    start: Int[Array, "..."],
    year_frac: Float[Array, "..."],
    denominator: float,
) -> Float[Array, "..."]:
    """Invert a fixed-denominator convention."""
    return (
        jnp.asarray(start, dtype=jnp.float64)
        + jnp.asarray(year_frac, dtype=jnp.float64) * denominator
    )


def _act_act_inverse(
    start: Int[Array, "..."],
    year_frac: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Invert ACT/ACT ISDA exactly.

    Because every whole calendar year contributes exactly 1.0 to the
    cumulative :func:`_act_act_elapsed_years`, the integer part of the
    target identifies the year and the fractional part locates the day
    within it — no search required.
    """
    target = _act_act_elapsed_years(start) + jnp.asarray(
        year_frac, dtype=jnp.float64
    )
    year_offset = jnp.floor(target)
    year = (1970 + year_offset).astype(jnp.int32)
    frac = target - year_offset
    return _jan1_ordinal(year).astype(jnp.float64) + frac * _days_in_year(
        year
    ).astype(jnp.float64)


#: Exact inverses of :func:`year_fraction`, keyed by convention.  A
#: convention appears here only if the round trip is exact; ``30_360``
#: is absent by construction.
DAY_COUNT_INVERSE_FNS = {
    "act_360": lambda s, t: _linear_inverse(s, t, 360.0),
    "act_365": lambda s, t: _linear_inverse(s, t, 365.0),
    "act_act": _act_act_inverse,
}


def year_fraction(
    start: Int[Array, "..."],
    end: Int[Array, "..."],
    convention: str = "act_365",
    calendar: Optional["HolidayCalendar"] = None,
) -> Float[Array, "..."]:
    """Compute year fraction between two ordinal dates.

    Args:
        start: Ordinal(s) of the period start.
        end: Ordinal(s) of the period end.
        convention: Day-count name; one of :data:`ALL_DAY_COUNT_NAMES`.
        calendar: Business-day calendar, required by the conventions in
            :data:`CALENDAR_DAY_COUNT_FNS` (``bus_252``) and unused by the
            elapsed-day conventions.

    Returns:
        Year fraction(s), broadcast over ``start`` and ``end``.

    Raises:
        ValueError: If ``convention`` is not a known day count, or if it
            needs a calendar and none was supplied.
    """
    if convention in CALENDAR_DAY_COUNT_FNS:
        if calendar is None:
            raise ValueError(
                f"Day count {convention!r} counts business days and so "
                "needs a calendar, but none was supplied. Pass "
                "calendar=valax.dates.brazil_calendar() (or the relevant "
                "market) rather than leaving the convention to guess."
            )
        return CALENDAR_DAY_COUNT_FNS[convention](start, end, calendar)
    try:
        return DAY_COUNT_FNS[convention](start, end)
    except KeyError:
        raise ValueError(
            f"Unknown day count {convention!r}; expected one of "
            f"{list(ALL_DAY_COUNT_NAMES)}."
        ) from None


def ordinal_from_year_fraction(
    start: Int[Array, "..."],
    year_frac: Float[Array, "..."],
    convention: str = "act_365",
) -> Float[Array, "..."]:
    """Inverse of :func:`year_fraction`: a year fraction back to an ordinal.

    The bridge between the two date conventions VALAX instruments use:
    equity contracts carry ``expiry`` as a **year fraction**, while
    :class:`~valax.curves.discount.DiscountCurve` is indexed by **date
    ordinals**.  Converting one to the other is a day-count question, and
    answering it with a hard-coded ``365`` is how a curve's own
    convention gets silently ignored (the defect measured in
    ``valax.risk.var._extract_short_rate``).  Always pass the curve's
    ``day_count``.

    Returns a **fractional** ordinal rather than rounding to a whole day,
    for two reasons: the round trip through :func:`year_fraction` is then
    exact, and the result stays differentiable, so a rate projected off a
    curve at this horizon still carries a gradient.  Both
    ``DiscountCurve.__call__`` and :func:`~valax.curves.discount.zero_rate`
    accept a fractional ordinal and interpolate identically to the integer
    case.  Round at the boundary if a calendar date is what you need.

    Args:
        start: Reference-date ordinal.
        year_frac: Year fraction from ``start``, in ``convention``.
        convention: Day-count name; must be one of the exactly
            invertible conventions in :data:`DAY_COUNT_INVERSE_FNS`.

    Returns:
        The fractional ordinal ``d`` such that
        ``year_fraction(start, d, convention) == year_frac``.

    Raises:
        ValueError: For ``"30_360"``, which has no closed-form inverse
            (it depends on both endpoints' calendar structure), or for an
            unknown convention.
    """
    if convention not in DAY_COUNT_INVERSE_FNS:
        if convention in DAY_COUNT_FNS:
            raise ValueError(
                f"Day count {convention!r} has no closed-form inverse: it "
                f"depends on the calendar structure of both endpoints. "
                f"Supply an explicit date ordinal instead of a year "
                f"fraction, or use one of "
                f"{sorted(DAY_COUNT_INVERSE_FNS)}."
            )
        raise ValueError(
            f"Unknown day count {convention!r}; expected one of "
            f"{sorted(DAY_COUNT_FNS)}."
        )
    return DAY_COUNT_INVERSE_FNS[convention](start, year_frac)
