"""Day count conventions for year fraction computation.

All functions take integer ordinals (days since epoch) and return
year fractions as JAX scalars. Pure functions, fully JIT-compatible.
"""

import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array


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
    """Actual/Actual (ISDA simplified) — uses 365.25 denominator."""
    return (end - start).astype(jnp.float64) / 365.25


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


# ── Registry ─────────────────────────────────────────────────────────

DAY_COUNT_FNS = {
    "act_360": act_360,
    "act_365": act_365,
    "act_act": act_act,
    "30_360": thirty_360,
}

#: Denominators of the conventions that are *linear* in elapsed days, and
#: therefore exactly invertible.  ``30_360`` is deliberately absent: it
#: depends on the calendar structure of both endpoints, so it has no
#: closed-form inverse.
DAY_COUNT_DENOMINATORS = {
    "act_360": 360.0,
    "act_365": 365.0,
    "act_act": 365.25,
}


def year_fraction(
    start: Int[Array, "..."],
    end: Int[Array, "..."],
    convention: str = "act_365",
) -> Float[Array, "..."]:
    """Compute year fraction between two ordinal dates."""
    return DAY_COUNT_FNS[convention](start, end)


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
        convention: Day-count name; must be one of the linear
            conventions in :data:`DAY_COUNT_DENOMINATORS`.

    Returns:
        The fractional ordinal ``d`` such that
        ``year_fraction(start, d, convention) == year_frac``.

    Raises:
        ValueError: For ``"30_360"``, which has no closed-form inverse
            (it depends on both endpoints' calendar structure), or for an
            unknown convention.
    """
    if convention not in DAY_COUNT_DENOMINATORS:
        if convention in DAY_COUNT_FNS:
            raise ValueError(
                f"Day count {convention!r} is not linear in elapsed days, "
                f"so it has no closed-form inverse: it depends on the "
                f"calendar structure of both endpoints. Supply an explicit "
                f"date ordinal instead of a year fraction, or use one of "
                f"{sorted(DAY_COUNT_DENOMINATORS)}."
            )
        raise ValueError(
            f"Unknown day count {convention!r}; expected one of "
            f"{sorted(DAY_COUNT_FNS)}."
        )
    denominator = DAY_COUNT_DENOMINATORS[convention]
    return (
        jnp.asarray(start, dtype=jnp.float64)
        + jnp.asarray(year_frac, dtype=jnp.float64) * denominator
    )
