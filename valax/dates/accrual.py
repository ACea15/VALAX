"""Accrual periods, coupon sizing, and the clean/dirty distinction.

Three things live here because they are the same question asked three
ways:

1. **What period does a coupon accrue over?** :func:`accrual_periods`.
2. **How large is that coupon?** :func:`fixed_coupon_amounts` — the year
   fraction of the period under the instrument's day count, never
   ``1 / frequency``.
3. **How much of the current period has already accrued?**
   :func:`accrued_interest`, and with it the clean/dirty conversion.

The third is not cosmetic. A model price is a present value of all
remaining cashflows, which is a **dirty** price; market quotes for
bonds, and the strike of an embedded call, are **clean**. Comparing one
against the other is an apples-to-oranges error whose size is exactly
the accrued interest — up to a full coupon, and systematically biased
rather than noisy.

All functions here are pure and JIT-safe: shapes are static, and the
period search is a masked reduction rather than a Python branch.
"""

from typing import Any, Optional

import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.dates.daycounts import year_fraction


def accrual_periods(
    payment_dates: Int[Array, " n"],
    accrual_start_dates: Optional[Int[Array, " n"]] = None,
    frequency: int = 2,
) -> tuple[Int[Array, " n"], Int[Array, " n"]]:
    """Resolve each coupon's accrual period.

    When ``accrual_start_dates`` is supplied it is authoritative and is
    returned unchanged. Otherwise each period is reconstructed as running
    from the previous payment date, with the first period extrapolated
    backward by the length of the second.

    That reconstruction is deliberate in two respects. It starts the
    first period at the **previous coupon date rather than at
    settlement**, because a bond pays its whole next coupon however late
    you buy it — charging the buyer only the remaining stub would
    under-price every seasoned bond. And it is only ever an
    approximation: a stubbed schedule is not recoverable from payment
    dates alone, which is precisely why ``accrual_start_dates`` exists.

    Args:
        payment_dates: Ordinal payment dates, ascending.
        accrual_start_dates: Explicit accrual starts, or ``None`` to
            reconstruct.
        frequency: Coupons per year; used only to extrapolate the first
            period of a single-coupon schedule, where there is no second
            period to measure.

    Returns:
        ``(accrual_start, accrual_end)``, both aligned with
        ``payment_dates``.
    """
    ends = jnp.asarray(payment_dates)
    if accrual_start_dates is not None:
        return jnp.asarray(accrual_start_dates), ends

    n = int(ends.shape[0])
    if n >= 2:
        first_start = ends[0] - (ends[1] - ends[0])
    else:
        first_start = ends[0] - int(round(365.0 / frequency))
    starts = jnp.concatenate([jnp.reshape(first_start, (1,)), ends[:-1]])
    return starts, ends


def fixed_coupon_amounts(bond: Any) -> Float[Array, " n"]:
    """Per-period coupon cash amounts for a fixed-coupon instrument.

    .. math::

        C_i = N \\cdot c \\cdot \\tau(\\text{start}_i, \\text{end}_i)

    The single source of truth for coupon sizing across the analytic,
    lattice, PDE and Monte Carlo pricers, so that a bond's coupon cannot
    depend on which engine prices it.

    Note the coupons are **not** all equal: under any actual/N convention
    a 181-day period and a 184-day period pay different amounts, which is
    the whole point of routing through the day count rather than
    dividing the annual rate by the frequency.

    Args:
        bond: Any instrument exposing ``face_value``, ``coupon_rate``,
            ``payment_dates``, ``accrual_start_dates``, ``day_count`` and
            ``frequency``.

    Returns:
        Coupon amount for each payment date.
    """
    starts, ends = accrual_periods(
        bond.payment_dates,
        bond.accrual_start_dates,
        getattr(bond, "frequency", 2),
    )
    tau = year_fraction(starts, ends, bond.day_count)
    return bond.face_value * bond.coupon_rate * tau


def floating_accrual_periods(bond: Any) -> tuple[Int[Array, " n"], Int[Array, " n"]]:
    """Resolve the accrual periods of a floating-rate instrument.

    The single convention for every FRN pricer in the library: a coupon
    accrues over ``[accrual_start, payment_date)``, and the reference
    rate applied to it is the forward over **that same period**.

    When ``accrual_start_dates`` is not supplied the fixing dates stand
    in for the accrual starts, which is exact for a plain
    fixing-in-advance note. Supply ``accrual_start_dates`` explicitly
    whenever a fixing lag, an in-arrears convention or a stub makes
    observation and accrual diverge.

    This exists because the analytic and Monte Carlo FRN pricers
    previously disagreed — one accrued from the fixing date, the other
    from the previous payment date — so the same note priced differently
    depending on the engine whenever those two dates were not identical.

    Args:
        bond: Instrument exposing ``payment_dates``, ``fixing_dates`` and
            ``accrual_start_dates``.

    Returns:
        ``(accrual_start, accrual_end)``, aligned with ``payment_dates``.
    """
    starts = (
        bond.fixing_dates
        if bond.accrual_start_dates is None
        else bond.accrual_start_dates
    )
    return jnp.asarray(starts), jnp.asarray(bond.payment_dates)


def accrued_interest(
    settlement_date: Int[Array, ""],
    payment_dates: Int[Array, " n"],
    coupon_rate: Float[Array, ""],
    face_value: Float[Array, ""],
    accrual_start_dates: Optional[Int[Array, " n"]] = None,
    day_count: str = "act_365",
    frequency: int = 2,
) -> Float[Array, ""]:
    """Interest accrued but not yet paid as at ``settlement_date``.

    Locates the period containing settlement and measures the elapsed
    fraction of it under ``day_count``:

    .. math::

        A = N \\cdot c \\cdot \\tau(\\text{start}_j, t), \\qquad
        \\text{start}_j \\le t < \\text{end}_j

    Returns zero outside the instrument's life, and — by construction —
    exactly zero **on** a coupon date, since that date is the end of one
    period and the start of the next.

    Implemented as a masked sum rather than a search so it stays
    JIT-safe and differentiable in ``coupon_rate`` and ``face_value``.

    Args:
        settlement_date: Ordinal at which accrual is measured.
        payment_dates: Ordinal payment dates, ascending.
        coupon_rate: Annual coupon rate.
        face_value: Par/face value.
        accrual_start_dates: Explicit accrual starts, or ``None``.
        day_count: Day count convention.
        frequency: Coupons per year; see :func:`accrual_periods`.

    Returns:
        Accrued interest in cash terms (same units as ``face_value``).
    """
    starts, ends = accrual_periods(
        payment_dates, accrual_start_dates, frequency
    )
    t = jnp.asarray(settlement_date)
    in_period = (starts <= t) & (t < ends)
    elapsed = year_fraction(starts, t, day_count)
    return jnp.sum(
        jnp.where(in_period, face_value * coupon_rate * elapsed, 0.0)
    )


def bond_accrued_interest(
    bond: Any,
    settlement_date: Optional[Int[Array, ""]] = None,
) -> Float[Array, ""]:
    """Accrued interest for a fixed-coupon bond instrument.

    Args:
        bond: Instrument exposing the fields used by
            :func:`fixed_coupon_amounts`.
        settlement_date: Ordinal at which to measure accrual; defaults to
            the bond's own ``settlement_date``.

    Returns:
        Accrued interest in cash terms.
    """
    date = (
        bond.settlement_date if settlement_date is None else settlement_date
    )
    return accrued_interest(
        settlement_date=date,
        payment_dates=bond.payment_dates,
        coupon_rate=bond.coupon_rate,
        face_value=bond.face_value,
        accrual_start_dates=bond.accrual_start_dates,
        day_count=bond.day_count,
        frequency=getattr(bond, "frequency", 2),
    )


def accrued_interest_at_dates(
    dates: Int[Array, " n_dates"],
    payment_dates: Int[Array, " n"],
    coupon_rate: Float[Array, ""],
    face_value: Float[Array, ""],
    accrual_start_dates: Optional[Int[Array, " n"]] = None,
    day_count: str = "act_365",
    frequency: int = 2,
) -> Float[Array, " n_dates"]:
    """Accrued interest at each of several calendar dates.

    The vectorised form of :func:`accrued_interest`, used to value the
    accrual owed on a set of exercise dates.

    Measuring accrual on the **exact exercise date** rather than at the
    nearest grid step is what lets a tree and a PDE with different time
    discretisations agree: snapping first and measuring afterwards makes
    the accrual a function of ``n_steps``, and a call date landing half a
    period from its step then mis-states the obstacle by a large fraction
    of a coupon.

    Args:
        dates: Ordinals at which to measure accrual.
        payment_dates: Ordinal payment dates, ascending.
        coupon_rate: Annual coupon rate.
        face_value: Par/face value.
        accrual_start_dates: Explicit accrual starts, or ``None``.
        day_count: Day count convention.
        frequency: Coupons per year; see :func:`accrual_periods`.

    Returns:
        Accrued interest at each element of ``dates``.
    """
    starts, ends = accrual_periods(
        payment_dates, accrual_start_dates, frequency
    )
    # (n_dates, 1) against (1, n_periods) → (n_dates, n_periods).
    t = jnp.asarray(dates)[:, None]
    s = starts[None, :]
    e = ends[None, :]
    in_period = (s <= t) & (t < e)
    elapsed = year_fraction(s, t, day_count)
    return jnp.sum(
        jnp.where(in_period, face_value * coupon_rate * elapsed, 0.0), axis=1
    )


def bond_accrued_interest_at_dates(
    bond: Any,
    dates: Int[Array, " n_dates"],
) -> Float[Array, " n_dates"]:
    """Accrued interest of a fixed-coupon bond at each of ``dates``.

    Args:
        bond: Instrument exposing the fields used by
            :func:`fixed_coupon_amounts`.
        dates: Ordinals at which to measure accrual.

    Returns:
        Accrued interest at each element of ``dates``.
    """
    return accrued_interest_at_dates(
        dates,
        bond.payment_dates,
        bond.coupon_rate,
        bond.face_value,
        bond.accrual_start_dates,
        bond.day_count,
        getattr(bond, "frequency", 2),
    )


def accrued_interest_by_time(
    times: Float[Array, " n_times"],
    accrual_start_times: Float[Array, " n_periods"],
    accrual_end_times: Float[Array, " n_periods"],
    coupon_rate: Float[Array, ""],
    face_value: Float[Array, ""],
) -> Float[Array, " n_times"]:
    """Accrued interest at each of ``times``, in year-fraction space.

    The counterpart of :func:`accrued_interest` for backward-induction
    engines, whose time axis is a grid of year fractions rather than
    calendar ordinals. Working in the engine's own time coordinate keeps
    the measure consistent with how it snaps coupon and exercise dates,
    and avoids inverting the day count — which ``30_360`` does not admit.

    Args:
        times: Year fractions at which to measure accrual, from the same
            reference date as the accrual times.
        accrual_start_times: Year fraction to each period's accrual start.
        accrual_end_times: Year fraction to each period's accrual end.
        coupon_rate: Annual coupon rate.
        face_value: Par/face value.

    Returns:
        Accrued interest at each element of ``times``; zero outside the
        instrument's life and at period boundaries.
    """
    t = jnp.asarray(times)[:, None]
    starts = jnp.asarray(accrual_start_times)[None, :]
    ends = jnp.asarray(accrual_end_times)[None, :]
    in_period = (starts <= t) & (t < ends)
    return jnp.sum(
        jnp.where(in_period, face_value * coupon_rate * (t - starts), 0.0),
        axis=1,
    )


def clean_from_dirty(
    dirty_price: Float[Array, ""],
    accrued: Float[Array, ""],
) -> Float[Array, ""]:
    """Strip accrued interest from a full (dirty) price.

    Args:
        dirty_price: Present value of all remaining cashflows.
        accrued: Accrued interest, as from :func:`accrued_interest`.

    Returns:
        The quoted (clean) price.
    """
    return dirty_price - accrued


def dirty_from_clean(
    clean_price: Float[Array, ""],
    accrued: Float[Array, ""],
) -> Float[Array, ""]:
    """Add accrued interest to a quoted (clean) price.

    Args:
        clean_price: Quoted price, excluding accrued interest.
        accrued: Accrued interest, as from :func:`accrued_interest`.

    Returns:
        The full (dirty) price actually settled.
    """
    return clean_price + accrued
