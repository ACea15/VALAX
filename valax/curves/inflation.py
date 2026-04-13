"""Inflation curve: term structure of forward CPI levels.

The :class:`InflationCurve` stores forward Consumer Price Index (CPI)
levels at pillar dates and interpolates in **log-CPI** space, which
gives smooth implied forward inflation rates — analogous to log-linear
discount factor interpolation on the nominal curve.

From the forward CPI curve three key quantities are derived:

1. **Forward CPI** at any date: ``forward_cpi(curve, dates)``.
2. **Zero-coupon (breakeven) inflation rate**: the annually-compounded
   rate ``z`` such that ``CPI(T) = base_cpi * (1 + z)^T``.
3. **Year-on-year forward inflation rate**: the single-period rate
   ``CPI(T_i) / CPI(T_{i-1}) - 1``.

Because the curve is an ``equinox.Module`` pytree, ``jax.grad`` through
any pricing function that takes an inflation curve gives inflation-
delta (IE01) sensitivities automatically.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.dates.daycounts import year_fraction


class InflationCurve(eqx.Module):
    """Term structure of forward CPI levels.

    Attributes:
        pillar_dates: Sorted ordinal dates for CPI pillars.
        forward_cpis: Forward CPI level at each pillar.
        base_cpi: CPI level at inception / base date.
        reference_date: Valuation date (ordinal).
        day_count: Day count convention for year fractions.
    """

    pillar_dates: Int[Array, " n"]
    forward_cpis: Float[Array, " n"]
    base_cpi: Float[Array, ""]
    reference_date: Int[Array, ""]
    day_count: str = eqx.field(static=True, default="act_act")


# ── Interpolation ────────────────────────────────────────────────────

def forward_cpi(
    curve: InflationCurve,
    dates: Int[Array, "..."],
) -> Float[Array, "..."]:
    """Interpolate forward CPI at arbitrary dates.

    Uses log-linear interpolation on CPI levels with flat
    extrapolation beyond the pillar range.
    """
    pillar_times = year_fraction(
        curve.reference_date, curve.pillar_dates, curve.day_count
    )
    query_times = year_fraction(curve.reference_date, dates, curve.day_count)
    log_cpis = jnp.log(curve.forward_cpis)
    log_cpi_interp = jnp.interp(query_times, pillar_times, log_cpis)
    return jnp.exp(log_cpi_interp)


def zc_inflation_rate(
    curve: InflationCurve,
    dates: Int[Array, "..."],
) -> Float[Array, "..."]:
    """Zero-coupon (breakeven) inflation rate to given dates.

    .. math::

        z(T) = \\left(\\frac{\\text{CPI}(T)}{\\text{CPI}(0)}\\right)^{1/T} - 1
    """
    cpi_T = forward_cpi(curve, dates)
    tau = year_fraction(curve.reference_date, dates, curve.day_count)
    tau_safe = jnp.maximum(tau, 1e-10)
    return (cpi_T / curve.base_cpi) ** (1.0 / tau_safe) - 1.0


def yoy_forward_rate(
    curve: InflationCurve,
    start_dates: Int[Array, "..."],
    end_dates: Int[Array, "..."],
) -> Float[Array, "..."]:
    """Year-on-year forward inflation rate between two date sets.

    .. math::

        \\text{YoY}(T_{i-1}, T_i) = \\frac{\\text{CPI}(T_i)}{\\text{CPI}(T_{i-1})} - 1
    """
    return forward_cpi(curve, end_dates) / forward_cpi(curve, start_dates) - 1.0


# ── Constructor helper ───────────────────────────────────────────────

def from_zc_rates(
    reference_date: Int[Array, ""],
    pillar_dates: Int[Array, " n"],
    zc_rates: Float[Array, " n"],
    base_cpi: Float[Array, ""],
    day_count: str = "act_act",
) -> InflationCurve:
    """Build an InflationCurve from zero-coupon breakeven rates.

    Converts each rate :math:`z_i` to a forward CPI via

    .. math::

        \\text{CPI}(T_i) = \\text{base\\_cpi} \\cdot (1 + z_i)^{T_i}

    Args:
        reference_date: Valuation date (ordinal).
        pillar_dates: Pillar dates (ordinals).
        zc_rates: Zero-coupon breakeven rates at each pillar.
        base_cpi: CPI level at inception.
        day_count: Day count convention.

    Returns:
        InflationCurve with forward CPI levels.
    """
    tau = year_fraction(reference_date, pillar_dates, day_count)
    forward_cpis = base_cpi * (1.0 + zc_rates) ** tau
    return InflationCurve(
        pillar_dates=pillar_dates,
        forward_cpis=forward_cpis,
        base_cpi=base_cpi,
        reference_date=reference_date,
        day_count=day_count,
    )
