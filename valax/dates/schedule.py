"""Coupon schedule generation for fixed income instruments.

Generates arrays of payment dates as integer ordinals, suitable for
use inside JIT-traced code.
"""

import jax.numpy as jnp
from jaxtyping import Int
from jax import Array

from valax.dates.daycounts import ymd_to_ordinal


def generate_schedule(
    start_year: int,
    start_month: int,
    start_day: int,
    end_year: int,
    end_month: int,
    end_day: int,
    frequency: int = 2,
) -> Int[Array, " n_dates"]:
    """Generate a coupon schedule as an array of ordinal dates.

    Produces dates from (exclusive) start to (inclusive) end at the
    given frequency (payments per year). Dates are generated backward
    from maturity, which is standard for bond schedules.

    Args:
        start_year, start_month, start_day: Issue/settlement date.
        end_year, end_month, end_day: Maturity date.
        frequency: Coupons per year (1=annual, 2=semi, 4=quarterly).

    Returns:
        1-D array of ordinal dates (payment dates including maturity).
    """
    months_between = (end_year - start_year) * 12 + (end_month - start_month)
    step_months = 12 // frequency
    n_periods = months_between // step_months

    dates = []
    for i in range(n_periods - 1, 0, -1):
        # Walk backward from maturity
        total_months_back = i * step_months
        m = end_month - (total_months_back % 12)
        y = end_year - (total_months_back // 12)
        if m <= 0:
            m += 12
            y -= 1
        dates.append(ymd_to_ordinal(y, m, min(end_day, 28)))

    # Always include maturity
    dates.append(ymd_to_ordinal(end_year, end_month, end_day))

    # Sort ascending
    dates.sort(key=lambda x: int(x))
    return jnp.stack(dates)
