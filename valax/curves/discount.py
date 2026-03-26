"""Discount curve: interpolated term structure of discount factors.

The DiscountCurve is a JAX pytree (eqx.Module) storing pillar dates
and corresponding discount factors. Interpolation is log-linear on
discount factors, which is equivalent to piecewise-constant
continuously-compounded forward rates.

Because DiscountCurve is a pytree with differentiable leaves,
jax.grad through a pricing function that takes a curve gives
key-rate durations for free.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.dates.daycounts import year_fraction


class DiscountCurve(eqx.Module):
    """Term structure of discount factors.

    Attributes:
        pillar_dates: Sorted ordinal dates for curve nodes.
        discount_factors: Discount factors at each pillar (first should be 1.0).
        reference_date: Valuation date (ordinal).
        day_count: Day count convention name for year fractions.
    """

    pillar_dates: Int[Array, " n_pillars"]
    discount_factors: Float[Array, " n_pillars"]
    reference_date: Int[Array, ""]
    day_count: str = eqx.field(static=True, default="act_365")

    def __call__(self, dates: Int[Array, "..."]) -> Float[Array, "..."]:
        """Interpolate discount factors at arbitrary dates.

        Uses log-linear interpolation (linear in log-DF space),
        with flat extrapolation beyond the curve range.
        """
        # Year fractions from reference date for pillars and query dates
        pillar_times = year_fraction(
            self.reference_date, self.pillar_dates, self.day_count
        )
        query_times = year_fraction(self.reference_date, dates, self.day_count)

        log_dfs = jnp.log(self.discount_factors)

        # Linear interpolation in log-DF space with flat extrapolation
        log_df_interp = jnp.interp(query_times, pillar_times, log_dfs)
        return jnp.exp(log_df_interp)


def forward_rate(
    curve: DiscountCurve,
    start: Int[Array, ""],
    end: Int[Array, ""],
) -> Float[Array, ""]:
    """Simply-compounded forward rate between two dates.

    F(t1, t2) = (DF(t1)/DF(t2) - 1) / tau(t1, t2)
    """
    df_start = curve(start)
    df_end = curve(end)
    tau = year_fraction(start, end, curve.day_count)
    return (df_start / df_end - 1.0) / tau


def zero_rate(
    curve: DiscountCurve,
    date: Int[Array, ""],
) -> Float[Array, ""]:
    """Continuously-compounded zero rate to a given date.

    r(t) = -ln(DF(t)) / tau(ref, t)
    """
    df = curve(date)
    tau = year_fraction(curve.reference_date, date, curve.day_count)
    return -jnp.log(df) / tau
