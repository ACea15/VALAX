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


#: The interpolation mode :meth:`DiscountCurve.__call__` implements.
#:
#: :class:`DiscountCurve` has no ``interp`` field — log-linear-in-DF is
#: hard-coded.  This constant names that fact so that code declaring an
#: interpolation mode (notably :class:`valax.curves.CurveSpec`) can be
#: checked against what the curve actually does, instead of the two
#: agreeing by coincidence.  Adding a genuine second mode means giving
#: :class:`DiscountCurve` a static ``interp`` field and dispatching on it
#: here; see the interpolation workstream on the roadmap (spec C-4/C-8).
DISCOUNT_CURVE_INTERP = "log_linear_df"


class DiscountCurve(eqx.Module):
    """Term structure of discount factors.

    Interpolation is always :data:`DISCOUNT_CURVE_INTERP` — log-linear in
    discount factors, equivalently piecewise-constant continuously
    compounded forward rates — with flat extrapolation.  It is not
    configurable; see that constant for why and for what adding a mode
    would entail.

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


def flat_curve(
    rate: float | Float[Array, ""],
    reference_date: int | Int[Array, ""] = 0,
    horizon_years: float = 100.0,
    day_count: str = "act_365",
) -> DiscountCurve:
    """Two-pillar curve at a constant continuously-compounded rate.

    The degenerate curve of a constant-parameter (Black-Scholes) market: log-
    linear interpolation between ``DF = 1`` at ``reference_date`` and
    ``exp(-rate * horizon)`` at the far pillar is exactly ``exp(-rate * t)``,
    and flat extrapolation beyond it keeps the zero rate constant. Fully
    differentiable in ``rate``.

    Args:
        rate: Continuously-compounded zero rate.
        reference_date: Valuation date ordinal.
        horizon_years: Far-pillar maturity in years; only needs to exceed the
            longest horizon priced.
        day_count: Day-count convention of the curve.

    Returns:
        A :class:`DiscountCurve`.
    """
    rate_arr = jnp.asarray(rate, dtype=jnp.float64)
    ref = jnp.asarray(reference_date, dtype=jnp.int32)
    far = ref + jnp.int32(int(horizon_years * 365.0))
    # Times measured in the curve's own day count, so the far-pillar DF is
    # exactly exp(-rate * t(far)) whatever the convention.
    far_time = year_fraction(ref, far, day_count)
    times = jnp.stack([jnp.zeros_like(far_time), far_time]).astype(jnp.float64)
    return DiscountCurve(
        pillar_dates=jnp.stack([ref, far]),
        discount_factors=jnp.exp(-rate_arr * times),
        reference_date=ref,
        day_count=day_count,
    )


def forward_rate(
    curve: DiscountCurve,
    start: Int[Array, ""],
    end: Int[Array, ""],
) -> Float[Array, ""]:
    r"""Simply-compounded forward rate between two dates.

    .. math::

        F(t_1, t_2) = \frac{1}{\tau(t_1, t_2)}
            \left(\frac{DF(t_1)}{DF(t_2)} - 1\right)

    Args:
        curve: Discount curve providing ``DF(t)`` via interpolation.
        start: Start date of the forward period (ordinal).
        end: End date of the forward period (ordinal). Must be strictly
            after ``start`` so that ``tau > 0``.

    Returns:
        Simply-compounded forward rate under the curve's day-count
        convention, as a scalar JAX array.
    """
    df_start = curve(start)
    df_end = curve(end)
    tau = year_fraction(start, end, curve.day_count)
    return (df_start / df_end - 1.0) / tau


def zero_rate(
    curve: DiscountCurve,
    date: Int[Array, ""],
) -> Float[Array, ""]:
    r"""Continuously-compounded zero rate to a given date.

    .. math::

        r(t) = -\frac{\ln DF(t)}{\tau(t_\mathrm{ref}, t)}

    Args:
        curve: Discount curve providing ``DF(t)`` via interpolation.
        date: Target date (ordinal). Must be strictly after
            ``curve.reference_date`` so that ``tau > 0``.

    Returns:
        Continuously-compounded zero rate under the curve's day-count
        convention, as a scalar JAX array.
    """
    df = curve(date)
    tau = year_fraction(curve.reference_date, date, curve.day_count)
    return -jnp.log(df) / tau
