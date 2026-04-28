"""Bootstrap instrument definitions (data-only pytrees with residual methods).

These represent market quotes used as inputs to curve construction, not
tradeable instruments for pricing.  They live in ``curves/`` rather
than ``instruments/`` because they are curve-building inputs, keeping
the dependency graph clean.

Each class implements the :class:`~valax.curves.bootstrap_proto.BootstrapInstrument`
protocol: it carries a static ``curves_touched`` tuple of curve
identifiers and a ``residual(graph, fixings, ref_date)`` method
returning zero when the graph correctly reprices the quote.

For backwards compatibility, the existing single-curve bootstrap
functions (``bootstrap_sequential``, ``bootstrap_simultaneous``)
continue to work without any explicit curve identifier — instruments
default to the sentinel id ``"_default_"`` and the bootstrap wraps
the in-progress curve in a single-element :class:`CurveGraph` keyed
by the same sentinel.

Multi-curve users override ``curves_touched`` at construction time::

    swap = SwapRate(
        start_date=ref,
        fixed_dates=fixed_dates,
        rate=rate_5y,
        curves_touched=("USD.SOFR.OIS",),
    )
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.curves.fixings import FixingHistory
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import year_fraction


# Sentinel curve identifier used by single-curve bootstraps.  Multi-curve
# code should pass an explicit identifier (e.g. ``"USD.SOFR.OIS"``).
_DEFAULT_CURVE_ID = "_default_"


class DepositRate(eqx.Module):
    """Money market deposit rate quote.

    The implied discount factor relationship is:

    .. math::
        DF(\\text{end}) = \\frac{DF(\\text{start})}{1 + r\\,\\tau}

    When ``start_date == reference_date``, ``DF(start) == 1``.

    Attributes:
        start_date: Deposit effective date (ordinal).
        end_date: Deposit maturity date (ordinal).
        rate: Simply-compounded deposit rate.
        day_count: Day count convention.
        curves_touched: Curve identifier this instrument prices against
            (single-element tuple).  Defaults to the single-curve
            sentinel for backwards compatibility.
    """

    start_date: Int[Array, ""]
    end_date: Int[Array, ""]
    rate: Float[Array, ""]
    day_count: str = eqx.field(static=True, default="act_360")
    curves_touched: tuple = eqx.field(
        static=True, default=(_DEFAULT_CURVE_ID,)
    )

    def residual(
        self,
        graph: CurveGraph,
        fixings: FixingHistory,
        ref_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        """Repricing residual: ``DF(end) * (1 + r*tau) - DF(start)``.

        ``fixings`` and ``ref_date`` are unused for deposits but are
        accepted to satisfy the :class:`BootstrapInstrument` protocol.
        """
        del fixings, ref_date  # unused for deposits
        curve = graph[self.curves_touched[0]]
        tau = year_fraction(self.start_date, self.end_date, self.day_count)
        df_start = curve(self.start_date)
        df_end = curve(self.end_date)
        return df_end * (1.0 + self.rate * tau) - df_start


class FRA(eqx.Module):
    """Forward Rate Agreement quote.

    Same residual shape as :class:`DepositRate`; the distinction is
    semantic (start date in the future vs. on the valuation date).
    Both impose the equation:

    .. math::
        DF(\\text{end})\\,(1 + r\\,\\tau) - DF(\\text{start}) = 0

    Attributes:
        start_date: FRA effective date (ordinal).
        end_date: FRA maturity date (ordinal).
        rate: Simply-compounded forward rate.
        day_count: Day count convention.
        curves_touched: Curve identifier this instrument prices against.
    """

    start_date: Int[Array, ""]
    end_date: Int[Array, ""]
    rate: Float[Array, ""]
    day_count: str = eqx.field(static=True, default="act_360")
    curves_touched: tuple = eqx.field(
        static=True, default=(_DEFAULT_CURVE_ID,)
    )

    def residual(
        self,
        graph: CurveGraph,
        fixings: FixingHistory,
        ref_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        """Repricing residual: ``DF(end) * (1 + r*tau) - DF(start)``."""
        del fixings, ref_date  # unused for FRAs
        curve = graph[self.curves_touched[0]]
        tau = year_fraction(self.start_date, self.end_date, self.day_count)
        df_start = curve(self.start_date)
        df_end = curve(self.end_date)
        return df_end * (1.0 + self.rate * tau) - df_start


class SwapRate(eqx.Module):
    """Par swap rate quote for curve bootstrap.

    The par condition is:

    .. math::
        r \\sum_i \\tau_i\\,DF(T_i) = DF(\\text{start}) - DF(\\text{maturity})

    where the annuity sum runs over the fixed-leg payment dates.  The
    residual is the difference of the two sides (zero when the curve
    is correctly calibrated).

    Distinct from :class:`~valax.instruments.rates.InterestRateSwap`,
    which represents a tradeable swap contract with notional and
    direction.  This class is purely a calibration input.

    Attributes:
        start_date: Swap effective date (ordinal).
        fixed_dates: Fixed leg payment dates including maturity
            (ordinal, shape ``n``).
        rate: Par swap rate (annualized).
        day_count: Day count convention for fixed leg accrual.
        curves_touched: Curve identifier this instrument prices against.
            For a single-curve build, the same curve is used for
            forward projection and discounting.
    """

    start_date: Int[Array, ""]
    fixed_dates: Int[Array, " n"]
    rate: Float[Array, ""]
    day_count: str = eqx.field(static=True, default="act_360")
    curves_touched: tuple = eqx.field(
        static=True, default=(_DEFAULT_CURVE_ID,)
    )

    def residual(
        self,
        graph: CurveGraph,
        fixings: FixingHistory,
        ref_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        """Repricing residual: ``r * annuity - (DF(start) - DF(maturity))``."""
        del fixings, ref_date  # single-curve swap residual ignores fixings
        curve = graph[self.curves_touched[0]]
        starts = jnp.concatenate(
            [self.start_date[None], self.fixed_dates[:-1]]
        )
        taus = year_fraction(starts, self.fixed_dates, self.day_count)
        fixed_dfs = curve(self.fixed_dates)
        annuity = jnp.sum(taus * fixed_dfs)
        df_start = curve(self.start_date)
        df_mat = curve(self.fixed_dates[-1])
        return self.rate * annuity - (df_start - df_mat)
