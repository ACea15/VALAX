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

from valax.curves.convexity import ConvexityAdjFn, no_convexity_adj
from valax.curves.discount import forward_rate
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


class OISSwapRate(eqx.Module):
    """Par OIS swap rate quote (single-curve, telescoping float leg).

    The floating leg compounds the overnight rate (SOFR, €STR, SONIA,
    TONA, etc.) over each accrual period.  Under a single OIS curve,
    the floating-leg present value collapses via the telescoping
    identity:

    .. math::
        \\text{PV}_{\\text{float}} = DF(T_0) - DF(T_n)

    where :math:`T_0` is ``start_date`` and :math:`T_n` is the swap's
    maturity (``fixed_dates[-1]``).  The par condition is therefore:

    .. math::
        r \\sum_i \\tau_i\\,DF(T_i) = DF(T_0) - DF(T_n)

    which is structurally identical to :class:`SwapRate`'s residual
    in the single-curve case.  The distinction is semantic and
    forward-looking: ``OISSwapRate`` is the canonical name for OIS
    quotes once the joint solver lands (MC-Curves-2), where IBOR
    swaps will be a *dual*-curve quote (:class:`IborSwapRate`,
    task #5) and OIS swaps remain single-curve.

    Cross-checks against :func:`valax.pricing.analytic.floating.ois_swap_price`:
    ``residual(...) == swap_price / notional`` when the schedules
    match.

    Partially-seasoned swaps (``start_date < ref_date``): the MVP
    residual ignores ``fixings`` and assumes the OIS curve already
    encodes the realised compounding through ``DF(start_date)``.
    Explicit per-period seasoning support is deferred to a follow-up
    PR; the ``index_id`` field is reserved for that purpose.

    Attributes:
        start_date: Swap effective date (ordinal).
        fixed_dates: Fixed leg payment dates including maturity
            (ordinal, shape ``n``).
        rate: Par fixed rate (annualized).
        day_count: Day count convention for the fixed leg accruals.
        curves_touched: Single-element tuple naming the OIS curve.
            Multi-curve consumers override this at construction.
        index_id: Identifier for the overnight index (e.g.
            ``"USD.SOFR"``, ``"EUR.ESTR"``).  Reserved for future
            seasoning support; not consumed by the current residual.
    """

    start_date: Int[Array, ""]
    fixed_dates: Int[Array, " n"]
    rate: Float[Array, ""]
    day_count: str = eqx.field(static=True, default="act_360")
    curves_touched: tuple = eqx.field(
        static=True, default=(_DEFAULT_CURVE_ID,)
    )
    index_id: str = eqx.field(static=True, default="OIS")

    def residual(
        self,
        graph: CurveGraph,
        fixings: FixingHistory,
        ref_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        """Repricing residual: ``r * annuity - (DF(start) - DF(maturity))``.

        Telescoping form — the float leg PV equals
        ``DF(start_date) - DF(fixed_dates[-1])``.  ``fixings`` is
        accepted for protocol conformance but not consumed in the
        MVP; see class docstring on partially-seasoned swaps.
        """
        del fixings, ref_date  # MVP: fixings deferred, telescope only
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


class MoneyMarketFuture(eqx.Module):
    """Money-market / SOFR / Euribor / short-sterling future quote.

    A futures contract whose price determines the simply-compounded
    rate over a period :math:`[T_0, T_1]`.  The quoted *futures rate*
    differs from the *forward rate* by the convexity adjustment
    (see ``theory.md`` §3.9):

    .. math::
        F^{fut} - F^{FRA} = \\text{convexity adjustment}.

    The residual subtracts the adjustment from the quote and equates
    the result to the curve-implied forward rate:

    .. math::
        F^{fut} - \\text{adj} - F^{curve}(T_0, T_1) = 0,

    where :math:`F^{curve}(T_0, T_1) = (DF(T_0)/DF(T_1) - 1)/\\tau`
    is the simply-compounded forward rate over the futures period.

    Convexity adjustment is pluggable via :data:`ConvexityAdjFn`
    (see :mod:`valax.curves.convexity`):

    * :func:`~valax.curves.convexity.no_convexity_adj` — short-dated
      futures or simplified calibration.
    * :func:`~valax.curves.convexity.constant_convexity_adj` —
      desk-supplied bps value, the production default for now.
    * Future: ``hull_white_convexity_adj(model)`` — derived from a
      calibrated short-rate model; deferred until the short-rate
      model is wired into the curve build.

    Attributes:
        start_date: Period start :math:`T_0` (ordinal).
        end_date: Period end :math:`T_1` (ordinal).
        futures_rate: The quoted futures rate, in rate units (e.g.
            ``0.0435`` for 4.35%).  Note that exchange quotes are
            typically in price terms (e.g. SOFR future at 95.65,
            implying 4.35%) — convert before constructing.
        day_count: Day count for :math:`\\tau(T_0, T_1)`.
        curves_touched: Single-element tuple naming the forward
            curve this future calibrates.  In a multi-curve build,
            this would be e.g. ``("USD.SOFR.3M",)``.
        convexity_adj_fn: The adjustment plug-in.  Defaults to
            :func:`no_convexity_adj` so the simplest construction
            (``MoneyMarketFuture(start, end, rate)``) treats the
            future as a pure forward.
    """

    start_date: Int[Array, ""]
    end_date: Int[Array, ""]
    futures_rate: Float[Array, ""]
    day_count: str = eqx.field(static=True, default="act_360")
    curves_touched: tuple = eqx.field(
        static=True, default=(_DEFAULT_CURVE_ID,)
    )
    convexity_adj_fn: ConvexityAdjFn = eqx.field(
        static=True, default_factory=no_convexity_adj
    )

    def residual(
        self,
        graph: CurveGraph,
        fixings: FixingHistory,
        ref_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        """Repricing residual: ``futures_rate - adj - forward_rate(curve)``.

        Returns zero when the curve's simply-compounded forward rate
        over :math:`[T_0, T_1]` equals the futures rate minus the
        convexity adjustment.
        """
        del fixings, ref_date  # not consumed by the futures residual
        curve = graph[self.curves_touched[0]]
        adj = self.convexity_adj_fn(graph, self.start_date, self.end_date)
        # forward_rate uses the curve's own day_count.  For futures with
        # a non-standard day count, override curves with the appropriate
        # convention or compute the forward inline below.
        tau = year_fraction(self.start_date, self.end_date, self.day_count)
        df_start = curve(self.start_date)
        df_end = curve(self.end_date)
        curve_forward = (df_start / df_end - 1.0) / tau
        return self.futures_rate - adj - curve_forward
