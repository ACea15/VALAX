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


def _floating_leg_pv(
    graph: CurveGraph,
    fixings: FixingHistory,
    start_date: Int[Array, ""],
    payment_dates: Int[Array, " n"],
    fixing_dates: Int[Array, " n"],
    day_count: str,
    index_id: str,
    discount_curve_id: str,
    forward_curve_id: str,
    spread_addon: Float[Array, ""],
) -> Float[Array, ""]:
    """Present value of a floating leg under the dual-curve framework.

    Helper shared by every instrument with at least one floating leg
    (:class:`IborSwapRate`, :class:`TenorBasisSwap`, both legs of
    :class:`CrossCurrencyBasisSwap`).  Computes

    .. math::
        \\text{PV} = \\sum_j (F_j + s)\\,\\tau_j\\,DF_{\\text{disc}}(T_j),

    where each :math:`F_j` is a realised fixing if recorded under
    ``index_id``, else the curve-projected forward
    :math:`(DF_{\\text{fwd}}(T_{j-1}) / DF_{\\text{fwd}}(T_j) - 1)/\\tau_j`.
    """
    discount_curve = graph[discount_curve_id]
    forward_curve = graph[forward_curve_id]

    starts = jnp.concatenate([start_date[None], payment_dates[:-1]])
    taus = year_fraction(starts, payment_dates, day_count)
    df_starts_fwd = forward_curve(starts)
    df_ends_fwd = forward_curve(payment_dates)
    projected = (df_starts_fwd / df_ends_fwd - 1.0) / taus

    if index_id in fixings.indices:
        realised = fixings.lookup_many(index_id, fixing_dates)
        rates = jnp.where(jnp.isnan(realised), projected, realised)
    else:
        rates = projected

    df_disc = discount_curve(payment_dates)
    return jnp.sum((rates + spread_addon) * taus * df_disc)


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


class IborSwapRate(eqx.Module):
    """Par IBOR swap rate quote (dual-curve).

    The post-2008 standard quote for an interest-rate swap whose
    floating leg pays a single tenor IBOR (3M, 6M, 12M).  Touches two
    curves:

    * a **discount curve** (typically OIS) that discounts every
      cashflow on both legs;
    * a **forward (projection) curve** for the floating-rate tenor,
      from which each coupon's forward rate is projected.

    The dual-curve par condition (theory.md §3.2) is

    .. math::
        r \\sum_i \\tau^{\\text{fixed}}_i\\,DF_{\\text{disc}}(T^{\\text{fixed}}_i)
        \\;=\\; \\sum_j F_j\\,\\tau^{\\text{float}}_j\\,DF_{\\text{disc}}(T^{\\text{float}}_j)

    where each forward :math:`F_j` is

    * a **realised fixing** if ``fixing_dates[j]`` is recorded in the
      :class:`FixingHistory` under ``index_id``;
    * the **projected forward**
      :math:`(DF_{\\text{fwd}}(T_{j-1}) / DF_{\\text{fwd}}(T_j) - 1) / \\tau`
      otherwise.

    Sign convention matches :class:`SwapRate`: ``residual = fixed_pv -
    float_pv``, so a positive residual means the quoted fixed rate is
    above the par rate that the curve graph reprices.

    Single-curve degenerate case: when
    ``curves_touched[0] == curves_touched[1]``, the dual-curve formula
    reduces to the float-leg telescoping identity, recovering the
    same residual as :class:`SwapRate` (modulo day-count differences).

    Attributes:
        start_date: Swap effective date (ordinal).
        fixed_dates: Fixed leg payment dates (ordinal, shape
            ``n_fixed``).
        float_dates: Floating leg accrual end / payment dates (ordinal,
            shape ``n_float``).  May differ from ``fixed_dates`` when
            payment frequencies differ.
        fixing_dates: Fixing date for each float coupon (ordinal,
            shape ``n_float``).  For an in-arrears IBOR swap, this is
            typically ``T_{j-1}`` (the period start); the convention
            is left to the caller.
        rate: Par fixed rate (annualised).
        fixed_day_count: Fixed leg day count convention.
        float_day_count: Floating leg day count convention.
        curves_touched: 2-tuple ``(discount_curve_id,
            forward_curve_id)``.  Defaults to
            ``("_default_", "_default_")`` so the residual reduces to
            the single-curve telescope when no override is supplied.
        index_id: Identifier under which realised fixings for this
            tenor are stored in :class:`FixingHistory`.  Reserved
            ``"IBOR"`` sentinel by default; production users specify
            e.g. ``"USD.SOFR.3M"`` or ``"EUR.EURIBOR.6M"``.
    """

    start_date: Int[Array, ""]
    fixed_dates: Int[Array, " n_fixed"]
    float_dates: Int[Array, " n_float"]
    fixing_dates: Int[Array, " n_float"]
    rate: Float[Array, ""]
    fixed_day_count: str = eqx.field(static=True, default="act_360")
    float_day_count: str = eqx.field(static=True, default="act_360")
    curves_touched: tuple = eqx.field(
        static=True,
        default=(_DEFAULT_CURVE_ID, _DEFAULT_CURVE_ID),
    )
    index_id: str = eqx.field(static=True, default="IBOR")

    def residual(
        self,
        graph: CurveGraph,
        fixings: FixingHistory,
        ref_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        """Repricing residual: ``fixed_pv - float_pv``.

        Float-leg coupons that have a realised fixing in
        ``fixings[index_id]`` use the realised rate; the rest are
        projected from the forward curve.
        """
        del ref_date  # temporal selection is delegated to fixings
        float_pv = _floating_leg_pv(
            graph,
            fixings,
            self.start_date,
            self.float_dates,
            self.fixing_dates,
            self.float_day_count,
            self.index_id,
            self.curves_touched[0],
            self.curves_touched[1],
            jnp.asarray(0.0),
        )
        # Fixed leg annuity (no spread, no fixings).
        discount_curve = graph[self.curves_touched[0]]
        fixed_starts = jnp.concatenate(
            [self.start_date[None], self.fixed_dates[:-1]]
        )
        fixed_taus = year_fraction(
            fixed_starts, self.fixed_dates, self.fixed_day_count
        )
        annuity = jnp.sum(fixed_taus * discount_curve(self.fixed_dates))
        fixed_pv = self.rate * annuity
        return fixed_pv - float_pv


class FXForward(eqx.Module):
    """FX forward outright quote (covered interest rate parity).

    Quotes the rate at which two currencies exchange at
    ``settle_date``.  Touches the OIS discount curves of both
    currencies; FX spot is supplied as instrument data rather than
    pulled from the graph (keeps the curve graph homogeneous —
    only DiscountCurves live in it).

    The residual is the deviation from covered interest rate parity
    (theory.md §3.7):

    .. math::
        F(0, T) - S(0) \\cdot \\frac{DF_{\\text{foreign}}(T)}{DF_{\\text{domestic}}(T)} = 0.

    Convention: ``fx_spot`` and ``quoted_forward`` are in units of
    *domestic currency per unit of foreign currency*.  For a EUR/USD
    quote of 1.10, USD is the domestic side (denominator currency in
    the trader's mental model) and EUR is foreign; ``fx_spot = 1.10``
    means 1.10 USD per 1 EUR.  By convention,
    ``curves_touched[0]`` is the domestic curve and
    ``curves_touched[1]`` is the foreign curve.

    Attributes:
        value_date: Spot value date (typically T+2 from the trade
            date).  Stored for completeness; not consumed by the
            residual under the standard collateralised CIP
            assumption.
        settle_date: Forward settlement date — the date at which
            the exchange occurs.
        quoted_forward: Quoted forward rate, in domestic per foreign
            units.
        fx_spot: Spot rate at the reference date, in the same units.
        curves_touched: 2-tuple ``(domestic_curve_id,
            foreign_curve_id)``.  Defaults to single-curve sentinel
            for protocol consistency; production callers always
            override.
    """

    value_date: Int[Array, ""]
    settle_date: Int[Array, ""]
    quoted_forward: Float[Array, ""]
    fx_spot: Float[Array, ""]
    curves_touched: tuple = eqx.field(
        static=True,
        default=(_DEFAULT_CURVE_ID, _DEFAULT_CURVE_ID),
    )

    def residual(
        self,
        graph: CurveGraph,
        fixings: FixingHistory,
        ref_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        """Repricing residual: ``quoted_forward - spot * DF_for / DF_dom``.

        Both DFs are evaluated at ``settle_date`` from the reference
        date, under the standard single-asof collateralised pricing
        assumption.  ``value_date`` is not consumed.
        """
        del fixings, ref_date  # CIP residual ignores both
        domestic_curve = graph[self.curves_touched[0]]
        foreign_curve = graph[self.curves_touched[1]]
        df_dom = domestic_curve(self.settle_date)
        df_for = foreign_curve(self.settle_date)
        return self.quoted_forward - self.fx_spot * df_for / df_dom


class FXSwap(eqx.Module):
    """FX swap quote — simultaneous near-leg and far-leg exchange.

    An FX swap is two simultaneous opposite-direction trades:

    * **Near leg**: exchange at ``near_date`` at ``near_rate``
      (typically spot).
    * **Far leg**: reverse the exchange at ``far_date`` at
      ``far_rate``.

    Market quote is the **swap points**: ``swap_points = far_rate -
    near_rate``.  The user supplies the absolute ``near_rate`` and
    ``far_rate`` (the absolute rates are what the residual uses).

    The CIP relation between the two legs is

    .. math::
        \\frac{F(0, T_{\\text{far}})}{F(0, T_{\\text{near}})}
        = \\frac{DF_{\\text{for}}(T_{\\text{far}}) / DF_{\\text{dom}}(T_{\\text{far}})}
               {DF_{\\text{for}}(T_{\\text{near}}) / DF_{\\text{dom}}(T_{\\text{near}})}.

    Rearranged for the residual:

    .. math::
        \\text{far\\_rate} \\cdot DF_{\\text{for}}(T_{\\text{near}})
        \\cdot DF_{\\text{dom}}(T_{\\text{far}})
        \\;-\\;
        \\text{near\\_rate} \\cdot DF_{\\text{dom}}(T_{\\text{near}})
        \\cdot DF_{\\text{for}}(T_{\\text{far}})
        \\;=\\; 0.

    When ``near_date == ref_date`` (the spot DFs equal 1), the
    residual collapses to :class:`FXForward`'s CIP form on the far
    leg with ``fx_spot = near_rate``.

    Attributes:
        near_date: Near-leg value date (ordinal).
        far_date: Far-leg value date (ordinal).
        near_rate: Near-leg exchange rate, domestic per foreign.
        far_rate: Far-leg exchange rate, domestic per foreign.
        curves_touched: 2-tuple ``(domestic_curve_id,
            foreign_curve_id)``.
    """

    near_date: Int[Array, ""]
    far_date: Int[Array, ""]
    near_rate: Float[Array, ""]
    far_rate: Float[Array, ""]
    curves_touched: tuple = eqx.field(
        static=True,
        default=(_DEFAULT_CURVE_ID, _DEFAULT_CURVE_ID),
    )

    def residual(
        self,
        graph: CurveGraph,
        fixings: FixingHistory,
        ref_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        """Repricing residual: CIP relation between the two legs.

        Returns zero when the curve graph satisfies covered interest
        rate parity at both ``near_date`` and ``far_date``.
        """
        del fixings, ref_date
        domestic_curve = graph[self.curves_touched[0]]
        foreign_curve = graph[self.curves_touched[1]]
        df_dom_near = domestic_curve(self.near_date)
        df_for_near = foreign_curve(self.near_date)
        df_dom_far = domestic_curve(self.far_date)
        df_for_far = foreign_curve(self.far_date)
        return (
            self.far_rate * df_for_near * df_dom_far
            - self.near_rate * df_dom_near * df_for_far
        )


class TenorBasisSwap(eqx.Module):
    """Tenor basis swap quote (e.g. 3M-vs-6M same-currency basis).

    Both legs are floating: one pays the IBOR rate of tenor :math:`a`,
    the other pays the IBOR rate of tenor :math:`b`, plus a basis
    **spread** added to one of them (by market convention, the
    funding-stressed tenor).  Both legs are discounted with the same
    OIS curve.

    Touches three curves:

    * the **discount curve** (typically OIS) used to discount each
      leg's coupons;
    * the **leg-A forward curve** for projecting tenor-:math:`a`
      forwards;
    * the **leg-B forward curve** for projecting tenor-:math:`b`
      forwards.

    The par condition (theory.md §3.7) is

    .. math::
        \\sum_i \\tau^a_i\\,(F^a_i + s_a)\\,DF_{\\text{disc}}(T^a_i)
        \\;=\\; \\sum_j \\tau^b_j\\,(F^b_j + s_b)\\,DF_{\\text{disc}}(T^b_j),

    where :math:`s_a`, :math:`s_b` are zero except on the leg
    indicated by ``spread_on_leg``.  The residual is

    .. math::
        \\text{leg-A PV} - \\text{leg-B PV},

    with the spread additive to whichever leg is selected.  Realised
    fixings on either leg override the curve-projected forward, per
    the convention in :class:`IborSwapRate`.

    Without any same-currency multi-tenor basis-swap quote, a curve
    graph containing two distinct tenor curves is mathematically
    underdetermined — the level of the second tenor cannot be
    resolved against the first.  This is the calibration role this
    instrument fills.

    Attributes:
        start_date: Swap effective date (ordinal).

        leg_a_dates: Leg-A payment dates (ordinal, shape ``n_a``).
        leg_a_fixing_dates: Leg-A fixing dates (shape ``n_a``).
        leg_a_day_count: Leg-A day count convention.
        leg_a_index_id: Identifier for leg-A's IBOR fixings.

        leg_b_dates: Leg-B payment dates (ordinal, shape ``n_b``).
        leg_b_fixing_dates: Leg-B fixing dates (shape ``n_b``).
        leg_b_day_count: Leg-B day count convention.
        leg_b_index_id: Identifier for leg-B's IBOR fixings.

        spread: The basis spread, in rate units (e.g. ``0.0021`` for
            21 bp).
        spread_on_leg: Which leg carries the spread — ``"a"`` or
            ``"b"``.  Convention: the funding-stressed tenor (the
            one with higher credit/liquidity premia per unit time).

        curves_touched: 3-tuple ``(discount_id, leg_a_id, leg_b_id)``.
            Defaults to ``(_default_, _default_, _default_)`` for
            single-curve degeneracy; multi-curve consumers override.
    """

    # Required (array) fields — must come first in the dataclass.
    start_date: Int[Array, ""]
    leg_a_dates: Int[Array, " n_a"]
    leg_a_fixing_dates: Int[Array, " n_a"]
    leg_b_dates: Int[Array, " n_b"]
    leg_b_fixing_dates: Int[Array, " n_b"]

    # Defaulted fields.
    spread: Float[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(0.0)
    )
    leg_a_day_count: str = eqx.field(static=True, default="act_360")
    leg_a_index_id: str = eqx.field(static=True, default="IBOR_A")
    leg_b_day_count: str = eqx.field(static=True, default="act_360")
    leg_b_index_id: str = eqx.field(static=True, default="IBOR_B")
    spread_on_leg: str = eqx.field(static=True, default="a")
    curves_touched: tuple = eqx.field(
        static=True,
        default=(
            _DEFAULT_CURVE_ID,
            _DEFAULT_CURVE_ID,
            _DEFAULT_CURVE_ID,
        ),
    )

    def residual(
        self,
        graph: CurveGraph,
        fixings: FixingHistory,
        ref_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        """Repricing residual: ``leg_a_pv - leg_b_pv`` with the spread
        added (per period) to the leg indicated by ``spread_on_leg``.
        """
        del ref_date  # temporal selection is delegated to fixings

        if self.spread_on_leg == "a":
            a_addon, b_addon = self.spread, jnp.asarray(0.0)
        elif self.spread_on_leg == "b":
            a_addon, b_addon = jnp.asarray(0.0), self.spread
        else:
            raise ValueError(
                f"spread_on_leg must be 'a' or 'b', got "
                f"'{self.spread_on_leg}'"
            )

        a_pv = _floating_leg_pv(
            graph,
            fixings,
            self.start_date,
            self.leg_a_dates,
            self.leg_a_fixing_dates,
            self.leg_a_day_count,
            self.leg_a_index_id,
            self.curves_touched[0],   # shared discount
            self.curves_touched[1],   # leg A forward
            a_addon,
        )
        b_pv = _floating_leg_pv(
            graph,
            fixings,
            self.start_date,
            self.leg_b_dates,
            self.leg_b_fixing_dates,
            self.leg_b_day_count,
            self.leg_b_index_id,
            self.curves_touched[0],   # shared discount
            self.curves_touched[2],   # leg B forward
            b_addon,
        )
        return a_pv - b_pv


class CrossCurrencyBasisSwap(eqx.Module):
    """Cross-currency basis swap quote (CCBS) — MTM and constant-notional
    variants.

    Two floating legs in different currencies, exchanged simultaneously.
    The basis spread is added per period to the funding-stressed leg
    (specified via ``spread_on_leg``).  Touches up to four curves:
    discount and forward curves in each of two currencies.

    Two market-quoted variants are supported:

    * ``"constant_notional"`` — Notional fixed at inception.  At
      :math:`t = 0` the parties exchange notionals at spot
      (:math:`N_{\\text{for}} = N_{\\text{dom}} / S(0)`); at
      maturity the notionals are returned in their respective
      currencies.  Under this convention, normalising
      :math:`N_{\\text{dom}} = 1` gives the residual

      .. math::
          R_{\\text{cn}} =
          \\big[\\text{coupon\\_pv}_{\\text{dom}} + (1 - DF_{\\text{dom}}(T_n))\\big]
          - \\big[\\text{coupon\\_pv}_{\\text{for}} + (1 - DF_{\\text{for}}(T_n))\\big],

      with the spread added to the indicated leg's coupons.  The
      ``fx_spot`` field cancels in this normalised form (it appears
      both as :math:`N_{\\text{for}} = N_{\\text{dom}}/S(0)` and
      as the conversion factor from foreign-leg PV to domestic
      units, and the two cancel).  We carry it as an explicit
      instrument field for documentation, audit, and future
      diagnostic use.

    * ``"mtm"`` — Notional resets at every fixing date to the
      prevailing spot rate, removing the principal-FX risk.  The
      MTM CCBS PV-decomposition (Henrard 2014, §10.5) shows that
      the foreign-leg PV in domestic terms simplifies to

      .. math::
          \\text{PV}^{\\text{MTM}}_{\\text{for, in dom}}
          = N_{\\text{dom}} \\sum_j F^{\\text{for}}_j\\,\\tau_j\\,DF_{\\text{dom}}(T_j),

      i.e. **discounting with the domestic OIS curve**.  The
      foreign discount curve does *not* enter the MTM residual:

      .. math::
          R_{\\text{mtm}} =
          \\sum_i (F^{\\text{dom}}_i + s_{\\text{dom}})\\,\\tau_i\\,DF_{\\text{dom}}(T_i)
          - \\sum_j (F^{\\text{for}}_j + s_{\\text{for}})\\,\\tau_j\\,DF_{\\text{dom}}(T_j).

      The two variants give measurably different par spreads
      (test ``test_variant_distinction``) — they are genuinely
      different equations.

    Convention: ``curves_touched`` is a 4-tuple
    ``(dom_discount_id, dom_forward_id, for_discount_id,
    for_forward_id)``.

    Attributes:
        start_date: Swap effective date (ordinal).

        dom_dates: Domestic-leg payment dates (ordinal, shape ``n_dom``).
        dom_fixing_dates: Domestic-leg fixing dates (shape ``n_dom``).
        for_dates: Foreign-leg payment dates (ordinal, shape ``n_for``).
        for_fixing_dates: Foreign-leg fixing dates (shape ``n_for``).
        fx_spot: Spot FX rate, in domestic per foreign units.  Stored
            for documentation; cancels in the normalised residual
            for both variants.

        spread: Basis spread (rate units, e.g. ``-0.0050`` for −50 bp).
        dom_day_count: Domestic-leg day count.
        dom_index_id: Identifier for domestic-leg fixings.
        for_day_count: Foreign-leg day count.
        for_index_id: Identifier for foreign-leg fixings.
        spread_on_leg: ``"domestic"`` or ``"foreign"``.
        variant: ``"constant_notional"`` or ``"mtm"``.
        curves_touched: 4-tuple of curve identifiers, in the order
            ``(dom_discount, dom_forward, for_discount, for_forward)``.
    """

    # Required (array) fields.
    start_date: Int[Array, ""]
    dom_dates: Int[Array, " n_dom"]
    dom_fixing_dates: Int[Array, " n_dom"]
    for_dates: Int[Array, " n_for"]
    for_fixing_dates: Int[Array, " n_for"]
    fx_spot: Float[Array, ""]

    # Defaulted fields.
    spread: Float[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(0.0)
    )
    dom_day_count: str = eqx.field(static=True, default="act_360")
    dom_index_id: str = eqx.field(static=True, default="DOM_FLOAT")
    for_day_count: str = eqx.field(static=True, default="act_360")
    for_index_id: str = eqx.field(static=True, default="FOR_FLOAT")
    spread_on_leg: str = eqx.field(static=True, default="domestic")
    variant: str = eqx.field(static=True, default="constant_notional")
    curves_touched: tuple = eqx.field(
        static=True,
        default=(
            _DEFAULT_CURVE_ID,  # dom discount
            _DEFAULT_CURVE_ID,  # dom forward
            _DEFAULT_CURVE_ID,  # for discount
            _DEFAULT_CURVE_ID,  # for forward
        ),
    )

    def residual(
        self,
        graph: CurveGraph,
        fixings: FixingHistory,
        ref_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        """CCBS residual; selects between MTM and constant-notional
        per the static ``variant`` field."""
        del ref_date  # temporal selection delegated to fixings

        if self.spread_on_leg == "domestic":
            dom_addon, for_addon = self.spread, jnp.asarray(0.0)
        elif self.spread_on_leg == "foreign":
            dom_addon, for_addon = jnp.asarray(0.0), self.spread
        else:
            raise ValueError(
                f"spread_on_leg must be 'domestic' or 'foreign', got "
                f"'{self.spread_on_leg}'"
            )

        dom_disc_id = self.curves_touched[0]
        dom_fwd_id = self.curves_touched[1]
        for_disc_id = self.curves_touched[2]
        for_fwd_id = self.curves_touched[3]

        # Domestic-leg coupons (always discounted with domestic OIS).
        dom_coupon_pv = _floating_leg_pv(
            graph,
            fixings,
            self.start_date,
            self.dom_dates,
            self.dom_fixing_dates,
            self.dom_day_count,
            self.dom_index_id,
            dom_disc_id,
            dom_fwd_id,
            dom_addon,
        )

        if self.variant == "mtm":
            # Foreign-leg coupons discounted with DOMESTIC OIS.
            # The foreign discount curve is not used in MTM.
            for_coupon_pv_in_dom = _floating_leg_pv(
                graph,
                fixings,
                self.start_date,
                self.for_dates,
                self.for_fixing_dates,
                self.for_day_count,
                self.for_index_id,
                dom_disc_id,   # ← key MTM detail
                for_fwd_id,
                for_addon,
            )
            return dom_coupon_pv - for_coupon_pv_in_dom

        if self.variant == "constant_notional":
            # Foreign-leg coupons discounted with FOREIGN OIS.
            for_coupon_pv = _floating_leg_pv(
                graph,
                fixings,
                self.start_date,
                self.for_dates,
                self.for_fixing_dates,
                self.for_day_count,
                self.for_index_id,
                for_disc_id,
                for_fwd_id,
                for_addon,
            )
            df_dom_mat = graph[dom_disc_id](self.dom_dates[-1])
            df_for_mat = graph[for_disc_id](self.for_dates[-1])
            return (
                (dom_coupon_pv + (1.0 - df_dom_mat))
                - (for_coupon_pv + (1.0 - df_for_mat))
            )

        raise ValueError(
            f"variant must be 'constant_notional' or 'mtm', got "
            f"'{self.variant}'"
        )


class TurnInstrument(eqx.Module):
    """Discrete forward-rate jump anchor for turn-of-year (or any
    calendar discontinuity) effects.

    Year-end and quarter-end liquidity strains produce visible spikes
    in overnight rates that an otherwise smooth curve cannot capture.
    The market resolution is to quote a **turn** instrument — the
    expected forward rate over the short window straddling the
    discontinuity — and to inject it as a calibration anchor.

    The residual pins the curve's simply-compounded forward rate over
    the window :math:`[T_{\\text{turn}} - \\text{days\\_before},
    T_{\\text{turn}} + \\text{days\\_after}]` to ``jump_size``:

    .. math::
        \\frac{1}{\\tau}\\!\\left(\\frac{DF(T_-)}{DF(T_+)} - 1\\right)
        - \\text{jump\\_size} = 0.

    .. note::

       Turn instruments require an interpolation method that admits
       discontinuities for the bootstrap to converge to a sharp jump
       at the turn date.  Log-linear-DF interpolation (VALAX's current
       only mode) produces *piecewise-constant* forward rates between
       pillars, so the jump can be modelled by placing two pillars
       tightly around the turn date — the segment between them is then
       a single isolated forward.  Genuine step-function interpolation
       (calendar-aware, with explicit jump segments) is part of
       MC-Curves-3.

    Attributes:
        turn_date: The date of the discontinuity (typically year-end
            or quarter-end), as an ordinal integer.
        jump_size: The simply-compounded forward rate over the turn
            window, in rate units (e.g. ``0.0150`` for a 150 bp
            year-end rate spike).
        accrual_days_before: Days before ``turn_date`` to start the
            window (default 1).  Static field — set at construction.
        accrual_days_after: Days after ``turn_date`` to end the
            window (default 1).  Static.
        day_count: Day count convention for :math:`\\tau`.
        curves_touched: 1-tuple naming the curve to which the jump
            applies.  Defaults to the single-curve sentinel.
    """

    turn_date: Int[Array, ""]
    jump_size: Float[Array, ""]
    accrual_days_before: int = eqx.field(static=True, default=1)
    accrual_days_after: int = eqx.field(static=True, default=1)
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
        """Repricing residual: window forward rate − ``jump_size``."""
        del fixings, ref_date  # turn instruments don't use either
        curve = graph[self.curves_touched[0]]
        t0 = self.turn_date - self.accrual_days_before
        t1 = self.turn_date + self.accrual_days_after
        tau = year_fraction(t0, t1, self.day_count)
        df0 = curve(t0)
        df1 = curve(t1)
        window_forward = (df0 / df1 - 1.0) / tau
        return window_forward - self.jump_size


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
