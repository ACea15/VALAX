"""Present value of swap legs and multi-leg trades.

Prices the data-only pytrees of :mod:`valax.instruments.legs` against a
:class:`~valax.curves.graph.CurveGraph`, reading each leg's *own*
``discount_id`` / ``forward_id`` so a trade whose legs project off
different indices is priced from one call.

Relationship to the existing pricers
------------------------------------
The arithmetic here is not new. It is the generalisation of three
expressions that already existed in narrower form, and each is
reproduced **bit for bit** on its own path:

* :func:`valax.pricing.analytic.swaptions.annuity` — the fixed-leg
  annuity, recovered by :func:`fixed_leg_pv` on a flat leg.
* :func:`valax.curves.instruments._compounded_float_leg_pv` — the
  compounded floating leg, recovered by :func:`floating_leg_pv` with
  ``compounding="compounded"`` and no valuation date.
* :func:`valax.curves.instruments._floating_leg_pv` — the dual-curve
  simple leg with realised fixings, recovered by :func:`floating_leg_pv`
  with ``compounding="simple"``.

Those functions are left in place and untouched; this module does not
refactor them. The parity is asserted with bare ``==`` in
``tests/test_pricing/test_legs.py`` so that the generalisation cannot
drift away from the numbers the curve bootstrap and the swap pricers
already produce.

The seasoning cut-over
----------------------
A floating period is realised, straddling or unborn depending on where
the valuation date :math:`t_0` falls, and the three cases are usually
written as three branches. They collapse into one expression by clipping
:math:`t_0` into each period:

.. math::
    c_i = \\min\\bigl(\\max(t_0, T_i^s),\\, T_i^e\\bigr),

so :math:`c_i = T_i^s` for a future period, :math:`t_0` for the
straddling one and :math:`T_i^e` for a settled one. A compounded coupon
is then :math:`A_i\\, P_f(0,c_i)/P_f(0,T_i^e) - 1` throughout, where
:math:`A_i` is the realised compounding factor over
:math:`[T_i^s, c_i)` taken from the fixing history. Each case falls out:
:math:`A_i = 1` leaves pure projection, and at :math:`c_i = T_i^e` the
projection ratio is exactly ``1.0`` and the coupon is the realised
:math:`A_i - 1`. The same cut-over date drives both the realised factor
and the projection, which is what keeps the two from double-counting the
day :math:`t_0` itself.
"""

from functools import singledispatch
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.curves.fixings import FixingHistory
from valax.curves.graph import CurveGraph

# The averaged-overnight convention already has exactly one
# implementation, written for averaged money-market futures.  Importing
# it keeps a single definition of "arithmetic average of the daily
# forwards"; re-deriving it here would create a second one that could
# drift.
from valax.curves.instruments import _arithmetic_average_forward
from valax.dates.daycounts import year_fraction
from valax.instruments.legs import FixedLeg, FloatingLeg, Leg, MultiLegSwap


def _scale_by_amortisation(
    flows: Float[Array, " n"],
    amortisation: Optional[Float[Array, " n"]],
) -> Float[Array, " n"]:
    """Scale per-period flows by outstanding-notional factors.

    Applied to the *fully formed* per-period flow, so the notional stays
    outside the summation and a leg with no amortisation keeps the exact
    operand order of the pricer it generalises. ``None`` short-circuits
    to the input array rather than multiplying by ones, which makes the
    preserved path a static Python branch instead of a numerical
    coincidence — though multiplying by exactly ``1.0`` would be exact
    in IEEE-754 anyway, and the tests pin both routes.

    Args:
        flows: Per-period cashflow contributions.
        amortisation: Outstanding-notional factors, or ``None`` for a
            flat leg.

    Returns:
        The scaled flows.
    """
    if amortisation is None:
        return flows
    return amortisation * flows


def _realised_factors(
    leg: FloatingLeg,
    fixings: Optional[FixingHistory],
    cut_dates: Int[Array, " n"],
) -> Float[Array, " n"]:
    """Realised compounding factor for each period, up to its cut-over.

    Period :math:`i` compounds the recorded fixings over
    :math:`[T_i^s, c_i)`. The window is half-open at the top, so a rate
    recorded *on* the cut-over date is excluded — the overnight rate set
    on the valuation date is earned over the following day and has not
    happened yet. See
    :meth:`valax.curves.fixings.FixingSeries.compounded_factor`.

    The loop is over the leg's period count, which is static, and each
    iteration is a vectorised reduction over the fixing series. An index
    with no recorded history yields exactly ``1.0`` for every period.

    Args:
        leg: The floating leg being priced.
        fixings: Recorded fixing histories, or ``None`` for none.
        cut_dates: Per-period cut-over ordinals.

    Returns:
        One realised compounding factor per period.
    """
    if fixings is None:
        return jnp.ones(leg.n_periods)
    return jnp.stack(
        [
            fixings.compounded_factor(
                leg.index_id,
                leg.accrual_start_dates[i],
                cut_dates[i],
                leg.day_count,
            )
            for i in range(leg.n_periods)
        ]
    )


def fixed_leg_pv(leg: FixedLeg, graph: CurveGraph) -> Float[Array, ""]:
    """Present value of a fixed leg.

    .. math::
        \\text{PV} = \\pm N \\sum_i a_i\\, c\\, \\tau_i\\, P_d(0,T_i^p)

    The accrual fraction :math:`\\tau_i` is measured over the accrual
    period while the discount factor is read at the payment date, so a
    payment lag is honoured by construction.

    Args:
        leg: The fixed leg to value.
        graph: Curve graph supplying the leg's discount curve.

    Returns:
        The leg's present value, negated when ``leg.pay`` is set.
    """
    curve = graph[leg.discount_id]
    taus = year_fraction(
        leg.accrual_start_dates, leg.accrual_end_dates, leg.day_count
    )
    flows = _scale_by_amortisation(
        taus * curve(leg.payment_dates), leg.amortisation
    )
    pv = leg.notional * leg.fixed_rate * jnp.sum(flows)
    return -pv if leg.pay else pv


def floating_leg_pv(
    leg: FloatingLeg,
    graph: CurveGraph,
    fixings: Optional[FixingHistory] = None,
    ref_date: Optional[Int[Array, ""]] = None,
) -> Float[Array, ""]:
    """Present value of a floating leg.

    Dispatches on ``leg.compounding`` (a static field, so the branch is
    resolved at trace time):

    * ``"compounded"`` — the coupon is the compounding ratio
      :math:`A_i P_f(0,c_i)/P_f(0,T_i^e) - 1`. The day count cancels out
      of the index term by the definition of the simply-compounded
      forward, so :math:`\\tau_i` appears only on the spread.
    * ``"simple"`` — one observed rate per period, with precedence
      *recorded fixing* → *straddling blend* → *projection*.
    * ``"averaged"`` — the arithmetic average of the daily forwards,
      which discards interest-on-interest.

    Seasoning requires **both** a ``ref_date`` and a history for the
    leg's index; absent either, every period is projected in full by a
    pure-projection path that reproduces the pre-existing pricers bit for
    bit. At a valuation date before the leg starts the two routes agree
    exactly, since nothing has been realised.

    Args:
        leg: The floating leg to value.
        graph: Curve graph supplying the discount and projection curves.
        fixings: Recorded fixing histories, consulted under
            ``leg.index_id``. ``None``, or a history with no entry for
            this leg's index, projects every period in full regardless of
            ``ref_date``.
        ref_date: Valuation date (ordinal). Needed, together with a
            history for the leg's index, to season the leg; without both
            every period is projected in full.

    Returns:
        The leg's present value, negated when ``leg.pay`` is set.

    Raises:
        ValueError: If an averaged leg is asked to season against a
            recorded history, which is not derivable from the available
            reduction.
    """
    discount_curve = graph[leg.discount_id]
    forward_curve = (
        discount_curve
        if leg.forward_id is None
        else graph[leg.forward_id]
    )

    starts = leg.accrual_start_dates
    ends = leg.accrual_end_dates
    taus = year_fraction(starts, ends, leg.day_count)
    df_ends_fwd = forward_curve(ends)

    # Whether this leg's index has any recorded history is a *static*
    # question -- a ``FixingHistory``'s keys are pytree metadata, not
    # leaves -- so it selects a branch at trace time.
    #
    # Seasoning needs both a valuation date and a history. Requiring the
    # history too is what keeps ``fixings=None`` meaning "project every
    # period in full" even when a ``ref_date`` is supplied: otherwise a
    # period already settled would take its cut-over at its own accrual
    # end, where the projection ratio is exactly 1.0, and silently pay a
    # zero index coupon instead of a projected one.
    has_history = fixings is not None and leg.index_id in fixings.indices
    seasoned = ref_date is not None and has_history

    if leg.compounding == "compounded":
        if not seasoned:
            # Preserved path: pure projection, in the exact operand
            # order of ``_compounded_float_leg_pv``.
            index_coupon = forward_curve(starts) / df_ends_fwd - 1.0
        else:
            cut = jnp.clip(jnp.asarray(ref_date), starts, ends)
            realised = _realised_factors(leg, fixings, cut)
            index_coupon = (
                realised * forward_curve(cut) / df_ends_fwd - 1.0
            )
        coupon = index_coupon + leg.spread * taus
        flows = _scale_by_amortisation(
            discount_curve(leg.payment_dates) * coupon, leg.amortisation
        )
        pv = leg.notional * jnp.sum(flows)
        return -pv if leg.pay else pv

    if leg.compounding == "averaged":
        if has_history:
            raise ValueError(
                f"Cannot season an averaged leg: index "
                f"{leg.index_id!r} has a recorded history, but the "
                f"realised part of an arithmetic average is a *sum* of "
                f"recorded rates and FixingSeries only reduces them to "
                f"a compounding *product*. Using that product here "
                f"would silently price the wrong convention. Price the "
                f"leg with compounding='compounded', or value it with "
                f"no history for its index."
            )
        rates = jnp.stack(
            [
                _arithmetic_average_forward(
                    forward_curve, starts[i], ends[i], taus[i]
                )
                for i in range(leg.n_periods)
            ]
        )
    else:  # "simple"
        projected = (forward_curve(starts) / df_ends_fwd - 1.0) / taus
        if has_history:
            realised_rates = fixings.lookup_many(
                leg.index_id, leg.fixing_dates
            )
            has_realised = ~jnp.isnan(realised_rates)
            # ``jnp.where`` over a NaN-free replacement keeps the
            # unselected branch's gradient at exactly zero, so a
            # recorded coupon carries no projection-curve sensitivity
            # (spec T-11).
            rates = jnp.where(
                has_realised, jnp.nan_to_num(realised_rates), projected
            )
        else:
            has_realised = jnp.zeros(taus.shape, dtype=bool)
            rates = projected
        if seasoned:
            ref = jnp.asarray(ref_date)
            # Half-open: closing both ends would match two periods
            # whenever the valuation date lands on an accrual boundary,
            # and they cannot share one realised factor.
            straddles = (starts <= ref) & (ends > ref)
            cut = jnp.clip(ref, starts, ends)
            blended = (
                _realised_factors(leg, fixings, cut)
                * forward_curve(ref)
                / df_ends_fwd
                - 1.0
            ) / taus
            rates = jnp.where(straddles & ~has_realised, blended, rates)

    flows = _scale_by_amortisation(
        (rates + leg.spread) * taus * discount_curve(leg.payment_dates),
        leg.amortisation,
    )
    pv = leg.notional * jnp.sum(flows)
    return -pv if leg.pay else pv


@singledispatch
def leg_pv(
    leg: Leg,
    graph: CurveGraph,
    fixings: Optional[FixingHistory] = None,
    ref_date: Optional[Int[Array, ""]] = None,
) -> Float[Array, ""]:
    """Present value of a leg of either kind.

    Dispatches on the leg's type, so a heterogeneous tuple of legs can
    be priced by a single call per leg. ``fixings`` and ``ref_date`` are
    accepted for both kinds and ignored by a fixed leg, which has nothing
    to fix.

    Args:
        leg: The leg to value.
        graph: Curve graph supplying the leg's curves.
        fixings: Recorded fixing histories.
        ref_date: Valuation date (ordinal).

    Returns:
        The leg's present value, signed by its ``pay`` flag.

    Raises:
        TypeError: If ``leg`` is not a known leg type.
    """
    raise TypeError(
        f"leg_pv has no implementation for {type(leg).__name__}; "
        f"expected a FixedLeg or a FloatingLeg."
    )


@leg_pv.register
def _fixed_leg_pv_dispatch(
    leg: FixedLeg,
    graph: CurveGraph,
    fixings: Optional[FixingHistory] = None,
    ref_date: Optional[Int[Array, ""]] = None,
) -> Float[Array, ""]:
    del fixings, ref_date  # a fixed leg has nothing to fix
    return fixed_leg_pv(leg, graph)


@leg_pv.register
def _floating_leg_pv_dispatch(
    leg: FloatingLeg,
    graph: CurveGraph,
    fixings: Optional[FixingHistory] = None,
    ref_date: Optional[Int[Array, ""]] = None,
) -> Float[Array, ""]:
    return floating_leg_pv(leg, graph, fixings, ref_date)


def swap_pv(
    swap: MultiLegSwap,
    graph: CurveGraph,
    fixings: Optional[FixingHistory] = None,
    ref_date: Optional[Int[Array, ""]] = None,
) -> Float[Array, ""]:
    """Present value of a multi-leg trade.

    The sum of the legs' present values, each already signed by its own
    ``pay`` flag. Every leg must discount in the same currency; see
    :class:`~valax.instruments.legs.MultiLegSwap` on why no currency
    conversion happens here.

    Args:
        swap: The trade to value.
        graph: Curve graph supplying every curve in
            ``swap.curves_touched``.
        fixings: Recorded fixing histories for ``swap.indices_touched``.
        ref_date: Valuation date (ordinal).

    Returns:
        The trade's present value.
    """
    total = leg_pv(swap.legs[0], graph, fixings, ref_date)
    for leg in swap.legs[1:]:
        total = total + leg_pv(leg, graph, fixings, ref_date)
    return total


def swap_fair_rate(
    swap: MultiLegSwap,
    graph: CurveGraph,
    fixings: Optional[FixingHistory] = None,
    ref_date: Optional[Int[Array, ""]] = None,
) -> Float[Array, ""]:
    """Fixed rate that sets a trade's present value to zero.

    The present value is affine in the fixed rate — a fixed leg's PV is
    proportional to it and no other leg depends on it — so the par rate
    follows from one annuity and the rest of the trade, with no solve:

    .. math::
        c^\\star = -\\frac{\\text{PV}_{\\text{other legs}}}
                          {\\partial \\text{PV} / \\partial c}.

    The derivative is taken with :func:`jax.grad` rather than rebuilt by
    hand, so it cannot drift from :func:`fixed_leg_pv`.

    Args:
        swap: The trade to solve. Must contain exactly one fixed leg.
        graph: Curve graph supplying every curve the trade reads.
        fixings: Recorded fixing histories.
        ref_date: Valuation date (ordinal).

    Returns:
        The par fixed rate.

    Raises:
        ValueError: If the trade does not have exactly one fixed leg, in
            which case "the" fixed rate is not well defined.
    """
    fixed_positions = [
        i for i, leg in enumerate(swap.legs) if isinstance(leg, FixedLeg)
    ]
    if len(fixed_positions) != 1:
        raise ValueError(
            f"swap_fair_rate needs exactly one fixed leg to solve for, "
            f"found {len(fixed_positions)}. With none there is no rate "
            f"to set; with several the par rate is underdetermined "
            f"unless you say how they move together."
        )
    position = fixed_positions[0]
    fixed_leg = swap.legs[position]

    def pv_at_rate(rate: Float[Array, ""]) -> Float[Array, ""]:
        repriced = eqx.tree_at(
            lambda s: s.legs[position].fixed_rate, swap, rate
        )
        return swap_pv(repriced, graph, fixings, ref_date)

    zero = jnp.zeros_like(fixed_leg.fixed_rate)
    pv_without_fixed = pv_at_rate(zero)
    sensitivity = jax.grad(pv_at_rate)(zero)
    return -pv_without_fixed / sensitivity
