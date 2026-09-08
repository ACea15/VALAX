r"""Model-agnostic node valuations for the short-rate PDE recipes.

The Hull-White, G2++ and Cheyette finite-difference recipes all price the same
family of vanilla-rates payoffs — coupon bonds, swaps, caplets/caps — and the
payoff arithmetic is *identical* across them: it only ever needs

1. the model's **affine bond value at the grid nodes**, and
2. the instrument **schedule** (year fractions, accruals, cash flows).

Everything model-specific (the affine coefficients, the exact-fit discount
shift :math:`\alpha` / :math:`\varphi`, the frozen- vs exact-``y`` accumulator)
lives *inside* the bond valuation; once a recipe hands this module a way to
value bonds at its grid nodes, the swap-tail, coupon-injection and caplet
payoffs are written once here and shared.

Per-model interface
-------------------
Recipes pass two thin closures over ``(model, grid)``:

``bond_portfolio(t, payment_times, cashflows) -> node values``
    Value at time ``t`` of :math:`\sum_i c_i P(t, T_i)` at every grid node,
    from the model's affine bond price. Used for swap tails and terminal swap
    payoffs.

``discount_factor(t, payment_time) -> node values``
    Value at time ``t`` of a single :math:`P(t, T)` at every grid node. Used
    for the cashflow-timing-exact coupon injection and the caplet ZCB.

The node array is 1-D (``(n,)``) for the one-factor recipes and 2-D
(``(n_x, n_y)``) for the two-factor / exact-``y`` recipes; the builders are
written to broadcast over either, so ``node_shape`` is passed where a zero
scatter target is needed.

Caplet as a bond option
-----------------------
A caplet paying :math:`N\,\tau\,(F - K)^+` at the accrual end :math:`T_1`, with
:math:`F` the simply-compounded forward observed at the fixing :math:`T_0`,
equals :math:`N\,(1 + K\tau)` put options on the zero-coupon bond
:math:`P(T_0, T_1)` struck at :math:`1/(1 + K\tau)`. Discounting that payoff
from :math:`T_1` back to :math:`T_0` collapses to a value **at the fixing** of

.. math::

    N\,\bigl(1 - P(T_0, T_1)\,(1 + K\tau)\bigr)^+

per unit at each node (a floorlet flips the sign inside the positive part).
Because this is already the fixing-date value as a function of the node state,
a cap prices as a single backward sweep with each caplet's value injected
additively at its fixing level and discounted home — the exact mechanism the
coupon injection uses.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

from valax.dates.daycounts import year_fraction

# A closure valuing a cash-flow portfolio at the grid nodes: given the
# valuation time, the payment times and the cash flows, it returns the node
# value array (1-D or 2-D).
BondPortfolio = Callable[
    [Float[Array, ""], Float[Array, " n_cf"], Float[Array, " n_cf"]],
    Float[Array, "..."],
]
# A closure valuing a single zero-coupon bond at the grid nodes.
DiscountFactor = Callable[
    [Float[Array, ""], Float[Array, ""]], Float[Array, "..."]
]


# ─────────────────────────────────────────────────────────────────────
# Schedule helpers
# ─────────────────────────────────────────────────────────────────────


def snap_to_levels(
    dates: Int[Array, " m"],
    reference_date: Int[Array, ""],
    day_count: str,
    dt: Float[Array, ""],
    n_time: int,
) -> Int[Array, " m"]:
    """Map ordinal dates to the nearest backward-sweep time level.

    Levels count forward time: ``0`` at the reference date, ``n_time`` at the
    horizon. Traced indices, so the recipes stay JIT/grad-friendly.

    Args:
        dates: Event dates as integer ordinals.
        reference_date: Curve reference date as an integer ordinal.
        day_count: Day-count convention used to convert to year fractions.
        dt: Time-step size in year fractions.
        n_time: Number of backward time steps.

    Returns:
        The clipped time-level index of each date.
    """
    times = year_fraction(reference_date, dates, day_count)
    return jnp.clip(jnp.round(times / dt).astype(jnp.int32), 0, n_time)


def swap_accruals(
    start_date: Int[Array, ""],
    payment_dates: Int[Array, " n"],
    day_count: str,
) -> Float[Array, " n"]:
    """Year fractions for each fixed-leg accrual period.

    The first period accrues from ``start_date`` (the swap start / option
    exercise date); subsequent periods run between consecutive payment dates.

    Args:
        start_date: Swap effective date as an integer ordinal.
        payment_dates: Fixed-leg payment dates as integer ordinals.
        day_count: Day-count convention.

    Returns:
        The accrual factor for each fixed payment.
    """
    starts = jnp.concatenate([start_date[jnp.newaxis], payment_dates[:-1]])
    return year_fraction(starts, payment_dates, day_count)


# ─────────────────────────────────────────────────────────────────────
# Swap payoffs
# ─────────────────────────────────────────────────────────────────────


def swap_tail_values(
    bond_portfolio: BondPortfolio,
    reference_date: Int[Array, ""],
    exercise_date: Int[Array, ""],
    payment_dates: Int[Array, " n_cf"],
    strike: Float[Array, ""],
    notional: Float[Array, ""],
    day_count: str,
    is_payer: bool,
) -> Float[Array, "..."]:
    r"""Value of the underlying swap at ``exercise_date``, at every grid node.

    On unit notional a fixed-for-float swap entered at :math:`T_0` is worth
    :math:`1 - \sum_i c_i P(T_0, T_i)` to the payer, with :math:`c_i = K\tau_i`
    and the principal folded into the final flow — the standard replication of
    the floating leg. The receiver's value is the negative of that. Computed
    analytically from the model's affine bond price via ``bond_portfolio``, so
    exercise decisions carry no discretisation error of their own.

    Args:
        bond_portfolio: Node-level valuer for :math:`\sum_i c_i P(t, T_i)`.
        reference_date: Curve reference date as an integer ordinal.
        exercise_date: Swap start / option exercise date as an integer ordinal.
        payment_dates: Fixed-leg payment dates as integer ordinals.
        strike: Fixed rate.
        notional: Notional principal.
        day_count: Day-count convention.
        is_payer: ``True`` for a payer swap, ``False`` for a receiver.

    Returns:
        The swap value at each grid node.
    """
    t = year_fraction(reference_date, exercise_date, day_count)
    payment_times = year_fraction(reference_date, payment_dates, day_count)

    taus = swap_accruals(exercise_date, payment_dates, day_count)
    cashflows = (strike * taus).at[-1].add(1.0)

    bond = bond_portfolio(t, payment_times, cashflows)
    payer = notional * (1.0 - bond)
    return payer if is_payer else -payer


# ─────────────────────────────────────────────────────────────────────
# Coupon injection
# ─────────────────────────────────────────────────────────────────────


def coupon_values_by_level(
    discount_factor: DiscountFactor,
    reference_date: Int[Array, ""],
    payment_dates: Int[Array, " n_cf"],
    day_count: str,
    coupon: Float[Array, ""],
    dt: Float[Array, ""],
    n_time: int,
    node_shape: tuple[int, ...],
) -> Float[Array, "n_levels ..."]:
    r"""Coupon value to inject at each time level, per grid node.

    A coupon due at :math:`t_c` is attached to the nearest time level
    :math:`t_k` and scaled by the analytic discount factor
    :math:`P(t_k, t_c \mid \text{node})`, which removes the cashflow-timing
    error that plain date-snapping would introduce. The factor is the correct
    analytic continuation for either sign of :math:`t_c - t_k`.

    Args:
        discount_factor: Node-level valuer for a single :math:`P(t, T)`.
        reference_date: Curve reference date as an integer ordinal.
        payment_dates: Coupon payment dates as integer ordinals.
        day_count: Day-count convention.
        coupon: Coupon amount (per payment date).
        dt: Time-step size in year fractions.
        n_time: Number of backward time steps.
        node_shape: Shape of the spatial grid (``(n,)`` or ``(n_x, n_y)``).

    Returns:
        Coupon value indexed by ``(time level, *node)``; zero where no coupon
        is due.
    """
    payment_times = year_fraction(reference_date, payment_dates, day_count)
    levels = jnp.clip(
        jnp.round(payment_times / dt).astype(jnp.int32), 0, n_time
    )
    factors = jax.vmap(discount_factor)(levels * dt, payment_times)
    return (
        jnp.zeros((n_time + 1, *node_shape)).at[levels].add(coupon * factors)
    )


# ─────────────────────────────────────────────────────────────────────
# Caplet / cap / floor payoffs
# ─────────────────────────────────────────────────────────────────────


def caplet_fixing_values(
    bond_at_fixing: Float[Array, "..."],
    strike: Float[Array, ""],
    tau: Float[Array, ""],
    notional: Float[Array, ""],
    is_cap: bool,
) -> Float[Array, "..."]:
    r"""Fixing-date value of a single caplet/floorlet at every grid node.

    A caplet is :math:`N` puts on the accrual-end zero-coupon bond, worth
    :math:`N\,(1 - P(T_0, T_1)(1 + K\tau))^+` at the fixing; a floorlet flips
    the sign inside the positive part.

    Args:
        bond_at_fixing: :math:`P(T_0, T_1)` at every grid node.
        strike: Cap/floor rate :math:`K`.
        tau: Accrual fraction :math:`\tau` for the period.
        notional: Notional principal.
        is_cap: ``True`` for a caplet, ``False`` for a floorlet.

    Returns:
        The caplet/floorlet value at each grid node.
    """
    moneyness = 1.0 - bond_at_fixing * (1.0 + strike * tau)
    payoff = (
        jnp.maximum(moneyness, 0.0) if is_cap else jnp.maximum(-moneyness, 0.0)
    )
    return notional * payoff


def caplet_values_by_level(
    discount_factor: DiscountFactor,
    reference_date: Int[Array, ""],
    fixing_dates: Int[Array, " n_caplets"],
    start_dates: Int[Array, " n_caplets"],
    end_dates: Int[Array, " n_caplets"],
    day_count: str,
    strike: Float[Array, ""],
    notional: Float[Array, ""],
    is_cap: bool,
    dt: Float[Array, ""],
    n_time: int,
    node_shape: tuple[int, ...],
) -> Float[Array, "n_levels ..."]:
    r"""Caplet fixing-value to inject at each time level, per grid node.

    Each caplet in the strip contributes its fixing-date value
    (:func:`caplet_fixing_values`) at its (snapped) fixing level; the backward
    sweep then discounts every contribution home, so a whole cap/floor prices
    in one pass. The zero-coupon bond :math:`P(t_k, T_1)` is evaluated at the
    snapped fixing level ``t_k`` against the true accrual end, mirroring the
    coupon-injection treatment.

    Args:
        discount_factor: Node-level valuer for a single :math:`P(t, T)`.
        reference_date: Curve reference date as an integer ordinal.
        fixing_dates: Rate fixing date for each caplet, as integer ordinals.
        start_dates: Accrual start date for each caplet, as integer ordinals.
        end_dates: Accrual end (= payment) date for each caplet, as ordinals.
        day_count: Day-count convention.
        strike: Uniform cap/floor rate :math:`K`.
        notional: Notional principal.
        is_cap: ``True`` for a cap, ``False`` for a floor.
        dt: Time-step size in year fractions.
        n_time: Number of backward time steps.
        node_shape: Shape of the spatial grid (``(n,)`` or ``(n_x, n_y)``).

    Returns:
        Caplet value indexed by ``(time level, *node)``; zero where no caplet
        fixes.
    """
    fixing_times = year_fraction(reference_date, fixing_dates, day_count)
    end_times = year_fraction(reference_date, end_dates, day_count)
    taus = year_fraction(start_dates, end_dates, day_count)
    levels = jnp.clip(
        jnp.round(fixing_times / dt).astype(jnp.int32), 0, n_time
    )

    def caplet_val(level_time, end_time, tau):
        bond = discount_factor(level_time, end_time)
        return caplet_fixing_values(bond, strike, tau, notional, is_cap)

    vals = jax.vmap(caplet_val)(levels * dt, end_times, taus)
    return jnp.zeros((n_time + 1, *node_shape)).at[levels].add(vals)
