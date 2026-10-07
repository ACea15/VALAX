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

from valax.dates.accrual import bond_accrued_interest_at_dates
from valax.dates.daycounts import year_fraction
from valax.pricing.multicurve import apply_swap_basis

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
    basis: Float[Array, " n_cf"] | None = None,
) -> Float[Array, "..."]:
    r"""Value of the underlying swap at ``exercise_date``, at every grid node.

    On unit notional a fixed-for-float swap entered at :math:`T_0` is worth
    :math:`1 - \sum_i c_i P(T_0, T_i)` to the payer, with :math:`c_i = K\tau_i`
    and the principal folded into the final flow — the standard replication of
    the floating leg. The receiver's value is the negative of that. Computed
    analytically from the model's affine bond price via ``bond_portfolio``, so
    exercise decisions carry no discretisation error of their own.

    **Dual curve.** When the floating leg projects off a curve distinct from
    the discount curve, that telescoping replication no longer collapses.
    Under the frozen tenor basis :math:`b_j` of
    :func:`~valax.pricing.multicurve.tenor_basis` the floating leg is

    .. math::

        \sum_j \bigl(b_j P(T_0, T_{j-1}) - P(T_0, T_j)\bigr)
          = b_1 + \sum_{j<n} (b_{j+1} - 1)\,P(T_0, T_j) - P(T_0, T_n),

    using :math:`P(T_0, T_0) = 1`. Folding the extra :math:`(1 - b_{j+1})`
    terms into the cash-flow vector keeps the payoff a *single* call to
    ``bond_portfolio``, so the dual-curve tail costs exactly what the
    single-curve one did; setting every :math:`b_j = 1` recovers the
    single-curve expression term by term. ``basis=None`` skips the arithmetic
    altogether, so a single-curve price is unchanged bit-for-bit.

    The floating leg is assumed to pay on the fixed leg's schedule — the same
    assumption the single-curve replication already makes.

    Args:
        bond_portfolio: Node-level valuer for :math:`\sum_i c_i P(t, T_i)`.
        reference_date: Curve reference date as an integer ordinal.
        exercise_date: Swap start / option exercise date as an integer ordinal.
        payment_dates: Fixed-leg payment dates as integer ordinals.
        strike: Fixed rate.
        notional: Notional principal.
        day_count: Day-count convention.
        is_payer: ``True`` for a payer swap, ``False`` for a receiver.
        basis: Per-period frozen tenor basis, period :math:`j` spanning
            :math:`(T_{j-1}, T_j)` with :math:`T_0` the exercise date.
            ``None`` for single-curve projection.

    Returns:
        The swap value at each grid node.
    """
    t = year_fraction(reference_date, exercise_date, day_count)
    payment_times = year_fraction(reference_date, payment_dates, day_count)

    taus = swap_accruals(exercise_date, payment_dates, day_count)
    cashflows = (strike * taus).at[-1].add(1.0)

    float_start, cashflows = apply_swap_basis(cashflows, basis)

    bond = bond_portfolio(t, payment_times, cashflows)
    payer = notional * (float_start - bond)
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
        coupon: Coupon amount. Either a scalar applied to every payment
            date, or one amount per payment date — the latter being the
            normal case, since day-count accrual makes coupon periods
            unequal.
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
    coupon = jnp.asarray(coupon)
    if coupon.ndim == 1:
        # Align the per-payment axis with `factors` before broadcasting
        # across the spatial grid.
        coupon = coupon.reshape(coupon.shape[0], *([1] * len(node_shape)))
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
    basis: Float[Array, ""] | None = None,
) -> Float[Array, "..."]:
    r"""Fixing-date value of a single caplet/floorlet at every grid node.

    A caplet is :math:`N` puts on the accrual-end zero-coupon bond, worth
    :math:`N\,(1 - P(T_0, T_1)(1 + K\tau))^+` at the fixing; a floorlet flips
    the sign inside the positive part.

    **Dual curve.** The rate that fixes is projected off the forward curve
    while the payment is discounted on the model's (discount) curve. Under
    the frozen tenor basis :math:`b` of
    :func:`~valax.pricing.multicurve.tenor_basis`,
    :math:`1 + \tau F = b / P(T_0, T_1)`, so the fixing value becomes

    .. math::

        N\,\bigl(b - P(T_0, T_1)\,(1 + K\tau)\bigr)^+ ,

    i.e. the basis simply displaces the strike side of the option. ``b = 1``
    is the single-curve case, and ``basis=None`` keeps that expression
    literally unchanged.

    Args:
        bond_at_fixing: :math:`P(T_0, T_1)` at every grid node.
        strike: Cap/floor rate :math:`K`.
        tau: Accrual fraction :math:`\tau` for the period.
        notional: Notional principal.
        is_cap: ``True`` for a caplet, ``False`` for a floorlet.
        basis: Frozen tenor basis over :math:`(T_0, T_1)`, or ``None`` for
            single-curve projection.

    Returns:
        The caplet/floorlet value at each grid node.
    """
    intrinsic = 1.0 if basis is None else basis
    moneyness = intrinsic - bond_at_fixing * (1.0 + strike * tau)
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
    basis: Float[Array, " n_caplets"] | None = None,
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
        basis: Per-caplet frozen tenor basis over each accrual period (see
            :func:`caplet_fixing_values`), or ``None`` for single-curve
            projection.

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
    def caplet_val(level_time, end_time, tau, b):
        bond = discount_factor(level_time, end_time)
        return caplet_fixing_values(bond, strike, tau, notional, is_cap, b)

    # ``basis is None`` is a *trace-time* branch, not a traced value: it keeps
    # the single-curve arithmetic literally unchanged rather than multiplying
    # through by a ones array.
    if basis is None:
        vals = jax.vmap(lambda lt, et, tau: caplet_val(lt, et, tau, None))(
            levels * dt, end_times, taus
        )
    else:
        vals = jax.vmap(caplet_val)(levels * dt, end_times, taus, basis)
    return jnp.zeros((n_time + 1, *node_shape)).at[levels].add(vals)


# ─────────────────────────────────────────────────────────────────────
# CMS-spread coupon injection
# ─────────────────────────────────────────────────────────────────────


def cms_rate_by_state(
    discount_factor: DiscountFactor,
    t_fix: Float[Array, ""],
    tenor: int,
) -> Float[Array, "..."]:
    r"""Par swap rate of an annual ``tenor``-year swap fixed at ``t_fix``, per node.

    Rebuilt at every grid node from the model's affine zero-coupon bonds — the
    exact per-node analogue of the Monte-Carlo CMS rate:

    .. math::

        S = \frac{1 - P(t_\text{fix},\, t_\text{fix} + \text{tenor})}
                 {\sum_{j=1}^{\text{tenor}} P(t_\text{fix},\, t_\text{fix} + j)}

    with annual accruals and a unit year fraction per period (the convention
    baked into the MC recipe). ``tenor`` is a static whole-year count, so the
    zero-coupon strip unrolls at trace time.

    Args:
        discount_factor: Node-level valuer for a single :math:`P(t, T)`.
        t_fix: Fixing time in year fractions.
        tenor: Underlying swap tenor in whole years (static).

    Returns:
        The forward par swap rate at each grid node.
    """
    js = jnp.arange(1, tenor + 1, dtype=jnp.float64)
    maturities = t_fix + js
    fixings = jnp.broadcast_to(t_fix, maturities.shape)
    zcb = jax.vmap(discount_factor)(fixings, maturities)  # (tenor, *node)
    annuity = jnp.sum(zcb, axis=0)
    return (1.0 - zcb[-1]) / annuity


def cms_spread_values_by_level(
    discount_factor: DiscountFactor,
    reference_date: Int[Array, ""],
    start_date: Int[Array, ""],
    payment_dates: Int[Array, " n"],
    day_count: str,
    fixed_rate: Float[Array, ""],
    notional: Float[Array, ""],
    cms_tenor_long: int,
    cms_tenor_short: int,
    pay_fixed: bool,
    dt: Float[Array, ""],
    n_time: int,
    node_shape: tuple[int, ...],
) -> Float[Array, "n_levels ..."]:
    r"""CMS-spread coupon value to inject at each time level, per grid node.

    Each period's coupon
    :math:`C_i = \pm N\,(S^\text{long} - S^\text{short} - K)\,\tau_i` is a
    *linear* function of two par swap rates (:func:`cms_rate_by_state`) observed
    at the accrual start, so — unlike a caplet — it carries no kink and injects
    additively with no positive part. The CMS rates fix at the accrual start
    (the previous payment date, or ``start_date`` for the first period) while
    each coupon pays at the accrual end; the fixing snaps to the nearest time
    level and the coupon is scaled by the node-level discount factor
    :math:`P(t_\text{fix}, t_\text{pay})` from fixing to payment, so the backward
    sweep discounts the whole strip home in one pass (mirroring
    :func:`coupon_values_by_level`). ``pay_fixed`` selects a steepener
    (:math:`+`, receive the spread) or a flattener (:math:`-`).

    Args:
        discount_factor: Node-level valuer for a single :math:`P(t, T)`.
        reference_date: Curve reference date as an integer ordinal.
        start_date: Swap effective date / first accrual start, as an ordinal.
        payment_dates: Period-end payment dates as integer ordinals.
        day_count: Day-count convention for the accrual fractions.
        fixed_rate: Fixed spread strike :math:`K`.
        notional: Notional principal.
        cms_tenor_long: Long reference swap tenor in whole years (static).
        cms_tenor_short: Short reference swap tenor in whole years (static).
        pay_fixed: ``True`` for a steepener, ``False`` for a flattener.
        dt: Time-step size in year fractions.
        n_time: Number of backward time steps.
        node_shape: Shape of the spatial grid (``(n_x, n_y)``).

    Returns:
        Coupon value indexed by ``(time level, *node)``; zero where no CMS
        fixes.
    """
    pay_times = year_fraction(reference_date, payment_dates, day_count)
    starts = jnp.concatenate([start_date[jnp.newaxis], payment_dates[:-1]])
    fixing_times = year_fraction(reference_date, starts, day_count)
    taus = year_fraction(starts, payment_dates, day_count)
    levels = jnp.clip(
        jnp.round(fixing_times / dt).astype(jnp.int32), 0, n_time
    )
    sign = 1.0 if pay_fixed else -1.0

    def period_val(level_time, pay_time, tau):
        s_long = cms_rate_by_state(discount_factor, level_time, cms_tenor_long)
        s_short = cms_rate_by_state(discount_factor, level_time, cms_tenor_short)
        coupon = sign * (s_long - s_short - fixed_rate) * tau
        pay_factor = discount_factor(level_time, pay_time)
        return notional * coupon * pay_factor

    vals = jax.vmap(period_val)(levels * dt, pay_times, taus)
    return jnp.zeros((n_time + 1, *node_shape)).at[levels].add(vals)


# ─────────────────────────────────────────────────────────────────────
# Callable / puttable exercise projection
# ─────────────────────────────────────────────────────────────────────


def exercise_obstacle_by_level(
    exercise_dates: Int[Array, " m"],
    exercise_prices: Float[Array, " m"],
    face_value: Float[Array, ""],
    reference_date: Int[Array, ""],
    day_count: str,
    dt: Float[Array, ""],
    n_time: int,
    is_call: bool,
    accrued_at_exercise: Float[Array, " m"] | None = None,
) -> Float[Array, " n_levels"]:
    """Per-level exercise obstacle for a callable/puttable bond.

    Exercise dates snap to the nearest backward-sweep level and carry their
    redemption amount (``exercise_price * face``) there. Non-exercise levels
    get a **neutral** obstacle that never wins the projection: ``+inf`` for an
    issuer call (a ``min`` caps nothing), ``-inf`` for a holder put (a ``max``
    floors nothing).

    **Clean vs dirty.** Call and put prices are quoted *clean*, while the
    continuation value the obstacle is compared against is a present value
    of all remaining cashflows and is therefore *dirty*. A holder redeemed
    mid-period receives the redemption amount **plus** the interest accrued
    since the last coupon, so the obstacle is raised by that accrual and the
    comparison becomes like-for-like. Accrual is zero on a coupon date, so
    this changes nothing when exercise and coupon dates coincide — which is
    exactly why the discrepancy went unnoticed.

    Args:
        exercise_dates: Call or put dates as integer ordinals.
        exercise_prices: Redemption price at each exercise date, as a fraction
            of face value.
        face_value: Bond face value.
        reference_date: Curve reference date as an integer ordinal.
        day_count: Day-count convention.
        dt: Time-step size in year fractions.
        n_time: Number of backward time steps.
        is_call: ``True`` for issuer-optimal (callable), ``False`` for
            holder-optimal (puttable).
        accrued_at_exercise: Accrued interest owed on each *exercise date*
            (from
            :func:`~valax.dates.accrual.bond_accrued_interest_at_dates`).
            Measured on the true date rather than at the snapped level, so
            the obstacle does not depend on ``n_time``. ``None`` treats the
            quoted price as already dirty, which is only correct when every
            exercise date is a coupon date.

    Returns:
        The obstacle value at each time level, indexed ``0 .. n_time``.
    """
    levels = snap_to_levels(
        exercise_dates, reference_date, day_count, dt, n_time
    )
    neutral = jnp.inf if is_call else -jnp.inf
    redemption = exercise_prices * face_value
    if accrued_at_exercise is not None:
        redemption = redemption + accrued_at_exercise
    return jnp.full(n_time + 1, neutral).at[levels].set(redemption)


def exercisable_terminal_and_event(
    coupons: Float[Array, "n_levels ..."],
    face_value: Float[Array, ""],
    strike_at_level: Float[Array, " n_levels"],
    is_call: bool,
) -> tuple[Float[Array, "..."], Callable]:
    r"""Terminal condition and exercise event for a callable/puttable bond.

    The ordering within a step follows the trinomial tree: exercise is decided
    on the *ex-coupon* continuation value and the coupon is added afterwards,
    because a holder redeemed on a coupon date still receives that coupon. The
    issuer caps the value at the call price (``is_call``); the holder floors it
    at the put price.

    Being ex-coupon is not the same as being clean, and the obstacle must
    account for the difference: between coupon dates the continuation value
    carries accrued interest that the quoted redemption price does not. That
    adjustment lives in :func:`exercise_obstacle_by_level`, which raises the
    obstacle by the accrual owed on each exercise date, so
    ``strike_at_level`` arrives here already on the same (dirty) footing as
    ``values``.

    Args:
        coupons: Coupon value to inject per ``(time level, *node)`` (from
            :func:`coupon_values_by_level`).
        face_value: Bond face value (redeemed at the horizon).
        strike_at_level: Per-level exercise obstacle (from
            :func:`exercise_obstacle_by_level`).
        is_call: ``True`` for issuer-optimal (callable), ``False`` for
            holder-optimal (puttable).

    Returns:
        ``(terminal, event_fn)`` — the horizon condition (redemption plus final
        coupon) and the per-level exercise-then-coupon event.
    """
    node_shape = coupons.shape[1:]
    terminal = jnp.full(node_shape, face_value) + coupons[-1]

    def event_fn(level, values):
        obstacle = strike_at_level[level]
        exercised = (
            jnp.minimum(values, obstacle)
            if is_call
            else jnp.maximum(values, obstacle)
        )
        return exercised + coupons[level]

    return terminal, event_fn
