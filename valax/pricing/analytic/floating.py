"""Analytical pricing for floating-rate instruments: FRN and OIS swap.

Pure functions. All pricers are differentiable via ``jax.grad``, jittable,
and ``vmap``-friendly, so portfolio risk and key-rate durations come for
free via autodiff through the curve pytree.

Both pricers assume a **single-curve** setup where the discount curve is
also used to project forward rates — i.e. the reference rate on the FRN
or OIS float leg is the same index as the discount curve.  Basis between
forecasting and discounting curves is a multi-curve extension (see
``valax/curves/multi_curve.py``) not yet wired into these pricers.

References:
    Hull (2018), *Options, Futures, and Other Derivatives*, ch. 7-9.
    Brigo & Mercurio (2006), *Interest Rate Models*, ch. 1.
    Henrard (2014), *Interest Rate Modelling in the Multi-Curve Framework*.
"""

import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.bonds import FloatingRateBond
from valax.instruments.rates import OISSwap
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction

# Reuse the fixed-leg annuity helper from the swaptions module rather than
# duplicating it. It is a private helper but the intent is the same here.
from valax.pricing.analytic.swaptions import _annuity


# ── Floating rate note ────────────────────────────────────────────────

def floating_rate_bond_price(
    bond: FloatingRateBond,
    curve: DiscountCurve,
) -> Float[Array, ""]:
    """Price a floating-rate note from a discount curve.

    The coupon for period *i* is

    .. math::

        C_i = N \\cdot (F_i + s) \\cdot \\tau_i

    where :math:`F_i` is either

    - a **known past fixing** taken from ``bond.fixing_rates[i]`` when
      that entry is finite, or
    - the **projected** simply-compounded forward rate from the curve:

      .. math::

          F_i = \\frac{1}{\\tau_i}\\left(\\frac{DF(T_{i-1})}{DF(T_i)} - 1\\right)

    with :math:`T_{i-1}` = ``fixing_dates[i]`` and :math:`T_i` =
    ``payment_dates[i]``.

    The present value is then

    .. math::

        P = \\sum_{i : T_i > t_{\\text{settle}}} C_i \\cdot DF(T_i)
            \\;+\\; N \\cdot DF(T_n)

    The sum runs over **future** payments only; past coupons (including
    the historical portion of a seasoned bond whose fixing has already
    been paid) are masked out with ``payment_date > settlement_date``.

    **Par-at-reset invariant.** For a zero-spread FRN valued on the
    first reset date, the telescoping sum collapses:

    .. math::

        \\sum_i \\left(\\frac{DF(T_{i-1})}{DF(T_i)} - 1\\right) DF(T_i)
        = DF(T_0) - DF(T_n)

    so :math:`P = N\\,DF(T_0) + N\\,DF(T_n) - N\\,DF(T_n) = N\\,DF(T_0)`,
    which equals the face value on the reset date itself.  With a
    non-zero spread, :math:`P = N\\,DF(T_0) + N\\,s \\cdot A` where
    :math:`A` is the discounted day-count annuity.

    Args:
        bond: Floating rate note contract.
        curve: Discount / projection curve (single-curve assumption).

    Returns:
        Present value at ``curve.reference_date``.
    """
    tau = year_fraction(bond.fixing_dates, bond.payment_dates, bond.day_count)

    df_start = curve(bond.fixing_dates)
    df_end = curve(bond.payment_dates)
    projected_rate = (df_start / df_end - 1.0) / tau

    if bond.fixing_rates is not None:
        rate = jnp.where(
            jnp.isnan(bond.fixing_rates), projected_rate, bond.fixing_rates
        )
    else:
        rate = projected_rate

    coupon = bond.face_value * (rate + bond.spread) * tau
    future_mask = (bond.payment_dates > bond.settlement_date).astype(jnp.float64)

    coupon_pv = jnp.sum(coupon * df_end * future_mask)

    # Redemption of face value at maturity (last payment date).
    maturity = bond.payment_dates[-1]
    df_maturity = curve(maturity)

    return coupon_pv + bond.face_value * df_maturity


# ── OIS swap ──────────────────────────────────────────────────────────

def ois_swap_price(
    swap: OISSwap,
    curve: DiscountCurve,
) -> Float[Array, ""]:
    """NPV of an Overnight Index Swap (OIS).

    The floating leg pays the **compounded** overnight rate over each
    accrual period.  Under a single OIS curve, the value of the floating
    leg telescopes exactly as in the vanilla IRS case:

    .. math::

        \\text{PV}_{\\text{float}} = N \\cdot (DF(T_0) - DF(T_n))

    where :math:`T_0` = ``start_date`` and :math:`T_n` = ``float_dates[-1]``.
    This identity is exact for ``compounding="compounded"`` under
    log-linear discount-factor interpolation.  The ``"averaged"``
    convention used in a few markets differs only by a (small)
    convexity correction not modelled here.

    The fixed leg is a standard annuity:

    .. math::

        \\text{PV}_{\\text{fixed}} = N \\cdot K \\cdot A

    with :math:`A = \\sum_i \\tau_i\\, DF(T_i^{\\text{fixed}})`.

    A positive return means the payer side (``pay_fixed=True``) is
    in-the-money — i.e. the par OIS rate exceeds the contract rate.

    Args:
        swap: OIS swap contract.
        curve: OIS discount curve.

    Returns:
        Swap NPV.
    """
    ann = _annuity(swap.start_date, swap.fixed_dates, curve, swap.day_count)
    df_start = curve(swap.start_date)
    df_end = curve(swap.float_dates[-1])

    float_pv = swap.notional * (df_start - df_end)
    fixed_pv = swap.notional * swap.fixed_rate * ann

    payer_pv = float_pv - fixed_pv
    if swap.pay_fixed:
        return payer_pv
    return -payer_pv


def ois_swap_rate(
    swap: OISSwap,
    curve: DiscountCurve,
) -> Float[Array, ""]:
    """Par OIS rate: fixed rate :math:`K^*` such that the swap NPV is zero.

    .. math::

        K^* = \\frac{DF(T_0) - DF(T_n)}{A}

    with the same :math:`T_0`, :math:`T_n`, and annuity :math:`A` used
    in :func:`ois_swap_price`.  Structurally identical to
    :func:`valax.pricing.analytic.swaptions.swap_rate` but keyed off the
    :class:`OISSwap` pytree so that fixed and floating legs can have
    distinct schedules.

    Args:
        swap: OIS swap contract.
        curve: OIS discount curve.

    Returns:
        Par swap rate (annualized).
    """
    ann = _annuity(swap.start_date, swap.fixed_dates, curve, swap.day_count)
    df_start = curve(swap.start_date)
    df_end = curve(swap.float_dates[-1])
    return (df_start - df_end) / ann
