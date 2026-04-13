"""Analytical pricers for inflation derivatives: ZCIS, YYIS, inflation cap/floor.

All pricers take an :class:`InflationCurve` (for forward CPI projection)
and a :class:`DiscountCurve` (for nominal discounting) plus the
instrument pytree.  They are pure functions — ``jax.jit``, ``jax.grad``,
and ``jax.vmap`` work out of the box.

References:
    Kerkhof (2005), "Inflation Derivatives Explained".
    Brigo & Mercurio (2006), *Interest Rate Models*, ch. 15.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.inflation import (
    ZeroCouponInflationSwap,
    YearOnYearInflationSwap,
    InflationCapFloor,
)
from valax.curves.discount import DiscountCurve
from valax.curves.inflation import InflationCurve, forward_cpi
from valax.dates.daycounts import year_fraction


# ─────────────────────────────────────────────────────────────────────
# Zero-coupon inflation swap
# ─────────────────────────────────────────────────────────────────────

def zcis_price(
    swap: ZeroCouponInflationSwap,
    inflation_curve: InflationCurve,
    discount_curve: DiscountCurve,
) -> Float[Array, ""]:
    """NPV of a zero-coupon inflation swap (ZCIS).

    At maturity the inflation receiver gets

    .. math::

        N \\cdot \\left(\\frac{\\text{CPI}(T)}{\\text{CPI}(0)} - 1\\right)

    and pays

    .. math::

        N \\cdot \\left((1 + K)^T - 1\\right)

    both settled at maturity and discounted at :math:`DF(T)`.

    Args:
        swap: Zero-coupon inflation swap instrument.
        inflation_curve: Forward CPI curve.
        discount_curve: Nominal discount curve.

    Returns:
        Swap NPV.  Sign follows ``is_inflation_receiver``.
    """
    T = year_fraction(
        swap.effective_date, swap.maturity_date, swap.day_count
    )
    cpi_ratio = forward_cpi(inflation_curve, swap.maturity_date) / swap.base_cpi
    df = discount_curve(swap.maturity_date)

    infl_leg = swap.notional * (cpi_ratio - 1.0) * df
    fixed_leg = swap.notional * ((1.0 + swap.fixed_rate) ** T - 1.0) * df

    receiver_pv = infl_leg - fixed_leg
    if swap.is_inflation_receiver:
        return receiver_pv
    return -receiver_pv


def zcis_breakeven_rate(
    swap: ZeroCouponInflationSwap,
    inflation_curve: InflationCurve,
) -> Float[Array, ""]:
    """Breakeven (par) rate :math:`K^*` of a ZCIS.

    .. math::

        K^* = \\left(\\frac{\\text{CPI}(T)}{\\text{CPI}(0)}\\right)^{1/T} - 1

    This is the fixed rate that makes the ZCIS NPV exactly zero
    (independent of the discount curve, since both legs settle at
    the same date).
    """
    T = year_fraction(
        swap.effective_date, swap.maturity_date, swap.day_count
    )
    cpi_ratio = forward_cpi(inflation_curve, swap.maturity_date) / swap.base_cpi
    return cpi_ratio ** (1.0 / T) - 1.0


# ─────────────────────────────────────────────────────────────────────
# Year-on-year inflation swap
# ─────────────────────────────────────────────────────────────────────

def yyis_price(
    swap: YearOnYearInflationSwap,
    inflation_curve: InflationCurve,
    discount_curve: DiscountCurve,
) -> Float[Array, ""]:
    """NPV of a year-on-year inflation swap (YYIS).

    At each payment date :math:`t_i` the inflation leg pays

    .. math::

        N \\cdot \\left(\\frac{\\text{CPI}(t_i)}{\\text{CPI}(t_{i-1})} - 1\\right)

    and the fixed leg pays :math:`N \\cdot K \\cdot \\tau_i`.

    The year-on-year forward inflation rate is taken directly from the
    ratio of forward CPIs.

    .. warning::

       **No convexity adjustment.**  The true expected YoY CPI ratio
       under the payment measure differs from the ratio of forward
       CPIs by a convexity correction that depends on the inflation
       volatility and the nominal-real rate correlation.  This pricer
       uses the forward ratio directly, which is a standard baseline.

    Args:
        swap: Year-on-year inflation swap.
        inflation_curve: Forward CPI curve.
        discount_curve: Nominal discount curve.

    Returns:
        Swap NPV.  Sign follows ``is_inflation_receiver``.
    """
    n = swap.payment_dates.shape[0]
    starts = jnp.concatenate([swap.effective_date[None], swap.payment_dates[:-1]])
    ends = swap.payment_dates

    yoy_fwd = (
        forward_cpi(inflation_curve, ends)
        / forward_cpi(inflation_curve, starts)
        - 1.0
    )
    tau = year_fraction(starts, ends, swap.day_count)
    dfs = discount_curve(ends)

    infl_leg = swap.notional * jnp.sum(yoy_fwd * dfs)
    fixed_leg = swap.notional * swap.fixed_rate * jnp.sum(tau * dfs)

    receiver_pv = infl_leg - fixed_leg
    if swap.is_inflation_receiver:
        return receiver_pv
    return -receiver_pv


# ─────────────────────────────────────────────────────────────────────
# Inflation cap / floor
# ─────────────────────────────────────────────────────────────────────

def inflation_cap_floor_price_black76(
    cap: InflationCapFloor,
    inflation_curve: InflationCurve,
    discount_curve: DiscountCurve,
    vol: Float[Array, ""],
) -> Float[Array, ""]:
    """Black-76 price of an inflation cap or floor on YoY CPI returns.

    Each caplet pays

    .. math::

        N \\cdot \\max\\!\\left(\\frac{\\text{CPI}(t_i)}{\\text{CPI}(t_{i-1})}
        - 1 - K,\\; 0\\right)

    priced via Black-76 on the forward YoY rate :math:`F_i`:

    .. math::

        \\text{caplet}_i = N \\cdot DF(t_i)
        \\left[F_i\\,\\Phi(d_1) - K\\,\\Phi(d_2)\\right]

    ``vol`` can be a scalar (flat vol) or per-period array.  Floor via
    put-call parity.

    Args:
        cap: Inflation cap or floor.
        inflation_curve: Forward CPI curve.
        discount_curve: Nominal discount curve.
        vol: Black-76 volatility on the YoY inflation rate.

    Returns:
        Cap or floor NPV.
    """
    starts = jnp.concatenate([cap.effective_date[None], cap.payment_dates[:-1]])
    ends = cap.payment_dates

    F = (
        forward_cpi(inflation_curve, ends)
        / forward_cpi(inflation_curve, starts)
        - 1.0
    )
    dfs = discount_curve(ends)

    # Time to fixing (start of each period).
    T = year_fraction(discount_curve.reference_date, starts, cap.day_count)
    T_safe = jnp.maximum(T, 1e-10)
    sqrt_T = jnp.sqrt(T_safe)

    K = cap.strike
    d1 = (jnp.log(F / K) + 0.5 * vol**2 * T_safe) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T

    Phi = jax.scipy.stats.norm.cdf
    caplet_pvs = cap.notional * dfs * (F * Phi(d1) - K * Phi(d2))

    if cap.is_cap:
        return jnp.sum(caplet_pvs)
    # Floor via put-call parity.
    floorlet_pvs = caplet_pvs - cap.notional * dfs * (F - K)
    return jnp.sum(floorlet_pvs)
