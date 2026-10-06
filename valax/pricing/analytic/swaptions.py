"""Black-76 and Bachelier pricing for European swaptions, plus swap utilities.

Swaption: European option on a vanilla fixed-for-float interest rate swap.
Payer swaption: right to pay fixed / receive float.
Receiver swaption: right to receive fixed / pay float.

The pricing model (Black-76 or Bachelier) is applied to the forward par
swap rate S, discounted by the physical-measure annuity A:

    payer = notional * A * Black76(S, K, vol, T)
    receiver = payer - notional * A * (S - K)    [put-call parity]

References:
    Black (1976), "The pricing of commodity contracts".
    Brigo & Mercurio (2006), "Interest Rate Models", ch. 6-7.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.instruments.rates import InterestRateSwap, Swaption
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction


# ── Forward / annuity observables ─────────────────────────────────────

def annuity(
    start_date: Int[Array, ""],
    fixed_dates: Int[Array, " n"],
    curve: DiscountCurve,
    day_count: str,
) -> Float[Array, ""]:
    """Fixed-leg annuity (PV01): PV of a unit stream of fixed payments.

    .. math::

        A_d = \\sum_i \\tau_i\\, P_d(0, T_i)

    where :math:`\\tau_i` is the year fraction from the previous date (or
    ``start_date``) to :math:`T_i`, and :math:`P_d` is the discount factor from
    the curve's reference date. This is the physical-measure numeraire the
    swaption formulas multiply through, and the quantity a premium-quoted vol
    residual has to divide by, so it is public rather than internal.

    Args:
        start_date: Accrual start of the first fixed period (day ordinal).
        fixed_dates: Fixed-leg payment dates (day ordinals), ascending.
        curve: Discount curve.
        day_count: Day-count convention for the accrual fractions.

    Returns:
        The annuity (PV01) per unit notional.
    """
    starts = jnp.concatenate([start_date[None], fixed_dates[:-1]])
    tau = year_fraction(starts, fixed_dates, day_count)
    return jnp.sum(tau * curve(fixed_dates))


#: Backwards-compatible private alias retained for existing call sites.
_annuity = annuity


def swaption_forward_and_annuity(
    swaption: Swaption,
    curve: DiscountCurve,
    forward_curve: DiscountCurve | None = None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Forward par swap rate and discount annuity of a swaption's underlying.

    The two curve-derived observables every swaption model consumes, computed
    together so a caller can read a smile *at* the forward and then price with
    the same forward and annuity — rather than recomputing them in the pricer
    and risking a vol/price mismatch.

    Single-curve (``forward_curve=None``):

    .. math::

        S_x = \\frac{P_d(0, T_x) - P_d(0, T_n)}{A_d},

    and with a distinct projection curve the floating leg no longer telescopes:

    .. math::

        S_x = \\frac{1}{A_d} \\sum_i
              \\left(\\frac{P_f(0, T_{i-1})}{P_f(0, T_i)} - 1\\right) P_d(0, T_i).

    Args:
        swaption: Swaption contract (its fixed schedule is the projection grid).
        curve: Discount curve.
        forward_curve: Optional projection curve for the underlying swap's
            floating leg. Defaults to ``curve`` (single-curve setup).

    Returns:
        Tuple ``(forward, annuity)`` — the forward par swap rate and the
        fixed-leg annuity per unit notional.
    """
    ann = annuity(
        swaption.expiry_date, swaption.fixed_dates, curve, swaption.day_count
    )
    if forward_curve is None:
        df_start = curve(swaption.expiry_date)
        df_end = curve(swaption.fixed_dates[-1])
        fwd = (df_start - df_end) / ann
    else:
        fwd = _dual_curve_float_pv(
            swaption.expiry_date, swaption.fixed_dates, curve, forward_curve
        ) / ann
    return fwd, ann


# ── Internal helpers ──────────────────────────────────────────────────


def _dual_curve_float_pv(
    start_date: Int[Array, ""],
    payment_dates: Int[Array, " n"],
    discount_curve: DiscountCurve,
    forward_curve: DiscountCurve,
) -> Float[Array, ""]:
    """Unit-notional floating-leg PV with distinct projection curve.

    Projects each period's simply-compounded forward off
    ``forward_curve`` and discounts on ``discount_curve``:

        PV = sum_i (DF_f(T_{i-1}) / DF_f(T_i) - 1) * DF_d(T_i)

    The accrual fraction cancels between projection and payoff, so no
    day count is needed.  When both curves coincide the sum telescopes
    to DF(start) - DF(maturity), recovering the single-curve identity.
    """
    starts = jnp.concatenate([start_date[None], payment_dates[:-1]])
    df_f_start = forward_curve(starts)
    df_f_end = forward_curve(payment_dates)
    df_d_end = discount_curve(payment_dates)
    return jnp.sum((df_f_start / df_f_end - 1.0) * df_d_end)


# ── Swap utilities ────────────────────────────────────────────────────

def swap_rate(
    swap: InterestRateSwap,
    curve: DiscountCurve,
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, ""]:
    """Par swap rate: fixed rate K* such that the swap NPV is zero.

    Single-curve (``forward_curve=None``):

        S = (DF(start) - DF(maturity)) / A

    where A is the fixed-leg annuity.  With a distinct projection
    curve the floating leg no longer telescopes and

        S = sum_i (DF_f(T_{i-1})/DF_f(T_i) - 1) * DF_d(T_i) / A

    using the fixed schedule as the projection grid (the
    :class:`InterestRateSwap` pytree carries no separate float
    schedule).

    Args:
        swap: Swap contract.
        curve: Discount curve.
        forward_curve: Optional projection curve for the floating leg.
            Defaults to ``curve`` (single-curve setup).

    Returns:
        Par swap rate (annualized).
    """
    ann = annuity(swap.start_date, swap.fixed_dates, curve, swap.day_count)
    if forward_curve is None:
        df_start = curve(swap.start_date)
        df_end = curve(swap.fixed_dates[-1])
        return (df_start - df_end) / ann
    float_pv = _dual_curve_float_pv(
        swap.start_date, swap.fixed_dates, curve, forward_curve
    )
    return float_pv / ann


def swap_price(
    swap: InterestRateSwap,
    curve: DiscountCurve,
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, ""]:
    """NPV of a vanilla fixed-for-float interest rate swap.

    In the single-curve setup (``forward_curve=None``) the floating leg
    uses the replication identity:
        PV(float) = notional * (DF(start) - DF(maturity))

    With a distinct projection curve the floating leg is projected off
    ``forward_curve`` and discounted on ``curve`` (fixed schedule used
    as the projection grid).  The fixed leg is always the discounted
    fixed cash flows:
        PV(fixed) = notional * fixed_rate * A

    A positive result means the payer perspective is in-the-money
    when pay_fixed=True (i.e., par rate > fixed_rate).

    Args:
        swap: Swap contract (pay_fixed field determines sign convention).
        curve: Discount curve.
        forward_curve: Optional projection curve for the floating leg.
            Defaults to ``curve`` (single-curve setup).

    Returns:
        Swap NPV.
    """
    ann = annuity(swap.start_date, swap.fixed_dates, curve, swap.day_count)
    if forward_curve is None:
        df_start = curve(swap.start_date)
        df_end = curve(swap.fixed_dates[-1])
        float_pv = swap.notional * (df_start - df_end)
    else:
        float_pv = swap.notional * _dual_curve_float_pv(
            swap.start_date, swap.fixed_dates, curve, forward_curve
        )
    fixed_pv = swap.notional * swap.fixed_rate * ann

    payer_pv = float_pv - fixed_pv
    if swap.pay_fixed:
        return payer_pv
    return -payer_pv


# ── Swaption pricing cores (forward and annuity supplied) ─────────────

def swaption_black76_from_forward(
    swaption: Swaption,
    forward: Float[Array, ""],
    ann: Float[Array, ""],
    vol: Float[Array, ""],
    expiry: Float[Array, ""],
) -> Float[Array, ""]:
    """Black-76 swaption value from a precomputed forward and annuity.

    The payoff core of :func:`swaption_price_black76` with the curve work
    hoisted out, so a smile-aware caller can read its vol *at* ``forward`` and
    then price with that very same ``forward`` and ``ann``.

    Args:
        swaption: Swaption contract (supplies strike, notional, direction).
        forward: Forward par swap rate :math:`S_x`.
        ann: Fixed-leg annuity per unit notional.
        vol: Black (lognormal) implied volatility.
        expiry: Time to expiry (year fraction).

    Returns:
        Payer or receiver swaption price.
    """
    K = swaption.strike
    sqrt_T = jnp.sqrt(expiry)
    d1 = (jnp.log(forward / K) + 0.5 * vol**2 * expiry) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T

    Phi = jax.scipy.stats.norm.cdf
    payer = swaption.notional * ann * (forward * Phi(d1) - K * Phi(d2))

    if swaption.is_payer:
        return payer
    # Receiver via payer-receiver parity: receiver = payer - notional*A*(S - K)
    return payer - swaption.notional * ann * (forward - K)


def swaption_bachelier_from_forward(
    swaption: Swaption,
    forward: Float[Array, ""],
    ann: Float[Array, ""],
    vol: Float[Array, ""],
    expiry: Float[Array, ""],
) -> Float[Array, ""]:
    """Bachelier swaption value from a precomputed forward and annuity.

    The normal-model counterpart of :func:`swaption_black76_from_forward`.

    Args:
        swaption: Swaption contract (supplies strike, notional, direction).
        forward: Forward par swap rate :math:`S_x`.
        ann: Fixed-leg annuity per unit notional.
        vol: Normal (Bachelier) volatility.
        expiry: Time to expiry (year fraction).

    Returns:
        Payer or receiver swaption price.
    """
    K = swaption.strike
    sigma_T = vol * jnp.sqrt(expiry)
    d = (forward - K) / sigma_T

    Phi = jax.scipy.stats.norm.cdf
    phi = jax.scipy.stats.norm.pdf

    intrinsic = forward - K
    payer = swaption.notional * ann * (intrinsic * Phi(d) + sigma_T * phi(d))

    if swaption.is_payer:
        return payer
    # Receiver via parity
    return payer - swaption.notional * ann * intrinsic


# ── Swaption pricing ──────────────────────────────────────────────────

def swaption_price_black76(
    swaption: Swaption,
    curve: DiscountCurve,
    vol: Float[Array, ""],
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, ""]:
    """Black-76 price for a European payer or receiver swaption.

    Applies the Black-76 model to the forward par swap rate S:

        payer  = notional * A * [S * N(d1) - K * N(d2)]
        d1 = [ln(S/K) + 0.5*vol^2*T] / (vol*sqrt(T)),  d2 = d1 - vol*sqrt(T)

    Requires S, K > 0 (lognormal model). Use Bachelier for near-zero rates.

    Args:
        swaption: Swaption contract.
        curve: Discount curve (used to compute S, annuity, and discounting).
        vol: Black (lognormal) swaption implied volatility.
        forward_curve: Optional projection curve for the underlying
            swap's floating leg. Defaults to ``curve`` (single-curve).

    Returns:
        Payer or receiver swaption price.
    """
    T = year_fraction(curve.reference_date, swaption.expiry_date, swaption.day_count)
    S, ann = swaption_forward_and_annuity(swaption, curve, forward_curve)
    return swaption_black76_from_forward(swaption, S, ann, vol, T)


def swaption_price_bachelier(
    swaption: Swaption,
    curve: DiscountCurve,
    vol: Float[Array, ""],
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, ""]:
    """Bachelier (normal model) price for a European payer or receiver swaption.

    Uses the normal dynamics dS = sigma * dW, suitable for near-zero
    or negative rates (SOFR/ESTR market):

        payer  = notional * A * [(S-K)*N(d) + sigma*sqrt(T)*n(d)]
        d = (S - K) / (sigma * sqrt(T))

    Args:
        swaption: Swaption contract.
        curve: Discount curve.
        vol: Normal (Bachelier) swaption volatility.
        forward_curve: Optional projection curve for the underlying
            swap's floating leg. Defaults to ``curve`` (single-curve).

    Returns:
        Payer or receiver swaption price.
    """
    T = year_fraction(curve.reference_date, swaption.expiry_date, swaption.day_count)
    S, ann = swaption_forward_and_annuity(swaption, curve, forward_curve)
    return swaption_bachelier_from_forward(swaption, S, ann, vol, T)
