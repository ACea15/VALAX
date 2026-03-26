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


# ── Internal helpers ──────────────────────────────────────────────────

def _annuity(
    start_date: Int[Array, ""],
    fixed_dates: Int[Array, " n"],
    curve: DiscountCurve,
    day_count: str,
) -> Float[Array, ""]:
    """Fixed-leg annuity (PV01): PV of a unit stream of fixed payments.

        A = sum_i  tau_i * DF(T_i)

    where tau_i is the year fraction from the previous date (or start_date)
    to T_i, and DF(T_i) is the discount factor from the reference date.
    """
    starts = jnp.concatenate([start_date[None], fixed_dates[:-1]])
    tau = year_fraction(starts, fixed_dates, day_count)
    return jnp.sum(tau * curve(fixed_dates))


# ── Swap utilities ────────────────────────────────────────────────────

def swap_rate(
    swap: InterestRateSwap,
    curve: DiscountCurve,
) -> Float[Array, ""]:
    """Par swap rate: fixed rate K* such that the swap NPV is zero.

        S = (DF(start) - DF(maturity)) / A

    where A is the fixed-leg annuity.

    Args:
        swap: Swap contract.
        curve: Discount curve.

    Returns:
        Par swap rate (annualized).
    """
    ann = _annuity(swap.start_date, swap.fixed_dates, curve, swap.day_count)
    df_start = curve(swap.start_date)
    df_end = curve(swap.fixed_dates[-1])
    return (df_start - df_end) / ann


def swap_price(
    swap: InterestRateSwap,
    curve: DiscountCurve,
) -> Float[Array, ""]:
    """NPV of a vanilla fixed-for-float interest rate swap.

    Uses the replication identity for the floating leg:
        PV(float) = notional * (DF(start) - DF(maturity))

    and the discounted fixed cash flows for the fixed leg:
        PV(fixed) = notional * fixed_rate * A

    A positive result means the payer perspective is in-the-money
    when pay_fixed=True (i.e., par rate > fixed_rate).

    Args:
        swap: Swap contract (pay_fixed field determines sign convention).
        curve: Discount curve.

    Returns:
        Swap NPV.
    """
    ann = _annuity(swap.start_date, swap.fixed_dates, curve, swap.day_count)
    df_start = curve(swap.start_date)
    df_end = curve(swap.fixed_dates[-1])

    float_pv = swap.notional * (df_start - df_end)
    fixed_pv = swap.notional * swap.fixed_rate * ann

    payer_pv = float_pv - fixed_pv
    if swap.pay_fixed:
        return payer_pv
    return -payer_pv


# ── Swaption pricing ──────────────────────────────────────────────────

def swaption_price_black76(
    swaption: Swaption,
    curve: DiscountCurve,
    vol: Float[Array, ""],
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

    Returns:
        Payer or receiver swaption price.
    """
    T = year_fraction(curve.reference_date, swaption.expiry_date, swaption.day_count)
    ann = _annuity(swaption.expiry_date, swaption.fixed_dates, curve, swaption.day_count)

    df_start = curve(swaption.expiry_date)
    df_end = curve(swaption.fixed_dates[-1])
    S = (df_start - df_end) / ann  # forward par swap rate

    K = swaption.strike
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(S / K) + 0.5 * vol**2 * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T

    Phi = jax.scipy.stats.norm.cdf
    payer = swaption.notional * ann * (S * Phi(d1) - K * Phi(d2))

    if swaption.is_payer:
        return payer
    # Receiver via payer-receiver parity: receiver = payer - notional*A*(S - K)
    return payer - swaption.notional * ann * (S - K)


def swaption_price_bachelier(
    swaption: Swaption,
    curve: DiscountCurve,
    vol: Float[Array, ""],
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

    Returns:
        Payer or receiver swaption price.
    """
    T = year_fraction(curve.reference_date, swaption.expiry_date, swaption.day_count)
    ann = _annuity(swaption.expiry_date, swaption.fixed_dates, curve, swaption.day_count)

    df_start = curve(swaption.expiry_date)
    df_end = curve(swaption.fixed_dates[-1])
    S = (df_start - df_end) / ann

    K = swaption.strike
    sigma_T = vol * jnp.sqrt(T)
    d = (S - K) / sigma_T

    Phi = jax.scipy.stats.norm.cdf
    phi = jax.scipy.stats.norm.pdf

    intrinsic = S - K
    payer = swaption.notional * ann * (intrinsic * Phi(d) + sigma_T * phi(d))

    if swaption.is_payer:
        return payer
    # Receiver via parity
    return payer - swaption.notional * ann * intrinsic
