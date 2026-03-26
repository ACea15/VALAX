"""Black-76 and Bachelier pricing for caplets, floorlets, caps, and floors.

Caplet/floorlet: single-period options on a simply-compounded forward rate.
Cap/floor: strips of caplets/floorlets priced under a flat volatility.

Reference: Black (1976), "The pricing of commodity contracts".
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.rates import Caplet, Cap
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction


# ── Caplet / Floorlet ─────────────────────────────────────────────────

def caplet_price_black76(
    caplet: Caplet,
    curve: DiscountCurve,
    vol: Float[Array, ""],
) -> Float[Array, ""]:
    """Black-76 price for a caplet or floorlet.

    Applies the Black-76 formula to the simply-compounded forward rate F
    over [start_date, end_date]:

        caplet = notional * tau * DF(end) * [F * N(d1) - K * N(d2)]
        d1 = [ln(F/K) + 0.5*vol^2*T] / (vol*sqrt(T)),  d2 = d1 - vol*sqrt(T)

    Args:
        caplet: Caplet/floorlet contract.
        curve: Discount curve for forward rates and discounting.
        vol: Black (lognormal) implied volatility of the forward rate.

    Returns:
        Caplet or floorlet price.
    """
    T = year_fraction(curve.reference_date, caplet.fixing_date, caplet.day_count)
    tau = year_fraction(caplet.start_date, caplet.end_date, caplet.day_count)

    df_start = curve(caplet.start_date)
    df_end = curve(caplet.end_date)
    F = (df_start / df_end - 1.0) / tau
    P = curve(caplet.end_date)

    K = caplet.strike
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(F / K) + 0.5 * vol**2 * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T

    Phi = jax.scipy.stats.norm.cdf
    cap_pv = caplet.notional * tau * P * (F * Phi(d1) - K * Phi(d2))

    if caplet.is_cap:
        return cap_pv
    # Floorlet via put-call parity: floorlet = caplet - notional*tau*P*(F - K)
    return cap_pv - caplet.notional * tau * P * (F - K)


def caplet_price_bachelier(
    caplet: Caplet,
    curve: DiscountCurve,
    vol: Float[Array, ""],
) -> Float[Array, ""]:
    """Bachelier (normal model) price for a caplet or floorlet.

    Uses the normal dynamics dF = sigma * dW, giving a formula that is
    well-behaved at zero and negative rates:

        caplet = notional * tau * DF(end) * [(F-K)*N(d) + sigma*sqrt(T)*n(d)]
        d = (F - K) / (sigma * sqrt(T))

    Args:
        caplet: Caplet/floorlet contract.
        curve: Discount curve.
        vol: Normal (Bachelier) volatility of the forward rate.

    Returns:
        Caplet or floorlet price.
    """
    T = year_fraction(curve.reference_date, caplet.fixing_date, caplet.day_count)
    tau = year_fraction(caplet.start_date, caplet.end_date, caplet.day_count)

    df_start = curve(caplet.start_date)
    df_end = curve(caplet.end_date)
    F = (df_start / df_end - 1.0) / tau
    P = curve(caplet.end_date)

    K = caplet.strike
    sigma_T = vol * jnp.sqrt(T)
    d = (F - K) / sigma_T

    Phi = jax.scipy.stats.norm.cdf
    phi = jax.scipy.stats.norm.pdf

    intrinsic = F - K
    cap_pv = caplet.notional * tau * P * (intrinsic * Phi(d) + sigma_T * phi(d))

    if caplet.is_cap:
        return cap_pv
    # Floorlet via parity
    return cap_pv - caplet.notional * tau * P * intrinsic


# ── Cap / Floor ───────────────────────────────────────────────────────

def cap_price_black76(
    cap: Cap,
    curve: DiscountCurve,
    vol: Float[Array, ""],
) -> Float[Array, ""]:
    """Black-76 price for a cap or floor (strip of caplets/floorlets).

    Prices each constituent caplet under a flat Black volatility and sums.
    For a term-structure of vols, pass vol as a 1-D array of shape (n,).

    Args:
        cap: Cap/floor contract.
        curve: Discount curve.
        vol: Flat Black volatility (scalar) or per-caplet vols (shape n).

    Returns:
        Cap or floor price.
    """
    T = year_fraction(curve.reference_date, cap.fixing_dates, cap.day_count)
    tau = year_fraction(cap.start_dates, cap.end_dates, cap.day_count)

    df_start = curve(cap.start_dates)
    df_end = curve(cap.end_dates)
    F = (df_start / df_end - 1.0) / tau
    P = curve(cap.end_dates)

    K = cap.strike
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(F / K) + 0.5 * vol**2 * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T

    Phi = jax.scipy.stats.norm.cdf
    caplet_pvs = cap.notional * tau * P * (F * Phi(d1) - K * Phi(d2))

    if cap.is_cap:
        return jnp.sum(caplet_pvs)
    floorlet_pvs = caplet_pvs - cap.notional * tau * P * (F - K)
    return jnp.sum(floorlet_pvs)


def cap_price_bachelier(
    cap: Cap,
    curve: DiscountCurve,
    vol: Float[Array, ""],
) -> Float[Array, ""]:
    """Bachelier (normal model) price for a cap or floor.

    For a term-structure of vols, pass vol as a 1-D array of shape (n,).

    Args:
        cap: Cap/floor contract.
        curve: Discount curve.
        vol: Flat normal volatility (scalar) or per-caplet vols (shape n).

    Returns:
        Cap or floor price.
    """
    T = year_fraction(curve.reference_date, cap.fixing_dates, cap.day_count)
    tau = year_fraction(cap.start_dates, cap.end_dates, cap.day_count)

    df_start = curve(cap.start_dates)
    df_end = curve(cap.end_dates)
    F = (df_start / df_end - 1.0) / tau
    P = curve(cap.end_dates)

    K = cap.strike
    sigma_T = vol * jnp.sqrt(T)
    d = (F - K) / sigma_T

    Phi = jax.scipy.stats.norm.cdf
    phi = jax.scipy.stats.norm.pdf

    intrinsic = F - K
    caplet_pvs = cap.notional * tau * P * (intrinsic * Phi(d) + sigma_T * phi(d))

    if cap.is_cap:
        return jnp.sum(caplet_pvs)
    floorlet_pvs = caplet_pvs - cap.notional * tau * P * intrinsic
    return jnp.sum(floorlet_pvs)
