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


# ── Forward observables ───────────────────────────────────────────────

def caplet_forward(
    caplet: Caplet,
    curve: DiscountCurve,
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, ""]:
    r"""Simply-compounded forward rate of a caplet's accrual period.

    .. math::

        F = \frac{1}{\tau}
            \left(\frac{P^{\text{fwd}}(0, T_{s})}{P^{\text{fwd}}(0, T_{e})} - 1\right)

    Exposed on its own because a smile-aware caller has to read its vol *at*
    the forward before it can price, and must use the identical forward in the
    option formula. The pricers below accept it back via their ``forward``
    keyword so the two can never drift apart.

    Args:
        caplet: Caplet/floorlet contract.
        curve: Discount curve; also the projection curve when
            ``forward_curve`` is not given.
        forward_curve: Optional tenor/projection curve.

    Returns:
        The period's forward rate.
    """
    tau = year_fraction(caplet.start_date, caplet.end_date, caplet.day_count)
    fwd = curve if forward_curve is None else forward_curve
    return (fwd(caplet.start_date) / fwd(caplet.end_date) - 1.0) / tau


def cap_forwards(
    cap: Cap,
    curve: DiscountCurve,
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, " n"]:
    r"""Simply-compounded forward rate of every period in a cap strip.

    The vectorised counterpart of :func:`caplet_forward`, and the at-the-money
    anchor a per-expiry smile fit calibrates around (see
    :func:`valax.surfaces.caplet_stripping.optionlet_forwards`, which
    delegates here).

    Args:
        cap: Cap/floor whose accrual schedule defines the periods.
        curve: Discount curve; also the projection curve when
            ``forward_curve`` is not given.
        forward_curve: Optional tenor/projection curve.

    Returns:
        Forward rate of each accrual period.
    """
    tau = year_fraction(cap.start_dates, cap.end_dates, cap.day_count)
    fwd = curve if forward_curve is None else forward_curve
    return (fwd(cap.start_dates) / fwd(cap.end_dates) - 1.0) / tau


# ── Caplet / Floorlet ─────────────────────────────────────────────────

def caplet_price_black76(
    caplet: Caplet,
    curve: DiscountCurve,
    vol: Float[Array, ""],
    forward_curve: DiscountCurve | None = None,
    *,
    forward: Float[Array, ""] | None = None,
) -> Float[Array, ""]:
    """Black-76 price for a caplet or floorlet.

    Applies the Black-76 formula to the simply-compounded forward rate F
    over [start_date, end_date]:

        caplet = notional * tau * DF(end) * [F * N(d1) - K * N(d2)]
        d1 = [ln(F/K) + 0.5*vol^2*T] / (vol*sqrt(T)),  d2 = d1 - vol*sqrt(T)

    Args:
        caplet: Caplet/floorlet contract.
        curve: Discount curve (also projects the forward when
            ``forward_curve`` is not given).
        vol: Black (lognormal) implied volatility of the forward rate.
        forward_curve: Optional projection curve for the forward rate F.
            Defaults to ``curve`` (single-curve setup).
        forward: Optional precomputed forward rate (see
            :func:`caplet_forward`). Supply it when the volatility was read at
            a specific forward so vol and payoff cannot see different ones;
            omitted, it is projected off the curves.

    Returns:
        Caplet or floorlet price.
    """
    T = year_fraction(curve.reference_date, caplet.fixing_date, caplet.day_count)
    tau = year_fraction(caplet.start_date, caplet.end_date, caplet.day_count)

    if forward is None:
        F = caplet_forward(caplet, curve, forward_curve)
    else:
        F = forward
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
    forward_curve: DiscountCurve | None = None,
    *,
    forward: Float[Array, ""] | None = None,
) -> Float[Array, ""]:
    """Bachelier (normal model) price for a caplet or floorlet.

    Uses the normal dynamics dF = sigma * dW, giving a formula that is
    well-behaved at zero and negative rates:

        caplet = notional * tau * DF(end) * [(F-K)*N(d) + sigma*sqrt(T)*n(d)]
        d = (F - K) / (sigma * sqrt(T))

    Args:
        caplet: Caplet/floorlet contract.
        curve: Discount curve (also projects the forward when
            ``forward_curve`` is not given).
        vol: Normal (Bachelier) volatility of the forward rate.
        forward_curve: Optional projection curve for the forward rate F.
            Defaults to ``curve`` (single-curve setup).
        forward: Optional precomputed forward rate (see
            :func:`caplet_forward`); omitted, it is projected off the curves.

    Returns:
        Caplet or floorlet price.
    """
    T = year_fraction(curve.reference_date, caplet.fixing_date, caplet.day_count)
    tau = year_fraction(caplet.start_date, caplet.end_date, caplet.day_count)

    if forward is None:
        F = caplet_forward(caplet, curve, forward_curve)
    else:
        F = forward
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
    forward_curve: DiscountCurve | None = None,
    *,
    forwards: Float[Array, " n"] | None = None,
) -> Float[Array, ""]:
    """Black-76 price for a cap or floor (strip of caplets/floorlets).

    Prices each constituent caplet under a flat Black volatility and sums.
    For a term-structure of vols, pass vol as a 1-D array of shape (n,).

    Args:
        cap: Cap/floor contract.
        curve: Discount curve (also projects the forwards when
            ``forward_curve`` is not given).
        vol: Flat Black volatility (scalar) or per-caplet vols (shape n).
        forward_curve: Optional projection curve for the forward rates.
            Defaults to ``curve`` (single-curve setup).
        forwards: Optional precomputed per-period forward rates (see
            :func:`cap_forwards`). Supply them when the per-caplet vols were
            read at those forwards; omitted, they are projected off the curves.

    Returns:
        Cap or floor price.
    """
    T = year_fraction(curve.reference_date, cap.fixing_dates, cap.day_count)
    tau = year_fraction(cap.start_dates, cap.end_dates, cap.day_count)

    if forwards is None:
        F = cap_forwards(cap, curve, forward_curve)
    else:
        F = forwards
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
    forward_curve: DiscountCurve | None = None,
    *,
    forwards: Float[Array, " n"] | None = None,
) -> Float[Array, ""]:
    """Bachelier (normal model) price for a cap or floor.

    For a term-structure of vols, pass vol as a 1-D array of shape (n,).

    Args:
        cap: Cap/floor contract.
        curve: Discount curve (also projects forwards when
            ``forward_curve`` is not given).
        vol: Flat normal volatility (scalar) or per-caplet vols (shape n).
        forward_curve: Optional projection curve for the forward rates.
            Defaults to ``curve`` (single-curve setup).
        forwards: Optional precomputed per-period forward rates (see
            :func:`cap_forwards`); omitted, they are projected off the curves.

    Returns:
        Cap or floor price.
    """
    T = year_fraction(curve.reference_date, cap.fixing_dates, cap.day_count)
    tau = year_fraction(cap.start_dates, cap.end_dates, cap.day_count)

    if forwards is None:
        F = cap_forwards(cap, curve, forward_curve)
    else:
        F = forwards
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
