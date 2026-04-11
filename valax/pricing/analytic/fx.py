"""Garman-Kohlhagen pricing for FX options, FX forward valuation,
and FX-specific delta conventions.

The Garman-Kohlhagen (1983) model is Black-Scholes with the foreign
risk-free rate playing the role of the continuous dividend yield:

.. math::

    C = S e^{-r_f T} \\Phi(d_1) - K e^{-r_d T} \\Phi(d_2)

    d_1 = \\frac{\\ln(S/K) + (r_d - r_f + \\sigma^2/2)T}{\\sigma\\sqrt{T}}

    d_2 = d_1 - \\sigma\\sqrt{T}

FX markets quote options in **delta space**, not strike space.  The
standard quoting points are 10Δ put, 25Δ put, ATM (DNS), 25Δ call,
10Δ call.  This module provides the delta ↔ strike conversion
utilities required for FX vol surface construction.

Delta conventions
-----------------
FX options use three delta conventions that differ by whether the
premium is included and whether the delta is expressed in spot or
forward terms:

- **Spot delta** (``"spot"``): ``Δ = ±e^{-r_f T} Φ(±d_1)``
- **Forward delta** (``"forward"``): ``Δ = ±Φ(±d_1)``
- **Premium-adjusted delta** (``"premium_adjusted"``):
  ``Δ = ±e^{-r_f T} [Φ(±d_1) - (V / (S · N))]``
  where ``V`` is the option premium in domestic terms.  Used when
  the premium is paid in foreign currency (the norm for most EM pairs).

References
----------
- Garman, M. and Kohlhagen, S. (1983). "Foreign currency option values."
  *Journal of International Money and Finance*.
- Clark, I. (2011). *Foreign Exchange Option Pricing: A Practitioner's
  Guide*. Wiley.
- Reiswich, D. and Wystup, U. (2010). "FX Volatility Smile Construction."
  *Wilmott*.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.fx import FXForward, FXVanillaOption


# ── FX Forward ───────────────────────────────────────────────────────


def fx_forward_rate(
    spot: Float[Array, ""],
    r_domestic: Float[Array, ""],
    r_foreign: Float[Array, ""],
    expiry: Float[Array, ""],
) -> Float[Array, ""]:
    """Fair FX forward rate via covered interest rate parity.

    .. math::

        F = S \\cdot e^{(r_d - r_f) T}

    Args:
        spot: Current spot FX rate (domestic per foreign).
        r_domestic: Domestic continuously-compounded risk-free rate.
        r_foreign: Foreign continuously-compounded risk-free rate.
        expiry: Time to maturity in year fractions.

    Returns:
        Fair forward FX rate.
    """
    return spot * jnp.exp((r_domestic - r_foreign) * expiry)


def fx_forward_price(
    fwd: FXForward,
    spot: Float[Array, ""],
    r_domestic: Float[Array, ""],
    r_foreign: Float[Array, ""],
) -> Float[Array, ""]:
    """NPV of an FX forward contract in domestic currency.

    .. math::

        NPV = N_f \\cdot e^{-r_d T} \\cdot (F - K) \\quad (\\text{if buying foreign})

    where :math:`F = S e^{(r_d - r_f)T}` is the fair forward rate.

    At inception, ``strike = F`` and ``NPV = 0``.

    Args:
        fwd: FX forward contract.
        spot: Current spot rate.
        r_domestic: Domestic risk-free rate.
        r_foreign: Foreign risk-free rate.

    Returns:
        NPV in domestic currency (positive = in-the-money for buyer).
    """
    F = fx_forward_rate(spot, r_domestic, r_foreign, fwd.expiry)
    df_dom = jnp.exp(-r_domestic * fwd.expiry)
    pv = fwd.notional_foreign * df_dom * (F - fwd.strike)
    if fwd.is_buy:
        return pv
    return -pv


# ── Garman-Kohlhagen ─────────────────────────────────────────────────


def _gk_d1d2(
    spot: Float[Array, ""],
    strike: Float[Array, ""],
    expiry: Float[Array, ""],
    vol: Float[Array, ""],
    r_domestic: Float[Array, ""],
    r_foreign: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute d₁ and d₂ for the Garman-Kohlhagen formula."""
    sqrt_T = jnp.sqrt(expiry)
    d1 = (
        jnp.log(spot / strike) + (r_domestic - r_foreign + 0.5 * vol**2) * expiry
    ) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T
    return d1, d2


def garman_kohlhagen_price(
    option: FXVanillaOption,
    spot: Float[Array, ""],
    vol: Float[Array, ""],
    r_domestic: Float[Array, ""],
    r_foreign: Float[Array, ""],
) -> Float[Array, ""]:
    """Garman-Kohlhagen price for an FX vanilla option.

    .. math::

        C = N \\cdot [S e^{-r_f T} \\Phi(d_1) - K e^{-r_d T} \\Phi(d_2)]

        P = N \\cdot [K e^{-r_d T} \\Phi(-d_2) - S e^{-r_f T} \\Phi(-d_1)]

    The price is in **domestic** currency units regardless of the
    ``premium_currency`` field.  To convert to foreign-currency premium,
    divide by ``spot``.

    Args:
        option: FX vanilla option contract.
        spot: Spot FX rate (domestic per foreign).
        vol: Black-Scholes implied volatility.
        r_domestic: Domestic risk-free rate.
        r_foreign: Foreign risk-free rate.

    Returns:
        Option price in domestic currency.
    """
    d1, d2 = _gk_d1d2(spot, option.strike, option.expiry, vol, r_domestic, r_foreign)

    df_dom = jnp.exp(-r_domestic * option.expiry)
    df_for = jnp.exp(-r_foreign * option.expiry)

    Phi = jax.scipy.stats.norm.cdf

    call = option.notional_foreign * (
        spot * df_for * Phi(d1) - option.strike * df_dom * Phi(d2)
    )

    if option.is_call:
        return call
    # Put via put-call parity:  P = C - N * (S * df_for - K * df_dom)
    return call - option.notional_foreign * (spot * df_for - option.strike * df_dom)


# ── Implied volatility ───────────────────────────────────────────────


def fx_implied_vol(
    option: FXVanillaOption,
    spot: Float[Array, ""],
    r_domestic: Float[Array, ""],
    r_foreign: Float[Array, ""],
    market_price: Float[Array, ""],
    n_iterations: int = 20,
) -> Float[Array, ""]:
    """Newton-Raphson implied vol inversion for FX options.

    Finds ``σ`` such that ``garman_kohlhagen_price(option, spot, σ, r_d, r_f) = market_price``.

    Args:
        option: FX option contract.
        spot: Spot FX rate.
        r_domestic: Domestic rate.
        r_foreign: Foreign rate.
        market_price: Observed option price in domestic currency.
        n_iterations: Newton steps.

    Returns:
        Implied Black-Scholes volatility.
    """
    price_fn = lambda v: garman_kohlhagen_price(option, spot, v, r_domestic, r_foreign)
    vega_fn = jax.grad(price_fn)

    def newton_step(vol, _):
        p = price_fn(vol)
        v = vega_fn(vol)
        v = jnp.maximum(v, 1e-10)
        return vol - (p - market_price) / v, None

    vol_init = jnp.array(0.10)
    vol, _ = jax.lax.scan(newton_step, vol_init, None, length=n_iterations)
    return vol


# ── FX Delta conventions ─────────────────────────────────────────────


def fx_delta(
    option: FXVanillaOption,
    spot: Float[Array, ""],
    vol: Float[Array, ""],
    r_domestic: Float[Array, ""],
    r_foreign: Float[Array, ""],
    convention: str = "spot",
) -> Float[Array, ""]:
    """FX option delta under various market conventions.

    **Spot delta** (``"spot"``):

    .. math::

        \\Delta_{\\text{spot}}^{\\text{call}} = e^{-r_f T} \\Phi(d_1)

        \\Delta_{\\text{spot}}^{\\text{put}} = -e^{-r_f T} \\Phi(-d_1)

    **Forward delta** (``"forward"``):

    .. math::

        \\Delta_{\\text{fwd}}^{\\text{call}} = \\Phi(d_1)

        \\Delta_{\\text{fwd}}^{\\text{put}} = -\\Phi(-d_1) = \\Phi(d_1) - 1

    **Premium-adjusted delta** (``"premium_adjusted"``):

    Accounts for the fact that the premium itself has FX exposure when
    paid in foreign currency:

    .. math::

        \\Delta_{\\text{pa}}^{\\text{call}} = e^{-r_f T} \\Phi(d_1) - \\frac{V}{S \\cdot N}

    where :math:`V` is the option value in domestic and :math:`N` is the
    foreign notional.

    Args:
        option: FX vanilla option.
        spot: Spot FX rate.
        vol: Implied volatility.
        r_domestic: Domestic rate.
        r_foreign: Foreign rate.
        convention: ``"spot"``, ``"forward"``, or ``"premium_adjusted"``.

    Returns:
        Delta (positive for calls, negative for puts by convention).
    """
    d1, _ = _gk_d1d2(spot, option.strike, option.expiry, vol, r_domestic, r_foreign)
    Phi = jax.scipy.stats.norm.cdf
    df_for = jnp.exp(-r_foreign * option.expiry)

    if convention == "spot":
        if option.is_call:
            return df_for * Phi(d1)
        return -df_for * Phi(-d1)

    elif convention == "forward":
        if option.is_call:
            return Phi(d1)
        return -Phi(-d1)

    elif convention == "premium_adjusted":
        price = garman_kohlhagen_price(option, spot, vol, r_domestic, r_foreign)
        premium_adj = price / (spot * option.notional_foreign)
        if option.is_call:
            return df_for * Phi(d1) - premium_adj
        return -df_for * Phi(-d1) + premium_adj

    else:
        raise ValueError(
            f"Unknown delta convention: {convention!r}. "
            "Use 'spot', 'forward', or 'premium_adjusted'."
        )


def strike_to_delta(
    strike: Float[Array, ""],
    spot: Float[Array, ""],
    vol: Float[Array, ""],
    r_domestic: Float[Array, ""],
    r_foreign: Float[Array, ""],
    expiry: Float[Array, ""],
    is_call: bool = True,
    convention: str = "spot",
) -> Float[Array, ""]:
    """Convert an FX strike to its delta under the given convention.

    Convenience function that constructs a unit-notional option and
    computes its delta.

    Args:
        strike: FX strike rate.
        spot: Spot rate.
        vol: Implied volatility.
        r_domestic: Domestic rate.
        r_foreign: Foreign rate.
        expiry: Time to expiry.
        is_call: Call or put.
        convention: Delta convention.

    Returns:
        Delta value.
    """
    opt = FXVanillaOption(
        strike=strike,
        expiry=expiry,
        notional_foreign=jnp.array(1.0),
        is_call=is_call,
    )
    return fx_delta(opt, spot, vol, r_domestic, r_foreign, convention)


def delta_to_strike(
    target_delta: Float[Array, ""],
    spot: Float[Array, ""],
    vol: Float[Array, ""],
    r_domestic: Float[Array, ""],
    r_foreign: Float[Array, ""],
    expiry: Float[Array, ""],
    is_call: bool = True,
    convention: str = "spot",
    n_iterations: int = 30,
) -> Float[Array, ""]:
    """Convert an FX delta to a strike via Newton-Raphson inversion.

    Finds the strike ``K`` such that ``strike_to_delta(K, ...) = target_delta``.

    This is the key utility for building FX vol surfaces from market
    quotes in delta space (10Δ put, 25Δ put, ATM, 25Δ call, 10Δ call).

    Args:
        target_delta: Desired delta value.
        spot: Spot FX rate.
        vol: Implied volatility at this delta point.
        r_domestic: Domestic rate.
        r_foreign: Foreign rate.
        expiry: Time to expiry.
        is_call: Call or put.
        convention: Delta convention.
        n_iterations: Newton steps.

    Returns:
        Strike that achieves the target delta.
    """
    delta_fn = lambda k: strike_to_delta(
        k, spot, vol, r_domestic, r_foreign, expiry, is_call, convention,
    )
    ddelta_dk = jax.grad(delta_fn)

    # Initial guess: ATM forward
    F = fx_forward_rate(spot, r_domestic, r_foreign, expiry)

    def newton_step(k, _):
        d = delta_fn(k)
        dd = ddelta_dk(k)
        dd = jnp.where(jnp.abs(dd) < 1e-15, -1e-15, dd)
        return k - (d - target_delta) / dd, None

    k, _ = jax.lax.scan(newton_step, F, None, length=n_iterations)
    return k
