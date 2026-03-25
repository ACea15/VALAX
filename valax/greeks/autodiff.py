"""Generic autodiff Greek computation for any pricing function.

Usage:
    from valax.greeks import greeks

    # Get all first-order Greeks at once
    result = greeks(black_scholes_price, option, spot, vol, rate, dividend)
    # result["price"], result["delta"], result["vega"], result["rho"], ...

    # Get a single Greek by name
    delta = greek(black_scholes_price, "delta", option, spot, vol, rate, dividend)
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

import equinox as eqx


# Maps Greek names to (argnums for first derivative, argnums for second derivative)
# argnums are relative to the market args (spot=0, vol=1, rate=2, dividend=3)
# after the instrument argument which is always arg 0 in the pricing function.
_GREEKS = {
    # First order
    "delta": (1,),
    "vega": (2,),
    "rho": (3,),
    "dividend_rho": (4,),
    # Second order
    "gamma": (1, 1),
    "vanna": (1, 2),
    "volga": (2, 2),
}


def greek(
    pricing_fn: Callable,
    name: str,
    instrument: eqx.Module,
    *market_args,
) -> Float[Array, ""]:
    """Compute a single named Greek via autodiff.

    Args:
        pricing_fn: Pure pricing function with signature
            (instrument, spot, vol, rate, dividend) -> price.
        name: Greek name — one of: delta, vega, rho, dividend_rho,
            gamma, vanna, volga.
        instrument: The instrument pytree (not differentiated).
        *market_args: Market inputs (spot, vol, rate, dividend, ...).

    Returns:
        The Greek value.
    """
    if name == "theta":
        return _theta(pricing_fn, instrument, *market_args)

    argnums = _GREEKS[name]

    # Shift argnums: pricing_fn's arg 0 is instrument; market args start at 1.
    # But we wrap to close over the instrument, so argnums index into market_args.
    fn = lambda *args: pricing_fn(instrument, *args)

    if len(argnums) == 1:
        return jax.grad(fn, argnums=argnums[0] - 1)(*market_args)
    else:
        a, b = argnums
        return jax.grad(jax.grad(fn, argnums=a - 1), argnums=b - 1)(*market_args)


def _theta(
    pricing_fn: Callable,
    instrument: eqx.Module,
    *market_args,
    dt: float = 1.0 / 365.0,
) -> Float[Array, ""]:
    """Theta via finite difference on expiry.

    Autodiff w.r.t. time requires differentiating through the instrument's
    expiry field. We bump expiry by -dt and compute the price change.
    """
    price_now = pricing_fn(instrument, *market_args)

    # Bump expiry down by dt
    bumped = eqx.tree_at(lambda t: t.expiry, instrument, instrument.expiry - dt)
    price_bumped = pricing_fn(bumped, *market_args)

    return (price_bumped - price_now) / dt


def greeks(
    pricing_fn: Callable,
    instrument: eqx.Module,
    *market_args,
) -> dict[str, Float[Array, ""]]:
    """Compute price and all standard Greeks at once.

    Args:
        pricing_fn: Pure pricing function with signature
            (instrument, spot, vol, rate, dividend) -> price.
        instrument: The instrument pytree.
        *market_args: Market inputs (spot, vol, rate, dividend, ...).

    Returns:
        Dict with keys: price, delta, gamma, vega, volga, vanna,
        rho, dividend_rho, theta.
    """
    fn = lambda *args: pricing_fn(instrument, *args)
    price = fn(*market_args)

    # First-order Greeks — one backward pass for all
    first_order_grad = jax.grad(fn, argnums=(0, 1, 2, 3))
    delta, vega, rho, dividend_rho = first_order_grad(*market_args)

    # Second-order Greeks
    gamma = jax.grad(jax.grad(fn, argnums=0), argnums=0)(*market_args)
    vanna = jax.grad(jax.grad(fn, argnums=0), argnums=1)(*market_args)
    volga = jax.grad(jax.grad(fn, argnums=1), argnums=1)(*market_args)

    # Theta via bump
    theta = _theta(pricing_fn, instrument, *market_args)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "volga": volga,
        "vanna": vanna,
        "rho": rho,
        "dividend_rho": dividend_rho,
        "theta": theta,
    }
