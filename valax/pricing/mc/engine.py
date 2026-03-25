"""Monte Carlo pricing engine.

Composes path generation + payoff evaluation + discounting.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.instruments.options import EuropeanOption
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.pricing.mc.paths import generate_gbm_paths, generate_heston_paths
from valax.pricing.mc.payoffs import european_payoff


class MCConfig(eqx.Module):
    """Monte Carlo simulation configuration."""

    n_paths: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)


def mc_price(
    option: EuropeanOption,
    spot: Float[Array, ""],
    model: eqx.Module,
    config: MCConfig,
    key: jax.Array,
    payoff_fn: Callable = european_payoff,
) -> Float[Array, ""]:
    """Price an option via Monte Carlo simulation.

    Dispatches path generation based on model type.

    Args:
        option: The option to price.
        spot: Current spot price.
        model: Stochastic model (BlackScholesModel or HestonModel).
        config: MC configuration (n_paths, n_steps).
        key: JAX PRNG key.
        payoff_fn: Payoff function mapping (paths, option) -> cashflows.

    Returns:
        Discounted expected payoff (MC price estimate).
    """
    T = option.expiry

    if isinstance(model, HestonModel):
        paths, _ = generate_heston_paths(
            model, spot, T, config.n_steps, config.n_paths, key
        )
        rate = model.rate
    else:
        paths = generate_gbm_paths(
            model, spot, T, config.n_steps, config.n_paths, key
        )
        rate = model.rate

    cashflows = payoff_fn(paths, option)
    df = jnp.exp(-rate * T)
    return df * jnp.mean(cashflows)


def mc_price_with_stderr(
    option: EuropeanOption,
    spot: Float[Array, ""],
    model: eqx.Module,
    config: MCConfig,
    key: jax.Array,
    payoff_fn: Callable = european_payoff,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """MC price with standard error estimate.

    Returns:
        (price, standard_error) tuple.
    """
    T = option.expiry

    if isinstance(model, HestonModel):
        paths, _ = generate_heston_paths(
            model, spot, T, config.n_steps, config.n_paths, key
        )
        rate = model.rate
    else:
        paths = generate_gbm_paths(
            model, spot, T, config.n_steps, config.n_paths, key
        )
        rate = model.rate

    cashflows = payoff_fn(paths, option)
    df = jnp.exp(-rate * T)

    price = df * jnp.mean(cashflows)
    stderr = df * jnp.std(cashflows) / jnp.sqrt(jnp.array(config.n_paths, dtype=jnp.float64))
    return price, stderr
