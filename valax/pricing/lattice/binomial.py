"""Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.

The tree is built forward (spot prices at each node), then rolled back from
terminal payoff. American exercise is handled by taking the max of continuation
value and intrinsic value at each node.

The entire computation uses jax.lax.scan/fori_loop for JIT compatibility
and differentiability.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.instruments.options import EuropeanOption


class BinomialConfig(eqx.Module):
    """Binomial tree configuration."""

    n_steps: int = eqx.field(static=True, default=200)
    american: bool = eqx.field(static=True, default=False)


def binomial_price(
    option: EuropeanOption,
    spot: Float[Array, ""],
    vol: Float[Array, ""],
    rate: Float[Array, ""],
    dividend: Float[Array, ""],
    config: BinomialConfig = BinomialConfig(),
) -> Float[Array, ""]:
    """Price an option via CRR binomial tree.

    Args:
        option: European option contract.
        spot: Current spot price.
        vol: Volatility.
        rate: Risk-free rate.
        dividend: Continuous dividend yield.
        config: Tree configuration (n_steps, american flag).

    Returns:
        Option price.
    """
    N = config.n_steps
    T = option.expiry
    K = option.strike
    dt = T / N

    # CRR parameters
    u = jnp.exp(vol * jnp.sqrt(dt))
    d = 1.0 / u
    disc = jnp.exp(-rate * dt)
    p = (jnp.exp((rate - dividend) * dt) - d) / (u - d)

    # Terminal spot prices: S * u^j * d^(N-j) for j = 0..N
    j = jnp.arange(N + 1, dtype=jnp.float64)
    S_terminal = spot * u ** j * d ** (N - j)

    # Terminal payoff
    if option.is_call:
        values = jnp.maximum(S_terminal - K, 0.0)
    else:
        values = jnp.maximum(K - S_terminal, 0.0)

    # Precompute spot prices at all tree nodes for American exercise.
    # At level i (0 = root, N = terminal), node j has spot = S * u^j * d^(i-j).
    # We store all N+1 possible node values; only the first (level+1) are valid.
    j_all = jnp.arange(N + 1, dtype=jnp.float64)

    # Backward induction using fixed-shape arrays (no dynamic slicing).
    # At each step, values[j] = f(values[j], values[j+1]) for j=0..n-1,
    # and values[n:] are unused. We compute over the full array and mask.
    def step(values, i):
        """Roll back one step. i counts from 0 to N-1."""
        # Continuation: new[j] = disc * (p * old[j+1] + (1-p) * old[j])
        cont = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        # Pad to N+1 to keep shape constant
        cont = jnp.concatenate([cont, jnp.zeros(1)])

        if config.american:
            # At level (N-1-i), node j has spot = S * u^j * d^(N-1-i-j)
            level = N - 1 - i
            S_level = spot * u ** j_all * d ** (level - j_all)
            if option.is_call:
                intrinsic = jnp.maximum(S_level - K, 0.0)
            else:
                intrinsic = jnp.maximum(K - S_level, 0.0)
            cont = jnp.maximum(cont, intrinsic)

        return cont, None

    values, _ = jax.lax.scan(step, values, jnp.arange(N))
    return values[0]
