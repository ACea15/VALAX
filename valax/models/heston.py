"""Heston stochastic volatility model.

dS = (r - q) S dt + sqrt(V) S dW_1
dV = kappa (theta - V) dt + xi sqrt(V) dW_2
Corr(dW_1, dW_2) = rho
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


class HestonModel(eqx.Module):
    """Heston model parameters."""

    v0: Float[Array, ""]       # initial variance
    kappa: Float[Array, ""]    # mean reversion speed
    theta: Float[Array, ""]    # long-run variance
    xi: Float[Array, ""]       # vol of vol
    rho: Float[Array, ""]      # correlation between spot and vol Brownians
    rate: Float[Array, ""]     # risk-free rate
    dividend: Float[Array, ""] # continuous dividend yield


class HestonDrift(eqx.Module):
    """Drift for the (log S, V) Heston system."""

    rate: Float[Array, ""]
    dividend: Float[Array, ""]
    kappa: Float[Array, ""]
    theta: Float[Array, ""]

    def __call__(self, t, y, args):
        log_s, v = y
        v_pos = jnp.maximum(v, 0.0)
        d_log_s = (self.rate - self.dividend) - 0.5 * v_pos
        d_v = self.kappa * (self.theta - v_pos)
        return jnp.array([d_log_s, d_v])


class HestonDiffusion(eqx.Module):
    """Diffusion for the (log S, V) Heston system.

    Returns a (2, 2) matrix for correlated Brownians.
    """

    xi: Float[Array, ""]
    rho: Float[Array, ""]

    def __call__(self, t, y, args):
        _, v = y
        sqrt_v = jnp.sqrt(jnp.maximum(v, 0.0))
        return jnp.array([
            [sqrt_v, 0.0],
            [self.xi * sqrt_v * self.rho, self.xi * sqrt_v * jnp.sqrt(1.0 - self.rho**2)],
        ])
