"""SABR stochastic volatility model.

dF = alpha * F^beta * dW_1
dalpha_t = nu * alpha_t * dW_2
Corr(dW_1, dW_2) = rho

where F is the forward rate, alpha is the stochastic vol,
beta controls the CEV backbone, nu is the vol-of-vol, rho is correlation.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


class SABRModel(eqx.Module):
    """SABR model parameters."""

    alpha: Float[Array, ""]    # initial volatility
    beta: Float[Array, ""]     # CEV exponent (0 = normal, 1 = lognormal)
    rho: Float[Array, ""]      # correlation between forward and vol Brownians
    nu: Float[Array, ""]       # vol of vol


class SABRDrift(eqx.Module):
    """Drift for the (F, alpha) SABR system — zero under the forward measure."""

    beta: Float[Array, ""]

    def __call__(self, t, y, args):
        return jnp.array([0.0, 0.0])


class SABRDiffusion(eqx.Module):
    """Diffusion for the (F, alpha) SABR system.

    Returns a (2, 2) matrix for correlated Brownians:
        dF      = alpha * F^beta * dW_1
        dalpha  = nu * alpha * (rho * dW_1 + sqrt(1 - rho^2) * dW_2)
    """

    beta: Float[Array, ""]
    nu: Float[Array, ""]
    rho: Float[Array, ""]

    def __call__(self, t, y, args):
        F, alpha = y
        F_pos = jnp.maximum(F, 1e-10)
        alpha_pos = jnp.maximum(alpha, 1e-10)
        F_beta = F_pos ** self.beta

        return jnp.array([
            [alpha_pos * F_beta, 0.0],
            [self.nu * alpha_pos * self.rho, self.nu * alpha_pos * jnp.sqrt(1.0 - self.rho**2)],
        ])
