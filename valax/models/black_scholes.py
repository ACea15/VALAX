"""Geometric Brownian Motion / Black-Scholes model."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


class BlackScholesModel(eqx.Module):
    """GBM model parameters: dS = (r - q) S dt + sigma S dW."""

    vol: Float[Array, ""]
    rate: Float[Array, ""]
    dividend: Float[Array, ""]


class GBMDrift(eqx.Module):
    """Drift coefficient for GBM: mu * S."""

    mu: Float[Array, ""]

    def __call__(self, t, y, args):
        return self.mu * y


class GBMDiffusion(eqx.Module):
    """Diffusion coefficient for GBM: sigma * S."""

    sigma: Float[Array, ""]

    def __call__(self, t, y, args):
        return self.sigma * y
