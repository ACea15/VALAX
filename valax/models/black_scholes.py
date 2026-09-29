"""Geometric Brownian Motion / Black-Scholes model."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


class BlackScholesModel(eqx.Module):
    """GBM model parameters: dS = (r - q) S dt + sigma S dW.

    Every parameter may be given as a constant — the textbook use — or
    left ``None`` to be sourced from the :class:`MarketState` at price
    time by :func:`valax.core.binding.bind_model`::

        BlackScholesModel(vol=0.2, rate=0.05, dividend=0.0)  # all constant
        BlackScholesModel(vol=0.2)                # rate + div from market
        BlackScholesModel(rate=0.05)              # vol from the surface
        BlackScholesModel()                       # everything from market

    A market-sourced ``rate`` is the continuously-compounded zero rate at
    the instrument's own maturity, and a market-sourced ``vol`` is the
    surface sampled at ``(strike, expiry)``.  Because the binding happens
    inside the differentiated path, those parameters carry curve and
    surface sensitivities — which is what lets an equity book report
    pillar deltas rather than a structural zero.

    Attributes:
        vol: Lognormal volatility, or ``None`` to read the market surface.
        rate: Continuously-compounded risk-free rate, or ``None`` to
            project the discount curve at the instrument's maturity.
        dividend: Continuous dividend yield, or ``None`` to read the
            market's dividend for the underlying.
    """

    vol: Float[Array, ""] | None = None
    rate: Float[Array, ""] | None = None
    dividend: Float[Array, ""] | None = None


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
