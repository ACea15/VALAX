"""Local volatility model.

A ``LocalVolModel`` carries an implied vol surface and a deterministic
rate / dividend pair. Under Dupire's construction, the spot SDE is

    dS_t = (r - q) S_t dt + sigma_loc(S_t, t) S_t dW_t,

where ``sigma_loc`` is extracted from the surface via the Dupire formula
(:func:`valax.pricing.analytic.dupire.dupire_local_vol`). The model
itself does **not** carry the leverage function ``sigma_loc`` — it is
recomputed from the surface on demand inside the path generator. This
keeps the model pytree small (no precomputed grid) and lets autodiff
flow through surface parameters cleanly.

The forward curve is implicit:

    F(t) = spot * exp((rate - dividend) * t).

For non-flat-curve cases (term-structure of rates / dividends) the
caller should pass an SVI / Grid surface whose ``total_variance``
already encodes the curve-dependent log-moneyness convention; the LV MC
path generator will compute ``k_t = log(S_t / F(t))`` using the flat
``rate - dividend`` drift baked into the model.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


class LocalVolModel(eqx.Module):
    """Local volatility model with surface-driven diffusion.

    Field order (immutable — downstream ``eqx.tree_at`` and pytree
    flatten/unflatten relies on it):

        1. ``surface``: any object exposing
           ``total_variance(log_moneyness, expiry) -> Float[""]``.
           ``SVIVolSurface``, ``SABRVolSurface``, and ``GridVolSurface``
           all comply. The surface lives at the front of the pytree
           because it carries the bulk of the differentiable leaves
           (SVI parameters, grid nodes, ...).
        2. ``rate``: continuously-compounded risk-free rate.
        3. ``dividend``: continuous dividend yield.

    Attributes:
        surface: Implied vol surface (duck-typed Dupire input).
        rate: Risk-free rate.
        dividend: Continuous dividend.
    """

    surface: eqx.Module
    rate: Float[Array, ""]
    dividend: Float[Array, ""]

    @classmethod
    def from_flat_rate(
        cls,
        surface: eqx.Module,
        rate: Float[Array, ""] | float,
        dividend: Float[Array, ""] | float = 0.0,
    ) -> "LocalVolModel":
        """Construct a LocalVolModel with scalar rate and dividend.

        Args:
            surface: A vol surface exposing ``total_variance``.
            rate: Risk-free rate.
            dividend: Continuous dividend yield (default 0).

        Returns:
            A ``LocalVolModel`` with rate / dividend coerced to scalar
            JAX arrays.
        """
        return cls(
            surface=surface,
            rate=jnp.asarray(rate, dtype=jnp.float64),
            dividend=jnp.asarray(dividend, dtype=jnp.float64),
        )
