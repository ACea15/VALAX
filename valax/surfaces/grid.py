"""Grid-based volatility surface with bilinear interpolation.

The simplest vol surface construction: store implied vols on a
(strike, expiry) grid and interpolate. No model assumptions.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


class GridVolSurface(eqx.Module):
    """Implied volatility surface on a regular grid.

    Stores a matrix of implied vols at discrete (strike, expiry) nodes.
    Interpolation is bilinear in (strike, expiry) space with flat
    extrapolation beyond the grid boundaries.

    All array fields are differentiable — ``jax.grad`` through a pricing
    function that queries this surface gives vega-like sensitivities to
    each grid node.

    Attributes:
        strikes: Sorted strike grid.
        expiries: Sorted expiry grid (year fractions).
        vols: Implied vol matrix, shape ``(n_expiries, n_strikes)``.
    """

    strikes: Float[Array, " n_strikes"]
    expiries: Float[Array, " n_expiries"]
    vols: Float[Array, "n_expiries n_strikes"]

    def __call__(
        self,
        strike: Float[Array, ""],
        expiry: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Interpolate implied vol at a (strike, expiry) point.

        Uses bilinear interpolation: first interpolate the vol smile
        at the two bracketing expiries, then interpolate between them
        in the expiry dimension.
        """
        # Interpolate each expiry slice at the query strike
        # jnp.interp does piecewise linear with flat extrapolation
        vol_at_strike = _interp_each_row(self.vols, self.strikes, strike)

        # Interpolate across expiries
        return jnp.interp(expiry, self.expiries, vol_at_strike)


def _interp_each_row(
    vols: Float[Array, "n_expiries n_strikes"],
    strikes: Float[Array, " n_strikes"],
    strike: Float[Array, ""],
) -> Float[Array, " n_expiries"]:
    """Interpolate each row of the vol grid at a single strike.

    This is equivalent to ``jnp.array([jnp.interp(strike, strikes, row) for row in vols])``
    but written as a vectorized operation for JIT efficiency.
    """
    # Clamp strike into the grid range for flat extrapolation
    k = jnp.clip(strike, strikes[0], strikes[-1])

    # Find the bracketing index
    idx = jnp.searchsorted(strikes, k, side="right") - 1
    idx = jnp.clip(idx, 0, strikes.shape[0] - 2)

    # Interpolation weight
    k_lo = strikes[idx]
    k_hi = strikes[idx + 1]
    w = (k - k_lo) / (k_hi - k_lo)
    w = jnp.clip(w, 0.0, 1.0)

    # Interpolate each row
    vol_lo = vols[:, idx]
    vol_hi = vols[:, idx + 1]
    return vol_lo + w * (vol_hi - vol_lo)
