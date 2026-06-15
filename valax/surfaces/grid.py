"""Grid-based volatility surface with bilinear interpolation.

The simplest vol surface construction: store implied vols on a
(strike, expiry) grid and interpolate. No model assumptions.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.surfaces._interp import bilinear_2d


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

        Uses bilinear interpolation in (strike, expiry) with flat
        extrapolation outside the grid.
        """
        return bilinear_2d(
            self.vols, self.strikes, self.expiries, strike, expiry
        )

    def total_variance(
        self,
        log_moneyness: Float[Array, ""],
        expiry: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Total variance ``w = sigma^2 * T`` at given log-moneyness and expiry.

        The grid is indexed by *strike* internally; this method converts
        from log-moneyness by assuming the surface is parameterised in
        absolute strike with forward = 1 (i.e. the caller has already
        normalised log-moneyness against the relevant forward). For an
        SLV / Dupire pipeline that needs a forward-relative quotation,
        either rebuild the grid against log-moneyness directly or compose
        with a forward curve at the call site.

        Args:
            log_moneyness: ``k = log(K / F)``. Used as the strike axis
                query directly — see note above on conventions.
            expiry: Year fraction.

        Returns:
            Scalar total variance.
        """
        sigma = bilinear_2d(
            self.vols, self.strikes, self.expiries, log_moneyness, expiry
        )
        return sigma * sigma * expiry
