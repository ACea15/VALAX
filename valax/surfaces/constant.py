"""Constant (flat) volatility as a callable vol source.

Lets a scalar volatility satisfy the same structural interface as the SABR smile
objects (:class:`~valax.surfaces.swaption_cube.SwaptionCube`,
:class:`~valax.surfaces.optionlet_surface.OptionletVolSurface`): a callable
returning an implied vol at a coordinate, carrying an ``is_normal`` quoting flag.
This keeps the curve-aware rates pricers convention-uniform -- a flat vol and a
full smile plug into the same entry point.

Deliberately self-contained (equinox + jax only) so it can be imported without
pulling in the rest of the surfaces or pricing packages.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


class ConstantVol(eqx.Module):
    """Flat volatility that ignores the query coordinate.

    Satisfies the "vol source" structural protocol used by the curve-aware
    rates pricers: it is callable and exposes an ``is_normal`` flag.

    Attributes:
        vol: The constant volatility.
        is_normal: Quoting convention -- True for normal (Bachelier), False for
            lognormal (Black-76). Selects which model a curve-aware pricer uses.
    """

    vol: Float[Array, ""]
    is_normal: bool = eqx.field(static=True, default=False)

    def __call__(
        self,
        *coords: Float[Array, ""],
        forward: Float[Array, ""] | None = None,
    ) -> Float[Array, ""]:
        """Return the constant vol.

        Args:
            *coords: Ignored ``(strike, expiry[, tenor])`` coordinates.
            forward: Ignored live forward. Accepted so a flat vol remains a
                drop-in replacement for a curve-aware smile under the vol-source
                protocol of :mod:`valax.pricing.analytic.rates_smile`.

        Returns:
            The constant volatility.
        """
        return jnp.asarray(self.vol)
