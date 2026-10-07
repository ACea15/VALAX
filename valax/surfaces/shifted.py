"""Parallel-shift decorator over any duck-typed vol surface.

A scenario has to be able to move *volatility* on a
:class:`~valax.core.market_state.MarketState`, where vols are surface
**objects** (``GridVolSurface``, ``SVIVolSurface``, ``SABRVolSurface``,
``SwaptionCube``, ``OptionletVolSurface``, ``ConstantVol``, ...) rather
than one scalar per asset.  Two representations were possible:

(a) shock the surface's **nodes** — expressive (a whole smile move can be
    described) but grid-specific: it has no meaning for SVI / SABR /
    parametric surfaces, whose leaves are model parameters, not vols;
(b) wrap the surface in a **parallel-shift decorator** — uniform across
    every surface type at the cost of expressiveness (one number per
    surface).

VALAX ships (b) as the scenario primitive: :class:`ShiftedVolSurface`
here, applied by :func:`valax.risk.state_shocks.shift_surface`.  It is
type-agnostic, composes with itself, and keeps the shock magnitude a
single differentiable scalar — the same shape the sensitivity ladders
already assume (``SensitivityLadder.delta_vol`` is one vega per asset).
Node-level shocks stay available per surface type via the surfaces'
own leaves (``eqx.tree_at(lambda s: s.vols, surface, ...)``) and are the
natural place to plug (a) in later without changing this type.

No floor is applied to the shifted vol: a shift that drives a lognormal
vol negative is the caller's problem, exactly as with the legacy
``MarketData.vols + scenario.vol_shocks`` arithmetic.
"""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


#: Floor on the expiry used to invert total variance into a vol.
_MIN_EXPIRY = 1e-12


class ShiftedVolSurface(eqx.Module):
    """A vol surface with an additive parallel shift on implied vol.

    Wraps any duck-typed vol source and adds ``shift`` to every quote it
    returns, leaving the shape of the smile/term-structure untouched:

    .. math::

        \\sigma_{\\text{new}}(K, T) = \\sigma_{\\text{base}}(K, T) + \\delta.

    ``total_variance`` is kept consistent with ``__call__`` by inverting
    the base surface's variance into a vol, shifting, and re-squaring, so
    a Dupire / SLV pipeline reading either entry point sees the same
    shifted surface.

    Attributes:
        base: The wrapped surface (any object that is callable, and
            optionally exposes ``total_variance`` / ``is_normal``).
        shift: Additive parallel vol shift, in the base surface's
            quoting convention (lognormal or normal).
    """

    base: Any
    shift: Float[Array, ""]

    def __call__(
        self,
        *coords: Float[Array, ""],
        forward: Float[Array, ""] | None = None,
    ) -> Float[Array, ""]:
        """Return the base surface's quote at ``coords`` plus the shift.

        Args:
            *coords: Whatever coordinate tuple the base surface takes
                (e.g. ``(strike, expiry)``, ``(strike, expiry, tenor)``).
            forward: Optional live forward from the curves, per the vol-source
                protocol of :mod:`valax.pricing.analytic.rates_smile`. Passed
                through to the base surface only when supplied, so wrapping an
                equity surface (which takes no forward) still works.

        Returns:
            The shifted implied volatility.
        """
        if forward is None:
            return jnp.asarray(self.base(*coords)) + self.shift
        return jnp.asarray(self.base(*coords, forward=forward)) + self.shift

    def total_variance(
        self,
        log_moneyness: Float[Array, ""],
        expiry: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Total variance of the shifted surface.

        Consistent with :meth:`__call__`: the base total variance is
        inverted to a vol, shifted, then re-squared.

        Args:
            log_moneyness: ``k = log(K / F)`` in the base surface's
                convention.
            expiry: Year fraction.

        Returns:
            Scalar total variance of the shifted surface.

        Raises:
            AttributeError: If the wrapped surface does not implement
                ``total_variance`` (e.g. a flat
                :class:`~valax.surfaces.constant.ConstantVol`, which
                :func:`valax.risk.state_shocks.shift_surface` shifts in
                place instead of wrapping).
        """
        w = jnp.asarray(self.base.total_variance(log_moneyness, expiry))
        safe_expiry = jnp.maximum(expiry, _MIN_EXPIRY)
        sigma = jnp.sqrt(jnp.maximum(w, 0.0) / safe_expiry) + self.shift
        return sigma * sigma * expiry

    @property
    def is_normal(self) -> bool:
        """Quoting convention inherited from the wrapped surface.

        Returns:
            The base surface's ``is_normal`` flag, or ``False`` when it
            declares none (the lognormal default).
        """
        return bool(getattr(self.base, "is_normal", False))
