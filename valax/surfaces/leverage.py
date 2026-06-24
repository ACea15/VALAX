"""Tabulated leverage function L(k, t) for stochastic-local volatility.

The leverage function is the multiplicative correction applied to the
Heston volatility under a Stochastic-Local Volatility (SLV) model so
that the marginal distribution of ``S_t`` reproduces a calibrated
implied-vol surface (Markovian projection / Gyöngy's lemma):

.. math::

    \\sigma_{loc}^2(k, t) \\;=\\; L^2(k, t) \\cdot \\mathbb{E}[V_t \\mid k_t = k].

We store ``L`` on a fixed ``(k, t)`` grid and interpolate with the
project-wide ``bilinear_2d`` helper. The grid axes are *static* (used as
the interpolation scaffolding), only ``values`` carries differentiable
leaves — gradients flow through autodiff into the per-node leverage
values, which is what calibration needs.

Storage convention (mirrors ``GridVolSurface.vols``):

    values.shape == (n_t, n_k)   # y outer, x inner
    values[i, j]  ==  L(log_moneyness_grid[j], time_grid[i])

Query convention mirrors ``dupire_local_vol``: scalar in / scalar out,
``jax.vmap`` to batch.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from valax.surfaces._interp import bilinear_2d


class LeverageGrid(eqx.Module):
    """Tabulated leverage ``L(k, t)`` with bilinear interpolation.

    Attributes:
        log_moneyness_grid: Sorted log-moneyness grid, shape ``(n_k,)``.
        time_grid: Sorted time grid (year fractions), shape ``(n_t,)``.
            ``time_grid[0]`` should be ``> 0`` (Dupire is singular at 0).
        values: Leverage values, shape ``(n_t, n_k)``. Strictly positive
            in any well-calibrated grid.
    """

    log_moneyness_grid: Float[Array, " n_k"]
    time_grid: Float[Array, " n_t"]
    values: Float[Array, "n_t n_k"]

    def __call__(
        self,
        log_moneyness: Float[Array, ""],
        time: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Interpolate ``L(k, t)`` at a single point.

        Flat extrapolation outside the grid (delegates to
        :func:`bilinear_2d`).
        """
        return bilinear_2d(
            self.values,
            self.log_moneyness_grid,
            self.time_grid,
            log_moneyness,
            time,
        )

    @classmethod
    def flat(
        cls,
        log_moneyness_grid: Float[Array, " n_k"],
        time_grid: Float[Array, " n_t"],
        value: Float[Array, ""] | float = 1.0,
    ) -> "LeverageGrid":
        """Build a constant-leverage grid (``L ≡ value``).

        ``value = 1`` recovers the pure-Heston limit of the SLV SDE —
        useful as a calibration warm-start and as a Heston-limit
        reduction test.
        """
        log_moneyness_grid = jnp.asarray(log_moneyness_grid, dtype=jnp.float64)
        time_grid = jnp.asarray(time_grid, dtype=jnp.float64)
        n_t = time_grid.shape[0]
        n_k = log_moneyness_grid.shape[0]
        values = jnp.full(
            (n_t, n_k),
            jnp.asarray(value, dtype=jnp.float64),
        )
        return cls(
            log_moneyness_grid=log_moneyness_grid,
            time_grid=time_grid,
            values=values,
        )
