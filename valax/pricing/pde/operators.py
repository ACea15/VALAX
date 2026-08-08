"""Spatial differential operators for 1-D finite-difference solvers.

An :class:`Operator1D` holds the per-row coefficients of the spatial operator

.. math::

    \\mathcal{L} V = \\tfrac{1}{2}\\sigma^2(x)\\, V_{xx} + \\mu(x)\\, V_x - r V,

discretised with central differences on a uniform grid. Coefficients may be
scalars (constant-coefficient Black-Scholes) or per-node arrays (local vol,
PR-2); both are broadcast to length ``n``.

The three bands are stored **per row** (length ``n`` each) so that row 0's
sub-diagonal coefficient and row ``n-1``'s super-diagonal coefficient — which
couple to the exterior Dirichlet boundaries — are available to the time-stepper.
The interior tridiagonal solve uses ``lower[1:]`` and ``upper[:-1]``.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.pricing.pde.grids import Grid1D, spacing


class Operator1D(eqx.Module):
    """Per-row coefficients of the discrete spatial operator ``L``.

    Attributes:
        lower: Sub-diagonal coefficient at each row (length ``n``).
        diag: Main-diagonal coefficient at each row (length ``n``).
        upper: Super-diagonal coefficient at each row (length ``n``).
    """

    lower: Float[Array, " n"]
    diag: Float[Array, " n"]
    upper: Float[Array, " n"]


def build_operator_1d(
    grid: Grid1D,
    drift: Float[Array, ""] | Float[Array, " n"],
    diffusion: Float[Array, ""] | Float[Array, " n"],
    discount: Float[Array, ""],
) -> Operator1D:
    """Assemble the central-difference operator from drift/diffusion/discount.

    With uniform spacing ``dx``, the standard second-order stencil gives, at
    each node,

    - ``lower = diffusion / (2 dx^2) - drift / (2 dx)``
    - ``diag  = -diffusion / dx^2 - discount``
    - ``upper = diffusion / (2 dx^2) + drift / (2 dx)``

    where ``diffusion`` is ``sigma^2`` (the ``1/2`` of the diffusion term is
    absorbed into the ``1/(2 dx^2)`` factor), matching the incumbent solver's
    convention.

    Args:
        grid: Uniform spatial grid.
        drift: Convection coefficient ``mu`` (scalar or per-node).
        diffusion: Diffusion coefficient ``sigma^2`` (scalar or per-node).
        discount: Discount rate ``r`` (scalar).

    Returns:
        The assembled :class:`Operator1D`.
    """
    dx = spacing(grid)
    ones = jnp.ones(grid.n)
    mu = ones * drift
    sig2 = ones * diffusion
    r = discount

    lower = sig2 / (2.0 * dx**2) - mu / (2.0 * dx)
    diag = -sig2 / dx**2 - r
    upper = sig2 / (2.0 * dx**2) + mu / (2.0 * dx)
    return Operator1D(lower=lower, diag=diag, upper=upper)
