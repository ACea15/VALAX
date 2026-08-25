"""Spatial differential operators for 1-D finite-difference solvers.

An :class:`Operator1D` holds the per-row coefficients of the spatial operator

.. math::

    \\mathcal{L} V = \\tfrac{1}{2}\\sigma^2(x)\\, V_{xx} + \\mu(x)\\, V_x - r V,

discretised with three-point central differences on a (possibly non-uniform)
grid. Coefficients may be scalars (constant-coefficient Black-Scholes) or
per-node arrays (local vol, PR-2); both are broadcast to length ``n``.

The three bands are stored **per row** (length ``n`` each) so that row 0's
sub-diagonal coefficient and row ``n-1``'s super-diagonal coefficient — which
couple to the exterior Dirichlet boundaries — are available to the time-stepper.
The interior tridiagonal solve uses ``lower[1:]`` and ``upper[:-1]``.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.pricing.pde.grids import Grid1D, boundary_coords


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
    discount: Float[Array, ""] | Float[Array, " n"],
) -> Operator1D:
    """Assemble the central-difference operator from drift/diffusion/discount.

    Uses the three-point central-difference stencil for a **non-uniform** grid.
    With backward spacing ``h_- = x_j - x_{j-1}``, forward spacing
    ``h_+ = x_{j+1} - x_j`` and ``h_s = h_- + h_+``, the second-order stencil for
    ``L V = 1/2 sigma^2 V_xx + mu V_x - r V`` gives, at each node,

    - ``lower = sigma^2 / (h_- h_s) - mu h_+ / (h_- h_s)``
    - ``diag  = -sigma^2 / (h_- h_+) + mu (h_+ - h_-) / (h_- h_+) - r``
    - ``upper = sigma^2 / (h_+ h_s) + mu h_- / (h_+ h_s)``

    where ``diffusion`` is ``sigma^2`` (the ``1/2`` of the diffusion term is
    absorbed into the stencil weights). For a uniform grid ``h_- = h_+ = dx``
    and this reduces exactly to the classic
    ``sigma^2 / (2 dx^2) -/+ mu / (2 dx)`` form used by the incumbent solver.

    The exterior spacings (``h_-`` at node 0 and ``h_+`` at node ``n-1``) come
    from :func:`~valax.pricing.pde.grids.boundary_coords`, so the stencil and
    the Dirichlet boundary values (which use the same helper) share a single,
    consistent boundary placement.

    Args:
        grid: Spatial grid (uniform or concentrated).
        drift: Convection coefficient ``mu`` (scalar or per-node).
        diffusion: Diffusion coefficient ``sigma^2`` (scalar or per-node).
        discount: Discount rate ``r`` (scalar or per-node). Short-rate models
            need the per-node form: in the Hull-White state variable ``x`` the
            discount rate *is* the state, ``r = x + alpha(t)``.

    Returns:
        The assembled :class:`Operator1D`.
    """
    x_lo, x_hi = boundary_coords(grid)
    # Augment interior nodes with the two exterior boundary coordinates so the
    # edge rows get their (possibly asymmetric) boundary spacings.
    full = jnp.concatenate([x_lo[jnp.newaxis], grid.nodes, x_hi[jnp.newaxis]])
    h_minus = grid.nodes - full[:-2]  # x_j - x_{j-1}, length n
    h_plus = full[2:] - grid.nodes    # x_{j+1} - x_j, length n
    h_sum = h_minus + h_plus

    ones = jnp.ones(grid.n)
    mu = ones * drift
    sig2 = ones * diffusion
    r = discount

    lower = sig2 / (h_minus * h_sum) - mu * h_plus / (h_minus * h_sum)
    diag = -sig2 / (h_minus * h_plus) + mu * (h_plus - h_minus) / (h_minus * h_plus) - r
    upper = sig2 / (h_plus * h_sum) + mu * h_minus / (h_plus * h_sum)
    return Operator1D(lower=lower, diag=diag, upper=upper)
