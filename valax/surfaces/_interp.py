"""Internal 2-D interpolation utilities for vol surfaces and leverage grids.

Pure functions, scalar-in / scalar-out, autodiff-clean. Mirrors the
``jnp.searchsorted`` + clip + linear-blend pattern used throughout VALAX
for 1-D ``jnp.interp``, lifted to a rank-2 value grid.

Used by:
    - ``valax.surfaces.grid.GridVolSurface`` for bilinear vol lookup.
    - Future SLV ``LeverageGrid`` for tabulated ``L(S, t)`` lookup.

Design notes:
    * Bilinear interpolation with **flat extrapolation** outside grid bounds
      (clip query to grid range). This matches the existing GridVolSurface
      behaviour exactly so the refactor is value-identical.
    * Pure ``jax.numpy``: no Python branching on traced values, no
      ``where`` on raw zeros that would poison autodiff.
    * Autodiff is well-defined w.r.t. ``values`` (grid node values) and
      w.r.t. the query point. Differentiability w.r.t. the *grid axes*
      themselves (``x_grid`` / ``y_grid``) is also clean but rarely
      useful — grids are typically static.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


def _interp1d_along_axis(
    values: Float[Array, " n"],
    grid: Float[Array, " n"],
    query: Float[Array, ""],
) -> Float[Array, ""]:
    """Single-axis linear interpolation with flat extrapolation.

    Equivalent to ``jnp.interp(query, grid, values)`` but written with the
    explicit ``searchsorted`` + clamp pattern so it composes cleanly with
    the 2-D version below.
    """
    q = jnp.clip(query, grid[0], grid[-1])
    idx = jnp.searchsorted(grid, q, side="right") - 1
    idx = jnp.clip(idx, 0, grid.shape[0] - 2)

    g_lo = grid[idx]
    g_hi = grid[idx + 1]
    w = (q - g_lo) / (g_hi - g_lo)
    w = jnp.clip(w, 0.0, 1.0)

    v_lo = values[idx]
    v_hi = values[idx + 1]
    return v_lo + w * (v_hi - v_lo)


def _interp1d_each_row(
    values: Float[Array, "m n"],
    grid: Float[Array, " n"],
    query: Float[Array, ""],
) -> Float[Array, " m"]:
    """Interpolate each row of a rank-2 grid at a single query point.

    Equivalent to ``vmap(lambda row: jnp.interp(query, grid, row))(values)``
    but written without a vmap so the partial result is a plain array
    indexable column slice — slightly faster under jit than a vmap.
    """
    q = jnp.clip(query, grid[0], grid[-1])
    idx = jnp.searchsorted(grid, q, side="right") - 1
    idx = jnp.clip(idx, 0, grid.shape[0] - 2)

    g_lo = grid[idx]
    g_hi = grid[idx + 1]
    w = (q - g_lo) / (g_hi - g_lo)
    w = jnp.clip(w, 0.0, 1.0)

    v_lo = values[:, idx]
    v_hi = values[:, idx + 1]
    return v_lo + w * (v_hi - v_lo)


def bilinear_2d(
    values: Float[Array, "n_y n_x"],
    x_grid: Float[Array, " n_x"],
    y_grid: Float[Array, " n_y"],
    x_query: Float[Array, ""],
    y_query: Float[Array, ""],
) -> Float[Array, ""]:
    """Bilinear interpolation on a regular 2-D grid with flat extrapolation.

    The value at ``(x_query, y_query)`` is computed by first interpolating
    each row of ``values`` at ``x_query`` (collapsing to a 1-D slice along
    ``y``), then interpolating that slice at ``y_query``.

    Args:
        values: Rank-2 grid of shape ``(n_y, n_x)`` — ``values[i, j]`` is
            the value at ``(x_grid[j], y_grid[i])``. The ``(n_y, n_x)``
            ordering (y outer, x inner) matches the existing
            ``GridVolSurface.vols`` convention.
        x_grid: Sorted x-axis grid, shape ``(n_x,)``.
        y_grid: Sorted y-axis grid, shape ``(n_y,)``.
        x_query: Scalar query x-coordinate.
        y_query: Scalar query y-coordinate.

    Returns:
        Scalar interpolated value.

    Notes:
        * Queries outside the grid bounds are clamped (flat extrapolation).
        * Autodiff-clean: no ``where`` on raw zeros, no ``cond``.
    """
    row_at_x = _interp1d_each_row(values, x_grid, x_query)
    return _interp1d_along_axis(row_at_x, y_grid, y_query)
