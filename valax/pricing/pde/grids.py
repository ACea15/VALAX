"""Spatial grid construction and price read-off.

Grid building (previously inlined in ``pde_price``) is factored into reusable
builders returning a lightweight :class:`Grid1D` pytree. The default is uniform
spacing in log-spot ``x = ln S`` (equal resolution in moneyness); a linear
builder is provided for domains bounded by a barrier.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


class Grid1D(eqx.Module):
    """A 1-D spatial grid of interior nodes.

    Attributes:
        nodes: Strictly increasing coordinates (e.g. log-spot values).
        n: Number of nodes (static).
    """

    nodes: Float[Array, " n"]
    n: int = eqx.field(static=True)


def spacing(grid: Grid1D) -> Float[Array, ""]:
    """Return the (uniform) node spacing ``dx`` of ``grid``."""
    return grid.nodes[1] - grid.nodes[0]


def boundary_coords(grid: Grid1D) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Return the two exterior boundary coordinates ``(x_min, x_max)``.

    The interior grid excludes the endpoints (Dirichlet boundary treatment);
    this returns the coordinates one spacing beyond the first/last nodes.
    """
    dx = spacing(grid)
    return grid.nodes[0] - dx, grid.nodes[-1] + dx


def uniform_log_spot_grid(
    spot: Float[Array, ""],
    vol: Float[Array, ""],
    expiry: Float[Array, ""],
    *,
    n: int,
    half_width: float,
) -> Grid1D:
    """Build a uniform log-spot grid centred on ``ln(spot)``.

    The grid spans ``ln(spot) +/- half_width * vol * sqrt(expiry)`` and returns
    the ``n`` interior nodes (endpoints excluded, matching the Dirichlet
    boundary treatment used by the solver).

    Args:
        spot: Current spot price.
        vol: Reference volatility used to scale the grid width.
        expiry: Time to expiry.
        n: Number of interior nodes.
        half_width: Half-width in std-dev units.

    Returns:
        A :class:`Grid1D` of ``n`` interior log-spot nodes.
    """
    x_center = jnp.log(spot)
    x_width = half_width * vol * jnp.sqrt(expiry)
    x_min = x_center - x_width
    x_max = x_center + x_width
    dx = (x_max - x_min) / (n + 1)
    nodes = x_min + dx * jnp.arange(1, n + 1)
    return Grid1D(nodes=nodes, n=n)


def uniform_linear_grid(
    lo: Float[Array, ""],
    hi: Float[Array, ""],
    *,
    n: int,
) -> Grid1D:
    """Build a uniform grid of ``n`` interior nodes on ``[lo, hi]``.

    Endpoints are excluded (they are handled as Dirichlet boundaries). Used for
    barrier problems where one edge of the domain is the barrier level.

    Args:
        lo: Lower domain boundary coordinate.
        hi: Upper domain boundary coordinate.
        n: Number of interior nodes.

    Returns:
        A :class:`Grid1D` of ``n`` interior nodes.
    """
    dx = (hi - lo) / (n + 1)
    nodes = lo + dx * jnp.arange(1, n + 1)
    return Grid1D(nodes=nodes, n=n)


def read_off_1d(
    grid: Grid1D,
    values: Float[Array, " n"],
    query: Float[Array, ""],
) -> Float[Array, ""]:
    """Interpolate grid ``values`` at a single ``query`` coordinate.

    Uses ``jnp.interp`` (piecewise-linear, differentiable) with flat
    extrapolation outside the grid.

    Args:
        grid: The spatial grid.
        values: Field values at the grid nodes.
        query: Coordinate at which to read off the value.

    Returns:
        The interpolated scalar value.
    """
    return jnp.interp(query, grid.nodes, values)
