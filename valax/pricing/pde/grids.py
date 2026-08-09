"""Spatial grid construction and price read-off.

Grid building (previously inlined in ``pde_price``) is factored into reusable
builders returning a lightweight :class:`Grid1D` pytree. The default is uniform
spacing in log-spot ``x = ln S`` (equal resolution in moneyness); a linear
builder is provided for domains bounded by a barrier.
"""

import equinox as eqx
import jax.numpy as jnp
from jax import Array, lax
from jaxtyping import Float


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

    **Spot is detached from autodiff** (via :func:`jax.lax.stop_gradient`) for
    the purpose of *grid placement only*: the nodes are still centred on the
    current spot value (unchanged forward price), but the grid no longer
    *co-moves* with ``spot`` under differentiation. This is what makes
    second-order spot Greeks (gamma) well defined. If the grid translated with
    ``spot``, the price read-off would sit at a frozen fractional cell position
    and the reconstruction would be piecewise-linear in ``spot`` -- exact delta
    but identically-zero pointwise gamma. Detaching leaves the only
    differentiable spot dependence in the read-off query ``ln(spot)``, so
    autodiff recovers ``delta = V_g' / S`` and ``gamma = (V_g'' - V_g') / S^2``
    directly (paired with the curvature-carrying :func:`read_off_1d`). ``vol``
    and ``expiry`` are intentionally left differentiable so vega/theta still
    capture the grid-width dependence.

    Args:
        spot: Current spot price.
        vol: Reference volatility used to scale the grid width.
        expiry: Time to expiry.
        n: Number of interior nodes.
        half_width: Half-width in std-dev units.

    Returns:
        A :class:`Grid1D` of ``n`` interior log-spot nodes.
    """
    x_center = jnp.log(lax.stop_gradient(spot))
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

    Uses a **Catmull-Rom cubic** (a 4-point Hermite spline on the uniform grid)
    rather than piecewise-linear ``jnp.interp``. The cubic is C1-continuous and
    -- crucially -- carries genuine within-cell *curvature*, so its second
    derivative is non-zero. Combined with the ``stop_gradient`` grid detachment
    in :func:`uniform_log_spot_grid`, this lets ``jax.grad(jax.grad(...))``
    recover a correct second-order spot Greek (gamma); piecewise-linear
    interpolation has no curvature and collapses gamma to ~0. Delta and price
    are unaffected (both improve slightly) and flat extrapolation outside the
    grid is preserved to match the previous behaviour.

    The Catmull-Rom coefficients assume (approximately) uniform node spacing,
    which holds for every grid builder in this module.

    Args:
        grid: The spatial grid.
        values: Field values at the grid nodes.
        query: Coordinate at which to read off the value.

    Returns:
        The interpolated scalar value.
    """
    xs = grid.nodes
    n = xs.shape[0]

    # Locate the cell [xs[j], xs[j+1]] containing ``query`` and clamp so the
    # 4-point stencil {j-1, j, j+1, j+2} stays in-bounds (needs n >= 4).
    j = jnp.clip(jnp.searchsorted(xs, query) - 1, 1, n - 3)
    dx = xs[j + 1] - xs[j]
    t = (query - xs[j]) / dx

    p0, p1, p2, p3 = values[j - 1], values[j], values[j + 1], values[j + 2]

    # Uniform Catmull-Rom cubic: interpolates p1 at t=0 and p2 at t=1.
    cubic = 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t**2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t**3
    )

    # Flat extrapolation outside the interior grid (matches ``jnp.interp``).
    result = jnp.where(query <= xs[0], values[0], cubic)
    result = jnp.where(query >= xs[-1], values[-1], result)
    return result
