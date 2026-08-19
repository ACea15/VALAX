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


def boundary_coords(grid: Grid1D) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Return the two exterior boundary coordinates ``(x_min, x_max)``.

    The interior grid excludes the endpoints (Dirichlet boundary treatment);
    this returns the coordinates one *edge* spacing beyond the first / last
    nodes. The lower boundary uses the first interior interval
    (``nodes[1] - nodes[0]``) and the upper boundary uses the last interior
    interval (``nodes[-1] - nodes[-2]``), so a non-uniform grid places each
    exterior node symmetrically with respect to its adjacent interval. For a
    uniform grid both intervals are equal and this reduces to the original
    ``nodes[0] - dx`` / ``nodes[-1] + dx``.
    """
    lo = grid.nodes[0] - (grid.nodes[1] - grid.nodes[0])
    hi = grid.nodes[-1] + (grid.nodes[-1] - grid.nodes[-2])
    return lo, hi


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


def sinh_concentrated_grid(
    lo: Float[Array, ""],
    hi: Float[Array, ""],
    center: Float[Array, ""],
    *,
    n: int,
    scale: Float[Array, ""] | float,
) -> Grid1D:
    """Build a grid on ``[lo, hi]`` with nodes concentrated around ``center``.

    Uses the Tavella-Randall ``sinh`` stretching: a uniform coordinate
    ``u in (0, 1)`` is mapped through

    .. math::

        x(u) = c + \\alpha\\,\\sinh\\!\\big(c_1 + u\\,(c_2 - c_1)\\big),\\quad
        c_1 = \\operatorname{asinh}\\tfrac{\\text{lo} - c}{\\alpha},\\;
        c_2 = \\operatorname{asinh}\\tfrac{\\text{hi} - c}{\\alpha},

    which places the finest resolution near ``center`` (where ``sinh`` is
    locally linear) and coarsens smoothly toward the domain edges. The
    stretching intensity is set by ``scale`` (``alpha``): **smaller ``scale``
    concentrates more tightly** around ``center``; as ``scale -> inf`` the grid
    approaches uniform.

    Endpoints are excluded (Dirichlet boundary treatment): the ``n`` interior
    nodes use ``u_i = i / (n + 1)`` for ``i = 1 .. n``, so ``u = 0`` and
    ``u = 1`` recover ``lo`` and ``hi`` as the exterior boundaries — consistent
    with :func:`boundary_coords`' edge extrapolation.

    Args:
        lo: Lower domain boundary coordinate.
        hi: Upper domain boundary coordinate.
        center: Coordinate around which to concentrate resolution.
        n: Number of interior nodes.
        scale: ``sinh`` stretching scale ``alpha`` (smaller = tighter).

    Returns:
        A :class:`Grid1D` of ``n`` interior nodes, strictly increasing.
    """
    c1 = jnp.arcsinh((lo - center) / scale)
    c2 = jnp.arcsinh((hi - center) / scale)
    u = jnp.arange(1, n + 1) / (n + 1)
    nodes = center + scale * jnp.sinh(c1 + u * (c2 - c1))
    return Grid1D(nodes=nodes, n=n)


def read_off_1d(
    grid: Grid1D,
    values: Float[Array, " n"],
    query: Float[Array, ""],
) -> Float[Array, ""]:
    """Interpolate grid ``values`` at a single ``query`` coordinate.

    Uses a **4-point cubic Hermite** spline (a non-uniform Catmull-Rom) rather
    than piecewise-linear ``jnp.interp``. The cubic is C1-continuous and --
    crucially -- carries genuine within-cell *curvature*, so its second
    derivative is non-zero. Combined with the ``stop_gradient`` grid detachment
    in :func:`uniform_log_spot_grid`, this lets ``jax.grad(jax.grad(...))``
    recover a correct second-order spot Greek (gamma); piecewise-linear
    interpolation has no curvature and collapses gamma to ~0. Delta and price
    are unaffected (both improve slightly) and flat extrapolation outside the
    grid is preserved to match the previous behaviour.

    The tangents at the two cell endpoints use the **second-order non-uniform
    three-point derivative** (the exact slope of the parabola through the three
    surrounding nodes), so the read-off reproduces quadratics exactly and stays
    accurate on concentrated (:func:`sinh_concentrated_grid`) as well as uniform
    grids. On a uniform grid these tangents collapse to the standard Catmull-Rom
    slopes ``(p2 - p0) / 2`` and ``(p3 - p1) / 2``, reproducing the incumbent
    formula exactly. (A naive central slope ``(p2 - p0) / (x2 - x0)`` is only
    first-order on a non-uniform grid and would bias the recovered gamma.)

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
    x0, x1, x2, x3 = xs[j - 1], xs[j], xs[j + 1], xs[j + 2]
    p0, p1, p2, p3 = values[j - 1], values[j], values[j + 1], values[j + 2]

    h = x2 - x1
    t = (query - x1) / h

    # Second-order non-uniform three-point derivative at the two cell endpoints
    # (exact for quadratics), scaled by the local cell width h to Hermite
    # parameter units. Tangent at x1 uses {x0, x1, x2}; at x2 uses {x1, x2, x3}.
    a1, b1 = x1 - x0, x2 - x1  # backward / forward spacings around x1
    m1 = h * (
        -b1 / (a1 * (a1 + b1)) * p0
        + (b1 - a1) / (a1 * b1) * p1
        + a1 / (b1 * (a1 + b1)) * p2
    )
    a2, b2 = x2 - x1, x3 - x2  # backward / forward spacings around x2
    m2 = h * (
        -b2 / (a2 * (a2 + b2)) * p1
        + (b2 - a2) / (a2 * b2) * p2
        + a2 / (b2 * (a2 + b2)) * p3
    )

    t2 = t * t
    t3 = t2 * t
    # Cubic Hermite basis: interpolates p1 at t=0 and p2 at t=1.
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    cubic = h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2

    # Flat extrapolation outside the interior grid (matches ``jnp.interp``).
    result = jnp.where(query <= xs[0], values[0], cubic)
    result = jnp.where(query >= xs[-1], values[-1], result)
    return result
