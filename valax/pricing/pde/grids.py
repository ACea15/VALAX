"""Spatial grid construction and price read-off.

Grid building (previously inlined in ``pde_price``) is factored into reusable
builders returning a lightweight :class:`Grid1D` pytree. The default is uniform
spacing in log-spot ``x = ln S`` (equal resolution in moneyness); a linear
builder is provided for domains bounded by a barrier, and
:func:`sinh_concentrated_grid` for non-uniform meshes that cluster resolution
around a point of interest.

Two-dimensional (e.g. stochastic-volatility) pricing uses a tensor-product
:class:`Grid2D` (log-spot :math:`\\times` variance) with a separable bicubic
read-off (:func:`read_off_2d`) that reuses the 1-D Hermite kernel per axis.
"""

import equinox as eqx
import jax
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


def centred_state_grid(
    std_dev: Float[Array, ""],
    *,
    n: int,
    half_width: float,
) -> Grid1D:
    """Build a uniform grid of ``n`` interior nodes on ``[-w, w]``, ``w = half_width * std_dev``.

    The mesh for a **mean-reverting state variable** that starts at zero, such
    as the centred Ornstein-Uhlenbeck factor ``x`` in the Hull-White
    decomposition ``r(t) = x(t) + alpha(t)``. Unlike the equity log-spot grid,
    the domain is anchored at the origin rather than at a market quote, so
    there is nothing to detach from autodiff: the read-off always happens at
    ``x = 0`` (a fixed coordinate), and the only differentiable dependence on
    the model parameters enters through ``std_dev``, which sizes the domain.

    Pass the state's terminal standard deviation for ``std_dev`` — for
    Hull-White, ``sqrt(sigma^2/(2a) (1 - e^{-2aT}))`` from
    :func:`~valax.models.hull_white.hw_short_rate_variance`. The truncation
    error decays like the Gaussian tail beyond ``half_width`` std devs, so 6-8
    is ample.

    Endpoints are excluded (Dirichlet / ghost boundary treatment), consistent
    with the other builders in this module.

    Args:
        std_dev: Standard deviation of the state at the horizon.
        n: Number of interior nodes.
        half_width: Half-width of the domain in std-dev units.

    Returns:
        A :class:`Grid1D` of ``n`` interior nodes symmetric about zero.
    """
    w = half_width * std_dev
    return uniform_linear_grid(-w, w, n=n)


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
    return _hermite_read(grid.nodes, values, query)


def _hermite_read(
    xs: Float[Array, " n"],
    values: Float[Array, " n"],
    query: Float[Array, ""],
) -> Float[Array, ""]:
    """4-point non-uniform cubic Hermite interpolation of ``values`` on ``xs``.

    The shared kernel behind :func:`read_off_1d` and each axis of
    :func:`read_off_2d`. Endpoint tangents use the second-order non-uniform
    three-point derivative (exact for quadratics), and values flat-extrapolate
    outside ``[xs[0], xs[-1]]``. Requires ``len(xs) >= 4``.

    Args:
        xs: Strictly increasing node coordinates.
        values: Field values at ``xs``.
        query: Coordinate at which to interpolate.

    Returns:
        The interpolated scalar value.
    """
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


# ─────────────────────────────────────────────────────────────────────
# Two-dimensional (tensor-product) grids
# ─────────────────────────────────────────────────────────────────────


class Grid2D(eqx.Module):
    """A tensor-product 2-D grid of interior nodes.

    The full node set is the Cartesian product ``x.nodes`` :math:`\\times`
    ``y.nodes``; a field is stored as a matrix of shape ``(n_x, n_y)`` with the
    first axis indexing ``x`` (e.g. log-spot) and the second indexing ``y``
    (e.g. variance). Each axis is an independent :class:`Grid1D`, so the 1-D
    operator / boundary / read-off machinery applies per axis unchanged.

    Attributes:
        x: First-axis grid (e.g. log-spot ``x = ln S``).
        y: Second-axis grid (e.g. variance ``v``).
    """

    x: Grid1D
    y: Grid1D

    @property
    def n_x(self) -> int:
        """Number of interior nodes along the first axis."""
        return self.x.n

    @property
    def n_y(self) -> int:
        """Number of interior nodes along the second axis."""
        return self.y.n

    @property
    def shape(self) -> tuple[int, int]:
        """Field shape ``(n_x, n_y)`` for values on this grid."""
        return (self.x.n, self.y.n)


def log_spot_variance_grid(
    spot: Float[Array, ""],
    expiry: Float[Array, ""],
    v0: Float[Array, ""],
    v_max: Float[Array, ""],
    *,
    n_x: int,
    n_y: int,
    x_half_width: float,
    v_scale: Float[Array, ""] | float,
    vol_ref: Float[Array, ""] | None = None,
) -> Grid2D:
    """Build a log-spot :math:`\\times` variance grid for 2-D (Heston) pricing.

    The first axis is a uniform log-spot grid centred on ``ln(spot)`` (via
    :func:`uniform_log_spot_grid`, inheriting its ``stop_gradient`` spot
    detachment so second-order spot Greeks stay well defined). The second axis
    is a :func:`sinh_concentrated_grid` on ``[0, v_max]`` clustered around the
    initial variance ``v0`` — where the value function has the most curvature —
    coarsening toward ``0`` and ``v_max``.

    The variance boundaries ``0`` and ``v_max`` are the *excluded* endpoints
    (consistent with the 1-D interior-node convention); their special treatment
    (the degenerate ``v = 0`` transport row and the ``v = v_max`` linearity
    condition) is applied by the 2-D operator / boundary layer, not here.

    Args:
        spot: Current spot price.
        expiry: Time to expiry (scales the log-spot half-width).
        v0: Initial variance; the variance axis concentrates around it.
        v_max: Upper variance bound (excluded boundary).
        n_x: Interior nodes along log-spot.
        n_y: Interior nodes along variance.
        x_half_width: Log-spot half-width in std-dev units.
        v_scale: ``sinh`` stretching scale for the variance axis (smaller =
            tighter concentration around ``v0``).
        vol_ref: Reference volatility for the log-spot half-width. Defaults to
            ``sqrt(v0)``; pass e.g. ``sqrt(max(v0, theta))`` to size the mesh
            for a mean-reverting variance that drifts away from ``v0``.

    Returns:
        The assembled :class:`Grid2D`.
    """
    ref = jnp.sqrt(v0) if vol_ref is None else vol_ref
    x = uniform_log_spot_grid(spot, ref, expiry, n=n_x, half_width=x_half_width)
    y = sinh_concentrated_grid(
        jnp.asarray(0.0), v_max, v0, n=n_y, scale=v_scale
    )
    return Grid2D(x=x, y=y)


def read_off_2d(
    grid: Grid2D,
    values: Float[Array, "n_x n_y"],
    query_x: Float[Array, ""],
    query_y: Float[Array, ""],
) -> Float[Array, ""]:
    """Interpolate a 2-D field at ``(query_x, query_y)`` (separable bicubic).

    Applies the 1-D non-uniform cubic Hermite kernel (:func:`_hermite_read`)
    tensor-wise: first collapse the variance axis at ``query_y`` for every
    log-spot row, then interpolate the resulting ``n_x``-vector along the
    log-spot axis at ``query_x``. Because each axis carries genuine within-cell
    curvature, ``jax.grad(jax.grad(...))`` recovers second-order Greeks
    (gamma in ``query_x``, and the mixed ``d^2/dx dy`` cross-sensitivity),
    exactly as the 1-D read-off enables PDE gamma.

    Separability makes the read-off exact for products of per-axis quadratics
    (and reproduces every nodal value), which is what the Greek-recovery tests
    rely on.

    Args:
        grid: The tensor-product grid.
        values: Field values of shape ``(n_x, n_y)`` (row = log-spot, column =
            variance).
        query_x: First-axis coordinate (e.g. ``ln(spot)``).
        query_y: Second-axis coordinate (e.g. current variance ``v0``).

    Returns:
        The interpolated scalar value.
    """
    # Collapse the variance axis for each log-spot row -> (n_x,) vector.
    row_values = jax.vmap(
        lambda row: _hermite_read(grid.y.nodes, row, query_y)
    )(values)
    # Interpolate the row values along the log-spot axis.
    return _hermite_read(grid.x.nodes, row_values, query_x)
