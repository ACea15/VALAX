"""Unit tests for the spatial grid builders and the price read-off.

Focus on :func:`read_off_1d`: it must reproduce nodal values, extrapolate
flat outside the grid, stay accurate on a smooth field, and -- unlike the old
piecewise-linear ``jnp.interp`` -- carry genuine *curvature* so that a second
derivative w.r.t. the query is non-zero (this is what enables PDE gamma).
"""

import jax
import jax.numpy as jnp

from valax.pricing.pde.grids import (
    Grid1D,
    read_off_1d,
    sinh_concentrated_grid,
    uniform_linear_grid,
    uniform_log_spot_grid,
)
from valax.pricing.pde.operators import build_operator_1d


def _grid(n=51, lo=-1.0, hi=1.0):
    nodes = jnp.linspace(lo, hi, n)
    return Grid1D(nodes=nodes, n=n)


def test_read_off_reproduces_node_values():
    grid = _grid()
    values = jnp.sin(grid.nodes)
    for k in (1, 10, 25, 40, grid.n - 2):
        got = float(read_off_1d(grid, values, grid.nodes[k]))
        assert abs(got - float(values[k])) < 1e-6


def test_read_off_flat_extrapolation():
    grid = _grid()
    values = jnp.cos(grid.nodes)
    below = float(read_off_1d(grid, values, jnp.array(-5.0)))
    above = float(read_off_1d(grid, values, jnp.array(5.0)))
    assert abs(below - float(values[0])) < 1e-12
    assert abs(above - float(values[-1])) < 1e-12


def test_read_off_accurate_on_smooth_field():
    grid = _grid(n=201)
    values = jnp.exp(grid.nodes)  # smooth reference
    q = jnp.array(0.123456)
    got = float(read_off_1d(grid, values, q))
    assert abs(got - float(jnp.exp(q))) < 1e-5


def test_read_off_has_curvature():
    """Second derivative of a quadratic field is recovered (was 0 for linear)."""
    grid = _grid(n=201)
    values = grid.nodes**2  # f(x)=x^2  ->  f''=2
    f = lambda x: read_off_1d(grid, values, x)
    second = float(jax.grad(jax.grad(f))(jnp.array(0.3)))
    assert abs(second - 2.0) < 1e-3


def test_read_off_first_derivative():
    grid = _grid(n=201)
    values = jnp.sin(grid.nodes)
    f = lambda x: read_off_1d(grid, values, x)
    d = float(jax.grad(f)(jnp.array(0.4)))
    assert abs(d - float(jnp.cos(jnp.array(0.4)))) < 1e-3


def test_log_spot_grid_detaches_spot_but_keeps_center():
    """The grid must be *centred* on spot yet carry no spot-gradient (scaffold)."""
    spot = jnp.array(100.0)
    vol, expiry = jnp.array(0.2), jnp.array(1.0)

    def center(s):
        g = uniform_log_spot_grid(s, vol, expiry, n=50, half_width=4.0)
        # midpoint of the interior nodes ~ ln(spot)
        return 0.5 * (g.nodes[g.n // 2 - 1] + g.nodes[g.n // 2])

    # Forward: centred on ln(spot).
    assert abs(float(center(spot)) - float(jnp.log(spot))) < 0.05
    # Backward: node positions do NOT co-move with spot under autodiff.
    assert float(jax.grad(center)(spot)) == 0.0


# ── Non-uniform (sinh-concentrated) grids ────────────────────────────


def test_sinh_grid_is_increasing_and_concentrated():
    lo, hi, c = jnp.array(-3.0), jnp.array(3.0), jnp.array(0.0)
    grid = sinh_concentrated_grid(lo, hi, c, n=101, scale=0.5)
    gaps = jnp.diff(grid.nodes)
    # Strictly increasing nodes.
    assert bool(jnp.all(gaps > 0))
    # Resolution is finest near the centre, coarsest at the edges.
    mid = grid.n // 2
    assert float(gaps[mid]) < float(gaps[0])
    assert float(gaps[mid]) < float(gaps[-1])
    # Interior nodes stay strictly inside (lo, hi).
    assert float(grid.nodes[0]) > float(lo)
    assert float(grid.nodes[-1]) < float(hi)


def test_sinh_grid_approaches_uniform_for_large_scale():
    lo, hi, c = jnp.array(-1.0), jnp.array(1.0), jnp.array(0.0)
    grid = sinh_concentrated_grid(lo, hi, c, n=50, scale=1e6)
    gaps = jnp.diff(grid.nodes)
    # As scale -> inf the sinh map is locally linear => (near) uniform spacing.
    assert float(jnp.max(gaps) - jnp.min(gaps)) < 1e-6


def test_read_off_accurate_on_concentrated_grid():
    grid = sinh_concentrated_grid(
        jnp.array(-2.0), jnp.array(2.0), jnp.array(0.0), n=201, scale=0.4
    )
    values = jnp.exp(grid.nodes)
    q = jnp.array(0.137)
    got = float(read_off_1d(grid, values, q))
    assert abs(got - float(jnp.exp(q))) < 1e-4


def test_read_off_curvature_on_concentrated_grid():
    """Non-uniform Hermite reproduces a quadratic => f'' = 2 recovered."""
    grid = sinh_concentrated_grid(
        jnp.array(-2.0), jnp.array(2.0), jnp.array(0.0), n=201, scale=0.4
    )
    values = grid.nodes**2
    f = lambda x: read_off_1d(grid, values, x)
    second = float(jax.grad(jax.grad(f))(jnp.array(0.2)))
    assert abs(second - 2.0) < 1e-3


# ── Non-uniform central-difference operator ──────────────────────────


def test_uniform_operator_matches_closed_form():
    """On a uniform grid the non-uniform stencil reduces to sigma^2/(2 dx^2)."""
    grid = uniform_linear_grid(jnp.array(-1.0), jnp.array(1.0), n=50)
    op = build_operator_1d(
        grid, drift=jnp.array(0.3), diffusion=jnp.array(0.8), discount=jnp.array(0.05)
    )
    dx = float(grid.nodes[1] - grid.nodes[0])
    lower = 0.8 / (2.0 * dx**2) - 0.3 / (2.0 * dx)
    diag = -0.8 / dx**2 - 0.05
    upper = 0.8 / (2.0 * dx**2) + 0.3 / (2.0 * dx)
    assert bool(jnp.allclose(op.lower, lower, atol=1e-9))
    assert bool(jnp.allclose(op.diag, diag, atol=1e-9))
    assert bool(jnp.allclose(op.upper, upper, atol=1e-9))


def test_nonuniform_operator_exact_on_quadratic():
    """The 3-point non-uniform stencil is exact for V(x) = x^2.

    For ``L V = 1/2 sigma^2 V_xx + mu V_x - r V`` with V = x^2 the exact value is
    ``sigma^2 + 2 mu x - r x^2``; the interior rows must reproduce it to fp.
    """
    grid = sinh_concentrated_grid(
        jnp.array(-2.0), jnp.array(2.0), jnp.array(0.0), n=61, scale=0.5
    )
    mu, sig2, r = 0.3, 0.8, 0.05
    op = build_operator_1d(
        grid, drift=jnp.array(mu), diffusion=jnp.array(sig2), discount=jnp.array(r)
    )
    x = grid.nodes
    v = x**2
    # Interior action (rows 1..n-2): excludes the Dirichlet-coupled edge rows.
    lv = op.lower[1:-1] * v[:-2] + op.diag[1:-1] * v[1:-1] + op.upper[1:-1] * v[2:]
    exact = sig2 + 2.0 * mu * x[1:-1] - r * x[1:-1] ** 2
    assert float(jnp.max(jnp.abs(lv - exact))) < 1e-8
