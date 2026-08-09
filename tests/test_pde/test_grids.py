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
    uniform_log_spot_grid,
)


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
