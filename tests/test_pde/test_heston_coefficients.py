"""Unit tests for the Heston 2-D operator coefficients (``heston_operator_2d``).

Each Heston coefficient is isolated by applying one split operator to a monomial
whose analytic action is known, evaluated on the interior (edge rows carry the
zero-ghost values handled later by the boundary layer). The three-point
non-uniform stencils are exact on these low-degree fields.
"""

import jax.numpy as jnp

from valax.models.heston import HestonModel
from valax.pricing.pde.coefficients import heston_operator_2d
from valax.pricing.pde.grids import (
    Grid2D,
    sinh_concentrated_grid,
    uniform_linear_grid,
)


def _model():
    return HestonModel(
        v0=jnp.array(0.04),
        kappa=jnp.array(1.5),
        theta=jnp.array(0.04),
        xi=jnp.array(0.5),
        rho=jnp.array(-0.6),
        rate=jnp.array(0.03),
        dividend=jnp.array(0.01),
    )


def _grid(n_x=45, n_y=39):
    x = uniform_linear_grid(jnp.array(-0.6), jnp.array(0.6), n=n_x)
    y = sinh_concentrated_grid(
        jnp.array(0.0), jnp.array(0.6), jnp.array(0.04), n=n_y, scale=0.06
    )
    return Grid2D(x=x, y=y)


def _interior(a):
    return a[1:-1, 1:-1]


def test_a1_matches_heston_log_spot_operator():
    """A1 on V = x^2 recovers 1/2 * v * 2 + (r-q-v/2) * 2x - r/2 * x^2."""
    m, grid = _model(), _grid()
    op = heston_operator_2d(m, grid)
    x = grid.x.nodes[:, None]
    v = grid.y.nodes[None, :]
    field = jnp.broadcast_to(x**2, grid.shape)
    got = _interior(op.apply_a1(field))
    mu = float(m.rate - m.dividend)
    exact = _interior(
        v + (mu - 0.5 * v) * 2.0 * x - 0.5 * float(m.rate) * x**2 + jnp.zeros(grid.shape)
    )
    assert float(jnp.max(jnp.abs(got - exact))) < 1e-8


def test_a1_drift_on_linear_field():
    """A1 on V = x recovers drift_x - r/2 * x = (r-q-v/2) - r/2 * x."""
    m, grid = _model(), _grid()
    op = heston_operator_2d(m, grid)
    x = grid.x.nodes[:, None]
    v = grid.y.nodes[None, :]
    field = jnp.broadcast_to(x, grid.shape)
    got = _interior(op.apply_a1(field))
    mu = float(m.rate - m.dividend)
    exact = _interior((mu - 0.5 * v) - 0.5 * float(m.rate) * x + jnp.zeros(grid.shape))
    assert float(jnp.max(jnp.abs(got - exact))) < 1e-8


def test_a2_matches_heston_variance_operator():
    """A2 on V = v^2 recovers 1/2 * xi^2 v * 2 + kappa(theta-v) * 2v - r/2 * v^2."""
    m, grid = _model(), _grid()
    op = heston_operator_2d(m, grid)
    v = grid.y.nodes[None, :]
    field = jnp.broadcast_to(v**2, grid.shape)
    got = _interior(op.apply_a2(field))
    xi2 = float(m.xi**2)
    exact = _interior(
        xi2 * v
        + float(m.kappa) * (float(m.theta) - v) * 2.0 * v
        - 0.5 * float(m.rate) * v**2
        + jnp.zeros(grid.shape)
    )
    assert float(jnp.max(jnp.abs(got - exact))) < 1e-8


def test_a2_drift_on_linear_field():
    """A2 on V = v recovers kappa(theta - v) - r/2 * v."""
    m, grid = _model(), _grid()
    op = heston_operator_2d(m, grid)
    v = grid.y.nodes[None, :]
    field = jnp.broadcast_to(v, grid.shape)
    got = _interior(op.apply_a2(field))
    exact = _interior(
        float(m.kappa) * (float(m.theta) - v) - 0.5 * float(m.rate) * v
        + jnp.zeros(grid.shape)
    )
    assert float(jnp.max(jnp.abs(got - exact))) < 1e-8


def test_a0_matches_mixed_coefficient():
    """A0 on V = x*v recovers rho*xi*v (since V_xv = 1)."""
    m, grid = _model(), _grid()
    op = heston_operator_2d(m, grid)
    x = grid.x.nodes[:, None]
    v = grid.y.nodes[None, :]
    field = jnp.broadcast_to(x, grid.shape) * jnp.broadcast_to(v, grid.shape)
    got = _interior(op.apply_a0(field))
    exact = _interior(float(m.rho) * float(m.xi) * v + jnp.zeros(grid.shape))
    assert float(jnp.max(jnp.abs(got - exact))) < 1e-8


def test_negative_rho_gives_negative_mixed_coefficient():
    """Sanity: rho<0 (leverage) yields a negative mixed coefficient rho*xi*v."""
    m, grid = _model(), _grid()
    op = heston_operator_2d(m, grid)
    # c0 = rho*xi*v with rho=-0.6, xi=0.5, v>0  =>  strictly negative interior.
    assert bool(jnp.all(_interior(op.c0) < 0.0))
