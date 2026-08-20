"""Unit tests for the 2-D ADI operator split (:mod:`valax.pricing.pde.operators2d`).

Each split operator is checked against a closed-form action on a known field,
evaluated on the *interior* only (edge rows use zero-ghost values that the
boundary layer / ADI stepper corrects later). Fields are chosen so the
three-point non-uniform stencils are exact:

- ``A1`` on ``V(x, v) = x^2``  ->  ``1/2 c_xx * 2 + c_x * 2x - 1/2 r x^2``.
- ``A2`` on ``V(x, v) = v^2``  ->  ``1/2 c_vv * 2 + c_v * 2v - 1/2 r v^2``.
- ``A0`` on ``V(x, v) = x v``  ->  ``c_xv`` (since ``V_xv = 1``).
"""

import jax.numpy as jnp
import pytest

from valax.pricing.pde.grids import (
    Grid2D,
    sinh_concentrated_grid,
    uniform_linear_grid,
)
from valax.pricing.pde.operators2d import build_operator_2d


def _grid(n_x=41, n_y=37):
    """A deliberately non-uniform tensor-product grid (concentrated v axis)."""
    x = uniform_linear_grid(jnp.array(-1.5), jnp.array(1.5), n=n_x)
    y = sinh_concentrated_grid(
        jnp.array(0.0), jnp.array(1.0), jnp.array(0.25), n=n_y, scale=0.3
    )
    return Grid2D(x=x, y=y)


def _interior(a):
    """Strip the one-cell border where zero-ghost edge rows live."""
    return a[1:-1, 1:-1]


def test_a1_action_on_x_squared():
    grid = _grid()
    c_xx, c_x, r = 0.7, 0.2, 0.05
    op = build_operator_2d(
        grid,
        diff_x=jnp.array(c_xx),
        drift_x=jnp.array(c_x),
        diff_v=jnp.array(0.0),
        drift_v=jnp.array(0.0),
        mixed=jnp.array(0.0),
        discount=jnp.array(r),
    )
    x = grid.x.nodes[:, None]
    v = jnp.broadcast_to(x**2, grid.shape)  # V = x^2, constant in v
    got = _interior(op.apply_a1(v))
    exact = _interior(
        0.5 * c_xx * 2.0 + c_x * 2.0 * x - 0.5 * r * x**2 + jnp.zeros(grid.shape)
    )
    assert float(jnp.max(jnp.abs(got - exact))) < 1e-8


def test_a2_action_on_v_squared():
    grid = _grid()
    c_vv, c_v, r = 0.9, 0.3, 0.05
    op = build_operator_2d(
        grid,
        diff_x=jnp.array(0.0),
        drift_x=jnp.array(0.0),
        diff_v=jnp.array(c_vv),
        drift_v=jnp.array(c_v),
        mixed=jnp.array(0.0),
        discount=jnp.array(r),
    )
    vv = grid.y.nodes[None, :]
    v = jnp.broadcast_to(vv**2, grid.shape)  # V = v^2, constant in x
    got = _interior(op.apply_a2(v))
    exact = _interior(
        0.5 * c_vv * 2.0 + c_v * 2.0 * vv - 0.5 * r * vv**2 + jnp.zeros(grid.shape)
    )
    assert float(jnp.max(jnp.abs(got - exact))) < 1e-8


def test_a0_action_on_bilinear():
    grid = _grid()
    # Per-node mixed coefficient rho*sigma*v -> here just an arbitrary field.
    xx = grid.x.nodes[:, None]
    vv = grid.y.nodes[None, :]
    c_xv = 0.4 * vv + 0.1 * xx  # arbitrary smooth per-node coefficient
    op = build_operator_2d(
        grid,
        diff_x=jnp.array(0.0),
        drift_x=jnp.array(0.0),
        diff_v=jnp.array(0.0),
        drift_v=jnp.array(0.0),
        mixed=c_xv,
        discount=jnp.array(0.0),
    )
    v_field = jnp.broadcast_to(xx, grid.shape) * jnp.broadcast_to(vv, grid.shape)
    # V = x*v  =>  V_xv = 1  =>  A0 V = c_xv.
    got = _interior(op.apply_a0(v_field))
    exact = _interior(jnp.broadcast_to(c_xv, grid.shape))
    assert float(jnp.max(jnp.abs(got - exact))) < 1e-8


def test_a0_zero_for_separable_x_only():
    """A field independent of v has zero mixed derivative."""
    grid = _grid()
    op = build_operator_2d(
        grid,
        diff_x=jnp.array(0.0),
        drift_x=jnp.array(0.0),
        diff_v=jnp.array(0.0),
        drift_v=jnp.array(0.0),
        mixed=jnp.array(1.0),
        discount=jnp.array(0.0),
    )
    x = grid.x.nodes[:, None]
    v_field = jnp.broadcast_to(jnp.sin(x), grid.shape)  # constant in v
    got = _interior(op.apply_a0(v_field))
    assert float(jnp.max(jnp.abs(got))) < 1e-10


def test_apply_is_sum_of_splits():
    grid = _grid()
    xx = grid.x.nodes[:, None]
    vv = grid.y.nodes[None, :]
    op = build_operator_2d(
        grid,
        diff_x=jnp.broadcast_to(vv, grid.shape),
        drift_x=jnp.array(0.03) - 0.5 * jnp.broadcast_to(vv, grid.shape),
        diff_v=0.04 * jnp.broadcast_to(vv, grid.shape),
        drift_v=1.5 * (0.04 - jnp.broadcast_to(vv, grid.shape)),
        mixed=-0.6 * 0.3 * jnp.broadcast_to(vv, grid.shape),
        discount=jnp.array(0.03),
    )
    v_field = jnp.exp(0.3 * xx) * (1.0 + vv)
    v_field = jnp.broadcast_to(v_field, grid.shape)
    total = op.apply(v_field)
    parts = op.apply_a0(v_field) + op.apply_a1(v_field) + op.apply_a2(v_field)
    assert float(jnp.max(jnp.abs(total - parts))) < 1e-12


def test_mixed_term_is_zero_on_boundaries():
    """A0 must vanish on all four domain edges (in't Hout ADI treatment).

    A one-sided cross stencil at the boundary injects spurious flux that
    otherwise destroys convergence -- this guards that regression.
    """
    grid = _grid()
    op = build_operator_2d(
        grid,
        diff_x=jnp.array(0.0),
        drift_x=jnp.array(0.0),
        diff_v=jnp.array(0.0),
        drift_v=jnp.array(0.0),
        mixed=jnp.array(2.5),  # non-zero everywhere before boundary zeroing
        discount=jnp.array(0.0),
    )
    assert bool(jnp.all(op.c0[0, :] == 0.0))
    assert bool(jnp.all(op.c0[-1, :] == 0.0))
    assert bool(jnp.all(op.c0[:, 0] == 0.0))
    assert bool(jnp.all(op.c0[:, -1] == 0.0))
    # Interior retains the coefficient.
    assert bool(jnp.all(op.c0[1:-1, 1:-1] == 2.5))


def test_bands_have_field_shape():
    grid = _grid(n_x=33, n_y=29)
    op = build_operator_2d(
        grid,
        diff_x=jnp.array(0.1),
        drift_x=jnp.array(0.0),
        diff_v=jnp.array(0.1),
        drift_v=jnp.array(0.0),
        mixed=jnp.array(0.0),
        discount=jnp.array(0.0),
    )
    assert op.a1_lower.shape == (33, 29)
    assert op.a2_upper.shape == (33, 29)
    assert op.c0.shape == (33, 29)
    assert op.sx_lower.shape == (33,)
    assert op.sv_lower.shape == (29,)
