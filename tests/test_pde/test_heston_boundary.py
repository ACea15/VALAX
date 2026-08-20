"""Unit tests for the 2-D Heston boundary conditions.

Covers the log-spot Dirichlet asymptotics (:func:`heston_boundary`) and the
variance-axis operator surgery (:func:`apply_heston_variance_bc`): the
degenerate ``v -> 0`` drift-only upwind row and the ``v = v_max`` linearity row,
including a Feller-violating parameter set (``2 kappa theta < xi^2``).
"""

import jax.numpy as jnp

from valax.models.heston import HestonModel
from valax.pricing.pde.boundary import (
    apply_heston_variance_bc,
    heston_boundary,
)
from valax.pricing.pde.coefficients import heston_operator_2d
from valax.pricing.pde.grids import (
    Grid2D,
    boundary_coords,
    sinh_concentrated_grid,
    uniform_linear_grid,
    uniform_log_spot_grid,
)


def _model(kappa=1.5, theta=0.04, xi=0.5):
    return HestonModel(
        v0=jnp.array(0.04),
        kappa=jnp.array(kappa),
        theta=jnp.array(theta),
        xi=jnp.array(xi),
        rho=jnp.array(-0.6),
        rate=jnp.array(0.03),
        dividend=jnp.array(0.01),
    )


def _grid(spot=100.0, n_x=41, n_y=39, v_max=0.6):
    x = uniform_log_spot_grid(
        jnp.array(spot), jnp.array(0.2), jnp.array(1.0), n=n_x, half_width=4.0
    )
    y = sinh_concentrated_grid(
        jnp.array(0.0), jnp.array(v_max), jnp.array(0.04), n=n_y, scale=0.06
    )
    return Grid2D(x=x, y=y)


# ── Log-spot Dirichlet asymptotics ───────────────────────────────────


def test_heston_boundary_call_asymptotics():
    grid = _grid()
    m = _model()
    bnd = heston_boundary(grid, jnp.array(100.0), m.rate, m.dividend, is_call=True)
    tau = jnp.array(0.5)
    x_min, x_max = boundary_coords(grid.x)
    s_hi = jnp.exp(x_max)
    # Call: worthless at S->0, discounted forward minus discounted strike at S->inf.
    assert abs(float(bnd.x_lower_fn(tau))) < 1e-12
    expected_hi = float(
        s_hi * jnp.exp(-m.dividend * tau) - 100.0 * jnp.exp(-m.rate * tau)
    )
    assert abs(float(bnd.x_upper_fn(tau)) - expected_hi) < 1e-9


def test_heston_boundary_put_asymptotics():
    grid = _grid()
    m = _model()
    bnd = heston_boundary(grid, jnp.array(100.0), m.rate, m.dividend, is_call=False)
    tau = jnp.array(0.5)
    x_min, _ = boundary_coords(grid.x)
    s_lo = jnp.exp(x_min)
    expected_lo = float(
        100.0 * jnp.exp(-m.rate * tau) - s_lo * jnp.exp(-m.dividend * tau)
    )
    assert abs(float(bnd.x_lower_fn(tau)) - expected_lo) < 1e-9
    assert abs(float(bnd.x_upper_fn(tau))) < 1e-12


# ── Variance-axis operator surgery ───────────────────────────────────


def test_low_variance_row_is_drift_only_upwind():
    grid = _grid()
    m = _model()
    op = apply_heston_variance_bc(heston_operator_2d(m, grid), grid, m)
    # No coupling to the sub-zero ghost.
    assert bool(jnp.all(op.a2_lower[:, 0] == 0.0))

    # A2 on a field linear in v: (A2 V)[i,0] = drift0 * slope - r/2 * V[i,0].
    v = grid.y.nodes
    a, b = 0.7, 1.3
    field = jnp.broadcast_to(a + b * v[None, :], grid.shape)
    got = op.apply_a2(field)[:, 0]
    drift0 = float(m.kappa) * (float(m.theta) - float(v[0]))
    exact = drift0 * b - 0.5 * float(m.rate) * (a + b * float(v[0]))
    assert float(jnp.max(jnp.abs(got - exact))) < 1e-9


def test_high_variance_row_linearity():
    grid = _grid()
    m = _model()
    op = apply_heston_variance_bc(heston_operator_2d(m, grid), grid, m)
    # No coupling to the exterior ghost after the linearity fold.
    assert bool(jnp.all(op.a2_upper[:, -1] == 0.0))

    # A2 on a field linear in v at the last row: V_vv = 0 so only drift survives:
    # (A2 V)[i,-1] = kappa(theta - v_max) * slope - r/2 * V[i,-1].
    v = grid.y.nodes
    a, b = -0.4, 0.9
    field = jnp.broadcast_to(a + b * v[None, :], grid.shape)
    got = op.apply_a2(field)[:, -1]
    drift_hi = float(m.kappa) * (float(m.theta) - float(v[-1]))
    exact = drift_hi * b - 0.5 * float(m.rate) * (a + b * float(v[-1]))
    assert float(jnp.max(jnp.abs(got - exact))) < 1e-9


def test_interior_variance_rows_unchanged():
    grid = _grid()
    m = _model()
    raw = heston_operator_2d(m, grid)
    op = apply_heston_variance_bc(raw, grid, m)
    # Only rows 0 and -1 of A2 are rewritten; the interior is identical.
    assert bool(jnp.allclose(op.a2_lower[:, 1:-1], raw.a2_lower[:, 1:-1]))
    assert bool(jnp.allclose(op.a2_diag[:, 1:-1], raw.a2_diag[:, 1:-1]))
    assert bool(jnp.allclose(op.a2_upper[:, 1:-1], raw.a2_upper[:, 1:-1]))
    # A1 / A0 untouched.
    assert bool(jnp.allclose(op.a1_diag, raw.a1_diag))
    assert bool(jnp.allclose(op.c0, raw.c0))


def test_feller_violating_boundary_is_finite_and_drift_only():
    """With 2 kappa theta < xi^2 the v=0 row must stay well posed (finite)."""
    # 2*kappa*theta = 2*1.0*0.04 = 0.08 < xi^2 = 0.9^2 = 0.81  -> Feller violated.
    m = _model(kappa=1.0, theta=0.04, xi=0.9)
    assert 2.0 * float(m.kappa) * float(m.theta) < float(m.xi) ** 2
    grid = _grid()
    op = apply_heston_variance_bc(heston_operator_2d(m, grid), grid, m)
    for band in (op.a2_lower, op.a2_diag, op.a2_upper):
        assert bool(jnp.all(jnp.isfinite(band)))
    # The degenerate row remains drift-only (no diffusion coupling below).
    assert bool(jnp.all(op.a2_lower[:, 0] == 0.0))
    # Upwind drift points inward (kappa*theta>0): super-diagonal coeff positive.
    assert bool(jnp.all(op.a2_upper[:, 0] > 0.0))
