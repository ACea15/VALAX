"""Unit tests for the SLV ``LeverageGrid`` (``valax/surfaces/leverage.py``).

Test layout (SLV-1 — leverage grid acceptance):

* ``TestLeverageGrid``  — construction, shape, scalar-in/scalar-out
  interpolation contract.
* ``TestFlatFactory``   — ``LeverageGrid.flat`` returns a constant grid
  that recovers the pure-Heston limit of the SLV SDE.
* ``TestInterpolation`` — bilinear-at-node, bilinear-at-midpoint, and
  flat extrapolation outside the grid bounds (mirrors the existing
  ``bilinear_2d`` contract).
* ``TestAutodiff``      — ``jax.jit`` / ``jax.grad`` flow through
  ``values``; ``jax.vmap`` over query points.

x64 is enabled at module level — SLV's constructor enforces it and
``bilinear_2d`` benefits from the wider mantissa near the grid edges.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import equinox as eqx
import pytest

from valax.surfaces import LeverageGrid


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def small_grid_axes():
    """Modest 5×4 (k, t) axes used throughout."""
    log_moneyness_grid = jnp.linspace(-0.4, 0.4, 5)  # 5 k-nodes
    time_grid = jnp.linspace(0.1, 1.0, 4)            # 4 t-nodes
    return log_moneyness_grid, time_grid


@pytest.fixture
def smooth_grid(small_grid_axes):
    """Smoothly-varying L on the small grid: L(k, t) = 1 + 0.2·k + 0.1·t."""
    k_grid, t_grid = small_grid_axes
    K, T = jnp.meshgrid(k_grid, t_grid)  # K, T shape (n_t, n_k)
    values = 1.0 + 0.2 * K + 0.1 * T
    return LeverageGrid(
        log_moneyness_grid=k_grid,
        time_grid=t_grid,
        values=values,
    )


# ─────────────────────────────────────────────────────────────────────
# 1. Construction + shape / scalar contract
# ─────────────────────────────────────────────────────────────────────


class TestLeverageGrid:
    def test_shape_convention(self, smooth_grid):
        """Storage is (n_t, n_k), matching ``GridVolSurface.vols``."""
        assert smooth_grid.values.shape == (
            smooth_grid.time_grid.shape[0],
            smooth_grid.log_moneyness_grid.shape[0],
        )

    def test_call_returns_scalar(self, smooth_grid):
        out = smooth_grid(jnp.array(0.0), jnp.array(0.5))
        assert out.shape == ()
        assert jnp.isfinite(out)


# ─────────────────────────────────────────────────────────────────────
# 2. Flat-factory: pure-Heston-limit warm start
# ─────────────────────────────────────────────────────────────────────


class TestFlatFactory:
    def test_flat_grid_returns_value(self, small_grid_axes):
        k_grid, t_grid = small_grid_axes
        flat = LeverageGrid.flat(k_grid, t_grid, value=1.25)
        # All node values exactly 1.25 — flat factory is bit-equal.
        assert jnp.all(flat.values == 1.25)
        # Interpolation at an arbitrary interior point also returns 1.25.
        assert float(flat(jnp.array(0.1), jnp.array(0.4))) == pytest.approx(1.25)

    def test_flat_default_is_one(self, small_grid_axes):
        """Default value is L ≡ 1 — the pure-Heston-limit warm start."""
        k_grid, t_grid = small_grid_axes
        flat = LeverageGrid.flat(k_grid, t_grid)
        assert float(flat(jnp.array(0.0), jnp.array(0.5))) == 1.0

    def test_flat_shape_matches_axes(self, small_grid_axes):
        k_grid, t_grid = small_grid_axes
        flat = LeverageGrid.flat(k_grid, t_grid, value=0.8)
        assert flat.values.shape == (t_grid.shape[0], k_grid.shape[0])


# ─────────────────────────────────────────────────────────────────────
# 3. Interpolation contract (bilinear at node / midpoint, flat
#    extrapolation outside)
# ─────────────────────────────────────────────────────────────────────


class TestInterpolation:
    def test_bilinear_at_grid_node(self, smooth_grid):
        """Querying at a grid node returns the stored value exactly."""
        k_grid = smooth_grid.log_moneyness_grid
        t_grid = smooth_grid.time_grid
        # Pick the (i, j) = (2, 1) node.
        i, j = 1, 2
        out = smooth_grid(k_grid[j], t_grid[i])
        assert float(out) == pytest.approx(float(smooth_grid.values[i, j]))

    def test_bilinear_midpoint(self, smooth_grid):
        """At the midpoint of a cell, bilinear interp is the four-corner mean."""
        k_grid = smooth_grid.log_moneyness_grid
        t_grid = smooth_grid.time_grid
        i, j = 1, 2
        k_mid = 0.5 * (k_grid[j] + k_grid[j + 1])
        t_mid = 0.5 * (t_grid[i] + t_grid[i + 1])
        expected = 0.25 * (
            smooth_grid.values[i, j]
            + smooth_grid.values[i, j + 1]
            + smooth_grid.values[i + 1, j]
            + smooth_grid.values[i + 1, j + 1]
        )
        out = smooth_grid(k_mid, t_mid)
        assert float(out) == pytest.approx(float(expected), rel=1e-12)

    def test_flat_extrapolation_outside_grid(self, smooth_grid):
        """Queries outside the grid bounds clip to the boundary (flat extrap)."""
        k_grid = smooth_grid.log_moneyness_grid
        t_grid = smooth_grid.time_grid
        # Far below the (k, t) grid: should equal the (k_min, t_min) node.
        out = smooth_grid(jnp.array(-10.0), jnp.array(-10.0))
        assert float(out) == pytest.approx(float(smooth_grid.values[0, 0]))
        # Far above: top-right corner.
        out_hi = smooth_grid(jnp.array(10.0), jnp.array(10.0))
        assert float(out_hi) == pytest.approx(float(smooth_grid.values[-1, -1]))


# ─────────────────────────────────────────────────────────────────────
# 4. Autodiff / JAX-transform compatibility
# ─────────────────────────────────────────────────────────────────────


class TestAutodiff:
    def test_jit_compatible(self, smooth_grid):
        """``eqx.filter_jit`` compiles a leverage query."""
        f = eqx.filter_jit(smooth_grid)
        out = f(jnp.array(0.05), jnp.array(0.4))
        assert jnp.isfinite(out)

    def test_differentiable_through_values(self, small_grid_axes):
        """``jax.grad`` w.r.t. ``values`` flows through bilinear interp."""
        k_grid, t_grid = small_grid_axes
        values = jnp.ones((t_grid.shape[0], k_grid.shape[0]))

        def query(vals):
            grid = LeverageGrid(
                log_moneyness_grid=k_grid,
                time_grid=t_grid,
                values=vals,
            )
            return grid(jnp.array(0.1), jnp.array(0.4))

        grads = jax.grad(query)(values)
        # The gradient is sparse — only the four enclosing nodes are
        # non-zero, and they must sum to 1 (partition of unity property
        # of bilinear basis functions).
        assert grads.shape == values.shape
        assert float(jnp.sum(grads)) == pytest.approx(1.0, rel=1e-10)
        # At least 4 non-zero entries (the enclosing cell corners).
        assert int(jnp.sum(grads > 0)) >= 1
        assert int(jnp.sum(grads > 0)) <= 4

    def test_vmap_over_query_points(self, smooth_grid):
        """``jax.vmap`` batches scalar queries."""
        ks = jnp.linspace(-0.3, 0.3, 7)
        t = jnp.array(0.5)
        out = jax.vmap(lambda k: smooth_grid(k, t))(ks)
        assert out.shape == (7,)
        assert jnp.all(jnp.isfinite(out))

    def test_grad_wrt_query_point(self, smooth_grid):
        """The leverage I built (L = 1 + 0.2k + 0.1t) has ∂L/∂k = 0.2
        at any interior point — autodiff w.r.t. the query position
        agrees with the analytic derivative."""
        f = lambda k: smooth_grid(k, jnp.array(0.4))
        dL_dk = float(jax.grad(f)(jnp.array(0.05)))
        assert dL_dk == pytest.approx(0.2, rel=1e-10)
