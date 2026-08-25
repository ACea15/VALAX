"""Tests for the 1-D boundary treatments in :mod:`valax.pricing.pde.boundary`.

Focuses on :func:`apply_linearity_bc_1d`, the zero-curvature (``V_xx = 0``)
ghost fold used by short-rate solvers that have no closed-form far-field value
to pin with Dirichlet data.

The defining property is exactness on **affine** fields: if ``V`` is linear in
``x`` then the linear extrapolation used to build the exterior ghost is not an
approximation at all — it is the true value — so the folded operator must
reproduce ``L V = mu V_x - r V`` *exactly* at the two edge rows, with no
boundary data supplied. That is a much sharper statement than "the price looks
about right", and it is what lets the fold stand in for Dirichlet data.
"""

import jax
import jax.numpy as jnp
import pytest

from valax.pricing.pde.boundary import apply_linearity_bc_1d, zero_boundary
from valax.pricing.pde.grids import (
    Grid1D,
    centred_state_grid,
    sinh_concentrated_grid,
    uniform_linear_grid,
)
from valax.pricing.pde.linalg import tridiagonal_matvec
from valax.pricing.pde.operators import build_operator_1d

DRIFT = 0.35
DIFFUSION = 0.04
DISCOUNT = 0.03


def _apply(operator, values):
    """Act with the tridiagonal operator on ``values`` (no boundary term)."""
    return tridiagonal_matvec(
        operator.lower[1:], operator.diag, operator.upper[:-1], values
    )


@pytest.fixture(params=["uniform", "concentrated"])
def grid(request) -> Grid1D:
    if request.param == "uniform":
        return uniform_linear_grid(jnp.array(-2.0), jnp.array(3.0), n=24)
    # Non-uniform spacing exercises the ghost-extrapolation ratios, which are
    # 1 on a uniform mesh and would hide an indexing slip.
    return sinh_concentrated_grid(
        jnp.array(-2.0), jnp.array(3.0), jnp.array(0.5), n=24, scale=1.0
    )


@pytest.fixture
def operator(grid):
    return build_operator_1d(
        grid,
        drift=jnp.array(DRIFT),
        diffusion=jnp.array(DIFFUSION),
        discount=jnp.array(DISCOUNT),
    )


class TestApplyLinearityBC1D:
    def test_edge_bands_are_zeroed(self, grid, operator):
        """The exterior couplings are folded away, so no ghost is referenced."""
        folded = apply_linearity_bc_1d(operator, grid)
        assert float(folded.lower[0]) == 0.0
        assert float(folded.upper[-1]) == 0.0

    def test_interior_rows_untouched(self, grid, operator):
        folded = apply_linearity_bc_1d(operator, grid)
        assert jnp.allclose(folded.lower[2:-1], operator.lower[2:-1], atol=0)
        assert jnp.allclose(folded.diag[1:-1], operator.diag[1:-1], atol=0)
        assert jnp.allclose(folded.upper[1:-2], operator.upper[1:-2], atol=0)

    @pytest.mark.parametrize("slope,intercept", [(0.0, 1.0), (1.0, 0.0), (-0.7, 2.5)])
    def test_exact_on_affine_fields(self, grid, operator, slope, intercept):
        """``L V = mu V_x - r V`` exactly, edges included, for affine ``V``."""
        values = intercept + slope * grid.nodes
        folded = apply_linearity_bc_1d(operator, grid)
        expected = DRIFT * slope - DISCOUNT * values
        assert float(jnp.max(jnp.abs(_apply(folded, values) - expected))) < 1e-12

    def test_interior_keeps_its_curvature(self, grid, operator):
        """A quadratic keeps the full ``1/2 sigma^2 V_xx`` term inside."""
        values = grid.nodes**2
        folded = apply_linearity_bc_1d(operator, grid)
        got = _apply(folded, values)
        interior = DIFFUSION + DRIFT * 2.0 * grid.nodes - DISCOUNT * values
        assert jnp.allclose(got[1:-1], interior[1:-1], atol=1e-10)

    def test_edge_rows_are_zero_curvature_and_one_sided(self, grid, operator):
        """Pins *exactly* what the fold does to the two edge rows.

        Zero curvature and a one-sided convection stencil: the forward
        difference at the lower edge, the backward difference at the upper.
        For ``V = x^2`` those evaluate to ``x_0 + x_1`` and
        ``x_{n-1} + x_{n-2}`` respectively.
        """
        x = grid.nodes
        values = x**2
        got = _apply(apply_linearity_bc_1d(operator, grid), values)

        lower_edge = DRIFT * float(x[0] + x[1]) - DISCOUNT * float(values[0])
        upper_edge = DRIFT * float(x[-1] + x[-2]) - DISCOUNT * float(values[-1])
        assert float(got[0]) == pytest.approx(lower_edge, abs=1e-12)
        assert float(got[-1]) == pytest.approx(upper_edge, abs=1e-12)

    def test_stacked_bands_fold_per_row(self, grid):
        """A time-dependent (stacked) operator folds along the trailing axis."""
        discounts = jnp.linspace(0.01, 0.05, 5)
        n_time = discounts.shape[0]
        stacked = jax.vmap(
            lambda r: build_operator_1d(
                grid,
                drift=jnp.array(DRIFT),
                diffusion=jnp.array(DIFFUSION),
                discount=r,
            )
        )(discounts)
        folded = apply_linearity_bc_1d(stacked, grid)
        assert folded.lower.shape == (n_time, grid.n)
        assert jnp.allclose(folded.lower[:, 0], 0.0)
        assert jnp.allclose(folded.upper[:, -1], 0.0)

        # Each row must independently be exact on an affine field.
        values = 2.5 - 0.7 * grid.nodes
        for m in range(n_time):
            row = tridiagonal_matvec(
                folded.lower[m][1:], folded.diag[m], folded.upper[m][:-1], values
            )
            expected = DRIFT * (-0.7) - discounts[m] * values
            assert float(jnp.max(jnp.abs(row - expected))) < 1e-12

    def test_matches_an_explicitly_extrapolated_ghost(self, grid, operator):
        """Cross-check: folding must equal supplying the extrapolated ghost.

        Independent route to the same numbers — build the *unfolded* operator's
        action by hand with a linearly extrapolated exterior value and compare.
        """
        x = grid.nodes
        values = jnp.exp(0.4 * x)  # generic curved field

        ratio_lo = float((x[0] - (x[0] - (x[1] - x[0]))) / (x[1] - x[0]))
        ratio_hi = float(((x[-1] + (x[-1] - x[-2])) - x[-1]) / (x[-1] - x[-2]))
        ghost_lo = values[0] - ratio_lo * (values[1] - values[0])
        ghost_hi = values[-1] + ratio_hi * (values[-1] - values[-2])

        manual = _apply(operator, values)
        manual = manual.at[0].add(operator.lower[0] * ghost_lo)
        manual = manual.at[-1].add(operator.upper[-1] * ghost_hi)

        folded = _apply(apply_linearity_bc_1d(operator, grid), values)
        assert float(jnp.max(jnp.abs(folded - manual))) < 1e-12


class TestZeroBoundary:
    def test_returns_zero_for_any_tau(self):
        b = zero_boundary()
        for tau in (0.0, 0.5, 7.0):
            assert float(b.lower_fn(jnp.array(tau))) == 0.0
            assert float(b.upper_fn(jnp.array(tau))) == 0.0


class TestCentredStateGrid:
    def test_symmetric_about_zero(self):
        grid = centred_state_grid(jnp.array(0.02), n=21, half_width=6.0)
        assert float(jnp.max(jnp.abs(grid.nodes + grid.nodes[::-1]))) < 1e-15

    def test_odd_node_count_straddles_the_origin(self):
        """An odd interior count puts a node exactly at x = 0, where the
        price is read off — worth knowing when choosing ``n_spot``."""
        grid = centred_state_grid(jnp.array(0.02), n=21, half_width=6.0)
        assert float(jnp.min(jnp.abs(grid.nodes))) < 1e-15

    def test_span_matches_requested_half_width(self):
        std, half = 0.02, 6.0
        grid = centred_state_grid(jnp.array(std), n=200, half_width=half)
        # Interior nodes exclude the endpoints, so the span is just inside.
        assert float(grid.nodes[-1]) < half * std
        assert float(grid.nodes[-1]) > half * std * 0.98
