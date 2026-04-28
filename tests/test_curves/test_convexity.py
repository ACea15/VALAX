"""Tests for convexity adjustment plug-ins."""

import jax
import jax.numpy as jnp
import pytest

from valax.curves.convexity import (
    ConvexityAdjFn,
    constant_convexity_adj,
    no_convexity_adj,
)
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal


# ── Helpers ──────────────────────────────────────────────────────────


@pytest.fixture
def graph():
    """Trivial single-curve graph (the convexity plug-ins ignore it)."""
    ref = ymd_to_ordinal(2025, 1, 1)
    pillars = jnp.array(
        [int(ref), int(ymd_to_ordinal(2030, 1, 1))], dtype=jnp.int32
    )
    dfs = jnp.array([1.0, 0.8])
    curve = DiscountCurve(
        pillar_dates=pillars, discount_factors=dfs, reference_date=ref
    )
    return CurveGraph(curves={"USD.SOFR.3M": curve})


@pytest.fixture
def t0():
    return jnp.asarray(int(ymd_to_ordinal(2025, 6, 1)), dtype=jnp.int32)


@pytest.fixture
def t1():
    return jnp.asarray(int(ymd_to_ordinal(2025, 9, 1)), dtype=jnp.int32)


# ── no_convexity_adj ─────────────────────────────────────────────────


class TestNoConvexityAdj:
    def test_returns_zero(self, graph, t0, t1):
        adj = no_convexity_adj()
        result = adj(graph, t0, t1)
        assert float(result) == 0.0

    def test_dtype_is_float(self, graph, t0, t1):
        adj = no_convexity_adj()
        result = adj(graph, t0, t1)
        assert jnp.issubdtype(result.dtype, jnp.floating)

    def test_jit_compatible(self, graph, t0, t1):
        adj = no_convexity_adj()
        jitted = jax.jit(adj)
        result = jitted(graph, t0, t1)
        assert float(result) == 0.0


# ── constant_convexity_adj ───────────────────────────────────────────


class TestConstantConvexityAdj:
    def test_5_bps(self, graph, t0, t1):
        adj = constant_convexity_adj(5.0)
        result = adj(graph, t0, t1)
        assert float(result) == pytest.approx(0.0005, abs=1e-12)

    def test_0_bps(self, graph, t0, t1):
        adj = constant_convexity_adj(0.0)
        result = adj(graph, t0, t1)
        assert float(result) == pytest.approx(0.0, abs=1e-12)

    def test_negative_bps(self, graph, t0, t1):
        # Some markets / models produce negative adjustments.
        adj = constant_convexity_adj(-2.5)
        result = adj(graph, t0, t1)
        assert float(result) == pytest.approx(-0.00025, abs=1e-12)

    def test_independent_of_dates(self, graph):
        """Constant adjustment is independent of t0, t1."""
        adj = constant_convexity_adj(7.5)
        for y in [2025, 2026, 2027, 2028]:
            t0 = jnp.asarray(
                int(ymd_to_ordinal(y, 1, 1)), dtype=jnp.int32
            )
            t1 = jnp.asarray(
                int(ymd_to_ordinal(y, 4, 1)), dtype=jnp.int32
            )
            assert float(adj(graph, t0, t1)) == pytest.approx(0.00075)

    def test_independent_of_graph(self, t0, t1):
        """Constant adjustment is independent of the curve graph."""
        adj = constant_convexity_adj(3.0)
        ref = ymd_to_ordinal(2025, 1, 1)
        # Two different graphs with different DF profiles.
        g1 = CurveGraph(
            curves={
                "X": DiscountCurve(
                    pillar_dates=jnp.array([int(ref)], dtype=jnp.int32),
                    discount_factors=jnp.array([1.0]),
                    reference_date=ref,
                )
            }
        )
        g2 = CurveGraph(
            curves={
                "Y": DiscountCurve(
                    pillar_dates=jnp.array(
                        [int(ref), int(ymd_to_ordinal(2030, 1, 1))],
                        dtype=jnp.int32,
                    ),
                    discount_factors=jnp.array([1.0, 0.5]),
                    reference_date=ref,
                )
            }
        )
        assert float(adj(g1, t0, t1)) == float(adj(g2, t0, t1))

    def test_jit_compatible(self, graph, t0, t1):
        adj = constant_convexity_adj(5.0)
        jitted = jax.jit(adj)
        result = jitted(graph, t0, t1)
        assert float(result) == pytest.approx(0.0005, abs=1e-12)


# ── ConvexityAdjFn type alias ────────────────────────────────────────


class TestConvexityAdjFnTypeAlias:
    def test_alias_imports_cleanly(self):
        # Just confirm the alias is exposed.
        assert ConvexityAdjFn is not None

    def test_factories_match_alias_shape(self, graph, t0, t1):
        """Both factories return callables with the (graph, t0, t1) → float
        signature implied by the type alias."""
        for adj in [no_convexity_adj(), constant_convexity_adj(5.0)]:
            result = adj(graph, t0, t1)
            assert result.shape == ()  # scalar
            assert jnp.issubdtype(result.dtype, jnp.floating)
