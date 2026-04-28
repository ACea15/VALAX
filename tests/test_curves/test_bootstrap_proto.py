"""Tests for the BootstrapInstrument protocol.

The protocol is structural: any class with ``curves_touched`` and
``residual`` qualifies.  These tests verify the runtime ``isinstance``
check, JIT-compatibility of a stub implementation, and that the
protocol's own contract is consistent with how the future joint solver
will dispatch.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Float, Int
from jax import Array

from valax.curves.bootstrap_proto import BootstrapInstrument
from valax.curves.discount import DiscountCurve
from valax.curves.fixings import FixingHistory, empty_fixing_history
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal


# ── Stub instrument satisfying the protocol ──────────────────────────


class _StubResidualOne(eqx.Module):
    """A trivial instrument whose residual is always 1.0.

    Used purely to verify the protocol's runtime contract: any class
    with ``curves_touched`` and a ``residual`` method conforming to the
    signature is a ``BootstrapInstrument``.
    """

    curve_id: str = eqx.field(static=True)
    curves_touched: tuple = eqx.field(static=True)

    def residual(
        self,
        graph: CurveGraph,
        fixings: FixingHistory,
        ref_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        return jnp.asarray(1.0)


class _StubDiscountResidual(eqx.Module):
    """A stub whose residual = curve(maturity) - target_df.

    Useful for testing that ``residual`` is differentiable and JIT-able.
    """

    target_df: Float[Array, ""]
    maturity: Int[Array, ""]
    curve_id: str = eqx.field(static=True)
    curves_touched: tuple = eqx.field(static=True)

    def residual(
        self,
        graph: CurveGraph,
        fixings: FixingHistory,
        ref_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        return graph[self.curve_id](self.maturity) - self.target_df


# ── Helpers ──────────────────────────────────────────────────────────


def _flat_curve(rate: float) -> DiscountCurve:
    ref = ymd_to_ordinal(2025, 1, 1)
    pillars = jnp.array(
        [
            int(ymd_to_ordinal(2025, 1, 1)),
            int(ymd_to_ordinal(2026, 1, 1)),
            int(ymd_to_ordinal(2030, 1, 1)),
        ],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref)).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rate * times),
        reference_date=ref,
    )


@pytest.fixture
def graph():
    return CurveGraph(curves={"USD.SOFR.OIS": _flat_curve(0.04)})


@pytest.fixture
def ref_date():
    return jnp.asarray(int(ymd_to_ordinal(2025, 1, 1)), dtype=jnp.int32)


# ── Protocol contract ────────────────────────────────────────────────


class TestProtocolContract:
    def test_runtime_isinstance(self):
        inst = _StubResidualOne(
            curve_id="USD.SOFR.OIS", curves_touched=("USD.SOFR.OIS",)
        )
        assert isinstance(inst, BootstrapInstrument)

    def test_negative_isinstance(self):
        # A class lacking ``residual`` is not a BootstrapInstrument.
        class NotAnInstrument(eqx.Module):
            curves_touched: tuple = eqx.field(static=True, default=())

        assert not isinstance(
            NotAnInstrument(), BootstrapInstrument
        )

    def test_curves_touched_is_static(self):
        # ``curves_touched`` must be treatable as static — i.e. it's
        # part of the pytree structure, not a leaf.  Concretely:
        # ``jax.tree_util.tree_leaves`` should not return the tuple
        # itself as a leaf.
        inst = _StubResidualOne(
            curve_id="USD.SOFR.OIS", curves_touched=("USD.SOFR.OIS",)
        )
        leaves = jax.tree_util.tree_leaves(inst)
        # Both fields are static — no array leaves.
        assert len(leaves) == 0


# ── Residual semantics ───────────────────────────────────────────────


class TestStubResidual:
    def test_constant_residual(self, graph, ref_date):
        inst = _StubResidualOne(
            curve_id="USD.SOFR.OIS", curves_touched=("USD.SOFR.OIS",)
        )
        r = inst.residual(graph, empty_fixing_history(), ref_date)
        assert float(r) == pytest.approx(1.0)

    def test_curve_dependent_residual_zero_on_match(self, graph, ref_date):
        # Build the stub with target_df = curve(maturity) so residual = 0.
        maturity = jnp.asarray(int(ymd_to_ordinal(2026, 1, 1)), dtype=jnp.int32)
        target = graph["USD.SOFR.OIS"](maturity)
        inst = _StubDiscountResidual(
            target_df=target,
            maturity=maturity,
            curve_id="USD.SOFR.OIS",
            curves_touched=("USD.SOFR.OIS",),
        )
        r = inst.residual(graph, empty_fixing_history(), ref_date)
        assert float(r) == pytest.approx(0.0, abs=1e-12)

    def test_curve_dependent_residual_nonzero_on_mismatch(self, graph, ref_date):
        maturity = jnp.asarray(int(ymd_to_ordinal(2026, 1, 1)), dtype=jnp.int32)
        target = jnp.asarray(0.5)  # very different from flat-4% DF
        inst = _StubDiscountResidual(
            target_df=target,
            maturity=maturity,
            curve_id="USD.SOFR.OIS",
            curves_touched=("USD.SOFR.OIS",),
        )
        r = inst.residual(graph, empty_fixing_history(), ref_date)
        # Curve gives ~0.961, target is 0.5 → residual ≈ 0.461.
        assert float(r) == pytest.approx(0.4608, abs=1e-3)


# ── JIT and autodiff ─────────────────────────────────────────────────


class TestJitAndGrad:
    def test_jit_residual(self, graph, ref_date):
        maturity = jnp.asarray(int(ymd_to_ordinal(2026, 1, 1)), dtype=jnp.int32)
        target = graph["USD.SOFR.OIS"](maturity)
        inst = _StubDiscountResidual(
            target_df=target,
            maturity=maturity,
            curve_id="USD.SOFR.OIS",
            curves_touched=("USD.SOFR.OIS",),
        )

        @jax.jit
        def residual_jit(g, f, t):
            return inst.residual(g, f, t)

        r = residual_jit(graph, empty_fixing_history(), ref_date)
        assert float(r) == pytest.approx(0.0, abs=1e-12)

    def test_grad_through_residual(self, graph, ref_date):
        # Differentiate the residual w.r.t. target_df.  Sign should be
        # negative (residual = curve(t) - target).
        maturity = jnp.asarray(int(ymd_to_ordinal(2026, 1, 1)), dtype=jnp.int32)

        def f(target_df):
            inst = _StubDiscountResidual(
                target_df=target_df,
                maturity=maturity,
                curve_id="USD.SOFR.OIS",
                curves_touched=("USD.SOFR.OIS",),
            )
            return inst.residual(graph, empty_fixing_history(), ref_date)

        g = jax.grad(f)(jnp.asarray(0.96))
        assert float(g) == pytest.approx(-1.0, abs=1e-10)

    def test_grad_through_graph(self, graph, ref_date):
        # Differentiate the residual w.r.t. the curve's DFs.  Use
        # eqx.filter_grad because the graph contains int32 leaves;
        # filter_grad differentiates only inexact-array leaves.
        maturity = jnp.asarray(int(ymd_to_ordinal(2026, 1, 1)), dtype=jnp.int32)
        inst = _StubDiscountResidual(
            target_df=jnp.asarray(0.95),
            maturity=maturity,
            curve_id="USD.SOFR.OIS",
            curves_touched=("USD.SOFR.OIS",),
        )

        def f(g):
            return inst.residual(g, empty_fixing_history(), ref_date)

        grads = eqx.filter_grad(f)(graph)
        ois_grad = grads.curves["USD.SOFR.OIS"].discount_factors
        assert jnp.any(jnp.abs(ois_grad) > 1e-12)
