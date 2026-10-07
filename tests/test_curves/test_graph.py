"""Tests for CurveGraph: identifier-keyed registry of discount curves."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DISCOUNT_CURVE_INTERP, DiscountCurve
from valax.curves.graph import CurveGraph, CurveSpec, _SUPPORTED_INTERP
from valax.dates.daycounts import ymd_to_ordinal


# ── Helpers ──────────────────────────────────────────────────────────


def _flat_curve(rate: float, day_count: str = "act_365") -> DiscountCurve:
    """Build a flat continuously-compounded discount curve at the given rate."""
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
    dfs = jnp.exp(-rate * times)
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=dfs,
        reference_date=ref,
        day_count=day_count,
    )


@pytest.fixture
def two_curve_graph():
    """USD OIS + USD 3M tenor curves at 4% and 4.4% flat."""
    return CurveGraph(
        curves={
            "USD.SOFR.OIS": _flat_curve(0.04),
            "USD.SOFR.3M": _flat_curve(0.044),
        }
    )


# ── Construction and lookup ──────────────────────────────────────────


class TestCurveGraphConstruction:
    def test_construct_with_one_curve(self):
        c = _flat_curve(0.05)
        g = CurveGraph(curves={"USD.SOFR.OIS": c})
        assert "USD.SOFR.OIS" in g
        assert g["USD.SOFR.OIS"] is c

    def test_construct_with_multiple_curves(self, two_curve_graph):
        assert "USD.SOFR.OIS" in two_curve_graph
        assert "USD.SOFR.3M" in two_curve_graph
        assert "EUR.ESTR.OIS" not in two_curve_graph

    def test_keys_values_items(self, two_curve_graph):
        assert set(two_curve_graph.keys()) == {"USD.SOFR.OIS", "USD.SOFR.3M"}
        assert len(list(two_curve_graph.values())) == 2
        items = dict(two_curve_graph.items())
        assert isinstance(items["USD.SOFR.OIS"], DiscountCurve)


class TestCurveGraphLookup:
    def test_get_existing_curve(self, two_curve_graph):
        c = two_curve_graph["USD.SOFR.OIS"]
        assert isinstance(c, DiscountCurve)

    def test_get_missing_curve_raises(self, two_curve_graph):
        with pytest.raises(KeyError):
            _ = two_curve_graph["GBP.SONIA.OIS"]

    def test_lookup_then_evaluate(self, two_curve_graph):
        # Round-trip: graph lookup → curve evaluation at a date.
        c = two_curve_graph["USD.SOFR.OIS"]
        df_1y = float(c(ymd_to_ordinal(2026, 1, 1)))
        # Flat 4% curve: DF(1Y) = exp(-0.04 * 1.0) ≈ 0.96079.
        assert df_1y == pytest.approx(jnp.exp(-0.04), abs=1e-8)


# ── Pytree properties ────────────────────────────────────────────────


class TestCurveGraphPytree:
    def test_is_pytree(self, two_curve_graph):
        leaves = jax.tree_util.tree_leaves(two_curve_graph)
        # 2 curves × (pillar_dates, discount_factors, reference_date) = 6.
        # day_count is static, not a leaf.
        assert len(leaves) == 6

    def test_tree_at_replaces_one_curve(self, two_curve_graph):
        new_3m = _flat_curve(0.05)
        bumped = eqx.tree_at(
            lambda g: g.curves["USD.SOFR.3M"], two_curve_graph, new_3m
        )
        # Other curve untouched (DFs equal element-wise; identity is
        # not preserved because eqx.tree_at rebuilds the pytree).
        old_ois_dfs = two_curve_graph["USD.SOFR.OIS"].discount_factors
        new_ois_dfs = bumped["USD.SOFR.OIS"].discount_factors
        assert jnp.array_equal(old_ois_dfs, new_ois_dfs)
        # Replaced curve has the new rate.
        df_1y = float(bumped["USD.SOFR.3M"](ymd_to_ordinal(2026, 1, 1)))
        assert df_1y == pytest.approx(jnp.exp(-0.05), abs=1e-8)

    def test_tree_at_replaces_dfs_in_place(self, two_curve_graph):
        # Bumping the discount_factors leaf of one curve.
        new_dfs = jnp.ones_like(
            two_curve_graph["USD.SOFR.OIS"].discount_factors
        )
        bumped = eqx.tree_at(
            lambda g: g.curves["USD.SOFR.OIS"].discount_factors,
            two_curve_graph,
            new_dfs,
        )
        assert jnp.all(bumped["USD.SOFR.OIS"].discount_factors == 1.0)
        # Other curve unchanged.
        ois3m = bumped["USD.SOFR.3M"]
        assert not jnp.all(ois3m.discount_factors == 1.0)

    def test_jit_compatible(self, two_curve_graph):
        # Pricing functions get the graph as a pytree input; the dict
        # keys are static, the curves themselves are dynamic leaves.
        @jax.jit
        def discount_at_1y(graph, date):
            return graph["USD.SOFR.OIS"](date)

        d = jnp.asarray(int(ymd_to_ordinal(2026, 1, 1)), dtype=jnp.int32)
        df = float(discount_at_1y(two_curve_graph, d))
        assert df == pytest.approx(jnp.exp(-0.04), abs=1e-8)

    def test_grad_through_graph(self, two_curve_graph):
        # Differentiate a function of the OIS curve's DFs through the
        # graph.  Use eqx.filter_grad because the graph contains int32
        # leaves (pillar_dates, reference_date) which jax.grad cannot
        # differentiate.  filter_grad differentiates only inexact-array
        # leaves and treats integer leaves as static.
        d = jnp.asarray(int(ymd_to_ordinal(2026, 1, 1)), dtype=jnp.int32)

        def f(graph):
            return graph["USD.SOFR.OIS"](d)

        grads = eqx.filter_grad(f)(two_curve_graph)
        ois_grad = grads.curves["USD.SOFR.OIS"].discount_factors
        # Some pillar must have non-zero sensitivity.
        assert jnp.any(jnp.abs(ois_grad) > 1e-12)
        # Other curve has zero gradient on its DFs.
        m3_grad = grads.curves["USD.SOFR.3M"].discount_factors
        assert jnp.all(m3_grad == 0.0)


class TestCurveSpecInterp:
    """``CurveSpec.interp`` is declared and checked but not forwarded.

    ``DiscountCurve`` has no ``interp`` field — log-linear-in-DF is
    hard-coded — so there is nowhere to forward a declared mode to.  That
    is not currently a defect, because the only admissible value is
    precisely what the curve implements.  It is a *latent* one: the day a
    second mode joins ``_SUPPORTED_INTERP``, the field becomes silently
    ignored and every curve quietly keeps interpolating log-linearly.

    These tests are the tripwire.  They are deliberately written to fail
    when someone widens the supported set, so the plumbing has to land in
    the same change as the new mode.
    """

    def test_declarable_modes_match_what_the_curve_implements(self):
        """The guard that matters.

        If this fails because ``_SUPPORTED_INTERP`` grew, do not relax
        it: give ``DiscountCurve`` a static ``interp`` field, dispatch on
        it in ``__call__``, and forward ``spec.interp`` in
        ``bootstrap_graph._build_graph_from_log_dfs``.  Then this can
        become a check that every declarable mode is dispatchable.
        """
        assert _SUPPORTED_INTERP == (DISCOUNT_CURVE_INTERP,), (
            "CurveSpec can declare an interpolation mode DiscountCurve "
            "cannot deliver; see the class docstring."
        )

    def test_the_default_is_the_implemented_mode(self):
        spec = CurveSpec(
            curve_id="USD.SOFR.OIS",
            currency="USD",
            pillar_dates=jnp.asarray([int(ymd_to_ordinal(2026, 1, 1))]),
        )
        assert spec.interp == DISCOUNT_CURVE_INTERP

    def test_an_unsupported_mode_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="not supported"):
            CurveSpec(
                curve_id="USD.SOFR.OIS",
                currency="USD",
                pillar_dates=jnp.asarray(
                    [int(ymd_to_ordinal(2026, 1, 1))]
                ),
                interp="monotone_convex",
            )

    def test_the_build_refuses_a_mode_it_cannot_honour(self):
        """The second half of the tripwire, at the point where the field
        would have to be forwarded.

        Exercised with a stand-in rather than a real ``CurveSpec``,
        because ``__check_init__`` makes the bad spec unconstructible —
        which is the point.  If the construction-time check is ever
        loosened, this one still fires.
        """
        from valax.curves.bootstrap_graph import _build_graph_from_log_dfs

        class _Spec:
            curve_id = "USD.SOFR.OIS"
            day_count = "act_365"
            interp = "monotone_convex"
            pillar_dates = jnp.asarray(
                [int(ymd_to_ordinal(2026, 1, 1))], dtype=jnp.int32
            )

        with pytest.raises(NotImplementedError, match="only implements"):
            _build_graph_from_log_dfs(
                jnp.asarray([-0.04]),
                [_Spec()],
                (0, 1),
                jnp.asarray(int(ymd_to_ordinal(2025, 1, 1))),
            )
