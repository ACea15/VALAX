"""Tests for the joint multi-curve Newton solver (MC-Curves-2).

Coverage:

* Single-curve degenerate case reproduces :func:`bootstrap_simultaneous`
  to machine precision.
* Two-curve same-currency dual-curve build (OIS + 3M) with
  :class:`IborSwapRate` closes joint residuals to ≤ 1e-10.
* Three-curve build (OIS + 3M + 6M) closed by a :class:`TenorBasisSwap`.
* Four-curve EUR/USD graph closed by an :class:`FXForward` and a
  :class:`CrossCurrencyBasisSwap`.
* ``jax.grad`` flows through :func:`bootstrap_curve_graph` via
  ``optimistix.ImplicitAdjoint`` (finite, non-zero gradients).
* Input validation errors (square-system, unknown curve id).
"""

import jax
import jax.numpy as jnp
import pytest

from valax.curves import (
    CrossCurrencyBasisSwap,
    CurveSpec,
    DepositRate,
    FRA,
    FXForward,
    IborSwapRate,
    OISSwapRate,
    SwapRate,
    TenorBasisSwap,
    bootstrap_curve_graph,
    bootstrap_simultaneous,
    empty_fixing_history,
)
from valax.dates.daycounts import ymd_to_ordinal


REF = ymd_to_ordinal(2025, 1, 1)


def _date(y, m, d):
    return ymd_to_ordinal(y, m, d)


# ── Fixtures / helpers ───────────────────────────────────────────────


def _quarterly_dates(end_date_ordinal: int) -> jnp.ndarray:
    """Quarterly dates strictly after REF up to (and including)
    ``end_date_ordinal``.  Uses 1st-of-Jan/Apr/Jul/Oct roll.
    """
    dates: list[int] = []
    for year in range(2025, 2036):
        for month in (1, 4, 7, 10):
            d = int(_date(year, month, 1))
            if d > int(REF) and d <= end_date_ordinal:
                dates.append(d)
    return jnp.array(dates, dtype=jnp.int32)


def _semiannual_dates(end_date_ordinal: int) -> jnp.ndarray:
    """Semi-annual dates strictly after REF, 1-Jul / 1-Jan roll."""
    dates: list[int] = []
    for year in range(2025, 2036):
        for month in (1, 7):
            d = int(_date(year, month, 1))
            if d > int(REF) and d <= end_date_ordinal:
                dates.append(d)
    return jnp.array(dates, dtype=jnp.int32)


# ── Test 1: single-curve degenerate case ─────────────────────────────


class TestSingleCurveDegeneracy:
    """Joint solver must agree with the (already-migrated) single-curve
    ``bootstrap_simultaneous`` to machine precision on the classic
    three quote types.
    """

    def test_deposit_strip_matches_bootstrap_simultaneous(self):
        pillars = jnp.array(
            [
                int(_date(2025, 4, 1)),
                int(_date(2025, 7, 1)),
                int(_date(2026, 1, 1)),
            ],
            dtype=jnp.int32,
        )
        # bootstrap_simultaneous uses the sentinel "_default_" implicitly.
        deps_default = [
            DepositRate(
                start_date=REF, end_date=int(pillars[i]),
                rate=jnp.array(r), day_count="act_365",
            )
            for i, r in enumerate([0.045, 0.048, 0.050])
        ]
        curve_ref = bootstrap_simultaneous(REF, pillars, deps_default)

        # Same instruments, explicit curve id, joint solver.
        deps_joint = [
            DepositRate(
                start_date=REF, end_date=int(pillars[i]),
                rate=jnp.array(r), day_count="act_365",
                curves_touched=("USD.SOFR.OIS",),
            )
            for i, r in enumerate([0.045, 0.048, 0.050])
        ]
        spec = CurveSpec(
            curve_id="USD.SOFR.OIS", currency="USD",
            pillar_dates=pillars, day_count="act_365",
        )
        graph, diag = bootstrap_curve_graph(REF, [spec], deps_joint)

        assert diag.converged
        assert float(diag.max_abs_residual) < 1e-12

        joint_curve = graph["USD.SOFR.OIS"]
        for p in pillars:
            diff = abs(float(joint_curve(p)) - float(curve_ref(p)))
            assert diff < 1e-12, (
                f"pillar {int(p)}: joint={float(joint_curve(p))} "
                f"seq={float(curve_ref(p))} diff={diff}"
            )


# ── Test 2: dual-curve OIS + 3M via IborSwapRate ─────────────────────


class TestDualCurve:
    """Two-curve same-currency joint bootstrap.

    The OIS side is anchored by two deposits (single-curve residuals).
    The 3M-projection side is anchored by two IBOR par swaps, each
    of which is a *dual-curve* residual involving both curves
    simultaneously — this is the case ``bootstrap_multi_curve``'s
    sequential dual-curve pipeline can also handle but the joint
    solver generalises.
    """

    def test_joint_dual_curve_closes(self):
        disc_pillars = jnp.array(
            [int(_date(2026, 1, 1)), int(_date(2027, 1, 1))],
            dtype=jnp.int32,
        )
        fwd_pillars = disc_pillars

        disc_spec = CurveSpec(
            curve_id="USD.SOFR.OIS", currency="USD",
            pillar_dates=disc_pillars, day_count="act_365",
        )
        fwd_spec = CurveSpec(
            curve_id="USD.SOFR.3M", currency="USD",
            pillar_dates=fwd_pillars, day_count="act_365",
        )

        ois_deps = [
            DepositRate(
                start_date=REF, end_date=int(disc_pillars[0]),
                rate=jnp.array(0.040), day_count="act_365",
                curves_touched=("USD.SOFR.OIS",),
            ),
            DepositRate(
                start_date=REF, end_date=int(disc_pillars[1]),
                rate=jnp.array(0.042), day_count="act_365",
                curves_touched=("USD.SOFR.OIS",),
            ),
        ]

        # 1Y and 2Y IBOR swap quotes.
        float_1y = _quarterly_dates(int(_date(2026, 1, 1)))
        # Fixing date = accrual start; first accrual starts at REF, then
        # each successive fixing is the previous payment date.
        fix_1y = jnp.concatenate([jnp.array([REF], dtype=jnp.int32), float_1y[:-1]])
        ibor_1y = IborSwapRate(
            start_date=REF,
            fixed_dates=jnp.array([int(_date(2026, 1, 1))], dtype=jnp.int32),
            float_dates=float_1y,
            fixing_dates=fix_1y,
            rate=jnp.array(0.045),
            fixed_day_count="act_365",
            float_day_count="act_365",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
            index_id="USD.SOFR.3M",
        )
        float_2y = _quarterly_dates(int(_date(2027, 1, 1)))
        fix_2y = jnp.concatenate([jnp.array([REF], dtype=jnp.int32), float_2y[:-1]])
        ibor_2y = IborSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(
                [int(_date(2026, 1, 1)), int(_date(2027, 1, 1))],
                dtype=jnp.int32,
            ),
            float_dates=float_2y,
            fixing_dates=fix_2y,
            rate=jnp.array(0.047),
            fixed_day_count="act_365",
            float_day_count="act_365",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
            index_id="USD.SOFR.3M",
        )

        instruments = ois_deps + [ibor_1y, ibor_2y]
        graph, diag = bootstrap_curve_graph(
            REF, [disc_spec, fwd_spec], instruments,
        )

        assert diag.converged
        assert float(diag.max_abs_residual) < 1e-10

        # OIS-leg residuals of the deposits alone must be zero exactly
        # (their residual depends only on the OIS curve).
        for dep in ois_deps:
            r = float(dep.residual(graph, empty_fixing_history(), REF))
            assert abs(r) < 1e-12

        # IBOR par residuals must be zero to solver tol.
        for swap in (ibor_1y, ibor_2y):
            r = float(swap.residual(graph, empty_fixing_history(), REF))
            assert abs(r) < 1e-9


# ── Test 3: three-curve build closed by a tenor-basis swap ───────────


class TestTenorBasisJointSolve:
    """A tenor-basis swap simultaneously constrains two forward curves.

    This case is *unreachable* by the sequential dual-curve
    ``bootstrap_multi_curve`` — the basis-swap residual is a joint
    constraint on the 3M and 6M forward curves with the OIS curve
    fixed.  Adding a :class:`TenorBasisSwap` anchor lets the joint
    solver strip a second forward tenor.
    """

    def test_three_curve_closes_to_machine_precision(self):
        # 1) OIS: two deposits.
        disc_pillars = jnp.array(
            [int(_date(2026, 1, 1)), int(_date(2027, 1, 1))],
            dtype=jnp.int32,
        )
        disc_spec = CurveSpec(
            curve_id="USD.SOFR.OIS", currency="USD",
            pillar_dates=disc_pillars, day_count="act_365",
        )

        # 2) 3M forward: one deposit + one IBOR swap.
        fwd_3m_pillars = jnp.array(
            [int(_date(2026, 1, 1)), int(_date(2027, 1, 1))],
            dtype=jnp.int32,
        )
        fwd_3m_spec = CurveSpec(
            curve_id="USD.SOFR.3M", currency="USD",
            pillar_dates=fwd_3m_pillars, day_count="act_365",
        )

        # 3) 6M forward: one deposit at 2027, anchored above the
        #    short end by a tenor-basis swap constraint.
        fwd_6m_pillars = jnp.array(
            [int(_date(2027, 1, 1))], dtype=jnp.int32,
        )
        fwd_6m_spec = CurveSpec(
            curve_id="USD.SOFR.6M", currency="USD",
            pillar_dates=fwd_6m_pillars, day_count="act_365",
        )

        # Instruments (5 = 2 + 2 + 1).
        ois_deps = [
            DepositRate(
                start_date=REF, end_date=int(disc_pillars[0]),
                rate=jnp.array(0.040), day_count="act_365",
                curves_touched=("USD.SOFR.OIS",),
            ),
            DepositRate(
                start_date=REF, end_date=int(disc_pillars[1]),
                rate=jnp.array(0.042), day_count="act_365",
                curves_touched=("USD.SOFR.OIS",),
            ),
        ]

        # 3M anchors
        float_1y = _quarterly_dates(int(_date(2026, 1, 1)))
        fix_1y = jnp.concatenate([jnp.array([REF], dtype=jnp.int32), float_1y[:-1]])
        ibor_1y_3m = IborSwapRate(
            start_date=REF,
            fixed_dates=jnp.array([int(_date(2026, 1, 1))], dtype=jnp.int32),
            float_dates=float_1y, fixing_dates=fix_1y,
            rate=jnp.array(0.045),
            fixed_day_count="act_365", float_day_count="act_365",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
            index_id="USD.SOFR.3M",
        )
        float_2y = _quarterly_dates(int(_date(2027, 1, 1)))
        fix_2y = jnp.concatenate([jnp.array([REF], dtype=jnp.int32), float_2y[:-1]])
        ibor_2y_3m = IborSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(
                [int(_date(2026, 1, 1)), int(_date(2027, 1, 1))],
                dtype=jnp.int32,
            ),
            float_dates=float_2y, fixing_dates=fix_2y,
            rate=jnp.array(0.047),
            fixed_day_count="act_365", float_day_count="act_365",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
            index_id="USD.SOFR.3M",
        )

        # 6M anchor via a 3M-vs-6M basis swap over 2Y.
        leg_a_dates = _quarterly_dates(int(_date(2027, 1, 1)))   # 3M leg
        leg_a_fix = jnp.concatenate([jnp.array([REF], dtype=jnp.int32), leg_a_dates[:-1]])
        leg_b_dates = _semiannual_dates(int(_date(2027, 1, 1)))  # 6M leg
        leg_b_fix = jnp.concatenate([jnp.array([REF], dtype=jnp.int32), leg_b_dates[:-1]])
        basis_2y = TenorBasisSwap(
            start_date=REF,
            leg_a_dates=leg_a_dates,
            leg_a_fixing_dates=leg_a_fix,
            leg_b_dates=leg_b_dates,
            leg_b_fixing_dates=leg_b_fix,
            spread=jnp.array(0.0015),  # 15 bp on the 3M leg
            leg_a_day_count="act_365",
            leg_a_index_id="USD.SOFR.3M",
            leg_b_day_count="act_365",
            leg_b_index_id="USD.SOFR.6M",
            spread_on_leg="a",
            curves_touched=(
                "USD.SOFR.OIS", "USD.SOFR.3M", "USD.SOFR.6M",
            ),
        )

        instruments = ois_deps + [ibor_1y_3m, ibor_2y_3m, basis_2y]
        graph, diag = bootstrap_curve_graph(
            REF, [disc_spec, fwd_3m_spec, fwd_6m_spec], instruments,
        )

        assert diag.converged
        assert float(diag.max_abs_residual) < 1e-10
        assert {"USD.SOFR.OIS", "USD.SOFR.3M", "USD.SOFR.6M"} == set(graph.keys())
        # Each curve must have distinct DFs at maturity (otherwise the
        # basis anchor has collapsed).
        df_ois = float(graph["USD.SOFR.OIS"](_date(2027, 1, 1)))
        df_3m = float(graph["USD.SOFR.3M"](_date(2027, 1, 1)))
        df_6m = float(graph["USD.SOFR.6M"](_date(2027, 1, 1)))
        assert df_ois != df_3m
        assert df_3m != df_6m


# ── Test 4: EUR/USD four-curve close via FXForward + CCBS ────────────


class TestCrossCurrencyJointSolve:
    """The CCBS is the only quote type that touches four curves at once.

    Combined with an FXForward on the short end, we close a full
    EUR/USD graph: {USD OIS, USD 3M, EUR OIS, EUR 6M}.  Sequential
    dual-curve pipelines cannot do this because the CCBS constraint
    couples both currencies.
    """

    def test_four_curve_closes(self):
        # USD side: OIS pillar at 2026, 2027; 3M pillar at 2027.
        usd_ois_pillars = jnp.array(
            [int(_date(2026, 1, 1)), int(_date(2027, 1, 1))],
            dtype=jnp.int32,
        )
        usd_3m_pillars = jnp.array(
            [int(_date(2027, 1, 1))], dtype=jnp.int32,
        )

        # EUR side: OIS pillar at 2026, 2027; 6M pillar at 2027.
        eur_ois_pillars = jnp.array(
            [int(_date(2026, 1, 1)), int(_date(2027, 1, 1))],
            dtype=jnp.int32,
        )
        eur_6m_pillars = jnp.array(
            [int(_date(2027, 1, 1))], dtype=jnp.int32,
        )

        specs = [
            CurveSpec(
                curve_id="USD.SOFR.OIS", currency="USD",
                pillar_dates=usd_ois_pillars, day_count="act_365",
            ),
            CurveSpec(
                curve_id="USD.SOFR.3M", currency="USD",
                pillar_dates=usd_3m_pillars, day_count="act_365",
            ),
            CurveSpec(
                curve_id="EUR.ESTR.OIS", currency="EUR",
                pillar_dates=eur_ois_pillars, day_count="act_365",
            ),
            CurveSpec(
                curve_id="EUR.EURIBOR.6M", currency="EUR",
                pillar_dates=eur_6m_pillars, day_count="act_365",
            ),
        ]

        # 6 = 2 + 2 pinning OIS + 1 IBOR (USD 3M) + 1 CCBS anchors 2 = 1+1
        # Total unknowns = 2 + 1 + 2 + 1 = 6.  We need 6 instruments.

        usd_ois_deps = [
            DepositRate(
                start_date=REF, end_date=int(usd_ois_pillars[0]),
                rate=jnp.array(0.040), day_count="act_365",
                curves_touched=("USD.SOFR.OIS",),
            ),
            DepositRate(
                start_date=REF, end_date=int(usd_ois_pillars[1]),
                rate=jnp.array(0.042), day_count="act_365",
                curves_touched=("USD.SOFR.OIS",),
            ),
        ]
        eur_ois_deps = [
            DepositRate(
                start_date=REF, end_date=int(eur_ois_pillars[0]),
                rate=jnp.array(0.028), day_count="act_365",
                curves_touched=("EUR.ESTR.OIS",),
            ),
            DepositRate(
                start_date=REF, end_date=int(eur_ois_pillars[1]),
                rate=jnp.array(0.030), day_count="act_365",
                curves_touched=("EUR.ESTR.OIS",),
            ),
        ]

        # USD 3M pillar via 2Y IBOR swap.
        float_2y = _quarterly_dates(int(_date(2027, 1, 1)))
        fix_2y = jnp.concatenate([jnp.array([REF], dtype=jnp.int32), float_2y[:-1]])
        usd_ibor_2y = IborSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(
                [int(_date(2026, 1, 1)), int(_date(2027, 1, 1))],
                dtype=jnp.int32,
            ),
            float_dates=float_2y, fixing_dates=fix_2y,
            rate=jnp.array(0.047),
            fixed_day_count="act_365", float_day_count="act_365",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
            index_id="USD.SOFR.3M",
        )

        # EUR 6M pillar via CCBS 2Y (USD 3M vs EUR 6M).
        dom_dates = _quarterly_dates(int(_date(2027, 1, 1)))   # USD leg (3M)
        dom_fix = jnp.concatenate([jnp.array([REF], dtype=jnp.int32), dom_dates[:-1]])
        for_dates = _semiannual_dates(int(_date(2027, 1, 1)))  # EUR leg (6M)
        for_fix = jnp.concatenate([jnp.array([REF], dtype=jnp.int32), for_dates[:-1]])
        ccbs_2y = CrossCurrencyBasisSwap(
            start_date=REF,
            dom_dates=dom_dates,
            dom_fixing_dates=dom_fix,
            for_dates=for_dates,
            for_fixing_dates=for_fix,
            fx_spot=jnp.array(1.10),   # USD per EUR
            spread=jnp.array(-0.0025),  # -25 bp on the EUR leg
            dom_day_count="act_365",
            dom_index_id="USD.SOFR.3M",
            for_day_count="act_365",
            for_index_id="EUR.EURIBOR.6M",
            spread_on_leg="foreign",
            variant="mtm",
            curves_touched=(
                "USD.SOFR.OIS", "USD.SOFR.3M",
                "EUR.ESTR.OIS", "EUR.EURIBOR.6M",
            ),
        )

        instruments = usd_ois_deps + eur_ois_deps + [usd_ibor_2y, ccbs_2y]
        assert len(instruments) == sum(int(s.pillar_dates.shape[0]) for s in specs)

        graph, diag = bootstrap_curve_graph(REF, specs, instruments)

        assert diag.converged
        assert float(diag.max_abs_residual) < 1e-10
        assert set(graph.keys()) == {
            "USD.SOFR.OIS", "USD.SOFR.3M", "EUR.ESTR.OIS", "EUR.EURIBOR.6M",
        }


# ── Test 5: gradient flow through the solver ─────────────────────────


class TestGradientFlow:
    """`jax.grad` must flow through the joint solver via
    `optimistix.ImplicitAdjoint`.
    """

    def test_grad_of_df_wrt_quote_is_finite_and_signed(self):
        pillars = jnp.array(
            [int(_date(2026, 1, 1))], dtype=jnp.int32,
        )
        spec = CurveSpec(
            curve_id="USD.SOFR.OIS", currency="USD",
            pillar_dates=pillars, day_count="act_365",
        )

        def price(r):
            dep = DepositRate(
                start_date=REF, end_date=int(pillars[0]),
                rate=r, day_count="act_365",
                curves_touched=("USD.SOFR.OIS",),
            )
            g, _ = bootstrap_curve_graph(REF, [spec], [dep])
            return g["USD.SOFR.OIS"](pillars[0])

        g = jax.grad(price)(jnp.array(0.05))
        assert bool(jnp.isfinite(g))
        # Higher rate → lower DF, so gradient must be negative.
        assert float(g) < -1e-3

    def test_grad_couples_across_curves_in_dual_solve(self):
        """A 2Y IBOR swap rate change must move the 3M-forward DF."""
        disc_pillars = jnp.array(
            [int(_date(2026, 1, 1)), int(_date(2027, 1, 1))],
            dtype=jnp.int32,
        )
        fwd_pillars = disc_pillars
        disc_spec = CurveSpec(
            curve_id="USD.SOFR.OIS", currency="USD",
            pillar_dates=disc_pillars, day_count="act_365",
        )
        fwd_spec = CurveSpec(
            curve_id="USD.SOFR.3M", currency="USD",
            pillar_dates=fwd_pillars, day_count="act_365",
        )
        ois_deps = [
            DepositRate(
                start_date=REF, end_date=int(disc_pillars[0]),
                rate=jnp.array(0.040), day_count="act_365",
                curves_touched=("USD.SOFR.OIS",),
            ),
            DepositRate(
                start_date=REF, end_date=int(disc_pillars[1]),
                rate=jnp.array(0.042), day_count="act_365",
                curves_touched=("USD.SOFR.OIS",),
            ),
        ]
        float_1y = _quarterly_dates(int(_date(2026, 1, 1)))
        fix_1y = jnp.concatenate([jnp.array([REF], dtype=jnp.int32), float_1y[:-1]])
        ibor_1y = IborSwapRate(
            start_date=REF,
            fixed_dates=jnp.array([int(_date(2026, 1, 1))], dtype=jnp.int32),
            float_dates=float_1y, fixing_dates=fix_1y,
            rate=jnp.array(0.045),
            fixed_day_count="act_365", float_day_count="act_365",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
            index_id="USD.SOFR.3M",
        )
        float_2y = _quarterly_dates(int(_date(2027, 1, 1)))
        fix_2y = jnp.concatenate([jnp.array([REF], dtype=jnp.int32), float_2y[:-1]])

        import equinox as eqx
        ibor_2y_template = IborSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(
                [int(_date(2026, 1, 1)), int(_date(2027, 1, 1))],
                dtype=jnp.int32,
            ),
            float_dates=float_2y, fixing_dates=fix_2y,
            rate=jnp.array(0.047),
            fixed_day_count="act_365", float_day_count="act_365",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
            index_id="USD.SOFR.3M",
        )

        def price(r2y):
            ib2 = eqx.tree_at(lambda x: x.rate, ibor_2y_template, r2y)
            g, _ = bootstrap_curve_graph(
                REF, [disc_spec, fwd_spec],
                ois_deps + [ibor_1y, ib2],
            )
            return g["USD.SOFR.3M"](fwd_pillars[1])

        g = jax.grad(price)(jnp.array(0.047))
        assert bool(jnp.isfinite(g))
        # Higher 2Y IBOR par rate → lower forward DF at 2027.
        assert float(g) < -1e-2


# ── Test 6: input validation ─────────────────────────────────────────


class TestValidation:
    def test_wrong_instrument_count_raises(self):
        spec = CurveSpec(
            curve_id="USD.SOFR.OIS", currency="USD",
            pillar_dates=jnp.array([int(_date(2026, 1, 1))], dtype=jnp.int32),
            day_count="act_365",
        )
        dep_a = DepositRate(
            start_date=REF, end_date=int(_date(2026, 1, 1)),
            rate=jnp.array(0.05), day_count="act_365",
            curves_touched=("USD.SOFR.OIS",),
        )
        dep_b = DepositRate(
            start_date=REF, end_date=int(_date(2026, 6, 1)),
            rate=jnp.array(0.05), day_count="act_365",
            curves_touched=("USD.SOFR.OIS",),
        )
        with pytest.raises(ValueError, match="one instrument per pillar"):
            bootstrap_curve_graph(REF, [spec], [dep_a, dep_b])

    def test_unknown_curve_id_raises(self):
        spec = CurveSpec(
            curve_id="USD.SOFR.OIS", currency="USD",
            pillar_dates=jnp.array([int(_date(2026, 1, 1))], dtype=jnp.int32),
            day_count="act_365",
        )
        # Instrument references a curve not in the graph spec set.
        dep = DepositRate(
            start_date=REF, end_date=int(_date(2026, 1, 1)),
            rate=jnp.array(0.05), day_count="act_365",
            curves_touched=("EUR.ESTR.OIS",),
        )
        with pytest.raises(ValueError, match="unknown curve"):
            bootstrap_curve_graph(REF, [spec], [dep])
