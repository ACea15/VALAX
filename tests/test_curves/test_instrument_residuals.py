"""Residual unit tests for the migrated bootstrap instruments.

For each of :class:`DepositRate`, :class:`FRA`, :class:`SwapRate`,
verify that:

* ``inst.residual(graph, fixings, ref)`` returns ~0 on a hand-built
  :class:`CurveGraph` whose discount factors satisfy the instrument's
  no-arb relation;

* ``inst.residual(...)`` returns a non-zero value when the curve is
  deliberately inconsistent with the quote;

* the residual definition is consistent with what
  ``bootstrap_sequential`` converges to (round-trip property test).

These tests are the contract that the future joint solver
(``bootstrap_curve_graph`` in MC-Curves-2) will rely on.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.bootstrap import (
    bootstrap_sequential,
    bootstrap_simultaneous,
)
from valax.curves.discount import DiscountCurve
from valax.curves.fixings import empty_fixing_history
from valax.curves.graph import CurveGraph
from valax.curves.convexity import (
    constant_convexity_adj,
    no_convexity_adj,
)
from valax.curves.fixings import FixingHistory, FixingSeries
from valax.curves.instruments import (
    CrossCurrencyBasisSwap,
    DepositRate,
    FRA,
    FXForward,
    FXSwap,
    IborSwapRate,
    MoneyMarketFuture,
    OISSwapRate,
    SwapRate,
    TenorBasisSwap,
    TurnInstrument,
)
from valax.dates.daycounts import year_fraction, ymd_to_ordinal


# ── Helpers ──────────────────────────────────────────────────────────


REF = ymd_to_ordinal(2025, 1, 1)
NO_FIXINGS = empty_fixing_history()


def _make_date(y: int, m: int, d: int):
    return ymd_to_ordinal(y, m, d)


def _flat_continuously_compounded(
    rate: float, day_count: str = "act_365"
) -> DiscountCurve:
    """Construct a flat continuously-compounded curve at ``rate``."""
    pillars = jnp.array(
        [
            int(_make_date(2025, 1, 1)),
            int(_make_date(2025, 7, 1)),
            int(_make_date(2026, 1, 1)),
            int(_make_date(2027, 1, 1)),
            int(_make_date(2030, 1, 1)),
            int(_make_date(2035, 1, 1)),
        ],
        dtype=jnp.int32,
    )
    times = (pillars - int(REF)).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-rate * times)
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=dfs,
        reference_date=REF,
        day_count=day_count,
    )


def _wrap_default(curve: DiscountCurve) -> CurveGraph:
    return CurveGraph(curves={"_default_": curve})


# ── DepositRate residual ─────────────────────────────────────────────


class TestDepositRateResidual:
    def test_residual_zero_on_match(self):
        """Build a deposit at the implied simple rate, residual ≈ 0."""
        curve = _flat_continuously_compounded(0.05)
        graph = _wrap_default(curve)
        end = _make_date(2025, 7, 1)
        # Implied simply-compounded rate from the flat-CC curve.
        df_end = float(curve(end))
        tau = float(year_fraction(REF, end, "act_360"))
        implied_rate = (1.0 / df_end - 1.0) / tau
        dep = DepositRate(
            start_date=REF,
            end_date=end,
            rate=jnp.asarray(implied_rate),
            day_count="act_360",
        )
        r = dep.residual(graph, NO_FIXINGS, REF)
        assert abs(float(r)) < 1e-10

    def test_residual_nonzero_on_mismatch(self):
        """Deliberately wrong rate → residual is non-zero with the
        expected sign (rate too high → DF(end)*(1+r*tau) > DF(start),
        so residual > 0)."""
        curve = _flat_continuously_compounded(0.05)
        graph = _wrap_default(curve)
        end = _make_date(2025, 7, 1)
        df_end = float(curve(end))
        tau = float(year_fraction(REF, end, "act_360"))
        implied_rate = (1.0 / df_end - 1.0) / tau
        dep = DepositRate(
            start_date=REF,
            end_date=end,
            rate=jnp.asarray(implied_rate + 0.01),  # +100 bps
            day_count="act_360",
        )
        r = float(dep.residual(graph, NO_FIXINGS, REF))
        assert r > 1e-4  # well above the 1e-10 zero threshold

    def test_curves_touched_default(self):
        dep = DepositRate(
            start_date=REF,
            end_date=_make_date(2025, 7, 1),
            rate=jnp.asarray(0.05),
        )
        assert dep.curves_touched == ("_default_",)

    def test_curves_touched_override(self):
        """A multi-curve consumer can override the default identifier."""
        curve = _flat_continuously_compounded(0.05)
        graph = CurveGraph(curves={"USD.SOFR.OIS": curve})
        end = _make_date(2025, 7, 1)
        df_end = float(curve(end))
        tau = float(year_fraction(REF, end, "act_360"))
        implied_rate = (1.0 / df_end - 1.0) / tau
        dep = DepositRate(
            start_date=REF,
            end_date=end,
            rate=jnp.asarray(implied_rate),
            day_count="act_360",
            curves_touched=("USD.SOFR.OIS",),
        )
        r = dep.residual(graph, NO_FIXINGS, REF)
        assert abs(float(r)) < 1e-10


# ── FRA residual ─────────────────────────────────────────────────────


class TestFRAResidual:
    def test_residual_zero_on_match(self):
        """FRA over [3M, 6M] at the implied simply-compounded forward."""
        curve = _flat_continuously_compounded(0.05)
        graph = _wrap_default(curve)
        start = _make_date(2025, 4, 1)
        end = _make_date(2025, 7, 1)
        df_start = float(curve(start))
        df_end = float(curve(end))
        tau = float(year_fraction(start, end, "act_360"))
        implied_rate = (df_start / df_end - 1.0) / tau
        fra = FRA(
            start_date=start,
            end_date=end,
            rate=jnp.asarray(implied_rate),
            day_count="act_360",
        )
        r = fra.residual(graph, NO_FIXINGS, REF)
        assert abs(float(r)) < 1e-10

    def test_residual_nonzero_on_mismatch(self):
        curve = _flat_continuously_compounded(0.05)
        graph = _wrap_default(curve)
        start = _make_date(2025, 4, 1)
        end = _make_date(2025, 7, 1)
        df_start = float(curve(start))
        df_end = float(curve(end))
        tau = float(year_fraction(start, end, "act_360"))
        implied_rate = (df_start / df_end - 1.0) / tau
        fra = FRA(
            start_date=start,
            end_date=end,
            rate=jnp.asarray(implied_rate + 0.01),
            day_count="act_360",
        )
        r = float(fra.residual(graph, NO_FIXINGS, REF))
        assert abs(r) > 1e-4


# ── SwapRate residual ────────────────────────────────────────────────


class TestSwapRateResidual:
    def test_residual_zero_at_par_rate(self):
        """For any curve, the par swap rate makes the residual ~0."""
        curve = _flat_continuously_compounded(0.05)
        graph = _wrap_default(curve)
        # 5Y annual swap.
        fixed_dates = jnp.array(
            [
                int(_make_date(2026, 1, 1)),
                int(_make_date(2027, 1, 1)),
                int(_make_date(2028, 1, 1)),
                int(_make_date(2029, 1, 1)),
                int(_make_date(2030, 1, 1)),
            ],
            dtype=jnp.int32,
        )
        # Compute the par rate: rate = (DF(start) - DF(maturity)) / annuity.
        starts = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], fixed_dates[:-1]]
        )
        taus = year_fraction(starts, fixed_dates, "act_360")
        annuity = float(jnp.sum(taus * curve(fixed_dates)))
        par_rate = float((curve(REF) - curve(fixed_dates[-1])) / annuity)

        swap = SwapRate(
            start_date=REF,
            fixed_dates=fixed_dates,
            rate=jnp.asarray(par_rate),
            day_count="act_360",
        )
        r = swap.residual(graph, NO_FIXINGS, REF)
        assert abs(float(r)) < 1e-10

    def test_residual_nonzero_off_par(self):
        curve = _flat_continuously_compounded(0.05)
        graph = _wrap_default(curve)
        fixed_dates = jnp.array(
            [
                int(_make_date(2026, 1, 1)),
                int(_make_date(2027, 1, 1)),
                int(_make_date(2028, 1, 1)),
                int(_make_date(2029, 1, 1)),
                int(_make_date(2030, 1, 1)),
            ],
            dtype=jnp.int32,
        )
        starts = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], fixed_dates[:-1]]
        )
        taus = year_fraction(starts, fixed_dates, "act_360")
        annuity = float(jnp.sum(taus * curve(fixed_dates)))
        par_rate = float((curve(REF) - curve(fixed_dates[-1])) / annuity)

        # Quote 50 bp above par: residual should be the over-paid annuity.
        swap = SwapRate(
            start_date=REF,
            fixed_dates=fixed_dates,
            rate=jnp.asarray(par_rate + 0.0050),
            day_count="act_360",
        )
        r = float(swap.residual(graph, NO_FIXINGS, REF))
        # Sign and magnitude: residual = (rate - par) * annuity.
        expected = 0.0050 * annuity
        assert r == pytest.approx(expected, rel=1e-6)


# ── OISSwapRate residual ─────────────────────────────────────────────


class TestOISSwapRateResidual:
    """OISSwapRate uses the float-leg telescoping identity, so the
    residual is structurally identical to SwapRate in the single-curve
    case.  These tests pin that contract and cross-check against the
    existing :func:`ois_swap_price` pricer.
    """

    def _ois_swap_at_par(self, curve: DiscountCurve):
        """Construct an OIS swap at the par rate implied by ``curve``."""
        fixed_dates = jnp.array(
            [
                int(_make_date(2026, 1, 1)),
                int(_make_date(2027, 1, 1)),
                int(_make_date(2028, 1, 1)),
                int(_make_date(2029, 1, 1)),
                int(_make_date(2030, 1, 1)),
            ],
            dtype=jnp.int32,
        )
        starts = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], fixed_dates[:-1]]
        )
        taus = year_fraction(starts, fixed_dates, "act_360")
        annuity = float(jnp.sum(taus * curve(fixed_dates)))
        par_rate = float((curve(REF) - curve(fixed_dates[-1])) / annuity)
        return (
            OISSwapRate(
                start_date=REF,
                fixed_dates=fixed_dates,
                rate=jnp.asarray(par_rate),
                day_count="act_360",
                index_id="USD.SOFR",
            ),
            fixed_dates,
            par_rate,
        )

    def test_residual_zero_at_par_rate(self):
        curve = _flat_continuously_compounded(0.04)
        graph = _wrap_default(curve)
        swap, _, _ = self._ois_swap_at_par(curve)
        r = swap.residual(graph, NO_FIXINGS, REF)
        assert abs(float(r)) < 1e-10

    def test_residual_off_par(self):
        """50 bp above par produces residual ≈ 0.0050 × annuity."""
        curve = _flat_continuously_compounded(0.04)
        graph = _wrap_default(curve)
        swap, fixed_dates, par_rate = self._ois_swap_at_par(curve)
        starts = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], fixed_dates[:-1]]
        )
        taus = year_fraction(starts, fixed_dates, "act_360")
        annuity = float(jnp.sum(taus * curve(fixed_dates)))
        off_par = OISSwapRate(
            start_date=REF,
            fixed_dates=fixed_dates,
            rate=jnp.asarray(par_rate + 0.0050),
            day_count="act_360",
            index_id="USD.SOFR",
        )
        r = float(off_par.residual(graph, NO_FIXINGS, REF))
        expected = 0.0050 * annuity
        assert r == pytest.approx(expected, rel=1e-6)

    def test_curves_touched_default_and_override(self):
        # Default sentinel id.
        s_default = OISSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(
                [int(_make_date(2026, 1, 1))], dtype=jnp.int32
            ),
            rate=jnp.asarray(0.04),
        )
        assert s_default.curves_touched == ("_default_",)
        assert s_default.index_id == "OIS"

        # Multi-curve override.
        s_named = OISSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(
                [int(_make_date(2026, 1, 1))], dtype=jnp.int32
            ),
            rate=jnp.asarray(0.04),
            curves_touched=("USD.SOFR.OIS",),
            index_id="USD.SOFR",
        )
        assert s_named.curves_touched == ("USD.SOFR.OIS",)
        assert s_named.index_id == "USD.SOFR"

    def test_cross_check_with_ois_swap_pricer(self):
        """OISSwapRate.residual on the calibrated curve agrees with
        :func:`ois_swap_price` divided by notional, when schedules match.
        """
        from valax.instruments.rates import OISSwap
        from valax.pricing.analytic.floating import ois_swap_price

        curve = _flat_continuously_compounded(0.04)
        graph = _wrap_default(curve)
        swap_quote, fixed_dates, par_rate = self._ois_swap_at_par(curve)

        # Bump 25 bp above par so both sides have a non-trivial value.
        bumped_quote = OISSwapRate(
            start_date=REF,
            fixed_dates=fixed_dates,
            rate=jnp.asarray(par_rate + 0.0025),
            day_count="act_360",
            index_id="USD.SOFR",
        )
        bumped_residual = float(
            bumped_quote.residual(graph, NO_FIXINGS, REF)
        )

        ois_swap = OISSwap(
            start_date=REF,
            fixed_dates=fixed_dates,
            float_dates=fixed_dates,  # share schedule for the cross-check
            fixed_rate=jnp.asarray(par_rate + 0.0025),
            notional=jnp.asarray(1.0),
            pay_fixed=True,
            day_count="act_360",
        )
        # ois_swap_price returns the payer NPV: float - fixed.
        # OISSwapRate.residual = fixed - float (rate * annuity - (DF_start - DF_end)).
        # So they have opposite sign.
        ois_npv = float(ois_swap_price(ois_swap, curve))
        assert bumped_residual == pytest.approx(-ois_npv, abs=1e-10)

    def test_jit_compat(self):
        curve = _flat_continuously_compounded(0.04)
        graph = _wrap_default(curve)
        swap, _, _ = self._ois_swap_at_par(curve)

        @jax.jit
        def f(g):
            return swap.residual(g, NO_FIXINGS, REF)

        r = f(graph)
        assert abs(float(r)) < 1e-10

    def test_fixings_currently_ignored(self):
        """The MVP residual ignores fixings (deferred for partially-
        seasoned swaps).  Passing a non-empty FixingHistory must not
        change the residual."""
        from valax.curves.fixings import FixingHistory, FixingSeries

        curve = _flat_continuously_compounded(0.04)
        graph = _wrap_default(curve)
        swap, _, _ = self._ois_swap_at_par(curve)

        non_empty = FixingHistory(
            indices={
                "USD.SOFR": FixingSeries(
                    fixing_dates=jnp.array(
                        [int(_make_date(2024, 12, 30))], dtype=jnp.int32
                    ),
                    fixings=jnp.array([0.0420], dtype=jnp.float64),
                )
            }
        )
        r_empty = float(swap.residual(graph, NO_FIXINGS, REF))
        r_with_fixings = float(swap.residual(graph, non_empty, REF))
        assert r_with_fixings == pytest.approx(r_empty, abs=1e-12)


# ── IborSwapRate residual ────────────────────────────────────────────


class TestIborSwapRateResidual:
    """Dual-curve IBOR swap quote.  Float leg projected from a tenor
    forward curve; both legs discounted with an OIS curve.  Residual
    is ``fixed_pv - float_pv`` (matches :class:`SwapRate` sign
    convention).
    """

    @staticmethod
    def _build_two_curve_graph(ois_rate=0.035, fwd_rate=0.040):
        """Construct (USD.SOFR.OIS, USD.SOFR.3M) at distinct flat rates."""
        ois = _flat_continuously_compounded(ois_rate)
        fwd = _flat_continuously_compounded(fwd_rate)
        return CurveGraph(
            curves={"USD.SOFR.OIS": ois, "USD.SOFR.3M": fwd}
        )

    @staticmethod
    def _annual_swap_schedule(years=5):
        """Annual fixed and float schedule out to ``years`` years."""
        dates = jnp.array(
            [
                int(_make_date(2025 + y + 1, 1, 1))
                for y in range(years)
            ],
            dtype=jnp.int32,
        )
        # Fixing date for each coupon = start of that period.
        fixing_dates = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], dates[:-1]]
        )
        return dates, fixing_dates

    def _par_rate_dual_curve(
        self,
        graph,
        fixed_dates,
        float_dates,
    ) -> float:
        """Compute the par fixed rate analytically from the dual-curve
        formula on a hand-built graph (no fixings)."""
        ois = graph["USD.SOFR.OIS"]
        fwd = graph["USD.SOFR.3M"]

        float_starts = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], float_dates[:-1]]
        )
        float_taus = year_fraction(float_starts, float_dates, "act_360")
        df_starts_fwd = fwd(float_starts)
        df_ends_fwd = fwd(float_dates)
        forwards = (df_starts_fwd / df_ends_fwd - 1.0) / float_taus
        df_float_disc = ois(float_dates)
        float_pv = float(jnp.sum(forwards * float_taus * df_float_disc))

        fixed_starts = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], fixed_dates[:-1]]
        )
        fixed_taus = year_fraction(fixed_starts, fixed_dates, "act_360")
        df_fixed_disc = ois(fixed_dates)
        annuity = float(jnp.sum(fixed_taus * df_fixed_disc))
        return float_pv / annuity

    def test_residual_zero_at_par_rate(self):
        graph = self._build_two_curve_graph()
        fixed_dates, fixing_dates = self._annual_swap_schedule(years=5)
        float_dates = fixed_dates  # share schedule for this test

        par = self._par_rate_dual_curve(graph, fixed_dates, float_dates)
        swap = IborSwapRate(
            start_date=REF,
            fixed_dates=fixed_dates,
            float_dates=float_dates,
            fixing_dates=fixing_dates,
            rate=jnp.asarray(par),
            fixed_day_count="act_360",
            float_day_count="act_360",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
            index_id="USD.SOFR.3M",
        )
        r = swap.residual(graph, NO_FIXINGS, REF)
        assert abs(float(r)) < 1e-10

    def test_residual_off_par_matches_annuity_times_rate_diff(self):
        """50 bp above par produces residual ≈ 0.0050 × annuity."""
        graph = self._build_two_curve_graph()
        fixed_dates, fixing_dates = self._annual_swap_schedule(years=5)
        float_dates = fixed_dates
        par = self._par_rate_dual_curve(graph, fixed_dates, float_dates)

        ois = graph["USD.SOFR.OIS"]
        fixed_starts = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], fixed_dates[:-1]]
        )
        fixed_taus = year_fraction(fixed_starts, fixed_dates, "act_360")
        annuity = float(jnp.sum(fixed_taus * ois(fixed_dates)))

        swap = IborSwapRate(
            start_date=REF,
            fixed_dates=fixed_dates,
            float_dates=float_dates,
            fixing_dates=fixing_dates,
            rate=jnp.asarray(par + 0.0050),
            fixed_day_count="act_360",
            float_day_count="act_360",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
        )
        r = float(swap.residual(graph, NO_FIXINGS, REF))
        assert r == pytest.approx(0.0050 * annuity, rel=1e-6)

    def test_dual_curve_reduces_to_single_when_curves_equal(self):
        """When discount == forward curve, the dual-curve residual
        equals the equivalent :class:`SwapRate` residual (single-curve
        telescoping form)."""
        curve = _flat_continuously_compounded(0.04)
        graph = CurveGraph(curves={"_default_": curve})
        fixed_dates, fixing_dates = self._annual_swap_schedule(years=3)

        rate = jnp.asarray(0.0399)
        swap_dual = IborSwapRate(
            start_date=REF,
            fixed_dates=fixed_dates,
            float_dates=fixed_dates,
            fixing_dates=fixing_dates,
            rate=rate,
            fixed_day_count="act_360",
            float_day_count="act_360",
            # Same id for both = single-curve degenerate case.
            curves_touched=("_default_", "_default_"),
        )
        swap_single = SwapRate(
            start_date=REF,
            fixed_dates=fixed_dates,
            rate=rate,
            day_count="act_360",
        )
        r_dual = float(swap_dual.residual(graph, NO_FIXINGS, REF))
        r_single = float(swap_single.residual(graph, NO_FIXINGS, REF))
        assert r_dual == pytest.approx(r_single, abs=1e-12)

    def test_residual_uses_forward_curve(self):
        """The forward curve genuinely participates in the residual:
        a swap that is at-par on graph A is off-par on graph B that
        shares OIS but has a different forward curve."""
        fixed_dates, fixing_dates = self._annual_swap_schedule(years=5)

        graph_a = self._build_two_curve_graph(ois_rate=0.035, fwd_rate=0.040)
        par_a = self._par_rate_dual_curve(
            graph_a, fixed_dates, fixed_dates
        )
        swap = IborSwapRate(
            start_date=REF,
            fixed_dates=fixed_dates,
            float_dates=fixed_dates,
            fixing_dates=fixing_dates,
            rate=jnp.asarray(par_a),
            fixed_day_count="act_360",
            float_day_count="act_360",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
        )

        # Same OIS, different forward curve.
        graph_b = CurveGraph(
            curves={
                "USD.SOFR.OIS": graph_a["USD.SOFR.OIS"],
                "USD.SOFR.3M": _flat_continuously_compounded(0.050),
            }
        )
        r_a = float(swap.residual(graph_a, NO_FIXINGS, REF))
        r_b = float(swap.residual(graph_b, NO_FIXINGS, REF))
        # At par on A, off par on B by ~(par_a - 5%) * annuity ≈
        # 100 bp × ~5 years ≈ a few % of notional.
        assert abs(r_a) < 1e-10
        assert abs(r_b) > 1e-3

    def test_off_par_residual_uses_ois_discount(self):
        """For an off-par swap (where DF_ois doesn't cancel across the
        legs), changing the OIS curve changes the residual.  At-par
        swaps with identical fixed/float schedules see no OIS effect
        because the discount factors cancel — that is a known
        property of dual-curve par swaps, *not* a sign that OIS is
        ignored."""
        fixed_dates, fixing_dates = self._annual_swap_schedule(years=5)
        graph_a = self._build_two_curve_graph(ois_rate=0.035, fwd_rate=0.040)
        par = self._par_rate_dual_curve(graph_a, fixed_dates, fixed_dates)

        # Quote 100 bp above par to make the residual depend on OIS.
        swap = IborSwapRate(
            start_date=REF,
            fixed_dates=fixed_dates,
            float_dates=fixed_dates,
            fixing_dates=fixing_dates,
            rate=jnp.asarray(par + 0.0100),
            fixed_day_count="act_360",
            float_day_count="act_360",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
        )

        # Same forward curve, different OIS.
        graph_b = CurveGraph(
            curves={
                "USD.SOFR.OIS": _flat_continuously_compounded(0.050),
                "USD.SOFR.3M": graph_a["USD.SOFR.3M"],
            }
        )
        r_a = float(swap.residual(graph_a, NO_FIXINGS, REF))
        r_b = float(swap.residual(graph_b, NO_FIXINGS, REF))
        # Both are non-zero (off-par); the OIS bump shifts annuity ≈
        # exp(-0.05*5)/exp(-0.035*5) ≈ 0.93×, so r_b ≈ 0.93 × r_a.
        # The exact ratio is not important — only that they differ.
        assert abs(r_a) > 1e-3
        assert abs(r_a - r_b) > 1e-4

    def test_partially_seasoned_uses_fixing_for_first_coupon(self):
        """First fixing already realised: residual differs from
        non-fixings case by exactly the expected amount."""
        graph = self._build_two_curve_graph()
        fixed_dates, fixing_dates = self._annual_swap_schedule(years=5)
        float_dates = fixed_dates

        par = self._par_rate_dual_curve(graph, fixed_dates, float_dates)
        swap = IborSwapRate(
            start_date=REF,
            fixed_dates=fixed_dates,
            float_dates=float_dates,
            fixing_dates=fixing_dates,
            rate=jnp.asarray(par),
            fixed_day_count="act_360",
            float_day_count="act_360",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
            index_id="USD.SOFR.3M",
        )

        # Compute the projected first-coupon rate from the curve.
        fwd = graph["USD.SOFR.3M"]
        first_start = REF
        first_end = float_dates[0]
        tau_first = float(year_fraction(first_start, first_end, "act_360"))
        df_start_fwd = float(fwd(first_start))
        df_end_fwd = float(fwd(first_end))
        projected_first = (df_start_fwd / df_end_fwd - 1.0) / tau_first

        # Override that projection with a realised value 100 bp above.
        realised_first = projected_first + 0.0100
        history = FixingHistory(
            indices={
                "USD.SOFR.3M": FixingSeries(
                    fixing_dates=jnp.array(
                        [int(REF)], dtype=jnp.int32
                    ),
                    fixings=jnp.array([realised_first]),
                )
            }
        )

        # With fixings the residual changes by exactly
        #     (projected - realised) * tau * DF_disc(first_end)
        # because the float leg's first coupon now uses ``realised``
        # instead of ``projected``, and the fixed leg is unchanged.
        ois = graph["USD.SOFR.OIS"]
        df_first_disc = float(ois(first_end))
        expected_delta = (projected_first - realised_first) * tau_first * df_first_disc

        r_no_fixings = float(swap.residual(graph, NO_FIXINGS, REF))
        r_with_fixings = float(swap.residual(graph, history, REF))
        assert (r_with_fixings - r_no_fixings) == pytest.approx(
            expected_delta, abs=1e-10
        )

    def test_no_fixings_for_index_falls_back_to_projection(self):
        """If FixingHistory contains other indices but not this one,
        all coupons project from the curve (no override)."""
        graph = self._build_two_curve_graph()
        fixed_dates, fixing_dates = self._annual_swap_schedule(years=3)
        float_dates = fixed_dates
        par = self._par_rate_dual_curve(graph, fixed_dates, float_dates)
        swap = IborSwapRate(
            start_date=REF,
            fixed_dates=fixed_dates,
            float_dates=float_dates,
            fixing_dates=fixing_dates,
            rate=jnp.asarray(par),
            fixed_day_count="act_360",
            float_day_count="act_360",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
            index_id="USD.SOFR.3M",
        )

        # FixingHistory has a different index — should be ignored.
        history = FixingHistory(
            indices={
                "USD.SOFR.6M": FixingSeries(
                    fixing_dates=jnp.array(
                        [int(REF)], dtype=jnp.int32
                    ),
                    fixings=jnp.array([0.10]),  # arbitrary, irrelevant
                )
            }
        )
        r_empty = float(swap.residual(graph, NO_FIXINGS, REF))
        r_other = float(swap.residual(graph, history, REF))
        assert r_other == pytest.approx(r_empty, abs=1e-12)

    def test_curves_touched_2tuple_default_and_override(self):
        s_default = IborSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(
                [int(_make_date(2026, 1, 1))], dtype=jnp.int32
            ),
            float_dates=jnp.array(
                [int(_make_date(2026, 1, 1))], dtype=jnp.int32
            ),
            fixing_dates=jnp.array([int(REF)], dtype=jnp.int32),
            rate=jnp.asarray(0.04),
        )
        assert s_default.curves_touched == (
            "_default_",
            "_default_",
        )
        assert s_default.index_id == "IBOR"

        s_named = IborSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(
                [int(_make_date(2026, 1, 1))], dtype=jnp.int32
            ),
            float_dates=jnp.array(
                [int(_make_date(2026, 1, 1))], dtype=jnp.int32
            ),
            fixing_dates=jnp.array([int(REF)], dtype=jnp.int32),
            rate=jnp.asarray(0.04),
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
            index_id="USD.SOFR.3M",
        )
        assert s_named.curves_touched == ("USD.SOFR.OIS", "USD.SOFR.3M")
        assert s_named.index_id == "USD.SOFR.3M"

    def test_jit_compat(self):
        graph = self._build_two_curve_graph()
        fixed_dates, fixing_dates = self._annual_swap_schedule(years=3)
        par = self._par_rate_dual_curve(graph, fixed_dates, fixed_dates)
        swap = IborSwapRate(
            start_date=REF,
            fixed_dates=fixed_dates,
            float_dates=fixed_dates,
            fixing_dates=fixing_dates,
            rate=jnp.asarray(par),
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
        )

        @jax.jit
        def f(g):
            return swap.residual(g, NO_FIXINGS, REF)

        r = f(graph)
        assert abs(float(r)) < 1e-10


# ── FXForward residual ───────────────────────────────────────────────


class TestFXForwardResidual:
    """FX forward residual is the deviation from covered interest
    rate parity.  Both DFs are evaluated at settle_date.
    """

    @staticmethod
    def _fx_graph(usd_rate=0.035, eur_rate=0.020):
        return CurveGraph(
            curves={
                "USD.SOFR.OIS": _flat_continuously_compounded(usd_rate),
                "EUR.ESTR.OIS": _flat_continuously_compounded(eur_rate),
            }
        )

    @staticmethod
    def _cip_forward(
        graph: CurveGraph, settle_date, fx_spot: float
    ) -> float:
        """Closed-form CIP forward rate."""
        df_dom = float(graph["USD.SOFR.OIS"](settle_date))
        df_for = float(graph["EUR.ESTR.OIS"](settle_date))
        return fx_spot * df_for / df_dom

    def test_residual_zero_at_cip_quote(self):
        graph = self._fx_graph()
        settle = _make_date(2026, 1, 1)
        spot = 1.10  # USD per EUR
        cip_fwd = self._cip_forward(graph, settle, spot)
        fxf = FXForward(
            value_date=_make_date(2025, 1, 3),
            settle_date=settle,
            quoted_forward=jnp.asarray(cip_fwd),
            fx_spot=jnp.asarray(spot),
            curves_touched=("USD.SOFR.OIS", "EUR.ESTR.OIS"),
        )
        r = fxf.residual(graph, NO_FIXINGS, REF)
        assert abs(float(r)) < 1e-12

    def test_residual_off_quote_changes_linearly(self):
        """Bumping the quoted forward by Δ shifts the residual by Δ."""
        graph = self._fx_graph()
        settle = _make_date(2026, 1, 1)
        spot = 1.10
        cip_fwd = self._cip_forward(graph, settle, spot)
        fxf = FXForward(
            value_date=_make_date(2025, 1, 3),
            settle_date=settle,
            quoted_forward=jnp.asarray(cip_fwd + 0.0010),  # +10 pips
            fx_spot=jnp.asarray(spot),
            curves_touched=("USD.SOFR.OIS", "EUR.ESTR.OIS"),
        )
        r = float(fxf.residual(graph, NO_FIXINGS, REF))
        assert r == pytest.approx(0.0010, abs=1e-12)

    def test_changing_fx_spot_breaks_cip(self):
        """Holding the quote fixed but changing spot makes residual
        non-zero by the expected amount."""
        graph = self._fx_graph()
        settle = _make_date(2026, 1, 1)
        cip_fwd = self._cip_forward(graph, settle, 1.10)
        fxf_bumped_spot = FXForward(
            value_date=_make_date(2025, 1, 3),
            settle_date=settle,
            quoted_forward=jnp.asarray(cip_fwd),
            fx_spot=jnp.asarray(1.20),  # bumped from 1.10
            curves_touched=("USD.SOFR.OIS", "EUR.ESTR.OIS"),
        )
        r = float(fxf_bumped_spot.residual(graph, NO_FIXINGS, REF))
        # Residual = cip_fwd_at_1.10 - 1.20 * df_for/df_dom
        #          = 1.10 * R - 1.20 * R = -0.10 * R, where R = df_for/df_dom.
        df_dom = float(graph["USD.SOFR.OIS"](settle))
        df_for = float(graph["EUR.ESTR.OIS"](settle))
        expected = cip_fwd - 1.20 * df_for / df_dom
        assert r == pytest.approx(expected, abs=1e-12)

    def test_changing_curves_changes_residual(self):
        """Holding spot and quote fixed, bumping the foreign curve
        changes the residual by the expected amount."""
        graph_a = self._fx_graph(usd_rate=0.035, eur_rate=0.020)
        settle = _make_date(2026, 1, 1)
        spot = 1.10
        cip_fwd_a = self._cip_forward(graph_a, settle, spot)
        fxf = FXForward(
            value_date=_make_date(2025, 1, 3),
            settle_date=settle,
            quoted_forward=jnp.asarray(cip_fwd_a),
            fx_spot=jnp.asarray(spot),
            curves_touched=("USD.SOFR.OIS", "EUR.ESTR.OIS"),
        )
        # Same USD curve, different EUR curve.
        graph_b = CurveGraph(
            curves={
                "USD.SOFR.OIS": graph_a["USD.SOFR.OIS"],
                "EUR.ESTR.OIS": _flat_continuously_compounded(0.030),
            }
        )
        r_a = float(fxf.residual(graph_a, NO_FIXINGS, REF))
        r_b = float(fxf.residual(graph_b, NO_FIXINGS, REF))
        assert abs(r_a) < 1e-12
        assert abs(r_b) > 1e-4  # measurably non-zero

    def test_curves_touched_default_and_override(self):
        f_default = FXForward(
            value_date=_make_date(2025, 1, 3),
            settle_date=_make_date(2026, 1, 1),
            quoted_forward=jnp.asarray(1.10),
            fx_spot=jnp.asarray(1.10),
        )
        assert f_default.curves_touched == (
            "_default_",
            "_default_",
        )

        f_named = FXForward(
            value_date=_make_date(2025, 1, 3),
            settle_date=_make_date(2026, 1, 1),
            quoted_forward=jnp.asarray(1.10),
            fx_spot=jnp.asarray(1.10),
            curves_touched=("USD.SOFR.OIS", "EUR.ESTR.OIS"),
        )
        assert f_named.curves_touched == (
            "USD.SOFR.OIS",
            "EUR.ESTR.OIS",
        )

    def test_jit_compat(self):
        graph = self._fx_graph()
        settle = _make_date(2026, 1, 1)
        spot = 1.10
        cip_fwd = self._cip_forward(graph, settle, spot)
        fxf = FXForward(
            value_date=_make_date(2025, 1, 3),
            settle_date=settle,
            quoted_forward=jnp.asarray(cip_fwd),
            fx_spot=jnp.asarray(spot),
            curves_touched=("USD.SOFR.OIS", "EUR.ESTR.OIS"),
        )

        @jax.jit
        def f(g):
            return fxf.residual(g, NO_FIXINGS, REF)

        r = f(graph)
        assert abs(float(r)) < 1e-12


# ── FXSwap residual ──────────────────────────────────────────────────


class TestFXSwapResidual:
    """FXSwap residual is the CIP relation between near and far legs.
    Reduces to FXForward when ``near_date == ref_date``.
    """

    @staticmethod
    def _fx_graph():
        return CurveGraph(
            curves={
                "USD.SOFR.OIS": _flat_continuously_compounded(0.035),
                "EUR.ESTR.OIS": _flat_continuously_compounded(0.020),
            }
        )

    def test_residual_zero_at_cip(self):
        graph = self._fx_graph()
        near = _make_date(2025, 1, 3)
        far = _make_date(2026, 1, 5)
        near_rate = 1.10
        df_dom_near = float(graph["USD.SOFR.OIS"](near))
        df_for_near = float(graph["EUR.ESTR.OIS"](near))
        df_dom_far = float(graph["USD.SOFR.OIS"](far))
        df_for_far = float(graph["EUR.ESTR.OIS"](far))
        # Solve far_rate * df_for_near * df_dom_far =
        #       near_rate * df_dom_near * df_for_far for far_rate.
        far_rate = (
            near_rate * df_dom_near * df_for_far
            / (df_for_near * df_dom_far)
        )
        swap = FXSwap(
            near_date=near,
            far_date=far,
            near_rate=jnp.asarray(near_rate),
            far_rate=jnp.asarray(far_rate),
            curves_touched=("USD.SOFR.OIS", "EUR.ESTR.OIS"),
        )
        r = swap.residual(graph, NO_FIXINGS, REF)
        assert abs(float(r)) < 1e-12

    def test_collapses_to_fx_forward_when_near_is_ref(self):
        """When ``near_date == ref_date``, both near DFs equal 1 and
        the FXSwap residual is identical to FXForward's, with
        ``near_rate`` playing the role of ``fx_spot``."""
        graph = self._fx_graph()
        far = _make_date(2026, 1, 1)
        spot = 1.10
        df_dom_far = float(graph["USD.SOFR.OIS"](far))
        df_for_far = float(graph["EUR.ESTR.OIS"](far))
        cip_fwd = spot * df_for_far / df_dom_far

        swap = FXSwap(
            near_date=REF,  # near = ref → DFs at near = 1
            far_date=far,
            near_rate=jnp.asarray(spot),
            far_rate=jnp.asarray(cip_fwd),
            curves_touched=("USD.SOFR.OIS", "EUR.ESTR.OIS"),
        )
        fxf = FXForward(
            value_date=REF,
            settle_date=far,
            quoted_forward=jnp.asarray(cip_fwd),
            fx_spot=jnp.asarray(spot),
            curves_touched=("USD.SOFR.OIS", "EUR.ESTR.OIS"),
        )
        r_swap = float(swap.residual(graph, NO_FIXINGS, REF))
        r_fxf = float(fxf.residual(graph, NO_FIXINGS, REF))
        assert abs(r_swap) < 1e-12
        assert abs(r_fxf) < 1e-12
        # Both zero — consistent.

    def test_residual_off_par(self):
        """Bumping ``far_rate`` by Δ moves residual by
        Δ * df_for_near * df_dom_far."""
        graph = self._fx_graph()
        near = _make_date(2025, 1, 3)
        far = _make_date(2026, 1, 5)
        near_rate = 1.10
        df_dom_near = float(graph["USD.SOFR.OIS"](near))
        df_for_near = float(graph["EUR.ESTR.OIS"](near))
        df_dom_far = float(graph["USD.SOFR.OIS"](far))
        df_for_far = float(graph["EUR.ESTR.OIS"](far))
        cip_far = (
            near_rate * df_dom_near * df_for_far
            / (df_for_near * df_dom_far)
        )
        bump = 0.0050
        swap = FXSwap(
            near_date=near,
            far_date=far,
            near_rate=jnp.asarray(near_rate),
            far_rate=jnp.asarray(cip_far + bump),
            curves_touched=("USD.SOFR.OIS", "EUR.ESTR.OIS"),
        )
        r = float(swap.residual(graph, NO_FIXINGS, REF))
        expected = bump * df_for_near * df_dom_far
        assert r == pytest.approx(expected, abs=1e-12)

    def test_jit_compat(self):
        graph = self._fx_graph()
        near = _make_date(2025, 1, 3)
        far = _make_date(2026, 1, 5)
        near_rate = 1.10
        df_dom_near = float(graph["USD.SOFR.OIS"](near))
        df_for_near = float(graph["EUR.ESTR.OIS"](near))
        df_dom_far = float(graph["USD.SOFR.OIS"](far))
        df_for_far = float(graph["EUR.ESTR.OIS"](far))
        far_rate = (
            near_rate * df_dom_near * df_for_far
            / (df_for_near * df_dom_far)
        )
        swap = FXSwap(
            near_date=near,
            far_date=far,
            near_rate=jnp.asarray(near_rate),
            far_rate=jnp.asarray(far_rate),
            curves_touched=("USD.SOFR.OIS", "EUR.ESTR.OIS"),
        )

        @jax.jit
        def f(g):
            return swap.residual(g, NO_FIXINGS, REF)

        r = f(graph)
        assert abs(float(r)) < 1e-12


# ── CrossCurrencyBasisSwap residual ──────────────────────────────────


class TestCrossCurrencyBasisSwapResidual:
    """Cross-currency basis swap with MTM and constant-notional
    variants.  Residual differs structurally between the two; both
    are exercised against hand-built four-curve graphs.
    """

    @staticmethod
    def _ccbs_graph(
        usd_ois=0.040,
        usd_fwd=0.045,
        eur_ois=0.020,
        eur_fwd=0.025,
    ):
        return CurveGraph(
            curves={
                "USD.SOFR.OIS": _flat_continuously_compounded(usd_ois),
                "USD.SOFR.3M": _flat_continuously_compounded(usd_fwd),
                "EUR.ESTR.OIS": _flat_continuously_compounded(eur_ois),
                "EUR.EURIBOR.3M": _flat_continuously_compounded(eur_fwd),
            }
        )

    @staticmethod
    def _quarterly_dates(years=5):
        """Quarterly dates from REF + 3M to REF + years*12M."""
        dates = []
        for q in range(1, years * 4 + 1):
            year = 2025 + (q // 4)
            month = (q * 3) % 12 + 1
            if (q * 3) % 12 == 0:
                month = 1
                year = 2025 + (q // 4)
            dates.append(int(_make_date(year, month, 1)))
        return jnp.array(sorted(set(dates)), dtype=jnp.int32)

    @staticmethod
    def _annual_dates(years=5):
        return jnp.array(
            [int(_make_date(2025 + y + 1, 1, 1)) for y in range(years)],
            dtype=jnp.int32,
        )

    def _build_swap(
        self,
        spread,
        spread_on_leg="domestic",
        variant="constant_notional",
        years=5,
    ):
        dom_dates = self._annual_dates(years)
        for_dates = self._annual_dates(years)
        dom_fixing_dates = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], dom_dates[:-1]]
        )
        for_fixing_dates = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], for_dates[:-1]]
        )
        return CrossCurrencyBasisSwap(
            start_date=REF,
            dom_dates=dom_dates,
            dom_fixing_dates=dom_fixing_dates,
            for_dates=for_dates,
            for_fixing_dates=for_fixing_dates,
            fx_spot=jnp.asarray(1.10),
            spread=jnp.asarray(spread),
            dom_day_count="act_360",
            dom_index_id="USD.SOFR.3M",
            for_day_count="act_360",
            for_index_id="EUR.EURIBOR.3M",
            spread_on_leg=spread_on_leg,
            variant=variant,
            curves_touched=(
                "USD.SOFR.OIS",
                "USD.SOFR.3M",
                "EUR.ESTR.OIS",
                "EUR.EURIBOR.3M",
            ),
        )

    @staticmethod
    def _solve_par_spread(swap, graph):
        """Newton-style: residual is linear in spread (annuity slope),
        so par = spread - residual / (∂residual/∂spread)."""
        # We can solve analytically: residual(s) = residual(0) + s * annuity.
        # First evaluate at spread=0, then evaluate at spread=1 to extract
        # the annuity slope.
        zero_swap = eqx.tree_at(
            lambda s: s.spread, swap, jnp.asarray(0.0)
        )
        one_swap = eqx.tree_at(
            lambda s: s.spread, swap, jnp.asarray(1.0)
        )
        r0 = float(zero_swap.residual(graph, NO_FIXINGS, REF))
        r1 = float(one_swap.residual(graph, NO_FIXINGS, REF))
        slope = r1 - r0
        return -r0 / slope

    def test_constant_notional_residual_zero_at_par(self):
        graph = self._ccbs_graph()
        swap = self._build_swap(
            spread=0.0, variant="constant_notional"
        )
        par_spread = self._solve_par_spread(swap, graph)
        swap_at_par = eqx.tree_at(
            lambda s: s.spread, swap, jnp.asarray(par_spread)
        )
        r = swap_at_par.residual(graph, NO_FIXINGS, REF)
        assert abs(float(r)) < 1e-10

    def test_mtm_residual_zero_at_par(self):
        graph = self._ccbs_graph()
        swap = self._build_swap(spread=0.0, variant="mtm")
        par_spread = self._solve_par_spread(swap, graph)
        swap_at_par = eqx.tree_at(
            lambda s: s.spread, swap, jnp.asarray(par_spread)
        )
        r = swap_at_par.residual(graph, NO_FIXINGS, REF)
        assert abs(float(r)) < 1e-10

    def test_variant_distinction(self):
        """The MTM and constant-notional variants give measurably
        different par spreads on the same graph.  A swap at the
        constant-notional par spread under the constant-notional
        variant has residual ~0, but under MTM at the same numerical
        spread the residual is non-zero (and vice versa)."""
        graph = self._ccbs_graph()
        cn_solve_swap = self._build_swap(
            spread=0.0, variant="constant_notional"
        )
        cn_par = self._solve_par_spread(cn_solve_swap, graph)

        # Construct two swaps directly with the same spread but
        # different ``variant`` static fields.  Cannot use
        # ``eqx.tree_at`` on a static field; must reconstruct.
        cn_at_par = self._build_swap(
            spread=cn_par, variant="constant_notional"
        )
        mtm_at_cn_par = self._build_swap(
            spread=cn_par, variant="mtm"
        )
        r_cn = float(cn_at_par.residual(graph, NO_FIXINGS, REF))
        r_mtm = float(mtm_at_cn_par.residual(graph, NO_FIXINGS, REF))
        assert abs(r_cn) < 1e-10
        assert abs(r_mtm) > 1e-4  # genuinely different equation

    def test_off_par_residual_scales_with_spread(self):
        """20 bp above the constant-notional par spread shifts the
        residual by 20 bp × domestic annuity_disc."""
        graph = self._ccbs_graph()
        swap = self._build_swap(
            spread=0.0, variant="constant_notional"
        )
        par_spread = self._solve_par_spread(swap, graph)

        ois = graph["USD.SOFR.OIS"]
        dom_dates = self._annual_dates(5)
        starts = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], dom_dates[:-1]]
        )
        taus = year_fraction(starts, dom_dates, "act_360")
        annuity_disc = float(jnp.sum(taus * ois(dom_dates)))

        bumped = eqx.tree_at(
            lambda s: s.spread,
            swap,
            jnp.asarray(par_spread + 0.0020),
        )
        r = float(bumped.residual(graph, NO_FIXINGS, REF))
        assert r == pytest.approx(0.0020 * annuity_disc, rel=1e-6)

    def test_invalid_spread_on_leg_raises(self):
        graph = self._ccbs_graph()
        swap = self._build_swap(
            spread=0.001, spread_on_leg="x"
        )  # invalid
        with pytest.raises(ValueError, match="spread_on_leg"):
            swap.residual(graph, NO_FIXINGS, REF)

    def test_invalid_variant_raises(self):
        graph = self._ccbs_graph()
        swap = self._build_swap(
            spread=0.001, variant="exotic"
        )  # invalid
        with pytest.raises(ValueError, match="variant"):
            swap.residual(graph, NO_FIXINGS, REF)

    def test_curves_touched_4tuple_default(self):
        swap = CrossCurrencyBasisSwap(
            start_date=REF,
            dom_dates=jnp.array([int(_make_date(2026, 1, 1))], dtype=jnp.int32),
            dom_fixing_dates=jnp.array([int(REF)], dtype=jnp.int32),
            for_dates=jnp.array([int(_make_date(2026, 1, 1))], dtype=jnp.int32),
            for_fixing_dates=jnp.array([int(REF)], dtype=jnp.int32),
            fx_spot=jnp.asarray(1.10),
        )
        assert swap.curves_touched == (
            "_default_",
            "_default_",
            "_default_",
            "_default_",
        )
        assert swap.variant == "constant_notional"
        assert swap.spread_on_leg == "domestic"

    def test_jit_compat(self):
        graph = self._ccbs_graph()
        swap = self._build_swap(spread=0.0, variant="mtm")
        par = self._solve_par_spread(swap, graph)
        swap_at_par = eqx.tree_at(
            lambda s: s.spread, swap, jnp.asarray(par)
        )

        @jax.jit
        def f(g):
            return swap_at_par.residual(g, NO_FIXINGS, REF)

        r = f(graph)
        assert abs(float(r)) < 1e-10


# ── TenorBasisSwap residual ──────────────────────────────────────────


class TestTenorBasisSwapResidual:
    """Two floating legs on the same currency / different tenors,
    discounted with a shared OIS curve, with a basis spread on one
    leg.  Residual is ``leg_a_pv - leg_b_pv``.
    """

    @staticmethod
    def _three_curve_graph(ois=0.035, fwd_a=0.040, fwd_b=0.042):
        return CurveGraph(
            curves={
                "USD.SOFR.OIS": _flat_continuously_compounded(ois),
                "USD.SOFR.3M": _flat_continuously_compounded(fwd_a),
                "USD.SOFR.6M": _flat_continuously_compounded(fwd_b),
            }
        )

    @staticmethod
    def _annual_schedule(years=5):
        dates = jnp.array(
            [int(_make_date(2025 + y + 1, 1, 1)) for y in range(years)],
            dtype=jnp.int32,
        )
        fixing_dates = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], dates[:-1]]
        )
        return dates, fixing_dates

    def _par_basis_spread(
        self, graph, dates_a, dates_b, dc_a="act_360", dc_b="act_360"
    ) -> float:
        """Closed-form par basis spread (added to leg A) given identical
        OIS discounting and arbitrary forward curves."""
        ois = graph["USD.SOFR.OIS"]
        fwd_a = graph["USD.SOFR.3M"]
        fwd_b = graph["USD.SOFR.6M"]

        a_starts = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], dates_a[:-1]]
        )
        a_taus = year_fraction(a_starts, dates_a, dc_a)
        a_fwds = (fwd_a(a_starts) / fwd_a(dates_a) - 1.0) / a_taus
        df_disc_a = ois(dates_a)
        a_pv_no_spread = float(jnp.sum(a_fwds * a_taus * df_disc_a))
        a_annuity = float(jnp.sum(a_taus * df_disc_a))

        b_starts = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], dates_b[:-1]]
        )
        b_taus = year_fraction(b_starts, dates_b, dc_b)
        b_fwds = (fwd_b(b_starts) / fwd_b(dates_b) - 1.0) / b_taus
        df_disc_b = ois(dates_b)
        b_pv_no_spread = float(jnp.sum(b_fwds * b_taus * df_disc_b))

        # Solve a_pv_no_spread + s_a * a_annuity = b_pv_no_spread.
        return (b_pv_no_spread - a_pv_no_spread) / a_annuity

    def test_residual_zero_at_par_basis(self):
        graph = self._three_curve_graph()
        dates, fixing_dates = self._annual_schedule(years=5)
        par_spread = self._par_basis_spread(graph, dates, dates)
        swap = TenorBasisSwap(
            start_date=REF,
            leg_a_dates=dates,
            leg_a_fixing_dates=fixing_dates,
            leg_a_index_id="USD.SOFR.3M",
            leg_b_dates=dates,
            leg_b_fixing_dates=fixing_dates,
            leg_b_index_id="USD.SOFR.6M",
            spread=jnp.asarray(par_spread),
            spread_on_leg="a",
            curves_touched=(
                "USD.SOFR.OIS",
                "USD.SOFR.3M",
                "USD.SOFR.6M",
            ),
        )
        r = swap.residual(graph, NO_FIXINGS, REF)
        assert abs(float(r)) < 1e-10

    def test_residual_off_par(self):
        """20 bp above par adds 20 bp × annuity to the leg-A side
        of the residual."""
        graph = self._three_curve_graph()
        dates, fixing_dates = self._annual_schedule(years=5)
        par_spread = self._par_basis_spread(graph, dates, dates)

        ois = graph["USD.SOFR.OIS"]
        starts = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], dates[:-1]]
        )
        taus = year_fraction(starts, dates, "act_360")
        annuity = float(jnp.sum(taus * ois(dates)))

        swap = TenorBasisSwap(
            start_date=REF,
            leg_a_dates=dates,
            leg_a_fixing_dates=fixing_dates,
            leg_a_index_id="USD.SOFR.3M",
            leg_b_dates=dates,
            leg_b_fixing_dates=fixing_dates,
            leg_b_index_id="USD.SOFR.6M",
            spread=jnp.asarray(par_spread + 0.0020),
            spread_on_leg="a",
            curves_touched=(
                "USD.SOFR.OIS",
                "USD.SOFR.3M",
                "USD.SOFR.6M",
            ),
        )
        r = float(swap.residual(graph, NO_FIXINGS, REF))
        # +20 bp on leg A increases leg-A PV by 0.0020 × annuity.
        assert r == pytest.approx(0.0020 * annuity, rel=1e-6)

    def test_spread_on_leg_b_flips_sign(self):
        """The same magnitude of spread, applied to leg B instead of
        leg A, changes the residual by the expected signed amount."""
        graph = self._three_curve_graph()
        dates, fixing_dates = self._annual_schedule(years=5)
        ois = graph["USD.SOFR.OIS"]
        starts = jnp.concatenate(
            [jnp.asarray(REF, dtype=jnp.int32)[None], dates[:-1]]
        )
        taus = year_fraction(starts, dates, "act_360")
        annuity = float(jnp.sum(taus * ois(dates)))

        # Build a swap at "par on leg A" (residual = 0).  Then build
        # the same swap with the spread moved to leg B (also at the
        # par value on A — i.e., the spread is no longer at the
        # par-on-B value): residual changes by ``-spread * annuity``
        # because the spread leg flipped from A to B.
        par_spread = self._par_basis_spread(graph, dates, dates)
        swap_a = TenorBasisSwap(
            start_date=REF,
            leg_a_dates=dates,
            leg_a_fixing_dates=fixing_dates,
            leg_a_index_id="USD.SOFR.3M",
            leg_b_dates=dates,
            leg_b_fixing_dates=fixing_dates,
            leg_b_index_id="USD.SOFR.6M",
            spread=jnp.asarray(par_spread),
            spread_on_leg="a",
            curves_touched=(
                "USD.SOFR.OIS",
                "USD.SOFR.3M",
                "USD.SOFR.6M",
            ),
        )
        swap_b = TenorBasisSwap(
            start_date=REF,
            leg_a_dates=dates,
            leg_a_fixing_dates=fixing_dates,
            leg_a_index_id="USD.SOFR.3M",
            leg_b_dates=dates,
            leg_b_fixing_dates=fixing_dates,
            leg_b_index_id="USD.SOFR.6M",
            spread=jnp.asarray(par_spread),
            spread_on_leg="b",
            curves_touched=(
                "USD.SOFR.OIS",
                "USD.SOFR.3M",
                "USD.SOFR.6M",
            ),
        )
        r_a = float(swap_a.residual(graph, NO_FIXINGS, REF))
        r_b = float(swap_b.residual(graph, NO_FIXINGS, REF))
        # r_a - r_b = par_spread * annuity + par_spread * annuity =
        # 2 * par_spread * annuity (A had +s on leg A; B has +s on
        # leg B which subtracts from the residual).
        assert r_a == pytest.approx(0.0, abs=1e-10)
        assert (r_a - r_b) == pytest.approx(
            2.0 * par_spread * annuity, rel=1e-6
        )

    def test_invalid_spread_on_leg_raises(self):
        graph = self._three_curve_graph()
        dates, fixing_dates = self._annual_schedule(years=2)
        swap = TenorBasisSwap(
            start_date=REF,
            leg_a_dates=dates,
            leg_a_fixing_dates=fixing_dates,
            leg_a_index_id="USD.SOFR.3M",
            leg_b_dates=dates,
            leg_b_fixing_dates=fixing_dates,
            leg_b_index_id="USD.SOFR.6M",
            spread=jnp.asarray(0.0010),
            spread_on_leg="x",  # invalid
            curves_touched=(
                "USD.SOFR.OIS",
                "USD.SOFR.3M",
                "USD.SOFR.6M",
            ),
        )
        with pytest.raises(ValueError, match="spread_on_leg"):
            swap.residual(graph, NO_FIXINGS, REF)

    def test_curves_touched_3tuple_default_and_override(self):
        dates, fixing_dates = self._annual_schedule(years=2)
        s_default = TenorBasisSwap(
            start_date=REF,
            leg_a_dates=dates,
            leg_a_fixing_dates=fixing_dates,
            leg_b_dates=dates,
            leg_b_fixing_dates=fixing_dates,
            spread=jnp.asarray(0.0010),
        )
        assert s_default.curves_touched == (
            "_default_",
            "_default_",
            "_default_",
        )
        assert s_default.spread_on_leg == "a"

        s_named = TenorBasisSwap(
            start_date=REF,
            leg_a_dates=dates,
            leg_a_fixing_dates=fixing_dates,
            leg_a_index_id="USD.SOFR.3M",
            leg_b_dates=dates,
            leg_b_fixing_dates=fixing_dates,
            leg_b_index_id="USD.SOFR.6M",
            spread=jnp.asarray(0.0010),
            curves_touched=(
                "USD.SOFR.OIS",
                "USD.SOFR.3M",
                "USD.SOFR.6M",
            ),
        )
        assert s_named.curves_touched == (
            "USD.SOFR.OIS",
            "USD.SOFR.3M",
            "USD.SOFR.6M",
        )

    def test_jit_compat(self):
        graph = self._three_curve_graph()
        dates, fixing_dates = self._annual_schedule(years=3)
        par_spread = self._par_basis_spread(graph, dates, dates)
        swap = TenorBasisSwap(
            start_date=REF,
            leg_a_dates=dates,
            leg_a_fixing_dates=fixing_dates,
            leg_a_index_id="USD.SOFR.3M",
            leg_b_dates=dates,
            leg_b_fixing_dates=fixing_dates,
            leg_b_index_id="USD.SOFR.6M",
            spread=jnp.asarray(par_spread),
            spread_on_leg="a",
            curves_touched=(
                "USD.SOFR.OIS",
                "USD.SOFR.3M",
                "USD.SOFR.6M",
            ),
        )

        @jax.jit
        def f(g):
            return swap.residual(g, NO_FIXINGS, REF)

        r = f(graph)
        assert abs(float(r)) < 1e-10


# ── MoneyMarketFuture residual ───────────────────────────────────────


class TestMoneyMarketFutureResidual:
    """The futures residual is ``F^fut - adj - F^curve(T0, T1)``.
    Tests cover both convexity-adjustment plug-ins shipped in
    MC-Curves-1 and confirm JIT-compatibility.
    """

    def test_residual_zero_no_convexity(self):
        """With ``no_convexity_adj``, residual = 0 when the curve's
        forward equals the quoted futures rate."""
        curve = _flat_continuously_compounded(0.05)
        graph = _wrap_default(curve)
        t0 = _make_date(2025, 6, 1)
        t1 = _make_date(2025, 9, 1)
        df0 = float(curve(t0))
        df1 = float(curve(t1))
        tau = float(year_fraction(t0, t1, "act_360"))
        forward_rate_curve = (df0 / df1 - 1.0) / tau
        future = MoneyMarketFuture(
            start_date=t0,
            end_date=t1,
            futures_rate=jnp.asarray(forward_rate_curve),
            day_count="act_360",
            convexity_adj_fn=no_convexity_adj(),
        )
        r = future.residual(graph, NO_FIXINGS, REF)
        assert abs(float(r)) < 1e-10

    def test_residual_zero_with_constant_adjustment(self):
        """With a 5 bp constant adjustment, residual = 0 when
        ``futures_rate = curve_forward + 0.0005``."""
        curve = _flat_continuously_compounded(0.05)
        graph = _wrap_default(curve)
        t0 = _make_date(2025, 6, 1)
        t1 = _make_date(2025, 9, 1)
        df0 = float(curve(t0))
        df1 = float(curve(t1))
        tau = float(year_fraction(t0, t1, "act_360"))
        forward_rate_curve = (df0 / df1 - 1.0) / tau
        future = MoneyMarketFuture(
            start_date=t0,
            end_date=t1,
            futures_rate=jnp.asarray(forward_rate_curve + 0.0005),
            day_count="act_360",
            convexity_adj_fn=constant_convexity_adj(5.0),
        )
        r = future.residual(graph, NO_FIXINGS, REF)
        assert abs(float(r)) < 1e-10

    def test_residual_off_quote(self):
        """5 bp above the curve's implied forward (with no adjustment)
        gives residual = +0.0005."""
        curve = _flat_continuously_compounded(0.05)
        graph = _wrap_default(curve)
        t0 = _make_date(2025, 6, 1)
        t1 = _make_date(2025, 9, 1)
        df0 = float(curve(t0))
        df1 = float(curve(t1))
        tau = float(year_fraction(t0, t1, "act_360"))
        forward_rate_curve = (df0 / df1 - 1.0) / tau
        future = MoneyMarketFuture(
            start_date=t0,
            end_date=t1,
            futures_rate=jnp.asarray(forward_rate_curve + 0.0005),
            day_count="act_360",
            convexity_adj_fn=no_convexity_adj(),
        )
        r = float(future.residual(graph, NO_FIXINGS, REF))
        assert r == pytest.approx(0.0005, abs=1e-10)

    def test_default_adjustment_is_no_op(self):
        """A future constructed without an explicit adjustment uses
        ``no_convexity_adj`` (defaults applied via default_factory)."""
        curve = _flat_continuously_compounded(0.05)
        graph = _wrap_default(curve)
        t0 = _make_date(2025, 6, 1)
        t1 = _make_date(2025, 9, 1)
        df0 = float(curve(t0))
        df1 = float(curve(t1))
        tau = float(year_fraction(t0, t1, "act_360"))
        forward_rate_curve = (df0 / df1 - 1.0) / tau
        future = MoneyMarketFuture(
            start_date=t0,
            end_date=t1,
            futures_rate=jnp.asarray(forward_rate_curve),
            day_count="act_360",
        )  # no convexity_adj_fn specified
        r = future.residual(graph, NO_FIXINGS, REF)
        assert abs(float(r)) < 1e-10

    def test_curves_touched_default_and_override(self):
        f_default = MoneyMarketFuture(
            start_date=_make_date(2025, 6, 1),
            end_date=_make_date(2025, 9, 1),
            futures_rate=jnp.asarray(0.05),
        )
        assert f_default.curves_touched == ("_default_",)

        f_named = MoneyMarketFuture(
            start_date=_make_date(2025, 6, 1),
            end_date=_make_date(2025, 9, 1),
            futures_rate=jnp.asarray(0.05),
            curves_touched=("USD.SOFR.3M",),
        )
        assert f_named.curves_touched == ("USD.SOFR.3M",)

    def test_jit_compat(self):
        curve = _flat_continuously_compounded(0.05)
        graph = _wrap_default(curve)
        t0 = _make_date(2025, 6, 1)
        t1 = _make_date(2025, 9, 1)
        df0 = float(curve(t0))
        df1 = float(curve(t1))
        tau = float(year_fraction(t0, t1, "act_360"))
        forward_rate_curve = (df0 / df1 - 1.0) / tau
        future = MoneyMarketFuture(
            start_date=t0,
            end_date=t1,
            futures_rate=jnp.asarray(forward_rate_curve + 0.0005),
            day_count="act_360",
            convexity_adj_fn=constant_convexity_adj(5.0),
        )

        @jax.jit
        def f(g):
            return future.residual(g, NO_FIXINGS, REF)

        r = f(graph)
        assert abs(float(r)) < 1e-10


# ── TurnInstrument residual ──────────────────────────────────────────


class TestTurnInstrumentResidual:
    """Turn instruments anchor the curve's forward rate over a short
    window straddling a calendar discontinuity (typically year-end).
    Tests construct a curve with two pillars tightly around the turn
    date, encoding a known forward, and verify the residual.
    """

    @staticmethod
    def _curve_with_turn_jump(window_forward_rate: float):
        """Build a curve with a sharp 2-day forward over Dec 31 / Jan 1
        equal to ``window_forward_rate`` (simply-compounded), and a
        flat ~3% surrounding forward.

        The curve has pillars at:
        - REF (DF=1)
        - 2025-12-30 (just before the turn window)
        - 2026-01-02 (just after the turn window)
        - 2027-01-01 (later anchor)

        The forward over [2025-12-30, 2026-01-02] (3-day window
        containing Dec 31) is set to ``window_forward_rate``.
        """
        ref = REF
        d_before = _make_date(2025, 12, 30)
        d_after = _make_date(2026, 1, 2)
        d_year = _make_date(2027, 1, 1)

        # Surrounding 3% flat continuously-compounded curve.
        t_before = float(d_before - int(ref)) / 365.0
        t_year = float(d_year - int(ref)) / 365.0
        df_before = float(jnp.exp(-0.03 * t_before))

        # Force the 3-day window forward to equal the prescribed value.
        # Simply-compounded forward over 3 days:
        #   F = (df_before / df_after - 1) / tau
        # => df_after = df_before / (1 + F * tau)
        tau = float(
            year_fraction(d_before, d_after, "act_360")
        )  # ≈ 3/360
        df_after = df_before / (1.0 + window_forward_rate * tau)

        # Continue flat 3% from d_after onward (df at d_year extrapolated
        # consistently with df_after).
        t_after = float(d_after - int(ref)) / 365.0
        delta_t = t_year - t_after
        df_year = df_after * float(jnp.exp(-0.03 * delta_t))

        pillars = jnp.array(
            [int(ref), int(d_before), int(d_after), int(d_year)],
            dtype=jnp.int32,
        )
        dfs = jnp.array(
            [1.0, df_before, df_after, df_year], dtype=jnp.float64
        )
        return DiscountCurve(
            pillar_dates=pillars,
            discount_factors=dfs,
            reference_date=ref,
            day_count="act_365",
        )

    def test_residual_zero_when_curve_matches(self):
        target = 0.0150  # 150 bp year-end spike
        curve = self._curve_with_turn_jump(target)
        graph = CurveGraph(curves={"USD.SOFR.OIS": curve})
        turn = TurnInstrument(
            turn_date=_make_date(2025, 12, 31),
            jump_size=jnp.asarray(target),
            accrual_days_before=1,
            accrual_days_after=2,
            day_count="act_360",
            curves_touched=("USD.SOFR.OIS",),
        )
        r = turn.residual(graph, NO_FIXINGS, REF)
        assert abs(float(r)) < 1e-10

    def test_residual_off_quote(self):
        """Curve encodes 150 bp; quoted jump 200 bp → residual = -50 bp."""
        curve_value = 0.0150
        curve = self._curve_with_turn_jump(curve_value)
        graph = CurveGraph(curves={"USD.SOFR.OIS": curve})
        quoted = 0.0200
        turn = TurnInstrument(
            turn_date=_make_date(2025, 12, 31),
            jump_size=jnp.asarray(quoted),
            accrual_days_before=1,
            accrual_days_after=2,
            day_count="act_360",
            curves_touched=("USD.SOFR.OIS",),
        )
        r = float(turn.residual(graph, NO_FIXINGS, REF))
        # residual = window_forward - quoted = curve_value - quoted = -50 bp.
        assert r == pytest.approx(curve_value - quoted, abs=1e-10)

    def test_curves_touched_default_and_override(self):
        t_default = TurnInstrument(
            turn_date=_make_date(2025, 12, 31),
            jump_size=jnp.asarray(0.015),
        )
        assert t_default.curves_touched == ("_default_",)
        assert t_default.accrual_days_before == 1
        assert t_default.accrual_days_after == 1

        t_named = TurnInstrument(
            turn_date=_make_date(2025, 12, 31),
            jump_size=jnp.asarray(0.015),
            accrual_days_before=2,
            accrual_days_after=3,
            curves_touched=("USD.SOFR.OIS",),
        )
        assert t_named.curves_touched == ("USD.SOFR.OIS",)
        assert t_named.accrual_days_before == 2
        assert t_named.accrual_days_after == 3

    def test_jit_compat(self):
        target = 0.0100
        curve = self._curve_with_turn_jump(target)
        graph = CurveGraph(curves={"USD.SOFR.OIS": curve})
        turn = TurnInstrument(
            turn_date=_make_date(2025, 12, 31),
            jump_size=jnp.asarray(target),
            accrual_days_before=1,
            accrual_days_after=2,
            day_count="act_360",
            curves_touched=("USD.SOFR.OIS",),
        )

        @jax.jit
        def f(g):
            return turn.residual(g, NO_FIXINGS, REF)

        r = f(graph)
        assert abs(float(r)) < 1e-10


# ── Round-trip: bootstrap_sequential output reprices to zero ─────────


class TestBootstrapResidualRoundTrip:
    """After bootstrap_sequential, every input instrument's residual on
    the resulting curve must be ~0.  This proves the migrated residual
    definition is consistent with the bootstrapper's notion of
    'calibrated'."""

    def test_deposits_only(self):
        deposits = [
            DepositRate(
                start_date=REF,
                end_date=_make_date(2025, 4, 1),
                rate=jnp.asarray(0.045),
                day_count="act_360",
            ),
            DepositRate(
                start_date=REF,
                end_date=_make_date(2025, 7, 1),
                rate=jnp.asarray(0.048),
                day_count="act_360",
            ),
            DepositRate(
                start_date=REF,
                end_date=_make_date(2026, 1, 1),
                rate=jnp.asarray(0.050),
                day_count="act_360",
            ),
        ]
        curve = bootstrap_sequential(REF, deposits, "act_365")
        graph = _wrap_default(curve)
        for dep in deposits:
            r = float(dep.residual(graph, NO_FIXINGS, REF))
            assert abs(r) < 1e-10, f"Deposit residual = {r:.2e}"

    def test_mixed_strip_simultaneous(self):
        """A deposit + FRA + swap strip bootstrapped with explicit pillars
        at every payment date reprices to machine precision.

        Sequential bootstrapping cannot guarantee this for swaps whose
        intermediate coupons fall between pillars — the bootstrapper
        uses the partial curve's flat-extrapolated DFs while the final
        curve interpolates them differently, leaving a small residual.
        Simultaneous bootstrapping with one pillar per coupon date
        sidesteps the problem.
        """
        instruments = [
            DepositRate(
                start_date=REF,
                end_date=_make_date(2025, 7, 1),
                rate=jnp.asarray(0.048),
                day_count="act_360",
            ),
            FRA(
                start_date=_make_date(2025, 7, 1),
                end_date=_make_date(2025, 10, 1),
                rate=jnp.asarray(0.046),
                day_count="act_360",
            ),
            SwapRate(
                start_date=REF,
                fixed_dates=jnp.array(
                    [int(_make_date(2026, 1, 1))], dtype=jnp.int32
                ),
                rate=jnp.asarray(0.050),
                day_count="act_360",
            ),
        ]
        pillar_dates = jnp.array(
            [
                int(_make_date(2025, 7, 1)),
                int(_make_date(2025, 10, 1)),
                int(_make_date(2026, 1, 1)),
            ],
            dtype=jnp.int32,
        )
        curve = bootstrap_simultaneous(
            REF, pillar_dates, instruments, day_count="act_365"
        )
        graph = _wrap_default(curve)
        for inst in instruments:
            r = float(inst.residual(graph, NO_FIXINGS, REF))
            assert abs(r) < 1e-10, (
                f"{type(inst).__name__} residual = {r:.2e}"
            )


# ── JIT and autodiff through the residual ────────────────────────────


class TestResidualUnderTransforms:
    def test_jit_compatible(self):
        curve = _flat_continuously_compounded(0.05)
        graph = _wrap_default(curve)
        end = _make_date(2025, 7, 1)
        df_end = float(curve(end))
        tau = float(year_fraction(REF, end, "act_360"))
        implied_rate = (1.0 / df_end - 1.0) / tau
        dep = DepositRate(
            start_date=REF,
            end_date=end,
            rate=jnp.asarray(implied_rate),
            day_count="act_360",
        )

        @jax.jit
        def f(g, ref):
            return dep.residual(g, NO_FIXINGS, ref)

        r = f(graph, jnp.asarray(REF, dtype=jnp.int32))
        assert abs(float(r)) < 1e-10

    def test_grad_through_residual_wrt_rate(self):
        """∂(residual)/∂(rate) = DF(end) * tau > 0 for a deposit."""
        curve = _flat_continuously_compounded(0.05)
        graph = _wrap_default(curve)
        end = _make_date(2025, 7, 1)
        tau = float(year_fraction(REF, end, "act_360"))

        def f(rate):
            dep = DepositRate(
                start_date=REF,
                end_date=end,
                rate=rate,
                day_count="act_360",
            )
            return dep.residual(graph, NO_FIXINGS, REF)

        g = float(jax.grad(f)(jnp.asarray(0.05)))
        df_end = float(curve(end))
        expected = df_end * tau
        assert g == pytest.approx(expected, rel=1e-6)

    def test_filter_grad_through_curve(self):
        """Gradient w.r.t. the underlying curve's DFs is non-zero on
        the pillars that the deposit's start/end dates fall between."""
        curve = _flat_continuously_compounded(0.05)
        graph = _wrap_default(curve)
        end = _make_date(2025, 7, 1)
        df_end = float(curve(end))
        tau = float(year_fraction(REF, end, "act_360"))
        implied_rate = (1.0 / df_end - 1.0) / tau
        dep = DepositRate(
            start_date=REF,
            end_date=end,
            rate=jnp.asarray(implied_rate),
            day_count="act_360",
        )

        def f(g):
            return dep.residual(g, NO_FIXINGS, REF)

        grads = eqx.filter_grad(f)(graph)
        ois_grad = grads.curves["_default_"].discount_factors
        assert jnp.any(jnp.abs(ois_grad) > 1e-12)


# ── Payment lag (spec N-3 / C-1) ─────────────────────────────────────


#: Tolerance for the T-5 telescoping identity, in units of a unit-notional
#: floating-leg PV.  The per-period sum and the closed telescope differ only
#: by floating-point summation order, so this is a round-off budget, not a
#: modelling tolerance.
T5_ATOL = 1e-14

#: Annual OIS schedule used throughout: 5 annual accrual periods from REF.
_ANNUAL = jnp.array(
    [int(_make_date(2025 + k, 1, 1)) for k in range(1, 6)],
    dtype=jnp.int32,
)


def _sloped_curve(short: float, slope: float) -> DiscountCurve:
    """Curve with zero rate ``z(t) = short + slope * t`` (t in years).

    ``slope == 0`` gives a flat continuously-compounded curve, for which
    the per-period payment-lag discount factor
    :math:`P(T^e + \\ell)/P(T^e)` is the *same* constant in every period.
    That degeneracy is what makes the flat case an exact invariant below.
    """
    pillars = jnp.array(
        [int(REF) + 365 * k for k in range(11)], dtype=jnp.int32
    )
    times = (pillars - int(REF)).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-(short + slope * times) * times)
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=dfs,
        reference_date=REF,
        day_count="act_365",
    )


def _ois_float_pv(curve: DiscountCurve, **kwargs) -> float:
    """Isolate the floating-leg PV from an :class:`OISSwapRate` residual.

    ``residual = r * A - PV_float``, so evaluating it at ``r = 0`` returns
    ``-PV_float``.  Reading the leg out of the residual (rather than
    re-implementing it) is deliberate: the identity under test is a
    property of the shipped residual, not of a test-local copy.
    """
    inst = OISSwapRate(
        start_date=REF,
        fixed_dates=_ANNUAL,
        rate=jnp.asarray(0.0),
        **kwargs,
    )
    return -float(inst.residual(_wrap_default(curve), NO_FIXINGS, REF))


def _ois_par_rate(curve: DiscountCurve, **kwargs) -> float:
    """Par fixed rate implied by an :class:`OISSwapRate` residual.

    The residual is affine in ``rate`` with slope equal to the annuity, so
    two evaluations recover both the annuity and the par rate exactly.
    """
    graph = _wrap_default(curve)

    def at(r: float) -> float:
        inst = OISSwapRate(
            start_date=REF,
            fixed_dates=_ANNUAL,
            rate=jnp.asarray(r),
            **kwargs,
        )
        return float(inst.residual(graph, NO_FIXINGS, REF))

    r0 = at(0.0)
    annuity = at(1.0) - r0
    return -r0 / annuity


class TestPaymentLagTelescoping:
    """Spec T-5: the zero-lag OIS floating leg telescopes.

    The generalised floating leg is
    :math:`\\sum_i P(0,T_i^p)\\,(P(0,T_i^s)/P(0,T_i^e) - 1)`.  Only when
    :math:`T^p = T^e` does each term collapse to
    :math:`P(0,T_i^s) - P(0,T_i^e)` and the sum telescope to
    :math:`P(0,T_0) - P(0,T_n)`.  Before payment lags existed the
    telescope was the *only* form in the code, so the test was vacuous;
    it now pins the per-period path against an independent reference.
    """

    @pytest.mark.audit
    def test_t5_zero_lag_float_leg_telescopes(self):
        """T-5 gate: per-period sum == P_d(0,T_0) - P_d(0,T_n)."""
        curve = _sloped_curve(0.03, 0.004)
        # Independent reference: two curve reads, no summation at all.
        expected = float(curve(REF) - curve(_ANNUAL[-1]))
        # Explicit payment dates equal to the accrual ends force the
        # per-period branch while leaving the economics lag-free.
        per_period = _ois_float_pv(curve, payment_dates=_ANNUAL)
        assert per_period == pytest.approx(expected, abs=T5_ATOL)

    def test_default_path_is_the_closed_telescope(self):
        """With no lag configured the residual takes the fast path and
        returns the closed telescope *exactly* — bit for bit, which is
        what preserves every pre-existing bootstrap number."""
        curve = _sloped_curve(0.03, 0.004)
        assert _ois_float_pv(curve) == -float(
            curve(_ANNUAL[-1]) - curve(REF)
        )

    def test_fast_path_and_per_period_path_agree(self):
        """The two branches are the same number up to summation order."""
        curve = _sloped_curve(0.03, 0.004)
        assert _ois_float_pv(curve) == pytest.approx(
            _ois_float_pv(curve, payment_dates=_ANNUAL), abs=T5_ATOL
        )

    def test_telescope_does_not_survive_a_lag(self):
        """A non-zero lag breaks the identity — the point of the gate."""
        curve = _sloped_curve(0.03, 0.004)
        telescoped = float(curve(REF) - curve(_ANNUAL[-1]))
        lagged = _ois_float_pv(curve, payment_lag=2)
        assert abs(lagged - telescoped) > 1e-7
        # Paying later is worth less on a positive-rate curve.
        assert lagged < telescoped

    def test_swap_rate_shares_the_treatment(self):
        """:class:`SwapRate`'s single-curve float leg is the same
        expression, so it gets the same two branches."""
        curve = _sloped_curve(0.03, 0.004)
        graph = _wrap_default(curve)

        def float_pv(**kw):
            inst = SwapRate(
                start_date=REF,
                fixed_dates=_ANNUAL,
                rate=jnp.asarray(0.0),
                **kw,
            )
            return -float(inst.residual(graph, NO_FIXINGS, REF))

        expected = float(curve(REF) - curve(_ANNUAL[-1]))
        assert float_pv() == pytest.approx(expected, abs=T5_ATOL)
        assert float_pv(payment_dates=_ANNUAL) == pytest.approx(
            expected, abs=T5_ATOL
        )
        assert float_pv(payment_lag=2) < expected


class TestPaymentLagDirection:
    """What a uniform payment lag does to the *par rate*.

    Testing the residual at a fixed quoted rate is a weak gate: a uniform
    lag multiplies both legs by very nearly the same factor, so the
    residual just rescales and the sign of the change tells you little.
    The par rate is the sharp statement.

    Write the par rate as a weighted average of the period forwards,
    :math:`r^{*} = \\sum_i w_i f_i / \\sum_i w_i` with
    :math:`w_i \\propto \\tau_i P(0,T_i^p)`.  Lagging payment multiplies
    each weight by :math:`\\delta_i = P(0,T_i^p)/P(0,T_i^e) \\approx
    e^{-f_i \\ell}`, a *decreasing function of that period's own forward*.
    So the lag always shifts weight away from the high-forward periods and
    the par rate always falls — whatever the shape of the curve — by an
    amount governed by the dispersion of the forwards, and by exactly
    nothing when they are all equal.
    """

    def test_flat_curve_par_rate_is_lag_invariant(self):
        """Zero dispersion, zero effect — and exactly so, not just
        approximately: on a flat continuously-compounded curve every
        period shares one delta, which cancels between numerator and
        denominator."""
        curve = _sloped_curve(0.03, 0.0)
        assert _ois_par_rate(curve, payment_lag=2) == pytest.approx(
            _ois_par_rate(curve), abs=1e-15
        )

    @pytest.mark.parametrize("slope", [0.004, -0.004])
    def test_lag_lowers_the_par_rate_whatever_the_slope(self, slope):
        """Upward or inverted, the par rate falls."""
        curve = _sloped_curve(0.03, slope)
        assert _ois_par_rate(curve, payment_lag=2) < _ois_par_rate(curve)

    def test_effect_grows_with_the_dispersion_of_the_forwards(self):
        """Doubling the slope roughly quadruples the shift — the
        signature of a variance-driven effect rather than a level one."""
        base = _sloped_curve(0.03, 0.002)
        steep = _sloped_curve(0.03, 0.004)
        d_base = _ois_par_rate(base) - _ois_par_rate(base, payment_lag=2)
        d_steep = _ois_par_rate(steep) - _ois_par_rate(
            steep, payment_lag=2
        )
        assert d_base > 0.0 and d_steep > 0.0
        assert d_steep / d_base == pytest.approx(4.0, rel=0.05)

    def test_lag_scales_the_effect(self):
        """Two days move the par rate about twice as far as one."""
        curve = _sloped_curve(0.03, 0.004)
        base = _ois_par_rate(curve)
        d1 = base - _ois_par_rate(curve, payment_lag=1)
        d2 = base - _ois_par_rate(curve, payment_lag=2)
        assert d2 / d1 == pytest.approx(2.0, rel=0.02)


class TestPaymentLagPerLeg:
    """Payment dates are resolved leg by leg, and discounting uses them."""

    def _ibor(self, **kwargs) -> IborSwapRate:
        semis = jnp.array(
            [int(REF) + 182 * k for k in range(1, 11)], dtype=jnp.int32
        )
        starts = jnp.concatenate([jnp.asarray(REF)[None], semis[:-1]])
        return IborSwapRate(
            start_date=REF,
            fixed_dates=_ANNUAL,
            float_dates=semis,
            fixing_dates=starts,
            rate=jnp.asarray(0.03),
            **kwargs,
        )

    def test_explicit_dates_equal_to_accrual_ends_are_a_no_op(self):
        """Handing back the accrual schedule as the payment schedule must
        change nothing — it is the lag-free convention written out."""
        curve = _sloped_curve(0.03, 0.004)
        graph = _wrap_default(curve)
        plain = self._ibor()
        spelled = self._ibor(
            fixed_payment_dates=plain.fixed_dates,
            float_payment_dates=plain.float_dates,
        )
        assert float(
            spelled.residual(graph, NO_FIXINGS, REF)
        ) == pytest.approx(
            float(plain.residual(graph, NO_FIXINGS, REF)), abs=T5_ATOL
        )

    def test_each_leg_moves_independently(self):
        """Lagging only the fixed leg, only the float leg, and both give
        three different residuals — so neither leg is silently reusing
        the other's payment schedule."""
        curve = _sloped_curve(0.03, 0.004)
        graph = _wrap_default(curve)
        plain = self._ibor()

        def resid(inst):
            return float(inst.residual(graph, NO_FIXINGS, REF))

        base = resid(plain)
        fixed_only = resid(
            self._ibor(fixed_payment_dates=plain.fixed_dates + 2)
        )
        float_only = resid(
            self._ibor(float_payment_dates=plain.float_dates + 2)
        )
        both = resid(self._ibor(payment_lag=2))

        for other in (fixed_only, float_only, both):
            assert abs(other - base) > 1e-9
        assert abs(fixed_only - float_only) > 1e-9
        # The two single-leg effects are additive to first order.
        assert (fixed_only - base) + (float_only - base) == pytest.approx(
            both - base, rel=1e-4
        )

    def test_flat_curve_par_rate_is_lag_invariant(self):
        """The dual-curve par rate inherits the flat-curve invariance."""
        curve = _sloped_curve(0.03, 0.0)
        graph = _wrap_default(curve)

        def par(**kw):
            def at(r):
                inst = eqx.tree_at(
                    lambda i: i.rate, self._ibor(**kw), jnp.asarray(r)
                )
                return float(inst.residual(graph, NO_FIXINGS, REF))

            r0 = at(0.0)
            return -r0 / (at(1.0) - r0)

        assert par(payment_lag=2) == pytest.approx(par(), abs=1e-15)

    def test_tenor_basis_swap_discounts_at_payment_dates(self):
        curve = _sloped_curve(0.03, 0.004)
        graph = _wrap_default(curve)
        quarters = jnp.array(
            [int(REF) + 91 * k for k in range(1, 21)], dtype=jnp.int32
        )
        semis = jnp.array(
            [int(REF) + 182 * k for k in range(1, 11)], dtype=jnp.int32
        )

        def resid(**kw):
            inst = TenorBasisSwap(
                start_date=REF,
                leg_a_dates=quarters,
                leg_a_fixing_dates=jnp.concatenate(
                    [jnp.asarray(REF)[None], quarters[:-1]]
                ),
                leg_b_dates=semis,
                leg_b_fixing_dates=jnp.concatenate(
                    [jnp.asarray(REF)[None], semis[:-1]]
                ),
                spread=jnp.asarray(0.002),
                **kw,
            )
            return float(inst.residual(graph, NO_FIXINGS, REF))

        base = resid()
        assert resid(
            leg_a_payment_dates=quarters, leg_b_payment_dates=semis
        ) == pytest.approx(base, abs=T5_ATOL)
        assert abs(resid(payment_lag=2) - base) > 1e-9

    def test_xccy_notional_redemption_follows_the_payment_date(self):
        """Under ``constant_notional`` the final notional re-exchange is
        paid with the last coupon, so a lag must move it too."""
        curve = _sloped_curve(0.03, 0.004)
        foreign = _sloped_curve(0.02, 0.003)
        graph = CurveGraph(
            curves={"_default_": curve, "FOR": foreign}
        )
        semis = jnp.array(
            [int(REF) + 182 * k for k in range(1, 11)], dtype=jnp.int32
        )
        fixings = jnp.concatenate([jnp.asarray(REF)[None], semis[:-1]])

        def resid(**kw):
            inst = CrossCurrencyBasisSwap(
                start_date=REF,
                dom_dates=semis,
                dom_fixing_dates=fixings,
                for_dates=semis,
                for_fixing_dates=fixings,
                fx_spot=jnp.asarray(1.1),
                spread=jnp.asarray(-0.005),
                curves_touched=(
                    "_default_", "_default_", "FOR", "FOR",
                ),
                **kw,
            )
            return float(inst.residual(graph, fixings=NO_FIXINGS, ref_date=REF))

        base = resid()
        assert resid(
            dom_payment_dates=semis, for_payment_dates=semis
        ) == pytest.approx(base, abs=T5_ATOL)
        lagged = resid(payment_lag=5)
        assert abs(lagged - base) > 1e-9

        # The redemption term is what carries most of the move: holding
        # the coupons lag-free but shifting only the last payment date
        # must still change the answer.
        shifted_last = semis.at[-1].add(5)
        only_redemption = resid(
            dom_payment_dates=shifted_last,
            for_payment_dates=shifted_last,
        )
        assert abs(only_redemption - base) > 1e-9


class TestAveragedFutures:
    """Compounded vs arithmetic-average money-market futures (spec § 4.2).

    The compounded forward is a *product* of daily discount-factor ratios
    minus one; the averaged forward is a *sum* of the same ratios minus
    one.  The gap is the interest-on-interest that averaging discards, so
    it survives on a perfectly flat curve — this is a compounding effect,
    not a curve-shape effect.

    The references below are **closed forms**, not rearrangements of the
    implementation.  On a flat continuously-compounded curve at rate
    :math:`r` with ``act_365`` curve basis and an ``act_360`` contract
    basis, every daily ratio is exactly :math:`e^{r/365}`, so for
    :math:`N` days

    .. math::
        \\bar{F} = 360\\left(e^{r/365} - 1\\right), \\qquad
        F = \\frac{360}{N}\\left(e^{rN/365} - 1\\right).

    Neither expression shares a line of code with the thing under test.
    """

    #: Flat curve rate used for the closed-form comparisons.
    RATE = 0.04

    def _curve(self, slope: float = 0.0) -> DiscountCurve:
        pillars = jnp.array(
            [int(REF) + 30 * k for k in range(0, 25)], dtype=jnp.int32
        )
        times = (pillars - int(REF)).astype(jnp.float64) / 365.0
        dfs = jnp.exp(-(self.RATE + slope * times) * times)
        return DiscountCurve(
            pillar_dates=pillars,
            discount_factors=dfs,
            reference_date=REF,
            day_count="act_365",
        )

    def _forward(self, curve, t0, t1, averaging) -> float:
        """Curve-implied forward, read out of the residual at a zero
        quote: ``residual = q - adj - F`` so ``F = -residual`` at
        ``q = 0`` with no adjustment."""
        future = MoneyMarketFuture(
            start_date=jnp.asarray(t0),
            end_date=jnp.asarray(t1),
            futures_rate=jnp.asarray(0.0),
            day_count="act_360",
            averaging=averaging,
        )
        return -float(future.residual(_wrap_default(curve), NO_FIXINGS, REF))

    def test_default_is_compounded(self):
        """The new field must not change anybody's existing instrument."""
        assert (
            MoneyMarketFuture(
                start_date=jnp.asarray(int(REF) + 30),
                end_date=jnp.asarray(int(REF) + 120),
                futures_rate=jnp.asarray(0.04),
            ).averaging
            == "compounded"
        )

    def test_compounded_branch_is_bit_for_bit_unchanged(self):
        """Exact equality, not approximate: the compounded branch is the
        same two-discount-factor expression it always was."""
        curve = self._curve()
        t0, t1 = int(REF) + 30, int(REF) + 120
        tau = year_fraction(
            jnp.asarray(t0), jnp.asarray(t1), "act_360"
        )
        expected = float(
            (curve(jnp.asarray(t0)) / curve(jnp.asarray(t1)) - 1.0) / tau
        )
        assert self._forward(curve, t0, t1, "compounded") == expected

    def test_averaged_forward_matches_its_closed_form(self):
        """1M contract, flat curve: 360 (e^{r/365} - 1)."""
        curve = self._curve()
        t0, t1 = int(REF) + 30, int(REF) + 60
        expected = 360.0 * (math.exp(self.RATE / 365.0) - 1.0)
        assert self._forward(curve, t0, t1, "arithmetic") == pytest.approx(
            expected, rel=1e-12
        )

    def test_compounded_forward_matches_its_closed_form(self):
        curve = self._curve()
        t0, t1, n = int(REF) + 30, int(REF) + 120, 90
        expected = (360.0 / n) * (
            math.exp(self.RATE * n / 365.0) - 1.0
        )
        assert self._forward(curve, t0, t1, "compounded") == pytest.approx(
            expected, rel=1e-12
        )

    def test_the_two_differ_on_a_flat_curve(self):
        """The distinction is compounding, so a flat curve does not make
        it go away — which is exactly why calibrating a 1M contract as
        though it were compounded is a real mis-specification."""
        curve = self._curve()
        t0, t1 = int(REF) + 30, int(REF) + 60
        compounded = self._forward(curve, t0, t1, "compounded")
        averaged = self._forward(curve, t0, t1, "arithmetic")
        assert compounded > averaged
        # Leading order is r^2 tau / 2; 1M at 4% is a fraction of a bp.
        tau = 30.0 / 360.0
        predicted = self.RATE**2 * tau / 2.0
        assert (compounded - averaged) == pytest.approx(
            predicted, rel=0.1
        )
        assert 1e-5 < (compounded - averaged) < 1e-4

    def test_gap_grows_with_the_period(self):
        """r^2 tau / 2 is linear in tau: a 3M contract shows ~3x a 1M."""
        curve = self._curve()
        t0 = int(REF) + 30
        gap_1m = self._forward(
            curve, t0, t0 + 30, "compounded"
        ) - self._forward(curve, t0, t0 + 30, "arithmetic")
        gap_3m = self._forward(
            curve, t0, t0 + 90, "compounded"
        ) - self._forward(curve, t0, t0 + 90, "arithmetic")
        assert gap_3m / gap_1m == pytest.approx(3.0, rel=0.05)

    def test_both_vanish_on_a_zero_rate_curve(self):
        """Every daily ratio is exactly 1, so sum and product agree."""
        pillars = jnp.array(
            [int(REF) + 30 * k for k in range(0, 25)], dtype=jnp.int32
        )
        flat_one = DiscountCurve(
            pillar_dates=pillars,
            discount_factors=jnp.ones_like(pillars, dtype=jnp.float64),
            reference_date=REF,
            day_count="act_365",
        )
        t0, t1 = int(REF) + 30, int(REF) + 120
        assert self._forward(flat_one, t0, t1, "compounded") == 0.0
        assert self._forward(flat_one, t0, t1, "arithmetic") == 0.0

    def test_daily_accruals_sum_to_the_period_accrual(self):
        """Justifies collapsing the day-count-weighted average to
        ``sum(rho_j - 1) / tau``: under ``act_360`` the daily accrual
        fractions add up to the period's exactly, so ``tau`` really is
        the sum of the weights."""
        t0, t1 = int(REF) + 30, int(REF) + 120
        daily = jnp.asarray(t0) + jnp.arange(t1 - t0 + 1)
        deltas = year_fraction(daily[:-1], daily[1:], "act_360")
        tau = year_fraction(jnp.asarray(t0), jnp.asarray(t1), "act_360")
        assert float(jnp.sum(deltas)) == float(tau)

    def test_only_the_averaged_contract_sees_inside_its_own_period(self):
        """A structural difference in the *risk*, not just the level.

        The compounded forward is a telescoping product of daily ratios,
        so interior pillars cancel identically and it is sensitive only
        to the two endpoint discount factors — on any curve.  The
        averaged forward is a sum, which does not telescope, so it picks
        up genuine exposure to the shape of the curve inside its own
        period.  That exposure vanishes on a flat curve (adjacent daily
        ratios are then equal and cancel), so this has to be checked on
        a sloped one.
        """
        curve = self._curve(slope=0.05)
        graph = _wrap_default(curve)
        t0, t1 = int(REF) + 30, int(REF) + 120

        def pillar_grad(averaging):
            future = MoneyMarketFuture(
                start_date=jnp.asarray(t0),
                end_date=jnp.asarray(t1),
                futures_rate=jnp.asarray(0.0),
                day_count="act_360",
                averaging=averaging,
            )
            grads = eqx.filter_grad(
                lambda g: future.residual(g, NO_FIXINGS, REF)
            )(graph)
            return grads.curves["_default_"].discount_factors

        pillars = curve.pillar_dates
        interior = jnp.asarray(
            [(int(p) > t0) and (int(p) < t1) for p in pillars]
        )
        assert bool(jnp.any(interior)), "need interior pillars to test"

        comp = pillar_grad("compounded")
        arith = pillar_grad("arithmetic")
        # Compounded: interior sensitivity is *identically* zero.
        assert jnp.all(comp[interior] == 0.0)
        # Averaged: it is not.
        assert jnp.any(jnp.abs(arith[interior]) > 1e-9)
        # Both still see the endpoints, and with opposite signs.
        assert float(jnp.sum(arith[pillars == t0])) < 0.0
        assert float(jnp.sum(arith[pillars == t1])) > 0.0

    def test_rejects_an_unknown_convention(self):
        with pytest.raises(ValueError, match="averaging must be one of"):
            MoneyMarketFuture(
                start_date=jnp.asarray(int(REF)),
                end_date=jnp.asarray(int(REF) + 30),
                futures_rate=jnp.asarray(0.04),
                averaging="average",
            )

    def test_rejects_a_degenerate_span(self):
        """An averaged contract needs at least one rate day."""
        future = MoneyMarketFuture(
            start_date=jnp.asarray(int(REF) + 30),
            end_date=jnp.asarray(int(REF) + 30),
            futures_rate=jnp.asarray(0.04),
            averaging="arithmetic",
        )
        with pytest.raises(ValueError, match="at least one"):
            future.residual(
                _wrap_default(self._curve()), NO_FIXINGS, REF
            )

    def test_traced_dates_raise_a_pointed_error(self):
        """The daily subdivision needs a concrete span.  Failing loudly
        with a message naming the way out beats silently averaging over
        the wrong number of days."""
        curve = self._curve()
        graph = _wrap_default(curve)

        def f(t0, t1):
            return MoneyMarketFuture(
                start_date=t0,
                end_date=t1,
                futures_rate=jnp.asarray(0.0),
                averaging="arithmetic",
            ).residual(graph, NO_FIXINGS, REF)

        with pytest.raises(ValueError, match="concrete day count"):
            jax.jit(f)(
                jnp.asarray(int(REF) + 30), jnp.asarray(int(REF) + 120)
            )

    def test_jit_and_grad_with_the_instrument_closed_over(self):
        """The realistic pattern: quotes are Python data, the graph is
        the traced argument."""
        curve = self._curve()
        future = MoneyMarketFuture(
            start_date=jnp.asarray(int(REF) + 30),
            end_date=jnp.asarray(int(REF) + 120),
            futures_rate=jnp.asarray(0.04),
            day_count="act_360",
            averaging="arithmetic",
        )

        def f(g):
            return future.residual(g, NO_FIXINGS, REF)

        graph = _wrap_default(curve)
        assert float(eqx.filter_jit(f)(graph)) == pytest.approx(
            float(f(graph)), abs=1e-14
        )
        dfs_bar = eqx.filter_grad(f)(graph).curves[
            "_default_"
        ].discount_factors
        assert jnp.all(jnp.isfinite(dfs_bar))
        assert jnp.any(jnp.abs(dfs_bar) > 1e-12)

    def test_bootstraps_to_a_different_curve(self):
        """End to end: an averaged strip converges, and lands somewhere
        measurably different from the same quotes read as compounded."""
        from valax.curves.bootstrap_graph import bootstrap_curve_graph
        from valax.curves.graph import CurveSpec

        pillars = jnp.array(
            [int(REF) + 90 * k for k in range(1, 5)], dtype=jnp.int32
        )
        spec = CurveSpec(
            curve_id="USD.SOFR.1M",
            currency="USD",
            pillar_dates=pillars,
            day_count="act_365",
        )

        def build(averaging):
            quotes = [
                MoneyMarketFuture(
                    start_date=jnp.asarray(int(p) - 90),
                    end_date=jnp.asarray(int(p)),
                    futures_rate=jnp.asarray(0.04 + 0.001 * k),
                    curves_touched=("USD.SOFR.1M",),
                    averaging=averaging,
                )
                for k, p in enumerate(pillars)
            ]
            return bootstrap_curve_graph(REF, [spec], quotes)

        averaged, diag_a = build("arithmetic")
        compounded, diag_c = build("compounded")
        assert diag_a.converged and diag_c.converged
        assert float(diag_a.max_abs_residual) < 1e-12
        gap = jnp.abs(
            averaged["USD.SOFR.1M"].discount_factors
            - compounded["USD.SOFR.1M"].discount_factors
        )
        assert float(jnp.max(gap)) > 1e-5


class TestPartlyAccruedFixings:
    """Spec C-3 / T-11: a coupon's rate has three possible sources.

    A recorded fixing fully determines the coupon.  A period that has
    *begun but not ended* is neither realised nor forward-looking: part
    of its overnight compounding has happened and the rest has not, and
    the two must be combined as

    .. math::
        1 + \\tau_j F_j = A \\cdot P_x(0,t_0)/P_x(0,T_j^e).

    Everything else is projected.  The gates below pin the two
    degeneracies of that formula, the consistency property that makes it
    credible, and the zero curve-sensitivity of a realised coupon.
    """

    #: Realised rate used for the daily history, deliberately below the
    #: projection curve so the blend moves the answer in a signed way.
    REALISED = 0.04

    def _graph(self, ref):
        pillars = jnp.array(
            [int(REF) + 30 * k for k in range(0, 40)], dtype=jnp.int32
        )
        times = (pillars - int(REF)).astype(jnp.float64) / 365.0
        return CurveGraph(
            curves={
                "DISC": DiscountCurve(
                    pillar_dates=pillars,
                    discount_factors=jnp.exp(-0.040 * times),
                    reference_date=ref,
                    day_count="act_365",
                ),
                "FWD": DiscountCurve(
                    pillar_dates=pillars,
                    discount_factors=jnp.exp(-0.045 * times),
                    reference_date=ref,
                    day_count="act_365",
                ),
            }
        )

    def _series(self, n_days=120):
        return FixingSeries(
            fixing_dates=jnp.arange(int(REF), int(REF) + n_days + 1),
            fixings=jnp.full(n_days + 1, self.REALISED),
        )

    def _leg(self, accrued=None, float_dates=None, **kwargs):
        """One-period float leg over [REF, REF+90] unless overridden."""
        floats = (
            jnp.asarray([int(REF) + 90], dtype=jnp.int32)
            if float_dates is None
            else float_dates
        )
        fixing_dates = jnp.concatenate(
            [jnp.asarray(REF)[None], floats[:-1]]
        )
        return IborSwapRate(
            start_date=REF,
            fixed_dates=jnp.asarray([int(REF) + 90], dtype=jnp.int32),
            float_dates=floats,
            fixing_dates=fixing_dates,
            rate=jnp.asarray(0.0),
            float_day_count="act_360",
            index_id="RFR.TENOR",
            curves_touched=("DISC", "FWD"),
            accrued_compounding=accrued,
            **kwargs,
        )

    def _float_pv(self, ref, accrued=None, fixings=None, **kwargs):
        """``residual`` at ``rate=0`` is ``-float_pv``."""
        leg = self._leg(accrued=accrued, **kwargs)
        return -float(
            leg.residual(
                self._graph(ref),
                fixings if fixings is not None else NO_FIXINGS,
                jnp.asarray(ref),
            )
        )

    # ── the two degeneracies ─────────────────────────────────────────

    def test_nothing_accrued_is_exactly_the_projected_forward(self):
        """At ``t_0 = T^s`` nothing has accrued, ``A = 1``, and the blend
        must reduce to pure projection — exactly, not approximately."""
        ref = int(REF)
        assert self._float_pv(
            ref, accrued=jnp.asarray(1.0)
        ) == self._float_pv(ref)

    def test_a_realised_part_matching_the_curve_is_an_exact_no_op(self):
        """The consistency property that makes the blend credible.

        If the realised rates happen to be exactly what the curve had
        projected for the elapsed part — i.e. ``A`` equals the curve's
        own compounding ``P_x(T^s)/P_x(t_0)`` — then substituting
        realised data for projected data cannot change the answer.  It
        holds to the last bit, which also pins down that the blend uses
        ``P_x(t_0)`` and not some neighbouring date.
        """
        ref = int(REF) + 45
        fwd = self._graph(ref)["FWD"]
        a_curve = fwd(jnp.asarray(int(REF))) / fwd(jnp.asarray(ref))
        assert self._float_pv(ref, accrued=a_curve) == self._float_pv(ref)

    def test_blend_tends_to_purely_realised_at_the_far_end(self):
        """As ``t_0`` approaches ``T^e`` the projected remainder shrinks
        to nothing and the coupon tends to ``(A - 1)/tau``."""
        end = int(REF) + 90
        graph_ref = end - 1
        series = self._series()
        full_a = float(
            series.compounded_factor(
                jnp.asarray(int(REF)), jnp.asarray(end), "act_360"
            )
        )
        realised_only = full_a - 1.0
        disc = float(self._graph(graph_ref)["DISC"](jnp.asarray(end)))

        def gap(ref):
            a = series.compounded_factor(
                jnp.asarray(int(REF)), jnp.asarray(ref), "act_360"
            )
            return abs(
                self._float_pv(ref, accrued=a) - realised_only * disc
            )

        # Monotonically closing on the purely-realised value.
        assert gap(int(REF) + 89) < gap(int(REF) + 60) < gap(int(REF) + 30)

    # ── signed behaviour ─────────────────────────────────────────────

    def test_blend_is_monotone_in_the_realised_factor(self):
        ref = int(REF) + 45
        pvs = [
            self._float_pv(ref, accrued=jnp.asarray(a))
            for a in (1.001, 1.005, 1.010)
        ]
        assert pvs[0] < pvs[1] < pvs[2]

    def test_realised_above_or_below_the_curve_moves_it_the_right_way(
        self,
    ):
        ref = int(REF) + 45
        fwd = self._graph(ref)["FWD"]
        a_curve = fwd(jnp.asarray(int(REF))) / fwd(jnp.asarray(ref))
        plain = self._float_pv(ref)
        assert self._float_pv(ref, accrued=a_curve * 1.001) > plain
        assert self._float_pv(ref, accrued=a_curve * 0.999) < plain

    # ── uniqueness of the straddling period ──────────────────────────

    def test_only_one_period_straddles_a_boundary_date(self):
        """The half-open test ``T^s <= t_0 < T^e`` must match exactly one
        period when ``t_0`` lands on an accrual boundary.  Closing both
        ends would match the period ending there *and* the one starting
        there, and they cannot share one realised factor.  Checked by
        supplying ``A = 1``: a unique match then makes the blend a
        provable no-op, whereas a double match would not be.
        """
        boundary = int(REF) + 45
        two_periods = jnp.asarray(
            [boundary, int(REF) + 90], dtype=jnp.int32
        )
        assert self._float_pv(
            boundary, accrued=jnp.asarray(1.0), float_dates=two_periods
        ) == self._float_pv(boundary, float_dates=two_periods)

    # ── T-11 ─────────────────────────────────────────────────────────

    @pytest.mark.audit
    def test_t11_recorded_coupon_has_no_projection_curve_sensitivity(
        self,
    ):
        """T-11 gate, first half.

        A coupon whose fixing is recorded is not a function of any curve,
        so its sensitivity to the *projection* curve must be identically
        zero — not small, zero.  It must still be discounted, so the
        discount-curve sensitivity has to survive; asserting both is what
        stops the test passing for the trivial reason that nothing is
        connected.

        Tolerance: exact equality to ``0.0``.  This is achievable because
        the selection is a ``jnp.where`` over a NaN-free replacement, so
        the unselected branch receives a zero cotangent rather than a
        small one.
        """
        ref = int(REF)
        history = FixingHistory(
            indices={
                "RFR.TENOR": FixingSeries(
                    fixing_dates=jnp.asarray([int(REF)]),
                    fixings=jnp.asarray([0.0375]),
                )
            }
        )
        leg = self._leg()
        grads = eqx.filter_grad(
            lambda g: leg.residual(g, history, jnp.asarray(ref))
        )(self._graph(ref))
        projection_bar = grads.curves["FWD"].discount_factors
        discount_bar = grads.curves["DISC"].discount_factors

        assert jnp.all(projection_bar == 0.0)
        assert jnp.any(jnp.abs(discount_bar) > 1e-12)

    @pytest.mark.audit
    def test_t11_sensitivity_to_the_stored_fixing_is_non_zero(self):
        """T-11 gate, second half — the complement that proves the first
        half is not vacuous.  The coupon *does* depend on the recorded
        value; what it does not depend on is the curve."""
        ref = int(REF)
        history = FixingHistory(
            indices={
                "RFR.TENOR": FixingSeries(
                    fixing_dates=jnp.asarray([int(REF)]),
                    fixings=jnp.asarray([0.0375]),
                )
            }
        )
        leg = self._leg()
        grads = eqx.filter_grad(
            lambda h: leg.residual(self._graph(ref), h, jnp.asarray(ref))
        )(history)
        assert abs(
            float(grads.indices["RFR.TENOR"].fixings[0])
        ) > 1e-6

    @pytest.mark.audit
    def test_t11_straddling_period_uses_both_sources(self):
        """T-11 gate, third part: on a date straddling a fixing the
        coupon draws on realised data *and* the curve.  Both
        sensitivities must be non-zero, and the answer must differ from
        both pure alternatives."""
        ref = int(REF) + 45
        series = self._series()
        a = series.compounded_factor(
            jnp.asarray(int(REF)), jnp.asarray(ref), "act_360"
        )

        # Depends on the curve ...
        leg = self._leg(accrued=a)
        curve_bar = eqx.filter_grad(
            lambda g: leg.residual(g, NO_FIXINGS, jnp.asarray(ref))
        )(self._graph(ref)).curves["FWD"].discount_factors
        assert jnp.any(jnp.abs(curve_bar) > 1e-9)

        # ... and on the realised factor.
        a_bar = eqx.filter_grad(
            lambda x: self._leg(accrued=x).residual(
                self._graph(ref), NO_FIXINGS, jnp.asarray(ref)
            )
        )(a)
        assert abs(float(a_bar)) > 1e-6

    def test_a_recorded_fixing_outranks_the_blend(self):
        """Documented precedence: a recorded value for the coupon is
        strictly more information than a partial accrual, so an absurd
        ``A`` must be ignored when a fixing exists."""
        ref = int(REF) + 45
        history = FixingHistory(
            indices={
                "RFR.TENOR": FixingSeries(
                    fixing_dates=jnp.asarray([int(REF)]),
                    fixings=jnp.asarray([0.0375]),
                )
            }
        )
        assert self._float_pv(
            ref, accrued=jnp.asarray(1.5), fixings=history
        ) == self._float_pv(ref, fixings=history)

    def test_accrued_compounding_without_a_ref_date_raises(self):
        from valax.curves.instruments import _floating_leg_pv

        with pytest.raises(ValueError, match="needs ref_date"):
            _floating_leg_pv(
                self._graph(int(REF)),
                NO_FIXINGS,
                REF,
                jnp.asarray([int(REF) + 90], dtype=jnp.int32),
                jnp.asarray([int(REF)], dtype=jnp.int32),
                "act_360",
                "RFR.TENOR",
                "DISC",
                "FWD",
                jnp.asarray(0.0),
                None,
                None,
                jnp.asarray(1.01),
            )

    def test_default_is_off_and_changes_nothing(self):
        """No field set means no blend: every coupon is either recorded
        or fully projected, exactly as before."""
        assert self._leg().accrued_compounding is None

    def test_jit_and_grad_through_the_blend(self):
        ref = int(REF) + 45
        leg = self._leg(accrued=jnp.asarray(1.005))

        def f(g):
            return leg.residual(g, NO_FIXINGS, jnp.asarray(ref))

        graph = self._graph(ref)
        assert float(eqx.filter_jit(f)(graph)) == pytest.approx(
            float(f(graph)), abs=1e-14
        )
        bars = eqx.filter_grad(f)(graph).curves["FWD"].discount_factors
        assert jnp.all(jnp.isfinite(bars))

    def test_basis_swap_legs_blend_independently(self):
        """Each leg of a tenor-basis swap resolves its own realised
        factor, so blending one must not move the other."""
        ref = int(REF) + 45
        graph = self._graph(ref)
        quarters = jnp.asarray(
            [int(REF) + 91 * k for k in range(1, 9)], dtype=jnp.int32
        )
        semis = jnp.asarray(
            [int(REF) + 182 * k for k in range(1, 5)], dtype=jnp.int32
        )

        def resid(**kw):
            swap = TenorBasisSwap(
                start_date=REF,
                leg_a_dates=quarters,
                leg_a_fixing_dates=jnp.concatenate(
                    [jnp.asarray(REF)[None], quarters[:-1]]
                ),
                leg_b_dates=semis,
                leg_b_fixing_dates=jnp.concatenate(
                    [jnp.asarray(REF)[None], semis[:-1]]
                ),
                spread=jnp.asarray(0.002),
                curves_touched=("DISC", "FWD", "FWD"),
                **kw,
            )
            return float(swap.residual(graph, NO_FIXINGS, jnp.asarray(ref)))

        base = resid()
        a_only = resid(leg_a_accrued_compounding=jnp.asarray(1.02))
        b_only = resid(leg_b_accrued_compounding=jnp.asarray(1.02))
        assert abs(a_only - base) > 1e-9
        assert abs(b_only - base) > 1e-9
        # Opposite signs: the legs enter the residual as a - b.
        assert (a_only - base) * (b_only - base) < 0.0


class TestPaymentLagValidation:
    """Configuration errors are raised at the user boundary."""

    def test_negative_lag_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            OISSwapRate(
                start_date=REF,
                fixed_dates=_ANNUAL,
                rate=jnp.asarray(0.03),
                payment_lag=-1,
            )

    def test_explicit_dates_and_lag_together_rejected(self):
        with pytest.raises(ValueError, match="contradict each other"):
            OISSwapRate(
                start_date=REF,
                fixed_dates=_ANNUAL,
                rate=jnp.asarray(0.03),
                payment_dates=_ANNUAL,
                payment_lag=2,
            )

    def test_mismatched_schedule_length_rejected(self):
        with pytest.raises(ValueError, match="one payment per period"):
            OISSwapRate(
                start_date=REF,
                fixed_dates=_ANNUAL,
                rate=jnp.asarray(0.03),
                payment_dates=_ANNUAL[:-1],
            )

    def test_wrong_leg_is_named(self):
        """The message says which leg is misconfigured."""
        with pytest.raises(ValueError, match="float leg"):
            IborSwapRate(
                start_date=REF,
                fixed_dates=_ANNUAL,
                float_dates=_ANNUAL,
                fixing_dates=_ANNUAL,
                rate=jnp.asarray(0.03),
                float_payment_dates=_ANNUAL[:-1],
            )


class TestPaymentLagUnderTransforms:
    """The lagged branch is JIT-safe and differentiable."""

    def test_jit(self):
        curve = _sloped_curve(0.03, 0.004)
        inst = OISSwapRate(
            start_date=REF,
            fixed_dates=_ANNUAL,
            rate=jnp.asarray(0.03),
            payment_lag=2,
        )

        @eqx.filter_jit
        def f(i, g):
            return i.residual(g, NO_FIXINGS, REF)

        traced = float(f(inst, _wrap_default(curve)))
        eager = float(inst.residual(_wrap_default(curve), NO_FIXINGS, REF))
        assert traced == pytest.approx(eager, abs=1e-14)

    def test_grad_flows_to_the_curve(self):
        curve = _sloped_curve(0.03, 0.004)
        inst = OISSwapRate(
            start_date=REF,
            fixed_dates=_ANNUAL,
            rate=jnp.asarray(0.03),
            payment_lag=2,
        )
        grads = eqx.filter_grad(
            lambda g: inst.residual(g, NO_FIXINGS, REF)
        )(_wrap_default(curve))
        dfs_bar = grads.curves["_default_"].discount_factors
        assert jnp.any(jnp.abs(dfs_bar) > 1e-12)
        assert jnp.all(jnp.isfinite(dfs_bar))

    def test_bootstrap_converges_with_a_lag(self):
        """End to end: a lagged OIS strip still solves to machine zero,
        and to a *different* curve than the lag-free strip."""
        from valax.curves.bootstrap_graph import bootstrap_curve_graph
        from valax.curves.graph import CurveSpec

        spec = CurveSpec(
            curve_id="USD.SOFR.OIS",
            currency="USD",
            pillar_dates=_ANNUAL,
            day_count="act_365",
        )

        def build(lag: int):
            quotes = [
                OISSwapRate(
                    start_date=REF,
                    fixed_dates=_ANNUAL[: k + 1],
                    rate=jnp.asarray(0.03 + 0.001 * k),
                    curves_touched=("USD.SOFR.OIS",),
                    payment_lag=lag,
                )
                for k in range(len(_ANNUAL))
            ]
            return bootstrap_curve_graph(REF, [spec], quotes)

        lagged, diag = build(2)
        assert diag.converged
        assert float(diag.max_abs_residual) < 1e-12
        plain, _ = build(0)
        a = lagged.curves["USD.SOFR.OIS"].discount_factors
        b = plain.curves["USD.SOFR.OIS"].discount_factors
        assert jnp.any(jnp.abs(a - b) > 1e-9)
