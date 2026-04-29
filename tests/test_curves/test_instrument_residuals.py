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
