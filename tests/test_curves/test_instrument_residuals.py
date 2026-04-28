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
from valax.curves.instruments import (
    DepositRate,
    FRA,
    MoneyMarketFuture,
    OISSwapRate,
    SwapRate,
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
