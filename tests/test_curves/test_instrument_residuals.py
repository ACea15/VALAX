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
from valax.curves.instruments import DepositRate, FRA, SwapRate
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
