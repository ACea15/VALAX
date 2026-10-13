"""Tests for the MultiCurveSet view over CurveGraph (MC-Curves-2b).

The two-tier set is built with :func:`valax.curves.bootstrap_curve_graph`
and viewed through :meth:`MultiCurveSet.from_graph`.  These tests pin:

1. the dual-curve calibration target (every input quote reprices on the
   curves it was quoted against);
2. :class:`MultiCurveSet` round-trips through :class:`CurveGraph`;
3. gradients flow through the joint build on the swap legs.
"""

import jax
import jax.numpy as jnp

from valax.curves.bootstrap_graph import bootstrap_curve_graph
from valax.curves.graph import CurveGraph, CurveSpec
from valax.curves.instruments import DepositRate, IborSwapRate, SwapRate
from valax.curves.multi_curve import (
    MultiCurveSet,
    _LEGACY_DISCOUNT_ID,
    _legacy_forward_id,
)
from valax.dates.daycounts import ymd_to_ordinal, year_fraction


REF = ymd_to_ordinal(2025, 1, 1)
DC = "act_365"
OIS_ID = "USD.SOFR.OIS"
FWD_ID = "USD.SOFR.3M"


def _deposit(y, m, d, rate):
    return DepositRate(
        start_date=REF,
        end_date=ymd_to_ordinal(y, m, d),
        rate=jnp.array(rate),
        day_count=DC,
    )


def _swap(end_year, rate):
    """Annual-pay par swap quote maturing on 1 Jan ``end_year``."""
    dates = [ymd_to_ordinal(y, 1, 1) for y in range(2026, end_year + 1)]
    return SwapRate(
        start_date=REF,
        fixed_dates=jnp.stack(dates).astype(jnp.int32),
        rate=jnp.array(rate),
        day_count=DC,
    )


def _legacy_inputs():
    """A two-tier build with deposits *and* swaps on both legs."""
    discount_instruments = [
        _deposit(2025, 4, 1, 0.040),
        _swap(2027, 0.042),
        _swap(2029, 0.043),
    ]
    forward_instruments = {
        "3M": [
            _deposit(2025, 4, 1, 0.045),
            _swap(2027, 0.047),
            _swap(2029, 0.048),
        ],
    }
    return discount_instruments, forward_instruments


def _graph_instruments(disc, fwd_insts):
    """Retarget the two-tier quotes onto the joint ``CurveGraph`` system.

    Discount quotes touch the OIS curve only; forward-leg deposits touch
    the projection curve only; forward-leg swaps become dual-curve
    (OIS-discounted) IBOR swaps.
    """
    instruments = [_retarget(i, (OIS_ID,)) for i in disc]
    for inst in fwd_insts:
        if isinstance(inst, DepositRate):
            instruments.append(_retarget(inst, (FWD_ID,)))
        else:
            instruments.append(_as_ibor(inst, OIS_ID, FWD_ID))
    return instruments


def _specs(disc, fwd_insts):
    def pillars(insts):
        return jnp.sort(
            jnp.stack(
                [jnp.asarray(_maturity(i), dtype=jnp.int32) for i in insts]
            )
        )

    return [
        CurveSpec(
            curve_id=OIS_ID, currency="USD",
            pillar_dates=pillars(disc), day_count=DC,
        ),
        CurveSpec(
            curve_id=FWD_ID, currency="USD",
            pillar_dates=pillars(fwd_insts), day_count=DC,
        ),
    ]


def _build_graph(disc, fwd_insts):
    return bootstrap_curve_graph(
        REF, _specs(disc, fwd_insts), _graph_instruments(disc, fwd_insts)
    )


def _build():
    disc, fwd = _legacy_inputs()
    res = _build_graph(disc, fwd["3M"])
    assert bool(res.converged)
    assert float(res.diagnostics.max_abs_residual) < 1e-10
    return MultiCurveSet.from_graph(res.value, OIS_ID, {"3M": FWD_ID})


# ── Calibration target is unchanged ──────────────────────────────────

class TestCalibrationTarget:
    def test_discount_leg_reprices(self):
        """Every discount quote reprices on the OIS curve alone."""
        disc, _ = _legacy_inputs()
        mcs = _build()
        graph = CurveGraph(curves={"_default_": mcs.discount_curve})
        for inst in disc:
            residual = inst.residual(graph, _empty_fixings(), REF)
            assert abs(float(residual)) < 1e-10

    def test_forward_deposits_reprice_on_their_own_curve(self):
        """Forward-leg deposits keep the single-curve DF relationship."""
        _, fwd = _legacy_inputs()
        mcs = _build()
        curve = mcs.forward_curves["3M"]
        for inst in fwd["3M"]:
            if not isinstance(inst, DepositRate):
                continue
            tau = year_fraction(REF, inst.end_date, inst.day_count)
            expected = 1.0 / (1.0 + float(inst.rate) * float(tau))
            assert abs(float(curve(inst.end_date)) - expected) < 1e-10

    def test_forward_swaps_reprice_dual_curve(self):
        """Forward-leg swaps satisfy the OIS-discounted par condition.

        float leg = sum_i (DF_fwd(T_{i-1})/DF_fwd(T_i) - 1) * DF_ois(T_i)
        fixed leg = r * sum_i tau_i * DF_ois(T_i)
        """
        _, fwd = _legacy_inputs()
        mcs = _build()
        ois = mcs.discount_curve
        proj = mcs.forward_curves["3M"]
        for inst in fwd["3M"]:
            if not isinstance(inst, SwapRate):
                continue
            starts = jnp.concatenate(
                [inst.start_date[None], inst.fixed_dates[:-1]]
            )
            taus = year_fraction(starts, inst.fixed_dates, inst.day_count)
            ois_dfs = ois(inst.fixed_dates)
            fixed_leg = float(inst.rate) * float(jnp.sum(taus * ois_dfs))
            fwd_rates = proj(starts) / proj(inst.fixed_dates) - 1.0
            float_leg = float(jnp.sum(fwd_rates * ois_dfs))
            assert abs(float_leg - fixed_leg) < 1e-10


def _empty_fixings():
    from valax.curves.fixings import empty_fixing_history

    return empty_fixing_history()


# ── Helpers for the joint-solver build ─────────────────────────────

def _maturity(inst):
    return inst.end_date if isinstance(inst, DepositRate) else inst.fixed_dates[-1]


def _retarget(inst, curves_touched):
    import dataclasses

    return dataclasses.replace(inst, curves_touched=curves_touched)


def _as_ibor(swap, discount_id, forward_id):
    return IborSwapRate(
        start_date=swap.start_date,
        fixed_dates=swap.fixed_dates,
        float_dates=swap.fixed_dates,
        fixing_dates=jnp.concatenate(
            [swap.start_date[None], swap.fixed_dates[:-1]]
        ),
        rate=swap.rate,
        fixed_day_count=swap.day_count,
        float_day_count=swap.day_count,
        curves_touched=(discount_id, forward_id),
    )


# ── MultiCurveSet as a view over CurveGraph ──────────────────────────

class TestGraphView:
    def test_to_graph_exposes_both_tiers(self):
        mcs = _build()
        graph = mcs.to_graph()
        assert isinstance(graph, CurveGraph)
        assert _LEGACY_DISCOUNT_ID in graph
        assert _legacy_forward_id("3M", 0) in graph
        assert graph[_LEGACY_DISCOUNT_ID] is mcs.discount_curve

    def test_round_trip_is_value_identical(self):
        mcs = _build()
        graph = mcs.to_graph()
        restored = MultiCurveSet.from_graph(
            graph,
            _LEGACY_DISCOUNT_ID,
            {"3M": _legacy_forward_id("3M", 0)},
        )
        assert jnp.array_equal(
            restored.discount_curve.discount_factors,
            mcs.discount_curve.discount_factors,
        )
        assert jnp.array_equal(
            restored.forward_curves["3M"].discount_factors,
            mcs.forward_curves["3M"].discount_factors,
        )

    def test_to_graph_accepts_explicit_ids(self):
        mcs = _build()
        graph = mcs.to_graph(
            discount_id="USD.SOFR.OIS", forward_ids={"3M": "USD.SOFR.3M"},
        )
        assert set(graph.keys()) == {"USD.SOFR.OIS", "USD.SOFR.3M"}

    def test_free_form_tenor_label_gets_a_valid_id(self):
        """Non-conformant tenor labels fall back to the qualifier form."""
        curve_id = _legacy_forward_id("three_month", 2)
        # Must be accepted by CurveSpec's frozen alphabet.
        CurveSpec(
            curve_id=curve_id,
            currency="XXX",
            pillar_dates=jnp.array([ymd_to_ordinal(2026, 1, 1)], jnp.int32),
        )


# ── Differentiability (new in MC-Curves-2b) ──────────────────────────

class TestGradients:
    def test_grad_flows_through_swap_quote(self):
        """jax.grad w.r.t. a forward-leg swap rate is finite and nonzero.

        The joint solver's ImplicitAdjoint differentiates through the
        dual-curve build.
        """
        disc, fwd = _legacy_inputs()

        def forward_df(rate):
            insts = list(fwd["3M"])
            insts[1] = SwapRate(
                start_date=insts[1].start_date,
                fixed_dates=insts[1].fixed_dates,
                rate=rate,
                day_count=insts[1].day_count,
            )
            graph = _build_graph(disc, insts).value
            mcs = MultiCurveSet.from_graph(graph, OIS_ID, {"3M": FWD_ID})
            return mcs.forward_curves["3M"](ymd_to_ordinal(2027, 1, 1))

        grad = jax.grad(forward_df)(jnp.array(0.047))
        assert jnp.isfinite(grad)
        assert float(grad) < 0.0  # DF falls as the projected rate rises
