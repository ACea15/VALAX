"""General Hull-White trinomial tree for piecewise-constant σ(t).

The tree has level-dependent spacing :math:`\\sqrt{3V(t_i, t_{i+1})}` and
moment-matched nearest-node branching. Validated as an *independent* engine
against the σ(t) closed form:

* the tree reprices zero-coupon bonds exactly (Arrow-Debreu α fit);
* with cell-averaged payoffs (default) the European swaption converges
  **monotonically at first order** to Jamshidian — the error halves with
  ``n_steps`` — so the Richardson extrapolate ``2V(2n) - V(n)`` is held to
  the measured ≤ 3e-6 (smooth and 5× steep σ(t));
* an undersized ``j_max`` yields NaN, never a wrong price;
* it is traceable (``filter_jit`` with traced instruments) and differentiable.

The Bermudan triangulation (tree vs PDE vs QuantLib ``Gsr``) lives in
``tests/test_quantlib_comparison/test_hull_white_piecewise_ql.py``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core.binding import bind_curves
from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.dates.schedule import generate_schedule
from valax.instruments.bonds import CallableBond
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.hull_white import HullWhiteModel, hw_market_df
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.lattice import (
    GeneralHullWhiteTree,
    bermudan_swaption_tree_price,
    build_hull_white_tree,
    callable_bond_price,
    hw_tree_width,
    swaption_tree_price,
)
from valax.pricing.pde import PDEConfig

REF = int(ymd_to_ordinal(2026, 1, 1))
RATE = 0.035
STEP_YEARS = (1.0, 2.0, 3.0, 4.0)
SMOOTH_VOLS = (0.012, 0.008, 0.015, 0.006, 0.010)
STEEP_VOLS = (0.004, 0.020, 0.004, 0.020, 0.004)


def _curve() -> DiscountCurve:
    years = [0.0] + [float(k) for k in range(1, 20)]
    pillars = jnp.array([REF + int(round(y * 365)) for y in years], dtype=jnp.int32)
    t = (pillars - REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-RATE * t),
        reference_date=jnp.int32(REF),
        day_count="act_365",
    )


CURVE = _curve()


def _free(vols):
    return HullWhiteModel(
        mean_reversion=jnp.asarray(0.05),
        volatility=jnp.asarray(vols),
        volatility_times=jnp.asarray(STEP_YEARS),
    )


def _model(vols):
    return bind_curves(_free(vols), CURVE)


def _swaption(exp_days, n, payer=True):
    expiry = REF + exp_days
    return Swaption(
        expiry_date=jnp.int32(expiry),
        fixed_dates=jnp.array([expiry + 365 * k for k in range(1, n + 1)], dtype=jnp.int32),
        strike=jnp.asarray(RATE),
        notional=jnp.asarray(1.0),
        is_payer=payer,
        day_count="act_365",
    )


class TestConstruction:
    def test_dispatches_to_general_tree(self):
        tree = build_hull_white_tree(_model(SMOOTH_VOLS), 5.0, 50)
        assert isinstance(tree, GeneralHullWhiteTree)
        assert bool(tree.valid)

    def test_probabilities(self):
        """Sum to one everywhere; positive wherever probability mass arrives.

        Edge nodes the tree never reaches have their targets clipped and may
        carry meaningless weights — harmless, and exactly what ``valid``
        guards — so positivity is checked on the reached nodes only, found by
        propagating the transition weights forward.
        """
        tree = build_hull_white_tree(_model(STEEP_VOLS), 5.0, 80)
        probs, targets = np.asarray(tree.probs), np.asarray(tree.targets)
        np.testing.assert_allclose(probs.sum(-1), 1.0, atol=1e-14)
        reach = np.zeros(2 * tree.j_max + 1)
        reach[tree.j_max] = 1.0
        for step in range(tree.n_steps):
            live = reach > 0.0
            assert np.all(probs[step][live] > 0.0)
            nxt = np.zeros_like(reach)
            for b in range(3):
                np.add.at(nxt, targets[step][:, b], reach * probs[step][:, b])
            reach = nxt
        assert bool(tree.valid)

    @pytest.mark.parametrize("vols", [SMOOTH_VOLS, STEEP_VOLS], ids=["smooth", "steep"])
    def test_reprices_the_curve(self, vols):
        """Rolling back 1 from level n gives P(0, t_n) (the α fit is exact)."""
        model = _model(vols)
        T, n = 4.5, 90
        tree = build_hull_white_tree(model, T, n)
        values = jnp.ones(2 * tree.j_max + 1)
        for step in range(n - 1, -1, -1):
            nxt = values[tree.targets[step]]
            values = jnp.exp(-tree.rates[step] * tree.dt) * jnp.sum(nxt * tree.probs[step], axis=1)
        assert float(values[tree.j_max]) == pytest.approx(
            float(hw_market_df(model, jnp.asarray(T))), rel=1e-12
        )

    def test_undersized_width_is_nan(self):
        """Anti-vacuity: a j_max below the reachable width must not price."""
        model = _model(STEEP_VOLS)
        sw = _swaption(1095, 4, payer=False)
        assert hw_tree_width(model, 3.0, 200) > 10
        assert np.isnan(float(swaption_tree_price(sw, model, n_steps=200, j_max=6)))
        assert np.isfinite(float(swaption_tree_price(sw, model, n_steps=200)))


class TestEuropeanConvergence:
    @pytest.mark.parametrize("vols", [SMOOTH_VOLS, STEEP_VOLS], ids=["smooth", "steep"])
    @pytest.mark.parametrize("exp_days,n", [(730, 5), (548, 5)])
    def test_first_order_and_richardson(self, vols, exp_days, n):
        model = _model(vols)
        sw = _swaption(exp_days, n)
        exact = float(hw_swaption_price(sw, model))
        v400 = float(swaption_tree_price(sw, model, n_steps=400))
        v800 = float(swaption_tree_price(sw, model, n_steps=800))
        e400, e800 = v400 / exact - 1.0, v800 / exact - 1.0
        assert 0.45 < e800 / e400 < 0.55          # clean first order
        assert abs((2.0 * v800 - v400) / exact - 1.0) < 5e-6


class TestBondsAndTraceability:
    def test_callable_bond_matches_pde(self):
        """σ(t) callable on the general tree vs the PDE (independent grids)."""
        sched = generate_schedule(2026, 1, 1, 2031, 1, 1, frequency=2).payment_dates
        bond = CallableBond(
            payment_dates=sched,
            settlement_date=jnp.int32(REF),
            coupon_rate=jnp.asarray(0.04),
            face_value=jnp.asarray(100.0),
            call_dates=jnp.array(
                [int(ymd_to_ordinal(y, 1, 1)) for y in (2028, 2029, 2030)], dtype=jnp.int32
            ),
            call_prices=jnp.array([1.0, 1.0, 1.0]),
            frequency=2,
        )
        tree = float(callable_bond_price(bond, _model(SMOOTH_VOLS), n_steps=400))
        pde = float(price(
            bond, _free(SMOOTH_VOLS), MarketState.single_curve(CURVE), method="pde",
            config=PDEConfig(n_spot=401, n_time=400),
        ).price)
        assert tree == pytest.approx(pde, rel=2e-4)

    def test_filter_jit_with_traced_instruments(self):
        model = _model(SMOOTH_VOLS)
        sw = _swaption(730, 5)
        berm = BermudanSwaption(
            exercise_dates=jnp.array([REF + 365 * k for k in (1, 2, 3, 4)], dtype=jnp.int32),
            fixed_dates=jnp.array([REF + 365 * k for k in range(2, 7)], dtype=jnp.int32),
            strike=jnp.asarray(RATE), notional=jnp.asarray(1.0), day_count="act_365",
        )
        j_euro = hw_tree_width(model, 2.0, 100)
        j_berm = hw_tree_width(model, 4.0, 100)
        euro = eqx.filter_jit(
            lambda s, m: swaption_tree_price(s, m, n_steps=100, j_max=j_euro)
        )(sw, model)
        berm_v = eqx.filter_jit(
            lambda b, m: bermudan_swaption_tree_price(b, m, n_steps=100, j_max=j_berm)
        )(berm, model)
        assert float(euro) == pytest.approx(
            float(swaption_tree_price(sw, model, n_steps=100)), rel=1e-12
        )
        assert float(berm_v) > float(euro)

    def test_vega_by_bucket(self):
        """∂V/∂σ_i: finite, zero for buckets starting after the expiry."""
        model = _model(SMOOTH_VOLS)
        sw = _swaption(730, 5)
        j = hw_tree_width(model, 2.0, 100)
        g = jax.grad(lambda v: swaption_tree_price(
            sw, eqx.tree_at(lambda m: m.volatility, model, v), n_steps=100, j_max=j
        ))(model.volatility)
        g = np.asarray(g)
        assert np.all(np.isfinite(g))
        assert np.all(g[:2] > 0.0)
        assert np.all(g[2:] == 0.0)
