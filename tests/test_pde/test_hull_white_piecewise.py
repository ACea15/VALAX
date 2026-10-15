"""Hull-White PDE with piecewise-constant σ(t), against the σ(t) closed form.

Each backward row diffuses at the exact step average of σ² (so a step that
straddles a breakpoint carries the right variance) and the domain is sized by
the state variance at the horizon. The European swaption must converge to
Jamshidian at second order under joint refinement — including a *steep*
σ(t) whose buckets differ by 5×, the case where a uniform time grid could be
expected to struggle — and the Bermudan must dominate every co-terminal.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core.binding import bind_curves
from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.pde import PDEConfig

REF = int(ymd_to_ordinal(2026, 1, 1))
RATE = 0.035
STEP_YEARS = (1.0, 2.0, 3.0, 4.0)
SMOOTH_VOLS = (0.012, 0.008, 0.015, 0.006, 0.010)
STEEP_VOLS = (0.004, 0.020, 0.004, 0.020, 0.004)
GRIDS = [(101, 60), (201, 120), (401, 240)]


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
MARKET = MarketState.single_curve(CURVE)


def _model(vols, a=0.05):
    return HullWhiteModel(
        mean_reversion=jnp.asarray(a),
        volatility=jnp.asarray(vols),
        volatility_times=jnp.asarray(STEP_YEARS),
    )


def _swaption(exp_days=730, n=5, strike=RATE, payer=True):
    expiry = REF + exp_days
    return Swaption(
        expiry_date=jnp.int32(expiry),
        fixed_dates=jnp.array([expiry + 365 * k for k in range(1, n + 1)], dtype=jnp.int32),
        strike=jnp.asarray(strike),
        notional=jnp.asarray(1.0),
        is_payer=payer,
        day_count="act_365",
    )


def _pde_errors(sw, model):
    exact = float(hw_swaption_price(sw, bind_curves(model, CURVE)))
    return exact, [
        float(price(sw, model, MARKET, method="pde",
                    config=PDEConfig(n_spot=ns, n_time=nt)).price) / exact - 1.0
        for ns, nt in GRIDS
    ]


class TestEuropeanConvergence:
    @pytest.mark.parametrize("vols", [SMOOTH_VOLS, STEEP_VOLS], ids=["smooth", "steep"])
    @pytest.mark.parametrize("exp_days,n,payer", [(730, 5, True), (548, 5, True), (1095, 4, False)])
    def test_second_order_to_jamshidian(self, vols, exp_days, n, payer):
        _, errs = _pde_errors(_swaption(exp_days, n, payer=payer), _model(vols))
        # Second order: each joint (space, time) halving divides the error ~4x.
        assert abs(errs[1]) < abs(errs[0]) / 3.0
        assert abs(errs[2]) < abs(errs[1]) / 3.0
        assert abs(errs[2]) < 3e-5

    def test_steep_case_measured(self):
        """Record the steep-σ(t) error on the finest grid (register evidence)."""
        _, errs = _pde_errors(_swaption(1095, 4, payer=False), _model(STEEP_VOLS))
        assert abs(errs[2]) < 3e-5


class TestBermudan:
    def test_dominates_coterminals(self):
        model = _model(SMOOTH_VOLS)
        berm = BermudanSwaption(
            exercise_dates=jnp.array([REF + 365 * k for k in (1, 2, 3, 4)], dtype=jnp.int32),
            fixed_dates=jnp.array([REF + 365 * k for k in range(2, 7)], dtype=jnp.int32),
            strike=jnp.asarray(RATE),
            notional=jnp.asarray(1.0),
            day_count="act_365",
        )
        pv = float(price(berm, model, MARKET, method="pde",
                         config=PDEConfig(n_spot=201, n_time=120)).price)
        euros = []
        for k in (1, 2, 3, 4):
            sw = Swaption(
                expiry_date=jnp.int32(REF + 365 * k),
                fixed_dates=jnp.array([REF + 365 * j for j in range(k + 1, 7)], dtype=jnp.int32),
                strike=jnp.asarray(RATE), notional=jnp.asarray(1.0), day_count="act_365",
            )
            euros.append(float(hw_swaption_price(sw, bind_curves(model, CURVE))))
        assert pv > max(euros)


def test_filter_jit_smoke():
    model = _model(SMOOTH_VOLS)
    sw = _swaption()
    f = eqx.filter_jit(
        lambda m, s: price(s, m, MARKET, method="pde",
                           config=PDEConfig(n_spot=101, n_time=60)).price
    )
    assert np.isfinite(float(f(model, sw)))
