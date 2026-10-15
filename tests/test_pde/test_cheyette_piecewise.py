"""One-factor Cheyette PDEs with a piecewise-constant volatility level σ(t).

* **Reduction.** At β = 0 the frozen-``y`` PDE *is* the Hull-White PDE with the
  same σ(t): the Hull-White twin carries the buckets, each row diffuses at the
  exact step average of σ², and the Gaussian mean / accumulator are the
  piecewise ones. Measured agreement on a 201 x 120 grid: ≤ 3.3e-16 relative
  (round-off) for European and Bermudan swaptions.
* **Exact-``y`` vs Jamshidian** (β = 0). The 2-D exact-``y`` solver (first-order
  upwinded ``y`` axis) converges to the σ(t) closed form under joint
  refinement, at the same rate as for a constant σ. Measured relative errors
  on the 3Yx4Y receiver (41/81/161/321 x-nodes): -8.2e-3, -2.4e-3, -1.7e-3,
  -7.2e-4 (constant σ = 1.05%: -7.1e-3, -1.9e-3, -1.4e-3, -6.2e-4).
* **Frozen- vs exact-``y`` at β = 0.5** — the frozen-``y`` modelling bias,
  measured as a relative gap that shrinks with the exact-``y`` grid
  (+1.8e-3, +1.0e-3, +1.2e-4 on the 3Yx4Y; +1.6e-3, +1.1e-3, +3.7e-4 on the
  18Mx5Y): below the exact-``y`` discretisation error (~6e-4 at the finest
  grid) for these at-the-money swaptions.
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
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.cheyette import CheyetteModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.pde import PDEConfig, PDEConfig2D
from valax.pricing.pde.cheyette import (
    cheyette_exact_operator_2d,
    cheyette_exact_price,
)
from valax.pricing.pde.grids import Grid2D, centred_state_grid, uniform_linear_grid

REF = int(ymd_to_ordinal(2026, 1, 1))
RATE = 0.035
KAPPA = 0.05
STEP_YEARS = (1.0, 2.0, 3.0, 4.0)
VOLS = (0.012, 0.008, 0.015, 0.006, 0.010)
CFG_1D = PDEConfig(n_spot=201, n_time=120)
CFG_1D_FAST = PDEConfig(n_spot=101, n_time=60)
EXACT_GRIDS = [(41, 31, 36), (81, 61, 72), (161, 121, 144), (321, 241, 288)]


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


def _cheyette(skew, vols=VOLS, times=STEP_YEARS):
    return CheyetteModel(
        mean_reversion=jnp.asarray(KAPPA),
        volatility=jnp.asarray(vols),
        skew=jnp.asarray(skew),
        volatility_times=jnp.asarray(times),
    )


def _hull_white(vols=VOLS, times=STEP_YEARS):
    return HullWhiteModel(
        mean_reversion=jnp.asarray(KAPPA),
        volatility=jnp.asarray(vols),
        volatility_times=jnp.asarray(times),
    )


def _swaption(exp_days, n, payer=True, strike=RATE):
    expiry = REF + exp_days
    return Swaption(
        expiry_date=jnp.int32(expiry),
        fixed_dates=jnp.array([expiry + 365 * k for k in range(1, n + 1)], dtype=jnp.int32),
        strike=jnp.asarray(strike),
        notional=jnp.asarray(1.0),
        is_payer=payer,
        day_count="act_365",
    )


SWAPTIONS = {
    "3Yx4Y_receiver": _swaption(1095, 4, payer=False),
    "18Mx5Y_payer": _swaption(548, 5, payer=True),
}

BERMUDAN = BermudanSwaption(
    exercise_dates=jnp.array([REF + 365 * k for k in (1, 2, 3, 4)], dtype=jnp.int32),
    fixed_dates=jnp.array([REF + 365 * k for k in range(2, 7)], dtype=jnp.int32),
    strike=jnp.asarray(RATE),
    notional=jnp.asarray(1.0),
    day_count="act_365",
)


def _coterminals():
    return [
        Swaption(
            expiry_date=jnp.int32(REF + 365 * k),
            fixed_dates=jnp.array([REF + 365 * j for j in range(k + 1, 7)], dtype=jnp.int32),
            strike=jnp.asarray(RATE), notional=jnp.asarray(1.0), day_count="act_365",
        )
        for k in (1, 2, 3, 4)
    ]


def _exact_cfg(nx, ny, nt):
    return PDEConfig2D(n_x=nx, n_y=ny, n_time=nt, x_range=6.0, y_max=3.0)


# ── β = 0: frozen-y is the Hull-White PDE ────────────────────────────


class TestHullWhiteReduction:
    @pytest.mark.parametrize("name", list(SWAPTIONS))
    def test_frozen_european_equals_hw_pde(self, name):
        sw = SWAPTIONS[name]
        ch = float(price(sw, _cheyette(0.0), MARKET, method="pde", config=CFG_1D).price)
        hw = float(price(sw, _hull_white(), MARKET, method="pde", config=CFG_1D).price)
        # Measured: <= 3.3e-16 relative (round-off).
        assert ch == pytest.approx(hw, rel=1e-13)

    def test_frozen_bermudan_equals_hw_pde(self):
        ch = float(price(BERMUDAN, _cheyette(0.0), MARKET, method="pde", config=CFG_1D).price)
        hw = float(price(BERMUDAN, _hull_white(), MARKET, method="pde", config=CFG_1D).price)
        assert ch == pytest.approx(hw, rel=1e-13)

    @pytest.mark.parametrize("name", list(SWAPTIONS))
    def test_exact_y_converges_to_jamshidian(self, name):
        sw = SWAPTIONS[name]
        model = bind_curves(_cheyette(0.0), CURVE)
        jam = float(hw_swaption_price(sw, bind_curves(_hull_white(), CURVE)))
        errs = [
            float(cheyette_exact_price(sw, model, _exact_cfg(*g)).price) / jam - 1.0
            for g in EXACT_GRIDS
        ]
        # Monotone under joint refinement (first-order upwinded y axis).
        assert all(abs(b) < abs(a) for a, b in zip(errs[:-1], errs[1:])), errs
        assert abs(errs[-1]) < 1e-3, errs


# ── β ≠ 0 ────────────────────────────────────────────────────────────


class TestSkewed:
    @pytest.mark.parametrize("name", list(SWAPTIONS))
    def test_frozen_vs_exact_gap(self, name):
        """The frozen-y modelling bias, measured against the exact-y PDE."""
        sw = SWAPTIONS[name]
        frozen = float(price(sw, _cheyette(0.5), MARKET, method="pde", config=CFG_1D).price)
        model = bind_curves(_cheyette(0.5), CURVE)
        gaps = [
            frozen / float(cheyette_exact_price(sw, model, _exact_cfg(*g)).price) - 1.0
            for g in EXACT_GRIDS[1:]
        ]
        # Measured: 1.8e-3 / 1.0e-3 / 1.2e-4 (3Yx4Y), 1.6e-3 / 1.1e-3 / 3.7e-4
        # (18Mx5Y); the finest is inside the exact-y grid error (~6e-4).
        assert abs(gaps[-1]) < abs(gaps[0]), gaps
        assert abs(gaps[-1]) < 1e-3, gaps

    def test_bermudan_dominates_coterminals(self):
        model = _cheyette(0.5)
        pv = float(price(BERMUDAN, model, MARKET, method="pde", config=CFG_1D).price)
        euros = [
            float(price(sw, model, MARKET, method="pde", config=CFG_1D).price)
            for sw in _coterminals()
        ]
        assert pv >= max(euros)

    def test_skew_moves_the_price(self):
        otm = _swaption(1095, 4, payer=True, strike=RATE + 0.01)
        p0 = float(price(otm, _cheyette(0.0), MARKET, method="pde", config=CFG_1D_FAST).price)
        p5 = float(price(otm, _cheyette(0.5), MARKET, method="pde", config=CFG_1D_FAST).price)
        assert p5 > p0  # positive skew fattens the high-rate tail


# ── Constant through the general path, gradients, JIT ────────────────


class TestGeneralPath:
    def test_equal_buckets_equal_constant_frozen(self):
        sw = SWAPTIONS["3Yx4Y_receiver"]
        const = _cheyette(0.5, vols=(0.011,), times=())
        split = _cheyette(0.5, vols=(0.011, 0.011), times=(1.5,))
        pc = float(price(sw, const, MARKET, method="pde", config=CFG_1D_FAST).price)
        ps = float(price(sw, split, MARKET, method="pde", config=CFG_1D_FAST).price)
        assert ps == pytest.approx(pc, rel=1e-12)

    def test_equal_buckets_equal_constant_exact(self):
        sw = SWAPTIONS["3Yx4Y_receiver"]
        cfg = _exact_cfg(41, 31, 36)
        const = bind_curves(_cheyette(0.5, vols=(0.011,), times=()), CURVE)
        split = bind_curves(_cheyette(0.5, vols=(0.011, 0.011), times=(1.5,)), CURVE)
        pc = float(cheyette_exact_price(sw, const, cfg).price)
        ps = float(cheyette_exact_price(sw, split, cfg).price)
        assert ps == pytest.approx(pc, rel=1e-12)

    def test_exact_operator_refuses_piecewise(self):
        model = _cheyette(0.5)
        grid = Grid2D(
            x=centred_state_grid(jnp.asarray(0.01), n=11, half_width=5.0),
            y=uniform_linear_grid(jnp.asarray(-1e-5), jnp.asarray(1e-4), n=11),
        )
        with pytest.raises(ValueError, match="cheyette_exact_operator_rows"):
            cheyette_exact_operator_2d(model, grid)

    def test_frozen_gradient_matches_fd(self):
        sw = SWAPTIONS["3Yx4Y_receiver"]

        def f(v):
            m = eqx.tree_at(lambda x: x.volatility, _cheyette(0.5), v)
            return price(sw, m, MARKET, method="pde", config=CFG_1D_FAST).price

        v = jnp.asarray(VOLS)
        grad = np.asarray(jax.grad(f)(v))
        assert np.all(np.isfinite(grad))
        h = 1e-6
        bucket_starts = (0.0,) + STEP_YEARS
        for i in range(len(VOLS)):
            fd = (float(f(v.at[i].add(h))) - float(f(v.at[i].add(-h)))) / (2 * h)
            if bucket_starts[i] >= 3.0:
                # Buckets starting at or after the 3Y expiry have no effect.
                assert grad[i] == 0.0 and fd == 0.0
            else:
                assert grad[i] == pytest.approx(fd, rel=1e-4)

    def test_filter_jit_smoke(self):
        sw = SWAPTIONS["3Yx4Y_receiver"]
        frozen = eqx.filter_jit(
            lambda m, s: price(s, m, MARKET, method="pde", config=CFG_1D_FAST).price
        )
        exact = eqx.filter_jit(
            lambda m, s: cheyette_exact_price(s, m, _exact_cfg(41, 31, 36)).price
        )
        assert np.isfinite(float(frozen(_cheyette(0.5), sw)))
        assert np.isfinite(float(exact(bind_curves(_cheyette(0.5), CURVE), sw)))
