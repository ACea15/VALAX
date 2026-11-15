"""Tests for revaluation grids (ladders) on a ``MarketState``.

Mirrors ``valax/risk/revaluation_grids.py``. What is gated:

1. **Exactness where it must be exact** — the grid's PVs are full
   revaluations, its node deltas match finite differences of the same shock
   map, and the Hermite P&L reproduces the full revaluation at every rung.
2. **Higher than second order** — on a large spot move of an option book the
   grid P&L is orders of magnitude closer to full revaluation than the
   second-order Taylor explain, and its error converges with the rung
   spacing at the interpolant's order (Hermite ``O(w^4)``, delta-integration
   ``O(w^2)``).
3. **Cross terms** — summed per-direction ladders miss cross-direction
   terms; adding them from second-order ``KeyedSensitivities`` closes the
   residual on a dual-curve swap book.
4. **The contract** — validation, out-of-range flagging, batching, and
   ``filter_jit``.
"""

import warnings

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core.market_state import MarketState
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.options import EuropeanOption
from valax.instruments.rates import InterestRateSwap
from valax.models.black_scholes import BlackScholesModel
from valax.risk import (
    GridPnL,
    KeyedFactorLayout,
    RevaluationGrid,
    key_rate_directions,
    keyed_sensitivities,
    parallel_direction,
    revaluation_grid,
    revaluation_grid_pnl,
)
from valax.risk.state_shocks import shock
from valax.risk.state_var import group_pv
from valax.surfaces.constant import ConstantVol


# ── Fixtures ─────────────────────────────────────────────────────────

REF = ymd_to_ordinal(2026, 1, 1)
PILLARS = jnp.array([
    ymd_to_ordinal(2026, 1, 1),
    ymd_to_ordinal(2026, 7, 1),
    ymd_to_ordinal(2027, 1, 1),
    ymd_to_ordinal(2028, 1, 1),
])
TIMES = (PILLARS - REF).astype(jnp.float64) / 365.0
OIS = "USD.SOFR.OIS"
FWD = "USD.SOFR.3M"
SPOT = "EQ.SPOT.SPX"


def _curve(rate: float) -> DiscountCurve:
    return DiscountCurve(
        pillar_dates=PILLARS,
        discount_factors=jnp.exp(-rate * TIMES),
        reference_date=REF,
    )


@pytest.fixture(scope="module")
def equity_state():
    return MarketState(
        curves=CurveGraph(curves={"_default_": _curve(0.05)}),
        surfaces={"_default_": ConstantVol(vol=jnp.asarray(0.2))},
        spots={"SPX": jnp.asarray(100.0)},
        dividends={"SPX": jnp.asarray(0.01)},
        asset_ids=("SPX",),
    )


@pytest.fixture(scope="module")
def options():
    return EuropeanOption(
        strike=jnp.array([100.0, 110.0]),
        expiry=jnp.array([1.0, 1.0]),
        is_call=True,
        underlying_id="SPX",
    )


@pytest.fixture(scope="module")
def bs():
    """Market-sourced Black-Scholes: spot, vol and rate come off the market."""
    return BlackScholesModel()


@pytest.fixture(scope="module")
def eq_layout(equity_state):
    return KeyedFactorLayout.from_market_state(equity_state)


@pytest.fixture(scope="module")
def spot_dir(eq_layout):
    return parallel_direction(eq_layout, "spot", "SPX")


@pytest.fixture(scope="module")
def spot_grid(options, bs, equity_state, eq_layout, spot_dir):
    return revaluation_grid(
        options, bs, equity_state, spot_dir, np.arange(-30.0, 30.0 + 1e-9, 10.0),
        layout=eq_layout,
    )


@pytest.fixture(scope="module")
def dual_state():
    return MarketState(
        curves=CurveGraph(curves={OIS: _curve(0.05), FWD: _curve(0.055)}),
    )


@pytest.fixture(scope="module")
def dual_swaps():
    pay = [ymd_to_ordinal(2027, 1, 1), ymd_to_ordinal(2028, 1, 1)]
    return InterestRateSwap(
        start_date=jnp.array([REF, REF]),
        fixed_dates=jnp.array([pay, pay]),
        fixed_rate=jnp.array([0.04, 0.05]),
        notional=jnp.array([1e6, 1e6]),
        discount_id=OIS,
        forward_id=FWD,
    )


def _full_pnl(instruments, model, market, layout, x):
    """Full-revaluation P&L of the factor move ``x``."""
    return float(
        group_pv(instruments, model, shock(market, layout.unflatten(x)))
        - group_pv(instruments, model, market)
    )


# ── 1. Exactness ─────────────────────────────────────────────────────


class TestExactness:
    def test_pv_at_every_rung_is_a_full_revaluation(
        self, spot_grid, options, bs, equity_state, eq_layout,
    ):
        d = spot_grid.directions[0]
        for j, h in enumerate(np.asarray(spot_grid.levels)):
            expected = _full_pnl(options, bs, equity_state, eq_layout, h * d)
            assert float(spot_grid.ladder(SPOT)[j]) == pytest.approx(
                expected, rel=1e-12, abs=1e-12)

    def test_base_pv_matches_group_pv(self, spot_grid, options, bs, equity_state):
        assert float(spot_grid.base_pv) == pytest.approx(
            float(group_pv(options, bs, equity_state)), rel=1e-12)

    def test_node_delta_matches_finite_differences(
        self, spot_grid, options, bs, equity_state, eq_layout,
    ):
        d, h = spot_grid.directions[0], 1e-4
        for j, lv in enumerate(np.asarray(spot_grid.levels)):
            fd = (_full_pnl(options, bs, equity_state, eq_layout, (lv + h) * d)
                  - _full_pnl(options, bs, equity_state, eq_layout, (lv - h) * d)) / (2 * h)
            assert float(spot_grid.delta_ladder(SPOT)[j]) == pytest.approx(fd, rel=1e-6)

    def test_hermite_pnl_is_exact_at_the_rungs(self, spot_grid):
        levels = np.asarray(spot_grid.levels)
        pnl = revaluation_grid_pnl(spot_grid, levels[:, None])
        np.testing.assert_allclose(
            np.asarray(pnl.by_direction[:, 0]), np.asarray(spot_grid.ladder(SPOT)),
            rtol=1e-12, atol=1e-12)

    def test_zero_move_is_zero_pnl(self, spot_grid):
        for mode in ("hermite", "delta"):
            pnl = revaluation_grid_pnl(spot_grid, {SPOT: 0.0}, interpolation=mode)
            assert float(pnl.total) == pytest.approx(0.0, abs=1e-12)


# ── 2. Higher than second order ──────────────────────────────────────


class TestHigherOrder:
    MOVE = 23.7  # between rungs, and large enough for the Taylor explain to break

    def test_grid_beats_second_order_taylor_on_a_large_move(
        self, spot_grid, options, bs, equity_state, eq_layout,
    ):
        x = self.MOVE * spot_grid.directions[0]
        exact = _full_pnl(options, bs, equity_state, eq_layout, x)
        sens = keyed_sensitivities(options, bs, equity_state, eq_layout, second_order=True)
        taylor = float(sens.delta @ x + 0.5 * x @ sens.gamma @ x)
        grid = float(revaluation_grid_pnl(spot_grid, {SPOT: self.MOVE}).total)

        taylor_err, grid_err = abs(taylor - exact), abs(grid - exact)
        assert taylor_err > 1.0                 # measured 1.76 on a 33.6 P&L
        assert grid_err < taylor_err / 1000.0   # measured 5.9e-4 at 10-point rungs

    @pytest.mark.parametrize("mode, min_ratio", [("hermite", 8.0), ("delta", 3.5)])
    def test_error_converges_at_the_interpolant_order(
        self, mode, min_ratio, options, bs, equity_state, eq_layout, spot_dir,
    ):
        """Halving the rung spacing cuts the error by ~2**4 (Hermite) / 2**2 (delta)."""
        x = self.MOVE * next(iter(spot_dir.values()))
        exact = _full_pnl(options, bs, equity_state, eq_layout, x)
        errs = []
        for step in (10.0, 5.0, 2.5):
            g = revaluation_grid(options, bs, equity_state, spot_dir,
                                 np.arange(-30.0, 30.0 + 1e-9, step), layout=eq_layout)
            got = float(revaluation_grid_pnl(g, {SPOT: self.MOVE}, interpolation=mode).total)
            errs.append(abs(got - exact))
        assert errs[0] / errs[1] > min_ratio
        assert errs[1] / errs[2] > min_ratio


# ── 3. Cross terms ───────────────────────────────────────────────────


class TestCrossTerms:
    def test_sensitivity_cross_terms_close_the_multi_curve_residual(
        self, dual_swaps, dual_state,
    ):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        dirs = key_rate_directions(layout, OIS) | key_rate_directions(layout, FWD)
        grid = revaluation_grid(dual_swaps, None, dual_state, dirs,
                                np.linspace(-0.02, 0.02, 9), layout=layout)
        sens = keyed_sensitivities(dual_swaps, None, dual_state, layout, second_order=True)
        moves = jnp.full((len(dirs),), 0.01)          # +100 bp on every pillar of both curves
        x = moves @ grid.directions
        exact = _full_pnl(dual_swaps, None, dual_state, layout, x)

        without = revaluation_grid_pnl(grid, moves)
        with_cross = revaluation_grid_pnl(grid, moves, sensitivities=sens)
        taylor = float(sens.delta @ x + 0.5 * x @ sens.gamma @ x)

        assert float(without.cross) == 0.0
        assert abs(float(without.total) - exact) > 100.0           # measured 966
        assert abs(float(with_cross.total) - exact) < 0.1          # measured 0.027
        assert abs(float(with_cross.total) - exact) < abs(taylor - exact) / 10.0
        assert float(with_cross.total) == pytest.approx(
            float(jnp.sum(with_cross.by_direction)) + float(with_cross.cross), rel=1e-12)

    def test_key_rate_labels_are_factor_ids(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        dirs = key_rate_directions(layout, OIS)
        assert tuple(dirs) == tuple(
            i for i in layout.factor_ids() if i.startswith(f"IR.CURVE.{OIS}."))


# ── 4. Contract ──────────────────────────────────────────────────────


class TestContract:
    def test_out_of_range_is_flagged_and_linearly_extrapolated(self, spot_grid):
        top = float(spot_grid.levels[-1])
        pnl = revaluation_grid_pnl(spot_grid, {SPOT: jnp.array([top - 1.0, top + 5.0])})
        assert np.asarray(pnl.out_of_range[:, 0]).tolist() == [False, True]
        expected = float(spot_grid.ladder(SPOT)[-1] + spot_grid.delta_ladder(SPOT)[-1] * 5.0)
        assert float(pnl.by_direction[1, 0]) == pytest.approx(expected, rel=1e-12)

    def test_batched_moves_keep_their_shape(self, spot_grid):
        moves = jnp.linspace(-25.0, 25.0, 12).reshape(3, 4, 1)
        pnl = revaluation_grid_pnl(spot_grid, moves)
        assert isinstance(pnl, GridPnL)
        assert pnl.by_direction.shape == (3, 4, 1)
        assert pnl.total.shape == (3, 4)

    def test_filter_jit_matches_eager(self, spot_grid, options, bs, equity_state, eq_layout, spot_dir):
        eager = revaluation_grid_pnl(spot_grid, {SPOT: 12.3}).total
        jitted = eqx.filter_jit(revaluation_grid_pnl)(spot_grid, {SPOT: 12.3}).total
        assert float(jitted) == pytest.approx(float(eager), rel=1e-12)
        build = eqx.filter_jit(
            lambda m: revaluation_grid(options, bs, m, spot_dir, [-10.0, 0.0, 10.0],
                                       layout=eq_layout))
        grid = build(equity_state)
        assert isinstance(grid, RevaluationGrid)
        assert float(grid.ladder(SPOT)[2]) == pytest.approx(
            float(spot_grid.ladder(SPOT)[4]), rel=1e-10)

    @pytest.mark.parametrize("levels, match", [
        ([0.0], "at least 2"),
        ([0.0, 1.0, 0.5], "strictly increasing"),
        ([1.0, 2.0], "contain 0.0"),
    ])
    def test_bad_levels_raise(self, options, bs, equity_state, spot_dir, levels, match):
        with pytest.raises(ValueError, match=match):
            revaluation_grid(options, bs, equity_state, spot_dir, levels)

    def test_bad_directions_raise(self, options, bs, equity_state, eq_layout):
        with pytest.raises(ValueError, match="at least one direction"):
            revaluation_grid(options, bs, equity_state, {}, [0.0, 1.0])
        with pytest.raises(ValueError, match="shape"):
            revaluation_grid(options, bs, equity_state, {"x": jnp.ones(3)}, [0.0, 1.0])
        with pytest.raises(ValueError, match="zero vector"):
            revaluation_grid(options, bs, equity_state,
                             {"x": jnp.zeros(eq_layout.n_factors)}, [0.0, 1.0])

    def test_overlapping_directions_warn_and_give_no_total(
        self, options, bs, equity_state, eq_layout, spot_dir,
    ):
        dirs = spot_dir | {"twice": 2.0 * next(iter(spot_dir.values()))}
        with pytest.warns(UserWarning, match="overlap"):
            grid = revaluation_grid(options, bs, equity_state, dirs, [-1.0, 0.0, 1.0],
                                    layout=eq_layout)
        pnl = revaluation_grid_pnl(grid, {SPOT: 0.5, "twice": 0.5})
        assert np.isfinite(np.asarray(pnl.by_direction)).all()
        assert np.isnan(float(pnl.total))

    def test_bad_moves_and_options_raise(self, spot_grid, options, bs, equity_state, eq_layout):
        with pytest.raises(ValueError, match="unknown directions"):
            revaluation_grid_pnl(spot_grid, {"nope": 1.0})
        with pytest.raises(ValueError, match="trailing dimension"):
            revaluation_grid_pnl(spot_grid, jnp.ones(3))
        with pytest.raises(ValueError, match="interpolation"):
            revaluation_grid_pnl(spot_grid, {SPOT: 1.0}, interpolation="spline")
        first = keyed_sensitivities(options, bs, equity_state, eq_layout)
        with pytest.raises(ValueError, match="second-order"):
            revaluation_grid_pnl(spot_grid, {SPOT: 1.0}, sensitivities=first)

    def test_keyed_scenario_as_direction(self, options, bs, equity_state, eq_layout, spot_grid):
        unit = eq_layout.unflatten(spot_grid.directions[0])
        grid = revaluation_grid(options, bs, equity_state, {"spot": unit},
                                [-10.0, 0.0, 10.0], layout=eq_layout)
        assert float(grid.ladder("spot")[2]) == pytest.approx(
            float(spot_grid.ladder(SPOT)[4]), rel=1e-12)


# ── Guide example (docs/risk-revaluation-grids.md) ───────────────────


def test_guide_example_runs(dual_swaps, dual_state):
    """The quick-start of the guide page, verbatim apart from the fixtures."""
    swaps, market = dual_swaps, dual_state
    layout = KeyedFactorLayout.from_market_state(market)

    dirs = key_rate_directions(layout, OIS) | key_rate_directions(layout, FWD)
    levels = jnp.linspace(-0.02, 0.02, 9)            # -200 bp … +200 bp, 50 bp rungs
    grid = revaluation_grid(swaps, None, market, dirs, levels, layout=layout)

    sens = keyed_sensitivities(swaps, None, market, layout, second_order=True)
    today = {fid: 0.0125 for fid in dirs}            # +125 bp everywhere: between rungs
    pnl = revaluation_grid_pnl(grid, today, sensitivities=sens)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = pnl.by_direction, pnl.cross, pnl.total
    assert not bool(jnp.any(pnl.out_of_range))
    x = jnp.asarray([today[f] for f in grid.labels]) @ grid.directions
    exact = _full_pnl(swaps, None, market, layout, x)
    assert float(pnl.total) == pytest.approx(exact, rel=1e-5)
