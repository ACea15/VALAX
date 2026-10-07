"""Tests for the keyed, per-curve factor layout and its generators.

Mirrors the keyed half of ``valax/risk/scenarios.py``.  The property under
test throughout is that a **named curve owns its own coordinates**: the
flat ``[spots, vols, rates, dividends]`` layout cannot say which curve
moved, and everything here exists to fix that.
"""

import jax
import jax.numpy as jnp
import pytest

from valax.core.market_state import MarketState
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import InterestRateSwap
from valax.market.keyed_scenario import KeyedScenario
from valax.risk.scenarios import (
    CURVE_SHAPES,
    FactorBlock,
    KeyedFactorLayout,
    as_basis_layout,
    basis_factor_prefix,
    block_covariance,
    butterfly,
    curve_shape_bumps,
    historical_block_covariance,
    keyed_basis_shift,
    keyed_butterfly,
    keyed_curve_stress,
    keyed_flattener,
    keyed_historical_scenarios,
    keyed_parametric_scenarios,
    keyed_steepener,
    keyed_stress_scenario,
    steepener,
)
from valax.risk.state_shocks import shock
from valax.risk.state_var import group_pv, group_scenario_pnl
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
N_PILLARS = 4

OIS = "USD.SOFR.OIS"
FWD = "USD.SOFR.3M"


def _curve(rate: float) -> DiscountCurve:
    return DiscountCurve(
        pillar_dates=PILLARS,
        discount_factors=jnp.exp(-rate * TIMES),
        reference_date=REF,
    )


@pytest.fixture
def dual_state():
    """Two-curve market: OIS at 5%, 3M projection 50 bp above."""
    return MarketState(
        curves=CurveGraph(curves={OIS: _curve(0.05), FWD: _curve(0.055)}),
    )


@pytest.fixture
def equity_state():
    """One curve, one asset, one surface — exercises every block kind."""
    return MarketState(
        curves=CurveGraph(curves={OIS: _curve(0.05)}),
        surfaces={"SPX.VOL": ConstantVol(vol=jnp.asarray(0.2))},
        spots={"SPX": jnp.asarray(100.0)},
        dividends={"SPX": jnp.asarray(0.01)},
        asset_ids=("SPX",),
    )


@pytest.fixture
def dual_swaps():
    """Two-swap group discounting on OIS, projecting on 3M."""
    return InterestRateSwap(
        start_date=jnp.array([REF, REF]),
        fixed_dates=jnp.array([
            [ymd_to_ordinal(2027, 1, 1), ymd_to_ordinal(2028, 1, 1)],
            [ymd_to_ordinal(2027, 1, 1), ymd_to_ordinal(2028, 1, 1)],
        ]),
        fixed_rate=jnp.array([0.04, 0.05]),
        notional=jnp.array([1e6, 1e6]),
        discount_id=OIS,
        forward_id=FWD,
    )


# ── Layout construction ──────────────────────────────────────────────


class TestLayoutFromMarketState:
    def test_every_curve_gets_its_own_block(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        assert layout.curve_ids() == (FWD, OIS)      # sorted
        assert layout.n_factors == 2 * N_PILLARS
        assert layout.block("curve", OIS).size == N_PILLARS

    def test_blocks_cover_every_kind(self, equity_state):
        layout = KeyedFactorLayout.from_market_state(equity_state)
        kinds = tuple(b.kind for b in layout.blocks)
        assert kinds == ("spot", "vol", "curve", "dividend")
        assert layout.n_factors == 1 + 1 + N_PILLARS + 1

    def test_pillar_times_are_carried(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        times = layout.block("curve", OIS).times
        assert len(times) == N_PILLARS
        assert times[-1] == pytest.approx(float(TIMES[-1]))

    def test_subsetting(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(
            dual_state, curve_ids=(FWD,),
        )
        assert layout.curve_ids() == (FWD,)
        with pytest.raises(KeyError):
            layout.index("curve", OIS)

    def test_factor_ids_are_hierarchical(self, equity_state):
        layout = KeyedFactorLayout.from_market_state(equity_state)
        ids = layout.factor_ids()
        assert len(ids) == layout.n_factors
        assert ids[0] == "EQ.SPOT.SPX"
        assert ids[1] == "EQ.VOL.SPX.VOL"
        assert ids[2].startswith(f"IR.CURVE.{OIS}.")
        assert ids[-1] == "EQ.DIV.SPX"

    def test_duplicate_block_rejected(self):
        with pytest.raises(ValueError, match="two blocks"):
            KeyedFactorLayout(blocks=(
                FactorBlock(kind="spot", key="SPX"),
                FactorBlock(kind="spot", key="SPX"),
            ))

    def test_block_validation(self):
        with pytest.raises(ValueError, match="kind must be"):
            FactorBlock(kind="rates", key="X")
        with pytest.raises(ValueError, match="size must be"):
            FactorBlock(kind="curve", key="X", size=0)
        with pytest.raises(ValueError, match="pillar times"):
            FactorBlock(kind="curve", key="X", size=3, times=(1.0, 2.0))
        with pytest.raises(ValueError, match="only meaningful for a curve"):
            FactorBlock(kind="spot", key="X", basis_of=OIS)


# ── flatten / unflatten ──────────────────────────────────────────────


class TestRoundTrip:
    def test_vector_round_trip(self, equity_state):
        layout = KeyedFactorLayout.from_market_state(equity_state)
        x = jnp.arange(layout.n_factors, dtype=float) * 0.001
        assert jnp.allclose(layout.flatten(layout.unflatten(x)), x, atol=0.0)

    def test_scenario_round_trip(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        scenario = KeyedScenario(curve_shocks={
            FWD: jnp.full(N_PILLARS, 0.0010),
            OIS: jnp.linspace(0.0, 0.0020, N_PILLARS),
        })
        x = layout.flatten(scenario)
        back = layout.unflatten(x)
        assert jnp.allclose(back.curve_shocks[FWD], scenario.curve_shocks[FWD])
        assert jnp.allclose(back.curve_shocks[OIS], scenario.curve_shocks[OIS])

    def test_absent_keys_are_zeros(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        x = layout.flatten(KeyedScenario(
            curve_shocks={FWD: jnp.full(N_PILLARS, 0.001)},
        ))
        assert jnp.allclose(x[layout.index("curve", OIS)], 0.0)
        assert jnp.allclose(x[layout.index("curve", FWD)], 0.001)

    def test_unknown_key_is_an_error_not_a_silent_drop(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        with pytest.raises(ValueError, match="no block in this layout"):
            layout.flatten(KeyedScenario(
                curve_shocks={"EUR.ESTR": jnp.zeros(N_PILLARS)},
            ))

    def test_wrong_width_rejected(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        with pytest.raises(ValueError, match="trailing axis"):
            layout.unflatten(jnp.zeros((3,)))

    def test_batched_round_trip(self, equity_state):
        layout = KeyedFactorLayout.from_market_state(equity_state)
        X = jnp.arange(3 * layout.n_factors, dtype=float).reshape(
            3, layout.n_factors,
        )
        batched = layout.unflatten(X)
        assert batched.spot_shocks["SPX"].shape == (3,)
        assert batched.curve_shocks[OIS].shape == (3, N_PILLARS)
        assert jnp.allclose(layout.flatten(batched), X)

    def test_zero_is_structurally_complete(self, equity_state):
        layout = KeyedFactorLayout.from_market_state(equity_state)
        zero = layout.zero()
        assert set(zero.curve_shocks) == {OIS}
        assert set(zero.spot_shocks) == {"SPX"}
        assert jnp.allclose(layout.flatten(zero), 0.0)

    def test_shock_of_zero_is_a_no_op(self, equity_state):
        layout = KeyedFactorLayout.from_market_state(equity_state)
        shocked = shock(equity_state, layout.zero())
        assert jnp.allclose(
            shocked.curves[OIS].discount_factors,
            equity_state.curves[OIS].discount_factors,
            atol=1e-15,
        )
        assert float(shocked.spots["SPX"]) == pytest.approx(100.0)


# ── The basis as a coordinate ────────────────────────────────────────


class TestBasisLayout:
    def test_factor_prefix_follows_the_curve_alphabet(self):
        assert basis_factor_prefix("USD.SOFR.3M", "USD.OIS") == (
            "IR.BASIS.USD.SOFR3M.OIS"
        )
        # Different currencies: not one currency's factor, so both ids stay.
        assert basis_factor_prefix("EUR.ESTR", "USD.OIS") == (
            "IR.BASIS.EUR.ESTR.USD.OIS"
        )

    def test_basis_block_is_renamed(self, dual_state):
        layout = as_basis_layout(
            KeyedFactorLayout.from_market_state(dual_state), FWD, OIS,
        )
        ids = layout.factor_ids()[layout.index("curve", FWD)]
        assert all(i.startswith("IR.BASIS.USD.SOFR3M.") for i in ids)

    def test_basis_coordinates_are_spreads(self, dual_state):
        layout = as_basis_layout(
            KeyedFactorLayout.from_market_state(dual_state), FWD, OIS,
        )
        # Discount +10bp, projection +10bp: parallel move, zero basis.
        parallel = KeyedScenario(curve_shocks={
            OIS: jnp.full(N_PILLARS, 0.0010),
            FWD: jnp.full(N_PILLARS, 0.0010),
        })
        x = layout.flatten(parallel)
        assert jnp.allclose(x[layout.index("curve", FWD)], 0.0)
        assert jnp.allclose(x[layout.index("curve", OIS)], 0.0010)

    def test_unflatten_re_adds_the_base_curve(self, dual_state):
        layout = as_basis_layout(
            KeyedFactorLayout.from_market_state(dual_state), FWD, OIS,
        )
        x = jnp.zeros((layout.n_factors,))
        x = x.at[layout.index("curve", OIS)].set(0.0010)   # discount move
        x = x.at[layout.index("curve", FWD)].set(0.0002)   # basis widening
        scenario = layout.unflatten(x)
        assert jnp.allclose(scenario.curve_shocks[OIS], 0.0010)
        assert jnp.allclose(scenario.curve_shocks[FWD], 0.0012)

    def test_round_trip_in_basis_coordinates(self, dual_state):
        layout = as_basis_layout(
            KeyedFactorLayout.from_market_state(dual_state), FWD, OIS,
        )
        x = jnp.linspace(-0.001, 0.002, layout.n_factors)
        assert jnp.allclose(layout.flatten(layout.unflatten(x)), x, atol=1e-18)

    def test_rejects_degenerate_or_unknown_pairs(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        with pytest.raises(ValueError, match="basis over itself"):
            as_basis_layout(layout, FWD, FWD)
        with pytest.raises(KeyError):
            as_basis_layout(layout, FWD, "EUR.ESTR")
        once = as_basis_layout(layout, FWD, OIS)
        with pytest.raises(ValueError, match="already a basis"):
            as_basis_layout(once, FWD, OIS)


# ── Per-curve covariance ─────────────────────────────────────────────


class TestBlockCovariance:
    def test_each_curve_owns_its_block(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        cov = block_covariance(layout, {
            ("curve", OIS): jnp.full(N_PILLARS, 1e-6),
            ("curve", FWD): jnp.full(N_PILLARS, 4e-8),
        })
        assert cov.shape == (layout.n_factors, layout.n_factors)
        ois = layout.index("curve", OIS)
        fwd = layout.index("curve", FWD)
        assert jnp.allclose(jnp.diag(cov)[ois], 1e-6)
        assert jnp.allclose(jnp.diag(cov)[fwd], 4e-8)
        # Unstated pair ⇒ uncorrelated.
        assert jnp.allclose(cov[ois, fwd], 0.0)

    def test_cross_correlation_is_symmetric(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        cov = block_covariance(
            layout,
            {
                ("curve", OIS): jnp.full(N_PILLARS, 1e-6),
                ("curve", FWD): jnp.full(N_PILLARS, 1e-6),
            },
            {(("curve", OIS), ("curve", FWD)): -0.5},
        )
        assert jnp.allclose(cov, cov.T)
        off = cov[layout.index("curve", OIS), layout.index("curve", FWD)]
        assert jnp.allclose(off, -0.5e-6)

    def test_scalar_vector_and_matrix_blocks(self, equity_state):
        layout = KeyedFactorLayout.from_market_state(equity_state)
        full = jnp.eye(N_PILLARS) * 1e-6
        cov = block_covariance(layout, {
            ("spot", "SPX"): 0.04,
            ("vol", "SPX.VOL"): 1e-4,
            ("curve", OIS): full,
            ("dividend", "SPX"): jnp.array([1e-8]),
        })
        assert float(cov[0, 0]) == pytest.approx(0.04)
        assert float(cov[1, 1]) == pytest.approx(1e-4)
        assert jnp.allclose(cov[layout.index("curve", OIS), layout.index("curve", OIS)], full)

    def test_unknown_block_rejected(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        with pytest.raises(KeyError):
            block_covariance(layout, {("curve", "EUR.ESTR"): 1e-6})

    def test_bad_block_shape_rejected(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        with pytest.raises(ValueError, match="expects"):
            block_covariance(layout, {("curve", OIS): jnp.zeros((2, 2))})

    def test_historical_block_covariance(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        key = jax.random.PRNGKey(0)
        returns = 1e-3 * jax.random.normal(key, (500, layout.n_factors))
        cov = historical_block_covariance(returns, layout)
        assert cov.shape == (layout.n_factors,) * 2
        assert jnp.allclose(cov, cov.T, atol=1e-18)
        assert float(jnp.min(jnp.linalg.eigvalsh(cov))) > -1e-18

    def test_historical_width_checked(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        with pytest.raises(ValueError, match="columns"):
            historical_block_covariance(jnp.zeros((10, 3)), layout)


# ── Generators ───────────────────────────────────────────────────────


class TestKeyedGenerators:
    def test_parametric_is_batched_per_block(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        cov = block_covariance(layout, {
            ("curve", OIS): jnp.full(N_PILLARS, 1e-6),
            ("curve", FWD): jnp.full(N_PILLARS, 1e-6),
        })
        scenarios = keyed_parametric_scenarios(
            jax.random.PRNGKey(0), cov, layout, 64,
        )
        assert scenarios.curve_shocks[OIS].shape == (64, N_PILLARS)
        assert scenarios.curve_shocks[FWD].shape == (64, N_PILLARS)
        # Independent blocks ⇒ low realised cross-correlation.
        a = scenarios.curve_shocks[OIS][:, 2]
        b = scenarios.curve_shocks[FWD][:, 2]
        corr = jnp.corrcoef(jnp.stack([a, b]))[0, 1]
        assert abs(float(corr)) < 0.4

    def test_parametric_t_has_fatter_tails(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(
            dual_state, curve_ids=(OIS,),
        )
        cov = block_covariance(layout, {("curve", OIS): jnp.full(N_PILLARS, 1e-6)})
        normal = keyed_parametric_scenarios(
            jax.random.PRNGKey(1), cov, layout, 2000,
        ).curve_shocks[OIS]
        student = keyed_parametric_scenarios(
            jax.random.PRNGKey(1), cov, layout, 2000, distribution="t", df=3.0,
        ).curve_shocks[OIS]
        assert float(jnp.max(jnp.abs(student))) > float(jnp.max(jnp.abs(normal)))

    def test_parametric_checks_cov_shape(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        with pytest.raises(ValueError, match="cov has shape"):
            keyed_parametric_scenarios(
                jax.random.PRNGKey(0), jnp.eye(3), layout, 4,
            )

    def test_historical(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        returns = jnp.zeros((7, layout.n_factors)).at[:, layout.index("curve", FWD)].set(0.001)
        scenarios = keyed_historical_scenarios(returns, layout)
        assert scenarios.curve_shocks[FWD].shape == (7, N_PILLARS)
        assert jnp.allclose(scenarios.curve_shocks[FWD], 0.001)
        assert jnp.allclose(scenarios.curve_shocks[OIS], 0.0)

    def test_stress_sizes_curves_from_the_layout(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        scenario = keyed_stress_scenario(
            layout, parallel_shift=0.0010, curve_shifts={FWD: 0.0025},
        )
        assert jnp.allclose(scenario.curve_shocks[OIS], 0.0010)
        assert jnp.allclose(scenario.curve_shocks[FWD], 0.0025)

    def test_stress_full_control(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        bumps = jnp.linspace(0.0, 0.003, N_PILLARS)
        scenario = keyed_stress_scenario(layout, curve_shocks={OIS: bumps})
        assert jnp.allclose(scenario.curve_shocks[OIS], bumps)

    def test_stress_rejects_unknown_curve(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        with pytest.raises(KeyError):
            keyed_stress_scenario(layout, curve_shifts={"EUR.ESTR": 0.001})

    def test_stress_wrong_pillar_count(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        with pytest.raises(ValueError, match="expected"):
            keyed_stress_scenario(layout, curve_shocks={OIS: jnp.zeros(2)})

    def test_equity_blocks_are_shocked(self, equity_state):
        layout = KeyedFactorLayout.from_market_state(equity_state)
        scenario = keyed_stress_scenario(
            layout, spot_shock=-5.0, vol_shock=0.03, dividend_shock=0.001,
        )
        shocked = shock(equity_state, scenario)
        assert float(shocked.spots["SPX"]) == pytest.approx(95.0)
        assert float(shocked.surfaces["SPX.VOL"].vol) == pytest.approx(0.23)
        assert float(shocked.dividends["SPX"]) == pytest.approx(0.011)


class TestCurveShapes:
    def test_shared_algebra_matches_the_legacy_presets(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        legacy = steepener(0, N_PILLARS, 0.0010, 0.0030).rate_shocks
        keyed = keyed_steepener(layout, OIS, 0.0010, 0.0030)
        assert jnp.allclose(keyed.curve_shocks[OIS], legacy)

        legacy_bf = butterfly(0, N_PILLARS, 0.0005, 0.0025).rate_shocks
        keyed_bf = keyed_butterfly(layout, OIS, 0.0005, 0.0025)
        assert jnp.allclose(keyed_bf.curve_shocks[OIS], legacy_bf)

    def test_shape_only_touches_the_named_curve(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        scenario = keyed_flattener(layout, FWD, 0.0030, -0.0010)
        assert jnp.allclose(scenario.curve_shocks[OIS], 0.0)
        assert float(scenario.curve_shocks[FWD][0]) == pytest.approx(0.0030)
        assert float(scenario.curve_shocks[FWD][-1]) == pytest.approx(-0.0010)

    def test_all_shapes_are_supported(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        for shape in CURVE_SHAPES:
            scenario = keyed_curve_stress(layout, OIS, shape, 0.001, 0.002)
            assert scenario.curve_shocks[OIS].shape == (N_PILLARS,)

    def test_unknown_shape_rejected(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        with pytest.raises(ValueError, match="shape must be"):
            keyed_curve_stress(layout, OIS, "twist", 0.001)
        with pytest.raises(ValueError, match="shape must be"):
            curve_shape_bumps(4, "twist")


# ── The point: a projection curve carries its own P&L ────────────────


class TestBasisRiskReachesPnL:
    def test_basis_shift_moves_the_book(self, dual_swaps, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        scenario = keyed_basis_shift(layout, FWD, 0.0025, OIS)
        base = group_pv(dual_swaps, None, dual_state)
        shocked = group_pv(dual_swaps, None, shock(dual_state, scenario))
        # Same number the hand-written keyed scenario produces in
        # tests/test_risk/test_state_var.py::TestMultiCurveBasisRisk.
        assert float(shocked - base) == pytest.approx(9817.3, rel=1e-3)

    def test_basis_shift_in_basis_coordinates_agrees(
        self, dual_swaps, dual_state,
    ):
        absolute = KeyedFactorLayout.from_market_state(dual_state)
        basis = as_basis_layout(absolute, FWD, OIS)
        a = keyed_basis_shift(absolute, FWD, 0.0025, OIS)
        b = keyed_basis_shift(basis, FWD, 0.0025, OIS)
        for curve_id in (OIS, FWD):
            assert jnp.allclose(a.curve_shocks[curve_id], b.curve_shocks[curve_id])

    def test_basis_shift_needs_two_curves(self, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        with pytest.raises(ValueError, match="two distinct curves"):
            keyed_basis_shift(layout, FWD, 0.001, FWD)
        with pytest.raises(KeyError):
            keyed_basis_shift(layout, FWD, 0.001, "EUR.ESTR")

    def test_projection_curve_carries_its_own_var_path(
        self, dual_swaps, dual_state,
    ):
        """A scenario *set* drawn on the basis block alone makes P&L.

        The hole the roadmap flags under § 8: a VaR path where a
        projection curve's own variance produces the loss distribution.
        """
        layout = as_basis_layout(
            KeyedFactorLayout.from_market_state(dual_state), FWD, OIS,
        )
        cov = block_covariance(layout, {
            ("curve", FWD): jnp.full(N_PILLARS, (2e-4) ** 2),  # basis only
        })
        scenarios = keyed_parametric_scenarios(
            jax.random.PRNGKey(0), cov, layout, 128,
        )
        pnl = group_scenario_pnl(dual_swaps, None, dual_state, scenarios)
        assert pnl.shape == (128,)
        assert float(jnp.std(pnl)) > 0.0
        # Nothing else moved, so the loss distribution is pure basis risk.
        assert jnp.allclose(scenarios.curve_shocks[OIS], 0.0)

    def test_discount_only_covariance_leaves_the_basis_flat(self, dual_state):
        layout = as_basis_layout(
            KeyedFactorLayout.from_market_state(dual_state), FWD, OIS,
        )
        cov = block_covariance(layout, {
            ("curve", OIS): jnp.full(N_PILLARS, (7e-4) ** 2),
        })
        scenarios = keyed_parametric_scenarios(
            jax.random.PRNGKey(2), cov, layout, 32,
        )
        # A pure discount move drags the projection curve with it: that is
        # what "the basis did not move" means, and it is inexpressible in
        # absolute coordinates without a unit correlation.
        assert jnp.allclose(
            scenarios.curve_shocks[OIS], scenarios.curve_shocks[FWD],
        )
