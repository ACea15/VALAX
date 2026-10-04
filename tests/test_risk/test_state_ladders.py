"""Tests for keyed sensitivity ladders on a ``MarketState``.

Mirrors ``valax/risk/state_ladders.py``.  Three things are gated:

1. **Correctness of the sensitivities** — against finite differences of the
   *same* shock-and-reprice map, and against the legacy single-curve
   numbers where the two coincide.
2. **Per-curve separation** — a dual-curve book's OIS and projection
   ladders are distinct, additive, and the projection one is zero for a
   trade that does not use it.  This is the thing the flat
   ``SensitivityLadder`` cannot express.
3. **Attribution closes** — predicted + unexplained == actual, per-block
   rungs sum to the total, and component VaR sums to VaR.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.core.market_state import MarketState
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.options import EuropeanOption
from valax.instruments.rates import InterestRateSwap
from valax.models.black_scholes import BlackScholesModel
from valax.risk.scenarios import (
    KeyedFactorLayout,
    as_basis_layout,
    block_covariance,
    keyed_basis_shift,
    keyed_parametric_scenarios,
    keyed_steepener,
    keyed_stress_scenario,
)
from valax.risk.state_ladders import (
    BASIS_POINT,
    keyed_component_var,
    keyed_ladder,
    keyed_parametric_var,
    keyed_waterfall_pnl,
    keyed_waterfall_report,
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
    return MarketState(
        curves=CurveGraph(curves={OIS: _curve(0.05), FWD: _curve(0.055)}),
    )


def _swaps(forward_id):
    return InterestRateSwap(
        start_date=jnp.array([REF, REF]),
        fixed_dates=jnp.array([
            [ymd_to_ordinal(2027, 1, 1), ymd_to_ordinal(2028, 1, 1)],
            [ymd_to_ordinal(2027, 1, 1), ymd_to_ordinal(2028, 1, 1)],
        ]),
        fixed_rate=jnp.array([0.04, 0.05]),
        notional=jnp.array([1e6, 1e6]),
        discount_id=OIS,
        forward_id=forward_id,
    )


@pytest.fixture
def dual_swaps():
    """Discounts on OIS, projects on 3M — genuinely dual-curve."""
    return _swaps(FWD)


@pytest.fixture
def single_curve_swaps():
    """Same trade, but projecting off its own discount curve."""
    return _swaps(None)


@pytest.fixture
def equity_state():
    """Single-curve equity market.

    The surface is filed under the sentinel because that is what
    ``valax.core.resolve.resolve_surface`` falls back to for a model with
    no ``surface_id`` (``BlackScholesModel`` has none); the spot and
    dividend are keyed by the option's ``underlying_id``.
    """
    return MarketState(
        curves=CurveGraph(curves={"_default_": _curve(0.05)}),
        surfaces={"_default_": ConstantVol(vol=jnp.asarray(0.2))},
        spots={"SPX": jnp.asarray(100.0)},
        dividends={"SPX": jnp.asarray(0.01)},
        asset_ids=("SPX",),
    )


@pytest.fixture
def options():
    return EuropeanOption(
        strike=jnp.array([100.0, 110.0]),
        expiry=jnp.array([1.0, 1.0]),
        is_call=True,
        underlying_id="SPX",
    )


@pytest.fixture
def market_sourced_bs():
    """A Black-Scholes model that reads spot / vol / rate off the market.

    Every parameter left ``None`` is bound from the market by
    ``valax.core.binding``, so a keyed vol or curve shock actually reaches
    the price instead of being shadowed by a model scalar.
    """
    return BlackScholesModel()


def _fd_block(instruments, model, market, layout, kind, key, index, h):
    """Central finite difference of the shock-and-reprice map."""
    def pv_at(bump):
        x = jnp.zeros((layout.n_factors,)).at[
            layout.index(kind, key).start + index
        ].set(bump)
        return group_pv(instruments, model, shock(market, layout.unflatten(x)))

    return (pv_at(h) - pv_at(-h)) / (2.0 * h)


# ── First-order sensitivities ────────────────────────────────────────


class TestLadderAgainstFiniteDifferences:
    def test_curve_pillar_sensitivities(self, dual_swaps, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        ladder = keyed_ladder(dual_swaps, None, dual_state, layout)
        for curve_id in (OIS, FWD):
            for i in range(N_PILLARS):
                got = float(ladder.block("curve", curve_id)[i])
                want = float(_fd_block(
                    dual_swaps, None, dual_state, layout,
                    "curve", curve_id, i, 1e-6,
                ))
                assert got == pytest.approx(want, rel=1e-4, abs=1e-6)

    def test_pv_is_the_base_valuation(self, dual_swaps, dual_state):
        ladder = keyed_ladder(dual_swaps, None, dual_state)
        assert float(ladder.pv) == pytest.approx(
            float(group_pv(dual_swaps, None, dual_state)), rel=1e-12,
        )

    def test_vega_and_spot_delta(self, options, market_sourced_bs, equity_state):
        layout = KeyedFactorLayout.from_market_state(equity_state)
        ladder = keyed_ladder(options, market_sourced_bs, equity_state, layout)

        vega = float(ladder.block("vol", "_default_")[0])
        fd_vega = float(_fd_block(
            options, market_sourced_bs, equity_state, layout,
            "vol", "_default_", 0, 1e-5,
        ))
        assert vega == pytest.approx(fd_vega, rel=1e-4)
        assert vega > 0.0            # long two calls

        delta = float(ladder.spot_delta("SPX"))
        fd_delta = float(_fd_block(
            options, market_sourced_bs, equity_state, layout,
            "spot", "SPX", 0, 1e-4,
        ))
        assert delta == pytest.approx(fd_delta, rel=1e-4)
        assert 0.0 < delta < 2.0     # two unit calls

    def test_dividend_sensitivity_is_negative_for_calls(
        self, options, market_sourced_bs, equity_state,
    ):
        ladder = keyed_ladder(options, market_sourced_bs, equity_state)
        assert float(ladder.dividend_delta("SPX")) < 0.0


class TestSecondOrder:
    def test_spot_gamma_is_positive(
        self, options, market_sourced_bs, equity_state,
    ):
        ladder = keyed_ladder(
            options, market_sourced_bs, equity_state, second_order=True,
        )
        assert float(ladder.gamma_block("spot", "SPX")[0, 0]) > 0.0

    def test_gamma_is_symmetric(self, dual_swaps, dual_state):
        ladder = keyed_ladder(
            dual_swaps, None, dual_state, second_order=True,
        )
        assert jnp.allclose(ladder.gamma, ladder.gamma.T, atol=1e-8)

    def test_first_order_ladder_has_no_hessian(self, dual_swaps, dual_state):
        ladder = keyed_ladder(dual_swaps, None, dual_state)
        assert ladder.gamma is None
        with pytest.raises(ValueError, match="first-order only"):
            ladder.gamma_block("curve", OIS)


# ── Per-curve separation: the point ──────────────────────────────────


class TestPerCurveLadders:
    def test_each_curve_has_its_own_dv01(self, dual_swaps, dual_state):
        ladder = keyed_ladder(dual_swaps, None, dual_state)
        ois = float(ladder.dv01(OIS))
        fwd = float(ladder.dv01(FWD))
        # Receiving the projected float leg: a projection move dominates,
        # and the two have opposite signs — information the flat layout
        # destroys by adding them into one rate block.
        assert fwd > 0.0
        assert ois < 0.0
        assert abs(fwd) > abs(ois)

    def test_per_curve_dv01s_add_to_the_joint_move(
        self, dual_swaps, dual_state,
    ):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        ladder = keyed_ladder(dual_swaps, None, dual_state, layout)
        joint = keyed_stress_scenario(layout, parallel_shift=BASIS_POINT)
        waterfall = keyed_waterfall_pnl(ladder, joint)
        assert float(waterfall.total_first_order) == pytest.approx(
            float(ladder.dv01(OIS) + ladder.dv01(FWD)), rel=1e-10,
        )

    def test_key_rate_ladder_sums_to_the_parallel_dv01(
        self, dual_swaps, dual_state,
    ):
        ladder = keyed_ladder(dual_swaps, None, dual_state)
        rungs = ladder.curve_ladder(FWD)
        assert rungs.shape == (N_PILLARS,)
        assert float(jnp.sum(rungs)) == pytest.approx(
            float(ladder.dv01(FWD)), rel=1e-12,
        )

    def test_unused_projection_curve_carries_no_risk(
        self, single_curve_swaps, dual_state,
    ):
        """A trade that never reads the 3M curve has zero 3M ladder.

        The honest control for the test above: per-curve attribution must
        not manufacture risk where the pricer has no dependency.
        """
        ladder = keyed_ladder(single_curve_swaps, None, dual_state)
        assert jnp.allclose(ladder.block("curve", FWD), 0.0, atol=1e-12)
        assert float(abs(ladder.dv01(OIS))) > 1.0

    def test_unknown_curve_raises(self, dual_swaps, dual_state):
        ladder = keyed_ladder(dual_swaps, None, dual_state)
        with pytest.raises(KeyError):
            ladder.dv01("EUR.ESTR")


# ── The IR.BASIS factor ──────────────────────────────────────────────


class TestBasisDV01:
    def test_basis_dv01_is_the_projection_dv01(self, dual_swaps, dual_state):
        ladder = keyed_ladder(dual_swaps, None, dual_state)
        assert float(ladder.basis_dv01(FWD, OIS)) == pytest.approx(
            float(ladder.dv01(FWD)), rel=1e-12,
        )

    def test_basis_dv01_is_zero_for_a_single_curve_trade(
        self, single_curve_swaps, dual_state,
    ):
        ladder = keyed_ladder(single_curve_swaps, None, dual_state)
        assert float(ladder.basis_dv01(FWD, OIS)) == pytest.approx(0.0, abs=1e-10)

    def test_basis_layout_names_the_factor(self, dual_swaps, dual_state):
        layout = as_basis_layout(
            KeyedFactorLayout.from_market_state(dual_state), FWD, OIS,
        )
        ladder = keyed_ladder(dual_swaps, None, dual_state, layout)
        prefixes = set(ladder.block_totals())
        assert "IR.BASIS.USD.SOFR3M.SOFROIS" in prefixes
        assert f"IR.CURVE.{OIS}" in prefixes

    def test_basis_coordinates_reallocate_the_rate_risk(
        self, dual_swaps, dual_state,
    ):
        """In basis coordinates the discount rung is the *total* rate risk.

        Moving OIS with the basis held fixed drags the projection curve
        with it, so ``dv01(OIS)`` in a basis layout equals the sum of the
        two absolute DV01s — the economically meaningful decomposition of
        "rates moved" vs "the spread moved".
        """
        absolute = KeyedFactorLayout.from_market_state(dual_state)
        basis = as_basis_layout(absolute, FWD, OIS)
        abs_ladder = keyed_ladder(dual_swaps, None, dual_state, absolute)
        basis_ladder = keyed_ladder(dual_swaps, None, dual_state, basis)

        assert float(basis_ladder.dv01(OIS)) == pytest.approx(
            float(abs_ladder.dv01(OIS) + abs_ladder.dv01(FWD)), rel=1e-8,
        )
        assert float(basis_ladder.basis_dv01(FWD, OIS)) == pytest.approx(
            float(abs_ladder.dv01(FWD)), rel=1e-8,
        )

    def test_basis_needs_two_distinct_known_curves(
        self, dual_swaps, dual_state,
    ):
        ladder = keyed_ladder(dual_swaps, None, dual_state)
        with pytest.raises(ValueError, match="two distinct curves"):
            ladder.basis_dv01(FWD, FWD)
        with pytest.raises(KeyError):
            ladder.basis_dv01(FWD, "EUR.ESTR")


# ── Waterfall attribution ────────────────────────────────────────────


class TestWaterfall:
    def test_second_order_explains_a_small_basis_move(
        self, dual_swaps, dual_state,
    ):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        scenario = keyed_basis_shift(layout, FWD, 0.0025, OIS)

        first = keyed_waterfall_report(
            dual_swaps, None, dual_state, scenario,
            layout=layout, second_order=False,
        )
        second = keyed_waterfall_report(
            dual_swaps, None, dual_state, scenario,
            layout=layout, second_order=True,
        )
        assert float(first.actual) == pytest.approx(9817.3, rel=1e-3)
        # Adding the Hessian must strictly improve the explain.
        assert abs(float(second.unexplained)) < abs(float(first.unexplained))
        assert abs(float(second.unexplained)) / abs(float(second.actual)) < 1e-5

    def test_rungs_sum_to_predicted(self, dual_swaps, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        ladder = keyed_ladder(
            dual_swaps, None, dual_state, layout, second_order=True,
        )
        scenario = keyed_steepener(layout, FWD, 0.0005, 0.0030)
        wf = keyed_waterfall_pnl(ladder, scenario)

        rungs = sum(float(v) for v in wf.delta_pnl.values())
        assert rungs == pytest.approx(float(wf.total_first_order), rel=1e-10)
        gammas = sum(float(v) for v in wf.gamma_pnl.values())
        assert gammas + float(wf.cross_pnl) == pytest.approx(
            float(wf.total_second_order), rel=1e-8, abs=1e-9,
        )
        assert float(wf.predicted) == pytest.approx(
            float(wf.total_first_order + wf.total_second_order), rel=1e-12,
        )

    def test_one_rung_per_named_curve(self, dual_swaps, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        ladder = keyed_ladder(dual_swaps, None, dual_state, layout)
        scenario = keyed_basis_shift(layout, FWD, 0.0010, OIS)
        wf = keyed_waterfall_pnl(ladder, scenario)

        assert set(wf.delta_pnl) == {f"IR.CURVE.{OIS}", f"IR.CURVE.{FWD}"}
        # A pure basis move leaves the discount rung at exactly zero.
        assert float(wf.delta_pnl[f"IR.CURVE.{OIS}"]) == pytest.approx(0.0)
        assert float(wf.delta_pnl[f"IR.CURVE.{FWD}"]) != 0.0

    def test_first_order_only_waterfall(self, dual_swaps, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        ladder = keyed_ladder(dual_swaps, None, dual_state, layout)
        wf = keyed_waterfall_pnl(
            ladder, keyed_stress_scenario(layout, parallel_shift=1e-4),
        )
        assert wf.gamma_pnl == {}
        assert float(wf.total_second_order) == 0.0
        assert jnp.isnan(wf.actual)
        assert jnp.isnan(wf.unexplained)

    def test_scenario_outside_the_layout_raises(self, dual_swaps, dual_state):
        layout = KeyedFactorLayout.from_market_state(
            dual_state, curve_ids=(OIS,),
        )
        ladder = keyed_ladder(dual_swaps, None, dual_state, layout)
        full = KeyedFactorLayout.from_market_state(dual_state)
        with pytest.raises(ValueError, match="no block in this layout"):
            keyed_waterfall_pnl(
                ladder, keyed_basis_shift(full, FWD, 0.001, OIS),
            )


# ── Delta-normal VaR in keyed coordinates ────────────────────────────


class TestKeyedVaR:
    @pytest.fixture
    def basis_setup(self, dual_swaps, dual_state):
        layout = as_basis_layout(
            KeyedFactorLayout.from_market_state(dual_state), FWD, OIS,
        )
        ladder = keyed_ladder(dual_swaps, None, dual_state, layout)
        cov = block_covariance(layout, {
            ("curve", OIS): jnp.full(N_PILLARS, (7e-4) ** 2),
            ("curve", FWD): jnp.full(N_PILLARS, (2e-4) ** 2),
        })
        return layout, ladder, cov

    def test_matches_the_closed_form(self, basis_setup):
        _, ladder, cov = basis_setup
        var = keyed_parametric_var(ladder, cov, confidence=0.99)
        z = -float(jax.scipy.stats.norm.ppf(0.01))
        want = z * float(jnp.sqrt(ladder.delta @ cov @ ladder.delta))
        assert float(var) == pytest.approx(want, rel=1e-12)
        assert float(var) > 0.0

    def test_horizon_scales_as_root_time(self, basis_setup):
        _, ladder, cov = basis_setup
        one = float(keyed_parametric_var(ladder, cov, horizon=1.0))
        ten = float(keyed_parametric_var(ladder, cov, horizon=10.0))
        assert ten == pytest.approx(one * jnp.sqrt(10.0), rel=1e-10)

    def test_components_sum_to_total(self, basis_setup):
        _, ladder, cov = basis_setup
        var = float(keyed_parametric_var(ladder, cov))
        components = keyed_component_var(ladder, cov)
        assert sum(float(v) for v in components.values()) == pytest.approx(
            var, rel=1e-10,
        )

    def test_the_basis_carries_its_own_var(self, basis_setup):
        """The number the flat layout cannot produce."""
        _, ladder, cov = basis_setup
        components = keyed_component_var(ladder, cov)
        basis = components["IR.BASIS.USD.SOFR3M.SOFROIS"]
        assert float(basis) != 0.0
        assert abs(float(basis)) > 1.0

    def test_zero_basis_variance_removes_the_basis_contribution(
        self, dual_swaps, dual_state,
    ):
        layout = as_basis_layout(
            KeyedFactorLayout.from_market_state(dual_state), FWD, OIS,
        )
        ladder = keyed_ladder(dual_swaps, None, dual_state, layout)
        cov = block_covariance(layout, {
            ("curve", OIS): jnp.full(N_PILLARS, (7e-4) ** 2),
        })
        components = keyed_component_var(ladder, cov)
        assert float(components["IR.BASIS.USD.SOFR3M.SOFROIS"]) == pytest.approx(
            0.0, abs=1e-8,
        )

    def test_cov_shape_is_checked(self, basis_setup):
        _, ladder, _ = basis_setup
        with pytest.raises(ValueError, match="covariance has shape"):
            keyed_parametric_var(ladder, jnp.eye(3))

    def test_agrees_with_full_revaluation_var_on_a_linear_book(
        self, dual_swaps, dual_state, basis_setup,
    ):
        """Swaps are near-linear, so delta-normal ≈ simulated VaR."""
        from valax.risk.state_var import group_scenario_pnl
        from valax.risk.var import value_at_risk

        layout, ladder, cov = basis_setup
        scenarios = keyed_parametric_scenarios(
            jax.random.PRNGKey(0), cov, layout, 20_000,
        )
        pnl = group_scenario_pnl(dual_swaps, None, dual_state, scenarios)
        simulated = float(value_at_risk(pnl, confidence=0.99))
        analytic = float(keyed_parametric_var(ladder, cov, confidence=0.99))
        assert simulated == pytest.approx(analytic, rel=0.05)


# ── Reporting, JIT and warnings ──────────────────────────────────────


class TestLadderContract:
    def test_to_dict_is_keyed_by_factor_id(self, dual_swaps, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)
        ladder = keyed_ladder(dual_swaps, None, dual_state, layout)
        table = ladder.to_dict()
        assert tuple(table) == layout.factor_ids()
        assert len(table) == layout.n_factors

    def test_block_totals_one_number_per_market_object(
        self, options, market_sourced_bs, equity_state,
    ):
        ladder = keyed_ladder(options, market_sourced_bs, equity_state)
        totals = ladder.block_totals()
        assert set(totals) == {
            "EQ.SPOT.SPX", "EQ.VOL._default_", "IR.CURVE._default_",
            "EQ.DIV.SPX",
        }

    def test_default_layout_covers_the_whole_market(
        self, dual_swaps, dual_state,
    ):
        ladder = keyed_ladder(dual_swaps, None, dual_state)
        assert ladder.layout.curve_ids() == (FWD, OIS)

    def test_filter_jit(self, dual_swaps, dual_state):
        layout = KeyedFactorLayout.from_market_state(dual_state)

        @eqx.filter_jit
        def run(market):
            return keyed_ladder(dual_swaps, None, market, layout).delta

        jitted = run(dual_state)
        eager = keyed_ladder(dual_swaps, None, dual_state, layout).delta
        assert jnp.allclose(jitted, eager, rtol=1e-12)

    def test_warns_when_the_model_shadows_the_market(
        self, options, equity_state,
    ):
        """A model carrying its own vol/rate makes those rungs structurally 0."""
        model = BlackScholesModel(
            vol=jnp.array(0.2), rate=jnp.array(0.05),
            dividend=jnp.array(0.0),
        )
        with pytest.warns(UserWarning, match="shadowed by BlackScholesModel"):
            ladder = keyed_ladder(options, model, equity_state)
        assert jnp.allclose(ladder.block("vol", "_default_"), 0.0)

    def test_no_warning_when_the_model_is_market_sourced(
        self, options, market_sourced_bs, equity_state,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            keyed_ladder(options, market_sourced_bs, equity_state)
