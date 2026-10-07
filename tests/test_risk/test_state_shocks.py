"""Tests for keyed shocks on a ``MarketState``."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.core.market_state import DEFAULT_ID, MarketState
from valax.curves.discount import DiscountCurve, zero_rate
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.market.data import MarketData
from valax.market.keyed_scenario import (
    KeyedScenario,
    keyed_from_market_scenario,
    stack_keyed_scenarios,
    zero_keyed_scenario,
)
from valax.market.scenario import MarketScenario
from valax.risk.shocks import apply_scenario
from valax.risk.state_shocks import shift_surface, shock
from valax.surfaces.constant import ConstantVol
from valax.surfaces.grid import GridVolSurface
from valax.surfaces.shifted import ShiftedVolSurface


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def flat_curve():
    """A flat 5% continuously-compounded curve with 3 pillars."""
    ref = ymd_to_ordinal(2026, 1, 1)
    pillars = jnp.array([
        ymd_to_ordinal(2026, 1, 1),
        ymd_to_ordinal(2027, 1, 1),
        ymd_to_ordinal(2028, 1, 1),
    ])
    times = (pillars - ref).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-0.05 * times),
        reference_date=ref,
    )


@pytest.fixture
def single(flat_curve):
    """Zero-config single-curve, single-asset market."""
    return MarketState.single_curve(
        flat_curve,
        spot=jnp.array(100.0),
        vol=jnp.array(0.2),
        dividend=jnp.array(0.01),
    )


@pytest.fixture
def dual(flat_curve):
    """Two-curve, two-asset market with explicit identifiers."""
    times = (
        (flat_curve.pillar_dates - flat_curve.reference_date).astype(jnp.float64)
        / 365.0
    )
    fwd = DiscountCurve(
        pillar_dates=flat_curve.pillar_dates,
        discount_factors=jnp.exp(-0.055 * times),  # +50bp basis
        reference_date=flat_curve.reference_date,
    )
    return MarketState(
        curves=CurveGraph(curves={
            "USD.SOFR.OIS": flat_curve,
            "USD.SOFR.3M": fwd,
        }),
        surfaces={
            "SPX.VOL": ConstantVol(vol=jnp.array(0.20)),
            "NDX.VOL": GridVolSurface(
                strikes=jnp.array([90.0, 100.0, 110.0]),
                expiries=jnp.array([0.5, 1.0]),
                vols=jnp.array([[0.30, 0.28, 0.27], [0.29, 0.27, 0.26]]),
            ),
        },
        spots={"SPX": jnp.array(5000.0), "NDX": jnp.array(18000.0)},
        dividends={"SPX": jnp.array(0.015), "NDX": jnp.array(0.008)},
        asset_ids=("SPX", "NDX"),
    )


# ── Zero-config sentinel behaviour ───────────────────────────────────


class TestSentinelZeroConfig:
    def test_empty_scenario_is_a_no_op(self, single):
        out = shock(single, zero_keyed_scenario())
        assert out.spot() == single.spot()
        assert out.discount_curve.discount_factors is (
            single.discount_curve.discount_factors
        )

    def test_sentinel_addresses_the_only_entry(self, flat_curve):
        """Named market objects still answer the sentinel when unique."""
        state = MarketState(
            curves=CurveGraph(curves={"USD.SOFR.OIS": flat_curve}),
            surfaces={"SPX.VOL": ConstantVol(vol=jnp.array(0.2))},
            spots={"SPX": jnp.array(100.0)},
            dividends={"SPX": jnp.array(0.01)},
            asset_ids=("SPX",),
        )
        out = shock(state, KeyedScenario(
            spot_shocks={DEFAULT_ID: jnp.array(5.0)},
            vol_shocks={DEFAULT_ID: jnp.array(0.01)},
            curve_shocks={DEFAULT_ID: jnp.full(3, 0.01)},
            dividend_shocks={DEFAULT_ID: jnp.array(0.002)},
        ))
        assert out.spots["SPX"] == 105.0
        assert jnp.isclose(out.surfaces["SPX.VOL"].vol, 0.21)
        assert jnp.isclose(out.dividends["SPX"], 0.012)
        assert jnp.isclose(
            zero_rate(out.curves["USD.SOFR.OIS"], flat_curve.pillar_dates[1]),
            0.06, atol=1e-4,
        )

    def test_asset_id_resolves_the_vol_surface_key(self, dual):
        """``"SPX"`` finds ``"SPX.VOL"`` — the from_market_data convention."""
        out = shock(dual, KeyedScenario(vol_shocks={"SPX": jnp.array(0.03)}))
        assert jnp.isclose(out.surfaces["SPX.VOL"].vol, 0.23)

    def test_sentinel_is_ambiguous_when_many_entries(self, dual):
        with pytest.raises(KeyError, match="not in the MarketState"):
            shock(dual, KeyedScenario(spot_shocks={DEFAULT_ID: jnp.array(1.0)}))


# ── No silent no-ops ─────────────────────────────────────────────────


class TestUnknownIdentifiersRaise:
    def test_unknown_curve(self, single):
        with pytest.raises(KeyError, match="shocks curve 'EUR.ESTR.OIS'"):
            shock(single, KeyedScenario(
                curve_shocks={"EUR.ESTR.OIS": jnp.zeros(3)},
            ))

    def test_unknown_surface(self, dual):
        with pytest.raises(KeyError, match="shocks surface"):
            shock(dual, KeyedScenario(vol_shocks={"RTY.VOL": jnp.array(0.01)}))

    def test_unknown_spot(self, dual):
        with pytest.raises(KeyError, match="shocks spot for asset"):
            shock(dual, KeyedScenario(spot_shocks={"RTY": jnp.array(1.0)}))

    def test_unknown_dividend(self, dual):
        with pytest.raises(KeyError, match="shocks dividend for asset"):
            shock(dual, KeyedScenario(dividend_shocks={"RTY": jnp.array(0.01)}))

    def test_legacy_scenario_is_refused(self, single):
        legacy = MarketScenario(
            spot_shocks=jnp.zeros(1),
            vol_shocks=jnp.zeros(1),
            rate_shocks=jnp.zeros(3),
            dividend_shocks=jnp.zeros(1),
        )
        with pytest.raises(TypeError, match="keyed_from_market_scenario"):
            shock(single, legacy)


# ── Purity ───────────────────────────────────────────────────────────


class TestPurity:
    def test_base_market_is_untouched(self, dual):
        before_spot = dual.spots["SPX"]
        before_vol = dual.surfaces["SPX.VOL"].vol
        before_dfs = dual.curves["USD.SOFR.OIS"].discount_factors
        out = shock(dual, KeyedScenario(
            spot_shocks={"SPX": jnp.array(100.0)},
            vol_shocks={"SPX.VOL": jnp.array(0.05)},
            curve_shocks={"USD.SOFR.OIS": jnp.full(3, 0.01)},
        ))
        assert dual.spots["SPX"] == before_spot
        assert dual.surfaces["SPX.VOL"].vol == before_vol
        assert jnp.array_equal(
            dual.curves["USD.SOFR.OIS"].discount_factors, before_dfs,
        )
        assert out is not dual

    def test_unshocked_leaves_are_shared_not_recomputed(self, dual):
        """A targeted shock leaves every other leaf alone (same objects)."""
        out = shock(dual, KeyedScenario(spot_shocks={"SPX": jnp.array(10.0)}))
        assert out.spots["NDX"] is dual.spots["NDX"]
        assert (
            out.curves["USD.SOFR.OIS"].discount_factors
            is dual.curves["USD.SOFR.OIS"].discount_factors
        )
        assert out.surfaces["SPX.VOL"].vol is dual.surfaces["SPX.VOL"].vol

    def test_untargeted_tables_keep_their_keys(self, dual):
        out = shock(dual, KeyedScenario(
            curve_shocks={"USD.SOFR.3M": jnp.full(3, 0.001)},
        ))
        assert set(out.curves.keys()) == set(dual.curves.keys())
        assert set(out.surfaces) == set(dual.surfaces)
        assert out.asset_ids == dual.asset_ids


# ── Per-curve (multi-curve) moves ────────────────────────────────────


class TestCurveShocksAreKeyed:
    def test_only_the_named_curve_moves(self, dual):
        """The whole point: OIS and projection are distinct risk factors."""
        out = shock(dual, KeyedScenario(
            curve_shocks={"USD.SOFR.3M": jnp.full(3, 0.01)},
        ))
        pillars = dual.curves["USD.SOFR.OIS"].pillar_dates
        assert jnp.allclose(
            out.curves["USD.SOFR.OIS"].discount_factors,
            dual.curves["USD.SOFR.OIS"].discount_factors,
        )
        assert jnp.isclose(
            zero_rate(out.curves["USD.SOFR.3M"], pillars[1]), 0.065, atol=1e-4,
        )

    def test_basis_move_is_expressible(self, dual):
        """A pure basis widening: forward up, discount down."""
        out = shock(dual, KeyedScenario(curve_shocks={
            "USD.SOFR.3M": jnp.full(3, 0.0025),
            "USD.SOFR.OIS": jnp.full(3, -0.0025),
        }))
        pillars = dual.curves["USD.SOFR.OIS"].pillar_dates
        basis_before = (
            zero_rate(dual.curves["USD.SOFR.3M"], pillars[1])
            - zero_rate(dual.curves["USD.SOFR.OIS"], pillars[1])
        )
        basis_after = (
            zero_rate(out.curves["USD.SOFR.3M"], pillars[1])
            - zero_rate(out.curves["USD.SOFR.OIS"], pillars[1])
        )
        assert jnp.isclose(basis_after - basis_before, 0.005, atol=1e-4)

    def test_key_rate_bump_hits_one_pillar(self, dual):
        bumps = jnp.zeros(3).at[2].set(0.01)
        out = shock(dual, KeyedScenario(
            curve_shocks={"USD.SOFR.OIS": bumps},
        ))
        pillars = dual.curves["USD.SOFR.OIS"].pillar_dates
        assert jnp.isclose(
            zero_rate(out.curves["USD.SOFR.OIS"], pillars[1]), 0.05, atol=1e-4,
        )
        assert jnp.isclose(
            zero_rate(out.curves["USD.SOFR.OIS"], pillars[2]), 0.06, atol=1e-4,
        )


# ── Vol shocks: the surface fork ─────────────────────────────────────


class TestVolShocks:
    def test_constant_vol_is_shifted_in_place(self, single):
        """ConstantVol stays a ConstantVol, so the legacy bridge survives."""
        out = shock(single, KeyedScenario(
            vol_shocks={DEFAULT_ID: jnp.array(0.01)},
        ))
        assert isinstance(out.surfaces[DEFAULT_ID], ConstantVol)
        assert out.surfaces[DEFAULT_ID].vol == single.surfaces[DEFAULT_ID].vol + 0.01
        out.to_market_data()  # must not raise

    def test_rich_surface_is_wrapped(self, dual):
        out = shock(dual, KeyedScenario(
            vol_shocks={"NDX.VOL": jnp.array(0.02)},
        ))
        wrapped = out.surfaces["NDX.VOL"]
        assert isinstance(wrapped, ShiftedVolSurface)
        k, t = jnp.array(100.0), jnp.array(1.0)
        assert jnp.isclose(wrapped(k, t), dual.surfaces["NDX.VOL"](k, t) + 0.02)

    def test_shifts_accumulate_without_nesting(self, dual):
        once = shock(dual, KeyedScenario(vol_shocks={"NDX.VOL": jnp.array(0.01)}))
        twice = shock(once, KeyedScenario(vol_shocks={"NDX.VOL": jnp.array(0.02)}))
        surface = twice.surfaces["NDX.VOL"]
        assert isinstance(surface, ShiftedVolSurface)
        assert not isinstance(surface.base, ShiftedVolSurface)
        assert jnp.isclose(surface.shift, 0.03)

    def test_shift_surface_is_pure(self, dual):
        base = dual.surfaces["NDX.VOL"]
        shift_surface(base, jnp.array(0.05))
        assert jnp.allclose(dual.surfaces["NDX.VOL"].vols, base.vols)


# ── Equivalence with the legacy flat representation ──────────────────


class TestLegacyEquivalence:
    def test_round_trip_matches_apply_scenario_exactly(self, flat_curve):
        """Keyed and flat scenarios must produce the same shocked market."""
        md = MarketData(
            spots=jnp.array([100.0, 200.0]),
            vols=jnp.array([0.20, 0.35]),
            dividends=jnp.array([0.01, 0.02]),
            discount_curve=flat_curve,
        )
        legacy = MarketScenario(
            spot_shocks=jnp.array([5.0, -10.0]),
            vol_shocks=jnp.array([0.01, -0.02]),
            rate_shocks=jnp.array([0.0, 0.005, 0.01]),
            dividend_shocks=jnp.array([0.0, 0.005]),
        )
        asset_ids = ("SPX", "NDX")
        state = MarketState.from_market_data(md, asset_ids=asset_ids)
        keyed = keyed_from_market_scenario(legacy, asset_ids=asset_ids)

        via_keyed = shock(state, keyed).to_market_data()
        via_legacy = apply_scenario(md, legacy)

        assert jnp.array_equal(via_keyed.spots, via_legacy.spots)
        assert jnp.array_equal(via_keyed.vols, via_legacy.vols)
        assert jnp.array_equal(via_keyed.dividends, via_legacy.dividends)
        assert jnp.array_equal(
            via_keyed.discount_curve.discount_factors,
            via_legacy.discount_curve.discount_factors,
        )

    def test_multiplicative_spots_match(self, flat_curve):
        md = MarketData(
            spots=jnp.array([100.0]),
            vols=jnp.array([0.2]),
            dividends=jnp.array([0.0]),
            discount_curve=flat_curve,
        )
        legacy = MarketScenario(
            spot_shocks=jnp.array([-0.10]),
            vol_shocks=jnp.zeros(1),
            rate_shocks=jnp.zeros(3),
            dividend_shocks=jnp.zeros(1),
            multiplicative=True,
        )
        state = MarketState.from_market_data(md, asset_ids=("SPX",))
        keyed = keyed_from_market_scenario(legacy, asset_ids=("SPX",))
        assert keyed.multiplicative
        out = shock(state, keyed)
        assert jnp.array_equal(
            out.spots["SPX"], apply_scenario(md, legacy).spots[0],
        )

    def test_lift_rejects_mismatched_asset_ids(self):
        legacy = MarketScenario(
            spot_shocks=jnp.zeros(2),
            vol_shocks=jnp.zeros(2),
            rate_shocks=jnp.zeros(3),
            dividend_shocks=jnp.zeros(2),
        )
        with pytest.raises(ValueError, match="asset_ids has length"):
            keyed_from_market_scenario(legacy, asset_ids=("SPX",))

    def test_lift_attributes_rates_to_the_named_curve(self, dual):
        legacy = MarketScenario(
            spot_shocks=jnp.zeros(2),
            vol_shocks=jnp.zeros(2),
            rate_shocks=jnp.full(3, 0.01),
            dividend_shocks=jnp.zeros(2),
        )
        keyed = keyed_from_market_scenario(
            legacy, asset_ids=("SPX", "NDX"), curve_id="USD.SOFR.3M",
        )
        out = shock(dual, keyed)
        pillars = dual.curves["USD.SOFR.OIS"].pillar_dates
        assert jnp.allclose(
            out.curves["USD.SOFR.OIS"].discount_factors,
            dual.curves["USD.SOFR.OIS"].discount_factors,
        )
        assert jnp.isclose(
            zero_rate(out.curves["USD.SOFR.3M"], pillars[1]), 0.065, atol=1e-4,
        )


# ── Batching ─────────────────────────────────────────────────────────


class TestStackKeyedScenarios:
    def test_leaves_gain_a_scenario_axis(self):
        scenarios = [
            KeyedScenario(
                spot_shocks={"SPX": jnp.array(float(i))},
                curve_shocks={"USD.SOFR.OIS": jnp.full(3, 0.001 * i)},
            )
            for i in range(4)
        ]
        stacked = stack_keyed_scenarios(scenarios)
        assert stacked.spot_shocks["SPX"].shape == (4,)
        assert stacked.curve_shocks["USD.SOFR.OIS"].shape == (4, 3)

    def test_vmap_over_the_scenario_axis(self, dual):
        scenarios = stack_keyed_scenarios([
            KeyedScenario(spot_shocks={"SPX": jnp.array(10.0)}),
            KeyedScenario(spot_shocks={"SPX": jnp.array(-10.0)}),
        ])
        spots = jax.vmap(lambda s: shock(dual, s).spots["SPX"])(scenarios)
        assert jnp.allclose(spots, jnp.array([5010.0, 4990.0]))

    def test_mismatched_keys_raise(self):
        with pytest.raises(ValueError, match="key sets differ"):
            stack_keyed_scenarios([
                KeyedScenario(spot_shocks={"SPX": jnp.array(1.0)}),
                KeyedScenario(spot_shocks={"NDX": jnp.array(1.0)}),
            ])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one scenario"):
            stack_keyed_scenarios([])

    def test_mismatched_multiplicative_raises(self):
        with pytest.raises(ValueError, match="multiplicative"):
            stack_keyed_scenarios([
                KeyedScenario(spot_shocks={"SPX": jnp.array(1.0)}),
                KeyedScenario(
                    spot_shocks={"SPX": jnp.array(1.0)}, multiplicative=True,
                ),
            ])


# ── JAX contract ─────────────────────────────────────────────────────


class TestJaxContract:
    def test_grad_flows_to_the_shock_magnitudes(self, dual):
        """The scenario itself is differentiable — shock-space sensitivities."""
        pillars = dual.curves["USD.SOFR.OIS"].pillar_dates

        def value(spot_shock, vol_shock, rate_bump):
            out = shock(dual, KeyedScenario(
                spot_shocks={"SPX": spot_shock},
                vol_shocks={"NDX.VOL": vol_shock},
                curve_shocks={"USD.SOFR.3M": jnp.full(3, rate_bump)},
            ))
            return (
                out.spots["SPX"]
                + 100.0 * out.surfaces["NDX.VOL"](
                    jnp.array(100.0), jnp.array(1.0),
                )
                + 1000.0 * out.curves["USD.SOFR.3M"](pillars[2])
            )

        g_spot, g_vol, g_rate = jax.grad(value, argnums=(0, 1, 2))(
            jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
        )
        assert jnp.isclose(g_spot, 1.0)
        assert jnp.isclose(g_vol, 100.0)
        assert g_rate < 0.0  # rates up ⇒ discount factor down

    def test_grad_flows_to_the_base_market(self, dual):
        """Shocking does not sever sensitivity to the calibrated market."""
        def value(spot):
            state = eqx.tree_at(lambda m: m.spots["SPX"], dual, spot)
            out = shock(state, KeyedScenario(
                spot_shocks={"SPX": jnp.array(10.0)},
            ))
            return out.spots["SPX"] ** 2

        g = jax.grad(value)(jnp.array(5000.0))
        assert jnp.isclose(g, 2.0 * 5010.0)

    def test_multiplicative_grad(self, dual):
        def value(ret):
            out = shock(dual, KeyedScenario(
                spot_shocks={"SPX": ret}, multiplicative=True,
            ))
            return out.spots["SPX"]

        assert jnp.isclose(jax.grad(value)(jnp.array(0.0)), 5000.0)

    def test_filter_jit_smoke(self, dual):
        @eqx.filter_jit
        def run(state, scenario):
            out = shock(state, scenario)
            return (
                out.spots["SPX"],
                out.curves["USD.SOFR.3M"].discount_factors,
                out.surfaces["NDX.VOL"](jnp.array(100.0), jnp.array(1.0)),
            )

        scenario = KeyedScenario(
            spot_shocks={"SPX": jnp.array(25.0)},
            vol_shocks={"NDX.VOL": jnp.array(0.01)},
            curve_shocks={"USD.SOFR.3M": jnp.full(3, 0.001)},
            dividend_shocks={"SPX": jnp.array(0.001)},
        )
        spot, dfs, vol = run(dual, scenario)
        assert spot == 5025.0
        assert dfs.shape == (3,)
        assert jnp.isfinite(vol)
