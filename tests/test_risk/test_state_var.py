"""Tests for group repricing / P&L / gradients through ``price(...)``.

The gates here are of four kinds:

1. **Legacy equivalence** — the ported path must reproduce
   :mod:`valax.risk.var` *exactly* wherever the arithmetic is unchanged.
   For rates groups that is the whole computation; both paths call
   ``swap_price`` on a curve shocked by the same zero-rate algebra.
2. **Quantified divergence** — where the paths genuinely differ, the
   difference is measured and named rather than absorbed into a loose
   ``atol``.  The one place that happens is
   :func:`valax.risk.var._extract_short_rate`.
3. **Multi-curve** — the payoff of the port: a basis move the legacy
   unkeyed scenario structurally cannot express.
4. **Common random numbers** — the trap that only appears once groups
   price through ``price(..., key=...)``.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.core.market_state import MarketState
from valax.curves.discount import DiscountCurve, zero_rate
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.bonds import FixedRateBond
from valax.instruments.options import EuropeanOption
from valax.instruments.rates import InterestRateSwap
from valax.market.data import MarketData
from valax.market.keyed_scenario import KeyedScenario, stack_keyed_scenarios
from valax.market.scenario import MarketScenario
from valax.models.black_scholes import BlackScholesModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.analytic.swaptions import swap_price
from valax.pricing.mc.engine import MCConfig
from valax.risk.state_var import (
    _shadowed_shocks,
    group_market_gradient,
    group_model_gradient,
    group_prices,
    group_pv,
    group_reprice,
    group_scenario_pnl,
    group_size,
)
from valax.risk.var import _extract_short_rate, reprice_under_scenario


REF = ymd_to_ordinal(2026, 1, 1)
PILLARS = jnp.array([
    ymd_to_ordinal(2026, 1, 1),
    ymd_to_ordinal(2026, 7, 1),
    ymd_to_ordinal(2027, 1, 1),
    ymd_to_ordinal(2028, 1, 1),
])
TIMES = (PILLARS - REF).astype(jnp.float64) / 365.0
N_PILLARS = 4


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def flat_curve():
    """Flat 5% continuously-compounded curve on an act/365 time axis."""
    return DiscountCurve(
        pillar_dates=PILLARS,
        discount_factors=jnp.exp(-0.05 * TIMES),
        reference_date=REF,
    )


@pytest.fixture
def sloped_curve():
    """Upward-sloping curve: 3% short, 4.5% at 1Y, 6% at 2Y."""
    zero_rates = jnp.array([0.03, 0.03, 0.045, 0.06])
    return DiscountCurve(
        pillar_dates=PILLARS,
        discount_factors=jnp.exp(-zero_rates * TIMES),
        reference_date=REF,
    )


@pytest.fixture
def swaps():
    """A two-swap homogeneous group, single-curve (no projection id)."""
    return InterestRateSwap(
        start_date=jnp.array([REF, REF]),
        fixed_dates=jnp.array([
            [ymd_to_ordinal(2027, 1, 1), ymd_to_ordinal(2028, 1, 1)],
            [ymd_to_ordinal(2027, 1, 1), ymd_to_ordinal(2028, 1, 1)],
        ]),
        fixed_rate=jnp.array([0.04, 0.05]),
        notional=jnp.array([1e6, 1e6]),
    )


@pytest.fixture
def options():
    """A two-option homogeneous group: same underlying, both calls."""
    return EuropeanOption(
        strike=jnp.array([100.0, 110.0]),
        expiry=jnp.array([1.0, 1.0]),
        is_call=True,
    )


@pytest.fixture
def bs_model():
    return BlackScholesModel(
        vol=jnp.array(0.2), rate=jnp.array(0.05), dividend=jnp.array(0.0),
    )


@pytest.fixture
def equity_state(flat_curve):
    return MarketState.single_curve(
        flat_curve,
        spot=jnp.array(100.0),
        vol=jnp.array(0.2),
        dividend=jnp.array(0.0),
    )


def _legacy_swap_fn(swap, market: MarketData):
    """Legacy rates pricing_fn: reads the full curve, no flattening."""
    return swap_price(swap, market.discount_curve)


def _legacy_bs_fn(option, market: MarketData):
    """Legacy equity pricing_fn, with the short-rate collapse."""
    rate = _extract_short_rate(market.discount_curve)
    return black_scholes_price(
        option, market.spots, market.vols, rate, market.dividends,
    )


def _zero_legacy_scenario(n_assets: int) -> MarketScenario:
    return MarketScenario(
        spot_shocks=jnp.zeros(n_assets),
        vol_shocks=jnp.zeros(n_assets),
        rate_shocks=jnp.zeros(N_PILLARS),
        dividend_shocks=jnp.zeros(n_assets),
    )


# ── Group validation ─────────────────────────────────────────────────


class TestGroupSize:
    def test_stacked_group(self, swaps):
        assert group_size(swaps) == 2

    def test_single_instrument_is_rejected(self):
        """A scalar-leaf pytree is one instrument, not a group of one."""
        single = EuropeanOption(
            strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True,
        )
        with pytest.raises(ValueError, match="single instrument"):
            group_size(single)

    def test_ragged_group_is_rejected(self):
        ragged = EuropeanOption(
            strike=jnp.array([100.0, 110.0]),
            expiry=jnp.array([1.0, 1.0, 1.0]),
            is_call=True,
        )
        with pytest.raises(ValueError, match="leading batch axis"):
            group_size(ragged)

    def test_no_array_leaves_is_rejected(self):
        with pytest.raises(ValueError, match="no array leaves"):
            group_size(object())


# ── Legacy equivalence: rates (the arithmetic is unchanged) ──────────


class TestLegacyEquivalenceRates:
    """Both paths call ``swap_price`` on the same curve — so: bit-exact.

    This is the regression gate for the port.  The legacy path shocks
    via ``apply_scenario`` → ``bump_curve_zero_rates``; the ported path
    via ``shock`` → ``bump_graph_curve``.  Part 1 made those two produce
    identical curves, so any difference here is a porting bug.
    """

    def test_base_pv_is_bit_identical(self, swaps, flat_curve):
        state = MarketState(
            curves=CurveGraph(curves={"_default_": flat_curve}),
        )
        ported = group_pv(swaps, None, state)

        legacy_market = MarketData(
            spots=jnp.zeros(2), vols=jnp.zeros(2), dividends=jnp.zeros(2),
            discount_curve=flat_curve,
        )
        legacy = reprice_under_scenario(
            _legacy_swap_fn, swaps, legacy_market, _zero_legacy_scenario(2),
        )
        assert ported == legacy

    def test_curve_shock_is_bit_identical(self, swaps, flat_curve):
        bumps = jnp.full(N_PILLARS, 0.0010)  # +10bp parallel

        state = MarketState(
            curves=CurveGraph(curves={"_default_": flat_curve}),
        )
        ported = group_reprice(
            swaps, None, state, KeyedScenario(curve_shocks={"_default_": bumps}),
        )

        legacy_market = MarketData(
            spots=jnp.zeros(2), vols=jnp.zeros(2), dividends=jnp.zeros(2),
            discount_curve=flat_curve,
        )
        legacy = reprice_under_scenario(
            _legacy_swap_fn, swaps, legacy_market,
            MarketScenario(
                spot_shocks=jnp.zeros(2), vol_shocks=jnp.zeros(2),
                rate_shocks=bumps, dividend_shocks=jnp.zeros(2),
            ),
        )
        assert ported == legacy

    def test_prices_sum_to_pv(self, swaps, flat_curve):
        state = MarketState(
            curves=CurveGraph(curves={"_default_": flat_curve}),
        )
        prices = group_prices(swaps, None, state)
        assert prices.shape == (2,)
        assert jnp.sum(prices) == group_pv(swaps, None, state)


# ── Legacy equivalence: equity ───────────────────────────────────────


class TestLegacyEquivalenceEquity:
    """Exact too — *provided* the model rate equals the collapsed rate.

    The legacy path derives its scalar rate from the curve with
    ``_extract_short_rate``; the ported path reads ``model.rate``.  Feed
    the ported path the same scalar and the two agree bit-for-bit, which
    isolates the collapse itself as the only difference between them
    (quantified in :class:`TestShortRateCollapse`).
    """

    def test_base_pv_is_bit_identical(self, options, flat_curve, equity_state):
        collapsed = _extract_short_rate(flat_curve)
        model = BlackScholesModel(
            vol=jnp.array(0.2), rate=collapsed, dividend=jnp.array(0.0),
        )
        ported = group_pv(options, model, equity_state)

        legacy_market = MarketData(
            spots=jnp.array([100.0, 100.0]), vols=jnp.array([0.2, 0.2]),
            dividends=jnp.zeros(2), discount_curve=flat_curve,
        )
        legacy = reprice_under_scenario(
            _legacy_bs_fn, options, legacy_market, _zero_legacy_scenario(2),
        )
        assert ported == legacy

    def test_spot_shock_is_bit_identical(
        self, options, flat_curve, equity_state,
    ):
        collapsed = _extract_short_rate(flat_curve)
        model = BlackScholesModel(
            vol=jnp.array(0.2), rate=collapsed, dividend=jnp.array(0.0),
        )
        ported = group_reprice(
            options, model, equity_state,
            KeyedScenario(spot_shocks={"_default_": jnp.array(5.0)}),
        )

        legacy_market = MarketData(
            spots=jnp.array([100.0, 100.0]), vols=jnp.array([0.2, 0.2]),
            dividends=jnp.zeros(2), discount_curve=flat_curve,
        )
        legacy = reprice_under_scenario(
            _legacy_bs_fn, options, legacy_market,
            MarketScenario(
                spot_shocks=jnp.full(2, 5.0), vol_shocks=jnp.zeros(2),
                rate_shocks=jnp.zeros(N_PILLARS),
                dividend_shocks=jnp.zeros(2),
            ),
        )
        assert ported == legacy


# ── The approximation that goes away, quantified ─────────────────────


class TestShortRateCollapse:
    """``_extract_short_rate`` collapses the term structure to one pillar.

    It returns ``zero_rates[1]`` — the *second* pillar — with a
    hard-coded ``/365.0`` that ignores the curve's own day count.  On the
    ``sloped_curve`` fixture (3% short, 4.5% at 1Y) that is a measured
    **150 bp** error for a 1Y option, worth **9.02%** of the group PV.

    The ported path has no such collapse. A rates group discounts on the
    whole curve. An equity group's rate is *either* a calibrated constant
    on the model (the mode measured below) *or*, when the model leaves it
    ``None``, the zero rate at that trade's **own** maturity — see
    ``tests/test_core/test_binding.py::TestProjectRate``, which recovers
    the same 9.02 % from the other side.
    """

    def test_collapsed_rate_is_the_short_pillar(self, sloped_curve):
        collapsed = _extract_short_rate(sloped_curve)
        true_1y = zero_rate(sloped_curve, ymd_to_ordinal(2027, 1, 1))

        assert float(collapsed) == pytest.approx(0.03, abs=1e-12)
        assert float(true_1y) == pytest.approx(0.045, abs=1e-12)
        gap_bp = float((true_1y - collapsed) * 1e4)
        assert gap_bp == pytest.approx(150.0, abs=0.1)

    def test_collapse_costs_nine_percent_of_pv(self, options, sloped_curve):
        state = MarketState.single_curve(
            sloped_curve, spot=jnp.array(100.0), vol=jnp.array(0.2),
            dividend=jnp.array(0.0),
        )
        collapsed = _extract_short_rate(sloped_curve)
        true_1y = zero_rate(sloped_curve, ymd_to_ordinal(2027, 1, 1))

        pv_collapsed = group_pv(
            options,
            BlackScholesModel(
                vol=jnp.array(0.2), rate=collapsed, dividend=jnp.array(0.0),
            ),
            state,
        )
        pv_true = group_pv(
            options,
            BlackScholesModel(
                vol=jnp.array(0.2), rate=true_1y, dividend=jnp.array(0.0),
            ),
            state,
        )

        relative = float((pv_true - pv_collapsed) / pv_collapsed)
        assert relative == pytest.approx(0.0902, abs=5e-4)

    def test_rates_group_sees_the_whole_curve(self, swaps, sloped_curve):
        """No single-pillar bottleneck: bump only the 2Y and the PV moves.

        Under the collapse a shock to any pillar but the second is
        invisible.  Through ``price`` every pillar carries risk.
        """
        state = MarketState(
            curves=CurveGraph(curves={"_default_": sloped_curve}),
        )
        base = group_pv(swaps, None, state)
        only_last = jnp.array([0.0, 0.0, 0.0, 0.0050])
        bumped = group_reprice(
            swaps, None, state,
            KeyedScenario(curve_shocks={"_default_": only_last}),
        )
        assert abs(float(bumped - base)) > 1.0


class TestEquityCurveShockIsInert:
    """True in **constant mode** only — pinned so the distinction cannot drift.

    A ``BlackScholesModel`` given an explicit ``rate`` / ``vol`` /
    ``dividend`` reads those parameters, not the market, so a curve or
    surface shock does not move its price and that risk sits on
    :func:`group_model_gradient` rather than
    :func:`group_market_gradient`.  That is the textbook, self-contained
    model, and it stays supported.

    Leaving a parameter ``None`` sources it from the ``MarketState``
    instead (:func:`valax.core.binding.bind_model`), and then the shock
    *does* reach the price — see
    ``tests/test_core/test_binding.py::TestCurveShockNowReachesEquity``.
    The tests below therefore fix the constant mode deliberately.
    """

    def test_curve_shock_does_not_move_a_black_scholes_price(
        self, options, bs_model, equity_state,
    ):
        base = group_pv(options, bs_model, equity_state)
        with pytest.warns(UserWarning, match="cannot reach the price"):
            shocked = group_reprice(
                options, bs_model, equity_state,
                KeyedScenario(
                    curve_shocks={"_default_": jnp.full(N_PILLARS, 0.01)},
                ),
            )
        assert shocked == base

    def test_rate_risk_is_on_the_model_gradient_in_constant_mode(
        self, options, bs_model, equity_state,
    ):
        market_grad = group_market_gradient(options, bs_model, equity_state)
        model_grad = group_model_gradient(options, bs_model, equity_state)

        # Spot risk is on the market...
        assert float(market_grad.spots["_default_"]) > 0.0
        # ...rate and vol risk are on the model.
        assert float(model_grad.rate) > 0.0
        assert float(model_grad.vol) > 0.0

    def test_model_gradient_refuses_a_model_free_group(self, swaps, flat_curve):
        state = MarketState(
            curves=CurveGraph(curves={"_default_": flat_curve}),
        )
        with pytest.raises(ValueError, match="no model parameters"):
            group_model_gradient(swaps, None, state)


# ── The silent zero, made loud ───────────────────────────────────────


class TestShadowedShockGuard:
    """A shock that cannot reach the price must not pass silently.

    Contributing exactly 0 P&L is the same failure class as the dropped
    `curve_graph` — a plausible number with no error — so it is warned
    about at the user boundary, detected structurally from the model
    pytree's own fields rather than guessed.
    """

    def test_curve_shock_on_an_equity_group_warns(
        self, options, bs_model, equity_state,
    ):
        with pytest.warns(
            UserWarning, match=r"curve_shocks.*BlackScholesModel\.rate",
        ):
            group_reprice(
                options, bs_model, equity_state,
                KeyedScenario(
                    curve_shocks={"_default_": jnp.full(N_PILLARS, 0.001)},
                ),
            )

    def test_vol_and_dividend_shocks_also_warn(
        self, options, bs_model, equity_state,
    ):
        with pytest.warns(UserWarning, match="vol_shocks") as record:
            group_reprice(
                options, bs_model, equity_state,
                KeyedScenario(
                    vol_shocks={"_default_": jnp.array(0.01)},
                    dividend_shocks={"_default_": jnp.array(0.005)},
                ),
            )
        assert "dividend_shocks" in str(record[0].message)

    def test_scenario_pnl_warns_once_naming_itself(
        self, options, bs_model, equity_state,
    ):
        scenarios = stack_keyed_scenarios([
            KeyedScenario(curve_shocks={"_default_": jnp.zeros(N_PILLARS)}),
            KeyedScenario(
                curve_shocks={"_default_": jnp.full(N_PILLARS, 0.001)},
            ),
        ])
        with pytest.warns(UserWarning, match="group_scenario_pnl") as record:
            pnl = group_scenario_pnl(options, bs_model, equity_state, scenarios)

        assert len(record) == 1
        # ...and the warning tells the truth: the P&L really is zero.
        assert jnp.all(pnl == 0.0)

    def test_spot_shock_on_an_equity_group_does_not_warn(
        self, options, bs_model, equity_state,
    ):
        """Spot *is* read from the market, so nothing is shadowed."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            group_reprice(
                options, bs_model, equity_state,
                KeyedScenario(spot_shocks={"_default_": jnp.array(1.0)}),
            )

    def test_model_free_group_does_not_warn(self, swaps, flat_curve):
        state = MarketState(
            curves=CurveGraph(curves={"_default_": flat_curve}),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            group_reprice(
                swaps, None, state,
                KeyedScenario(
                    curve_shocks={"_default_": jnp.full(N_PILLARS, 0.001)},
                ),
            )

    def test_curve_referencing_model_does_not_warn(self, flat_curve):
        """Hull-White reads the market curve, so its shocks are not shadowed.

        Verified behaviourally as well as structurally: the HW price moves
        under a curve shock because the model is curve-free and ``price()``
        binds its ``initial_curve`` from the (shocked) market.
        """
        state = MarketState(
            curves=CurveGraph(curves={"_default_": flat_curve}),
        )
        hw = HullWhiteModel(
            mean_reversion=jnp.array(0.03), volatility=jnp.array(0.01),
        )
        bonds = FixedRateBond(
            payment_dates=jnp.tile(PILLARS[1:3], (2, 1)),
            settlement_date=jnp.array([REF, REF]),
            coupon_rate=jnp.array([0.04, 0.05]),
            face_value=jnp.array([100.0, 100.0]),
            frequency=1,
        )
        scenario = KeyedScenario(
            curve_shocks={"_default_": jnp.full(N_PILLARS, 0.001)},
        )

        assert _shadowed_shocks(hw, scenario) == []

        config = MCConfig(n_paths=2_000, n_steps=6)
        key = jax.random.PRNGKey(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            base = group_pv(
                bonds, hw, state, method="mc", config=config, key=key,
            )
            moved = group_reprice(
                bonds, hw, state, scenario,
                method="mc", config=config, key=key,
            )
        assert moved != base


# ── Multi-curve: the payoff ──────────────────────────────────────────


class TestMultiCurveBasisRisk:
    """A basis move is expressible, and it reaches the P&L."""

    @pytest.fixture
    def dual_state(self, flat_curve):
        fwd = DiscountCurve(
            pillar_dates=PILLARS,
            discount_factors=jnp.exp(-0.055 * TIMES),  # +50bp basis
            reference_date=REF,
        )
        return MarketState(curves=CurveGraph(curves={
            "USD.SOFR.OIS": flat_curve,
            "USD.SOFR.3M": fwd,
        }))

    @pytest.fixture
    def dual_swaps(self):
        return InterestRateSwap(
            start_date=jnp.array([REF, REF]),
            fixed_dates=jnp.array([
                [ymd_to_ordinal(2027, 1, 1), ymd_to_ordinal(2028, 1, 1)],
                [ymd_to_ordinal(2027, 1, 1), ymd_to_ordinal(2028, 1, 1)],
            ]),
            fixed_rate=jnp.array([0.04, 0.05]),
            notional=jnp.array([1e6, 1e6]),
            discount_id="USD.SOFR.OIS",
            forward_id="USD.SOFR.3M",
        )

    def test_basis_widening_moves_the_book(self, dual_swaps, dual_state):
        """+25bp on the projection curve only: measured ≈ +9817 on the book."""
        base = group_pv(dual_swaps, None, dual_state)
        widened = group_reprice(
            dual_swaps, None, dual_state,
            KeyedScenario(curve_shocks={"USD.SOFR.3M": jnp.full(N_PILLARS, 0.0025)}),
        )
        delta = float(widened - base)
        assert delta > 0.0  # receiving a higher projected float leg
        assert delta == pytest.approx(9817.3, rel=1e-3)

    def test_each_curve_carries_its_own_gradient(self, dual_swaps, dual_state):
        """The gradient pytree keys risk by curve — not one rate block."""
        grad = group_market_gradient(dual_swaps, None, dual_state)
        ois = grad.curves["USD.SOFR.OIS"].discount_factors
        fwd = grad.curves["USD.SOFR.3M"].discount_factors

        assert ois.shape == (N_PILLARS,)
        assert fwd.shape == (N_PILLARS,)
        assert jnp.any(ois != 0.0)
        assert jnp.any(fwd != 0.0)

    def test_legacy_path_cannot_express_the_basis_move(
        self, dual_swaps, flat_curve,
    ):
        """The same intent, through the unkeyed vector, is a different trade.

        Handed a 25bp ``rate_shocks`` vector the legacy path moves the
        *discount* curve and leaves the projection curve frozen — a
        measured −150 P&L where the basis move is +9817.  It warns, but
        the number it returns answers a different question.
        """
        fwd = DiscountCurve(
            pillar_dates=PILLARS,
            discount_factors=jnp.exp(-0.055 * TIMES),
            reference_date=REF,
        )
        legacy_market = MarketData(
            spots=jnp.zeros(2), vols=jnp.zeros(2), dividends=jnp.zeros(2),
            discount_curve=flat_curve,
            curve_graph=CurveGraph(curves={
                "_default_": flat_curve, "USD.SOFR.3M": fwd,
            }),
        )
        single_curve_swaps = InterestRateSwap(
            start_date=dual_swaps.start_date,
            fixed_dates=dual_swaps.fixed_dates,
            fixed_rate=dual_swaps.fixed_rate,
            notional=dual_swaps.notional,
        )

        def graph_fn(swap, market: MarketData):
            return swap_price(
                swap, market.discount_curve, market.curve_graph["USD.SOFR.3M"],
            )

        with pytest.warns(UserWarning, match="held fixed"):
            base = reprice_under_scenario(
                graph_fn, single_curve_swaps, legacy_market,
                _zero_legacy_scenario(2),
            )
            shocked = reprice_under_scenario(
                graph_fn, single_curve_swaps, legacy_market,
                MarketScenario(
                    spot_shocks=jnp.zeros(2), vol_shocks=jnp.zeros(2),
                    rate_shocks=jnp.full(N_PILLARS, 0.0025),
                    dividend_shocks=jnp.zeros(2),
                ),
            )

        legacy_delta = float(shocked - base)
        assert legacy_delta < 0.0        # opposite sign to the basis move
        assert abs(legacy_delta) < 500.0  # and two orders of magnitude smaller


# ── Scenario P&L vectors ─────────────────────────────────────────────


class TestScenarioPnl:
    def test_shape_and_zero_scenario(self, swaps, flat_curve):
        state = MarketState(
            curves=CurveGraph(curves={"_default_": flat_curve}),
        )
        scenarios = stack_keyed_scenarios([
            KeyedScenario(curve_shocks={"_default_": jnp.zeros(N_PILLARS)}),
            KeyedScenario(curve_shocks={"_default_": jnp.full(N_PILLARS, 0.0010)}),
            KeyedScenario(curve_shocks={"_default_": jnp.full(N_PILLARS, -0.0010)}),
        ])
        pnl = group_scenario_pnl(swaps, None, state, scenarios)

        assert pnl.shape == (3,)
        assert pnl[0] == 0.0          # zero scenario is exactly flat
        assert pnl[1] > 0.0 > pnl[2]  # payer swaps gain when rates rise

    def test_matches_portfolio_pnl_elementwise(self, swaps, flat_curve):
        """Vectorised P&L equals the per-scenario repricing, exactly."""
        state = MarketState(
            curves=CurveGraph(curves={"_default_": flat_curve}),
        )
        shocks = [jnp.zeros(N_PILLARS), jnp.full(N_PILLARS, 0.0025)]
        scenarios = stack_keyed_scenarios(
            [KeyedScenario(curve_shocks={"_default_": s}) for s in shocks]
        )
        vectorised = group_scenario_pnl(swaps, None, state, scenarios)

        base = group_pv(swaps, None, state)
        one_by_one = jnp.array([
            group_reprice(
                swaps, None, state,
                KeyedScenario(curve_shocks={"_default_": s}),
            ) - base
            for s in shocks
        ])
        assert jnp.array_equal(vectorised, one_by_one)


# ── Common random numbers ────────────────────────────────────────────


class TestCommonRandomNumbers:
    """Fix the key per group, reuse it for base and every revaluation.

    Analytic pricers have hidden this: with an MC group, drawing a fresh
    key per revaluation turns scenario P&L into sampling noise.  The
    measured noise below is −0.207 on a zero scenario whose true P&L is
    0, against a +1.16 P&L for a genuine +1 spot move — i.e. the noise is
    ~18% of the signal.
    """

    @pytest.fixture
    def mc_config(self):
        return MCConfig(n_paths=20_000, n_steps=1)

    def test_zero_scenario_pnl_is_exactly_zero(
        self, options, bs_model, equity_state, mc_config,
    ):
        scenarios = stack_keyed_scenarios([
            KeyedScenario(spot_shocks={"_default_": jnp.array(0.0)}),
            KeyedScenario(spot_shocks={"_default_": jnp.array(1.0)}),
        ])
        pnl = group_scenario_pnl(
            options, bs_model, equity_state, scenarios,
            method="mc", config=mc_config, key=jax.random.PRNGKey(7),
        )

        # Exactly zero: same key ⇒ same paths ⇒ same price, to the bit.
        assert pnl[0] == 0.0
        assert float(pnl[1]) > 0.5

    def test_fresh_key_per_revaluation_is_noise(
        self, options, bs_model, equity_state, mc_config,
    ):
        """The test that fails without common random numbers.

        Same market, same instruments, *zero* shock — but a different key
        for the base and the revaluation.  The P&L should be 0 and is
        not.
        """
        base = group_pv(
            options, bs_model, equity_state,
            method="mc", config=mc_config, key=jax.random.PRNGKey(1),
        )
        shocked = group_reprice(
            options, bs_model, equity_state,
            KeyedScenario(spot_shocks={"_default_": jnp.array(0.0)}),
            method="mc", config=mc_config, key=jax.random.PRNGKey(2),
        )
        spurious = float(shocked - base)
        assert abs(spurious) > 0.05  # measured −0.207

    def test_same_key_reprices_identically(
        self, options, bs_model, equity_state, mc_config,
    ):
        key = jax.random.PRNGKey(3)
        a = group_pv(
            options, bs_model, equity_state,
            method="mc", config=mc_config, key=key,
        )
        b = group_reprice(
            options, bs_model, equity_state, KeyedScenario(),
            method="mc", config=mc_config, key=key,
        )
        assert a == b


# ── JAX contract ─────────────────────────────────────────────────────


class TestJaxContract:
    def test_grad_wrt_market(self, swaps, flat_curve):
        state = MarketState(
            curves=CurveGraph(curves={"_default_": flat_curve}),
        )
        grad = group_market_gradient(swaps, None, state)
        dfs = grad.curves["_default_"].discount_factors
        assert dfs.shape == (N_PILLARS,)
        assert jnp.all(jnp.isfinite(dfs))
        assert jnp.any(dfs != 0.0)

    def test_grad_wrt_shock_magnitude(self, swaps, flat_curve):
        """``jax.grad`` flows through the ported loop into the scenario."""
        state = MarketState(
            curves=CurveGraph(curves={"_default_": flat_curve}),
        )

        def pnl_of_bump(size):
            scenario = KeyedScenario(
                curve_shocks={"_default_": jnp.full(N_PILLARS, size)},
            )
            return group_reprice(swaps, None, state, scenario)

        sensitivity = jax.grad(pnl_of_bump)(jnp.array(0.0))
        assert jnp.isfinite(sensitivity)
        assert sensitivity != 0.0

    def test_grad_through_the_scenario_pnl_vector(self, swaps, flat_curve):
        state = MarketState(
            curves=CurveGraph(curves={"_default_": flat_curve}),
        )
        scenarios = stack_keyed_scenarios([
            KeyedScenario(curve_shocks={"_default_": jnp.zeros(N_PILLARS)}),
            KeyedScenario(curve_shocks={"_default_": jnp.full(N_PILLARS, 0.001)}),
        ])

        def total_pnl(notional):
            group = eqx.tree_at(lambda s: s.notional, swaps, notional)
            return jnp.sum(group_scenario_pnl(group, None, state, scenarios))

        g = jax.grad(total_pnl)(swaps.notional)
        assert g.shape == (2,)
        assert jnp.all(jnp.isfinite(g))

    def test_filter_jit_smoke(self, swaps, flat_curve):
        state = MarketState(
            curves=CurveGraph(curves={"_default_": flat_curve}),
        )
        scenario = KeyedScenario(
            curve_shocks={"_default_": jnp.full(N_PILLARS, 0.001)},
        )

        @eqx.filter_jit
        def run(group, market, scen):
            return group_reprice(group, None, market, scen)

        jitted = run(swaps, state, scenario)
        eager = group_reprice(swaps, None, state, scenario)
        assert jnp.allclose(jitted, eager, atol=1e-10)

    def test_filter_jit_smoke_equity(self, options, bs_model, equity_state):
        @eqx.filter_jit
        def run(group, model, market):
            return group_pv(group, model, market)

        assert jnp.isfinite(run(options, bs_model, equity_state))


# ── Aggregation contract (what the portfolio layer will rely on) ─────


class TestAggregationContract:
    """Per-group gradients share a structure and add — § 0.1 point 1."""

    def test_gradients_tree_map_add_across_groups(
        self, swaps, flat_curve,
    ):
        import operator

        state = MarketState(
            curves=CurveGraph(curves={"_default_": flat_curve}),
        )
        first = eqx.tree_at(
            lambda s: s.fixed_rate, swaps, jnp.array([0.04, 0.05]),
        )
        second = eqx.tree_at(
            lambda s: s.fixed_rate, swaps, jnp.array([0.03, 0.06]),
        )

        g1 = group_market_gradient(first, None, state)
        g2 = group_market_gradient(second, None, state)
        summed = jax.tree_util.tree_map(operator.add, g1, g2)

        expected = (
            g1.curves["_default_"].discount_factors
            + g2.curves["_default_"].discount_factors
        )
        assert jnp.array_equal(
            summed.curves["_default_"].discount_factors, expected,
        )
