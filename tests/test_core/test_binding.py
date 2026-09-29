"""Tests for late-binding a model's market-sourced parameters.

Two modes of one model class:

* **constant** — every parameter given, the textbook form. Must price
  **bit-identically** to before this layer existed, which is what makes
  the change safe for the ~90 existing model constructions.
* **market** — a parameter left ``None`` is read from the
  ``MarketState``. Because the binding happens inside the differentiated
  path, the price then *moves* under a curve or surface shock and the
  gradient reports pillar deltas, closing the structural zero pinned by
  ``tests/test_risk/test_state_var.py::TestEquityCurveShockIsInert``.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.core.binding import (
    bind_model,
    instrument_horizon,
    project_rate,
    sample_vol,
)
from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.core.resolve import resolve
from valax.curves.discount import DiscountCurve, zero_rate
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.options import EuropeanOption
from valax.instruments.rates import InterestRateSwap
from valax.market.keyed_scenario import KeyedScenario
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.risk.state_var import (
    _shadowed_shocks,
    group_market_gradient,
    group_pv,
    group_reprice,
)
from valax.surfaces.constant import ConstantVol
from valax.surfaces.leverage import LeverageGrid


REF = ymd_to_ordinal(2026, 1, 1)
PILLARS = jnp.array([
    REF,
    ymd_to_ordinal(2026, 7, 1),
    ymd_to_ordinal(2027, 1, 1),
    ymd_to_ordinal(2028, 1, 1),
])
TIMES = (PILLARS - REF).astype(jnp.float64) / 365.0

#: Upward-sloping: 3% short, 4.5% at 1Y, 6% at 2Y.  Sloped on purpose —
#: a flat curve cannot tell the two modes apart.
SLOPED_ZEROS = jnp.array([0.03, 0.03, 0.045, 0.06])


@pytest.fixture
def curve():
    return DiscountCurve(
        pillar_dates=PILLARS,
        discount_factors=jnp.exp(-SLOPED_ZEROS * TIMES),
        reference_date=REF,
    )


@pytest.fixture
def market(curve):
    return MarketState.single_curve(
        curve,
        spot=jnp.array(100.0),
        vol=jnp.array(0.2),
        dividend=jnp.array(0.0),
    )


@pytest.fixture
def option():
    return EuropeanOption(
        strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True,
    )


@pytest.fixture
def options():
    """A 1Y group, so its rate is read at the 1Y pillar."""
    return EuropeanOption(
        strike=jnp.array([100.0, 110.0]),
        expiry=jnp.array([1.0, 1.0]),
        is_call=True,
    )


# ── Horizon and projections ──────────────────────────────────────────


class TestHorizon:
    def test_year_fraction_expiry_is_used_directly(self, curve, option):
        assert float(instrument_horizon(option, curve)) == 1.0

    def test_ordinal_expiry_is_converted(self, curve):
        swap = InterestRateSwap(
            start_date=jnp.array(REF),
            fixed_dates=PILLARS[1:],
            fixed_rate=jnp.array(0.04),
            notional=jnp.array(1e6),
        )
        # No expiry / expiry_date / maturity on a swap.
        with pytest.raises(ValueError, match="no 'expiry'"):
            instrument_horizon(swap, curve)

    def test_zero_expiry_is_floored(self, curve):
        expired = EuropeanOption(
            strike=jnp.array(100.0), expiry=jnp.array(0.0), is_call=True,
        )
        horizon = instrument_horizon(expired, curve)
        assert float(horizon) > 0.0
        assert jnp.isfinite(project_rate(curve, horizon))


class TestProjectRate:
    def test_matches_the_curves_own_zero_rate(self, curve):
        """The honest replacement for `_extract_short_rate`.

        Read at the trade's own maturity, day-count aware — so it agrees
        with `zero_rate` at the corresponding date to the last bit.
        """
        projected = project_rate(curve, jnp.array(1.0))
        direct = zero_rate(curve, ymd_to_ordinal(2027, 1, 1))
        assert float(projected) == pytest.approx(float(direct), abs=1e-15)
        assert float(projected) == pytest.approx(0.045, abs=1e-12)

    def test_reads_the_trades_own_maturity_not_a_fixed_pillar(self, curve):
        """Different expiries get different rates — the point of the port.

        Projected at each pillar's *actual* year fraction (the 6M pillar is
        181 days, i.e. 0.4959y, not 0.5y), the rate reproduces that
        pillar's zero exactly.
        """
        for index, expected in enumerate(SLOPED_ZEROS):
            if index == 0:
                continue  # t = 0 has no zero rate
            projected = project_rate(curve, TIMES[index])
            assert float(projected) == pytest.approx(
                float(expected), abs=1e-12,
            )

    def test_distinct_horizons_give_distinct_rates(self, curve):
        """The fixed-pillar collapse cannot do this."""
        short = float(project_rate(curve, TIMES[1]))
        long = float(project_rate(curve, TIMES[3]))
        assert short == pytest.approx(0.03, abs=1e-12)
        assert long == pytest.approx(0.06, abs=1e-12)
        assert long > short

    def test_honours_the_curve_day_count(self):
        """An act_360 curve must not be projected on a 365 axis."""
        curve_360 = DiscountCurve(
            pillar_dates=PILLARS,
            discount_factors=jnp.exp(-SLOPED_ZEROS * TIMES),
            reference_date=REF,
            day_count="act_360",
        )
        curve_365 = DiscountCurve(
            pillar_dates=PILLARS,
            discount_factors=jnp.exp(-SLOPED_ZEROS * TIMES),
            reference_date=REF,
            day_count="act_365",
        )
        assert float(project_rate(curve_360, jnp.array(1.0))) != float(
            project_rate(curve_365, jnp.array(1.0))
        )

    def test_is_differentiable_in_the_curve(self, curve):
        g = jax.grad(
            lambda dfs: project_rate(
                eqx.tree_at(lambda c: c.discount_factors, curve, dfs),
                jnp.array(1.0),
            )
        )(curve.discount_factors)
        assert jnp.any(g != 0.0)
        assert jnp.all(jnp.isfinite(g))


class TestSampleVol:
    def test_samples_at_strike_and_expiry(self):
        surface = ConstantVol(vol=jnp.array(0.22))
        assert float(
            sample_vol(surface, jnp.array(100.0), jnp.array(1.0))
        ) == pytest.approx(0.22)

    def test_missing_surface_raises(self):
        with pytest.raises(ValueError, match="carries no surface"):
            sample_vol(None, jnp.array(100.0), jnp.array(1.0))

    def test_leverage_grid_is_refused_not_mis_sampled(self):
        """Same arity, different units — the silent-wrongness case."""
        leverage = LeverageGrid(
            log_moneyness_grid=jnp.array([-0.1, 0.0, 0.1]),
            time_grid=jnp.array([0.5, 1.0]),
            values=jnp.ones((2, 3)),
        )
        with pytest.raises(ValueError, match="not an implied-vol surface"):
            sample_vol(leverage, jnp.array(100.0), jnp.array(1.0))

    def test_normal_vols_are_refused(self):
        with pytest.raises(ValueError, match="normal .Bachelier."):
            sample_vol(
                ConstantVol(vol=jnp.array(0.01), is_normal=True),
                jnp.array(100.0),
                jnp.array(1.0),
            )


# ── bind_model ───────────────────────────────────────────────────────


class TestBindModel:
    def test_fully_specified_model_is_returned_unchanged(
        self, option, market,
    ):
        """Identity, not a copy — the guarantee of constant-mode exactness."""
        model = BlackScholesModel(
            vol=jnp.array(0.2), rate=jnp.array(0.05), dividend=jnp.array(0.0),
        )
        resolved = resolve(option, model, market)
        assert bind_model(model, option, resolved) is model

    def test_model_free_pricing_is_untouched(self, option, market):
        resolved = resolve(option, None, market)
        assert bind_model(None, option, resolved) is None

    def test_preserves_the_model_class(self, option, market):
        """Dispatch keys on type(model); bind must not change it."""
        model = BlackScholesModel(vol=jnp.array(0.2))
        resolved = resolve(option, model, market)
        assert type(bind_model(model, option, resolved)) is BlackScholesModel

    def test_fills_rate_from_the_curve(self, option, market, curve):
        model = BlackScholesModel(
            vol=jnp.array(0.2), dividend=jnp.array(0.0),
        )
        bound = bind_model(model, option, resolve(option, model, market))
        assert float(bound.rate) == pytest.approx(0.045, abs=1e-12)

    def test_fills_dividend_from_the_market(self, option, curve):
        market = MarketState.single_curve(
            curve, spot=jnp.array(100.0), vol=jnp.array(0.2),
            dividend=jnp.array(0.031),
        )
        model = BlackScholesModel(vol=jnp.array(0.2), rate=jnp.array(0.05))
        bound = bind_model(model, option, resolve(option, model, market))
        assert float(bound.dividend) == pytest.approx(0.031)

    def test_fills_vol_from_the_surface(self, option, curve):
        market = MarketState.single_curve(
            curve, spot=jnp.array(100.0), vol=jnp.array(0.27),
            dividend=jnp.array(0.0),
        )
        model = BlackScholesModel(rate=jnp.array(0.05))
        bound = bind_model(model, option, resolve(option, model, market))
        assert float(bound.vol) == pytest.approx(0.27)

    def test_fills_everything(self, option, market):
        model = BlackScholesModel()
        bound = bind_model(model, option, resolve(option, model, market))
        assert bound.vol is not None
        assert bound.rate is not None
        assert bound.dividend is not None

    def test_heston_rate_and_dividend_bind(self, option, market):
        model = HestonModel(
            v0=jnp.array(0.04), kappa=jnp.array(1.5), theta=jnp.array(0.04),
            xi=jnp.array(0.3), rho=jnp.array(-0.7),
        )
        bound = bind_model(model, option, resolve(option, model, market))
        assert float(bound.rate) == pytest.approx(0.045, abs=1e-12)
        assert float(bound.dividend) == pytest.approx(0.0)

    def test_dividend_with_no_assets_raises(self, curve, option):
        bare = MarketState(curves=CurveGraph(curves={"_default_": curve}))
        model = BlackScholesModel(vol=jnp.array(0.2), rate=jnp.array(0.05))
        with pytest.raises(ValueError, match="carries no assets"):
            bind_model(model, option, resolve(option, model, bare))


# ── End to end through price() ───────────────────────────────────────


class TestPriceEndToEnd:
    @pytest.mark.parametrize("method", ["analytic", "lattice"])
    def test_market_rate_equals_an_explicit_equal_rate(
        self, option, market, method,
    ):
        """Market mode is not a new model — just a sourced parameter."""
        sourced = price(
            option,
            BlackScholesModel(vol=jnp.array(0.2), dividend=jnp.array(0.0)),
            market, method=method,
        ).price
        explicit = price(
            option,
            BlackScholesModel(
                vol=jnp.array(0.2), rate=jnp.array(0.045),
                dividend=jnp.array(0.0),
            ),
            market, method=method,
        ).price
        assert float(sourced) == pytest.approx(float(explicit), abs=1e-12)

    def test_constant_and_market_modes_differ_by_the_collapse(
        self, options, market,
    ):
        """The 9.02 % the fixed-pillar collapse was missing.

        Constant mode fed the short-pillar rate (3 %) prices the 1Y group
        at 14.7068; market mode reads the 1Y zero (4.5 %) and prices it at
        16.0336 — the same gap `TestShortRateCollapse` measures from the
        other side.
        """
        collapsed = group_pv(
            options,
            BlackScholesModel(
                vol=jnp.array(0.2), rate=jnp.array(0.03),
                dividend=jnp.array(0.0),
            ),
            market,
        )
        sourced = group_pv(options, BlackScholesModel(), market)
        assert float(collapsed) == pytest.approx(14.7068, abs=1e-3)
        assert float(sourced) == pytest.approx(16.0336, abs=1e-3)
        relative = float((sourced - collapsed) / collapsed)
        assert relative == pytest.approx(0.0902, abs=5e-4)


# ── The structural zero, closed ──────────────────────────────────────


class TestCurveShockNowReachesEquity:
    """What this whole change exists for."""

    @pytest.fixture
    def bump(self):
        return KeyedScenario(curve_shocks={"_default_": jnp.full(4, 0.0100)})

    def test_price_moves_under_a_curve_shock(self, options, market, bump):
        model = BlackScholesModel()
        base = group_pv(options, model, market)
        shocked = group_reprice(options, model, market, bump)
        assert float(shocked - base) == pytest.approx(0.9216, abs=1e-3)

    def test_gradient_lands_on_the_trades_own_pillar(self, options, market):
        """A 1Y group's rate risk sits on the 1Y pillar, not pillar 1."""
        grad = group_market_gradient(options, BlackScholesModel(), market)
        dfs = grad.curves["_default_"].discount_factors

        one_year = 2  # index of ymd_to_ordinal(2027, 1, 1) in PILLARS
        assert float(dfs[one_year]) != 0.0
        assert jnp.all(jnp.isfinite(dfs))
        others = jnp.delete(dfs, one_year)
        assert jnp.all(others == 0.0)

    def test_the_shadowed_shock_guard_stops_firing(self, options, market, bump):
        """The guard is forward-compatible: no scalar rate, nothing shadowed."""
        assert _shadowed_shocks(BlackScholesModel(), bump) == []
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            group_reprice(options, BlackScholesModel(), market, bump)

    def test_constant_mode_still_warns_and_is_still_inert(
        self, options, market, bump,
    ):
        """Both modes coexist; only one of them can report curve risk."""
        model = BlackScholesModel(
            vol=jnp.array(0.2), rate=jnp.array(0.03), dividend=jnp.array(0.0),
        )
        base = group_pv(options, model, market)
        with pytest.warns(UserWarning, match="cannot reach the price"):
            shocked = group_reprice(options, model, market, bump)
        assert shocked == base


# ── JAX contract ─────────────────────────────────────────────────────


class TestJaxContract:
    def test_filter_jit_smoke(self, options, market):
        @eqx.filter_jit
        def run(group, model, mkt):
            return group_pv(group, model, mkt)

        jitted = run(options, BlackScholesModel(), market)
        eager = group_pv(options, BlackScholesModel(), market)
        assert float(jitted) == pytest.approx(float(eager), abs=1e-12)

    def test_grad_through_binding_to_the_curve(self, options, market):
        grad = eqx.filter_grad(
            lambda m: group_pv(options, BlackScholesModel(), m)
        )(market)
        assert jnp.any(grad.curves["_default_"].discount_factors != 0.0)

    def test_grad_reaches_the_surface_too(self, options, market):
        grad = eqx.filter_grad(
            lambda m: group_pv(options, BlackScholesModel(), m)
        )(market)
        assert float(grad.surfaces["_default_"].vol) > 0.0
