"""QuantLib cross-validation of Hull-White with piecewise-constant σ(t).

QuantLib's ``Gsr`` is the Gaussian short-rate model with piecewise-constant
volatility on step dates and a constant reversion — the same model as a
multi-bucket :class:`~valax.models.hull_white.HullWhiteModel`. It is an
independent implementation, so it validates the σ(t) closed form
(:func:`~valax.pricing.analytic.hull_white_swaptions.hw_swaption_price`) end
to end.

**Which QuantLib engine.** ``Gaussian1dJamshidianSwaptionEngine`` is *not* a
tight oracle: its zero-bond options come from ``Gaussian1dModel``'s numerical
integration on a fixed grid, and on a constant σ it already differs from
QuantLib's own exact ``JamshidianSwaptionEngine`` by 1e-4 – 5e-4. The oracle
used here is ``Gaussian1dSwaptionEngine`` with a dense integration
(:data:`_POINTS` points over :data:`_STDDEVS` standard deviations), whose own
error on the constant case is measured by :class:`TestOracleAccuracy` against
the exact engine (≈ 2e-6) and sets :data:`_RTOL`.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import QuantLib as ql

from valax.core.binding import bind_curves
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Swaption
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    ql_flat_curve,
    reset_evaluation_date,
)

_REF = int(ymd_to_ordinal(2026, 1, 1))
_RATE = 0.035
_STEP_DAYS = (365, 730, 1095, 1460)
_VOLS = (0.012, 0.008, 0.015, 0.006, 0.010)
_POINTS, _STDDEVS = 2048, 12.0
#: Oracle budget: the dense Gaussian1d engine's measured error on the
#: constant case is 1.8e-6 – 2.5e-6 (TestOracleAccuracy); the σ(t) agreement
#: (measured ≤ 1.6e-6) is held to the same budget.
_RTOL = 5e-6

# (mean reversion, expiry days, n annual periods, strike offset, payer)
_CASES = [
    (0.05, 730, 5, 0.0, True),     # expiry on a breakpoint
    (0.05, 548, 5, 0.0, True),     # expiry mid-bucket
    (0.05, 1095, 4, 0.01, False),  # receiver, out of the money
    (0.05, 1825, 5, -0.005, True),  # expiry past the last breakpoint
    (0.05, 200, 3, 0.0, False),    # first bucket only
    (0.20, 730, 5, 0.0, True),
    (0.20, 548, 5, 0.0, True),
    (0.20, 1825, 5, -0.005, True),
]


def _valax_curve() -> DiscountCurve:
    years = [0.0] + [float(k) for k in range(1, 20)]
    pillars = jnp.array([_REF + int(round(y * 365)) for y in years], dtype=jnp.int32)
    t = (pillars - _REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-_RATE * t),
        reference_date=jnp.int32(_REF),
        day_count="act_365",
    )


def _gsr(disc, a: float, vols=_VOLS, step_days=_STEP_DAYS) -> ql.Gsr:
    steps = [DEFAULT_QL_DATE + ql.Period(d, ql.Days) for d in step_days]
    return ql.Gsr(
        disc, steps,
        [ql.QuoteHandle(ql.SimpleQuote(v)) for v in vols],
        [ql.QuoteHandle(ql.SimpleQuote(a))], 60.0,
    )


def _ql_swaption(disc, exp_days: int, n: int, strike: float, payer: bool) -> ql.Swaption:
    expiry = DEFAULT_QL_DATE + ql.Period(exp_days, ql.Days)
    fixed = [expiry + ql.Period(365 * k, ql.Days) for k in range(1, n + 1)]
    schedule = ql.Schedule(ql.DateVector([expiry] + fixed))
    index = ql.IborIndex(
        "Flat", ql.Period(365, ql.Days), 0, ql.EURCurrency(), ql.NullCalendar(),
        ql.Unadjusted, False, ql.Actual365Fixed(), disc,
    )
    swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer if payer else ql.VanillaSwap.Receiver, 1.0,
        schedule, strike, ql.Actual365Fixed(), schedule, index, 0.0,
        ql.Actual365Fixed(),
    )
    return ql.Swaption(swap, ql.EuropeanExercise(expiry))


def _valax_swaption(exp_days: int, n: int, strike: float, payer: bool) -> Swaption:
    expiry = _REF + exp_days
    return Swaption(
        expiry_date=jnp.int32(expiry),
        fixed_dates=jnp.array([expiry + 365 * k for k in range(1, n + 1)], dtype=jnp.int32),
        strike=jnp.asarray(strike),
        notional=jnp.asarray(1.0),
        is_payer=payer,
        day_count="act_365",
    )


def _valax_model(a: float, vols=_VOLS, step_days=_STEP_DAYS) -> HullWhiteModel:
    return HullWhiteModel(
        mean_reversion=jnp.asarray(a),
        volatility=jnp.asarray(vols),
        volatility_times=jnp.asarray([d / 365.0 for d in step_days]),
    )


@pytest.fixture(scope="module")
def disc():
    reset_evaluation_date()
    return ql_flat_curve(_RATE)


class TestOracleAccuracy:
    """The dense Gaussian1d engine against QuantLib's exact Jamshidian (σ const)."""

    @pytest.mark.parametrize("a", [0.05, 0.2])
    def test_oracle_error_on_constant_sigma(self, disc, a):
        sw = _ql_swaption(disc, 730, 5, _RATE, True)
        sw.setPricingEngine(ql.JamshidianSwaptionEngine(ql.HullWhite(disc, a, 0.012), disc))
        exact = sw.NPV()
        sw.setPricingEngine(ql.Gaussian1dSwaptionEngine(
            _gsr(disc, a, vols=(0.012,) * 5), _POINTS, _STDDEVS, True, False, disc
        ))
        # Measured 1.8e-6 (a = 0.05) and 2.5e-6 (a = 0.2): the oracle's own
        # error must sit inside the budget the σ(t) comparison is held to.
        assert abs(sw.NPV() / exact - 1.0) < _RTOL


class TestPiecewiseJamshidianVsGsr:
    @pytest.mark.parametrize("a,exp_days,n,k_off,payer", _CASES)
    def test_matches_gsr(self, disc, a, exp_days, n, k_off, payer):
        strike = _RATE + k_off
        ql_sw = _ql_swaption(disc, exp_days, n, strike, payer)
        ql_sw.setPricingEngine(
            ql.Gaussian1dSwaptionEngine(_gsr(disc, a), _POINTS, _STDDEVS, True, False, disc)
        )
        q = ql_sw.NPV()
        model = bind_curves(_valax_model(a), _valax_curve())
        v = float(hw_swaption_price(_valax_swaption(exp_days, n, strike, payer), model))
        assert abs(v / q - 1.0) < _RTOL, f"VALAX={v:.12f} QL={q:.12f}"

    def test_detects_misplaced_breakpoints_european(self, disc):
        """Anti-vacuity: buckets shifted by one year miss the oracle badly."""
        ql_sw = _ql_swaption(disc, 1095, 4, _RATE, True)
        ql_sw.setPricingEngine(
            ql.Gaussian1dSwaptionEngine(_gsr(disc, 0.05), _POINTS, _STDDEVS, True, False, disc)
        )
        q = ql_sw.NPV()
        shifted = _valax_model(0.05, step_days=tuple(d + 365 for d in _STEP_DAYS))
        v = float(hw_swaption_price(
            _valax_swaption(1095, 4, _RATE, True), bind_curves(shifted, _valax_curve())
        ))
        assert abs(v / q - 1.0) > 1e-3


_STEEP_VOLS = (0.004, 0.020, 0.004, 0.020, 0.004)


def _ql_bermudan(disc) -> ql.Swaption:
    """Payer Bermudan, exercise annually at 1..4Y into a swap ending 6Y."""
    first = DEFAULT_QL_DATE + ql.Period(365, ql.Days)
    schedule = ql.Schedule(ql.DateVector(
        [first] + [DEFAULT_QL_DATE + ql.Period(365 * k, ql.Days) for k in range(2, 7)]
    ))
    index = ql.IborIndex(
        "Flat", ql.Period(365, ql.Days), 0, ql.EURCurrency(), ql.NullCalendar(),
        ql.Unadjusted, False, ql.Actual365Fixed(), disc,
    )
    swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer, 1.0, schedule, _RATE, ql.Actual365Fixed(),
        schedule, index, 0.0, ql.Actual365Fixed(),
    )
    exercise = ql.BermudanExercise(
        [DEFAULT_QL_DATE + ql.Period(365 * k, ql.Days) for k in (1, 2, 3, 4)]
    )
    return ql.Swaption(swap, exercise)


class TestBermudanTriangulation:
    """σ(t) Bermudan: VALAX tree, VALAX PDE and QuantLib ``Gsr`` agree.

    Three independent engines. Measured (relative to QuantLib, whose own
    error is ~2.5e-6): tree Richardson extrapolate of 400 / 800 steps
    -4.5e-6 (smooth σ(t)) and -1.7e-5 (5× steep); PDE on 401×240
    +1.1e-5 and +4.3e-6.
    """

    @pytest.mark.parametrize("vols", [_VOLS, _STEEP_VOLS], ids=["smooth", "steep"])
    def test_tree_pde_and_quantlib_agree(self, disc, vols):
        from valax.core.market_state import MarketState
        from valax.core.pricing import price
        from valax.instruments.rates import BermudanSwaption
        from valax.pricing.lattice import bermudan_swaption_tree_price
        from valax.pricing.pde import PDEConfig

        ql_sw = _ql_bermudan(disc)
        ql_sw.setPricingEngine(ql.Gaussian1dSwaptionEngine(
            _gsr(disc, 0.05, vols=vols), _POINTS, _STDDEVS, True, False, disc
        ))
        q = ql_sw.NPV()

        berm = BermudanSwaption(
            exercise_dates=jnp.array([_REF + 365 * k for k in (1, 2, 3, 4)], dtype=jnp.int32),
            fixed_dates=jnp.array([_REF + 365 * k for k in range(2, 7)], dtype=jnp.int32),
            strike=jnp.asarray(_RATE), notional=jnp.asarray(1.0), day_count="act_365",
        )
        free = _valax_model(0.05, vols=vols)
        bound = bind_curves(free, _valax_curve())
        v400 = float(bermudan_swaption_tree_price(berm, bound, n_steps=400))
        v800 = float(bermudan_swaption_tree_price(berm, bound, n_steps=800))
        tree = 2.0 * v800 - v400
        pde = float(price(
            berm, free, MarketState.single_curve(_valax_curve()), method="pde",
            config=PDEConfig(n_spot=401, n_time=240),
        ).price)
        assert abs(tree / q - 1.0) < 3e-5, f"tree={tree:.10f} QL={q:.10f}"
        assert abs(pde / q - 1.0) < 2e-5, f"pde={pde:.10f} QL={q:.10f}"
        assert abs(pde / tree - 1.0) < 3e-5
