"""QuantLib cross-validation for the G2++ **2-D finite-difference** pricer.

VALAX's G2++ PDE (:mod:`valax.pricing.pde.g2pp`) is compared against QuantLib's
own 2-D finite-difference G2 engine, ``ql.FdG2SwaptionEngine`` — an *independent
numerical method*, so agreement is a strong correctness statement for both the
European and the Bermudan swaption.

The tolerance is deliberately loose. Two different ADI discretisations on
finite grids never agree tightly: QL's engine on a ``50 x 100 x 100`` grid alone
carries ~1% discretisation error (see ``test_g2pp_swaptions_ql.py``), so this
class only guards against gross errors (wrong scaling, a mis-wired cross term,
a broken exercise projection). The *tight* European gate lives in
``tests/test_pde/test_g2pp.py``, which checks the PDE against the analytic
Gauss-Hermite price to a few ``e-3``.

Conventions match ``_ql_adapters.py``: evaluation date 2026-01-01,
``Actual365Fixed``, ``NullCalendar``, integer-day-aligned schedules.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import QuantLib as ql

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.g2pp import G2PPModel
from valax.pricing.analytic.swaptions import _annuity
from valax.pricing.pde import PDEConfig2D, Scheme, pde_price_dispatch

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    ql_flat_curve,
    reset_evaluation_date,
)

_REF_DATE_ORD = int(ymd_to_ordinal(2026, 1, 1))
_ACT365 = "act_365"
_NOTIONAL = 1_000_000.0

# Loose guard: both sides are finite-difference solvers on finite grids.
_FD_RTOL = 1.5e-2

# Moderate mesh — the QL oracle is only good to ~1% anyway.
_CONFIG = PDEConfig2D(
    n_x=121, n_y=121, n_time=100, x_range=6.0, scheme=Scheme.CRAIG_SNEYD
)


def _flat_valax_curve(rate: float, n_years: int) -> DiscountCurve:
    years = [0.0] + [float(k) for k in range(1, n_years + 3)]
    pillars = jnp.array(
        [_REF_DATE_ORD + int(round(y * 365)) for y in years], dtype=jnp.int32
    )
    times = (pillars - _REF_DATE_ORD).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-jnp.asarray(rate) * times),
        reference_date=jnp.int32(_REF_DATE_ORD),
        day_count=_ACT365,
    )


def _ql_swap(rate, expiry_y, tenor_y, strike, is_payer, ql_disc):
    expiry_ql = DEFAULT_QL_DATE + ql.Period(expiry_y * 365, ql.Days)
    fixed_ql = expiry_ql + ql.Period(int(round(tenor_y * 365)), ql.Days)
    schedule = ql.Schedule(
        expiry_ql, fixed_ql, ql.Period(ql.Annual), ql.NullCalendar(),
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False,
    )
    idx = ql.IborIndex(
        "Flat", ql.Period(ql.Annual), 0, ql.EURCurrency(), ql.NullCalendar(),
        ql.Unadjusted, False, ql.Actual365Fixed(), ql_disc,
    )
    swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer if is_payer else ql.VanillaSwap.Receiver,
        _NOTIONAL, schedule, strike, ql.Actual365Fixed(),
        schedule, idx, 0.0, ql.Actual365Fixed(),
    )
    return swap, schedule


# ── European ─────────────────────────────────────────────────────────

# (rate, a, sigma, b, eta, rho, expiry_y, tenor_y, strike_offset, is_payer)
_EURO_CASES = [
    (0.03, 0.50, 0.010, 0.10, 0.008, -0.70, 3, 5, 0.000, True),
    (0.03, 0.50, 0.010, 0.10, 0.008, -0.70, 3, 5, 0.000, False),
    (0.04, 0.30, 0.012, 0.05, 0.009, -0.50, 2, 5, 0.010, True),
    (0.02, 0.60, 0.009, 0.15, 0.006, 0.30, 4, 6, -0.010, True),
]
_EURO_IDS = [f"euro{i}" for i in range(len(_EURO_CASES))]


@pytest.fixture(params=_EURO_CASES, ids=_EURO_IDS)
def euro_setup(request):
    rate, a, sig, b, eta, rho, expiry_y, tenor_y, strike_off, is_payer = request.param
    reset_evaluation_date()

    curve = _flat_valax_curve(rate, expiry_y + tenor_y)
    expiry_ord = _REF_DATE_ORD + expiry_y * 365
    fixed_ords = jnp.array(
        [expiry_ord + int(round(k * 365)) for k in range(1, tenor_y + 1)],
        dtype=jnp.int32,
    )
    # ATM forward from the curve, then offset.
    ann = float(_annuity(jnp.int32(expiry_ord), fixed_ords, curve, _ACT365))
    fwd = (float(curve(jnp.int32(expiry_ord))) - float(curve(fixed_ords[-1]))) / ann
    strike = fwd + strike_off

    valax_sw = Swaption(
        expiry_date=jnp.int32(expiry_ord),
        fixed_dates=fixed_ords,
        strike=jnp.asarray(strike),
        notional=jnp.asarray(_NOTIONAL),
        is_payer=is_payer,
        day_count=_ACT365,
    )
    model = G2PPModel(
        mean_reversion_x=jnp.asarray(a),
        mean_reversion_y=jnp.asarray(b),
        volatility_x=jnp.asarray(sig),
        volatility_y=jnp.asarray(eta),
        correlation=jnp.asarray(rho),
        initial_curve=curve,
    )

    ql_disc = ql_flat_curve(rate)
    swap, _ = _ql_swap(rate, expiry_y, tenor_y, strike, is_payer, ql_disc)
    expiry_ql = DEFAULT_QL_DATE + ql.Period(expiry_y * 365, ql.Days)
    ql_swaption = ql.Swaption(swap, ql.EuropeanExercise(expiry_ql))

    return {
        "valax_sw": valax_sw,
        "model": model,
        "ql_swaption": ql_swaption,
        "ql_g2": ql.G2(ql_disc, float(a), float(sig), float(b), float(eta), float(rho)),
    }


class TestEuropeanFdQL:
    """VALAX 2-D PDE vs QL's independent 2-D finite-difference G2 engine."""

    def test_matches_ql_fd_engine(self, euro_setup):
        v = float(
            pde_price_dispatch(
                euro_setup["valax_sw"], euro_setup["model"], _CONFIG
            ).price
        )
        euro_setup["ql_swaption"].setPricingEngine(
            ql.FdG2SwaptionEngine(euro_setup["ql_g2"], 50, 100, 100)
        )
        q = euro_setup["ql_swaption"].NPV()
        rel = abs(v - q) / max(abs(q), 1.0)
        assert rel < _FD_RTOL, f"euro FD: VALAX={v:.2f} QL={q:.2f} rel={rel:.2e}"


# ── Bermudan ─────────────────────────────────────────────────────────

# (rate, a, sigma, b, eta, rho, first_ex_y, final_y, strike_offset)
_BERM_CASES = [
    (0.03, 0.50, 0.010, 0.10, 0.008, -0.70, 3, 8, 0.000),
    (0.04, 0.30, 0.012, 0.05, 0.009, -0.50, 2, 7, 0.005),
]
_BERM_IDS = [f"berm{i}" for i in range(len(_BERM_CASES))]


@pytest.fixture(params=_BERM_CASES, ids=_BERM_IDS)
def berm_setup(request):
    rate, a, sig, b, eta, rho, first_y, final_y, strike_off = request.param
    reset_evaluation_date()

    tenor = final_y - first_y
    curve = _flat_valax_curve(rate, final_y + 1)

    start_ord = _REF_DATE_ORD + first_y * 365
    # Fixed-leg payment dates D_1..D_n and exercise dates D_0..D_{n-1}.
    fixed_ords = jnp.array(
        [start_ord + int(round(k * 365)) for k in range(1, tenor + 1)],
        dtype=jnp.int32,
    )
    exercise_ords = jnp.array(
        [start_ord + int(round(k * 365)) for k in range(0, tenor)],
        dtype=jnp.int32,
    )

    # ATM co-terminal forward (swap starts at the first exercise date).
    ann = float(_annuity(jnp.int32(start_ord), fixed_ords, curve, _ACT365))
    fwd = (float(curve(jnp.int32(start_ord))) - float(curve(fixed_ords[-1]))) / ann
    strike = fwd + strike_off

    valax_berm = BermudanSwaption(
        exercise_dates=exercise_ords,
        fixed_dates=fixed_ords,
        strike=jnp.asarray(strike),
        notional=jnp.asarray(_NOTIONAL),
        is_payer=True,
        day_count=_ACT365,
    )
    model = G2PPModel(
        mean_reversion_x=jnp.asarray(a),
        mean_reversion_y=jnp.asarray(b),
        volatility_x=jnp.asarray(sig),
        volatility_y=jnp.asarray(eta),
        correlation=jnp.asarray(rho),
        initial_curve=curve,
    )

    ql_disc = ql_flat_curve(rate)
    start_ql = DEFAULT_QL_DATE + ql.Period(first_y * 365, ql.Days)
    end_ql = DEFAULT_QL_DATE + ql.Period(final_y * 365, ql.Days)
    schedule = ql.Schedule(
        start_ql, end_ql, ql.Period(ql.Annual), ql.NullCalendar(),
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False,
    )
    idx = ql.IborIndex(
        "Flat", ql.Period(ql.Annual), 0, ql.EURCurrency(), ql.NullCalendar(),
        ql.Unadjusted, False, ql.Actual365Fixed(), ql_disc,
    )
    ql_swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer, _NOTIONAL, schedule, strike, ql.Actual365Fixed(),
        schedule, idx, 0.0, ql.Actual365Fixed(),
    )
    exercise_dates_ql = list(schedule)[:-1]  # period starts D_0..D_{n-1}
    ql_berm = ql.Swaption(ql_swap, ql.BermudanExercise(exercise_dates_ql))

    return {
        "valax_berm": valax_berm,
        "model": model,
        "ql_swaption": ql_berm,
        "ql_g2": ql.G2(ql_disc, float(a), float(sig), float(b), float(eta), float(rho)),
    }


class TestBermudanFdQL:
    """VALAX Bermudan PDE vs QL's 2-D FD G2 engine — the headline exotic."""

    def test_matches_ql_fd_engine(self, berm_setup):
        v = float(
            pde_price_dispatch(
                berm_setup["valax_berm"], berm_setup["model"], _CONFIG
            ).price
        )
        berm_setup["ql_swaption"].setPricingEngine(
            ql.FdG2SwaptionEngine(berm_setup["ql_g2"], 50, 100, 100)
        )
        q = berm_setup["ql_swaption"].NPV()
        rel = abs(v - q) / max(abs(q), 1.0)
        assert rel < _FD_RTOL, f"berm FD: VALAX={v:.2f} QL={q:.2f} rel={rel:.2e}"
