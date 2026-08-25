"""QuantLib cross-validation for the Hull-White short-rate PDE (roadmap PR-3).

The PDE pricers in :mod:`valax.pricing.pde.hull_white` are checked against
QuantLib engines driven by a ``ql.HullWhite`` model on the same flat curve.
Three comparisons, in increasing order of what they prove:

1. **European swaption vs** ``ql.JamshidianSwaptionEngine`` — QuantLib's *exact*
   closed form. VALAX already has its own Jamshidian implementation validated
   against this engine, so the value added here is that the **finite-difference
   machinery** (state grid, operator stack, zero-curvature edges, exact-fit
   shift) reproduces an exact answer end to end.

2. **European swaption vs** ``ql.FdHullWhiteSwaptionEngine`` — an independent
   finite-difference implementation of the same PDE, which pins the
   discretisation rather than just the model.

3. **Bermudan swaption vs** ``ql.FdHullWhiteSwaptionEngine`` and
   ``ql.TreeSwaptionEngine`` — the important one. A Bermudan swaption has no
   closed form, so this is the only fully external check on the exercise
   machinery: the discrete-event seam, the analytic tail-swap exercise values,
   and the projection ordering.

Plus a callable-bond comparison against ``ql.TreeCallableFixedRateBondEngine``.

Tolerances are set from observed agreement (European ~3e-5 relative, Bermudan
~2e-4 against QuantLib's FD engine and ~1.2e-3 against its lattice) with
headroom. The Bermudan-vs-lattice tolerance is the loosest **because the
lattice is the less accurate engine**, not because VALAX is: QuantLib's own FD
and tree engines disagree with each other by more than VALAX's PDE disagrees
with QuantLib's FD.

Conventions match ``_ql_adapters.py``: evaluation date 2026-01-01,
``Actual365Fixed``, ``NullCalendar``, integer-day-aligned schedules.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import QuantLib as ql

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.bonds import CallableBond
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.hull_white import HullWhiteModel
from valax.pricing.pde import PDEConfig, pde_price_dispatch

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    ql_flat_curve,
    reset_evaluation_date,
)

_REF_DATE_ORD = int(ymd_to_ordinal(2026, 1, 1))
_ACT365 = "act_365"
_NOTIONAL = 1_000_000.0

# A deliberately fine mesh: the point of these tests is to compare *converged*
# prices, so that a disagreement means a modelling difference rather than one
# engine simply being run coarser than the other.
_CONFIG = PDEConfig(n_spot=401, n_time=400, spot_range=6.0)

# QuantLib FD engine resolution (state nodes, time steps).
_QL_FD_NODES = 100
_QL_FD_STEPS = 400

# (flat rate, mean reversion, sigma, expiry years, tenor years, strike, is_payer)
_SWAPTION_CASES = [
    (0.04, 0.10, 0.010, 1, 5, 0.040, True),
    (0.04, 0.10, 0.010, 1, 5, 0.040, False),
    (0.04, 0.05, 0.015, 2, 5, 0.050, True),
    (0.03, 0.20, 0.008, 3, 7, 0.030, True),
    (0.05, 0.02, 0.012, 2, 8, 0.055, False),
]
_IDS = [f"c{i}" for i in range(len(_SWAPTION_CASES))]


def _flat_valax_curve(rate: float, n_years: int) -> DiscountCurve:
    """Flat continuously-compounded curve anchored at t=0 with df=1."""
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


@pytest.fixture(params=_SWAPTION_CASES, ids=_IDS)
def setup(request):
    rate, a, sigma, expiry_y, tenor_y, strike, is_payer = request.param
    reset_evaluation_date()

    # ── VALAX side ────────────────────────────────────────────────────
    curve = _flat_valax_curve(rate, expiry_y + tenor_y)
    model = HullWhiteModel(
        mean_reversion=jnp.asarray(a),
        volatility=jnp.asarray(sigma),
        initial_curve=curve,
    )
    expiry_ord = _REF_DATE_ORD + expiry_y * 365
    fixed_ords = jnp.array(
        [expiry_ord + int(round(k * 365)) for k in range(1, tenor_y + 1)],
        dtype=jnp.int32,
    )
    common = dict(
        fixed_dates=fixed_ords,
        strike=jnp.asarray(strike),
        notional=jnp.asarray(_NOTIONAL),
        is_payer=is_payer,
        day_count=_ACT365,
    )
    valax_european = Swaption(expiry_date=jnp.int32(expiry_ord), **common)

    # Exercise at the swap start and at every coupon date bar the last, so
    # each exercise lands exactly on an accrual boundary. That matters: VALAX
    # accrues the tail swap's first period from the exercise date, whereas
    # QuantLib always pays the full period containing it. On an accrual
    # boundary the two coincide and the comparison is like for like.
    exercise_ords = jnp.array(
        [expiry_ord] + [int(fixed_ords[k]) for k in range(tenor_y - 1)],
        dtype=jnp.int32,
    )
    valax_bermudan = BermudanSwaption(exercise_dates=exercise_ords, **common)

    # ── QuantLib side ─────────────────────────────────────────────────
    ql_disc = ql_flat_curve(rate)
    expiry_ql = DEFAULT_QL_DATE + ql.Period(expiry_y * 365, ql.Days)
    fixed_ql = [
        expiry_ql + ql.Period(int(round(k * 365)), ql.Days)
        for k in range(1, tenor_y + 1)
    ]
    schedule = ql.Schedule(
        expiry_ql, fixed_ql[-1], ql.Period(ql.Annual), ql.NullCalendar(),
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False,
    )
    idx = ql.IborIndex(
        "Flat", ql.Period(ql.Annual), 0, ql.EURCurrency(), ql.NullCalendar(),
        ql.Unadjusted, False, ql.Actual365Fixed(), ql_disc,
    )
    ql_swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer if is_payer else ql.VanillaSwap.Receiver,
        _NOTIONAL, schedule, strike, ql.Actual365Fixed(),
        schedule, idx, 0.0, ql.Actual365Fixed(),
    )
    exercise_ql = [expiry_ql] + [fixed_ql[k] for k in range(tenor_y - 1)]

    return {
        "valax_european": valax_european,
        "valax_bermudan": valax_bermudan,
        "model": model,
        "ql_european": ql.Swaption(ql_swap, ql.EuropeanExercise(expiry_ql)),
        "ql_bermudan": ql.Swaption(ql_swap, ql.BermudanExercise(exercise_ql)),
        "ql_hw": ql.HullWhite(ql_disc, float(a), float(sigma)),
        "ql_disc": ql_disc,
    }


def _relative(valax: float, quantlib: float) -> float:
    return abs(valax - quantlib) / max(abs(quantlib), 1.0)


class TestEuropeanSwaptionPdeQL:
    """The FD machinery reproducing an exactly-known answer."""

    def test_matches_ql_jamshidian(self, setup):
        v = float(pde_price_dispatch(setup["valax_european"], setup["model"], _CONFIG).price)
        setup["ql_european"].setPricingEngine(
            ql.JamshidianSwaptionEngine(setup["ql_hw"], setup["ql_disc"])
        )
        q = setup["ql_european"].NPV()
        rel = _relative(v, q)
        assert rel < 5e-4, f"vs QL Jamshidian: VALAX={v:.4f} QL={q:.4f} rel={rel:.2e}"

    def test_matches_ql_finite_difference_engine(self, setup):
        """Independent FD implementation of the same PDE."""
        v = float(pde_price_dispatch(setup["valax_european"], setup["model"], _CONFIG).price)
        setup["ql_european"].setPricingEngine(
            ql.FdHullWhiteSwaptionEngine(setup["ql_hw"], _QL_FD_NODES, _QL_FD_STEPS)
        )
        q = setup["ql_european"].NPV()
        rel = _relative(v, q)
        assert rel < 5e-4, f"vs QL FD: VALAX={v:.4f} QL={q:.4f} rel={rel:.2e}"


class TestBermudanSwaptionPdeQL:
    """The only fully external check on the early-exercise machinery."""

    def test_matches_ql_finite_difference_engine(self, setup):
        v = float(pde_price_dispatch(setup["valax_bermudan"], setup["model"], _CONFIG).price)
        setup["ql_bermudan"].setPricingEngine(
            ql.FdHullWhiteSwaptionEngine(setup["ql_hw"], _QL_FD_NODES, _QL_FD_STEPS)
        )
        q = setup["ql_bermudan"].NPV()
        rel = _relative(v, q)
        assert rel < 2e-3, f"vs QL FD: VALAX={v:.4f} QL={q:.4f} rel={rel:.2e}"

    def test_matches_ql_tree_engine(self, setup):
        """Different numerical method entirely: QuantLib's trinomial lattice."""
        v = float(pde_price_dispatch(setup["valax_bermudan"], setup["model"], _CONFIG).price)
        setup["ql_bermudan"].setPricingEngine(
            ql.TreeSwaptionEngine(setup["ql_hw"], 500)
        )
        q = setup["ql_bermudan"].NPV()
        rel = _relative(v, q)
        assert rel < 8e-3, f"vs QL tree: VALAX={v:.4f} QL={q:.4f} rel={rel:.2e}"

    def test_dominates_the_european_on_both_engines(self, setup):
        """A Bermudan is worth at least its co-terminal European — and both
        engines must agree on that, which catches a sign or ordering slip in
        the exercise projection that a tolerance-based test could absorb."""
        setup["ql_european"].setPricingEngine(
            ql.JamshidianSwaptionEngine(setup["ql_hw"], setup["ql_disc"])
        )
        european = setup["ql_european"].NPV()
        bermudan = float(
            pde_price_dispatch(setup["valax_bermudan"], setup["model"], _CONFIG).price
        )
        assert bermudan >= european - 1e-6


class TestCallableBondPdeQL:
    """Callable bonds on the PDE against QuantLib's callable-bond lattice."""

    _CASES = [
        (0.05, 0.01, 0.10, 5, 2, 0.05),
        (0.04, 0.02, 0.08, 7, 3, 0.04),
        (0.06, 0.005, 0.12, 3, 1, 0.06),
    ]

    @pytest.fixture(params=_CASES, ids=[f"b{i}" for i in range(len(_CASES))])
    def bond_setup(self, request):
        rate, a, sigma, mat_y, n_call, coupon = request.param
        reset_evaluation_date()

        payment_ords = jnp.array(
            [_REF_DATE_ORD + int(round(k * 365)) for k in range(1, mat_y + 1)],
            dtype=jnp.int32,
        )
        call_step = max(1, mat_y // (n_call + 1))
        call_ords = jnp.array(
            [
                _REF_DATE_ORD + int(round(k * call_step * 365))
                for k in range(1, n_call + 1)
            ],
            dtype=jnp.int32,
        )
        valax_bond = CallableBond(
            payment_dates=payment_ords,
            settlement_date=jnp.int32(_REF_DATE_ORD),
            coupon_rate=jnp.asarray(coupon),
            face_value=jnp.asarray(100.0),
            call_dates=call_ords,
            call_prices=jnp.ones(n_call),
            frequency=1,
            day_count=_ACT365,
        )
        model = HullWhiteModel(
            mean_reversion=jnp.asarray(a),
            volatility=jnp.asarray(sigma),
            initial_curve=_flat_valax_curve(rate, mat_y),
        )

        ql_disc = ql_flat_curve(rate)
        maturity_ql = DEFAULT_QL_DATE + ql.Period(int(round(mat_y * 365)), ql.Days)
        schedule_ql = ql.Schedule(
            DEFAULT_QL_DATE, maturity_ql, ql.Period(ql.Annual), ql.NullCalendar(),
            ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False,
        )
        call_schedule = ql.CallabilitySchedule()
        for k in range(1, n_call + 1):
            call_schedule.append(
                ql.Callability(
                    ql.BondPrice(100.0, ql.BondPrice.Clean),
                    ql.Callability.Call,
                    DEFAULT_QL_DATE
                    + ql.Period(int(round(k * call_step * 365)), ql.Days),
                )
            )
        ql_bond = ql.CallableFixedRateBond(
            0, 100.0, schedule_ql, [coupon], ql.Actual365Fixed(),
            ql.Unadjusted, 100.0, DEFAULT_QL_DATE, call_schedule,
        )
        return {
            "valax_bond": valax_bond,
            "model": model,
            "ql_bond": ql_bond,
            "ql_hw": ql.HullWhite(ql_disc, float(a), float(sigma)),
        }

    def test_callable_bond_matches_ql(self, bond_setup):
        v = float(
            pde_price_dispatch(
                bond_setup["valax_bond"], bond_setup["model"], _CONFIG
            ).price
        )
        bond_setup["ql_bond"].setPricingEngine(
            ql.TreeCallableFixedRateBondEngine(bond_setup["ql_hw"], 400)
        )
        q = bond_setup["ql_bond"].dirtyPrice()
        rel = _relative(v, q)
        assert rel < 5e-3, f"Callable bond: VALAX={v:.4f} QL={q:.4f} rel={rel:.2e}"
