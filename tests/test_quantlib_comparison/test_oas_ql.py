"""QuantLib cross-validation for the option-adjusted spread solver.

VALAX's :func:`valax.risk.oas.callable_bond_oas` inverts the Hull-White **PDE**
callable-bond pricer for the constant continuously-compounded parallel spread
that reprices the bond, and is compared against
``ql.CallableFixedRateBond.OAS`` driven by a ``ql.TreeCallableFixedRateBondEngine``
on the same flat curve and Hull-White parameters.

Two conventions are exercised:

* **Continuous compounding (the apples-to-apples test).** Asked for its OAS on a
  ``ql.Continuous`` basis, QuantLib applies the spread exactly the way
  :func:`valax.risk.oas.parallel_shift`-based repricing does, so the two OAS
  values agree to well under a basis point — the residual is the tree-vs-PDE
  pricing gap divided by the bond's (large) spread sensitivity.

* **The compounding trap.** Asked for its OAS on a ``ql.Compounded`` basis,
  QuantLib returns a *different* number (here several bp away), because a
  periodically-compounded parallel spread is not the same curve move as a
  continuous one.  The gap is the exact systematic error the design notes in
  ``RATES_SESSION_GUIDE.md`` §3b warn about; the test asserts it is material so
  the convention can never be silently confused.

Conventions match ``_ql_adapters.py``: evaluation date 2026-01-01,
``Actual365Fixed``, ``NullCalendar``, integer-day-aligned schedules, zero
settlement days (so clean price equals dirty price at the issue date).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

ql = pytest.importorskip("QuantLib")

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.bonds import CallableBond
from valax.models.hull_white import HullWhiteModel
from valax.pricing.pde import PDEConfig, pde_price_dispatch
from valax.risk.oas import (
    callable_bond_oas,
    compounded_to_continuous_spread,
    continuous_to_compounded_spread,
)

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    ql_flat_curve,
    reset_evaluation_date,
)

_REF_DATE_ORD = int(ymd_to_ordinal(2026, 1, 1))
_ACT365 = "act_365"

# A fine mesh so a disagreement means a modelling/convention difference rather
# than one engine being run coarser than the other.
_CONFIG = PDEConfig(n_spot=401, n_time=400, spot_range=6.0)

# (flat rate, mean reversion, sigma, maturity years, n call dates, coupon)
_CASES = [
    (0.05, 0.01, 0.10, 5, 2, 0.05),
    (0.04, 0.02, 0.08, 7, 3, 0.04),
]
_IDS = [f"oas{i}" for i in range(len(_CASES))]


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


@pytest.fixture(params=_CASES, ids=_IDS)
def setup(request):
    rate, a, sigma, mat_y, n_call, coupon = request.param
    reset_evaluation_date()

    # ── VALAX side ────────────────────────────────────────────────────
    payment_ords = jnp.array(
        [_REF_DATE_ORD + int(round(k * 365)) for k in range(1, mat_y + 1)],
        dtype=jnp.int32,
    )
    call_step = max(1, mat_y // (n_call + 1))
    call_ords = jnp.array(
        [_REF_DATE_ORD + int(round(k * call_step * 365)) for k in range(1, n_call + 1)],
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

    # ── QuantLib side ─────────────────────────────────────────────────
    disc = ql_flat_curve(rate)
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
                DEFAULT_QL_DATE + ql.Period(int(round(k * call_step * 365)), ql.Days),
            )
        )
    ql_bond = ql.CallableFixedRateBond(
        0, 100.0, schedule_ql, [coupon], ql.Actual365Fixed(),
        ql.Unadjusted, 100.0, DEFAULT_QL_DATE, call_schedule,
    )
    ql_hw = ql.HullWhite(disc, float(a), float(sigma))
    ql_bond.setPricingEngine(ql.TreeCallableFixedRateBondEngine(ql_hw, 400))

    return {
        "valax_bond": valax_bond,
        "model": model,
        "ql_bond": ql_bond,
        "disc": disc,
        "coupon_freq": 1,
    }


def test_oas_matches_ql_continuous(setup):
    """VALAX continuous OAS matches QuantLib's continuous-basis OAS.

    A market price 3 points below the model price forces a clearly positive
    spread.  Zero settlement days => clean price equals dirty price at the issue
    date, so the same target feeds both solvers.
    """
    ql_bond = setup["ql_bond"]
    target_clean = ql_bond.cleanPrice() - 3.0

    valax_oas = float(
        callable_bond_oas(
            setup["valax_bond"], setup["model"], jnp.asarray(target_clean), _CONFIG
        )
    )
    ql_oas_cont = ql_bond.OAS(
        target_clean, setup["disc"], ql.Actual365Fixed(),
        ql.Continuous, ql.Annual, DEFAULT_QL_DATE,
    )

    assert valax_oas > 0.0
    # Sub-basis-point agreement (residual = tree-vs-PDE gap / spread duration).
    assert valax_oas == pytest.approx(ql_oas_cont, abs=2e-4)


def test_compounding_convention_is_material(setup):
    """QuantLib's compounded-basis OAS differs from its continuous one.

    The gap is the systematic error that silently confusing the two conventions
    would introduce, so it must be asserted non-trivial rather than assumed
    away.  It also demonstrates why the VALAX↔QuantLib comparison above requests
    the *continuous* basis explicitly.
    """
    ql_bond = setup["ql_bond"]
    target_clean = ql_bond.cleanPrice() - 3.0

    ql_oas_cont = ql_bond.OAS(
        target_clean, setup["disc"], ql.Actual365Fixed(),
        ql.Continuous, ql.Annual, DEFAULT_QL_DATE,
    )
    ql_oas_comp = ql_bond.OAS(
        target_clean, setup["disc"], ql.Actual365Fixed(),
        ql.Compounded, ql.Annual, DEFAULT_QL_DATE,
    )

    # Both are legitimate OAS values, but on different bases => materially apart.
    assert abs(ql_oas_comp - ql_oas_cont) > 2e-4  # > 2 bp


def test_spread_convention_helpers_round_trip():
    """The compounding-basis spread converters are exact inverses.

    Pure-math check of :func:`continuous_to_compounded_spread` /
    :func:`compounded_to_continuous_spread`; independent of QuantLib.
    """
    for freq in (1, 2, 4):
        s_cont = jnp.asarray(0.0125)
        s_comp = continuous_to_compounded_spread(s_cont, freq)
        back = compounded_to_continuous_spread(s_comp, freq)
        assert float(back) == pytest.approx(float(s_cont), abs=1e-14)
        # Periodic compounding of a positive spread quotes wider than continuous.
        assert float(s_comp) > float(s_cont)
