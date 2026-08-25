"""QuantLib cross-validation for the Hull-White Jamshidian swaption pricer.

VALAX's ``hw_swaption_price`` is compared against
``ql.JamshidianSwaptionEngine`` driven by a ``ql.HullWhite`` model on the same
flat curve.  Both implement the same decomposition, so agreement should be at
solver-tolerance level rather than model level -- ``rel < 1e-4``.

A second class checks against ``ql.TreeSwaptionEngine``, which is an
*independent numerical method* (trinomial lattice) rather than the same closed
form, and therefore a stronger statement about correctness. Its tolerance is
looser because of lattice discretisation.

Conventions match ``_ql_adapters.py``: evaluation date 2026-01-01,
``Actual365Fixed``, ``NullCalendar``, integer-day-aligned schedules.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import QuantLib as ql

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Swaption
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.analytic.swaptions import _annuity

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    ql_flat_curve,
    reset_evaluation_date,
)

_REF_DATE_ORD = int(ymd_to_ordinal(2026, 1, 1))
_ACT365 = "act_365"
_NOTIONAL = 1_000_000.0

# (flat rate, mean reversion, sigma, expiry years, tenor years, strike offset, is_payer)
_CASES = [
    (0.04, 0.10, 0.010, 1, 5, 0.000, True),
    (0.04, 0.10, 0.010, 1, 5, 0.000, False),
    (0.04, 0.05, 0.015, 2, 5, 0.010, True),
    (0.04, 0.05, 0.015, 2, 5, -0.010, True),
    (0.03, 0.20, 0.008, 3, 7, 0.000, True),
    (0.05, 0.02, 0.012, 5, 10, 0.005, False),
    (0.04, 0.10, 0.010, 1, 5, 0.010, False),
    (0.02, 0.30, 0.020, 2, 3, 0.000, True),
]
_IDS = [f"c{i}" for i in range(len(_CASES))]


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
    rate, a, sigma, expiry_y, tenor_y, strike_off, is_payer = request.param
    reset_evaluation_date()

    # ── VALAX side ────────────────────────────────────────────────────
    curve = _flat_valax_curve(rate, expiry_y + tenor_y)
    expiry_ord = _REF_DATE_ORD + expiry_y * 365
    fixed_ords = jnp.array(
        [expiry_ord + int(round(k * 365)) for k in range(1, tenor_y + 1)],
        dtype=jnp.int32,
    )
    ann = float(_annuity(jnp.int32(expiry_ord), fixed_ords, curve, _ACT365))
    fwd = (
        float(curve(jnp.int32(expiry_ord))) - float(curve(fixed_ords[-1]))
    ) / ann
    strike = fwd + strike_off

    valax_sw = Swaption(
        expiry_date=jnp.int32(expiry_ord),
        fixed_dates=fixed_ords,
        strike=jnp.asarray(strike),
        notional=jnp.asarray(_NOTIONAL),
        is_payer=is_payer,
        day_count=_ACT365,
    )
    model = HullWhiteModel(
        mean_reversion=jnp.asarray(a),
        volatility=jnp.asarray(sigma),
        initial_curve=curve,
    )

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
    ql_swaption = ql.Swaption(ql_swap, ql.EuropeanExercise(expiry_ql))

    return {
        "valax_sw": valax_sw,
        "model": model,
        "ql_swaption": ql_swaption,
        "ql_hw": ql.HullWhite(ql_disc, float(a), float(sigma)),
        "ql_disc": ql_disc,
    }


class TestJamshidianQL:
    """Same closed form on both sides — expect solver-tolerance agreement."""

    def test_matches_ql_jamshidian_engine(self, setup):
        v = float(hw_swaption_price(setup["valax_sw"], setup["model"]))
        setup["ql_swaption"].setPricingEngine(
            ql.JamshidianSwaptionEngine(setup["ql_hw"], setup["ql_disc"])
        )
        q = setup["ql_swaption"].NPV()
        rel = abs(v - q) / max(abs(q), 1.0)
        assert rel < 1e-4, f"Jamshidian: VALAX={v:.4f}  QL={q:.4f}  rel={rel:.2e}"


class TestTreeSwaptionQL:
    """Independent numerical method: QL's trinomial lattice."""

    def test_matches_ql_tree_engine(self, setup):
        v = float(hw_swaption_price(setup["valax_sw"], setup["model"]))
        setup["ql_swaption"].setPricingEngine(
            ql.TreeSwaptionEngine(setup["ql_hw"], 400)
        )
        q = setup["ql_swaption"].NPV()
        rel = abs(v - q) / max(abs(q), 1.0)
        assert rel < 5e-3, f"Tree: VALAX={v:.4f}  QL={q:.4f}  rel={rel:.2e}"
