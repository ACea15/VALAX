"""QuantLib cross-validation for the G2++ semi-analytic swaption pricer.

VALAX's ``g2pp_swaption_price`` is compared against ``ql.G2SwaptionEngine``
driven by a ``ql.G2`` model on the same flat curve.  Both evaluate the same
Brigo-Mercurio one-dimensional integral, so agreement should be at
integration-tolerance level -- ``rel < 1e-4``.

A second class checks against ``ql.FdG2SwaptionEngine`` (a 2-D finite-
difference solver), an *independent numerical method* and therefore a stronger
correctness statement; its tolerance is looser because of PDE discretisation.

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
from valax.models.g2pp import G2PPModel
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price
from valax.pricing.analytic.swaptions import _annuity

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    ql_flat_curve,
    reset_evaluation_date,
)

_REF_DATE_ORD = int(ymd_to_ordinal(2026, 1, 1))
_ACT365 = "act_365"
_NOTIONAL = 1_000_000.0

# (flat rate, a, sigma, b, eta, rho, expiry years, tenor years, strike offset, is_payer)
_CASES = [
    (0.03, 0.50, 0.010, 0.10, 0.008, -0.70, 1, 5, 0.000, True),
    (0.03, 0.50, 0.010, 0.10, 0.008, -0.70, 1, 5, 0.000, False),
    (0.04, 0.30, 0.012, 0.05, 0.009, -0.50, 2, 5, 0.010, True),
    (0.04, 0.30, 0.012, 0.05, 0.009, -0.50, 2, 5, -0.010, True),
    (0.03, 0.80, 0.015, 0.20, 0.007, -0.80, 3, 7, 0.000, True),
    (0.05, 0.20, 0.011, 0.02, 0.010, 0.30, 5, 10, 0.005, False),
    (0.02, 0.60, 0.009, 0.15, 0.006, -0.60, 2, 3, 0.000, True),
    (0.04, 0.40, 0.013, 0.08, 0.008, -0.40, 1, 5, 0.010, False),
]
_IDS = [f"c{i}" for i in range(len(_CASES))]


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


@pytest.fixture(params=_CASES, ids=_IDS)
def setup(request):
    rate, a, sig, b, eta, rho, expiry_y, tenor_y, strike_off, is_payer = request.param
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
    model = G2PPModel(
        mean_reversion_x=jnp.asarray(a),
        mean_reversion_y=jnp.asarray(b),
        volatility_x=jnp.asarray(sig),
        volatility_y=jnp.asarray(eta),
        correlation=jnp.asarray(rho),
        initial_curve=curve,
    )

    # ── QuantLib side ─────────────────────────────────────────────────
    ql_disc = ql_flat_curve(rate)
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
        "ql_g2": ql.G2(ql_disc, float(a), float(sig), float(b), float(eta), float(rho)),
    }


class TestG2AnalyticQL:
    """Same B-M integral on both sides -- expect integration-tolerance match."""

    def test_matches_ql_g2_engine(self, setup):
        v = float(g2pp_swaption_price(setup["valax_sw"], setup["model"]))
        setup["ql_swaption"].setPricingEngine(
            ql.G2SwaptionEngine(setup["ql_g2"], 6.0, 64)
        )
        q = setup["ql_swaption"].NPV()
        rel = abs(v - q) / max(abs(q), 1.0)
        assert rel < 1e-4, f"G2 analytic: VALAX={v:.4f}  QL={q:.4f}  rel={rel:.2e}"


class TestG2FdQL:
    """Independent numerical method: QL's 2-D finite-difference G2 engine."""

    def test_matches_ql_fd_engine(self, setup):
        v = float(g2pp_swaption_price(setup["valax_sw"], setup["model"]))
        setup["ql_swaption"].setPricingEngine(
            ql.FdG2SwaptionEngine(setup["ql_g2"], 50, 100, 100)
        )
        q = setup["ql_swaption"].NPV()
        rel = abs(v - q) / max(abs(q), 1.0)
        # Loose: QL's FD G2 engine on a 50x100x100 grid carries ~1% PDE
        # discretisation error; this only guards against gross errors.
        assert rel < 1.5e-2, f"G2 FD: VALAX={v:.4f}  QL={q:.4f}  rel={rel:.2e}"
