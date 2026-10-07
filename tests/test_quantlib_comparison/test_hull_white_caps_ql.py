r"""Hull-White cap PDE vs QuantLib's analytic cap/floor engine.

The VALAX cap/floor finite-difference recipe (a single backward sweep with a
caplet payoff injected at each fixing level) is checked against
``ql.AnalyticCapFloorEngine`` driven by the same ``ql.HullWhite`` model on the
same flat curve — an independent, closed-form Hull-White cap pricer.

Conventions match the rest of ``test_quantlib_comparison``: evaluation date
2026-01-01, ``Actual365Fixed``, ``NullCalendar``, integer-day annual schedules,
continuous-compounding flat curve. The first accrual starts one year out so
every caplet has a strictly-future fixing (no already-fixed stub to reconcile).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import QuantLib as ql

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Cap
from valax.models.hull_white import HullWhiteModel
from valax.pricing.pde import PDEConfig, pde_price_dispatch

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    ql_annual_ibor_index,
    reset_evaluation_date,
)

_REF = int(ymd_to_ordinal(2026, 1, 1))
_ACT365 = "act_365"
_NOTIONAL = 1_000_000.0
_CONFIG = PDEConfig(n_spot=401, n_time=800, spot_range=6.0)

# (flat rate, mean reversion, sigma, strike, is_cap)
_CASES = [
    (0.03, 0.10, 0.010, 0.03, True),
    (0.03, 0.10, 0.010, 0.04, True),
    (0.03, 0.05, 0.015, 0.02, True),
    (0.03, 0.10, 0.010, 0.03, False),
]
_IDS = [f"c{i}" for i in range(len(_CASES))]


def _valax_flat_curve(rate: float, n_years: int) -> DiscountCurve:
    years = [0.0] + [float(k) for k in range(1, n_years + 2)]
    pillars = jnp.array(
        [_REF + int(round(y * 365)) for y in years], dtype=jnp.int32
    )
    times = (pillars - _REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-jnp.asarray(rate) * times),
        reference_date=jnp.int32(_REF),
        day_count=_ACT365,
    )


@pytest.fixture(params=_CASES, ids=_IDS)
def setup(request):
    rate, a, sigma, strike, is_cap = request.param
    reset_evaluation_date()
    n_years = 5  # caplets fixing at 1y, 2y, 3y, 4y; last accrual ends at 5y

    # ── VALAX ─────────────────────────────────────────────────────────
    curve = _valax_flat_curve(rate, n_years)
    model = HullWhiteModel(
        mean_reversion=jnp.asarray(a), volatility=jnp.asarray(sigma),
        initial_curve=curve,
    )
    fix = jnp.array(
        [_REF + int(round(k * 365)) for k in range(1, n_years)], dtype=jnp.int32
    )
    end = jnp.array(
        [_REF + int(round((k + 1) * 365)) for k in range(1, n_years)],
        dtype=jnp.int32,
    )
    valax_cap = Cap(
        fixing_dates=fix, start_dates=fix, end_dates=end,
        strike=jnp.asarray(strike), notional=jnp.asarray(_NOTIONAL),
        is_cap=is_cap, day_count=_ACT365,
    )

    # ── QuantLib ──────────────────────────────────────────────────────
    idx, disc = ql_annual_ibor_index(rate)
    start_ql = DEFAULT_QL_DATE + ql.Period(365, ql.Days)
    end_ql = DEFAULT_QL_DATE + ql.Period(n_years * 365, ql.Days)
    schedule = ql.Schedule(
        start_ql, end_ql, ql.Period(ql.Annual), ql.NullCalendar(),
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False,
    )
    leg = ql.IborLeg([_NOTIONAL], schedule, idx)
    ctor = ql.Cap if is_cap else ql.Floor
    ql_cap = ctor(leg, [strike])
    ql_cap.setPricingEngine(
        ql.AnalyticCapFloorEngine(ql.HullWhite(disc, float(a), float(sigma)), disc)
    )

    return {"valax_cap": valax_cap, "model": model, "ql_cap": ql_cap}


def test_cap_matches_ql_analytic(setup):
    v = float(pde_price_dispatch(setup["valax_cap"], setup["model"], _CONFIG).price)
    q = setup["ql_cap"].NPV()
    rel = abs(v - q) / max(abs(q), 1.0)
    assert rel < 5e-3, f"VALAX={v:.4f} QL={q:.4f} rel={rel:.2e}"
