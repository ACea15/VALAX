r"""QuantLib cross-validation for the Cheyette PDE via the beta = 0 reduction.

There is no QuantLib Cheyette engine — the model's skew has no QuantLib
counterpart. But at zero skew (:math:`\beta = 0`) the Cheyette model is exactly
Gaussian: one factor reduces to Hull-White and two factors to G2++. So the
Cheyette **finite-difference** pricers can be validated end-to-end against
QuantLib's Gaussian engines by pricing at :math:`\beta = 0`:

- **One factor** vs ``ql.JamshidianSwaptionEngine`` (exact) and
  ``ql.FdHullWhiteSwaptionEngine`` (independent FD).
- **Two factors** vs ``ql.FdG2SwaptionEngine`` (independent 2-D FD).

This is the same reduction the internal ``tests/test_pde/test_cheyette.py``
checks against the VALAX Hull-White / G2++ PDEs; here the chain is closed all the
way to QuantLib. Conventions match ``_ql_adapters.py`` (evaluation date
2026-01-01, ``Actual365Fixed``, ``NullCalendar``, integer-day schedules).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import QuantLib as ql

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Swaption
from valax.models.cheyette import CheyetteModel, CheyetteModel2F
from valax.pricing.analytic.swaptions import _annuity
from valax.pricing.pde import PDEConfig, PDEConfig2D, Scheme, pde_price_dispatch

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    ql_flat_curve,
    reset_evaluation_date,
)

_REF_DATE_ORD = int(ymd_to_ordinal(2026, 1, 1))
_ACT365 = "act_365"
_NOTIONAL = 1_000_000.0

# 1-D reduction is an exact model match, so use a fine mesh and a tight gate.
_CONFIG_1F = PDEConfig(n_spot=401, n_time=400, spot_range=6.0)
# 2-D reduction: both sides are FD on finite grids (QL good to ~1%).
_CONFIG_2F = PDEConfig2D(
    n_x=121, n_y=121, n_time=100, x_range=6.0, scheme=Scheme.CRAIG_SNEYD
)
_QL_FD_NODES = 100
_QL_FD_STEPS = 400
_FD_RTOL_2F = 1.5e-2


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


def _valax_swaption(rate, expiry_y, tenor_y, strike_off, is_payer, curve):
    expiry_ord = _REF_DATE_ORD + expiry_y * 365
    fixed_ords = jnp.array(
        [expiry_ord + int(round(k * 365)) for k in range(1, tenor_y + 1)],
        dtype=jnp.int32,
    )
    ann = float(_annuity(jnp.int32(expiry_ord), fixed_ords, curve, _ACT365))
    fwd = (float(curve(jnp.int32(expiry_ord))) - float(curve(fixed_ords[-1]))) / ann
    strike = fwd + strike_off
    sw = Swaption(
        expiry_date=jnp.int32(expiry_ord),
        fixed_dates=fixed_ords,
        strike=jnp.asarray(strike),
        notional=jnp.asarray(_NOTIONAL),
        is_payer=is_payer,
        day_count=_ACT365,
    )
    return sw, strike


def _ql_swaption(rate, expiry_y, tenor_y, strike, is_payer, ql_disc):
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
    return ql.Swaption(swap, ql.EuropeanExercise(expiry_ql))


def _relative(v, q):
    return abs(v - q) / max(abs(q), 1.0)


# ── One factor: Cheyette(beta=0) == Hull-White ───────────────────────

# (rate, kappa, sigma, expiry_y, tenor_y, strike_offset, is_payer)
_HW_CASES = [
    (0.04, 0.10, 0.010, 1, 5, 0.000, True),
    (0.04, 0.10, 0.010, 1, 5, 0.000, False),
    (0.03, 0.20, 0.008, 3, 7, 0.000, True),
    (0.05, 0.05, 0.012, 2, 8, 0.005, True),
]
_HW_IDS = [f"hw{i}" for i in range(len(_HW_CASES))]


@pytest.fixture(params=_HW_CASES, ids=_HW_IDS)
def hw_setup(request):
    rate, kappa, sigma, expiry_y, tenor_y, strike_off, is_payer = request.param
    reset_evaluation_date()
    curve = _flat_valax_curve(rate, expiry_y + tenor_y)
    sw, strike = _valax_swaption(rate, expiry_y, tenor_y, strike_off, is_payer, curve)
    model = CheyetteModel(
        mean_reversion=jnp.asarray(kappa),
        volatility=jnp.asarray(sigma),
        skew=jnp.asarray(0.0),
        initial_curve=curve,
    )
    ql_disc = ql_flat_curve(rate)
    return {
        "valax_sw": sw,
        "model": model,
        "ql_swaption": _ql_swaption(rate, expiry_y, tenor_y, strike, is_payer, ql_disc),
        "ql_hw": ql.HullWhite(ql_disc, float(kappa), float(sigma)),
        "ql_disc": ql_disc,
    }


class TestOneFactorReductionQL:
    def test_matches_ql_jamshidian(self, hw_setup):
        v = float(pde_price_dispatch(hw_setup["valax_sw"], hw_setup["model"], _CONFIG_1F).price)
        hw_setup["ql_swaption"].setPricingEngine(
            ql.JamshidianSwaptionEngine(hw_setup["ql_hw"], hw_setup["ql_disc"])
        )
        q = hw_setup["ql_swaption"].NPV()
        rel = _relative(v, q)
        assert rel < 5e-4, f"Cheyette(beta0) vs QL Jamshidian: {v:.4f} vs {q:.4f} rel={rel:.2e}"

    def test_matches_ql_fd_engine(self, hw_setup):
        v = float(pde_price_dispatch(hw_setup["valax_sw"], hw_setup["model"], _CONFIG_1F).price)
        hw_setup["ql_swaption"].setPricingEngine(
            ql.FdHullWhiteSwaptionEngine(hw_setup["ql_hw"], _QL_FD_NODES, _QL_FD_STEPS)
        )
        q = hw_setup["ql_swaption"].NPV()
        rel = _relative(v, q)
        assert rel < 5e-4, f"Cheyette(beta0) vs QL FD-HW: {v:.4f} vs {q:.4f} rel={rel:.2e}"


# ── Two factors: Cheyette(beta1=beta2=0) == G2++ ─────────────────────

# (rate, k1, s1, k2, s2, rho, expiry_y, tenor_y, strike_offset, is_payer)
_G2_CASES = [
    (0.03, 0.50, 0.010, 0.10, 0.008, -0.70, 3, 5, 0.000, True),
    (0.03, 0.50, 0.010, 0.10, 0.008, -0.70, 3, 5, 0.000, False),
    (0.04, 0.30, 0.012, 0.05, 0.009, -0.50, 2, 5, 0.010, True),
    (0.02, 0.60, 0.009, 0.15, 0.006, 0.30, 4, 6, -0.010, True),
]
_G2_IDS = [f"g2_{i}" for i in range(len(_G2_CASES))]


@pytest.fixture(params=_G2_CASES, ids=_G2_IDS)
def g2_setup(request):
    rate, k1, s1, k2, s2, rho, expiry_y, tenor_y, strike_off, is_payer = request.param
    reset_evaluation_date()
    curve = _flat_valax_curve(rate, expiry_y + tenor_y)
    sw, strike = _valax_swaption(rate, expiry_y, tenor_y, strike_off, is_payer, curve)
    model = CheyetteModel2F(
        mean_reversion_1=jnp.asarray(k1),
        mean_reversion_2=jnp.asarray(k2),
        volatility_1=jnp.asarray(s1),
        volatility_ratio=jnp.asarray(s2) / jnp.asarray(s1),
        skew_1=jnp.asarray(0.0),
        skew_2=jnp.asarray(0.0),
        correlation=jnp.asarray(rho),
        initial_curve=curve,
    )
    ql_disc = ql_flat_curve(rate)
    return {
        "valax_sw": sw,
        "model": model,
        "ql_swaption": _ql_swaption(rate, expiry_y, tenor_y, strike, is_payer, ql_disc),
        "ql_g2": ql.G2(ql_disc, float(k1), float(s1), float(k2), float(s2), float(rho)),
    }


class TestTwoFactorReductionQL:
    def test_matches_ql_fd_g2_engine(self, g2_setup):
        v = float(pde_price_dispatch(g2_setup["valax_sw"], g2_setup["model"], _CONFIG_2F).price)
        g2_setup["ql_swaption"].setPricingEngine(
            ql.FdG2SwaptionEngine(g2_setup["ql_g2"], 50, 100, 100)
        )
        q = g2_setup["ql_swaption"].NPV()
        rel = _relative(v, q)
        assert rel < _FD_RTOL_2F, f"Cheyette2F(beta0) vs QL FD-G2: {v:.2f} vs {q:.2f} rel={rel:.2e}"
