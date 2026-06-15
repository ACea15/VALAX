"""Cross-validation: VALAX Dupire / LV MC vs QuantLib.

Two tests:

* ``TestDupireVsQL_FlatLimit``
    Build constant-IV surfaces (σ_iv = c) in BOTH libraries.  Dupire's
    invariant guarantees ``σ_loc ≡ c`` at every (K, T).  Sweep five
    (σ, K/F, T) probes and verify both implementations return ``c`` to
    1e-6 absolute.  This pins the analytic limit and validates the
    library-to-library convention layer (forward, day-count, expiry-
    snapping).

* ``TestLVMCvsQL_FlatSurface``
    With the same flat surface, run VALAX LV MC and compare against
    QL's analytic Black-Scholes price (the analytical fixed point of
    constant local vol).  Single case at default MC sizing; 3·stderr
    gate.

Why we do NOT include a non-trivial smile cross-check
-----------------------------------------------------
``ql.BlackVarianceSurface`` interpolates *total variance* bilinearly,
whereas ``valax.surfaces.GridVolSurface`` interpolates *implied vol*
bilinearly.  On any non-flat grid the two libraries' Dupire outputs
disagree at the O(1%) level *purely* because of interpolation, not
because either's Dupire kernel is wrong.  Pinning a smile case would
test the surfaces' interpolation conventions, not Dupire itself.  The
SVI-self-consistency gate in ``test_local_vol_paths.py`` is the right
smile-Dupire validation; the QL gate stays focused on the analytic
limit where the libraries must agree exactly.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest
import QuantLib as ql

from valax.instruments.options import EuropeanOption
from valax.models.local_vol import LocalVolModel
from valax.pricing.analytic.dupire import dupire_local_vol
from valax.pricing.mc.local_vol_paths import generate_local_vol_paths
from valax.surfaces import SVIVolSurface

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    reset_evaluation_date,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _build_ql_flat_vol_surface(
    spot: float,
    rate: float,
    dividend: float,
    sigma: float,
    today: ql.Date = DEFAULT_QL_DATE,
):
    """Build a (process, local_vol_surface) pair for QL with constant IV."""
    reset_evaluation_date(today)
    dc = ql.Actual365Fixed()
    calendar = ql.NullCalendar()

    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend, dc))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, sigma, dc)
    )
    process = ql.BlackScholesMertonProcess(spot_h, q_ts, r_ts, vol_ts)
    local_vol = ql.LocalVolSurface(vol_ts, r_ts, q_ts, spot_h)
    return process, local_vol


def _build_valax_flat_svi(
    spot: float, rate: float, dividend: float, sigma: float
) -> SVIVolSurface:
    """Match the QL flat surface above with a flat (b=0) SVI surface."""
    mu = rate - dividend
    expiries = jnp.array([0.05, 0.25, 0.5, 1.0, 2.0])
    return SVIVolSurface(
        expiries=expiries,
        forwards=jnp.array(spot) * jnp.exp(mu * expiries),
        a_vec=sigma ** 2 * expiries,  # w = σ²·T at every slice
        b_vec=jnp.zeros_like(expiries),
        rho_vec=jnp.zeros_like(expiries),
        m_vec=jnp.zeros_like(expiries),
        sigma_vec=jnp.full_like(expiries, 0.1),  # ignored at b=0
    )


# ─────────────────────────────────────────────────────────────────────
# 1. Flat-surface Dupire equivalence
# ─────────────────────────────────────────────────────────────────────


class TestDupireVsQL_FlatLimit:
    """For σ_iv ≡ c, both libraries' Dupire should return c exactly."""

    @pytest.mark.parametrize("sigma,moneyness,T_days", [
        (0.15, 1.00, 90),
        (0.25, 0.90, 180),
        (0.25, 1.10, 180),
        (0.35, 1.00, 365),
        (0.40, 0.95, 730),
    ])
    def test_flat_local_vol(self, sigma, moneyness, T_days):
        spot = 100.0
        rate, dividend = 0.03, 0.01
        K = spot * moneyness
        T_years = T_days / 365.0

        # ── QL side
        today = DEFAULT_QL_DATE
        _, ql_lv = _build_ql_flat_vol_surface(spot, rate, dividend, sigma, today)
        expiry_date = today + ql.Period(T_days, ql.Days)
        ql_sigma_loc = ql_lv.localVol(expiry_date, K, True)

        # ── VALAX side
        svi = _build_valax_flat_svi(spot, rate, dividend, sigma)
        # Forward at expiry T in VALAX (deterministic-rate equity forward).
        F_T = float(jnp.array(spot) * jnp.exp((rate - dividend) * T_years))
        k = float(jnp.log(K / F_T))
        valax_sigma_loc = float(dupire_local_vol(
            svi, jnp.array(k), jnp.array(T_years)
        ))

        # Both should equal σ exactly.
        assert abs(ql_sigma_loc - sigma) < 1e-6, (
            f"QL deviates: {ql_sigma_loc:.8f} vs σ={sigma}"
        )
        assert abs(valax_sigma_loc - sigma) < 1e-10, (
            f"VALAX deviates: {valax_sigma_loc:.10f} vs σ={sigma}"
        )
        # Library-to-library agreement.
        assert abs(ql_sigma_loc - valax_sigma_loc) < 1e-6, (
            f"QL {ql_sigma_loc:.8f} vs VALAX {valax_sigma_loc:.10f}"
        )


# ─────────────────────────────────────────────────────────────────────
# 2. LV MC reprice vs QL analytic BSM (the flat-surface fixed point)
# ─────────────────────────────────────────────────────────────────────


class TestLVMCvsQL_FlatSurface:
    """Flat surface LV MC reprices QL's analytic BSM call within 3·stderr."""

    def test_atm_call_matches_ql_bsm(self):
        spot, rate, dividend, sigma = 100.0, 0.03, 0.01, 0.25
        K = 100.0
        T_days = 365
        T_years = T_days / 365.0

        # ── QL analytic BSM price
        today = DEFAULT_QL_DATE
        process, _ = _build_ql_flat_vol_surface(spot, rate, dividend, sigma, today)
        expiry_date = today + ql.Period(T_days, ql.Days)
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
        exercise = ql.EuropeanExercise(expiry_date)
        ql_option = ql.VanillaOption(payoff, exercise)
        ql_option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        ql_price = ql_option.NPV()

        # ── VALAX LV MC on the matching flat SVI surface
        svi = _build_valax_flat_svi(spot, rate, dividend, sigma)
        model = LocalVolModel.from_flat_rate(svi, rate=rate, dividend=dividend)
        key = jax.random.PRNGKey(20260101)
        paths = generate_local_vol_paths(
            model, jnp.array(spot), T_years, n_steps=200, n_paths=50_000, key=key
        )
        terminal = paths[:, -1]
        cashflow = jnp.maximum(terminal - K, 0.0)
        df = jnp.exp(-rate * T_years)
        mc_price = float(df * jnp.mean(cashflow))
        mc_se = float(df * jnp.std(cashflow) / jnp.sqrt(cashflow.shape[0]))

        n_se = abs(mc_price - ql_price) / max(mc_se, 1e-12)
        assert n_se < 3.0, (
            f"VALAX LV MC = {mc_price:.4f} ± {mc_se:.4f}, "
            f"QL BSM = {ql_price:.4f}, {n_se:.2f}·σ"
        )
