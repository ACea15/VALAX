"""Tests for the Cheyette (quasi-Gaussian) short-rate model.

Covers the exact-fit ZCB property, the affine bond reconstruction and its
autodiff Greeks, the Gaussian-limit reductions to Hull-White (one factor) and
G2++ (two factors), the local-volatility skew, and JIT compatibility.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from valax.models.cheyette import (
    CheyetteModel,
    CheyetteModel2F,
    cheyette_G,
    cheyette_market_df,
    cheyette_instantaneous_forward,
    cheyette_local_vol,
    cheyette_local_vol_2f,
    cheyette_y_gaussian,
    cheyette_y_gaussian_2f,
    cheyette_gaussian_mean,
    cheyette_gaussian_mean_2f,
    cheyette_short_rate,
    cheyette_bond_price,
    cheyette_bond_price_2f,
)
from valax.models.hull_white import HullWhiteModel, hw_alpha, hw_bond_price
from valax.models.g2pp import G2PPModel, g2pp_bond_price
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def ref_date():
    return ymd_to_ordinal(2025, 1, 1)


def _flat_curve(ref_date, rate, n_years=16):
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(n_years)],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rate * times),
        reference_date=ref_date,
    )


def _steep_curve(ref_date, n_years=16):
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(n_years)],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    zero = 0.02 + 0.03 * (1.0 - jnp.exp(-0.3 * times))
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-zero * times),
        reference_date=ref_date,
    )


@pytest.fixture
def flat_curve(ref_date):
    return _flat_curve(ref_date, 0.03)


@pytest.fixture
def model(flat_curve):
    return CheyetteModel(
        mean_reversion=jnp.array(0.10),
        volatility=jnp.array(0.010),
        skew=jnp.array(-3.0),
        initial_curve=flat_curve,
    )


@pytest.fixture
def model_2f(flat_curve):
    return CheyetteModel2F(
        mean_reversion_1=jnp.array(0.50),
        mean_reversion_2=jnp.array(0.10),
        volatility_1=jnp.array(0.010),
        volatility_ratio=jnp.array(0.008) / jnp.array(0.010),
        skew_1=jnp.array(-2.0),
        skew_2=jnp.array(1.5),
        correlation=jnp.array(-0.70),
        initial_curve=flat_curve,
    )


# ── G function ───────────────────────────────────────────────────────

class TestCheyetteG:
    def test_g_zero_at_tau_zero(self):
        assert float(cheyette_G(jnp.array(0.1), jnp.array(0.0))) == pytest.approx(0.0, abs=1e-15)

    def test_g_known_value(self):
        expected = (1.0 - float(jnp.exp(-0.5))) / 0.1
        assert float(cheyette_G(jnp.array(0.1), jnp.array(5.0))) == pytest.approx(expected, rel=1e-10)


# ── Exact-fit ZCB ────────────────────────────────────────────────────

class TestExactFit:
    def test_zcb_recovers_flat_curve_1f(self, model):
        for T in [0.5, 1.0, 2.0, 5.0, 10.0]:
            P = cheyette_bond_price(
                model, jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(T)
            )
            PM = cheyette_market_df(model, jnp.array(T))
            assert float(P) == pytest.approx(float(PM), abs=1e-10)

    def test_zcb_recovers_steep_curve_1f(self, ref_date):
        model = CheyetteModel(
            mean_reversion=jnp.array(0.10),
            volatility=jnp.array(0.010),
            skew=jnp.array(-3.0),
            initial_curve=_steep_curve(ref_date),
        )
        for T in [0.5, 1.0, 3.0, 7.0, 12.0]:
            P = cheyette_bond_price(
                model, jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(T)
            )
            PM = cheyette_market_df(model, jnp.array(T))
            assert float(P) == pytest.approx(float(PM), abs=1e-4)

    def test_zcb_recovers_flat_curve_2f(self, model_2f):
        for T in [0.5, 1.0, 5.0, 10.0]:
            P = cheyette_bond_price_2f(
                model_2f, jnp.array(0.0), jnp.array(0.0),
                jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
                jnp.array(0.0), jnp.array(T),
            )
            PM = cheyette_market_df(model_2f, jnp.array(T))
            assert float(P) == pytest.approx(float(PM), abs=1e-10)

    def test_zcb_recovers_steep_curve_2f(self, ref_date):
        model = CheyetteModel2F(
            mean_reversion_1=jnp.array(0.50), mean_reversion_2=jnp.array(0.10),
            volatility_1=jnp.array(0.010), volatility_ratio=jnp.array(0.008) / jnp.array(0.010),
            skew_1=jnp.array(-2.0), skew_2=jnp.array(1.5),
            correlation=jnp.array(-0.70), initial_curve=_steep_curve(ref_date),
        )
        for T in [0.5, 3.0, 7.0, 12.0]:
            P = cheyette_bond_price_2f(
                model, jnp.array(0.0), jnp.array(0.0),
                jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
                jnp.array(0.0), jnp.array(T),
            )
            PM = cheyette_market_df(model, jnp.array(T))
            assert float(P) == pytest.approx(float(PM), abs=1e-4)


# ── Instantaneous forward ────────────────────────────────────────────

class TestInstantaneousForward:
    def test_flat_forward(self, model):
        for t in [0.0, 1.0, 5.0, 10.0]:
            f = cheyette_instantaneous_forward(model, jnp.array(t))
            assert float(f) == pytest.approx(0.03, abs=1e-4)

    def test_short_rate_at_zero(self, model):
        r0 = cheyette_short_rate(model, jnp.array(0.0), jnp.array(0.0))
        assert float(r0) == pytest.approx(0.03, abs=1e-4)


# ── Local volatility skew ────────────────────────────────────────────

class TestLocalVol:
    def test_no_skew_is_constant(self, flat_curve):
        m = CheyetteModel(
            mean_reversion=jnp.array(0.1), volatility=jnp.array(0.012),
            skew=jnp.array(0.0), initial_curve=flat_curve,
        )
        for x in [-0.02, 0.0, 0.03]:
            assert float(cheyette_local_vol(m, jnp.array(x))) == pytest.approx(0.012, rel=1e-12)

    def test_skew_tilts_with_state(self, model):
        # Negative skew: vol falls as x rises.
        v_lo = float(cheyette_local_vol(model, jnp.array(-0.01)))
        v_hi = float(cheyette_local_vol(model, jnp.array(0.01)))
        assert v_lo > v_hi

    def test_floored_nonnegative(self, model):
        # Large positive x with negative skew would send vol negative; floored.
        assert float(cheyette_local_vol(model, jnp.array(100.0))) == 0.0

    def test_local_vol_2f(self, model_2f):
        s1, s2 = cheyette_local_vol_2f(model_2f, jnp.array(0.0), jnp.array(0.0))
        assert float(s1) == pytest.approx(0.010, rel=1e-12)
        assert float(s2) == pytest.approx(0.008, rel=1e-12)


# ── Hull-White reduction limit (1F, skew = 0) ────────────────────────

class TestHullWhiteReduction:
    """With constant volatility the Cheyette state is Gaussian and the affine
    bond collapses onto Hull-White under the state map x = x_HW + m(t)."""

    def test_zcb_matches_hw(self, flat_curve):
        ch = CheyetteModel(
            mean_reversion=jnp.array(0.30), volatility=jnp.array(0.012),
            skew=jnp.array(0.0), initial_curve=flat_curve,
        )
        hw = HullWhiteModel(
            mean_reversion=jnp.array(0.30), volatility=jnp.array(0.012),
            initial_curve=flat_curve,
        )
        for t, T in [(1.0, 5.0), (2.0, 10.0), (0.5, 3.0)]:
            m_t = cheyette_gaussian_mean(ch, jnp.array(t))
            y_t = cheyette_y_gaussian(ch, jnp.array(t))
            for x_hw in [-0.01, 0.0, 0.02]:
                P_ch = cheyette_bond_price(
                    ch, jnp.array(x_hw) + m_t, y_t, jnp.array(t), jnp.array(T)
                )
                r = jnp.array(x_hw) + hw_alpha(hw, jnp.array(t))
                P_hw = hw_bond_price(hw, r, jnp.array(t), jnp.array(T))
                assert float(P_ch) == pytest.approx(float(P_hw), rel=1e-9)

    def test_gaussian_mean_matches_hw_convexity(self, flat_curve):
        ch = CheyetteModel(
            mean_reversion=jnp.array(0.30), volatility=jnp.array(0.012),
            skew=jnp.array(0.0), initial_curve=flat_curve,
        )
        hw = HullWhiteModel(
            mean_reversion=jnp.array(0.30), volatility=jnp.array(0.012),
            initial_curve=flat_curve,
        )
        for t in [0.5, 2.0, 5.0, 10.0]:
            m_t = float(cheyette_gaussian_mean(ch, jnp.array(t)))
            conv = float(hw_alpha(hw, jnp.array(t)) - cheyette_instantaneous_forward(hw, jnp.array(t)))
            assert m_t == pytest.approx(conv, rel=1e-10, abs=1e-14)


# ── G2++ reduction limit (2F, skews = 0) ─────────────────────────────

class TestG2PPReduction:
    """With constant volatilities the two-factor Cheyette collapses onto G2++
    under the state map x_i = x_i^{G2} + m_i(t)."""

    def test_zcb_matches_g2pp(self, flat_curve):
        ch = CheyetteModel2F(
            mean_reversion_1=jnp.array(0.50), mean_reversion_2=jnp.array(0.10),
            volatility_1=jnp.array(0.010), volatility_ratio=jnp.array(0.008) / jnp.array(0.010),
            skew_1=jnp.array(0.0), skew_2=jnp.array(0.0),
            correlation=jnp.array(-0.70), initial_curve=flat_curve,
        )
        g2 = G2PPModel(
            mean_reversion_x=jnp.array(0.50), mean_reversion_y=jnp.array(0.10),
            volatility_x=jnp.array(0.010), volatility_ratio=jnp.array(0.8),
            correlation=jnp.array(-0.70), initial_curve=flat_curve,
        )
        for t, T in [(1.0, 5.0), (2.0, 10.0), (0.5, 7.0)]:
            m1, m2 = cheyette_gaussian_mean_2f(ch, jnp.array(t))
            Y = cheyette_y_gaussian_2f(ch, jnp.array(t))
            for x1, x2 in [(0.005, -0.003), (0.0, 0.0), (0.01, 0.004)]:
                P_ch = cheyette_bond_price_2f(
                    ch, jnp.array(x1) + m1, jnp.array(x2) + m2,
                    Y[0, 0], Y[1, 1], Y[0, 1], jnp.array(t), jnp.array(T),
                )
                P_g2 = g2pp_bond_price(
                    g2, jnp.array(x1), jnp.array(x2), jnp.array(t), jnp.array(T)
                )
                assert float(P_ch) == pytest.approx(float(P_g2), rel=1e-9)


# ── Gaussian accumulator vs quadrature ───────────────────────────────

class TestGaussianAccumulator:
    def test_y_gaussian_matches_ode_quadrature(self, flat_curve):
        """y(t) = ∫_0^t σ² e^{-2κ(t-s)} ds against the closed form."""
        ch = CheyetteModel(
            mean_reversion=jnp.array(0.25), volatility=jnp.array(0.011),
            skew=jnp.array(0.0), initial_curve=flat_curve,
        )
        kappa, sigma, t = 0.25, 0.011, 4.0
        n = 4000
        grid = jnp.linspace(0.0, t, n)
        integrand = sigma**2 * jnp.exp(-2.0 * kappa * (t - grid))
        y_quad = float(jnp.trapezoid(integrand, grid))
        y_closed = float(cheyette_y_gaussian(ch, jnp.array(t)))
        assert y_quad == pytest.approx(y_closed, rel=1e-5)

    def test_y_gaussian_2f_cross_matches_g2pp_covariance(self, model_2f):
        # Off-diagonal accumulator equals the G2++ cross covariance form.
        t = 3.0
        Y = cheyette_y_gaussian_2f(model_2f, jnp.array(t))
        k1, k2 = 0.5, 0.1
        s1, s2, rho = 0.010, 0.008, -0.70
        cross = rho * s1 * s2 / (k1 + k2) * (1.0 - jnp.exp(-(k1 + k2) * t))
        assert float(Y[0, 1]) == pytest.approx(float(cross), rel=1e-12)


# ── Autodiff & JIT ───────────────────────────────────────────────────

class TestAutodiff:
    def test_grad_wrt_x(self, model):
        """dP/dx = -G(t, T) * P."""
        t, T = jnp.array(1.0), jnp.array(5.0)
        x0, y0 = jnp.array(0.004), jnp.array(1e-4)
        P = cheyette_bond_price(model, x0, y0, t, T)
        dP = jax.grad(lambda x: cheyette_bond_price(model, x, y0, t, T))(x0)
        expected = -cheyette_G(model.mean_reversion, T - t) * P
        assert float(dP) == pytest.approx(float(expected), rel=1e-9)

    def test_grad_wrt_y(self, model):
        """dP/dy = -0.5 G(t, T)^2 * P."""
        t, T = jnp.array(1.0), jnp.array(7.0)
        x0, y0 = jnp.array(0.003), jnp.array(2e-4)
        P = cheyette_bond_price(model, x0, y0, t, T)
        dP = jax.grad(lambda y: cheyette_bond_price(model, x0, y, t, T))(y0)
        G = cheyette_G(model.mean_reversion, T - t)
        expected = -0.5 * G**2 * P
        assert float(dP) == pytest.approx(float(expected), rel=1e-9)

    def test_differentiable_in_model_params(self, model):
        def price(m):
            return cheyette_bond_price(
                m, jnp.array(0.01), jnp.array(1e-4), jnp.array(1.0), jnp.array(5.0)
            )
        grads = eqx.filter_grad(price)(model)
        assert jnp.isfinite(grads.mean_reversion)
        assert jnp.isfinite(grads.volatility)

    def test_differentiable_in_model_params_2f(self, model_2f):
        def price(m):
            return cheyette_bond_price_2f(
                m, jnp.array(0.005), jnp.array(-0.002),
                jnp.array(1e-4), jnp.array(1e-4), jnp.array(-1e-5),
                jnp.array(1.0), jnp.array(5.0),
            )
        grads = eqx.filter_grad(price)(model_2f)
        assert jnp.isfinite(grads.correlation)
        assert jnp.isfinite(grads.mean_reversion_1)


class TestJIT:
    def test_jit_compatible_1f(self, model):
        args = (jnp.array(0.01), jnp.array(1e-4), jnp.array(1.0), jnp.array(5.0))
        eager = cheyette_bond_price(model, *args)
        jitted = eqx.filter_jit(cheyette_bond_price)(model, *args)
        assert float(jitted) == pytest.approx(float(eager), rel=1e-10)

    def test_jit_compatible_2f(self, model_2f):
        args = (
            jnp.array(0.005), jnp.array(-0.002),
            jnp.array(1e-4), jnp.array(1e-4), jnp.array(-1e-5),
            jnp.array(1.0), jnp.array(5.0),
        )
        eager = cheyette_bond_price_2f(model_2f, *args)
        jitted = eqx.filter_jit(cheyette_bond_price_2f)(model_2f, *args)
        assert float(jitted) == pytest.approx(float(eager), rel=1e-10)
