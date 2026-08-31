"""Tests for the G2++ two-factor Gaussian short-rate model.

Covers the exact-fit ZCB property, the Gaussian variance term ``V(t, T)``
against brute-force covariance quadrature, the one-factor (Hull-White)
reduction limit, autodiff Greeks, and JIT compatibility.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from valax.models.g2pp import (
    G2PPModel,
    g2pp_B,
    g2pp_V,
    g2pp_phi,
    g2pp_bond_price,
    g2pp_market_df,
    g2pp_instantaneous_forward,
    g2pp_factor_covariance,
    g2pp_short_rate_variance,
)
from valax.models.hull_white import HullWhiteModel, hw_alpha, hw_bond_price
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
    # Upward-sloping zero rate 2% -> 5%.
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
    return G2PPModel(
        mean_reversion_x=jnp.array(0.50),
        mean_reversion_y=jnp.array(0.10),
        volatility_x=jnp.array(0.010),
        volatility_y=jnp.array(0.008),
        correlation=jnp.array(-0.70),
        initial_curve=flat_curve,
    )


# ── B function ───────────────────────────────────────────────────────

class TestG2PPB:
    def test_b_zero_at_tau_zero(self):
        assert float(g2pp_B(jnp.array(0.1), jnp.array(0.0))) == pytest.approx(0.0, abs=1e-15)

    def test_b_known_value(self):
        expected = (1.0 - float(jnp.exp(-0.5))) / 0.1
        assert float(g2pp_B(jnp.array(0.1), jnp.array(5.0))) == pytest.approx(expected, rel=1e-10)


# ── Exact-fit ZCB ────────────────────────────────────────────────────

class TestExactFit:
    def test_zcb_recovers_flat_curve(self, model):
        """At t=0, x=y=0 the ZCB must equal P^M(0, T) exactly."""
        for T in [0.5, 1.0, 2.0, 5.0, 10.0]:
            P = g2pp_bond_price(
                model, jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(T)
            )
            PM = g2pp_market_df(model, jnp.array(T))
            assert float(P) == pytest.approx(float(PM), abs=1e-10)

    def test_zcb_recovers_steep_curve(self, ref_date):
        model = G2PPModel(
            mean_reversion_x=jnp.array(0.50),
            mean_reversion_y=jnp.array(0.10),
            volatility_x=jnp.array(0.010),
            volatility_y=jnp.array(0.008),
            correlation=jnp.array(-0.70),
            initial_curve=_steep_curve(ref_date),
        )
        for T in [0.5, 1.0, 3.0, 7.0, 12.0]:
            P = g2pp_bond_price(
                model, jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(T)
            )
            PM = g2pp_market_df(model, jnp.array(T))
            assert float(P) == pytest.approx(float(PM), abs=1e-4)


# ── Variance term V(t, T) ────────────────────────────────────────────

class TestVarianceTerm:
    def test_v_zero_at_tau_zero(self, model):
        assert float(g2pp_V(model, jnp.array(2.0), jnp.array(2.0))) == pytest.approx(0.0, abs=1e-14)

    def test_v_positive(self, model):
        assert float(g2pp_V(model, jnp.array(1.0), jnp.array(6.0))) > 0.0

    def test_v_matches_covariance_quadrature(self, model):
        """V(t, T) = Var[∫_t^T (x(u)+y(u)) du | F_t] via the OU covariance kernel.

        Independently double-integrates the analytic conditional covariance of
        the two correlated OU factors over [t, T]^2 and compares to the closed
        form ``g2pp_V``.
        """
        a = float(model.mean_reversion_x)
        b = float(model.mean_reversion_y)
        sig = float(model.volatility_x)
        eta = float(model.volatility_y)
        rho = float(model.correlation)
        t, T = 1.0, 5.0

        n = 400
        grid = jnp.linspace(t, T, n)
        u = grid[:, None]
        v = grid[None, :]

        cov_xx = (sig**2 / (2 * a)) * (
            jnp.exp(-a * jnp.abs(u - v)) - jnp.exp(-a * (u + v - 2 * t))
        )
        cov_yy = (eta**2 / (2 * b)) * (
            jnp.exp(-b * jnp.abs(u - v)) - jnp.exp(-b * (u + v - 2 * t))
        )
        # Cov(x(u), y(v)) = rho*sig*eta*e^{-a u - b v}*(e^{(a+b)min(u,v)}-e^{(a+b)t})/(a+b)
        m = jnp.minimum(u, v)
        cov_xy = rho * sig * eta * jnp.exp(-a * u - b * v) * (
            jnp.exp((a + b) * m) - jnp.exp((a + b) * t)
        ) / (a + b)
        cov_yx = rho * sig * eta * jnp.exp(-b * u - a * v) * (
            jnp.exp((a + b) * m) - jnp.exp((a + b) * t)
        ) / (a + b)

        kernel = cov_xx + cov_yy + cov_xy + cov_yx
        inner = jnp.trapezoid(kernel, grid, axis=1)
        v_quad = float(jnp.trapezoid(inner, grid))

        v_closed = float(g2pp_V(model, jnp.array(t), jnp.array(T)))
        assert v_quad == pytest.approx(v_closed, rel=1e-4)


# ── Hull-White reduction limit ───────────────────────────────────────

class TestHullWhiteReduction:
    """As the second factor vanishes (eta -> 0, rho = 0), G2++ collapses to HW-1F."""

    def test_phi_matches_hw_alpha(self, flat_curve):
        g2 = G2PPModel(
            mean_reversion_x=jnp.array(0.30),
            mean_reversion_y=jnp.array(0.10),
            volatility_x=jnp.array(0.012),
            volatility_y=jnp.array(0.0),
            correlation=jnp.array(0.0),
            initial_curve=flat_curve,
        )
        hw = HullWhiteModel(
            mean_reversion=jnp.array(0.30),
            volatility=jnp.array(0.012),
            initial_curve=flat_curve,
        )
        for t in [0.5, 2.0, 5.0, 10.0]:
            phi = float(g2pp_phi(g2, jnp.array(t)))
            alpha = float(hw_alpha(hw, jnp.array(t)))
            assert phi == pytest.approx(alpha, rel=1e-10, abs=1e-12)

    def test_zcb_matches_hw(self, flat_curve):
        g2 = G2PPModel(
            mean_reversion_x=jnp.array(0.30),
            mean_reversion_y=jnp.array(0.10),
            volatility_x=jnp.array(0.012),
            volatility_y=jnp.array(0.0),
            correlation=jnp.array(0.0),
            initial_curve=flat_curve,
        )
        hw = HullWhiteModel(
            mean_reversion=jnp.array(0.30),
            volatility=jnp.array(0.012),
            initial_curve=flat_curve,
        )
        for t, T in [(1.0, 5.0), (2.0, 10.0), (0.5, 3.0)]:
            for x in [-0.01, 0.0, 0.02]:
                P_g2 = g2pp_bond_price(
                    g2, jnp.array(x), jnp.array(0.0), jnp.array(t), jnp.array(T)
                )
                # HW is parameterised by the short rate r = x + alpha(t).
                r = jnp.array(x) + hw_alpha(hw, jnp.array(t))
                P_hw = hw_bond_price(hw, r, jnp.array(t), jnp.array(T))
                assert float(P_g2) == pytest.approx(float(P_hw), rel=1e-9)


# ── Factor covariance ────────────────────────────────────────────────

class TestFactorCovariance:
    def test_covariance_symmetric_psd(self, model):
        cov = g2pp_factor_covariance(model, jnp.array(0.25))
        assert float(cov[0, 1]) == pytest.approx(float(cov[1, 0]), rel=1e-12)
        # Positive semi-definite: eigenvalues non-negative.
        eigs = jnp.linalg.eigvalsh(cov)
        assert float(eigs.min()) > 0.0

    def test_cross_sign_follows_rho(self, flat_curve):
        pos = G2PPModel(
            mean_reversion_x=jnp.array(0.5), mean_reversion_y=jnp.array(0.1),
            volatility_x=jnp.array(0.01), volatility_y=jnp.array(0.01),
            correlation=jnp.array(0.6), initial_curve=flat_curve,
        )
        assert float(g2pp_factor_covariance(pos, jnp.array(0.5))[0, 1]) > 0.0
        neg = eqx.tree_at(lambda m: m.correlation, pos, jnp.array(-0.6))
        assert float(g2pp_factor_covariance(neg, jnp.array(0.5))[0, 1]) < 0.0

    def test_short_rate_variance_positive(self, model):
        for t in [0.25, 1.0, 5.0]:
            assert float(g2pp_short_rate_variance(model, jnp.array(t))) > 0.0


# ── Autodiff & JIT ───────────────────────────────────────────────────

class TestAutodiff:
    def test_grad_wrt_x(self, model):
        """dP/dx = -B(a, t, T) * P."""
        t, T = jnp.array(1.0), jnp.array(5.0)
        x0 = jnp.array(0.005)
        P = g2pp_bond_price(model, x0, jnp.array(0.0), t, T)
        dP = jax.grad(
            lambda x: g2pp_bond_price(model, x, jnp.array(0.0), t, T)
        )(x0)
        expected = -g2pp_B(model.mean_reversion_x, T - t) * P
        assert float(dP) == pytest.approx(float(expected), rel=1e-9)

    def test_grad_wrt_y(self, model):
        t, T = jnp.array(1.0), jnp.array(7.0)
        y0 = jnp.array(0.003)
        P = g2pp_bond_price(model, jnp.array(0.0), y0, t, T)
        dP = jax.grad(
            lambda y: g2pp_bond_price(model, jnp.array(0.0), y, t, T)
        )(y0)
        expected = -g2pp_B(model.mean_reversion_y, T - t) * P
        assert float(dP) == pytest.approx(float(expected), rel=1e-9)

    def test_differentiable_in_model_params(self, model):
        def price(m):
            return g2pp_bond_price(
                m, jnp.array(0.01), jnp.array(0.0), jnp.array(1.0), jnp.array(5.0)
            )
        grads = eqx.filter_grad(price)(model)
        assert jnp.isfinite(grads.correlation)
        assert jnp.isfinite(grads.volatility_x)


class TestJIT:
    def test_jit_compatible(self, model):
        args = (jnp.array(0.01), jnp.array(0.0), jnp.array(1.0), jnp.array(5.0))
        eager = g2pp_bond_price(model, *args)
        jitted = eqx.filter_jit(g2pp_bond_price)(model, *args)
        assert float(jitted) == pytest.approx(float(eager), rel=1e-10)

    def test_instantaneous_forward_flat(self, model):
        for t in [0.0, 1.0, 5.0, 10.0]:
            f = g2pp_instantaneous_forward(model, jnp.array(t))
            assert float(f) == pytest.approx(0.03, abs=1e-4)
