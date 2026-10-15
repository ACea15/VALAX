"""G2++ with piecewise-constant σ(t) and volatility ratio r(t): model-level maths.

Oracles are independent of the implementation's closed forms:

* Gauss-Legendre quadrature of the defining integrals, split at the
  breakpoints (the integrands are smooth inside a bucket);
* the exact-fit identity
  :math:`\\int_0^T \\varphi = -\\ln P^M(0,T) + \\tfrac12 V(0,T)`;
* the tower property :math:`P^M(0,T) = E[e^{-\\int_0^t r} P(t,T \\mid x_t, y_t)]`,
  evaluated as a Gaussian expectation with quadrature moments;
* the constant case run through the general code (two equal buckets);
* the one-factor limit (ratio 0, ρ = 0) against piecewise Hull-White.

QuantLib has no time-dependent two-factor G2 model (its ``G2`` takes constant
parameters; ``Gsr`` is one-factor), so there is no external oracle here.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.models.g2pp import (
    G2PPModel,
    g2pp_B,
    g2pp_V,
    g2pp_bond_price,
    g2pp_diffusion_rates,
    g2pp_factor_covariance,
    g2pp_forward_measure_moments,
    g2pp_instantaneous_forward,
    g2pp_market_df,
    g2pp_phi,
    g2pp_phi_integral,
    g2pp_short_rate_variance,
    g2pp_state_covariance,
)
from valax.models.hull_white import HullWhiteModel, hw_alpha, hw_bond_price

A, B, RHO = 0.5, 0.1, -0.7
VOLS = (0.012, 0.008, 0.015, 0.010)
RATIOS = (0.8, 0.6, 0.9, 1.3)
TIMES = (0.8, 2.0, 3.5)
EVAL_T = (0.5, 0.8, 1.3, 2.0, 3.0, 3.5, 5.0, 9.0)

_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)


def _quad(f, lo, hi, breaks=TIMES):
    """Gauss-Legendre on [lo, hi], split at every break inside it."""
    pts = sorted({lo, hi, *[b for b in breaks if lo < b < hi]})
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        u = 0.5 * (b - a) * _GL_X + 0.5 * (b + a)
        total += 0.5 * (b - a) * np.sum(_GL_W * f(u))
    return total


def _bucket(u, values, times=TIMES):
    return np.asarray(values)[np.searchsorted(np.asarray(times), u, side="right")]


def _sig(u):
    return _bucket(u, VOLS)


def _eta(u):
    return _bucket(u, RATIOS) * _bucket(u, VOLS)


def _b(z, u, T):
    return -np.expm1(-z * (T - u)) / z


def _v_integrand(T):
    """Integrand of V(., T): sigma^2 B_a^2 + eta^2 B_b^2 + 2 rho sigma eta B_a B_b."""
    return lambda u: (
        _sig(u) ** 2 * _b(A, u, T) ** 2
        + _eta(u) ** 2 * _b(B, u, T) ** 2
        + 2 * RHO * _sig(u) * _eta(u) * _b(A, u, T) * _b(B, u, T)
    )


def _curve():
    ref = int(ymd_to_ordinal(2026, 1, 1))
    years = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0)
    zeros = (0.020, 0.021, 0.022, 0.025, 0.027, 0.031, 0.034, 0.038, 0.040)
    pillars = jnp.array([ref + int(round(y * 365)) for y in years], dtype=jnp.int32)
    times = (pillars - ref).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-jnp.asarray(zeros) * times),
        reference_date=jnp.int32(ref),
        day_count="act_365",
    ), tuple(float(t) for t in times)


CURVE, PILLAR_T = _curve()


def _model(vols=VOLS, ratios=RATIOS, times=TIMES, rho=RHO):
    return G2PPModel(
        mean_reversion_x=jnp.asarray(A),
        mean_reversion_y=jnp.asarray(B),
        volatility_x=jnp.asarray(vols),
        volatility_ratio=jnp.asarray(ratios),
        correlation=jnp.asarray(rho),
        initial_curve=CURVE,
        volatility_times=jnp.asarray(times),
    )


def _with_buckets(model, vols, ratios):
    return eqx.tree_at(
        lambda m: (m.volatility_x, m.volatility_ratio), model, (vols, ratios)
    )


# ── G2++ quantities against quadrature ───────────────────────────────


class TestAgainstQuadrature:
    @pytest.mark.parametrize(
        "t,T",
        [(0.0, 0.5), (0.0, 1.0), (0.0, 5.0), (0.3, 0.6), (0.5, 0.75),
         (1.0, 3.0), (1.9, 2.1), (2.0, 12.0), (4.0, 9.0)],
    )
    def test_variance_term(self, t, T):
        ref = _quad(_v_integrand(T), t, T)
        got = float(g2pp_V(_model(), jnp.asarray(t), jnp.asarray(T)))
        assert got == pytest.approx(ref, rel=1e-12)

    def test_variance_term_vectorised(self):
        t = jnp.asarray([0.0, 0.5, 2.0])
        T = jnp.asarray([3.0, 4.0, 9.0])
        got = np.asarray(g2pp_V(_model(), t, T))
        ref = [_quad(_v_integrand(TT), tt, TT) for tt, TT in zip(np.asarray(t), np.asarray(T))]
        np.testing.assert_allclose(got, ref, rtol=1e-12)

    @pytest.mark.parametrize("s,t", [(0.0, 0.5), (0.0, 3.0), (0.3, 1.1), (1.5, 4.2), (4.0, 9.0)])
    def test_state_covariance(self, s, t):
        cov = np.asarray(g2pp_state_covariance(_model(), jnp.asarray(s), jnp.asarray(t)))
        var_x = _quad(lambda u: _sig(u) ** 2 * np.exp(-2 * A * (t - u)), s, t)
        var_y = _quad(lambda u: _eta(u) ** 2 * np.exp(-2 * B * (t - u)), s, t)
        cov_xy = RHO * _quad(lambda u: _sig(u) * _eta(u) * np.exp(-(A + B) * (t - u)), s, t)
        np.testing.assert_allclose(cov, [[var_x, cov_xy], [cov_xy, var_y]], rtol=1e-12)

    def test_factor_covariance_is_from_origin(self):
        m = _model()
        np.testing.assert_array_equal(
            np.asarray(g2pp_factor_covariance(m, jnp.asarray(2.7))),
            np.asarray(g2pp_state_covariance(m, jnp.asarray(0.0), jnp.asarray(2.7))),
        )

    def test_short_rate_variance(self):
        m = _model()
        for t in EVAL_T:
            ref = _quad(
                lambda u: _sig(u) ** 2 * np.exp(-2 * A * (t - u))
                + _eta(u) ** 2 * np.exp(-2 * B * (t - u))
                + 2 * RHO * _sig(u) * _eta(u) * np.exp(-(A + B) * (t - u)),
                0.0, t,
            )
            got = float(g2pp_short_rate_variance(m, jnp.asarray(t)))
            assert got == pytest.approx(ref, rel=1e-12)

    @pytest.mark.parametrize("T", [0.5, 1.0, 2.0, 3.0, 4.5, 9.0])
    def test_forward_measure_moments(self, T):
        """Brigo-Mercurio T-forward moments with sigma(t), eta(t)."""
        mu_x, mu_y, sd_x, sd_y, rho_xy = (
            float(v) for v in g2pp_forward_measure_moments(_model(), jnp.asarray(T))
        )
        e = lambda z, u: np.exp(-z * (T - u))
        mu_x_ref = -_quad(
            lambda u: e(A, u) * _sig(u)
            * (_sig(u) * _b(A, u, T) + RHO * _eta(u) * _b(B, u, T)), 0.0, T,
        )
        mu_y_ref = -_quad(
            lambda u: e(B, u) * _eta(u)
            * (_eta(u) * _b(B, u, T) + RHO * _sig(u) * _b(A, u, T)), 0.0, T,
        )
        var_x = _quad(lambda u: _sig(u) ** 2 * e(2 * A, u), 0.0, T)
        var_y = _quad(lambda u: _eta(u) ** 2 * e(2 * B, u), 0.0, T)
        cov = RHO * _quad(lambda u: _sig(u) * _eta(u) * e(A + B, u), 0.0, T)
        assert mu_x == pytest.approx(mu_x_ref, rel=1e-12)
        assert mu_y == pytest.approx(mu_y_ref, rel=1e-12)
        assert sd_x == pytest.approx(np.sqrt(var_x), rel=1e-12)
        assert sd_y == pytest.approx(np.sqrt(var_y), rel=1e-12)
        assert rho_xy == pytest.approx(cov / np.sqrt(var_x * var_y), rel=1e-12)

    def test_phi_convexity(self):
        """phi - f^M = (1/2) dV(0,t)/dt, written as an integral over u."""
        m = _model()
        for t in EVAL_T:
            e = lambda z, u: np.exp(-z * (t - u))
            ref = _quad(
                lambda u: _sig(u) ** 2 * _b(A, u, t) * e(A, u)
                + _eta(u) ** 2 * _b(B, u, t) * e(B, u)
                + RHO * _sig(u) * _eta(u) * (_b(A, u, t) * e(B, u) + _b(B, u, t) * e(A, u)),
                0.0, t,
            )
            got = float(g2pp_phi(m, jnp.asarray(t)) - g2pp_instantaneous_forward(m, jnp.asarray(t)))
            assert got == pytest.approx(ref, rel=1e-12)

    def test_diffusion_rates(self):
        m = _model()
        for t0, t1 in [(0.1, 0.5), (0.5, 2.5), (1.9, 3.6), (3.0, 7.0)]:
            got = [float(v) for v in g2pp_diffusion_rates(m, jnp.asarray(t0), jnp.asarray(t1))]
            dt = t1 - t0
            ref = [
                _quad(lambda u: _sig(u) ** 2, t0, t1) / dt,
                _quad(lambda u: _eta(u) ** 2, t0, t1) / dt,
                RHO * _quad(lambda u: _sig(u) * _eta(u), t0, t1) / dt,
            ]
            np.testing.assert_allclose(got, ref, rtol=1e-13)


# ── Exact fit and the bond ───────────────────────────────────────────


class TestExactFitAndBond:
    @pytest.mark.parametrize("T", [0.6, 2.0, 3.0, 7.5, 11.0])
    def test_exact_fit_identity(self, T):
        """∫_0^T φ = -ln P(0,T) + ½ V(0,T), both sides independently."""
        m = _model()
        f = lambda u: np.asarray(g2pp_phi(m, jnp.asarray(u)))
        lhs = _quad(f, 0.0, T, breaks=TIMES + PILLAR_T)
        rhs = -float(jnp.log(g2pp_market_df(m, jnp.asarray(T)))) + 0.5 * _quad(_v_integrand(T), 0.0, T)
        assert lhs == pytest.approx(rhs, rel=1e-12)
        assert float(g2pp_phi_integral(m, jnp.asarray(T))) == pytest.approx(lhs, rel=1e-12)

    def test_zcb_recovers_curve(self):
        m = _model()
        for T in (0.5, 1.7, 4.0, 11.0):
            p = g2pp_bond_price(m, jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(T))
            assert float(p) == pytest.approx(float(g2pp_market_df(m, jnp.asarray(T))), rel=1e-14)

    @pytest.mark.parametrize("t,T", [(1.0, 4.0), (2.5, 7.0), (4.0, 10.0)])
    def test_bond_price_tower_property(self, t, T):
        """P(0,T) = E[exp(-∫_0^t r) P(t,T | x_t, y_t)] with quadrature moments.

        ``∫_0^t (x + y) + B_a(t,T) x_t + B_b(t,T) y_t`` is the stochastic
        integral of ``σ B_a(u,T) dW1 + η B_b(u,T) dW2`` over ``[0, t]``.
        """
        m = _model()
        var = _quad(_v_integrand(T), 0.0, t)
        f = lambda u: np.asarray(g2pp_phi(m, jnp.asarray(u)))
        int_phi = _quad(f, 0.0, t, breaks=TIMES + PILLAR_T)
        a_tt = float(g2pp_bond_price(m, jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(t), jnp.asarray(T)))
        expected = np.exp(-int_phi) * a_tt * np.exp(0.5 * var)
        assert expected == pytest.approx(float(g2pp_market_df(m, jnp.asarray(T))), rel=1e-12)

    def test_one_factor_limit_is_piecewise_hull_white(self):
        """Ratio 0 and ρ = 0: G2++ is Hull-White with the same σ(t)."""
        g2 = _model(ratios=(0.0,) * 4, rho=0.0)
        hw = HullWhiteModel(
            mean_reversion=jnp.asarray(A), volatility=jnp.asarray(VOLS),
            initial_curve=CURVE, volatility_times=jnp.asarray(TIMES),
        )
        t_grid = jnp.asarray(EVAL_T)
        np.testing.assert_allclose(
            np.asarray(g2pp_phi(g2, t_grid)), np.asarray(hw_alpha(hw, t_grid)), rtol=1e-12
        )
        for t, T in [(1.0, 5.0), (2.2, 10.0), (0.5, 3.0)]:
            for x in (-0.01, 0.0, 0.02):
                p_g2 = g2pp_bond_price(g2, jnp.asarray(x), jnp.asarray(0.0), jnp.asarray(t), jnp.asarray(T))
                r = jnp.asarray(x) + hw_alpha(hw, jnp.asarray(t))
                p_hw = hw_bond_price(hw, r, jnp.asarray(t), jnp.asarray(T))
                assert float(p_g2) == pytest.approx(float(p_hw), rel=1e-12)


# ── Constant case through the general path ───────────────────────────


class TestConstantThroughGeneralPath:
    """Two equal buckets run the general code and must equal the closed form."""

    def test_equal_buckets_equal_constant(self):
        const = _model(vols=(0.011,), ratios=(0.7,), times=())
        split = _model(vols=(0.011, 0.011), ratios=(0.7, 0.7), times=(1.7,))
        t = jnp.asarray(EVAL_T)
        zero = jnp.asarray(0.0)
        pairs = [
            (g2pp_phi(const, t), g2pp_phi(split, t)),
            (g2pp_phi_integral(const, t), g2pp_phi_integral(split, t)),
            (g2pp_short_rate_variance(const, t), g2pp_short_rate_variance(split, t)),
            (g2pp_state_covariance(const, t * 0.5, t), g2pp_state_covariance(split, t * 0.5, t)),
            (g2pp_factor_covariance(const, jnp.asarray(2.3)), g2pp_factor_covariance(split, jnp.asarray(2.3))),
            (jnp.stack(g2pp_diffusion_rates(const, t * 0.5, t)), jnp.stack(g2pp_diffusion_rates(split, t * 0.5, t))),
            (
                g2pp_bond_price(const, jnp.asarray(0.01), jnp.asarray(-0.004), jnp.asarray(2.5), jnp.asarray(8.0)),
                g2pp_bond_price(split, jnp.asarray(0.01), jnp.asarray(-0.004), jnp.asarray(2.5), jnp.asarray(8.0)),
            ),
            (
                g2pp_bond_price(const, zero, zero, jnp.asarray(0.5), jnp.asarray(0.75)),
                g2pp_bond_price(split, zero, zero, jnp.asarray(0.5), jnp.asarray(0.75)),
            ),
        ]
        for c, s in pairs:
            np.testing.assert_allclose(np.asarray(s), np.asarray(c), rtol=1e-13)
        for T in (0.5, 1.7, 3.0, 9.0):
            # mu_x, mu_y are differences of O((sigma^2 + eta^2) T^2) terms
            # (mu_y nearly cancels at T = 0.5 with rho = -0.7): 1e-13 of the
            # term scale, rtol 1e-13 on everything else.
            np.testing.assert_allclose(
                np.asarray(jnp.stack(g2pp_forward_measure_moments(split, jnp.asarray(T)))),
                np.asarray(jnp.stack(g2pp_forward_measure_moments(const, jnp.asarray(T)))),
                rtol=1e-13, atol=1e-13 * (0.011**2 + 0.0077**2) * T**2,
            )

    def test_variance_term_equal_buckets(self):
        """V's constant closed form cancels at short windows (legacy); compare
        both against quadrature there, and with each other at 1e-13 beyond."""
        const = _model(vols=(0.011,), ratios=(0.7,), times=())
        split = _model(vols=(0.011, 0.011), ratios=(0.7, 0.7), times=(1.7,))
        t = jnp.asarray([0.0, 0.0, 1.0, 2.0, 1.5])
        T = jnp.asarray([2.0, 5.0, 3.0, 12.0, 9.0])
        np.testing.assert_allclose(
            np.asarray(g2pp_V(split, t, T)), np.asarray(g2pp_V(const, t, T)), rtol=1e-13
        )
        sig, eta = 0.011, 0.0077

        def integrand(TT):
            return lambda u: (
                sig**2 * _b(A, u, TT) ** 2 + eta**2 * _b(B, u, TT) ** 2
                + 2 * RHO * sig * eta * _b(A, u, TT) * _b(B, u, TT)
            )

        for tt, TT in [(0.5, 0.75), (0.0, 0.25), (3.0, 3.1)]:
            ref = _quad(integrand(TT), tt, TT)
            got = float(g2pp_V(split, jnp.asarray(tt), jnp.asarray(TT)))
            assert got == pytest.approx(ref, rel=1e-12)


# ── Differentiation, triangularity, transforms ───────────────────────


class TestDifferentiation:
    def _bond(self, m):
        return g2pp_bond_price(m, jnp.asarray(0.01), jnp.asarray(-0.004), jnp.asarray(2.7), jnp.asarray(8.0))

    @pytest.mark.parametrize("field", ["volatility_x", "volatility_ratio"])
    def test_bond_price_gradient_matches_fd(self, field):
        m = _model()

        def f(v):
            return self._bond(eqx.tree_at(lambda x: getattr(x, field), m, v))

        v0 = getattr(m, field)
        grad = np.asarray(jax.grad(f)(v0))
        assert np.all(np.isfinite(grad))
        h = 1e-6 * float(jnp.max(v0))
        for i in range(v0.shape[0]):
            e = jnp.zeros(v0.shape[0]).at[i].set(h)
            fd = (float(f(v0 + e)) - float(f(v0 - e))) / (2 * h)
            if abs(fd) < 1e-14:
                assert abs(grad[i]) < 1e-12
            else:
                assert grad[i] == pytest.approx(fd, rel=1e-4)

    @pytest.mark.parametrize("field", ["volatility_x", "volatility_ratio"])
    def test_moments_gradient_matches_fd(self, field):
        m = _model()

        def f(v):
            mm = eqx.tree_at(lambda x: getattr(x, field), m, v)
            mu_x, mu_y, sd_x, sd_y, rho_xy = g2pp_forward_measure_moments(mm, jnp.asarray(2.9))
            return mu_x + 2 * mu_y + 3 * sd_x + 5 * sd_y + 0.01 * rho_xy

        v0 = getattr(m, field)
        grad = np.asarray(jax.grad(f)(v0))
        h = 1e-6 * float(jnp.max(v0))
        for i in range(v0.shape[0]):
            e = jnp.zeros(v0.shape[0]).at[i].set(h)
            fd = (float(f(v0 + e)) - float(f(v0 - e))) / (2 * h)
            if abs(fd) < 1e-14:
                assert abs(grad[i]) < 1e-12
            else:
                assert grad[i] == pytest.approx(fd, rel=1e-4)

    def test_later_buckets_have_zero_effect(self):
        """Quantities at t depend only on buckets that start before t."""
        m = _model()
        t = jnp.asarray(1.3)  # inside bucket 1; buckets 2, 3 start at 2.0, 3.5

        def quantities(vols, ratios):
            mm = _with_buckets(m, vols, ratios)
            cov = g2pp_state_covariance(mm, jnp.asarray(0.0), t)
            moments = g2pp_forward_measure_moments(mm, t)
            return jnp.concatenate([
                cov.reshape(-1), jnp.stack(moments),
                jnp.stack([g2pp_V(mm, jnp.asarray(0.0), t), g2pp_phi(mm, t),
                           g2pp_phi_integral(mm, t), g2pp_short_rate_variance(mm, t)]),
            ])

        vols, ratios = jnp.asarray(VOLS), jnp.asarray(RATIOS)
        jac_v, jac_r = jax.jacobian(quantities, argnums=(0, 1))(vols, ratios)
        assert np.all(np.asarray(jac_v)[:, 2:] == 0.0)
        assert np.all(np.asarray(jac_r)[:, 2:] == 0.0)
        assert np.any(np.asarray(jac_v)[:, :2] != 0.0)
        # Perturbing the later buckets leaves every value bit-identical.
        bumped = quantities(vols.at[2:].mul(1.7), ratios.at[2:].mul(0.3))
        np.testing.assert_array_equal(np.asarray(bumped), np.asarray(quantities(vols, ratios)))

    def test_filter_jit_smoke(self):
        m = _model()
        f = eqx.filter_jit(g2pp_bond_price)
        args = (jnp.asarray(0.01), jnp.asarray(0.0), jnp.asarray(2.7), jnp.asarray(8.0))
        assert float(f(m, *args)) == pytest.approx(float(g2pp_bond_price(m, *args)), rel=1e-13)
        g = eqx.filter_jit(g2pp_forward_measure_moments)(m, jnp.asarray(3.0))
        assert all(np.isfinite(float(v)) for v in g)

    def test_vmap_over_different_breakpoints(self):
        m1 = _model(times=(0.8, 2.0, 3.5))
        m2 = _model(times=(1.0, 2.5, 4.0))
        stacked = jax.tree.map(lambda a, b: jnp.stack([a, b]), m1, m2)

        def f(m):
            return jnp.stack([
                g2pp_V(m, jnp.asarray(0.5), jnp.asarray(6.0)),
                g2pp_short_rate_variance(m, jnp.asarray(3.0)),
                g2pp_phi(m, jnp.asarray(2.2)),
            ])

        batched = np.asarray(eqx.filter_vmap(f)(stacked))
        np.testing.assert_array_equal(batched, np.stack([np.asarray(f(m1)), np.asarray(f(m2))]))
        assert not np.array_equal(batched[0], batched[1])


# ── Construction ─────────────────────────────────────────────────────


class TestConstruction:
    def test_scalars_promoted_to_one_bucket(self):
        m = G2PPModel(jnp.asarray(0.5), jnp.asarray(0.1), jnp.asarray(0.01),
                      jnp.asarray(0.8), jnp.asarray(-0.7))
        assert m.volatility_x.shape == (1,)
        assert m.volatility_ratio.shape == (1,)
        assert m.volatility_times.shape == (0,)
        assert m.initial_curve is None and m.curve_id is None

    def test_positional_curve_and_id(self):
        m = G2PPModel(0.5, 0.1, 0.01, 0.8, -0.7, CURVE, "OIS")
        assert m.initial_curve is CURVE and m.curve_id == "OIS"

    def test_volatility_y_property(self):
        m = _model()
        np.testing.assert_array_equal(
            np.asarray(m.volatility_y), np.asarray(m.volatility_ratio * m.volatility_x)
        )

    @pytest.mark.parametrize("ratio", [0.8, jnp.asarray(0.8), jnp.asarray([0.8])])
    def test_one_bucket_ratio_broadcast(self, ratio):
        m = G2PPModel(0.5, 0.1, jnp.asarray([0.01, 0.02]), ratio, -0.7,
                      volatility_times=jnp.asarray([1.0]))
        assert m.volatility_ratio.shape == (2,)
        np.testing.assert_array_equal(np.asarray(m.volatility_ratio), [0.8, 0.8])

    def test_several_buckets_need_breakpoints(self):
        with pytest.raises(ValueError, match="breakpoints"):
            G2PPModel(0.5, 0.1, jnp.asarray([0.01, 0.02]), 0.8, -0.7)

    def test_breakpoint_count_checked(self):
        with pytest.raises(ValueError, match="breakpoints"):
            G2PPModel(0.5, 0.1, jnp.asarray([0.01, 0.02]), 0.8, -0.7,
                      volatility_times=jnp.asarray([1.0, 2.0]))

    def test_ratio_bucket_count_checked(self):
        with pytest.raises(ValueError, match="volatility_ratio"):
            G2PPModel(0.5, 0.1, jnp.asarray([0.01, 0.02]), jnp.asarray([0.8, 0.7, 0.6]), -0.7,
                      volatility_times=jnp.asarray([1.0]))

    def test_no_volatility_y_keyword(self):
        with pytest.raises(TypeError):
            G2PPModel(mean_reversion_x=0.5, mean_reversion_y=0.1, volatility_x=0.01,
                      volatility_y=0.008, correlation=-0.7)

    def test_constant_operator_rejects_piecewise(self):
        from valax.pricing.pde.coefficients import g2pp_operator_2d
        from valax.pricing.pde.grids import Grid2D, centred_state_grid

        g = centred_state_grid(jnp.asarray(0.01), n=11, half_width=4.0)
        with pytest.raises(ValueError, match="g2pp_operator_rows"):
            g2pp_operator_2d(_model(), Grid2D(x=g, y=g))

    def test_b_factor_unchanged(self):
        assert float(g2pp_B(jnp.asarray(A), jnp.asarray(2.0))) == pytest.approx(
            -np.expm1(-2 * A) / A, rel=1e-15
        )


# ── Traced-graph size (compile-time regression guard) ────────────────


def test_piecewise_variance_term_graph_stays_small():
    """The piecewise V(t, T) must not expand into an unrolled double sum.

    Its small-window series was once written as a nested binomial sum and
    evaluated six times per call: a 3-bucket V lowered to 12x the constant
    one's HLO, and a reverse-mode compile of a G2++ Bermudan PDE through
    ``price`` took 427 s (2 s at constant vol). With the A_k recurrence and
    one broadcast evaluation it is 2.7x (4 s). Lowering only — no compile.
    """
    t = jnp.linspace(0.0, 4.0, 41)

    def hlo_lines(model):
        lowered = eqx.filter_jit(lambda m: g2pp_V(m, t, t + 1.0)).lower(model)
        return len(lowered.as_text().splitlines())

    constant = hlo_lines(_model(vols=(0.01,), ratios=(0.8,), times=()))
    piecewise = hlo_lines(_model())
    assert piecewise < 5 * constant, (piecewise, constant)
