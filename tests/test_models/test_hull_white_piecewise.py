"""Hull-White with piecewise-constant volatility σ(t): the model-level maths.

Oracles are independent of the implementation's recurrences:

* Gauss-Legendre quadrature of the defining integrals, split at the
  breakpoints (the integrands are smooth inside a bucket);
* the exact-fit identity
  :math:`\\int_0^T \\alpha = -\\ln P^M(0,T) + \\tfrac12\\int_0^T \\sigma^2 B^2`;
* the tower property :math:`P^M(0,T) = E[e^{-\\int_0^t r} P(t,T \\mid r_t)]`,
  evaluated as a Gaussian expectation with quadrature moments;
* the constant case run through the general code (two equal buckets).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.models.hull_white import (
    HullWhiteModel,
    hw_B,
    hw_alpha,
    hw_alpha_average,
    hw_bond_price,
    hw_instantaneous_forward,
    hw_market_df,
    hw_mean_variance_rate,
    hw_short_rate_variance,
    hw_state_variance,
)
from valax.models.short_rate import decayed_integral, piecewise_integral

A = 0.07
VOLS = (0.012, 0.007, 0.015, 0.009)
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


def _sigma_np(u, vols=VOLS, times=TIMES):
    return np.asarray(vols)[np.searchsorted(np.asarray(times), u, side="right")]


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


def _model(vols=VOLS, times=TIMES, a=A):
    return HullWhiteModel(
        mean_reversion=jnp.asarray(a),
        volatility=jnp.asarray(vols),
        initial_curve=CURVE,
        volatility_times=jnp.asarray(times),
    )


# ── The primitive ────────────────────────────────────────────────────


class TestDecayedIntegral:
    @pytest.mark.parametrize("rate", [0.3, 0.07, 1e-3, 1e-5])
    def test_matches_quadrature(self, rate):
        values = jnp.asarray([1.3, 0.4, 2.1, 0.9])
        times = jnp.asarray(TIMES)
        i_val, j_val = decayed_integral(values, times, jnp.asarray(rate), jnp.asarray(EVAL_T))
        vals_np = np.asarray(values)
        for t, i_t, j_t in zip(EVAL_T, np.asarray(i_val), np.asarray(j_val)):
            v = lambda u: _sigma_np(u, vals_np)
            i_ref = _quad(lambda u: v(u) * np.exp(-rate * (t - u)), 0.0, t)
            # J(t) = ∫_0^t I = ∫_0^t v(u) (1 - e^{-k(t-u)})/k du (order swapped).
            j_ref = _quad(lambda u: v(u) * -np.expm1(-rate * (t - u)) / rate, 0.0, t)
            assert i_t == pytest.approx(i_ref, rel=1e-12)
            assert j_t == pytest.approx(j_ref, rel=1e-12)

    def test_zero_at_origin(self):
        i0, j0 = decayed_integral(
            jnp.asarray(VOLS), jnp.asarray(TIMES), jnp.asarray(0.1), jnp.asarray(0.0)
        )
        assert float(i0) == 0.0 and float(j0) == 0.0

    def test_piecewise_integral(self):
        got = piecewise_integral(jnp.asarray(VOLS), jnp.asarray(TIMES), jnp.asarray(EVAL_T))
        for t, g in zip(EVAL_T, np.asarray(got)):
            assert g == pytest.approx(_quad(_sigma_np, 0.0, t), rel=1e-13)


# ── Hull-White quantities ────────────────────────────────────────────


class TestHullWhitePiecewise:
    def test_short_rate_variance(self):
        m = _model()
        for t in EVAL_T:
            ref = _quad(lambda u: _sigma_np(u) ** 2 * np.exp(-2 * A * (t - u)), 0.0, t)
            assert float(hw_short_rate_variance(m, jnp.asarray(t))) == pytest.approx(ref, rel=1e-12)

    def test_state_variance(self):
        m = _model()
        for s, t in [(0.3, 1.1), (0.8, 2.0), (1.5, 4.2), (4.0, 9.0)]:
            ref = _quad(lambda u: _sigma_np(u) ** 2 * np.exp(-2 * A * (t - u)), s, t)
            got = float(hw_state_variance(m, jnp.asarray(s), jnp.asarray(t)))
            assert got == pytest.approx(ref, rel=1e-12)

    def test_alpha_convexity(self):
        m = _model()
        for t in EVAL_T:
            b = lambda u: -np.expm1(-A * (t - u)) / A
            ref = _quad(lambda u: _sigma_np(u) ** 2 * np.exp(-A * (t - u)) * b(u), 0.0, t)
            got = float(hw_alpha(m, jnp.asarray(t)) - hw_instantaneous_forward(m, jnp.asarray(t)))
            assert got == pytest.approx(ref, rel=1e-12)

    @pytest.mark.parametrize("T", [0.6, 2.0, 3.0, 7.5, 11.0])
    def test_exact_fit_identity(self, T):
        """∫_0^T α = -ln P(0,T) + ½ ∫_0^T σ(u)² B(u,T)² du."""
        m = _model()
        lhs = float(hw_alpha_average(m, jnp.asarray(0.0), jnp.asarray(T))) * T
        var = _quad(lambda u: _sigma_np(u) ** 2 * (-np.expm1(-A * (T - u)) / A) ** 2, 0.0, T)
        rhs = -float(jnp.log(hw_market_df(m, jnp.asarray(T)))) + 0.5 * var
        assert lhs == pytest.approx(rhs, rel=1e-12)

    @pytest.mark.parametrize("t0,t1", [(0.2, 0.7), (0.5, 1.5), (1.7, 4.1), (3.0, 9.0)])
    def test_alpha_average_straddling(self, t0, t1):
        """Step averages across breakpoints and curve pillars."""
        m = _model()
        f = lambda u: np.asarray(hw_alpha(m, jnp.asarray(u)))
        ref = _quad(f, t0, t1, breaks=TIMES + PILLAR_T) / (t1 - t0)
        got = float(hw_alpha_average(m, jnp.asarray(t0), jnp.asarray(t1)))
        assert got == pytest.approx(ref, rel=1e-12)

    @pytest.mark.parametrize("t,T", [(1.0, 4.0), (2.5, 7.0), (4.0, 10.0)])
    def test_bond_price_tower_property(self, t, T):
        """P(0,T) = E[exp(-∫_0^t r) P(t,T | r_t)] with Gaussian moments by quadrature."""
        m = _model()
        e = lambda u: np.exp(-A * (t - u))
        b_ut = lambda u: -np.expm1(-A * (t - u)) / A
        v_int = _quad(lambda u: _sigma_np(u) ** 2 * b_ut(u) ** 2, 0.0, t)
        v_x = _quad(lambda u: _sigma_np(u) ** 2 * e(u) ** 2, 0.0, t)
        cov = _quad(lambda u: _sigma_np(u) ** 2 * b_ut(u) * e(u), 0.0, t)
        b = float(hw_B(jnp.asarray(A), jnp.asarray(T - t)))
        alpha_t = float(hw_alpha(m, jnp.asarray(t)))
        int_alpha = float(hw_alpha_average(m, jnp.asarray(0.0), jnp.asarray(t))) * t
        # P(t,T|r) = A e^{-B r}: read A at r = 0.
        a_tt = float(hw_bond_price(m, jnp.asarray(0.0), jnp.asarray(t), jnp.asarray(T)))
        expected = (
            np.exp(-int_alpha) * a_tt * np.exp(-b * alpha_t)
            * np.exp(0.5 * (v_int + 2 * b * cov + b * b * v_x))
        )
        assert expected == pytest.approx(float(hw_market_df(m, jnp.asarray(T))), rel=1e-12)

    def test_mean_variance_rate(self):
        m = _model()
        for t0, t1 in [(0.1, 0.5), (0.5, 2.5), (1.9, 3.6)]:
            ref = _quad(lambda u: _sigma_np(u) ** 2, t0, t1) / (t1 - t0)
            got = float(hw_mean_variance_rate(m, jnp.asarray(t0), jnp.asarray(t1)))
            assert got == pytest.approx(ref, rel=1e-13)


class TestConstantThroughGeneralPath:
    """Equal buckets run the general code and must equal the closed form."""

    def test_equal_buckets_equal_constant(self):
        const = _model(vols=(0.011,), times=())
        split = _model(vols=(0.011, 0.011, 0.011), times=(1.0, 4.0))
        t = jnp.asarray(EVAL_T)
        pairs = [
            (hw_alpha(const, t), hw_alpha(split, t)),
            (hw_short_rate_variance(const, t), hw_short_rate_variance(split, t)),
            (hw_alpha_average(const, t * 0.5, t), hw_alpha_average(split, t * 0.5, t)),
            (hw_state_variance(const, t * 0.5, t), hw_state_variance(split, t * 0.5, t)),
            (
                hw_bond_price(const, jnp.asarray(0.03), jnp.asarray(2.5), jnp.asarray(8.0)),
                hw_bond_price(split, jnp.asarray(0.03), jnp.asarray(2.5), jnp.asarray(8.0)),
            ),
        ]
        for c, s in pairs:
            np.testing.assert_allclose(np.asarray(s), np.asarray(c), rtol=1e-13)


class TestDifferentiation:
    def test_bond_price_gradient_matches_fd(self):
        m = _model()

        def f(vols):
            mm = eqx.tree_at(lambda x: x.volatility, m, vols)
            return hw_bond_price(mm, jnp.asarray(0.03), jnp.asarray(2.7), jnp.asarray(8.0))

        vols = jnp.asarray(VOLS)
        grad = np.asarray(jax.grad(f)(vols))
        h = 1e-7
        for i in range(len(VOLS)):
            e = jnp.zeros(len(VOLS)).at[i].set(h)
            fd = (float(f(vols + e)) - float(f(vols - e))) / (2 * h)
            if abs(fd) < 1e-14:
                assert abs(grad[i]) < 1e-12
            else:
                assert grad[i] == pytest.approx(fd, rel=1e-4)

    def test_later_buckets_have_zero_sensitivity(self):
        """Var x(t) depends only on buckets that start before t (triangularity)."""
        m = _model()
        g = jax.grad(
            lambda v: hw_short_rate_variance(
                eqx.tree_at(lambda x: x.volatility, m, v), jnp.asarray(1.3)
            )
        )(jnp.asarray(VOLS))
        assert np.all(np.asarray(g)[:2] > 0.0)
        assert np.all(np.asarray(g)[2:] == 0.0)

    def test_filter_jit_smoke(self):
        m = _model()
        f = eqx.filter_jit(hw_bond_price)
        p = f(m, jnp.asarray(0.03), jnp.asarray(2.7), jnp.asarray(8.0))
        assert np.isfinite(float(p))

    def test_vmap_over_different_breakpoints(self):
        m1 = _model(times=(0.8, 2.0, 3.5))
        m2 = _model(times=(1.0, 2.5, 4.0))
        stacked = jax.tree.map(lambda a, b: jnp.stack([a, b]), m1, m2)
        batched = eqx.filter_vmap(
            lambda m: hw_short_rate_variance(m, jnp.asarray(3.0))
        )(stacked)
        np.testing.assert_array_equal(
            np.asarray(batched),
            [float(hw_short_rate_variance(m, jnp.asarray(3.0))) for m in (m1, m2)],
        )


class TestConstruction:
    def test_scalar_promoted_to_one_bucket(self):
        m = HullWhiteModel(mean_reversion=jnp.asarray(0.05), volatility=jnp.asarray(0.01))
        assert m.volatility.shape == (1,)
        assert m.volatility_times.shape == (0,)

    def test_positional_curve_unchanged(self):
        m = HullWhiteModel(jnp.asarray(0.05), jnp.asarray(0.01), CURVE)
        assert m.initial_curve is CURVE

    def test_several_buckets_need_breakpoints(self):
        with pytest.raises(ValueError, match="breakpoints"):
            HullWhiteModel(mean_reversion=jnp.asarray(0.05), volatility=jnp.asarray([0.01, 0.02]))

    def test_breakpoint_count_checked(self):
        with pytest.raises(ValueError, match="breakpoints"):
            HullWhiteModel(
                mean_reversion=jnp.asarray(0.05),
                volatility=jnp.asarray([0.01, 0.02]),
                volatility_times=jnp.asarray([1.0, 2.0]),
            )
