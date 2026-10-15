"""Two-factor Cheyette with piecewise-constant σ₁(t) and volatility ratio r(t).

Model-level maths of the Gaussian (β = 0) quantities. Oracles are independent
of the implementation's closed forms:

* Gauss-Legendre quadrature of the defining integrals, split at the
  breakpoints (the integrands are smooth inside a bucket);
* the mean ODE :math:`\\dot m_i = y_{ii} + y_{12} - \\kappa_i m_i`, by autodiff;
* the constant case run through the general code (equal buckets);
* the Gaussian G2++ twin with the same σ(t), r(t) (state covariance and the
  φ convexity).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.models.cheyette import (
    CheyetteModel2F,
    cheyette_gaussian_mean_2f,
    cheyette_instantaneous_forward,
    cheyette_local_vol_2f,
    cheyette_mean_variance_rates_2f,
    cheyette_y_gaussian_2f,
)
from valax.models.g2pp import G2PPModel, g2pp_phi, g2pp_state_covariance
from valax.models.short_rate import volatility_bucket_fields

K1, K2, RHO = 0.5, 0.1, -0.7
VOLS = (0.012, 0.008, 0.015, 0.010)
RATIOS = (0.8, 0.6, 0.9, 1.3)
TIMES = (0.8, 2.0, 3.5)
EVAL_T = (0.25, 0.5, 0.8, 1.3, 2.0, 3.0, 3.5, 5.0, 9.0)

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


def _s1(u):
    return _bucket(u, VOLS)


def _s2(u):
    return _bucket(u, RATIOS) * _bucket(u, VOLS)


def _model(vols=VOLS, ratios=RATIOS, times=TIMES, curve=None):
    return CheyetteModel2F(
        mean_reversion_1=jnp.asarray(K1),
        mean_reversion_2=jnp.asarray(K2),
        volatility_1=jnp.asarray(vols),
        volatility_ratio=jnp.asarray(ratios),
        skew_1=jnp.asarray(0.0),
        skew_2=jnp.asarray(0.0),
        correlation=jnp.asarray(RHO),
        initial_curve=curve,
        volatility_times=jnp.asarray(times),
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
    )


def _summary(model, t=4.0):
    """A scalar touching every Gaussian quantity (for gradient checks)."""
    y = cheyette_y_gaussian_2f(model, jnp.asarray(t))
    m1, m2 = cheyette_gaussian_mean_2f(model, jnp.asarray(t))
    return y[0, 0] + y[1, 1] + y[0, 1] + m1 + m2


# ── Construction ─────────────────────────────────────────────────────


class TestConstruction:
    def test_scalars_promote_to_one_bucket(self):
        m = CheyetteModel2F(
            jnp.asarray(K1), jnp.asarray(K2), jnp.asarray(0.01), jnp.asarray(0.8),
            jnp.asarray(0.1), jnp.asarray(0.05), jnp.asarray(RHO),
        )
        assert m.volatility_1.shape == (1,)
        assert m.volatility_ratio.shape == (1,)
        assert m.volatility_times.shape == (0,)
        assert float(m.volatility_2[0]) == pytest.approx(0.008, rel=1e-15)

    def test_one_bucket_ratio_broadcasts(self):
        m = _model(ratios=(0.7,))
        assert m.volatility_ratio.shape == (4,)
        np.testing.assert_array_equal(np.asarray(m.volatility_ratio), 0.7)
        np.testing.assert_allclose(
            np.asarray(m.volatility_2), 0.7 * np.asarray(VOLS), rtol=1e-15
        )

    def test_bucket_fields_registered(self):
        assert volatility_bucket_fields(_model()) == ("volatility_1", "volatility_ratio")

    def test_several_buckets_need_breakpoints(self):
        with pytest.raises(ValueError, match="breakpoints"):
            CheyetteModel2F(
                jnp.asarray(K1), jnp.asarray(K2), jnp.asarray(VOLS),
                jnp.asarray(RATIOS), jnp.asarray(0.0), jnp.asarray(0.0),
                jnp.asarray(RHO),
            )

    def test_wrong_breakpoint_count(self):
        with pytest.raises(ValueError, match="CheyetteModel2F"):
            _model(times=(1.0, 2.0))

    def test_ratio_bucket_mismatch(self):
        with pytest.raises(ValueError, match="volatility_ratio"):
            _model(ratios=(0.8, 0.6))

    def test_local_vol_needs_time(self):
        with pytest.raises(ValueError, match="time t"):
            cheyette_local_vol_2f(_model(), jnp.asarray(0.0), jnp.asarray(0.0))

    def test_local_vol_selects_bucket(self):
        m = eqx.tree_at(lambda mm: (mm.skew_1, mm.skew_2), _model(),
                        (jnp.asarray(2.0), jnp.asarray(-1.0)))
        for t in (0.0, 0.8, 1.9, 2.0, 7.0):
            s1, s2 = cheyette_local_vol_2f(m, jnp.asarray(0.01), jnp.asarray(0.02), jnp.asarray(t))
            assert float(s1) == pytest.approx(_s1(t) * 1.02, rel=1e-14)
            assert float(s2) == pytest.approx(_s2(t) * 0.98, rel=1e-14)


# ── Gaussian quantities vs quadrature ────────────────────────────────


class TestQuadrature:
    @pytest.mark.parametrize("t", EVAL_T)
    def test_accumulators(self, t):
        y = np.asarray(cheyette_y_gaussian_2f(_model(), jnp.asarray(t)))
        q11 = _quad(lambda u: _s1(u) ** 2 * np.exp(-2 * K1 * (t - u)), 0.0, t)
        q22 = _quad(lambda u: _s2(u) ** 2 * np.exp(-2 * K2 * (t - u)), 0.0, t)
        q12 = RHO * _quad(lambda u: _s1(u) * _s2(u) * np.exp(-(K1 + K2) * (t - u)), 0.0, t)
        np.testing.assert_allclose(y, [[q11, q12], [q12, q22]], rtol=1e-12)

    @pytest.mark.parametrize("t", EVAL_T)
    def test_means(self, t):
        # m_i(t) = int_0^t e^{-k_i (t-s)} (y_ii + y_12)(s) ds, inner s-integral
        # done by hand: int_u^t e^{-k_i(t-s)} e^{-c(s-u)} ds.
        def inner(k, c, u):
            return np.exp(-k * (t - u)) * -np.expm1(-(c - k) * (t - u)) / (c - k)

        q1 = _quad(lambda u: _s1(u) ** 2 * inner(K1, 2 * K1, u)
                   + RHO * _s1(u) * _s2(u) * inner(K1, K1 + K2, u), 0.0, t)
        q2 = _quad(lambda u: _s2(u) ** 2 * inner(K2, 2 * K2, u)
                   + RHO * _s1(u) * _s2(u) * inner(K2, K1 + K2, u), 0.0, t)
        m1, m2 = cheyette_gaussian_mean_2f(_model(), jnp.asarray(t))
        assert float(m1) == pytest.approx(q1, rel=1e-12)
        assert float(m2) == pytest.approx(q2, rel=1e-12)

    @pytest.mark.parametrize("t", (0.5, 1.3, 3.0, 5.0))
    def test_mean_ode(self, t):
        """dm_i/dt = y_ii + y_12 - k_i m_i (inside a bucket, by autodiff)."""
        model = _model()
        dm1, dm2 = jax.jacfwd(lambda s: jnp.stack(cheyette_gaussian_mean_2f(model, s)))(
            jnp.asarray(t)
        )
        y = cheyette_y_gaussian_2f(model, jnp.asarray(t))
        m1, m2 = cheyette_gaussian_mean_2f(model, jnp.asarray(t))
        assert float(dm1) == pytest.approx(float(y[0, 0] + y[0, 1] - K1 * m1), rel=1e-10)
        assert float(dm2) == pytest.approx(float(y[1, 1] + y[0, 1] - K2 * m2), rel=1e-10)

    @pytest.mark.parametrize("t0,t1", [(0.0, 0.5), (0.5, 1.7), (1.9, 2.1), (3.0, 6.0)])
    def test_step_average_rates(self, t0, t1):
        v1, v2, c12 = cheyette_mean_variance_rates_2f(_model(), jnp.asarray(t0), jnp.asarray(t1))
        dt = t1 - t0
        assert float(v1) == pytest.approx(_quad(lambda u: _s1(u) ** 2, t0, t1) / dt, rel=1e-13)
        assert float(v2) == pytest.approx(_quad(lambda u: _s2(u) ** 2, t0, t1) / dt, rel=1e-13)
        assert float(c12) == pytest.approx(
            RHO * _quad(lambda u: _s1(u) * _s2(u), t0, t1) / dt, rel=1e-13
        )


# ── Constant case and the G2++ twin ──────────────────────────────────


class TestConstantAndTwin:
    def test_equal_buckets_match_constant_closed_form(self):
        const = _model(vols=(0.011,), ratios=(0.7,), times=())
        split = _model(vols=(0.011,) * 4, ratios=(0.7,) * 4)
        for t in EVAL_T:
            t = jnp.asarray(t)
            np.testing.assert_allclose(
                np.asarray(cheyette_y_gaussian_2f(split, t)),
                np.asarray(cheyette_y_gaussian_2f(const, t)), rtol=1e-13,
            )
        # The means from t = 0.8 on: below that m_2 is a cancellation of its
        # own and (rho < 0) cross terms -- see test_short_time_mean_conditioning.
        for t in (0.8, 1.3, 2.0, 3.0, 3.5, 5.0, 9.0):
            t = jnp.asarray(t)
            np.testing.assert_allclose(
                np.asarray(cheyette_gaussian_mean_2f(split, t)),
                np.asarray(cheyette_gaussian_mean_2f(const, t)), rtol=1e-13,
            )
        rates_split = cheyette_mean_variance_rates_2f(split, jnp.asarray(0.5), jnp.asarray(2.5))
        rates_const = cheyette_mean_variance_rates_2f(const, jnp.asarray(0.5), jnp.asarray(2.5))
        np.testing.assert_allclose(np.asarray(rates_split), np.asarray(rates_const), rtol=1e-13)

    @pytest.mark.parametrize("t_str", ["0.25", "0.5"])
    def test_short_time_mean_conditioning(self, t_str):
        """At short times m_2 is 16-30x smaller than its own term (the rho < 0
        cross term nearly cancels it) and the general path's I_k - I_2k
        difference cancels too, so both forms lose digits. Against a 50-digit
        reference of the constant closed form both stay within 1e-12
        (measured, general / constant: t = 0.25: 2.4e-13 / 9.6e-13;
        t = 0.5: 2.0e-13 / 6.6e-15)."""
        from decimal import Decimal, getcontext

        getcontext().prec = 50
        k1, k2, s1 = Decimal("0.5"), Decimal("0.1"), Decimal("0.011")
        s2, rho, t = Decimal("0.7") * s1, Decimal("-0.7"), Decimal(t_str)
        e1, e2 = 1 - (-k1 * t).exp(), 1 - (-k2 * t).exp()
        cross = rho * s1 * s2 / (k1 + k2)
        exact = float(
            s2**2 / (2 * k2**2) * e2**2 + cross * (e2 / k2 - (-k2 * t).exp() * e1 / k1)
        )
        const = _model(vols=(0.011,), ratios=(0.7,), times=())
        split = _model(vols=(0.011,) * 4, ratios=(0.7,) * 4)
        tj = jnp.asarray(float(t_str))
        m2_split = float(cheyette_gaussian_mean_2f(split, tj)[1])
        m2_const = float(cheyette_gaussian_mean_2f(const, tj)[1])
        assert m2_split == pytest.approx(exact, rel=1e-12)
        assert m2_const == pytest.approx(exact, rel=1e-12)
        assert m2_split == pytest.approx(m2_const, rel=1e-12)

    def test_matches_g2pp_twin(self):
        curve = _curve()
        ch = _model(curve=curve)
        g2 = G2PPModel(
            mean_reversion_x=jnp.asarray(K1), mean_reversion_y=jnp.asarray(K2),
            volatility_x=jnp.asarray(VOLS), volatility_ratio=jnp.asarray(RATIOS),
            correlation=jnp.asarray(RHO), initial_curve=curve,
            volatility_times=jnp.asarray(TIMES),
        )
        for t in EVAL_T:
            t = jnp.asarray(t)
            np.testing.assert_allclose(
                np.asarray(cheyette_y_gaussian_2f(ch, t)),
                np.asarray(g2pp_state_covariance(g2, jnp.asarray(0.0), t)), rtol=1e-13,
            )
            m1, m2 = cheyette_gaussian_mean_2f(ch, t)
            convexity = g2pp_phi(g2, t) - cheyette_instantaneous_forward(ch, t)
            assert float(m1 + m2) == pytest.approx(float(convexity), rel=1e-11)


# ── Differentiation, causality, JIT, vmap ────────────────────────────


class TestDifferentiation:
    @pytest.mark.parametrize("field", ["volatility_1", "volatility_ratio"])
    def test_bucket_gradients_match_fd(self, field):
        model = _model()
        v0 = getattr(model, field)

        def f(v):
            return _summary(eqx.tree_at(lambda m: getattr(m, field), model, v))

        grad = np.asarray(jax.grad(f)(v0))
        assert np.all(np.isfinite(grad))
        for i in range(v0.shape[0]):
            h = 1e-6 * float(v0[i])
            e = jnp.zeros(v0.shape[0]).at[i].set(h)
            fd = (float(f(v0 + e)) - float(f(v0 - e))) / (2 * h)
            assert grad[i] == pytest.approx(fd, rel=1e-4)

    def test_later_buckets_have_no_effect_earlier(self):
        """Before the start of bucket j its value has exactly zero effect."""
        model = _model()
        t = 1.9  # inside bucket 1: buckets 2 and 3 have not started
        for field in ("volatility_1", "volatility_ratio"):
            v0 = getattr(model, field)
            grad = jax.grad(
                lambda v: _summary(eqx.tree_at(lambda m: getattr(m, field), model, v), t)
            )(v0)
            assert float(grad[2]) == 0.0 and float(grad[3]) == 0.0
            assert float(grad[0]) != 0.0
            bumped = eqx.tree_at(
                lambda m: getattr(m, field), model, v0.at[2:].multiply(1.7)
            )
            assert float(_summary(bumped, t)) == float(_summary(model, t))

    def test_filter_jit_smoke(self):
        f = eqx.filter_jit(_summary)
        model = _model()
        assert float(f(model)) == pytest.approx(float(_summary(model)), rel=1e-14)

    def test_vmap_over_breakpoints(self):
        """Models whose buckets sit on different dates stack under vmap."""
        times_a, times_b = (0.8, 2.0, 3.5), (0.5, 1.5, 4.0)
        models = [_model(times=times_a), _model(times=times_b)]
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *models)
        t = jnp.asarray(3.0)
        batched = jax.vmap(lambda m: cheyette_y_gaussian_2f(m, t))(stacked)
        means = jax.vmap(lambda m: jnp.stack(cheyette_gaussian_mean_2f(m, t)))(stacked)
        for k, m in enumerate(models):
            np.testing.assert_allclose(
                np.asarray(batched[k]), np.asarray(cheyette_y_gaussian_2f(m, t)), rtol=1e-15
            )
            np.testing.assert_allclose(
                np.asarray(means[k]),
                np.asarray(jnp.stack(cheyette_gaussian_mean_2f(m, t))), rtol=1e-15,
            )
        assert not np.allclose(np.asarray(batched[0]), np.asarray(batched[1]))
