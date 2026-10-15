"""One-factor Cheyette with piecewise-constant volatility level σ(t).

Oracles are independent of the implementation's recurrences:

* Gauss-Legendre quadrature of the defining integrals, split at the
  breakpoints (the integrands are smooth inside a bucket):
  :math:`y(t) = \\int_0^t \\sigma^2 e^{-2\\kappa(t-u)}du` and
  :math:`m(t) = \\int_0^t e^{-\\kappa(t-s)} y(s) ds
  = \\int_0^t \\sigma(u)^2 (e^{-\\kappa(t-u)} - e^{-2\\kappa(t-u)})/\\kappa\\,du`;
* the constant case run through the general code (two equal buckets);
* the β = 0 identity with Hull-White: the Cheyette Gaussian mean is the
  Hull-White convexity and :math:`y` the Hull-White state variance.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.models.cheyette import (
    CheyetteModel,
    cheyette_gaussian_mean,
    cheyette_local_vol,
    cheyette_mean_variance_rate,
    cheyette_y_gaussian,
)
from valax.models.hull_white import (
    HullWhiteModel,
    hw_alpha,
    hw_instantaneous_forward,
    hw_short_rate_variance,
)
from valax.models.short_rate import volatility_bucket_fields

KAPPA = 0.07
SKEW = 0.5
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
    )


CURVE = _curve()


def _model(vols=VOLS, times=TIMES, kappa=KAPPA, skew=SKEW):
    return CheyetteModel(
        mean_reversion=jnp.asarray(kappa),
        volatility=jnp.asarray(vols),
        skew=jnp.asarray(skew),
        initial_curve=CURVE,
        volatility_times=jnp.asarray(times),
    )


def _y_ref(t):
    return _quad(lambda u: _sigma_np(u) ** 2 * np.exp(-2 * KAPPA * (t - u)), 0.0, t)


def _mean_ref(t):
    kern = lambda u: (np.exp(-KAPPA * (t - u)) - np.exp(-2 * KAPPA * (t - u))) / KAPPA
    return _quad(lambda u: _sigma_np(u) ** 2 * kern(u), 0.0, t)


# ── Gaussian-limit quantities vs quadrature ──────────────────────────


class TestGaussianLimitQuantities:
    def test_y_gaussian_matches_quadrature(self):
        got = np.asarray(cheyette_y_gaussian(_model(), jnp.asarray(EVAL_T)))
        for t, g in zip(EVAL_T, got):
            assert g == pytest.approx(_y_ref(t), rel=1e-12)

    def test_gaussian_mean_matches_quadrature(self):
        got = np.asarray(cheyette_gaussian_mean(_model(), jnp.asarray(EVAL_T)))
        for t, g in zip(EVAL_T, got):
            assert g == pytest.approx(_mean_ref(t), rel=1e-12)

    def test_gaussian_mean_is_the_ode_solution(self):
        """m(t) = ∫_0^t e^{-κ(t-s)} y(s) ds, with y itself from the code."""
        m = _model()
        for t in (1.3, 3.0, 5.0):
            y = lambda s: np.asarray(cheyette_y_gaussian(m, jnp.asarray(s)))
            ref = _quad(lambda s: np.exp(-KAPPA * (t - s)) * y(s), 0.0, t)
            got = float(cheyette_gaussian_mean(m, jnp.asarray(t)))
            assert got == pytest.approx(ref, rel=1e-12)

    def test_mean_variance_rate(self):
        m = _model()
        for t0, t1 in [(0.1, 0.5), (0.5, 2.5), (1.9, 3.6), (4.0, 6.0)]:
            ref = _quad(lambda u: _sigma_np(u) ** 2, t0, t1) / (t1 - t0)
            got = float(cheyette_mean_variance_rate(m, jnp.asarray(t0), jnp.asarray(t1)))
            assert got == pytest.approx(ref, rel=1e-13)

    def test_local_vol_selects_bucket(self):
        m = _model()
        x = jnp.asarray(0.01)
        for t in (0.0, 0.5, 0.8, 2.7, 3.5, 10.0):
            expected = _sigma_np(t) * (1.0 + SKEW * 0.01)
            got = float(cheyette_local_vol(m, x, jnp.asarray(t)))
            assert got == pytest.approx(expected, rel=1e-15)

    def test_local_vol_needs_time_when_piecewise(self):
        with pytest.raises(ValueError, match="time"):
            cheyette_local_vol(_model(), jnp.asarray(0.0))

    def test_beta_zero_matches_hull_white(self):
        """y(t) is the HW state variance and m(t) the HW convexity."""
        ch = _model(skew=0.0)
        hw = HullWhiteModel(
            mean_reversion=jnp.asarray(KAPPA), volatility=jnp.asarray(VOLS),
            initial_curve=CURVE, volatility_times=jnp.asarray(TIMES),
        )
        t = jnp.asarray(EVAL_T)
        np.testing.assert_allclose(
            np.asarray(cheyette_y_gaussian(ch, t)),
            np.asarray(hw_short_rate_variance(hw, t)), rtol=1e-14,
        )
        convexity = np.asarray(hw_alpha(hw, t)) - np.asarray(
            [float(hw_instantaneous_forward(hw, s)) for s in t]
        )
        np.testing.assert_allclose(
            np.asarray(cheyette_gaussian_mean(ch, t)), convexity, rtol=1e-12,
        )


class TestConstantThroughGeneralPath:
    """Equal buckets run the general code and must equal the closed form."""

    def test_equal_buckets_equal_constant(self):
        const = _model(vols=(0.011,), times=())
        split = _model(vols=(0.011, 0.011), times=(2.0,))
        t = jnp.asarray(EVAL_T)
        pairs = [
            (cheyette_y_gaussian(const, t), cheyette_y_gaussian(split, t)),
            (cheyette_gaussian_mean(const, t), cheyette_gaussian_mean(split, t)),
            (
                cheyette_mean_variance_rate(const, t * 0.5, t),
                cheyette_mean_variance_rate(split, t * 0.5, t),
            ),
            (
                cheyette_local_vol(const, jnp.asarray(0.02), t),
                cheyette_local_vol(split, jnp.asarray(0.02), t),
            ),
        ]
        for c, s in pairs:
            np.testing.assert_allclose(np.asarray(s), np.asarray(c), rtol=1e-13)


class TestDifferentiation:
    @pytest.mark.parametrize("fn", [cheyette_y_gaussian, cheyette_gaussian_mean],
                             ids=["y", "mean"])
    def test_gradient_matches_fd(self, fn):
        m = _model()

        def f(vols):
            return fn(eqx.tree_at(lambda x: x.volatility, m, vols), jnp.asarray(4.2))

        vols = jnp.asarray(VOLS)
        grad = np.asarray(jax.grad(f)(vols))
        assert np.all(np.isfinite(grad))
        h = 1e-7
        for i in range(len(VOLS)):
            e = jnp.zeros(len(VOLS)).at[i].set(h)
            fd = (float(f(vols + e)) - float(f(vols - e))) / (2 * h)
            assert grad[i] == pytest.approx(fd, rel=1e-4)

    @pytest.mark.parametrize("fn", [cheyette_y_gaussian, cheyette_gaussian_mean],
                             ids=["y", "mean"])
    def test_later_buckets_have_zero_effect(self, fn):
        """Quantities at t depend only on buckets that start before t."""
        m = _model()
        g = jax.grad(
            lambda v: fn(eqx.tree_at(lambda x: x.volatility, m, v), jnp.asarray(1.3))
        )(jnp.asarray(VOLS))
        assert np.all(np.asarray(g)[:2] > 0.0)
        assert np.all(np.asarray(g)[2:] == 0.0)
        # ...and changing them leaves the value bit-identical.
        bumped = eqx.tree_at(
            lambda x: x.volatility, m, jnp.asarray((VOLS[0], VOLS[1], 0.05, 0.002))
        )
        assert float(fn(bumped, jnp.asarray(1.3))) == float(fn(m, jnp.asarray(1.3)))

    def test_filter_jit_smoke(self):
        f = eqx.filter_jit(cheyette_gaussian_mean)
        assert np.isfinite(float(f(_model(), jnp.asarray(2.7))))

    def test_vmap_over_different_breakpoints(self):
        m1 = _model(times=(0.8, 2.0, 3.5))
        m2 = _model(times=(1.0, 2.5, 4.0))
        stacked = jax.tree.map(lambda a, b: jnp.stack([a, b]), m1, m2)
        batched = eqx.filter_vmap(
            lambda m: cheyette_gaussian_mean(m, jnp.asarray(3.0))
        )(stacked)
        np.testing.assert_array_equal(
            np.asarray(batched),
            [float(cheyette_gaussian_mean(m, jnp.asarray(3.0))) for m in (m1, m2)],
        )


class TestConstruction:
    def test_scalar_promoted_to_one_bucket(self):
        m = CheyetteModel(
            mean_reversion=jnp.asarray(0.05), volatility=jnp.asarray(0.01),
            skew=jnp.asarray(0.2),
        )
        assert m.volatility.shape == (1,)
        assert m.volatility_times.shape == (0,)
        assert m.initial_curve is None and m.curve_id is None

    def test_python_float_promoted(self):
        m = CheyetteModel(jnp.asarray(0.05), 0.01, jnp.asarray(0.2))
        assert m.volatility.shape == (1,)

    def test_positional_order_unchanged(self):
        m = CheyetteModel(jnp.asarray(0.05), jnp.asarray(0.01), jnp.asarray(0.2), CURVE, "ois")
        assert m.initial_curve is CURVE
        assert m.curve_id == "ois"
        assert float(m.skew) == 0.2

    def test_several_buckets_need_breakpoints(self):
        with pytest.raises(ValueError, match="breakpoints"):
            CheyetteModel(
                mean_reversion=jnp.asarray(0.05), volatility=jnp.asarray([0.01, 0.02]),
                skew=jnp.asarray(0.0),
            )

    def test_breakpoint_count_checked(self):
        with pytest.raises(ValueError, match="breakpoints"):
            CheyetteModel(
                mean_reversion=jnp.asarray(0.05),
                volatility=jnp.asarray([0.01, 0.02]),
                skew=jnp.asarray(0.0),
                volatility_times=jnp.asarray([1.0, 2.0]),
            )

    def test_bucket_fields_registered(self):
        assert volatility_bucket_fields(_model()) == ("volatility",)
