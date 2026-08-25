"""Tests for Hull-White one-factor model: analytics and ZCB pricing."""

import jax
import jax.numpy as jnp
import pytest

from valax.models.hull_white import (
    HullWhiteModel,
    hw_B,
    hw_alpha,
    hw_alpha_average,
    hw_bond_price,
    hw_instantaneous_forward,
    hw_market_df,
    hw_short_rate_variance,
    _instantaneous_forward,
)
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


@pytest.fixture
def flat_curve(ref_date):
    return _flat_curve(ref_date, 0.05)


@pytest.fixture
def model(flat_curve):
    return HullWhiteModel(
        mean_reversion=jnp.array(0.10),
        volatility=jnp.array(0.01),
        initial_curve=flat_curve,
    )


# ── B function ───────────────────────────────────────────────────────

class TestHWB:
    def test_b_zero_at_tau_zero(self):
        assert float(hw_B(jnp.array(0.1), jnp.array(0.0))) == pytest.approx(0.0, abs=1e-15)

    def test_b_positive(self):
        assert float(hw_B(jnp.array(0.1), jnp.array(5.0))) > 0.0

    def test_b_limit_large_tau(self):
        """B(a, ∞) → 1/a."""
        a = jnp.array(0.10)
        b = hw_B(a, jnp.array(100.0))
        assert float(b) == pytest.approx(1.0 / float(a), rel=1e-4)

    def test_b_known_value(self):
        """B(0.1, 5) = (1 - exp(-0.5)) / 0.1."""
        expected = (1.0 - float(jnp.exp(-0.5))) / 0.1
        assert float(hw_B(jnp.array(0.1), jnp.array(5.0))) == pytest.approx(expected, rel=1e-10)


# ── Instantaneous forward ───────────────────────────────────────────

class TestInstantaneousForward:
    def test_flat_curve_forward(self, model):
        """On a flat 5% CC curve, f(0,t) = 0.05 for all t."""
        for t in [0.0, 1.0, 5.0, 10.0]:
            f = _instantaneous_forward(model, jnp.array(t))
            assert float(f) == pytest.approx(0.05, abs=1e-4)

    def test_short_end_without_t0_pillar(self, ref_date):
        """Regression: forwards before the first pillar must not collapse to 0.

        ``DiscountCurve`` extrapolates flat, so a curve whose first pillar sits
        strictly after the reference date used to give ``f(0, t) = 0`` on that
        leading stub (``jnp.interp`` is constant there, so its gradient
        vanishes).  ``_log_df_grid`` now anchors the interpolation at
        ``ln P(0,0) = 0``.
        """
        pillars = jnp.array(
            [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(1, 8)],
            dtype=jnp.int32,
        )
        times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
        curve = DiscountCurve(
            pillar_dates=pillars,
            discount_factors=jnp.exp(-0.05 * times),
            reference_date=ref_date,
        )
        m = HullWhiteModel(
            mean_reversion=jnp.array(0.10),
            volatility=jnp.array(0.01),
            initial_curve=curve,
        )
        # t = 0 and t = 0.5 both fall strictly before the first pillar.
        for t in [0.0, 0.25, 0.5]:
            f = _instantaneous_forward(m, jnp.array(t))
            assert float(f) == pytest.approx(0.05, abs=1e-4)

    def test_t0_pillar_is_idempotent(self, ref_date):
        """Anchoring must not perturb a curve that already has a t=0 pillar."""
        with_t0 = _flat_curve(ref_date, 0.05)
        m = HullWhiteModel(
            mean_reversion=jnp.array(0.10),
            volatility=jnp.array(0.01),
            initial_curve=with_t0,
        )
        for t in [0.0, 1.0, 7.5]:
            assert float(_instantaneous_forward(m, jnp.array(t))) == pytest.approx(
                0.05, abs=1e-4
            )


# ── Exact-fit property ───────────────────────────────────────────────

class TestHWBondPrice:
    def test_exact_fit_flat_curve(self, model):
        """P_HW(0, T | r=f(0,0)) recovers the initial curve DF for flat curve."""
        f0 = _instantaneous_forward(model, jnp.array(0.0))
        for T in [0.5, 1.0, 3.0, 5.0, 10.0]:
            p_hw = hw_bond_price(model, f0, jnp.array(0.0), jnp.array(T))
            p_mkt = float(jnp.exp(-0.05 * T))
            assert float(p_hw) == pytest.approx(p_mkt, abs=1e-6)

    def test_exact_fit_steep_curve(self, ref_date):
        """Exact-fit holds on a non-flat curve (within tree tolerance)."""
        pillars = jnp.array(
            [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(16)],
            dtype=jnp.int32,
        )
        times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
        rates = 0.03 + 0.005 * times
        curve = DiscountCurve(
            pillar_dates=pillars,
            discount_factors=jnp.exp(-rates * times),
            reference_date=ref_date,
        )
        model = HullWhiteModel(
            mean_reversion=jnp.array(0.10),
            volatility=jnp.array(0.01),
            initial_curve=curve,
        )
        f0 = _instantaneous_forward(model, jnp.array(0.0))
        for T in [1.0, 5.0, 10.0]:
            p_hw = hw_bond_price(model, f0, jnp.array(0.0), jnp.array(T))
            rate = 0.03 + 0.005 * T
            p_mkt = float(jnp.exp(-rate * T))
            assert float(p_hw) == pytest.approx(p_mkt, abs=1e-4)

    def test_bond_price_positive(self, model):
        """Bond prices are always positive."""
        for r in [-0.01, 0.0, 0.05, 0.10, 0.15]:
            p = hw_bond_price(model, jnp.array(r), jnp.array(0.0), jnp.array(5.0))
            assert float(p) > 0.0

    def test_bond_price_decreasing_in_r(self, model):
        """Higher short rate → lower bond price."""
        prices = [
            float(hw_bond_price(model, jnp.array(r), jnp.array(0.0), jnp.array(5.0)))
            for r in [0.02, 0.04, 0.06, 0.08]
        ]
        assert all(p1 > p2 for p1, p2 in zip(prices, prices[1:]))

    def test_jit_compatible(self, model):
        f0 = _instantaneous_forward(model, jnp.array(0.0))
        eager = hw_bond_price(model, f0, jnp.array(0.0), jnp.array(5.0))
        jitted = jax.jit(hw_bond_price)(model, f0, jnp.array(0.0), jnp.array(5.0))
        assert float(jitted) == pytest.approx(float(eager), rel=1e-10)

    def test_grad_wrt_r(self, model):
        """dP/dr = -B * P (standard HW sensitivity)."""
        r = jnp.array(0.05)
        t = jnp.array(0.0)
        T = jnp.array(5.0)
        P = hw_bond_price(model, r, t, T)
        dPdr = jax.grad(hw_bond_price, argnums=1)(model, r, t, T)
        B = hw_B(model.mean_reversion, T - t)
        assert float(dPdr) == pytest.approx(-float(B) * float(P), rel=1e-8)


# ── Variance ─────────────────────────────────────────────────────────

class TestHWVariance:
    def test_variance_zero_at_zero(self, model):
        v = hw_short_rate_variance(model, jnp.array(0.0))
        assert float(v) == pytest.approx(0.0, abs=1e-15)

    def test_variance_known_value(self, model):
        """sigma^2/(2a) * (1 - exp(-2at))."""
        t = 5.0
        expected = 0.01**2 / (2 * 0.10) * (1.0 - float(jnp.exp(-2 * 0.10 * t)))
        assert float(hw_short_rate_variance(model, jnp.array(t))) == pytest.approx(expected, rel=1e-10)

    def test_variance_increasing(self, model):
        """Variance increases monotonically with time."""
        times = [0.5, 1.0, 2.0, 5.0, 10.0]
        vars_ = [float(hw_short_rate_variance(model, jnp.array(t))) for t in times]
        assert all(v1 < v2 for v1, v2 in zip(vars_, vars_[1:]))


# ── Exact-fit shift alpha(t) ─────────────────────────────────────────
#
# ``hw_alpha`` is the bridge between the short rate and the centred OU state
# used by every numerical scheme: r(t) = x(t) + alpha(t) with
# dx = -a x dt + sigma dW, x(0) = 0. The defining property is that this drift
# is *exactly* -a x — no residual time-dependent term — which holds iff
# theta(t) = alpha'(t) + a alpha(t) for the exact-fit theta. The tests below
# pin that identity numerically (via autodiff of alpha) rather than trusting
# the closed form.

class TestHWAlpha:
    def test_alpha_at_zero_is_the_short_rate(self, model):
        """alpha(0) = f^M(0,0) = r(0); the convexity term vanishes at t = 0."""
        a0 = float(hw_alpha(model, jnp.array(0.0)))
        f0 = float(hw_instantaneous_forward(model, jnp.array(0.0)))
        assert a0 == pytest.approx(f0, abs=1e-14)

    def test_alpha_exceeds_the_forward(self, model):
        """The convexity term is strictly positive for t > 0."""
        for t in (0.5, 2.0, 10.0):
            alpha = float(hw_alpha(model, jnp.array(t)))
            fwd = float(hw_instantaneous_forward(model, jnp.array(t)))
            assert alpha > fwd

    def test_alpha_closed_form(self, model):
        """alpha(t) = f^M(0,t) + sigma^2/(2a^2) (1 - e^{-at})^2."""
        a = float(model.mean_reversion)
        sigma = float(model.volatility)
        for t in (0.25, 3.0, 12.0):
            expected = float(
                hw_instantaneous_forward(model, jnp.array(t))
            ) + sigma**2 / (2.0 * a**2) * (1.0 - float(jnp.exp(-a * t))) ** 2
            assert float(hw_alpha(model, jnp.array(t))) == pytest.approx(
                expected, rel=1e-12
            )

    def test_alpha_vectorises(self, model):
        """Scalar and vector calls agree, and the scalar call stays 0-d."""
        times = jnp.array([0.0, 0.5, 3.0, 9.0])
        vec = hw_alpha(model, times)
        assert vec.shape == (4,)
        assert jnp.shape(hw_alpha(model, jnp.array(3.0))) == ()
        for i, t in enumerate(times):
            assert float(vec[i]) == pytest.approx(
                float(hw_alpha(model, t)), rel=1e-12
            )

    def test_x_drift_is_exactly_mean_reverting(self, model):
        """theta(t) = alpha'(t) + a alpha(t) — the identity that makes
        dx = -a x dt + sigma dW hold with no residual drift term.

        The right-hand side is computed by differentiating ``hw_alpha``; the
        left-hand side uses the standard Hull-White exact-fit
        theta(t) = f_t(0,t) + a f(0,t) + sigma^2/(2a) (1 - e^{-2at}).
        """
        a = model.mean_reversion
        sigma = model.volatility

        d_alpha = jax.grad(lambda s: hw_alpha(model, s))
        d_fwd = jax.grad(lambda s: hw_instantaneous_forward(model, s))

        # Sample away from the curve's pillar kinks, where the log-linear
        # interpolant's forward is piecewise constant and f_t is a delta.
        for t in (0.4, 2.4, 6.4):
            tt = jnp.array(t)
            lhs = (
                float(d_fwd(tt))
                + float(a) * float(hw_instantaneous_forward(model, tt))
                + float(sigma) ** 2 / (2.0 * float(a))
                * (1.0 - float(jnp.exp(-2.0 * float(a) * t)))
            )
            rhs = float(d_alpha(tt)) + float(a) * float(hw_alpha(model, tt))
            assert rhs == pytest.approx(lhs, rel=1e-10)

    def test_alpha_is_differentiable_in_model_params(self, model):
        """Greeks must flow through alpha into the model parameters."""
        import equinox as eqx

        g = eqx.filter_grad(lambda m: hw_alpha(m, jnp.array(5.0)))(model)
        # d alpha / d sigma = sigma/a^2 (1 - e^{-at})^2 > 0.
        assert float(g.volatility) > 0.0


# ── Exact step-averaged alpha ────────────────────────────────────────
#
# ``hw_alpha_average`` exists because midpoint sampling of alpha is *not*
# second-order accurate on a real curve: a log-linear discount curve has a
# piecewise-constant instantaneous forward that jumps at every pillar. A
# discretised scheme built on midpoint sampling stalls (the Hull-White PDE
# plateaued at ~4e-6 bond-repricing error under time refinement); averaging
# exactly restores clean second-order convergence. These tests pin the closed
# form against brute-force quadrature, including across a pillar.

class TestHWAlphaAverage:
    @staticmethod
    def _quadrature(model, t0, t1, n=200_001):
        """Trapezoidal average of alpha over [t0, t1] as an independent check."""
        s = jnp.linspace(t0, t1, n)
        return float(jnp.trapezoid(hw_alpha(model, s), s) / (t1 - t0))

    @pytest.mark.parametrize("t0,t1", [(0.0, 0.25), (1.2, 1.7), (4.0, 9.0)])
    def test_matches_quadrature(self, model, t0, t1):
        got = float(hw_alpha_average(model, jnp.array(t0), jnp.array(t1)))
        assert got == pytest.approx(self._quadrature(model, t0, t1), rel=1e-8)

    def test_exact_across_a_curve_pillar(self, model):
        """The motivating case: an interval straddling a pillar discontinuity.

        The pillars sit at annual spacing, so [0.8, 1.3] contains the jump in
        f^M(0, t). Midpoint sampling is only first-order accurate here; the
        closed form must still match quadrature.
        """
        got = float(hw_alpha_average(model, jnp.array(0.8), jnp.array(1.3)))
        assert got == pytest.approx(self._quadrature(model, 0.8, 1.3), rel=1e-7)

    def test_forward_part_telescopes_to_the_discount_ratio(self, model):
        """The market-forward half integrates to ln P(0,t0) - ln P(0,t1)."""
        a = float(model.mean_reversion)
        sigma = float(model.volatility)
        t0, t1 = 1.5, 4.5

        def convexity_integral(t):
            return (
                t
                - 2.0 * (1.0 - float(jnp.exp(-a * t))) / a
                + (1.0 - float(jnp.exp(-2.0 * a * t))) / (2.0 * a)
            )

        convexity = (
            sigma**2 / (2.0 * a**2)
            * (convexity_integral(t1) - convexity_integral(t0))
            / (t1 - t0)
        )
        forward = float(
            hw_alpha_average(model, jnp.array(t0), jnp.array(t1))
        ) - convexity
        expected = float(
            jnp.log(hw_market_df(model, jnp.array(t0)))
            - jnp.log(hw_market_df(model, jnp.array(t1)))
        ) / (t1 - t0)
        assert forward == pytest.approx(expected, rel=1e-12)

    def test_shrinks_to_the_pointwise_value(self, model):
        """As the interval collapses, the average tends to alpha(t)."""
        t = 3.0
        for eps in (1e-2, 1e-3, 1e-4):
            avg = float(
                hw_alpha_average(model, jnp.array(t - eps), jnp.array(t + eps))
            )
            assert avg == pytest.approx(
                float(hw_alpha(model, jnp.array(t))), abs=1e-6
            )

    def test_vectorises_over_intervals(self, model):
        edges = jnp.linspace(0.0, 5.0, 11)
        avgs = hw_alpha_average(model, edges[:-1], edges[1:])
        assert avgs.shape == (10,)
        for i in range(10):
            assert float(avgs[i]) == pytest.approx(
                float(hw_alpha_average(model, edges[i], edges[i + 1])), rel=1e-12
            )
