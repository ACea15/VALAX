"""Tests for the smoothed positive part (``valax.pricing.pde.payoff_smoothing``).

Two layers:

* **The helpers** — exact cell average for a linear ``f`` (1-D kernel) and
  zero integrated bias (2-D kernel); :math:`C^1` at the branch switch; the
  :math:`\\delta \\to 0` limit; finite gradients at ``f = 0``; the grid
  half-variation on linear fields (1-D, non-uniform, 2-D).
* **What it is for** — the short-rate Bermudan PDE price is :math:`C^1` in
  its inputs with smoothing on: central finite differences agree across
  steps and with ``jax.grad`` (pointwise sampling gives a piecewise-smooth
  price whose FD depends on whether the step straddles a node crossing), a
  fine scan has no jumps in its first differences, and the European
  swaption converges at a clean second order to the closed form.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.analytic.swaptions import _annuity
from valax.pricing.pde import PDEConfig, PDEConfig2D, pde_price_dispatch
from valax.pricing.pde.payoff_smoothing import (
    bias_neutral_positive_part,
    cell_half_variation,
    positive_part,
    project_max,
    project_min,
    smooth_positive_part,
)

jax.config.update("jax_enable_x64", True)


# ── The helper ───────────────────────────────────────────────────────


class TestSmoothPositivePart:
    @pytest.mark.parametrize("f0", [-0.7, -0.3, 0.0, 0.2, 0.49, 0.8])
    @pytest.mark.parametrize("slope", [1.0, -2.5])
    def test_exact_cell_average_for_linear_f(self, f0, slope):
        h = 0.4
        delta = abs(slope) * h / 2.0
        u = (np.arange(200_000) + 0.5) / 200_000 - 0.5  # midpoint rule on the cell
        average = np.mean(np.maximum(f0 + slope * h * u, 0.0))
        got = float(smooth_positive_part(jnp.asarray(f0), jnp.asarray(delta)))
        assert got == pytest.approx(average, abs=1e-10)

    def test_c1_at_the_branch_switch(self):
        delta = 0.3
        dfun = jax.grad(smooth_positive_part, argnums=(0, 1))
        for edge in (-delta, delta):
            for eps in (1e-9, -1e-9):
                f = jnp.asarray(edge + eps)
                value = float(smooth_positive_part(f, jnp.asarray(delta)))
                assert value == pytest.approx(max(edge, 0.0), abs=1e-8)
            lo = dfun(jnp.asarray(edge - 1e-9), jnp.asarray(delta))
            hi = dfun(jnp.asarray(edge + 1e-9), jnp.asarray(delta))
            for a, b in zip(lo, hi):
                assert float(a) == pytest.approx(float(b), abs=1e-8)

    def test_delta_to_zero_recovers_max(self):
        f = jnp.linspace(-1.0, 1.0, 41)
        for delta in (1e-6, 1e-12, 0.0):
            got = smooth_positive_part(f, jnp.asarray(delta))
            np.testing.assert_allclose(got, jnp.maximum(f, 0.0), atol=delta)

    @pytest.mark.parametrize("delta", [0.0, 1e-40, 0.1])
    def test_gradient_finite_at_zero(self, delta):
        g = jax.grad(smooth_positive_part, argnums=(0, 1))(
            jnp.asarray(0.0), jnp.asarray(delta)
        )
        assert all(np.isfinite(float(x)) for x in g)
        if delta == 0.1:
            assert float(g[0]) == pytest.approx(0.5)

    def test_bounds(self):
        """max(f, 0) <= cell average <= max(f, 0) + delta / 4."""
        f = jnp.linspace(-1.0, 1.0, 201)
        delta = jnp.asarray(0.3)
        got = smooth_positive_part(f, delta)
        assert bool(jnp.all(got >= jnp.maximum(f, 0.0) - 1e-15))
        assert bool(jnp.all(got <= jnp.maximum(f, 0.0) + delta / 4 + 1e-15))


class TestBiasNeutralPositivePart:
    """The 2-D kernel: C^1, same support, zero integrated bias."""

    def test_zero_integrated_bias(self):
        delta = 0.3
        f = np.linspace(-delta, delta, 400_001)
        s = np.asarray(bias_neutral_positive_part(jnp.asarray(f), jnp.asarray(delta)))
        assert np.trapezoid(s - np.maximum(f, 0.0), f) == pytest.approx(0.0, abs=1e-12)
        # ...where the cell average carries delta^2 / 6.
        c = np.asarray(smooth_positive_part(jnp.asarray(f), jnp.asarray(delta)))
        assert np.trapezoid(c - np.maximum(f, 0.0), f) == pytest.approx(delta**2 / 6)

    def test_c1_at_the_branch_switch(self):
        delta = 0.3
        dfun = jax.grad(bias_neutral_positive_part, argnums=(0, 1))
        for edge in (-delta, delta):
            value = float(bias_neutral_positive_part(jnp.asarray(edge - 1e-9), jnp.asarray(delta)))
            assert value == pytest.approx(max(edge, 0.0), abs=1e-8)
            lo = dfun(jnp.asarray(edge - 1e-9), jnp.asarray(delta))
            hi = dfun(jnp.asarray(edge + 1e-9), jnp.asarray(delta))
            for a, b in zip(lo, hi):
                assert float(a) == pytest.approx(float(b), abs=1e-7)

    def test_bounds_and_limits(self):
        f = jnp.linspace(-1.0, 1.0, 2001)
        delta = jnp.asarray(0.5)
        excess = bias_neutral_positive_part(f, delta) - jnp.maximum(f, 0.0)
        assert float(excess.min()) == pytest.approx(-0.02558 * 0.5, rel=1e-3)
        assert float(excess.max()) == pytest.approx(0.09375 * 0.5, rel=1e-6)
        np.testing.assert_array_equal(
            bias_neutral_positive_part(f, jnp.asarray(0.0)), jnp.maximum(f, 0.0)
        )

    @pytest.mark.parametrize("delta", [0.0, 0.1])
    def test_gradient_finite_at_zero(self, delta):
        g = jax.grad(bias_neutral_positive_part, argnums=(0, 1))(
            jnp.asarray(0.0), jnp.asarray(delta)
        )
        assert all(np.isfinite(float(x)) for x in g)

    def test_positive_part_picks_the_kernel_by_rank(self):
        x = jnp.linspace(-1.0, 1.0, 11)
        f2 = x[:, None] + 0.0 * x[None, :]
        np.testing.assert_allclose(
            positive_part(f2, True),
            bias_neutral_positive_part(f2, cell_half_variation(f2)),
        )
        np.testing.assert_allclose(
            positive_part(x, True), smooth_positive_part(x, cell_half_variation(x))
        )


class TestCellHalfVariation:
    def test_linear_1d_uniform(self):
        x = jnp.linspace(-1.0, 1.0, 21)
        h = float(x[1] - x[0])
        delta = cell_half_variation(3.0 * x - 0.2)
        np.testing.assert_allclose(delta, 3.0 * h / 2.0, rtol=1e-12)

    def test_linear_1d_nonuniform_is_half_slope_times_control_volume(self):
        x = jnp.sinh(jnp.linspace(-2.0, 2.0, 15))
        delta = cell_half_variation(2.0 * x)
        control_volume = (x[2:] - x[:-2]) / 2.0
        np.testing.assert_allclose(delta[1:-1], 2.0 * control_volume / 2.0, rtol=1e-12)

    def test_2d_combines_axes_in_quadrature(self):
        x = jnp.linspace(-1.0, 1.0, 11)
        y = jnp.linspace(-2.0, 2.0, 9)
        hx, hy = float(x[1] - x[0]), float(y[1] - y[0])
        f = 1.5 * x[:, None] - 0.5 * y[None, :]
        expected = np.hypot(1.5 * hx / 2.0, 0.5 * hy / 2.0)
        np.testing.assert_allclose(cell_half_variation(f), expected, rtol=1e-12)

    def test_gradient_finite_on_a_flat_2d_field(self):
        g = jax.grad(lambda c: jnp.sum(cell_half_variation(c * jnp.ones((5, 4)))))(1.0)
        assert np.isfinite(float(g))


class TestSwitchesAndProjections:
    def test_off_is_pointwise_max(self):
        f = jnp.linspace(-1.0, 1.0, 11)
        assert bool(jnp.all(positive_part(f, False) == jnp.maximum(f, 0.0)))

    def test_projection_identities(self):
        c = jnp.linspace(0.0, 2.0, 21)
        e = jnp.full_like(c, 1.03)
        for smooth in (False, True):
            hi, lo = project_max(c, e, smooth), project_min(c, e, smooth)
            assert bool(jnp.all(hi >= jnp.maximum(c, e) - 1e-15))
            assert bool(jnp.all(lo <= jnp.minimum(c, e) + 1e-15))
            # Away from the crossing both are exact.
            far = jnp.abs(c - 1.03) > 0.2
            np.testing.assert_array_equal(hi[far], jnp.maximum(c, e)[far])
            np.testing.assert_array_equal(lo[far], jnp.minimum(c, e)[far])


# ── The PDE price is C^1 in its inputs ───────────────────────────────

_REF = int(ymd_to_ordinal(2026, 1, 1))
_DC = "act_365"


def _curve():
    pillars = jnp.array([_REF + int(round(y * 365)) for y in range(26)], jnp.int32)
    t = (pillars - _REF).astype(jnp.float64) / 365.0
    zero = 0.03 + 0.01 * jnp.tanh(t / 4.0)
    return DiscountCurve(
        pillar_dates=pillars, discount_factors=jnp.exp(-zero * t),
        reference_date=jnp.int32(_REF), day_count=_DC,
    )


_CURVE = _curve()
_EX = jnp.array([int(ymd_to_ordinal(y, 1, 1)) for y in range(2028, 2033)], jnp.int32)
_FIXED = jnp.array([int(ymd_to_ordinal(y, 1, 1)) for y in range(2029, 2034)], jnp.int32)
_ATM = float((_CURVE(_EX[0]) - _CURVE(_FIXED[-1])) / _annuity(_EX[0], _FIXED, _CURVE, _DC))
_BERMUDAN = BermudanSwaption(
    exercise_dates=_EX, fixed_dates=_FIXED, strike=jnp.asarray(_ATM),
    notional=jnp.array(1.0), is_payer=True, day_count=_DC,
)
_HW_CFG = PDEConfig(n_spot=101, n_time=60, spot_range=5.0)
_G2_CFG = PDEConfig2D(n_x=41, n_y=41, n_time=40, x_range=5.0)


def _hw_price_in_strike(config):
    @eqx.filter_jit
    def f(strike):
        trade = eqx.tree_at(lambda b: b.strike, _BERMUDAN, strike)
        model = HullWhiteModel(
            mean_reversion=jnp.array(0.05), volatility=jnp.array(0.01),
            initial_curve=_CURVE,
        )
        return pde_price_dispatch(trade, model, config).price

    return f


def _g2_price_in_sigma(config):
    @eqx.filter_jit
    def f(sigma):
        model = G2PPModel(
            mean_reversion_x=jnp.array(0.5), mean_reversion_y=jnp.array(0.1),
            volatility_x=sigma, volatility_y=jnp.array(0.008),
            correlation=jnp.array(-0.7), initial_curve=_CURVE,
        )
        return pde_price_dispatch(_BERMUDAN, model, config).price

    return f


def _fd_by_step(f, x0, rel_steps):
    return np.array([
        float((f(jnp.asarray(x0 + h * x0)) - f(jnp.asarray(x0 - h * x0))) / (2 * h * x0))
        for h in rel_steps
    ])


# The base point for G2++ sits next to a node crossing of the pointwise
# price (found by a 401-point scan of sigma_x on [0.0095, 0.0105]).
_G2_SIGMA_AT_A_KINK = 0.0095225
_STEPS = (1e-3, 1e-4, 1e-5, 1e-6)


class TestBermudanIsC1:
    def test_hw_fd_consistent_across_steps_and_equals_grad(self):
        f = _hw_price_in_strike(_HW_CFG)
        grad = float(jax.grad(f)(jnp.asarray(_ATM)))
        fds = _fd_by_step(f, _ATM, _STEPS)
        np.testing.assert_allclose(fds, grad, rtol=1e-6)

    def test_hw_scan_has_no_jumps_in_its_first_differences(self):
        strikes = jnp.linspace(_ATM - 0.002, _ATM + 0.002, 161)
        prices = {
            smooth: np.asarray(jax.vmap(_hw_price_in_strike(
                PDEConfig(n_spot=101, n_time=60, spot_range=5.0, smooth_payoff=smooth)
            ))(strikes))
            for smooth in (True, False)
        }

        def jump_ratio(v):
            d2 = np.abs(np.diff(np.diff(v)))
            return d2.max() / np.median(d2)

        assert jump_ratio(prices[True]) < 3.0
        # Anti-vacuity: the pointwise price over the same scan does jump.
        assert jump_ratio(prices[False]) > 100.0

    def test_g2pp_fd_consistent_across_steps_and_equals_grad(self):
        f = _g2_price_in_sigma(_G2_CFG)
        x0 = _G2_SIGMA_AT_A_KINK
        grad = float(jax.grad(f)(jnp.asarray(x0)))
        fds = _fd_by_step(f, x0, _STEPS[1:])
        np.testing.assert_allclose(fds, grad, rtol=1e-6)
        # The 1e-3 step sees genuine curvature (O(h^2)), not a kink.
        assert _fd_by_step(f, x0, _STEPS[:1])[0] == pytest.approx(grad, rel=5e-5)

    def test_g2pp_pointwise_price_is_kinked_there(self):
        """Anti-vacuity: without smoothing the same point straddles a kink."""
        f = _g2_price_in_sigma(PDEConfig2D(
            n_x=41, n_y=41, n_time=40, x_range=5.0, smooth_payoff=False
        ))
        fds = _fd_by_step(f, _G2_SIGMA_AT_A_KINK, _STEPS)
        assert np.ptp(fds) / abs(np.mean(fds)) > 1e-4


class TestSecondOrderConvergence:
    def test_hw_european_error_falls_by_four(self):
        """Smoothed: the error vs Jamshidian falls ~4x per refinement."""
        swaption = Swaption(
            expiry_date=jnp.int32(int(ymd_to_ordinal(2029, 1, 1))),
            fixed_dates=jnp.array(
                [int(ymd_to_ordinal(2029 + k, 1, 1)) for k in range(1, 6)], jnp.int32
            ),
            strike=jnp.asarray(0.045), notional=jnp.array(1.0), is_payer=True,
            day_count=_DC,
        )
        model = HullWhiteModel(
            mean_reversion=jnp.array(0.05), volatility=jnp.array(0.01),
            initial_curve=_CURVE,
        )
        exact = float(hw_swaption_price(swaption, model))
        errors = [
            float(pde_price_dispatch(
                swaption, model, PDEConfig(n_spot=n, n_time=n // 2, spot_range=6.0)
            ).price) - exact
            for n in (100, 200, 400)
        ]
        for coarse, fine in zip(errors, errors[1:]):
            assert coarse / fine == pytest.approx(4.0, rel=0.15)
