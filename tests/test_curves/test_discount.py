"""Tests for discount curve interpolation and rates."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve, forward_rate, zero_rate
from valax.dates.daycounts import ymd_to_ordinal


@pytest.fixture
def flat_curve():
    """Flat 5% continuously-compounded curve."""
    ref = ymd_to_ordinal(2025, 1, 1)
    rate = 0.05
    pillars = jnp.array([
        int(ymd_to_ordinal(2025, 1, 1)),
        int(ymd_to_ordinal(2026, 1, 1)),
        int(ymd_to_ordinal(2027, 1, 1)),
        int(ymd_to_ordinal(2030, 1, 1)),
        int(ymd_to_ordinal(2035, 1, 1)),
    ], dtype=jnp.int32)
    times = (pillars - int(ref)).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-rate * times)
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=dfs,
        reference_date=ref,
    )


class TestDiscountCurve:
    def test_at_pillars(self, flat_curve):
        """DF at pillar dates should match exactly."""
        dfs = flat_curve(flat_curve.pillar_dates)
        expected = flat_curve.discount_factors
        assert jnp.allclose(dfs, expected, atol=1e-10)

    def test_interpolation_flat_curve(self, flat_curve):
        """On a flat curve, interpolated DF should match exp(-r*t)."""
        mid_date = ymd_to_ordinal(2028, 7, 1)
        df = flat_curve(mid_date)
        ref = int(flat_curve.reference_date)
        t = (int(mid_date) - ref) / 365.0
        expected = jnp.exp(-0.05 * t)
        assert abs(float(df) - float(expected)) < 1e-6

    def test_df_at_reference_is_one(self, flat_curve):
        df = flat_curve(flat_curve.reference_date)
        assert abs(float(df) - 1.0) < 1e-10

    def test_df_decreasing(self, flat_curve):
        """Discount factors should decrease with time for positive rates."""
        dates = jnp.array([
            int(ymd_to_ordinal(2025, 7, 1)),
            int(ymd_to_ordinal(2026, 7, 1)),
            int(ymd_to_ordinal(2028, 1, 1)),
        ], dtype=jnp.int32)
        dfs = flat_curve(dates)
        for i in range(len(dfs) - 1):
            assert float(dfs[i]) > float(dfs[i + 1])

    def test_jit_compatible(self, flat_curve):
        jitted = eqx.filter_jit(flat_curve)
        date = ymd_to_ordinal(2026, 6, 15)
        df = jitted(date)
        assert jnp.isfinite(df)


class TestForwardRate:
    def test_flat_curve_forward_equals_spot(self, flat_curve):
        """On a flat curve, forward rate ≈ spot rate."""
        start = ymd_to_ordinal(2026, 1, 1)
        end = ymd_to_ordinal(2027, 1, 1)
        fwd = forward_rate(flat_curve, start, end)
        # For 5% CC rate, simply-compounded 1Y forward ≈ exp(0.05)-1 ≈ 0.05127
        expected = jnp.exp(0.05) - 1.0
        assert abs(float(fwd) - float(expected)) < 1e-3


class TestZeroRate:
    def test_flat_curve(self, flat_curve):
        """Zero rate on flat 5% curve should be ~5%."""
        date = ymd_to_ordinal(2027, 1, 1)
        r = zero_rate(flat_curve, date)
        assert abs(float(r) - 0.05) < 1e-4


class TestCurveDifferentiability:
    def test_grad_through_curve(self, flat_curve):
        """Can differentiate price w.r.t. discount factors."""
        date = ymd_to_ordinal(2027, 1, 1)

        def price_fn(dfs):
            c = DiscountCurve(
                pillar_dates=flat_curve.pillar_dates,
                discount_factors=dfs,
                reference_date=flat_curve.reference_date,
            )
            return c(date)

        grad = jax.grad(price_fn)(flat_curve.discount_factors)
        assert grad.shape == flat_curve.discount_factors.shape
        assert jnp.all(jnp.isfinite(grad))
