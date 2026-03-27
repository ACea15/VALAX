"""Tests for shock application to curves and market data."""

import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve, zero_rate
from valax.dates.daycounts import ymd_to_ordinal
from valax.market.data import MarketData
from valax.market.scenario import MarketScenario
from valax.risk.shocks import (
    apply_scenario,
    bump_curve_zero_rates,
    key_rate_bump,
    parallel_shift,
)


@pytest.fixture
def flat_curve():
    """A flat 5% continuously-compounded curve with 5 pillars."""
    ref = ymd_to_ordinal(2026, 1, 1)
    pillars = jnp.array([
        ymd_to_ordinal(2026, 1, 1),
        ymd_to_ordinal(2027, 1, 1),
        ymd_to_ordinal(2028, 1, 1),
        ymd_to_ordinal(2029, 1, 1),
        ymd_to_ordinal(2030, 1, 1),
    ])
    rate = 0.05
    times = (pillars - ref).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-rate * times)
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=dfs,
        reference_date=ref,
    )


class TestBumpCurveZeroRates:
    def test_zero_bump_is_identity(self, flat_curve):
        bumped = bump_curve_zero_rates(flat_curve, jnp.zeros(5))
        assert jnp.allclose(bumped.discount_factors, flat_curve.discount_factors, atol=1e-12)

    def test_parallel_bump_shifts_zero_rates(self, flat_curve):
        bump = 0.01  # +100bp
        bumped = bump_curve_zero_rates(flat_curve, jnp.full(5, bump))
        # Zero rate at pillar 1 (1Y) should be ~6%
        zr = zero_rate(bumped, flat_curve.pillar_dates[1])
        assert jnp.isclose(zr, 0.06, atol=1e-4)

    def test_single_pillar_bump(self, flat_curve):
        bumps = jnp.zeros(5).at[2].set(0.02)
        bumped = bump_curve_zero_rates(flat_curve, bumps)
        # Pillar 2 should be bumped, pillar 1 should be unchanged
        zr1 = zero_rate(bumped, flat_curve.pillar_dates[1])
        zr2 = zero_rate(bumped, flat_curve.pillar_dates[2])
        assert jnp.isclose(zr1, 0.05, atol=1e-4)
        assert jnp.isclose(zr2, 0.07, atol=1e-4)

    def test_differentiable(self, flat_curve):
        """jax.grad through bump_curve_zero_rates should work."""
        def f(bumps):
            c = bump_curve_zero_rates(flat_curve, bumps)
            return c(flat_curve.pillar_dates[2])  # DF at 2Y

        grad = jax.grad(f)(jnp.zeros(5))
        # Gradient should be non-zero at pillar 2 (and possibly neighbours
        # due to interpolation), zero far away
        assert grad[2] != 0.0


class TestParallelShift:
    def test_shifts_all_rates(self, flat_curve):
        bumped = parallel_shift(flat_curve, jnp.array(0.01))
        for i in range(1, 5):
            zr = zero_rate(bumped, flat_curve.pillar_dates[i])
            assert jnp.isclose(zr, 0.06, atol=1e-4)


class TestKeyRateBump:
    def test_bumps_single_pillar(self, flat_curve):
        bumped = key_rate_bump(flat_curve, 3, jnp.array(0.02))
        zr1 = zero_rate(bumped, flat_curve.pillar_dates[1])
        zr3 = zero_rate(bumped, flat_curve.pillar_dates[3])
        assert jnp.isclose(zr1, 0.05, atol=1e-4)
        assert jnp.isclose(zr3, 0.07, atol=1e-4)


class TestApplyScenario:
    def test_additive_spots(self, flat_curve):
        base = MarketData(
            spots=jnp.array([100.0, 200.0]),
            vols=jnp.array([0.2, 0.3]),
            dividends=jnp.array([0.01, 0.02]),
            discount_curve=flat_curve,
        )
        scenario = MarketScenario(
            spot_shocks=jnp.array([5.0, -10.0]),
            vol_shocks=jnp.array([0.01, -0.02]),
            rate_shocks=jnp.zeros(5),
            dividend_shocks=jnp.array([0.0, 0.005]),
        )
        shocked = apply_scenario(base, scenario)
        assert jnp.allclose(shocked.spots, jnp.array([105.0, 190.0]))
        assert jnp.allclose(shocked.vols, jnp.array([0.21, 0.28]))
        assert jnp.allclose(shocked.dividends, jnp.array([0.01, 0.025]))

    def test_multiplicative_spots(self, flat_curve):
        base = MarketData(
            spots=jnp.array([100.0]),
            vols=jnp.array([0.2]),
            dividends=jnp.array([0.0]),
            discount_curve=flat_curve,
        )
        scenario = MarketScenario(
            spot_shocks=jnp.array([-0.10]),  # -10% return
            vol_shocks=jnp.zeros(1),
            rate_shocks=jnp.zeros(5),
            dividend_shocks=jnp.zeros(1),
            multiplicative=True,
        )
        shocked = apply_scenario(base, scenario)
        assert jnp.isclose(shocked.spots[0], 90.0)

    def test_rate_shocks_applied(self, flat_curve):
        base = MarketData(
            spots=jnp.array([100.0]),
            vols=jnp.array([0.2]),
            dividends=jnp.array([0.0]),
            discount_curve=flat_curve,
        )
        scenario = MarketScenario(
            spot_shocks=jnp.zeros(1),
            vol_shocks=jnp.zeros(1),
            rate_shocks=jnp.full(5, 0.01),
            dividend_shocks=jnp.zeros(1),
        )
        shocked = apply_scenario(base, scenario)
        zr = zero_rate(shocked.discount_curve, flat_curve.pillar_dates[1])
        assert jnp.isclose(zr, 0.06, atol=1e-4)
