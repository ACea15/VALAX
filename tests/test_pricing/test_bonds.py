"""Tests for fixed income pricing: bonds, duration, convexity, key-rate durations."""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.bonds import ZeroCouponBond, FixedRateBond
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.dates.schedule import generate_schedule
from valax.pricing.analytic.bonds import (
    zero_coupon_bond_price,
    fixed_rate_bond_price,
    fixed_rate_bond_price_from_yield,
    yield_to_maturity,
    modified_duration,
    convexity,
    key_rate_durations,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def ref_date():
    return ymd_to_ordinal(2025, 1, 1)


@pytest.fixture
def flat_curve(ref_date):
    """Flat 5% CC curve."""
    pillars = jnp.array([
        int(ymd_to_ordinal(2025, 1, 1)),
        int(ymd_to_ordinal(2026, 1, 1)),
        int(ymd_to_ordinal(2027, 1, 1)),
        int(ymd_to_ordinal(2028, 1, 1)),
        int(ymd_to_ordinal(2029, 1, 1)),
        int(ymd_to_ordinal(2030, 1, 1)),
    ], dtype=jnp.int32)
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-0.05 * times)
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=dfs,
        reference_date=ref_date,
    )


@pytest.fixture
def zcb(ref_date):
    """5-year zero-coupon bond, face=100."""
    return ZeroCouponBond(
        maturity=ymd_to_ordinal(2030, 1, 1),
        face_value=jnp.array(100.0),
    )


@pytest.fixture
def coupon_bond(ref_date):
    """5-year, 4% semi-annual coupon bond, face=100."""
    sched = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2)
    return FixedRateBond(
        payment_dates=sched,
        settlement_date=ref_date,
        coupon_rate=jnp.array(0.04),
        face_value=jnp.array(100.0),
        frequency=2,
    )


# ── Zero-coupon bond ─────────────────────────────────────────────────

class TestZeroCouponBond:
    def test_price_positive(self, zcb, flat_curve):
        p = zero_coupon_bond_price(zcb, flat_curve)
        assert float(p) > 0.0

    def test_price_less_than_face(self, zcb, flat_curve):
        p = zero_coupon_bond_price(zcb, flat_curve)
        assert float(p) < 100.0

    def test_price_matches_analytical(self, zcb, flat_curve):
        """ZCB price = 100 * exp(-0.05 * 5)."""
        p = zero_coupon_bond_price(zcb, flat_curve)
        expected = 100.0 * jnp.exp(-0.05 * 5.0)
        assert abs(float(p) - float(expected)) < 0.1  # small day-count mismatch ok

    def test_jit(self, zcb, flat_curve):
        p = jax.jit(zero_coupon_bond_price)(zcb, flat_curve)
        assert jnp.isfinite(p)


# ── Fixed-rate bond ──────────────────────────────────────────────────

class TestFixedRateBond:
    def test_price_positive(self, coupon_bond, flat_curve):
        p = fixed_rate_bond_price(coupon_bond, flat_curve)
        assert float(p) > 0.0

    def test_par_bond_near_100(self):
        """A bond with coupon = yield should price near par."""
        ref = ymd_to_ordinal(2025, 1, 1)
        sched = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2)
        bond = FixedRateBond(
            payment_dates=sched,
            settlement_date=ref,
            coupon_rate=jnp.array(0.05),
            face_value=jnp.array(100.0),
            frequency=2,
        )
        # Build curve at 5% CC
        pillars = jnp.array([
            int(ref),
            int(ymd_to_ordinal(2026, 1, 1)),
            int(ymd_to_ordinal(2028, 1, 1)),
            int(ymd_to_ordinal(2030, 1, 1)),
        ], dtype=jnp.int32)
        times = (pillars - int(ref)).astype(jnp.float64) / 365.0
        dfs = jnp.exp(-0.05 * times)
        curve = DiscountCurve(pillar_dates=pillars, discount_factors=dfs, reference_date=ref)
        p = fixed_rate_bond_price(bond, curve)
        # Not exactly 100 due to CC vs discrete compounding, but close
        assert abs(float(p) - 100.0) < 2.0

    def test_higher_coupon_higher_price(self, flat_curve):
        """Bond with higher coupon should have higher price."""
        ref = ymd_to_ordinal(2025, 1, 1)
        sched = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2)
        bond_low = FixedRateBond(
            payment_dates=sched, settlement_date=ref,
            coupon_rate=jnp.array(0.02), face_value=jnp.array(100.0), frequency=2,
        )
        bond_high = FixedRateBond(
            payment_dates=sched, settlement_date=ref,
            coupon_rate=jnp.array(0.08), face_value=jnp.array(100.0), frequency=2,
        )
        p_low = fixed_rate_bond_price(bond_low, flat_curve)
        p_high = fixed_rate_bond_price(bond_high, flat_curve)
        assert float(p_high) > float(p_low)

    def test_jit(self, coupon_bond, flat_curve):
        p = jax.jit(fixed_rate_bond_price)(coupon_bond, flat_curve)
        assert jnp.isfinite(p)


# ── Yield-based pricing ──────────────────────────────────────────────

class TestYieldPricing:
    def test_par_bond(self):
        """Bond priced at coupon rate yield should be at par."""
        ref = ymd_to_ordinal(2025, 1, 1)
        sched = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2)
        bond = FixedRateBond(
            payment_dates=sched, settlement_date=ref,
            coupon_rate=jnp.array(0.06), face_value=jnp.array(100.0), frequency=2,
        )
        p = fixed_rate_bond_price_from_yield(bond, jnp.array(0.06))
        assert abs(float(p) - 100.0) < 1e-6

    def test_discount_bond(self):
        """Yield > coupon => price < par."""
        ref = ymd_to_ordinal(2025, 1, 1)
        sched = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2)
        bond = FixedRateBond(
            payment_dates=sched, settlement_date=ref,
            coupon_rate=jnp.array(0.04), face_value=jnp.array(100.0), frequency=2,
        )
        p = fixed_rate_bond_price_from_yield(bond, jnp.array(0.06))
        assert float(p) < 100.0

    def test_premium_bond(self):
        """Yield < coupon => price > par."""
        ref = ymd_to_ordinal(2025, 1, 1)
        sched = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2)
        bond = FixedRateBond(
            payment_dates=sched, settlement_date=ref,
            coupon_rate=jnp.array(0.06), face_value=jnp.array(100.0), frequency=2,
        )
        p = fixed_rate_bond_price_from_yield(bond, jnp.array(0.04))
        assert float(p) > 100.0


# ── Yield-to-maturity ────────────────────────────────────────────────

class TestYTM:
    def test_roundtrip(self):
        """Price at yield y, then solve for y — should recover."""
        ref = ymd_to_ordinal(2025, 1, 1)
        sched = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2)
        bond = FixedRateBond(
            payment_dates=sched, settlement_date=ref,
            coupon_rate=jnp.array(0.05), face_value=jnp.array(100.0), frequency=2,
        )
        target_yield = jnp.array(0.06)
        price = fixed_rate_bond_price_from_yield(bond, target_yield)
        recovered = yield_to_maturity(bond, price)
        assert abs(float(recovered) - float(target_yield)) < 1e-8

    def test_par_bond_ytm_equals_coupon(self):
        ref = ymd_to_ordinal(2025, 1, 1)
        sched = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2)
        bond = FixedRateBond(
            payment_dates=sched, settlement_date=ref,
            coupon_rate=jnp.array(0.05), face_value=jnp.array(100.0), frequency=2,
        )
        ytm = yield_to_maturity(bond, jnp.array(100.0))
        assert abs(float(ytm) - 0.05) < 1e-8


# ── Risk measures via autodiff ────────────────────────────────────────

class TestRiskMeasures:
    def test_modified_duration_positive(self):
        ref = ymd_to_ordinal(2025, 1, 1)
        sched = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2)
        bond = FixedRateBond(
            payment_dates=sched, settlement_date=ref,
            coupon_rate=jnp.array(0.05), face_value=jnp.array(100.0), frequency=2,
        )
        md = modified_duration(bond, jnp.array(0.05))
        assert float(md) > 0.0

    def test_duration_increases_with_maturity(self):
        """Longer bonds should have higher duration."""
        ref = ymd_to_ordinal(2025, 1, 1)
        sched_2y = generate_schedule(2025, 1, 1, 2027, 1, 1, frequency=2)
        sched_10y = generate_schedule(2025, 1, 1, 2035, 1, 1, frequency=2)
        bond_2y = FixedRateBond(
            payment_dates=sched_2y, settlement_date=ref,
            coupon_rate=jnp.array(0.05), face_value=jnp.array(100.0), frequency=2,
        )
        bond_10y = FixedRateBond(
            payment_dates=sched_10y, settlement_date=ref,
            coupon_rate=jnp.array(0.05), face_value=jnp.array(100.0), frequency=2,
        )
        d_2y = modified_duration(bond_2y, jnp.array(0.05))
        d_10y = modified_duration(bond_10y, jnp.array(0.05))
        assert float(d_10y) > float(d_2y)

    def test_convexity_positive(self):
        ref = ymd_to_ordinal(2025, 1, 1)
        sched = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2)
        bond = FixedRateBond(
            payment_dates=sched, settlement_date=ref,
            coupon_rate=jnp.array(0.05), face_value=jnp.array(100.0), frequency=2,
        )
        c = convexity(bond, jnp.array(0.05))
        assert float(c) > 0.0

    def test_modified_duration_analytical(self):
        """Compare autodiff duration against textbook formula.

        For a par bond (ytm = coupon), modified duration has a known formula:
        MD = (1/y) * [1 - 1/(1+y/f)^n]  (for par bond)
        """
        ref = ymd_to_ordinal(2025, 1, 1)
        sched = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2)
        bond = FixedRateBond(
            payment_dates=sched, settlement_date=ref,
            coupon_rate=jnp.array(0.06), face_value=jnp.array(100.0), frequency=2,
        )
        y = jnp.array(0.06)
        md_autodiff = float(modified_duration(bond, y))

        # Analytical: for par bond
        f = 2
        n = 10  # 5 years * 2
        analytical = (1.0 / 0.06) * (1.0 - 1.0 / (1.0 + 0.06 / f) ** n)
        assert abs(md_autodiff - analytical) < 0.05

    def test_key_rate_durations_shape(self, coupon_bond, flat_curve):
        krd = key_rate_durations(coupon_bond, flat_curve)
        assert krd.shape == flat_curve.discount_factors.shape

    def test_key_rate_durations_finite(self, coupon_bond, flat_curve):
        krd = key_rate_durations(coupon_bond, flat_curve)
        assert jnp.all(jnp.isfinite(krd))

    def test_key_rate_durations_sum_approx_duration(self, coupon_bond, flat_curve):
        """Sum of KRDs should approximate total duration."""
        krd = key_rate_durations(coupon_bond, flat_curve)
        ytm = yield_to_maturity(
            coupon_bond,
            fixed_rate_bond_price(coupon_bond, flat_curve),
        )
        md = modified_duration(coupon_bond, ytm)
        # On a flat curve, sum of KRDs ≈ modified duration (loose tolerance)
        assert abs(float(jnp.sum(krd)) - float(md)) < 1.0
