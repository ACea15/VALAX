"""Tests for Hull-White trinomial tree: construction and callable/puttable bonds."""

import jax
import jax.numpy as jnp
import pytest

from valax.models.hull_white import HullWhiteModel
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.dates.schedule import generate_schedule
from valax.instruments.bonds import FixedRateBond, CallableBond, PuttableBond
from valax.pricing.lattice.hull_white_tree import (
    build_hull_white_tree,
    callable_bond_price,
    puttable_bond_price,
)
from valax.pricing.analytic.bonds import fixed_rate_bond_price


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def ref_date():
    return ymd_to_ordinal(2025, 1, 1)


@pytest.fixture
def flat_curve(ref_date):
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(16)],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-0.05 * times),
        reference_date=ref_date,
    )


@pytest.fixture
def model(flat_curve):
    return HullWhiteModel(
        mean_reversion=jnp.array(0.10),
        volatility=jnp.array(0.01),
        initial_curve=flat_curve,
    )


@pytest.fixture
def bond_schedule():
    return generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2)


@pytest.fixture
def straight_bond(ref_date, bond_schedule):
    return FixedRateBond(
        payment_dates=bond_schedule,
        settlement_date=ref_date,
        coupon_rate=jnp.array(0.04),
        face_value=jnp.array(100.0),
        frequency=2,
    )


@pytest.fixture
def call_dates():
    return jnp.array([
        int(ymd_to_ordinal(2027, 1, 1)),
        int(ymd_to_ordinal(2028, 1, 1)),
        int(ymd_to_ordinal(2029, 1, 1)),
    ], dtype=jnp.int32)


@pytest.fixture
def callable_bond(ref_date, bond_schedule, call_dates):
    return CallableBond(
        payment_dates=bond_schedule,
        settlement_date=ref_date,
        coupon_rate=jnp.array(0.04),
        face_value=jnp.array(100.0),
        call_dates=call_dates,
        call_prices=jnp.array([1.0, 1.0, 1.0]),
        frequency=2,
    )


@pytest.fixture
def puttable_bond(ref_date, bond_schedule, call_dates):
    return PuttableBond(
        payment_dates=bond_schedule,
        settlement_date=ref_date,
        coupon_rate=jnp.array(0.04),
        face_value=jnp.array(100.0),
        put_dates=call_dates,
        put_prices=jnp.array([1.0, 1.0, 1.0]),
        frequency=2,
    )


# ── Tree construction ────────────────────────────────────────────────

class TestTreeConstruction:
    def test_tree_shape(self, model):
        tree = build_hull_white_tree(model, 5.0, 100)
        n_states = 2 * tree.j_max + 1
        assert tree.rates.shape == (101, n_states)
        assert tree.alpha.shape == (100,)
        assert tree.probs.shape == (n_states, 3)
        assert tree.targets.shape == (n_states, 3)

    def test_probabilities_sum_to_one(self, model):
        tree = build_hull_white_tree(model, 5.0, 100)
        sums = jnp.sum(tree.probs, axis=1)
        assert jnp.allclose(sums, 1.0, atol=1e-12)

    def test_probabilities_non_negative(self, model):
        tree = build_hull_white_tree(model, 5.0, 100)
        assert jnp.all(tree.probs >= -1e-14)

    def test_alpha_reasonable(self, model):
        """Alpha values should be near the flat rate (0.05) for a flat curve."""
        tree = build_hull_white_tree(model, 5.0, 100)
        assert jnp.all(tree.alpha > 0.03)
        assert jnp.all(tree.alpha < 0.08)


# ── Callable bond ────────────────────────────────────────────────────

class TestCallableBond:
    def test_callable_less_than_straight(
        self, callable_bond, straight_bond, model, flat_curve
    ):
        """Callable bond price < straight bond price (call reduces holder value)."""
        p_call = callable_bond_price(callable_bond, model, n_steps=100)
        p_str = fixed_rate_bond_price(straight_bond, flat_curve)
        assert float(p_call) < float(p_str)

    def test_call_option_value_positive(
        self, callable_bond, straight_bond, model, flat_curve
    ):
        """The embedded call option has positive value."""
        p_call = callable_bond_price(callable_bond, model, n_steps=100)
        p_str = fixed_rate_bond_price(straight_bond, flat_curve)
        option_val = float(p_str) - float(p_call)
        assert option_val > 0.0

    def test_high_call_price_approaches_straight(
        self, ref_date, bond_schedule, call_dates, straight_bond, model, flat_curve
    ):
        """With an extremely high call price, callable ≈ straight."""
        cb_high = CallableBond(
            payment_dates=bond_schedule,
            settlement_date=ref_date,
            coupon_rate=jnp.array(0.04),
            face_value=jnp.array(100.0),
            call_dates=call_dates,
            call_prices=jnp.array([100.0, 100.0, 100.0]),
            frequency=2,
        )
        p_high = callable_bond_price(cb_high, model, n_steps=100)
        p_str = fixed_rate_bond_price(straight_bond, flat_curve)
        assert float(p_high) == pytest.approx(float(p_str), rel=0.005)

    def test_convergence(self, callable_bond, model):
        """Finer tree gives a stable price (< 1% diff between 50 and 200 steps)."""
        p50 = callable_bond_price(callable_bond, model, n_steps=50)
        p200 = callable_bond_price(callable_bond, model, n_steps=200)
        rel_diff = abs(float(p50) - float(p200)) / float(p200)
        assert rel_diff < 0.01

    def test_price_positive(self, callable_bond, model):
        p = callable_bond_price(callable_bond, model, n_steps=50)
        assert float(p) > 0.0

    def test_grad_wrt_coupon_rate_positive(
        self, ref_date, bond_schedule, call_dates, model
    ):
        """Higher coupon → higher callable bond price."""
        def price_from_coupon(c):
            cb = CallableBond(
                payment_dates=bond_schedule,
                settlement_date=ref_date,
                coupon_rate=c,
                face_value=jnp.array(100.0),
                call_dates=call_dates,
                call_prices=jnp.array([1.0, 1.0, 1.0]),
                frequency=2,
            )
            return callable_bond_price(cb, model, n_steps=50)

        g = jax.grad(price_from_coupon)(jnp.array(0.04))
        assert float(g) > 0.0


# ── Puttable bond ────────────────────────────────────────────────────

class TestPuttableBond:
    def test_puttable_greater_than_straight(
        self, puttable_bond, straight_bond, model, flat_curve
    ):
        """Puttable bond price > straight bond price (put increases holder value)."""
        p_put = puttable_bond_price(puttable_bond, model, n_steps=100)
        p_str = fixed_rate_bond_price(straight_bond, flat_curve)
        assert float(p_put) > float(p_str)

    def test_put_option_value_positive(
        self, puttable_bond, straight_bond, model, flat_curve
    ):
        p_put = puttable_bond_price(puttable_bond, model, n_steps=100)
        p_str = fixed_rate_bond_price(straight_bond, flat_curve)
        assert float(p_put) - float(p_str) > 0.0

    def test_low_put_price_approaches_straight(
        self, ref_date, bond_schedule, call_dates, straight_bond, model, flat_curve
    ):
        """With a very low put price (0), the put is worthless → puttable ≈ straight."""
        pb_low = PuttableBond(
            payment_dates=bond_schedule,
            settlement_date=ref_date,
            coupon_rate=jnp.array(0.04),
            face_value=jnp.array(100.0),
            put_dates=call_dates,
            put_prices=jnp.array([0.001, 0.001, 0.001]),
            frequency=2,
        )
        p_low = puttable_bond_price(pb_low, model, n_steps=100)
        p_str = fixed_rate_bond_price(straight_bond, flat_curve)
        assert float(p_low) == pytest.approx(float(p_str), rel=0.005)

    def test_price_positive(self, puttable_bond, model):
        p = puttable_bond_price(puttable_bond, model, n_steps=50)
        assert float(p) > 0.0
