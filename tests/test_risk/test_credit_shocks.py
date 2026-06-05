"""Tests for credit shock primitives operating on SurvivalCurve."""

import jax
import jax.numpy as jnp
import pytest

from valax.curves.survival import SurvivalCurve, from_hazard_rates
from valax.dates.daycounts import ymd_to_ordinal
from valax.risk.shocks import (
    bump_hazard_rates,
    key_rate_hazard_bump,
    parallel_credit_spread_shift,
)


@pytest.fixture
def flat_curve():
    """Flat 2% hazard curve, pillars at 0, 1, 2, 5 years."""
    ref = ymd_to_ordinal(2026, 1, 1)
    pillars = jnp.array([
        ref,
        ymd_to_ordinal(2027, 1, 1),
        ymd_to_ordinal(2028, 1, 1),
        ymd_to_ordinal(2031, 1, 1),
    ])
    hazards = jnp.array([0.0, 0.02, 0.02, 0.02])
    return from_hazard_rates(ref, pillars, hazards)


class TestBumpHazardRates:
    def test_zero_bump_identity(self, flat_curve):
        bumps = jnp.zeros(4)
        bumped = bump_hazard_rates(flat_curve, bumps)
        assert jnp.allclose(
            bumped.survival_probabilities,
            flat_curve.survival_probabilities,
            atol=1e-12,
        )

    def test_positive_bump_reduces_survival(self, flat_curve):
        bumps = jnp.array([0.0, 0.01, 0.01, 0.01])
        bumped = bump_hazard_rates(flat_curve, bumps)
        # Survival at every later pillar must decrease
        diff = bumped.survival_probabilities[1:] - flat_curve.survival_probabilities[1:]
        assert jnp.all(diff < 0.0)

    def test_uniform_bump_matches_analytic_formula(self, flat_curve):
        """h = 0.02 → 0.03 should give S(t) ≈ exp(-0.03 t)."""
        bumps = jnp.array([0.0, 0.01, 0.01, 0.01])
        bumped = bump_hazard_rates(flat_curve, bumps)
        # Expected survival at 5y with h=0.03
        pillar_times = jnp.array([0.0, 1.0, 2.0, 5.0])
        expected = jnp.exp(-0.03 * pillar_times)
        assert jnp.allclose(
            bumped.survival_probabilities, expected, atol=5e-4,
        )

    def test_preserves_pillar_dates_and_reference(self, flat_curve):
        bumps = jnp.array([0.0, 0.005, 0.005, 0.005])
        bumped = bump_hazard_rates(flat_curve, bumps)
        assert jnp.array_equal(bumped.pillar_dates, flat_curve.pillar_dates)
        assert bumped.reference_date == flat_curve.reference_date
        assert bumped.day_count == flat_curve.day_count


class TestParallelCreditSpreadShift:
    def test_spread_to_hazard_conversion(self, flat_curve):
        """Δs = 6 bp, R = 0.4 ⇒ Δh = 10 bp."""
        spread_bump = jnp.array(0.0006)
        recovery = jnp.array(0.4)
        bumped = parallel_credit_spread_shift(flat_curve, spread_bump, recovery)

        # Reference: same effect as bumping all hazards by 10 bp
        reference = bump_hazard_rates(
            flat_curve,
            jnp.array([0.0, 0.001, 0.001, 0.001]),
        )
        assert jnp.allclose(
            bumped.survival_probabilities,
            reference.survival_probabilities,
            atol=1e-12,
        )

    def test_widening_reduces_survival(self, flat_curve):
        bumped = parallel_credit_spread_shift(
            flat_curve, jnp.array(0.0050), jnp.array(0.4),
        )
        diff = bumped.survival_probabilities[1:] - flat_curve.survival_probabilities[1:]
        assert jnp.all(diff < 0.0)


class TestKeyRateHazardBump:
    def test_only_targeted_interval_changes(self, flat_curve):
        """Bumping interval 2 should change S at pillars 2, 3 but not 0, 1."""
        bumped = key_rate_hazard_bump(flat_curve, pillar_index=2, bump=jnp.array(0.01))
        # Pillar 0 and 1: unchanged
        assert jnp.isclose(
            bumped.survival_probabilities[0],
            flat_curve.survival_probabilities[0],
            atol=1e-12,
        )
        assert jnp.isclose(
            bumped.survival_probabilities[1],
            flat_curve.survival_probabilities[1],
            atol=1e-12,
        )
        # Pillar 2 and 3: changed (lower)
        assert bumped.survival_probabilities[2] < flat_curve.survival_probabilities[2]
        assert bumped.survival_probabilities[3] < flat_curve.survival_probabilities[3]

    def test_grad_w_r_t_bump(self, flat_curve):
        """Sensitivity of a survival probability to a hazard bump is well-defined."""

        def pv(bump):
            bumped = key_rate_hazard_bump(flat_curve, 2, bump)
            return bumped.survival_probabilities[3]

        g = jax.grad(pv)(jnp.array(0.0))
        # Negative: bumping hazard reduces survival
        assert g < 0.0
        assert jnp.isfinite(g)
