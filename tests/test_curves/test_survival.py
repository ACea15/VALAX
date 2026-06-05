"""Tests for SurvivalCurve and credit-curve constructors."""

import jax
import jax.numpy as jnp
import pytest

from valax.curves.survival import (
    SurvivalCurve,
    from_cds_spreads,
    from_hazard_rates,
    hazard_rate,
    piecewise_hazards,
    survival_probability,
)
from valax.dates.daycounts import ymd_to_ordinal


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def simple_curve():
    ref = ymd_to_ordinal(2026, 1, 1)
    pillars = jnp.array([
        ymd_to_ordinal(2026, 1, 1),  # t=0
        ymd_to_ordinal(2027, 1, 1),  # t=1
        ymd_to_ordinal(2028, 1, 1),  # t=2
        ymd_to_ordinal(2031, 1, 1),  # t=5
    ])
    # Constant hazard of 2% ⇒ S(t) = exp(-0.02 t)
    hazards = jnp.array([0.0, 0.02, 0.02, 0.02])
    return from_hazard_rates(
        reference_date=ref, pillar_dates=pillars, hazards=hazards,
    )


# ── Constructor tests ───────────────────────────────────────────────


class TestFromHazardRates:
    def test_first_survival_is_one(self, simple_curve):
        assert jnp.isclose(simple_curve.survival_probabilities[0], 1.0)

    def test_constant_hazard_exponential_decay(self, simple_curve):
        # S(t) = exp(-0.02 * t)
        pillar_times = jnp.array([0.0, 1.0, 2.0, 5.0])
        expected = jnp.exp(-0.02 * pillar_times)
        # Within day-count tolerance (act_365 with leap year)
        assert jnp.allclose(
            simple_curve.survival_probabilities, expected, atol=5e-4,
        )

    def test_survival_monotone_decreasing(self, simple_curve):
        s = simple_curve.survival_probabilities
        assert jnp.all(jnp.diff(s) <= 0.0)


class TestFromCDSSpreads:
    def test_spreads_to_hazards_via_triangle(self):
        ref = ymd_to_ordinal(2026, 1, 1)
        pillars = jnp.array([
            ref,
            ymd_to_ordinal(2027, 1, 1),
            ymd_to_ordinal(2031, 1, 1),
        ])
        spreads = jnp.array([0.0, 0.012, 0.012])  # 120 bp flat
        R = jnp.array(0.4)
        curve = from_cds_spreads(ref, pillars, spreads, R)
        # h = s / (1-R) = 0.012 / 0.6 = 0.02
        # S(t) = exp(-0.02 t)
        pillar_times = jnp.array([0.0, 1.0, 5.0])
        expected = jnp.exp(-0.02 * pillar_times)
        assert jnp.allclose(
            curve.survival_probabilities, expected, atol=5e-4,
        )

    def test_monotone_decreasing(self):
        ref = ymd_to_ordinal(2026, 1, 1)
        pillars = jnp.array([
            ref,
            ymd_to_ordinal(2027, 1, 1),
            ymd_to_ordinal(2028, 1, 1),
            ymd_to_ordinal(2031, 1, 1),
        ])
        spreads = jnp.array([0.0, 0.010, 0.015, 0.020])
        curve = from_cds_spreads(ref, pillars, spreads)
        assert jnp.all(jnp.diff(curve.survival_probabilities) <= 0.0)


# ── Interpolation tests ─────────────────────────────────────────────


class TestInterpolation:
    def test_at_pillars_returns_pillar_values(self, simple_curve):
        out = simple_curve(simple_curve.pillar_dates)
        assert jnp.allclose(out, simple_curve.survival_probabilities, atol=1e-10)

    def test_constant_hazard_between_pillars(self, simple_curve):
        # Mid-2027 = 1.5 years; with h=0.02 expect S ≈ exp(-0.03)
        mid_date = ymd_to_ordinal(2027, 7, 1)
        S = simple_curve(mid_date)
        # Actual year-fraction is ~1.498 (act_365 across leap year);
        # use a wider tolerance.
        assert jnp.isclose(S, jnp.exp(-0.02 * 1.5), atol=5e-3)

    def test_flat_extrapolation_after_last_pillar(self, simple_curve):
        # Beyond the last pillar (2031), curve flat-extrapolates
        far = ymd_to_ordinal(2035, 1, 1)
        last = simple_curve.survival_probabilities[-1]
        assert jnp.isclose(simple_curve(far), last, atol=1e-10)


# ── Hazard rate helpers ─────────────────────────────────────────────


class TestHazardRate:
    def test_average_hazard_matches_constant_hazard(self, simple_curve):
        date = ymd_to_ordinal(2027, 1, 1)
        h = hazard_rate(simple_curve, date)
        assert jnp.isclose(h, 0.02, atol=2e-4)

    def test_piecewise_hazards(self, simple_curve):
        h = piecewise_hazards(simple_curve)
        assert h.shape == (4,)
        # First slot is sentinel zero; later slots are ~2%
        assert jnp.isclose(h[0], 0.0)
        assert jnp.allclose(h[1:], 0.02, atol=2e-4)


# ── Differentiability ──────────────────────────────────────────────


class TestDifferentiability:
    def test_grad_w_r_t_pillar_survival(self, simple_curve):
        date = ymd_to_ordinal(2027, 1, 1)

        def pv(s):
            # Build a curve from a survival vector and evaluate
            curve = SurvivalCurve(
                pillar_dates=simple_curve.pillar_dates,
                survival_probabilities=s,
                reference_date=simple_curve.reference_date,
                day_count=simple_curve.day_count,
            )
            return survival_probability(curve, date)

        g = jax.grad(pv)(simple_curve.survival_probabilities)
        assert jnp.all(jnp.isfinite(g))
        # PV should rise with each pillar's survival
        assert jnp.any(g > 0.0)
