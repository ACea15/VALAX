"""Tests for coupon schedule generation."""

import jax.numpy as jnp
import pytest

from valax.dates.schedule import generate_schedule
from valax.dates.daycounts import ymd_to_ordinal


class TestGenerateSchedule:
    def test_semiannual_2y(self):
        """2-year bond, semi-annual => 4 payment dates."""
        sched = generate_schedule(2025, 1, 15, 2027, 1, 15, frequency=2)
        assert sched.shape == (4,)
        # Last date should be maturity
        assert int(sched[-1]) == int(ymd_to_ordinal(2027, 1, 15))

    def test_annual_5y(self):
        """5-year bond, annual => 5 payment dates."""
        sched = generate_schedule(2025, 6, 1, 2030, 6, 1, frequency=1)
        assert sched.shape == (5,)

    def test_quarterly_1y(self):
        """1-year bond, quarterly => 4 payment dates."""
        sched = generate_schedule(2025, 3, 15, 2026, 3, 15, frequency=4)
        assert sched.shape == (4,)

    def test_ascending_order(self):
        sched = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2)
        for i in range(len(sched) - 1):
            assert int(sched[i]) < int(sched[i + 1])
