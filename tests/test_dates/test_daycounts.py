"""Tests for day count conventions and date utilities."""

import jax.numpy as jnp
import pytest

from valax.dates.daycounts import (
    DAY_COUNT_DENOMINATORS,
    act_360,
    act_365,
    ordinal_from_year_fraction,
    thirty_360,
    year_fraction,
    ymd_to_ordinal,
)


class TestYmdToOrdinal:
    def test_epoch(self):
        assert int(ymd_to_ordinal(1970, 1, 1)) == 0

    def test_known_date(self):
        # 2025-01-01 = ordinal 20089 (days since 1970-01-01)
        import datetime
        expected = (datetime.date(2025, 1, 1) - datetime.date(1970, 1, 1)).days
        assert int(ymd_to_ordinal(2025, 1, 1)) == expected

    def test_leap_year(self):
        import datetime
        d = datetime.date(2024, 2, 29)
        expected = (d - datetime.date(1970, 1, 1)).days
        assert int(ymd_to_ordinal(2024, 2, 29)) == expected

    def test_roundtrip_multiple_dates(self):
        import datetime
        test_dates = [
            (2000, 3, 15), (2010, 7, 4), (2023, 12, 31),
            (1999, 1, 1), (2030, 6, 30),
        ]
        for y, m, d in test_dates:
            expected = (datetime.date(y, m, d) - datetime.date(1970, 1, 1)).days
            assert int(ymd_to_ordinal(y, m, d)) == expected, f"Failed for {y}-{m}-{d}"


class TestAct365:
    def test_one_year(self):
        start = ymd_to_ordinal(2024, 1, 1)
        end = ymd_to_ordinal(2025, 1, 1)
        frac = act_365(start, end)
        # 2024 is a leap year: 366 days
        assert abs(float(frac) - 366.0 / 365.0) < 1e-10

    def test_half_year(self):
        start = ymd_to_ordinal(2025, 1, 1)
        end = ymd_to_ordinal(2025, 7, 2)
        frac = act_365(start, end)
        # 31+28+31+30+31+1 = 182 days
        assert abs(float(frac) - 182.0 / 365.0) < 1e-10


class TestAct360:
    def test_90_days(self):
        start = ymd_to_ordinal(2025, 1, 1)
        end = ymd_to_ordinal(2025, 4, 1)
        frac = act_360(start, end)
        assert abs(float(frac) - 90.0 / 360.0) < 1e-10


class TestThirty360:
    def test_standard(self):
        start = ymd_to_ordinal(2025, 1, 15)
        end = ymd_to_ordinal(2025, 7, 15)
        frac = thirty_360(start, end)
        # 30/360: 6 months = 180/360 = 0.5
        assert abs(float(frac) - 0.5) < 1e-10

    def test_full_year(self):
        start = ymd_to_ordinal(2025, 3, 1)
        end = ymd_to_ordinal(2026, 3, 1)
        frac = thirty_360(start, end)
        assert abs(float(frac) - 1.0) < 1e-10


class TestYearFraction:
    def test_dispatch(self):
        start = ymd_to_ordinal(2025, 1, 1)
        end = ymd_to_ordinal(2025, 7, 1)
        f365 = year_fraction(start, end, "act_365")
        f360 = year_fraction(start, end, "act_360")
        # act_360 should give a larger fraction for the same period
        assert float(f360) > float(f365)


class TestOrdinalFromYearFraction:
    """The inverse: year fraction → (fractional) ordinal.

    The bridge between an equity instrument's year-fraction ``expiry`` and
    a curve's date-ordinal index. Returns a *fractional* ordinal so the
    round trip is exact and the path stays differentiable.
    """

    @pytest.mark.parametrize("convention", sorted(DAY_COUNT_DENOMINATORS))
    @pytest.mark.parametrize("years", [0.25, 0.5, 1.0, 2.0, 7.5])
    def test_round_trip_is_exact(self, convention, years):
        start = ymd_to_ordinal(2026, 1, 1)
        target = jnp.asarray(years)
        ordinal = ordinal_from_year_fraction(start, target, convention)
        assert float(year_fraction(start, ordinal, convention)) == pytest.approx(
            years, abs=1e-12,
        )

    def test_uses_the_conventions_denominator(self):
        """Not a hard-coded 365: act_360 lands 360 days out, act_365 365."""
        start = ymd_to_ordinal(2026, 1, 1)
        one_year = jnp.asarray(1.0)
        assert float(
            ordinal_from_year_fraction(start, one_year, "act_360") - start
        ) == pytest.approx(360.0)
        assert float(
            ordinal_from_year_fraction(start, one_year, "act_365") - start
        ) == pytest.approx(365.0)
        assert float(
            ordinal_from_year_fraction(start, one_year, "act_act") - start
        ) == pytest.approx(365.25)

    def test_is_differentiable_in_the_year_fraction(self):
        """A rate projected at this horizon must keep its gradient."""
        import jax

        start = ymd_to_ordinal(2026, 1, 1)
        g = jax.grad(
            lambda t: ordinal_from_year_fraction(start, t, "act_365")
        )(jnp.asarray(1.5))
        assert float(g) == pytest.approx(365.0)

    def test_thirty_360_has_no_closed_form_inverse(self):
        start = ymd_to_ordinal(2026, 1, 1)
        with pytest.raises(ValueError, match="no closed-form inverse"):
            ordinal_from_year_fraction(start, jnp.asarray(1.0), "30_360")

    def test_unknown_convention_raises(self):
        start = ymd_to_ordinal(2026, 1, 1)
        with pytest.raises(ValueError, match="Unknown day count"):
            ordinal_from_year_fraction(start, jnp.asarray(1.0), "act_366")
