"""Tests for caplet, floorlet, cap, and floor pricing."""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.rates import Caplet, Cap
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal

from valax.pricing.analytic.caplets import (
    caplet_price_black76,
    caplet_price_bachelier,
    cap_price_black76,
    cap_price_bachelier,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def ref_date():
    return ymd_to_ordinal(2025, 1, 1)


@pytest.fixture
def flat_curve(ref_date):
    """Flat 5% continuously-compounded curve out to 5 years."""
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
    return DiscountCurve(pillar_dates=pillars, discount_factors=dfs, reference_date=ref_date)


@pytest.fixture
def caplet(ref_date):
    """3m x 6m caplet: fixes in 3 months, covers 3m→6m period."""
    fixing = ymd_to_ordinal(2025, 4, 1)   # 3m
    start = ymd_to_ordinal(2025, 4, 1)
    end = ymd_to_ordinal(2025, 7, 1)      # 6m
    return Caplet(
        fixing_date=fixing,
        start_date=start,
        end_date=end,
        strike=jnp.array(0.05),
        notional=jnp.array(1_000_000.0),
        is_cap=True,
    )


@pytest.fixture
def floorlet(ref_date):
    fixing = ymd_to_ordinal(2025, 4, 1)
    start = ymd_to_ordinal(2025, 4, 1)
    end = ymd_to_ordinal(2025, 7, 1)
    return Caplet(
        fixing_date=fixing,
        start_date=start,
        end_date=end,
        strike=jnp.array(0.05),
        notional=jnp.array(1_000_000.0),
        is_cap=False,
    )


@pytest.fixture
def cap(ref_date):
    """1-year cap with quarterly resets (4 caplets), all fixings in the future."""
    start_dates = jnp.array([
        int(ymd_to_ordinal(2025, 4, 1)),
        int(ymd_to_ordinal(2025, 7, 1)),
        int(ymd_to_ordinal(2025, 10, 1)),
        int(ymd_to_ordinal(2026, 1, 1)),
    ], dtype=jnp.int32)
    end_dates = jnp.array([
        int(ymd_to_ordinal(2025, 7, 1)),
        int(ymd_to_ordinal(2025, 10, 1)),
        int(ymd_to_ordinal(2026, 1, 1)),
        int(ymd_to_ordinal(2026, 4, 1)),
    ], dtype=jnp.int32)
    return Cap(
        fixing_dates=start_dates,   # fix at period start
        start_dates=start_dates,
        end_dates=end_dates,
        strike=jnp.array(0.05),
        notional=jnp.array(1_000_000.0),
        is_cap=True,
    )


@pytest.fixture
def floor(ref_date):
    """1-year floor with quarterly resets (4 floorlets), same schedule as cap."""
    start_dates = jnp.array([
        int(ymd_to_ordinal(2025, 4, 1)),
        int(ymd_to_ordinal(2025, 7, 1)),
        int(ymd_to_ordinal(2025, 10, 1)),
        int(ymd_to_ordinal(2026, 1, 1)),
    ], dtype=jnp.int32)
    end_dates = jnp.array([
        int(ymd_to_ordinal(2025, 7, 1)),
        int(ymd_to_ordinal(2025, 10, 1)),
        int(ymd_to_ordinal(2026, 1, 1)),
        int(ymd_to_ordinal(2026, 4, 1)),
    ], dtype=jnp.int32)
    return Cap(
        fixing_dates=start_dates,
        start_dates=start_dates,
        end_dates=end_dates,
        strike=jnp.array(0.05),
        notional=jnp.array(1_000_000.0),
        is_cap=False,
    )


# ── Caplet: Black-76 ──────────────────────────────────────────────────

class TestCapletBlack76:
    def test_price_positive(self, caplet, flat_curve):
        p = caplet_price_black76(caplet, flat_curve, jnp.array(0.20))
        assert float(p) > 0.0

    def test_put_call_parity(self, caplet, floorlet, flat_curve):
        """Caplet - Floorlet = PV(F - K) * tau * notional."""
        vol = jnp.array(0.20)
        cap_p = caplet_price_black76(caplet, flat_curve, vol)
        flo_p = caplet_price_black76(floorlet, flat_curve, vol)

        from valax.dates.daycounts import year_fraction
        tau = year_fraction(caplet.start_date, caplet.end_date, caplet.day_count)
        df_start = flat_curve(caplet.start_date)
        df_end = flat_curve(caplet.end_date)
        F = (df_start / df_end - 1.0) / tau
        P = flat_curve(caplet.end_date)
        expected_diff = float(caplet.notional * tau * P * (F - caplet.strike))

        assert abs(float(cap_p) - float(flo_p) - expected_diff) < 1e-4

    def test_higher_vol_higher_price(self, caplet, flat_curve):
        p_low = caplet_price_black76(caplet, flat_curve, jnp.array(0.10))
        p_high = caplet_price_black76(caplet, flat_curve, jnp.array(0.40))
        assert float(p_high) > float(p_low)

    def test_deep_itm_approximates_intrinsic(self, flat_curve):
        """Very low strike => caplet ≈ PV(F - K) * tau * notional."""
        fixing = ymd_to_ordinal(2025, 4, 1)
        start = ymd_to_ordinal(2025, 4, 1)
        end = ymd_to_ordinal(2025, 7, 1)
        caplet = Caplet(
            fixing_date=fixing, start_date=start, end_date=end,
            strike=jnp.array(0.001),    # very low strike
            notional=jnp.array(1_000_000.0),
        )
        from valax.dates.daycounts import year_fraction
        tau = year_fraction(start, end, caplet.day_count)
        df_start = flat_curve(start)
        df_end = flat_curve(end)
        F = (df_start / df_end - 1.0) / tau
        P = flat_curve(end)
        intrinsic = float(caplet.notional * tau * P * (F - caplet.strike))

        p = caplet_price_black76(caplet, flat_curve, jnp.array(0.20))
        assert abs(float(p) - intrinsic) / intrinsic < 0.01  # within 1% of intrinsic

    def test_jit(self, caplet, flat_curve):
        p = jax.jit(caplet_price_black76)(caplet, flat_curve, jnp.array(0.20))
        assert jnp.isfinite(p)

    def test_grad_vol(self, caplet, flat_curve):
        """Vega (dPrice/dvol) should be positive for caplet."""
        vega = jax.grad(caplet_price_black76, argnums=2)(caplet, flat_curve, jnp.array(0.20))
        assert float(vega) > 0.0


# ── Caplet: Bachelier ─────────────────────────────────────────────────

class TestCapletBachelier:
    def test_price_positive(self, caplet, flat_curve):
        p = caplet_price_bachelier(caplet, flat_curve, jnp.array(0.005))
        assert float(p) > 0.0

    def test_put_call_parity(self, caplet, floorlet, flat_curve):
        """Bachelier caplet - floorlet = PV(F - K) * tau * notional."""
        vol = jnp.array(0.005)
        cap_p = caplet_price_bachelier(caplet, flat_curve, vol)
        flo_p = caplet_price_bachelier(floorlet, flat_curve, vol)

        from valax.dates.daycounts import year_fraction
        tau = year_fraction(caplet.start_date, caplet.end_date, caplet.day_count)
        df_start = flat_curve(caplet.start_date)
        df_end = flat_curve(caplet.end_date)
        F = (df_start / df_end - 1.0) / tau
        P = flat_curve(caplet.end_date)
        expected_diff = float(caplet.notional * tau * P * (F - caplet.strike))

        assert abs(float(cap_p) - float(flo_p) - expected_diff) < 1e-4

    def test_higher_vol_higher_price(self, caplet, flat_curve):
        p_low = caplet_price_bachelier(caplet, flat_curve, jnp.array(0.002))
        p_high = caplet_price_bachelier(caplet, flat_curve, jnp.array(0.010))
        assert float(p_high) > float(p_low)

    def test_jit(self, caplet, flat_curve):
        p = jax.jit(caplet_price_bachelier)(caplet, flat_curve, jnp.array(0.005))
        assert jnp.isfinite(p)

    def test_grad_vol(self, caplet, flat_curve):
        vega = jax.grad(caplet_price_bachelier, argnums=2)(caplet, flat_curve, jnp.array(0.005))
        assert float(vega) > 0.0


# ── Cap / Floor: Black-76 ─────────────────────────────────────────────

class TestCapBlack76:
    def test_price_positive(self, cap, flat_curve):
        p = cap_price_black76(cap, flat_curve, jnp.array(0.20))
        assert float(p) > 0.0

    def test_cap_floor_parity(self, cap, floor, flat_curve):
        """Cap - Floor = PV(floating leg - fixed leg) = sum of forward-rate differences."""
        vol = jnp.array(0.20)
        cap_p = cap_price_black76(cap, flat_curve, vol)
        flo_p = cap_price_black76(floor, flat_curve, vol)

        # Parity: cap - floor = sum_i [notional * tau_i * P_i * (F_i - K)]
        from valax.dates.daycounts import year_fraction
        tau = year_fraction(cap.start_dates, cap.end_dates, cap.day_count)
        df_start = flat_curve(cap.start_dates)
        df_end = flat_curve(cap.end_dates)
        F = (df_start / df_end - 1.0) / tau
        P = flat_curve(cap.end_dates)
        expected = float(jnp.sum(cap.notional * tau * P * (F - cap.strike)))

        assert abs(float(cap_p) - float(flo_p) - expected) < 1e-2

    def test_cap_geq_any_caplet(self, cap, flat_curve):
        """Cap price >= any individual caplet price."""
        vol = jnp.array(0.20)
        cap_p = cap_price_black76(cap, flat_curve, vol)

        # Price first caplet individually
        from valax.instruments.rates import Caplet
        caplet = Caplet(
            fixing_date=cap.fixing_dates[0],
            start_date=cap.start_dates[0],
            end_date=cap.end_dates[0],
            strike=cap.strike,
            notional=cap.notional,
        )
        single = caplet_price_black76(caplet, flat_curve, vol)
        assert float(cap_p) >= float(single)

    def test_higher_vol_higher_price(self, cap, flat_curve):
        p_low = cap_price_black76(cap, flat_curve, jnp.array(0.10))
        p_high = cap_price_black76(cap, flat_curve, jnp.array(0.40))
        assert float(p_high) > float(p_low)

    def test_jit(self, cap, flat_curve):
        p = jax.jit(cap_price_black76)(cap, flat_curve, jnp.array(0.20))
        assert jnp.isfinite(p)

    def test_grad_vol(self, cap, flat_curve):
        vega = jax.grad(cap_price_black76, argnums=2)(cap, flat_curve, jnp.array(0.20))
        assert float(vega) > 0.0


# ── Cap / Floor: Bachelier ────────────────────────────────────────────

class TestCapBachelier:
    def test_price_positive(self, cap, flat_curve):
        p = cap_price_bachelier(cap, flat_curve, jnp.array(0.005))
        assert float(p) > 0.0

    def test_cap_floor_parity(self, cap, floor, flat_curve):
        vol = jnp.array(0.005)
        cap_p = cap_price_bachelier(cap, flat_curve, vol)
        flo_p = cap_price_bachelier(floor, flat_curve, vol)

        from valax.dates.daycounts import year_fraction
        tau = year_fraction(cap.start_dates, cap.end_dates, cap.day_count)
        df_start = flat_curve(cap.start_dates)
        df_end = flat_curve(cap.end_dates)
        F = (df_start / df_end - 1.0) / tau
        P = flat_curve(cap.end_dates)
        expected = float(jnp.sum(cap.notional * tau * P * (F - cap.strike)))

        assert abs(float(cap_p) - float(flo_p) - expected) < 1e-2

    def test_jit(self, cap, flat_curve):
        p = jax.jit(cap_price_bachelier)(cap, flat_curve, jnp.array(0.005))
        assert jnp.isfinite(p)
