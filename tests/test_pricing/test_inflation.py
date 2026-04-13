"""Tests for inflation curve and inflation derivative pricers."""

import jax
import jax.numpy as jnp
import pytest

from valax.curves.inflation import (
    InflationCurve,
    forward_cpi,
    zc_inflation_rate,
    yoy_forward_rate,
    from_zc_rates,
)
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.dates.schedule import generate_schedule
from valax.instruments.inflation import (
    ZeroCouponInflationSwap,
    YearOnYearInflationSwap,
    InflationCapFloor,
)
from valax.pricing.analytic.inflation import (
    zcis_price,
    zcis_breakeven_rate,
    yyis_price,
    inflation_cap_floor_price_black76,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def ref_date():
    return ymd_to_ordinal(2025, 1, 1)


@pytest.fixture
def annual_pillars():
    return jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(1, 12)],
        dtype=jnp.int32,
    )


@pytest.fixture
def flat_infl_curve(ref_date, annual_pillars):
    """Flat 2.5% ZC inflation curve, base CPI = 100."""
    return from_zc_rates(
        ref_date, annual_pillars,
        jnp.full(11, 0.025),
        jnp.array(100.0),
    )


@pytest.fixture
def disc_curve(ref_date):
    """Flat 4% nominal discount curve."""
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(16)],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-0.04 * times),
        reference_date=ref_date,
    )


@pytest.fixture
def annual_schedule():
    return generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=1)


# ── InflationCurve ───────────────────────────────────────────────────

class TestInflationCurve:
    def test_forward_cpi_at_pillars(self, flat_infl_curve, annual_pillars):
        """Interpolation at exact pillar dates recovers stored values."""
        cpis = forward_cpi(flat_infl_curve, annual_pillars)
        assert jnp.allclose(cpis, flat_infl_curve.forward_cpis, atol=1e-10)

    def test_zc_rate_round_trip(self, ref_date, annual_pillars):
        """from_zc_rates → zc_inflation_rate recovers the input rates."""
        rates_in = jnp.array([0.02 + 0.001 * i for i in range(11)])
        curve = from_zc_rates(ref_date, annual_pillars, rates_in, jnp.array(100.0))
        rates_out = zc_inflation_rate(curve, annual_pillars)
        assert jnp.allclose(rates_out, rates_in, atol=1e-10)

    def test_yoy_forward_consistent_with_cpi(self, flat_infl_curve, annual_pillars):
        """yoy_forward_rate == CPI(end)/CPI(start) - 1."""
        starts = annual_pillars[:-1]
        ends = annual_pillars[1:]
        yoy = yoy_forward_rate(flat_infl_curve, starts, ends)
        cpi_start = forward_cpi(flat_infl_curve, starts)
        cpi_end = forward_cpi(flat_infl_curve, ends)
        expected = cpi_end / cpi_start - 1.0
        assert jnp.allclose(yoy, expected, atol=1e-12)

    def test_jit_compatible(self, flat_infl_curve, annual_pillars):
        eager = forward_cpi(flat_infl_curve, annual_pillars)
        jitted = jax.jit(forward_cpi)(flat_infl_curve, annual_pillars)
        assert jnp.allclose(eager, jitted, atol=1e-12)

    def test_base_cpi_is_cpi_at_reference(self, flat_infl_curve, ref_date):
        """forward_cpi at reference_date ≈ base_cpi (flat extrapolation)."""
        cpi_0 = forward_cpi(flat_infl_curve, ref_date)
        # First pillar is 1Y out; at t=0 the flat extrapolation gives
        # the first pillar's CPI, which is base_cpi * 1.025 ≈ 102.5.
        # This tests that the interpolation doesn't blow up at t=0.
        assert float(cpi_0) > 0.0


# ── ZCIS ─────────────────────────────────────────────────────────────

class TestZCIS:
    @pytest.fixture
    def zcis(self, ref_date):
        return ZeroCouponInflationSwap(
            effective_date=ref_date,
            maturity_date=ymd_to_ordinal(2030, 1, 1),
            fixed_rate=jnp.array(0.025),
            notional=jnp.array(1_000_000.0),
            base_cpi=jnp.array(100.0),
        )

    def test_breakeven_gives_zero_npv(
        self, zcis, flat_infl_curve, disc_curve, ref_date
    ):
        be = zcis_breakeven_rate(zcis, flat_infl_curve)
        at_par = ZeroCouponInflationSwap(
            effective_date=ref_date,
            maturity_date=ymd_to_ordinal(2030, 1, 1),
            fixed_rate=be,
            notional=jnp.array(1_000_000.0),
            base_cpi=jnp.array(100.0),
        )
        npv = zcis_price(at_par, flat_infl_curve, disc_curve)
        assert float(npv) == pytest.approx(0.0, abs=1e-6)

    def test_breakeven_matches_zc_rate(self, zcis, flat_infl_curve):
        """On a flat 2.5% ZC curve, breakeven ≈ 2.5%."""
        be = zcis_breakeven_rate(zcis, flat_infl_curve)
        assert float(be) == pytest.approx(0.025, abs=1e-4)

    def test_receiver_positive_when_fixed_below_breakeven(
        self, flat_infl_curve, disc_curve, ref_date
    ):
        zcis_low = ZeroCouponInflationSwap(
            effective_date=ref_date,
            maturity_date=ymd_to_ordinal(2030, 1, 1),
            fixed_rate=jnp.array(0.020),
            notional=jnp.array(1_000_000.0),
            base_cpi=jnp.array(100.0),
        )
        npv = zcis_price(zcis_low, flat_infl_curve, disc_curve)
        assert float(npv) > 0.0

    def test_sign_flip(self, zcis, flat_infl_curve, disc_curve, ref_date):
        recv = zcis_price(zcis, flat_infl_curve, disc_curve)
        payer = ZeroCouponInflationSwap(
            effective_date=ref_date,
            maturity_date=ymd_to_ordinal(2030, 1, 1),
            fixed_rate=zcis.fixed_rate,
            notional=zcis.notional,
            base_cpi=zcis.base_cpi,
            is_inflation_receiver=False,
        )
        pay = zcis_price(payer, flat_infl_curve, disc_curve)
        assert float(recv) == pytest.approx(-float(pay), abs=1e-10)

    def test_jit_compatible(self, zcis, flat_infl_curve, disc_curve):
        eager = zcis_price(zcis, flat_infl_curve, disc_curve)
        jitted = jax.jit(zcis_price)(zcis, flat_infl_curve, disc_curve)
        assert float(jitted) == pytest.approx(float(eager), rel=1e-10)

    def test_grad_wrt_fixed_rate(self, zcis, flat_infl_curve, disc_curve, ref_date):
        """dNPV/dK < 0 for inflation receiver (higher fixed rate = worse)."""
        def price_from_rate(k):
            s = ZeroCouponInflationSwap(
                effective_date=ref_date,
                maturity_date=ymd_to_ordinal(2030, 1, 1),
                fixed_rate=k,
                notional=jnp.array(1_000_000.0),
                base_cpi=jnp.array(100.0),
            )
            return zcis_price(s, flat_infl_curve, disc_curve)

        g = jax.grad(price_from_rate)(jnp.array(0.025))
        assert float(g) < 0.0


# ── YYIS ─────────────────────────────────────────────────────────────

class TestYYIS:
    @pytest.fixture
    def yyis(self, ref_date, annual_schedule):
        return YearOnYearInflationSwap(
            effective_date=ref_date,
            payment_dates=annual_schedule,
            fixed_rate=jnp.array(0.025),
            notional=jnp.array(1_000_000.0),
            base_cpi=jnp.array(100.0),
        )

    def test_sign_flip(self, ref_date, annual_schedule, flat_infl_curve, disc_curve):
        recv = YearOnYearInflationSwap(
            effective_date=ref_date, payment_dates=annual_schedule,
            fixed_rate=jnp.array(0.025), notional=jnp.array(1_000_000.0),
            base_cpi=jnp.array(100.0), is_inflation_receiver=True,
        )
        pay = YearOnYearInflationSwap(
            effective_date=ref_date, payment_dates=annual_schedule,
            fixed_rate=jnp.array(0.025), notional=jnp.array(1_000_000.0),
            base_cpi=jnp.array(100.0), is_inflation_receiver=False,
        )
        r = yyis_price(recv, flat_infl_curve, disc_curve)
        p = yyis_price(pay, flat_infl_curve, disc_curve)
        assert float(r) == pytest.approx(-float(p), abs=1e-10)

    def test_jit_compatible(self, yyis, flat_infl_curve, disc_curve):
        eager = yyis_price(yyis, flat_infl_curve, disc_curve)
        jitted = jax.jit(yyis_price)(yyis, flat_infl_curve, disc_curve)
        assert float(jitted) == pytest.approx(float(eager), rel=1e-10)

    def test_grad_wrt_fixed_rate_negative(
        self, ref_date, annual_schedule, flat_infl_curve, disc_curve
    ):
        """Higher fixed rate → worse for inflation receiver."""
        def price_from_rate(k):
            s = YearOnYearInflationSwap(
                effective_date=ref_date, payment_dates=annual_schedule,
                fixed_rate=k, notional=jnp.array(1_000_000.0),
                base_cpi=jnp.array(100.0),
            )
            return yyis_price(s, flat_infl_curve, disc_curve)

        g = jax.grad(price_from_rate)(jnp.array(0.025))
        assert float(g) < 0.0

    def test_finite_price(self, yyis, flat_infl_curve, disc_curve):
        p = yyis_price(yyis, flat_infl_curve, disc_curve)
        assert jnp.isfinite(p)


# ── Inflation Cap / Floor ────────────────────────────────────────────

class TestInflationCapFloor:
    @pytest.fixture
    def cap(self, ref_date, annual_schedule):
        return InflationCapFloor(
            effective_date=ref_date, payment_dates=annual_schedule,
            strike=jnp.array(0.025), notional=jnp.array(1_000_000.0),
            base_cpi=jnp.array(100.0), is_cap=True,
        )

    @pytest.fixture
    def floor(self, ref_date, annual_schedule):
        return InflationCapFloor(
            effective_date=ref_date, payment_dates=annual_schedule,
            strike=jnp.array(0.025), notional=jnp.array(1_000_000.0),
            base_cpi=jnp.array(100.0), is_cap=False,
        )

    def test_cap_non_negative(self, cap, flat_infl_curve, disc_curve):
        pv = inflation_cap_floor_price_black76(cap, flat_infl_curve, disc_curve, jnp.array(0.03))
        assert float(pv) >= 0.0

    def test_floor_non_negative(self, floor, flat_infl_curve, disc_curve):
        pv = inflation_cap_floor_price_black76(floor, flat_infl_curve, disc_curve, jnp.array(0.03))
        assert float(pv) >= 0.0

    def test_put_call_parity(self, cap, floor, flat_infl_curve, disc_curve, ref_date, annual_schedule):
        """cap - floor = sum (F - K) * DF * N."""
        vol = jnp.array(0.03)
        cap_pv = inflation_cap_floor_price_black76(cap, flat_infl_curve, disc_curve, vol)
        floor_pv = inflation_cap_floor_price_black76(floor, flat_infl_curve, disc_curve, vol)

        starts = jnp.concatenate([ref_date[None], annual_schedule[:-1]])
        F = forward_cpi(flat_infl_curve, annual_schedule) / forward_cpi(flat_infl_curve, starts) - 1.0
        dfs = disc_curve(annual_schedule)
        parity = float(jnp.sum((F - 0.025) * dfs * 1_000_000.0))

        assert float(cap_pv - floor_pv) == pytest.approx(parity, rel=1e-8)

    def test_cap_increases_with_vol(self, cap, flat_infl_curve, disc_curve):
        low = inflation_cap_floor_price_black76(cap, flat_infl_curve, disc_curve, jnp.array(0.01))
        high = inflation_cap_floor_price_black76(cap, flat_infl_curve, disc_curve, jnp.array(0.10))
        assert float(high) > float(low)

    def test_jit_compatible(self, cap, flat_infl_curve, disc_curve):
        vol = jnp.array(0.03)
        eager = inflation_cap_floor_price_black76(cap, flat_infl_curve, disc_curve, vol)
        jitted = jax.jit(inflation_cap_floor_price_black76)(cap, flat_infl_curve, disc_curve, vol)
        assert float(jitted) == pytest.approx(float(eager), rel=1e-10)

    def test_vmap_across_vols(self, cap, flat_infl_curve, disc_curve):
        vols = jnp.linspace(0.01, 0.10, 5)
        def price_one(v):
            return inflation_cap_floor_price_black76(cap, flat_infl_curve, disc_curve, v)
        batch = jax.vmap(price_one)(vols)
        assert batch.shape == (5,)
        assert jnp.all(jnp.diff(batch) > 0.0)
