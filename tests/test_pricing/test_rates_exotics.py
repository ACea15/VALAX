"""Tests for exotic rates pricers: XCCY, TRS, CMS swap/cap/floor, range accrual."""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.rates import (
    CrossCurrencySwap,
    TotalReturnSwap,
    CMSSwap,
    CMSCapFloor,
    RangeAccrual,
)
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.dates.schedule import generate_schedule
from valax.pricing.analytic.rates_exotics import (
    cross_currency_swap_price,
    cross_currency_basis_spread,
    total_return_swap_price,
    cms_swap_price,
    cms_cap_floor_price_black76,
    range_accrual_price_black76,
    _annuity,
    _cms_forward_rates,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def ref_date():
    return ymd_to_ordinal(2025, 1, 1)


def _flat_curve(ref_date, rate, n_years=16):
    """Build a flat CC curve at `rate` with `n_years` annual pillars."""
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(n_years)],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-rate * times)
    return DiscountCurve(
        pillar_dates=pillars, discount_factors=dfs, reference_date=ref_date,
    )


@pytest.fixture
def dom_curve(ref_date):
    """Flat 5% domestic (USD) curve."""
    return _flat_curve(ref_date, 0.05)


@pytest.fixture
def for_curve(ref_date):
    """Flat 2% foreign (EUR) curve."""
    return _flat_curve(ref_date, 0.02)


@pytest.fixture
def quarterly_schedule():
    return generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=4)


# ── Cross-currency swap ─────────────────────────────────────────────

class TestCrossCurrencySwap:
    """XCCY basis swap under single-curve-per-currency + spot conversion."""

    @pytest.fixture
    def spot(self):
        return jnp.array(1.10)  # 1.10 USD per EUR

    @pytest.fixture
    def matched_xccy(self, ref_date, quarterly_schedule, spot):
        """Matched notionals (N_d = spot * N_f), zero spread, exch_notional."""
        return CrossCurrencySwap(
            start_date=ref_date,
            payment_dates=quarterly_schedule,
            maturity_date=quarterly_schedule[-1],
            domestic_notional=spot * jnp.array(100_000_000.0),
            foreign_notional=jnp.array(100_000_000.0),
            basis_spread=jnp.array(0.0),
            exchange_notional=True,
        )

    def test_par_xccy_zero_spread_matched_notionals(
        self, matched_xccy, dom_curve, for_curve, spot
    ):
        """With matched notionals, zero spread, and exchange_notional=True,
        all DF terms cancel and NPV = 0."""
        npv = cross_currency_swap_price(matched_xccy, dom_curve, for_curve, spot)
        assert float(npv) == pytest.approx(0.0, abs=1e-4)

    def test_spread_identity_with_exchange(
        self, ref_date, quarterly_schedule, dom_curve, for_curve, spot
    ):
        """NPV = N_d * s * A_d exactly when exchange_notional=True."""
        s = -0.002
        xccy = CrossCurrencySwap(
            start_date=ref_date,
            payment_dates=quarterly_schedule,
            maturity_date=quarterly_schedule[-1],
            domestic_notional=spot * jnp.array(100_000_000.0),
            foreign_notional=jnp.array(100_000_000.0),
            basis_spread=jnp.array(s),
            exchange_notional=True,
        )
        npv = cross_currency_swap_price(xccy, dom_curve, for_curve, spot)
        N_d = float(spot) * 100_000_000.0
        A_d = float(_annuity(xccy.start_date, xccy.payment_dates, dom_curve, xccy.day_count))
        expected = N_d * s * A_d
        assert float(npv) == pytest.approx(expected, rel=1e-8)

    def test_basis_spread_solver_round_trip(
        self, ref_date, quarterly_schedule, dom_curve, for_curve, spot
    ):
        """cross_currency_basis_spread returns s* such that NPV ≈ 0."""
        xccy = CrossCurrencySwap(
            start_date=ref_date,
            payment_dates=quarterly_schedule,
            maturity_date=quarterly_schedule[-1],
            domestic_notional=jnp.array(100_000_000.0),  # NOT matched
            foreign_notional=jnp.array(100_000_000.0),
            basis_spread=jnp.array(0.0),
            exchange_notional=True,
        )
        par = cross_currency_basis_spread(xccy, dom_curve, for_curve, spot)
        at_par = CrossCurrencySwap(
            start_date=ref_date,
            payment_dates=quarterly_schedule,
            maturity_date=quarterly_schedule[-1],
            domestic_notional=jnp.array(100_000_000.0),
            foreign_notional=jnp.array(100_000_000.0),
            basis_spread=par,
            exchange_notional=True,
        )
        npv = cross_currency_swap_price(at_par, dom_curve, for_curve, spot)
        assert float(npv) == pytest.approx(0.0, abs=1e-4)

    def test_two_distinct_curves_nonzero_npv(
        self, matched_xccy, dom_curve, for_curve, spot
    ):
        """Without exchange_notional, different rates give nonzero NPV."""
        no_exch = CrossCurrencySwap(
            start_date=matched_xccy.start_date,
            payment_dates=matched_xccy.payment_dates,
            maturity_date=matched_xccy.maturity_date,
            domestic_notional=matched_xccy.domestic_notional,
            foreign_notional=matched_xccy.foreign_notional,
            basis_spread=jnp.array(0.0),
            exchange_notional=False,
        )
        npv = cross_currency_swap_price(no_exch, dom_curve, for_curve, spot)
        # 5% dom vs 2% for ⇒ dom float leg is worth more ⇒ receiver > 0
        assert float(npv) > 0.0

    def test_jit_compatible(self, matched_xccy, dom_curve, for_curve, spot):
        eager = cross_currency_swap_price(matched_xccy, dom_curve, for_curve, spot)
        jitted = jax.jit(cross_currency_swap_price, static_argnames=())(
            matched_xccy, dom_curve, for_curve, spot
        )
        assert float(jitted) == pytest.approx(float(eager), rel=1e-10)

    def test_grad_wrt_basis_spread(
        self, ref_date, quarterly_schedule, dom_curve, for_curve, spot
    ):
        """dNPV/ds = N_d * A_d (linear in spread)."""
        def price_from_spread(s):
            xccy = CrossCurrencySwap(
                start_date=ref_date,
                payment_dates=quarterly_schedule,
                maturity_date=quarterly_schedule[-1],
                domestic_notional=jnp.array(100_000_000.0),
                foreign_notional=jnp.array(100_000_000.0),
                basis_spread=s,
                exchange_notional=True,
            )
            return cross_currency_swap_price(xccy, dom_curve, for_curve, spot)

        g = jax.grad(price_from_spread)(jnp.array(0.0))
        A_d = float(_annuity(ref_date, quarterly_schedule, dom_curve, "act_360"))
        expected = 100_000_000.0 * A_d
        assert float(g) == pytest.approx(expected, rel=1e-8)

    def test_vmap_across_spots(
        self, ref_date, quarterly_schedule, dom_curve, for_curve
    ):
        """vmap over different spot rates."""
        spots = jnp.linspace(0.90, 1.30, 5)

        def price_one(s):
            xccy = CrossCurrencySwap(
                start_date=ref_date,
                payment_dates=quarterly_schedule,
                maturity_date=quarterly_schedule[-1],
                domestic_notional=jnp.array(100_000_000.0),
                foreign_notional=jnp.array(100_000_000.0),
                basis_spread=jnp.array(0.0),
                exchange_notional=False,
            )
            return cross_currency_swap_price(xccy, dom_curve, for_curve, s)

        batch = jax.vmap(price_one)(spots)
        assert batch.shape == (5,)
        assert jnp.all(jnp.isfinite(batch))


# ── Total return swap ────────────────────────────────────────────────

class TestTotalReturnSwap:
    """TRS pricing under self-financing asset assumption."""

    @pytest.fixture
    def trs(self, ref_date, quarterly_schedule):
        return TotalReturnSwap(
            start_date=ref_date,
            payment_dates=quarterly_schedule,
            notional=jnp.array(10_000_000.0),
            funding_spread=jnp.array(0.005),
        )

    def test_zero_spread_zero_npv(self, ref_date, quarterly_schedule, dom_curve):
        """At reset with zero funding spread → NPV = 0."""
        trs0 = TotalReturnSwap(
            start_date=ref_date,
            payment_dates=quarterly_schedule,
            notional=jnp.array(10_000_000.0),
            funding_spread=jnp.array(0.0),
        )
        npv = total_return_swap_price(trs0, dom_curve)
        assert float(npv) == pytest.approx(0.0, abs=1e-10)

    def test_spread_identity(self, trs, dom_curve):
        """NPV = -N * s * A for the TR receiver."""
        npv = total_return_swap_price(trs, dom_curve)
        A = float(_annuity(trs.start_date, trs.payment_dates, dom_curve, trs.day_count))
        expected = -10_000_000.0 * 0.005 * A
        assert float(npv) == pytest.approx(expected, rel=1e-10)

    def test_sign_flip(self, ref_date, quarterly_schedule, dom_curve):
        """Payer and receiver have opposite NPV."""
        trs_recv = TotalReturnSwap(
            start_date=ref_date, payment_dates=quarterly_schedule,
            notional=jnp.array(10_000_000.0), funding_spread=jnp.array(0.005),
            is_total_return_receiver=True,
        )
        trs_pay = TotalReturnSwap(
            start_date=ref_date, payment_dates=quarterly_schedule,
            notional=jnp.array(10_000_000.0), funding_spread=jnp.array(0.005),
            is_total_return_receiver=False,
        )
        r = total_return_swap_price(trs_recv, dom_curve)
        p = total_return_swap_price(trs_pay, dom_curve)
        assert float(r) == pytest.approx(-float(p), abs=1e-10)

    def test_unrealized_return_adds_accrued_pv(self, trs, dom_curve):
        """unrealized_return = u adds N * u * DF(first_payment) to NPV."""
        npv_base = total_return_swap_price(trs, dom_curve)
        u = jnp.array(0.02)
        npv_u = total_return_swap_price(trs, dom_curve, unrealized_return=u)
        accrued = 10_000_000.0 * 0.02 * float(dom_curve(trs.payment_dates[0]))
        assert float(npv_u) == pytest.approx(float(npv_base) + accrued, rel=1e-10)

    def test_grad_wrt_funding_spread(self, trs, dom_curve):
        """dNPV/d(spread) = -N * A for receiver."""
        def price_from_spread(s):
            t = TotalReturnSwap(
                start_date=trs.start_date, payment_dates=trs.payment_dates,
                notional=trs.notional, funding_spread=s,
            )
            return total_return_swap_price(t, dom_curve)

        g = jax.grad(price_from_spread)(trs.funding_spread)
        A = float(_annuity(trs.start_date, trs.payment_dates, dom_curve, trs.day_count))
        expected = -10_000_000.0 * A
        assert float(g) == pytest.approx(expected, rel=1e-10)

    def test_jit_compatible(self, trs, dom_curve):
        eager = total_return_swap_price(trs, dom_curve)
        jitted = jax.jit(total_return_swap_price)(trs, dom_curve)
        assert float(jitted) == pytest.approx(float(eager), rel=1e-10)


# ── CMS swap ─────────────────────────────────────────────────────────

class TestCMSSwap:
    """CMS swap pricing (no convexity adjustment)."""

    @pytest.fixture
    def cms(self, ref_date, quarterly_schedule):
        return CMSSwap(
            start_date=ref_date,
            payment_dates=quarterly_schedule,
            fixed_rate=jnp.array(0.05),
            notional=jnp.array(1_000_000.0),
            cms_tenor=5,
        )

    @pytest.fixture
    def flat_curve_15y(self, ref_date):
        return _flat_curve(ref_date, 0.05, 16)

    def test_finite_price(self, cms, flat_curve_15y):
        """Price is a finite real number."""
        p = cms_swap_price(cms, flat_curve_15y)
        assert jnp.isfinite(p)

    def test_sign_flip(self, ref_date, quarterly_schedule, flat_curve_15y):
        """pay_fixed=True vs False gives opposite NPVs."""
        payer = CMSSwap(
            start_date=ref_date, payment_dates=quarterly_schedule,
            fixed_rate=jnp.array(0.05), notional=jnp.array(1_000_000.0),
            cms_tenor=5, pay_fixed=True,
        )
        receiver = CMSSwap(
            start_date=ref_date, payment_dates=quarterly_schedule,
            fixed_rate=jnp.array(0.05), notional=jnp.array(1_000_000.0),
            cms_tenor=5, pay_fixed=False,
        )
        p1 = cms_swap_price(payer, flat_curve_15y)
        p2 = cms_swap_price(receiver, flat_curve_15y)
        assert float(p1) == pytest.approx(-float(p2), rel=1e-10)

    def test_grad_wrt_fixed_rate(self, cms, flat_curve_15y):
        """dNPV/d(fixed_rate) ≈ -notional * annuity."""
        def price_from_rate(k):
            s = CMSSwap(
                start_date=cms.start_date, payment_dates=cms.payment_dates,
                fixed_rate=k, notional=cms.notional, cms_tenor=cms.cms_tenor,
            )
            return cms_swap_price(s, flat_curve_15y)

        g = jax.grad(price_from_rate)(cms.fixed_rate)
        A = float(_annuity(cms.start_date, cms.payment_dates, flat_curve_15y, cms.day_count))
        expected = -float(cms.notional) * A
        assert float(g) == pytest.approx(expected, rel=1e-8)

    def test_jit_compatible(self, cms, flat_curve_15y):
        eager = cms_swap_price(cms, flat_curve_15y)
        jitted = jax.jit(cms_swap_price)(cms, flat_curve_15y)
        assert float(jitted) == pytest.approx(float(eager), rel=1e-10)

    def test_vmap_across_fixed_rates(self, cms, flat_curve_15y):
        rates = jnp.linspace(0.03, 0.07, 9)

        def price_one(k):
            s = CMSSwap(
                start_date=cms.start_date, payment_dates=cms.payment_dates,
                fixed_rate=k, notional=cms.notional, cms_tenor=cms.cms_tenor,
            )
            return cms_swap_price(s, flat_curve_15y)

        batch = jax.vmap(price_one)(rates)
        assert batch.shape == (9,)
        # Payer NPV decreases monotonically as fixed rate rises.
        assert jnp.all(jnp.diff(batch) < 0.0)


# ── CMS cap / floor ─────────────────────────────────────────────────

class TestCMSCapFloor:
    """Black-76 CMS cap/floor (no convexity adjustment)."""

    @pytest.fixture
    def cap(self, quarterly_schedule):
        return CMSCapFloor(
            payment_dates=quarterly_schedule[-4:],
            strike=jnp.array(0.05),
            notional=jnp.array(1_000_000.0),
            cms_tenor=5,
            is_cap=True,
        )

    @pytest.fixture
    def floor(self, quarterly_schedule):
        return CMSCapFloor(
            payment_dates=quarterly_schedule[-4:],
            strike=jnp.array(0.05),
            notional=jnp.array(1_000_000.0),
            cms_tenor=5,
            is_cap=False,
        )

    @pytest.fixture
    def flat_curve_15y(self, ref_date):
        return _flat_curve(ref_date, 0.05, 16)

    def test_cap_non_negative(self, cap, flat_curve_15y):
        pv = cms_cap_floor_price_black76(cap, flat_curve_15y, jnp.array(0.25))
        assert float(pv) >= 0.0

    def test_floor_non_negative(self, floor, flat_curve_15y):
        pv = cms_cap_floor_price_black76(floor, flat_curve_15y, jnp.array(0.25))
        assert float(pv) >= 0.0

    def test_put_call_parity(self, cap, floor, flat_curve_15y):
        """cap - floor = sum_i (F_cms_i - K) * tau_i * DF_i * N."""
        vol = jnp.array(0.25)
        cap_pv = cms_cap_floor_price_black76(cap, flat_curve_15y, vol)
        floor_pv = cms_cap_floor_price_black76(floor, flat_curve_15y, vol)

        # Manual parity computation
        F = _cms_forward_rates(cap.payment_dates, cap.cms_tenor, flat_curve_15y)
        from valax.dates.daycounts import year_fraction
        diffs = jnp.diff(cap.payment_dates)
        tau_days = jnp.concatenate([diffs[:1], diffs])
        starts = cap.payment_dates - tau_days
        tau = year_fraction(starts, cap.payment_dates, cap.day_count)
        dfs = flat_curve_15y(cap.payment_dates)
        parity = float(jnp.sum((F - cap.strike) * tau * dfs * cap.notional))

        assert float(cap_pv - floor_pv) == pytest.approx(parity, rel=1e-10)

    def test_cap_increases_with_vol(self, cap, flat_curve_15y):
        """Higher vol → higher cap price (standard vega > 0)."""
        p_low = cms_cap_floor_price_black76(cap, flat_curve_15y, jnp.array(0.10))
        p_high = cms_cap_floor_price_black76(cap, flat_curve_15y, jnp.array(0.50))
        assert float(p_high) > float(p_low)

    def test_jit_compatible(self, cap, flat_curve_15y):
        vol = jnp.array(0.25)
        eager = cms_cap_floor_price_black76(cap, flat_curve_15y, vol)
        jitted = jax.jit(cms_cap_floor_price_black76)(cap, flat_curve_15y, vol)
        assert float(jitted) == pytest.approx(float(eager), rel=1e-10)

    def test_vmap_across_vols(self, cap, flat_curve_15y):
        vols = jnp.linspace(0.10, 0.50, 5)

        def price_one(v):
            return cms_cap_floor_price_black76(cap, flat_curve_15y, v)

        batch = jax.vmap(price_one)(vols)
        assert batch.shape == (5,)
        # Cap price monotonically increases with vol.
        assert jnp.all(jnp.diff(batch) > 0.0)


# ── Range accrual ────────────────────────────────────────────────────

class TestRangeAccrual:
    """Black-76 digital-replication range accrual."""

    @pytest.fixture
    def ra(self, quarterly_schedule):
        return RangeAccrual(
            payment_dates=quarterly_schedule,
            coupon_rate=jnp.array(0.08),
            lower_barrier=jnp.array(0.01),
            upper_barrier=jnp.array(0.10),
            notional=jnp.array(1_000_000.0),
        )

    def test_wide_range_approaches_full_coupon(self, quarterly_schedule, dom_curve):
        """Very wide range → prob ≈ 1 → price ≈ full coupon annuity."""
        ra_wide = RangeAccrual(
            payment_dates=quarterly_schedule,
            coupon_rate=jnp.array(0.08),
            lower_barrier=jnp.array(1e-6),
            upper_barrier=jnp.array(100.0),
            notional=jnp.array(1_000_000.0),
        )
        pv = range_accrual_price_black76(ra_wide, dom_curve, jnp.array(0.30))

        # Expected full coupon annuity
        diffs = jnp.diff(quarterly_schedule)
        tau_days = jnp.concatenate([diffs[:1], diffs])
        starts = quarterly_schedule - tau_days
        from valax.dates.daycounts import year_fraction
        tau = year_fraction(starts, quarterly_schedule, "act_360")
        full = float(1_000_000.0 * 0.08 * jnp.sum(tau * dom_curve(quarterly_schedule)))
        assert float(pv) == pytest.approx(full, rel=1e-6)

    def test_narrow_range_small_price(self, quarterly_schedule, dom_curve):
        """Very narrow range → small price."""
        ra_narrow = RangeAccrual(
            payment_dates=quarterly_schedule,
            coupon_rate=jnp.array(0.08),
            lower_barrier=jnp.array(0.04999),
            upper_barrier=jnp.array(0.05001),
            notional=jnp.array(1_000_000.0),
        )
        pv_narrow = range_accrual_price_black76(ra_narrow, dom_curve, jnp.array(0.30))

        # Wide range for comparison
        ra_wide = RangeAccrual(
            payment_dates=quarterly_schedule,
            coupon_rate=jnp.array(0.08),
            lower_barrier=jnp.array(1e-6),
            upper_barrier=jnp.array(100.0),
            notional=jnp.array(1_000_000.0),
        )
        pv_wide = range_accrual_price_black76(ra_wide, dom_curve, jnp.array(0.30))
        # Narrow should be a tiny fraction of wide.
        assert float(pv_narrow) < 0.05 * float(pv_wide)

    def test_non_negative(self, ra, dom_curve):
        pv = range_accrual_price_black76(ra, dom_curve, jnp.array(0.30))
        assert float(pv) >= 0.0

    def test_monotone_in_range_width(self, quarterly_schedule, dom_curve):
        """Wider range → higher price."""
        def price_for_upper(u):
            ra = RangeAccrual(
                payment_dates=quarterly_schedule,
                coupon_rate=jnp.array(0.08),
                lower_barrier=jnp.array(0.03),
                upper_barrier=u,
                notional=jnp.array(1_000_000.0),
            )
            return range_accrual_price_black76(ra, dom_curve, jnp.array(0.30))

        uppers = jnp.array([0.06, 0.08, 0.10, 0.15])
        prices = jax.vmap(price_for_upper)(uppers)
        assert jnp.all(jnp.diff(prices) > 0.0)

    def test_jit_compatible(self, ra, dom_curve):
        vol = jnp.array(0.30)
        eager = range_accrual_price_black76(ra, dom_curve, vol)
        jitted = jax.jit(range_accrual_price_black76)(ra, dom_curve, vol)
        assert float(jitted) == pytest.approx(float(eager), rel=1e-10)
