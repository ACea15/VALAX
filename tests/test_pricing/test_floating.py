"""Tests for floating-rate instrument pricing: FRN and OIS swap."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.instruments.bonds import FloatingRateBond
from valax.instruments.rates import InterestRateSwap, OISSwap
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.dates.schedule import generate_schedule
from valax.pricing.analytic.floating import (
    floating_rate_bond_price,
    ois_swap_price,
    ois_swap_rate,
)
from valax.pricing.analytic.swaptions import swap_price, swap_rate


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def ref_date():
    return ymd_to_ordinal(2025, 1, 1)


@pytest.fixture
def flat_curve(ref_date):
    """Flat 5% continuously-compounded curve, 6-year coverage."""
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(7)],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-0.05 * times)
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=dfs,
        reference_date=ref_date,
    )


@pytest.fixture
def steep_curve(ref_date):
    """Upward-sloping curve 3% → 6% CC over 6 years."""
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(7)],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    rates = 0.03 + 0.005 * times  # 3% → 6% linearly in time
    dfs = jnp.exp(-rates * times)
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=dfs,
        reference_date=ref_date,
    )


@pytest.fixture
def quarterly_schedule():
    """5-year quarterly payment schedule."""
    return generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=4)


@pytest.fixture
def quarterly_fixings(ref_date, quarterly_schedule):
    """Fixing dates = start of each accrual period.

    First fixing happens on the reference date; subsequent fixings are
    the previous payment date.
    """
    sched_np = np.asarray(quarterly_schedule)
    fixings = [int(ref_date)] + list(sched_np[:-1].tolist())
    return jnp.array(fixings, dtype=jnp.int32)


@pytest.fixture
def par_frn(ref_date, quarterly_schedule, quarterly_fixings):
    """Par-issue FRN, zero spread, face = 100, valued at first reset."""
    return FloatingRateBond(
        payment_dates=quarterly_schedule,
        fixing_dates=quarterly_fixings,
        settlement_date=ref_date,
        spread=jnp.array(0.0),
        face_value=jnp.array(100.0),
    )


@pytest.fixture
def spread_frn(ref_date, quarterly_schedule, quarterly_fixings):
    """FRN with 50 bps spread, face = 100."""
    return FloatingRateBond(
        payment_dates=quarterly_schedule,
        fixing_dates=quarterly_fixings,
        settlement_date=ref_date,
        spread=jnp.array(0.005),
        face_value=jnp.array(100.0),
    )


@pytest.fixture
def vanilla_ois(ref_date, quarterly_schedule):
    """Vanilla 5Y OIS, 5% fixed, 1MM notional."""
    return OISSwap(
        start_date=ref_date,
        fixed_dates=quarterly_schedule,
        float_dates=quarterly_schedule,
        fixed_rate=jnp.array(0.05),
        notional=jnp.array(1_000_000.0),
    )


# ── FloatingRateBond tests ───────────────────────────────────────────

class TestFloatingRateBond:
    def test_par_at_reset_zero_spread(self, par_frn, flat_curve):
        """Par FRN with zero spread valued on reset date = face value."""
        price = floating_rate_bond_price(par_frn, flat_curve)
        assert float(price) == pytest.approx(100.0, abs=1e-8)

    def test_par_at_reset_steep_curve(
        self, ref_date, quarterly_schedule, quarterly_fixings, steep_curve
    ):
        """Par-at-reset invariant holds for a non-flat curve too."""
        frn = FloatingRateBond(
            payment_dates=quarterly_schedule,
            fixing_dates=quarterly_fixings,
            settlement_date=ref_date,
            spread=jnp.array(0.0),
            face_value=jnp.array(100.0),
        )
        price = floating_rate_bond_price(frn, steep_curve)
        assert float(price) == pytest.approx(100.0, abs=1e-8)

    def test_spread_identity(
        self, spread_frn, flat_curve, quarterly_schedule, ref_date
    ):
        """P - face ≈ face * spread * annuity (par-at-reset + spread)."""
        price = floating_rate_bond_price(spread_frn, flat_curve)
        # Annuity = sum_i tau_i * DF(T_i), act_360, from ref to each payment
        starts = jnp.concatenate(
            [jnp.array([int(ref_date)], dtype=jnp.int32), quarterly_schedule[:-1]]
        )
        tau = (quarterly_schedule - starts).astype(jnp.float64) / 360.0
        annuity = float(jnp.sum(tau * flat_curve(quarterly_schedule)))
        expected = 100.0 + 100.0 * 0.005 * annuity
        assert float(price) == pytest.approx(expected, rel=1e-10)

    def test_spread_increases_price(self, par_frn, spread_frn, flat_curve):
        """Positive spread → price above par."""
        p0 = floating_rate_bond_price(par_frn, flat_curve)
        p1 = floating_rate_bond_price(spread_frn, flat_curve)
        assert float(p1) > float(p0)

    def test_known_fixing_used_for_current_period(
        self, ref_date, quarterly_schedule, quarterly_fixings, flat_curve
    ):
        """A known fixing overrides the projected forward rate."""
        # Put an artificially high fixing on period 0 and NaN elsewhere
        known = jnp.array(
            [0.10] + [float("nan")] * (len(quarterly_schedule) - 1)
        )
        frn_known = FloatingRateBond(
            payment_dates=quarterly_schedule,
            fixing_dates=quarterly_fixings,
            settlement_date=ref_date,
            spread=jnp.array(0.0),
            face_value=jnp.array(100.0),
            fixing_rates=known,
        )
        frn_none = FloatingRateBond(
            payment_dates=quarterly_schedule,
            fixing_dates=quarterly_fixings,
            settlement_date=ref_date,
            spread=jnp.array(0.0),
            face_value=jnp.array(100.0),
        )
        price_known = floating_rate_bond_price(frn_known, flat_curve)
        price_none = floating_rate_bond_price(frn_none, flat_curve)
        # The inflated fixing on period 0 should push the price above par
        assert float(price_known) > float(price_none)
        # And above face value (par-at-reset baseline)
        assert float(price_known) > 100.0

    def test_all_nan_fixings_equivalent_to_none(self, par_frn, flat_curve):
        """Passing an all-NaN fixing_rates array is equivalent to None."""
        n = par_frn.payment_dates.shape[0]
        all_nan = jnp.full((n,), jnp.nan)
        frn_all_nan = FloatingRateBond(
            payment_dates=par_frn.payment_dates,
            fixing_dates=par_frn.fixing_dates,
            settlement_date=par_frn.settlement_date,
            spread=par_frn.spread,
            face_value=par_frn.face_value,
            fixing_rates=all_nan,
        )
        p_none = floating_rate_bond_price(par_frn, flat_curve)
        p_all_nan = floating_rate_bond_price(frn_all_nan, flat_curve)
        assert float(p_all_nan) == pytest.approx(float(p_none), abs=1e-12)

    def test_jit_compatible(self, par_frn, flat_curve):
        """Pricer is jit-compilable."""
        jitted = jax.jit(floating_rate_bond_price)
        p_eager = floating_rate_bond_price(par_frn, flat_curve)
        p_jit = jitted(par_frn, flat_curve)
        assert float(p_jit) == pytest.approx(float(p_eager), abs=1e-12)

    def test_grad_wrt_face_value(self, par_frn, flat_curve):
        """For a par floater at reset, dP/dN = 1 exactly."""
        def price_from_face(face):
            frn = FloatingRateBond(
                payment_dates=par_frn.payment_dates,
                fixing_dates=par_frn.fixing_dates,
                settlement_date=par_frn.settlement_date,
                spread=par_frn.spread,
                face_value=face,
            )
            return floating_rate_bond_price(frn, flat_curve)

        g = jax.grad(price_from_face)(jnp.array(100.0))
        assert float(g) == pytest.approx(1.0, abs=1e-10)

    def test_grad_wrt_spread_positive(self, par_frn, flat_curve):
        """dP/d spread > 0 and approximately equals annuity * face."""
        def price_from_spread(s):
            frn = FloatingRateBond(
                payment_dates=par_frn.payment_dates,
                fixing_dates=par_frn.fixing_dates,
                settlement_date=par_frn.settlement_date,
                spread=s,
                face_value=par_frn.face_value,
            )
            return floating_rate_bond_price(frn, flat_curve)

        g = jax.grad(price_from_spread)(jnp.array(0.0))
        assert float(g) > 0.0

        # Expected: face * annuity (act_360, from ref to each payment)
        starts = jnp.concatenate(
            [jnp.array([int(par_frn.settlement_date)], dtype=jnp.int32),
             par_frn.payment_dates[:-1]]
        )
        tau = (par_frn.payment_dates - starts).astype(jnp.float64) / 360.0
        annuity = float(jnp.sum(tau * flat_curve(par_frn.payment_dates)))
        expected = 100.0 * annuity
        assert float(g) == pytest.approx(expected, rel=1e-10)

    def test_vmap_across_spreads(self, par_frn, flat_curve):
        """vmap prices a batch of spreads in one call."""
        spreads = jnp.linspace(0.0, 0.01, 5)

        def price_one(s):
            frn = FloatingRateBond(
                payment_dates=par_frn.payment_dates,
                fixing_dates=par_frn.fixing_dates,
                settlement_date=par_frn.settlement_date,
                spread=s,
                face_value=par_frn.face_value,
            )
            return floating_rate_bond_price(frn, flat_curve)

        batch = jax.vmap(price_one)(spreads)
        assert batch.shape == (5,)
        # Monotone in spread
        diffs = jnp.diff(batch)
        assert jnp.all(diffs > 0.0)


# ── OIS swap tests ───────────────────────────────────────────────────

class TestOISSwap:
    def test_par_rate_zeroes_npv(self, vanilla_ois, flat_curve):
        """Pricing an OIS at its par rate gives near-zero NPV."""
        par = ois_swap_rate(vanilla_ois, flat_curve)
        at_par = OISSwap(
            start_date=vanilla_ois.start_date,
            fixed_dates=vanilla_ois.fixed_dates,
            float_dates=vanilla_ois.float_dates,
            fixed_rate=par,
            notional=vanilla_ois.notional,
        )
        npv = ois_swap_price(at_par, flat_curve)
        assert float(npv) == pytest.approx(0.0, abs=1e-6)

    def test_par_rate_positive_and_sensible(self, vanilla_ois, flat_curve):
        """Par rate should be close to the flat curve rate."""
        par = ois_swap_rate(vanilla_ois, flat_curve)
        # 5% CC rate with act_360 day-count gives a par rate near but
        # not exactly equal to 5%; just check it's in the right ballpark.
        assert 0.04 < float(par) < 0.06

    def test_sign_flip_on_pay_receive(self, vanilla_ois, flat_curve):
        """Receiver swap has the opposite NPV of the equivalent payer swap."""
        payer = vanilla_ois  # pay_fixed=True by default
        receiver = OISSwap(
            start_date=vanilla_ois.start_date,
            fixed_dates=vanilla_ois.fixed_dates,
            float_dates=vanilla_ois.float_dates,
            fixed_rate=vanilla_ois.fixed_rate,
            notional=vanilla_ois.notional,
            pay_fixed=False,
        )
        p_payer = ois_swap_price(payer, flat_curve)
        p_receiver = ois_swap_price(receiver, flat_curve)
        assert float(p_payer) == pytest.approx(-float(p_receiver), abs=1e-10)

    def test_agrees_with_vanilla_irs(
        self, vanilla_ois, flat_curve, quarterly_schedule, ref_date
    ):
        """Under single-curve, OIS and IRS with same schedule and rate match."""
        irs = InterestRateSwap(
            start_date=ref_date,
            fixed_dates=quarterly_schedule,
            fixed_rate=vanilla_ois.fixed_rate,
            notional=vanilla_ois.notional,
        )
        p_ois = float(ois_swap_price(vanilla_ois, flat_curve))
        p_irs = float(swap_price(irs, flat_curve))
        assert p_ois == pytest.approx(p_irs, abs=1e-8)

        # Par rates must agree as well.
        par_ois = float(ois_swap_rate(vanilla_ois, flat_curve))
        par_irs = float(swap_rate(irs, flat_curve))
        assert par_ois == pytest.approx(par_irs, rel=1e-12)

    def test_jit_compatible(self, vanilla_ois, flat_curve):
        """Both pricers are jittable (equal up to float rearrangement)."""
        p_eager = ois_swap_price(vanilla_ois, flat_curve)
        r_eager = ois_swap_rate(vanilla_ois, flat_curve)
        p_jit = jax.jit(ois_swap_price)(vanilla_ois, flat_curve)
        r_jit = jax.jit(ois_swap_rate)(vanilla_ois, flat_curve)
        assert float(p_jit) == pytest.approx(float(p_eager), rel=1e-10)
        assert float(r_jit) == pytest.approx(float(r_eager), rel=1e-10)

    def test_grad_wrt_fixed_rate(self, vanilla_ois, flat_curve):
        """dNPV/d(fixed_rate) = -notional * annuity for a payer swap."""
        def price_from_rate(k):
            s = OISSwap(
                start_date=vanilla_ois.start_date,
                fixed_dates=vanilla_ois.fixed_dates,
                float_dates=vanilla_ois.float_dates,
                fixed_rate=k,
                notional=vanilla_ois.notional,
            )
            return ois_swap_price(s, flat_curve)

        g = jax.grad(price_from_rate)(vanilla_ois.fixed_rate)

        # Expected annuity
        starts = jnp.concatenate(
            [jnp.array([int(vanilla_ois.start_date)], dtype=jnp.int32),
             vanilla_ois.fixed_dates[:-1]]
        )
        tau = (vanilla_ois.fixed_dates - starts).astype(jnp.float64) / 360.0
        annuity = float(jnp.sum(tau * flat_curve(vanilla_ois.fixed_dates)))
        expected = -float(vanilla_ois.notional) * annuity
        assert float(g) == pytest.approx(expected, rel=1e-10)

    def test_vmap_across_fixed_rates(self, vanilla_ois, flat_curve):
        """vmap prices a batch of OIS swaps with different fixed rates."""
        rates = jnp.linspace(0.03, 0.07, 9)

        def price_one(k):
            s = OISSwap(
                start_date=vanilla_ois.start_date,
                fixed_dates=vanilla_ois.fixed_dates,
                float_dates=vanilla_ois.float_dates,
                fixed_rate=k,
                notional=vanilla_ois.notional,
            )
            return ois_swap_price(s, flat_curve)

        batch = jax.vmap(price_one)(rates)
        assert batch.shape == (9,)
        # Payer NPV decreases monotonically as fixed rate rises.
        diffs = jnp.diff(batch)
        assert jnp.all(diffs < 0.0)

    def test_key_rate_sensitivity_via_log_dfs(self, vanilla_ois, flat_curve):
        """grad w.r.t. curve log-DFs runs and returns a finite vector."""
        def price_from_log_dfs(log_dfs):
            shifted = DiscountCurve(
                pillar_dates=flat_curve.pillar_dates,
                discount_factors=jnp.exp(log_dfs),
                reference_date=flat_curve.reference_date,
                day_count=flat_curve.day_count,
            )
            return ois_swap_price(vanilla_ois, shifted)

        log_dfs = jnp.log(flat_curve.discount_factors)
        g = jax.grad(price_from_log_dfs)(log_dfs)
        assert g.shape == log_dfs.shape
        assert jnp.all(jnp.isfinite(g))
