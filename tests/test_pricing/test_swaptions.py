"""Tests for swap pricing, swap rate, and European swaption pricing."""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.rates import InterestRateSwap, Swaption
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.dates.schedule import generate_schedule

from valax.pricing.analytic.swaptions import (
    swap_rate,
    swap_price,
    swaption_price_black76,
    swaption_price_bachelier,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def ref_date():
    return ymd_to_ordinal(2025, 1, 1)


@pytest.fixture
def flat_curve(ref_date):
    """Flat 5% continuously-compounded curve out to 10 years."""
    pillars = jnp.array([
        int(ymd_to_ordinal(2025, 1, 1)),
        int(ymd_to_ordinal(2026, 1, 1)),
        int(ymd_to_ordinal(2027, 1, 1)),
        int(ymd_to_ordinal(2028, 1, 1)),
        int(ymd_to_ordinal(2029, 1, 1)),
        int(ymd_to_ordinal(2030, 1, 1)),
        int(ymd_to_ordinal(2032, 1, 1)),
        int(ymd_to_ordinal(2035, 1, 1)),
    ], dtype=jnp.int32)
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-0.05 * times)
    return DiscountCurve(pillar_dates=pillars, discount_factors=dfs, reference_date=ref_date)


@pytest.fixture
def swap_5y(ref_date):
    """5-year annual fixed-for-float payer swap, notional=1M."""
    fixed_dates = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=1)
    return InterestRateSwap(
        start_date=ref_date,
        fixed_dates=fixed_dates,
        fixed_rate=jnp.array(0.05),
        notional=jnp.array(1_000_000.0),
        pay_fixed=True,
    )


@pytest.fixture
def payer_swaption(ref_date):
    """1y x 5y payer swaption: expiry in 1 year, 5-year underlying."""
    expiry = ymd_to_ordinal(2026, 1, 1)
    fixed_dates = generate_schedule(2026, 1, 1, 2031, 1, 1, frequency=1)
    return Swaption(
        expiry_date=expiry,
        fixed_dates=fixed_dates,
        strike=jnp.array(0.05),
        notional=jnp.array(1_000_000.0),
        is_payer=True,
    )


@pytest.fixture
def receiver_swaption(ref_date):
    """1y x 5y receiver swaption: same terms, opposite direction."""
    expiry = ymd_to_ordinal(2026, 1, 1)
    fixed_dates = generate_schedule(2026, 1, 1, 2031, 1, 1, frequency=1)
    return Swaption(
        expiry_date=expiry,
        fixed_dates=fixed_dates,
        strike=jnp.array(0.05),
        notional=jnp.array(1_000_000.0),
        is_payer=False,
    )


# ── Swap utilities ────────────────────────────────────────────────────

class TestSwap:
    def test_par_swap_rate_positive(self, swap_5y, flat_curve):
        S = swap_rate(swap_5y, flat_curve)
        assert float(S) > 0.0

    def test_par_swap_rate_close_to_yield(self, swap_5y, flat_curve):
        """On a flat 5% curve, the par swap rate should be ~5%."""
        S = swap_rate(swap_5y, flat_curve)
        # On a flat CC curve, the simply-compounded par rate is slightly above
        # the CC rate due to compounding. Allow 50bps tolerance.
        assert abs(float(S) - 0.05) < 0.005

    def test_at_par_rate_swap_npv_zero(self, flat_curve, ref_date):
        """Swap at the par rate should have NPV = 0."""
        fixed_dates = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=1)
        # Compute par rate first
        dummy_swap = InterestRateSwap(
            start_date=ref_date,
            fixed_dates=fixed_dates,
            fixed_rate=jnp.array(0.05),
            notional=jnp.array(1_000_000.0),
            pay_fixed=True,
        )
        S = swap_rate(dummy_swap, flat_curve)
        # Build swap at par rate
        par_swap = InterestRateSwap(
            start_date=ref_date,
            fixed_dates=fixed_dates,
            fixed_rate=S,
            notional=jnp.array(1_000_000.0),
            pay_fixed=True,
        )
        npv = swap_price(par_swap, flat_curve)
        assert abs(float(npv)) < 1.0  # < $1 on $1M notional

    def test_payer_receiver_sign_opposite(self, flat_curve, ref_date):
        """Payer and receiver swaps at same rate should have opposite NPVs."""
        fixed_dates = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=1)
        payer = InterestRateSwap(
            start_date=ref_date, fixed_dates=fixed_dates,
            fixed_rate=jnp.array(0.04), notional=jnp.array(1_000_000.0),
            pay_fixed=True,
        )
        receiver = InterestRateSwap(
            start_date=ref_date, fixed_dates=fixed_dates,
            fixed_rate=jnp.array(0.04), notional=jnp.array(1_000_000.0),
            pay_fixed=False,
        )
        npv_payer = swap_price(payer, flat_curve)
        npv_receiver = swap_price(receiver, flat_curve)
        assert abs(float(npv_payer) + float(npv_receiver)) < 1e-4

    def test_higher_fixed_rate_worse_for_payer(self, flat_curve, ref_date):
        """Payer pays fixed: higher fixed rate => lower NPV."""
        fixed_dates = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=1)
        swap_low = InterestRateSwap(
            start_date=ref_date, fixed_dates=fixed_dates,
            fixed_rate=jnp.array(0.03), notional=jnp.array(1_000_000.0),
            pay_fixed=True,
        )
        swap_high = InterestRateSwap(
            start_date=ref_date, fixed_dates=fixed_dates,
            fixed_rate=jnp.array(0.07), notional=jnp.array(1_000_000.0),
            pay_fixed=True,
        )
        assert float(swap_price(swap_low, flat_curve)) > float(swap_price(swap_high, flat_curve))

    def test_jit(self, swap_5y, flat_curve):
        p = jax.jit(swap_price)(swap_5y, flat_curve)
        assert jnp.isfinite(p)


# ── Swaption: Black-76 ────────────────────────────────────────────────

class TestSwaptionBlack76:
    def test_payer_price_positive(self, payer_swaption, flat_curve):
        p = swaption_price_black76(payer_swaption, flat_curve, jnp.array(0.20))
        assert float(p) > 0.0

    def test_receiver_price_positive(self, receiver_swaption, flat_curve):
        p = swaption_price_black76(receiver_swaption, flat_curve, jnp.array(0.20))
        assert float(p) > 0.0

    def test_payer_receiver_parity(self, payer_swaption, receiver_swaption, flat_curve, ref_date):
        """Payer - Receiver = PV of underlying payer swap at strike K.

        payer_swaption - receiver_swaption = notional * A * (S - K)
        """
        vol = jnp.array(0.20)
        payer_p = swaption_price_black76(payer_swaption, flat_curve, vol)
        recv_p = swaption_price_black76(receiver_swaption, flat_curve, vol)

        # Compute forward swap NPV = notional * A * (S - K)
        expiry = payer_swaption.expiry_date
        fixed_dates = payer_swaption.fixed_dates
        from valax.pricing.analytic.swaptions import _annuity
        ann = _annuity(expiry, fixed_dates, flat_curve, payer_swaption.day_count)
        df_start = flat_curve(expiry)
        df_end = flat_curve(fixed_dates[-1])
        S = (df_start - df_end) / ann
        expected = float(payer_swaption.notional * ann * (S - payer_swaption.strike))

        assert abs(float(payer_p) - float(recv_p) - expected) < 1.0

    def test_higher_vol_higher_price(self, payer_swaption, flat_curve):
        p_low = swaption_price_black76(payer_swaption, flat_curve, jnp.array(0.10))
        p_high = swaption_price_black76(payer_swaption, flat_curve, jnp.array(0.40))
        assert float(p_high) > float(p_low)

    def test_atm_payer_approx_equal_receiver(self, flat_curve, ref_date):
        """ATM swaption (K = S): payer ≈ receiver."""
        expiry = ymd_to_ordinal(2026, 1, 1)
        fixed_dates = generate_schedule(2026, 1, 1, 2031, 1, 1, frequency=1)

        # Compute par swap rate
        dummy = Swaption(
            expiry_date=expiry, fixed_dates=fixed_dates,
            strike=jnp.array(0.05), notional=jnp.array(1_000_000.0), is_payer=True,
        )
        from valax.pricing.analytic.swaptions import _annuity
        ann = _annuity(expiry, fixed_dates, flat_curve, dummy.day_count)
        df_start = flat_curve(expiry)
        df_end = flat_curve(fixed_dates[-1])
        S = (df_start - df_end) / ann  # ATM strike

        payer = Swaption(
            expiry_date=expiry, fixed_dates=fixed_dates,
            strike=S, notional=jnp.array(1_000_000.0), is_payer=True,
        )
        receiver = Swaption(
            expiry_date=expiry, fixed_dates=fixed_dates,
            strike=S, notional=jnp.array(1_000_000.0), is_payer=False,
        )
        vol = jnp.array(0.20)
        p = swaption_price_black76(payer, flat_curve, vol)
        r = swaption_price_black76(receiver, flat_curve, vol)
        assert abs(float(p) - float(r)) < 1.0  # < $1 on $1M

    def test_jit(self, payer_swaption, flat_curve):
        p = jax.jit(swaption_price_black76)(payer_swaption, flat_curve, jnp.array(0.20))
        assert jnp.isfinite(p)

    def test_grad_vol(self, payer_swaption, flat_curve):
        """Vega should be positive."""
        vega = jax.grad(swaption_price_black76, argnums=2)(
            payer_swaption, flat_curve, jnp.array(0.20)
        )
        assert float(vega) > 0.0

    def test_grad_curve(self, payer_swaption, flat_curve):
        """DV01: gradient of swaption price w.r.t. discount factors is finite."""
        import equinox as eqx
        price_fn = lambda curve: swaption_price_black76(payer_swaption, curve, jnp.array(0.20))
        grads = eqx.filter_grad(price_fn)(flat_curve)
        assert jnp.all(jnp.isfinite(grads.discount_factors))


# ── Swaption: Bachelier ───────────────────────────────────────────────

class TestSwaptionBachelier:
    def test_payer_price_positive(self, payer_swaption, flat_curve):
        p = swaption_price_bachelier(payer_swaption, flat_curve, jnp.array(0.005))
        assert float(p) > 0.0

    def test_receiver_price_positive(self, receiver_swaption, flat_curve):
        p = swaption_price_bachelier(receiver_swaption, flat_curve, jnp.array(0.005))
        assert float(p) > 0.0

    def test_payer_receiver_parity(self, payer_swaption, receiver_swaption, flat_curve):
        """Bachelier payer - receiver = notional * A * (S - K)."""
        vol = jnp.array(0.005)
        payer_p = swaption_price_bachelier(payer_swaption, flat_curve, vol)
        recv_p = swaption_price_bachelier(receiver_swaption, flat_curve, vol)

        expiry = payer_swaption.expiry_date
        fixed_dates = payer_swaption.fixed_dates
        from valax.pricing.analytic.swaptions import _annuity
        ann = _annuity(expiry, fixed_dates, flat_curve, payer_swaption.day_count)
        df_start = flat_curve(expiry)
        df_end = flat_curve(fixed_dates[-1])
        S = (df_start - df_end) / ann
        expected = float(payer_swaption.notional * ann * (S - payer_swaption.strike))

        assert abs(float(payer_p) - float(recv_p) - expected) < 1.0

    def test_higher_vol_higher_price(self, payer_swaption, flat_curve):
        p_low = swaption_price_bachelier(payer_swaption, flat_curve, jnp.array(0.002))
        p_high = swaption_price_bachelier(payer_swaption, flat_curve, jnp.array(0.010))
        assert float(p_high) > float(p_low)

    def test_jit(self, payer_swaption, flat_curve):
        p = jax.jit(swaption_price_bachelier)(payer_swaption, flat_curve, jnp.array(0.005))
        assert jnp.isfinite(p)

    def test_grad_vol(self, payer_swaption, flat_curve):
        vega = jax.grad(swaption_price_bachelier, argnums=2)(
            payer_swaption, flat_curve, jnp.array(0.005)
        )
        assert float(vega) > 0.0
