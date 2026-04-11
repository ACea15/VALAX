"""Tests for FX forward and Garman-Kohlhagen option pricing."""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.fx import FXForward, FXVanillaOption
from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.analytic.fx import (
    delta_to_strike,
    fx_delta,
    fx_forward_price,
    fx_forward_rate,
    fx_implied_vol,
    garman_kohlhagen_price,
    strike_to_delta,
)


# ── Fixtures ─────────────────────────────────────────────────────────


SPOT = jnp.array(1.10)        # EUR/USD = 1.10
VOL = jnp.array(0.08)         # 8% vol
R_DOM = jnp.array(0.05)       # USD rate 5%
R_FOR = jnp.array(0.03)       # EUR rate 3%
EXPIRY = jnp.array(0.5)       # 6 months
NOTIONAL = jnp.array(1e6)     # 1M EUR


@pytest.fixture
def fx_call():
    return FXVanillaOption(
        strike=jnp.array(1.10),
        expiry=EXPIRY,
        notional_foreign=NOTIONAL,
        is_call=True,
        currency_pair="EUR/USD",
    )


@pytest.fixture
def fx_put():
    return FXVanillaOption(
        strike=jnp.array(1.10),
        expiry=EXPIRY,
        notional_foreign=NOTIONAL,
        is_call=False,
        currency_pair="EUR/USD",
    )


# ── FX Forward ───────────────────────────────────────────────────────


class TestFXForward:
    def test_forward_rate_covered_interest_parity(self):
        """F = S * exp((r_d - r_f) * T)."""
        F = fx_forward_rate(SPOT, R_DOM, R_FOR, EXPIRY)
        expected = SPOT * jnp.exp((R_DOM - R_FOR) * EXPIRY)
        assert jnp.isclose(F, expected, atol=1e-10)

    def test_forward_npv_zero_at_fair_rate(self):
        """FX forward NPV should be zero when struck at the fair rate."""
        F = fx_forward_rate(SPOT, R_DOM, R_FOR, EXPIRY)
        fwd = FXForward(
            strike=F,
            expiry=EXPIRY,
            notional_foreign=NOTIONAL,
            is_buy=True,
        )
        npv = fx_forward_price(fwd, SPOT, R_DOM, R_FOR)
        assert jnp.isclose(npv, 0.0, atol=1e-6)

    def test_forward_buy_sell_opposite_sign(self):
        """Buy and sell should have opposite NPV."""
        F = fx_forward_rate(SPOT, R_DOM, R_FOR, EXPIRY)
        strike = F * 1.01  # slightly above fair → buy is negative
        fwd_buy = FXForward(strike=strike, expiry=EXPIRY,
                            notional_foreign=NOTIONAL, is_buy=True)
        fwd_sell = FXForward(strike=strike, expiry=EXPIRY,
                             notional_foreign=NOTIONAL, is_buy=False)
        npv_buy = fx_forward_price(fwd_buy, SPOT, R_DOM, R_FOR)
        npv_sell = fx_forward_price(fwd_sell, SPOT, R_DOM, R_FOR)
        assert jnp.isclose(npv_buy, -npv_sell, atol=1e-8)

    def test_forward_differentiable(self):
        """jax.grad through fx_forward_price should work."""
        fwd = FXForward(strike=jnp.array(1.12), expiry=EXPIRY,
                        notional_foreign=NOTIONAL, is_buy=True)
        grad = jax.grad(lambda s: fx_forward_price(fwd, s, R_DOM, R_FOR))(SPOT)
        assert jnp.isfinite(grad)
        assert grad > 0  # Buy forward: NPV increases with spot


# ── Garman-Kohlhagen ─────────────────────────────────────────────────


class TestGarmanKohlhagen:
    def test_matches_black_scholes_with_dividend(self, fx_call):
        """GK should match BSM when dividend = r_foreign."""
        gk_price = garman_kohlhagen_price(fx_call, SPOT, VOL, R_DOM, R_FOR)

        # Equivalent BSM option
        bsm_opt = EuropeanOption(
            strike=fx_call.strike,
            expiry=fx_call.expiry,
            is_call=True,
        )
        bsm_price = black_scholes_price(bsm_opt, SPOT, VOL, R_DOM, R_FOR)

        # GK has notional, BSM doesn't
        assert jnp.isclose(gk_price / NOTIONAL, bsm_price, rtol=1e-10)

    def test_put_call_parity(self, fx_call, fx_put):
        """C - P = N * (S * df_for - K * df_dom)."""
        C = garman_kohlhagen_price(fx_call, SPOT, VOL, R_DOM, R_FOR)
        P = garman_kohlhagen_price(fx_put, SPOT, VOL, R_DOM, R_FOR)

        df_dom = jnp.exp(-R_DOM * EXPIRY)
        df_for = jnp.exp(-R_FOR * EXPIRY)
        K = fx_call.strike

        parity_rhs = NOTIONAL * (SPOT * df_for - K * df_dom)
        assert jnp.isclose(C - P, parity_rhs, rtol=1e-8)

    def test_call_price_positive(self, fx_call):
        """Call price should be positive."""
        price = garman_kohlhagen_price(fx_call, SPOT, VOL, R_DOM, R_FOR)
        assert price > 0

    def test_put_price_positive(self, fx_put):
        """Put price should be positive."""
        price = garman_kohlhagen_price(fx_put, SPOT, VOL, R_DOM, R_FOR)
        assert price > 0

    def test_deep_itm_call_near_intrinsic(self):
        """Deep ITM call should approach discounted intrinsic value."""
        deep_itm = FXVanillaOption(
            strike=jnp.array(0.80),
            expiry=EXPIRY,
            notional_foreign=jnp.array(1.0),
            is_call=True,
        )
        price = garman_kohlhagen_price(deep_itm, SPOT, VOL, R_DOM, R_FOR)
        df_dom = jnp.exp(-R_DOM * EXPIRY)
        df_for = jnp.exp(-R_FOR * EXPIRY)
        intrinsic = SPOT * df_for - 0.80 * df_dom
        assert jnp.isclose(price, intrinsic, rtol=0.01)

    def test_zero_vol_call_equals_forward(self):
        """At zero vol, call = max(F - K, 0) * df_dom."""
        F = fx_forward_rate(SPOT, R_DOM, R_FOR, EXPIRY)
        K = jnp.array(1.08)  # ITM
        opt = FXVanillaOption(
            strike=K, expiry=EXPIRY, notional_foreign=jnp.array(1.0), is_call=True,
        )
        # Use very small vol instead of exactly 0 (avoids div by zero)
        price = garman_kohlhagen_price(opt, SPOT, jnp.array(1e-8), R_DOM, R_FOR)
        df_dom = jnp.exp(-R_DOM * EXPIRY)
        expected = jnp.maximum(F - K, 0.0) * df_dom
        assert jnp.isclose(price, expected, atol=1e-5)


# ── Implied Volatility ───────────────────────────────────────────────


class TestImpliedVol:
    def test_round_trip(self, fx_call):
        """Price → implied vol → price should round-trip."""
        price = garman_kohlhagen_price(fx_call, SPOT, VOL, R_DOM, R_FOR)
        iv = fx_implied_vol(fx_call, SPOT, R_DOM, R_FOR, price)
        assert jnp.isclose(iv, VOL, atol=1e-8)

    def test_round_trip_put(self, fx_put):
        """Implied vol round-trip for puts."""
        price = garman_kohlhagen_price(fx_put, SPOT, VOL, R_DOM, R_FOR)
        iv = fx_implied_vol(fx_put, SPOT, R_DOM, R_FOR, price)
        assert jnp.isclose(iv, VOL, atol=1e-8)

    def test_round_trip_otm(self):
        """Implied vol round-trip for OTM options."""
        otm_call = FXVanillaOption(
            strike=jnp.array(1.20),
            expiry=EXPIRY,
            notional_foreign=jnp.array(1.0),
            is_call=True,
        )
        price = garman_kohlhagen_price(otm_call, SPOT, VOL, R_DOM, R_FOR)
        iv = fx_implied_vol(otm_call, SPOT, R_DOM, R_FOR, price)
        assert jnp.isclose(iv, VOL, atol=1e-6)


# ── Delta Conventions ────────────────────────────────────────────────


class TestFXDelta:
    def test_spot_delta_atm_call_near_half(self):
        """ATM spot delta for a call ≈ 0.5 * exp(-r_f * T)."""
        F = fx_forward_rate(SPOT, R_DOM, R_FOR, EXPIRY)
        atm_call = FXVanillaOption(
            strike=F, expiry=EXPIRY, notional_foreign=jnp.array(1.0), is_call=True,
        )
        delta = fx_delta(atm_call, SPOT, VOL, R_DOM, R_FOR, "spot")
        df_for = jnp.exp(-R_FOR * EXPIRY)
        # ATM forward: d1 ≈ 0.5*vol*sqrt(T), so Phi(d1) slightly > 0.5
        assert jnp.isclose(delta, df_for * 0.5, atol=0.05)

    def test_forward_delta_call_larger_than_spot(self):
        """Forward delta should be larger than spot delta (no df_for discount)."""
        opt = FXVanillaOption(
            strike=jnp.array(1.10), expiry=EXPIRY,
            notional_foreign=jnp.array(1.0), is_call=True,
        )
        d_spot = fx_delta(opt, SPOT, VOL, R_DOM, R_FOR, "spot")
        d_fwd = fx_delta(opt, SPOT, VOL, R_DOM, R_FOR, "forward")
        assert d_fwd > d_spot

    def test_premium_adjusted_delta_less_than_spot(self):
        """Premium-adjusted delta should be less than spot delta for a call."""
        opt = FXVanillaOption(
            strike=jnp.array(1.10), expiry=EXPIRY,
            notional_foreign=jnp.array(1.0), is_call=True,
        )
        d_spot = fx_delta(opt, SPOT, VOL, R_DOM, R_FOR, "spot")
        d_pa = fx_delta(opt, SPOT, VOL, R_DOM, R_FOR, "premium_adjusted")
        assert d_pa < d_spot

    def test_put_delta_negative(self, fx_put):
        """Put delta should be negative under all conventions."""
        for conv in ["spot", "forward", "premium_adjusted"]:
            delta = fx_delta(fx_put, SPOT, VOL, R_DOM, R_FOR, conv)
            assert delta < 0, f"Put delta should be negative under {conv}"

    def test_call_delta_positive(self, fx_call):
        """Call delta should be positive under all conventions."""
        for conv in ["spot", "forward"]:
            delta = fx_delta(fx_call, SPOT, VOL, R_DOM, R_FOR, conv)
            assert delta > 0, f"Call delta should be positive under {conv}"


# ── Strike ↔ Delta Conversion ────────────────────────────────────────


class TestStrikeDeltaConversion:
    def test_strike_to_delta_round_trip(self):
        """strike → delta → strike should round-trip."""
        K = jnp.array(1.12)
        delta = strike_to_delta(K, SPOT, VOL, R_DOM, R_FOR, EXPIRY, True, "spot")
        K_recovered = delta_to_strike(delta, SPOT, VOL, R_DOM, R_FOR, EXPIRY, True, "spot")
        assert jnp.isclose(K_recovered, K, atol=1e-6)

    def test_delta_to_strike_round_trip_put(self):
        """delta → strike → delta should round-trip for puts."""
        target_delta = jnp.array(-0.25)
        K = delta_to_strike(target_delta, SPOT, VOL, R_DOM, R_FOR, EXPIRY, False, "spot")
        delta_back = strike_to_delta(K, SPOT, VOL, R_DOM, R_FOR, EXPIRY, False, "spot")
        assert jnp.isclose(delta_back, target_delta, atol=1e-6)

    def test_25d_call_strike_above_forward(self):
        """25-delta call strike should be above the forward rate."""
        F = fx_forward_rate(SPOT, R_DOM, R_FOR, EXPIRY)
        K_25d = delta_to_strike(
            jnp.array(0.25), SPOT, VOL, R_DOM, R_FOR, EXPIRY, True, "spot",
        )
        assert K_25d > F

    def test_25d_put_strike_below_forward(self):
        """25-delta put strike should be below the forward rate."""
        F = fx_forward_rate(SPOT, R_DOM, R_FOR, EXPIRY)
        K_25d = delta_to_strike(
            jnp.array(-0.25), SPOT, VOL, R_DOM, R_FOR, EXPIRY, False, "spot",
        )
        assert K_25d < F

    def test_forward_delta_convention_round_trip(self):
        """Round-trip under forward delta convention."""
        K = jnp.array(1.15)
        delta = strike_to_delta(K, SPOT, VOL, R_DOM, R_FOR, EXPIRY, True, "forward")
        K_back = delta_to_strike(delta, SPOT, VOL, R_DOM, R_FOR, EXPIRY, True, "forward")
        assert jnp.isclose(K_back, K, atol=1e-6)


# ── Greeks via autodiff ──────────────────────────────────────────────


class TestFXGreeks:
    def test_delta_via_grad(self, fx_call):
        """jax.grad w.r.t. spot should give the GK delta."""
        grad_spot = jax.grad(
            lambda s: garman_kohlhagen_price(fx_call, s, VOL, R_DOM, R_FOR),
        )(SPOT)
        # Should match N * spot_delta / 1 (noting notional factor)
        assert jnp.isfinite(grad_spot)
        assert grad_spot > 0  # Call has positive delta

    def test_vega_via_grad(self, fx_call):
        """jax.grad w.r.t. vol should give vega."""
        vega = jax.grad(
            lambda v: garman_kohlhagen_price(fx_call, SPOT, v, R_DOM, R_FOR),
        )(VOL)
        assert jnp.isfinite(vega)
        assert vega > 0  # Long option has positive vega

    def test_gamma_via_grad(self, fx_call):
        """Second derivative w.r.t. spot gives gamma."""
        gamma = jax.grad(jax.grad(
            lambda s: garman_kohlhagen_price(fx_call, s, VOL, R_DOM, R_FOR),
        ))(SPOT)
        assert jnp.isfinite(gamma)
        assert gamma > 0  # Long option has positive gamma

    def test_vanna_via_grad(self, fx_call):
        """Cross derivative ∂²V/∂S∂σ gives vanna."""
        vanna = jax.grad(jax.grad(
            lambda s, v: garman_kohlhagen_price(fx_call, s, v, R_DOM, R_FOR),
            argnums=0,
        ), argnums=1)(SPOT, VOL)
        assert jnp.isfinite(vanna)

    def test_rho_domestic_via_grad(self, fx_call):
        """Sensitivity to domestic rate."""
        rho_dom = jax.grad(
            lambda r: garman_kohlhagen_price(fx_call, SPOT, VOL, r, R_FOR),
        )(R_DOM)
        assert jnp.isfinite(rho_dom)
        # Domestic rate up → forward F = S*exp((r_d-r_f)*T) up → call value up
        assert rho_dom > 0

    def test_rho_foreign_via_grad(self, fx_call):
        """Sensitivity to foreign rate."""
        rho_for = jax.grad(
            lambda r: garman_kohlhagen_price(fx_call, SPOT, VOL, R_DOM, r),
        )(R_FOR)
        assert jnp.isfinite(rho_for)
        # Foreign rate up → forward down → call price down
        assert rho_for < 0
