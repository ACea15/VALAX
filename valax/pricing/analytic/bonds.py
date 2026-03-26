"""Analytical pricing for fixed income instruments.

Pure functions: price = f(bond, curve_or_yield). All are differentiable
via jax.grad, giving duration, convexity, and key-rate durations for free.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.instruments.bonds import ZeroCouponBond, FixedRateBond
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction


# ── Zero-coupon bond pricing ─────────────────────────────────────────

def zero_coupon_bond_price(
    bond: ZeroCouponBond,
    curve: DiscountCurve,
) -> Float[Array, ""]:
    """Price a zero-coupon bond from a discount curve.

    Price = face_value * DF(maturity)
    """
    df = curve(bond.maturity)
    return bond.face_value * df


# ── Fixed-rate bond pricing ──────────────────────────────────────────

def fixed_rate_bond_price(
    bond: FixedRateBond,
    curve: DiscountCurve,
) -> Float[Array, ""]:
    """Price a fixed-rate coupon bond from a discount curve.

    Discounts each coupon and the face value at maturity using the
    curve's interpolated discount factors.

    Only future cash flows (payment_date > settlement_date) are included.
    """
    coupon = bond.face_value * bond.coupon_rate / bond.frequency

    dfs = curve(bond.payment_dates)
    # Mask out past payments
    future_mask = (bond.payment_dates > bond.settlement_date).astype(jnp.float64)

    # Coupon PV
    coupon_pv = jnp.sum(coupon * dfs * future_mask)

    # Redemption at maturity
    maturity = bond.payment_dates[-1]
    df_maturity = curve(maturity)

    return coupon_pv + bond.face_value * df_maturity


# ── Yield-based pricing ──────────────────────────────────────────────

def fixed_rate_bond_price_from_yield(
    bond: FixedRateBond,
    ytm: Float[Array, ""],
) -> Float[Array, ""]:
    """Price a fixed-rate bond given a yield-to-maturity.

    Uses standard bond pricing formula:
        P = sum_i [C / (1 + y/f)^i] + F / (1 + y/f)^n

    where i counts future periods from settlement.
    """
    coupon = bond.face_value * bond.coupon_rate / bond.frequency
    future_mask = (bond.payment_dates > bond.settlement_date).astype(jnp.float64)
    n_future = jnp.sum(future_mask)

    # Period indices: 1, 2, ..., n_future for future coupons
    cumulative = jnp.cumsum(future_mask)
    periods = cumulative * future_mask  # zero for past payments

    disc = (1.0 + ytm / bond.frequency) ** (-periods)

    coupon_pv = jnp.sum(coupon * disc * future_mask)
    redemption_pv = bond.face_value * (1.0 + ytm / bond.frequency) ** (-n_future)

    return coupon_pv + redemption_pv


# ── Yield-to-maturity solver ─────────────────────────────────────────

def yield_to_maturity(
    bond: FixedRateBond,
    market_price: Float[Array, ""],
    n_iterations: int = 50,
) -> Float[Array, ""]:
    """Newton-Raphson yield-to-maturity solver.

    Finds y such that fixed_rate_bond_price_from_yield(bond, y) = market_price.
    """
    price_fn = lambda y: fixed_rate_bond_price_from_yield(bond, y)
    dprice_dy = jax.grad(price_fn)

    def newton_step(y, _):
        p = price_fn(y)
        dp = dprice_dy(y)
        dp = jnp.where(jnp.abs(dp) < 1e-12, -1e-12, dp)
        return y - (p - market_price) / dp, None

    y_init = bond.coupon_rate  # initial guess = coupon rate
    y, _ = jax.lax.scan(newton_step, y_init, None, length=n_iterations)
    return y


# ── Risk measures via autodiff ────────────────────────────────────────

def modified_duration(
    bond: FixedRateBond,
    ytm: Float[Array, ""],
) -> Float[Array, ""]:
    """Modified duration: -1/P * dP/dy."""
    price_fn = lambda y: fixed_rate_bond_price_from_yield(bond, y)
    p = price_fn(ytm)
    dp = jax.grad(price_fn)(ytm)
    return -dp / p


def convexity(
    bond: FixedRateBond,
    ytm: Float[Array, ""],
) -> Float[Array, ""]:
    """Convexity: 1/P * d^2P/dy^2."""
    price_fn = lambda y: fixed_rate_bond_price_from_yield(bond, y)
    p = price_fn(ytm)
    d2p = jax.grad(jax.grad(price_fn))(ytm)
    return d2p / p


def key_rate_durations(
    bond: FixedRateBond,
    curve: DiscountCurve,
) -> Float[Array, " n_pillars"]:
    """Key-rate durations: sensitivity of bond price to each curve zero rate.

    KRD_i = -1/P * dP/dr_i

    where r_i is the continuously-compounded zero rate at pillar i.
    Since log(DF_i) = -r_i * t_i, we have dP/dr_i = -t_i * dP/d(log DF_i).

    This is the core advantage of autodiff over finite differences:
    one backward pass gives all key-rate sensitivities simultaneously.
    """
    pillar_times = year_fraction(
        curve.reference_date, curve.pillar_dates, curve.day_count
    )

    def price_from_log_dfs(log_dfs):
        shifted_curve = DiscountCurve(
            pillar_dates=curve.pillar_dates,
            discount_factors=jnp.exp(log_dfs),
            reference_date=curve.reference_date,
            day_count=curve.day_count,
        )
        return fixed_rate_bond_price(bond, shifted_curve)

    log_dfs = jnp.log(curve.discount_factors)
    price = price_from_log_dfs(log_dfs)
    grad_log_dfs = jax.grad(price_from_log_dfs)(log_dfs)

    # dP/dr_i = dP/d(log DF_i) * d(log DF_i)/dr_i = dP/d(log DF_i) * (-t_i)
    # KRD_i = -1/P * dP/dr_i = 1/P * t_i * dP/d(log DF_i)
    return pillar_times * grad_log_dfs / price
