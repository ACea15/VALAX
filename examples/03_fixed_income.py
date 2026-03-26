# %% [markdown]
# # Fixed Income: Curves, Bonds, and Autodiff Risk Measures
#
# This example covers:
# - Building discount curves from synthetic market data
# - Pricing zero-coupon and fixed-rate bonds
# - Duration, convexity, and key-rate durations via autodiff
# - Yield-to-maturity solver
# - Curve sensitivity analysis

# %% Imports
import jax
import jax.numpy as jnp
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.curves.discount import DiscountCurve, forward_rate, zero_rate
from valax.instruments.bonds import ZeroCouponBond, FixedRateBond
from valax.pricing.analytic.bonds import (
    zero_coupon_bond_price,
    fixed_rate_bond_price,
    fixed_rate_bond_price_from_yield,
    yield_to_maturity,
    modified_duration,
    convexity,
    key_rate_durations,
)

# ============================================================================
# 1. BUILDING A DISCOUNT CURVE
# ============================================================================

# %% Define a curve from synthetic market zero rates
# Simulate a typical upward-sloping yield curve
today = ymd_to_ordinal(2026, 3, 26)

# Pillar dates: 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y
pillar_dates = jnp.array([
    ymd_to_ordinal(2026, 9, 26),   # 6M
    ymd_to_ordinal(2027, 3, 26),   # 1Y
    ymd_to_ordinal(2028, 3, 26),   # 2Y
    ymd_to_ordinal(2029, 3, 26),   # 3Y
    ymd_to_ordinal(2031, 3, 26),   # 5Y
    ymd_to_ordinal(2033, 3, 26),   # 7Y
    ymd_to_ordinal(2036, 3, 26),   # 10Y
    ymd_to_ordinal(2046, 3, 26),   # 20Y
    ymd_to_ordinal(2056, 3, 26),   # 30Y
])

# Zero rates: typical upward-sloping curve
zero_rates_bp = jnp.array([425, 410, 395, 385, 375, 370, 365, 360, 355])  # basis points
zero_rates_cont = zero_rates_bp / 10000.0  # convert to decimal

# Convert zero rates to discount factors: DF(T) = exp(-r * T)
pillar_times = year_fraction(today, pillar_dates, "act_365")
discount_factors = jnp.exp(-zero_rates_cont * pillar_times)

curve = DiscountCurve(
    pillar_dates=pillar_dates,
    discount_factors=discount_factors,
    reference_date=today,
    day_count="act_365",
)

print("--- Discount Curve ---")
print(f"{'Tenor':>6} {'Zero Rate':>10} {'DF':>10}")
tenors = ["6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
for i, tenor in enumerate(tenors):
    print(f"{tenor:>6} {float(zero_rates_cont[i])*100:9.2f}% {float(discount_factors[i]):10.6f}")

# %% Query the curve at arbitrary dates
# The curve interpolates via log-linear DF (piecewise-constant forward rates)
query_date = ymd_to_ordinal(2030, 3, 26)  # 4Y point (between pillars)
df_4y = curve(query_date)
r_4y = zero_rate(curve, query_date)
print(f"\n4Y interpolated: DF={float(df_4y):.6f}, zero rate={float(r_4y)*100:.2f}%")

# %% Forward rates
# Simply-compounded forward rate between 2Y and 5Y
start_2y = ymd_to_ordinal(2028, 3, 26)
end_5y = ymd_to_ordinal(2031, 3, 26)
fwd_2y5y = forward_rate(curve, start_2y, end_5y)
print(f"Forward rate 2Y-5Y: {float(fwd_2y5y)*100:.2f}%")

# ============================================================================
# 2. ZERO-COUPON BOND PRICING
# ============================================================================

# %% Price a 5-year zero-coupon bond
zcb = ZeroCouponBond(
    maturity=ymd_to_ordinal(2031, 3, 26),
    face_value=jnp.array(1_000_000.0),  # $1M face
)

zcb_price = zero_coupon_bond_price(zcb, curve)
print(f"\n--- Zero-Coupon Bond ---")
print(f"Face value: ${float(zcb.face_value):,.0f}")
print(f"Maturity:   5Y")
print(f"Price:      ${float(zcb_price):,.2f}")

# ============================================================================
# 3. FIXED-RATE COUPON BOND
# ============================================================================

# %% Build a 10-year 4% semiannual bond
# Payment dates every 6 months for 10 years
payment_dates = jnp.array([
    ymd_to_ordinal(2026 + (i // 2), 3 + 6 * (i % 2), 26)
    if (3 + 6 * (i % 2)) <= 12
    else ymd_to_ordinal(2026 + (i // 2) + 1, (3 + 6 * (i % 2)) - 12, 26)
    for i in range(1, 21)  # 20 semiannual payments
])

bond = FixedRateBond(
    payment_dates=payment_dates,
    settlement_date=today,
    coupon_rate=jnp.array(0.04),      # 4% annual coupon
    face_value=jnp.array(1_000_000.0),
    frequency=2,                       # semiannual
    day_count="act_365",
)

# %% Price from the discount curve
bond_price = fixed_rate_bond_price(bond, curve)
print(f"\n--- 10Y 4% Semiannual Bond ---")
print(f"Coupon rate: 4.00%")
print(f"Face value:  ${float(bond.face_value):,.0f}")
print(f"Price:       ${float(bond_price):,.2f}")
print(f"Clean price: {float(bond_price) / float(bond.face_value) * 100:.4f}%")

# ============================================================================
# 4. YIELD AND RISK MEASURES VIA AUTODIFF
# ============================================================================

# %% Yield-to-maturity (Newton-Raphson with autodiff Jacobian)
ytm = yield_to_maturity(bond, bond_price)
print(f"\nYield-to-maturity: {float(ytm)*100:.4f}%")

# Verify round-trip: price from yield should match curve price
price_from_ytm = fixed_rate_bond_price_from_yield(bond, ytm)
print(f"Price from YTM:    ${float(price_from_ytm):,.2f}  (error: ${abs(float(bond_price - price_from_ytm)):.2f})")

# %% Modified duration via autodiff
# This uses jax.grad(price_fn)(ytm), not the closed-form formula.
# The answer is exact, not an approximation.
mod_dur = modified_duration(bond, ytm)
print(f"\nModified duration: {float(mod_dur):.4f} years")

# %% Convexity via autodiff (second derivative)
conv = convexity(bond, ytm)
print(f"Convexity:         {float(conv):.4f}")

# %% Price impact of a 50bp yield shock
dy = 0.005  # 50 bps
price_approx_1st = float(bond_price) * (1.0 - float(mod_dur) * dy)
price_approx_2nd = float(bond_price) * (1.0 - float(mod_dur) * dy + 0.5 * float(conv) * dy**2)
price_exact = float(fixed_rate_bond_price_from_yield(bond, ytm + dy))

print(f"\nPrice impact of +50bp yield shock:")
print(f"  Duration approx:           ${price_approx_1st:,.2f}")
print(f"  Duration+convexity approx: ${price_approx_2nd:,.2f}")
print(f"  Exact (repriced):          ${price_exact:,.2f}")

# ============================================================================
# 5. KEY-RATE DURATIONS
# ============================================================================

# %% Key-rate durations: sensitivity to each pillar on the curve
# This is the "killer feature" of autodiff in fixed income.
# One backward pass through jax.grad gives sensitivities to ALL curve points.
krd = key_rate_durations(bond, curve)

print(f"\n--- Key-Rate Durations ---")
print(f"{'Tenor':>6} {'KRD':>10}")
for i, tenor in enumerate(tenors):
    print(f"{tenor:>6} {float(krd[i]):10.4f}")
print(f"{'Sum':>6} {float(jnp.sum(krd)):10.4f}  (should ≈ modified duration {float(mod_dur):.4f})")

# ============================================================================
# 6. CURVE SENSITIVITY ANALYSIS
# ============================================================================

# %% How does the bond price change if the 5Y rate moves?
# Autodiff gives this for free — differentiate price w.r.t. curve DFs.
def price_from_log_dfs(log_dfs):
    shifted_curve = DiscountCurve(
        pillar_dates=curve.pillar_dates,
        discount_factors=jnp.exp(log_dfs),
        reference_date=curve.reference_date,
        day_count=curve.day_count,
    )
    return fixed_rate_bond_price(bond, shifted_curve)

log_dfs = jnp.log(curve.discount_factors)
grad_log_dfs = jax.grad(price_from_log_dfs)(log_dfs)

print(f"\n--- dP / d(log DF) at each pillar ---")
print(f"{'Tenor':>6} {'dP/d(logDF)':>14}")
for i, tenor in enumerate(tenors):
    print(f"{tenor:>6} ${float(grad_log_dfs[i]):>13,.2f}")

# %% Parallel shift: impact of a 1bp parallel shift in zero rates
# dP/dr * dr = dP/d(logDF) * d(logDF)/dr * dr = dP/d(logDF) * (-t) * dr
dv01_per_pillar = -pillar_times * grad_log_dfs * 0.0001  # 1bp = 0.01%
total_dv01 = float(jnp.sum(dv01_per_pillar))
print(f"\nDV01 (parallel 1bp shift): ${total_dv01:,.2f}")
