# %% [markdown]
# # VALAX vs QuantLib: Fixed Income
#
# Side-by-side comparison of:
# - Discount curve construction
# - Bond pricing (zero-coupon and fixed-rate)
# - Yield-to-maturity
# - Duration, convexity
# - Key-rate durations (VALAX autodiff advantage)
#
# Validated by: tests/test_quantlib_comparison/test_fixed_income_ql.py

# %% Imports
import jax
import jax.numpy as jnp
import QuantLib as ql
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.curves.discount import DiscountCurve, zero_rate, forward_rate
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
# 1. COMMON MARKET DATA
# ============================================================================

print("=" * 70)
print("VALAX vs QuantLib: Fixed Income Comparison")
print("=" * 70)

# Zero rates for curve construction (continuously compounded)
tenors_years = [0.5, 1, 2, 3, 5, 7, 10]
zero_rates = [0.0425, 0.0410, 0.0395, 0.0385, 0.0375, 0.0370, 0.0365]

print(f"\nInput zero rates (continuous compounding, Act/365):")
tenor_labels = ["6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]
for lbl, r_val in zip(tenor_labels, zero_rates):
    print(f"  {lbl:>4}: {r_val*100:.2f}%")

# ============================================================================
# 2. VALAX — CURVE CONSTRUCTION
# ============================================================================

print(f"\n{'-' * 70}")
print("VALAX: Curve Construction")
print("-" * 70)

today_valax = ymd_to_ordinal(2026, 3, 26)

pillar_dates_valax = jnp.array([
    ymd_to_ordinal(2026, 9, 26),   # 6M
    ymd_to_ordinal(2027, 3, 26),   # 1Y
    ymd_to_ordinal(2028, 3, 26),   # 2Y
    ymd_to_ordinal(2029, 3, 26),   # 3Y
    ymd_to_ordinal(2031, 3, 26),   # 5Y
    ymd_to_ordinal(2033, 3, 26),   # 7Y
    ymd_to_ordinal(2036, 3, 26),   # 10Y
])

pillar_times = year_fraction(today_valax, pillar_dates_valax, "act_365")
discount_factors_valax = jnp.exp(-jnp.array(zero_rates) * pillar_times)

curve_valax = DiscountCurve(
    pillar_dates=pillar_dates_valax,
    discount_factors=discount_factors_valax,
    reference_date=today_valax,
    day_count="act_365",
)

print("Discount factors:")
for i, lbl in enumerate(tenor_labels):
    df = float(discount_factors_valax[i])
    print(f"  {lbl:>4}: {df:.8f}")

# ============================================================================
# 3. QUANTLIB — CURVE CONSTRUCTION
# ============================================================================

print(f"\n{'-' * 70}")
print("QuantLib: Curve Construction")
print("-" * 70)

today_ql = ql.Date(26, 3, 2026)
ql.Settings.instance().evaluationDate = today_ql

# Build zero curve from rates
dates_ql = [today_ql + ql.Period(int(t * 12 + 0.5), ql.Months) for t in tenors_years]
# Manually set dates to match VALAX exactly
dates_ql = [
    ql.Date(26, 9, 2026),   # 6M
    ql.Date(26, 3, 2027),   # 1Y
    ql.Date(26, 3, 2028),   # 2Y
    ql.Date(26, 3, 2029),   # 3Y
    ql.Date(26, 3, 2031),   # 5Y
    ql.Date(26, 3, 2033),   # 7Y
    ql.Date(26, 3, 2036),   # 10Y
]

curve_ql = ql.ZeroCurve(
    [today_ql] + dates_ql,
    [zero_rates[0]] + zero_rates,   # QuantLib needs today's point
    ql.Actual365Fixed(),
    ql.NullCalendar(),
    ql.Linear(),
    ql.Continuous,
)
curve_handle = ql.YieldTermStructureHandle(curve_ql)

print("Discount factors:")
for i, (lbl, dt) in enumerate(zip(tenor_labels, dates_ql)):
    df = curve_ql.discount(dt)
    print(f"  {lbl:>4}: {df:.8f}")

# ============================================================================
# 4. COMPARE DISCOUNT FACTORS
# ============================================================================

print(f"\n{'=' * 70}")
print("CURVE COMPARISON")
print("=" * 70)
print(f"\n{'Tenor':>6} {'VALAX DF':>14} {'QuantLib DF':>14} {'Diff':>14}")
print("-" * 50)
for i, lbl in enumerate(tenor_labels):
    v_df = float(discount_factors_valax[i])
    q_df = curve_ql.discount(dates_ql[i])
    print(f"{lbl:>6} {v_df:>14.8f} {q_df:>14.8f} {abs(v_df - q_df):>14.2e}")

# ============================================================================
# 5. FIXED-RATE BOND PRICING
# ============================================================================

print(f"\n{'=' * 70}")
print("FIXED-RATE BOND: 5Y 4% Semiannual")
print("=" * 70)

# --- VALAX ---
payment_dates_valax = jnp.array([
    ymd_to_ordinal(2026, 9, 26),   # 6M
    ymd_to_ordinal(2027, 3, 26),   # 1Y
    ymd_to_ordinal(2027, 9, 26),   # 1.5Y
    ymd_to_ordinal(2028, 3, 26),   # 2Y
    ymd_to_ordinal(2028, 9, 26),   # 2.5Y
    ymd_to_ordinal(2029, 3, 26),   # 3Y
    ymd_to_ordinal(2029, 9, 26),   # 3.5Y
    ymd_to_ordinal(2030, 3, 26),   # 4Y
    ymd_to_ordinal(2030, 9, 26),   # 4.5Y
    ymd_to_ordinal(2031, 3, 26),   # 5Y
])

bond_valax = FixedRateBond(
    payment_dates=payment_dates_valax,
    settlement_date=today_valax,
    coupon_rate=jnp.array(0.04),
    face_value=jnp.array(100.0),
    frequency=2,
    day_count="act_365",
)

valax_bond_price = float(fixed_rate_bond_price(bond_valax, curve_valax))
valax_ytm = float(yield_to_maturity(bond_valax, jnp.array(valax_bond_price)))
valax_dur = float(modified_duration(bond_valax, jnp.array(valax_ytm)))
valax_conv = float(convexity(bond_valax, jnp.array(valax_ytm)))

print(f"\nVALAX:")
print(f"  Dirty price:       {valax_bond_price:.6f}")
print(f"  YTM:               {valax_ytm*100:.4f}%")
print(f"  Modified duration: {valax_dur:.4f}")
print(f"  Convexity:         {valax_conv:.4f}")

# --- QuantLib ---
schedule = ql.Schedule(
    today_ql,
    ql.Date(26, 3, 2031),
    ql.Period(ql.Semiannual),
    ql.NullCalendar(),
    ql.Unadjusted,
    ql.Unadjusted,
    ql.DateGeneration.Forward,
    False,
)

bond_ql = ql.FixedRateBond(
    0,                          # settlement days
    100.0,                      # face value
    schedule,
    [0.04],                     # coupon rate
    ql.Actual365Fixed(),
)

bond_engine = ql.DiscountingBondEngine(curve_handle)
bond_ql.setPricingEngine(bond_engine)

ql_bond_price = bond_ql.dirtyPrice()

# QuantLib yield: use the 3-arg overload (DayCounter, Compounding, Frequency)
ql_ytm = bond_ql.bondYield(ql.Actual365Fixed(), ql.Continuous, ql.NoFrequency)

# Duration & convexity via QuantLib's BondFunctions
ql_dur = ql.BondFunctions.duration(
    bond_ql, ql_ytm, ql.Actual365Fixed(), ql.Continuous, ql.NoFrequency, ql.Duration.Modified
)
ql_conv = ql.BondFunctions.convexity(
    bond_ql, ql_ytm, ql.Actual365Fixed(), ql.Continuous, ql.NoFrequency
)

print(f"\nQuantLib:")
print(f"  Dirty price:       {ql_bond_price:.6f}")
print(f"  YTM:               {ql_ytm*100:.4f}%")
print(f"  Modified duration: {ql_dur:.4f}")
print(f"  Convexity:         {ql_conv:.4f}")

# --- Comparison ---
print(f"\n{'Metric':<22} {'VALAX':>14} {'QuantLib':>14} {'Diff':>14}")
print("-" * 66)
print(f"{'Dirty Price':<22} {valax_bond_price:>14.6f} {ql_bond_price:>14.6f} {abs(valax_bond_price - ql_bond_price):>14.2e}")
print(f"{'YTM':<22} {valax_ytm*100:>13.4f}% {ql_ytm*100:>13.4f}% {abs(valax_ytm - ql_ytm)*10000:>12.2f}bp")
print(f"{'Mod. Duration':<22} {valax_dur:>14.4f} {ql_dur:>14.4f} {abs(valax_dur - ql_dur):>14.2e}")
print(f"{'Convexity':<22} {valax_conv:>14.4f} {ql_conv:>14.4f} {abs(valax_conv - ql_conv):>14.2e}")

# ============================================================================
# 6. KEY-RATE DURATIONS — VALAX AUTODIFF ADVANTAGE
# ============================================================================

print(f"\n{'=' * 70}")
print("KEY-RATE DURATIONS — VALAX Autodiff Advantage")
print("=" * 70)

krd = key_rate_durations(bond_valax, curve_valax)

print(f"\nVALAX: One backward pass gives sensitivities to ALL curve pillars.")
print(f"\n{'Tenor':>6} {'KRD':>10}")
print("-" * 18)
for i, lbl in enumerate(tenor_labels):
    print(f"{lbl:>6} {float(krd[i]):10.6f}")
print(f"{'Sum':>6} {float(jnp.sum(krd)):10.6f}  (≈ modified duration: {valax_dur:.4f})")

print(f"""
QuantLib approach for key-rate durations:
  → Requires N separate curve bumps (one per pillar)
  → Each bump requires rebuilding the curve and repricing
  → For N pillars, that's N+1 pricings

VALAX approach:
  → One call to jax.grad gives all {len(tenor_labels)} sensitivities
  → Uses reverse-mode autodiff (backpropagation)
  → Cost ≈ 2-3x a single pricing call, regardless of N
""")

# ============================================================================
# 7. API DESIGN COMPARISON
# ============================================================================

print(f"{'=' * 70}")
print("API DESIGN: Fixed Income")
print("=" * 70)

print("""
┌──────────────────────┬─────────────────────────────────┬─────────────────────────────────┐
│ Aspect               │ VALAX                           │ QuantLib                        │
├──────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ Curve construction   │ DiscountCurve(dates, DFs)       │ ZeroCurve + Handle + Settings   │
│ Bond definition      │ FixedRateBond(dates, rate)      │ Schedule + Bond + Engine setup  │
│ Pricing              │ fixed_rate_bond_price(bond, crv)│ bond.setPricingEngine(); NPV()  │
│ YTM                  │ yield_to_maturity(bond, price)  │ bond.bondYield(price, ...)      │
│ Duration             │ jax.grad(price_fn)(ytm)         │ BondFunctions.duration(...)     │
│ Key-rate durations   │ One jax.grad call (all pillars) │ N bumps × full reprice          │
│ Curve sensitivities  │ Autodiff through interpolation  │ Manual perturbation             │
│ Differentiable       │ Yes — entire pipeline           │ No                              │
└──────────────────────┴─────────────────────────────────┴─────────────────────────────────┘
""")
