# %% [markdown]
# # Interest Rate Derivatives: Caps, Floors, Swaps, and Swaptions
#
# This example covers:
# - Building a realistic rate curve
# - Cap/floor pricing with Black-76 and Bachelier
# - Swap pricing and par swap rates
# - Swaption pricing
# - Greeks via autodiff through the rate curve

# %% Imports
import jax
import jax.numpy as jnp
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.curves.discount import DiscountCurve, forward_rate
from valax.instruments.rates import Caplet, Cap, InterestRateSwap, Swaption
from valax.pricing.analytic.caplets import (
    caplet_price_black76,
    caplet_price_bachelier,
    cap_price_black76,
)
from valax.pricing.analytic.swaptions import (
    swap_rate,
    swap_price,
    swaption_price_black76,
    swaption_price_bachelier,
)

# ============================================================================
# 1. BUILD A RATE CURVE
# ============================================================================

# %% Synthetic USD curve as of March 2026
today = ymd_to_ordinal(2026, 3, 26)

pillar_dates = jnp.array([
    ymd_to_ordinal(2026, 6, 26),   # 3M
    ymd_to_ordinal(2026, 9, 26),   # 6M
    ymd_to_ordinal(2027, 3, 26),   # 1Y
    ymd_to_ordinal(2028, 3, 26),   # 2Y
    ymd_to_ordinal(2029, 3, 26),   # 3Y
    ymd_to_ordinal(2031, 3, 26),   # 5Y
    ymd_to_ordinal(2033, 3, 26),   # 7Y
    ymd_to_ordinal(2036, 3, 26),   # 10Y
])

# Inverted curve — short rates higher than long rates
zero_rates = jnp.array([0.0440, 0.0430, 0.0415, 0.0395, 0.0385, 0.0375, 0.0370, 0.0365])
pillar_times = year_fraction(today, pillar_dates, "act_365")
discount_factors = jnp.exp(-zero_rates * pillar_times)

curve = DiscountCurve(
    pillar_dates=pillar_dates,
    discount_factors=discount_factors,
    reference_date=today,
    day_count="act_365",
)

print("--- USD Rate Curve ---")
tenors = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]
for i, t in enumerate(tenors):
    print(f"  {t:>4}: {float(zero_rates[i])*100:.2f}%")

# ============================================================================
# 2. CAPLET PRICING
# ============================================================================

# %% Price a single caplet
# A 1Y caplet on 3M LIBOR: fixes in 1Y, pays in 1Y+3M
caplet = Caplet(
    fixing_date=ymd_to_ordinal(2027, 3, 26),
    start_date=ymd_to_ordinal(2027, 3, 26),
    end_date=ymd_to_ordinal(2027, 6, 26),
    strike=jnp.array(0.04),        # 4% cap rate
    notional=jnp.array(10_000_000.0),  # $10M notional
    is_cap=True,
    day_count="act_360",
)

# Black-76 vol for the caplet (lognormal vol of the forward rate)
cap_vol = jnp.array(0.45)  # 45% Black vol — typical for short-dated caps

caplet_black = caplet_price_black76(caplet, curve, cap_vol)
print(f"\n--- Single Caplet (1Y x 3M, K=4%) ---")
print(f"Notional:    ${float(caplet.notional):,.0f}")
print(f"Black vol:   {float(cap_vol)*100:.0f}%")
print(f"Price:       ${float(caplet_black):,.2f}")

# %% Also price with Bachelier (normal vol)
# Normal vol in absolute terms (e.g., 80 bps annualized)
normal_vol = jnp.array(0.0080)  # 80 bps normal vol
caplet_bach = caplet_price_bachelier(caplet, curve, normal_vol)
print(f"Normal vol:  {float(normal_vol)*10000:.0f} bps")
print(f"Bachelier:   ${float(caplet_bach):,.2f}")

# %% Price a floorlet (same terms, is_cap=False)
floorlet = Caplet(
    fixing_date=caplet.fixing_date,
    start_date=caplet.start_date,
    end_date=caplet.end_date,
    strike=caplet.strike,
    notional=caplet.notional,
    is_cap=False,
    day_count="act_360",
)
floorlet_price = caplet_price_black76(floorlet, curve, cap_vol)
print(f"\nFloorlet:    ${float(floorlet_price):,.2f}")

# ============================================================================
# 3. CAP PRICING (STRIP OF CAPLETS)
# ============================================================================

# %% Build a 3-year quarterly cap at 4%
# This is a strip of 11 caplets (first period is spot, so 11 forward periods)
n_periods = 11
cap_start = ymd_to_ordinal(2026, 6, 26)
fixing_dates = jnp.array([ymd_to_ordinal(2026, 3 + 3 * i, 26) if (3 + 3 * i) <= 12
                           else ymd_to_ordinal(2026 + (3 + 3 * i - 1) // 12,
                                                ((3 + 3 * i - 1) % 12) + 1, 26)
                           for i in range(1, n_periods + 1)])
start_dates = fixing_dates
end_dates = jnp.array([ymd_to_ordinal(2026, 3 + 3 * i, 26) if (3 + 3 * i) <= 12
                         else ymd_to_ordinal(2026 + (3 + 3 * i - 1) // 12,
                                              ((3 + 3 * i - 1) % 12) + 1, 26)
                         for i in range(2, n_periods + 2)])

cap = Cap(
    fixing_dates=fixing_dates,
    start_dates=start_dates,
    end_dates=end_dates,
    strike=jnp.array(0.04),
    notional=jnp.array(10_000_000.0),
    is_cap=True,
    day_count="act_360",
)

flat_vol = jnp.array(0.45)
cap_price = cap_price_black76(cap, curve, flat_vol)
print(f"\n--- 3Y Quarterly Cap (K=4%) ---")
print(f"Notional:    ${float(cap.notional):,.0f}")
print(f"Flat vol:    {float(flat_vol)*100:.0f}%")
print(f"Cap price:   ${float(cap_price):,.2f}")
print(f"Running:     {float(cap_price / cap.notional) * 10000:.1f} bps")

# ============================================================================
# 4. SWAP PRICING
# ============================================================================

# %% Price a 5-year payer swap (pay fixed 3.75%, receive float)
swap_fixed_dates = jnp.array([
    ymd_to_ordinal(2026 + i // 2, 3 + 6 * (i % 2), 26)
    if (3 + 6 * (i % 2)) <= 12
    else ymd_to_ordinal(2026 + i // 2 + 1, (3 + 6 * (i % 2)) - 12, 26)
    for i in range(1, 11)  # 10 semiannual payments
])

swap = InterestRateSwap(
    start_date=today,
    fixed_dates=swap_fixed_dates,
    fixed_rate=jnp.array(0.0375),
    notional=jnp.array(50_000_000.0),  # $50M
    pay_fixed=True,
    day_count="act_360",
)

# Par swap rate (rate that makes NPV = 0)
par_rate = swap_rate(swap, curve)
print(f"\n--- 5Y Interest Rate Swap ---")
print(f"Notional:      ${float(swap.notional):,.0f}")
print(f"Fixed rate:    {float(swap.fixed_rate)*100:.2f}%")
print(f"Par swap rate: {float(par_rate)*100:.4f}%")

# Swap NPV
npv = swap_price(swap, curve)
print(f"Swap NPV:      ${float(npv):,.2f}")
print(f"  (positive = fixed rate is below par => payer benefits)")

# %% Swap DV01: sensitivity to a parallel 1bp shift
def swap_from_log_dfs(log_dfs):
    c = DiscountCurve(
        pillar_dates=curve.pillar_dates,
        discount_factors=jnp.exp(log_dfs),
        reference_date=curve.reference_date,
        day_count=curve.day_count,
    )
    return swap_price(swap, c)

log_dfs = jnp.log(curve.discount_factors)
swap_grad = jax.grad(swap_from_log_dfs)(log_dfs)
swap_dv01 = float(jnp.sum(-pillar_times * swap_grad * 0.0001))
print(f"Swap DV01:     ${swap_dv01:,.2f}")

# ============================================================================
# 5. SWAPTION PRICING
# ============================================================================

# %% Price a 1Y-into-5Y payer swaption
# Right to enter a 5Y payer swap in 1 year
swaption = Swaption(
    expiry_date=ymd_to_ordinal(2027, 3, 26),   # expires in 1Y
    fixed_dates=jnp.array([                     # underlying swap: 5Y semiannual
        ymd_to_ordinal(2027 + i // 2, 3 + 6 * (i % 2), 26)
        if (3 + 6 * (i % 2)) <= 12
        else ymd_to_ordinal(2027 + i // 2 + 1, (3 + 6 * (i % 2)) - 12, 26)
        for i in range(1, 11)
    ]),
    strike=jnp.array(0.04),           # 4% strike
    notional=jnp.array(50_000_000.0),
    is_payer=True,
    day_count="act_360",
)

swaption_vol = jnp.array(0.35)  # 35% Black vol on the swap rate
swaption_black = swaption_price_black76(swaption, curve, swaption_vol)
print(f"\n--- 1Y x 5Y Payer Swaption (K=4%) ---")
print(f"Notional:     ${float(swaption.notional):,.0f}")
print(f"Black vol:    {float(swaption_vol)*100:.0f}%")
print(f"Black price:  ${float(swaption_black):,.2f}")
print(f"Running:      {float(swaption_black / swaption.notional) * 10000:.1f} bps")

# %% Bachelier swaption price
swaption_normal_vol = jnp.array(0.0070)  # 70 bps normal vol
swaption_bach = swaption_price_bachelier(swaption, curve, swaption_normal_vol)
print(f"Normal vol:   {float(swaption_normal_vol)*10000:.0f} bps")
print(f"Bach price:   ${float(swaption_bach):,.2f}")

# %% Swaption Greeks via autodiff
# Vega: sensitivity to Black vol
swaption_vega_fn = jax.grad(
    lambda v: swaption_price_black76(swaption, curve, v)
)
swaption_vega = swaption_vega_fn(swaption_vol)
print(f"\nSwaption vega (dP/dvol): ${float(swaption_vega):,.2f}")
print(f"Vega per 1% vol move:   ${float(swaption_vega) * 0.01:,.2f}")
