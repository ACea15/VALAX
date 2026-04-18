# FX Derivatives

The FX market is the largest financial market by volume, with over \$7.5 trillion
daily turnover. VALAX's base FX instruments (forwards and vanilla options) are
implemented and documented in [Analytical Pricing § FX Options](analytical.md#fx-options-garman-kohlhagen).
This page covers the extended set: quanto options, TARFs, and FX swaps. For the
full mathematical treatment (Garman-Kohlhagen SDE, three delta conventions,
premium-adjusted delta, ATM delta-neutral straddle), see
[Models & Theory §2.7](../theory.md#27-garman-kohlhagen-fx-options).

All FX instruments use the **FOR/DOM** convention: the foreign currency is the asset,
and the domestic currency is the numeraire. For example, EUR/USD = 1.10 means 1 EUR
costs 1.10 USD.

!!! info "Implementation status"
    - **Implemented**: `FXForward`, `FXVanillaOption`, `FXBarrierOption`
      (instrument defined; analytical barrier pricing TBD). See
      `valax/pricing/analytic/fx.py` for Garman-Kohlhagen pricing, forward valuation,
      implied vol inversion, and strike↔delta conversion in all three delta
      conventions.
    - **Planned**: Quanto, TARF, FX Swap (see [Roadmap Tier 3.3](../roadmap.md#33-fx-derivatives)).

## Quanto Option

!!! warning "Planned"

**Market context.** A quanto (quantity-adjusted) option provides exposure to a
**foreign asset** with the payoff converted to the **domestic currency** at a
**fixed FX rate** rather than the prevailing spot rate. This eliminates FX risk
for the investor while maintaining the foreign asset exposure. Quanto options are
common in cross-border equity structured products and commodity-linked notes.

### Pricing

The modified Garman-Kohlhagen formula for a quanto call is:

$$C_{\text{quanto}} = Q \cdot e^{-r_d T} \left[ F'\, \Phi(d_1') - K\, \Phi(d_2') \right]$$

where:

$$F' = S \cdot e^{(r_f - q - \rho\, \sigma_S\, \sigma_{\text{FX}})T}$$

- $Q$ — fixed quanto FX rate (domestic per foreign)
- $S$ — foreign-currency asset price
- $r_d$ — domestic risk-free rate
- $r_f$ — foreign risk-free rate
- $q$ — asset dividend yield
- $\sigma_S$ — volatility of the foreign asset
- $\sigma_{\text{FX}}$ — volatility of the FX rate
- $\rho$ — correlation between the foreign asset and the FX rate

The term $-\rho\, \sigma_S\, \sigma_{\text{FX}}$ is the **quanto adjustment**. Its
intuition: when $\rho > 0$ (asset and FX move together), a rise in the asset price
is accompanied by a depreciation of the foreign currency, reducing the domestic-
currency value. The drift adjustment compensates for this effect.

### Code Example

```python
from valax.instruments import QuantoOption
import jax.numpy as jnp

# Quanto call on Nikkei 225 (JPY asset), payout in USD
quanto_call = QuantoOption(
    strike=jnp.array(38_000.0),     # strike in JPY
    expiry=jnp.array(0.5),          # 6-month expiry
    notional=jnp.array(100_000.0),  # USD notional
    quanto_fx_rate=jnp.array(0.0067),  # fixed rate: 1 JPY = 0.0067 USD
    is_call=True,
    currency_pair="JPY/USD",
)

# The correlation and FX vol are passed to the pricing function,
# not stored in the instrument (separation of data and model)
```

---

## Target Accrual Range Forward (TARF)

!!! warning "Planned"

**Market context.** TARFs are popular structured FX products, particularly in
Asian markets (USD/CNH, USD/KRW, USD/TWD). They offer leveraged exposure to an FX
rate with an automatic termination feature when cumulative gains reach a target.
TARFs became notorious during the 2008 and 2015 FX crises for their asymmetric
risk profile.

### Payoff Mechanics (Step by Step)

At each fixing date $t_k$ ($k = 1, \ldots, n$):

1. **If $S(t_k) > K$ (favorable):** Gain $= (S(t_k) - K) \times N_k$. Add to
   accumulated gains.
   - **If accumulated gains $\geq$ target:** Contract **terminates**. The final
     fixing may be scaled down (partial redemption) so cumulative gains exactly
     equal the target.

2. **If $S(t_k) \leq K$ (unfavorable):** Loss $= (K - S(t_k)) \times N_k \times \lambda$
   where $\lambda$ is the **leverage factor** (typically 2×). The client must pay
   the leveraged loss.

The asymmetry — gains are capped (by the target) while losses are leveraged — makes
TARFs a net short volatility / short gamma position for the client.

### Pricing

TARFs are highly **path-dependent** due to the target accumulation and early
termination feature. Pricing requires Monte Carlo simulation with:

- Simulation of FX spot at each fixing date
- Tracking of cumulative gains
- Early termination logic
- Partial redemption at the terminal fixing

No closed-form or PDE approach is feasible.

### Code Example

```python
from valax.instruments import TARF
from valax.dates import ymd_to_ordinal
import jax.numpy as jnp

tarf = TARF(
    fixing_dates=jnp.array([
        ymd_to_ordinal(2025, 4, 15),
        ymd_to_ordinal(2025, 5, 15),
        ymd_to_ordinal(2025, 6, 15),
        ymd_to_ordinal(2025, 7, 15),
        ymd_to_ordinal(2025, 8, 15),
        ymd_to_ordinal(2025, 9, 15),
        ymd_to_ordinal(2025, 10, 15),
        ymd_to_ordinal(2025, 11, 15),
        ymd_to_ordinal(2025, 12, 15),
        ymd_to_ordinal(2026, 1, 15),
        ymd_to_ordinal(2026, 2, 15),
        ymd_to_ordinal(2026, 3, 15),
    ]),
    strike=jnp.array(7.20),           # USD/CNH strike
    target=jnp.array(0.50),           # target accrual (in FX points)
    notional_per_fixing=jnp.array(1_000_000.0),
    leverage=jnp.array(2.0),          # 2x leverage on losses
    is_buy=True,
    currency_pair="USD/CNH",
)
```

!!! warning
    TARFs have **unlimited downside** (leveraged losses with no floor) and **capped
    upside** (target termination). The client is net short volatility and short
    gamma. Ensure that risk limits and scenario analysis cover extreme FX moves.

---

## FX Swap

!!! warning "Planned"

**Market context.** FX swaps are the **most traded FX instrument by volume**,
representing roughly half of all FX market turnover. They consist of two
simultaneous and opposite FX transactions at different value dates. FX swaps are
primarily used for **short-term funding**, **hedging**, and **rolling forward
exposures**.

### Cashflows

An FX swap has two legs:

| Leg | Action | Rate | Settlement |
|-----|--------|------|------------|
| **Near leg** | Buy (sell) foreign, sell (buy) domestic | Spot rate $S$ | Near date $T_1$ |
| **Far leg** | Sell (buy) foreign, buy (sell) domestic | Forward rate $F$ | Far date $T_2$ |

The **swap points** (forward points) represent the interest rate differential
between the two currencies:

$$\text{Swap points} = F - S = S \cdot \left(e^{(r_d - r_f)\, T} - 1\right)$$

where $r_d$ and $r_f$ are the domestic and foreign interest rates and $T = T_2 - T_1$
in year fractions.

When $r_d > r_f$, swap points are positive (forward premium); when $r_d < r_f$,
swap points are negative (forward discount).

### Code Example

```python
from valax.instruments import FXSwap
from valax.dates import ymd_to_ordinal
import jax.numpy as jnp

# EUR/USD spot-1M FX swap (buy EUR spot, sell EUR 1M forward)
fx_swap = FXSwap(
    near_date=ymd_to_ordinal(2025, 3, 19),   # T+2 spot
    far_date=ymd_to_ordinal(2025, 4, 17),     # 1-month forward
    spot_rate=jnp.array(1.0850),              # EUR/USD spot
    forward_rate=jnp.array(1.0862),           # EUR/USD 1M forward
    notional_foreign=jnp.array(10_000_000.0), # 10M EUR
    is_buy_near=True,                         # buy EUR on near leg
    currency_pair="EUR/USD",
)
```
