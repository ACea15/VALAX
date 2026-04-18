# Inflation Derivatives

Inflation derivatives are linked to a **Consumer Price Index (CPI)** such as
US CPI-U, Euro HICP, or UK RPI. The key conventions are:

- **Publication lag**: CPI values are published with a 2–3 month delay (e.g., the
  March CPI is published in April).
- **Interpolation**: for daily valuation, the CPI is linearly interpolated between
  monthly publications.
- **Seasonality**: CPI exhibits seasonal patterns (e.g., energy prices, sales
  periods) that must be stripped from market quotes before bootstrapping an
  inflation curve.

The two fundamental inflation swap types — zero-coupon and year-on-year — form the
building blocks for all inflation products. For the mathematical framework (forward
CPI curves, breakeven rates, convexity adjustments, and seasonality), see
[Models & Theory §3.6](../theory.md#36-inflation-curves-and-breakeven-pricing).

## Zero-Coupon Inflation Swap (ZCIS)

**Market context.** The ZCIS is the most liquid inflation derivative and the
primary instrument for bootstrapping an inflation forward curve. A single exchange
occurs at maturity: one party pays (or receives) the cumulative inflation return,
while the other pays a compounded fixed rate.

### Cashflows

At maturity $T$:

$$\text{Inflation leg} = N \cdot \left(\frac{I(T)}{I(0)} - 1\right)$$

$$\text{Fixed leg} = N \cdot \left((1 + K)^T - 1\right)$$

where:

- $I(T)$ — CPI index level at maturity (subject to publication lag)
- $I(0)$ — base CPI index level at inception
- $K$ — fixed (break-even) inflation rate
- $N$ — notional principal
- $T$ — swap tenor in years

The **break-even inflation rate** is the value of $K$ that makes the swap NPV
zero at inception. It represents the market's expectation of average annual
inflation over the swap tenor (plus an inflation risk premium).

### Pricing

ZCIS is priced from a **real rate curve** or, equivalently, an inflation forward
curve. Given a nominal discount factor $\mathrm{DF}_n(T)$ and a real discount
factor $\mathrm{DF}_r(T)$:

$$\frac{I(T)}{I(0)} = \frac{\mathrm{DF}_r(T)}{\mathrm{DF}_n(T)}$$

The NPV for the inflation receiver is:

$$\text{NPV} = \mathrm{DF}_n(T) \cdot N \cdot \left(\frac{\mathrm{DF}_r(T)}{\mathrm{DF}_n(T)} - (1 + K)^T\right)$$

### Code Example

```python
from valax.instruments import ZeroCouponInflationSwap
from valax.dates import ymd_to_ordinal
import jax.numpy as jnp

zcis = ZeroCouponInflationSwap(
    effective_date=ymd_to_ordinal(2025, 1, 15),
    maturity_date=ymd_to_ordinal(2030, 1, 15),
    fixed_rate=jnp.array(0.025),   # 2.5% break-even inflation
    notional=jnp.array(50_000_000.0),
    base_cpi=jnp.array(311.2),     # CPI-U level at inception
    is_inflation_receiver=True,
    index_lag=3,                    # 3-month publication lag
    day_count="act_act",
)
```

!!! note
    The `index_lag` field controls which CPI publication maps to a given swap date.
    With a 3-month lag, the CPI used for a January payment date is the October
    publication (interpolated between September and October monthly values).

---

## Year-on-Year Inflation Swap (YYIS)

**Market context.** The YYIS exchanges **annual** inflation returns periodically,
rather than a single cumulative return at maturity. It is the inflation analogue
of a standard fixed-for-floating interest rate swap and is commonly used for
hedging inflation-linked coupon obligations.

### Cashflows

At each payment date $t_i$:

$$C_i^{\text{infl}} = N \cdot \left(\frac{I(t_i)}{I(t_{i-1})} - 1\right)$$

$$C_i^{\text{fixed}} = N \cdot K \cdot \tau_i$$

where $\tau_i$ is the day count fraction for the period $[t_{i-1}, t_i]$.

### Convexity Adjustment

A critical subtlety: the expectation of a **ratio** of CPI indices is not equal to
the ratio of expectations:

$$E\!\left[\frac{I(t_i)}{I(t_{i-1})}\right] \neq \frac{E[I(t_i)]}{E[I(t_{i-1})]}$$

This **convexity adjustment** arises because the YoY payment depends on the CPI
ratio observed at $t_i$ but discounted from $t_i$ to today. The adjustment depends
on the volatility and correlation structure of inflation rates and nominal rates.
For typical market parameters, the adjustment is on the order of 1–5 bps per annum,
but it can be significant for long-dated swaps. See
[theory §3.6](../theory.md#year-on-year-inflation-swaps-yyis) for the full
Jarrow-Yildirim treatment. VALAX's current `yyis_price` uses the forward ratio
directly (the standard baseline); a convexity-adjusted variant is on the roadmap.

### Code Example

```python
from valax.instruments import YearOnYearInflationSwap
from valax.dates import ymd_to_ordinal, generate_schedule
import jax.numpy as jnp

yyis = YearOnYearInflationSwap(
    effective_date=ymd_to_ordinal(2025, 1, 15),
    payment_dates=generate_schedule(2026, 1, 15, 2030, 1, 15, frequency=1),
    fixed_rate=jnp.array(0.024),    # 2.4% fixed rate
    notional=jnp.array(25_000_000.0),
    base_cpi=jnp.array(311.2),
    is_inflation_receiver=True,
    index_lag=3,
    day_count="act_act",
)
```

---

## Inflation Cap/Floor

**Market context.** Inflation caps and floors are option overlays on year-on-year
inflation rates. An inflation cap protects against unexpectedly high inflation; an
inflation floor protects against deflation. Together, they form the basis for
inflation collars (long cap + short floor) used extensively by pension funds and
inflation-linked bond issuers.

### Pricing

Each **caplet** (or floorlet) in the strip is priced via Black-76 on the forward
year-on-year inflation rate:

$$\text{Caplet}_i = \mathrm{DF}(t_i) \cdot N \cdot \tau_i \cdot \text{Black76}(F_i, K, \sigma_i, \tau_i)$$

where:

- $F_i$ — forward year-on-year inflation rate for period $[t_{i-1}, t_i]$
- $K$ — strike inflation rate
- $\sigma_i$ — implied normal or lognormal volatility of the YoY rate
- $\tau_i$ — day count fraction

The total cap (or floor) value is the sum of the individual caplet (floorlet) values.

### Code Example

```python
from valax.instruments import InflationCapFloor
from valax.dates import ymd_to_ordinal, generate_schedule
import jax.numpy as jnp

# 5-year inflation cap struck at 3%
inflation_cap = InflationCapFloor(
    effective_date=ymd_to_ordinal(2025, 1, 15),
    payment_dates=generate_schedule(2026, 1, 15, 2030, 1, 15, frequency=1),
    strike=jnp.array(0.03),         # 3% strike
    notional=jnp.array(100_000_000.0),
    base_cpi=jnp.array(311.2),
    is_cap=True,
    index_lag=3,
    day_count="act_act",
)
```

!!! tip
    Use inflation put-call parity to convert between caps and floors:
    $\text{Cap} - \text{Floor} = \text{YYIS}$ (at the same strike). This is useful
    for calibration and relative-value checks.
