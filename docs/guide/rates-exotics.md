# Interest Rate Exotics

Interest rate derivatives are the largest derivatives market by notional. VALAX's
core IR instruments (caps, floors, swaptions, vanilla IRS) are documented in the
[Fixed Income](fixed-income.md) guide and the [Analytical Pricing](analytical.md)
guide. This page covers the extended set of rate derivatives implemented in
`valax/pricing/analytic/rates_exotics.py`: OIS, cross-currency, total return,
CMS swaps, CMS caps/floors, and range accruals.

!!! success "All instruments on this page are implemented."
    See `valax/instruments/rates.py` and `valax/pricing/analytic/rates_exotics.py`.

## OIS Swap

**Market context.** The Overnight Index Swap (OIS) is the dominant swap type in
the post-LIBOR world. The floating leg references the **compounded daily overnight
rate** — SOFR (USD), €STR (EUR), SONIA (GBP), or TONAR (JPY) — rather than a
term IBOR rate. OIS swaps are the primary instrument for constructing the risk-free
discount curve.

### Floating Leg Valuation

The floating leg value for an accrual period $[t_{j_0}, t_{j_D}]$ with $D$ business
days is:

$$V_{\text{float}} = N \cdot \left(\prod_{j=1}^{D} \bigl(1 + r_j \cdot \delta_j\bigr) - 1\right)$$

where:

- $r_j$ — overnight rate on business day $j$
- $\delta_j$ — day count fraction for one business day (typically $1/360$ or $1/365$)
- $N$ — notional principal

For past dates, $r_j$ is the realized overnight fixing. For future dates, $r_j$ is
the forward overnight rate implied by the OIS curve. VALAX's pricer exploits the
**telescoping identity** $\prod_j (1 + r_j \delta_j) = DF(t_{j_0})/DF(t_{j_D})$
to evaluate the leg in closed form (single-curve).

### Contrast with IBOR Swaps

| Feature | OIS Swap | IBOR Swap (legacy) |
|---------|----------|--------------------|
| Floating rate | Compounded daily overnight | Term rate (3M/6M LIBOR) |
| Fixing | Daily (compounded in arrears) | Once per period (in advance) |
| Credit risk | Near risk-free (overnight lending) | Includes bank credit premium |
| Curve | Single-curve discounting | Dual-curve (OIS discount + IBOR projection) |

!!! note
    Since the LIBOR transition (completed for most currencies by mid-2024), OIS
    swaps have replaced IBOR swaps as the benchmark. VALAX's curve bootstrapper
    uses OIS instruments as the foundation for the discount curve — see
    [Fixed Income → Multi-curve bootstrapping](fixed-income.md).

### Code Example

```python
from valax.instruments import OISSwap
from valax.dates import ymd_to_ordinal, generate_schedule
import jax.numpy as jnp

# 5-year SOFR OIS swap, quarterly fixed payments, quarterly float
fixed_dates = generate_schedule(2025, 6, 15, 2030, 3, 15, frequency=4)
float_dates = generate_schedule(2025, 6, 15, 2030, 3, 15, frequency=4)

ois = OISSwap(
    start_date=ymd_to_ordinal(2025, 3, 15),
    fixed_dates=fixed_dates,
    float_dates=float_dates,
    fixed_rate=jnp.array(0.038),    # 3.8% fixed rate
    notional=jnp.array(50_000_000.0),
    pay_fixed=True,
    compounding="compounded",
    day_count="act_360",
)
```

---

## Cross-Currency Swap

**Market context.** Cross-currency swaps (XCCY) exchange floating-rate cashflows
in two different currencies, with notional exchanges at inception and maturity. They
are essential for hedging foreign-currency debt issuance and for funding in a
non-domestic currency. The **basis spread** (often negative for EUR/USD) reflects
structural supply/demand imbalances for dollar funding.

### Cashflow Structure

A cross-currency swap has three components:

| Component | Domestic Leg | Foreign Leg |
|-----------|-------------|-------------|
| **Initial exchange** ($t_0$) | Pay $N_d$ | Receive $N_f$ |
| **Periodic coupons** | $N_d \cdot (r_d^{\text{float}} + s) \cdot \tau_i$ | $N_f \cdot r_f^{\text{float}} \cdot \tau_i$ |
| **Final re-exchange** ($T$) | Receive $N_d$ | Pay $N_f$ |

where $s$ is the **basis spread** — the spread added to the domestic floating leg
that makes the swap NPV zero at inception:

$$\text{Basis spread} = s \;\text{ such that }\; PV_{\text{dom}}(\text{dom leg} + s) = PV_{\text{dom}}(\text{for leg})$$

The basis spread is a key observable for cross-currency curve construction. VALAX's
pricer uses a **two-curve telescoping identity** on each leg plus the spot FX
conversion, and provides a par basis solver via `optimistix.root_find`.

### Code Example

```python
from valax.instruments import CrossCurrencySwap
from valax.dates import ymd_to_ordinal, generate_schedule
import jax.numpy as jnp

# 5-year EUR/USD cross-currency basis swap
payment_dates = generate_schedule(2025, 9, 15, 2030, 3, 15, frequency=4)

xccy = CrossCurrencySwap(
    start_date=ymd_to_ordinal(2025, 3, 15),
    payment_dates=payment_dates,
    maturity_date=ymd_to_ordinal(2030, 3, 15),
    domestic_notional=jnp.array(50_000_000.0),   # 50M USD
    foreign_notional=jnp.array(46_000_000.0),    # ~46M EUR at 1.087
    basis_spread=jnp.array(-0.0015),             # -15 bps basis
    exchange_notional=True,
    currency_pair="EUR/USD",
    day_count="act_360",
)
```

!!! tip
    The notional exchange feature means cross-currency swaps have **significant
    FX exposure** at maturity, unlike single-currency swaps. The MTM of the
    notional re-exchange depends on how much the FX rate has moved since inception.

---

## Total Return Swap (TRS)

**Market context.** A Total Return Swap transfers the **total economic exposure**
of a reference asset (price change + income) from one party to another in exchange
for a funding rate. TRS are widely used for synthetic exposure (e.g., gaining equity
index exposure without buying shares), prime brokerage financing, and balance sheet
management.

### Cashflow Legs

| Leg | Cashflow | Timing |
|-----|----------|--------|
| **Total return leg** | $N \cdot \frac{P_{\text{end}} - P_{\text{start}}}{P_{\text{start}}} + \text{income}$ | Each reset date |
| **Funding leg** | $N \cdot (r_{\text{float}} + s) \cdot \tau_i$ | Each reset date |

where:

- $P_{\text{start}}, P_{\text{end}}$ — reference asset price at start and end of each period
- income = coupons (bonds) or dividends (equities) during the period
- $r_{\text{float}}$ — floating reference rate (e.g., SOFR)
- $s$ — funding spread
- $\tau_i$ — day count fraction

The **total return receiver** has economic exposure equivalent to owning the
reference asset funded at $r_{\text{float}} + s$. VALAX's pricer uses the
**self-financing reduction**: the total return leg's expected PV collapses to an
annuity plus the unrealized return (at trade inception, assuming martingale total
returns under the funding measure), leaving the fair funding spread as a linear
problem.

### Code Example

```python
from valax.instruments import TotalReturnSwap
from valax.dates import ymd_to_ordinal, generate_schedule
import jax.numpy as jnp

# 1-year TRS on investment-grade bond index, quarterly resets
trs = TotalReturnSwap(
    start_date=ymd_to_ordinal(2025, 3, 15),
    payment_dates=generate_schedule(2025, 6, 15, 2026, 3, 15, frequency=4),
    notional=jnp.array(100_000_000.0),
    funding_spread=jnp.array(0.004),    # SOFR + 40 bps funding
    is_total_return_receiver=True,
    day_count="act_360",
)
```

---

## CMS Swap

**Market context.** A Constant Maturity Swap (CMS) has a floating leg that pays
the **par swap rate** for a given tenor (e.g., the 10-year swap rate) observed at
each fixing date. CMS swaps allow investors to express views on the shape and
level of the swap curve without entering into a series of forward-starting swaps.

### CMS Rate and Convexity Adjustment

The CMS rate $S(T_f)$ is the par swap rate for a specific tenor (e.g., 10Y)
observed at the fixing date $T_f$. Under the payment measure $T_p$, the expected
CMS rate differs from the forward swap rate $S_0$ by a **convexity adjustment**:

$$E^{T_p}[S(T_f)] = S_0 + \text{convexity adjustment}$$

The convexity adjustment arises because:

1. The swap rate is a **nonlinear function** of discount factors.
2. The payment date $T_p$ generally differs from the natural measure of the swap
   rate.

### Replication Method (Hagan) — Roadmap

The standard production approach is **static replication** (Hagan, 2003): express
the CMS payoff as an integral over European swaptions, then use the SABR smile to
evaluate the integral:

$$E^{T_p}[S(T_f)] = S_0 + \int_0^{\infty} w(K) \cdot \text{SwaptionVol}(K)\, dK$$

where $w(K)$ is a weight function derived from the annuity mapping. This approach
naturally captures the smile dependence of the convexity adjustment.

!!! note "Current implementation"
    VALAX's `cms_swap_price` uses the **forward par swap rate with no convexity
    adjustment** — the baseline. Integrating the Hagan replication formula against
    a SABR swaption surface is a roadmap item (it falls out cleanly from the
    existing SABR surface implementation).

### Code Example

```python
from valax.instruments import CMSSwap
from valax.dates import ymd_to_ordinal, generate_schedule
import jax.numpy as jnp

# Pay fixed, receive 10Y CMS rate quarterly for 5 years
cms_swap = CMSSwap(
    start_date=ymd_to_ordinal(2025, 3, 15),
    payment_dates=generate_schedule(2025, 6, 15, 2030, 3, 15, frequency=4),
    fixed_rate=jnp.array(0.042),     # 4.2% fixed leg
    notional=jnp.array(25_000_000.0),
    cms_tenor=10,                    # reference: 10-year swap rate
    pay_fixed=True,
    day_count="act_360",
)
```

---

## CMS Cap/Floor

**Market context.** CMS caps and floors are strips of caplets/floorlets on a CMS
rate. They provide protection against rising (cap) or falling (floor) long-term
interest rates. CMS cap/floor pricing inherits the same convexity adjustment
issues as CMS swaps.

### Pricing

Each CMS caplet pays:

$$\text{CMS caplet}_i = \mathrm{DF}(t_i) \cdot N \cdot \tau_i \cdot E^{T_i}\!\left[\max(S(T_f) - K,\; 0)\right]$$

VALAX currently prices this via **Black-76 on the forward CMS rate** (no
convexity adjustment) — analogous to the CMS swap treatment. The swaption
replication refinement is a roadmap item.

### Code Example

```python
from valax.instruments import CMSCapFloor
from valax.dates import generate_schedule
import jax.numpy as jnp

# CMS floor struck at 3% on the 10Y rate, 5-year tenor
cms_floor = CMSCapFloor(
    payment_dates=generate_schedule(2025, 6, 15, 2030, 3, 15, frequency=4),
    strike=jnp.array(0.03),         # 3% floor strike
    notional=jnp.array(50_000_000.0),
    cms_tenor=10,
    is_cap=False,                    # floor
    day_count="act_360",
)
```

---

## Range Accrual

**Market context.** Range accrual notes pay a coupon proportional to the number of
days a reference index stays within a specified range. They are popular in low-
volatility environments when investors seek yield enhancement by selling optionality.
Reference indices include overnight rates, CMS rates, FX rates, or equity indices.

### Coupon Structure

The coupon for period $i$ is:

$$C_i = N \cdot R \cdot \tau_i \cdot \frac{n_{\text{in range}}}{n_{\text{total}}}$$

where:

- $R$ — maximum coupon rate (earned if the index is always in range)
- $\tau_i$ — day count fraction for the period
- $n_{\text{in range}}$ — number of business days the index is within
  $[\text{lower\_barrier}, \text{upper\_barrier}]$
- $n_{\text{total}}$ — total business days in the period

### Pricing Decomposition

Each business day in the accrual period contributes a **digital option** to the
total coupon. The range accrual value can be decomposed as:

$$V = \sum_{i=1}^{n} \sum_{j=1}^{D_i} \frac{R \cdot \delta_j \cdot N}{D_i} \cdot \mathrm{DF}(t_j) \cdot P\!\bigl(L \leq X(t_j) \leq U\bigr)$$

where $X(t_j)$ is the reference index level on day $j$, $L$ and $U$ are the lower
and upper barriers, and $\delta_j$ is the daily accrual fraction. Each probability
term is the value of a digital range option (long digital at $L$, short digital
at $U$).

VALAX's implementation takes a single **snapshot probability** per period
(evaluated at the accrual midpoint) rather than summing across all daily
observations, which is the standard fast pricing method. Full daily integration
would be a refinement.

### Code Example

```python
from valax.instruments import RangeAccrual
from valax.dates import generate_schedule
import jax.numpy as jnp

# 2-year range accrual on SOFR, pays 5% if SOFR stays between 3% and 5%
range_note = RangeAccrual(
    payment_dates=generate_schedule(2025, 6, 15, 2027, 3, 15, frequency=4),
    coupon_rate=jnp.array(0.05),         # 5% max coupon
    lower_barrier=jnp.array(0.03),       # 3% lower bound
    upper_barrier=jnp.array(0.05),       # 5% upper bound
    notional=jnp.array(10_000_000.0),
    reference_index="rate",              # SOFR-linked
    day_count="act_360",
)
```

!!! warning
    Range accruals have **significant gamma exposure** near the barriers. As the
    reference index approaches a barrier, the daily digitals become highly sensitive
    to small moves. This can create large hedging costs in volatile markets.
