# Instruments Guide

VALAX now covers **35+ instruments** across **6 asset classes**: equity options and
exotics, fixed income, interest rate derivatives, FX derivatives, credit derivatives,
and inflation derivatives. Every instrument is an `equinox.Module` subclass — a
frozen dataclass that is automatically a valid JAX pytree. Instruments carry **no
pricing logic**; they are pure data containers describing the contractual terms of a
trade. Pricing is performed by separate pure functions (see
[Analytical Pricing](analytical.md), [Monte Carlo](monte-carlo.md),
[PDE Methods](pde.md), and [Lattice Methods](lattice.md)).

This page documents the instruments added beyond the original vanilla set
(European/American options, fixed-rate bonds, caps, swaps, swaptions). For each
instrument you will find: market context, cashflow mathematics, pricing approach,
a complete code example, and any key conventions.

!!! tip
    All instruments are immutable pytrees. To modify a field, use `equinox.tree_at`
    or create a new instance. This guarantees safe `jax.jit`, `jax.vmap`, and
    `jax.grad` usage.

---

## Credit Derivatives

Credit derivatives transfer **credit risk** — the risk that a borrower defaults —
between counterparties without transferring the underlying debt instrument.
The fundamental building block is the **survival curve**: the probability
$P(\tau > t)$ that the reference entity has not defaulted by time $t$. The curve
is characterized by a **hazard rate** $h(t)$ such that

$$P(\tau > t) = \exp\!\left(-\int_0^t h(s)\,ds\right)$$

Survival curves are bootstrapped from observed CDS spreads. The **recovery rate**
$R$ (typically 40 % for senior unsecured corporates) determines the loss-given-default.

### Credit Default Swap (CDS)

**Market context.** The Credit Default Swap is the most liquid credit derivative,
with roughly \$3.8 trillion notional outstanding globally. A CDS is economically
equivalent to credit insurance: the **protection buyer** pays a periodic premium
(the *spread*) in exchange for a contingent payment if the reference entity defaults.
CDS spreads are the primary observable for single-name credit risk and feed into
index products (CDX, iTraxx), structured credit, and CVA calculations.

#### Cashflows

A CDS has two legs:

| Leg | Cashflow | Timing |
|-----|----------|--------|
| **Protection leg** | $N \cdot (1 - R)$ | At default (if $\tau < T$) |
| **Premium leg** | $s \cdot N \cdot \tau_i \cdot P(\tau > t_i)$ per period | Quarterly to maturity |
| **Accrued premium** | Fractional spread from last coupon date to default date | At default |

where $N$ is the notional, $R$ the recovery rate, $s$ the annual spread, and
$\tau_i$ the day count fraction for period $i$.

#### Fair Spread

The **par CDS spread** is the value of $s$ that makes the NPV zero at inception.
Setting the present value of the protection leg equal to the premium leg:

$$s = \frac{\displaystyle\sum_{i=1}^{n} \mathrm{DF}(t_i) \cdot q(t_i) \cdot \Delta t_i \cdot (1 - R)}{\displaystyle\sum_{i=1}^{n} \mathrm{DF}(t_i) \cdot P(\tau > t_i) \cdot \Delta\tau_i}$$

where:

- $\mathrm{DF}(t_i)$ — risk-free discount factor to time $t_i$
- $q(t_i)$ — marginal default probability density in period $i$, i.e.,
  $q(t_i) \approx P(\tau > t_{i-1}) - P(\tau > t_i)$
- $P(\tau > t_i)$ — survival probability to $t_i$
- $R$ — recovery rate
- $\Delta\tau_i$ — day count fraction for premium accrual in period $i$

#### Pricing Approaches

1. **Deterministic rates + bootstrapped survival curve.** The standard market
   approach. Hazard rates are piece-wise constant, bootstrapped from quoted CDS
   spreads. Interest rate and credit risk are assumed independent.
2. **Reduced-form model with stochastic hazard rates.** Required for wrong-way
   risk, CVA, and portfolio credit models where default correlation matters.

See [Analytical Pricing](analytical.md) for the deterministic-rates implementation
and [Monte Carlo](monte-carlo.md) for stochastic hazard rate simulation.

#### Code Example

```python
from valax.instruments import CDS
from valax.dates import ymd_to_ordinal, generate_schedule
import jax.numpy as jnp

cds = CDS(
    effective_date=ymd_to_ordinal(2025, 3, 20),
    maturity_date=ymd_to_ordinal(2030, 3, 20),
    premium_dates=generate_schedule(2025, 6, 20, 2030, 3, 20, frequency=4),
    spread=jnp.array(0.01),       # 100 bps annual spread
    notional=jnp.array(10_000_000.0),
    recovery_rate=jnp.array(0.4), # 40% — market standard for senior unsecured
    is_protection_buyer=True,
    day_count="act_360",
)
```

#### Key Conventions

| Convention | Detail |
|------------|--------|
| Standard maturities | Mar 20, Jun 20, Sep 20, Dec 20 (IMM dates) |
| Premium frequency | Quarterly (4× per year) |
| Day count | ACT/360 |
| Recovery rate | 40 % (senior unsecured), 25 % (subordinated), 35 % (senior secured) |
| Quoting | Index CDS: upfront points + 100 bps or 500 bps running. Single-name: running spread |
| Business day | Modified Following |

!!! note
    The ISDA standard model uses piece-wise flat hazard rates and deterministic
    discounting. VALAX's CDS pricer is consistent with the ISDA CDS Standard Model
    (also known as the ISDA Calculator).

---

### CDO Tranche

**Market context.** A Collateralized Debt Obligation (CDO) pools credit risk from
a portfolio of reference entities (typically 125 names in standard index tranches)
and distributes losses to **tranches** ordered by seniority. Each tranche absorbs
portfolio losses within an **attachment** $a$ and **detachment** $d$ range. The
equity tranche (e.g., 0–3 %) takes the first losses and receives the highest
spread; the super-senior tranche (e.g., 15–100 %) is exposed only to catastrophic
losses.

#### Tranche Loss

Given a portfolio loss fraction $L$, the loss absorbed by the tranche $[a, d]$ is:

$$\text{Tranche loss}(L) = \min\!\bigl(\max(L - a,\; 0),\; d - a\bigr)$$

The tranche notional is $N \cdot (d - a)$ where $N$ is the total portfolio notional.
The tranche coupon is paid on the *outstanding* tranche notional (reduced by losses).

#### Gaussian Copula Pricing

The standard market model is the **one-factor Gaussian copula**. The joint default
probability of $n$ names is:

$$P(\text{joint default}) = \Phi_n\!\bigl(\Phi^{-1}(q_1),\, \ldots,\, \Phi^{-1}(q_n);\; \Sigma\bigr)$$

where $q_i$ is the marginal default probability of name $i$, $\Phi$ the standard
normal CDF, and $\Sigma$ the correlation matrix. Under the one-factor assumption
with uniform pairwise correlation $\rho$:

$$P(\text{default}_i \mid M) = \Phi\!\left(\frac{\Phi^{-1}(q_i) - \sqrt{\rho}\, M}{\sqrt{1 - \rho}}\right)$$

where $M \sim \mathcal{N}(0,1)$ is the common factor. The conditional independence
simplifies the portfolio loss distribution to a mixture of independent Bernoulli trials.

The market convention is to quote **base correlation**: for each detachment point,
the implied correlation that reprices the $[0, d]$ tranche. Base correlation is
monotonically increasing in $d$ (unlike compound correlation, which can be
non-monotonic).

#### Code Example

```python
from valax.instruments import CDOTranche
from valax.dates import ymd_to_ordinal, generate_schedule
import jax.numpy as jnp

# 3%-7% mezzanine tranche on a 125-name portfolio
tranche = CDOTranche(
    effective_date=ymd_to_ordinal(2025, 3, 20),
    maturity_date=ymd_to_ordinal(2030, 3, 20),
    premium_dates=generate_schedule(2025, 6, 20, 2030, 3, 20, frequency=4),
    attachment=jnp.array(0.03),     # 3%
    detachment=jnp.array(0.07),     # 7%
    spread=jnp.array(0.05),         # 500 bps running
    notional=jnp.array(500_000_000.0),
    recovery_rate=jnp.array(0.4),
    n_names=125,
    day_count="act_360",
)
```

!!! warning
    The Gaussian copula model is known for its limitations (static correlation,
    inability to capture correlation skew). It remains the market standard for
    quoting and relative value, but should not be used for absolute risk measurement
    without supplementary stress testing.

---

## Fixed Income

Fixed income instruments generate predictable streams of cashflows (coupons and
principal). VALAX's base fixed-rate bond and zero-coupon bond are documented in the
[Fixed Income](fixed-income.md) guide. This section covers the extended set:
floating-rate bonds, callable bonds, puttable bonds, and convertible bonds.

### Floating Rate Bond (FRN)

**Market context.** Floating-rate notes (FRNs) pay coupons that reset periodically
based on a **reference rate** — SOFR (USD), EURIBOR (EUR), SONIA (GBP), or TONAR
(JPY). FRNs represent a major segment of the bond market, particularly in
leveraged finance (floating-rate bank loans) and government-sponsored issuance
(US Treasury FRNs).

#### Cashflow Mathematics

The price of an FRN with $n$ remaining coupon periods is:

$$P = \sum_{i=1}^{n} (F_i + s) \cdot \tau_i \cdot \mathrm{DF}(t_i) \cdot N \;+\; \mathrm{DF}(t_n) \cdot N$$

where:

- $F_i$ — forward reference rate for period $[t_{i-1}, t_i]$, projected from the discount curve
- $s$ — fixed spread over the reference rate
- $\tau_i$ — day count fraction for period $i$ (typically ACT/360)
- $\mathrm{DF}(t_i)$ — discount factor to $t_i$
- $N$ — face value / notional

For **seasoned** FRNs, the current period's rate is already fixed. Past fixings are
stored in the `fixing_rates` field; future fixings are projected from the forward curve.

#### Duration Properties

An FRN has two distinct duration measures:

| Measure | Value | Intuition |
|---------|-------|-----------|
| **Interest rate duration** | Very low (≈ time to next reset) | At each reset date, the coupon adjusts to the current market rate, pulling the price back toward par |
| **Spread duration** | ≈ Macaulay duration of equivalent fixed bond | The spread $s$ is fixed for the life of the bond; a widening in spread lowers the price |

!!! note
    At a reset date (assuming no credit deterioration), an FRN prices at exactly
    par plus accrued interest. Between resets, it trades close to par with
    sensitivity only to the remaining stub period. The deviation from par is driven
    entirely by the spread component and any change in credit quality.

#### Code Example

```python
from valax.instruments import FloatingRateBond
from valax.dates import ymd_to_ordinal, generate_schedule
import jax.numpy as jnp

# 3-year SOFR + 50 bps FRN, quarterly resets
payment_dates = generate_schedule(2025, 6, 15, 2028, 3, 15, frequency=4)
fixing_dates = generate_schedule(2025, 3, 15, 2027, 12, 15, frequency=4)

# First two fixings are known (seasoned bond)
n = len(payment_dates)
fixing_rates = jnp.full(n, jnp.nan)
fixing_rates = fixing_rates.at[0].set(0.0435)  # 4.35% SOFR
fixing_rates = fixing_rates.at[1].set(0.0428)  # 4.28% SOFR

frn = FloatingRateBond(
    payment_dates=payment_dates,
    fixing_dates=fixing_dates,
    settlement_date=ymd_to_ordinal(2025, 9, 15),
    spread=jnp.array(0.005),       # 50 bps over SOFR
    face_value=jnp.array(1_000.0),
    fixing_rates=fixing_rates,
    frequency=4,
    day_count="act_360",
)
```

---

### Callable Bond

**Market context.** The majority of US corporate bonds are callable — the issuer
retains the right to redeem the bond early, typically after an initial non-call
period. Issuers exercise this right when interest rates fall (they can refinance
at a lower coupon). From the investor's perspective, the callable bond offers a
higher yield than an equivalent non-callable bond to compensate for the embedded
short call position.

#### Pricing

Callable bonds require a model that handles the embedded optionality. The standard
approach is **backward induction** on a short-rate tree (Hull-White trinomial) or
PDE:

1. Build a recombining tree calibrated to the yield curve (and optionally
   swaption volatilities).
2. At each node and each call date, the issuer decides whether to call:
   $V_{\text{node}} = \min(V_{\text{continuation}}, \text{call price})$.
3. The resulting tree price includes the impact of the embedded option.

The **option-adjusted spread (OAS)** is the constant spread added to the discount
curve at every node that equates the model price to the observed market price:

$$P_{\text{market}} = \text{TreePrice}\!\bigl(\text{curve} + \text{OAS},\; \text{call schedule}\bigr)$$

OAS isolates the pure credit/liquidity component of the bond's yield premium,
stripping out the optionality.

**Effective duration** is computed via autodiff — differentiate the OAS-based
model price with respect to a parallel shift of the yield curve. Because VALAX
pricing functions are pure JAX functions, this is a single `jax.grad` call with
no finite-difference approximation:

$$D_{\text{eff}} = -\frac{1}{P}\,\frac{\partial P}{\partial \Delta r}$$

See [Greeks & Risk Sensitivities](greeks.md) for details on autodiff-based duration.

#### Code Example

```python
from valax.instruments import CallableBond
from valax.dates import ymd_to_ordinal, generate_schedule
import jax.numpy as jnp

# 10-year callable corporate bond, callable after year 3
payment_dates = generate_schedule(2025, 9, 15, 2035, 3, 15, frequency=2)
call_dates = jnp.array([
    ymd_to_ordinal(2028, 3, 15),
    ymd_to_ordinal(2029, 3, 15),
    ymd_to_ordinal(2030, 3, 15),
    ymd_to_ordinal(2031, 3, 15),
    ymd_to_ordinal(2032, 3, 15),
])

callable = CallableBond(
    payment_dates=payment_dates,
    settlement_date=ymd_to_ordinal(2025, 3, 15),
    coupon_rate=jnp.array(0.055),  # 5.5% coupon
    face_value=jnp.array(100.0),
    call_dates=call_dates,
    call_prices=jnp.array([102.0, 101.5, 101.0, 100.5, 100.0]),
    frequency=2,
    day_count="act_365",
)
```

| Parameter | Description |
|-----------|-------------|
| `payment_dates` | Semi-annual coupon dates (ordinals) |
| `call_dates` | Dates on which the issuer may call (subset of payment dates) |
| `call_prices` | Clean call price at each call date (often declining schedule toward par) |
| `coupon_rate` | Annual coupon rate |

!!! tip
    Use `jax.grad` through the Hull-White tree pricer to compute effective duration
    and effective convexity — no finite-difference bumps required. See
    [Lattice Methods](lattice.md) for the tree implementation.

---

### Puttable Bond

**Market context.** A puttable bond is the mirror image of a callable bond — the
**bondholder** has the right to sell the bond back to the issuer at a predetermined
price on specified put dates. This protects the investor against rising interest
rates or credit deterioration. Puttable bonds are less common than callable bonds
but appear in corporate and municipal markets.

#### Valuation

The value of a puttable bond decomposes as:

$$V_{\text{puttable}} = V_{\text{straight}} + V_{\text{put option}}$$

The embedded put **increases** the bond value relative to an otherwise identical
straight bond. The put option value is highest when rates are high (the holder
would exercise to reinvest at higher yields) or when credit quality has deteriorated.

Pricing mechanics are analogous to callable bonds (Hull-White tree or PDE with
backward induction), but the exercise decision now belongs to the **bondholder**:
at each put date, $V_{\text{node}} = \max(V_{\text{continuation}}, \text{put price})$.

#### Code Example

```python
from valax.instruments import PuttableBond
from valax.dates import ymd_to_ordinal, generate_schedule
import jax.numpy as jnp

payment_dates = generate_schedule(2025, 9, 15, 2032, 3, 15, frequency=2)

puttable = PuttableBond(
    payment_dates=payment_dates,
    settlement_date=ymd_to_ordinal(2025, 3, 15),
    coupon_rate=jnp.array(0.04),  # 4.0% coupon
    face_value=jnp.array(100.0),
    put_dates=jnp.array([
        ymd_to_ordinal(2028, 3, 15),
        ymd_to_ordinal(2030, 3, 15),
    ]),
    put_prices=jnp.array([100.0, 100.0]),  # puttable at par
    frequency=2,
    day_count="act_365",
)
```

!!! note
    Some bonds have both call and put provisions. VALAX models these as separate
    instruments; for dual-option bonds, combine the logic in a custom pricing
    function that checks both exercise boundaries at each node.

---

### Convertible Bond

**Market context.** A convertible bond is an equity-credit hybrid — a fixed-rate
bond with an embedded equity conversion option. The bondholder may convert the bond
into a fixed number of the issuer's shares. Convertibles appeal to investors
seeking bond-like downside protection with equity upside participation. They are
widely issued by growth companies (lower coupon than straight debt) and actively
traded by convertible arbitrage hedge funds.

#### Conversion Value

The conversion value at any point is:

$$\text{Conversion value} = \text{conversion\_ratio} \times S$$

where $S$ is the current stock price. The bondholder converts when the conversion
value exceeds the bond's straight debt value (plus any accrued coupon).

The **conversion premium** is the excess of the bond price over conversion value,
expressed as a percentage:

$$\text{Conversion premium} = \frac{P_{\text{bond}} - \text{conversion value}}{\text{conversion value}}$$

#### Pricing

Convertible bond pricing requires a model that captures **equity risk**, **credit
risk**, and **interest rate risk** simultaneously. The standard approach is a
**1D PDE** in stock price with a credit-adjusted boundary:

1. The stock price follows GBM with a default intensity $h$: at default, the stock
   drops to zero and the bond recovers $R \cdot \text{face value}$.
2. The PDE is solved backward in time with boundary conditions for conversion
   (max of continuation and conversion value) and optional issuer call.
3. Credit spread enters through the discounting: the risky discount rate is
   $r + h$ where $h$ is the hazard rate.

#### Code Example

```python
from valax.instruments import ConvertibleBond
from valax.dates import ymd_to_ordinal, generate_schedule
import jax.numpy as jnp

payment_dates = generate_schedule(2025, 9, 15, 2030, 3, 15, frequency=2)

convertible = ConvertibleBond(
    payment_dates=payment_dates,
    settlement_date=ymd_to_ordinal(2025, 3, 15),
    coupon_rate=jnp.array(0.015),        # 1.5% coupon (low — equity optionality)
    face_value=jnp.array(1_000.0),
    conversion_ratio=jnp.array(12.5),    # 12.5 shares per bond
    call_dates=jnp.array([
        ymd_to_ordinal(2027, 3, 15),
        ymd_to_ordinal(2028, 3, 15),
        ymd_to_ordinal(2029, 3, 15),
    ]),
    call_prices=jnp.array([1030.0, 1020.0, 1010.0]),
    frequency=2,
    day_count="act_365",
)
```

| Parameter | Description |
|-----------|-------------|
| `conversion_ratio` | Number of shares received per unit of face value upon conversion |
| `call_dates` | Issuer call dates — allows forced conversion if stock price is high |
| `call_prices` | Clean call prices (often declining toward par) |

!!! tip
    The Tsiveriotis-Fernandes decomposition splits the convertible into a cash-only
    component (discounted at the risky rate $r + h$) and an equity component
    (discounted at the risk-free rate $r$). This is the standard implementation in
    VALAX's PDE pricer. See [PDE Methods](pde.md).

---

## Inflation Derivatives

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
building blocks for all inflation products.

### Zero-Coupon Inflation Swap (ZCIS)

**Market context.** The ZCIS is the most liquid inflation derivative and the
primary instrument for bootstrapping an inflation forward curve. A single exchange
occurs at maturity: one party pays (or receives) the cumulative inflation return,
while the other pays a compounded fixed rate.

#### Cashflows

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

#### Pricing

ZCIS is priced from a **real rate curve** or, equivalently, an inflation forward
curve. Given a nominal discount factor $\mathrm{DF}_n(T)$ and a real discount
factor $\mathrm{DF}_r(T)$:

$$\frac{I(T)}{I(0)} = \frac{\mathrm{DF}_r(T)}{\mathrm{DF}_n(T)}$$

The NPV for the inflation receiver is:

$$\text{NPV} = \mathrm{DF}_n(T) \cdot N \cdot \left(\frac{\mathrm{DF}_r(T)}{\mathrm{DF}_n(T)} - (1 + K)^T\right)$$

#### Code Example

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

### Year-on-Year Inflation Swap (YYIS)

**Market context.** The YYIS exchanges **annual** inflation returns periodically,
rather than a single cumulative return at maturity. It is the inflation analogue
of a standard fixed-for-floating interest rate swap and is commonly used for
hedging inflation-linked coupon obligations.

#### Cashflows

At each payment date $t_i$:

$$C_i^{\text{infl}} = N \cdot \left(\frac{I(t_i)}{I(t_{i-1})} - 1\right)$$

$$C_i^{\text{fixed}} = N \cdot K \cdot \tau_i$$

where $\tau_i$ is the day count fraction for the period $[t_{i-1}, t_i]$.

#### Convexity Adjustment

A critical subtlety: the expectation of a **ratio** of CPI indices is not equal to
the ratio of expectations:

$$E\!\left[\frac{I(t_i)}{I(t_{i-1})}\right] \neq \frac{E[I(t_i)]}{E[I(t_{i-1})]}$$

This **convexity adjustment** arises because the YoY payment depends on the CPI
ratio observed at $t_i$ but discounted from $t_i$ to today. The adjustment depends
on the volatility and correlation structure of inflation rates and nominal rates.
For typical market parameters, the adjustment is on the order of 1–5 bps per annum,
but it can be significant for long-dated swaps.

#### Code Example

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

### Inflation Cap/Floor

**Market context.** Inflation caps and floors are option overlays on year-on-year
inflation rates. An inflation cap protects against unexpectedly high inflation; an
inflation floor protects against deflation. Together, they form the basis for
inflation collars (long cap + short floor) used extensively by pension funds and
inflation-linked bond issuers.

#### Pricing

Each **caplet** (or floorlet) in the strip is priced via Black-76 on the forward
year-on-year inflation rate:

$$\text{Caplet}_i = \mathrm{DF}(t_i) \cdot N \cdot \tau_i \cdot \text{Black76}(F_i, K, \sigma_i, \tau_i)$$

where:

- $F_i$ — forward year-on-year inflation rate for period $[t_{i-1}, t_i]$
- $K$ — strike inflation rate
- $\sigma_i$ — implied normal or lognormal volatility of the YoY rate
- $\tau_i$ — day count fraction

The total cap (or floor) value is the sum of the individual caplet (floorlet) values.

#### Code Example

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

---

## Equity Exotics

VALAX supports a comprehensive set of exotic equity options beyond the standard
vanillas, barriers, Asians, and lookbacks documented in the core
[Analytical Pricing](analytical.md) guide. Each exotic below has unique payoff
features that require specialized pricing methods.

### Digital Option

**Market context.** A digital (binary) option pays a **fixed cash amount** if the
underlying finishes in-the-money, regardless of how far. Digitals are used in
structured products, sports betting analogues in finance, and as building blocks for
more complex payoffs (e.g., range accruals decompose into strips of digitals).

#### Payoff

Cash-or-nothing digital call:

$$C_{\text{digital}} = e^{-rT} \cdot \text{payout} \cdot \Phi(d_2)$$

Cash-or-nothing digital put:

$$P_{\text{digital}} = e^{-rT} \cdot \text{payout} \cdot \Phi(-d_2)$$

where $d_2 = \frac{\ln(S/K) + (r - q - \sigma^2/2)T}{\sigma\sqrt{T}}$ is the
standard Black-Scholes-Merton $d_2$.

#### Greeks and Hedging

The digital payoff is **discontinuous** at the strike, which creates singular
Greeks:

- **Delta** has a spike near $K$ (approaches a Dirac delta as $T \to 0$).
- **Gamma** changes sign at the strike.
- **Vega** is also discontinuous — a small change in vol dramatically affects the
  probability mass at the boundary.

In practice, digitals are hedged as tight **call spreads** (or put spreads):

$$\text{Digital call} \approx \frac{1}{\epsilon}\bigl[C(K - \epsilon/2) - C(K + \epsilon/2)\bigr]$$

where $\epsilon$ is a small notional-adjusted spread width. This converts the
discontinuous payoff into a smooth one for hedging purposes.

#### Code Example

```python
from valax.instruments import DigitalOption
import jax.numpy as jnp

digital_call = DigitalOption(
    strike=jnp.array(100.0),
    expiry=jnp.array(0.5),       # 6-month expiry
    payout=jnp.array(1_000.0),   # pays $1,000 if ITM
    is_call=True,
)
```

---

### Compound Option

**Market context.** A compound option is an **option on an option**. There are four
types: call-on-call (CoC), call-on-put (CoP), put-on-call (PoC), and put-on-put
(PoP). Compound options appear in real options analysis (staged investment decisions),
FX markets (options on currency option premiums), and corporate finance (equity as
a call on firm value, making equity options compound options on assets).

#### Pricing (Geske Formula)

The Geske (1979) closed-form solution for a call on a call uses the **bivariate
normal distribution** $N_2$:

$$V = S\, e^{-qT_2}\, N_2\!\bigl(a_1,\, b_1;\; \sqrt{T_1/T_2}\,\bigr) - K_2\, e^{-rT_2}\, N_2\!\bigl(a_2,\, b_2;\; \sqrt{T_1/T_2}\,\bigr) - K_1\, e^{-rT_1}\, N(a_2)$$

where:

- $T_1$ = outer expiry (compound option maturity)
- $T_2$ = inner expiry (underlying option maturity), $T_2 > T_1$
- $K_1$ = outer strike (premium to acquire the underlying option)
- $K_2$ = inner strike (strike of the underlying option)
- $a_{1,2}$ and $b_{1,2}$ are adjusted $d_1, d_2$ terms
- $N_2(\cdot, \cdot; \rho)$ = bivariate normal CDF with correlation
  $\rho = \sqrt{T_1 / T_2}$

The formula requires solving for $S^*$, the critical stock price at $T_1$ at which
the underlying option value equals $K_1$.

#### Code Example

```python
from valax.instruments import CompoundOption
import jax.numpy as jnp

# Call on a call: right to buy a 1-year call in 3 months for $5
coc = CompoundOption(
    outer_expiry=jnp.array(0.25),   # 3 months
    outer_strike=jnp.array(5.0),    # premium to acquire underlying option
    inner_expiry=jnp.array(1.0),    # underlying option expires in 1 year
    inner_strike=jnp.array(100.0),  # underlying option strike
    outer_is_call=True,
    inner_is_call=True,
)
```

---

### Chooser Option

**Market context.** A chooser option gives the holder the right to **choose**
whether the option becomes a call or a put at a future "choose date." This is
valuable when the direction of the underlying is uncertain but a large move is
expected (e.g., ahead of an earnings announcement, regulatory decision, or
election).

#### Simple Chooser Decomposition

For a **simple chooser** (same strike $K$ and same expiry $T$ for both the call
and the put), put-call parity yields an elegant decomposition:

$$V_{\text{chooser}} = C(S, K, T) + e^{-q(T - T_c)}\, P\!\left(S,\; K\, e^{-(r-q)(T - T_c)},\; T_c\right)$$

where $T_c$ is the choose date. The chooser equals a call plus a discounted put
with an adjusted strike and shorter maturity. This means a simple chooser can be
priced in closed form using two Black-Scholes evaluations.

For **complex choosers** (different strikes or expiries for the call and put arms),
no decomposition exists and numerical methods (PDE or MC) are required.

#### Code Example

```python
from valax.instruments import ChooserOption
import jax.numpy as jnp

chooser = ChooserOption(
    choose_date=jnp.array(0.25),   # choose in 3 months
    expiry=jnp.array(1.0),         # option expires in 1 year
    strike=jnp.array(100.0),
)
```

---

### Spread Option

**Market context.** Spread options pay the difference between two asset prices minus
a fixed strike. They are ubiquitous in commodity markets (**crack spreads**: crude
oil vs. refined products; **spark spreads**: natural gas vs. electricity), equity
relative-value trading, and interest rate markets (CMS spread options).

#### Kirk's Approximation

For $K \neq 0$, no exact closed form exists. **Kirk's approximation** treats
$S_2 + K$ as a single asset and applies Black-76:

$$V \approx \text{Black76}(S_1,\; S_2 + K,\; \sigma_{\text{spread}},\; T)$$

where the effective spread volatility is:

$$\sigma_{\text{spread}} = \sqrt{\sigma_1^2 - 2\rho\,\sigma_1\,\sigma_2\,\frac{S_2}{S_2 + K} + \left(\sigma_2\,\frac{S_2}{S_2 + K}\right)^2}$$

- $\sigma_1, \sigma_2$ — volatilities of $S_1$ and $S_2$
- $\rho$ — correlation between $S_1$ and $S_2$

#### Margrabe Special Case

When $K = 0$ (pure exchange option), the **Margrabe formula** gives an exact
closed-form solution:

$$V = S_1\, e^{-q_1 T}\, \Phi(d_1) - S_2\, e^{-q_2 T}\, \Phi(d_2)$$

where $d_{1,2}$ use $\sigma_{\text{spread}} = \sqrt{\sigma_1^2 - 2\rho\sigma_1\sigma_2 + \sigma_2^2}$.

#### Code Example

```python
from valax.instruments import SpreadOption
import jax.numpy as jnp

# Crack spread call: profit if gasoline - crude > $5
crack_spread = SpreadOption(
    expiry=jnp.array(0.25),
    strike=jnp.array(5.0),         # spread strike
    notional=jnp.array(1_000.0),
    is_call=True,
)
```

---

### Autocallable

**Market context.** Autocallable structured notes are among the **most traded
structured products globally**, particularly in Asia (Korea, Japan, Hong Kong)
and Europe. A typical autocallable offers an attractive coupon in exchange for
conditional downside exposure. The product is path-dependent with multiple
observation dates.

#### Payoff Mechanics (Step by Step)

At each **observation date** $t_k$ ($k = 1, \ldots, n$):

1. **Autocall check**: If $S(t_k) \geq B_{\text{autocall}} \cdot S_0$, the note
   redeems early at par plus accrued coupons. The investor receives
   $N \cdot (1 + k \cdot c)$ where $c$ is the periodic coupon rate.

2. **Coupon check**: If $S(t_k) \geq B_{\text{coupon}} \cdot S_0$ (but below the
   autocall barrier), a coupon $N \cdot c$ is paid for this period.
   - If `has_memory=True` (phoenix/memory feature): any previously missed coupons
     are also paid.
   - If `has_memory=False`: missed coupons are forfeited.

3. **Knock-in check** (continuous or discrete): If $S(t) < B_{\text{ki}} \cdot S_0$
   at *any* monitoring point, the knock-in put is activated.

4. **At maturity** (if not autocalled):
   - If knock-in was **not** triggered: investor receives par ($N$).
   - If knock-in **was** triggered: investor receives
     $N \cdot \min\!\left(1,\; \frac{S(T)}{K \cdot S_0}\right)$, i.e., exposed
     to downside below the strike.

#### Pricing

No closed-form solution exists. Autocallables require **Monte Carlo simulation**
with path monitoring at all observation dates (and continuous monitoring for the
knock-in barrier if specified). Stochastic local volatility (SLV) models are
preferred for accurate barrier pricing.

See [Monte Carlo](monte-carlo.md) for the simulation engine.

#### Code Example

```python
from valax.instruments import Autocallable
from valax.dates import ymd_to_ordinal
import jax.numpy as jnp

autocallable = Autocallable(
    observation_dates=jnp.array([
        ymd_to_ordinal(2025, 9, 15),
        ymd_to_ordinal(2026, 3, 15),
        ymd_to_ordinal(2026, 9, 15),
        ymd_to_ordinal(2027, 3, 15),
    ]),
    autocall_barrier=jnp.array(1.0),    # 100% of initial spot
    coupon_barrier=jnp.array(0.70),     # 70% of initial spot
    coupon_rate=jnp.array(0.08),        # 8% per period
    ki_barrier=jnp.array(0.60),         # 60% knock-in put barrier
    strike=jnp.array(1.0),             # at-the-money put strike
    notional=jnp.array(100_000.0),
    has_memory=False,
)
```

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| `autocall_barrier` | 0.95–1.05 | Early redemption trigger (fraction of $S_0$) |
| `coupon_barrier` | 0.60–0.80 | Coupon payment trigger |
| `ki_barrier` | 0.50–0.70 | Knock-in put barrier |
| `coupon_rate` | 0.05–0.15 | Per-period coupon rate |
| `has_memory` | True/False | Phoenix (memory) coupon feature |

!!! warning
    Autocallables have **significant vega and correlation exposure** that is hidden
    from simple scenario analysis. A 5-point increase in implied volatility can
    reduce the note value substantially due to increased knock-in probability.
    Always run full Greeks analysis via `jax.grad` through the MC pricer.

---

### Worst-of Basket Option

**Market context.** Worst-of options are common building blocks in structured
products (particularly autocallables referenced to a basket). The payoff depends
on the **worst-performing** asset in a multi-asset basket.

#### Payoff

$$\text{Payoff} = N \cdot \max\!\left(\min_i\!\left(\frac{S_i(T)}{S_i(0)}\right) - K,\; 0\right) \quad \text{(call)}$$

where $S_i(0)$ and $S_i(T)$ are the initial and terminal prices of asset $i$,
and $K$ is the strike expressed as a return level.

#### Correlation Sensitivity

Worst-of options are **highly correlation-sensitive**:

| Correlation | Call Price | Put Price | Intuition |
|-------------|-----------|-----------|-----------|
| High ($\rho \to 1$) | Higher | Lower | Assets move together — worst performer is close to average |
| Low ($\rho \to 0$) | Lower | Higher | Higher chance at least one asset underperforms badly |

Pricing requires **correlated multi-asset Monte Carlo**. The standard approach is
Cholesky decomposition of the correlation matrix $\Sigma = LL^T$ to generate
correlated Brownian increments: $\mathbf{Z}_{\text{corr}} = L \cdot \mathbf{Z}_{\text{indep}}$.

#### Code Example

```python
from valax.instruments import WorstOfBasketOption
import jax.numpy as jnp

# Worst-of put on 3 stocks, strike at 100% (ATM)
wof_put = WorstOfBasketOption(
    expiry=jnp.array(1.0),
    strike=jnp.array(1.0),          # 100% — at-the-money
    notional=jnp.array(1_000_000.0),
    n_assets=3,
    is_call=False,
)

# Correlation matrix and spots passed to the MC pricer, not the instrument
# See Monte Carlo guide for multi-asset simulation
```

---

### Cliquet / Ratchet

**Market context.** Cliquet (ratchet) options are popular in structured notes and
insurance products (equity-indexed annuities). The payoff is the sum of
**capped and floored periodic returns** — the investor participates in each period's
return within bounds, making the product sensitive to the **forward volatility smile**.

#### Payoff

$$\text{Payoff} = N \cdot \max\!\left(\sum_{i=1}^{n} \min\!\left(\max\!\left(\frac{S(t_i)}{S(t_{i-1})} - 1,\; f\right),\; c\right),\; g\right)$$

where:

- $c$ — local cap (maximum credited return per period)
- $f$ — local floor (minimum credited return per period, often 0 %)
- $g$ — global floor (minimum total accumulated return)
- $S(t_i)/S(t_{i-1})$ — periodic return

#### Forward Vol Sensitivity

Cliquets are **forward-starting** options: each period is an option that starts at
the beginning of the period. The value depends on the implied volatility for each
**future** period — not just the current smile. This makes cliquets sensitive to:

- Forward volatility skew
- Term structure of volatility
- Local volatility surface curvature

Standard Black-Scholes with flat vol significantly misprices cliquets. **Local
volatility** or **stochastic local volatility (SLV)** models are required.

#### Code Example

```python
from valax.instruments import Cliquet
from valax.dates import ymd_to_ordinal
import jax.numpy as jnp

cliquet = Cliquet(
    observation_dates=jnp.array([
        ymd_to_ordinal(2025, 6, 15),
        ymd_to_ordinal(2025, 9, 15),
        ymd_to_ordinal(2025, 12, 15),
        ymd_to_ordinal(2026, 3, 15),
        ymd_to_ordinal(2026, 6, 15),
        ymd_to_ordinal(2026, 9, 15),
        ymd_to_ordinal(2026, 12, 15),
        ymd_to_ordinal(2027, 3, 15),
    ]),
    local_cap=jnp.array(0.04),      # 4% cap per quarter
    local_floor=jnp.array(0.0),     # 0% floor (no negative contribution)
    global_floor=jnp.array(0.0),    # 0% global floor
    notional=jnp.array(1_000_000.0),
)
```

!!! tip
    To check model sensitivity, price the same cliquet under Black-Scholes (flat
    vol), local volatility, and Heston. The difference reveals the model risk
    inherent in the product. Use `jax.vmap` to batch across models efficiently.

---

## FX Derivatives

The FX market is the largest financial market by volume, with over \$7.5 trillion
daily turnover. VALAX's base FX instruments (forwards, vanilla options, barrier
options) are documented in the API reference. This section covers the extended set:
quanto options, TARFs, and FX swaps.

All FX instruments use the **FOR/DOM** convention: the foreign currency is the asset,
and the domestic currency is the numeraire. For example, EUR/USD = 1.10 means 1 EUR
costs 1.10 USD.

### Quanto Option

**Market context.** A quanto (quantity-adjusted) option provides exposure to a
**foreign asset** with the payoff converted to the **domestic currency** at a
**fixed FX rate** rather than the prevailing spot rate. This eliminates FX risk
for the investor while maintaining the foreign asset exposure. Quanto options are
common in cross-border equity structured products and commodity-linked notes.

#### Pricing

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

#### Code Example

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

### Target Accrual Range Forward (TARF)

**Market context.** TARFs are popular structured FX products, particularly in
Asian markets (USD/CNH, USD/KRW, USD/TWD). They offer leveraged exposure to an FX
rate with an automatic termination feature when cumulative gains reach a target.
TARFs became notorious during the 2008 and 2015 FX crises for their asymmetric
risk profile.

#### Payoff Mechanics (Step by Step)

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

#### Pricing

TARFs are highly **path-dependent** due to the target accumulation and early
termination feature. Pricing requires Monte Carlo simulation with:

- Simulation of FX spot at each fixing date
- Tracking of cumulative gains
- Early termination logic
- Partial redemption at the terminal fixing

No closed-form or PDE approach is feasible.

#### Code Example

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

### FX Swap

**Market context.** FX swaps are the **most traded FX instrument by volume**,
representing roughly half of all FX market turnover. They consist of two
simultaneous and opposite FX transactions at different value dates. FX swaps are
primarily used for **short-term funding**, **hedging**, and **rolling forward
exposures**.

#### Cashflows

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

#### Code Example

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

---

## Interest Rate Derivatives

Interest rate derivatives are the largest derivatives market by notional. VALAX's
core IR instruments (caps, floors, swaptions, vanilla IRS) are documented in the
[Fixed Income](fixed-income.md) guide and the [Analytical Pricing](analytical.md)
guide. This section covers the extended set of rate derivatives.

### OIS Swap

**Market context.** The Overnight Index Swap (OIS) is the dominant swap type in
the post-LIBOR world. The floating leg references the **compounded daily overnight
rate** — SOFR (USD), €STR (EUR), SONIA (GBP), or TONAR (JPY) — rather than a
term IBOR rate. OIS swaps are the primary instrument for constructing the risk-free
discount curve.

#### Floating Leg Valuation

The floating leg value for an accrual period $[t_{j_0}, t_{j_D}]$ with $D$ business
days is:

$$V_{\text{float}} = N \cdot \left(\prod_{j=1}^{D} \bigl(1 + r_j \cdot \delta_j\bigr) - 1\right)$$

where:

- $r_j$ — overnight rate on business day $j$
- $\delta_j$ — day count fraction for one business day (typically $1/360$ or $1/365$)
- $N$ — notional principal

For past dates, $r_j$ is the realized overnight fixing. For future dates, $r_j$ is
the forward overnight rate implied by the OIS curve.

#### Contrast with IBOR Swaps

| Feature | OIS Swap | IBOR Swap (legacy) |
|---------|----------|--------------------|
| Floating rate | Compounded daily overnight | Term rate (3M/6M LIBOR) |
| Fixing | Daily (compounded in arrears) | Once per period (in advance) |
| Credit risk | Near risk-free (overnight lending) | Includes bank credit premium |
| Curve | Single-curve discounting | Dual-curve (OIS discount + IBOR projection) |

!!! note
    Since the LIBOR transition (completed for most currencies by mid-2024), OIS
    swaps have replaced IBOR swaps as the benchmark. VALAX's curve bootstrapper
    uses OIS instruments as the foundation for the discount curve.

#### Code Example

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

### Cross-Currency Swap

**Market context.** Cross-currency swaps (XCCY) exchange floating-rate cashflows
in two different currencies, with notional exchanges at inception and maturity. They
are essential for hedging foreign-currency debt issuance and for funding in a
non-domestic currency. The **basis spread** (often negative for EUR/USD) reflects
structural supply/demand imbalances for dollar funding.

#### Cashflow Structure

A cross-currency swap has three components:

| Component | Domestic Leg | Foreign Leg |
|-----------|-------------|-------------|
| **Initial exchange** ($t_0$) | Pay $N_d$ | Receive $N_f$ |
| **Periodic coupons** | $N_d \cdot (r_d^{\text{float}} + s) \cdot \tau_i$ | $N_f \cdot r_f^{\text{float}} \cdot \tau_i$ |
| **Final re-exchange** ($T$) | Receive $N_d$ | Pay $N_f$ |

where $s$ is the **basis spread** — the spread added to the domestic floating leg
that makes the swap NPV zero at inception:

$$\text{Basis spread} = s \;\text{ such that }\; PV_{\text{dom}}(\text{dom leg} + s) = PV_{\text{dom}}(\text{for leg})$$

The basis spread is a key observable for cross-currency curve construction.

#### Code Example

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

### Total Return Swap (TRS)

**Market context.** A Total Return Swap transfers the **total economic exposure**
of a reference asset (price change + income) from one party to another in exchange
for a funding rate. TRS are widely used for synthetic exposure (e.g., gaining equity
index exposure without buying shares), prime brokerage financing, and balance sheet
management.

#### Cashflow Legs

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
reference asset funded at $r_{\text{float}} + s$.

#### Code Example

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

### CMS Swap

**Market context.** A Constant Maturity Swap (CMS) has a floating leg that pays
the **par swap rate** for a given tenor (e.g., the 10-year swap rate) observed at
each fixing date. CMS swaps allow investors to express views on the shape and
level of the swap curve without entering into a series of forward-starting swaps.

#### CMS Rate and Convexity Adjustment

The CMS rate $S(T_f)$ is the par swap rate for a specific tenor (e.g., 10Y)
observed at the fixing date $T_f$. Under the payment measure $T_p$, the expected
CMS rate differs from the forward swap rate $S_0$ by a **convexity adjustment**:

$$E^{T_p}[S(T_f)] = S_0 + \text{convexity adjustment}$$

The convexity adjustment arises because:

1. The swap rate is a **nonlinear function** of discount factors.
2. The payment date $T_p$ generally differs from the natural measure of the swap
   rate.

#### Replication Method (Hagan)

The standard approach is **static replication** (Hagan, 2003): express the CMS
payoff as an integral over European swaptions, then use the SABR smile to evaluate
the integral:

$$E^{T_p}[S(T_f)] = S_0 + \int_0^{\infty} w(K) \cdot \text{SwayptionVol}(K)\, dK$$

where $w(K)$ is a weight function derived from the annuity mapping. This approach
naturally captures the smile dependence of the convexity adjustment.

#### Code Example

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

### CMS Cap/Floor

**Market context.** CMS caps and floors are strips of caplets/floorlets on a CMS
rate. They provide protection against rising (cap) or falling (floor) long-term
interest rates. CMS cap/floor pricing inherits the same convexity adjustment
issues as CMS swaps.

#### Pricing

Each CMS caplet pays:

$$\text{CMS caplet}_i = \mathrm{DF}(t_i) \cdot N \cdot \tau_i \cdot E^{T_i}\!\left[\max(S(T_f) - K,\; 0)\right]$$

The expectation is computed via swaption replication with SABR smile integration,
identical to the CMS swap convexity adjustment but applied to the option payoff
rather than the linear payoff.

#### Code Example

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

### Range Accrual

**Market context.** Range accrual notes pay a coupon proportional to the number of
days a reference index stays within a specified range. They are popular in low-
volatility environments when investors seek yield enhancement by selling optionality.
Reference indices include overnight rates, CMS rates, FX rates, or equity indices.

#### Coupon Structure

The coupon for period $i$ is:

$$C_i = N \cdot R \cdot \tau_i \cdot \frac{n_{\text{in range}}}{n_{\text{total}}}$$

where:

- $R$ — maximum coupon rate (earned if the index is always in range)
- $\tau_i$ — day count fraction for the period
- $n_{\text{in range}}$ — number of business days the index is within
  $[\text{lower\_barrier}, \text{upper\_barrier}]$
- $n_{\text{total}}$ — total business days in the period

#### Pricing Decomposition

Each business day in the accrual period contributes a **digital option** to the
total coupon. The range accrual value can be decomposed as:

$$V = \sum_{i=1}^{n} \sum_{j=1}^{D_i} \frac{R \cdot \delta_j \cdot N}{D_i} \cdot \mathrm{DF}(t_j) \cdot P\!\bigl(L \leq X(t_j) \leq U\bigr)$$

where $X(t_j)$ is the reference index level on day $j$, $L$ and $U$ are the lower
and upper barriers, and $\delta_j$ is the daily accrual fraction. Each probability
term is the value of a digital range option (long digital at $L$, short digital
at $U$).

For CMS-linked range accruals, the distribution of the CMS rate at each daily
observation requires the same convexity-adjusted smile used for CMS caps/floors.

#### Code Example

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

---

## Summary

The table below summarizes all instruments covered in this guide, their primary
pricing methods, and the relevant VALAX modules.

| Asset Class | Instrument | Pricing Method | Module |
|-------------|-----------|---------------|--------|
| **Credit** | CDS | Survival curve + discounting | `valax.instruments.credit` |
| **Credit** | CDO Tranche | Gaussian copula (base correlation) | `valax.instruments.credit` |
| **Fixed Income** | Floating Rate Bond | Forward curve projection | `valax.instruments.bonds` |
| **Fixed Income** | Callable Bond | Hull-White tree / PDE + OAS | `valax.instruments.bonds` |
| **Fixed Income** | Puttable Bond | Hull-White tree / PDE | `valax.instruments.bonds` |
| **Fixed Income** | Convertible Bond | Equity-credit PDE | `valax.instruments.bonds` |
| **Inflation** | ZCIS | Inflation forward curve | `valax.instruments.inflation` |
| **Inflation** | YYIS | Inflation curve + convexity adj. | `valax.instruments.inflation` |
| **Inflation** | Inflation Cap/Floor | Black-76 on YoY forward | `valax.instruments.inflation` |
| **Equity Exotics** | Digital Option | BSM closed-form | `valax.instruments.options` |
| **Equity Exotics** | Compound Option | Bivariate normal (Geske) | `valax.instruments.options` |
| **Equity Exotics** | Chooser Option | BSM decomposition | `valax.instruments.options` |
| **Equity Exotics** | Spread Option | Kirk / Margrabe / 2D MC | `valax.instruments.options` |
| **Equity Exotics** | Autocallable | Monte Carlo (SLV) | `valax.instruments.options` |
| **Equity Exotics** | Worst-of Basket | Correlated MC (Cholesky) | `valax.instruments.options` |
| **Equity Exotics** | Cliquet | Local vol / SLV MC | `valax.instruments.options` |
| **FX** | Quanto Option | Modified GK (quanto adj.) | `valax.instruments.fx` |
| **FX** | TARF | Monte Carlo | `valax.instruments.fx` |
| **FX** | FX Swap | Discounted cashflows | `valax.instruments.fx` |
| **Rates** | OIS Swap | OIS curve (daily compounding) | `valax.instruments.rates` |
| **Rates** | Cross-Currency Swap | Multi-curve + FX | `valax.instruments.rates` |
| **Rates** | Total Return Swap | Reference asset + funding | `valax.instruments.rates` |
| **Rates** | CMS Swap | Hagan replication + SABR | `valax.instruments.rates` |
| **Rates** | CMS Cap/Floor | Swaption replication | `valax.instruments.rates` |
| **Rates** | Range Accrual | Digital decomposition | `valax.instruments.rates` |

!!! tip
    All instruments in this guide are **fully compatible** with JAX transformations.
    Use `jax.jit` for speed, `jax.grad` for Greeks, `jax.vmap` for batch pricing
    across strikes/maturities/scenarios, and `jax.pmap` for multi-GPU distribution.
    See [Greeks & Risk Sensitivities](greeks.md) and [Risk Management](risk.md)
    for details.
