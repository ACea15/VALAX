# Callable, Puttable, Floating, and Convertible Bonds

Fixed income instruments generate predictable streams of cashflows (coupons and
principal). VALAX's base fixed-rate bond and zero-coupon bond are documented in the
[Fixed Income](fixed-income.md) guide. This page covers the extended set:
floating-rate bonds, callable bonds, puttable bonds, and convertible bonds.

Callable and puttable bonds are priced via the Hull-White trinomial tree — see the
[Short-Rate Models](short-rate.md) guide for the end-to-end Hull-White workflow.

## Floating Rate Bond (FRN)

**Market context.** Floating-rate notes (FRNs) pay coupons that reset periodically
based on a **reference rate** — SOFR (USD), EURIBOR (EUR), SONIA (GBP), or TONAR
(JPY). FRNs represent a major segment of the bond market, particularly in
leveraged finance (floating-rate bank loans) and government-sponsored issuance
(US Treasury FRNs).

### Cashflow Mathematics

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

### Duration Properties

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

### Code Example

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

## Callable Bond

**Market context.** The majority of US corporate bonds are callable — the issuer
retains the right to redeem the bond early, typically after an initial non-call
period. Issuers exercise this right when interest rates fall (they can refinance
at a lower coupon). From the investor's perspective, the callable bond offers a
higher yield than an equivalent non-callable bond to compensate for the embedded
short call position.

### Pricing

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

### Code Example

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
    [Lattice Methods](lattice.md) and [Short-Rate Models](short-rate.md) for the
    tree implementation.

---

## Puttable Bond

**Market context.** A puttable bond is the mirror image of a callable bond — the
**bondholder** has the right to sell the bond back to the issuer at a predetermined
price on specified put dates. This protects the investor against rising interest
rates or credit deterioration. Puttable bonds are less common than callable bonds
but appear in corporate and municipal markets.

### Valuation

The value of a puttable bond decomposes as:

$$V_{\text{puttable}} = V_{\text{straight}} + V_{\text{put option}}$$

The embedded put **increases** the bond value relative to an otherwise identical
straight bond. The put option value is highest when rates are high (the holder
would exercise to reinvest at higher yields) or when credit quality has deteriorated.

Pricing mechanics are analogous to callable bonds (Hull-White tree or PDE with
backward induction), but the exercise decision now belongs to the **bondholder**:
at each put date, $V_{\text{node}} = \max(V_{\text{continuation}}, \text{put price})$.

### Code Example

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

## Convertible Bond

!!! warning "Planned — not yet implemented"
    Convertible bonds are on the roadmap (Tier 3.8). The section below documents
    the intended interface and pricing approach.

**Market context.** A convertible bond is an equity-credit hybrid — a fixed-rate
bond with an embedded equity conversion option. The bondholder may convert the bond
into a fixed number of the issuer's shares. Convertibles appeal to investors
seeking bond-like downside protection with equity upside participation. They are
widely issued by growth companies (lower coupon than straight debt) and actively
traded by convertible arbitrage hedge funds.

### Conversion Value

The conversion value at any point is:

$$\text{Conversion value} = \text{conversion\_ratio} \times S$$

where $S$ is the current stock price. The bondholder converts when the conversion
value exceeds the bond's straight debt value (plus any accrued coupon).

The **conversion premium** is the excess of the bond price over conversion value,
expressed as a percentage:

$$\text{Conversion premium} = \frac{P_{\text{bond}} - \text{conversion value}}{\text{conversion value}}$$

### Pricing

Convertible bond pricing requires a model that captures **equity risk**, **credit
risk**, and **interest rate risk** simultaneously. The standard approach is a
**1D PDE** in stock price with a credit-adjusted boundary:

1. The stock price follows GBM with a default intensity $h$: at default, the stock
   drops to zero and the bond recovers $R \cdot \text{face value}$.
2. The PDE is solved backward in time with boundary conditions for conversion
   (max of continuation and conversion value) and optional issuer call.
3. Credit spread enters through the discounting: the risky discount rate is
   $r + h$ where $h$ is the hazard rate.

### Code Example

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
    (discounted at the risk-free rate $r$). This is the standard implementation
    target for VALAX's future PDE pricer. See [PDE Methods](pde.md).
