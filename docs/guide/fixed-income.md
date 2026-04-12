# Fixed Income

VALAX provides a full fixed income stack: date utilities, discount curves, bond instruments, and pricing with autodiff risk measures (duration, convexity, key-rate durations).

## Date Utilities

All dates inside JAX-traced code are **integer ordinals** (days since 1970-01-01). This makes date arithmetic JIT-compatible via pure integer ops.

### Converting Dates

```python
from valax.dates import ymd_to_ordinal

settlement = ymd_to_ordinal(2025, 1, 15)  # -> jnp.int32 scalar
maturity = ymd_to_ordinal(2030, 1, 15)
```

### Day Count Conventions

Four conventions are available, all operating on ordinal dates:

| Convention | Function | Description |
|---|---|---|
| ACT/365 | `act_365(start, end)` | Actual days / 365 |
| ACT/360 | `act_360(start, end)` | Actual days / 360 |
| ACT/ACT | `act_act(start, end)` | Actual days / 365.25 |
| 30/360 | `thirty_360(start, end)` | 30/360 Bond Basis |

```python
from valax.dates import year_fraction, ymd_to_ordinal

start = ymd_to_ordinal(2025, 1, 15)
end = ymd_to_ordinal(2025, 7, 15)

yf = year_fraction(start, end, "act_365")   # 0.4959...
yf = year_fraction(start, end, "30_360")    # 0.5 (exactly 6 months)
```

### Schedule Generation

Generate coupon payment dates backward from maturity (standard bond convention):

```python
from valax.dates import generate_schedule

# 5-year semi-annual bond
schedule = generate_schedule(
    start_year=2025, start_month=1, start_day=15,
    end_year=2030, end_month=1, end_day=15,
    frequency=2,  # semi-annual
)
# Returns: array of 10 ordinal dates (every 6 months, excluding issue date)
```

## Discount Curves

`DiscountCurve` is a JAX pytree storing pillar dates and discount factors. Interpolation is **log-linear** (piecewise-constant forward rates), with flat extrapolation.

```python
import jax.numpy as jnp
from valax.dates import ymd_to_ordinal
from valax.curves import DiscountCurve, forward_rate, zero_rate

ref = ymd_to_ordinal(2025, 1, 1)

# Build a curve from market discount factors
curve = DiscountCurve(
    pillar_dates=jnp.array([
        int(ymd_to_ordinal(2025, 1, 1)),
        int(ymd_to_ordinal(2026, 1, 1)),
        int(ymd_to_ordinal(2028, 1, 1)),
        int(ymd_to_ordinal(2030, 1, 1)),
        int(ymd_to_ordinal(2035, 1, 1)),
    ], dtype=jnp.int32),
    discount_factors=jnp.array([1.0, 0.9512, 0.9048, 0.8607, 0.7788]),
    reference_date=ref,
    day_count="act_365",
)

# Interpolate at any date
df = curve(ymd_to_ordinal(2027, 6, 15))

# Forward and zero rates
fwd = forward_rate(curve, ymd_to_ordinal(2026, 1, 1), ymd_to_ordinal(2027, 1, 1))
zr = zero_rate(curve, ymd_to_ordinal(2028, 1, 1))
```

Because `DiscountCurve` is a pytree with differentiable discount factor leaves, `jax.grad` through any pricing function that takes a curve gives **key-rate durations for free**.

## Bond Instruments

Bonds are data-only pytrees, following the same pattern as `EuropeanOption`.

### Zero-Coupon Bond

```python
from valax.instruments import ZeroCouponBond

zcb = ZeroCouponBond(
    maturity=ymd_to_ordinal(2030, 1, 1),
    face_value=jnp.array(100.0),
)
```

### Fixed-Rate Coupon Bond

```python
from valax.instruments import FixedRateBond
from valax.dates import generate_schedule, ymd_to_ordinal

bond = FixedRateBond(
    payment_dates=generate_schedule(2025, 1, 15, 2030, 1, 15, frequency=2),
    settlement_date=ymd_to_ordinal(2025, 1, 15),
    coupon_rate=jnp.array(0.04),   # 4% annual coupon
    face_value=jnp.array(100.0),
    frequency=2,                    # semi-annual
    day_count="act_365",
)
```

## Bond Pricing

### From a Discount Curve

```python
from valax.pricing.analytic import zero_coupon_bond_price, fixed_rate_bond_price

# Zero-coupon bond
zcb_price = zero_coupon_bond_price(zcb, curve)

# Fixed-rate bond — discounts each coupon and redemption
bond_price = fixed_rate_bond_price(bond, curve)
```

### From a Yield-to-Maturity

```python
from valax.pricing.analytic import fixed_rate_bond_price_from_yield

# Standard bond pricing formula: P = sum C/(1+y/f)^i + F/(1+y/f)^n
price = fixed_rate_bond_price_from_yield(bond, ytm=jnp.array(0.05))
```

### Yield-to-Maturity Solver

Newton-Raphson solver using autodiff — no hand-coded derivative:

```python
from valax.pricing.analytic import yield_to_maturity

ytm = yield_to_maturity(bond, market_price=jnp.array(95.50))
```

## Risk Measures via Autodiff

This is where VALAX's autodiff approach shines. Duration, convexity, and key-rate durations are computed by differentiating the pricing function — no separate formulas needed.

### Modified Duration and Convexity

```python
from valax.pricing.analytic import modified_duration, convexity

ytm = jnp.array(0.05)

# -1/P * dP/dy  (via jax.grad)
md = modified_duration(bond, ytm)

# 1/P * d²P/dy²  (via nested jax.grad)
cx = convexity(bond, ytm)

# Price change approximation for a 50bp yield shock
dy = 0.005
dp_approx = -md * dy + 0.5 * cx * dy**2
```

### Key-Rate Durations

Sensitivity of bond price to each pillar on the discount curve. One backward pass gives all sensitivities simultaneously:

```python
from valax.pricing.analytic import key_rate_durations

krd = key_rate_durations(bond, curve)
# krd.shape == (n_pillars,)
# krd[i] = sensitivity to the zero rate at pillar i
```

$$\text{KRD}_i = -\frac{1}{P} \frac{\partial P}{\partial r_i}$$

This is computed via `jax.grad` through the discount curve pytree — the exact same mechanism that gives delta and gamma for options.

!!! tip "Autodiff Advantage"
    In a traditional library, key-rate durations require N+1 curve builds (one per pillar bump). VALAX computes all of them in a **single backward pass** through `jax.grad`, which is both faster and exact to machine precision.

## Full Example

```python
import jax.numpy as jnp
from valax.dates import ymd_to_ordinal, generate_schedule
from valax.instruments import FixedRateBond
from valax.curves import DiscountCurve
from valax.pricing.analytic import (
    fixed_rate_bond_price,
    yield_to_maturity,
    modified_duration,
    convexity,
    key_rate_durations,
)

# 5-year, 4% semi-annual bond
ref = ymd_to_ordinal(2025, 1, 1)
bond = FixedRateBond(
    payment_dates=generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2),
    settlement_date=ref,
    coupon_rate=jnp.array(0.04),
    face_value=jnp.array(100.0),
    frequency=2,
)

# Flat 5% curve
pillars = jnp.array([int(ymd_to_ordinal(2025+i, 1, 1)) for i in range(11)], dtype=jnp.int32)
times = (pillars - int(ref)).astype(jnp.float64) / 365.0
curve = DiscountCurve(
    pillar_dates=pillars,
    discount_factors=jnp.exp(-0.05 * times),
    reference_date=ref,
)

# Price and risk
price = fixed_rate_bond_price(bond, curve)
ytm = yield_to_maturity(bond, price)
md = modified_duration(bond, ytm)
cx = convexity(bond, ytm)
krd = key_rate_durations(bond, curve)

print(f"Price:    {price:.4f}")
print(f"YTM:      {ytm:.4f}")
print(f"Duration: {md:.4f}")
print(f"Convexity:{cx:.4f}")
print(f"KRDs:     {krd}")
```

## Floating Rate Notes & OIS Swaps

Floating-rate notes and overnight-index swaps share the same core identity: under a **single curve** (the discount curve is also the projection curve for the floating index), their float legs telescope. This gives exact, closed-form pricing without any cashflow simulation.

### The telescoping identity

For a single accrual period $[T_{i-1}, T_i]$ the simply-compounded forward rate satisfies

$$F_i \cdot \tau_i = \frac{DF(T_{i-1})}{DF(T_i)} - 1.$$

Multiplying by $N \cdot DF(T_i)$ and summing over the schedule:

$$\sum_i N \cdot F_i \cdot \tau_i \cdot DF(T_i) = N \cdot \sum_i \bigl(DF(T_{i-1}) - DF(T_i)\bigr) = N \cdot \bigl(DF(T_0) - DF(T_n)\bigr).$$

The floating leg PV collapses to **two discount factors** — the start and the end of the schedule.

### Floating rate note (FRN)

`floating_rate_bond_price(bond, curve)` applies this identity coupon-by-coupon so that a fixed **spread** and any **known past fixings** can be layered on top.  For each period $i$,

$$C_i = N \cdot (F_i + s) \cdot \tau_i$$

where $F_i$ is either taken from `bond.fixing_rates[i]` (when that entry is finite) or projected from the curve. The PV is then $\sum_i C_i \cdot DF(T_i) + N \cdot DF(T_n)$, summed over **future** cash flows only.

**Par-at-reset invariant.** For a zero-spread FRN valued on its first reset date, the coupon sum collapses via telescoping to $N \cdot DF(T_0) - N \cdot DF(T_n)$, and adding the redemption $N \cdot DF(T_n)$ gives exactly the face value. With a non-zero spread this becomes $P = N + N \cdot s \cdot A$, where $A$ is the discounted day-count annuity of the schedule.

```python
from valax.instruments import FloatingRateBond
from valax.pricing.analytic import floating_rate_bond_price
from valax.dates import generate_schedule, ymd_to_ordinal
import jax.numpy as jnp
import numpy as np

ref = ymd_to_ordinal(2025, 1, 1)
sched = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=4)  # quarterly
# Fixing dates = start of each accrual period (previous payment date, or ref)
fixings = jnp.array([int(ref)] + list(np.asarray(sched[:-1]).tolist()), dtype=jnp.int32)

frn = FloatingRateBond(
    payment_dates=sched,
    fixing_dates=fixings,
    settlement_date=ref,
    spread=jnp.array(0.005),     # 50 bps over the index
    face_value=jnp.array(100.0),
)
price = floating_rate_bond_price(frn, curve)
# At issue, price ≈ 100 + 100 · 0.005 · annuity
```

**Seasoned FRNs.** Pass the historical fixings as a 1-D array in `fixing_rates`, using `NaN` for periods that have not yet fixed. The pricer uses known rates where available and projects from the curve elsewhere:

```python
known = jnp.array([0.045, float("nan"), float("nan"), ...])  # period 0 already fixed
frn_seasoned = FloatingRateBond(
    payment_dates=sched, fixing_dates=fixings,
    settlement_date=ref, spread=jnp.array(0.005),
    face_value=jnp.array(100.0),
    fixing_rates=known,
)
```

### OIS swap

`ois_swap_price(swap, curve)` values an overnight index swap under the same single-curve assumption. The floating leg pays the daily-compounded overnight rate, but under log-linear DF interpolation the compounded rate *exactly* matches the simply-compounded forward, so the same telescoping identity applies:

$$\text{PV}_{\text{float}} = N \cdot \bigl(DF(T_0) - DF(T_n^{\text{float}})\bigr), \qquad \text{PV}_{\text{fixed}} = N \cdot K \cdot A^{\text{fixed}}.$$

```python
from valax.instruments import OISSwap
from valax.pricing.analytic import ois_swap_price, ois_swap_rate

ois = OISSwap(
    start_date=ref,
    fixed_dates=sched,
    float_dates=sched,           # same schedule on both legs
    fixed_rate=jnp.array(0.05),
    notional=jnp.array(1_000_000.0),
)
npv = ois_swap_price(ois, curve)
par = ois_swap_rate(ois, curve)   # par rate makes NPV exactly zero
```

`ois_swap_rate` is the companion par-rate solver: $K^\ast = \bigl(DF(T_0) - DF(T_n)\bigr) / A$. Structurally identical to `swap_rate` from the vanilla IRS pricer, but keyed off the `OISSwap` pytree so fixed and floating legs can carry distinct schedules (e.g. annual fixed vs. quarterly float).

!!! note "Single-curve assumption"
    Both `floating_rate_bond_price` and `ois_swap_price` assume the discount curve also forecasts the floating index. Separating a tenor-specific forecasting curve from the OIS discount curve (a **multi-curve** setup) is the right next step for basis-spread products such as cross-currency and Libor-OIS basis swaps — the `valax.curves.multi_curve` module has the primitives but is not yet wired into these pricers.

!!! tip "Autodiff Greeks come for free"
    Because these pricers are ordinary JAX functions of the curve pytree, `jax.grad` with respect to the curve's log-discount-factors gives the full **key-rate sensitivity vector** in a single backward pass — the same technique used for fixed-rate bonds above.

## Rates Exotics

VALAX now prices five additional rates and cross-asset swap products:
**cross-currency basis swaps**, **total return swaps**, **CMS swaps**,
**CMS caps/floors**, and **range accrual notes**. All pricers live in
`valax.pricing.analytic.rates_exotics` and follow the same
pure-function, single-curve-per-currency philosophy as the FRN and OIS
pricers above.

### Cross-currency basis swap

A cross-currency swap exchanges floating-rate coupons in two currencies,
with optional notional exchanges at inception and maturity.
`cross_currency_swap_price(swap, domestic_curve, foreign_curve, spot)`
applies the telescoping identity **in each currency independently**,
then converts the foreign leg to domestic at the prevailing spot rate:

$$\text{PV}_{\text{receive dom}} = \underbrace{N_d(DF_d(T_0) - DF_d(T_n))}_{\text{dom float}} + \underbrace{N_d \cdot s \cdot A_d}_{\text{basis spread}} - \underbrace{\text{spot} \cdot N_f(DF_f(T_0) - DF_f(T_n))}_{\text{for float}} + \text{notional exchanges}$$

When `exchange_notional=True` the notional exchange terms cancel exactly
against the respective float legs, collapsing the entire NPV to

$$\text{NPV} = N_d \cdot s \cdot A_d.$$

This is the classic XCCY identity — the par basis spread is determined
entirely by the funding-value mismatch between the two currencies.

```python
from valax.instruments import CrossCurrencySwap
from valax.pricing.analytic import cross_currency_swap_price, cross_currency_basis_spread

xccy = CrossCurrencySwap(
    start_date=ref, payment_dates=sched, maturity_date=sched[-1],
    domestic_notional=jnp.array(110_000_000.0),  # spot × 100M
    foreign_notional=jnp.array(100_000_000.0),
    basis_spread=jnp.array(-0.002),              # -20 bps
    exchange_notional=True,
)
npv = cross_currency_swap_price(xccy, usd_curve, eur_curve, jnp.array(1.10))
par_spread = cross_currency_basis_spread(xccy, usd_curve, eur_curve, jnp.array(1.10))
```

### Total return swap

A total return swap exchanges the full economic return (price change +
income) on a reference asset against a funding leg paying floating +
spread. Under the **self-financing** assumption (the asset's expected
return equals the risk-free rate), the TR and floating legs telescope
identically and the NPV at a reset date is:

$$\text{NPV}_{\text{TR receiver}} = -N \cdot s \cdot A$$

An optional `unrealized_return` argument captures mid-period mark-to-market:

```python
from valax.instruments import TotalReturnSwap
from valax.pricing.analytic import total_return_swap_price

trs = TotalReturnSwap(
    start_date=ref, payment_dates=sched,
    notional=jnp.array(10_000_000.0), funding_spread=jnp.array(0.005),
)
npv = total_return_swap_price(trs, curve)
npv_live = total_return_swap_price(trs, curve, unrealized_return=jnp.array(0.02))
```

### CMS swap and CMS cap/floor

A CMS swap pays the N-year par swap rate at each fixing date vs. a fixed
rate. `cms_swap_price` computes the forward par swap rate of a synthetic
annual N-year underlying swap for each fixing date:

$$S_i = \frac{DF(t_i) - DF(t_i + N \cdot Y)}{\sum_{k=1}^{N} DF(t_i + k \cdot Y)}$$

and sums the discounted CMS vs. fixed legs.

`cms_cap_floor_price_black76` applies Black-76 to each per-period CMS
rate, giving European caplets/floorlets on the forward CMS rate:

```python
from valax.instruments import CMSSwap, CMSCapFloor
from valax.pricing.analytic import cms_swap_price, cms_cap_floor_price_black76

cms = CMSSwap(
    start_date=ref, payment_dates=sched,
    fixed_rate=jnp.array(0.05), notional=jnp.array(1_000_000.0), cms_tenor=10,
)
npv = cms_swap_price(cms, curve)

cms_cap = CMSCapFloor(
    payment_dates=sched, strike=jnp.array(0.05),
    notional=jnp.array(1_000_000.0), cms_tenor=10, is_cap=True,
)
cap_pv = cms_cap_floor_price_black76(cms_cap, curve, vol=jnp.array(0.25))
```

!!! warning "No convexity adjustment"
    The true expected CMS rate under each payment measure differs from
    the forward par swap rate by a **convexity term** that depends on
    the swap-rate volatility surface.  The current pricers use the
    forward directly, which is a standard baseline but not
    market-accurate.  Full Hagan-replication / SABR-integration is a
    legitimate larger piece of work tracked in the roadmap.

### Range accrual note

A range accrual pays a coupon proportional to the fraction of time the
reference rate spends inside a range $[L, U]$.
`range_accrual_price_black76` replaces true day-by-day monitoring with
the **snapshot probability** under Black-76 that the forward rate is
in-range at the start of each period:

$$\mathbb{P}(L < F_i < U) = \Phi(-d_{2,U}) - \Phi(-d_{2,L})$$

$$\text{Coupon PV}_i = N \cdot R \cdot \tau_i \cdot \mathbb{P}(L < F_i < U) \cdot DF(t_i)$$

```python
from valax.instruments import RangeAccrual
from valax.pricing.analytic import range_accrual_price_black76

ra = RangeAccrual(
    payment_dates=sched, coupon_rate=jnp.array(0.08),
    lower_barrier=jnp.array(0.01), upper_barrier=jnp.array(0.10),
    notional=jnp.array(1_000_000.0),
)
pv = range_accrual_price_black76(ra, curve, vol=jnp.array(0.30))
```

!!! note "Snapshot approximation"
    The digital-replication approach gives a single probability per
    period rather than per-day monitoring.  For short accrual periods
    and moderate vol this is a good approximation; for production-grade
    per-day range accruals, use Monte Carlo simulation.
