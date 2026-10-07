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

### Day Counts and Schedules

Day-count conventions, stub placement, end-of-month rolls, accrued interest and
the clean/dirty distinction are all documented on
**[Dates and Conventions](dates-and-conventions.md)**. In brief, a schedule is
three aligned arrays, because sizing a coupon needs the period it accrued over
and not just the date the cash arrives:

```python
from valax.dates import generate_schedule

# 5-year semi-annual bond
accrual_start, accrual_end, payment_dates = generate_schedule(
    start_year=2025, start_month=1, start_day=15,
    end_year=2030, end_month=1, end_day=15,
    frequency=2,  # semi-annual
)
# 10 periods. accrual_start[0] is the dated date (2025-01-15); every date
# keeps its day-of-month, so payments fall on the 15th, and accrual_end[-1]
# is exactly the stated maturity.
```

Pass `accrual_start` to the instrument as `accrual_start_dates` whenever the
schedule has a stub, since a stub period cannot be recovered from the payment
dates alone.

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

accrual_start, _, payment_dates = generate_schedule(
    2025, 1, 15, 2030, 1, 15, frequency=2
)

bond = FixedRateBond(
    payment_dates=payment_dates,
    accrual_start_dates=accrual_start,   # sizes each coupon; required for stubs
    settlement_date=ymd_to_ordinal(2025, 1, 15),
    coupon_rate=jnp.array(0.04),   # 4% annual coupon
    face_value=jnp.array(100.0),
    frequency=2,                    # yield compounding only
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

### Z-Spread and Spread DV01

The **Z-spread** is the constant continuously-compounded spread added to every point of the discount curve that reprices the bond to its market price — the standard measure of a bond's credit/liquidity premium over the risk-free curve. Its risk sensitivities (spread duration, spread DV01, spread convexity) fall out of the same autodiff machinery.

Pass `price_type="clean"` for a conventional bond quote: the solver inverts a dirty repricing map, and treating a clean quote as dirty biases the recovered spread by the accrued interest — tens of basis points mid-period. See [Dates and Conventions](dates-and-conventions.md#accrued-interest-clean-and-dirty).

```python
from valax.risk import (
    bond_z_spread,
    z_spread_dv01,
    z_spread_duration,
    z_spread_convexity,
)

# Solve the Z-spread that matches a market *clean* quote of 96.0.
# Bonds are quoted clean while the pricer produces a dirty price, so the
# price type is stated explicitly — see Dates and Conventions.
z = bond_z_spread(bond, curve, jnp.array(96.0), price_type="clean")

# Spread risk evaluated at the bond's own Z-spread
dv01 = z_spread_dv01(bond, curve, z)        # cash P&L per +1 bp of spread
dur  = z_spread_duration(bond, curve, z)    # -1/P dP/dz
cvx  = z_spread_convexity(bond, curve, z)   #  1/P d²P/dz²
```

`z_spread_dv01` is reported positive for a long position (price falls as the spread widens) and satisfies $\text{DV01}_z = D_z \, P \times 10^{-4}$. For a bond discounted off a single curve, a Z-spread shift is the same curve move as a parallel zero-rate shift, so `z_spread_dv01` equals the bond's parallel-curve DV01 — the *spread* framing is what separates credit/liquidity risk from pure rate risk in a report. For a **callable** bond the analogous, option-aware measure is the OAS; see [Callable Bonds](callable-bonds.md#option-adjusted-spread-oas) and [Models & Theory §7.9](../theory/risk-measures.md#79-option-adjusted-spread-and-z-spread).

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
    payment_dates=generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2).payment_dates,
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
sched = generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=4).payment_dates  # quarterly
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

## Rates & cross-currency exotics

Cross-currency basis swaps, total return swaps, CMS swaps, CMS caps/floors, and
range accrual notes extend the same single-curve telescoping identity used for the
FRN and OIS above. To keep each instrument documented in exactly one place, their
cashflow mechanics, pricing formulas, and runnable examples live in the dedicated
[Interest Rate Exotics guide](rates-exotics.md).

## Inflation Derivatives

Inflation is projected off an `InflationCurve` (forward CPI term structure), which
prices the three standard inflation products — zero-coupon inflation swaps (ZCIS),
year-on-year inflation swaps (YYIS), and inflation caps/floors. These are
documented, with breakeven pricing and runnable examples, in the dedicated
[Inflation Derivatives guide](inflation.md).
