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
