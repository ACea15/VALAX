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
    Both `floating_rate_bond_price` and `ois_swap_price` assume the discount curve also forecasts the floating index, and neither accepts an amortising notional or a fixing history. For a dual-curve, amortising or seasoned trade use the [leg engine](#swap-legs-and-multi-leg-trades) below, which takes a projection curve per leg.

!!! tip "Autodiff Greeks come for free"
    Because these pricers are ordinary JAX functions of the curve pytree, `jax.grad` with respect to the curve's log-discount-factors gives the full **key-rate sensitivity vector** in a single backward pass — the same technique used for fixed-rate bonds above.

## Swap legs and multi-leg trades

The pricers above each hard-code a trade *shape*: `OISSwap` carries one
`fixed_dates` array, one `float_dates` array and a single `forward_id`. That is
enough for a vanilla swap and not enough for three common cases — an
**amortising** notional, a **seasoned** leg whose current coupon is part-earned,
and a trade whose legs project off **different indices**.

`valax.instruments.legs` represents a trade as a tuple of legs instead, each
naming its own curves:

| Class | What it carries |
|---|---|
| `FixedLeg` | accrual + payment schedule, fixed rate, notional, amortisation, `discount_id` |
| `FloatingLeg` | the same plus `compounding`, `index_id` and `forward_id` |
| `MultiLegSwap` | `legs: tuple[...]`, and the union of the curves and indices they read |

Pricing lives in `valax.pricing.analytic.legs` (`fixed_leg_pv`,
`floating_leg_pv`, the type-dispatched `leg_pv`, `swap_pv`, `swap_fair_rate`),
keeping instruments data-only.

### Compounding conventions

`FloatingLeg.compounding` selects how the daily index becomes a coupon:

| Value | Coupon for period $i$ | Use |
|---|---|---|
| `"simple"` | $(r_i + s)\,\tau_i$ | one observed fixing per period (IBOR-style) |
| `"compounded"` | $\prod_j (1 + r_j \delta_j) - 1 + s\,\tau_i$ | the post-LIBOR RFR standard |
| `"averaged"` | $(\bar r_i + s)\,\tau_i$ | arithmetic average, discarding interest-on-interest |

The compounded coupon is evaluated as the discount-factor ratio
$P_f(0,T_i^s)/P_f(0,T_i^e) - 1$, since the daily product telescopes; the day
count therefore cancels out of the index term and appears only on the spread.
The averaged convention is genuinely different — a sum of daily ratios rather
than a product — and is worth *less*. The two formulas, and why the gap
survives even on a perfectly flat curve, are derived once under
[Compounded vs averaged futures](curves.md#compounded-vs-averaged-futures);
the leg engine reuses that module's implementation rather than restating it.

!!! warning "Two limitations, stated rather than defaulted"
    **`"averaged"` needs a shape at trace time.** The average is defined per
    rate day, so a period must be subdivided into its actual days, and that
    array's length depends on the dates' *values*. An averaged leg therefore
    prices eagerly, or closed over a traced function, or — as a traced
    argument and hence under `vmap` — with `max_days`, a static bound on the
    subdivision that you derive with
    `valax.instruments.legs.max_period_days`:

    ```python
    from valax.instruments.legs import max_period_days

    leg = FloatingLeg(..., compounding="averaged",
                      max_days=max_period_days(starts, ends))
    ```

    Only the *bound* is static — the dates stay dynamic, which is what keeps
    one compilation serving every schedule and keeps a book of legs stackable.
    A bound that is too small raises rather than silently truncating the
    average. Background:
    [JAX Patterns § 2.5](../architecture/jax-patterns.md#25-when-you-genuinely-need-a-value-at-trace-time-data-dependent-shapes).

    **LIBOR-in-arrears is absent, not approximated.** A leg whose rate is
    observed at $T_i$ for $[T_i, T_i+\text{tenor}]$ but applied to
    $[T_{i-1}, T_i]$ is not worth its forward — it needs a convexity
    adjustment. Rather than project it off the curve and be silently wrong,
    there is no such convention. Fixing *timing* is otherwise carried by the
    `fixing_dates` array, not by a flag: an in-advance simple leg sets them to
    the accrual starts, while a compounded leg observes the index across its
    accrual period and so rejects `fixing_dates` outright.

### Amortisation is a factor, not a notional

`notional` is a scalar reference notional and `amortisation` holds
**dimensionless outstanding-notional factors**, one per period. A 100mm swap
stepping down to 20mm carries `notional=1e8` and
`amortisation=[1.0, 0.8, 0.6, 0.4, 0.2]`.

This keeps the notional *outside* the summation, which is what makes a
non-amortising leg reproduce the older pricers bit for bit:
$\sum_i (N x_i)$ and $N \sum_i x_i$ differ in the last bit for most inputs, so
absolute per-period notionals could not have matched exactly. Multiplying by
exactly `1.0` is exact in IEEE-754, so the overlay is inert when absent.

### Seasoning

Realised fixings are supplied at pricing time from a
[`FixingHistory`](curves.md) keyed by the leg's `index_id`, not stored on the
leg, so one trade can be revalued against successive histories. Each period is
settled, straddling or unborn according to where the valuation date falls, and
all three collapse into one expression by clipping the valuation date into the
period:

$$c_i = \min\bigl(\max(t_0, T_i^s),\, T_i^e\bigr),$$

making the coupon $A_i\,P_f(0,c_i)/P_f(0,T_i^e) - 1$ throughout, with $A_i$ the
realised compounding factor over $[T_i^s, c_i)$. A future period has $A_i = 1$
and is pure projection; a settled one has $c_i = T_i^e$, a projection ratio of
exactly `1.0`, and pays the realised $A_i - 1$. The same cut-over drives both
the realised factor and the projection, so the day $t_0$ itself is never
double-counted.

```python
import equinox as eqx
import jax.numpy as jnp

from valax.curves.discount import DiscountCurve
from valax.curves.fixings import FixingHistory, FixingSeries
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.legs import FixedLeg, FloatingLeg, MultiLegSwap
from valax.pricing.analytic.legs import leg_pv, swap_fair_rate, swap_pv

ref = ymd_to_ordinal(2026, 1, 1)


def curve(short, slope):
    pillars = jnp.asarray(
        [int(ref) - 365 + 365 * k for k in range(13)], dtype=jnp.int32
    )
    t = (pillars - (int(ref) - 365)).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-(short + slope * t) * t),
        reference_date=jnp.asarray(int(ref) - 365, dtype=jnp.int32),
        day_count="act_365",
    )


graph = CurveGraph(curves={
    "USD.OIS": curve(0.030, 0.0040),
    "USD.SOFR": curve(0.030, 0.0040),
})

# A 5-year annual grid that started 200 days ago, so the first period is
# still accruing at the valuation date.
start = int(ref) - 200
ends = jnp.asarray([start + 365 * k for k in range(1, 6)], dtype=jnp.int32)
starts = jnp.concatenate([jnp.asarray([start], dtype=jnp.int32), ends[:-1]])
notional = jnp.asarray(1.0e8)
steps = jnp.asarray([1.0, 0.8, 0.6, 0.4, 0.2])

# Pay an amortising fixed leg, receive compounded SOFR on the same profile.
fixed = FixedLeg(
    accrual_start_dates=starts, accrual_end_dates=ends, payment_dates=ends,
    fixed_rate=jnp.asarray(0.0325), notional=notional, amortisation=steps,
    pay=True, day_count="act_360", discount_id="USD.OIS",
)
floating = FloatingLeg(
    accrual_start_dates=starts, accrual_end_dates=ends, payment_dates=ends,
    spread=jnp.asarray(0.0), notional=notional, amortisation=steps,
    compounding="compounded", day_count="act_360",
    index_id="USD.SOFR", discount_id="USD.OIS", forward_id="USD.SOFR",
)
swap = MultiLegSwap(legs=(fixed, floating))

# Realised overnight fixings for the part of period 1 already accrued.
days = jnp.arange(start, int(ref), dtype=jnp.int32)
history = FixingHistory(indices={
    "USD.SOFR": FixingSeries(
        fixing_dates=days, fixings=jnp.full(days.shape, 0.0318)
    )
})

print("curves touched :", swap.curves_touched)
print(f"fixed leg PV   : {float(leg_pv(fixed, graph, history, ref)):>16,.2f}")
print(f"floating leg PV: {float(leg_pv(floating, graph, history, ref)):>16,.2f}")
print(f"swap PV        : {float(swap_pv(swap, graph, history, ref)):>16,.2f}")
print(f"par fixed rate : {float(swap_fair_rate(swap, graph, history, ref)):>16.6%}")
print(f"PV ignoring fixings: {float(swap_pv(swap, graph, None, ref)):>11,.2f}")


# The whole trade is one jit-able, differentiable function.
@eqx.filter_jit
def priced(trade, g, h):
    return swap_pv(trade, g, h, ref)


print(f"under filter_jit   : {float(priced(swap, graph, history)):>11,.2f}")
sens = eqx.filter_grad(lambda t: swap_pv(t, graph, history, ref))(swap)
print(f"d(PV)/d(fixed rate): {float(sens.legs[0].fixed_rate):>11,.2f}")
```

```
curves touched : ('USD.OIS', 'USD.SOFR')
fixed leg PV   :    -8,772,264.01
floating leg PV:    12,900,737.02
swap PV        :     4,128,473.01
par fixed rate :        4.779541%
PV ignoring fixings:  2,405,767.00
under filter_jit   :  4,128,473.01
d(PV)/d(fixed rate): -269,915,815.82
```

Note the 1.72mm gap between the seasoned and unseasoned PVs: that is the part of
the first coupon already earned at 3.18%, which a pricer with no fixing history
has no way to represent.

### Per-leg index descriptors

The reason the index lives on the *leg* rather than the trade is that a singular
`forward_id` can only express a single-index trade. A single-currency tenor
basis swap needs **three** curves — OIS discounting plus two projections — and a
CMS-spread leg structure three as well. With per-leg descriptors that is just
two legs with different `forward_id`s:

```python
basis = MultiLegSwap(legs=(
    FloatingLeg(
        accrual_start_dates=starts, accrual_end_dates=ends, payment_dates=ends,
        spread=jnp.asarray(0.0), notional=notional, compounding="compounded",
        day_count="act_360", index_id="USD.SOFR",
        discount_id="USD.OIS", forward_id="USD.SOFR",
    ),
    FloatingLeg(
        accrual_start_dates=starts, accrual_end_dates=ends, payment_dates=ends,
        spread=jnp.asarray(0.0), notional=notional, compounding="simple",
        fixing_dates=starts, day_count="act_360", index_id="USD.LIBOR.3M",
        discount_id="USD.OIS", forward_id="USD.LIBOR.3M", pay=True,
    ),
))
print(basis.curves_touched)   # ('USD.OIS', 'USD.SOFR', 'USD.LIBOR.3M')
print(basis.indices_touched)  # ('USD.SOFR', 'USD.LIBOR.3M')
```

`curves_touched` and `indices_touched` de-duplicate in order of first
appearance, so they are deterministic rather than hash-dependent — which is what
lets a caller assemble exactly the curves a trade needs before pricing it.

### Legs through `price()`

`price(swap, None, market)` routes a `MultiLegSwap` to `swap_pv` with every
curve its legs name (`ResolvedMarket.curves`, keyed by id —
[D16](../architecture/calibration-pricing-risk.md#3-decisions)) and the
market's fixing history, valued at the discount curve's reference date. Which
curves is fixed by the legs' static ids, so the same call works with the trade
as a traced `jit` argument, its curve risk lands on exactly the curves it
names, and trades sharing ids batch under `vmap`:

```python
import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from valax.core import MarketState, price
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.legs import FloatingLeg, MultiLegSwap

ref = int(ymd_to_ordinal(2026, 1, 1))


def curve(short, slope):
    pillars = jnp.array([ref + 365 * k for k in range(8)], dtype=jnp.int32)
    t = (pillars - ref) / 365.0
    return DiscountCurve(pillar_dates=pillars,
                         discount_factors=jnp.exp(-(short + slope * t) * t),
                         reference_date=jnp.int32(ref), day_count="act_365")


market = MarketState(curves=CurveGraph(curves={
    "USD.SOFR.OIS": curve(0.030, 0.002),
    "USD.SOFR.3M": curve(0.033, 0.002),
    "USD.SOFR.6M": curve(0.035, 0.002),
}))
starts = jnp.array([ref + 2 + 182 * k for k in range(10)], dtype=jnp.int32)
ends = starts + 182


def leg(forward_id, **kw):
    return FloatingLeg(accrual_start_dates=starts, accrual_end_dates=ends,
                       payment_dates=ends, notional=jnp.asarray(1e8),
                       compounding="simple", fixing_dates=starts,
                       index_id=forward_id, discount_id="USD.SOFR.OIS",
                       forward_id=forward_id, **kw)


def basis(spread):
    """Receive 3M flat, pay 6M + spread: three curves."""
    return MultiLegSwap(legs=(leg("USD.SOFR.3M", spread=jnp.asarray(0.0)),
                              leg("USD.SOFR.6M", spread=spread, pay=True)))


swap = basis(jnp.asarray(-0.0010))
value = eqx.filter_jit(lambda s, m: price(s, None, m).price)
print(f"basis swap PV : {float(value(swap, market)):,.2f}")
risk = eqx.filter_grad(lambda m: value(swap, m))(market).curves.curves
print({cid: float(jnp.sum(c.discount_factors)) for cid, c in risk.items()})
pvs = eqx.filter_vmap(lambda s: value(basis(s), market))(
    jnp.array([-0.002, -0.001, 0.0]))
print(pvs)                      # one compiled program, three spreads
```

A seasoned leg reads its realised fixings from `market.fixings` by its
`index_id`; the index id never selects a curve.

!!! note "What the leg engine does not do yet"
    `MultiLegSwap` adds its legs' present values directly, so **every leg must
    discount in the same currency**. A cross-currency swap additionally needs an
    FX rate and notional exchange; no `currency` field is carried rather than
    carrying one nothing reads. Payment lags must also still be supplied as
    explicit `payment_dates` — see
    [Dates and Conventions](dates-and-conventions.md#business-day-calendars-and-adjustment)
    for rolling a lag onto business days.

    A leg carries **no calendar**, so `day_count="bus_252"` is refused at
    construction rather than given a default calendar the caller never chose.
    Brazilian Bus/252 legs therefore remain out of reach until a leg can name
    a calendar.

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
