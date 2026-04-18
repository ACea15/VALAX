# Yield Curves and Bootstrapping

The yield curve is the foundation of every fixed-income, rates, and FX product
in VALAX. This guide covers how curves are represented, how to **bootstrap** them
from market quotes (deposits, FRAs, par swaps), and how to build a **multi-curve
set** (OIS discount + tenor-specific forward curves).

If you want the mathematical framework — discount factors vs. zero rates vs.
forward rates, single- vs. multi-curve, interpolation methods, and day count
conventions — see [Models & Theory §3](../theory.md#3-curve-framework). For using
a curve to price a bond and extract duration / KRDs, see the
[Fixed Income guide](fixed-income.md).

## 1. The `DiscountCurve` pytree

The base representation is a frozen `equinox.Module`:

```python
import jax.numpy as jnp
from valax.dates import ymd_to_ordinal
from valax.curves import DiscountCurve

ref = ymd_to_ordinal(2025, 1, 1)

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

# Curve is callable — interpolates at any date.
df_5y = curve(ymd_to_ordinal(2030, 1, 1))
```

Three things to know:

1. **Interpolation is log-linear on discount factors.** This gives piecewise-constant continuously-compounded forward rates between pillars. Simple, monotone, stable.
2. **Extrapolation is flat** beyond the pillar range.
3. **The curve is a JAX pytree with `discount_factors` as a dynamic leaf.** That means `jax.grad` of any price function that takes this curve will produce one sensitivity per pillar — key-rate durations for free.

### Deriving zero and forward rates

```python
from valax.curves import zero_rate, forward_rate

z_5y = zero_rate(curve, ymd_to_ordinal(2030, 1, 1))

f_1y2y = forward_rate(
    curve,
    start=ymd_to_ordinal(2026, 1, 1),
    end=ymd_to_ordinal(2027, 1, 1),
)
```

| Quantity | Formula | VALAX helper |
|----------|---------|--------------|
| Discount factor $DF(T)$ | curve definition | `curve(date)` |
| Continuously-compounded zero rate | $r(T) = -\ln DF(T) / \tau$ | `zero_rate(curve, date)` |
| Simply-compounded forward rate | $F(T_1, T_2) = (DF(T_1)/DF(T_2) - 1) / \tau$ | `forward_rate(curve, start, end)` |

## 2. Bootstrap instruments

Market quotes are represented as pytrees in `valax/curves/instruments.py`.
These are **inputs to bootstrapping** — distinct from the tradeable instruments
in `valax/instruments/rates.py`, which carry notional and direction.

```python
from valax.curves.instruments import DepositRate, FRA, SwapRate
from valax.dates import ymd_to_ordinal, generate_schedule
import jax.numpy as jnp

ref = ymd_to_ordinal(2025, 1, 1)

# Short end: money-market deposits.
depo_3m = DepositRate(
    start_date=ref,
    end_date=ymd_to_ordinal(2025, 4, 1),
    rate=jnp.array(0.0435),      # 4.35% simple
    day_count="act_360",
)
depo_6m = DepositRate(
    start_date=ref,
    end_date=ymd_to_ordinal(2025, 7, 1),
    rate=jnp.array(0.0425),
    day_count="act_360",
)

# Mid: a forward rate agreement.
fra_6x9 = FRA(
    start_date=ymd_to_ordinal(2025, 7, 1),
    end_date=ymd_to_ordinal(2025, 10, 1),
    rate=jnp.array(0.0410),
    day_count="act_360",
)

# Long end: par swap rate.
swap_5y = SwapRate(
    start_date=ref,
    fixed_dates=generate_schedule(2025, 7, 1, 2030, 1, 1, frequency=2),
    rate=jnp.array(0.0385),      # 3.85% par
    day_count="act_360",
)
```

Each one encodes a single no-arbitrage equation on the curve:

| Instrument | Equation |
|------------|----------|
| `DepositRate` | $DF(\text{end}) = DF(\text{start}) / (1 + r \tau)$ |
| `FRA` | $DF(\text{end}) = DF(\text{start}) / (1 + r \tau)$ (with $\text{start} > t_0$) |
| `SwapRate` | $r \sum_i \tau_i\,DF(T_i) = DF(\text{start}) - DF(T_n)$ |

## 3. Sequential bootstrap

For non-overlapping instruments sorted by maturity, each instrument adds one
new pillar and the discount factor at that pillar can be solved **analytically**
from the others. This is `bootstrap_sequential`:

```python
from valax.curves.bootstrap import bootstrap_sequential

curve = bootstrap_sequential(
    reference_date=ref,
    instruments=[depo_3m, depo_6m, fra_6x9, swap_5y],
    day_count="act_365",
)
```

The returned `DiscountCurve` has one pillar per input instrument (plus
`DF = 1` at the reference date). It is JAX-traceable end-to-end — gradients
flow through the analytic solve for each pillar.

**When sequential is sufficient:**

- Deposits at the short end (always sequential — they just set an anchor $DF$).
- FRAs that tile the curve without overlap.
- A single swap strip where each swap adds exactly one new maturity pillar.

**When sequential breaks down:**

- Two swaps share intermediate coupon dates (e.g., a 2Y and a 3Y swap both have a 1Y and 18M coupon). The sequential formula needs the DF at each intermediate date, which becomes ambiguous.
- Basis swaps or overlapping FRAs.
- Any time you want to move pillars around independently of the instrument set.

For those cases, use the simultaneous bootstrap.

## 4. Simultaneous bootstrap (Newton solve)

`bootstrap_simultaneous` solves for **all** pillar discount factors at once by
minimising the vector of repricing residuals to zero:

$$
\mathbf{R}(\mathbf{x}) = \mathbf{0}, \qquad \mathbf{x} = (\ln DF_1, \ldots, \ln DF_N)
$$

where each $R_i$ is instrument $i$'s repricing error. VALAX uses
`optimistix.Newton` working in **log-DF space** — this guarantees positive
discount factors at every Newton step and matches the log-linear interpolation
inside `DiscountCurve`.

```python
from valax.curves.bootstrap import bootstrap_simultaneous
import jax.numpy as jnp

pillar_dates = jnp.array([
    int(ymd_to_ordinal(2025, 4, 1)),
    int(ymd_to_ordinal(2025, 7, 1)),
    int(ymd_to_ordinal(2025, 10, 1)),
    int(ymd_to_ordinal(2030, 1, 1)),
], dtype=jnp.int32)

curve = bootstrap_simultaneous(
    reference_date=ref,
    pillar_dates=pillar_dates,
    instruments=[depo_3m, depo_6m, fra_6x9, swap_5y],
    day_count="act_365",
    # Optional: custom solver and initial guess.
    # solver=optx.Newton(rtol=1e-10, atol=1e-10),
    # initial_guess=jnp.zeros_like(pillar_dates, dtype=jnp.float64),
    max_steps=256,
)
```

The constraint is that the instrument set must be **square** — exactly one
instrument per pillar — so the system is well-determined.

### The Jacobian is exact

Newton's method needs $\partial R_i / \partial x_j$. VALAX gets this from
`jax.jacobian` — the Jacobian is the exact analytic derivative, not a finite
difference. This is faster and more numerically stable than production
bootstrappers that use numerical Jacobians, and it composes naturally:
differentiating **through** `bootstrap_simultaneous` w.r.t. the input quotes
gives the Jacobian of the resulting discount factors w.r.t. the market quotes
(via `optimistix`'s `ImplicitAdjoint`).

```python
import jax

def curve_dfs_from_quotes(depo_rate_1, depo_rate_2, fra_rate, swap_rate):
    d1 = DepositRate(ref, ymd_to_ordinal(2025, 4, 1), depo_rate_1, "act_360")
    d2 = DepositRate(ref, ymd_to_ordinal(2025, 7, 1), depo_rate_2, "act_360")
    f = FRA(ymd_to_ordinal(2025, 7, 1), ymd_to_ordinal(2025, 10, 1), fra_rate, "act_360")
    s = SwapRate(ref, generate_schedule(2025, 7, 1, 2030, 1, 1, frequency=2),
                 swap_rate, "act_360")
    curve = bootstrap_simultaneous(ref, pillar_dates, [d1, d2, f, s], "act_365")
    return curve.discount_factors

# Jacobian: ∂DF_i / ∂quote_j.
J = jax.jacobian(curve_dfs_from_quotes, argnums=(0, 1, 2, 3))(
    jnp.array(0.0435), jnp.array(0.0425), jnp.array(0.0410), jnp.array(0.0385),
)
```

Each column of `J` is the sensitivity vector of the whole curve to a 1-unit
move in one input quote. This is the matrix a trading desk uses to hedge curve
risk with liquid instruments — normally an expensive numerical object, free
here.

### Sequential vs. simultaneous: when to use which

| Situation | Preferred method |
|-----------|------------------|
| Simple strip of non-overlapping instruments | `bootstrap_sequential` — faster, trivially debuggable |
| Overlapping swaps, basis pillars decoupled from instrument schedule | `bootstrap_simultaneous` |
| Need to differentiate **through** the bootstrap | Either works; simultaneous is smoother because `ImplicitAdjoint` gives exact gradients without unrolling the Newton iteration |
| Want pillars at specific dates (e.g., 1Y, 2Y, 5Y, 10Y) different from instrument maturities | `bootstrap_simultaneous` |

## 5. Multi-curve bootstrap

Post-2008, single-curve discounting is dead. A EUR/USD fixed-income desk now
runs:

- **OIS discount curve** (SOFR for USD, €STR for EUR) — used to discount every cashflow.
- **Tenor-specific forward curves** (3M SOFR, 6M EURIBOR, ...) — used to project forward rates for floating coupons.

Under a CSA with daily collateral, the OIS rate is the funding rate for
discounted cashflows, so OIS discounting is the **arbitrage-free** choice.
See [theory §3.2](../theory.md#32-single-curve-vs-multi-curve-framework).

VALAX's `MultiCurveSet` holds one discount curve plus a dict of forward
curves keyed by tenor label:

```python
from valax.curves.multi_curve import bootstrap_multi_curve

# OIS short end + OIS par swaps.
ois_instruments = [
    DepositRate(ref, ymd_to_ordinal(2025, 2, 1), jnp.array(0.0415), "act_360"),
    DepositRate(ref, ymd_to_ordinal(2025, 4, 1), jnp.array(0.0410), "act_360"),
    SwapRate(ref, generate_schedule(2026, 1, 1, 2030, 1, 1, frequency=4),
             jnp.array(0.0385), "act_360"),
    SwapRate(ref, generate_schedule(2026, 1, 1, 2035, 1, 1, frequency=4),
             jnp.array(0.0395), "act_360"),
]

# 3M SOFR forward curve instruments.
sofr3m_instruments = [
    DepositRate(ref, ymd_to_ordinal(2025, 4, 1), jnp.array(0.0440), "act_360"),
    FRA(ymd_to_ordinal(2025, 4, 1), ymd_to_ordinal(2025, 7, 1),
        jnp.array(0.0435), "act_360"),
    SwapRate(ref, generate_schedule(2026, 1, 1, 2030, 1, 1, frequency=4),
             jnp.array(0.0405), "act_360"),  # 3M SOFR swap par
]

# Optionally: 6M EURIBOR forward curve instruments.
# euribor6m_instruments = [...]

multi = bootstrap_multi_curve(
    reference_date=ref,
    discount_instruments=ois_instruments,
    forward_instruments={
        "3M": sofr3m_instruments,
        # "6M": euribor6m_instruments,
    },
    day_count="act_365",
)

# Use it:
ois = multi.discount_curve
fwd_3m = multi.forward_curves["3M"]

# Price a 3M SOFR swap: project forwards with fwd_3m, discount with ois.
```

### What the dual-curve bootstrap does

For each forward curve:

1. **Deposits and FRAs** are bootstrapped sequentially (their DF formula is independent of the discount curve).
2. **Swaps** are bootstrapped via a simultaneous Newton solve, where the par swap condition is the **dual-curve** version:

$$
\sum_{i=1}^{n} \left(\frac{DF_{\text{fwd}}(T_{i-1})}{DF_{\text{fwd}}(T_i)} - 1\right) DF_{\text{ois}}(T_i) \;=\; r_{\text{swap}} \sum_{i=1}^{n} \tau_i\,DF_{\text{ois}}(T_i)
$$

Forward rates come from the forward curve; discounting comes from the OIS
curve. The single-curve identity $PV_{\text{float}} = DF(\text{start}) - DF(\text{end})$
**does not hold** here — forward projection and discounting are no longer the
same curve.

## 6. Autodiff: key-rate durations on the bootstrapped curve

Because `DiscountCurve.discount_factors` is a JAX leaf, **any** pricing function
that takes a curve automatically supports per-pillar sensitivities. Combined
with `bootstrap_*`, the full workflow is differentiable from market quotes to
bond price:

```python
import jax
import equinox as eqx
from valax.instruments import FixedRateBond
from valax.pricing.analytic.bonds import fixed_rate_bond_price

def bond_price_from_quotes(depo_rate, swap_rate):
    """Build a curve from two quotes, then price a bond."""
    d = DepositRate(ref, ymd_to_ordinal(2025, 4, 1), depo_rate, "act_360")
    s = SwapRate(ref, generate_schedule(2025, 7, 1, 2030, 1, 1, frequency=2),
                 swap_rate, "act_360")
    curve = bootstrap_sequential(ref, [d, s], "act_365")
    bond = FixedRateBond(
        payment_dates=generate_schedule(2025, 7, 1, 2030, 1, 1, frequency=2),
        settlement_date=ref,
        coupon_rate=jnp.array(0.04),
        face_value=jnp.array(100.0),
        frequency=2,
        day_count="act_365",
    )
    return fixed_rate_bond_price(bond, curve)

# DV01 w.r.t. each input quote — one reverse-mode pass.
dprice_dquotes = jax.grad(bond_price_from_quotes, argnums=(0, 1))(
    jnp.array(0.0435), jnp.array(0.0385),
)
```

The same idea scales: for any set of $N$ market quotes and any pricing function,
one `jax.grad` call returns the full vector of quote sensitivities. In a
traditional bump-and-reprice system this is $2N$ bootstraps + $2N$ repricings.
Here it is one bootstrap + one reverse-mode pass.

You can also take sensitivities directly through the `discount_factors` of the
output curve (key-rate DV01 on the curve pillars rather than on the input
quotes):

```python
def price_from_dfs(dfs):
    new_curve = eqx.tree_at(lambda c: c.discount_factors, curve, dfs)
    return fixed_rate_bond_price(bond, new_curve)

krds = jax.grad(price_from_dfs)(curve.discount_factors)
# one entry per pillar; sum ≈ modified duration (up to sign/convention).
```

## 7. Inflation curves

Inflation uses a parallel type, `InflationCurve` (`valax/curves/inflation.py`),
storing **forward CPI levels** interpolated in log-CPI space. The API mirrors
`DiscountCurve`:

```python
from valax.curves.inflation import from_zc_rates, forward_cpi, yoy_forward_rate

infl = from_zc_rates(
    reference_date=ref,
    pillar_dates=jnp.array([
        int(ymd_to_ordinal(2026, 1, 1)),
        int(ymd_to_ordinal(2028, 1, 1)),
        int(ymd_to_ordinal(2030, 1, 1)),
        int(ymd_to_ordinal(2035, 1, 1)),
    ], dtype=jnp.int32),
    zc_rates=jnp.array([0.025, 0.024, 0.024, 0.023]),
    base_cpi=jnp.array(311.2),
)

cpi_5y = forward_cpi(infl, ymd_to_ordinal(2030, 1, 1))
yoy = yoy_forward_rate(
    infl,
    start_dates=jnp.array([int(ymd_to_ordinal(2027, 1, 1))]),
    end_dates=jnp.array([int(ymd_to_ordinal(2028, 1, 1))]),
)
```

For the mathematical framework (forward CPI vs. real rate, ZCIS breakeven
identity, YoY convexity, seasonality) see
[theory §3.6](../theory.md#36-inflation-curves-and-breakeven-pricing). For
pricing inflation swaps and caps/floors against a built curve, see
[Inflation Derivatives](inflation.md).

## 8. What is not yet implemented

From the [roadmap (Tier 1.1)](../roadmap.md#11-multi-curve-bootstrapping):

- **Basis curves** (e.g., 1M vs. 3M SOFR basis) as first-class pytrees.
- **Turn-of-year jumps** and stub-period handling inside the bootstrap.
- **Alternative interpolation methods**: linear on zero rates, cubic splines on log-DF, monotone-convex, tension splines. The log-linear baseline is stable and monotone but produces discontinuous forward rates at pillars.
- **Constrained curves** (forward-rate positivity, calendar-spread no-arbitrage for inflation curves).

Contributions welcome — see `CONTRIBUTING.md` at the repository root.

## 9. Further reading

- [Models & Theory §3](../theory.md#3-curve-framework) — mathematical framework: DFs, zero/forward rates, single vs. multi-curve, interpolation families, day count conventions.
- [Fixed Income guide](fixed-income.md) — using a curve to price bonds, extract YTM, duration, KRDs.
- [Interest Rate Exotics guide](rates-exotics.md) — OIS swaps, XCCY, CMS, range accrual against a multi-curve set.
- [Inflation Derivatives guide](inflation.md) — ZCIS, YYIS, inflation caps against an `InflationCurve`.
- [Short-Rate Models guide](short-rate.md) — Hull-White workflow, which consumes a `DiscountCurve` for exact-fit $\theta(t)$.
