# Yield Curves and Bootstrapping

The yield curve is the foundation of every fixed-income, rates, and FX product
in VALAX. This guide covers how curves are represented, how to **bootstrap** them
from market quotes (deposits, FRAs, par swaps), and how to build a **multi-curve
set** (OIS discount + tenor-specific forward curves).

If you want the mathematical framework — discount factors vs. zero rates vs.
forward rates, single- vs. multi-curve, interpolation methods, and day count
conventions — see [Models & Theory §3](../theory/curves.md#3-curve-framework). For using
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
    fixed_dates=generate_schedule(2025, 7, 1, 2030, 1, 1, frequency=2).payment_dates,
    rate=jnp.array(0.0385),      # 3.85% par
    day_count="act_360",
)
```

Each one encodes a single no-arbitrage equation on the curve:

| Instrument | Equation |
|------------|----------|
| `DepositRate` | $DF(\text{end}) = DF(\text{start}) / (1 + r \tau)$ |
| `FRA` | $DF(\text{end}) = DF(\text{start}) / (1 + r \tau)$ (with $\text{start} > t_0$) |
| `SwapRate` | $r \sum_i \tau_i\,DF(T_i^p) = \sum_i DF(T_i^p)\left(DF(T_i^s)/DF(T_i^e) - 1\right)$ |

### Payment lag: accrual end vs payment date

Overnight-index swaps do not pay on the day the accrual period ends.  USD SOFR
OIS pays two business days later, EUR €STR and GBP SONIA one.  So each coupon
has *two* dates that matter — the accrual end $T^e_i$, which determines what
the coupon is worth, and the payment date $T^p_i$, which determines when you
receive it and therefore how it is discounted.

Every swap quote type keeps these in separate roles.  Projection and the
accrual fraction $\tau_i$ come from the accrual schedule; discounting comes
from the payment schedule:

$$
S^{\mathrm{OIS}}(\theta_d)
= \frac{\sum_i P_d(0,T_i^p)\left(\dfrac{P_d(0,T_i^s)}{P_d(0,T_i^e)} - 1\right)}
       {\sum_j \tau_j\, P_d(0,T_j^p)}
$$

There are two ways to say where the payments fall, and supplying both raises:

```python
from valax.curves.instruments import OISSwapRate
from valax.dates import generate_schedule, ymd_to_ordinal
import jax.numpy as jnp

ref = ymd_to_ordinal(2025, 1, 1)
accruals = generate_schedule(
    2025, 1, 1, 2030, 1, 1, frequency=1
).payment_dates

# (a) A uniform lag in *calendar* days — convenient, approximate.
ois_lagged = OISSwapRate(
    start_date=ref,
    fixed_dates=accruals,          # accrual ends
    rate=jnp.array(0.0385),
    day_count="act_360",
    payment_lag=2,
)

# (b) An explicit payment schedule you have already rolled over a
#     holiday calendar — the production route.
ois_exact = OISSwapRate(
    start_date=ref,
    fixed_dates=accruals,
    rate=jnp.array(0.0385),
    day_count="act_360",
    payment_dates=accruals + 2,    # substitute your own adjusted dates
)
```

`payment_lag` shifts by calendar days, which is only an approximation to a
business-day roll; VALAX has no holiday calendars (see
[§8](#8-what-is-not-yet-implemented)).  When the distinction matters, compute
the adjusted dates yourself and pass `payment_dates`.  Legs are resolved
independently, so `IborSwapRate` takes `fixed_payment_dates` and
`float_payment_dates`, `TenorBasisSwap` takes `leg_a_payment_dates` /
`leg_b_payment_dates`, and `CrossCurrencyBasisSwap` takes
`dom_payment_dates` / `for_payment_dates`.  Under the constant-notional
cross-currency variant the final notional re-exchange travels with the last
coupon, so it is discounted at the last *payment* date.

!!! note "Defaults are unchanged, and the telescope survives as a fast path"
    With neither field set, $T^p = T^e$ and the OIS floating leg collapses to
    $DF(T_0) - DF(T_n)$ by the telescoping identity.  That closed form is kept
    as an explicit branch, so every pre-existing curve reproduces bit for bit.
    The identity holds *only* at zero lag — which is the point of the
    conformance gate in
    `tests/test_curves/test_instrument_residuals.py::TestPaymentLagTelescoping`.

    A uniform lag always *lowers* the par rate, whatever the shape of the
    curve, and by exactly nothing on a flat one.  The weights in the par-rate
    average pick up a factor $P(0,T^p_i)/P(0,T^e_i) \approx e^{-f_i \ell}$,
    which is a decreasing function of that period's own forward, so lagging
    payment shifts weight away from the high-forward periods.  The size of the
    effect is driven by the *dispersion* of the forwards, not their level.

    What is **not** modelled is the timing convexity: the compounded rate is
    no longer a martingale under the $T^p$-forward measure, and no drift
    correction is applied.  This is the standard simplification, recorded in
    the [conformance map](../architecture/aad-audit-conformance.md#documented-choices)
    under C-7.

### Compounded vs averaged futures

Three-month RFR futures settle on the *compounded* overnight index.
One-month RFR futures settle on the *arithmetic average* of the daily index
values.  Writing $\rho_j = DF(t_j)/DF(t_{j+1})$ for the daily discount-factor
ratios, the two are a product and a sum of the same quantities:

$$
F = \frac{1}{\tau}\Big(\underbrace{\textstyle\prod_j \rho_j}_{\text{compounded}} - 1\Big),
\qquad
\bar{F} = \frac{1}{\tau}\sum_j \big(\rho_j - 1\big).
$$

Select the convention with the static `averaging` field, which defaults to
`"compounded"`:

```python
from valax.curves.instruments import MoneyMarketFuture
from valax.dates import ymd_to_ordinal
import jax.numpy as jnp

sofr_1m = MoneyMarketFuture(
    start_date=ymd_to_ordinal(2025, 3, 1),
    end_date=ymd_to_ordinal(2025, 4, 1),
    futures_rate=jnp.array(0.0432),
    day_count="act_360",
    averaging="arithmetic",      # 1M SOFR settles on the average
)
```

Two consequences worth internalising:

**The difference does not vanish on a flat curve.** The gap between a product
and a sum of the same ratios is the cross terms — the interest-on-interest
that averaging discards — of order $r^2\tau/2$. On a one-month contract at 4%
that is about 0.6 bp, and it is there even when every daily forward is
identical. This is a *compounding* effect, not a curve-shape effect, so
calibrating a 1M contract as though it were compounded mis-places the short
end by a real amount rather than by something that averages out.

**Only the averaged contract has exposure inside its own period.** The
compounded forward is a telescoping product, so it is sensitive to exactly two
discount factors — the endpoints — and interior pillars cancel identically.
The averaged forward is a sum, which does not telescope, so it picks up
genuine sensitivity to the *shape* of the curve between its start and end
dates.  (That shape exposure vanishes on a flat curve, where adjacent $\rho_j$
are equal and cancel pairwise.)  Both facts are gated in
`tests/test_curves/test_instrument_residuals.py::TestAveragedFutures`.

!!! note "Calendars are not needed here, but a concrete day span is"
    The market convention carries the preceding business day's rate across
    weekends and holidays *weighted by calendar days*, which is exactly what a
    calendar-daily subdivision of a smooth curve reproduces — so the absence
    of holiday calendars costs nothing in this particular calculation.

    What the averaged branch does need is a concrete number of days over
    `[start_date, end_date]`, since it reads one discount factor per day.  The
    span is read pre-trace; building the instrument inside a `jit` whose
    arguments include its dates raises a `ValueError` naming the way out
    rather than silently averaging over the wrong number of days.  Closing
    over the instrument and tracing only the curve — the normal pattern, since
    quotes are Python data — works fine.

    The convexity adjustment is a separate, orthogonal choice: an averaged
    contract has its own, different from the compounded one.
    `convexity_adj_fn` is the plug-in point, and the default for both is
    explicitly none.

### Partly-accrued periods

A coupon whose fixing date has passed uses the stored fixing rather than the
curve — that much is just a lookup, and `FixingHistory` has always done it.
But an *overnight-compounded* coupon whose period has begun and not yet ended
is neither realised nor forward-looking. Part of its compounding has happened
and the rest has not, and the two have to be combined:

$$
1 + \tau_j F_j \;=\; \underbrace{A}_{\text{realised}} \cdot
\underbrace{\frac{P_x(0,t_0)}{P_x(0,T_j^e)}}_{\text{projected remainder}}
$$

where $A = \prod_j (1 + \hat{r}_j \delta_j)$ is the realised compounding factor
over the elapsed part. This is exact rather than an approximation: the
compounding factor over the whole period *is* the product of the factor over
the realised part and the factor over the remainder.

Build $A$ from the recorded history and hand it to the leg:

```python
from valax.curves.fixings import FixingHistory, FixingSeries
from valax.curves.instruments import IborSwapRate
from valax.dates import ymd_to_ordinal
import jax.numpy as jnp

t0 = ymd_to_ordinal(2025, 6, 15)          # valuation date
accrual_start = ymd_to_ordinal(2025, 5, 1)  # period began six weeks ago

sofr = FixingSeries(
    fixing_dates=jnp.arange(accrual_start, t0),
    fixings=jnp.full(t0 - accrual_start, 0.0433),
)

realised = sofr.compounded_factor(accrual_start, t0, "act_360")

swap = IborSwapRate(
    start_date=accrual_start,
    fixed_dates=jnp.array([ymd_to_ordinal(2026, 5, 1)]),
    float_dates=jnp.array([ymd_to_ordinal(2025, 8, 1),
                           ymd_to_ordinal(2025, 11, 1),
                           ymd_to_ordinal(2026, 2, 1),
                           ymd_to_ordinal(2026, 5, 1)]),
    fixing_dates=jnp.array([accrual_start,
                            ymd_to_ordinal(2025, 8, 1),
                            ymd_to_ordinal(2025, 11, 1),
                            ymd_to_ordinal(2026, 2, 1)]),
    rate=jnp.array(0.0385),
    index_id="USD.SOFR.3M",
    accrued_compounding=realised,      # ← the blend
)
```

`TenorBasisSwap` takes `leg_a_accrued_compounding` /
`leg_b_accrued_compounding` and `CrossCurrencyBasisSwap` takes
`dom_accrued_compounding` / `for_accrued_compounding`, each resolved
independently per leg.

**Precedence.** Three sources, strictly ordered: a recorded fixing for the
coupon wins, then the blend for the straddling period, then curve projection.
A recorded value outranks a partial accrual because it is strictly more
information — for a tenor index such as `USD.SOFR.3M` the recorded value *is*
the compounded coupon rate for the whole period.

**Which period straddles.** The test is half-open, $T^s_j \le t_0 < T^e_j$,
which makes the straddling period unique. Closing both ends would match two
periods whenever $t_0$ lands exactly on an accrual boundary — the one ending
there and the one starting there — and they cannot share one realised factor.
A consequence worth knowing: a period that has *fully* ended is deliberately
outside the blend's scope, because its rate is entirely known. If a payment
lag leaves such a period ended but unpaid, record its compounded coupon rate
under `index_id` rather than reaching for `accrued_compounding`.

!!! note "Two degeneracies and one consistency check"
    The formula is pinned at both ends and in the middle, in
    `tests/test_curves/test_instrument_residuals.py::TestPartlyAccruedFixings`:

    - At $t_0 = T^s$ nothing has accrued, $A = 1$, and the blend reduces to
      the fully projected forward — **exactly**, to the last bit.
    - As $t_0 \to T^e$ the projected remainder shrinks and the coupon tends to
      the purely realised $(A-1)/\tau$.
    - If the realised rates happen to be exactly what the curve had projected
      for the elapsed part — $A = P_x(T^s)/P_x(t_0)$ — then substituting
      realised data for projected data cannot change the answer, and it
      doesn't, again exactly. This is the property that would break first if
      the blend read the curve at the wrong date.

    Because $A$ is computed from recorded values only, with no curve input, a
    realised coupon carries **identically zero** projection-curve sensitivity
    while still being discounted. That is the
    [T-11 gate](../architecture/aad-audit-conformance.md#t-verification-tests).

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
    s = SwapRate(ref, generate_schedule(2025, 7, 1, 2030, 1, 1, frequency=2).payment_dates,
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

> **Why this section exists.** For the end-to-end narrative of *why*
> VALAX ships a whole multi-curve solver instead of a single discount
> curve — what breaks in 2008, which instruments couple which curves,
> and why sequential bootstraps cannot represent tenor-basis or
> cross-currency constraints — see
> [`why-multicurve.md`](why-multicurve.md). This section covers the
> API and a worked example.

Post-2008, single-curve discounting is dead. A EUR/USD fixed-income desk now
runs:

- **OIS discount curves** (SOFR for USD, €STR for EUR) — used to discount every
  cashflow under CSA.
- **Tenor-specific forward curves** (3M SOFR, 6M EURIBOR, ...) — used to project
  forward rates for floating coupons.
- **Cross-currency links** (FX forwards, CCBS) — tying the two currencies
  together.

Under a CSA with daily collateral, the OIS rate is the funding rate for
discounted cashflows, so OIS discounting is the **arbitrage-free** choice.
See [theory §3.2](../theory/curves.md#32-single-curve-vs-multi-curve-framework).

The MC-Curves-2 API is
[`bootstrap_curve_graph`](../api/curves.md#valax.curves.bootstrap_curve_graph): a **joint
Newton solve** over the concatenated log-DFs of every curve in the graph.
This replaces the older sequential `bootstrap_multi_curve` (see [§5.4
Deprecated path](#54-deprecated-path-bootstrap_multi_curve) below) — and
crucially handles cases the sequential pipeline **could not**: tenor-basis
swaps (3M-vs-6M), cross-currency basis swaps, FX forwards on the short end,
and futures convexity in one go.

### 5.1 Curve-id alphabet

Every curve in the graph has a string identifier following the frozen
alphabet (see [`production.md` §11.3](../architecture/production.md#113-curve-graph-data-model)):

```
<CCY>.<INDEX>.<TENOR>[.<QUALIFIER>]
```

- `CCY` — ISO-4217 currency code (`USD`, `EUR`, `GBP`, `JPY`, ...).
- `INDEX` — reference index name (`SOFR`, `ESTR`, `EURIBOR`, `SONIA`, ...).
- `TENOR` — either the literal `OIS` (discount role) or a numeric tenor
  label (`1M`, `3M`, `6M`, `12M`, `1Y`, ...).
- `QUALIFIER` — optional (`FIXINGS`, `CLEAN`, ...).

`CurveSpec` validates identifiers against this pattern on construction.

### 5.2 A worked USD dual-curve build

This example builds a small but complete USD dual-curve (OIS + 3M SOFR)
system, inspects the extended diagnostics, and prices a 2-year IRS
by hand from the calibrated graph.  The whole block is copy-paste
runnable — the fixture mirrors
`tests/test_curves/test_bootstrap_graph.py::TestDualCurve` verbatim.

```python
import jax.numpy as jnp
from valax.curves import (
    CurveSpec, DepositRate, IborSwapRate,
    bootstrap_curve_graph, empty_fixing_history,
)
from valax.dates.daycounts import ymd_to_ordinal, year_fraction

# ------------------------------------------------------------------
# 0. Reference date and a helper for quarterly SOFR-style schedules.
# ------------------------------------------------------------------
REF = ymd_to_ordinal(2025, 1, 1)

def _date(y, m, d):
    return int(ymd_to_ordinal(y, m, d))

def quarterly_dates_to(end_ordinal: int) -> jnp.ndarray:
    """1-Jan / 1-Apr / 1-Jul / 1-Oct dates strictly after REF, up to end."""
    dates = []
    for year in range(2025, 2036):
        for month in (1, 4, 7, 10):
            d = _date(year, month, 1)
            if d > int(REF) and d <= end_ordinal:
                dates.append(d)
    return jnp.array(dates, dtype=jnp.int32)

# ------------------------------------------------------------------
# 1. Declare the two curves we want to build.
#    Same pillar grid on both sides here — that isn't required in
#    general, only for the illustration.
# ------------------------------------------------------------------
pillars = jnp.array(
    [_date(2026, 1, 1), _date(2027, 1, 1)], dtype=jnp.int32,
)
ois_spec = CurveSpec(
    curve_id="USD.SOFR.OIS", currency="USD",
    pillar_dates=pillars, day_count="act_365",
)
fwd_spec = CurveSpec(
    curve_id="USD.SOFR.3M", currency="USD",
    pillar_dates=pillars, day_count="act_365",
)

# ------------------------------------------------------------------
# 2. OIS side — two deposits anchoring the discount curve.
# ------------------------------------------------------------------
ois_deps = [
    DepositRate(
        start_date=REF, end_date=_date(2026, 1, 1),
        rate=jnp.array(0.040), day_count="act_365",
        curves_touched=("USD.SOFR.OIS",),
    ),
    DepositRate(
        start_date=REF, end_date=_date(2027, 1, 1),
        rate=jnp.array(0.042), day_count="act_365",
        curves_touched=("USD.SOFR.OIS",),
    ),
]

# ------------------------------------------------------------------
# 3. 3M-projection side — two IBOR par swaps.  Each has a *dual-curve*
#    residual (OIS discount + 3M forward projection) that couples both
#    curves.  The joint solver handles that natively.
#
#    Fixing convention here: fixing_dates[k] = accrual_start[k]
#    (start of the k-th accrual period).
# ------------------------------------------------------------------
float_1y = quarterly_dates_to(_date(2026, 1, 1))
fix_1y = jnp.concatenate([jnp.array([REF], dtype=jnp.int32), float_1y[:-1]])
ibor_1y = IborSwapRate(
    start_date=REF,
    fixed_dates=jnp.array([_date(2026, 1, 1)], dtype=jnp.int32),
    float_dates=float_1y,
    fixing_dates=fix_1y,
    rate=jnp.array(0.045),
    fixed_day_count="act_365", float_day_count="act_365",
    curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
    index_id="USD.SOFR.3M",
)
float_2y = quarterly_dates_to(_date(2027, 1, 1))
fix_2y = jnp.concatenate([jnp.array([REF], dtype=jnp.int32), float_2y[:-1]])
ibor_2y = IborSwapRate(
    start_date=REF,
    fixed_dates=jnp.array(
        [_date(2026, 1, 1), _date(2027, 1, 1)], dtype=jnp.int32,
    ),
    float_dates=float_2y,
    fixing_dates=fix_2y,
    rate=jnp.array(0.047),
    fixed_day_count="act_365", float_day_count="act_365",
    curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
    index_id="USD.SOFR.3M",
)

# ------------------------------------------------------------------
# 4. One joint Newton solve — every residual zeroed simultaneously.
# ------------------------------------------------------------------
graph, diag = bootstrap_curve_graph(
    reference_date=REF,
    curve_specs=[ois_spec, fwd_spec],
    instruments=ois_deps + [ibor_1y, ibor_2y],
)

assert diag.converged
assert float(diag.max_abs_residual) < 1e-10

ois_curve = graph["USD.SOFR.OIS"]
fwd_curve = graph["USD.SOFR.3M"]

# ------------------------------------------------------------------
# 5. Inspect the extended CurveBuildDiagnostics — the full 8 fields.
# ------------------------------------------------------------------
print(f"converged                    = {diag.converged}")
print(f"n_steps                      = {int(diag.n_steps)}")
print(f"max_abs_residual             = {float(diag.max_abs_residual):.2e}")
print(f"rmse                         = {float(diag.rmse):.2e}")
print(f"jacobian_condition_number    = {float(diag.jacobian_condition_number):.4f}")
print("per-instrument fit (fitted − quoted, in native quote units):")
for i, (q, f) in enumerate(zip(diag.quoted_values, diag.fitted_quotes)):
    print(f"  instrument {i}: quoted={float(q):.6f}  fitted={float(f):.6f}"
          f"  err={float(f - q):+.2e}")
```

`fitted_quotes[i] − quoted_values[i]` is exactly the per-instrument
fit error in quote-native units (rate points here; basis points for
`TenorBasisSwap` and `CrossCurrencyBasisSwap`; FX units for
`FXForward`). This holds *exactly* — not just to first order —
because each VALAX residual is linear in its primary quote scalar,
so the Newton correction
`fitted = quoted − residual / (∂residual/∂quote)` is exact. That
derivation is what the `_fitted_quote` helper in
`valax/curves/bootstrap_graph.py` computes via `jax.grad` on each
instrument's own residual.

```python
# ------------------------------------------------------------------
# 6. Price a 2Y IRS *from the calibrated graph* with the dual-curve
#    pricer.
#
# The analytic rates pricers accept an optional ``forward_curve``
# argument, and each has a ``*_from_graph`` wrapper that resolves
# curve ids on the graph and delegates.  Under the hood this is the
# dual-curve par formula:
#
#     PV_fixed  = K · Σ_i τ^fixed_i · DF_OIS(T^fixed_i)
#     PV_float  = Σ_j (DF_3M(T_{j-1}) / DF_3M(T_j) − 1) · DF_OIS(T^float_j)
#     PV_swap   = PV_float − PV_fixed        (receive-float, pay-fixed)
#
# ------------------------------------------------------------------
import equinox as eqx
from valax.instruments.rates import OISSwap
from valax.pricing.analytic import ois_swap_price_from_graph

notional = 10_000_000.0             # USD 10 mm
fixed_rate = 0.045                  # off-market (par is 0.047 by the 2Y quote)

# Despite its name, OISSwap is VALAX's generic dual-schedule
# fixed-vs-float pytree — the only swap instrument carrying distinct
# fixed and float schedules (annual fixed here, quarterly float).
swap = OISSwap(
    start_date=REF,
    fixed_dates=ibor_2y.fixed_dates,     # annual
    float_dates=ibor_2y.float_dates,     # quarterly
    fixed_rate=jnp.array(fixed_rate),
    notional=jnp.array(notional),
    pay_fixed=True,                      # receive-float, pay-fixed
    day_count="act_365",
)

pv_swap = ois_swap_price_from_graph(
    swap, graph,
    discount_id="USD.SOFR.OIS",          # discounting: OIS
    forward_id="USD.SOFR.3M",            # projection: 3M forwards
)
print(f"\n2Y IRS PV (receive 3M SOFR, pay {fixed_rate:.3%}):")
print(f"  PV_swap      = {float(pv_swap):+,.2f} USD")

# Equivalent long-hand form, without the graph wrapper:
#   ois_swap_price(swap, graph["USD.SOFR.OIS"],
#                  forward_curve=graph["USD.SOFR.3M"])

# ------------------------------------------------------------------
# 7. Sanity cross-check: setting fixed_rate to the calibrated par rate
#    should give PV ≈ 0 (par swap).  This is the definition of par.
# ------------------------------------------------------------------
par_swap = eqx.tree_at(lambda s: s.fixed_rate, swap, jnp.array(0.047))
pv_par_swap = ois_swap_price_from_graph(
    par_swap, graph,
    discount_id="USD.SOFR.OIS",
    forward_id="USD.SOFR.3M",
)
print(f"\nSame swap at the calibrated par rate (4.700%):")
print(f"  PV_swap      = {float(pv_par_swap):+,.4f} USD   (≈ 0 by construction)")
```

**What this example illustrates**:

1. Two `CurveSpec` declarations describe the target — one OIS
   discount curve, one 3M forwarding curve — each with its own
   pillar grid, day count, and identifier.
2. The instrument list is *flat*: OIS-side deposits and dual-curve
   IBOR swaps go into the same list.  The solver dispatches by
   asking each instrument for its own residual via the
   [`BootstrapInstrument`](../api/curves.md#valax.curves.bootstrap_proto.BootstrapInstrument)
   protocol — no `isinstance` chains, no per-tenor loops.
3. The system is well-determined when
   `len(instruments) == sum(spec.pillar_dates.shape[0])`; here that's
   4 == 2 + 2.  `bootstrap_curve_graph` raises `ValueError` otherwise.
4. `CurveBuildDiagnostics` exposes both convergence health
   (`residuals`, `max_abs_residual`, `rmse`, `n_steps`,
   `converged`) and curve well-posedness
   (`jacobian_condition_number`), plus a per-instrument fit table
   in each quote's native units (`fitted_quotes`, `quoted_values`).
   That's what a desk sees on the calibration sign-off sheet.
5. Pricing against the graph goes through the optional dual-curve
   seam: every analytic rates pricer accepts a `forward_curve`
   argument (defaulting to the discount curve, i.e. the classic
   single-curve behaviour), and the `*_from_graph` wrappers in
   `valax.pricing.analytic.graph_pricing` resolve curve identifiers
   on the graph and delegate — as `ois_swap_price_from_graph` does
   above.  Explicit `DiscountCurve` signatures remain the primary
   API; the graph route is opt-in (see §8).

### 5.3 What used to be impossible

`bootstrap_multi_curve` could only build **one forward curve at a time**,
sequentially after the OIS curve. Two structural cases were unreachable:

**Tenor-basis stripping** (3M vs 6M SOFR).  A [`TenorBasisSwap`](../api/curves.md#valax.curves.TenorBasisSwap)
constrains both forward curves *simultaneously*; no ordering of one-curve
solves gives the right answer. In the joint solver this is one more
instrument in the list, touching three curves.

**Cross-currency joint solve** (EUR-USD).  A [`CrossCurrencyBasisSwap`](../api/curves.md#valax.curves.CrossCurrencyBasisSwap)
touches four curves at once (both OIS + both forwards). Combined with an
FXForward on the short end, the four-curve graph closes with residuals
≤ 1e-10 in a single Newton solve — see
[`tests/test_curves/test_bootstrap_graph.py`](https://github.com/acea/VALAX/blob/main/tests/test_curves/test_bootstrap_graph.py)
for a working four-curve example.

### 5.4 Deprecated path: `bootstrap_multi_curve`

`bootstrap_multi_curve` (and its return type `MultiCurveSet`) is retained for
one deprecation cycle so existing user code continues to work.  It emits a
`DeprecationWarning` on every call.  New code should call
`bootstrap_curve_graph` directly.

**Since MC-Curves-2b the deprecated path is no longer a separate solver.**
`bootstrap_multi_curve` translates its two-tier input into `CurveSpec`s plus
graph instruments and makes a single `bootstrap_curve_graph` call; the old
sequential OIS-then-tenor pipeline (and its packed-swap residual machinery)
has been deleted.  Three consequences for callers:

- **Same numbers, one solve.** The calibration target is unchanged — discount
  quotes price on the OIS curve alone; forward-leg deposits/FRAs keep their
  single-curve DF relationship; forward-leg `SwapRate` quotes are reinterpreted
  as `IborSwapRate` (floating schedule = fixed schedule, OIS discounting),
  which is exactly the dual-curve par condition the sequential code solved.
  Curves agree with an explicit joint-solver build to ≤ 1e-12.
- **Gradients now flow.** Because the solve is one `optimistix.root_find` under
  `ImplicitAdjoint`, `jax.grad` of a calibrated DF w.r.t. any input quote —
  including the swap legs — is finite and cheap.
- **`MultiCurveSet` is a view, not a copy.** `MultiCurveSet.to_graph()` and
  `MultiCurveSet.from_graph(graph, discount_id, forward_ids)` convert between
  the two-tier container and a `CurveGraph` while sharing the same
  `DiscountCurve` objects:

```python
graph = mcs.to_graph(discount_id="USD.SOFR.OIS",
                     forward_ids={"3M": "USD.SOFR.3M"})
# ... use the graph-aware pricers / CurveGraph shocks ...
mcs_again = MultiCurveSet.from_graph(
    graph, "USD.SOFR.OIS", {"3M": "USD.SOFR.3M"},
)
```

Both APIs are expected to coexist long-term: `CurveGraph` is the engine,
`MultiCurveSet` the two-tier pedagogical lens on it.

#### 5.4a Migrating from `bootstrap_multi_curve`

Existing code that uses the sequential dual-curve pipeline typically
looks like this:

```python
# BEFORE — sequential dual-curve, single-currency, DeprecationWarning
from valax.curves import (
    DepositRate, SwapRate, bootstrap_multi_curve,
)

mcs = bootstrap_multi_curve(
    reference_date=REF,
    discount_instruments=[
        DepositRate(start_date=REF, end_date=..., rate=..., day_count="act_360"),
        SwapRate(start_date=REF, fixed_dates=..., rate=..., day_count="act_360"),
        # ...
    ],
    forward_instruments={
        "3M": [
            DepositRate(start_date=REF, end_date=..., rate=..., day_count="act_360"),
            SwapRate(start_date=REF, fixed_dates=..., rate=..., day_count="act_360"),
            # ...
        ],
    },
    day_count="act_360",
)

discount_curve = mcs.discount_curve
forward_3m     = mcs.forward_curves["3M"]
```

Migrated to the joint solver, the same build looks like:

```python
# AFTER — one joint Newton solve; no DeprecationWarning
from valax.curves import (
    CurveSpec, DepositRate, SwapRate, bootstrap_curve_graph,
)

ois_spec = CurveSpec(
    curve_id="USD.SOFR.OIS", currency="USD",
    pillar_dates=..., day_count="act_360",
)
fwd_spec = CurveSpec(
    curve_id="USD.SOFR.3M", currency="USD",
    pillar_dates=..., day_count="act_360",
)

# Explicit ``curves_touched`` on every instrument replaces the
# ``discount_instruments`` / ``forward_instruments`` split.
ois_insts = [
    DepositRate(..., curves_touched=("USD.SOFR.OIS",)),
    SwapRate(..., curves_touched=("USD.SOFR.OIS",)),
    # ...
]
fwd_insts = [
    DepositRate(..., curves_touched=("USD.SOFR.3M",)),
    SwapRate(..., curves_touched=("USD.SOFR.3M",)),
    # ...
]

graph, diag = bootstrap_curve_graph(
    reference_date=REF,
    curve_specs=[ois_spec, fwd_spec],
    instruments=ois_insts + fwd_insts,
)

discount_curve = graph["USD.SOFR.OIS"]
forward_3m     = graph["USD.SOFR.3M"]
```

If your downstream code expects a `MultiCurveSet`-shaped object,
recover the two halves from the graph in one line:

```python
mcs_like = {
    "discount": graph["USD.SOFR.OIS"],
    "forward_3M": graph["USD.SOFR.3M"],
}
# Or, keeping the MultiCurveSet type for legacy callers:
from valax.curves import MultiCurveSet
mcs = MultiCurveSet(
    discount_curve=graph["USD.SOFR.OIS"],
    forward_curves={"3M": graph["USD.SOFR.3M"]},
)
```

**Behavioural differences to be aware of after migration:**

1. Return value changes from `MultiCurveSet` to
   `(CurveGraph, CurveBuildDiagnostics)`.  You gain the diagnostics —
   `residuals`, `max_abs_residual`, `rmse`, `n_steps`,
   `fitted_quotes`, `quoted_values`, `jacobian_condition_number`,
   `converged` — that the sequential pipeline never produced.
2. Every instrument now needs an explicit `curves_touched` tuple.
   The single-curve default `("_default_",)` will not match any of
   your curve specs; the solver raises `ValueError` up front rather
   than silently mis-routing.
3. The system must be square: `len(instruments) ==
   sum(spec.pillar_dates.shape[0])`.  In practice this matches what
   the sequential pipeline did implicitly (one instrument per pillar
   per curve).
4. `IborSwapRate` — a proper dual-curve residual — is the natural
   replacement for the `SwapRate` quotes on the forward-curve leg.
   `SwapRate` on a forward-curve tenor still works (it degenerates
   to the single-curve residual on that tenor's own curve), but
   you'll want to switch to `IborSwapRate` to get the OIS-discount /
   3M-project decomposition explicitly.  See the [Dual-curve build
   in §5.2](#52-a-worked-usd-dual-curve-build) for the shape.

### 5.5 Quote-sensitivity Jacobian

Because the Newton solve is wrapped by `optimistix.ImplicitAdjoint`,
`jax.grad` and `jax.jacrev` flow through the calibrated graph in **one linear
solve** per output, independent of Newton iteration count. The convenience
wrapper [`quote_jacobian`](../api/curves.md#valax.curves.quote_jacobian) exposes this:

```python
from valax.curves import quote_jacobian

# ∂DF / ∂(quote rate) for every pillar × every quote.
J = quote_jacobian(
    reference_date=ref,
    curve_specs=[ois_spec, fwd_spec],
    instruments=ois_deps + [ibor_1y, ibor_2y],
    by="df",           # or "log_df" or "zero_rate"
)
# J.shape == (n_pillars_total, n_quotes)
```

This is the matrix a rates desk uses to hedge curve risk with liquid
instruments. Cheap via autodiff, prohibitively expensive via finite
differences.

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
    s = SwapRate(ref, generate_schedule(2025, 7, 1, 2030, 1, 1, frequency=2).payment_dates,
                 swap_rate, "act_360")
    curve = bootstrap_sequential(ref, [d, s], "act_365")
    bond = FixedRateBond(
        payment_dates=generate_schedule(2025, 7, 1, 2030, 1, 1, frequency=2).payment_dates,
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

!!! warning "`from_zc_rates` is algebraic, not a bootstrap"

    Unlike the rates curves above, `from_zc_rates` applies the closed-form
    identity $\mathrm{CPI}(T) = \mathrm{CPI}_0 (1 + z_T)^{\tau_T}$ directly to
    the quoted zero-coupon rates. There is no residual, no root-find and no
    `BootstrapInstrument`, so inflation quotes do not participate in
    `bootstrap_curve_graph` and carry no `quote_jacobian` row. Routing ZCIS /
    YYIS quotes through the joint solver is on the
    [roadmap](../roadmap.md).

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
[theory §3.6](../theory/curves.md#36-inflation-curves-and-breakeven-pricing). For
pricing inflation swaps and caps/floors against a built curve, see
[Inflation Derivatives](inflation.md).

## 8. What is not yet implemented

!!! note "Multi-curve pricing is **opt-in**: single-curve remains the default"
    VALAX ships production-grade multi-curve *calibration*
    (`bootstrap_curve_graph`, `CurveGraph`, `quote_jacobian`, extended
    `CurveBuildDiagnostics`) **and** an optional dual-curve pricing seam
    (the MC-Pricing workstream).  The analytic rates pricers in
    `floating.py`, `caplets.py`, `swaptions.py`, and `rates_exotics.py`
    accept an optional `forward_curve: DiscountCurve | None = None`
    argument: `None` (the default) reproduces the classic single-curve
    behaviour exactly, while passing a projection curve prices the
    floating/forward legs off that curve and discounts on `curve`.
    `bonds.py` and `inflation.py` are discount-only and unchanged;
    `cross_currency_swap_price` in `rates_exotics.py` keeps its explicit
    (`domestic_curve` + `foreign_curve`) pair.

    **How to consume a calibrated graph:**

    - Directly: `swap_price(swap, graph["USD.SOFR.OIS"],
      forward_curve=graph["USD.SOFR.3M"])`.
    - Via the thin wrappers in `valax.pricing.analytic.graph_pricing`:
      `swap_price_from_graph(swap, graph, "USD.SOFR.OIS", "USD.SOFR.3M")`
      — pure delegation, no numeric logic (see §5.2 step 6 for a worked
      example).
    - Risk: bump individual graph curves with
      `valax.risk.shocks.bump_graph_curve` / `parallel_graph_shift` /
      `key_rate_graph_bump`, and carry a graph on
      `MarketData.curve_graph` (optional field, defaults to `None`).

    Note that `InterestRateSwap` / `Swaption` carry no separate float
    schedule, so their dual-curve floating legs use the fixed schedule as
    the projection grid; `OISSwap` and `FloatingRateBond` carry explicit
    float schedules and project on them.

The MC-Curves-1 PR series shipped the `BootstrapInstrument` protocol, eleven
calibration quote types (`DepositRate`, `FRA`, `SwapRate`, `OISSwapRate`,
`IborSwapRate`, `MoneyMarketFuture`, `TenorBasisSwap`, `FXForward`, `FXSwap`,
`CrossCurrencyBasisSwap`, `TurnInstrument`), and supporting infrastructure
(`CurveGraph`, `FixingHistory`, convexity-adjustment plug-ins).

The MC-Curves-2 PR series (see [`production.md` §11.4](../architecture/production.md#114-joint-global-solver))
shipped the joint multi-curve Newton solver
[`bootstrap_curve_graph`](../api/curves.md#valax.curves.bootstrap_curve_graph), the
declarative curve descriptor `CurveSpec`, the implicit-adjoint quote-Jacobian
helper [`quote_jacobian`](../api/curves.md#valax.curves.quote_jacobian), and the
`CurveBuildDiagnostics` container.  The two-tier `bootstrap_multi_curve`
entry point is retained for one deprecation cycle and emits a
`DeprecationWarning` on every call; since MC-Curves-2b its sequential
internals are gone and it solves through the joint solver (see
[§5.4](#54-deprecated-path-bootstrap_multi_curve)).

The extended `CurveBuildDiagnostics` (MC-Curves-3) now ships production-grade
convergence and well-posedness diagnostics: `residuals`, `max_abs_residual`,
`rmse`, `n_steps`, `converged`, plus the per-instrument fit table
(`fitted_quotes` and `quoted_values` in each instrument's native quote units)
and the residual-Jacobian condition number (`jacobian_condition_number`).  See
[`CurveBuildDiagnostics`](../api/curves.md#valax.curves.CurveBuildDiagnostics).
`quote_jacobian` covers every registered quote type — including `.spread`
(`TenorBasisSwap`, `CrossCurrencyBasisSwap`), `.futures_rate`
(`MoneyMarketFuture`), `.quoted_forward` (`FXForward`), `.far_rate`
(`FXSwap`), and `.jump_size` (`TurnInstrument`).

What is still pending on the **build** side — alternative interpolation
modes on `CurveSpec.interp`, turn-of-year step interpolation,
Hull-White-derived futures convexity, CSA / collateralised discounting,
business-day calendars, an over-determined least-squares variant, fixings
seasoning, and constrained curves — is enumerated once in
[Roadmap → Multi-Curve Framework § Curve build](../roadmap.md#6-curve-build-absorbed-from-the-retired-mc-curves-3-notes),
which is also the single owner of the *downstream* gaps (risk,
calibration, vol surfaces, portfolio).

Contributions welcome — see `CONTRIBUTING.md` at the repository root.

## 9. Computing residuals on a hand-built graph

Each `BootstrapInstrument` exposes a `residual(graph, fixings, ref_date)`
method that returns zero when the curve graph correctly reprices the quote.
The joint solver
([`bootstrap_curve_graph`](../api/curves.md#valax.curves.bootstrap_curve_graph))
collects these residuals from a list of instruments and finds the curve graph
that zeroes them out simultaneously.

You don't need to wait for the solver to use the residual interface — you can
construct any graph by hand and verify (or compute Greeks against) any
instrument's residual directly.  This section walks through that workflow,
which is exactly what the per-instrument unit tests in
`tests/test_curves/test_instrument_residuals.py` do.

### The `CurveGraph` container

```python
import jax.numpy as jnp
from valax.curves import (
    CurveGraph,
    DiscountCurve,
    OISSwapRate,
    IborSwapRate,
    empty_fixing_history,
)
from valax.dates import ymd_to_ordinal

ref = ymd_to_ordinal(2025, 1, 1)

# Two flat continuously-compounded curves, hand-built.
def flat_curve(rate):
    pillars = jnp.array([
        int(ref),
        int(ymd_to_ordinal(2030, 1, 1)),
    ], dtype=jnp.int32)
    times = (pillars - int(ref)).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rate * times),
        reference_date=ref,
    )

graph = CurveGraph(curves={
    "USD.SOFR.OIS": flat_curve(0.035),
    "USD.SOFR.3M":  flat_curve(0.040),
})
```

### Single-curve residual

```python
fixed_dates = jnp.array([
    int(ymd_to_ordinal(2026, 1, 1)),
    int(ymd_to_ordinal(2027, 1, 1)),
], dtype=jnp.int32)

ois_swap = OISSwapRate(
    start_date=ref,
    fixed_dates=fixed_dates,
    rate=jnp.asarray(0.035),
    day_count="act_360",
    curves_touched=("USD.SOFR.OIS",),
    index_id="USD.SOFR",
)

# Compute residual against the graph.  ~1e-3 because the chosen rate
# isn't exactly the curve's par swap rate; for an actual bootstrap run
# the joint solver would adjust the curve until this is zero.
r = ois_swap.residual(graph, empty_fixing_history(), ref)
print(float(r))   # e.g. -0.0006
```

### Dual-curve residual

```python
ibor_swap = IborSwapRate(
    start_date=ref,
    fixed_dates=fixed_dates,
    float_dates=fixed_dates,
    fixing_dates=jnp.concatenate(
        [jnp.asarray(ref, dtype=jnp.int32)[None], fixed_dates[:-1]]
    ),
    rate=jnp.asarray(0.040),
    fixed_day_count="act_360",
    float_day_count="act_360",
    curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
    index_id="USD.SOFR.3M",
)

r = ibor_swap.residual(graph, empty_fixing_history(), ref)
# Same residual contract: zero when the dual-curve par condition holds
# for these two curves and this swap rate.
```

### Autodiff through the residual

Because the residual is a pure JAX function, differentiating it w.r.t. the
graph leaves or the instrument's array fields is a one-liner:

```python
import equinox as eqx

# Sensitivity of the residual to each pillar of the OIS curve.
def f(g):
    return ibor_swap.residual(g, empty_fixing_history(), ref)

grads = eqx.filter_grad(f)(graph)
ois_pillar_sensitivities = grads.curves["USD.SOFR.OIS"].discount_factors
```

This is also the mechanism by which the joint solver's
`optimistix.ImplicitAdjoint` produces quote-sensitivity Jacobians for free.
The residual is the contract; everything above (sensitivities, JIT
compilation, the eventual joint solver) composes on top of it without
requiring any further per-instrument code.

For the no-arbitrage motivation behind each residual, see
[theory §3.7](../theory/curves.md#37-no-arbitrage-relations-across-curves)
(CIP, tenor basis, XCCY basis) and
[theory §3.8](../theory/curves.md#38-joint-multi-curve-calibration) (the
joint residual system).  For full instrument-by-instrument signatures and
residual equations, see [API: Bootstrap Instruments](../api/curves.md#bootstrap-instruments).

## 10. Further reading

- [Models & Theory §3](../theory/curves.md#3-curve-framework) — mathematical framework: DFs, zero/forward rates, single vs. multi-curve, interpolation families, day count conventions.
- [Fixed Income guide](fixed-income.md) — using a curve to price bonds, extract YTM, duration, KRDs.
- [Interest Rate Exotics guide](rates-exotics.md) — OIS swaps, XCCY, CMS, range accrual against a multi-curve set.
- [Inflation Derivatives guide](inflation.md) — ZCIS, YYIS, inflation caps against an `InflationCurve`.
- [Lattice Methods](lattice.md#hull-white-trinomial-tree) — the Hull-White tree, which consumes a `DiscountCurve` for exact-fit $\theta(t)$.
