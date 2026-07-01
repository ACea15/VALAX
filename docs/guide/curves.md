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
See [theory §3.2](../theory.md#32-single-curve-vs-multi-curve-framework).

The MC-Curves-2 API is
[`bootstrap_curve_graph`](../api/curves.md#bootstrap_curve_graph): a **joint
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

### 5.2 A dual-curve USD SOFR build

```python
import jax.numpy as jnp
from valax.curves import (
    CurveSpec, DepositRate, IborSwapRate, bootstrap_curve_graph,
)
from valax.dates.daycounts import ymd_to_ordinal

ref = ymd_to_ordinal(2025, 1, 1)

# 1) Declare the two curves we want to build.
ois_spec = CurveSpec(
    curve_id="USD.SOFR.OIS", currency="USD",
    pillar_dates=jnp.array([
        ymd_to_ordinal(2026, 1, 1),
        ymd_to_ordinal(2027, 1, 1),
    ], dtype=jnp.int32),
    day_count="act_365",
)
fwd_spec = CurveSpec(
    curve_id="USD.SOFR.3M", currency="USD",
    pillar_dates=jnp.array([
        ymd_to_ordinal(2026, 1, 1),
        ymd_to_ordinal(2027, 1, 1),
    ], dtype=jnp.int32),
    day_count="act_365",
)

# 2) OIS side: two deposits anchoring the discount curve.
ois_deps = [
    DepositRate(
        start_date=ref, end_date=ymd_to_ordinal(2026, 1, 1),
        rate=jnp.array(0.040), day_count="act_365",
        curves_touched=("USD.SOFR.OIS",),
    ),
    DepositRate(
        start_date=ref, end_date=ymd_to_ordinal(2027, 1, 1),
        rate=jnp.array(0.042), day_count="act_365",
        curves_touched=("USD.SOFR.OIS",),
    ),
]

# 3) 3M-projection side: two IBOR par swaps.  Each has a *dual-curve*
#    residual — OIS discount + 3M projection — that couples both
#    curves.  The joint solver handles that natively.
ibor_1y = IborSwapRate(
    start_date=ref,
    fixed_dates=jnp.array([ymd_to_ordinal(2026, 1, 1)], dtype=jnp.int32),
    float_dates=quarterly_dates_to(ymd_to_ordinal(2026, 1, 1)),
    fixing_dates=fixing_dates_from(float_dates),
    rate=jnp.array(0.045),
    curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
    index_id="USD.SOFR.3M",
)
ibor_2y = IborSwapRate(...)  # analogous for 2Y

# 4) One joint Newton solve — every residual zeroed simultaneously.
graph, diagnostics = bootstrap_curve_graph(
    reference_date=ref,
    curve_specs=[ois_spec, fwd_spec],
    instruments=ois_deps + [ibor_1y, ibor_2y],
)

assert diagnostics.converged
assert float(diagnostics.max_abs_residual) < 1e-10

ois_curve = graph["USD.SOFR.OIS"]
fwd_curve = graph["USD.SOFR.3M"]
```

The system is well-determined when `len(instruments) == sum(spec.pillar_dates.shape[0])`;
`bootstrap_curve_graph` raises `ValueError` otherwise. Each instrument
carries a static `curves_touched: tuple[str, ...]` naming the curves whose
DFs enter its residual — the solver dispatches by asking every instrument
for its own residual and never `isinstance`-checks.

### 5.3 What used to be impossible

`bootstrap_multi_curve` could only build **one forward curve at a time**,
sequentially after the OIS curve. Two structural cases were unreachable:

**Tenor-basis stripping** (3M vs 6M SOFR).  A [`TenorBasisSwap`](../api/curves.md#tenorbasisswap-three-curve)
constrains both forward curves *simultaneously*; no ordering of one-curve
solves gives the right answer. In the joint solver this is one more
instrument in the list, touching three curves.

**Cross-currency joint solve** (EUR-USD).  A [`CrossCurrencyBasisSwap`](../api/curves.md#crosscurrencybasisswap-ccbs)
touches four curves at once (both OIS + both forwards). Combined with an
FXForward on the short end, the four-curve graph closes with residuals
≤ 1e-10 in a single Newton solve — see
[`tests/test_curves/test_bootstrap_graph.py`](https://github.com/acea/VALAX/blob/main/tests/test_curves/test_bootstrap_graph.py)
for a working four-curve example.

### 5.4 Deprecated path: `bootstrap_multi_curve`

`bootstrap_multi_curve` (and its return type `MultiCurveSet`) is retained for
one deprecation cycle so existing user code continues to work.  It emits a
`DeprecationWarning` on every call.  New code should call
`bootstrap_curve_graph` directly. The forthcoming MC-Curves-2b workstream will
remove the legacy path entirely once callers have migrated.

### 5.5 Quote-sensitivity Jacobian

Because the Newton solve is wrapped by `optimistix.ImplicitAdjoint`,
`jax.grad` and `jax.jacrev` flow through the calibrated graph in **one linear
solve** per output, independent of Newton iteration count. The convenience
wrapper [`quote_jacobian`](../api/curves.md#quote_jacobian) exposes this:

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

The MC-Curves-1 PR series shipped the `BootstrapInstrument` protocol, eleven
calibration quote types (`DepositRate`, `FRA`, `SwapRate`, `OISSwapRate`,
`IborSwapRate`, `MoneyMarketFuture`, `TenorBasisSwap`, `FXForward`, `FXSwap`,
`CrossCurrencyBasisSwap`, `TurnInstrument`), and supporting infrastructure
(`CurveGraph`, `FixingHistory`, convexity-adjustment plug-ins).

The MC-Curves-2 PR series (see [`production.md` §11.4](../architecture/production.md#114-joint-global-solver))
shipped the joint multi-curve Newton solver
[`bootstrap_curve_graph`](../api/curves.md#bootstrap_curve_graph), the
declarative curve descriptor `CurveSpec`, the implicit-adjoint quote-Jacobian
helper [`quote_jacobian`](../api/curves.md#quote_jacobian), and the
`CurveBuildDiagnostics` container.  The pre-MC-Curves-2 sequential
`bootstrap_multi_curve` is retained for one deprecation cycle and emits a
`DeprecationWarning` on every call.

From the [roadmap (Tier 1.1)](../roadmap.md#11-multi-curve-bootstrapping)
and the production design, items still pending (MC-Curves-3+):

- **Extended calibration diagnostics** (per-instrument fitted-vs-quoted table,
  RMSE, Jacobian condition number).  The MVP `CurveBuildDiagnostics` today
  ships `residuals`, `max_abs_residual`, `n_steps`, `converged`.
- **Alternative interpolation methods**: linear on zero rates, cubic splines
  on log-DF, monotone-convex (Hagan-West), tension splines.  The log-linear
  baseline is stable and monotone but produces discontinuous forward rates at
  pillars; `CurveSpec.interp` is already the dispatch site.
- **Hull-White-derived futures convexity adjustment** — gated on the
  short-rate-model integration with the curve build.  The plug-in framework
  is in place ([`valax/curves/convexity.py`](../api/curves.md#convexity-adjustment-plug-ins));
  only the closed-form factory is missing.
- **CSA / collateralised discounting** — currency-of-collateral selection and
  FX-implied foreign-currency-collateralised discount curves.  Design note
  still to be written.
- **Business-day calendars, holidays, roll conventions** — schedules today
  ignore holidays entirely.
- **Constrained curves** (forward-rate positivity, calendar-spread
  no-arbitrage for inflation curves).

Contributions welcome — see `CONTRIBUTING.md` at the repository root.

## 9. Computing residuals on a hand-built graph

Each `BootstrapInstrument` exposes a `residual(graph, fixings, ref_date)`
method that returns zero when the curve graph correctly reprices the quote.
The forthcoming joint solver
([MC-Curves-2](../architecture/production.md#114-joint-global-solver))
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
[theory §3.7](../theory.md#37-no-arbitrage-relations-across-curves)
(CIP, tenor basis, XCCY basis) and
[theory §3.8](../theory.md#38-joint-multi-curve-calibration) (the
joint residual system).  For full instrument-by-instrument signatures and
residual equations, see [API: Bootstrap Instruments](../api/curves.md#bootstrap-instruments).

## 10. Further reading

- [Models & Theory §3](../theory.md#3-curve-framework) — mathematical framework: DFs, zero/forward rates, single vs. multi-curve, interpolation families, day count conventions.
- [Fixed Income guide](fixed-income.md) — using a curve to price bonds, extract YTM, duration, KRDs.
- [Interest Rate Exotics guide](rates-exotics.md) — OIS swaps, XCCY, CMS, range accrual against a multi-curve set.
- [Inflation Derivatives guide](inflation.md) — ZCIS, YYIS, inflation caps against an `InflationCurve`.
- [Short-Rate Models guide](short-rate.md) — Hull-White workflow, which consumes a `DiscountCurve` for exact-fit $\theta(t)$.
