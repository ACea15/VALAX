# Curves

Discount curve construction and interpolation. All curve objects are `equinox.Module` pytrees — fully differentiable for key-rate duration computation.

## `DiscountCurve`

```python
class DiscountCurve(eqx.Module):
    pillar_dates: Int[Array, "n_pillars"]       # sorted ordinal dates
    discount_factors: Float[Array, "n_pillars"] # DF at each pillar
    reference_date: Int[Array, ""]              # valuation date
    day_count: str = "act_365"                  # static field
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `pillar_dates` | `Int[Array, "n"]` | No | Ordinal dates for curve nodes, sorted ascending. |
| `discount_factors` | `Float[Array, "n"]` | No | Discount factors at each pillar. First should be 1.0. **Differentiable.** |
| `reference_date` | `Int[Array, ""]` | No | Valuation date as ordinal. |
| `day_count` | `str` | Yes | Day count convention name. |

**Interpolation**: Log-linear (linear in log-DF space), equivalent to piecewise-constant continuously-compounded forward rates. Flat extrapolation beyond curve range.

**Usage**:

```python
# Callable — interpolate at arbitrary dates
df = curve(date)           # single date
dfs = curve(date_array)    # vectorized
```

!!! note "Differentiability"
    `discount_factors` are differentiable leaves. `jax.grad` through any function that uses a `DiscountCurve` gives sensitivities to each pillar — this is how key-rate durations work.

## `forward_rate`

```python
forward_rate(curve, start, end) -> Float[Array, ""]
```

Simply-compounded forward rate between two ordinal dates:

$$F(t_1, t_2) = \frac{DF(t_1)/DF(t_2) - 1}{\tau(t_1, t_2)}$$

## `zero_rate`

```python
zero_rate(curve, date) -> Float[Array, ""]
```

Continuously-compounded zero rate to a given date:

$$r(t) = -\frac{\ln DF(t)}{\tau(\text{ref}, t)}$$

---

## `InflationCurve`

Term structure of forward CPI (Consumer Price Index) levels. Interpolates in **log-CPI** space for smooth implied forward inflation rates.

```python
class InflationCurve(eqx.Module):
    pillar_dates: Int[Array, "n"]       # sorted ordinal dates
    forward_cpis: Float[Array, "n"]     # forward CPI at each pillar
    base_cpi: Float[Array, ""]          # CPI at inception
    reference_date: Int[Array, ""]      # valuation date
    day_count: str = "act_act"          # static field
```

**Functions**:

| Function | Description |
|---|---|
| `forward_cpi(curve, dates)` | Interpolated forward CPI at arbitrary dates |
| `zc_inflation_rate(curve, dates)` | Zero-coupon breakeven rate: $(CPI(T)/CPI(0))^{1/T} - 1$ |
| `yoy_forward_rate(curve, starts, ends)` | Year-on-year forward: $CPI(T_i)/CPI(T_{i-1}) - 1$ |
| `from_zc_rates(ref, pillars, rates, base_cpi)` | Constructor from ZC breakeven rates |

**Usage**:

```python
from valax.curves import InflationCurve, forward_cpi, from_zc_rates

curve = from_zc_rates(ref_date, pillar_dates, zc_rates, base_cpi=jnp.array(100.0))
cpi_5y = forward_cpi(curve, maturity_date)
```

!!! note "Differentiability"
    `forward_cpis` and `base_cpi` are differentiable leaves. `jax.grad` gives inflation-delta (IE01) sensitivities.

---

## Bootstrap Instruments

Bootstrap instruments are the *quotes* used to calibrate curves. Each one
implements the [`BootstrapInstrument`](#bootstrapinstrument-protocol)
protocol — it carries a static `curves_touched` tuple of curve identifiers
and exposes a `residual(graph, fixings, ref_date)` method that returns
zero when the curve graph correctly reprices the quote.

These are calibration *inputs*, distinct from the tradeable instruments in
`valax.instruments` (which carry notional and direction). They live in
`valax.curves` so the dependency graph stays clean.

For the no-arbitrage motivation behind each instrument's residual, see
[theory §3.7](../theory.md#37-no-arbitrage-relations-across-curves) and
[§3.8](../theory.md#38-joint-multi-curve-calibration). For the production
roadmap that uses these instruments in a joint solver, see
[`production.md` §11](../architecture/production.md#11-multi-curve-framework).

### `BootstrapInstrument` protocol

Every quote type is a structural [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol):

```python
class BootstrapInstrument(Protocol):
    curves_touched: tuple[str, ...]   # static
    def residual(
        self,
        graph: CurveGraph,
        fixings: FixingHistory,
        ref_date: Int[Array, ""],
    ) -> Float[Array, ""]: ...
```

`@runtime_checkable` — `isinstance(inst, BootstrapInstrument)` works as a
weak check on attribute presence.

### `CurveGraph`

```python
class CurveGraph(eqx.Module):
    curves: dict[str, DiscountCurve]
```

A flat, identifier-keyed registry of discount curves. Lookup via `graph[curve_id]`,
membership via `curve_id in graph`. JAX-pytree native — `jax.tree_util` traverses
the dict values; string keys are static metadata. Single-curve callers use
the sentinel id `"_default_"`; multi-curve callers use semantic identifiers
like `"USD.SOFR.OIS"`, `"EUR.EURIBOR.6M"`.

### `FixingSeries` / `FixingHistory`

```python
class FixingSeries(eqx.Module):
    fixing_dates: Int[Array, " n"]
    fixings: Float[Array, " n"]

class FixingHistory(eqx.Module):
    indices: dict[str, FixingSeries]
```

Realised fixings keyed by index id. Methods:

| Method | Returns | Notes |
|---|---|---|
| `series.lookup(date)` | `Float[Array, ""]` | Realised value at `date`, or `nan` if absent. |
| `series.has_fixing(date)` | `Bool[Array, ""]` | JIT-friendly presence check. |
| `series.lookup_many(dates)` | `Float[Array, " m"]` | Vectorised lookup; `nan` for absent. |
| `history.lookup(idx, date)` | `Float[Array, ""]` | `KeyError` (trace-time) if `idx` absent. |
| `history.lookup_many(idx, dates)` | `Float[Array, " m"]` | Vectorised. |
| `empty_fixing_history()` | `FixingHistory` | Default arg for forward-only swaps. |

### Convexity-adjustment plug-ins

```python
ConvexityAdjFn = Callable[
    [CurveGraph, Int[Array, ""], Int[Array, ""]],
    Float[Array, ""],
]
```

Used by `MoneyMarketFuture`. Two factories:

| Factory | Returns | Use case |
|---|---|---|
| `no_convexity_adj()` | Always 0.0 | Short-dated futures; simplified calibration. |
| `constant_convexity_adj(bps)` | Always `bps * 1e-4` | Desk-supplied bps; production default for now. |

Reserved for a follow-up PR (gated on the short-rate-model integration with
the curve build): `hull_white_convexity_adj(model)` — closed-form derivation
in [theory §3.9](../theory.md#39-futures-convexity-adjustment-and-fixings).

### Single-curve quote types

Each type's `curves_touched` is a 1-tuple defaulting to `("_default_",)`.

#### `DepositRate`

```python
class DepositRate(eqx.Module):
    start_date:    Int[Array, ""]
    end_date:      Int[Array, ""]
    rate:          Float[Array, ""]
    day_count:     str = "act_360"            # static
    curves_touched: tuple = ("_default_",)    # static
```

**Residual**: $DF(\text{end})\,(1 + r\,\tau) - DF(\text{start})$.

#### `FRA`

Same field layout and same residual as `DepositRate`. Distinction is semantic
(`start_date > ref_date`).

#### `SwapRate`

```python
class SwapRate(eqx.Module):
    start_date:     Int[Array, ""]
    fixed_dates:    Int[Array, " n"]
    rate:           Float[Array, ""]
    day_count:      str = "act_360"           # static
    curves_touched: tuple = ("_default_",)    # static
```

**Residual**: $r \sum_i \tau_i\,DF(T_i) - (DF(\text{start}) - DF(T_n))$.

#### `OISSwapRate`

```python
class OISSwapRate(eqx.Module):
    start_date:     Int[Array, ""]
    fixed_dates:    Int[Array, " n"]
    rate:           Float[Array, ""]
    day_count:      str = "act_360"           # static
    curves_touched: tuple = ("_default_",)    # static
    index_id:       str = "OIS"               # static (reserved for fixings)
```

**Residual**: same as `SwapRate` (the daily-compounded float leg telescopes
exactly to $DF(\text{start}) - DF(T_n)$ on a single OIS curve). Distinct
class because the multi-curve world treats OIS swaps as single-curve and
IBOR swaps as dual-curve. See `IborSwapRate`.

`fixings` is accepted for protocol conformance but not consumed in MC-Curves-1;
seasoned-OIS support is deferred.

#### `MoneyMarketFuture`

```python
class MoneyMarketFuture(eqx.Module):
    start_date:        Int[Array, ""]
    end_date:          Int[Array, ""]
    futures_rate:      Float[Array, ""]
    day_count:         str = "act_360"            # static
    curves_touched:    tuple = ("_default_",)     # static
    convexity_adj_fn:  ConvexityAdjFn             # static, default no_convexity_adj
```

**Residual**: $\text{futures\_rate} - \text{adj}(graph, T_0, T_1) - F^{\text{curve}}(T_0, T_1)$.

The convexity adjustment is pluggable — see the table above.

#### `TurnInstrument`

```python
class TurnInstrument(eqx.Module):
    turn_date:           Int[Array, ""]
    jump_size:           Float[Array, ""]
    accrual_days_before: int = 1                  # static
    accrual_days_after:  int = 1                  # static
    day_count:           str = "act_360"          # static
    curves_touched:      tuple = ("_default_",)   # static
```

**Residual**: $F^{\text{curve}}(T_- ,T_+) - \text{jump\_size}$ where the
window straddles the turn date.

Turn instruments require an interpolation method that admits discontinuities;
under VALAX's current log-linear-DF interpolation, the jump can be modelled by
placing two pillars tightly around the turn date.

### Multi-curve same-currency quote types

#### `IborSwapRate` (dual-curve)

```python
class IborSwapRate(eqx.Module):
    start_date:        Int[Array, ""]
    fixed_dates:       Int[Array, " n_fixed"]
    float_dates:       Int[Array, " n_float"]
    fixing_dates:      Int[Array, " n_float"]
    rate:              Float[Array, ""]
    fixed_day_count:   str = "act_360"            # static
    float_day_count:   str = "act_360"            # static
    curves_touched:    tuple = ("_default_", "_default_")   # static
    index_id:          str = "IBOR"               # static
```

**`curves_touched`** convention: `(discount_curve_id, forward_curve_id)`.

**Residual** (theory.md §3.2 dual-curve par condition):
$\text{fixed\_pv} - \text{float\_pv}$ with float coupons projected from
`forward_curve_id` and discounted (along with the fixed leg) with
`discount_curve_id`. Past coupons (`fixing_date <= ref_date` and recorded
in `FixingHistory[index_id]`) use the realised rate instead of the
projection.

#### `TenorBasisSwap` (three-curve)

```python
class TenorBasisSwap(eqx.Module):
    start_date:           Int[Array, ""]
    leg_a_dates:          Int[Array, " n_a"]
    leg_a_fixing_dates:   Int[Array, " n_a"]
    leg_b_dates:          Int[Array, " n_b"]
    leg_b_fixing_dates:   Int[Array, " n_b"]
    spread:               Float[Array, ""]    # default 0.0
    leg_a_day_count:      str = "act_360"     # static
    leg_a_index_id:       str = "IBOR_A"      # static
    leg_b_day_count:      str = "act_360"     # static
    leg_b_index_id:       str = "IBOR_B"      # static
    spread_on_leg:        str = "a"           # static, "a" or "b"
    curves_touched:       tuple = (3 × "_default_",)     # static
```

**`curves_touched`** convention: `(discount_id, leg_a_curve_id, leg_b_curve_id)`.

**Residual** (theory.md §3.7): $\text{leg\_a\_pv} - \text{leg\_b\_pv}$,
with the spread added per period to the leg indicated by `spread_on_leg`.
Both legs use forward projection from their respective forward curves and
discounting from the shared OIS curve.

### Cross-currency quote types

#### `FXForward`

```python
class FXForward(eqx.Module):
    value_date:      Int[Array, ""]
    settle_date:     Int[Array, ""]
    quoted_forward:  Float[Array, ""]
    fx_spot:         Float[Array, ""]
    curves_touched:  tuple = ("_default_", "_default_")    # static
```

**`curves_touched`** convention: `(domestic_curve_id, foreign_curve_id)`.
`fx_spot` and `quoted_forward` are in *domestic per foreign* units.

**Residual** (theory.md §3.7 CIP):
$F^{\text{quote}} - S(0) \cdot DF_{\text{foreign}}(T) / DF_{\text{domestic}}(T)$.

#### `FXSwap`

```python
class FXSwap(eqx.Module):
    near_date:       Int[Array, ""]
    far_date:        Int[Array, ""]
    near_rate:       Float[Array, ""]
    far_rate:        Float[Array, ""]
    curves_touched:  tuple = ("_default_", "_default_")    # static
```

**Residual**: CIP relation between near and far legs:

$$
\text{far\_rate} \cdot DF_{\text{for}}(T_{\text{near}})\, DF_{\text{dom}}(T_{\text{far}})
- \text{near\_rate} \cdot DF_{\text{dom}}(T_{\text{near}})\, DF_{\text{for}}(T_{\text{far}}) = 0.
$$

Reduces to `FXForward`'s residual when `near_date == ref_date`.

#### `CrossCurrencyBasisSwap` (CCBS)

```python
class CrossCurrencyBasisSwap(eqx.Module):
    start_date:        Int[Array, ""]
    dom_dates:         Int[Array, " n_dom"]
    dom_fixing_dates:  Int[Array, " n_dom"]
    for_dates:         Int[Array, " n_for"]
    for_fixing_dates:  Int[Array, " n_for"]
    fx_spot:           Float[Array, ""]
    spread:            Float[Array, ""]    # default 0.0
    dom_day_count:     str = "act_360"     # static
    dom_index_id:      str = "DOM_FLOAT"   # static
    for_day_count:     str = "act_360"     # static
    for_index_id:      str = "FOR_FLOAT"   # static
    spread_on_leg:     str = "domestic"    # static, "domestic" or "foreign"
    variant:           str = "constant_notional"   # static, or "mtm"
    curves_touched:    tuple = (4 × "_default_",)  # static
```

**`curves_touched`** convention:
`(dom_discount_id, dom_forward_id, for_discount_id, for_forward_id)`.

**Residuals** (theory.md §3.7 XCCY formula):

* `variant = "constant_notional"`:

    $$
    [\text{dom\_coupon\_pv} + (1 - DF_{\text{dom}}(T_n))]
    - [\text{for\_coupon\_pv} + (1 - DF_{\text{for}}(T_n))].
    $$

    `fx_spot` cancels in the normalised residual.

* `variant = "mtm"` (mark-to-market — notional resets at each fixing):

    $$
    \sum_i (F^{\text{dom}}_i + s_{\text{dom}})\,\tau_i\,DF_{\text{dom}}(T_i)
    - \sum_j (F^{\text{for}}_j + s_{\text{for}})\,\tau_j\,DF_{\text{dom}}(T_j).
    $$

    The foreign discount curve does not enter the MTM residual (Henrard 2014
    §10.5).

---

## SurvivalCurve

Term structure of survival probabilities for a single credit entity. Mirrors `DiscountCurve` in design: log-linear interpolation between pillars (= piecewise-constant hazard rate), drop-in target for `jax.grad` to obtain per-pillar credit-delta sensitivities.

### `SurvivalCurve`

```python
class SurvivalCurve(eqx.Module):
    pillar_dates: Int[Array, "n_pillars"]
    survival_probabilities: Float[Array, "n_pillars"]
    reference_date: Int[Array, ""]
    day_count: str = "act_365"

    def __call__(dates) -> Float[Array, "..."]  # interpolated S(t)
```

### Constructors and helpers

```python
from_hazard_rates(reference_date, pillar_dates, hazards, day_count="act_365") -> SurvivalCurve
from_cds_spreads(reference_date, pillar_dates, spreads, recovery_rate=0.4, day_count="act_365") -> SurvivalCurve
```

`from_hazard_rates` treats `hazards[i]` as the constant hazard on `(pillar_{i-1}, pillar_i]`. `from_cds_spreads` uses the credit-triangle `h ≈ s / (1 − R)` to bootstrap survival from a flat-spread CDS curve.

```python
survival_probability(curve, date) -> Float[Array, ""]
hazard_rate(curve, date) -> Float[Array, ""]                # average h on [0, t]
piecewise_hazards(curve) -> Float[Array, "n_pillars"]       # per-interval h
```

See [Risk Factors](../risk-factors.md) — Credit section — for the role of the survival curve in the wider risk-factor catalogue, and `valax.risk.shocks` for the credit-side bump primitives.
