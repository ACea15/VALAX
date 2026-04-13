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
