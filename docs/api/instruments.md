# Instruments

All instruments are `equinox.Module` subclasses — frozen dataclasses registered as JAX pytrees. They carry no pricing logic.

## `EuropeanOption`

```python
from valax.instruments import EuropeanOption

class EuropeanOption(eqx.Module):
    strike: Float[Array, ""]                          # strike price
    expiry: Float[Array, ""]                          # time to expiry (year fractions)
    is_call: bool = eqx.field(static=True, default=True)  # True=call, False=put
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `strike` | `Float[Array, ""]` | No | Strike price. Differentiable. |
| `expiry` | `Float[Array, ""]` | No | Time to expiry in year fractions. |
| `is_call` | `bool` | Yes | Call (`True`) or put (`False`). Not differentiable. |

**Notes**:

- `is_call` is marked `static=True` because it controls branching logic (call vs put formula). JAX traces separate code paths for `True` and `False`.
- `strike` and `expiry` are JAX arrays, so you can differentiate through them (e.g., strike sensitivity, theta via autodiff on expiry).
- For batch pricing via `vmap`, create a single `EuropeanOption` with batched arrays: `EuropeanOption(strike=jnp.array([90, 100, 110]), ...)`.

## `ZeroCouponBond`

```python
from valax.instruments import ZeroCouponBond

class ZeroCouponBond(eqx.Module):
    maturity: Int[Array, ""]     # maturity date (ordinal)
    face_value: Float[Array, ""] # par value paid at maturity
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `maturity` | `Int[Array, ""]` | No | Maturity date as ordinal (days since epoch). |
| `face_value` | `Float[Array, ""]` | No | Par/face value. Differentiable. |

## `FixedRateBond`

```python
from valax.instruments import FixedRateBond

class FixedRateBond(eqx.Module):
    payment_dates: Int[Array, "n_payments"]  # coupon + maturity dates
    settlement_date: Int[Array, ""]          # valuation date
    coupon_rate: Float[Array, ""]            # annual coupon rate
    face_value: Float[Array, ""]             # par value
    frequency: int = eqx.field(static=True, default=2)   # coupons/year
    day_count: str = eqx.field(static=True, default="act_365")
```

**Fields**:

| Field | Type | Static | Description |
|---|---|---|---|
| `payment_dates` | `Int[Array, "n"]` | No | Ordinal dates for coupon payments (includes maturity). |
| `settlement_date` | `Int[Array, ""]` | No | Settlement/valuation date as ordinal. |
| `coupon_rate` | `Float[Array, ""]` | No | Annual coupon rate (e.g., 0.05 for 5%). Differentiable. |
| `face_value` | `Float[Array, ""]` | No | Par/face value. Differentiable. |
| `frequency` | `int` | Yes | Coupons per year (1, 2, or 4). |
| `day_count` | `str` | Yes | Day count convention for accrual. |

**Notes**:

- Use `generate_schedule()` from `valax.dates` to build `payment_dates`.
- Only future cash flows (`payment_date > settlement_date`) are included in pricing.
- `frequency` and `day_count` are static because they control computation structure, not values.
