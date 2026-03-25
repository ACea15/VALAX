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
