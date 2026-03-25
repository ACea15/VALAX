# Greeks

## `greeks`

```python
greeks(pricing_fn, instrument, *market_args) -> dict[str, Float[Array, ""]]
```

Compute price and all standard Greeks at once.

**Returns** a dict with keys: `price`, `delta`, `gamma`, `vega`, `volga`, `vanna`, `rho`, `dividend_rho`, `theta`.

**Arguments**:

- `pricing_fn` — Any pure pricing function with signature `(instrument, *market_args) -> scalar`.
- `instrument` — The instrument pytree (not differentiated).
- `*market_args` — Market inputs in order: `spot/forward, vol, rate, [dividend]`.

!!! warning "Argument order matters"
    The `greeks()` function maps positional arguments to Greek names:
    arg 0 = delta, arg 1 = vega, arg 2 = rho, arg 3 = dividend_rho.
    This matches the signature `(spot, vol, rate, dividend)` for Black-Scholes.
    For Black-76/Bachelier with 3 args, `dividend_rho` will error.

## `greek`

```python
greek(pricing_fn, name, instrument, *market_args) -> Float[Array, ""]
```

Compute a single named Greek.

**Names**: `delta`, `gamma`, `vega`, `volga`, `vanna`, `rho`, `dividend_rho`, `theta`.

```python
from valax.greeks import greek
delta = greek(black_scholes_price, "delta", option, spot, vol, rate, div)
```
