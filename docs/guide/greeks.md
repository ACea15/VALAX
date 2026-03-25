# Greeks via Autodiff

VALAX computes Greeks by differentiating pricing functions with `jax.grad`. This works for **every** pricing method — no method-specific code needed.

## Quick Start

```python
from valax.greeks import greeks, greek
from valax.pricing.analytic import black_scholes_price
from valax.instruments import EuropeanOption
import jax.numpy as jnp

option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
args = (jnp.array(100.0), jnp.array(0.20), jnp.array(0.05), jnp.array(0.02))

# All Greeks at once
g = greeks(black_scholes_price, option, *args)
# g["delta"], g["gamma"], g["vega"], g["theta"], g["rho"], ...

# Single Greek
delta = greek(black_scholes_price, "delta", option, *args)
```

## Available Greeks

### First Order

| Name | Definition | `argnums` |
|---|---|---|
| `delta` | $\partial V / \partial S$ | spot |
| `vega` | $\partial V / \partial \sigma$ | vol |
| `rho` | $\partial V / \partial r$ | rate |
| `dividend_rho` | $\partial V / \partial q$ | dividend |
| `theta` | $\partial V / \partial T$ | time (bump-based) |

### Second Order

| Name | Definition |
|---|---|
| `gamma` | $\partial^2 V / \partial S^2$ |
| `vanna` | $\partial^2 V / \partial S \partial \sigma$ |
| `volga` | $\partial^2 V / \partial \sigma^2$ |

## How It Works

First-order Greeks use a single `jax.grad` call:

```python
import jax

# Delta: differentiate price w.r.t. spot (arg index 1)
delta_fn = jax.grad(black_scholes_price, argnums=1)

# All first-order Greeks in one backward pass
all_grads = jax.grad(black_scholes_price, argnums=(1, 2, 3, 4))
delta, vega, rho, div_rho = all_grads(option, *args)
```

Second-order Greeks use nested `jax.grad`:

```python
# Gamma = d(delta)/d(spot)
gamma_fn = jax.grad(jax.grad(black_scholes_price, argnums=1), argnums=1)

# Vanna = d(delta)/d(vol)
vanna_fn = jax.grad(jax.grad(black_scholes_price, argnums=1), argnums=2)
```

!!! note "Theta"
    Theta is computed via finite difference (bump expiry by 1/365) rather than autodiff, since expiry lives inside the instrument pytree rather than as a separate market argument. The returned value is **per-year** theta.

## Greeks Through Any Pricing Method

The same `greeks()` function works with MC, PDE, and lattice pricers:

```python
import jax

# MC delta — autodiff through the full simulation
def mc_price_fn(spot):
    return mc_price(option, spot, model, config, key)

mc_delta = jax.grad(mc_price_fn)(jnp.array(100.0))

# PDE delta
def pde_price_fn(spot):
    return pde_price(option, spot, vol, rate, dividend, config)

pde_delta = jax.grad(pde_price_fn)(jnp.array(100.0))
```

## Accuracy

Autodiff Greeks match closed-form analytical solutions to machine precision (~1e-10 in float64). This is validated across multiple moneyness/vol/rate regimes in the test suite.
