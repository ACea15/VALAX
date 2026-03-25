# PDE Solvers

VALAX solves the Black-Scholes PDE using Crank-Nicolson finite differences in log-spot space, with tridiagonal systems solved via [lineax](https://docs.kidger.site/lineax/).

## The Black-Scholes PDE

In log-spot space $x = \ln S$:

$$\frac{\partial V}{\partial t} + \left(r - q - \frac{\sigma^2}{2}\right) \frac{\partial V}{\partial x} + \frac{\sigma^2}{2} \frac{\partial^2 V}{\partial x^2} - rV = 0$$

The PDE is solved backward in time from the terminal payoff using Crank-Nicolson ($\theta = 0.5$), which is unconditionally stable and second-order accurate in both time and space.

## Usage

```python
import jax.numpy as jnp
from valax.instruments import EuropeanOption
from valax.pricing.pde import pde_price, PDEConfig

option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)

price = pde_price(
    option,
    spot=jnp.array(100.0),
    vol=jnp.array(0.20),
    rate=jnp.array(0.05),
    dividend=jnp.array(0.02),
    config=PDEConfig(n_spot=400, n_time=400, spot_range=4.0),
)
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `n_spot` | 200 | Number of spatial grid points (interior) |
| `n_time` | 200 | Number of time steps |
| `spot_range` | 4.0 | Grid extends `spot_range * vol * sqrt(T)` standard deviations from `ln(spot)` |

Finer grids give more accurate prices at the cost of computation time.

## Greeks via Autodiff

The entire PDE solve — grid construction, Crank-Nicolson time-stepping, tridiagonal solves, and interpolation — is differentiable:

```python
import jax

delta = jax.grad(lambda s: pde_price(option, s, vol, rate, div, config))(spot)
vega = jax.grad(lambda v: pde_price(option, spot, v, rate, div, config))(vol)
```

## Implementation Details

- **Log-spot grid**: Uniform spacing in $x = \ln S$ ensures equal resolution across all spot levels.
- **Tridiagonal solver**: Each Crank-Nicolson time step requires solving a tridiagonal linear system, handled by `lineax.Tridiagonal()`.
- **`jax.lax.scan`**: The backward time-stepping loop uses `lax.scan` for JIT compilation — no Python-level loop overhead.
- **Boundary conditions**: Derived from BS asymptotics for very small/large spot.
