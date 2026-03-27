# Architecture

## Design Principles

VALAX follows a purely functional design imposed by JAX. Understanding these principles is essential for contributing.

### Pure Functions, No Mutable State

JAX's `jit`, `grad`, and `vmap` transformations require pure functions — no side effects, no mutable state. Every VALAX data structure is an `equinox.Module`, which is a frozen dataclass automatically registered as a JAX pytree.

```python
import equinox as eqx

class EuropeanOption(eqx.Module):
    strike: Float[Array, ""]
    expiry: Float[Array, ""]
    is_call: bool = eqx.field(static=True)  # not differentiable
```

### Instruments Are Data, Not Logic

Unlike QuantLib's `instrument.setPricingEngine(engine)` pattern, VALAX instruments carry no pricing logic. They are data-only containers describing a contract. Pricing is a standalone function:

```python
# QuantLib style (NOT how VALAX works)
instrument.setPricingEngine(engine)
price = instrument.NPV()

# VALAX style
price = black_scholes_price(option, spot, vol, rate, dividend)
```

This separation enables composability — any pricing function works with `jax.grad`, `jax.vmap`, and `jax.jit` without special handling.

### Greeks via Autodiff

The core value proposition. Every pricing function is differentiable:

```python
import jax

# First-order
delta = jax.grad(price_fn, argnums=1)(option, spot, vol, rate, div)

# Second-order
gamma = jax.grad(jax.grad(price_fn, argnums=1), argnums=1)(option, spot, vol, rate, div)

# All at once
delta, vega, rho = jax.grad(price_fn, argnums=(1, 2, 3))(option, spot, vol, rate, div)
```

This works for **all** pricing methods — analytical, Monte Carlo, PDE, and lattice. No method-specific Greek implementations needed.

## Package Structure

```
valax/
├── core/          # Type aliases, constants
├── dates/         # Day count conventions, ordinal dates, schedule generation
├── curves/        # DiscountCurve, interpolation, forward/zero rates
├── instruments/   # Data-only pytrees (options, bonds, swaps)
├── models/        # Stochastic process definitions (GBM, Heston)
├── pricing/
│   ├── analytic/  # Black-Scholes, Black-76, Bachelier, bond pricing
│   ├── mc/        # Monte Carlo engine (diffrax SDE paths)
│   ├── pde/       # Crank-Nicolson finite differences (lineax)
│   └── lattice/   # CRR binomial tree
├── greeks/        # Generic autodiff wrappers
├── market/        # MarketData container, scenario definitions
├── risk/          # Scenario generation, curve shocks, VaR/ES
└── portfolio/     # vmap batch pricing
```

## Dependency Roles

| Package | Role |
|---|---|
| **equinox** | All data structures. `eqx.Module` = frozen dataclass + JAX pytree. Use `eqx.filter_jit`/`eqx.filter_grad` for modules with static fields. |
| **diffrax** | SDE path simulation for Monte Carlo. Provides Euler-Maruyama, Milstein, and higher-order methods with full differentiability. |
| **optimistix** | Root-finding for curve bootstrapping. Supports implicit differentiation through the solve. |
| **optax** | Gradient-based optimization for model calibration. |
| **lineax** | Tridiagonal linear solvers for Crank-Nicolson PDE stepping. |
| **jaxtyping** | Shape/dtype annotations: `Float[Array, "n_paths n_steps"]`. |

## Data Flow

```
Instrument (pytree)  ─┐
                       ├─→  pricing_fn()  ─→  price (scalar)
Market Data (arrays)  ─┘        │
                                │
                           jax.grad()  ─→  Greeks
                           jax.vmap()  ─→  Portfolio prices
                           jax.jit()   ─→  Compiled execution
```
