# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VALAX is a quantitative finance valuation engine similar to QuantLib, built entirely on JAX and its ecosystem. It leverages JAX's automatic differentiation for Greeks computation, JIT compilation for performance, `vmap` for portfolio-level vectorized pricing, and GPU/TPU accelerator support. The library integrates with the ML ecosystem for model calibration and surrogate pricing.

## Development Commands

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_pricing/test_black_scholes.py

# Run a single test
pytest tests/test_pricing/test_black_scholes.py::test_call_price -v

# Run benchmarks
pytest --benchmark-only

# Environment: pyenv virtualenv "valax" on Python 3.13.12
pyenv activate valax
```

## Architecture

### Core Design: Pure Functional on JAX Pytrees

Every data structure is an `equinox.Module` (frozen dataclass, auto-registered as a JAX pytree). There is no mutable state, no observer pattern, no lazy evaluation. This is a hard constraint imposed by JAX — all functions must be pure for `jit`/`grad`/`vmap` to work.

**Instruments are data-only pytrees** — they describe a contract (strike, expiry, notional) but carry no pricing logic. Pricing is a standalone pure function: `price = pricing_fn(instrument, market_data, [model_params])`. This is a deliberate departure from QuantLib's `instrument.setPricingEngine(engine)` pattern.

**Greeks come from autodiff, not finite differences.** `jax.grad(pricing_fn)` returns exact sensitivities. Higher-order Greeks (gamma, vanna, volga) use nested `jax.grad`. Differentiating through a `DiscountCurve` pytree gives key-rate durations for free. This is the library's primary advantage over QuantLib.

**Dates are integer ordinals** (days since epoch) stored in JAX arrays. This makes all date arithmetic JIT-compatible via pure integer ops. Calendars use precomputed boolean holiday arrays, not runtime lookups.

### Dependency Roles

| Package | Role in VALAX |
|---|---|
| **equinox** | All data structures (`eqx.Module`); use `eqx.filter_jit`, `eqx.filter_grad` for modules with static fields |
| **diffrax** | SDE path simulation for Monte Carlo (GBM, Heston, etc.) — do not hand-write Euler-Maruyama |
| **optimistix** | Root-finding for curve bootstrapping; least-squares for model calibration |
| **optax** | Gradient-based optimization for calibration and ML training loops |
| **lineax** | Structured linear solvers for PDE/finite-difference methods |
| **jaxtyping + beartype** | Shape/dtype annotations on array arguments (e.g., `Float[Array, "n_paths n_steps"]`) |

These are all from the same ecosystem (Patrick Kidger's JAX libraries) and share consistent conventions.

### Package Layout

```
valax/
├── core/          # Type aliases, constants, shared utilities
├── dates/         # Day count conventions, calendars, schedule generation
├── curves/        # DiscountCurve, interpolation, bootstrapping
├── instruments/   # Data-only pytrees: options, bonds, swaps, caps, swaptions
├── models/        # Stochastic process definitions: BS, Heston, Hull-White, SABR
├── pricing/
│   ├── analytic/  # Closed-form: Black-Scholes, Black-76, Bachelier, bonds
│   ├── mc/        # Monte Carlo: path generation (diffrax), payoffs, variance reduction
│   ├── pde/       # Finite difference solvers (Crank-Nicolson)
│   └── lattice/   # Tree methods (CRR binomial)
├── greeks/        # Generic autodiff wrappers around jax.grad/jacobian
├── calibration/   # Model fitting: loss functions, optimizer wrappers
├── market/        # MarketData container (curves + vol surfaces + spots)
├── ml/            # Neural surrogate pricers, learned volatility surfaces
└── portfolio/     # vmap batch pricing, risk aggregation (VaR, ES)
```

## Conventions

### Equinox Modules
- All structured data (instruments, curves, models, market data) must be `eqx.Module` subclasses
- Mark non-differentiable fields with `eqx.field(static=True)` (e.g., `is_call: bool`, `interp_method: str`)
- Use `eqx.filter_jit` and `eqx.filter_grad` instead of raw `jax.jit`/`jax.grad` when operating on modules with static fields

### Type Annotations
- Annotate all array arguments with jaxtyping shapes: `Float[Array, ""]` (scalar), `Float[Array, "n"]` (vector), `Float[Array, "n m"]` (matrix)
- Use named dimensions for clarity: `Float[Array, "n_paths n_steps"]` not `Float[Array, "a b"]`

### Pricing Functions
- Signature: `def price(instrument, *market_args) -> Float[Array, ""]`
- Must be pure — no side effects, no global state, no print statements inside JIT-traced code
- Must be differentiable — avoid `jnp.where` with integer indexing; use smooth approximations for discontinuous payoffs (e.g., `jax.nn.sigmoid` instead of Heaviside)

### Testing
- Validate Greeks against known closed-form solutions (not just finite differences)
- Use `hypothesis` for property-based tests (e.g., put-call parity holds for random inputs)
- MC tests: assert convergence within 2 standard errors of analytical solutions
- Tests mirror src structure under `tests/`

## What NOT to Do

- **No mutable state**: Never use Python lists/dicts that change during computation. No `self.cache`, no `__setattr__`
- **No class hierarchies for dispatch**: Don't create `PricingEngine` base classes. Use plain functions. If you need dispatch, use `functools.singledispatch` or pattern match on module type
- **No stdlib `datetime`**: Dates inside JAX-traced code must be integer ordinals. Use `datetime` only at the boundary (user-facing conversion utilities)
- **No hand-rolled SDE solvers**: Use diffrax. It handles higher-order methods, adaptive stepping, and differentiability
- **No scipy inside JIT**: Use optimistix/lineax instead of scipy.optimize/scipy.linalg — scipy is not JAX-traceable
