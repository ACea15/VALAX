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

# VALAX style: one front door over (instrument, model, market)
pv = price(option, BlackScholesModel(), market).price
# ...or a kernel directly
pv = black_scholes_price(option, spot, vol, rate, dividend)
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

## Invariants for contributors

The [seven architectural choices](../positioning/index.md#4-the-seven-choices-valax-commits-to-and-what-each-one-buys) only hold if every new module keeps them. These are acceptance criteria, not style guidelines; the detailed rules, with examples, are in [`CONTRIBUTING.md` § 2](https://github.com/ACea15/VALAX/blob/main/CONTRIBUTING.md#2-architectural-rules).

1. **Every pricing function is `jax.grad`-able.** A workaround (`stop_gradient`, a finite-difference fallback) breaks the Greek-consistency property, and the change should not land.
2. **Every domain type is an `equinox.Module` pytree.** Anything that does not flatten / unflatten under `jax.tree_util` cannot be `vmap`-ed across scenarios and instruments.
3. **No global state, no caches that survive between calls, no observer pattern.** Determinism is by construction; any leaked state means yesterday's risk run might not replay tomorrow.
4. **Dates inside traced code are integer ordinals.** `datetime.date` appears at the user-facing boundary only.
5. **No `scipy` inside JIT-traced code.** Use `optimistix` / `lineax` / `jax.scipy`, or the trader-hot-path / batch-risk equivalence silently breaks.
6. **New instruments declare their risk-factor consumption** in [Risk Factors § 5](../risk-factors.md#5-instrument-factor-matrix). This keeps the bucketing layer of [Theory § 7.8](../theory/risk-measures.md#78-risk-bucketing-linear-and-jacobian-transformations) honest at scale.

The JAX idioms that satisfy these invariants (static trace-time dispatch, the double-where autodiff-safety pattern, JIT cache semantics, autodiff through autodiff) are catalogued in [JAX Patterns](jax-patterns.md). The test-layer counterpart is [Validation Strategy](validation-strategy.md). If an invariant is hard to satisfy for a new module, the module is fighting the architecture: redesign the module rather than weaken the architecture.

## Package Structure

```
valax/
├── core/          # MarketState, price() front door, resolve / bind, pricer registry
├── dates/         # Day counts, calendars, schedules, accrual
├── curves/        # DiscountCurve, CurveGraph, joint multi-curve bootstrap
├── instruments/   # Data-only pytrees (options, bonds, swaps, legs, FX, credit)
├── models/        # Model parameters (BS, Heston, SABR, HW, G2++, Cheyette, LMM)
├── calibration/   # Solver settings, levels, CalibrationResult, chains + nodes, per-model fits
├── pricing/
│   ├── analytic/  # Closed forms and semi-analytics
│   ├── mc/        # Monte Carlo engine (diffrax SDE paths)
│   ├── pde/       # Finite differences (1D, ADI 2D)
│   └── lattice/   # Binomial and Hull-White trees
├── greeks/        # Market / model greeks, IFT adjoints, quote risk through chains
├── surfaces/      # Vol surfaces and cubes (constant, grid, SABR, SVI, optionlet)
├── market/        # QuoteSet, provenance, keyed scenarios, synthetic markets
├── risk/          # Keyed shocks, repricing, sensitivities, VaR/ES, bucketing
└── portfolio/     # Homogeneous vmap batch pricing (legacy)
```

## Dependency Roles

| Package | Role |
|---|---|
| **equinox** | All data structures. `eqx.Module` = frozen dataclass + JAX pytree. Use `eqx.filter_jit`/`eqx.filter_grad` for modules with static fields. |
| **diffrax** | SDE path simulation for Monte Carlo. Provides Euler-Maruyama, Milstein, and higher-order methods with full differentiability. |
| **optimistix** | Every calibration solve (root-find, least squares, minimise) through `valax.calibration.run_solver`. Its implicit adjoint is the autodiff path of quote risk. |
| **optax** | First-order optimisers, selectable as a calibration `SolverSettings.method`. |
| **lineax** | Tridiagonal linear solvers for Crank-Nicolson PDE stepping. |
| **jaxtyping** | Shape/dtype annotations: `Float[Array, "n_paths n_steps"]`. |

## Data Flow

```
quotes q, fixed inputs p (QuoteSet)
      │  calibrate_market(chain, q, p)         curves → spots → vol objects
      ▼
MarketState ─────────────┐
      │  calibrate_model(node, q, market, p)   per basket / underlying
      ▼                  │
model ───────────────────┤
Instrument (pytree) ─────┤
                         ▼
          price(instrument, model, market)  ─→  PricingResult
                         │
                   eqx.filter_grad  ─→  market / model cotangents
                         │
                   chain_adjoint    ─→  quote risk q̄, fixed-input risk p̄
                   (or quote_risk_autodiff, reconciled)
```

`jax.jit`, `jax.vmap` and `jax.grad` apply at every stage: the calibration is
traceable, so a scenario batch of quote sets recalibrates and reprices in one
compiled program, and a book's cotangents add before a single sweep. A
constant-parameter Black-Scholes market is the degenerate case of the same
flow (`MarketState.flat`, or a chain with one explicit level). The full design
— contracts, decisions and how to extend the framework — is on
[Calibration, Pricing & Risk](calibration-pricing-risk.md).
