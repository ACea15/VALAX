# VALAX

A quantitative finance valuation engine built entirely on [JAX](https://github.com/jax-ml/jax).

VALAX takes a functional approach to derivatives pricing: instruments are data-only pytrees, pricing is done by pure functions, and Greeks come from automatic differentiation — not finite differences.

## Why JAX?

- **Exact Greeks via `jax.grad`** — delta, gamma, vanna, volga, key-rate durations, all from autodiff. No bumping.
- **JIT compilation** — pricing functions compile to XLA for native-speed execution.
- **`vmap` for portfolios** — vectorize a single-instrument pricer across thousands of trades with one line.
- **GPU/TPU support** — the same code runs on accelerators without modification.
- **ML integration** — calibrate models with gradient-based optimizers, train neural surrogate pricers.

## Quick Start

```bash
pip install -e ".[dev]"
```

### Price a European option

```python
import jax.numpy as jnp
from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.greeks.autodiff import greeks

# Define the contract
option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)

# Price it
price = black_scholes_price(option, spot=jnp.array(100.0), vol=jnp.array(0.20),
                            rate=jnp.array(0.05), dividend=jnp.array(0.02))

# Get all Greeks via autodiff — no finite differences
g = greeks(black_scholes_price, option, jnp.array(100.0), jnp.array(0.20),
           jnp.array(0.05), jnp.array(0.02))
# g["delta"], g["gamma"], g["vega"], g["vanna"], g["volga"], ...
```

### Vectorize across a portfolio

```python
from valax.portfolio.batch import batch_price

options = EuropeanOption(
    strike=jnp.array([95.0, 100.0, 105.0]),
    expiry=jnp.array([0.5, 1.0, 1.5]),
    is_call=True,
)
prices = batch_price(black_scholes_price, options,
                     jnp.array([100.0]*3), jnp.array([0.20]*3),
                     jnp.array([0.05]*3), jnp.array([0.02]*3))
```

## What's Included

### Instruments
- **Equity options** — European calls/puts
- **Fixed income** — zero coupon bonds, fixed/floating rate bonds, caps/floors, swaptions

### Models
- Black-Scholes-Merton
- Heston stochastic volatility
- LIBOR Market Model (LMM)

### Pricing Engines
| Engine | Description |
|--------|-------------|
| **Analytic** | Black-Scholes, Black-76, Bachelier, bond pricing, caplet/swaption formulas |
| **Monte Carlo** | GBM and Heston path generation via [diffrax](https://github.com/patrick-kidger/diffrax), LMM simulation |
| **PDE** | Crank-Nicolson finite difference solver |
| **Lattice** | CRR binomial tree (European and American options) |

### Supporting Infrastructure
- **Curves** — discount curve construction with log-linear interpolation and bootstrapping
- **Dates** — JIT-compatible integer ordinal dates, day count conventions (Act/360, Act/365, 30/360), schedule generation
- **Greeks** — generic autodiff wrappers (`greeks`, `greek`) for any pricing function
- **Portfolio** — `vmap`-based batch pricing and risk aggregation

## Architecture

Every data structure is an [`equinox.Module`](https://github.com/patrick-kidger/equinox) — a frozen dataclass that is automatically registered as a JAX pytree. There is no mutable state, no observer pattern, no lazy evaluation.

```
valax/
├── core/          # Type aliases, constants
├── dates/         # Day counts, schedule generation
├── curves/        # Discount curves, interpolation, bootstrapping
├── instruments/   # Data-only pytrees: options, bonds, caps, swaptions
├── models/        # Black-Scholes, Heston, LMM
├── pricing/
│   ├── analytic/  # Closed-form solutions
│   ├── mc/        # Monte Carlo (diffrax-based path generation)
│   ├── pde/       # Finite difference (Crank-Nicolson)
│   └── lattice/   # Binomial trees (CRR)
├── greeks/        # Autodiff wrappers
└── portfolio/     # vmap batch pricing, risk aggregation
```

## Key Dependencies

| Package | Role |
|---------|------|
| [equinox](https://github.com/patrick-kidger/equinox) | Pytree dataclasses for all structured data |
| [diffrax](https://github.com/patrick-kidger/diffrax) | SDE solvers for Monte Carlo path simulation |
| [optimistix](https://github.com/patrick-kidger/optimistix) | Root-finding and least-squares for calibration |
| [optax](https://github.com/google-deepmind/optax) | Gradient-based optimization |
| [lineax](https://github.com/patrick-kidger/lineax) | Linear solvers for PDE methods |
| [jaxtyping](https://github.com/patrick-kidger/jaxtyping) | Shape/dtype annotations |

## Development

```bash
# Run tests
pytest

# Run a specific test
pytest tests/test_pricing/test_black_scholes.py -v

# Build docs
mkdocs build --strict

# Serve docs locally
mkdocs serve
```

## License

Apache 2.0
