# VALAX

A quantitative finance valuation and risk engine built entirely on [JAX](https://github.com/jax-ml/jax).

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
- **Equity options** — European/American vanillas, barriers, Asians, lookbacks, digitals, variance swaps, compound/chooser options, autocallables, cliquets, worst-of baskets, spread options
- **Fixed income** — zero coupon, fixed/floating, callable/puttable and convertible bonds; caps/floors; IRS, OIS, CMS, cross-currency and total-return swaps; European and Bermudan swaptions; range accruals
- **Credit** — CDS, CDO tranches
- **FX** — forwards, FX swaps, vanilla/barrier options, quantos, TARFs
- **Inflation** — zero-coupon and year-on-year inflation swaps, inflation caps/floors

### Models
- Black-Scholes-Merton (single- and multi-asset GBM)
- Heston stochastic volatility
- SABR
- Local volatility (Dupire) and stochastic-local volatility (SLV)
- Hull-White one-factor
- LIBOR Market Model (LMM)

### Pricing Engines
| Engine | Description |
|--------|-------------|
| **Analytic** | Black-Scholes, Black-76, Bachelier, bond pricing, caplet/swaption formulas |
| **Monte Carlo** | GBM, Heston, SABR, local-vol and SLV path generation via [diffrax](https://github.com/patrick-kidger/diffrax), LMM simulation, LSM for Bermudans |
| **PDE** | Crank-Nicolson finite difference solver |
| **Lattice** | CRR binomial tree (European and American options), Hull-White trinomial tree |

### Supporting Infrastructure
- **Curves** — discount curve construction with log-linear interpolation, single- and multi-curve bootstrapping, inflation and survival curves
- **Surfaces** — SVI, SABR and grid volatility surfaces, leverage surfaces for SLV
- **Calibration** — gradient-based calibration for Heston, SABR and SLV (optimistix/optax)
- **Market** — market data and scenario containers, synthetic market data generation
- **Risk** — VaR, bucketed sensitivities, bucketing, PCA-based rates factor models
- **Dates** — JIT-compatible integer ordinal dates, day count conventions (Act/360, Act/365, 30/360), schedule generation
- **Greeks** — generic autodiff wrappers (`greeks`, `greek`) for any pricing function
- **Portfolio** — `vmap`-based batch pricing and risk aggregation

## Architecture

Every data structure is an [`equinox.Module`](https://github.com/patrick-kidger/equinox) — a frozen dataclass that is automatically registered as a JAX pytree. There is no mutable state, no observer pattern, no lazy evaluation.

```
valax/
├── core/          # Type aliases, constants, arbitrage diagnostics
├── dates/         # Day counts, schedule generation
├── curves/        # Discount/inflation/survival curves, bootstrapping, multi-curve
├── surfaces/      # SVI, SABR and grid vol surfaces, SLV leverage
├── instruments/   # Data-only pytrees: options, bonds, rates, credit, FX, inflation
├── models/        # Black-Scholes, Heston, SABR, local vol, SLV, Hull-White, LMM
├── calibration/   # Heston/SABR/SLV calibration, losses, parameter transforms
├── market/        # Market data, scenarios, synthetic market generation
├── pricing/
│   ├── analytic/  # Closed-form solutions
│   ├── mc/        # Monte Carlo (diffrax-based path generation)
│   ├── pde/       # Finite difference (Crank-Nicolson)
│   └── lattice/   # Binomial trees (CRR), Hull-White trinomial tree
├── greeks/        # Autodiff wrappers
├── risk/          # VaR, sensitivities, bucketing, PCA factor models
└── portfolio/     # vmap batch pricing, risk aggregation
```

> **Note:** importing `valax` enables 64-bit precision globally via
> `jax.config.update("jax_enable_x64", True)` — a process-wide setting that
> affects all JAX code in the same process.

## Examples

Runnable scripts in `examples/` demonstrate the full library with synthetic market data. Use `# %%` cell markers for interactive execution in VS Code or PyCharm.

| Example | Topics |
|---------|--------|
| `01_equity_options.py` | Black-Scholes pricing, all Greeks via autodiff, implied vol, portfolio vmap, JIT |
| `02_sabr_smile.py` | SABR smile generation, parameter sensitivity, calibration (LM/BFGS) |
| `03_fixed_income.py` | Discount curves, bond pricing, YTM, duration/convexity/KRD via autodiff |
| `04_rates_derivatives.py` | Caplet/floor pricing, cap strips, swaps, swaptions, rate Greeks |
| `05_monte_carlo.py` | GBM/Heston/SABR paths, convergence, Asian + barrier exotics |
| `06_pde_and_lattice.py` | Crank-Nicolson PDE, binomial trees, American options, method comparison |
| `10_unified_market_tiers.py` | One option on three markets — flat Black-Scholes, bootstrapped curve + SABR smile, + calibrated Heston — through the same quotes → calibrate → price → quote-risk workflow |
| `11_unified_multicurve_rates.py` | Synthetic OIS + 3M quotes → joint bootstrap → swap book → book quote risk in one adjoint sweep (vs autodiff and FD), single-curve limit |
| `12_unified_equity_smile_and_model.py` | Noisy equity quotes → curve, SABR smiles, Heston → European / barrier / Asian prices → Monte Carlo quote risk with fixed-input sensitivities |
| `13_unified_quote_scenarios.py` | Correlated quote scenarios → `vmap` recalibration + repricing compiled once → P&L explain vs first-order quote risk → VaR / ES |
| `14_unified_rates_per_trade_model.py` | Curves → swaption cube → per-trade Hull–White σ(t) co-terminal bootstrap → Bermudan through `price()` (PDE) → quote risk and vega profile |
| `15_unified_rates_caps_optionlet_chain.py` | Cap strip → optionlet SABR as chain nodes → caps / floors through `price()` → quote risk |
| `16_unified_calibration_diagnostics.py` | Bad quotes named by `check()` / `failed_labels()`, conditioning alerts, engine provenance |
| `17_unified_slv_fixed_point.py` | SLV leverage as a fixed-point level, LV / Heston / SLV barrier, risk by AD |
| `18_unified_rates_per_trade_g2pp_cheyette.py` | Per-trade G2++ and Cheyette σ(t), exact vs engine-consistent calibration, model risk vs Hull–White |
| `19_unified_book_workflow.py` | Rates book end to end: quotes → curves → `price()` → `market_risk` → bond kernels → PCA + basis `KeyedScenario`s, P&L explain, VaR / ES |
| `20_unified_fx_inflation_xccy.py` | FX forward / option, cross-currency and basis swaps, ZCIS / YoY / caps on one multi-currency market; parity checks; risk exactly zero outside declared entries |
| `21_unified_rates_engine_ladder.py` | One European and one Bermudan swaption under Hull–White, G2++, Cheyette 1F/2F and LMM by every engine; engine agreement, the `auto` policy, curve risk across engines |
| `22_unified_multi_asset_exotics.py` | Spread / worst-of under multi-asset GBM; barrier / Asian / lookback / variance swap under BS, Heston, LV, SLV; spot risk; a kernel-built dual digital |

**Two layers, both supported.** Examples 01–06 call the pricing kernels
directly. That layer is public API, not legacy, and it is the one to compose
when `price()` does not express what you need. Examples 10–22 use the unified
`price(instrument, model, market)` contract on a keyed `MarketState`. They run
end to end on synthetic *quotes* generated from a known true market
(`valax.market.rates_snapshot` / `equity_snapshot`) and assert their own
claims; `tests/test_examples/` runs them in fast mode
(`VALAX_EXAMPLES_FAST=1`). Examples 07–09 were retired with the `MarketData`
deprecation (see the *Retired examples* table in `docs/examples.md`).

`examples/comparisons/` additionally contains side-by-side comparison scripts
(e.g. against QuantLib) covering options, fixed income, SABR, caps/swaptions,
Monte Carlo, PDE/lattice, Heston smiles, local-vol PDE, and
Hull-White, G2++ and Cheyette across MC, tree, PDE and analytic engines.

```bash
python examples/01_equity_options.py
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
mkdocs serve --livereload
```

## License

Apache 2.0
