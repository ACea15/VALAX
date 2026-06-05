# Getting Started

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.11. This installs JAX (CPU), equinox, diffrax, and all other dependencies.

For GPU support, install JAX with CUDA separately first:

```bash
pip install jax[cuda12]
pip install -e ".[dev]"
```

## First Steps

### Price a European Option

```python
import jax.numpy as jnp
from valax.instruments import EuropeanOption
from valax.pricing.analytic import black_scholes_price

option = EuropeanOption(
    strike=jnp.array(100.0),
    expiry=jnp.array(1.0),  # 1 year
    is_call=True,
)

price = black_scholes_price(
    option,
    spot=jnp.array(100.0),
    vol=jnp.array(0.20),
    rate=jnp.array(0.05),
    dividend=jnp.array(0.02),
)
print(f"Price: {price:.4f}")
```

### Compute Greeks

```python
from valax.greeks import greeks

g = greeks(black_scholes_price, option,
           jnp.array(100.0), jnp.array(0.20),
           jnp.array(0.05), jnp.array(0.02))

for name, value in g.items():
    print(f"{name:>12}: {value:.6f}")
```

### Price a Portfolio

```python
import jax
from valax.portfolio import batch_price

n = 10_000
options = EuropeanOption(
    strike=jnp.linspace(80.0, 120.0, n),
    expiry=jnp.ones(n),
    is_call=True,
)

prices = batch_price(
    black_scholes_price, options,
    spots=jnp.full(n, 100.0),
    vols=jnp.full(n, 0.20),
    rates=jnp.full(n, 0.05),
    dividends=jnp.full(n, 0.02),
)
print(f"Priced {n} options, shape: {prices.shape}")
```

### Price a Bond

```python
import jax.numpy as jnp
from valax.dates import ymd_to_ordinal, generate_schedule
from valax.instruments import FixedRateBond
from valax.curves import DiscountCurve
from valax.pricing.analytic import fixed_rate_bond_price, modified_duration, yield_to_maturity

# 5-year, 4% semi-annual bond
ref = ymd_to_ordinal(2025, 1, 1)
bond = FixedRateBond(
    payment_dates=generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2),
    settlement_date=ref,
    coupon_rate=jnp.array(0.04),
    face_value=jnp.array(100.0),
    frequency=2,
)

# Flat 5% discount curve
pillars = jnp.array([int(ymd_to_ordinal(2025+i, 1, 1)) for i in range(6)], dtype=jnp.int32)
times = (pillars - int(ref)).astype(jnp.float64) / 365.0
curve = DiscountCurve(
    pillar_dates=pillars,
    discount_factors=jnp.exp(-0.05 * times),
    reference_date=ref,
)

price = fixed_rate_bond_price(bond, curve)
ytm = yield_to_maturity(bond, price)
dur = modified_duration(bond, ytm)
print(f"Price: {price:.4f}, YTM: {ytm:.4f}, Duration: {dur:.4f}")
```

### Monte Carlo Pricing

```python
from valax.models import BlackScholesModel
from valax.pricing.mc import mc_price, MCConfig

model = BlackScholesModel(
    vol=jnp.array(0.20),
    rate=jnp.array(0.05),
    dividend=jnp.array(0.02),
)

price = mc_price(
    option, spot=jnp.array(100.0), model=model,
    config=MCConfig(n_paths=100_000, n_steps=100),
    key=jax.random.PRNGKey(42),
)
print(f"MC Price: {price:.4f}")
```

### American Options via Binomial Tree

```python
from valax.pricing.lattice import binomial_price, BinomialConfig

put = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)

american_price = binomial_price(
    put, jnp.array(100.0), jnp.array(0.20), jnp.array(0.05), jnp.array(0.0),
    config=BinomialConfig(n_steps=500, american=True),
)
print(f"American Put: {american_price:.4f}")
```

### Test with Synthetic Data

VALAX ships a complete synthetic-market generator so any component can
be exercised end-to-end without an external data source. Every random
draw goes through a `SeedRegistry`, so the same `(master_seed,
library_version)` produces identical numbers on any machine.

```python
import jax.numpy as jnp
import valax
from valax.market import (
    SeedRegistry, SyntheticMarketConfig,
    sample_market_with_correlation, sample_option_portfolio,
    OptionPortfolioSpec,
)
from valax.portfolio import batch_price
from valax.pricing.analytic import black_scholes_price

registry = SeedRegistry(master_seed=20260101,
                        library_version=valax.__version__)
cfg = SyntheticMarketConfig(n_assets=3)

# Stage 1: ground truth (snapshot + correlation)
md, corr = sample_market_with_correlation(registry, cfg)

# Stage 4: portfolio
port = sample_option_portfolio(registry, md,
                                OptionPortfolioSpec(n_per_asset=4,
                                                    call_probability=1.0))
calls, idx = port["calls"]

# Stage 5: batched analytic pricing
n = calls.strike.shape[0]
prices = batch_price(
    black_scholes_price, calls,
    md.spots[idx], md.vols[idx],
    jnp.full((n,), 0.03), md.dividends[idx],
)
print(f"portfolio PV: {float(jnp.sum(prices)):.4f}")
```

See [User Guide → Synthetic Market Data](guide/synthetic_market.md) for
the full six-stage workflow (calibration on noisy quotes, market tapes,
risk scenarios, arbitrage stress tests) and [`examples/08_end_to_end_workflow.py`](examples.md#end-to-end-workflow)
for the runnable counterpart.

## Running Tests

```bash
# All tests
pytest

# Single file
pytest tests/test_pricing/test_black_scholes.py -v

# Single test
pytest tests/test_greeks/test_autodiff.py::TestBSAnalyticalGreeks::test_delta -v

# Just the synthetic market / arbitrage / golden suites
pytest tests/test_market/ -v
pytest -m arbitrage tests/

# Regenerate golden datasets after an intentional numerical change
REGEN_GOLDEN=1 python scripts/regen_goldens.py
```
