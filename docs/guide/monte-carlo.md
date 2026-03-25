# Monte Carlo Pricing

VALAX uses [diffrax](https://docs.kidger.site/diffrax/) for SDE path simulation, giving access to higher-order solvers and full differentiability through the simulation.

## Basic Usage

```python
import jax
import jax.numpy as jnp
from valax.instruments import EuropeanOption
from valax.models import BlackScholesModel
from valax.pricing.mc import mc_price, MCConfig

option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
model = BlackScholesModel(vol=jnp.array(0.20), rate=jnp.array(0.05), dividend=jnp.array(0.02))

price = mc_price(
    option, spot=jnp.array(100.0), model=model,
    config=MCConfig(n_paths=100_000, n_steps=100),
    key=jax.random.PRNGKey(42),
)
```

## Price with Standard Error

```python
from valax.pricing.mc import mc_price_with_stderr

price, stderr = mc_price_with_stderr(
    option, jnp.array(100.0), model,
    MCConfig(n_paths=100_000, n_steps=100),
    jax.random.PRNGKey(42),
)
print(f"Price: {price:.4f} +/- {stderr:.4f}")
```

## Models

### Geometric Brownian Motion

$$dS = (r - q) S\, dt + \sigma S\, dW$$

```python
from valax.models import BlackScholesModel

model = BlackScholesModel(
    vol=jnp.array(0.20),
    rate=jnp.array(0.05),
    dividend=jnp.array(0.02),
)
```

### Heston Stochastic Volatility

$$dS = (r - q) S\, dt + \sqrt{V} S\, dW_1$$
$$dV = \kappa(\theta - V)\, dt + \xi \sqrt{V}\, dW_2$$
$$\text{Corr}(dW_1, dW_2) = \rho$$

```python
from valax.models import HestonModel

model = HestonModel(
    v0=jnp.array(0.04),       # initial variance (vol = 20%)
    kappa=jnp.array(2.0),     # mean reversion speed
    theta=jnp.array(0.04),    # long-run variance
    xi=jnp.array(0.3),        # vol of vol
    rho=jnp.array(-0.7),      # spot-vol correlation
    rate=jnp.array(0.05),
    dividend=jnp.array(0.0),
)
```

## Payoff Functions

### European

```python
from valax.pricing.mc.payoffs import european_payoff
price = mc_price(option, spot, model, config, key, payoff_fn=european_payoff)
```

### Asian (Arithmetic Average)

```python
from valax.pricing.mc.payoffs import asian_payoff
price = mc_price(option, spot, model, config, key, payoff_fn=asian_payoff)
```

### Barrier

```python
from valax.pricing.mc.payoffs import barrier_payoff
from functools import partial

# Down-and-out call with smoothing for differentiability
payoff = partial(barrier_payoff, barrier=jnp.array(80.0),
                  is_up=False, is_knock_in=False, smoothing=1.0)
price = mc_price(option, spot, model, config, key, payoff_fn=payoff)
```

!!! tip "Smoothing for Greeks"
    Barrier payoffs have discontinuities that break pathwise differentiation. Set `smoothing > 0` to use a sigmoid approximation, enabling autodiff Greeks at the cost of a small bias.

## MC Greeks via Autodiff

JAX differentiates through the entire simulation — path generation, payoff evaluation, and discounting:

```python
# Delta through MC
delta = jax.grad(lambda s: mc_price(option, s, model, config, key))(jnp.array(100.0))

# Vega through MC
vega = jax.grad(lambda v: mc_price(
    option, spot,
    BlackScholesModel(vol=v, rate=model.rate, dividend=model.dividend),
    config, key
))(jnp.array(0.20))
```
