# Binomial Trees

VALAX implements the Cox-Ross-Rubinstein (CRR) binomial tree for European and American option pricing.

## Usage

```python
import jax.numpy as jnp
from valax.instruments import EuropeanOption
from valax.pricing.lattice import binomial_price, BinomialConfig

option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)

# European
euro = binomial_price(option, jnp.array(100.0), jnp.array(0.20),
                       jnp.array(0.05), jnp.array(0.02),
                       BinomialConfig(n_steps=500, american=False))

# American
put = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)
amer = binomial_price(put, jnp.array(100.0), jnp.array(0.20),
                       jnp.array(0.05), jnp.array(0.0),
                       BinomialConfig(n_steps=500, american=True))
```

## CRR Parameters

At each time step $\Delta t = T/N$:

$$u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{(r-q)\Delta t} - d}{u - d}$$

The tree recombines: an up move followed by a down move reaches the same node as down-then-up.

## American Exercise

When `american=True`, at each node the algorithm takes the maximum of the continuation value and the intrinsic value:

$$V_i = \max\left(\text{disc} \cdot [p \cdot V_{i+1}^u + (1-p) \cdot V_{i+1}^d],\; \text{intrinsic}_i\right)$$

Key properties:

- **American put >= European put** — early exercise premium exists.
- **American call without dividends = European call** — early exercise is never optimal.
- **American call with dividends >= European call** — may be optimal to exercise before ex-dividend.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `n_steps` | 200 | Number of time steps in the tree |
| `american` | `False` | Enable American early exercise |

## Greeks via Autodiff

```python
import jax

delta = jax.grad(lambda s: binomial_price(
    option, s, vol, rate, div, config
))(spot)
```

## Implementation Notes

- The backward induction uses `jax.lax.scan` with fixed-shape arrays for JIT compatibility. At each step, `values[j] = disc * (p * values[j+1] + (1-p) * values[j])` is computed over the full array; unused trailing entries are harmless.
- European prices converge to Black-Scholes within 0.5% at 500 steps.
