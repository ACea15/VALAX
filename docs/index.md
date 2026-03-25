# VALAX

**JAX-native quantitative finance valuation engine.**

VALAX is a quantitative finance library built entirely on JAX and its ecosystem. It provides option pricing, fixed income analytics, and risk computation with automatic differentiation for Greeks — no finite differences needed.

## Key Features

- **Autodiff Greeks** — `jax.grad` gives exact delta, gamma, vega, vanna, volga, rho and more. Higher-order Greeks via nested differentiation.
- **Multiple pricing methods** — Analytical (Black-Scholes, Black-76, Bachelier), Monte Carlo (GBM, Heston via diffrax), PDE (Crank-Nicolson), and lattice (CRR binomial with American exercise).
- **Portfolio vectorization** — `jax.vmap` prices thousands of instruments in a single call.
- **GPU/TPU ready** — All code runs on accelerators with zero changes.
- **ML integration** — Gradient-based model calibration, neural surrogate pricers.
- **Pure functional** — No mutable state. Every data structure is a JAX pytree via equinox.

## Quick Example

```python
import jax.numpy as jnp
from valax.instruments import EuropeanOption
from valax.pricing.analytic import black_scholes_price
from valax.greeks import greeks

# Define an ATM European call
option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)

# Price it
price = black_scholes_price(option, spot=jnp.array(100.0), vol=jnp.array(0.20),
                             rate=jnp.array(0.05), dividend=jnp.array(0.02))

# Get all Greeks at once via autodiff
g = greeks(black_scholes_price, option, jnp.array(100.0), jnp.array(0.20),
           jnp.array(0.05), jnp.array(0.02))

print(f"Price: {g['price']:.4f}")
print(f"Delta: {g['delta']:.4f}")
print(f"Gamma: {g['gamma']:.6f}")
print(f"Vega:  {g['vega']:.4f}")
```

## Why JAX?

Traditional quant libraries compute Greeks via finite differences (bump-and-reprice) or hand-derived closed-form expressions. VALAX uses JAX's automatic differentiation instead:

| Approach | Effort | Accuracy | Speed |
|---|---|---|---|
| Finite differences | Low | O(h) error, tuning needed | 2N+1 evaluations for N Greeks |
| Closed-form | High (per model) | Exact | Fast, but limited to simple models |
| **Autodiff (VALAX)** | **Zero** | **Machine precision** | **One backward pass for all Greeks** |

This means every pricing function — analytical, Monte Carlo, PDE, or lattice — automatically supports Greeks with no additional code.
