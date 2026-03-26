# Model Calibration

VALAX calibrates model parameters to market data using gradient-based optimization. Because all pricing functions are differentiable via JAX, the optimizer gets exact Jacobians for free — no finite-difference bumping.

## SABR Calibration

Fit SABR parameters ($\alpha$, $\rho$, $\nu$) to an observed volatility smile. Beta is typically fixed.

```python
import jax.numpy as jnp
from valax.calibration import calibrate_sabr

# Market data: strikes and observed implied vols
strikes = jnp.array([80., 85., 90., 95., 100., 105., 110., 115., 120.])
market_vols = jnp.array([0.28, 0.26, 0.24, 0.22, 0.21, 0.205, 0.21, 0.22, 0.235])
forward = jnp.array(100.0)
expiry = jnp.array(1.0)

# Calibrate with beta fixed at 0.5
fitted, sol = calibrate_sabr(
    strikes, market_vols, forward, expiry,
    fixed_beta=jnp.array(0.5),
)

print(f"alpha={float(fitted.alpha):.4f}")
print(f"rho={float(fitted.rho):.4f}")
print(f"nu={float(fitted.nu):.4f}")
```

### Solvers

Three backends are available:

| Solver | Method | Best for |
|--------|--------|----------|
| `"levenberg_marquardt"` | Least-squares (optimistix) | Default — fast, exploits Jacobian |
| `"bfgs"` | Quasi-Newton (optimistix) | Fallback when LM struggles |
| `"optax_adam"` | Gradient descent (optax) | Research / experimentation |

```python
# Use BFGS instead
fitted, sol = calibrate_sabr(
    strikes, market_vols, forward, expiry,
    fixed_beta=jnp.array(0.5),
    solver="bfgs",
)
```

### Weighted Calibration

Emphasize ATM strikes by passing per-strike weights:

```python
weights = jnp.exp(-0.5 * ((strikes - forward) / 10.0) ** 2)
fitted, sol = calibrate_sabr(
    strikes, market_vols, forward, expiry,
    fixed_beta=jnp.array(0.5),
    weights=weights,
)
```

## Heston Calibration

Fit Heston parameters to option prices. Requires a pricing function to be injected — this allows using any Heston pricer (semi-analytic, Monte Carlo, or a neural surrogate).

```python
from valax.calibration import calibrate_heston

fitted, sol = calibrate_heston(
    strikes, market_prices, spot, rate, dividend, expiry,
    pricing_fn=my_heston_pricer,
)
```

## How It Works

### Parameter Transforms

Model parameters have natural constraints (e.g., $\alpha > 0$, $-1 < \rho < 1$). Rather than using constrained optimization, VALAX reparametrizes to unconstrained space:

| Constraint | Transform | Inverse |
|-----------|-----------|---------|
| $x > 0$ | softplus | inverse softplus |
| $a < x < b$ | scaled sigmoid | logit |
| $-1 < x < 1$ | tanh | arctanh |

The optimizer works in unconstrained $\mathbb{R}^n$; results are mapped back to valid parameters automatically.

```python
from valax.calibration import positive, correlation, bounded

# Custom transforms
pos = positive()                    # x > 0
corr = correlation()                # -1 < x < 1
frac = bounded(0.0, 1.0)           # 0 < x < 1
```

### Loss Functions

Calibration minimizes weighted residuals in implied-vol space (default) or price space:

$$\min_\theta \sum_i w_i \left( \sigma^\text{model}(K_i; \theta) - \sigma^\text{market}(K_i) \right)^2$$

Vol-space calibration is more stable than price-space for SABR because it removes the option's moneyness dependence from the residuals.
