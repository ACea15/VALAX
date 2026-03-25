# Models

Stochastic process definitions used by the Monte Carlo engine. Each model is an `equinox.Module` containing the process parameters.

## `BlackScholesModel`

Geometric Brownian Motion: $dS = (r - q) S\, dt + \sigma S\, dW$

```python
from valax.models import BlackScholesModel

model = BlackScholesModel(
    vol: Float[Array, ""],       # annualized volatility
    rate: Float[Array, ""],      # risk-free rate
    dividend: Float[Array, ""],  # continuous dividend yield
)
```

## `HestonModel`

Stochastic volatility with mean-reverting variance:

$$dS = (r - q) S\, dt + \sqrt{V} S\, dW_1$$
$$dV = \kappa(\theta - V)\, dt + \xi \sqrt{V}\, dW_2, \quad \text{Corr}(dW_1, dW_2) = \rho$$

```python
from valax.models import HestonModel

model = HestonModel(
    v0: Float[Array, ""],       # initial variance
    kappa: Float[Array, ""],    # mean reversion speed
    theta: Float[Array, ""],    # long-run variance
    xi: Float[Array, ""],       # vol of vol
    rho: Float[Array, ""],      # spot-vol correlation (typically negative)
    rate: Float[Array, ""],     # risk-free rate
    dividend: Float[Array, ""], # continuous dividend yield
)
```

**Notes**:

- Path generation works in log-spot space for numerical stability.
- Variance is floored at zero (`jnp.maximum(v, 0)`) to prevent square root of negative numbers — this is the absorption scheme.
- All parameters are differentiable, enabling gradient-based calibration.
