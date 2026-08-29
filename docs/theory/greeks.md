[← Models & Theory index](index.md)

# 6. Greeks and Automatic Differentiation

### 6.1 Greeks as Derivatives

Greeks measure the sensitivity of an option's price to changes in inputs:

| Greek | Definition | What it measures |
|-------|-----------|------------------|
| Delta ($\Delta$) | $\frac{\partial V}{\partial S}$ | Sensitivity to spot price |
| Gamma ($\Gamma$) | $\frac{\partial^2 V}{\partial S^2}$ | Curvature of delta (hedging cost) |
| Vega ($\mathcal{V}$) | $\frac{\partial V}{\partial \sigma}$ | Sensitivity to implied volatility |
| Theta ($\Theta$) | $\frac{\partial V}{\partial t}$ | Time decay |
| Rho ($\rho$) | $\frac{\partial V}{\partial r}$ | Sensitivity to interest rates |
| Vanna | $\frac{\partial^2 V}{\partial S \,\partial \sigma}$ | Cross-sensitivity of delta to vol |
| Volga | $\frac{\partial^2 V}{\partial \sigma^2}$ | Sensitivity of vega to vol |

### 6.2 Automatic Differentiation

Traditional libraries (QuantLib, etc.) compute Greeks via **finite differences** (bump-and-reprice):

$$
\Delta \approx \frac{V(S + h) - V(S - h)}{2h}
$$

This requires choosing $h$ (too large = truncation error; too small = floating-point cancellation), computing the price twice per Greek, and scales linearly with the number of risk factors.

VALAX uses **automatic differentiation** via `jax.grad`, which computes exact derivatives by applying the chain rule through the computational graph of the pricing function.

**Forward mode AD:** Propagates derivatives forward through the computation. Computes $\partial V / \partial x_i$ for a single input $x_i$ in one pass. Efficient when there are few inputs and many outputs.

**Reverse mode AD (backpropagation):** Propagates derivatives backward from the output. Computes $\partial V / \partial x_i$ for *all* inputs in one pass. Efficient when there is one output (a price) and many inputs (all risk factors). This is what `jax.grad` uses by default.

**Cost:** One reverse-mode pass costs approximately 2–4x the cost of the forward evaluation. This gives *all* first-order Greeks simultaneously — versus $2N$ evaluations for $N$ Greeks via central finite differences.

**Higher-order Greeks** use nested differentiation: `jax.grad(jax.grad(price_fn))` gives gamma. The computational cost grows linearly with the nesting depth, but each level is exact.

**VALAX implementation:** `valax/greeks/autodiff.py` provides `greeks()` (all Greeks at once) and `greek()` (single Greek by name). These are thin wrappers around `jax.grad` with appropriate `argnums` selection. Because every VALAX data structure is a JAX pytree, differentiation works through curves (`DiscountCurve`), surfaces, and model parameters — giving key-rate durations, surface sensitivities, and model parameter Greeks automatically.

### 6.3 Pathwise Method for MC Greeks

When computing Greeks of MC prices, `jax.grad` differentiates through the entire path simulation:

$$
\frac{\partial}{\partial \theta}\mathbb{E}[g(S_T(\theta))] = \mathbb{E}\!\left[\frac{\partial g}{\partial S_T} \cdot \frac{\partial S_T}{\partial \theta}\right]
$$

This is the **pathwise (infinitesimal perturbation analysis)** estimator. It works when:

- The payoff $g$ is differentiable w.r.t. $S_T$ (or smoothed to be so)
- The path $S_T(\theta)$ is differentiable w.r.t. the parameter $\theta$

**When pathwise fails:** Discontinuous payoffs (digital options, barrier knock-in/out). The derivative of an indicator function is zero almost everywhere and infinite at the barrier — the estimator has zero variance but is biased (always returns zero). VALAX addresses this via smooth sigmoid approximations to barriers in `valax/pricing/mc/payoffs.py`. The alternative **likelihood ratio method** differentiates the probability density instead of the payoff, but is not yet implemented.
