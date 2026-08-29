[← Models & Theory index](index.md)

# 8. Calibration Theory

### 8.1 The Calibration Problem

Model calibration finds parameters $\boldsymbol{\theta}$ such that model prices match market prices:

$$
\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \sum_{i=1}^{N} w_i \left(\sigma_{\text{model}}(K_i, T_i; \boldsymbol{\theta}) - \sigma_{\text{market}}(K_i, T_i)\right)^2
$$

Calibration is typically done in **implied volatility space** rather than price space because:

- Vols are more homogeneous in scale (a 0.1% vol error has consistent meaning across strikes)
- Price errors are dominated by ATM options (highest vega), under-weighting wings
- Vol-space residuals produce a better-conditioned Jacobian

### 8.2 Levenberg-Marquardt Algorithm

For least-squares problems $\min \|\mathbf{r}(\boldsymbol{\theta})\|^2$, the Levenberg-Marquardt (LM) algorithm interpolates between Gauss-Newton and gradient descent:

$$
\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - (\mathbf{J}^T\mathbf{J} + \lambda \mathbf{I})^{-1} \mathbf{J}^T \mathbf{r}
$$

where $\mathbf{J} = \partial \mathbf{r} / \partial \boldsymbol{\theta}$ is the Jacobian of residuals. When $\lambda \to 0$, this is Gauss-Newton (fast near the solution). When $\lambda \to \infty$, this is gradient descent (robust far from the solution). $\lambda$ is adapted automatically.

**VALAX advantage:** The Jacobian $\mathbf{J}$ is computed exactly via `jax.jacobian` — no finite differences. This gives faster convergence (accurate search directions) and is cheaper for models with many market quotes (one reverse-mode pass per residual, vs. $2p$ evaluations for $p$-parameter finite differences).

**VALAX implementation:** `valax/calibration/sabr.py` and `valax/calibration/heston.py` use `optimistix.least_squares` (which implements LM). Alternative solvers: BFGS via `optimistix.minimise`, Adam via `optax`.

### 8.3 Parameter Constraints and Transforms

Many model parameters have natural bounds:

| Parameter | Constraint | Transform |
|-----------|-----------|-----------|
| Volatility $\sigma$ | $> 0$ | $\sigma = \text{softplus}(x) = \ln(1 + e^x)$ |
| Correlation $\rho$ | $(-1, 1)$ | $\rho = \tanh(x)$ |
| CEV exponent $\beta$ | $[0, 1]$ | $\beta = \text{sigmoid}(x)$ |
| Mean-reversion $\kappa$ | $> 0$ | $\kappa = \text{softplus}(x)$ |

VALAX optimizes over the **unconstrained** variable $x$ and applies the transform to get the **constrained** parameter. The transforms are smooth and differentiable — autodiff flows through them seamlessly.

**VALAX implementation:** `valax/calibration/transforms.py` defines `to_unconstrained` and `to_constrained` for each transform type.

### 8.4 Identifiability and Ill-Conditioning

**Identifiability:** A model is identifiable if different parameter values produce different prices. If parameters are non-identifiable (or nearly so), the calibration problem has multiple solutions and the optimizer may find any of them — producing unstable Greeks.

**Known issues:**

- **Heston:** $\kappa$ and $\theta$ are weakly identified from vanilla options alone (the term structure of smile is needed to separate them). This often manifests as a flat direction in the loss surface.
- **SABR:** With $\beta$ fixed (standard practice), the remaining three parameters ($\alpha$, $\rho$, $\nu$) are well-identified from three or more strikes at a single expiry.
- **SVI:** Five parameters for a single-expiry smile can be over-parameterized when only a few strikes are liquid.

**Regularization** (not yet implemented): Adding a penalty term $\lambda \|\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{prior}}\|^2$ to the loss function biases the solution toward a prior (e.g., yesterday's parameters), improving stability at the cost of fit quality. This is standard practice for Heston calibration in production.
