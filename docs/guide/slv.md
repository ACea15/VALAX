# Stochastic-Local Volatility

Stochastic-Local Volatility (SLV) is the industry workhorse for exotic equity option pricing — it combines a calibrated implied-vol surface (so all vanillas reprice exactly by construction, like Dupire local vol) with a stochastic volatility backbone (so forward-skew dynamics are realistic, unlike pure LV). VALAX implements the Heston × leverage variant: Heston stochastic volatility under a multiplicative deterministic leverage function calibrated by Markovian projection.

This guide walks through the SDE, the two-pass calibration, the choice of estimator (particle vs kernel), the role of the fixed-point iteration, the MC discretisation, and an end-to-end worked example. See [theory §4.5](../theory.md#45-stochastic-local-volatility) for the underlying mathematics and [api/models.md#SLVModel](../api/models.md#slvmodel) / [api/surfaces.md#leverage-grid-slv](../api/surfaces.md#leverage-grid-slv) / [api/calibration.md#calibrate_slv](../api/calibration.md#calibrate_slv_leverage) for type-level reference.

## When to use SLV

| Model | Marginals reproduce SVI? | Forward-skew dynamics | Cost (calibration + MC) | Suitable for |
|---|---|---|---|---|
| Black–Scholes | ✗ (single vol) | trivial | cheap | vanillas only |
| Heston | partial (one fit) | stochastic, mean-reverting | one calibrator | vanillas, simple exotics |
| Dupire LV | ✓ (by construction) | **deterministic** | one calibrator | barriers, lookbacks under static-smile assumption |
| **SLV** | ✓ (by construction) | **stochastic** | two calibrators | autocallables, forward-starting, cliquets, vol-of-vol exotics |

SLV is the right model when (i) vanilla repricing must be exact, **and** (ii) the payoff is sensitive to how the smile rolls forward in time — autocallables and forward-starting options are the canonical cases. A pure LV will fit today's smile but predict a flattening smile at future dates that often understates the forward skew; SLV preserves Heston's stochastic forward-skew dynamics while pinning the marginals.

See the [Limitations and known approximations](#limitations-and-known-approximations) section at the end for the calibration-accuracy ceiling and the QE-coupling approximation.

## The two-pass calibration

SLV calibration is a sequential two-stage procedure (Guyon-Henry-Labordère 2012):

### Pass 1 — Heston to vanillas

Calibrate the Heston backbone `(v0, κ, θ, ξ, ρ)` to vanilla option prices (or implied vols) using VALAX's existing `calibrate_heston`. This is the same Heston calibration used elsewhere; the SLV machinery adds nothing here. After pass 1, the Heston marginal of `S_T` matches the input vanillas only approximately — pass 2 closes the gap.

### Pass 2 — leverage to the Dupire surface

Build a deterministic leverage function `L(k, t)` such that the **Markovian projection** of the SLV SDE reproduces the Dupire local volatility:

$$L^2(k, t) \;=\; \frac{\sigma_{\mathrm{Dupire}}^2(k, t)}{\mathbb{E}[V_t \mid k_t = k]}$$

where `k = log(S/F(t))`. VALAX provides this as `calibrate_slv_leverage` (or end-to-end `calibrate_slv`, which wraps Pass 1 + Pass 2).

The conditional expectation `E[V_t | k_t = k]` is estimated from a simulated particle swarm — see "Choosing particle vs kernel" below. The simulation is performed time-slice by time-slice on a fixed `(k, t)` grid, filling one row of `LeverageGrid.values` per slice.

```python
import jax, jax.numpy as jnp
from valax.calibration.slv import calibrate_slv_leverage
from valax.models import HestonModel, SLVModel
from valax.surfaces import SVIVolSurface

# Pass 1 output: a calibrated HestonModel (omitted — use calibrate_heston).
heston = HestonModel(
    v0=jnp.array(0.04), kappa=jnp.array(2.0), theta=jnp.array(0.04),
    xi=jnp.array(0.3), rho=jnp.array(-0.6),
    rate=jnp.array(0.03), dividend=jnp.array(0.01),
)
svi_surface: SVIVolSurface = ...  # calibrated to market vanillas

# Pass 2: build the leverage grid.
leverage = calibrate_slv_leverage(
    heston, svi_surface, spot=jnp.array(100.0),
    log_moneyness_grid=jnp.linspace(-0.25, 0.25, 11),
    time_grid=jnp.linspace(0.05, 2.0, 10),
    n_paths=10_000,
    key=jax.random.PRNGKey(42),
    method="kernel",     # or "particle"
    n_iterations=2,      # outer fixed-point iterations
    ridge=1e-3,          # used only when method="kernel"
)

slv = SLVModel.from_heston_and_leverage(heston, svi_surface, leverage)
```

The output is a `valax.surfaces.LeverageGrid` — a small `(n_t, n_k)` matrix that the path generator queries by bilinear interpolation.

## Choosing particle vs kernel

Both estimators target the same conditional expectation; they differ in how they handle the **tails of the simulated swarm**.

* **`method="particle"`** (default): pure Nadaraya-Watson Gaussian kernel on the particles. The bandwidth defaults to Silverman's rule of thumb per time slice (override via `bandwidth=`). Unbiased asymptotically, but the variance of the estimator inflates in regions where particle density is low — typically the wings of the `k`-grid at short expiries.

* **`method="kernel"`**: same Nadaraya-Watson core, plus a small ridge term `ridge` added to the kernel-mass denominator (Tikhonov-style regularisation toward the empirical particle mean of `V`). This biases the estimator toward the prior in low-density regions, suppressing the wild swings the particle method exhibits in the tails. Trade-off: a small bias is added in well-populated regions (rarely material for `ridge ≤ 1e-2`).

Recommendation:

* Calibrating to a tight strike range (`|k| < 0.20`) → either method is fine.
* Extending to OTM wings (`|k| > 0.30`) at short expiries → use `method="kernel"` with `ridge ≈ 1e-3`.
* Reproducibility-critical / golden-grade calibrations → use `method="particle"` (no implicit prior; the estimator's behaviour depends only on the data and the bandwidth).

The `tests/test_calibration/test_slv_calibration.py::TestKernelMethod::test_kernel_smoother_in_tails` test asserts the kernel method produces a strictly more stable leverage across seeds than the particle method on the wide-`k` fixture.

## Fixed-point iteration (`n_iterations`)

The Markovian projection identity is a **fixed-point equation**: `L` is defined in terms of `E[V_t | k_t]`, but that conditional expectation depends on the SLV dynamics, which themselves depend on `L`. The classical Guyon-Henry-Labordère particle method takes one shot at this — initialise with `L ≡ 1` (Heston warm-start), simulate one time-sweep, and read off the calibrated `L`.

VALAX exposes an outer fixed-point loop via `n_iterations`:

* `n_iterations=1` (default): classical one-shot particle method.
* `n_iterations ≥ 2`: re-simulate the swarm under the previous iteration's `L`, then rebuild the grid. Empirically the second iteration cuts the residual Markovian-projection error by 30-50% on a typical equity smile; the third iteration brings additional but diminishing returns.

A practical choice is `n_iterations=2` or `3`. The test `tests/test_calibration/test_slv_calibration.py::TestFixedPoint::test_iterations_converge` checks that the per-iteration `L` update contracts in sup-norm.

## SDE and discretisation

Under risk-neutral measure Q:

$$
\begin{aligned}
\frac{dS_t}{S_t} &= (r - q)\,dt \;+\; L(k_t, t)\,\sqrt{V_t}\,dW_1, \\
dV_t &= \kappa\,(\theta - V_t)\,dt \;+\; \xi\,\sqrt{V_t}\,dW_2, \\
\langle dW_1, dW_2 \rangle &= \rho\,dt.
\end{aligned}
$$

`valax.pricing.mc.generate_slv_paths` discretises this with:

* **Variance leg:** Andersen-QE (same scheme as `generate_heston_paths`). Exact in distribution at each `dt` step regardless of Feller's condition.
* **Log-spot leg:** Selectable via `scheme=`.
  * `"midpoint_euler"` (default): log-Euler with the Itô correction and `L` queried at the midpoint in time `t + dt/2` (mirrors LV-1's `lv_scheme` convention — avoids the Dupire `T=0` singularity). Weak order 1, 1× cost.
  * `"milstein"`: adds the strong-order correction `½ · σ · (∂L/∂k) · √V · dt · (Z² − 1)` to the log-spot update. One extra `jax.value_and_grad` of `L` per step, ~2× per-step cost. Strong order 1, same weak order. Pick this for path-dependent payoffs (barriers, lookbacks).
* **Correlation:** `Z_1 = ρ·Z_v + √(1−ρ²)·Z_⊥`, where `Z_v` is the standard normal driving the QE quadratic branch. This is exact when QE selects the quadratic branch (the typical case for equity-grade parameters) and is the standard approximation on the exponential branch — matches QuantLib's `HestonSLVProcess` convention.

The dispatcher routes the scheme keyword through the recipe layer:

```python
from valax.pricing.mc import mc_price_dispatch, MCConfig
from valax.instruments.options import EquityBarrierOption

barrier = EquityBarrierOption(...)
result = mc_price_dispatch(
    barrier, slv,
    config=MCConfig(n_paths=100_000, n_steps=500),
    key=jax.random.PRNGKey(0),
    spot=jnp.array(100.0),
    slv_scheme="milstein",   # barrier — strong-order matters
)
```

Five SLV recipes are registered out of the box: `EuropeanOption`, `AsianOption`, `EquityBarrierOption`, `LookbackOption`, `VarianceSwap`. Same set as LV-1.

## End-to-end worked example

```python
import jax, jax.numpy as jnp
from valax.calibration.slv import calibrate_slv_leverage
from valax.instruments.options import EuropeanOption
from valax.models import HestonModel, SLVModel
from valax.pricing.analytic.black_scholes import black_scholes_implied_vol
from valax.pricing.mc import mc_price_dispatch, MCConfig
from valax.surfaces import SVIVolSurface

# 1. Inputs: a calibrated SVI surface (assume already fit to market) and
#    a Heston backbone (assume already fit to vanillas via calibrate_heston).
expiries = jnp.array([0.05, 0.25, 0.5, 1.0, 2.0])
surface = SVIVolSurface(
    expiries=expiries,
    forwards=jnp.array(100.0) * jnp.exp(0.02 * expiries),
    a_vec=jnp.array([0.001, 0.006, 0.014, 0.030, 0.062]),
    b_vec=jnp.array([0.04, 0.05, 0.06, 0.07, 0.08]),
    rho_vec=jnp.array([-0.3] * 5),
    m_vec=jnp.zeros_like(expiries),
    sigma_vec=jnp.array([0.1] * 5),
)
heston = HestonModel(
    v0=jnp.array(0.04), kappa=jnp.array(2.0), theta=jnp.array(0.04),
    xi=jnp.array(0.3), rho=jnp.array(-0.6),
    rate=jnp.array(0.03), dividend=jnp.array(0.01),
)

# 2. Calibrate the leverage function.
leverage = calibrate_slv_leverage(
    heston, surface, spot=jnp.array(100.0),
    log_moneyness_grid=jnp.linspace(-0.25, 0.25, 11),
    time_grid=jnp.linspace(0.05, 2.0, 10),
    n_paths=10_000, key=jax.random.PRNGKey(20260101),
    method="kernel", n_iterations=2, ridge=1e-3,
)

# 3. Wrap as an SLVModel.
slv = SLVModel.from_heston_and_leverage(heston, surface, leverage)

# 4. Price a European via the dispatcher and verify the vanilla reprice.
opt = EuropeanOption(
    strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True,
)
result = mc_price_dispatch(
    opt, slv,
    config=MCConfig(n_paths=50_000, n_steps=200),
    key=jax.random.PRNGKey(20260201),
    spot=jnp.array(100.0),
)
iv_slv = black_scholes_implied_vol(
    opt, jnp.array(100.0), jnp.array(0.03), jnp.array(0.01), result.price,
)
iv_target = surface(jnp.array(100.0), jnp.array(1.0))
print(f"SLV IV: {float(iv_slv):.4f}, SVI target: {float(iv_target):.4f}, "
      f"diff: {float(iv_slv - iv_target) * 1e4:.1f} bp")
```

Once the leverage is calibrated, the SLV model behaves like any other `eqx.Module` pytree: JIT-compatible, vmappable, and differentiable through `leverage.values` and through the Heston block.

## Limitations and known approximations

* **Calibration accuracy at moderate budgets.** The particle method's MC-noise floor at `n_paths_cal = 10 000` typically gives ~100-200 bp absolute IV residual when SLV is repriced against the input surface — not the sub-20 bp gate LV-1 hits. This is the well-known particle-method limit; for sub-50 bp production accuracy use Fokker-Planck PDE calibration (QuantLib's `HestonSLVProcess`). VALAX's headline `TestDupireConsistency` gate is pinned at 250 bp accordingly; tightening to sub-100 bp is a roadmap item (SLV-2).
* **Correlation on the QE exponential branch.** The `Z_1 = ρ·Z_v + √(1−ρ²)·Z_⊥` coupling is exact when QE selects the quadratic branch (the typical case) and is approximate on the exponential branch. The exponential branch activates only at extreme variance excursions (`ψ > 1.5`) which are rare for production equity parameters; if your fixture triggers it routinely, expect a sub-bp additional bias.
* **Bandwidth selection.** Silverman's rule is robust for unimodal swarms but conservative — wide bandwidths over-smooth the conditional expectation at sharp-skew points. If you see leverage values close to `L_min` or `L_max` (the default clip range `[0.05, 5.0]`) in regions where the surface is well-behaved, narrow the bandwidth via the explicit `bandwidth=` argument.
* **`jax_enable_x64=True` is required.** `SLVModel.from_heston_and_leverage` raises `RuntimeError` otherwise (mirrors the Dupire-layer guard). `valax/__init__.py` enables x64 globally on import; the guard is a belt-and-braces check against caller-side overrides.
* **Time-grid extrapolation.** Querying `LeverageGrid` at `t < time_grid[0]` returns `L(k, time_grid[0])` (flat extrapolation). For a path generator running from `t = 0`, set `time_grid[0]` close to the simulation step `dt/2` to avoid a constant-leverage bias on the first few steps.

## Reference

* Guyon, J., & Henry-Labordère, P. (2012). "Being Particular About Calibration." *Risk*, January.
* Henry-Labordère, P. (2009). *Analysis, Geometry, and Modeling in Finance.* CRC, Chapter 12.
* Andersen, L. (2008). "Simple and Efficient Simulation of the Heston Stochastic Volatility Model." *Journal of Computational Finance* 11(3): 1–42.
* Gyöngy, I. (1986). "Mimicking the One-Dimensional Marginal Distributions of Processes Having an Itô Differential." *Probability Theory and Related Fields* 71: 501–516.
