# Stochastic-Local Volatility

Stochastic-Local Volatility (SLV) is the industry workhorse for exotic equity option pricing — it combines a calibrated implied-vol surface (so all vanillas reprice exactly by construction, like Dupire local vol) with a stochastic volatility backbone (so forward-skew dynamics are realistic, unlike pure LV). VALAX implements the Heston × leverage variant: Heston stochastic volatility under a multiplicative deterministic leverage function calibrated by Markovian projection.

This guide walks through the SDE, the two-pass calibration, the choice of estimator (particle vs kernel), the role of the fixed-point iteration, the MC discretisation, and an end-to-end worked example. See [theory §4.5](../theory/volatility.md#45-stochastic-local-volatility) for the underlying mathematics and [api/models.md#SLVModel](../api/models.md#valax.models.SLVModel) / [api/surfaces.md#leverage-grid-slv](../api/surfaces.md#leverage-grid-slv) / [api/calibration.md#calibrate_slv](../api/calibration.md#valax.calibration.calibrate_slv_leverage) for type-level reference.

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

**Condition the backbone.** On a single smile the five Heston parameters lie in a flat valley (`κ`, `ξ` and `v0` trade off). On example 17's SABR market the five-parameter fit to the 1Y smile alone exhausts its 512 Levenberg–Marquardt steps at `κ ≈ 8.9`, `ξ ≈ 4.3`. Two remedies, combinable:

* **several expiries** — `expiry` may be one expiry per quote, aligned with `strikes`; the same five-parameter fit on the 0.5Y + 1Y smiles converges in 27 steps (`κ = 1.39`, `ξ = 1.06`);
* **hold weakly identified parameters fixed** — `fixed_params={"kappa": 1.5}`; fixed values are reported as fixed inputs (`result.inputs.fixed["kappa"]`), the same convention as `basket_fit_level`.

```python
from valax.calibration import calibrate_heston

fit = calibrate_heston(
    strikes, prices, spot, rate, dividend,
    expiry=expiries,                 # one per quote: a multi-expiry fit
    pricing_fn=cos_pricer,           # (model, K, S, r, q, T) -> price, vmapped per quote
    fixed_params={"kappa": 1.5},
)
```

`calibrate_slv` passes both through (`expiry=`, `fixed_params=`), or takes a backbone you fitted yourself — e.g. a `price_fit_node` result inside a calibration chain — with `heston=` (see [below](#pass-2-leverage-to-the-dupire-surface)).

### Pass 2 — leverage to the Dupire surface

Build a deterministic leverage function `L(k, t)` such that the **Markovian projection** of the SLV SDE reproduces the Dupire local volatility:

$$L^2(k, t) \;=\; \frac{\sigma_{\mathrm{Dupire}}^2(k, t)}{\mathbb{E}[V_t \mid k_t = k]}$$

where `k = log(S/F(t))`. VALAX provides this as `calibrate_slv_leverage` (or end-to-end `calibrate_slv`, which wraps Pass 1 + Pass 2).

The conditional expectation `E[V_t | k_t = k]` is estimated from a simulated particle swarm — see "Choosing particle vs kernel" below. The simulation is the Guyon–Henry-Labordère sweep: the swarm marches forward through the time grid, and at each row time `t_i` the row `L(·, t_i)` is built from the *current* particles and immediately used to move them on. One sweep is therefore already self-consistent (see [the passes](#fixed-point-iteration-n_iterations)).

**Time convention (shared with the pricer).** A particle step from `t_{i−1}` to `t_i` queries `L` exactly as `generate_slv_paths` does: bilinearly interpolated in the `LeverageGrid` at the midpoint time and at `k = log S − log F(t_mid)`. That blends the built row `i−1` with row `i`, which needs the particles at `t_i` and is not built yet; it is *predicted* — on the first pass by `σ_Dupire(·, t_i)/√Ê_{i−1}[V | ·]` (with `Ê_0 = v_0` exactly), on later passes by the previous pass's row. Before `time_grid[0]` the grid is flat at row 0, as in the pricer. Use one row per pricing step (or `n_substeps=` particle steps per row, each querying its own midpoint) so the calibration and the pricer discretise the SDE identically: on example 17's market at `ξ ≈ 1.3`, 16 rows × 1 step gave 47/39 bp vanilla RMS error, 16 rows × 3 sub-steps 21/26 bp, and 48 rows 16/11 bp (0.5Y/1Y, priced with 48 steps).

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
fit = calibrate_slv_leverage(
    heston, svi_surface, spot=jnp.array(100.0),
    log_moneyness_grid=jnp.linspace(-0.25, 0.25, 11),
    time_grid=jnp.linspace(0.05, 2.0, 10),
    n_paths=10_000,
    key=jax.random.PRNGKey(42),
    method="kernel",     # or "particle"
    n_iterations=2,      # GHL passes (the default; 2 measures MC stability)
    ridge=30.0,          # pseudo-particles (the default); kernel only
)
leverage = fit.value     # LeverageGrid
fit.converged            # max|L_2 - L_1| <= settings.atol (traceable)

slv = SLVModel.from_heston_and_leverage(heston, svi_surface, leverage)
```

The calibrated object is `fit.value`, a `valax.surfaces.LeverageGrid` — a small `(n_t, n_k)` matrix that the path generator queries by bilinear interpolation. `fit` itself is a `CalibrationResult` reporting the fixed point (see [below](#how-the-fixed-point-is-reported)).

`calibrate_slv` runs both passes and returns one result: its value is the assembled `SLVModel`, the Heston level is `result.children[0]`, and `result.check()` / `result.converged` cover both passes. Pass 1's solver is configured with `heston_settings=SolverSettings(...)` (plus `fixed_params=`, `heston_weights=`, and a per-quote `expiry`), pass 2's tolerance with `settings=`. To skip pass 1, supply the backbone:

```python
res = calibrate_slv(
    spot=spot, surface=surface,
    log_moneyness_grid=k_grid, time_grid=t_grid, n_paths=50_000, key=key,
    heston=heston_result,          # a CalibrationResult (recorded as the child as is)
                                   # or a HestonModel (recorded as an input level)
    rate=r, dividend=q,            # optional; required if the backbone leaves them None
)
```

Heston-fit arguments together with `heston=`, or a missing required argument, raise `ValueError` before anything is traced.

## Choosing particle vs kernel

Both estimators target the same conditional expectation; they differ in how they handle the **tails of the simulated swarm**.

* **`method="particle"`** (default): pure Nadaraya-Watson Gaussian kernel on the particles. The bandwidth defaults to Silverman's rule of thumb per time slice (override via `bandwidth=`). Unbiased asymptotically, but the variance of the estimator inflates in regions where particle density is low — typically the wings of the `k`-grid at short expiries.

* **`method="kernel"`**: the same estimator shrunk toward the unconditional particle mean $\bar V$ by `ridge` = $\lambda$ **pseudo-particles**:

  $$\widehat{\mathbb E}[V \mid k] = \frac{\sum_j w_j(k)\,V_j + \lambda\,\bar V}{\sum_j w_j(k) + \lambda}, \qquad w_j(k) = e^{-\frac12((k-k_j)/h)^2} \in (0, 1].$$

  $\sum_j w_j(k)$ is the effective number of particles near $k$. At the money it is in the thousands (≈ 6 000 at 50 000 particles), so $\lambda = 30$ is a < 1 % weight; in a sparse wing cell it is comparable to or below $\lambda$, and the estimate is pulled to $\bar V$ instead of following a handful of particles. $\lambda$ is dimensionless and does not scale with `n_paths`, so its bias vanishes as the swarm grows. (Before this convention the weights were rescaled so the nearest particle had weight 1, which made the denominator ≥ 1 everywhere — the old documented `ridge=1e-3` was a thousandth of one particle and had no effect; example 17 needed 50–200.)

**The default, `ridge=30`, was chosen by measurement** on example 17's SABR market (50 000 particles, 16 rows × 3 sub-steps, two calibration seeds; vanillas repriced with 3 × 100 000 paths):

| `ridge` | seed-to-seed max\|ΔL\|, ξ ≈ 1.3 | same, ξ ≈ 0.65 | ATM `L(1Y)` shift vs. no ridge | vanilla RMS 0.5Y/1Y (ξ ≈ 1.3, two seeds) |
|---|---|---|---|---|
| 0 (`"particle"`) | 0.185 | 0.271 | — | 21/26, 28/26 bp |
| 10 | 0.153 | 0.068 | −0.06 % | 21/26, 28/26 bp |
| **30** | **0.109** | **0.052** | **−0.2 %** | **20/26, 27/26 bp** |
| 100 | 0.067 | 0.038 | −0.7 % | 22/27, 28/28 bp |

30 stabilises the wings 1.7–5× while leaving the money untouched and the vanilla fit unchanged or better; 100 starts to move ATM.

Recommendation:

* Calibrating to a tight strike range (`|k| < 0.20`) → either method is fine.
* Leverage grid extending into the wings, or short first rows (few particles away from the money) → `method="kernel"` with the default ridge.
* Reproducibility-critical / golden-grade calibrations → use `method="particle"` (no implicit prior; the estimator's behaviour depends only on the data and the bandwidth).

The `tests/test_calibration/test_slv_calibration.py::TestKernelMethod::test_kernel_smoother_in_tails` test asserts the kernel method produces a more stable leverage across seeds than the particle method on the wide-`k` fixture, and `TestRidge` that `ridge=0` *is* the particle method while the default moves the sparse wings measurably and the money by < 1 %.

## Fixed-point iteration (`n_iterations`)

The Markovian projection identity is a **fixed-point equation**: `L` is defined in terms of `E[V_t | k_t]`, but that conditional expectation depends on the SLV dynamics, which themselves depend on `L`. The Guyon–Henry-Labordère particle method resolves it *inside one forward sweep*: the rows are built in time order, and the particles that estimate row `i` were simulated under rows `0 … i−1` of the same leverage. Each VALAX pass is exactly that sweep.

* `n_iterations=1`: one self-consistent sweep — the classical one-shot calibration. Usable, but always reported **not converged**: there is nothing to compare it with except the `L ≡ 1` warm start.
* `n_iterations ≥ 2` (default 2): further sweeps, each with fresh randomness (`fold_in(key, pass)`); the previous pass enters only as the predictor of the not-yet-built row in the midpoint interpolation (an `O(Δt)` effect). The increment `L_K − L_{K−1}` is therefore the **Monte Carlo noise of the estimator** — a stability check, not the contraction of a Picard iteration. More passes do not reduce it; more particles (∝ 1/√`n_paths`) or a larger ridge do.

Why this matters: before, a pass moved the particles under the leverage *frozen at the start of the pass*, so pass 1 was a pure-Heston projection and the scheme was a Picard iteration on the whole grid. At high vol-of-vol it did not settle: on example 17's market at ξ = 1.3 it was observed to diverge (max\|ΔL\| 1.32 → 2.68 → 3.53), and a re-measurement at `ridge=1e-3` gave max\|ΔL\| 0.77 → 0.57 → 0.44 while the 1Y vanilla RMS error went 34 → 54 → 47 bp (100 000 pricing paths) — the example had to halve ξ with a mixing fraction. On the same market at the test budget (20 000 particles, 24 rows; RMS over the ten quoted vanillas), the old scheme gave 39.4 bp after one pass and 31.9 bp after two, and the sweep gives 31.3 bp after one pass (`TestGHLSweep::test_single_pass_reprices_high_vol_of_vol_smile`). Example 17 now runs the full ξ ≈ 1.1 backbone with no mixing: max\|ΔL\| 1.08 (vs `L ≡ 1`) → 0.14 → 0.12 → 0.11 → 0.14, clip fraction 0 %, vanillas 9–24 bp RMS across passes (100 000 pricing paths).

### How the fixed point is reported

The leverage calibration is **not** a residual driven to zero, and VALAX does not pretend it is. Each pass `L_{k+1} = G(L_k)` is a self-consistent sweep with fresh randomness, so the increment `L_{k+1} − L_k` is (and never falls below) the Monte Carlo noise of the leverage estimate (roughly `1/√n_paths` relative per cell). There is no solution `R(L) = 0` at which the implicit function theorem could be applied. The strategy is therefore:

1. **An explicit level.** `calibrate_slv_leverage` returns a `CalibrationResult` with `theta=None` and `problem="fixed_point"`. Its value is the output of the `n_iterations` unrolled passes — a differentiable map of the inputs (the Heston model, the target surface, spot).
2. **Risk by plain AAD, never by IFT.** Sensitivities of anything priced off the leverage are AAD (reverse-mode autodiff) through the passes actually computed (unrolled `lax.scan`s; jit-able), which is exact for the grid actually returned wherever the Monte Carlo sample is smooth. Through the surface, the leverage passes and the pricer it is: example 17 matches a same-random-numbers central FD to 2e-4 with the Heston backbone held. Through the Heston *parameters* it is only piecewise smooth: each Andersen-QE variance step picks a branch by `ψ ≷ 1.5`, and `ψ` depends on `(κ, θ, ξ)`, so at high vol-of-vol a parameter move flips branches for some particles. AAD is then the pathwise derivative of the smooth pieces, and a bump FD (dominated by the flips) does not converge as the bump shrinks.
3. **Convergence on the last increment.** `fixed_point_diagnostics` reports
    * `diagnostics.residuals` — the last increment `L_K − L_{K−1}` on the grid;
    * `diagnostics.converged` — `max|L_K − L_{K−1}| ≤ settings.atol`. The tolerance must sit **above** the Monte Carlo noise floor of the *maximum over the grid*, which grows with the grid and is set by its sparsest cells: on example 17's 48 × 17 grid at 50 000 particles and the default ridge it is 0.11–0.14 (typical cells move by ~0.01–0.02), so the example uses `atol = 0.2`; the default `SLV_SETTINGS` (`atol = 5e-2`) suits small grids or ~10× more particles. Raise `n_paths` before tightening it;
    * `n_iterations=1` — always `converged=False`: a single pass has nothing to compare with but the `L ≡ 1` warm start;
    * `diagnostics.extras["clip_fraction"]` — the share of grid cells whose raw `L` hit `[L_min, L_max]` in the last pass. Reported, not used for convergence: a high value means the estimator is degenerate somewhere (see *Bandwidth selection* below), whatever the increment says.

```python
from valax.calibration import SolverSettings

fit = calibrate_slv_leverage(
    heston, svi_surface, spot=jnp.array(100.0),
    log_moneyness_grid=jnp.linspace(-0.25, 0.25, 11),
    time_grid=jnp.linspace(0.05, 2.0, 10),
    n_paths=50_000, key=jax.random.PRNGKey(42),
    n_iterations=3,
    settings=SolverSettings(atol=3e-2, rtol=0.0),  # only atol is read
)
float(fit.diagnostics.max_abs_residual)      # max|L_3 - L_2|
float(fit.diagnostics.extras["clip_fraction"])
fit.check()                                  # CalibrationError if not converged
```

The rationale and the maths are recorded once, in [Calibration, Pricing & Risk § D9](../architecture/calibration-pricing-risk.md#3-decisions) and [Calibration Chains § I.8](../theory/calibration-chains.md#i8-fixed-point-levels).

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

# 2. Calibrate the leverage function (a fixed-point level; .value is the grid).
fit = calibrate_slv_leverage(
    heston, surface, spot=jnp.array(100.0),
    log_moneyness_grid=jnp.linspace(-0.25, 0.25, 11),
    time_grid=jnp.linspace(0.05, 2.0, 10),
    n_paths=10_000, key=jax.random.PRNGKey(20260101),
    method="kernel", n_iterations=2,   # default ridge (30 pseudo-particles)
)
print(f"fixed point: max|ΔL| = {float(fit.diagnostics.max_abs_residual):.3f}, "
      f"converged = {bool(fit.converged)}")
leverage = fit.value

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

* **Calibration accuracy at moderate budgets.** The particle method's MC-noise floor at `n_paths_cal = 10 000` typically gives ~100-200 bp absolute IV residual when SLV is repriced against the input surface — not the sub-20 bp gate LV-1 hits. At 50 000 particles with one leverage row per pricing step, example 17's SABR market reprices to 9–24 bp RMS per expiry (spread across independent passes = the estimator's noise); at 20 000 particles and 24 rows, ~25–37 bp at ξ ≈ 1.3. This is the well-known particle-method limit; for sub-10 bp production accuracy use Fokker-Planck PDE calibration (QuantLib's `HestonSLVProcess`). VALAX's headline `TestDupireConsistency` gate is pinned at 250 bp accordingly; tightening to sub-100 bp is a roadmap item (SLV-2).
* **Time discretisation of the swarm.** The calibration swarm takes one step per leverage row (or `n_substeps`); if that is coarser than the pricer's step the two discretise the SDE differently, which costs tens of bp at high vol-of-vol (see [the time convention](#pass-2-leverage-to-the-dupire-surface)). Match them.
* **Noise floor of `converged`.** The increment is a maximum over the grid; its floor is set by the sparsest cells (early rows, wings), not by the money. Size `atol` for your grid and budget (see [How the fixed point is reported](#how-the-fixed-point-is-reported)).
* **AAD through the backbone is pathwise.** The QE variance branch switch makes each sample only piecewise smooth in `(κ, θ, ξ)`; see item 2 of [How the fixed point is reported](#how-the-fixed-point-is-reported).
* **Local-vol target in the wings.** The Dupire target is evaluated through the default `LocalVolGuard` (flat beyond eight ATM standard deviations, implied-variance fallback where Gatheral's denominator is not positive, variance clipped to `[1e-8, 25 × ATM]`); pass `local_vol_guard=None` for raw Dupire. See the [Monte Carlo guide](monte-carlo.md#local-volatility).
* **Correlation on the QE exponential branch.** The `Z_1 = ρ·Z_v + √(1−ρ²)·Z_⊥` coupling is exact when QE selects the quadratic branch (the typical case) and is approximate on the exponential branch. The exponential branch activates only at extreme variance excursions (`ψ > 1.5`) which are rare for production equity parameters; if your fixture triggers it routinely, expect a sub-bp additional bias.
* **Bandwidth selection.** Silverman's rule is robust for unimodal swarms but conservative — wide bandwidths over-smooth the conditional expectation at sharp-skew points. If you see leverage values close to `L_min` or `L_max` (the default clip range `[0.05, 5.0]`) in regions where the surface is well-behaved — a non-trivial `diagnostics.extras["clip_fraction"]` is the signal — narrow the bandwidth via the explicit `bandwidth=` argument.
* **`jax_enable_x64=True` is required.** `SLVModel.from_heston_and_leverage` raises `RuntimeError` otherwise (mirrors the Dupire-layer guard). `valax/__init__.py` enables x64 globally on import; the guard is a belt-and-braces check against caller-side overrides.
* **Time-grid extrapolation.** Querying `LeverageGrid` at `t < time_grid[0]` returns `L(k, time_grid[0])` (flat extrapolation). For a path generator running from `t = 0`, set `time_grid[0]` close to the simulation step `dt/2` to avoid a constant-leverage bias on the first few steps.

## Reference

* Guyon, J., & Henry-Labordère, P. (2012). "Being Particular About Calibration." *Risk*, January.
* Henry-Labordère, P. (2009). *Analysis, Geometry, and Modeling in Finance.* CRC, Chapter 12.
* Andersen, L. (2008). "Simple and Efficient Simulation of the Heston Stochastic Volatility Model." *Journal of Computational Finance* 11(3): 1–42.
* Gyöngy, I. (1986). "Mimicking the One-Dimensional Marginal Distributions of Processes Having an Itô Differential." *Probability Theory and Related Fields* 71: 501–516.
