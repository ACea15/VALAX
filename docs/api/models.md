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

## `SABRModel`

Stochastic Alpha Beta Rho model for volatility smiles:

$$dF = \alpha F^\beta\, dW_1, \quad d\alpha_t = \nu \alpha_t\, dW_2, \quad \text{Corr}(dW_1, dW_2) = \rho$$

```python
from valax.models import SABRModel

model = SABRModel(
    alpha: Float[Array, ""],  # initial volatility
    beta: Float[Array, ""],   # CEV exponent (0 = normal, 1 = lognormal)
    rho: Float[Array, ""],    # forward-vol correlation (typically negative)
    nu: Float[Array, ""],     # vol of vol
)
```

**Notes**:

- `beta` controls the backbone: $\beta = 0$ gives normal dynamics, $\beta = 1$ gives lognormal. Equity typically uses $\beta = 0.5$; rates often use $\beta = 0$.
- Pricing uses Hagan's asymptotic implied vol expansion fed into Black-76 (`sabr_price`).
- Monte Carlo paths available via `generate_sabr_paths` using diffrax.
- All parameters are differentiable — Greeks via `jax.grad` work through the full SABR → Black-76 chain.
- Calibration via `calibrate_sabr` in `valax.calibration`.

## `LocalVolModel`

Dupire local volatility — a state- and time-dependent diffusion calibrated to fit the entire vanilla surface by construction:

$$dS = (r - q) S\, dt + \sigma_{\text{loc}}(S, t)\, S\, dW$$

where $\sigma_{\text{loc}}$ is extracted on demand from an implied vol surface via Gatheral's IV-space Dupire formula (see [theory §4.4](../theory.md#44-local-volatility-dupire) and the [`dupire_local_vol`](pricing.md#dupire_local_vol) entry).

```python
from valax.models import LocalVolModel
from valax.surfaces import SVIVolSurface

model = LocalVolModel(
    surface: eqx.Module,             # any object exposing total_variance(k, T)
    rate: Float[Array, ""],          # risk-free rate
    dividend: Float[Array, ""],      # continuous dividend yield
)

# Convenience factory with scalar / Python-float coercion:
model = LocalVolModel.from_flat_rate(
    surface,
    rate=0.03,
    dividend=0.01,
)
```

**The surface protocol.** Any object with a `total_variance(log_moneyness, expiry) -> Float[Array, ""]` method satisfies the duck-typed contract. `SVIVolSurface`, `SABRVolSurface`, and `GridVolSurface` all comply — see [`api/surfaces.md`](surfaces.md). The model intentionally does **not** store a precomputed leverage / local-vol grid: $\sigma_{\text{loc}}$ is recomputed from the surface on every path step, so `jax.grad` flows directly from MC prices into surface parameters (SVI $a$, $b$, $\rho$, $m$, $\sigma$ per slice) for vega-bucketed risk attribution and SLV calibration.

**Forward curve.** The implicit forward is $F(t) = S_0\,e^{(r - q)t}$ — a deterministic-rate equity forward. Term-structured rates and dividends are a planned extension via a `forward_curve: Callable` field (see backlog LV-1 follow-ups in the roadmap).

**Notes**:

- Requires `jax_enable_x64=True` (enforced at the Dupire layer — Dupire's second derivatives of total variance are precision-sensitive). `valax/__init__.py` enables x64 globally on import.
- Path generator: `generate_local_vol_paths` in `valax.pricing.mc` (`lax.scan` + log-Euler with **midpoint-in-time σ**). See [theory §4.4](../theory.md#44-local-volatility-dupire) for why midpoint and not left-endpoint.
- MC dispatcher recipes registered for `EuropeanOption`, `AsianOption`, `EquityBarrierOption`, `LookbackOption`, `VarianceSwap` — barrier is the canonical exotic where LV differs materially from BSM.
- All surface parameters remain differentiable through the model — `jax.grad` of a LV MC price w.r.t. `model.surface.a_vec` gives a vector of per-slice variance-level sensitivities for free.

## `SLVModel`

Stochastic-Local Volatility — Heston stochastic vol times a calibrated leverage function `L(k, t)`, with the leverage chosen so that the SLV marginals reproduce a calibrated implied-vol surface by Markovian projection:

$$\frac{dS_t}{S_t} = (r - q)\,dt + L(k_t, t)\,\sqrt{V_t}\,dW_1, \qquad dV_t = \kappa(\theta - V_t)\,dt + \xi\,\sqrt{V_t}\,dW_2, \qquad \langle dW_1, dW_2 \rangle = \rho\,dt,$$

with `k_t = log(S_t / F(t))`. The leverage function is calibrated in pass 2 of the SLV calibration (see [`calibrate_slv_leverage`](calibration.md#calibrate_slv_leverage)). See [theory §4.5](../theory.md#45-stochastic-local-volatility) and the [SLV guide](../guide/slv.md) for the underlying mathematics and end-to-end recipe.

```python
from valax.models import SLVModel
from valax.surfaces import LeverageGrid, SVIVolSurface

model = SLVModel(
    v0:       Float[Array, ""],       # Heston initial variance
    kappa:    Float[Array, ""],       # mean-reversion speed
    theta:    Float[Array, ""],       # long-run variance
    xi:       Float[Array, ""],       # vol of vol
    rho:      Float[Array, ""],       # spot/vol correlation
    rate:     Float[Array, ""],       # risk-free rate
    dividend: Float[Array, ""],       # continuous dividend yield
    surface:  eqx.Module,             # implied-vol surface (duck-typed
                                       # via total_variance(k, T))
    leverage: LeverageGrid,           # calibrated leverage L(k, t)
)

# Convenience factory — copies the Heston block field-by-field, asserts
# jax_enable_x64=True, and attaches the surface and calibrated leverage.
model = SLVModel.from_heston_and_leverage(
    heston:   HestonModel,            # pass-1 calibrated Heston
    surface:  SVIVolSurface,          # the calibration target
    leverage: LeverageGrid,           # pass-2 calibration output
)
```

**Notes**

- The path generator `generate_slv_paths` never queries the `surface` field — only `calibrate_slv_leverage` does. The surface is kept on the model so the leverage can be re-calibrated against the same target without external bookkeeping.
- Requires `jax_enable_x64=True` (enforced at construction). `valax/__init__.py` enables x64 globally on import.
- MC dispatcher recipes registered for `EuropeanOption`, `AsianOption`, `EquityBarrierOption`, `LookbackOption`, `VarianceSwap` — same set as LV-1. Forward `slv_scheme=` (`"midpoint_euler"` default, `"milstein"` opt-in) via `mc_price_dispatch`.
- All fields are differentiable through the pytree. `jax.grad` w.r.t. `model.leverage.values` gives node-level leverage sensitivities; `jax.grad` w.r.t. `model.surface.a_vec` / `b_vec` / etc. flows through the (re)calibration if you wrap it.
- See the [SLV guide](../guide/slv.md) for the two-pass calibration walkthrough and the [Limitations and known approximations](../guide/slv.md#limitations-and-known-approximations) section for the QE-coupling approximation and calibration-accuracy ceiling.

## `HullWhiteModel`

Hull-White one-factor short-rate model (extended Vasicek):

$$dr(t) = [\theta(t) - a\,r(t)]\,dt + \sigma\,dW(t)$$

```python
from valax.models import HullWhiteModel

model = HullWhiteModel(
    mean_reversion: Float[Array, ""],  # a — mean-reversion speed
    volatility: Float[Array, ""],      # σ — short-rate volatility
    initial_curve: DiscountCurve,      # P^M(0,t) for exact-fit θ(t)
)
```

**Key analytics** (in `valax.models.hull_white`):

| Function | Description |
|---|---|
| `hw_bond_price(model, r, t, T)` | Analytic ZCB price $P(t,T \mid r) = A(t,T)\,e^{-B(t,T)\,r}$ |
| `hw_B(a, tau)` | Mean-reversion decay $(1 - e^{-a\tau})/a$ |
| `hw_short_rate_variance(model, t)` | $\sigma^2/(2a)(1 - e^{-2at})$ |

**Notes**:

- **Exact-fit property**: at $t = 0$ with $r = f^M(0, 0)$, `hw_bond_price` recovers the initial curve $P^M(0, T)$ to machine precision.
- The model is a full JAX pytree — `jax.grad` with respect to `mean_reversion` or `volatility` gives parameter sensitivities.
- Used by `callable_bond_price` and `puttable_bond_price` in `valax.pricing.lattice` for trinomial-tree backward induction.
- Swaption calibration (Jamshidian decomposition) and G2++ extension are roadmap items.

## `LMMModel`

LIBOR Market Model (Brace-Gatarek-Musiela) for forward-rate simulation:

$$dF_i = \mu_i\,dt + \sigma_i(t)\,F_i \sum_j L_{ij}\,dW_j$$

where $L$ is the Cholesky (or PCA) factor of the correlation matrix and $\mu_i$ is the spot-measure drift.

```python
from valax.models import LMMModel, build_lmm_model

model = build_lmm_model(
    initial_forwards: Float[Array, "N"],  # initial forward rates
    tenors: Float[Array, "N"],            # accrual periods τ_i
    vol_fn: PiecewiseConstantVol | RebonatoVol,  # volatility structure
    corr_fn: ExponentialCorrelation | TwoParameterCorrelation,  # correlation
    curve: DiscountCurve,                 # initial discount curve
)
```

**Volatility structures**:

| Class | Description |
|---|---|
| `PiecewiseConstantVol(vols)` | Vol matrix $\sigma_{i,k}$ for forward $i$ during period $k$ |
| `RebonatoVol(a, b, c, d)` | Parametric $(a + b\tau)e^{-c\tau} + d$ |

**Correlation structures**:

| Class | Description |
|---|---|
| `ExponentialCorrelation(decay)` | $\rho_{ij} = e^{-\beta |i-j|}$ |
| `TwoParameterCorrelation(rho_inf, decay)` | $\rho_{ij} = \rho_\infty + (1-\rho_\infty)e^{-\beta|i-j|}$ |

**Notes**:

- Path simulation uses log-Euler discretization for guaranteed positivity.
- PCA-based factor reduction for efficient high-dimensional simulation.
- Used by the Bermudan swaption pricer (Longstaff-Schwartz on LMM paths).

## `MultiAssetGBMModel`

$N$-asset correlated geometric Brownian motion under a single risk-neutral measure:

$$dS_i(t) = (r - q_i) S_i(t)\,dt + \sigma_i S_i(t)\,dW_i(t), \qquad \langle dW_i, dW_j \rangle = \rho_{ij}\,dt$$

```python
from valax.models import MultiAssetGBMModel

model = MultiAssetGBMModel(
    vols: Float[Array, " n_assets"],           # per-asset volatilities
    rate: Float[Array, ""],                    # single risk-free rate (scalar)
    dividends: Float[Array, " n_assets"],      # per-asset continuous dividend yields
    correlation: Float[Array, "n_assets n_assets"],  # symmetric, unit diagonal, PSD
)
```

Used by the multi-asset MC recipes for `SpreadOption` (validates Margrabe / Kirk analytical) and `WorstOfBasketOption` (correlation-sensitive baskets). See the [Monte Carlo guide](../guide/monte-carlo.md#correlated-multi-asset-gbm) for a full walkthrough.

**Correlation validator**:

```python
from valax.models import validate_correlation

# Returns the smallest eigenvalue; negative means the matrix is not PSD.
min_eig = validate_correlation(correlation_matrix, tol=1e-6)
assert float(min_eig) >= -1e-6
```

**Notes**:

- The Cholesky factor of `correlation` is computed inside `generate_correlated_gbm_paths`; for a fixed correlation matrix across many repricings, use `jax.jit` to amortize.
- `jax.grad` through the `correlation` field gives per-entry correlation Greeks automatically.
