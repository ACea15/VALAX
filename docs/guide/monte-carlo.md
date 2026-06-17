# Monte Carlo Pricing

VALAX provides a three-layer Monte Carlo stack:

1. **Path generators** — low-level SDE simulators, built on
   [diffrax](https://docs.kidger.site/diffrax/).
2. **Payoff functions** — path → cashflow transformers, written for
   pathwise differentiability (`jax.grad` friendly).
3. **Unified dispatcher** — a single `mc_price_dispatch(instrument, model, ...)`
   entry point backed by an `(instrument, model)` → recipe registry. This
   is the preferred user-facing API.

For the mathematical framework (risk-neutral expectation, SDE discretization,
pathwise vs. likelihood-ratio Greeks, variance reduction), see
[Models & Theory §5.3](../theory.md#53-monte-carlo-simulation).

## 1. Quick start: the dispatcher

```python
import jax, jax.numpy as jnp
from valax.instruments import EuropeanOption
from valax.models import BlackScholesModel
from valax.pricing.mc import mc_price_dispatch, MCConfig

option = EuropeanOption(
    strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True,
)
model = BlackScholesModel(
    vol=jnp.array(0.20), rate=jnp.array(0.05), dividend=jnp.array(0.02),
)

result = mc_price_dispatch(
    option, model,
    config=MCConfig(n_paths=100_000, n_steps=100),
    key=jax.random.PRNGKey(42),
    spot=jnp.array(100.0),
)

print(f"Price:  {float(result.price):.4f}")
print(f"StdErr: {float(result.stderr):.4f}")
print(f"Paths:  {result.n_paths}")
```

`mc_price_dispatch` looks up the recipe for `(type(option), type(model))`,
runs path generation + payoff + discounting, and returns an `MCResult`
(`price`, `stderr`, `n_paths`). Use `float(result)` as a shortcut for
`float(result.price)`.

VALAX ships with **16 built-in recipes** (10 single-asset equity × GBM/Heston,
4 rates × LMM, 2 multi-asset × `MultiAssetGBMModel`). See the coverage map
in §2 and the built-in recipe list in the [API reference](../api/pricing.md#built-in-recipes-16).

To swap the model, just pass a different one:

```python
from valax.models import HestonModel

heston = HestonModel(
    v0=jnp.array(0.04), kappa=jnp.array(2.0), theta=jnp.array(0.04),
    xi=jnp.array(0.3), rho=jnp.array(-0.7),
    rate=jnp.array(0.05), dividend=jnp.array(0.02),
)
result = mc_price_dispatch(
    option, heston,
    config=MCConfig(n_paths=20_000, n_steps=100),
    key=jax.random.PRNGKey(1),
    spot=jnp.array(100.0),
)
```

The same `EuropeanOption` works with BSM or Heston because both models
have a registered recipe for it. To see what's available:

```python
from valax.pricing.mc import registered_recipes

for instr, model in registered_recipes():
    print(f"  ({instr}, {model})")
```

## 2. Coverage map

Legend: ✓ = dispatcher recipe exists, 🟡 = planned (payoff exists, recipe
not yet wired), 🟠 = needs a new path generator, 🔴 = needs major infra.

### Equity (single asset)

| Instrument | `BlackScholesModel` | `HestonModel` | `LocalVolModel` | `MultiAssetGBMModel` | Notes |
|-----------|:---:|:---:|:---:|:---:|---|
| `EuropeanOption` | ✓ | ✓ | ✓ | — | |
| `AsianOption` | ✓ | ✓ | ✓ | — | Arithmetic + geometric averaging |
| `EquityBarrierOption` | ✓ | ✓ | ✓ | — | KI/KO with sigmoid smoothing; LV is the canonical model when smile shape at the barrier matters |
| `LookbackOption` | ✓ | ✓ | ✓ | — | Floating + fixed strike |
| `VarianceSwap` | ✓ | ✓ | ✓ | — | Realized variance from log-returns |
| `SpreadOption` | — | — | — | ✓ | Validates Margrabe (K=0) and Kirk (K≠0) analytical |
| `WorstOfBasketOption` | — | — | — | ✓ | 2+ asset basket; cross-asset correlation Greeks via `jax.grad` |
| `AmericanOption` | 🟡 | 🟡 | 🟡 | — | LSM engine exists for rates — needs lifting |
| `DigitalOption` | 🟡 | 🟡 | 🟡 | — | Payoff is smoothed Heaviside |
| `Autocallable` / `Cliquet` | 🟠 | 🟠 | 🟠 | 🟠 | Needs multi-observation / forward-start engine |

### Rates (LMM)

| Instrument | `LMMModel` | Notes |
|-----------|:---:|---|
| `Caplet` | ✓ | Caller provides `forward_index` + `tau` |
| `Cap` | ✓ | Caller provides `forward_indices` + `taus` |
| `Swaption` (European) | ✓ | |
| `BermudanSwaption` | ✓ | Longstaff-Schwartz, `stderr = 0.0` sentinel |
| `CMSSwap` / `CMSCapFloor` | 🟠 | Replication payoff needs to be added |
| `RangeAccrual` | 🟠 | Full daily-observation MC on LMM |

### Fixed income (stochastic rates)

| Instrument | MC pricing | Blocker |
|-----------|:---:|---|
| `FixedRateBond` / `FloatingRateBond` | 🟠 | Needs Hull-White MC path generator |
| `CallableBond` / `PuttableBond` | 🟠 | LSM on HW MC paths (LSM engine already exists) |
| `ConvertibleBond` | 🔴 | Equity + rates + credit coupled MC |

### FX / Inflation / Credit

| Instrument | Status | Blocker |
|-----------|:---:|---|
| `FXVanillaOption`, `FXBarrierOption` | 🟡 | Literally reuse `generate_gbm_paths` with drift $r_d - r_f$ |
| `QuantoOption`, `TARF` | 🟠 | Correlated FX + asset MC |
| `YearOnYearInflationSwap`, `InflationCapFloor` | 🔴 | Jarrow-Yildirim 3-factor MC |
| `CDS`, `CDOTranche` | 🔴 | Survival curve + hazard-rate MC, copula simulation |

See the [Roadmap](../roadmap.md) for the execution order of the 🟡/🟠/🔴
items.

## 3. Path generators (low-level)

Each generator returns a fixed-shape JAX array (or a tuple of arrays)
so the whole pipeline remains `jax.jit` / `jax.vmap` friendly.

| Function | Model | Output shape |
|----------|-------|--------------|
| `generate_gbm_paths(model, spot, T, n_steps, n_paths, key)` | `BlackScholesModel` | `(n_paths, n_steps+1)` |
| `generate_heston_paths(model, spot, T, n_steps, n_paths, key)` | `HestonModel` | `(paths, variances)` each `(n_paths, n_steps+1)` |
| `generate_sabr_paths(model, forward, T, n_steps, n_paths, key)` | `SABRModel` | `(forwards, vols)` each `(n_paths, n_steps+1)` |
| `generate_lmm_paths(model, n_steps_per_period, n_paths, key)` | `LMMModel` | `LMMPathResult` with `forwards_at_fixing`, `forwards_at_tenors`, `discount_factors` |
| `generate_correlated_gbm_paths(model, spots, T, n_steps, n_paths, key)` | `MultiAssetGBMModel` | `(n_paths, n_steps+1, n_assets)` |
| `generate_local_vol_paths(model, spot, T, n_steps, n_paths, key)` | `LocalVolModel` | `(n_paths, n_steps+1)` |

### Geometric Brownian Motion

$$dS = (r - q) S\, dt + \sigma S\, dW$$

```python
from valax.pricing.mc import generate_gbm_paths

paths = generate_gbm_paths(
    model=bs_model,
    spot=jnp.array(100.0),
    T=1.0,
    n_steps=100,
    n_paths=50_000,
    key=jax.random.PRNGKey(0),
)
# paths.shape == (50_000, 101); paths[:, 0] == 100.0
```

### Heston

$$dS = (r - q) S\, dt + \sqrt{V} S\, dW_1, \quad dV = \kappa(\theta - V) dt + \xi \sqrt{V}\, dW_2, \quad \langle dW_1, dW_2 \rangle = \rho\, dt$$

```python
from valax.pricing.mc import generate_heston_paths

spot_paths, var_paths = generate_heston_paths(
    heston_model,
    spot=jnp.array(100.0),
    T=1.0, n_steps=100, n_paths=50_000,
    key=jax.random.PRNGKey(0),
)
```

See [theory §2.4](../theory.md#24-heston-stochastic-volatility) for the
Feller condition and numerical caveats near the variance boundary.

### Local volatility

$$dS = (r - q) S\, dt + \sigma_{\text{loc}}(S, t)\, S\, dW$$

where $\sigma_{\text{loc}}$ is recomputed from a calibrated implied-vol surface
at every path step via Gatheral's IV-space Dupire formula (see
[theory §4.4](../theory.md#44-local-volatility-dupire)). The model itself
carries no precomputed grid; autodiff flows from MC prices straight into the
surface parameters.

```python
import jax, jax.numpy as jnp
from valax.surfaces import calibrate_svi_surface
from valax.models import LocalVolModel
from valax.pricing.mc import generate_local_vol_paths

# 1. Build (or calibrate) an implied-vol surface
svi = calibrate_svi_surface(
    strikes_per_expiry=[...],     # market data
    market_vols_per_expiry=[...],
    forwards=jnp.array([...]),
    expiries=jnp.array([...]),
)

# 2. Wrap it in a LocalVolModel
model = LocalVolModel.from_flat_rate(svi, rate=0.03, dividend=0.01)

# 3. Simulate
paths = generate_local_vol_paths(
    model,
    spot=jnp.array(100.0),
    T=1.0, n_steps=500, n_paths=100_000,
    key=jax.random.PRNGKey(0),
)
# paths.shape == (100_000, 501); paths[:, 0] == 100.0
```

**Scheme.** `jax.lax.scan` over time with log-Euler and Itô correction:
$\ln S_{t_{n+1}} = \ln S_{t_n} + (r - q - \tfrac{1}{2}\sigma_n^2)\,\Delta t + \sigma_n\sqrt{\Delta t}\,Z_n$,
where $\sigma_n = \sigma_{\text{loc}}\!\big(S_{t_n},\,t_n + \tfrac{1}{2}\Delta t\big)$ —
**midpoint in time**, *not* left endpoint. The midpoint convention avoids
querying Dupire at $T = 0$ (where the $1/w$ terms in the denominator
diverge) and gives a better weak-error constant than left-endpoint Euler.

**Scheme selection.** `generate_local_vol_paths` exposes a
`scheme=` kwarg (forwarded as `lv_scheme=` through the unified
dispatcher) with two options:

| Scheme | Weak order | Strong order | Per-step cost | When to use |
|---|:---:|:---:|:---:|---|
| `"midpoint_euler"` (default) | 1 | 0.5 | 1× | Vanilla / smile pricing, SLV calibration — the typical case |
| `"milstein"` (opt-in) | 1 | 1.0 | ~2× | Realized-variance pricing, path-statistics analysis, any payoff sensitive to *strong* (path-wise) accuracy |

Empirically, both schemes give indistinguishable vanilla-reprice
accuracy at typical MC budgets (the noise floor dominates the
constant-factor improvement). The default is Euler because it is
cheaper for no measurable accuracy loss on the common case. See the
[`generate_local_vol_paths`](../api/pricing.md#generate_local_vol_paths)
module docstring for the full empirical sweep and
[theory §4.4](../theory.md#44-local-volatility-dupire) for the weak-
vs strong-order discussion.

**Surface choice.** `SVIVolSurface` is the recommended Dupire input —
closed-form $w(k, T)$ gives exact derivatives via `jax.grad`. `SABRVolSurface`
and `GridVolSurface` also satisfy the protocol but typically give noisier
local-vol estimates at extreme strikes.

**Dupire-consistency gate.** SVI → Dupire → LV MC reprices the input vanilla
IV grid to < 20 bp absolute IV (4 seeds × 100k paths × 500 steps; see
`tests/test_pricing/test_local_vol_paths.py::TestDupireConsistency`).
The headline gate to remember: a typical LV MC budget gives you smile
reprice accuracy of a few tens of basis points, not single-digit.

### Correlated multi-asset GBM

$$dS_i(t) = (r - q_i) S_i\,dt + \sigma_i S_i\,dW_i, \quad \langle dW_i, dW_j \rangle = \rho_{ij}\,dt$$

Build a model with per-asset vols/dividends and a symmetric positive
semi-definite correlation matrix:

```python
from valax.models import MultiAssetGBMModel, validate_correlation
import jax.numpy as jnp

C = jnp.array([
    [1.0, 0.6, 0.3],
    [0.6, 1.0, 0.4],
    [0.3, 0.4, 1.0],
])
# Sanity check the correlation matrix before building the model.
assert float(validate_correlation(C)) > 0, "C is not PSD"

multi = MultiAssetGBMModel(
    vols=jnp.array([0.25, 0.30, 0.35]),
    rate=jnp.array(0.05),
    dividends=jnp.zeros(3),
    correlation=C,
)

from valax.pricing.mc import generate_correlated_gbm_paths
paths = generate_correlated_gbm_paths(
    multi,
    spots=jnp.array([100.0, 100.0, 100.0]),
    T=1.0, n_steps=50, n_paths=20_000,
    key=jax.random.PRNGKey(0),
)
# paths.shape == (20_000, 51, 3)
```

Since pure GBM has a log-linear SDE, the log-Euler step is **exact** for
any `n_steps`. Use large `n_steps` only when the payoff monitors
intermediate observations (barriers, Asians, autocallables).

See [theory §2.9](../theory.md#29-two-asset-correlated-bsm-and-spread-options)
for the analytical spread-option formulas (Margrabe / Kirk) that the
multi-asset MC recipes are validated against.

### SABR

```python
from valax.pricing.mc import generate_sabr_paths

fwd_paths, vol_paths = generate_sabr_paths(
    sabr_model,
    forward=jnp.array(100.0),
    T=1.0, n_steps=100, n_paths=50_000,
    key=jax.random.PRNGKey(0),
)
```

### LMM

Simulates $N$ correlated forward rates under the spot LIBOR measure.
Uses log-Euler for positivity:

```python
from valax.pricing.mc import generate_lmm_paths

result = generate_lmm_paths(
    lmm_model,
    n_steps_per_period=20,
    n_paths=20_000,
    key=jax.random.PRNGKey(0),
)
# result.forwards_at_fixing.shape == (n_paths, N)
# result.discount_factors.shape == (n_paths, N+1)
```

## 4. Payoff functions (low-level)

Every payoff is a pure function `(paths, instrument, *extras) -> cashflows`
returning a per-path array. Payoffs are written to be **pathwise
differentiable** — `jax.grad` through the pricing pipeline gives
autodiff Greeks whenever the payoff is (or is smoothed to be) continuous.

### Equity payoffs

| Function | Instrument | Notes |
|----------|-----------|-------|
| `european_payoff(paths, option)` | `EuropeanOption` | Terminal-only |
| `asian_option_payoff(paths, option)` | `AsianOption` | `option.averaging` ∈ {"arithmetic", "geometric"} |
| `equity_barrier_payoff(paths, option)` | `EquityBarrierOption` | Reads barrier params from instrument |
| `barrier_payoff(paths, option, barrier, is_up, is_knock_in, smoothing)` | any | Manual barrier construction |
| `lookback_payoff(paths, option)` | `LookbackOption` | Floating + fixed strike |
| `variance_swap_payoff(paths, swap, annual_factor)` | `VarianceSwap` | Mean-zero realized variance estimator |

### Multi-asset equity payoffs

| Function | Instrument | Notes |
|----------|-----------|-------|
| `spread_option_mc_payoff(paths, option, asset1_index, asset2_index)` | `SpreadOption` | Payoff on $S_1(T) - S_2(T)$; `asset*_index` pick columns of a multi-asset paths array |
| `worst_of_basket_payoff(paths, option, initial_spots)` | `WorstOfBasketOption` | Payoff on $\min_i S_i(T)/S_i(0)$; `initial_spots` normalises to return space |

### Rate payoffs (LMM)

Rate payoffs consume the `LMMPathResult` and the instrument, and take
extra args that map the instrument's accrual periods to the LMM tenor
structure (indices + accrual fractions).

| Function | Instrument | Extra args |
|----------|-----------|-----------|
| `caplet_mc_payoff(result, caplet, forward_index, tau)` | `Caplet` | `forward_index`, `tau` |
| `cap_mc_payoff(result, cap, forward_indices, taus)` | `Cap` | `forward_indices`, `taus` |
| `swaption_mc_payoff(result, swaption, forward_indices, taus)` | `Swaption` (European) | same |
| `bermudan_swaption_lsm(result, swaption, exercise_indices, taus, config)` | `BermudanSwaption` | LSM regression config |

All LMM rate payoffs return cashflows **already discounted to time 0**
using path-wise discount factors derived from the realized forwards.
That's why the dispatcher recipes use `mean(cashflows)` directly
without a separate `exp(-rT)` term.

## 5. Greeks through Monte Carlo

Because the entire pipeline is pure JAX, `jax.grad` flows through path
generation, payoff evaluation, and discounting. This is the **pathwise
method** (§[theory 6.3](../theory.md#63-pathwise-method-for-mc-greeks)).

```python
def price_fn(spot):
    return mc_price_dispatch(
        option, model,
        MCConfig(n_paths=20_000, n_steps=100),
        jax.random.PRNGKey(42),
        spot=spot,
    ).price

delta = jax.grad(price_fn)(jnp.array(100.0))
gamma = jax.grad(jax.grad(price_fn))(jnp.array(100.0))
```

Same idea for vega — differentiate w.r.t. a model parameter:

```python
def price_fn(vol):
    m = BlackScholesModel(vol=vol, rate=jnp.array(0.05), dividend=jnp.array(0.02))
    return mc_price_dispatch(
        option, m,
        MCConfig(n_paths=20_000, n_steps=100),
        jax.random.PRNGKey(42),
        spot=jnp.array(100.0),
    ).price

vega = jax.grad(price_fn)(jnp.array(0.20))
```

### When pathwise fails — discontinuous payoffs

Pathwise differentiation assumes the payoff is continuous in the path.
Hard barriers and digitals violate this. Two options in VALAX today:

1. **Smooth the payoff.** `EquityBarrierOption.smoothing > 0` replaces
   the indicator function with a sigmoid — biased but differentiable.
   Digital options will work the same way once implemented.
2. **Use more paths.** The bias from smoothing decreases as the
   smoothing width shrinks; the MC error shrinks as $1/\sqrt{N}$. Tune
   both together.

The **likelihood-ratio (score function) method** differentiates the
density rather than the payoff, handling indicators exactly — that's a
planned addition (see §5.3 theory caveats).

### Common-seed Greeks

For all derivatives taken with the same `key`, the path realizations
are identical, so the Greeks are consistent with the price — no
random-seed noise between `price_fn(x)` and `price_fn(x + ε)`.
**Always pass the same `key`** when computing bumped finite-difference
benchmarks for validation.

## 6. Adding a new MC recipe (contributor cookbook)

Say you want to price a new `MyInstrument` under `MyModel`. Four steps:

### 6.1 Write the payoff function

Pure function, pathwise-differentiable, returning per-path cashflows:

```python
# valax/pricing/mc/payoffs.py

def my_instrument_payoff(
    paths: Float[Array, "n_paths n_steps"],
    instrument: MyInstrument,
) -> Float[Array, " n_paths"]:
    ...
```

If the payoff contains indicators, multiply by a `jax.nn.sigmoid(...)`
smoothing term controlled by an `instrument.smoothing` field — like
`equity_barrier_payoff`.

### 6.2 (If needed) add a path generator

If `MyModel` is not covered by an existing generator
(`generate_gbm_paths`, `generate_heston_paths`, ...), add a new one.
Use `diffrax` with `diffrax.VirtualBrownianTree` + `ControlTerm`; see
`paths.py` and `sabr_paths.py` for templates.

### 6.3 Register the recipe

In `valax/pricing/mc/recipes.py`:

```python
from valax.pricing.mc.dispatch import MCResult, register

@register(MyInstrument, MyModel)
def _my_recipe(
    *, instrument, model, config, key, spot, **kwargs,
) -> MCResult:
    T = instrument.expiry
    paths = generate_my_paths(model, spot, T, config.n_steps, config.n_paths, key)
    cashflows = my_instrument_payoff(paths, instrument)
    df = jnp.exp(-model.rate * T)
    from valax.pricing.mc.dispatch import discounted_mean_and_stderr
    price, stderr = discounted_mean_and_stderr(cashflows, df, config.n_paths)
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)
```

The `**kwargs` sink is important — the dispatcher passes every
`market_args` keyword through, and your recipe should ignore any it
doesn't use.

### 6.4 Add tests

Under `tests/test_mc/`:

1. Smoke test: recipe runs without error via
   `mc_price_dispatch(instrument, model, config, key, ...)`.
2. Agreement with an analytical reference if one exists (tolerance of
   `2–3 σ_MC`).
3. `jax.grad` smoke test: dispatched price is differentiable.

See `tests/test_mc/test_dispatch.py` for templates.

## 7. Legacy entry points

The older `mc_price` / `mc_price_with_stderr` in
`valax.pricing.mc.engine` are still exported and unchanged. New code
should use `mc_price_dispatch` for:

- Automatic recipe selection based on instrument type.
- Consistent `MCResult` return type (price + stderr + n_paths).
- An extension point (`register`) that does not require editing the
  engine module.

```python
# Legacy (still works)
from valax.pricing.mc import mc_price_with_stderr
price, stderr = mc_price_with_stderr(option, spot, bsm_model, config, key)

# Preferred
from valax.pricing.mc import mc_price_dispatch
result = mc_price_dispatch(option, bsm_model, config, key, spot=spot)
```

## 8. Variance reduction (planned)

Neither the dispatcher nor the low-level pipeline currently applies
variance reduction. The following are roadmap items:

| Technique | Status | Notes |
|-----------|:------:|-------|
| Antithetic variates | 🟡 | 2× for smooth payoffs; easy plug-in at the path-generator level |
| Control variates | 🟡 | Black-Scholes as control for Heston/SABR; 5–50× |
| Importance sampling | 🟡 | Problem-specific |
| Stratified / quasi-MC (Sobol, Halton) | 🟡 | `diffrax` does not yet expose quasi-random Brownian increments |

See [theory §5.3](../theory.md#53-monte-carlo-simulation) for the
underlying theory.

## 9. Performance notes

- **JIT compile once per shape.** `jax.jit(mc_price_dispatch, ...)`
  compiles a specialized kernel per `(n_paths, n_steps)` pair. Reuse
  the same `MCConfig` for many pricings to amortize compile time.
- **`vmap` across scenarios.** To reprice an instrument under 1000
  market scenarios, `vmap` the dispatcher over the scenario dimension.
  This is the foundation of the full-reval VaR/ES in
  [Risk & Scenarios](risk.md).
- **GPU.** Every MC pipeline in VALAX runs on GPU without source
  changes — `jax.config.update("jax_platform_name", "gpu")` or set
  `JAX_PLATFORMS=gpu`. See the [benchmarks page](../benchmarks.md)
  for the (planned) GPU vs. CPU numbers.

## 10. Further reading

- [Models & Theory §5.3](../theory.md#53-monte-carlo-simulation) — risk-neutral expectation, SDE discretization, convergence, variance reduction, pathwise vs. likelihood-ratio Greeks.
- [Models & Theory §6.3](../theory.md#63-pathwise-method-for-mc-greeks) — when pathwise differentiation works and when it doesn't.
- [Greeks via Autodiff](greeks.md) — how `jax.grad` composes with the MC pipeline.
- [Risk & Scenarios](risk.md) — vmapped MC pricing for VaR / ES.
- [Roadmap](../roadmap.md) — execution order for the 🟡 / 🟠 / 🔴 items in the coverage map.
