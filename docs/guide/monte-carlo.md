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

| Instrument | `BlackScholesModel` | `HestonModel` | Notes |
|-----------|:---:|:---:|---|
| `EuropeanOption` | ✓ | ✓ | |
| `AsianOption` | ✓ | ✓ | Arithmetic + geometric averaging |
| `EquityBarrierOption` | ✓ | ✓ | KI/KO with sigmoid smoothing |
| `LookbackOption` | ✓ | ✓ | Floating + fixed strike |
| `VarianceSwap` | ✓ | ✓ | Realized variance from log-returns |
| `AmericanOption` | 🟡 | 🟡 | LSM engine exists for rates — needs lifting |
| `DigitalOption` | 🟡 | 🟡 | Payoff is smoothed Heaviside |
| `SpreadOption` | 🟠 | 🟠 | Needs correlated 2-asset GBM paths |
| `Autocallable` / `WorstOfBasket` / `Cliquet` | 🟠 | 🟠 | Need correlated multi-asset GBM + multi-observation engine |

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
