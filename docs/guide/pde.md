# PDE Solvers

VALAX prices derivatives by solving the pricing PDE with finite differences —
Crank–Nicolson in 1-D and Alternating-Direction Implicit (ADI) schemes in 2-D —
with the tridiagonal systems solved via [lineax](https://docs.kidger.site/lineax/).
Because the entire solve is written in JAX, prices are differentiable
end-to-end: **Greeks come from `jax.grad` through the solver**, never from
bumping.

!!! note "What ships today vs. the multivariate design"
    The **1-D European Black–Scholes** solver described in
    [§1–§4](#1-the-black-scholes-pde) is available now via
    `valax.pricing.pde.pde_price`. The **multivariate subsystem** (2-D ADI,
    early exercise, dispatch across models/instruments) described from
    [§5](#5-the-multivariate-subsystem) onward is a design specification —
    the interface below is the intended API and is being delivered in phases.
    See the [PDE design doc](../architecture/pde-design.md) for the full
    architecture and [theory §5.2](../theory.md#52-pde-finite-differences) for
    the math.

## 1. The Black-Scholes PDE

In log-spot space $x = \ln S$:

$$\frac{\partial V}{\partial t} + \left(r - q - \frac{\sigma^2}{2}\right) \frac{\partial V}{\partial x} + \frac{\sigma^2}{2} \frac{\partial^2 V}{\partial x^2} - rV = 0$$

The PDE is solved backward in time from the terminal payoff using Crank–Nicolson
($\theta = 0.5$), which is unconditionally stable and second-order accurate in
both time and space.

## 2. Usage (1-D, available today)

```python
import jax.numpy as jnp
from valax.instruments import EuropeanOption
from valax.pricing.pde import pde_price, PDEConfig

option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)

price = pde_price(
    option,
    spot=jnp.array(100.0),
    vol=jnp.array(0.20),
    rate=jnp.array(0.05),
    dividend=jnp.array(0.02),
    config=PDEConfig(n_spot=400, n_time=400, spot_range=4.0),
)
```

## 3. Configuration

| Parameter | Default | Description |
|---|---|---|
| `n_spot` | 200 | Number of spatial grid points (interior) |
| `n_time` | 200 | Number of time steps |
| `spot_range` | 4.0 | Grid extends `spot_range * vol * sqrt(T)` standard deviations from `ln(spot)` |

Finer grids give more accurate prices at the cost of computation time. Because
the PDE is second-order, the error scales like $O(\Delta x^2) + O(\Delta t^2)$ —
doubling `n_spot` and `n_time` roughly quarters the error.

## 4. Greeks via Autodiff

The entire PDE solve — grid construction, Crank–Nicolson time-stepping,
tridiagonal solves, and interpolation — is differentiable:

```python
import jax

delta = jax.grad(lambda s: pde_price(option, s, vol, rate, div, config))(spot)
vega = jax.grad(lambda v: pde_price(option, spot, v, rate, div, config))(vol)
gamma = jax.grad(jax.grad(lambda s: pde_price(option, s, vol, rate, div, config)))(spot)
```

### Implementation details (1-D)

- **Log-spot grid**: Uniform spacing in $x = \ln S$ ensures equal resolution
  across all spot levels.
- **Tridiagonal solver**: Each Crank–Nicolson time step solves a tridiagonal
  linear system, handled by `lineax.Tridiagonal()`.
- **`jax.lax.scan`**: The backward time-stepping loop uses `lax.scan` for JIT
  compilation — no Python-level loop overhead.
- **Boundary conditions**: Derived from Black–Scholes asymptotics for very
  small/large spot.

---

## 5. The multivariate subsystem

!!! warning "Design-stage interface"
    Everything from here on describes the **intended** API of the multivariate
    PDE package. Signatures are illustrative and subject to revision during
    implementation. Track status in the [design doc](../architecture/pde-design.md#10-delivery-phases).

The multivariate subsystem generalises the 1-D solver along three axes:

1. **Dimension** — 2-D backward PDEs for stochastic-volatility `(S, v)` and
   two-asset `(S₁, S₂)` problems, solved with ADI operator splitting.
2. **Exercise** — European, American (continuous free boundary), and Bermudan
   (discrete exercise dates).
3. **Model & instrument coverage** — a dispatcher routes each
   `(instrument, model)` pair to the right solver, mirroring the Monte Carlo
   `mc_price_dispatch` design.

### 5.1 The dispatcher

```python
from valax.pricing.pde import pde_price_dispatch, PDEConfig, PDEConfig2D

result = pde_price_dispatch(instrument, model, config, spot=spot)
price = float(result)          # PDEResult unwraps to a scalar
```

`pde_price_dispatch` looks up a recipe keyed on
`(type(instrument), type(model))` and raises a `ValueError` listing the
available recipes if the pair is not registered — identical ergonomics to the
MC dispatcher.

### 5.2 Coverage matrix (target)

| Instrument | Model | Dimension | Scheme | Exercise | Cross-checked against |
|---|---|---|---|---|---|
| `EuropeanOption` | `BlackScholesModel` | 1-D | Crank–Nicolson | European | analytic BS |
| `AmericanOption` | `BlackScholesModel` | 1-D | CN + penalty | American | binomial (`american=True`) |
| `DigitalOption` | `BlackScholesModel` | 1-D | CN + Rannacher | European | tight call-spread proxy |
| `EquityBarrierOption` | `BlackScholesModel` | 1-D | CN + absorbing BC | European | MC within `2·stderr` |
| `EuropeanOption` | `HestonModel` | 2-D `(S,v)` | ADI (Craig–Sneyd) | European | Heston COS, QuantLib FD |
| `AmericanOption` | `HestonModel` | 2-D `(S,v)` | ADI + penalty | American | MC / QuantLib FD |
| `EuropeanOption` | `LocalVolModel` | 1-D | CN | European | Dupire MC |
| `EuropeanOption` | `SLVModel` | 2-D `(S,v)` | ADI | European | SLV MC |
| `SpreadOption` | `MultiAssetGBMModel` | 2-D `(S₁,S₂)` | ADI | European | Margrabe / Kirk |
| `CallableBond` | `HullWhiteModel` | 1-D `r` | CN + projection | Bermudan | Hull–White tree |
| `PuttableBond` | `HullWhiteModel` | 1-D `r` | CN + projection | Bermudan | Hull–White tree |

### 5.3 Early exercise

Two exercise styles are exposed through the config:

- **American** (continuous exercise) uses a **penalty method** — a smooth
  forcing term keeps the solution above the payoff at every node, so Greeks stay
  differentiable.
- **Bermudan / callable / puttable** (discrete dates) uses **explicit
  projection** — the continuation value is compared with the exercise value only
  at the exercise dates (issuer-optimal `min` for calls, holder-optimal `max`
  for puts).

```python
from valax.pricing.pde import PDEConfig, Exercise

config = PDEConfig(n_spot=400, n_time=400, exercise=Exercise.AMERICAN)
result = pde_price_dispatch(american_option, bs_model, config, spot=spot)
```

### 5.4 Stochastic volatility (2-D ADI)

For Heston and (local-)stochastic vol, the price solves a 2-D backward PDE on a
`(log S, v)` grid. Each time step is split by **ADI** into per-axis tridiagonal
solves, with the spot–vol correlation entering through an explicitly-treated
mixed-derivative term. The `PDEConfig2D` config controls both axes:

```python
from valax.pricing.pde import PDEConfig2D, Scheme

config = PDEConfig2D(
    n_x=128,          # log-spot nodes
    n_y=64,           # variance nodes
    n_time=200,
    scheme=Scheme.CRAIG_SNEYD,   # 2nd-order in the cross term
    rannacher_steps=2,           # damp payoff-kink oscillations
)
result = pde_price_dispatch(european_option, heston_model, config, spot=spot)
```

Scheme choice:

- `Scheme.DOUGLAS` — robust default for weakly correlated problems.
- `Scheme.CRAIG_SNEYD` — restores second-order accuracy with the mixed term
  (accuracy default).
- `Scheme.HV` (Hundsdorfer–Verwer) — better damping for strongly correlated /
  convection-dominated regimes (large `|ρ|`).

### 5.5 Non-smooth payoffs and Rannacher start-up

Crank–Nicolson can oscillate near a payoff discontinuity (digitals) or kink
(vanillas at the strike), polluting gamma and vega. Setting `rannacher_steps`
runs that many fully-implicit steps first to damp the oscillations before
switching to CN — enabled by default for digital and barrier recipes.

## 6. When to use PDE

- **Low-dimensional problems** (single underlying, or one spot + one variance
  factor) where you want the whole `(S, t)` (or `(S, v, t)`) solution surface.
- **Early exercise** — American/Bermudan features are natural on a backward
  grid.
- **Barriers and digitals** — an absorbing boundary and Rannacher start-up give
  cleaner values and Greeks than MC or CRR near the barrier/strike.

PDE is subject to the curse of dimensionality: beyond ~2–3 state variables,
[Monte Carlo](monte-carlo.md) is the right tool. For American options on a
single asset, the [binomial tree](lattice.md) is a lightweight alternative.

## 7. See also

- [PDE design doc](../architecture/pde-design.md) — full architecture, ADI and
  early-exercise rationale, delivery phases.
- [Theory §5.2](../theory.md#52-pde-finite-differences) — derivations, ADI
  schemes, penalty method, Rannacher start-up.
- [Monte Carlo](monte-carlo.md) and [Binomial Trees](lattice.md) — complementary
  numerical methods and cross-checks.
