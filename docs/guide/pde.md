# PDE Solvers

VALAX prices derivatives by solving the pricing PDE with finite differences —
Crank–Nicolson in 1-D and Alternating-Direction Implicit (ADI) schemes in 2-D —
with the tridiagonal systems solved via [lineax](https://docs.kidger.site/lineax/).
Because the entire solve is written in JAX, prices are differentiable
end-to-end: **Greeks come from `jax.grad` through the solver**, never from
bumping.

!!! note "What ships today"
    Beyond the **1-D European Black–Scholes** solver of [§1–§4](#1-the-black-scholes-pde)
    (`valax.pricing.pde.pde_price`), most of the **multivariate subsystem** now ships
    through `pde_price_dispatch`: 1-D equity (European / American / Digital / Barrier
    under Black–Scholes, European under local vol), 2-D Heston European (ADI), and the
    full short-rate PDE (Hull-White / G2++ / Cheyette across bonds, caps, swaptions,
    Bermudans and CMS-spread; §5.6). The remaining 2-D items (American Heston, SLV,
    spread) are in progress — see the per-row status in the §5.2 coverage matrix and
    the [PDE design doc](../architecture/pde-design.md) and
    [theory §5.2](../theory/pricing-methods.md#52-pde-finite-differences) for the math.

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

!!! note "Second-order spot Greeks (gamma)"

    Gamma works because the price read-off is a **curvature-carrying cubic**
    and the grid is *detached* from the differentiated spot (built on
    `lax.stop_gradient(spot)`, so it stays centred on spot but does not
    co-move with it). A piecewise-linear read-off on a co-moving grid makes the
    priced value piecewise-linear in `S` — exact delta but gamma `≡ 0`. See the
    [PDE design doc](../architecture/pde-design.md#read-off-grid-detachment-and-second-order-greeks-gamma)
    for the full derivation ($\Gamma = (V_g'' - V_g')/S^2$).

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

!!! note "Mostly shipped — a few 2-D rows still in progress"
    Most of the multivariate subsystem below ships today via `pde_price_dispatch`
    (see the per-row status in the [coverage matrix](#52-coverage-matrix)). A few
    2-D rows — American Heston, SLV, and spread options — remain in progress and
    their signatures may still change. Track status in the
    [design doc](../architecture/pde-design.md#10-delivery-phases).

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

Through `valax.price(..., method="pde")` the config is optional: `config=None`
takes the pair's default from `valax.core.policy` — `PDEConfig()` for the 1-D
equity and Hull-White grids, `PDEConfig2D()` for Heston and G2++, and for
Cheyette 1F / 2F the calibrator's own frozen-y grids, so a Cheyette fit and a
price both left at `config=None` share one grid (and `check_engine` accepts
it). The grid used is recorded on `PricingResult.config`
([D15](../architecture/calibration-pricing-risk.md#3-decisions)).

### 5.2 Coverage matrix

Status: ✅ shipped via `pde_price_dispatch` · 🟡 in progress (design-stage).

| Instrument | Model | Dimension | Scheme | Exercise | Status | Cross-checked against |
|---|---|---|---|---|:---:|---|
| `EuropeanOption` | `BlackScholesModel` | 1-D | Crank–Nicolson | European | ✅ | analytic BS |
| `AmericanOption` | `BlackScholesModel` | 1-D | CN + penalty | American | ✅ | binomial (`american=True`) |
| `DigitalOption` | `BlackScholesModel` | 1-D | CN + Rannacher | European | ✅ | tight call-spread proxy |
| `EquityBarrierOption` | `BlackScholesModel` | 1-D | CN + absorbing BC | European | ✅ | MC within `2·stderr` |
| `EuropeanOption` | `HestonModel` | 2-D `(S,v)` | ADI (Craig–Sneyd) | European | ✅ | Heston COS, QuantLib FD |
| `AmericanOption` | `HestonModel` | 2-D `(S,v)` | ADI + penalty | American | 🟡 | MC / QuantLib FD |
| `EuropeanOption` | `LocalVolModel` | 1-D | CN | European | ✅ | Dupire MC |
| `EuropeanOption` | `SLVModel` | 2-D `(S,v)` | ADI | European | 🟡 | SLV MC |
| `SpreadOption` | `MultiAssetGBMModel` | 2-D `(S₁,S₂)` | ADI | European | 🟡 | Margrabe / Kirk |
| `CallableBond` | `HullWhiteModel` | 1-D `r` | CN + projection | Bermudan | ✅ | Hull–White tree |
| `PuttableBond` | `HullWhiteModel` | 1-D `r` | CN + projection | Bermudan | ✅ | Hull–White tree |

The short-rate rows above are the equity-facing subset; the Hull-White / G2++ /
Cheyette PDE also prices fixed-rate bonds, caplets/caps, European and Bermudan
swaptions, and CMS-spread swaps (see §5.6 below and [API: Pricing](../api/pricing.md)),
all with piecewise-constant volatility \(\sigma(t)\) ([§5.8](#58-time-dependent-volatility)).

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

### 5.5 Non-smooth payoffs, Rannacher start-up and cell averaging

Crank–Nicolson can oscillate near a payoff discontinuity (digitals) or kink
(vanillas at the strike), polluting gamma and vega. Setting `rannacher_steps`
runs that many fully-implicit steps first to damp the oscillations before
switching to CN — enabled by default for digital and barrier recipes.

Rannacher damps the *time* stepping; it does nothing about where the kink sits
relative to the *nodes*. The short-rate engines (Hull–White, G2++, Cheyette 1F/2F
and exact-\(y\)) therefore also **cell-average** every kinked payoff they sample:

- the terminal \(\max(\cdot, 0)\) of swaptions and the Bermudan's last
  exercise;
- caplet / floorlet fixing values, both the single caplet and each caplet of a
  cap strip;
- the Bermudan exercise projection \(\max(C, E)\), and the callable / puttable
  bond projections \(\min(C, K)\) / \(\max(C, K)\).

Each node near the kink gets a \(C^1\) function of the payoff and of its
variation across the node's cell, instead of the point value. Without this,
the discrete price is piecewise smooth, with a kink whenever a node crosses
the exercise boundary. With it, the price is \(C^1\) in the model parameters,
curve and strike, and finite-difference risk no longer depends on the step: it
equals `jax.grad`. The bias is \(O(h^2)\), the order of the schemes.

- **1-D grids** (Hull–White, Cheyette 1F) use the exact cell average. This
  also cleans up convergence: the European swaption error against Jamshidian
  falls by 4.0 per refinement, where pointwise sampling gave ratios such as
  −63 and 212.
- **2-D grids** (G2++, Cheyette 2F, exact-\(y\)) use a bias-neutral variant.
  It has the same band and is \(C^1\), with zero integrated bias, so prices
  stay within a few percent of the pointwise error of the pointwise prices.
  There the cell average's one-signed bias measured larger than the error it
  removes: on the G2++ dual-curve receiver checked against the closed form
  it took the error from \(4.6\times10^{-4}\) (pointwise) to
  \(6.6\times10^{-3}\), against \(3.8\times10^{-4}\) for the bias-neutral
  kernel. The kernel, its \(O(h)\) nodal dip (at most \(0.0256\,\delta\)) and
  the design decision are in
  [Hull-White PDE § 4](../theory/hull-white-pde.md#kinks-cell-averaged-payoff-and-exercise).

The switch is a static field on both configs and is on by default:

```python
from valax.pricing.pde import PDEConfig, PDEConfig2D

PDEConfig(n_spot=201, n_time=150, spot_range=5.0)                       # smoothed (default)
PDEConfig2D(n_x=81, n_y=81, n_time=100, x_range=5.0, smooth_payoff=False)  # pointwise
```

`smooth_payoff=False` restores the pointwise arithmetic bit-for-bit. Prices
move by \(O(h^2)\) between the two settings, so a calibration bound to a grid
(`engine_kind="discretised"` / `"modelling_bias"`) is bound to the setting too:
the flag is part of the config the provenance records. The equity and Heston
recipes ignore the flag. The formula, the choice of the cell half-variation
\(\delta\) (including the 2-D quadrature approximation) and the convergence
evidence are in
[Hull-White Finite Differences §4](../theory/hull-white-pde.md#kinks-cell-averaged-payoff-and-exercise);
the implementation is `valax.pricing.pde.payoff_smoothing`.

## 5.6 Interest rates: the Hull–White short-rate PDE

The same 1-D substrate prices interest-rate instruments under
`HullWhiteModel`, including the ones no closed form reaches —
**Bermudan swaptions** and callable/puttable bonds:

```python
from valax.pricing.pde import PDEConfig, pde_price_dispatch

config = PDEConfig(n_spot=401, n_time=400, spot_range=6.0)
# Kernel-level call: hw_model must be bound to its curve,
# e.g. hw_model = bind_curves(HullWhiteModel(a, sigma), curve).
price = pde_price_dispatch(bermudan_swaption, hw_model, config).price

# Front door: a curve-free model; price() binds the market's curve.
# price(bermudan_swaption, HullWhiteModel(a, sigma), market, method="pde", config=config)
```

Three things differ from the equity recipes, each for a concrete reason:

- **The mesh is in the centred state \(x\)**, not the short rate, using
  \(r(t) = x(t) + \alpha(t)\). The drift is then time-independent, the
  diffusion is too for a constant \(\sigma\) (with \(\sigma(t)\) it varies
  per step only — see [§5.8](#58-time-dependent-volatility)), and the state
  starts at the origin, so `n_spot` counts state
  nodes and `spot_range` is a half-width in standard deviations of the short
  rate at the horizon. No `stop_gradient` scaffolding is needed: the read-off
  point \(x = 0\) is fixed.
- **The edges impose zero curvature rather than a Dirichlet value**, because a
  callable bond has no closed-form far-field value — it depends on the whole
  remaining exercise schedule.
- **Coupons and exercise use a discrete-event hook** on the backward sweep,
  with exercise decided ex-coupon and cashflows scaled by the analytic
  Hull-White bond price from their snapped level to their true date (so
  cashflow timing contributes no discretisation error). Ex-coupon is not the
  same as clean: the obstacle also carries the accrued interest owed on the
  exercise date, measured on the true date rather than the snapped level, so
  the exercise boundary does not depend on `n_time`. See
  [Dates and Conventions](dates-and-conventions.md#accrued-interest-clean-and-dirty).

Because schedules are scattered with traced indices, nothing has to be
concrete — unlike the trinomial tree, whose half-width is an array shape. The
pricers compose with `eqx.filter_jit` and `eqx.filter_grad` directly, and
sensitivities to \(a\) and \(\sigma\) match finite differences to
\(10^{-8}\)–\(10^{-11}\).

See [Hull-White Finite Differences](../theory/hull-white-pde.md) for the
derivation and validation, and [Callable Bonds](callable-bonds.md) for how the
tree, PDE and Monte-Carlo engines compare on the same instrument.

## 5.7 Multi-curve: OIS discounting with a separate projection curve

Every Hull-White and G2++ recipe takes two optional market arguments,
`discount_curve` and `forward_curve`:

```python
from valax.pricing.pde import PDEConfig, pde_price_dispatch

config = PDEConfig(n_spot=401, n_time=400, spot_range=6.0)
hw_model = HullWhiteModel(mean_reversion=a, volatility=sigma)   # curve-free
result = pde_price_dispatch(
    payer_swaption, hw_model, config,
    discount_curve=ois,          # binds the unbound model to OIS
    forward_curve=sofr_3m,       # tenor curve the floating leg projects off
)
```

Or, through the unified front door, by letting the instrument name its own
curves — `price(...)` resolves `discount_id` / `forward_id` on the
`MarketState` and forwards them for you:

```python
from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.curves.graph import CurveGraph

swaption = Swaption(..., discount_id="USD.SOFR.OIS", forward_id="USD.LIBOR.3M")
market = MarketState(
    curves=CurveGraph(curves={"USD.SOFR.OIS": ois, "USD.LIBOR.3M": sofr_3m})
)
result = price(swaption, hw_model, market, method="pde", config=config)
```

Two mechanisms are at work, and they are different in kind:

- **Discounting is a binding, not a re-weighting.** Hull-White and G2++ are
  exact-fit models, so their `initial_curve` *is* the discount curve, and
  \(\alpha\) (resp. \(\varphi\)) is fitted to whichever curve is bound.
  `price()` binds it from the `MarketState` (the curve `model.curve_id`, else
  the trade's `discount_id`) and refuses a pre-bound model; called directly, a
  recipe's `discount_curve` fills an **unbound** model via
  `valax.pricing.multicurve.with_discount_curve` and never re-fits a bound
  one. The calibrated dynamics are untouched.
- **Projection is a deterministic overlay.** A one-factor short rate produces
  one term structure, so the tenor basis is *frozen* at its time-zero value
  and enters each payoff as a constant multiplier. See
  [Multi-Curve Short Rates](../theory/multi-curve-short-rate.md) for the
  derivation and — importantly — for what it cannot price (basis volatility).

Both default to `None`, which is the single-curve case: the recipes then run
the arithmetic they always ran, so single-curve prices are unchanged
bit-for-bit. Whether the overlay applies is a property of the model type
(`valax.pricing.multicurve.basis_overlay`: Hull–White and G2++ yes, Cheyette
no), and a trade that resolves a distinct projection curve the recipe/model
cannot use **raises** rather than being silently priced single-curve — e.g.
the G2++ CMS-spread recipe, which rebuilds its par swap rates from the model's
own bond strip, or any Cheyette recipe.

## 5.8 Time-dependent volatility

Every short-rate PDE recipe accepts a **piecewise-constant volatility**: the
volatility fields are arrays of buckets with interior breakpoints
`volatility_times` (`HullWhiteModel(a, volatility=[...], volatility_times=[...])`,
`G2PPModel(..., volatility_x=[...], volatility_ratio=[...], ...)` with
\(\eta = \text{ratio}\cdot\sigma\), `CheyetteModel(kappa, volatility=[...], skew, ...)`,
`CheyetteModel2F(..., volatility_1=[...], volatility_ratio=[...], ...)`). The
time grid stays uniform (`n_time` steps to the horizon); what changes is that
each backward step diffuses at the **exact step average** of the variance
rates, so a step straddling a breakpoint carries the right variance. A
one-bucket model takes the original constant-operator path, unchanged. The
maths is in [Time-Dependent Volatility](../theory/time-dependent-volatility.md).

| Model | Grid | Operator for \(\sigma(t)\) | Constant-only builder (raises `ValueError` for > 1 bucket) | Measured accuracy |
|---|---|---|---|---|
| `HullWhiteModel` | 1-D CN | `hw_operator_stack` — stacked rows, step-averaged \(\sigma^2\) | — | second order to Jamshidian even for a 5× steep \(\sigma(t)\); finest grid (401 × 240) \(1.4\times10^{-5}\) relative |
| `G2PPModel` | 2-D ADI | `g2pp_operator_rows` — one operator per step, assembled inside the ADI loop (step averages of \(\sigma^2, \eta^2, \rho\sigma\eta\)) | `g2pp_operator_2d` | converges to Brigo–Mercurio: \(-3.2\times10^{-3}\), \(-1.1\times10^{-3}\), \(-4.3\times10^{-4}\) on 61²×50 / 111²×90 / 161²×140 |
| `CheyetteModel` (frozen-\(y\)) | 1-D CN | `cheyette_operator_stack` | — | at \(\beta = 0\) equals the Hull-White PDE to round-off (\(\le 3.3\times10^{-16}\)) |
| `CheyetteModel` (exact-\(y\), `cheyette_exact_price`) | 2-D | `cheyette_exact_operator_rows` | `cheyette_exact_operator_2d` | ~first order (upwinded \(y\) axis), same rate as constant \(\sigma\): \(-8.2\times10^{-3}\) … \(-7.2\times10^{-4}\) on 41 … 321 \(x\)-nodes |
| `CheyetteModel2F` (frozen-\(y\)) | 2-D ADI | `cheyette_operator_rows_2d` | `cheyette_operator_2d` | at \(\beta_i = 0\) equals the G2++ PDE to round-off |

The recipes choose the right builder themselves; the "constant-only" column
matters only if you assemble operators by hand. Evidence:
`tests/test_pde/test_{hull_white,g2pp,cheyette,cheyette2f}_piecewise.py`.

Bermudan swaption through the front door, on a 3-bucket Hull-White model,
with a bucketed vega:

```python
import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp

from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import BermudanSwaption
from valax.models import HullWhiteModel
from valax.pricing.pde import PDEConfig

ref = int(ymd_to_ordinal(2026, 1, 1))
pillars = jnp.array([ref + 365 * k for k in range(21)], dtype=jnp.int32)
curve = DiscountCurve(
    pillar_dates=pillars,
    discount_factors=jnp.exp(-0.035 * (pillars - ref) / 365.0),
    reference_date=jnp.int32(ref),
    day_count="act_365",
)
market = MarketState.single_curve(curve)

# Curve-free model (price() binds the market curve); sigma(t) in three buckets.
model = HullWhiteModel(
    mean_reversion=jnp.array(0.05),
    volatility=jnp.array([0.012, 0.008, 0.015]),
    volatility_times=jnp.array([1.0, 3.0]),
)
days = lambda years: jnp.array([ref + 365 * y for y in years], dtype=jnp.int32)
bermudan = BermudanSwaption(
    exercise_dates=days(range(2, 7)), fixed_dates=days(range(3, 8)),
    strike=jnp.array(0.035), notional=jnp.array(1.0), is_payer=True,
    day_count="act_365",
)
config = PDEConfig(n_spot=401, n_time=600, spot_range=6.0)


@eqx.filter_jit
def bermudan_pde(m):
    return price(bermudan, m, market, method="pde", config=config).price


print(f"Bermudan, PDE: {float(bermudan_pde(model)):.8f}")
vega = eqx.filter_grad(bermudan_pde)(model).volatility
print("Bermudan vega by bucket:", jnp.round(vega, 6))
```

```text
Bermudan, PDE: 0.02995125
Bermudan vega by bucket: [0.931533 1.023886 0.636331]
```

`n_time=600` over the 6Y horizon puts every annual exercise date exactly on a
level (exercise dates snap to the nearest level); with `n_time=400` the same
trade priced 0.02995486. The general Hull-White tree prices this trade at
0.02994999 (Richardson 600/1200 steps) — see
[Lattice → Swaptions on the tree](lattice.md#swaptions-on-the-tree).

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
- [Hull-White Finite Differences](../theory/hull-white-pde.md) — the short-rate
  PDE: state variable, boundary treatment, discrete events, validation.
- [Multi-Curve Short Rates](../theory/multi-curve-short-rate.md) — the frozen
  tenor-basis overlay behind `discount_curve` / `forward_curve`.
- [Numerical Pitfalls](../architecture/numerical-pitfalls.md) — a register of
  discretisation bugs this solver has actually shipped and fixed, each with the
  debugging playbook that localised it. Read before adding a scheme or a
  boundary condition.
- [Theory §5.2](../theory/pricing-methods.md#52-pde-finite-differences) — derivations, ADI
  schemes, penalty method, Rannacher start-up.
- [Monte Carlo](monte-carlo.md) and [Binomial Trees](lattice.md) — complementary
  numerical methods and cross-checks.
