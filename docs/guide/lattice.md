# Lattice Methods

VALAX provides two recombining-lattice pricing engines: a Cox-Ross-Rubinstein
(CRR) binomial tree for equity vanillas, and a Hull-White trinomial tree for
short-rate bonds with embedded optionality and for European / Bermudan
swaptions, with constant or piecewise-constant volatility \(\sigma(t)\).

## Recipe details

**What the lattice engines cover** is stated once, in the *Lattice* and
*`price()` routes* columns of the
[coverage matrix](../applications/instrument-pricing.md#coverage-matrix),
checked against the registry. This table gives the engine-specific details:

| Instrument | Model | Dimension | Scheme | Exercise | Cross-checked against |
|---|---|---|---|---|---|
| `EuropeanOption` | `BlackScholesModel` | 1-D | CRR binomial | European | analytic BS |
| `AmericanOption` (kernel: `binomial_price(..., american=True)` on a `EuropeanOption`) | `BlackScholesModel` | 1-D | CRR binomial | American | PDE (`AmericanOption`, CN + penalty) |
| `CallableBond` | `HullWhiteModel` | 1-D `r` | trinomial | Bermudan (call dates) | PDE / MC |
| `PuttableBond` | `HullWhiteModel` | 1-D `r` | trinomial | Bermudan (put dates) | PDE / MC |
| `Swaption` | `HullWhiteModel` | 1-D `r` | trinomial | European | Jamshidian (`hw_swaption_price`) |
| `BermudanSwaption` | `HullWhiteModel` | 1-D `r` | trinomial | Bermudan | PDE, QuantLib `Gaussian1dSwaptionEngine` |

Every Hull-White row accepts a **piecewise-constant volatility** \(\sigma(t)\)
(`HullWhiteModel(volatility=[...], volatility_times=[...])`): a one-bucket
model runs the classical Hull & White (1994) tree, a multi-bucket model the
[general tree](#general-tree-for-piecewise-constant-volatility). All four
Hull-White rows are also reachable through `price(..., method="lattice")` —
see [Hull-White tree through `price()`](#hull-white-tree-through-price).

## When to use a tree

The lattice engine is deliberately narrower than the [PDE](pde.md) — it carries
no local-vol/stochastic-vol models, no barriers or digitals, and no 2-D grids.
Reach for a tree when its specific strengths matter:

- **Lightweight single-asset early exercise** — a CRR tree is one self-contained
  backward induction with no grid/operator/boundary/scheme machinery, making it
  the low-overhead route to American vanillas on a single underlying.
- **Independent cross-validation oracle** — a recombining tree is a
  *structurally different* numerical method from finite differences, so it
  catches discretisation bugs a second FD scheme would share. The PDE
  American-option, Hull-White callable/puttable and $\sigma(t)$ Bermudan-swaption
  recipes are validated against these trees.
- **No far-field boundary to model** — trees need no artificial truncation of
  the state space, removing one error source (relevant for the short-rate tree
  below, whose callable payoff has no closed-form far field).

For breadth — LocalVol/Heston, barriers, digitals, 2-D ADI, the two-factor
and Cheyette models, and cleaner Greeks near kinks/barriers — prefer the
[PDE](pde.md). The Hull-White tree's swaption pricers exist mainly as that
independent oracle for the PDE and the Jamshidian closed form.

---

## Binomial Trees

VALAX implements the Cox-Ross-Rubinstein (CRR) binomial tree for European and American option pricing.

### Usage

```python
import jax.numpy as jnp
from valax.instruments import EuropeanOption
from valax.pricing.lattice import binomial_price, BinomialConfig

option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)

# European
euro = binomial_price(option, jnp.array(100.0), jnp.array(0.20),
                       jnp.array(0.05), jnp.array(0.02),
                       BinomialConfig(n_steps=500, american=False))

# American
put = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)
amer = binomial_price(put, jnp.array(100.0), jnp.array(0.20),
                       jnp.array(0.05), jnp.array(0.0),
                       BinomialConfig(n_steps=500, american=True))
```

### CRR Parameters

At each time step $\Delta t = T/N$:

$$u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{(r-q)\Delta t} - d}{u - d}$$

The tree recombines: an up move followed by a down move reaches the same node as down-then-up.

### American Exercise

When `american=True`, at each node the algorithm takes the maximum of the continuation value and the intrinsic value:

$$V_i = \max\left(\text{disc} \cdot [p \cdot V_{i+1}^u + (1-p) \cdot V_{i+1}^d],\; \text{intrinsic}_i\right)$$

Key properties:

- **American put >= European put** — early exercise premium exists.
- **American call without dividends = European call** — early exercise is never optimal.
- **American call with dividends >= European call** — may be optimal to exercise before ex-dividend.

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `n_steps` | 200 | Number of time steps in the tree |
| `american` | `False` | Enable American early exercise |

### Greeks via Autodiff

```python
import jax

delta = jax.grad(lambda s: binomial_price(
    option, s, vol, rate, div, config
))(spot)
```

### Implementation Notes

- The backward induction uses `jax.lax.scan` with fixed-shape arrays for JIT compatibility. At each step, `values[j] = disc * (p * values[j+1] + (1-p) * values[j])` is computed over the full array; unused trailing entries are harmless.
- European prices converge to Black-Scholes within 0.5% at 500 steps.

---

## Hull-White Trinomial Tree

The Hull-White one-factor model is the standard short-rate model for pricing bonds with embedded optionality. VALAX provides:

1. **`HullWhiteModel`** — an `equinox.Module` pytree carrying the mean-reversion speed $a$, the short-rate volatility $\sigma(t)$ (an array of piecewise-constant buckets — a scalar is promoted to one bucket — with `volatility_times` holding the `n_buckets - 1` interior breakpoints in year fractions; see [Time-Dependent Volatility](../theory/time-dependent-volatility.md)), and the initial discount curve $P^M(0,t)$ it is exact-fitted to. The curve is market data: `valax.price` binds it from the `MarketState` (and refuses a pre-bound model), while the kernel-level functions below need a bound model — build it with `initial_curve=curve` or `bind_curves(model, curve)` (see [API → Core § Curve fields](../api/core.md#curve-fields-exact-fit-short-rate-models)).
2. **Analytic zero-coupon bond prices** — $P(t,T \mid r) = A(t,T)\,e^{-B(t,T)\,r}$ with exact-fit to the initial curve via the Brigo-Mercurio $A(t,T)$ formula.
3. **Trinomial tree** — a recombining tree for backward induction, with Arrow-Debreu forward propagation to calibrate $\alpha(t_i)$ at each step and match market discount factors exactly.
4. **European and Bermudan swaption pricing** — `swaption_tree_price` and `bermudan_swaption_tree_price`, independent checks of the Jamshidian closed form and the PDE (see [Swaptions on the tree](#swaptions-on-the-tree)).
5. **Callable and puttable bond pricing** — backward induction with exercise decisions: `callable_bond_price` (issuer calls when continuation > call price, reducing holder value) and `puttable_bond_price` (holder puts when continuation < put price, increasing holder value). Call and put prices are quoted clean while the continuation value is dirty, so the obstacle carries accrued interest; see [Dates and Conventions](dates-and-conventions.md#accrued-interest-clean-and-dirty).

### Model

```python
import jax.numpy as jnp
from valax.models import HullWhiteModel
from valax.models.hull_white import hw_bond_price, hw_B, _instantaneous_forward

from valax.core.binding import bind_curves

model = HullWhiteModel(
    mean_reversion=jnp.array(0.10),
    volatility=jnp.array(0.01),      # scalar -> one bucket: model.volatility.shape == (1,)
)                                    # curve-free: what valax.price expects
model = bind_curves(model, curve)    # bound to a DiscountCurve for kernel calls

# Analytic ZCB price: recovers the initial curve at t=0
f0 = _instantaneous_forward(model, jnp.array(0.0))
p = hw_bond_price(model, r=f0, t=jnp.array(0.0), T=jnp.array(5.0))
# p ≈ curve(ymd_to_ordinal(2030, 1, 1))
```

### Tree construction

For a one-bucket (constant-$\sigma$) model `build_hull_white_tree` follows
Hull & White (1994), and its output is bit-identical to the pre-$\sigma(t)$
builder:

1. **Symmetric *x*-tree** — with time step $\Delta t = T/N$ and state step
   $\Delta x = \sigma\sqrt{3\Delta t}$, three branch types (normal / up / down)
   are chosen so the transition probabilities stay non-negative.
2. **Truncation** — above/below $j_{\max} \approx \lceil 0.1835/(a\,\Delta t)\rceil$
   the tree switches to up- or down-branching, capping the state width.
3. **Forward Arrow-Debreu sweep** — each shift $\alpha_i$ is solved in closed form
   so the tree-implied $P(0, t_i)$ matches the market curve; the tree therefore
   **reprices the initial curve exactly by construction**.

```python
from valax.pricing.lattice.hull_white_tree import build_hull_white_tree

tree = build_hull_white_tree(model, T=10.0, n_steps=200)
# tree.rates[i, j]    short rate at node (i, j)
# tree.probs[j, k]    (up, mid, down) transition probabilities from state j
# tree.targets[j, k]  destination state index for each branch
# tree.alpha[i]       curve-fitting shifts
```

The bond pricers drive the backward induction internally, so you rarely touch
these fields — they are exposed for inspection and for building custom rate
exotics.

### General tree for piecewise-constant volatility

With more than one volatility bucket, `build_hull_white_tree` dispatches to
`build_general_hull_white_tree` (Hull & White 2001), which returns a
`GeneralHullWhiteTree`. What changes against the constant-$\sigma$ tree, as
engine mechanics (the variance and branching formulas are in the
[`build_general_hull_white_tree`](../api/pricing.md#valax.pricing.lattice.hull_white_tree.build_general_hull_white_tree)
docstring and the maths of $\sigma(t)$ in
[Time-Dependent Volatility](../theory/time-dependent-volatility.md)):

- **The level spacing follows $\sigma(t)$.** Each level's spacing is set from
  the exact conditional variance of the centred state over the step
  (`hw_state_variance`), so a step that straddles a bucket breakpoint carries
  the right variance. Spacing, probabilities and targets therefore all carry
  a leading step axis (`tree.nodes`, `tree.probs`, `tree.targets` are
  `(n_steps[+1], n_states[, 3])`).
- **Nearest-node branching, no separate edge rule.** Each node branches around
  the node nearest its exact conditional mean, with moment-matched
  probabilities; mean reversion makes the reachable width saturate, which is
  what bounds the tree.
- **The width comes from the volatility path, not from $0.1835/(a\Delta t)$.**
  A *drop* in $\sigma(t)$ makes the next level finer, so the reachable
  half-width jumps there. `hw_tree_width(model, T, n_steps)` propagates the
  reachable width level by level and returns the smallest valid $J$; it is
  eager (NumPy on concrete parameters). In the example below, 400 steps to a
  2Y expiry with $\sigma$ falling from 1.2% to 0.8% at 1Y needs $J = 500$.
- **An undersized $J$ is NaN, never a wrong price.** A static `j_max` below
  the reachable width clips targets; if a clipped node with probability mass
  gets a negative probability, `tree.valid` is `False` and every price off
  the tree is NaN (`tests/test_lattice/test_hull_white_tree_piecewise.py`).
- **The curve is still repriced exactly**: $\alpha_i$ is fitted by the same
  Arrow-Debreu forward sweep, so rolling back a unit from level $n$ returns
  $P^M(0, t_n)$ to $10^{-12}$.

`callable_bond_price` and `puttable_bond_price` call `build_hull_white_tree`,
so they take a $\sigma(t)$ model unchanged; on a 5-bucket model the callable
agrees with the PDE to $2\times10^{-4}$ relative at 400 steps.

!!! tip "Tree resolution vs. accuracy"
    For semi-annual callables over 10 years, `n_steps=200` (≈ 18-day step) gives
    prices accurate to a few basis points. Increase to `n_steps=500+` for
    fine-grained call schedules or when computing convexity via
    `jax.grad(jax.grad(...))`.

### Callable and Puttable Bonds

```python
from valax.instruments import CallableBond, PuttableBond
from valax.pricing.lattice import callable_bond_price, puttable_bond_price

cb = CallableBond(
    payment_dates=sched, settlement_date=ref,
    coupon_rate=jnp.array(0.04), face_value=jnp.array(100.0),
    call_dates=call_dates, call_prices=jnp.array([1.0, 1.0, 1.0]),  # at par
    frequency=2,
)
p_callable = callable_bond_price(cb, model, n_steps=100)
# p_callable < straight bond price (the call option reduces holder value)
```

The tree is fully JAX-compatible: `jax.grad` through `callable_bond_price` with respect to `coupon_rate`, `face_value`, or the model's `mean_reversion` / `volatility` gives sensitivities automatically.

!!! tip "Convergence"
    100 tree steps is typically sufficient for callable/puttable bonds
    (< 0.5% deviation from the 200-step price).  For OAS or calibration
    loops where speed matters, 50 steps gives a reasonable initial
    estimate.

### Swaptions on the tree

`swaption_tree_price` (European) and `bermudan_swaption_tree_price` price on
either tree — constant or piecewise $\sigma$ — and follow the PDE recipes'
contract:

- **European**: the tree's horizon is the expiry; the underlying swap is
  valued analytically (affine Hull-White bonds) at every terminal node and
  rolled back. It is redundant with the exact Jamshidian closed form and kept
  as its independent check.
- **Bermudan**: the horizon is the last exercise date; exercise $e$ enters the
  tail swap on `fixed_dates[e:]`, valued analytically at the true exercise
  date on the nodes of the level it snaps to, and the holder takes
  $\max(C, E)$ there.
- **Smoothing**: `smooth_payoff=True` (the default, static) samples the
  kinked terminal payoff and the exercise projection as cell averages over each
  node's cell — the same machinery as the
  [PDE](pde.md#55-non-smooth-payoffs-rannacher-start-up-and-cell-averaging).
  This removes the lattice's oscillating convergence in `n_steps`: the error
  then decreases **monotonically at first order** (it halves when `n_steps`
  doubles), so the Richardson extrapolate $2V(2n) - V(n)$ is the number to
  quote. `smooth_payoff=False` restores pointwise sampling.
- **Multi-curve**: both take an optional `forward_curve` (frozen tenor basis,
  as on the PDE).
- **Traceable**: neither calls `int()` on dates, so the instrument can be a
  traced pytree; only `j_max` must be static (see
  [the caveat below](#greeks-and-the-j_max-caveat)).

**Choose `n_steps` so the exercise dates land on levels.** Dates are snapped to
the nearest level of the uniform grid `horizon / n_steps`; an off-grid date
adds a timing error that spoils the Richardson extrapolate. In the example
below the Bermudan's horizon is 6Y with annual exercise, so it uses 600 / 1200
steps (`dt = 0.01` / `0.005`); at 400 / 800 steps (`dt = 0.015`, which does
not divide 2Y) the extrapolate was off by $4\times10^{-4}$ relative.

**Accuracy evidence.**

| Check | Result | Test |
|---|---|---|
| European vs Jamshidian, smooth and 5× steep $\sigma(t)$ | first order (error ratio 0.45–0.55 per doubling); Richardson 400/800 within $5\times10^{-6}$ relative | `tests/test_lattice/test_hull_white_tree_piecewise.py` |
| Bermudan vs QuantLib `Gaussian1dSwaptionEngine` (`Gsr`), relative | tree Richardson 400/800: $-4.5\times10^{-6}$ (smooth), $-1.7\times10^{-5}$ (5× steep); PDE 401×240: $+1.1\times10^{-5}$, $+4.3\times10^{-6}$ | `tests/test_quantlib_comparison/test_hull_white_piecewise_ql.py` |
| Undersized `j_max` | NaN | `tests/test_lattice/test_hull_white_tree_piecewise.py` |
| Bucketed vega via `jax.grad` | finite; exactly zero for buckets starting after the expiry | `tests/test_lattice/test_hull_white_tree_piecewise.py` |

**Example** — a European and a Bermudan payer on a 3-bucket Hull-White model,
under `eqx.filter_jit`, with the European checked against Jamshidian:

```python
import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp

from valax.core.binding import bind_curves
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models import HullWhiteModel
from valax.pricing.analytic import hw_swaption_price
from valax.pricing.lattice import (
    bermudan_swaption_tree_price,
    hw_tree_width,
    swaption_tree_price,
)

ref = int(ymd_to_ordinal(2026, 1, 1))
pillars = jnp.array([ref + 365 * k for k in range(21)], dtype=jnp.int32)
curve = DiscountCurve(
    pillar_dates=pillars,
    discount_factors=jnp.exp(-0.035 * (pillars - ref) / 365.0),
    reference_date=jnp.int32(ref),
    day_count="act_365",
)

# sigma(t): 1.2% on [0, 1y), 0.8% on [1y, 3y), 1.5% from 3y on.
model = bind_curves(
    HullWhiteModel(
        mean_reversion=jnp.array(0.05),
        volatility=jnp.array([0.012, 0.008, 0.015]),
        volatility_times=jnp.array([1.0, 3.0]),
    ),
    curve,
)

days = lambda years: jnp.array([ref + 365 * y for y in years], dtype=jnp.int32)
european = Swaption(                       # 2y x 5y payer
    expiry_date=jnp.int32(ref + 730), fixed_dates=days(range(3, 8)),
    strike=jnp.array(0.035), notional=jnp.array(1.0), is_payer=True,
    day_count="act_365",
)
bermudan = BermudanSwaption(               # 2y-into-5y, annual exercise 2y..6y
    exercise_dates=days(range(2, 7)), fixed_dates=days(range(3, 8)),
    strike=jnp.array(0.035), notional=jnp.array(1.0), is_payer=True,
    day_count="act_365",
)


def european_on_tree(n):
    j = hw_tree_width(model, 2.0, n)       # static half-width, sized eagerly
    return eqx.filter_jit(lambda s, m: swaption_tree_price(s, m, n_steps=n, j_max=j))


def bermudan_on_tree(n):
    j = hw_tree_width(model, 6.0, n)       # horizon = last exercise date
    return eqx.filter_jit(
        lambda b, m: bermudan_swaption_tree_price(b, m, n_steps=n, j_max=j)
    )


# n_steps chosen so every exercise date falls exactly on a tree level.
e1, e2 = european_on_tree(400)(european, model), european_on_tree(800)(european, model)
b1, b2 = bermudan_on_tree(600)(bermudan, model), bermudan_on_tree(1200)(bermudan, model)

print(f"European, tree n=400:          {float(e1):.8f}")
print(f"European, Richardson 400/800:  {float(2 * e2 - e1):.8f}")
print(f"European, Jamshidian:          {float(hw_swaption_price(european, model)):.8f}")
print(f"Bermudan, tree n=600:          {float(b1):.8f}")
print(f"Bermudan, Richardson 600/1200: {float(2 * b2 - b1):.8f}")

# Bucketed vega through the tree: J is held fixed, so the lattice geometry is too.
berm_600 = bermudan_on_tree(600)
vega = eqx.filter_grad(lambda m: berm_600(bermudan, m))(model).volatility
print("Bermudan vega by bucket:", vega)
```

```text
European, tree n=400:          0.02228488
European, Richardson 400/800:  0.02227937
European, Jamshidian:          0.02227938
Bermudan, tree n=600:          0.02996657
Bermudan, Richardson 600/1200: 0.02994999
Bermudan vega by bucket: [0.93165654 1.02472917 0.63700188]
```

The Bermudan extrapolate agrees with the PDE example in the
[PDE guide](pde.md#58-time-dependent-volatility) (0.02995125 on 401 × 600)
to $4\times10^{-5}$ relative, and with a finer 801 × 1200 PDE (0.02995037) to
$1.3\times10^{-5}$.

### Hull-White tree through `price()`

The four Hull-White tree pricers are routed through
`valax.price(..., method="lattice")` for `CallableBond`, `PuttableBond`,
`Swaption` and `BermudanSwaption` under `HullWhiteModel`, configured by a
static `TreeConfig(n_steps=100, j_max=None, width_factor=2.0,
smooth_payoff=True)` (the pair's default when `config=None`). `method="auto"`
is unchanged — the PDE for the bonds and the Bermudan, Jamshidian for the
European swaption — so the tree stays the independent check, now one keyword
away. The routes differ from the direct calls in two ways:

- **Always the general tree**, for a constant $\sigma$ too (`general=True` on
  the direct functions does the same): its nearest-node branching is valid at
  any half-width, which the classical tree's is not (beyond about
  $0.816/(a\Delta t)$ its standard branching has negative probabilities). At
  100 steps it is also slightly *closer* to Jamshidian than the classical
  tree in the cases measured while choosing the default.
- **The half-width comes from the config alone**:
  $J = \lceil w \cdot n_\text{steps} \rceil$ with `width_factor` $w = 2$, so
  no eager sizing of a concrete model is needed and the route runs on a traced
  model under `eqx.filter_jit` and `jax.grad`. A step moves at most one level,
  but where $\sigma(t)$ drops the finer levels need more nodes for the same
  rates; the measured `hw_tree_width / n_steps` was ≤ 1.00 for a constant
  $\sigma$, 1.17 for a 0.012 → 0.009 drop, 1.25–1.30 for ×1.5 zig-zags and ×2
  humps, and 2.00 for a ×3 drop. Outer nodes beyond the reachable width
  carry no probability mass, so they cost time, not accuracy; a tree too
  narrow for its model is flagged invalid and prices NaN — raise
  `width_factor` or pass `j_max` (e.g. the tight `hw_tree_width`).

The bond routes price single-curve: a bond that resolves a projection curve
distinct from its discount curve is refused (the swaption routes take the
projection curve as a frozen tenor basis, like the PDE).

```python
import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp

from valax.core import MarketState, price
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Swaption
from valax.models import HullWhiteModel
from valax.pricing.lattice import TreeConfig

ref = int(ymd_to_ordinal(2025, 1, 1))
pillars = jnp.array([int(ymd_to_ordinal(2025 + y, 1, 1)) for y in range(12)],
                    dtype=jnp.int32)
t = (pillars - ref) / 365.0
market = MarketState.single_curve(DiscountCurve(
    pillar_dates=pillars, discount_factors=jnp.exp(-0.03 * t),
    reference_date=jnp.int32(ref), day_count="act_365"))
swaption = Swaption(
    expiry_date=pillars[2], fixed_dates=pillars[3:8], strike=jnp.asarray(0.03),
    notional=jnp.asarray(1.0), day_count="act_365")
# sigma(t) falls from 1.2% to 0.9% at 1Y: the tree needs J > n_steps there.
model = HullWhiteModel(mean_reversion=jnp.asarray(0.05),
                       volatility=jnp.asarray([0.012, 0.009, 0.011]),
                       volatility_times=jnp.asarray([1.0, 3.0]))

tree = price(swaption, model, market, method="lattice",
             config=TreeConfig(n_steps=200))
exact = price(swaption, model, market)            # auto: Jamshidian
print(tree.price, exact.price, tree.config.half_width)
# 0.0232907 vs 0.0232775 (first order in n_steps); J = 400, tight width 233

# Traced model, default config: jit and bucketed vega with no j_max.
value = lambda m: price(swaption, m, market, method="lattice").price
vega = eqx.filter_grad(eqx.filter_jit(value))(model).volatility   # (3,)
```

### Greeks and the `j_max` caveat

Because the whole pipeline is pure JAX, `jax.grad` flows through both the tree
construction and the backward induction — effective duration, convexity, and
key-rate durations come with no finite-difference bumping (see
[Callable Bonds → OAS](callable-bonds.md#option-adjusted-spread-oas) for the
option-aware measures and how they generalise the Z-spread).

!!! warning "Differentiating w.r.t. model parameters needs an explicit `j_max`"
    The lattice half-width $j_{\max}$ sets array shapes, so it must be a concrete
    Python `int`. `build_hull_white_tree` derives it from `mean_reversion` via
    `float(a)`, which raises `ConcretizationTypeError` the moment `a` becomes a
    tracer. Curve shifts leave `a` concrete and are unaffected, but
    `eqx.filter_grad` over the model pytree is not — precompute the width and pass
    it in:

    ```python
    from valax.pricing.lattice import hw_tree_j_max

    j_max = hw_tree_j_max(float(model.mean_reversion), T / n_steps)
    dP_dsigma = eqx.filter_grad(
        lambda m: callable_bond_price(bond, m, n_steps=n_steps, j_max=j_max)
    )(model).volatility
    ```

    Holding $j_{\max}$ fixed is also the *correct* sensitivity: it keeps the
    lattice geometry constant so the derivative measures the parameter's effect,
    not a discretisation that jumped a step.

    `hw_tree_j_max` is the constant-$\sigma$ rule. For a **piecewise
    $\sigma(t)$** model the general tree's width depends on the whole volatility
    path, so size it with `hw_tree_width(model, T, n_steps)` instead (eager,
    concrete model; `T` is the tree horizon — the bond maturity, the swaption
    expiry, or the Bermudan's last exercise date). The bond and swaption
    pricers pick the right rule themselves when `j_max=None`, which is only
    possible outside a trace. Because `model.volatility` is an array of
    buckets, the gradient above is the bucketed vega (shape `(n_buckets,)`;
    `(1,)` for a constant model). A `j_max` that is too small for the
    differentiated model returns NaN rather than a wrong number.
