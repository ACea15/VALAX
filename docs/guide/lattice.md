# Lattice Methods

VALAX provides two recombining-lattice pricing engines: a Cox-Ross-Rubinstein
(CRR) binomial tree for equity vanillas, and a Hull-White trinomial tree for
short-rate bonds with embedded optionality.

## Coverage matrix

| Instrument | Model | Dimension | Scheme | Exercise | Cross-checked against |
|---|---|---|---|---|---|
| `EuropeanOption` | `BlackScholesModel` | 1-D | CRR binomial | European | analytic BS |
| `EuropeanOption` | `BlackScholesModel` | 1-D | CRR binomial | American | PDE (`AmericanOption`, CN + penalty) |
| `CallableBond` | `HullWhiteModel` | 1-D `r` | trinomial | Bermudan (call dates) | PDE / MC |
| `PuttableBond` | `HullWhiteModel` | 1-D `r` | trinomial | Bermudan (put dates) | PDE / MC |

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
  American-option and Hull-White callable/puttable recipes are validated against
  these trees.
- **No far-field boundary to model** — trees need no artificial truncation of
  the state space, removing one error source (relevant for the short-rate tree
  below, whose callable payoff has no closed-form far field).

For breadth — LocalVol/Heston, barriers, digitals, 2-D ADI, Bermudan swaptions,
and cleaner Greeks near kinks/barriers — prefer the [PDE](pde.md).

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

1. **`HullWhiteModel`** — an `equinox.Module` pytree carrying the mean-reversion speed $a$, short-rate volatility $\sigma$, and the initial discount curve $P^M(0,t)$.
2. **Analytic zero-coupon bond prices** — $P(t,T \mid r) = A(t,T)\,e^{-B(t,T)\,r}$ with exact-fit to the initial curve via the Brigo-Mercurio $A(t,T)$ formula.
3. **Trinomial tree** — a recombining tree for backward induction, with Arrow-Debreu forward propagation to calibrate $\alpha(t_i)$ at each step and match market discount factors exactly.
4. **Callable and puttable bond pricing** — backward induction with exercise decisions: `callable_bond_price` (issuer calls when continuation > call price, reducing holder value) and `puttable_bond_price` (holder puts when continuation < put price, increasing holder value).

### Model

```python
import jax.numpy as jnp
from valax.models import HullWhiteModel
from valax.models.hull_white import hw_bond_price, hw_B, _instantaneous_forward

model = HullWhiteModel(
    mean_reversion=jnp.array(0.10),
    volatility=jnp.array(0.01),
    initial_curve=curve,   # a DiscountCurve
)

# Analytic ZCB price: recovers the initial curve at t=0
f0 = _instantaneous_forward(model, jnp.array(0.0))
p = hw_bond_price(model, r=f0, t=jnp.array(0.0), T=jnp.array(5.0))
# p ≈ curve(ymd_to_ordinal(2030, 1, 1))
```

### Tree construction

`build_hull_white_tree` follows Hull & White (1994):

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
