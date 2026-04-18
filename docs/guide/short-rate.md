# Short-Rate Models: Hull-White Workflow

The Hull-White one-factor model is VALAX's workhorse for products with embedded
optionality on the interest-rate curve — callable bonds, puttable bonds, and
(soon) Bermudan swaptions. This guide walks through the end-to-end workflow:
**build the model → build the trinomial tree → price callable/puttable bonds →
extract Greeks via autodiff**.

For the mathematical background (SDE, affine bond price formula, Jamshidian
decomposition, $\theta(t)$ exact-fit calibration), see
[Models & Theory §2.8](../theory.md#28-hull-white-one-factor-short-rate-model).

## 1. Why Hull-White?

The model specifies the risk-neutral short rate as an extended Vasicek process:

$$
dr_t = [\theta(t) - a\,r_t]\,dt + \sigma\,dW_t
$$

with three pieces of state:

| Component | Role | VALAX field |
|-----------|------|-------------|
| Mean-reversion speed $a$ | Controls how strongly $r_t$ pulls back to its long-run trend | `HullWhiteModel.mean_reversion` |
| Short-rate volatility $\sigma$ | Sets the scale of rate moves | `HullWhiteModel.volatility` |
| Drift $\theta(t)$ | Implicit, calibrated by VALAX so that the model exactly reprices the initial discount curve | `HullWhiteModel.initial_curve` |

The exact-fit property is what makes Hull-White suitable for production: when you
price a callable bond, the underlying non-callable cashflows are guaranteed to
discount back to today's market values. The optionality value falls out cleanly
on top.

## 2. Build the Model

Start from a discount curve. In VALAX the curve is itself a JAX pytree, so it
participates in autodiff; key-rate sensitivities of any HW-priced bond come from
`jax.grad` through the curve.

```python
import jax.numpy as jnp
from valax.curves.discount import DiscountCurve
from valax.models.hull_white import HullWhiteModel

# Today's curve — flat 4% as a toy example.
pillar_dates = jnp.array([0, 90, 180, 365, 730, 1825, 3650])  # days from ref
zero_rates = jnp.full(pillar_dates.shape, 0.04)
ref_date = jnp.array(0)

curve = DiscountCurve(
    pillar_dates=ref_date + pillar_dates,
    discount_factors=jnp.exp(-zero_rates * pillar_dates / 365.0),
    reference_date=ref_date,
    day_count="act_365",
)

model = HullWhiteModel(
    mean_reversion=jnp.array(0.05),  # a = 5%
    volatility=jnp.array(0.01),      # σ = 1% (100 bps short-rate vol)
    initial_curve=curve,
)
```

!!! note "Calibrating $a$ and $\sigma$"
    Currently the user supplies $a$ and $\sigma$ directly. **Calibration to a
    swaption surface** via the Jamshidian decomposition is on the roadmap (P1.4
    follow-up). The decomposition turns each swaption price into a portfolio of
    options on individual zero-coupon bonds, each priced by Black-76 with an
    integrated short-rate variance — fast enough to drive a least-squares fit.
    Until then, typical bank values are $a \in [0.01, 0.10]$ and
    $\sigma \in [0.005, 0.02]$.

### Sanity check: closed-form bond reprice

The analytic Hull-White zero-coupon bond price reproduces today's curve when
evaluated at $t = 0$ with $r_0 = f^M(0, 0)$:

```python
from valax.models.hull_white import hw_bond_price, _instantaneous_forward

t = jnp.array(0.0)
T = jnp.array(5.0)
r0 = _instantaneous_forward(model, t)        # f^M(0, 0)
P_hw = hw_bond_price(model, r0, t, T)        # closed-form HW bond price
P_curve = curve(jnp.array(int(365 * 5)))     # discount curve at the same date

print(f"HW bond price:    {P_hw:.6f}")
print(f"Curve DF(5Y):      {P_curve:.6f}")
# Both should match to ~1e-6.
```

## 3. Build the Trinomial Tree

For products with early exercise (callable bonds, puttable bonds, Bermudans),
VALAX builds a recombining trinomial tree via `build_hull_white_tree`. The
construction follows Hull & White (1994):

1. **Symmetric *x*-tree**: time step $\Delta t = T/N$, state step $\Delta x = \sigma\sqrt{3\Delta t}$, three branch types (normal/up/down) chosen so transition probabilities stay non-negative.
2. **Truncation**: above and below $j_{\max} \approx \lceil 0.1835 / (a\,\Delta t)\rceil$, the tree switches to up- or down-branching.
3. **Forward Arrow-Debreu sweep**: VALAX solves for each $\alpha_i$ in closed form so that the tree-implied $P(0, t_i)$ matches the market curve. The result is a tree that **reprices the initial curve exactly by construction**.

```python
from valax.pricing.lattice.hull_white_tree import build_hull_white_tree

tree = build_hull_white_tree(model, T=10.0, n_steps=200)

print(f"Time step:   {tree.dt:.4f} years")
print(f"State step:  {tree.dx:.6f}")
print(f"Tree shape:  {tree.rates.shape}  # (n_steps+1, 2*j_max+1)")
print(f"j_max:       {tree.j_max}")
```

The returned `HullWhiteTree` is a frozen pytree containing:

- `rates[i, j]` — the short rate at node $(i, j)$
- `probs[j, k]` — transition probabilities (up, mid, down) from state $j$
- `targets[j, k]` — destination state index for each branch
- `alpha[i]` — the curve-fitting shifts

You typically don't interact with these directly — the bond pricers handle
backward induction internally — but they're exposed for inspection and for
building custom rate exotics.

!!! tip "Tree resolution vs. accuracy"
    For callable bonds with semi-annual coupons over 10 years, `n_steps=200`
    (≈ 18 day step) gives prices accurate to a few basis points. Increase to
    `n_steps=500+` for fine-grained call schedules or for computing convexity
    via `jax.grad(jax.grad(...))`.

## 4. Price a Callable Bond

A callable bond is a fixed-rate bond plus a short call held by the issuer. At
each call date the issuer takes the **min** of continuation value and call
price — exercising when rates have fallen enough to make refinancing attractive.

```python
import jax.numpy as jnp
from valax.dates import ymd_to_ordinal, generate_schedule
from valax.instruments import CallableBond
from valax.pricing.lattice.hull_white_tree import callable_bond_price

settlement = ymd_to_ordinal(2025, 3, 15)

# 10Y, 5.5% coupon, semi-annual; callable at par on each coupon date from year 3.
payment_dates = generate_schedule(2025, 9, 15, 2035, 3, 15, frequency=2)
call_dates = jnp.array([
    ymd_to_ordinal(2028, 3, 15),
    ymd_to_ordinal(2029, 3, 15),
    ymd_to_ordinal(2030, 3, 15),
    ymd_to_ordinal(2031, 3, 15),
    ymd_to_ordinal(2032, 3, 15),
])

bond = CallableBond(
    payment_dates=payment_dates,
    settlement_date=settlement,
    coupon_rate=jnp.array(0.055),
    face_value=jnp.array(100.0),
    call_dates=call_dates,
    call_prices=jnp.array([1.02, 1.015, 1.01, 1.005, 1.00]),  # premium → par
    frequency=2,
    day_count="act_365",
)

price = callable_bond_price(bond, model, n_steps=200)
print(f"Callable bond price: {price:.4f}")
```

The pricer:

1. Builds the tree from $t = 0$ to the bond's maturity.
2. Initializes the terminal value to face + final coupon.
3. Rolls back via `jax.lax.fori_loop` — at each step, computes the discounted
   continuation value, adds the coupon if this step is a coupon date, and
   applies $\min(\text{cont}, \text{call})$ if it's a call date.
4. Returns the value at the root node.

## 5. Price a Puttable Bond

Symmetric to callable. The bondholder holds a long put: at each put date take the
**max** of continuation value and put price — exercising when rates have risen.

```python
from valax.instruments import PuttableBond
from valax.pricing.lattice.hull_white_tree import puttable_bond_price

put_bond = PuttableBond(
    payment_dates=payment_dates,
    settlement_date=settlement,
    coupon_rate=jnp.array(0.04),
    face_value=jnp.array(100.0),
    put_dates=jnp.array([
        ymd_to_ordinal(2028, 3, 15),
        ymd_to_ordinal(2030, 3, 15),
    ]),
    put_prices=jnp.array([1.0, 1.0]),  # puttable at par
    frequency=2,
    day_count="act_365",
)

put_price = puttable_bond_price(put_bond, model, n_steps=200)
print(f"Puttable bond price: {put_price:.4f}")
```

The puttable price always **exceeds** the price of an otherwise identical
non-puttable bond — the embedded put is held by the bondholder, so it adds value.

## 6. Greeks via Autodiff Through the Tree

The whole pricing pipeline is pure JAX, so `jax.grad` flows through both the
tree construction and the backward induction. You get **effective duration,
effective convexity, and key-rate durations of callable bonds** with no
finite-difference bumping.

### Effective duration: parallel curve shift

```python
import jax
import equinox as eqx

def callable_price_under_shift(shift):
    """Reprice with a parallel shift to all discount factors."""
    # DF(t) → DF(t) * exp(-shift * t)
    pillar_t = (model.initial_curve.pillar_dates - model.initial_curve.reference_date) / 365.0
    new_dfs = model.initial_curve.discount_factors * jnp.exp(-shift * pillar_t)
    new_curve = eqx.tree_at(lambda c: c.discount_factors, model.initial_curve, new_dfs)
    new_model = eqx.tree_at(lambda m: m.initial_curve, model, new_curve)
    return callable_bond_price(bond, new_model, n_steps=200)

# Effective duration: -∂P/∂(parallel shift) / P
dprice_dshift = jax.grad(callable_price_under_shift)(jnp.array(0.0))
eff_duration = -dprice_dshift / price
print(f"Effective duration: {eff_duration:.3f}")
```

### Effective convexity

```python
d2price = jax.grad(jax.grad(callable_price_under_shift))(jnp.array(0.0))
eff_convexity = d2price / price
print(f"Effective convexity: {eff_convexity:.3f}")
```

### Key-rate durations: per-pillar sensitivities

Differentiating directly through the curve's `discount_factors` array gives a
**vector** of sensitivities, one per pillar — no looping, no bumping:

```python
def price_from_dfs(dfs):
    new_curve = eqx.tree_at(lambda c: c.discount_factors, model.initial_curve, dfs)
    new_model = eqx.tree_at(lambda m: m.initial_curve, model, new_curve)
    return callable_bond_price(bond, new_model, n_steps=200)

# Single reverse-mode pass returns ∂P / ∂DF_i for every pillar i.
dP_dDF = jax.grad(price_from_dfs)(model.initial_curve.discount_factors)
print(f"Key-rate sensitivities: {dP_dDF}")
```

This is the autodiff payoff: in a traditional system you'd build the tree once
per pillar shift (so 2N tree builds for N pillars). Here it costs one reverse-mode
pass over a single tree build.

## 7. What's Next

| Capability | Status | Notes |
|------------|--------|-------|
| Hull-White trinomial tree | ✅ Implemented | `valax/pricing/lattice/hull_white_tree.py` |
| Callable / puttable bonds | ✅ Implemented | This guide |
| Analytic ZCB pricing | ✅ Implemented | `hw_bond_price` |
| OAS solver | 🟡 Roadmap | Trivial extension once we add a curve-shift parameter to the pricer |
| Jamshidian swaption decomposition | 🟡 Roadmap | Required for HW calibration to swaption surfaces |
| Bermudan swaptions on the HW tree | 🟡 Roadmap | Currently priced via LSM on LMM paths (`valax/pricing/mc/bermudan.py`) |
| G2++ (two-factor) | 🟡 Roadmap | Adds smile-fitting flexibility |

See the [Roadmap](../roadmap.md) for tracking.
