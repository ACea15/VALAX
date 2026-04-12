# Binomial Trees

VALAX implements the Cox-Ross-Rubinstein (CRR) binomial tree for European and American option pricing.

## Usage

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

## CRR Parameters

At each time step $\Delta t = T/N$:

$$u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{(r-q)\Delta t} - d}{u - d}$$

The tree recombines: an up move followed by a down move reaches the same node as down-then-up.

## American Exercise

When `american=True`, at each node the algorithm takes the maximum of the continuation value and the intrinsic value:

$$V_i = \max\left(\text{disc} \cdot [p \cdot V_{i+1}^u + (1-p) \cdot V_{i+1}^d],\; \text{intrinsic}_i\right)$$

Key properties:

- **American put >= European put** — early exercise premium exists.
- **American call without dividends = European call** — early exercise is never optimal.
- **American call with dividends >= European call** — may be optimal to exercise before ex-dividend.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `n_steps` | 200 | Number of time steps in the tree |
| `american` | `False` | Enable American early exercise |

## Greeks via Autodiff

```python
import jax

delta = jax.grad(lambda s: binomial_price(
    option, s, vol, rate, div, config
))(spot)
```

## Implementation Notes

- The backward induction uses `jax.lax.scan` with fixed-shape arrays for JIT compatibility. At each step, `values[j] = disc * (p * values[j+1] + (1-p) * values[j])` is computed over the full array; unused trailing entries are harmless.
- European prices converge to Black-Scholes within 0.5% at 500 steps.

---

# Hull-White Trinomial Tree

The Hull-White one-factor model is the standard short-rate model for pricing bonds with embedded optionality. VALAX provides:

1. **`HullWhiteModel`** — an `equinox.Module` pytree carrying the mean-reversion speed $a$, short-rate volatility $\sigma$, and the initial discount curve $P^M(0,t)$.
2. **Analytic zero-coupon bond prices** — $P(t,T \mid r) = A(t,T)\,e^{-B(t,T)\,r}$ with exact-fit to the initial curve via the Brigo-Mercurio $A(t,T)$ formula.
3. **Trinomial tree** — a recombining tree for backward induction, with Arrow-Debreu forward propagation to calibrate $\alpha(t_i)$ at each step and match market discount factors exactly.
4. **Callable and puttable bond pricing** — backward induction with exercise decisions: `callable_bond_price` (issuer calls when continuation > call price, reducing holder value) and `puttable_bond_price` (holder puts when continuation < put price, increasing holder value).

## Model

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

## Callable and Puttable Bonds

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
