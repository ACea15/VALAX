# Volatility Surfaces

A volatility surface maps (strike, expiry) pairs to implied volatilities. Every option pricing function needs one — a flat scalar vol is a toy assumption that ignores the smile (wings trade at higher vol than ATM) and the term structure (short-dated vs long-dated options have different vol levels).

VALAX provides three vol surface implementations, all as callable `eqx.Module` pytrees that are JIT-compilable, differentiable, and vmappable.

## Grid Surface

The simplest construction: store implied vols on a discrete (strike, expiry) grid and bilinearly interpolate. No model assumptions.

```python
from valax.surfaces import GridVolSurface
import jax.numpy as jnp

strikes = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0])
expiries = jnp.array([0.25, 0.5, 1.0])
vols = jnp.array([
    [0.30, 0.24, 0.20, 0.22, 0.28],  # T=0.25: steep smile
    [0.28, 0.23, 0.19, 0.21, 0.26],  # T=0.5: moderate smile
    [0.26, 0.22, 0.18, 0.20, 0.24],  # T=1.0: flatter smile
])

surface = GridVolSurface(strikes=strikes, expiries=expiries, vols=vols)

# Query at any (strike, expiry) — bilinear interpolation
vol = surface(jnp.array(95.0), jnp.array(0.75))

# Vectorize across strikes
import jax
smile = jax.vmap(lambda K: surface(K, jnp.array(0.5)))(strikes)
```

The `vols` grid is a differentiable leaf. `jax.grad` through a pricing function that queries this surface gives **vega bucketed by (strike, expiry) node** — the sensitivity to each grid point individually.

**When to use:** Direct construction from broker quotes or when you have a dense grid of market-observed implied vols and don't want to impose parametric assumptions.

## SABR Surface

Industry standard for rates (swaption cubes) and common for equity. Fits a SABR model per expiry slice, interpolates parameters across expiries.

```python
from valax.surfaces import SABRVolSurface, calibrate_sabr_surface

# Market data: strikes and vols at each expiry
strikes_3m = jnp.linspace(80.0, 120.0, 11)
strikes_1y = jnp.linspace(80.0, 120.0, 11)
vols_3m = ...  # observed smile at 3M
vols_1y = ...  # observed smile at 1Y

surface = calibrate_sabr_surface(
    strikes_per_expiry=[strikes_3m, strikes_1y],
    market_vols_per_expiry=[vols_3m, vols_1y],
    forwards=jnp.array([100.0, 100.0]),
    expiries=jnp.array([0.25, 1.0]),
    fixed_beta=jnp.array(0.5),  # common: 0.5 for equity, 0.0 for rates
)

# Query: SABR params are interpolated to the expiry, then Hagan formula applied
vol = surface(jnp.array(95.0), jnp.array(0.5))
```

You can also construct directly from known parameters:

```python
surface = SABRVolSurface(
    expiries=jnp.array([0.25, 1.0]),
    forwards=jnp.array([100.0, 100.0]),
    alphas=jnp.array([0.30, 0.25]),
    betas=jnp.array([0.5, 0.5]),
    rhos=jnp.array([-0.2, -0.3]),
    nus=jnp.array([0.4, 0.35]),
)
```

All SABR parameters are differentiable. `jax.grad` through the surface gives sensitivities to alpha, rho, nu at each expiry — useful for SABR vega risk.

**When to use:** Rates desks (swaption/cap vol cubes), or when you want a parsimonious parametric fit that extrapolates sensibly beyond the observed strike range.

## SVI Surface

Gatheral's Stochastic Volatility Inspired parameterization. Five parameters per expiry define total implied variance as a function of log-moneyness:

$$w(k) = a + b \left(\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right)$$

where $k = \log(K/F)$ and implied vol $= \sqrt{w / T}$.

```python
from valax.surfaces import SVIVolSurface, calibrate_svi_slice, calibrate_svi_surface

# Calibrate a single slice
params, sol = calibrate_svi_slice(
    strikes=jnp.linspace(80.0, 120.0, 15),
    market_vols=observed_vols,
    forward=jnp.array(100.0),
    expiry=jnp.array(1.0),
)

# Or calibrate an entire surface
surface = calibrate_svi_surface(
    strikes_per_expiry=[strikes_3m, strikes_6m, strikes_1y],
    market_vols_per_expiry=[vols_3m, vols_6m, vols_1y],
    forwards=jnp.array([100.0, 100.0, 100.0]),
    expiries=jnp.array([0.25, 0.5, 1.0]),
)
```

SVI interpolates **total variance** (not implied vol) across expiries. This preserves the calendar spread no-arbitrage condition: total variance must be non-decreasing in expiry.

| Parameter | Meaning | Constraint |
|---|---|---|
| $a$ | Overall variance level | Real |
| $b$ | Slope of the wings | $b \geq 0$ |
| $\rho$ | Left/right asymmetry | $-1 < \rho < 1$ |
| $m$ | Horizontal shift of the smile vertex | Real |
| $\sigma$ | Smoothing at the vertex | $\sigma > 0$ |

**When to use:** Equity options desks. SVI fits market smiles well with only 5 parameters, extrapolates cleanly, and can be constrained for arbitrage-freeness. The linear wings match the Roger Lee moment formula for extreme strikes.

## Common Patterns

### Pricing with a surface

All surfaces share the same callable interface — `surface(strike, expiry) -> vol`:

```python
from valax.pricing.analytic import black_scholes_price
from valax.instruments import EuropeanOption

option = EuropeanOption(strike=jnp.array(95.0), expiry=jnp.array(0.5), is_call=True)

# Look up the smile-consistent vol for this option
vol = surface(option.strike, option.expiry)
price = black_scholes_price(option, spot, vol, rate, dividend)
```

### Surface Greeks via autodiff

Because surfaces are differentiable pytrees, `jax.grad` flows through them:

```python
# Vega: sensitivity to parallel vol shift
def price_with_surface(vol_grid):
    s = GridVolSurface(strikes=..., expiries=..., vols=vol_grid)
    vol = s(option.strike, option.expiry)
    return black_scholes_price(option, spot, vol, rate, dividend)

vega_bucketed = jax.grad(price_with_surface)(surface.vols)
# Shape: (n_expiries, n_strikes) — sensitivity to each grid node
```

For SABR surfaces, you get sensitivities to each SABR parameter at each expiry:

```python
def price_with_alphas(alphas):
    s = SABRVolSurface(expiries=..., forwards=..., alphas=alphas, ...)
    vol = s(option.strike, option.expiry)
    return black_scholes_price(option, spot, vol, rate, dividend)

d_price_d_alpha = jax.grad(price_with_alphas)(surface.alphas)
# Shape: (n_expiries,) — SABR alpha sensitivity per expiry bucket
```

### Vectorization

`jax.vmap` across strikes to compute the full smile:

```python
strikes = jnp.linspace(80.0, 120.0, 50)
smile = jax.vmap(lambda K: surface(K, jnp.array(0.5)))(strikes)
```
