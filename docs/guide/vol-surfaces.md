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

## Using a Surface as a Dupire Input

Beyond `surface(K, T) → σ_IV`, all three surface types expose

```python
surface.total_variance(log_moneyness, expiry) -> Float[Array, ""]
```

returning total variance $w(k, T) = \sigma_{\text{IV}}^2 \cdot T$ directly.
This is the duck-typed input expected by
[`dupire_local_vol`](../api/pricing.md#dupire_local_vol) and by SLV's
leverage-function calibration ([SLV guide](slv.md),
[`calibrate_slv_leverage`](../api/calibration.md#calibrate_slv_leverage)).
The consistency identity

```python
surface.total_variance(jnp.log(K / F_T), T) == surface(K, T) ** 2 * T
```

holds to machine precision.

### Which surface to pick

| Use case | Recommended surface | Why |
|---|---|---|
| Dupire local vol extraction | **`SVIVolSurface`** | $w(k)$ is closed-form $C^\infty$ in $k$; `jax.grad` gives exact $\partial w / \partial k$ and $\partial^2 w / \partial k^2$. |
| SLV (supported, see [SLV guide](slv.md)) | `SVIVolSurface` | Same reasons; differentiability also flows through to leverage-grid calibration via [`calibrate_slv_leverage`](../api/calibration.md#calibrate_slv_leverage). |
| Quick prototyping on broker quotes | `GridVolSurface` (in log-moneyness) | No calibration needed; bilinear interpolation. |
| Rates desks | `SABRVolSurface` | Industry standard for swaption cubes. |

`SABRVolSurface.total_variance` works but uses Hagan's asymptotic IV
formula internally — at extreme strikes the resulting "local vol" is a
measurement of the expansion's residual rather than a clean market signal.

### End-to-end: SVI → Dupire → Local-vol MC

The Dupire-consistency loop is the canonical sanity check that the
extraction is plumbed correctly. Calibrate SVI to a market smile, wrap it
in a `LocalVolModel`, run LV MC, and reprice the input smile — the result
should match the input to within a few tens of basis points at typical
MC budgets.

```python
import jax, jax.numpy as jnp
from valax.surfaces import calibrate_svi_surface
from valax.models import LocalVolModel
from valax.pricing.mc import generate_local_vol_paths
from valax.instruments import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_implied_vol

# 1. Calibrate SVI to a market smile (or surface)
svi = calibrate_svi_surface(
    strikes_per_expiry=[strikes_3m, strikes_6m, strikes_1y],
    market_vols_per_expiry=[vols_3m, vols_6m, vols_1y],
    forwards=jnp.array([100.0, 100.5, 101.0]),
    expiries=jnp.array([0.25, 0.5, 1.0]),
)

# 2. Wrap in a local-vol model
model = LocalVolModel.from_flat_rate(svi, rate=0.03, dividend=0.01)

# 3. Simulate
paths = generate_local_vol_paths(
    model, spot=jnp.array(100.0),
    T=1.0, n_steps=500, n_paths=100_000,
    key=jax.random.PRNGKey(20260101),
)

# 4. Reprice the input smile via MC → implied vol → compare
strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])
df = jnp.exp(-0.03 * 1.0)
for K in strikes:
    payoff = jnp.maximum(paths[:, -1] - K, 0.0)
    mc_price = df * jnp.mean(payoff)
    opt = EuropeanOption(strike=K, expiry=jnp.array(1.0), is_call=True)
    iv_mc = black_scholes_implied_vol(
        opt, jnp.array(100.0),
        jnp.array(0.03), jnp.array(0.01), mc_price,
    )
    iv_market = svi(K, jnp.array(1.0))
    diff_bp = float(jnp.abs(iv_mc - iv_market)) * 1e4
    print(f"K={float(K):5.0f}  market={float(iv_market):.4f}  mc={float(iv_mc):.4f}  diff={diff_bp:.1f} bp")
```

Single-seed runs at this size show ~15–30 bp worst-case noise; the
[Dupire-consistency gate](monte-carlo.md#local-volatility) averages 4
seeds to hit < 20 bp. See [theory §4.4](../theory.md#44-local-volatility-dupire)
for the full discussion of MC bias floors and the Milstein scheme as
the path to sub-5-bp accuracy.

### Vega-bucketed Greeks through the LV pipeline

Because every leaf of the surface pytree is differentiable, `jax.grad`
of a LV MC price w.r.t. SVI parameters gives a vector of per-slice,
per-parameter sensitivities for free:

```python
def lv_mc_price(svi_params: SVIVolSurface, key):
    model = LocalVolModel.from_flat_rate(svi_params, rate=0.03, dividend=0.01)
    paths = generate_local_vol_paths(
        model, jnp.array(100.0), 1.0, 500, 100_000, key,
    )
    payoff = jnp.maximum(paths[:, -1] - 100.0, 0.0)
    return jnp.exp(-0.03 * 1.0) * jnp.mean(payoff)

# Sensitivity to the variance level (a) at each calibrated expiry
sens_a = jax.grad(lambda a: lv_mc_price(
    eqx.tree_at(lambda s: s.a_vec, svi, a),
    jax.random.PRNGKey(42),
))(svi.a_vec)
# shape (n_expiries,) — vega-bucketed by expiry slice
```
