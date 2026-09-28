# Scenarios & Shocks

*How VALAX builds market scenarios, shocks curves and other risk factors, and repackages everything into a single `MarketData` pytree ready for pricing. For the systems-level picture read [Risk: End-to-End](risk-overview.md) first; for the underlying maths see [Models & Theory § 7](theory/risk-measures.md#7-risk-measures).*

VALAX provides scenario generation, multi-curve shocks, and Value-at-Risk computation — all built on `jax.vmap` for massively parallel portfolio repricing.

!!! note "Why multi-curve is a risk question"
    Separate discount and forward curves are distinct risk factors — a parallel OIS move need not move SOFR forwards, and the basis is itself a hedgeable risk. The full argument and where VALAX stands lives on [Why the Multi-Curve Framework § What this means for risk](guide/why-multicurve.md#what-this-means-for-risk).

## Market Data Container

`MarketData` bundles all market state into a single pytree:

```python
from valax.market import MarketData
from valax.curves import DiscountCurve
from valax.dates import ymd_to_ordinal
import jax.numpy as jnp

ref = ymd_to_ordinal(2026, 1, 1)
pillars = jnp.array([
    ymd_to_ordinal(2026, 1, 1),
    ymd_to_ordinal(2027, 1, 1),
    ymd_to_ordinal(2028, 1, 1),
    ymd_to_ordinal(2030, 1, 1),
])
dfs = jnp.exp(-0.05 * (pillars - ref).astype(jnp.float64) / 365.0)

base = MarketData(
    spots=jnp.array([100.0, 50.0]),
    vols=jnp.array([0.20, 0.35]),
    dividends=jnp.array([0.02, 0.0]),
    discount_curve=DiscountCurve(
        pillar_dates=pillars,
        discount_factors=dfs,
        reference_date=ref,
    ),
)
```

Because `MarketData` is a pytree with differentiable leaves (spots, vols, dividends, and the curve's discount factors), `jax.grad` through any pricing function that consumes it gives sensitivities to every market input in one backward pass.

## Scenarios

A `MarketScenario` represents **additive changes** to risk factors — not absolute levels:

```python
from valax.market import MarketScenario

scenario = MarketScenario(
    spot_shocks=jnp.array([5.0, -2.0]),     # spots move +5, -2
    vol_shocks=jnp.array([0.02, -0.01]),     # vols bump +2%, -1%
    rate_shocks=jnp.array([0.01, 0.01, 0.01, 0.01]),  # +100bp parallel
    dividend_shocks=jnp.array([0.0, 0.0]),
)
```

For spot shocks, set `multiplicative=True` to interpret them as returns: `new_spot = old_spot * (1 + shock)`.

### Scenario Generation

Three methods for producing scenario sets:

**Parametric (Monte Carlo)** — correlated samples from a multivariate distribution:

```python
from valax.risk import parametric_scenarios

key = jax.random.PRNGKey(0)
cov = ...  # (n_factors x n_factors) covariance matrix
scenarios = parametric_scenarios(
    key, cov, n_scenarios=10_000,
    n_assets=2, n_pillars=4,
    distribution="normal",  # or "t" for fat tails
)
```

Column ordering in the covariance matrix: `[spots, vols, rates, dividends]`.

**Historical simulation** — replay observed risk factor changes:

```python
from valax.risk import historical_scenarios

# returns: (n_days, n_factors) of daily changes
scenarios = historical_scenarios(returns, n_assets=2, n_pillars=4)
```

**Stress scenarios** — deterministic named shocks:

```python
from valax.risk import stress_scenario, steepener, butterfly, flattener
from valax.market import stack_scenarios

parallel_up = stress_scenario(2, 4, parallel_rate_shift=0.01)
steep = steepener(2, 4, short_bump=-0.005, long_bump=0.015)
fly = butterfly(2, 4, wing_bump=0.01, belly_bump=-0.005)
crash = stress_scenario(2, 4, spot_shock=-20.0, vol_shock=0.15)

stress_set = stack_scenarios([parallel_up, steep, fly, crash])
```

## Curve Shocks

The core operation converts zero-rate bumps into discount factor adjustments:

$$DF_{\text{new}}(t_i) = DF_{\text{old}}(t_i) \cdot e^{-\Delta r_i \cdot t_i}$$

This is exact: bumping the continuously-compounded zero rate at pillar $i$ by $\Delta r_i$ multiplies the discount factor by $e^{-\Delta r_i \cdot t_i}$.

```python
from valax.risk import bump_curve_zero_rates, parallel_shift, key_rate_bump

# Arbitrary per-pillar bumps
bumped = bump_curve_zero_rates(curve, jnp.array([0.0, 0.005, 0.01, 0.015]))

# Parallel shift: all pillars by +50bp
bumped = parallel_shift(curve, jnp.array(0.005))

# Key-rate bump: only pillar 2 by +25bp
bumped = key_rate_bump(curve, pillar_index=2, bump=jnp.array(0.0025))
```

The bumped curve is a new `DiscountCurve` with the same pillar dates — a drop-in replacement. Its log-linear interpolation produces smooth shocked rates between pillars.

Structured shocks are just specific patterns of per-pillar bumps:

| Shock | Rate profile across pillars |
|-------|---------------------------|
| Parallel shift | Uniform: `[dr, dr, ..., dr]` |
| Steepener | Linear: short end down, long end up |
| Flattener | Linear: short end up, long end down |
| Butterfly | Quadratic: wings up, belly down (or vice versa) |
| Key-rate | Zero everywhere except one pillar |

!!! tip "Differentiability"
    `bump_curve_zero_rates` is fully differentiable. `jax.grad` through a pricing function that uses a bumped curve gives the sensitivity of the price to each bump magnitude — this is key-rate DV01 by construction.

## Applying Scenarios

`apply_scenario` applies all shocks at once and returns a new `MarketData`:

```python
from valax.risk import apply_scenario

shocked_market = apply_scenario(base, scenario)
# shocked_market.spots, .vols, .dividends are bumped
# shocked_market.discount_curve has bumped zero rates
```

Because `rate_shocks` is one unkeyed pillar vector, it can only move the
**default** curve: `discount_curve`, plus the `"_default_"` entry of an
attached `curve_graph` — the same curve seen through the legacy and the
graph view, so the two always move together. Named projection / basis
curves on the graph are carried through *unshocked*; an unkeyed vector
cannot say which curve it means. That is what the keyed scenario below is
for.

!!! warning "Named graph curves are held fixed, loudly"
    `reprice_under_scenario`, `portfolio_pnl`, `parametric_var` and
    `pnl_attribution` thread the `curve_graph` all the way to the pricer
    (they used to drop it, silently repricing a dual-curve book
    single-curve). If the graph carries curves a `MarketScenario` cannot
    address, they emit a `UserWarning` naming them: the risk to those
    curves is *not* in the reported number.

## Keyed scenarios on a `MarketState`

[`MarketState`](api/core.md#marketstate) — the container the unified
`price(instrument, model, market)` takes — is keyed, not positional: it
holds a whole [`CurveGraph`](api/curves.md#valax.curves.CurveGraph) (so
"rates" is not one block), vol **surface objects** rather than one scalar
per asset, and identifier-keyed spots. `KeyedScenario` + `shock` are the
scenario layer for it:

```python
import jax.numpy as jnp
from valax.core import MarketState
from valax.market import KeyedScenario
from valax.risk import shock

state = MarketState.single_curve(          # zero-config single curve
    base.discount_curve,
    spot=jnp.array(100.0),
    vol=jnp.array(0.20),
    dividend=jnp.array(0.02),
)

scenario = KeyedScenario(
    spot_shocks={"_default_": jnp.array(5.0)},      # +5 spot
    vol_shocks={"_default_": jnp.array(0.01)},      # +1 vol point, parallel
    curve_shocks={"_default_": jnp.full(4, 0.0010)},  # +10bp, all pillars
    dividend_shocks={"_default_": jnp.array(0.0)},
)

shocked = shock(state, scenario)           # a NEW MarketState
```

Three rules make it safe to use:

- **Absent key = no shock.** `KeyedScenario()` is the no-op scenario, so
  a scenario only names what it moves.
- **Unknown key = `KeyError`.** A shock that addresses nothing is never
  silently dropped — the failure mode this API exists to remove.
- **`"_default_"` is zero-config.** The sentinel resolves the sole entry
  of a one-entry table, so a single-curve / single-asset market needs no
  identifiers at all, and reduces to exactly today's numbers (a
  `MarketScenario` lifted with `keyed_from_market_scenario` produces a
  bit-identical shocked market).

### Per-curve moves are the point

Curve shocks are keyed by curve id and routed through `bump_graph_curve`,
so OIS and projection curves are finally *distinct* risk factors — a pure
basis widening is one scenario:

```python
from valax.curves import CurveGraph, DiscountCurve, zero_rate

times = (pillars - ref).astype(jnp.float64) / 365.0
fwd_3m = DiscountCurve(                     # 50bp above OIS
    pillar_dates=pillars,
    discount_factors=jnp.exp(-0.055 * times),
    reference_date=ref,
)
dual = MarketState(curves=CurveGraph(curves={
    "USD.SOFR.OIS": base.discount_curve,
    "USD.SOFR.3M": fwd_3m,
}))

basis_widening = KeyedScenario(curve_shocks={
    "USD.SOFR.3M":  jnp.full(4,  0.0025),   # forwards up 25bp
    "USD.SOFR.OIS": jnp.full(4, -0.0025),   # discounting down 25bp
})
widened = shock(dual, basis_widening)

before = zero_rate(dual.curves["USD.SOFR.3M"], pillars[2]) - zero_rate(
    dual.curves["USD.SOFR.OIS"], pillars[2])
after = zero_rate(widened.curves["USD.SOFR.3M"], pillars[2]) - zero_rate(
    widened.curves["USD.SOFR.OIS"], pillars[2])
print(f"2Y basis: {before * 1e4:.1f}bp → {after * 1e4:.1f}bp")  # 50.0bp → 100.0bp
```

Everything not named is shared with the base market, untouched: `shock`
rebuilds only the affected branches via `eqx.tree_at` and never mutates.
`jax.grad` flows to the shock magnitudes themselves (shock-space
sensitivities) as well as to the base market, and the whole path is
`eqx.filter_jit`-safe.

### Vol shocks: parallel shift, not node shock

Under `MarketState` a vol is a duck-typed **surface object**
(`GridVolSurface`, `SVIVolSurface`, `SABRVolSurface`, `SwaptionCube`,
`ConstantVol`, ...), so "bump the vol" needs a definition. Two were
possible:

| | Representation | Expressiveness | Works for |
|---|---|---|---|
| (a) | shock surface **nodes** | a whole smile/term-structure move | grids only — SVI / SABR leaves are model parameters, not vols |
| (b) | wrap in a **parallel-shift decorator** | one number per surface | every surface type |

**VALAX ships (b)** — [`ShiftedVolSurface`](api/surfaces.md#shifted-surface-scenario-decorator),
applied by `shift_surface` — because a scenario must mean the same thing
for every surface a book can hold, and because the shock magnitude then
stays a single differentiable scalar. A `ConstantVol` is shifted *in
place* (`vol + shift`), which is bit-for-bit the legacy
`vols + vol_shocks` arithmetic and keeps `MarketState.to_market_data()`
working; repeated shifts accumulate instead of nesting wrappers. Option
(a) is not lost: node-level moves remain available per surface type via
`eqx.tree_at(lambda s: s.vols, surface, ...)` and are the natural thing to
plug into `shift_surface` later.

**Consequence for the ladders.** (b) keeps the vol axis one scalar per
surface, which is exactly the shape
[`SensitivityLadder`](risk-ladders.md)'s `delta_vol`, `vanna` and `volga`
fields already assume — so they are unchanged by this work, but their
*meaning* under a `MarketState` is a per-surface **parallel-shift vega**
(a bucket vega), not a node vega. Had (a) been chosen, those three fields
would have had to become identifier-keyed per-node arrays. Node-level vega
ladders therefore remain future work, tracked in the
[roadmap](roadmap.md#1-risk-engine-is-still-single-curve-largest-gap).

!!! note "The VaR engine still consumes the legacy view"
    `reprice_under_scenario` / `portfolio_pnl` / `value_at_risk` take a
    `MarketData` and a `MarketScenario` and have not moved onto
    `price(...)` yet, so `KeyedScenario` is today the scenario
    representation for `MarketState`-based repricing and greeks rather
    than a drop-in for the VaR loop. `keyed_from_market_scenario` bridges
    the two representations in the meantime.

## Pricing Functions for Risk

The risk engine accepts any pricing function with the signature:

```python
def my_pricing_fn(instrument, market: MarketData) -> Float[Array, ""]:
    ...
```

Each instrument receives a per-instrument `MarketData` with scalar `spots`, `vols`, `dividends` and the **full shared discount curve**. This means both equity and rates products work through the same risk framework:

```python
# Equity: extract spot/vol/rate from market
def bs_pricer(option, market):
    from valax.risk.var import _extract_short_rate
    rate = _extract_short_rate(market.discount_curve)
    return black_scholes_price(option, market.spots, market.vols, rate, market.dividends)

# Rates: use the full curve directly
def swap_pricer(swap, market):
    return swap_price(swap, market.discount_curve)
```

For existing equity-style functions with signature `(instrument, spot, vol, rate, dividend) -> price`, use the adapter:

```python
from valax.risk import wrap_equity_pricing_fn

market_fn = wrap_equity_pricing_fn(black_scholes_price)
```

---

*Next: turn scenarios into loss distributions in [VaR & P&L](risk-var.md), or bucket sensitivities in [Risk Bucketing](risk-bucketing.md).*
