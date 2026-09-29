# VaR & P&L

*Turning a scenario set into a loss distribution and explaining the moves. Prerequisites: build the scenarios and shocks in [Scenarios & Shocks](risk-scenarios.md). Maths: [Models & Theory § 7.1–7.5](theory/risk-measures.md#7-risk-measures).*

## Repricing through `price()`

A book is repriced by the **same call** that produces a front-office
price. `valax.risk.state_var` takes a `MarketState` and a
[`KeyedScenario`](risk-scenarios.md#keyed-scenarios-on-a-marketstate)
and goes straight through [`price`](api/core.md#the-unified-signature) — no
pricing-function adapter, and every named curve of the `CurveGraph` is a
risk factor in its own right.

The unit of work is a **homogeneous group**: `price` dispatches on type,
so one `vmap` covers instruments sharing
`(instrument type, static fields, model type, method, config)`, and a
heterogeneous book is a sum over groups. Calls and puts are different
groups, because `is_call` is a static field.

```python
import jax.numpy as jnp
from valax.core import MarketState
from valax.curves import CurveGraph, DiscountCurve
from valax.dates import ymd_to_ordinal
from valax.instruments import InterestRateSwap
from valax.market import KeyedScenario, stack_keyed_scenarios
from valax.risk import group_pv, group_market_gradient, group_scenario_pnl, value_at_risk

ref = ymd_to_ordinal(2026, 1, 1)
pillars = jnp.array([ref, ymd_to_ordinal(2027, 1, 1), ymd_to_ordinal(2028, 1, 1)])
times = (pillars - ref).astype(jnp.float64) / 365.0

market = MarketState(curves=CurveGraph(curves={
    "USD.SOFR.OIS": DiscountCurve(pillar_dates=pillars,
                                  discount_factors=jnp.exp(-0.050 * times),
                                  reference_date=ref),
    "USD.SOFR.3M": DiscountCurve(pillar_dates=pillars,
                                 discount_factors=jnp.exp(-0.055 * times),
                                 reference_date=ref),
}))

swaps = InterestRateSwap(                      # one homogeneous group
    start_date=jnp.array([ref, ref]),
    fixed_dates=jnp.tile(pillars[1:], (2, 1)),
    fixed_rate=jnp.array([0.04, 0.05]),
    notional=jnp.array([1e6, 1e6]),
    discount_id="USD.SOFR.OIS",
    forward_id="USD.SOFR.3M",
)

print(f"PV: {float(group_pv(swaps, None, market)):,.2f}")   # PV: 40,520.22

# A basis widening: the projection curve moves, discounting does not.
scenarios = stack_keyed_scenarios([
    KeyedScenario(curve_shocks={"USD.SOFR.3M": jnp.full(3, bp)})
    for bp in (0.0, 0.0025, -0.0025)
])
pnl = group_scenario_pnl(swaps, None, market, scenarios)
# P&L: [0.00, 9,817.32, -9,792.80]   VaR(99%): 9,596.95
var_99 = value_at_risk(pnl, confidence=0.99)

grad = group_market_gradient(swaps, None, market)
grad.curves["USD.SOFR.OIS"].discount_factors   # OIS pillar deltas
grad.curves["USD.SOFR.3M"].discount_factors    # projection pillar deltas
```

Three things follow, and they are the reason the port matters:

- **The gradient is a `MarketState`-shaped pytree**, carrying a separate
  cotangent per named curve. Per-group gradients therefore aggregate with
  `jax.tree_util.tree_map(operator.add, ...)` in the market's own
  coordinates, and [bucketing](risk-bucketing.md) stays a coordinate
  change applied *after* aggregation.
- **Risk measures apply at the top only.** `group_scenario_pnl` returns a
  P&L *vector*; add vectors across groups and take
  `value_at_risk` / `expected_shortfall` once. A quantile of a sum is not
  a sum of quantiles, so adding per-desk VaRs produces a plausible wrong
  number.
- **MC groups need common random numbers.** `key` is fixed per group and
  reused for the base and every shocked revaluation. Drawing a fresh key
  per revaluation turns scenario P&L into sampling noise — measured at
  −0.207 on a zero scenario whose true P&L is 0, against a +1.16 P&L for
  a genuine spot move.

!!! note "What has not moved yet"
    [Sensitivity ladders](risk-ladders.md), the scenario generators in
    `valax.risk.scenarios`, and `parametric_var`'s covariance layout are
    still positional and still take a `MarketData`. Tracked in the
    [roadmap](roadmap.md#1-risk-engine-is-still-single-curve-largest-gap).

!!! warning "A shock that cannot reach the price says so"
    An equity model given an explicit scalar `rate` / `vol` / `dividend`
    reads *that*, not the market, so for such a group a curve, vol or
    dividend shock is **not an input to the price** and contributes exactly
    zero P&L — that risk is on `group_model_gradient` instead. Rather than
    report a silent zero, `group_reprice` / `group_scenario_pnl` raise a
    `UserWarning` naming the shadowed shock family and the model field
    responsible.

    The fix is to let the model read the market: leave the parameter
    `None` and it is sourced from the `MarketState` at price time
    ([API → Core § Model binding](api/core.md#model-binding-constants-or-market-sourced)).
    A market-sourced rate is projected at the trade's own maturity, and
    because that happens inside the differentiated path the shock reaches
    the price and the pillar deltas appear on `group_market_gradient` — at
    which point the warning stops firing by itself.

    Models that reference the market by id — `HullWhiteModel`,
    `G2PPModel`, or no model at all — were never affected: their curve
    shocks already reach the price, and the embedded `initial_curve` is
    vestigial in the price path because the recipe takes `discount_curve`
    from the market.

---

## Full-Revaluation VaR

!!! warning "Legacy view"
    The rest of this page documents the `MarketData` / `MarketScenario`
    path, frozen for one deprecation cycle. It cannot be re-expressed in
    terms of the section above: a `pricing_fn` is an opaque callable
    carrying no model, and `price` needs one.

The end-to-end VaR workflow: generate scenarios, reprice the portfolio under each one via `jax.vmap`, compute risk measures on the P&L vector.

```python
from valax.risk import portfolio_pnl, value_at_risk, expected_shortfall

# Reprice portfolio under all scenarios (vmapped)
pnl = portfolio_pnl(my_pricing_fn, instruments, base, scenarios)

# Risk measures
var_99 = value_at_risk(pnl, confidence=0.99)
es_99 = expected_shortfall(pnl, confidence=0.99)
```

`portfolio_pnl` uses `jax.vmap` over the scenario axis — each iteration applies one scenario, reprices via `jax.vmap` over instruments, and returns a scalar P&L. The base `MarketData` is closed over (constant); only the scenario varies.

This means **10,000 scenarios x 100 instruments = 1,000,000 repricings** compile down to a single JIT-compiled, vectorized computation — no Python loops.

## Parametric VaR (Delta-Normal)

For fast VaR without repricing, `parametric_var` uses autodiff to compute portfolio sensitivities (delta vector) and the covariance matrix to estimate portfolio variance:

$$\text{VaR}_\alpha = z_\alpha \cdot \sqrt{\boldsymbol{\delta}^T \Sigma \boldsymbol{\delta}}$$

where $\boldsymbol{\delta}$ is the gradient of portfolio value w.r.t. all risk factors and $\Sigma$ is the covariance matrix.

```python
from valax.risk import parametric_var

pvar_99 = parametric_var(my_pricing_fn, instruments, base, cov, confidence=0.99)
```

This is a first-order (linear) approximation — fast and accurate for well-hedged portfolios. For highly convex positions (e.g., short gamma), use full-revaluation VaR instead.

Internally, the sensitivity to curve pillars is converted from DF-space to zero-rate-space: $\frac{\partial P}{\partial r_i} = \frac{\partial P}{\partial DF_i} \cdot (-t_i \cdot DF_i)$, matching the covariance matrix's column ordering.

## P&L Attribution

`pnl_attribution` decomposes a scenario's P&L into risk factor contributions using a second-order Taylor expansion with autodiff-computed sensitivities:

$$\Delta P \approx \underbrace{\sum_i \delta_i \Delta x_i}_{\text{first order}} + \underbrace{\frac{1}{2} \sum_i \gamma_i (\Delta x_i)^2}_{\text{second order}} + \underbrace{\text{remainder}}_{\text{unexplained}}$$

```python
from valax.risk import pnl_attribution

attr = pnl_attribution(my_pricing_fn, instruments, base, scenario)

print(f"Delta (spot):  {attr['delta_spot']:.2f}")
print(f"Vega:          {attr['delta_vol']:.2f}")
print(f"Rho (rates):   {attr['delta_rate']:.2f}")
print(f"Gamma (spot):  {attr['gamma_spot']:.2f}")
print(f"Total approx:  {attr['total_second_order']:.2f}")
print(f"Actual P&L:    {attr['actual']:.2f}")
print(f"Unexplained:   {attr['unexplained']:.2f}")
```

The returned dict contains:

| Key | Description |
|-----|------------|
| `delta_spot` | P&L from spot moves (first order) |
| `delta_vol` | P&L from vol moves (vega) |
| `delta_rate` | P&L from rate moves (rho / DV01) |
| `delta_div` | P&L from dividend moves |
| `gamma_spot` | P&L from spot convexity (second order) |
| `total_first_order` | Sum of all delta terms |
| `total_second_order` | First order + gamma |
| `actual` | True P&L from full repricing |
| `unexplained` | `actual - total_second_order` |

All sensitivities are computed via `jax.grad` and `jax.hessian` — no finite differences. For the fully bucketed, ten-rung version with all cross terms, see [Sensitivity Ladders](risk-ladders.md#sensitivity-ladders-and-waterfall-pl).

---

## P&L Vectors: Predict vs Actual

Every risk metric in VALAX — VaR, ES, backtests, the FRTB PLA test — reduces to a single primitive: a **P&L vector** of length $N$, with one entry per scenario or per historical day. Once you have the vector, the metrics are just sample statistics.

For the theoretical framing of HPL / RTPL / APL and why these three series are kept separate, see [Models & Theory § 7.5](theory/risk-measures.md#75-pl-vectors-hypothetical-risk-theoretical-actual).

### Hypothetical P&L (full revaluation)

`hypothetical_pnl_vector` reprices the portfolio under every scenario via `jax.vmap`:

```python
from valax.risk import hypothetical_pnl_vector

hpl = hypothetical_pnl_vector(pricing_fn, instruments, base_market, scenarios)
# hpl.shape == (n_scenarios,)
```

This is identical to `portfolio_pnl` — the alias exists so risk-engine code reads consistently when both HPL and RTPL vectors appear side by side.

### Risk-theoretical P&L (ladder-based prediction)

`risk_theoretical_pnl_vector` is the vectorised version of the waterfall: one precomputed ladder, $N$ cheap arithmetic contractions:

```python
from valax.risk import compute_ladder, risk_theoretical_pnl_vector

ladder = compute_ladder(pricing_fn, instruments, base_market)
rtpl = risk_theoretical_pnl_vector(ladder, scenarios, base_market)
# rtpl.shape == (n_scenarios,)
```

For a 10 000-scenario VaR run on an option-heavy portfolio, RTPL is typically two to three orders of magnitude cheaper than HPL because the expensive call is the ladder build, not the per-scenario evaluation.

### Comparing both at once

`explained_unexplained_vector` returns the matched HPL/RTPL pair plus the per-scenario unexplained residual:

```python
from valax.risk import explained_unexplained_vector

report = explained_unexplained_vector(
    pricing_fn, instruments, base_market, scenarios,
)
report["rtpl"]         # shape (n_scenarios,) — ladder prediction
report["hpl"]          # shape (n_scenarios,) — full-revaluation
report["unexplained"]  # shape (n_scenarios,) — hpl - rtpl
```

The unexplained vector is a direct diagnostic: a large or systematically biased unexplained means the second-order ladder is missing something (third-order convexity, missing risk factors, or a scenario regime the ladder was not built for).

### From P&L vector to VaR/ES

Both HPL and RTPL vectors plug straight into the existing risk measures:

```python
from valax.risk import value_at_risk, expected_shortfall

var_hpl = value_at_risk(hpl, confidence=0.99)
es_hpl  = expected_shortfall(hpl, confidence=0.99)

var_rtpl = value_at_risk(rtpl, confidence=0.99)
es_rtpl  = expected_shortfall(rtpl, confidence=0.99)
```

The difference `var_rtpl - var_hpl` is the **model-induced VaR bias**: how much your risk engine's Taylor approximation under- or over-states the tail. For well-hedged portfolios it should be near zero.

---

*Next: validate the forecast in [Backtesting & FRTB PLA](risk-backtesting.md), or standardise the sensitivities in [Risk Bucketing](risk-bucketing.md).*
