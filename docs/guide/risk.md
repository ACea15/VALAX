# Risk: Scenarios, Shocks, and VaR

VALAX provides scenario generation, multi-curve shocks, and Value-at-Risk computation — all built on `jax.vmap` for massively parallel portfolio repricing.

## Why Multi-Curve?

### The pre-2008 world: one curve does everything

Before the financial crisis, the industry used a single yield curve for both **discounting** (computing present values) and **forecasting** (projecting future floating rates). This worked because the spread between LIBOR and OIS (the overnight indexed swap rate, a proxy for the risk-free rate) was negligible — typically 1-2 basis points.

Under this single-curve framework, the discount factor to time $t$ and the forward rate between $t_1$ and $t_2$ are derived from the same set of zero rates:

$$
DF(t) = e^{-r(t) \cdot t}, \quad F(t_1, t_2) = \frac{DF(t_1)/DF(t_2) - 1}{\tau(t_1, t_2)}
$$

One curve, built from deposits, FRAs, and swaps against the same index. Simple.

### The crisis: LIBOR-OIS blew out

In 2007-2008, the LIBOR-OIS spread widened from 2bp to over 350bp. This was the market pricing in **bank credit risk** — LIBOR is an unsecured interbank lending rate, and banks were suddenly risky counterparties. The spread revealed that LIBOR is not risk-free; it embeds a credit and liquidity premium that varies by tenor.

This meant:

- **3-month LIBOR and 6-month LIBOR are different risk factors.** A bank borrowing for 6 months faces more credit risk than one borrowing for 3 months. The two rates don't collapse to a single curve.
- **Discounting should use a risk-free (or near-risk-free) rate.** Under CSA (Credit Support Annex) agreements, collateralized derivatives are funded at the overnight rate. Discounting at LIBOR overstates the present value.
- **Forward projection should use the index-specific curve.** A swap paying 3M LIBOR should project forwards from a 3M LIBOR curve, not from the OIS curve.

### The post-crisis standard: multi-curve

The modern framework uses **separate curves for separate purposes**:

| Curve | Built from | Used for |
|-------|-----------|----------|
| **OIS discount curve** | Overnight index swaps (SOFR, ESTR, SONIA) | Discounting all collateralized cashflows |
| **SOFR forward curve** | SOFR swaps, futures | Projecting SOFR floating rates |
| **Term rate curves** (if still used) | Term SOFR, Euribor swaps | Projecting term fixings |
| **Basis curves** | Basis swaps (1M vs 3M, SOFR vs FF) | Expressing the spread between indices |
| **Cross-currency basis** | XCCY basis swaps | FX-adjusted discounting for foreign-currency legs |

A single interest rate swap now requires at least two curves: one to project the floating leg's cashflows (the forward curve for the relevant index) and one to discount all cashflows to present value (the OIS curve tied to the collateral agreement).

### Why this matters for risk

Each curve responds differently to market shocks. A parallel shift in OIS rates does not equally affect SOFR forwards — the basis can widen or narrow independently. Proper risk measurement requires:

1. **Independent shocks per curve.** A steepener on the OIS curve may coincide with a flattener on the SOFR curve. Single-curve VaR misses this.
2. **Basis risk.** The spread between curves is itself a risk factor. Hedging a SOFR swap with OIS instruments leaves residual basis risk that single-curve models cannot capture.
3. **Correct hedge ratios.** Differentiating price with respect to OIS pillar rates gives the OIS DV01; differentiating with respect to SOFR pillar rates gives the SOFR DV01. These are different numbers, and you need both to construct the right hedge.

### Where VALAX stands today

VALAX currently provides a **single `DiscountCurve`** with full pillar-level shocks — you can bump each zero rate independently (parallel shift, steepener, butterfly, key-rate bumps). This is sufficient for:

- Single-index portfolios (e.g., pure equity options using one risk-free rate)
- Learning and prototyping rate risk workflows
- Stress testing a single yield curve

The architecture is designed so that adding multi-curve support is a natural extension: `MarketData` can hold multiple `DiscountCurve` instances (one for discounting, one per forward index), and `MarketScenario` can carry separate `rate_shocks` vectors for each curve. The `bump_curve_zero_rates` function already works on any `DiscountCurve` — it just needs to be applied to each curve independently.

!!! note "Roadmap"
    Full multi-curve bootstrapping (simultaneous OIS + SOFR from market instruments) is a Tier 1 priority. See the [Roadmap](../roadmap.md) for details.

---

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

## Full-Revaluation VaR

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

All sensitivities are computed via `jax.grad` and `jax.hessian` — no finite differences.

---

## Risk Factors: What Gets Shocked

### What is a risk factor?

A risk factor is any market observable that, when it moves, changes the value of a position. VaR answers the question "how much could we lose?" by shocking these factors according to their historical or modelled joint distribution.

The choice of risk factors determines the fidelity of the VaR number. Too few factors and you miss real risks (basis risk, smile risk, cross-gamma). Too many and the covariance matrix becomes singular or unstable.

### Industry-standard factor counts

A real bank VaR system tracks thousands to tens of thousands of risk factors. The exact number depends on the desk and portfolio:

| Factor type | Typical count | Details |
|---|---|---|
| **IR curve pillars** (per curve) | 15-20 | Standard tenors: ON, 1W, 1M, 2M, 3M, 6M, 9M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y |
| **IR curves** (per currency) | 2-5 | OIS discount, SOFR/ESTR forward, basis, cross-currency basis |
| **Currencies covered** | 10-30 | G10 + major EM |
| **IR subtotal** | ~1,000-2,000 | 20 pillars x 3 curves x 20 currencies |
| **Equity spots** | 100-5,000+ | Individual names or sector indices |
| **Equity vol surfaces** | 500-10,000+ | 5-10 strikes x 5-8 expiries per underlier |
| **FX spots** | 20-30 | Against USD or EUR |
| **FX vol surfaces** | ~500 | 5 deltas x 8 expiries x 12 major pairs |
| **Credit spreads** | 500-5,000 | By issuer x tenor bucket |
| **Commodities** | 100-500 | Forward curves per commodity |
| **Inflation** | 50-100 | Breakeven curves per currency |

**Rates-only desk:** ~1,000-3,000 factors. **Multi-asset bank-wide VaR:** 5,000-50,000 factors. JP Morgan's firm-wide RiskMetrics system famously tracked 10,000+ factors.

### Factor reduction: why you don't use raw factors

Nobody runs VaR on 10,000 raw factors with a full covariance matrix — it would be rank-deficient (you rarely have 10,000 independent daily observations). Industry uses dimensionality reduction:

- **PCA on yield curves.** The first 3 principal components (level, slope, curvature) explain ~95-99% of yield curve variance. Instead of shocking 20 pillar rates independently, shock 3-5 PC scores and reconstruct pillar moves. This is both more stable and more interpretable. See the [PCA Curve Shocks guide](pca-rates.md) for the end-to-end VALAX workflow.
- **Factor models for equities.** Decompose returns into systematic factors (market, sector, style) plus idiosyncratic. Shock the systematic factors jointly, add independent idiosyncratic noise. This is the Barra/Axioma approach.
- **Filtered historical simulation.** Weight recent observations more heavily (EWMA), or fit a GARCH model to each factor and standardize returns before drawing scenarios. This captures volatility clustering without inflating the factor count.

### Current VALAX factor structure

VALAX currently uses a flat factor vector with fixed ordering:

```
[spot_0..spot_n, vol_0..vol_n, rate_0..rate_p, div_0..div_n]
```

Total factors = `3 * n_assets + n_pillars`. The covariance matrix columns follow this ordering. This is sufficient for prototyping and single-desk risk, but has known limitations:

| Limitation | Impact | Future direction |
|---|---|---|
| No factor metadata | Covariance matrix columns are unlabelled — error-prone for large factor sets | Named factor registry |
| Scalar vol per asset | Can't shock the vol surface (ATM vs wings, short vs long expiry) | Vol surface risk factors |
| Single rate curve | Can't capture basis risk between OIS and SOFR | Multi-curve `MarketData` with per-curve shocks |
| No PCA / factor reduction | Full covariance matrix is impractical beyond ~500 factors | PCA yield curve factors, equity factor models |

!!! note "Practical guidance"
    For a small equity portfolio (10-50 names, one rate curve with 5-10 pillars), the current framework gives reasonable VaR numbers with ~100-200 factors. For anything larger, factor reduction (PCA on curves, systematic equity factors) is essential — and is a natural fit for JAX, since PCA is just `jnp.linalg.svd`.

---

## Sensitivity Ladders and Waterfall P&L

The basic `pnl_attribution` function (shown above) gives a scalar decomposition: one number for delta-spot, one for delta-vol, one for gamma. **Sensitivity ladders** extend this to a fully bucketed, multi-rung P&L decomposition with all second-order cross terms — vanna, volga, rate convexity, and cross spot×rate / vol×rate effects.

For the mathematical foundations, see [Models & Theory § 7.4](../theory.md#74-sensitivity-ladders).

### Computing a Ladder

A `SensitivityLadder` holds bucketed first- and second-order sensitivities for every risk factor in the portfolio:

```python
from valax.risk import compute_ladder

ladder = compute_ladder(pricing_fn, instruments, base_market)

# First-order (delta) ladders — one value per bucket
ladder.delta_spot    # shape (n_assets,)  — equity deltas
ladder.delta_vol     # shape (n_assets,)  — vega ladder
ladder.delta_rate    # shape (n_pillars,) — DV01 / rho ladder
ladder.delta_div     # shape (n_assets,)  — dividend sensitivity

# Second-order (gamma) ladders — diagonals
ladder.gamma_spot    # shape (n_assets,)  — spot gammas
ladder.gamma_rate    # shape (n_pillars,) — rate convexity
ladder.vanna         # shape (n_assets,)  — ∂²V/∂S∂σ
ladder.volga         # shape (n_assets,)  — ∂²V/∂σ²

# Cross blocks
ladder.cross_spot_rate  # shape (n_assets, n_pillars) — ∂²V/∂S∂r
ladder.cross_vol_rate   # shape (n_assets, n_pillars) — ∂²V/∂σ∂r
```

All sensitivities are computed via `jax.grad` (first order) and `jax.hessian` (second order) — no bump-and-reprice. Rate sensitivities are in zero-rate space (not DF space), matching the convention used by `pnl_attribution` and `parametric_var`.

### Waterfall P&L Decomposition

Given a precomputed ladder and a scenario, the waterfall breaks down the P&L into 10 rungs:

```python
from valax.risk import waterfall_pnl, waterfall_pnl_report

# Fast: arithmetic only, no repricing
wf = waterfall_pnl(ladder, scenario, base_market)

# Full: includes actual repricing and unexplained
wf = waterfall_pnl_report(pricing_fn, instruments, base_market, scenario, ladder=ladder)
```

The `WaterfallPnL` pytree contains every rung:

```python
# First-order rungs
wf.delta_spot            # Rung 1: Σ δ_S · ΔS
wf.delta_vol             # Rung 2: Σ ν · Δσ       (vega P&L)
wf.delta_rate            # Rung 3: Σ ρ · Δr       (DV01 P&L)
wf.delta_div             # Rung 4: Σ δ_q · Δq

# Second-order rungs
wf.gamma_spot            # Rung 5: ½Σ γ · ΔS²
wf.gamma_rate            # Rung 6: ½Σ γ_r · Δr²   (rate convexity)
wf.vanna_pnl             # Rung 7: Σ vanna · ΔS·Δσ
wf.volga_pnl             # Rung 8: ½Σ volga · Δσ²

# Cross rungs
wf.cross_spot_rate_pnl   # Rung 9:  Σ ∂²V/∂S∂r · ΔS·Δr
wf.cross_vol_rate_pnl    # Rung 10: Σ ∂²V/∂σ∂r · Δσ·Δr

# Aggregates
wf.total_first_order     # Rungs 1-4 summed
wf.total_second_order    # Rungs 5-10 summed
wf.predicted             # All rungs summed
wf.actual                # Full-repricing P&L (from waterfall_pnl_report)
wf.unexplained           # actual - predicted
```

### Ladder vs. pnl_attribution

The existing `pnl_attribution` computes a scalar decomposition with spot gamma only. The ladder adds:

| Existing (`pnl_attribution`) | New (`waterfall_pnl_report`) |
|-----|-----|
| Scalar delta_spot | **Bucketed** delta per asset |
| Scalar delta_vol | **Bucketed** vega per asset |
| Scalar delta_rate | **Bucketed** DV01 per pillar |
| Spot gamma only | Spot gamma **+ rate gamma + vanna + volga** |
| No cross terms | **Full cross-gamma** (spot×rate, vol×rate) |
| 5 output fields | **15 output fields** with full waterfall |

For small moves, both give similar results. For large moves (crash scenarios, rate shocks ≥ 100bp, vol moves ≥ 5pts), the ladder's second-order terms significantly reduce the unexplained residual.

### Reusing Ladders Across Scenarios

Computing the ladder is the expensive step (requires Hessian evaluation). Once computed, the waterfall for each scenario is just cheap array arithmetic:

```python
# Compute once
ladder = compute_ladder(pricing_fn, instruments, base_market)

# Decompose many scenarios (near-instant per scenario)
for scenario in scenarios:
    wf = waterfall_pnl(ladder, scenario, base_market)
    print(f"Predicted: {wf.predicted:.2f},  Gamma: {wf.gamma_spot:.2f}")
```

This pattern is natural for end-of-day P&L explain: compute the ladder at the close, then attribute P&L to each risk factor's daily move.

---

## P&L Vectors: Predict vs Actual

Every risk metric in VALAX — VaR, ES, backtests, the FRTB PLA test — reduces to a single primitive: a **P&L vector** of length $N$, with one entry per scenario or per historical day. Once you have the vector, the metrics are just sample statistics.

For the theoretical framing of HPL / RTPL / APL and why these three series are kept separate, see [Models & Theory § 7.5](../theory.md#75-pl-vectors-hypothetical-risk-theoretical-actual).

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

## VaR Backtesting

A VaR forecast is only as good as its track record. The Basel framework requires 99% one-day VaR to be backtested on a rolling 250-day window, with capital multipliers driven by the breach count. VALAX provides the standard Kupiec, Christoffersen, and traffic-light tools in `valax/risk/backtesting.py`.

For the underlying statistics, see [Models & Theory § 7.6](../theory.md#76-var-backtesting).

### Counting breaches

A breach occurs on a day where the realised loss exceeds that day's VaR forecast:

```python
import jax.numpy as jnp
from valax.risk import var_breaches

# Both vectors have shape (n_days,):
#   actual_pnl[t] is realised one-day P&L (negative = loss)
#   var_forecast[t] is that morning's VaR forecast (positive number = loss threshold)
breaches = var_breaches(actual_pnl, var_forecast)
# breaches.shape == (n_days,), boolean
n_breaches = int(jnp.sum(breaches))
```

### Kupiec proportion-of-failures (POF)

Tests whether the *count* of breaches is consistent with the VaR confidence level:

```python
from valax.risk import kupiec_pof

result = kupiec_pof(breaches, confidence=0.99)
result["n"]          # number of observation days
result["x"]          # number of breaches
result["lr_uc"]      # likelihood-ratio statistic, χ²₁ under H₀
result["p_value"]    # right-tail p-value
```

Reject the model at the 5% level if `lr_uc > 3.84` (or equivalently `p_value < 0.05`).

### Christoffersen independence and conditional coverage

Detects *clustering* of breaches (a model that breaches twice in a week then never again):

```python
from valax.risk import christoffersen_independence, christoffersen_conditional_coverage

ind = christoffersen_independence(breaches)
ind["lr_ind"]        # χ²₁ statistic, tests pi_01 = pi_11
ind["p_value"]

cc = christoffersen_conditional_coverage(breaches, confidence=0.99)
cc["lr_cc"]          # χ²₂ — joint test of correct rate AND independence
cc["p_value"]
```

### Basel traffic light

The regulatory zoning is just a count-based lookup on the cumulative binomial:

```python
from valax.risk import basel_traffic_light

zone = basel_traffic_light(n_breaches=n_breaches, n_obs=250, confidence=0.99)
# zone in {"green", "yellow", "red"}
```

For a 250-day window at 99% VaR: 0–4 breaches green, 5–9 yellow, ≥10 red — the multiplier on capital scales accordingly.

---

## FRTB P&L Attribution Test

The FRTB Internal Models Approach requires a second daily validation on top of the VaR backtest: each desk must demonstrate that its **risk-theoretical P&L (RTPL)** tracks its **hypothetical P&L (HPL)** in both rank order (Spearman) and distribution (Kolmogorov–Smirnov). Failing the PLA test forces the desk off internal models and onto the (usually more punitive) standardized approach.

For the regulatory background and threshold derivation, see [Models & Theory § 7.7](../theory.md#77-frtb-pl-attribution-test).

### Spearman rank correlation

```python
from valax.risk import pla_spearman

rho = pla_spearman(rtpl_series, hpl_series)
# rho is a scalar in [-1, 1]
```

Tests monotonic agreement: do RTPL and HPL produce the same ordering of best-to-worst days?

### Kolmogorov–Smirnov statistic

```python
from valax.risk import pla_ks, ks_statistic

D = pla_ks(rtpl_series, hpl_series)
# D is the max distance between empirical CDFs in [0, 1]
```

`ks_statistic(x, y)` is the underlying two-sample KS computation, exposed for general use.

### Traffic-light zone

```python
from valax.risk import pla_traffic_light

zone = pla_traffic_light(spearman=rho, ks_stat=D, n_obs=len(rtpl_series))
# zone in {"green", "amber", "red"}
```

The zone applies the BCBS d558 thresholds: green if Spearman ≥ 0.80 *and* KS $p$-value ≥ 0.264; red if either Spearman < 0.70 or KS $p$-value < 0.055; amber otherwise.

### End-to-end PLA workflow

Putting the pieces together: compute the ladder once, build both vectors over the same scenario set (the 250-day historical window), and read the test outputs:

```python
import jax.numpy as jnp
from valax.risk import (
    compute_ladder,
    risk_theoretical_pnl_vector,
    hypothetical_pnl_vector,
    pla_spearman,
    pla_ks,
    pla_traffic_light,
)

# scenarios: 250-day historical ScenarioSet
ladder = compute_ladder(pricing_fn, instruments, base_market)
rtpl   = risk_theoretical_pnl_vector(ladder, scenarios, base_market)
hpl    = hypothetical_pnl_vector(pricing_fn, instruments, base_market, scenarios)

rho = pla_spearman(rtpl, hpl)
D   = pla_ks(rtpl, hpl)
zone = pla_traffic_light(rho, D, n_obs=rtpl.shape[0])
print(f"Spearman={float(rho):.3f}, KS={float(D):.3f}, zone={zone}")
```

If the zone is red, inspect the unexplained vector first — large systematic residuals point to missing risk factors or third-order convexity that the ladder doesn't capture.

---

## Risk Bucketing

Raw autodiff sensitivities live in the *finest* possible factor space — one DV01 per pillar, one vega per asset, one credit delta per hazard pillar. For reporting, regulatory capital, and stable VaR, every desk re-expresses the same risk in coarser coordinates:

- **Regulatory bucketing** (FRTB SBA, ISDA SIMM) — standard tenor vertices, equity sectors, credit rating buckets.
- **Trader-friendly bucketing** — "short / belly / wings" on a curve, "tech / energy / financials" on equities.
- **Factor reduction** — PCA scores (level / slope / curvature) for well-conditioned VaR on a 30-pillar curve.
- **Parametric vol bucketing** — push grid-vol sensitivities into SABR or SVI parameter space via the calibration Jacobian.

VALAX provides two transformation families in `valax/risk/bucketing.py`. They share the same matrix algebra, but they correspond to two different mental models and are named accordingly. See [Models & Theory § 7.8](../theory.md#78-risk-bucketing-linear-and-jacobian-transformations) for the derivation.

### Linear aggregation: `BucketMap`

A `BucketMap` is a thin wrapper around an aggregation matrix `A` of shape `(n_buckets, n_factors)`:

```python
from valax.risk import BucketMap, aggregate, pushforward_scenario, aggregate_covariance
import jax.numpy as jnp

# Map 4 pillars → 2 buckets (short / long)
A = jnp.array([
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
])
rate_bm = BucketMap(matrix=A, bucket_labels=("short", "long"))

# Bucket a DV01 ladder
bucket_dv01 = aggregate(rate_bm, ladder.delta_rate)         # shape (2,)

# Push a bucket-level shock back to raw factors (the dual operation)
bucket_shock = jnp.array([0.0010, 0.0025])                  # +10 bp / +25 bp
raw_shock = pushforward_scenario(rate_bm, bucket_shock)     # shape (4,)

# Aggregate a 4×4 covariance to a 2×2 bucket covariance
bucket_cov = aggregate_covariance(rate_bm, raw_cov)
```

`pushforward_scenario` is the *dual* of `aggregate`: it is the unique factor shock that makes the bucket P&L equal to the raw P&L. Use it when you have a stress defined in bucket terms (e.g. "+10 bp on the short tenor") and need to apply it via `apply_scenario` / `portfolio_pnl`.

### Standard bucket builders

```python
from valax.risk import tenor_bucket_map, equal_weight_bucket_map

# FRTB-style tenor vertices with indicator weights
pillar_times = jnp.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
frtb_vertices = jnp.array([1.0, 5.0, 30.0])  # short, belly, long
rate_bm = tenor_bucket_map(pillar_times, frtb_vertices, weight="indicator")

# Linear (piecewise-linear) splitting — smooth re-binning
rate_bm_smooth = tenor_bucket_map(pillar_times, frtb_vertices, weight="linear")

# Sector / currency / rating bucketing
sector_bm = equal_weight_bucket_map(
    group_membership=(0, 0, 1, 1, 0),    # 5 stocks → 2 sectors
    n_buckets=2,
    bucket_labels=("tech", "energy"),
)
```

### Jacobian reparameterization

When the new factors are a smooth (possibly nonlinear) function of the raw factors — PCA scores, level/slope/curvature, SVI/SABR parameters — `BucketMap` is not the right abstraction. Use a Jacobian instead:

```python
from valax.risk import (
    pushforward_sensitivities,
    pullback_shocks,
    reparameterize_covariance,
    jacobian_from_fn,
    level_slope_curvature_jacobian,
    pca_jacobian,
)

# Hand-built level / slope / curvature factors (fixed Jacobian)
J = level_slope_curvature_jacobian(pillar_times)  # (n_pillars, 3)
lsc_delta = pushforward_sensitivities(J, ladder.delta_rate)  # 3 numbers

# PCA on 250 days of curve returns → top 3 components
J_pca, eigvals, frac_explained = pca_jacobian(
    curve_returns,        # shape (250, n_pillars)
    n_components=3,
)
pca_delta = pushforward_sensitivities(J_pca, ladder.delta_rate)
pca_cov = reparameterize_covariance(J_pca, raw_cov)

# Stress scenario in PCA space: +1 std on level only
pca_shock = jnp.array([1.0, 0.0, 0.0]) * jnp.sqrt(eigvals)
raw_shock_from_pca = pullback_shocks(J_pca, pca_shock)
```

For nonlinear reparameterizations (e.g. an SVI vol-grid function), build the Jacobian on-the-fly via autodiff:

```python
def svi_to_grid(svi_params):
    # svi_params -> vol[grid_strike, expiry] reconstruction
    return svi_vol_function(svi_params, strikes, expiries).flatten()

J = jacobian_from_fn(svi_to_grid, svi_params_base)
# Now J is the autodiff Jacobian; push grid-vega through it.
svi_param_vega = pushforward_sensitivities(J, raw_grid_vega)
```

### Bucketing a full sensitivity ladder

`bucket_sensitivity_ladder` applies independent bucket maps to every component of a `SensitivityLadder`, including bilateral aggregation of the cross-gamma blocks:

```python
from valax.risk import compute_ladder, bucket_sensitivity_ladder

ladder = compute_ladder(pricing_fn, instruments, base_market)

bucketed = bucket_sensitivity_ladder(
    ladder,
    rate_bucket=rate_bm,        # bucket pillar dimension
    spot_bucket=sector_bm,      # bucket asset dimension
    # vol_bucket / div_bucket: omitted ⇒ left at full granularity
)

bucketed.delta_rate          # (n_rate_buckets,)
bucketed.delta_spot          # (n_spot_buckets,)
bucketed.cross_spot_rate     # (n_spot_buckets, n_rate_buckets) — bilaterally bucketed
bucketed.rate_bucket_labels  # human-readable bucket names
```

The bucketed ladder is itself a pytree — it can be fed back into `waterfall_pnl`-style arithmetic when paired with bucket-level scenarios.

### When to use which

| Goal | Use |
|---|---|
| Sum DV01s into FRTB tenor vertices | `tenor_bucket_map` + `aggregate` |
| Aggregate equity Greeks by sector | `equal_weight_bucket_map` + `aggregate` |
| Apply a bucket-level stress to raw factors | `pushforward_scenario` |
| Project a 30 × 30 covariance to a 10 × 10 regulatory covariance | `aggregate_covariance` |
| Yield-curve PCA factors (level/slope/curvature from data) | `pca_jacobian` + `pushforward_sensitivities` |
| Hand-picked level/slope/curvature | `level_slope_curvature_jacobian` + `pushforward_sensitivities` |
| SABR/SVI parameter Greeks from grid Greeks | `jacobian_from_fn` + `pushforward_sensitivities` |
| Stress in bucket / PC coords, evaluate raw P&L | `pushforward_scenario` (linear) or `pullback_shocks` (Jacobian) |
| Variance-reduced parametric VaR on a sparse factor set | `pca_jacobian` + `reparameterize_covariance` |
