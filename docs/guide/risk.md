# Risk: Scenarios, Shocks, and VaR

VALAX provides scenario generation, multi-curve shocks, and Value-at-Risk computation — all built on `jax.vmap` for massively parallel portfolio repricing.

## Why Multi-Curve?

### The pre-2008 world: one curve does everything

Before the financial crisis, the industry used a single yield curve for both **discounting** (computing present values) and **forecasting** (projecting future floating rates). This worked because the spread between LIBOR and OIS (the overnight indexed swap rate, a proxy for the risk-free rate) was negligible — typically 1-2 basis points.

Under this single-curve framework, the discount factor to time $t$ and the forward rate between $t_1$ and $t_2$ are derived from the same set of zero rates:

$$DF(t) = e^{-r(t) \cdot t}, \quad F(t_1, t_2) = \frac{DF(t_1)/DF(t_2) - 1}{\tau(t_1, t_2)}$$

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

- **PCA on yield curves.** The first 3 principal components (level, slope, curvature) explain ~95-99% of yield curve variance. Instead of shocking 20 pillar rates independently, shock 3-5 PC scores and reconstruct pillar moves. This is both more stable and more interpretable.
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
