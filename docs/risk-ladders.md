# Sensitivity Ladders

*The bucketed, multi-rung generalisation of scalar P&L attribution. Prerequisites: [Scenarios & Shocks](risk-scenarios.md) and [VaR & P&L](risk-var.md). Maths: [Models & Theory § 7.4](theory/risk-measures.md#74-sensitivity-ladders).*

## Sensitivity Ladders and Waterfall P&L

The basic `pnl_attribution` function (see [VaR & P&L](risk-var.md#pl-attribution)) gives a scalar decomposition: one number for delta-spot, one for delta-vol, one for gamma. **Sensitivity ladders** extend this to a fully bucketed, multi-rung P&L decomposition with all second-order cross terms — vanna, volga, rate convexity, and cross spot×rate / vol×rate effects.

For the mathematical foundations, see [Models & Theory § 7.4](theory/risk-measures.md#74-sensitivity-ladders).

!!! note "Two ladders: keyed (current) and flat (legacy)"
    This section documents `compute_ladder` / `SensitivityLadder` /
    `waterfall_pnl` / `waterfall_pnl_report`, which run on the legacy flat
    `MarketData` path with an opaque `pricing_fn`. On a `MarketState`,
    priced through `price(instrument, model, market)`, use
    [`keyed_ladder` → `KeyedLadder`](#keyed-ladders-on-a-marketstate) and
    `keyed_waterfall_pnl` / `keyed_waterfall_report` below. The flat ladder
    is still what `risk_theoretical_pnl_vector` and
    `bucket_sensitivity_ladder` consume; their keyed port is pending.

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

Both of these are laid out **positionally**, with one rate block; for a
multi-curve book see [§ Keyed ladders](#keyed-ladders-on-a-marketstate)
below.

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

## Keyed ladders on a `MarketState`

`SensitivityLadder` is laid out positionally: `delta_rate` is one vector,
so on a dual-curve book the OIS and projection sensitivities are **added
together before anyone sees them**. `KeyedLadder`
(`valax.risk.state_ladders`) is the counterpart for the container the
unified pricer takes, indexed by the same
[`KeyedFactorLayout`](risk-scenarios.md#keyed-generators-and-the-per-curve-covariance-layout)
as the scenarios and the covariance matrix — one block per named curve,
surface and asset.

```python
from valax.risk import keyed_ladder

ladder = keyed_ladder(swaps, None, dual_market, second_order=True)

ladder.dv01("USD.SOFR.OIS")        # discount DV01, projection held fixed
ladder.dv01("USD.SOFR.3M")         # projection DV01, discounting held fixed
ladder.curve_ladder("USD.SOFR.3M") # per-pillar key-rate ladder
ladder.vega("SPX.VOL")             # parallel-shift vega of one surface
ladder.block_totals()              # one number per named market object
ladder.to_dict()                   # keyed by hierarchical factor id
```

It prices through the one front door (`valax.core.price`, via
`group_pv`), so one `jax.grad` produces the whole first-order ladder and
one `jax.hessian` the whole second-order block matrix.

### Differentiating the shock, not the leaf

The design difference that matters: `compute_ladder` differentiates with
respect to *discount factors* and then converts to zero-rate space by hand
($\partial V/\partial r_i = \partial V/\partial DF_i \cdot (-t_i DF_i)$,
plus a correction term for gamma). `keyed_ladder` differentiates

$$
x \;\longmapsto\; \mathrm{PV}\bigl(\mathrm{shock}(\text{market},\;
  \mathrm{unflatten}(x))\bigr)
$$

at $x = 0$, where $x$ is the layout's factor vector. So:

- **the chain rule cannot be wrong** — the zero-rate convention is whatever
  `bump_graph_curve` means by it, exactly, to all orders;
- **vega is defined for every surface type**, because a vol coordinate is
  the magnitude of the `shift_surface` parallel shift rather than a leaf of
  a particular surface class;
- **ladder and scenario cannot disagree** — the predicted P&L is a Taylor
  expansion of the same function a full revaluation evaluates, so
  `unexplained` is the genuine higher-order remainder.

### Per-block waterfall and per-curve VaR

`keyed_waterfall_pnl` applies the algebra of the waterfall above, but with
**one rung per named market object** instead of one per asset class:

```python
from valax.risk import keyed_waterfall_report

wf = keyed_waterfall_report(swaps, None, dual_market, basis_widening,
                            ladder=ladder)
wf.delta_pnl        # {'IR.CURVE.USD.SOFR.OIS': 0.0, 'IR.BASIS...': 9805.1}
wf.predicted, wf.actual, wf.unexplained
```

A pure basis widening puts exactly zero on the discount rung — the
attribution a single `delta_rate` vector cannot produce. And because the
ladder and the covariance share coordinates, VaR decomposes the same way:

```python
from valax.risk import keyed_parametric_var, keyed_component_var

keyed_parametric_var(ladder, cov, confidence=0.99)
keyed_component_var(ladder, cov)   # Euler allocation; sums to the total
# {'IR.CURVE.USD.SOFR.OIS': 5922.2, 'IR.BASIS.USD.SOFR3M.SOFROIS': 493.7}
```

That last line is the multi-curve risk number the flat layout cannot
produce: a projection curve's own contribution to the book's VaR, additive
with the discount curve's. Build the covariance with
[`block_covariance`](risk-scenarios.md#keyed-generators-and-the-per-curve-covariance-layout)
and, for a basis rung, build the ladder on an `as_basis_layout` layout.

!!! note "Which ladder to use"
    `compute_ladder` / `waterfall_pnl` take a `MarketData` and an opaque
    `pricing_fn`, and remain for one deprecation cycle. New code should use
    `keyed_ladder` on a `MarketState`: it needs no pricing-function
    adapter, it keys risk by curve, and it goes through `price(...)`.

---

*Next: coarsen the ladder for reporting and capital in [Risk Bucketing](risk-bucketing.md).*
