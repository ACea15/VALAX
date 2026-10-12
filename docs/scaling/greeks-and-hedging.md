# Reliable Greeks & Hedging at Scale

"Reliable at scale" means three things at once: the book's sensitivities are
**exact** (no bump-and-reprice noise), **consistent** (every trade differentiated
against the *same* market object), and **cheap** (the whole ladder in one or two
autodiff passes). The JAX-native design gives all three for free — the work is in
aggregating and mapping them to risk factors, and in turning them into hedges.

## Why book Greeks compose exactly

Portfolio Greeks are computed by differentiating a **summed portfolio value that
reads a shared market pytree**. `valax/risk/ladders.py::compute_ladder` builds a
`pv(...)` closure that rebuilds the shared `DiscountCurve` via `eqx.tree_at`,
`vmap`s a single pricer across the instruments, and **sums**:

```python
from valax.risk import compute_ladder

ladder = compute_ladder(pricing_fn, instruments, base)   # SensitivityLadder
# first order  : jax.grad over (spots, vols, dfs, dividends)
# second order : jax.hessian blocks (gamma, vanna, volga, cross_spot_rate, ...)
```

Sensitivities aggregate correctly because every instrument differentiates the
**same curve/surface leaves** — the ones `jax.grad` traces. Rate Greeks are
converted from discount-factor space to zero-rate space by the chain rule
$\partial V/\partial r_i = \partial V/\partial \mathrm{DF}_i \cdot (-t_i\,\mathrm{DF}_i)$.
The result is a `SensitivityLadder` (per-asset / per-pillar buckets) and, via
`waterfall_pnl`, a `WaterfallPnL` P&L decomposition — the full runnable how-to is
in the [Scenarios & Shocks guide](../risk-scenarios.md).

!!! tip "Two passes, whole book"
    First-order sensitivities come from one reverse-mode pass over the summed
    value; second-order blocks from one `jax.hessian`. A traditional
    bump-and-reprice ladder needs *O(factors)* full revaluations — VALAX needs
    *O(1)* AD passes, exact to machine precision.

## Mapping to risk factors: bucketing and PCA

Raw per-pillar sensitivities are too granular for reporting, limits, or
regulatory bucketing. `valax/risk/bucketing.py` supplies the linear algebra:
`BucketMap` aggregates a sensitivity vector ($\delta_b = A\,\delta_x$), pushes
scenarios forward, and reparameterises covariance ($A\,\Sigma\,A^\top$);
`bucket_sensitivity_ladder` maps a `SensitivityLadder` onto tenor/sector/currency
buckets (FRTB SBA, SIMM). `valax/risk/factors.py::RatesFactorModel` collapses many
curve pillars onto a few PCA factors (level/slope/curvature). The maths is in
[Theory → Risk Measures](../theory/risk-measures.md); the how-to in the
[Risk section](../risk-overview.md).

This factor space is also where **heterogeneous** groups combine — but "sum the
Greeks" is only correct **component-wise in a shared factor basis**. Two
conditions must hold: every group is differentiated against the *same* market
object, and sensitivities are expressed against the *same* factors (curve
pillars, per-underlying spot, vol-surface nodes) in the same units. Same-factor
sensitivities add — instruments that don't touch a factor contribute exact
zeros — while sensitivities to *different* underlyings, currencies, tenors, or
quoting conventions are separate components of a vector, never a scalar sum.
Crucially, **model-parameter Greeks are not additive across models** (a
swaption's $\partial/\partial\sigma_{\text{HW}}$ and an option's
$\partial/\partial\sigma_{\text{BS}}$ are different factors); they must first be
mapped to shared market factors — see the next section.

## Market-quote (hedge-space) risk

The most actionable basis is not curve pillars or model parameters but the
**market quotes** that calibrate them — par swap rates, cap/swaption vols,
risk-reversals/butterflies, CDS spreads. Sensitivities in this space *are* hedge
ratios ("trade X of the 5y5y swaption to flatten this vega"), and they aggregate
cleanly across a heterogeneous book because every instrument differentiates the
*same* quote leaves.

The principled way to get them is **adjoint/algorithmic differentiation through
the calibration pipeline** — quotes → calibrate (bootstrap curves, fit surfaces,
calibrate model params) → price → PV — differentiated end to end:

$$\frac{\partial \text{PV}}{\partial q}
  = \frac{\partial \text{PV}}{\partial \theta}\,\frac{\partial \theta}{\partial q},$$

where $\theta$ are the calibrated objects and $q$ the market quotes. Two rules
make this both correct and cheap:

- **Use implicit differentiation, not solver unrolling.** A bootstrap is defined
  by $F(\theta; q) = 0$ and a least-squares fit by its first-order condition; the
  implicit function theorem yields $\partial\theta/\partial q$ from the solver
  Jacobian at the converged point (no unrolling, exact at the optimum).
  `optimistix` differentiates its solvers this way out of the box, and the curve
  bootstrap exposes the implicit adjoint as `quote_jacobian`
  ([Why Multi-Curve?](../guide/why-multicurve.md)).
- **Calibrate once, then push forward.** Rather than re-running calibration inside
  every risk pass, take price-space Greeks against the *cached* calibrated objects
  and map them to quote space through the cached calibration Jacobian
  (`pushforward_sensitivities` in `valax/risk/bucketing.py`). This is identical to
  the full end-to-end gradient but decouples the expensive fit from every
  revaluation.

**Vega, concretely,** is then $\partial \text{PV}/\partial(\text{vol quotes})$ — a
matrix bucketed by strike/expiry/tenor, directly hedgeable and summable across the
whole book. Two caveats: an imperfect calibration (SABR/Heston rarely fit exactly)
gives the sensitivity of the *best-fit* surface — correct, but unhedgeable fit
error lives outside this basis; and over-parameterised fits (dense SVI, SLV
leverage grids) need ridge/regularisation to keep the calibration Jacobian
well-conditioned.

!!! note "Built vs planned"
    **Built:** the keyed quote container `QuoteSet`, calibration chains
    (`calibrate_market` → curves, spots, SABR smiles; `calibrate_model` → any
    model through `price`), and quote risk through the whole chain by two
    reconciled paths — the explicit adjoint sweep `chain_adjoint` (one sweep
    for a whole book's summed cotangent) and autodiff through the solves
    (`quote_risk_autodiff`); plus the curve-only adjoints and the
    pushforward/pullback algebra in `valax/risk/bucketing.py`. See
    [From Quotes to Risk](../guide/calibration-to-risk.md). **Planned:** FX and
    credit quote levels, rates vol cubes as chain nodes, per-trade model baskets
    (see the [roadmap](../roadmap.md#unified-calibration-pricing-and-risk)).

## Reliability caveats to respect at scale

- **Discontinuous payoffs.** Digitals and hard barriers have singular pathwise
  Greeks; smooth them with a sigmoid (see the MC payoff library) so `jax.grad`
  stays finite. The likelihood-ratio estimator is a planned alternative.
- **Common-seed MC Greeks.** For Monte-Carlo pricers, differentiate with a *fixed*
  PRNG key so the price and its Greeks share path realisations — see
  [Monte Carlo → Greeks](../guide/monte-carlo.md).
- **Shape/JIT stability.** Greeks are only as warm as the pricing kernel; the
  [batching](batching.md) shape rules apply to `grad`/`hessian` too.

## Hedging (planned)

The book Greek ladder is exactly the input a hedge engine needs, but the hedge
*construction* layer is not yet built. The intended scope:

- **Hedge ratios** straight from the ladder — e.g. key-rate DV01 neutralised with
  a basket of swaps, or delta/vega neutralised with liquid vanillas.
- **Netting sets** — aggregate offsetting risk before hedging so only residual
  risk is traded.
- **Target-Greek / minimum-variance hedges** solved with `optimistix`, fully
  differentiable so hedge sensitivities are themselves available.
- **Deep hedging** (RL, transaction-cost-aware) — the frontier item, roadmap
  [P5.3](../roadmap.md).

!!! note "Built vs planned"
    **Built:** exact book-level ladders (`compute_ladder`), waterfall P&L,
    bucketing/PCA aggregation. **Planned:** hedge-ratio and netting-set
    construction, target-Greek optimisation, deep hedging.
