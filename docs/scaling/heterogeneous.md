# Heterogeneous Portfolios

A real book is not homogeneous. It holds bonds, swaps, swaptions, options,
FX forwards and exotics — each a different `equinox.Module` with different static
fields, and each routed to a different model and pricing engine. This page is the
design for pricing such a book *and its Greeks* in one coherent pass. Most of it
is **planned**; the section is explicit about the seam between what ships and what
does not.

## Why a mixed book can't share one `vmap`

`jax.vmap` and `jax.jit` require a single pytree structure and a single traced
function. Two consequences:

- Instruments of **different types** have different pytree structure (different
  fields, different static metadata), so they cannot be stacked into one batched
  pytree.
- Instruments routed to **different models/engines** call different functions,
  so there is no single `pricing_fn` to `vmap`.

Everything built today assumes past this problem: `batch_price`,
[`compute_ladder`](../risk-ladders.md), and the VaR engine each take **one**
`pricing_fn` and **one** homogeneous instrument batch. Composing a mixed book is
currently left to the caller.

The single-instrument [`mc_price_dispatch`](../guide/monte-carlo.md) /
`pde_price_dispatch` registries *do* know how to route one `(instrument, model)`
pair to the right recipe — but they resolve one pair per call. They are the right
primitive to build on, not yet a batched book-level engine.

## The design: group → batch → dispatch → aggregate (planned)

The tractable pattern, and the one VALAX intends to ship:

1. **Group** the book by *pricing key* `(instrument type, model)`. Every group is
   internally homogeneous.
2. **Batch** each group into one stacked instrument pytree (leading axis = number
   of trades in the group).
3. **Dispatch** each group to its recipe via the existing registry, compiled once
   per group with `eqx.filter_jit` and vectorised with `jax.vmap`.
4. **Aggregate** the per-group results — sum PVs to the book value, and combine
   Greeks in a shared risk-factor space (see
   [Reliable Greeks & Hedging at Scale](greeks-and-hedging.md)).

```text
      mixed book
          │  group by (instrument, model)
          ▼
 ┌─────────────┬─────────────┬─────────────┐
 │ Swap×Curve  │ Swaption×HW │ Option×BSM  │   ← homogeneous groups
 │  vmap+jit   │  vmap+jit   │  vmap+jit   │   ← one compiled kernel each
 └─────────────┴─────────────┴─────────────┘
          │  aggregate (sum PV; combine sensitivities)
          ▼
   book value + book Greeks
```

Because compilation is per group (a handful of shapes), the compile cost is
bounded no matter how many trades the book holds; padding each group to a
[stable shape](batching.md) keeps the JIT cache warm across revaluations.

## The `Portfolio` container (planned)

The missing data structure is a first-class `Portfolio` — an `equinox.Module`
holding trade identifiers, the batched instruments per group, and the pricing key
for each group, so a book is itself a pytree that flows through `vmap`/`jit`/`grad`
like everything else. The design (fields `trade_ids`, grouped `instruments`,
`pricing_keys`, and a shared `MarketState`) is written up in
[Productionisation → the trade container](../architecture/production.md); the
container and the group-and-aggregate engine above are not yet implemented.

## What exists to build on today

- Homogeneous [`batch_price` / `batch_greeks`](batching.md) and the risk engine's
  `vmap`-over-instruments.
- The `(instrument, model)` recipe registries (`mc_price_dispatch`,
  `pde_price_dispatch`).
- The aggregation algebra — `SensitivityLadder`, `BucketMap`, and the PCA factor
  model — which already combines sensitivities in factor space once each group has
  produced its Greeks (see [Greeks & Hedging](greeks-and-hedging.md)).

!!! note "Built vs planned"
    **Built:** homogeneous batching, single-pair MC/PDE dispatch, factor-space
    aggregation. **Planned:** the `Portfolio` container, the
    group-by-`(instrument, model)` dispatch-and-aggregate engine over a mixed
    book, and shape-stable padding. Tracked in the
    [roadmap](../roadmap.md).
