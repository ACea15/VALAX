# Scaling Up: From Instrument to Business

Pricing one instrument is the easy part. A production system has to price a
*book* — thousands of heterogeneous trades — revalue it under thousands of
scenarios, and return a consistent set of Greeks, fast enough to matter. This
section is the plan for that journey: **single instrument → homogeneous batch →
heterogeneous book → enterprise scale**.

It is deliberately honest about status. Some rungs of the ladder ship today;
others are a design that the JAX-native architecture makes tractable. Each page
carries an explicit **Built / Planned** split.

## Why it composes at all

VALAX has one structural advantage that makes scaling a *composition* problem
rather than a *rewrite* problem: every pricer is a **pure function** of
**pytree** inputs, so the three JAX transforms stack without new code.

- `jax.vmap` — vectorise one pricer across a batch of instruments or scenarios.
- `eqx.filter_jit` — compile the whole batched graph to one XLA kernel.
- `jax.grad` / `jax.hessian` — exact Greeks of the *aggregated* book value.

The mechanics of writing `vmap`-safe functions live in
[Architecture → JAX Patterns](../architecture/jax-patterns.md); this section is
about applying them at book and business scale.

## The scaling ladder

<!-- coverage-map -->

| Rung | What it means | Status |
|---|---|---|
| **1. Instrument** | `price(instrument, *market_args)` — one pure function returning a scalar | ✅ the whole library |
| **2. Homogeneous batch** | `vmap` one pricer over many instruments of the *same type/model* sharing one market | ✅ `batch_price` / `batch_greeks`, book-level bucketed sensitivities |
| **3. Heterogeneous book** | a *mixed* collection of instrument types and models priced and aggregated together | 🟡 planned (`Portfolio` container + group-and-aggregate) |
| **4. Enterprise / business** | many books, scenarios, XVA nested MC — sharded across devices | 🟡 planned (multi-device; single-device today) |

## What each page covers

- **[Batch Pricing & JIT at Scale](batching.md)** — `vmap` over homogeneous
  batches, compile-once-per-shape, and keeping the JIT cache warm.
- **[Heterogeneous Portfolios](heterogeneous.md)** — the core unsolved problem:
  pricing a book of mixed instrument types/models in one pass, and the
  group-by-`(instrument, model)` dispatch-and-aggregate design.
- **[Reliable Greeks & Hedging at Scale](greeks-and-hedging.md)** — why a shared
  market pytree makes book Greeks compose exactly, bucketed sensitivities and
  risk-factor bucketing, and hedge construction.
- **[Parallelism & Devices](parallelism.md)** — single-device XLA + `vmap` today,
  and the multi-device (`jax.sharding`) plan for whole-bank books and XVA.

## Related, deliberately not duplicated here

The audience-facing "reprice the whole book in one call" framing and the runnable
risk how-to already live elsewhere — this section links to them rather than
restating:

- [Scenarios & Shocks guide](../risk-scenarios.md) — the how-to for full-revaluation
  VaR/ES, bucketed sensitivities, and `BucketMap` aggregation.
- [Market Risk](../applications/market-risk.md) ·
  [Front-Office Trading](../applications/trading-desk.md) ·
  [Treasury & ALM](../applications/treasury.md) — book-scale revaluation from
  each desk's point of view.
- [Roadmap](../roadmap.md) — the sequencing of the planned scaling items
  (`Portfolio` container, heterogeneous dispatch, multi-device, GPU benchmarks).
