# Parallelism & Devices

Scaling is not only *vectorisation* — eventually a book or a nested-MC XVA run
outgrows a single device. This page separates what runs in parallel today
(single-device XLA + `vmap`) from the multi-device plan.

## Today: single-device XLA + `vmap`

VALAX gets its parallelism from XLA. A `vmap`ped, `jit`ted graph is compiled to
one kernel that the backend runs across the cores/SMs of **one device**. Full
book revaluation is exactly this: `vmap` over the scenario axis *and* the
instrument axis, summed to a P&L vector, in a single compiled call — e.g. the
full-revaluation VaR/ES engine reprices *scenarios × instruments* at once (see
[Scenarios & Shocks](../risk-scenarios.md) and [Market Risk](../applications/market-risk.md)).

**GPU/TPU is a backend swap, not a rewrite.** The same pure functions run on GPU
or TPU with no source change — set `JAX_PLATFORMS=gpu` (or
`jax.config.update("jax_platform_name", "gpu")`). This is a structural advantage
over C++ stacks that need bespoke CUDA kernels.

```python
# No code change — the same batched, jitted pricer runs on whichever
# backend JAX is configured for.
import jax
print(jax.devices())          # [CpuDevice] or [cuda:0, ...]
prices = price_book(...)      # runs on the active device
```

**What single-device buys you and where it stops.** One modern GPU already
reprices large books × thousands of scenarios in one call. The ceiling is device
memory and a single accelerator's throughput — a whole-bank book, or an XVA
exposure simulation (paths × dates × netting-set revaluations), can exceed it.

## Planned: multi-device parallelism

None of the following is implemented yet — there is **no `pmap`, `jax.sharding`,
`Mesh`, or `shard_map`** anywhere in the source today. The plan:

- **Shard the batch axis across devices.** Place the instrument or scenario axis
  on a device `Mesh` with `jax.sharding.NamedSharding`, so `vmap`/`jit` graphs run
  SPMD across GPUs/TPUs with the same code. `shard_map` gives explicit control
  where collective operations (e.g. a book-level sum) are needed.
- **Nested-MC XVA as the prime target.** Exposure simulation — outer market paths
  × inner repricing at each date × netting-set aggregation — is the most
  compute-intensive workload and the best fit for device sharding plus the
  neural-surrogate acceleration in roadmap P5.1.
- **GPU/TPU benchmark suite (roadmap P5.2).** Published GPU-vs-CPU numbers for
  batch pricing, full-reval VaR, and XVA exposure — the evidence for the
  performance claim. Currently planned, not yet produced (see
  [Benchmarks](../benchmarks.md)).

## Determinism at scale

Reproducibility is a hard requirement for risk and regulatory sign-off. Floating
-point reductions on GPU are not bit-identical to CPU (and can vary with
reduction order), so book values can differ in the last few digits across
backends. Treat the device as part of the reproducibility contract and pin it for
audited runs; the determinism caveats are detailed in
[Productionisation](../architecture/production.md).

!!! note "Built vs planned"
    **Built:** single-device XLA parallelism via `vmap`/`jit`; GPU/TPU by backend
    swap with no code change. **Planned:** all multi-device work — `jax.sharding`
    `Mesh` / `shard_map` book and scenario sharding, XVA nested-MC scaling, and
    the GPU/TPU benchmark suite (roadmap P5.2). Tracked under
    [Portfolio & Multi-Device Scaling](../roadmap.md).
