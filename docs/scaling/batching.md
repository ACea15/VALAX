# Batch Pricing & JIT at Scale

The first rung above a single instrument is a **homogeneous batch**: many
instruments of the *same type*, priced under the *same model/market*, in one
vectorised call. This is fully implemented and is the workhorse behind
full-revaluation risk.

## `batch_price` and `batch_greeks` (built)

`valax/portfolio/batch.py` wraps `jax.vmap` around any pricing function and the
autodiff Greeks helper. The batched `instruments` argument is a single
`equinox.Module` whose leaves each carry a leading batch axis of length `n`; the
market arguments are length-`n` vectors.

```python
import jax, jax.numpy as jnp
from valax.instruments import EuropeanOption
from valax.pricing.analytic import black_scholes_price
from valax.portfolio import batch_price, batch_greeks

# A batch of 10,000 options as ONE pytree with leading axis 10_000.
options = EuropeanOption(
    strike=jnp.linspace(80.0, 120.0, 10_000),
    expiry=jnp.full(10_000, 1.0),
    is_call=True,  # static field — shared across the batch
)
spots = jnp.full(10_000, 100.0)
vols  = jnp.full(10_000, 0.20)
rates = jnp.full(10_000, 0.05)
divs  = jnp.full(10_000, 0.02)

prices = batch_price(black_scholes_price, options, spots, vols, rates, divs)
greeks = batch_greeks(black_scholes_price, options, spots, vols, rates, divs)
# prices.shape == (10_000,);  greeks["delta"].shape == (10_000,)
```

Two constraints follow directly from `vmap`:

- **One pricing function, one instrument type.** Every element must be priced by
  the same `pricing_fn` and share the same pytree structure and static fields
  (`is_call`, day-counts, `n_steps`). Mixing types/models needs the
  [heterogeneous](heterogeneous.md) layer.
- **Static fields are shared, not batched.** Anything marked
  `eqx.field(static=True)` is part of the compiled graph, so it cannot vary
  element-by-element within one call.

The same pattern powers book-level risk: [`compute_ladder`](../risk-ladders.md) and
the VaR engine `vmap` a single pricer across the instrument axis and sum to a
portfolio value — see [Reliable Greeks & Hedging at Scale](greeks-and-hedging.md).

## JIT at scale: compile once per shape

`jax.jit` / `eqx.filter_jit` compiles a **specialised kernel per input shape**.
The economics of scale are therefore: pay compilation once, then amortise it over
the whole batch and every subsequent repricing of the same shape.

```python
import equinox as eqx

price_book = eqx.filter_jit(batch_price)
price_book(black_scholes_price, options, spots, vols, rates, divs)  # compiles
price_book(black_scholes_price, options, spots2, vols2, rates2, divs2)  # reuses
```

Rules of thumb that keep the cache warm:

- **Reuse the same config objects.** `MCConfig(n_paths, n_steps)` and the PDE
  configs are static — a new `(n_paths, n_steps)` pair triggers a fresh
  compile. Fix them across a run.
- **Keep batch shapes stable.** A book that grows from 9,998 to 10,001 trades is
  a new shape and recompiles. In a warm service you want a small set of stable
  shapes, not one shape per request.
- **Don't put per-element data in static fields.** A static field that varies
  across calls forces recompilation; keep varying data in JAX arrays.

## Shape-stable / padded batching (planned)

The clean way to bound compilation in a live service is to **pad every batch to a
small set of fixed shapes** (e.g. powers of two) and carry a validity mask, so a
book of any size maps onto an already-compiled kernel. VALAX does not ship a
padding/bucketing helper yet — today the caller chooses batch shapes. This is
tracked under [Portfolio & Multi-Device Scaling](../roadmap.md) and is a
prerequisite for the warm-service design in
[Productionisation](../architecture/production.md).

!!! note "Built vs planned"
    **Built:** `batch_price`, `batch_greeks`, and `vmap`-over-instruments inside
    the risk engine — all single-device. **Planned:** shape-stable padding for
    JIT-cache reuse across differently-sized books.
