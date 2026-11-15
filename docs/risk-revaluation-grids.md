# Revaluation Grids (Ladders)

*P&L to **all orders** along a direction, from full revaluation at shifted levels. Prerequisites: [Scenarios & Shocks](risk-scenarios.md) and [Bucketed Sensitivities](risk-sensitivities.md). Maths: [Theory § 7.4](theory/risk-measures.md#revaluation-grids-pl-beyond-second-order). Assumptions and limitations: [Risk Register R-11](applications/risk-register.md#r-11-revaluation-grids).*

## Why a grid, and not more Greeks

[Bucketed sensitivities](risk-sensitivities.md) are derivatives **at today's
market**. A P&L built from them is a Taylor expansion that VALAX truncates at
**second order** (delta and gamma, plus vanna, volga and the cross-gammas).
That is excellent for small moves and degrades as the move grows: the
`unexplained` residual of a waterfall is exactly the third- and higher-order
terms the expansion threw away.

A **revaluation grid** keeps those terms. Pick a direction in factor space
(a curve pillar, a whole curve in parallel, a spot) and a set of shift
levels along it, the **rungs**. The grid reprices the book at every rung —
present value *and* the exact directional delta — and the P&L of any move
up to the outer rung is read off by interpolation:

$$
\mathrm{P\&L}(m) \;=\; \int_0^{m} \delta(h)\,dh \;=\; V(m) - V(0).
$$

Every rung is a full revaluation, so **the P&L along the direction contains
every order** — gamma, but also the third, fourth, … order convexity of the
book — and the only error left is the interpolation error *between* rungs,
which shrinks with the rung spacing. A one-dimensional grid is what desks
call a **ladder**: a spot ladder, a vol ladder, a rate ladder at shifted
curve levels.

| | Bucketed sensitivities | Revaluation grid (ladder) |
|---|---|---|
| Evaluated at | the base market only | every rung (shifted level) |
| Order of the P&L along a direction | 2 (Taylor, truncated) | **all orders** (interpolation error only) |
| Cross-direction terms | 2nd order (full Hessian) | none by itself; add the 2nd-order cross terms from the sensitivities |
| Cost | 1 gradient (+1 Hessian) | `n_dirs × n_levels` repricings with tangent |
| Best for | small moves, hedging, VaR by covariance | large moves, stress, convex books, FRTB curvature |

## Quick start: a two-curve key-rate ladder

```python
import jax.numpy as jnp
from valax.risk import (
    KeyedFactorLayout, key_rate_directions, keyed_sensitivities,
    revaluation_grid, revaluation_grid_pnl,
)

layout = KeyedFactorLayout.from_market_state(market)

# One ladder per pillar of each curve: OIS discounting and 3M projection.
dirs = key_rate_directions(layout, "USD.SOFR.OIS") | key_rate_directions(layout, "USD.SOFR.3M")
levels = jnp.linspace(-0.02, 0.02, 9)        # -200 bp … +200 bp, 50 bp rungs
grid = revaluation_grid(swaps, None, market, dirs, levels, layout=layout)

grid.ladder("IR.CURVE.USD.SOFR.OIS.1Y")      # full-revaluation P&L at each rung
grid.delta_ladder("IR.CURVE.USD.SOFR.OIS.1Y")  # exact delta at each rung

# On the day: today's moves per direction, read off the grid.
sens = keyed_sensitivities(swaps, None, market, layout, second_order=True)
today = {fid: 0.0125 for fid in dirs}        # +125 bp everywhere: between rungs
pnl = revaluation_grid_pnl(grid, today, sensitivities=sens)

pnl.by_direction    # per pillar, all orders along that pillar
pnl.cross           # 2nd-order cross-pillar / cross-curve terms (from sens)
pnl.total           # sum + cross
pnl.out_of_range    # True where a move went past the outer rung
```

This is the quick-start of `tests/test_risk/test_revaluation_grids.py::test_guide_example_runs`,
which checks the total against full revaluation to `rel=1e-5`.

## Directions

A direction is a vector in the coordinates of a
[`KeyedFactorLayout`](risk-scenarios.md#keyed-generators-and-the-per-curve-covariance-layout),
so a level means what a shock of that size means everywhere else in the
risk layer.

| Helper | Directions it returns | A level of `h` means |
|---|---|---|
| `key_rate_directions(layout, curve_id)` | one per pillar, labelled by factor id (`IR.CURVE.<curve>.<t>Y`) | a zero-rate move of `h` at that pillar alone (`0.01` = 100 bp) |
| `parallel_direction(layout, "curve", curve_id)` | one, labelled by the block prefix | a parallel zero-rate move of the whole curve |
| `parallel_direction(layout, "spot", asset_id)` | one | a spot move (absolute; relative if the layout is `multiplicative`) |
| `parallel_direction(layout, "vol", surface_id)` | one | a parallel vol shift of the surface |
| any `KeyedScenario` | one, the scenario as a unit move | `h` times that scenario (a steepener, a basis widening, a crash) |

Combine helpers with `|`. Directions with **disjoint** support (different
pillars, different curves, different assets) add up to a total; overlapping
directions (say a parallel shift *and* key-rate shifts of the same curve)
are allowed, each ladder is still valid on its own, but
`revaluation_grid` warns and `GridPnL.total` is NaN, because summing them
would double count.

## Rungs, and how many you need

`levels` is shared by every direction, must be strictly increasing and must
contain `0.0` (the base market). The interpolation error falls with the
rung spacing $w$:

| `interpolation=` | Uses | Error | Measured (option book, spot +23.7, rungs at −30…+30) |
|---|---|---|---|
| `"hermite"` (default) | PV **and** delta at each rung (cubic Hermite) | $O(w^4)$, exact at every rung | 5.9e-4 → 2.3e-5 → 2.5e-6 for $w$ = 10 → 5 → 2.5 |
| `"delta"` | delta rungs only (integrate the piecewise-linear delta) | $O(w^2)$ | 0.172 → 0.045 → 0.011 |

For comparison, the second-order Taylor explain of the same move is off by
**1.76** on a P&L of 33.6: the Hermite grid at the coarsest spacing is
about 3,000× closer. The `"delta"` mode is the classical "sum the delta
ladder" recipe; it is offered because desks know it, but with the PVs
already on the grid, Hermite is the better reader.

A move beyond the outer rung is extrapolated **linearly** from that rung's
PV and delta — first order only — and flagged in `GridPnL.out_of_range`.
Size the outer rungs to the largest move you need to explain (a stress
scenario, the FRTB risk weight).

## Cross terms: what a set of ladders does not see

Each ladder captures **its own direction** to all orders. A sum of ladders
captures **no interaction between directions**: the
$\partial^2 V / \partial x_i \partial x_j$ terms between pillars, between
the discount and projection curves, or between spot and vol. Pass
second-order `KeyedSensitivities` to add them at second order:

$$
\mathrm{P\&L} \;\approx\; \sum_k \mathrm{P\&L}_k(m_k)
\;+\; \Bigl[\tfrac12 x^{\top} H x - \sum_k \tfrac12 m_k^2\, d_k^{\top} H d_k\Bigr],
\qquad x = \sum_k m_k d_k .
$$

On the dual-curve swap book of the quick start, +100 bp on every pillar of
both curves (P&L 38,238): the ladders alone are off by **966** (the OIS ×
projection cross terms), ladders plus cross terms by **0.027**, the
second-order Taylor explain by **2.5**. Mixed terms of third and higher
order between directions are still outside the number; a two-dimensional
product grid (spot × vol, curve × curve) is the remedy and is
[on the roadmap](roadmap.md#1-risk-engine-keyed-end-to-end-bar-portfolio-aggregation).

## Cost and engines

A grid costs `n_dirs × n_levels` repricings, each carrying one forward-mode
tangent (`jax.jvp`) for the exact delta, `vmap`-ped over levels and
directions through `price()`. The quick start is 8 directions × 9 rungs =
72 nodes. Any engine `price()` routes works; for an MC group, pass `key`:
it is reused at every rung, so the ladder is a response to the shifts and
not to sampling noise (the same common-random-numbers rule as
`group_scenario_pnl`). `revaluation_grid_pnl` is pure arithmetic on the grid
and is `eqx.filter_jit`-safe; moves can carry leading batch dimensions (a
whole scenario set at once).

## Where grids are used

- **Desk risk.** Spot, vol and rate ladders for convex books; stress P&L
  that is honest for large moves.
- **P&L explain.** Per-direction P&L to all orders, with the cross terms
  from the sensitivities; the residual left is cross-direction higher order
  only.
- **FRTB curvature.** The standardised approach's curvature charge reprices
  each bucket at ± its risk weight — a three-rung grid per bucket. The
  risk weights and the CVR aggregation are planned; status in the
  [FRTB capability matrix](applications/regulatory.md#frtb-capability-matrix).

---

*Reference: [API → Risk § Revaluation grids](api/risk.md#revaluation-grids). Next: [VaR & P&L](risk-var.md).*
