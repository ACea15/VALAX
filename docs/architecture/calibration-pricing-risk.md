# Calibration, Pricing & Risk Architecture

How VALAX connects market quotes to prices and back to risk, and how the
framework grows: new instruments, models, numerical methods, calibration levels
and risk factors each plug in through one small contract. The mathematics is on
[Calibration Chains](../theory/calibration-chains.md); a runnable walk-through
is [From Quotes to Risk](../guide/calibration-to-risk.md).

## 1. Three maps

Every risk number in VALAX is built from three maps:

```text
 quotes q, fixed inputs p
        │  calibrate_market(chain, q, p)          ── shared, once per snapshot
        ▼
 MarketState  ──────────────┐
        │  calibrate_model(node, q, market, p)    ── per basket / underlying
        ▼                   │
 model ─────────────────────┤
                            ▼
              price(instrument, model, market)     ── per trade
                            │
                            ▼  Σ quantity × price   ── per book (linear)
                         book value
```

1. **Calibration** — quotes and fixed inputs → calibrated objects, through a
   chain of levels ``R(θ; q, upstream, p) = 0``.
2. **Pricing** — ``price(instrument, model, market)``; reverse-mode AD gives the
   market and model cotangents.
3. **Aggregation** — linear, so it commutes with differentiation: a book's
   cotangent is the sum of its trades'.

The consequences are the design:

* **Separation.** Instruments, models and methods never see calibration;
  calibration never sees the trades being priced. Each axis grows on its own.
* **Risk in quote space is one sweep.** The chain adjoint is a linear map from
  the market (and model) cotangent to ``(q̄, p̄)``; it does not depend on the
  instrument, so a book is swept once with its summed cotangent.
* **Degenerate markets are data, not code paths.** Constant-parameter
  Black-Scholes is a chain with one explicit level; single-curve is a chain with
  one curve block; multi-curve is the same node with more specs.

## 2. Contracts

| Concern | Type / function | Module |
| --- | --- | --- |
| Quotes ``q`` and fixed inputs ``p`` | `QuoteSet` (keyed, differentiable, vmappable) | `valax.market.quotes` |
| Calibrated market | `MarketState` (`flat`, `single_curve`, keyed multi-curve) | `valax.core.market_state` |
| One level | `CalibrationLevel`, `LevelInputs(quotes, upstream, fixed)` | `valax.calibration.level` |
| Solver choice and tolerances | `SolverSettings`, `run_solver` | `valax.calibration.settings` |
| Result of a calibration | `CalibrationResult` (value, level, traceable diagnostics, `check()`) | `valax.calibration.result` |
| Calibration step | `CalibrationNode` (market scope: `install_fn`; model scope: none) | `valax.calibration.chain` |
| Market calibration | `CalibrationChain`, `calibrate_market` → `CalibratedMarket` | `valax.calibration.chain` |
| Model calibration | `calibrate_model(node, q, market, p)` | `valax.calibration.chain` |
| Node factories | `market_inputs_node`, `curve_graph_node`, `sabr_smile_node`, `price_fit_node` | `valax.calibration.nodes` |
| Pricing | `price(instrument, model, market, method=, config=)` | `valax.core.pricing` |
| Quote risk (adjoint) | `chain_adjoint`, `quote_risk_adjoint` | `valax.greeks.quote_risk` |
| Quote risk (autodiff) | `quote_risk_autodiff`, `reconcile_quote_risk` | `valax.greeks.quote_risk` |

## 3. Decisions

**D1 — Two differentiation paths, both first-class.** The explicit adjoint
sweep (`chain_adjoint`) reuses each level's residual at the solution, reports
per-level stationarity, and supports book-level aggregation. The autodiff path
(`quote_risk_autodiff`) differentiates through the solves with the implicit
adjoint `optimistix` attaches to each. They share no differentiation code, fail
in different ways (an ill-conditioned level vs a non-traceable calibrator), and
agree to solver tolerance — `reconcile_quote_risk` reports both and the
mismatch, never silently preferring one. Neither path survives a non-converged
calibration, which is what the convergence flag is for.

**D2 — The model is a separate input from the market.** The market is shared
by every trade; a model is calibrated on a basket chosen by the trade (rates
exotics) or by the underlying (equity books). ``price`` takes them as separate
arguments, `eqx.filter_grad` over ``(model, market)`` returns both cotangents,
and `chain_adjoint` folds model levels in through `model_levels`. Market-scope
and model-scope calibrations are the *same* `CalibrationNode` contract; only
the destination of the value differs, so one model class (e.g. Heston) can
calibrate a market surface or price an exotic.

**D3 — One solver factory, all-static settings, traceable convergence.**
Every calibration routes through `run_solver` with a `SolverSettings`
(algorithm, `rtol`, `atol`, `max_steps`, `throw`). Convergence is reported as a
JAX boolean on `CalibrationResult.diagnostics.converged`, so it survives
`jit` and `vmap`; `result.check()` / `calibrate_market(..., strict=True)` raise
eagerly. Exact levels default to Newton at ``1e-10``; least-squares levels to
Levenberg-Marquardt at ``1e-8`` (price-based fits have a noise floor a tighter
tolerance cannot get under).

**D4 — The level is emitted by the calibrator.** A level's residual is the
function the solver drove to zero, packaged with its solution by
`solve_level`; nothing rebuilds a residual for the adjoint.

**D5 — No `custom_vjp`.** Calibrations are differentiated by the implicit
function theorem (explicitly, or through `optimistix`), pricers by plain AD.
A custom rule would be reverse-only and could hide non-convergence.

**D6 — Validation happens when a node is built, not when it runs.** Node
factories check shapes, squareness and key counts eagerly; the calibration
itself is traceable, so a whole chain can be `jit`-ed, differentiated and
`vmap`-ed over quote scenarios.

## 4. Extending the framework

| To add… | You provide | It then works with |
| --- | --- | --- |
| **Instrument** | An `eqx.Module` (data only, static selectors such as `discount_id`) and `@register_pricer` adapters | `price`, both risk paths, `price_fit_node` baskets |
| **Model** | An `eqx.Module` with no embedded market objects (fields left `None` are bound from the market) plus pricers | `price`, `price_fit_node` calibration at either scope |
| **Numerical method** | A config class and pricers registered under the method name | `price(..., method=...)`, calibration through `price_fit_node(method=...)` |
| **Calibration level** | A node factory: residual ``(θ, inputs) → vector``, `assemble` ``(θ, inputs) → object``, `install` ``(market, object) → market`` (or none), routed through `solve_level` / `explicit_level` | `calibrate_market`, both risk paths, diagnostics, conditioning |
| **Risk factor** | A quote key and the node that consumes it (for a new market slot: an installer and a `MarketState` field) | quote risk keyed by that identifier |

A new level needs **no adjoint code**: `solve_level` packages the residual,
`chain_adjoint` pulls cotangents back through `assemble`, and the
`LevelInputs` split delivers ``q̄``, the upstream cotangent and ``p̄`` from one
VJP. The tests for a new level are the same three every level has: the two
paths reconcile, both match full-recalibration finite differences, and a
deliberately unconvergeable fit is reported (see
`tests/test_greeks/test_quote_risk.py` and
`tests/test_calibration/test_settings.py`).

## 5. Scaling

* **Many trades, one market.** Sum the trades' market cotangents and sweep the
  chain once (`chain_adjoint`); per-trade pricing batches with
  `eqx.filter_vmap` over homogeneous groups (`valax.risk.state_var`).
* **Many scenarios.** `QuoteSet` values are leaves, so a stacked quote set is a
  scenario batch: `jax.vmap` over `calibrate_market` recalibrates every
  scenario in one compiled program.
* **Many baskets.** Same-shaped model calibrations (one per trade or per
  underlying) batch with `eqx.filter_vmap(calibrate_model)`; a failed member
  flags itself through its `converged` entry.
* **Heterogeneous books** group by structure and array shape, with manual
  schedule padding where it pays; see
  [Heterogeneous Portfolios](../scaling/heterogeneous.md).
* **Devices.** Multi-device sharding is out of scope for now; see
  [Parallelism & Devices](../scaling/parallelism.md).

## 6. Open work

Tracked on the [roadmap](../roadmap.md#unified-calibration-pricing-and-risk):
migrating the per-model calibrators onto `CalibrationResult` and
`SolverSettings`; swaption-cube, optionlet and SVI levels as chain nodes;
per-trade Hull-White basket calibration; reaching every pricer through
`price`; derivative convergence of the numerical engines; the position /
portfolio layer.
