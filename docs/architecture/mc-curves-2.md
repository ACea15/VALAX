# MC-Curves-2 — Joint Multi-Curve Solver (Next Session Handoff)

> **Status:** Queued — ready to start.
> **Predecessor:** [MC-Curves-1](production.md#mc-curves-1-bootstrap-instrument-expansion)
> shipped (commits `9607392` → `d6c98f7`).
> **Estimated effort:** ~2 weeks.
> **Reading time before starting:** ~30 min.

This document is a **session-handoff note** — a self-contained briefing
that lets the next session resume MC-Curves-2 without re-deriving
context from scratch. It is not part of the architecture canon
(`production.md` §11 is); it is a working artefact that becomes
obsolete the moment MC-Curves-2 lands.

---

## 1. Where MC-Curves-1 leaves the codebase

The `BootstrapInstrument` protocol surface is complete. Every quote
type a real desk uses is representable as an `eqx.Module` carrying
`curves_touched: tuple[str, ...]` and a
`residual(graph, fixings, ref_date) -> Float[Array, ""]` method.
What's missing is the **solver** that consumes a list of these
instruments and finds the curve graph that zeroes their residuals
simultaneously.

### Public surface available to MC-Curves-2

```python
from valax.curves import (
    # Foundation
    CurveGraph,           # eqx.Module: dict[str, DiscountCurve]
    BootstrapInstrument,  # @runtime_checkable Protocol

    # Fixings
    FixingSeries,
    FixingHistory,
    empty_fixing_history,

    # Convexity adjustment plug-ins
    ConvexityAdjFn,
    no_convexity_adj,
    constant_convexity_adj,

    # 11 quote types
    DepositRate, FRA, SwapRate,
    OISSwapRate, MoneyMarketFuture,
    IborSwapRate, TenorBasisSwap,
    FXForward, FXSwap, CrossCurrencyBasisSwap,
    TurnInstrument,
)
```

### Internal helper to be aware of

```python
# valax/curves/instruments.py
def _floating_leg_pv(
    graph, fixings, start_date, payment_dates, fixing_dates,
    day_count, index_id,
    discount_curve_id, forward_curve_id, spread_addon,
) -> Float[Array, ""]: ...
```

Module-level helper used by every floating-leg instrument.
Not exported. The joint solver should *not* duplicate this —
each instrument's `residual` already calls it.

### Existing solver to be replaced (carefully)

```python
# valax/curves/multi_curve.py
class MultiCurveSet(eqx.Module): ...
def bootstrap_multi_curve(...) -> MultiCurveSet: ...
```

This is the predecessor: a single-currency, sequential dual-curve
bootstrapper (OIS first, then each tenor independently). MC-Curves-2's
deliverable is to replace its internals with a thin wrapper over the
new joint solver, while preserving the public function signature
for one minor release with a deprecation warning. **Do not delete it**
in MC-Curves-2 — the deprecation cycle matters.

### What's still mathematically locked

Without MC-Curves-2 you **cannot**:

- Bootstrap a tenor-basis curve graph (3M-and-6M on the same
  currency) — the basis swap touches two unknown forward curves
  simultaneously.
- Bootstrap a cross-currency curve graph — the CCBS touches up to
  four curves and FX spot.
- Compute the quote-sensitivity Jacobian via implicit-adjoint
  through a single Newton iteration.

After MC-Curves-2, all three are one function call.

---

## 2. Reading list — load these first

Order matters. Read top-down before any code.

| File | Why |
|---|---|
| `docs/architecture/production.md` §11.3, §11.4 | Data model and joint-solver interface design. The function signature `bootstrap_curve_graph(...)` is specified there. |
| `docs/architecture/production.md` §13 | Phased delivery plan; MC-Curves-2's scope and acceptance criteria. |
| `docs/architecture/production.md` §14 Q7–Q11 | Multi-curve open questions. **Q7 (instrument abstraction) is now resolved by MC-Curves-1.** Q8, Q10, Q11 are still open and gate parts of MC-Curves-2 — see §3 below. |
| `docs/theory.md` §3.7, §3.8 | No-arb identities (CIP, tenor basis, XCCY) and the joint residual system formalism. The math justifying why the solver must be joint, not sequential. |
| `valax/curves/multi_curve.py` | The current sequential dual-curve bootstrapper. Read end-to-end. The `_dual_curve_swap_residuals` function is the structural ancestor of what MC-Curves-2 generalises. |
| `valax/curves/bootstrap.py` | The current `bootstrap_simultaneous` Newton solve. The joint-solver implementation will reuse the `optimistix.Newton` + log-DF pattern with `ImplicitAdjoint`. |
| `valax/curves/instruments.py` | Spot-check the `_floating_leg_pv` helper and at least one instrument from each of the four classes (single-curve / dual-curve / three-curve / four-curve). |
| `tests/test_curves/test_instrument_residuals.py` | The existence proof that every residual is correct on a hand-built graph. MC-Curves-2's tests build on these patterns (build a *target* graph, construct instruments at the par values implied by it, verify the joint solver recovers it). |

---

## 3. Design questions to resolve before coding starts

From `production.md` §14, with MC-Curves-1's resolutions and the
remaining decisions for MC-Curves-2:

| # | Question | MC-Curves-1 status | MC-Curves-2 needs |
|---|---|---|---|
| Q7 | Instrument abstraction (Protocol vs. central dispatch) | ✅ **Resolved**: Protocol pattern, every quote type self-describes. The solver doesn't `isinstance` dispatch. | — |
| Q8 | Curve identifier scheme — lock now or evolve | 🟡 Still open: `<ccy>.<index>.<tenor>` is *used* throughout MC-Curves-1's tests but the docs don't formally lock it. | **Lock the scheme before MC-Curves-2 lands.** The joint solver returns `dict[str, DiscountCurve]` partitioned by id prefix in the workflow layer, and that partitioning logic depends on the scheme. |
| Q10 | Single-curve bootstrap retention | 🟡 Still open. | **Recommended: keep `bootstrap_sequential` as a thin user-facing API for pedagogy; reroute `bootstrap_simultaneous` to `bootstrap_curve_graph` internally as a one-instrument-per-pillar special case.** Do this in MC-Curves-2 so the deprecation surface is small. |
| Q11 | Default interpolation per curve type | 🟡 Still open. | **Recommended: log-linear DF for OIS/discount, monotone-convex for forward.** This is MC-Curves-3's territory; in MC-Curves-2 every curve still uses log-linear DF. |

Before starting MC-Curves-2 in the next session: pick a position on
Q8. The proposal is "freeze the `<ccy>.<index>.<tenor>` scheme as the
internal alphabet, with vendor-specific aliases handled at the
adapter layer". One sentence committed to the open-questions section
unblocks the rest.

---

## 4. Initial task breakdown (paste-ready for `eca__task plan`)

Six tasks, ordered by dependency. Each is independently testable.

```yaml
1. Define CurveSpec
   - File: valax/curves/graph.py (extend) — CurveSpec(eqx.Module).
     Static description of one curve: curve_id, currency, pillar_dates,
     interp ('log_linear_df' default), day_count.
   - Test: pytree round-trip, JIT compat.
   - Acceptance: from valax.curves.graph import CurveSpec succeeds;
     no impact on existing CurveGraph users.
   - Blocks: tasks 2, 3.

2. Implement bootstrap_curve_graph (the Newton joint solver)
   - File: valax/curves/bootstrap_graph.py (new).
   - Concatenate per-curve log-DFs into one state vector x. Each
     instrument contributes one residual; total system is square
     when len(instruments) == sum(spec.pillars).
   - Use optimistix.Newton with ImplicitAdjoint (matches existing
     bootstrap_simultaneous pattern).
   - Returns: (CurveGraph, sol.stats). Stats become part of
     CurveBuildDiagnostics in MC-Curves-3.
   - Tests:
     - Single-curve degenerate case: reproduces bootstrap_simultaneous
       to 1e-12.
     - Two-curve same-currency (USD OIS + USD 3M with futures + IBOR
       swaps): reproduces today's bootstrap_multi_curve.
     - Two-curve same-currency with TenorBasisSwap: third curve added.
     - EUR-USD four-curve graph closed by one CCBS: residuals < 1e-10
       on every input quote.
     - Gradients flow through bootstrap_curve_graph via
       ImplicitAdjoint (jax.grad of curve(date) w.r.t. one quote rate).
   - Blocks: tasks 3, 4, 5.

3. quote_jacobian convenience helper
   - File: valax/curves/bootstrap_graph.py (extend).
   - Wraps jax.jacobian over bootstrap_curve_graph with `by` switch
     ('log_df' / 'df' / 'zero_rate').
   - Tests: shape correctness; agreement with finite-difference for
     a small graph.

4. Migrate bootstrap_multi_curve to wrap the joint solver
   - File: valax/curves/multi_curve.py (modify).
   - Internals call bootstrap_curve_graph; public signature unchanged.
   - Add DeprecationWarning naming bootstrap_curve_graph as the
     replacement.
   - Tests: existing test_curves/test_bootstrap.py multi-curve tests
     pass unchanged.

5. Migrate bootstrap_simultaneous to wrap the joint solver (optional;
   defer to a later cleanup PR if scope creeps)
   - Same pattern as task 4 but for the single-curve simultaneous
     bootstrap.

6. Documentation update
   - docs/api/curves.md: add a Joint Solver section with
     bootstrap_curve_graph signature and CurveSpec.
   - docs/guide/curves.md: replace §5 "Multi-curve bootstrap" with a
     joint-solver walkthrough; preserve the existing
     bootstrap_multi_curve example as a deprecated path.
   - mkdocs build --strict.
```

Total: ~6 tasks, ~2 weeks of focused work. Tasks 4 and 5 are the
deprecation/migration pieces; the structural payoff is in tasks 1–3.

---

## 5. Modules MC-Curves-2 will touch

```
NEW
  valax/curves/bootstrap_graph.py    # the joint solver
  tests/test_curves/test_bootstrap_graph.py

EXTENDED
  valax/curves/graph.py              # add CurveSpec
  valax/curves/__init__.py           # export bootstrap_curve_graph,
                                       quote_jacobian, CurveSpec
  docs/api/curves.md                 # Joint Solver section
  docs/guide/curves.md               # §5 rewrite

MIGRATED (signature preserved, internals replaced)
  valax/curves/multi_curve.py        # wrap joint solver,
                                       emit DeprecationWarning
```

`valax/curves/instruments.py` does **not** change. Its instruments
are already protocol-conformant; the joint solver just iterates over
a list of them and asks for residuals.

---

## 6. Workstream-level acceptance criteria

MC-Curves-2 is "done" when:

1. ✅ `bootstrap_curve_graph(specs, instruments, *, fixings=None, ...)`
   exists, returns `(CurveGraph, stats)`, and zeroes every input
   instrument's residual to ≤ 1e-10 on a hand-built target graph.
2. ✅ A USD-only two-curve build (OIS + 3M tenor) using the new
   solver agrees with `bootstrap_multi_curve` to ≤ 1e-12.
3. ✅ A EUR-USD four-curve graph closed by one CCBS calibrates with
   residuals ≤ 1e-10 on every quote.
4. ✅ `jax.grad` through `bootstrap_curve_graph` produces finite,
   non-zero gradients (verified by `optimistix.ImplicitAdjoint`).
5. ✅ All existing `tests/test_curves/test_bootstrap.py` and
   `bootstrap_multi_curve` tests pass unchanged.
6. ✅ A deprecation warning is emitted by `bootstrap_multi_curve`
   pointing to `bootstrap_curve_graph`.
7. ✅ `mkdocs build --strict` is green; the user guide demonstrates
   a multi-currency build worked end-to-end.

---

## 7. Starter prompt for the next session

Paste the following into the start of the next session — it loads
the necessary context with one round-trip:

> We're picking up MC-Curves-2 (the joint multi-curve Newton solver).
>
> MC-Curves-1 was shipped in commits 9607392 → d6c98f7. Read
> `docs/architecture/mc-curves-2.md` first — that's the handoff
> note with the full state snapshot, reading list, open design
> questions, and the initial six-task plan ready to drop into
> `eca__task plan`. Then resolve open question Q8 (curve identifier
> scheme freeze) before any code lands. After that, plan and start
> Task #1 (`CurveSpec` definition).

That's it. The handoff doc above carries everything else.
