# Numerical Pitfalls & Debugging Playbook

This page is a **register of hard-won numerical lessons** — discretisation bugs
that were subtle, expensive to find, and *generalisable*. It is a different
register from the other docs:

- It is **not** [`research-ideas.md`](../research-ideas.md) — those are future
  explorations. These are mistakes already made and fixed.
- It is **not** the [architecture specs](pde-design.md) — those describe how the
  code is *meant* to work. This page records how it *failed* to, and how we
  found out.

The value is twofold: the specific gotcha (so it is never reintroduced), and the
**debugging methodology** that localised it (so the next subtle discretisation
bug is cheaper to find). Each entry follows the same shape: symptom → misleading
signals → root cause → fix → the generalisable lesson.

---

## 1. A locally-correct operator can be globally catastrophic: the ADI mixed-term boundary

**Context.** The 2-D Heston ADI pricer (`valax/pricing/pde/`). Operator split
`L = A0 + A1 + A2`, where `A0` is the mixed second derivative
`rho*xi*v * d²V/dS dv`.

### Symptom

Every *unit* test was green, and the end-to-end pricer matched the analytic COS
oracle to `rel < 1.5e-3` **as long as `rho = 0`**. Turn correlation on and the
price was **~33% wrong** at the money — an enormous error for a term that, in
the true model, barely moves the ATM level at all.

### The misleading signals (what it was *not*)

Three observations each *individually* pointed away from the real cause:

1. **The `A0` operator had passing unit tests.** `apply_a0` was verified exact
   on `V = S·v` (`V_Sv = 1`) on a *non-uniform* grid. The operator's *interior*
   action was genuinely correct — so attention naturally went elsewhere.
2. **All three ADI schemes (Douglas, Craig–Sneyd, HV) gave the *same* wrong
   answer.** That rules out the scheme-specific correctors and points at a
   shared component — but which one?
3. **The error scaled linearly with `rho`.** Since `rho` only enters through
   `A0`'s coefficient, the error *was* the mixed term — but the mixed operator
   was "known good" from (1).

The trap: (1) proves the operator is correct *on the interior*. The bug lived on
the **boundary rows**, which the interior-only unit test deliberately excluded.

### Root cause

`apply_a0` computes the cross derivative with a **zero-ghost** fallback at the
domain edges: a value beyond the boundary is treated as `0`. On the interior
that never fires; on the four boundary rows it produces a **wrong, non-zero**
one-sided cross stencil. The spurious flux is worst at `v = v_max`, where the
coefficient `rho*xi*v_max` is largest, and it leaks into the interior through the
explicit mixed coupling — an `O(1)`, mesh-refining-away-only-slowly error.

The standard in't Hout & Foulon ADI treatment is that the mixed term is **zero on
the whole boundary**: the cross derivative is applied explicitly only and needs
no boundary data of its own, so suppressing it at the edge is both correct and
stable.

### The fix

One line in `build_operator_2d`: zero `c0` on all four edges after assembly.

```python
c0 = c0.at[0, :].set(0.0).at[-1, :].set(0.0).at[:, 0].set(0.0).at[:, -1].set(0.0)
```

Result: **33% → 0.03%** at the money, with clean second-order convergence
restored. (Regression-guarded by `test_mixed_term_is_zero_on_boundaries`.)

### The generalisable lesson

> **"Unit-correct" is not "correct". A discretisation can be exact on every
> interior node and still be catastrophically wrong because the bug lives at the
> boundary — the one place the tidy interior test omitted.**

Corollaries worth internalising:

- **Boundaries are where discretisation bugs hide.** The interior stencil is the
  part that is easy to reason about and easy to test; the boundary rows are
  where conventions, ghosts, and one-sided fallbacks live, and where "obviously
  fine" defaults (a zero ghost) are silently wrong.
- **A convenient default is a landmine.** `apply_a0`'s zero-ghost made the
  operator *runnable* everywhere, which is exactly why the bug was invisible. A
  default that "just works" removes the crash that would have pointed at the
  problem. Prefer defaults that are *correct* or that *fail loudly*, not merely
  ones that produce a number.
- **Interior-only unit tests must be paired with a global oracle.** The
  `_interior(...)` slicing that made the operator tests clean is the same
  slicing that hid the bug. Every operator that has non-trivial boundary
  behaviour needs at least one end-to-end test against an independent reference
  (here: the COS price), because that is the only test that exercises the edges
  in anger.

### The debugging playbook that localised it

This sequence turned "the pricer is 33% wrong" into "one line in
`build_operator_2d`" in a handful of runs. It generalises to any multi-term,
multi-scheme numerical solver:

1. **Toggle terms to zero, one at a time.** Setting `rho = 0` (killing `A0`)
   restored `rel < 1.5e-3`. That single experiment isolated the guilty *term*
   before touching any code.
2. **Check the error's functional form.** The error was *linear in `rho`* — i.e.
   proportional to the `A0` coefficient — confirming the mixed term was the
   whole story, not an interaction.
3. **Vary the two discretisations independently.** Refining `n_time` (160 → 800)
   changed *nothing*; refining the *spatial* grid *did* help, slowly. That split
   "time-stepping bug" from "spatial-operator bug" — it was spatial.
4. **Check scheme-independence.** Douglas, CS and HV agreed with each other and
   were all wrong → the fault is in a component they *share* (the operator /
   boundary), not in a scheme's corrector.
5. **Localise interior vs boundary with a targeted patch.** A throwaway
   experiment that zeroed `c0` only on the boundary rows dropped the error from
   33% to 0.2% — proving the boundary, not the interior, was the culprit before
   committing to a fix.

The meta-point: **each step halves the search space along a different axis**
(term, order, time-vs-space, scheme, interior-vs-boundary). None of them require
reading the code — they are black-box experiments on a differentiable,
composable pricer, which is exactly the kind of thing a JAX-native design makes
cheap to run.
