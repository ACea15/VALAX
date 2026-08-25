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

| # | Bug | One-line lesson |
|---|---|---|
| [1](#1-a-locally-correct-operator-can-be-globally-catastrophic-the-adi-mixed-term-boundary) | ADI mixed term wrong on the boundary rows | "Unit-correct" is not "correct" — bugs hide where the tidy interior test stops |
| [2](#2-a-passing-test-at-the-default-setting-is-not-evidence-time-reversed-boundary-sampling) | Backward steppers sampled boundaries at the mirrored time level | Sweep the numerical knobs; a default can hide an error by two orders of magnitude |
| [3](#3-degenerate-fixtures-hide-first-order-errors-midpoint-sampling-of-a-kinked-curve) | Midpoint-sampled `alpha(t)` stalled Hull-White's convergence | A flat-curve fixture is degenerate — it can make a first-order term vanish identically |
| [4](#4-measure-the-convergence-order-not-the-error-cashflow-date-snapping) | Coupon-date snapping swamped a second-order scheme | Small absolute error ≠ correct; and agreeing with an engine that shares your approximation is not validation |

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

---

## 2. A passing test at the default setting is not evidence: time-reversed boundary sampling

**Context.** The shared backward time-steppers, `solve_backward_1d`
(`valax/pricing/pde/schemes.py`) and `solve_backward_2d` (`schemes2d.py`). Both
feed Dirichlet boundary values into the theta-scheme as an affine source term,
sampling the user's `Boundary1D` callables at the time-remaining `tau` of the
known and solved levels.

Found while auditing the substrate *before* building the Hull-White PDE on it —
not by a failing test.

### Symptom

None. The full suite was green, including **1298 QuantLib comparisons**. The bug
was found by deliberately sweeping a knob nothing else swept: the domain
half-width `spot_range`.

ATM Black-Scholes European call, `n_spot = n_time = 400`, error vs the analytic
price:

| `spot_range` | before | after |
|---|---|---|
| 2.0 | +1.397e-01 | −2.67e-04 |
| 3.0 | +1.368e-02 | +9.73e-06 |
| **4.0 (default)** | **+4.46e-04** | +1.61e-05 |
| 6.0 | +3.44e-05 | +3.44e-05 |

### The misleading signals (what it was *not*)

1. **Everything passed.** The default `spot_range=4.0` put the error at 4.5e-4 —
   comfortably inside every tolerance in the repo, and *below* the tolerance of
   the QuantLib tests that were supposed to be the safety net.
2. **The error decayed with grid width**, which is exactly what correct
   truncation error does. At `spot_range=6` the buggy and fixed code agree to
   all printed digits. So any test that happened to use a wide domain would
   have seen nothing.
3. **The variable was *named* correctly.** The code read
   `tau_new = (n_time - m) * dt  # time-remaining at the known level`. The
   comment asserted the right semantics; only the arithmetic disagreed.

### Root cause

`lax.scan` marches `m = 0 … n_time-1` starting from the **terminal payoff**, so
the level entering step `m` has time-remaining `m*dt` (`m = 0` is expiry,
`tau = 0`) and the level being solved sits at `(m+1)*dt`. The code passed
`(n_time - m)*dt` and `(n_time - m - 1)*dt` — exactly mirrored. Every
time-dependent boundary was therefore evaluated with the wrong discount factor:
the far field was off by `K(1 - e^{-rT})`, and that error leaked inward.

Identical error in both steppers, because the 2-D one was written to mirror the
1-D one.

### The fix

```python
tau_known  = m * dt          # time-remaining at the known level
tau_solved = (m + 1) * dt    # time-remaining at the solved level
```

### The generalisable lesson

> **A tolerance that passes at the default configuration is not evidence. Sweep
> the numerical knobs — the ones that are *supposed* not to matter are precisely
> the ones that expose boundary and convergence bugs.**

Corollaries:

- **Parameters that should be irrelevant make the best probes.** A converged
  price must be insensitive to `spot_range`, `n_time`, mesh concentration, and
  domain placement. Each such invariance is a free, oracle-free test — and it
  fails loudly for boundary bugs that an absolute-tolerance comparison absorbs.
  This is now guarded by `TestPDEDomainWidthIndependence`.
- **An external oracle is only as strong as the configuration you run it in.**
  1298 QuantLib comparisons did not catch this, because they all ran the default
  grid. Breadth of instruments is not breadth of *numerics*.
- **Don't trust a comment over the arithmetic.** The comment was right and had
  been right since the code was written; it documented intent, and intent is not
  execution.

### The debugging playbook

1. **Sweep the knob that shouldn't matter.** The error fell roughly
   geometrically with domain width — the signature of a boundary-sourced error,
   not an interior one.
2. **Test the hypothesis without editing the source.** Monkeypatch the boundary
   factory to wrap its callables with `tau -> T - tau`. The `spot_range=2` error
   collapsed from 1.4e-1 to 2.7e-4 (500×) and the residual went *flat* in width.
   That confirmed the diagnosis before touching the stepper.
3. **Only then read the code**, knowing exactly what to look for.
4. **Write a decisive unit test, then verify it fails on the old code.** Both
   halves matter — the nine new guards were checked against a `git stash` of the
   fix.

### A sub-lesson from writing the test

The sharpest probe is a **spatially constant solution**: with terminal data
`V = C` and pure discounting, the exact solution has *no* spatial variation, and
the discrete operator annihilates a constant field exactly
(`lower + diag + upper = -r` on every row). So the field must stay constant in
`x` to machine precision — but only if the edge ghosts are taken at the
interior's time level. Mirrored `tau` makes the edge rows inconsistent and
spatial structure appears immediately.

The first draft of that test failed at 1.7e-5 on *correct* code. The reason is
worth recording: it fed the boundary the **exact** exponential `C e^{-r tau}`
while the interior evolves by the **discrete** theta-scheme recursion. The gap
was the scheme's own time-discretisation error, not a bug.

> **When asserting that a discrete scheme is exact, the reference must be
> discrete too.** Feeding an analytic boundary into a discretised interior
> builds a mismatch into the test and blunts it.

---

## 3. Degenerate fixtures hide first-order errors: midpoint sampling of a kinked curve

**Context.** The Hull-White PDE operator stack
(`valax/pricing/pde/coefficients.py`). The discount coefficient is
`r = x + alpha(t)`, where `alpha` is the exact-fit shift carrying the initial
curve. Following the local-volatility recipe's convention, `alpha` was first
sampled at each step's **midpoint in time**.

### Symptom

Hull-White is defined by fitting the initial curve *exactly*, so a zero-coupon
bond is the sharpest possible oracle: the PDE must reproduce `P^M(0,T)`. It did
— to ~4e-6 — and then **refused to improve**. Under time refinement the error
ratios decayed toward 1:

| `n_x = n_t` | error | ratio |
|---|---|---|
| 100 | 1.93e-05 | — |
| 200 | 7.51e-06 | 2.57× |
| 400 | 4.55e-06 | 1.65× |
| 800 | 3.82e-06 | **1.19×** |

A second-order scheme should show 4× per halving. It was converging to the
wrong answer.

### The misleading signals

1. **4e-6 looks converged.** On a discount factor of ~0.64 that is 6e-6
   relative — inside any sane tolerance. Only the *ratio* revealed it.
2. **Spatial refinement was innocent.** Holding `n_t = 800` and varying `n_x`
   changed nothing (3.91e-6 → 3.82e-6). Holding `n_x = 800` and varying `n_t`
   reproduced the stall exactly. So: a time-integration defect.
3. **Domain width was innocent** (4σ through 9σ all gave 4.5e-6), ruling out the
   new zero-curvature boundary treatment, which was the obvious suspect since it
   was the newest code.

### Root cause

Running the same sweep on a **flat** curve gave textbook 4.00× convergence. That
single contrast localised it: the defect is in how the *curve* enters, not in
the scheme.

`alpha(t) = f^M(0,t) + (sigma²/2a²)(1 - e^{-at})²`, and `f^M(0,t)` — the
instantaneous forward of a **log-linear** discount curve, VALAX's interpolation
and the market standard — is **piecewise constant with a jump at every pillar**.
The midpoint rule is second-order only for smooth integrands; on each step
straddling a pillar it is first-order. A flat curve has a *constant* forward, so
the error term vanishes identically and the fixture is blind to it.

### The fix

Integrate `alpha` exactly across each step instead of sampling it
(`hw_alpha_average`). Both halves are closed-form, and the market-forward half
telescopes into a discount ratio:

$$\int_{t_0}^{t_1} f^M(0,s)\,ds = \ln\frac{P^M(0,t_0)}{P^M(0,t_1)}$$

so each step discounts by precisely the market forward discount factor across
it, for any curve shape. Clean second order returned on flat, sloped and humped
curves alike (8.2e-7, 9.8e-7, 2.9e-6 at `n = 400`, each quartering thereafter).

### The generalisable lesson

> **A degenerate fixture can make an error term vanish identically. "Flat",
> "zero", "constant" and "symmetric" test inputs are exactly the ones that
> annihilate the terms you most need to see.**

Corollaries:

- **Every convergence test needs a non-degenerate input.** Flat curves, zero
  correlation, zero dividend, ATM strikes and uniform meshes are all fixtures
  that can silently zero out a coefficient. Keep them for readability, but never
  let them be the *only* case.
- **Interpolation choices are part of the model.** The bug is not in the PDE; it
  is in the interaction between the solver's quadrature and the curve object's
  interpolation. Anything sampling a curve-derived quantity inherits its
  smoothness — log-linear discount factors mean piecewise-constant forwards
  means kinks at pillars.
- **A model's exact-fit property is a free oracle with zero tolerance.** If a
  model claims to reproduce the initial curve exactly, any residual is a bug
  budget rather than noise — which is what made a 4e-6 discrepancy worth chasing
  at all.

---

## 4. Measure the convergence order, not the error: cashflow-date snapping

**Context.** The Hull-White PDE bond recipes
(`valax/pricing/pde/hull_white.py`). Coupon dates rarely coincide with a time
level, so — following the trinomial tree — they were **snapped to the nearest
one**.

### Symptom

With entry 3 fixed, a *zero-coupon* bond repriced the curve to ~1e-6, but a
five-year *coupon* bond was off by **2.2e-3** against the analytic curve price.
Same solver, same curve, same mesh.

### The misleading signals

1. **2.2e-3 on a price of 95.35 is 2.3e-5 relative** — it would pass essentially
   any relative tolerance in the repo, and did pass the first draft of the test
   suite.
2. **The trinomial tree agreed to ~1e-3.** This is the dangerous one: a second,
   independently-written engine corroborating the number *looks* like
   validation. It was not — the tree snaps coupon dates too, so both engines
   shared the same approximation and therefore the same error mode.
3. **The convergence order looked plausible in isolation.** The error did shrink
   with refinement; only by comparing against the *coupon-free* case was it
   obvious that a whole error term had appeared out of nowhere.

### Root cause

Snapping displaces each coupon by up to `dt/2` in time. Discounting a cashflow
over a wrong interval is an **`O(dt)` error** — and an `O(dt)` term inside an
otherwise second-order scheme dominates everything else long before it becomes
visible in absolute terms. At `n = 400` on a 5-year bond, `dt/2 ≈ 2.3` days;
across ten coupons the residuals accumulated to ~2e-3, roughly **1000× the
scheme's own error**.

### The fix

Don't snap — *correct*. Hull-White's affine structure gives the exact discount
factor from the snapped level `t_k` to the true payment date `t_c` at every
node:

$$P(t_k, t_c \mid x) = A(t_k,t_c)\,e^{-B(t_k,t_c)\,x}$$

A coupon is attached to the nearest level and scaled by that factor, which is
the correct analytic continuation for **either sign** of `t_c - t_k` (a payment
before the level becomes an accumulation factor > 1). Cashflow timing then
contributes no discretisation error at all, and second-order convergence is
restored: 2.12e-3 → 5.31e-4 → 1.33e-4 → 3.32e-5 → 8.31e-6, quartering cleanly.

Exercise dates are still snapped — a decision must happen *at* a level — but
that error is second order, because the exercise boundary is smooth in time.

### The generalisable lesson

> **Test the convergence *order*, not the error magnitude. A small absolute
> error is compatible with a completely wrong error term; only the ratio under
> refinement distinguishes "accurate" from "accidentally close".**

Corollaries:

- **Corroboration from an engine that shares your approximation is worthless.**
  Tree and PDE agreeing to 1e-3 meant only that both snapped dates. Independence
  has to be checked at the level of *assumptions*, not implementations. The
  discriminating test was the instrument that differed in exactly one respect —
  a bond with no intermediate cashflows.
- **When two engines disagree, ask which is wrong.** The reflex is to assume the
  new code is. Here, pricing an effectively option-free bond *on the tree*
  showed −2.2e-3 to +3.4e-5 error depending on step count, non-monotone because
  the count decides where dates land. The PDE was the more accurate engine, and
  the callable-bond test tolerances are consequently set by the **tree's**
  accuracy — documented in the test rather than left as an unexplained 5e-3.
- **Prefer an exact correction to a finer mesh.** Snapping error can only be
  refined away at `O(dt)`; the analytic factor removes it outright at no cost.
  Wherever a model has closed-form structure available, use it for the
  *contractual* details and reserve the mesh for the genuinely unknown part —
  here, the continuation value.
