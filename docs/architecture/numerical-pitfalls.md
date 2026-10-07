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
| [5](#5-a-value-correct-guard-with-no-derivative-jnpwhere-on-a-removable-singularity) | `jnp.where` pinned Hagan's `z/x(z)` to a constant at the money | A guard can be exact in value and empty in derivative; and two adjoints agreeing proves nothing if they share the broken primitive |

## Runnable reproductions

This page is the prose register. Several of these failures also live as
**self-contained, runnable scripts** you can open and perturb — flip a solver,
tighten a tolerance, re-break the fix — in
[`examples/pitfalls/`](https://github.com/ACea15/VALAX/tree/main/examples/pitfalls).
Each script carries a SYMPTOM / ROOT CAUSE / KNOBS header and a **WAY OUT**
section, and is wrapped so it runs end-to-end even where the underlying call
diverges: the point is to *observe* the failure, not to avoid it.

| Script | Register entry |
|---|---|
| [`01_sabr_vol_convention_mismatch.py`](https://github.com/ACea15/VALAX/blob/main/examples/pitfalls/01_sabr_vol_convention_mismatch.py) | — (calibration-convention mismatch; no PDE counterpart here) |
| [`02_where_guard_kills_the_derivative.py`](https://github.com/ACea15/VALAX/blob/main/examples/pitfalls/02_where_guard_kills_the_derivative.py) | [entry 5](#5-a-value-correct-guard-with-no-derivative-jnpwhere-on-a-removable-singularity) |

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

### Where this lesson is now enforced

Both halves of it are load-bearing in the audit T-suite
([conformance map](aad-audit-conformance.md)), which is where the lesson stops
being advice and becomes a gate:

| Gate | Which half | How |
|---|---|---|
| **T-8** (smoothness scan) | Convergence *order* | "No steps" is not assertable at any tolerance: every smooth function has non-zero second differences, and a bound on near-linearity would fail on the price's genuine curvature across the scan. The discriminator is the refinement ratio: halving the grid must shrink `max|Δ²price|` fourfold and `max|Δdelta|` twofold, which a single discontinuity pins near one. Observed 3.99 and 2.00. Its anti-vacuity companion — asserting the scan actually moves the price — is entry 3's lesson |
| **T-9** (basket repriced by the PDE) | Convergence *order* | A single residual cannot say whether 1.5e-4 is discretisation error or calibration misfit. Doubling the grid drops it to 3.7e-5 — ratio 4 — which attributes it to the mesh |
| **T-6** (put–call parity) | Shared approximation | VALAX derives the floorlet and the receiver *from* parity, so the identity is exact by construction and asserting it tests nothing. Each leg is checked against an independently written closed form instead, with the put as $K\Phi(-d_2) - F\Phi(-d_1)$ rather than $\text{call} - (F-K)$ |

---

## 5. A value-correct guard with no derivative: `jnp.where` on a removable singularity

**Context.** Hagan's SABR implied-vol expansions
(`valax/pricing/analytic/sabr.py`). Both the lognormal and normal forms contain
the ratio

$$
\frac{z}{x(z)}, \qquad
x(z) = \log\frac{\sqrt{1 - 2\rho z + z^2} + z - \rho}{1 - \rho},
\qquad z \propto \log(F/K),
$$

which is $0/0$ at the money. It was guarded with the pattern every JAX codebase
reaches for:

```python
is_small = jnp.abs(z) < 1e-7
safe_z = jnp.where(is_small, 1.0, z)
safe_x = jnp.where(is_small, 1.0, x_z)
z_over_x = jnp.where(is_small, 1.0, safe_z / safe_x)   # limit is 1
```

Found while building the volatility layer's adjoint — not by a failing test.

### Symptom

$\partial\sigma/\partial F$ at the at-the-money strike was wrong by **18%** in
the lognormal branch (AD $-10.910$ against a true $-9.222$) and **identically
zero** in the normal branch (AD $0.000$ against a true $0.051$). Every *price*
was correct to machine precision, at every strike, in both conventions.

### The misleading signals (what it was *not*)

This one produced three false exonerations in a row, and the order matters.

1. **A full-rebuild finite difference disagreed, and was dismissed.** The FD was
   stable across four decades of $h$ — the textbook evidence that it has
   converged — and *that stability was read as the FD being suspicious* rather
   than authoritative, because two adjoints disagreed with it.
2. **A second, "independent" adjoint agreed with the first, bit-for-bit.**
   `jax.grad` straight through `calibrate_sabr`, i.e. optimistix's own
   `ImplicitAdjoint` on the Levenberg-Marquardt problem in its *unconstrained*
   parametrisation — different code, different variables, different formulation
   of the implicit function theorem. It shared `sabr_implied_vol` underneath, so
   it shared the defect. Two votes that were really one.
3. **The least-squares Hessian was genuinely ill-conditioned** ($\kappa \approx
   2\times 10^6$), which supplied a *plausible mechanism* for the FD to be
   untrustworthy: a flat valley, a solver stopping short of the optimum, a
   difference quotient measuring the stopping point rather than the minimum.
   The mechanism was real in general and false here — Newton on the first-order
   condition to $10^{-14}$ reproduced the FD exactly. A believable story for why
   the inconvenient evidence is wrong is the most expensive kind of misleading
   signal.

### Root cause

A constant has zero derivative. `jnp.where(is_small, 1.0, ...)` returns the
right *value* at $z = 0$ and propagates **nothing**, so AD discarded the entire
$z/x$ contribution. The true limiting slope is
$\mathrm{d}(z/x)/\mathrm{d}z \to -\rho/2 \neq 0$.

Why only $\partial\sigma/\partial F$, and only at the money: $z$ carries a
factor $\log(F/K)$, which vanishes at $F = K$. So every *parameter* derivative
of $z$ (in $\alpha$, $\rho$, $\nu$) is multiplied by zero and genuinely
vanishes — those derivatives were always correct. $\partial z/\partial F$ is the
one that survives, because $\partial \log(F/K)/\partial F = 1/F$. One entry of
one row of the Jacobian, at one strike.

### The fix

Hand over to the **series**, not to a constant, so both branches are smooth
functions that agree to $O(z^3)$ *and* carry the right slope:

$$
\frac{z}{x(z)} = 1 - \frac{\rho}{2} z
  + \left(\frac{1}{6} - \frac{\rho^2}{4}\right) z^2 + O(z^3).
$$

Extracted as `_z_over_x`, shared by both expansions. The outer double-`where`
is retained — that part was doing real work, keeping NaN gradients from leaking
out of the inactive branch.

The crossover moved from `1e-7` to `_ATM_Z_TOL = 1e-4`, chosen by balancing the
two error sources rather than by picking a small-looking number: series
truncation grows like $z^3$, the closed form's cancellation like $\epsilon/z$,
and $z^3 \sim \epsilon/z$ gives $z \sim \epsilon^{1/4} \sim 10^{-4}$, where both
branches are accurate to ~$10^{-12}$. The original `1e-7` was *too small* — it
kept the ill-conditioned closed form in play across a decade where the series
was already the better approximation.

At-the-money *values* are bit-identical before and after, because the series is
exactly $1$ at $z = 0$.

### The generalisable lesson

> **A guard can be exact in value and empty in derivative. And corroboration
> between two implementations is worth nothing when they share the primitive
> that is broken — independence has to hold at the level of assumptions, not
> code.**

Corollaries:

- **Never `jnp.where` a removable singularity onto a constant.** Use the limit's
  series. The constant is right at exactly one point and wrong in every
  derivative; the series is right in a neighbourhood and differentiable.
- **Rank disagreeing evidence by independence, not by vote count.** A finite
  difference shares *no* code with an adjoint, which is precisely why it is
  worth keeping around despite being the noisier instrument. This is entry 4's
  lesson — "corroboration from an engine that shares your approximation is
  worthless" — in a second costume, and it was walked into anyway.
- **Be suspicious of a satisfying explanation for inconvenient data.** The
  ill-conditioning story was coherent, quantitative and wrong. The cheap test
  that would have killed it immediately — solve the first-order condition with
  Newton to $10^{-14}$ and re-difference — took one run.
- **Value tests cannot see derivative bugs.** A differentiable library needs
  derivative tests *at* its singular points, not merely near them. Guarded by
  `tests/test_models/test_sabr.py::TestATMDerivatives`, which checks AD against
  FD at the money and asserts the ATM forward derivative is non-zero.

### The debugging playbook

1. **Reduce to the smallest quantity that shows the discrepancy.** From "the
   calibration adjoint is 80% off" down to "$\partial\sigma/\partial F$ of a
   single `sabr_implied_vol` call", by removing the optimiser entirely.
2. **Sweep the strike.** The error appeared only within $|K - F| < 10^{-7}$ —
   i.e. exactly where `is_small` fires. That localised it to the guard without
   reading the expansion.
3. **Check *which* derivatives are affected.** $\alpha$, $\rho$, $\nu$ were
   fine; only $F$ was wrong. That asymmetry is only explicable if the lost term
   carries a $\log(F/K)$ factor, which named the culprit outright.
4. **Verify the suspected mechanism in isolation.** `jax.grad` of the guard
   alone at $z = 0$: `0.0` against the analytic $-\rho/2$. Two lines, no SABR.
5. **Confirm the new branch against the old one where they overlap**, so the fix
   cannot introduce a jump or a kink at the crossover.

A runnable reproduction, with the broken and fixed guards side by side and the
knobs to re-break it, is
[`examples/pitfalls/02_where_guard_kills_the_derivative.py`](https://github.com/ACea15/VALAX/blob/main/examples/pitfalls/02_where_guard_kills_the_derivative.py).
