# Research Ideas

This page is a **parking lot for exploratory ideas** — directions that could be
valuable but are **not committed and carry no priority right now**. It exists so
that a promising line of thinking is written down and can be evaluated later,
rather than lost or prematurely promoted into planned work.

It is deliberately a different register from the other forward-looking docs:

- It is **not** [`roadmap.md`](roadmap.md) — the roadmap is time-ordered,
  prioritized work we intend to do. Nothing here is a commitment.
- It is **not** [`vision.md`](vision.md) — the vision is the untimed
  *destination*, the kind of thing VALAX is becoming. Ideas here are concrete
  technical explorations that may or may not serve that destination.
- It is **not** [`design-rationale.md`](design-rationale.md) or the
  [architecture](architecture/overview.md) specs — those justify decisions we
  have already made.

**Promotion path.** When an idea earns a priority — because a use case demands
it, a dependency lands, or a quick experiment proves it out — it graduates into
a [`roadmap.md`](roadmap.md) tier or session backlog, and its entry here is
updated with a link to that committed work. Until then, entries stay here with
`Status: unprioritized`.

**Adding an idea.** Keep it cheap. Copy the template below, fill in what you
know, and leave the rest blank. An idea does not need to be fully worked out to
be worth recording — half of the value is capturing *why* it was interesting and
*what the open questions are*.

```markdown
### <short title>

**Context / motivation.** Why this came up; what problem it might solve.

**The idea.** The proposed approach in a few sentences.

**Where it helps.** The specific situations in which it would pay off.

**Open questions & risks.** What we do not know; what could kill it.

**Rough cost / complexity.** Order-of-magnitude effort and the hard parts.

**Status.** unprioritized | experimenting | promoted → <roadmap link>
```

---

## Numerical methods

### Exact-$y$ two-factor Cheyette PDE (the 5-D accumulator problem)

**Context / motivation.** The two-factor Cheyette PDE
(`valax/pricing/pde/cheyette.py`) ships as **frozen-$y$** (production, 2-D) plus
an **exact-$y$ one-factor** validation solver (2-D). The natural next question is
the *exact-$y$ two-factor* PDE, whose state
$(x_1, x_2, y_{11}, y_{22}, y_{12})$ is **5-D**. It would remove the frozen-$y$
approximation for the decorrelation-capable two-factor model and give a
zero-Monte-Carlo-noise benchmark for the frozen-$y$ bias in 2F (as the exact-$y$
1F solver already does for 1F).

**The idea.** Exploit the structure: the three $y$-axes are **advection-only**
(pure-drift accumulators, no Brownian term), so the PDE is degenerate parabolic —
2 diffusive dimensions $(x_1, x_2)$ plus 3 hyperbolic ones. That means the
$y$-axes need far less refinement than a diffusive axis and admit upwinding or
semi-Lagrangian / method-of-characteristics stepping. A first spike could freeze
only the small cross-accumulator $y_{12}$ and carry $(y_{11}, y_{22})$ exactly, a
**4-D** compromise, before attempting the full 5-D solve.

**Where it helps.** A deterministic (noise-free) measurement of the frozen-$y$
bias for two-factor swaptions / Bermudans, and a research-grade oracle for
decorrelation products where the skew and the two-factor structure interact.

**Open questions & risks.** The binding constraint is **not** the forward FLOPs
but reverse-mode autodiff through the time-marching `lax.scan`: a 5-D grid at
$n \approx 48$/axis is $\sim 2.5\times10^8$ nodes ($\sim$2 GB per `float64`
field), and gradient checkpointing stores several such copies — borderline on a
single accelerator, and prohibitive inside a calibration loop. So this is a
**single-shot, likely non-differentiable benchmark**, not a calibration engine;
2-factor calibration stays on the frozen-$y$ engine. Upwinding on the degenerate
$y$-axes (no diffusive smoothing) is the main scheme-design risk.

**Rough cost / complexity.** Medium–large: a bespoke degenerate-parabolic
stepper (the existing ADI substrate assumes diffusion on every axis), careful
$y$-domain sizing, and memory engineering. Deliberately excluded from the shipped
work; the frozen-$y$ 2F engine plus the exact-$y$ 1F oracle cover the practical
needs.

**Status.** unprioritized.

### High-order & finite-element methods for multi-dimensional PDE pricing

**Context / motivation.** VALAX's PDE subsystem is a 1-D log-spot theta-scheme
(Crank–Nicolson / implicit, with Rannacher start-up) over a uniform grid, solved
tridiagonally via `lineax`
(see [PDE Solvers — Design](architecture/pde-design.md) and the
[PDE guide](guide/pde.md)). Meanwhile the library already carries the *models*
whose natural PDE formulation is **2–3 dimensional** — Heston (`S, v`),
Heston–Hull–White (`S, v, r`), SLV + stochastic rates, and three-asset baskets
(`S₁, S₂, S₃`) — but these are reachable only via Monte Carlo. The open question
is whether a **higher-dimensional, higher-order** PDE engine would be worth
building, and whether **finite-element methods (FEM)** are the right vehicle for
it (as opposed to finite differences, which we already use).

**The idea.** Build a multi-dimensional PDE pricer and use it to test the
hypothesis that *higher-order* spatial discretization buys disproportionately
more accuracy in higher dimensions. The prize is a deterministic, low-variance,
autodiff-friendly Greeks engine for early-exercise and barrier features on
multi-factor models — precisely where Monte Carlo struggles (American exercise,
sharp gammas near barriers).

**Where it helps.**

- **Accuracy-per-degree-of-freedom compounds with dimension.** Any grid/mesh
  method has `~Nᵈ` unknowns. If a 2nd-order method needs `N≈200` per axis but a
  high-order/spectral method reaches the same accuracy at `N≈40`, that is
  `200³ ≈ 8·10⁶` vs `40³ ≈ 6.4·10⁴` unknowns — a ~125× reduction in 3D. This
  exponential leverage is the strongest argument for high order in ≥3D.
- **FEM handles correlation cross-terms naturally.** The mixed `∂²V/∂S∂v` terms
  from correlation are a well-known source of instability and positivity loss in
  finite differences (they are why ADI schemes such as Craig–Sneyd exist). In
  FEM's variational (weak) form these terms fall out cleanly.
- **FEM enables adaptive local refinement and conforming boundaries.** DOFs can
  be concentrated at strikes/barriers/kinks (in 3D these are 2-D manifolds where
  uniform refinement is exponentially wasteful), and multi-asset barrier
  knock-out regions can be meshed exactly rather than staircased.

**Open questions & risks.**

- **FEM ≠ high order — they are orthogonal choices.** The "higher order for
  accuracy" goal may be better served by **high-order structured / spectral
  methods** than by FEM proper. Bundling the two is a design trap to avoid.
- **The JAX constraint is likely the binding one, not the math.** JAX/JIT wants
  static shapes and dense, vectorizable arrays. Dense *structured* 3-D grids
  (`10⁶–10⁷` nodes) are a sweet spot for JAX-on-GPU, but **unstructured and
  adaptive** FEM (variable connectivity, gather/scatter assembly, data-dependent
  shapes) fights JAX at every step, and the ecosystem's sparse-LA and
  preconditioner support is immature. FEM's full mass matrix also loses the
  **ADI dimension-splitting** that keeps each implicit step a cheap sequence of
  1-D tridiagonal solves.
- **Nothing here beats the curse of dimensionality.** FEM/high-order improve
  *constants and convergence order*, not the `Nᵈ` asymptotic. For ≥4–5 factors
  the honest tools remain Monte Carlo and, potentially, deep-BSDE / neural
  methods. PDE's justification in 3D is *deterministic sharp Greeks and
  early-exercise*, not raw dimensionality.
- **Greeks through the solve change shape.** The current design differentiates
  straight through a tridiagonal solve; a sparse iterative solve needs implicit
  differentiation (supported by `lineax`, but a different code path) and careful
  handling to keep gamma well-defined.

**Rough cost / complexity.** Large, and best staged with decision gates so it
can be abandoned cheaply:

1. **2-D Heston ADI baseline** — the real unlock step: build the
   cross-derivative operator, operator-splitting time-march, and non-tridiagonal
   boundaries at a scale that validates against the analytic Heston COS pricer.
   Activates the already-scaffolded `Scheme.DOUGLAS / CRAIG_SNEYD / HV` and
   `PDEConfig2D`. *Gate: if Craig–Sneyd Heston will not match analytic + MC, 3-D
   is not worth attempting.*
2. **3-D ADI-FD baseline** (`S, v, r`, e.g. Hundsdorfer–Verwer) — the pragmatic
   3-D pricer and the reference against which any higher-order or FEM method must
   prove itself. Reuses the tridiagonal solver per sub-step.
3. **High-order experiment** — an error-vs-total-DOF study, 2nd-order ADI-FD vs a
   high-order structured / spectral-element method on the *same* problem. *Gate:
   this single plot decides whether the whole direction is worth pursuing.*
4. **FEM proper (unstructured/adaptive)** — only if step 3 shows high order wins
   *and* a requirement appears that structured grids cannot meet (conforming
   irregular boundaries for multi-asset barriers, or adaptive refinement at
   kinks), accepting the JAX sparse-solver R&D cost.

**Status.** unprioritized. Related committed context: the multivariate PDE
design already anticipates 2-D/3-D schemes in
[PDE Solvers — Design](architecture/pde-design.md); if this idea is promoted it
would extend that specification and land as roadmap tiers.

### Solution-dependent (nonlinear) PDE operators via a per-step rebuild hook

**Context / motivation.** This surfaced while building the Dupire local-vol PDE
recipe. That recipe needs *time-dependent* operator coefficients, and the chosen
implementation precomputes a **stack** of operators (one per time level) before
the `lax.scan` backward sweep — because the local-vol diffusion `σ²_loc(x, τ)`
depends only on space and time, both known ahead of the scan. The alternative we
rejected for that task was to pass an `operator_fn(τ) -> Operator1D` callable
that rebuilds the operator *inside* each step. Rebuilding inside the step is
strictly more expressive, and that extra expressiveness is exactly what a whole
class of **nonlinear pricing PDEs** requires. Worth capturing before the seam
gets designed away.

**The idea.** Give `solve_backward_1d` (and the eventual 2-D/3-D steppers) an
optional per-step operator-construction hook of the form
`operator_fn(τ, v) -> Operator1D`, invoked inside the scan step where the current
solution vector `v` is in scope. When the hook depends only on `τ` it reduces to
the precomputed-stack case (linear, time-dependent); when it depends on `v` it
unlocks **solution-dependent, nonlinear** operators that a precomputed stack
cannot represent, because the coefficients do not exist until partway through the
backward sweep.

**Where it helps.** Concretely, nonlinear models where diffusion/drift depend on
the evolving solution or its derivatives:

- **Uncertain-volatility / Black–Scholes–Barenblatt** — diffusion switches
  between `σ_min` and `σ_max` on the sign of Gamma (`v_xx`), known only mid-scan.
- **Transaction-cost models** (Leland, Hoggard–Whalley–Wilmott) — effective
  volatility is adjusted by a term in the sign/size of Gamma.
- **Penalty / free-boundary formulations** — the operator picks up a penalty
  band that depends on `v` at each step (a cleaner seam than the current
  `solver_fn` swap used by the American penalty method).

**Open questions & risks.**

- **Differentiability through a `v`-dependent operator.** Autodiff for Greeks
  must stay well-behaved when the operator itself is a function of the solution;
  the nonlinear dependence changes the adjoint and needs care to keep gamma
  meaningful (cf. the same concern raised for iterative/sparse solves in the
  high-order-PDE idea above).
- **Nonlinear PDEs are not a single linear solve per step.** BSB and
  transaction-cost equations may need an inner iteration (policy/Newton) per
  time step; the hook is necessary but not sufficient.
- **Cost inside the traced loop.** Rebuilding the stencil every step (vs the
  precomputed stack) re-traces work per step; acceptable for genuinely nonlinear
  problems, wasteful for the linear time-dependent case, so the hook must remain
  *optional* and coexist with the stack path.
- **Convergence/stability of theta-schemes on nonlinear operators** is less
  clean than the linear theory; Rannacher start-up interactions are unexplored.

**Rough cost / complexity.** Small-to-moderate for the *hook* itself (extend the
stepper signature to accept a callable alongside the existing single-operator and
stacked-operator paths, thread it through the scan). The real cost is per-model:
each nonlinear PDE brings its own inner-iteration scheme, stability analysis, and
validation reference. Best staged as (1) land the optional hook when the local-vol
stack work goes in, keeping it dormant; (2) prove it out on uncertain-volatility
(the simplest nonlinearity, with a known analytic bound to validate against);
(3) generalize to transaction-cost and penalty formulations.

**Status.** unprioritized. Emerged from the Dupire local-vol PDE recipe, whose
precomputed-operator-stack stepper is the natural place to add the optional
callable seam later.

### Discrete (Andreasen-Huge) local-vol calibration for exact FD surface repricing

**Context / motivation.** The Dupire local-vol PDE recipe feeds the *continuous*
Dupire local volatility into a *discrete* backward finite-difference scheme. This
reprices the input vanilla surface exactly only when the local vol is constant in
log-spot (flat or pure-term-structure surfaces). For **skewed** surfaces a
grid-*independent*, skew-proportional repricing gap remains even in the mesh
limit — the continuous Dupire formula inverts the continuous forward
(Fokker-Planck) equation, which is not the adjoint of the discrete backward
operator. This was characterised while building the LV PDE recipe: the gap
converges (does not shrink as `O(dx^2)`), an independent hand-rolled FD solver
reproduces it, and **QuantLib's `FdBlackScholesVanillaEngine` exhibits a gap of
the same magnitude** — so it is inherent to FD-Dupire, not a VALAX bug. Monte
Carlo (sampling the true continuous SDE) remains the faithful surface-repricer.

**The idea.** Calibrate the local volatility directly to the **discrete** forward
operator rather than via the continuous Dupire formula, following Andreasen &
Huge (2011), *Volatility Interpolation*. One implicit finite-difference step of
the forward (Dupire) equation is used as the pricing/interpolation operator, and
the per-slice local variance is solved so that the discrete operator reprices the
input option quotes exactly. The resulting local-vol field is, by construction,
the one the backward FD scheme needs to reprice the surface to machine precision.

**Where it helps.**

- **Exact FD repricing of skewed surfaces** — closes the wing gap that both VALAX
  and QuantLib currently exhibit, so the LV PDE (and PDE-priced exotics on the
  same surface) are consistent with the calibrating vanillas.
- **Arbitrage-free interpolation for free** — the Andreasen-Huge operator yields a
  smooth, calendar/butterfly arbitrage-free surface, side-stepping the wing
  pathologies (σ_loc blow-ups, QL's "decreasing variance" refusals) seen when
  differentiating a raw SVI/bicubic surface for Dupire.
- **A single discrete object serving calibration, PDE, and Greeks** — the same
  operator underlies fitting and pricing, and it is `lineax`-friendly (each step
  is a tridiagonal solve) and autodiff-friendly.

**Open questions & risks.**

- **Calibration inside a JAX graph.** The per-slice least-squares fit of local
  variance to quotes must be `optimistix`-based and differentiable w.r.t. market
  quotes if surface-parameter Greeks are wanted; nested solves raise memory and
  implicit-diff considerations.
- **1-D forward operator vs the backward pricer.** The AH operator is a specific
  implicit forward step; making the *backward* European/exotic pricer share the
  exact same discretisation (grid, θ, boundaries) is required for the exactness to
  hold and constrains the recipe layer.
- **Scope of exactness.** AH reprices the *calibrating* strikes/expiries exactly;
  off-grid strikes and path-dependent exotics still carry interpolation/scheme
  error — the guarantee is narrower than "reprices everything".

**Rough cost / complexity.** Moderate. The forward AH operator + per-slice
calibration is a self-contained module; wiring it so the existing LV PDE recipe
consumes an AH-calibrated `LeverageGrid`/local-vol field (instead of continuous
Dupire) is a bounded change. Validation is clean: assert exact reprice of the
calibrating quotes, then re-run the skew cross-check in
`tests/test_pde/test_local_vol_pde.py` with a tight (not ATM-only) gate.

**Status.** unprioritized. Directly motivated by the FD-Dupire skew gap
documented in the LV PDE recipe (`valax/pricing/pde/recipes.py`) and its
QuantLib cross-check (`tests/test_quantlib_comparison/test_local_vol_pde_ql.py`).

---

## Rates

### Recovering textbook negative effective convexity for callable bonds

**Context / motivation.** The OAS work (`valax/risk/oas.py`) set out to
reproduce the classic callable-bond signature — **negative** effective convexity
near the call boundary — as an autodiff second derivative of price w.r.t. a
parallel OAS shift. It did not appear. In the exact-fit Hull-White framework the
capped redemption is still discounted *convexly*, so effective convexity to a
parallel curve spread floors at the small positive value of the near-term call
stub rather than turning negative. The honest, reproducible signature is
convexity *compression* (bullet ~70 → callable ~0) plus duration compression,
and the tests assert that compression rather than a sign flip. The textbook
"negative convexity" is a price-vs-yield-to-call statement, not this
term-structure OAS derivative — but whether *some* tractable, differentiable
framework recovers the sign flip as a first-class autodiff Greek is genuinely
open.

**The idea.** Investigate which risk-metric definition and/or model, if any,
makes effective convexity go strictly negative as a computed derivative (not a
yield-axis picture). Candidate axes to sweep:

- **Shift convention.** Redefine the OAS bump against a *compounded* /
  lognormal-quoted spread, or against yield rather than a continuous parallel
  zero shift, and differentiate through that map.
- **Curve-move shape.** Replace the parallel shift with a level/slope shock (or a
  small PCA-shaped move) and look at convexity to the level factor.
- **Model richness.** Repeat under G2++ (now delivered) so the exercise region
  is two-factor — decorrelation may change the sign story that single-factor HW
  forces.

**Where it helps.** Would let a callable/MBS-style book display the risk metric
practitioners expect (negative convexity) as an exact, GPU-friendly autodiff
Greek, and would pin down *precisely* which convention the "negative convexity"
folklore actually refers to — a clarification with teaching and validation value
regardless of outcome.

**Open questions & risks.** It may simply be **true** that no continuous
parallel-spread definition yields negative convexity in an exact-fit Gaussian
short-rate model — in which case the deliverable is a crisp *negative result*
("compression, not sign flip, is the correct term-structure signature") rather
than a new metric. Differentiating through a changed shift convention (compounded
/ yield) reintroduces the compounding-convention trap already documented in the
OAS work. Second derivatives through the exercise projection need a gamma sanity
check against central differences (test-device only, per `AGENTS.md`).

**Rough cost / complexity.** Small-to-moderate and mostly *analysis*: the pricer,
OAS solver, and autodiff plumbing already exist, so this is a definition sweep
plus a validation write-up, not a build. Highest-value output is likely a
documented finding (extending gap §2.3b of the rates guide) rather than new code.

**Status.** unprioritized. Emerged from the OAS deliverable
(`valax/risk/oas.py`, `tests/test_risk/test_oas.py`); see gap §2.3b of the rates
session guide for the reproduced compression-not-sign-flip result.

### Stochastic tenor-basis with a low-dimensional shared basis factor

**Context / motivation.** VALAX evolves a single discount curve (HW-1F, or two
factors under G2++), and every tenor forward curve is that curve's implied
forward plus a **deterministic** spread. The tenor bases therefore cannot move on
their own — they are frozen functions of time layered on the rate dynamics. Real
basis (1M/3M/6M/OIS) spreads *do* move, and decorrelate from the level of rates;
products explicitly on the basis (tenor-basis swaps, basis options) have no
volatility in the current framework.

**The idea.** Add stochastic dynamics to the basis while keeping the factor count
low. Rather than an independent driver per tenor (an over-parameterized,
poorly-identified `n`-factors-for-`n`-tenors model), introduce **one shared basis
factor** (optionally a second for basis-slope) with tenor-dependent loadings:
`s_τ(t) = ℓ_τ · z(t)`, `z` mean-reverting and correlated with the rate factors.
For a G2++ core this is a 2+1 factor model whose factor count is *independent of
the number of tenors carried*. Kept affine/Gaussian, the multi-curve ZCB and
swaption tractability that G2++ already has should largely survive.

**Where it helps.** Prices and risks basis-sensitive instruments that are
currently vol-less — tenor-basis swaps and basis swaptions/options — and lets the
curve graph produce a *stochastic* basis for scenario/VaR generation instead of a
frozen spread. Complements the deterministic-basis multi-curve bootstrap already
shipped (`bootstrap_curve_graph`).

**Open questions & risks.** **Calibratability is the real risk, not the math:**
basis-swap *volatility* is not liquidly quoted across all tenor pairs, so
per-tenor vol/correlation parameters are largely unidentifiable — which is the
core argument *for* the low-factor form and *against* the full per-tenor model.
Empirically bases are strongly co-moving, so even a single shared factor may
capture most of the structure; the open question is whether a second (slope)
factor earns its keep. Because the standard low-factor recipe is fairly
well-established in the literature (Mercurio-style multi-curve LMM/HJM, spread
formulations), this leans closer to a *deferred implementation* than a pure
exploration — its uncertainty is **whether it is worth building for VALAX's
audiences**, not how. Correlation between the basis factor and the rate factors
adds a mixed term that must stay consistent with the existing dual-curve
discounting discipline (discounting on OIS only).

**Rough cost / complexity.** Moderate-to-large: a new model pytree
(`G2PPModel`-style, extended with the basis factor + loadings), the analytic
multi-curve ZCB/annuity under the extended dynamics, and a calibration story that
honestly confronts the thin basis-vol data. Best gated behind a *demand* signal
(a real basis-sensitive product or risk requirement) given the calibration
uncertainty.

**Status.** unprioritized. Surfaced from the G2++ multi-curve guardrails in the
rates session guide (deterministic basis "on top" of a single-curve dynamics
engine); the deterministic-basis half already ships via `bootstrap_curve_graph`.
