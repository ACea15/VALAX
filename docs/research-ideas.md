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
