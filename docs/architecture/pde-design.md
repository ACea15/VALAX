# PDE Solvers — Design

!!! note "Status: design specification (prose-first)"
    This document is the design specification for VALAX's **multivariate** PDE
    pricing subsystem. It is written **before** the implementation so the
    architecture can be reviewed against readable prose and math rather than
    diffs. Public signatures shown here are **illustrative** — they describe the
    intended API and are not yet importable. As each layer lands, its symbols
    gain Google-style docstrings and are switched on in
    [`api/pricing.md`](../api/pricing.md) so `mkdocs build --strict` stays green.

    The theory behind the equations is in [theory §5.2](../theory.md#52-pde-finite-differences);
    the user-facing walkthrough is in the [PDE guide](../guide/pde.md).

## 1. Motivation

VALAX today ships a single PDE solver: a 1-D Crank–Nicolson scheme for a
European option under Black–Scholes, living in one file
(`valax/pricing/pde/solvers.py`). Everything — grid construction, coefficient
assembly, boundary conditions, the tridiagonal solve, and the price read-off —
is inlined in one `pde_price` function. It covers exactly one model
(`BlackScholesModel`) and one instrument (`EuropeanOption`), with no early
exercise.

Meanwhile the rest of the library already carries the *models* (Heston, local
vol, SLV, Hull–White, multi-asset GBM) and the *instruments* (American,
barrier, digital, spread, worst-of, callable/puttable bonds, Bermudan
swaptions) that a real PDE engine would price. Those instruments are currently
reachable only via Monte Carlo, analytic approximations, or lattices. A proper
finite-difference subsystem "lights up" that existing surface area:

- **Early exercise done right.** American/Bermudan exercise is a natural
  free-boundary problem on a PDE grid. Today it rides on the binomial tree
  (`BinomialConfig.american`) keyed on `EuropeanOption`; the `AmericanOption`
  type has no pricer at all.
- **Barriers and digitals.** These are notoriously inaccurate under MC and CRR
  near the barrier/strike. A PDE with an absorbing boundary and Rannacher
  start-up is the right tool. `DigitalOption` currently has **no** pricer
  anywhere in VALAX.
- **Stochastic-volatility grids.** Heston and (local-)stochastic vol admit
  2-D backward PDEs that complement the COS and MC pricers and cross-check
  them.

This document specifies a layered, multi-file `valax/pricing/pde/` package that
supports 1-D and 2-D (and, where it makes sense, n-D) problems, early exercise,
multiple models, and a dispatcher — mirroring how the `mc/` package is
factored.

## 2. Design goals and non-goals

**Goals.**

1. **Multivariate.** First-class 1-D *and* 2-D solvers, with the 2-D layer
   general enough for Heston `(S, v)`, local-stochastic vol, and two-asset
   `(S₁, S₂)` problems.
2. **Composable numerics.** Grids, linear algebra, time-stepping, spatial
   operators, boundary conditions, terminal conditions, and early-exercise
   projection are **separate, reusable modules** — not inlined per pricer.
3. **Autodiff-clean.** The whole solve is differentiable end-to-end so Greeks
   come from `jax.grad`/`jax.jacobian`, never finite differences (per
   `AGENTS.md`). This constrains our choice of early-exercise method (§7).
4. **Convention-consistent.** Configs are `eqx.Module`s with `static=True`
   grid sizes/flags; entry points are pure functions `price(instrument,
   model, ...)`; dispatch mirrors `mc/dispatch.py`.
5. **Backward-compatible.** The existing `pde_price(option, spot, vol, rate,
   dividend, config)` keeps its exact signature and behaviour, re-expressed as
   a thin wrapper over the new core.

**Non-goals (for this design).**

- No general n-D (d > 3) solver — the curse of dimensionality makes MC the
  right tool there. We scope to d ≤ 2 concretely, with the operator layer left
  open to a third axis where a specific product demands it.
- No adaptive/AMR meshing. Grids are structured (optionally non-uniform via a
  fixed analytic stretch), not adaptively refined.
- No sparse/algebraic multigrid. Multi-D implicit solves are done by
  **operator splitting (ADI)** into per-axis tridiagonal solves (§6), reusing
  the machinery we already have.

## 3. Where this sits in the tree

```
valax/pricing/pde/
├── __init__.py        # public exports + recipe import side-effect
├── config.py          # PDEConfig (1-D, back-compat), PDEConfig2D, scheme/exercise enums
├── grids.py           # Grid1D / Grid2D builders (uniform + stretched), price read-off
├── linalg.py          # reusable tridiagonal (Thomas / lineax) solve + matvec
├── schemes.py         # theta-scheme time-stepping + Rannacher start-up (1-D)
├── operators.py       # spatial operator assembly from drift/diffusion coefficients
├── adi.py             # 2-D ADI schemes (Douglas / Craig–Sneyd / Hundsdorfer–Verwer)
├── boundary.py        # Dirichlet / Neumann / linearity BCs (1-D endpoints, 2-D edges)
├── terminal.py        # terminal (payoff) conditions evaluated on the mesh
├── exercise.py        # early-exercise projection (penalty method, explicit projection)
├── coefficients.py    # model → PDE operator coefficients (BS, Heston, LV, SLV, HW, multi-asset)
├── dispatch.py        # (instrument_cls, model_cls) registry + pde_price_dispatch
├── recipes.py         # per-(instrument, model) registered pricers
└── solvers.py         # high-level drivers + back-compat pde_price wrapper
```

The layering is strict: a module only depends on modules **above** it in this
list (with `coefficients.py`/`recipes.py`/`dispatch.py`/`solvers.py` as the
"assembly" layer that ties the numerics to VALAX models and instruments).

```
config → grids → linalg → schemes → operators → adi → boundary → terminal → exercise
                                                                                  │
                                          coefficients ────────────────┐         │
                                                                       ▼         ▼
                                                          dispatch ← recipes ← solvers
```

## 4. Layer-by-layer specification

The tone below mirrors `mc/`: small pure functions, `eqx.Module` configs with
static grid fields, and JIT applied by the *caller* (no `@jax.jit` baked into
pricing functions — consistent with the rest of `valax/pricing/`).

### 4.1 `config.py` — configuration

A single place for grid sizes, scheme selection, and exercise style. Grid
counts and enums are `static=True` so they specialise the compiled graph.

```python
class Scheme(enum.Enum):        # time-stepping scheme
    IMPLICIT = "implicit"       # backward Euler (θ=1)
    CRANK_NICOLSON = "cn"       # θ=1/2
    DOUGLAS = "douglas"         # 2-D ADI, θ per axis
    CRAIG_SNEYD = "craig_sneyd" # 2-D ADI, 2nd-order cross-term
    HV = "hv"                   # 2-D ADI, Hundsdorfer–Verwer

class Exercise(enum.Enum):
    EUROPEAN = "european"
    AMERICAN = "american"       # continuous free boundary (penalty)
    BERMUDAN = "bermudan"       # discrete exercise dates (explicit projection)

class PDEConfig(eqx.Module):                       # 1-D — unchanged public shape
    n_spot: int = eqx.field(static=True, default=200)
    n_time: int = eqx.field(static=True, default=200)
    spot_range: float = eqx.field(static=True, default=4.0)
    scheme: Scheme = eqx.field(static=True, default=Scheme.CRANK_NICOLSON)
    rannacher_steps: int = eqx.field(static=True, default=0)   # 0 = off

class PDEConfig2D(eqx.Module):
    n_x: int = eqx.field(static=True, default=128)     # first axis (e.g. log-spot)
    n_y: int = eqx.field(static=True, default=64)      # second axis (e.g. variance)
    n_time: int = eqx.field(static=True, default=200)
    x_range: float = eqx.field(static=True, default=4.0)
    y_max: float = eqx.field(static=True, default=1.0) # e.g. max variance
    scheme: Scheme = eqx.field(static=True, default=Scheme.CRAIG_SNEYD)
    theta: float = eqx.field(static=True, default=0.5)
    rannacher_steps: int = eqx.field(static=True, default=2)
```

Design note: `PDEConfig` keeps its three original fields *first and defaulted
identically*, so all existing call sites and tests are untouched; `scheme` and
`rannacher_steps` are additive with backward-compatible defaults.

### 4.2 `grids.py` — meshes

Grid construction (currently inlined in `pde_price`) becomes reusable builders
returning lightweight `eqx.Module` meshes. Uniform spacing in `x = ln S` is the
default (equal resolution in moneyness); an optional analytic **sinh-stretch**
concentrates nodes near a focal point (strike or barrier) for sharper Greeks
without extra nodes.

```python
class Grid1D(eqx.Module):
    nodes: Float[Array, " n"]          # spatial coordinates (e.g. log-spot)
    n: int = eqx.field(static=True)

class Grid2D(eqx.Module):
    x_nodes: Float[Array, " nx"]
    y_nodes: Float[Array, " ny"]
    nx: int = eqx.field(static=True)
    ny: int = eqx.field(static=True)

def uniform_log_spot_grid(spot, vol, expiry, *, n, half_width) -> Grid1D: ...
def stretched_grid(center, lo, hi, *, n, concentration) -> Grid1D: ...
def read_off_1d(grid, values, query) -> Float[Array, ""]:      # jnp.interp
    ...
def read_off_2d(grid, values, x_query, y_query) -> Float[Array, ""]:
    ...   # reuses valax.surfaces._interp.bilinear_2d
```

We deliberately **reuse `valax/surfaces/_interp.py::bilinear_2d`** for the 2-D
read-off — it is already autodiff-clean with flat extrapolation.

### 4.3 `linalg.py` — the tridiagonal workhorse

Today the tridiagonal LHS assembly and the RHS matvec are inlined. ADI needs
per-axis tridiagonal solves *many* times per step, so this is extracted once:

```python
def tridiagonal_solve(lower, diag, upper, rhs) -> Float[Array, " n"]:
    """Solve a tridiagonal system via lineax.Tridiagonal()."""
    op = lx.TridiagonalLinearOperator(diag, lower, upper)
    return lx.linear_solve(op, rhs, solver=lx.Tridiagonal()).value

def tridiagonal_matvec(lower, diag, upper, v) -> Float[Array, " n"]:
    """Multiply a tridiagonal operator by a vector (the RHS build)."""
    ...
```

This is the **only** place that touches `lineax` (it currently appears in
exactly one file across the whole codebase), keeping the dependency surface
tiny and the solve testable in isolation.

### 4.4 `schemes.py` — 1-D time-stepping

A θ-scheme stepper generalising the current CN loop, plus **Rannacher
start-up**: run `rannacher_steps` fully-implicit (θ=1) sub-steps first to damp
oscillations from non-smooth terminal data, then switch to CN. The backward
loop is a `jax.lax.scan` (as today).

```python
def theta_step(operator, boundary, v, tau_new, tau_old, dt, theta) -> Float[Array, " n"]: ...
def solve_backward_1d(operator, boundary, terminal, grid, *, n_time, dt,
                      scheme, rannacher_steps, project=None) -> Float[Array, " n"]:
    """Backward time-march terminal→0 via lax.scan; `project` hook = early exercise."""
```

The optional `project` callable is the seam for early exercise (§7) — it is
`None` for European, and a value-projection closure otherwise.

### 4.5 `operators.py` — spatial operators

Builds the discrete spatial operator $\mathcal{L}$ from **drift** and
**diffusion** coefficient fields sampled on the grid. In 1-D this yields the
`(lower, diag, upper)` tridiagonal bands (generalising today's constant
`alpha/beta/gamma`); coefficients may be space- **and** time-dependent (needed
for local vol). In 2-D it yields the per-axis tridiagonal parts **plus a
cross-derivative stencil** for the mixed term $\partial_{xy}$.

```python
class Operator1D(eqx.Module):
    lower: Float[Array, " n"]; diag: Float[Array, " n"]; upper: Float[Array, " n"]

class Operator2D(eqx.Module):
    x_bands: Operator1D           # A_x (acts along x for each fixed y)
    y_bands: Operator1D           # A_y (acts along y for each fixed x)
    cross: Float[Array, "nx ny"]  # A_0 mixed-derivative stencil weights

def build_operator_1d(grid, drift, diffusion, discount) -> Operator1D: ...
def build_operator_2d(grid, drift_x, drift_y, diff_xx, diff_yy, diff_xy, discount) -> Operator2D: ...
```

The mixed-derivative stencil uses a standard four-point centred difference; it
is treated **explicitly** in the ADI schemes (§6), which is why we need
Craig–Sneyd/HV corrections to recover second order.

### 4.6 `adi.py` — 2-D solvers

Alternating-Direction Implicit operator splitting. Each time step decomposes
into a sequence of **1-D tridiagonal solves per axis** (via `linalg.py`) with
the cross term applied explicitly. Three schemes, increasing in accuracy/cost:

- **Douglas** — first predictor + one implicit correction per axis; robust,
  first-order in the cross term.
- **Craig–Sneyd** — adds a second explicit stage that restores **second-order**
  accuracy in the presence of the mixed derivative.
- **Hundsdorfer–Verwer** — a two-parameter scheme with better damping for
  strongly correlated / convection-dominated problems (e.g. large `|ρ|` Heston).

```python
def adi_step(op2d, boundary, v, dt, *, scheme, theta) -> Float[Array, "nx ny"]: ...
def solve_backward_2d(op2d, boundary, terminal, grid, *, n_time, dt,
                      scheme, theta, rannacher_steps, project=None) -> Float[Array, "nx ny"]: ...
```

The `project` hook mirrors the 1-D case, so American exercise on a 2-D grid
(e.g. American under Heston) reuses the same seam.

### 4.7 `boundary.py` — boundary conditions

An abstraction over the two 1-D endpoints and the four 2-D edges (plus
corners). Three BC families cover our instruments:

- **Dirichlet** — fixed value (barrier level; deep-ITM/OTM asymptotics).
- **Neumann** — fixed first derivative (linearity of value in $S$ at the far
  field).
- **Linearity / "PDE-at-boundary"** — impose $\partial_{xx}V = 0$ at the far
  edge (the common robust choice for the variance axis in Heston).

```python
class Dirichlet(eqx.Module): value_fn: Callable   # tau -> value
class Neumann(eqx.Module):   slope_fn: Callable
class Linearity(eqx.Module): pass

class Boundary1D(eqx.Module): lower: ...; upper: ...
class Boundary2D(eqx.Module): x_lower: ...; x_upper: ...; y_lower: ...; y_upper: ...
```

This generalises today's inline `boundary_lower`/`boundary_upper` closures.

### 4.8 `terminal.py` — payoff on the mesh

Terminal (payoff) conditions evaluated on the grid, per instrument. Unlike the
MC payoff functions (which return per-path cashflows), these return
**grid-shaped** arrays: shape `(n,)` in 1-D, `(nx, ny)` in 2-D.

```python
def vanilla_terminal(grid, strike, is_call) -> Float[Array, " n"]: ...
def digital_terminal(grid, strike, payout, is_call) -> Float[Array, " n"]: ...   # Rannacher-smoothed
def heston_terminal(grid, strike, is_call) -> Float[Array, "nx ny"]: ...          # payoff independent of v
def spread_terminal(grid, strike, is_call) -> Float[Array, "nx ny"]: ...
```

Barriers are handled as an **absorbing Dirichlet boundary** at the barrier
level (set in `boundary.py`) rather than only through the terminal condition —
the correct treatment for continuously-monitored knock-outs.

### 4.9 `exercise.py` — early exercise

See §7 for the method rationale. Two projection strategies:

```python
def penalty_project(v, payoff, *, rho, iters) -> Float[Array, " n"]:
    """Smooth American projection: add rho*max(payoff - v, 0) forcing term."""

def explicit_project(v, exercise_value, is_min) -> Float[Array, " n"]:
    """Bermudan/callable: v <- max/min(v, exercise_value) at snapped event steps."""
```

Both are `jax.grad`-friendly. `explicit_project` reuses the date→step snapping
idiom from `hull_white_tree.py` (`round(times / dt)`), so discrete exercise
dates map to specific `lax.scan` steps.

### 4.10 `coefficients.py` — model → operator coefficients

The adapter layer mapping each VALAX model onto the drift/diffusion fields the
operator builder expects. It reuses the models' existing coefficient helpers
wherever possible.

| Model | Axes | Coefficients (source) |
|---|---|---|
| `BlackScholesModel` | `x = ln S` | drift `r−q−½σ²`, diffusion `½σ²`, discount `r` (from `GBMDrift`/`GBMDiffusion`) |
| `HestonModel` | `(ln S, v)` | from `HestonDrift`/`HestonDiffusion`: `½v ∂ₓₓ`, `½ξ²v ∂ᵥᵥ`, `ρξv ∂ₓᵥ` |
| `LocalVolModel` | `x = ln S` | per-node `σ_loc = dupire_local_vol(surface, k, t)` at each `(x, t)` |
| `SLVModel` | `(ln S, v)` | Heston block × leverage `L(k,t)·√v` via `LeverageGrid.__call__` |
| `HullWhiteModel` | `r` | drift `θ(t)−a r`, diffusion `½σ²`, discount `r` |
| `MultiAssetGBMModel` | `(ln S₁, ln S₂)` | covariance `Σ_ij = ρ_ij σ_i σ_j` (cross term = `ρσ₁σ₂`) |

Local vol is the interesting case: coefficients are **time- and
space-dependent**, evaluated at each grid node per time step via
`jax.vmap(dupire_local_vol)`. This is why `operators.py` accepts callable
coefficients, not just constant bands.

### 4.11 `dispatch.py` + `recipes.py` — instrument×model routing

A direct mirror of `mc/dispatch.py`: a module-level dict registry keyed on
`(type(instrument), type(model))`, a `register` decorator, and a
`pde_price_dispatch` entry point that raises a helpful `ValueError` listing
available recipes on a miss. `recipes.py` is imported for its side effects
(populating the registry), exactly like `mc/recipes.py`.

```python
_REGISTRY: dict[tuple[type, type], Callable[..., PDEResult]] = {}

def register(instrument_cls, model_cls, *, overwrite=False): ...
def pde_price_dispatch(instrument, model, config, **market_args) -> PDEResult: ...

class PDEResult(eqx.Module):
    price: Float[Array, ""]
    def __float__(self) -> float: return float(self.price)
```

Illustrative recipe registrations for the first delivery:

```python
@register(AmericanOption, BlackScholesModel)
def _american_bs(*, instrument, model, config, spot): ...

@register(DigitalOption, BlackScholesModel)
def _digital_bs(*, instrument, model, config, spot): ...

@register(EquityBarrierOption, BlackScholesModel)
def _barrier_bs(*, instrument, model, config, spot): ...

@register(EuropeanOption, HestonModel)
def _european_heston(*, instrument, model, config, spot): ...   # 2-D ADI
```

### 4.12 `solvers.py` — drivers + back-compat

High-level 1-D/2-D drivers that assemble grid → operator → boundary → terminal
→ time-march, plus the **unchanged** public `pde_price`:

```python
def pde_price(option, spot, vol, rate, dividend, config=PDEConfig()) -> Float[Array, ""]:
    """Back-compat façade: builds a BlackScholesModel + EuropeanOption path
    through the new 1-D driver. Identical signature and results to today."""
```

## 5. Coordinate and discretisation conventions

- **Equity spot axis:** log-spot `x = ln S` (uniform grid = uniform in
  moneyness; kills the `S²` coefficient). Matches today's solver and Heston's
  `log S` state.
- **Variance axis (Heston/SLV):** `v` directly on `[0, y_max]`, with a
  linearity BC at `v = y_max` and the natural (degenerate) behaviour at
  `v = 0` handled by the Feller-aware upwinding of the convection term.
- **Short-rate axis (Hull–White):** `r` directly, centred on the forward curve;
  discount term `r·V`.
- **Time:** backward from the terminal condition, `dt = T / n_time`, via
  `jax.lax.scan`.
- **Read-off:** `jnp.interp` in 1-D; `bilinear_2d` in 2-D — both differentiable.

## 6. Multi-dimensional scheme: why ADI

The central numerical decision. A 2-D implicit step in principle inverts a
large block-banded operator; we instead **split** it into per-axis tridiagonal
solves.

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **ADI (Douglas / Craig–Sneyd / HV)** | Each step = a few **1-D tridiagonal solves per axis** → reuses `linalg.py`; unconditionally stable; the industry standard for Heston/basket FD | Cross term treated explicitly → needs CS/HV to recover 2nd order | **Chosen** |
| Full 2-D implicit (block/sparse) | Cross term implicit; conceptually uniform | No sparse/block operator exists in VALAX; large systems; harder to keep autodiff-clean | Rejected |
| Explicit (FTCS) | Trivial, no solves | CFL-restricted `Δt`; unstable/slow for realistic grids | Rejected |

ADI factors cleanly into the tridiagonal building blocks we already have, and
Craig–Sneyd/HV correctly handle the $\partial_{xy}$ mixed term arising from
spot–vol correlation $\rho$. Douglas is the robust default for weakly
correlated problems; Craig–Sneyd is the accuracy default; HV is available for
strongly correlated / convection-dominated regimes.

## 7. Early exercise: method rationale

| Method | Autodiff | Smoothness | Fit |
|---|---|---|---|
| **Penalty method** | Clean (fixed iteration of smooth ops) | Smooth Greeks | **American (continuous)** |
| **Explicit projection** | Clean (`max`/`min` at event steps) | Kinked but fine | **Bermudan / callable / puttable (discrete dates)** |
| PSOR | Awkward (data-dependent iteration count) | Non-smooth | Rejected |

- **Penalty method** for continuous American exercise: add a forcing term
  $\rho\,\max(g - V, 0)$ that pushes the solution above the payoff, solved with
  a small fixed number of iterations per step. Fixed iteration count + smooth
  operations ⇒ clean `jax.grad` (important because `AGENTS.md` mandates autodiff
  Greeks, not FD).
- **Explicit projection** for discrete-date exercise (Bermudan swaptions,
  callable/puttable bonds): apply `V ← max/min(V_continuation, exercise_value)`
  **only** at snapped event steps — matching the existing lattice pattern in
  `hull_white_tree.py`.
- **PSOR is rejected**: its data-dependent iteration count and non-smooth
  updates are hard to make cleanly differentiable in JAX.

## 8. Non-smooth payoffs: Rannacher start-up

Crank–Nicolson oscillates on discontinuous terminal data (digitals) and kinks
(vanillas at the strike), polluting gamma/vega. The fix is **Rannacher
start-up**: replace the first `rannacher_steps` CN steps (typically 2) with
fully-implicit (backward-Euler) steps, which damp high-frequency modes, then
resume CN. This is essential for `DigitalOption` (whose terminal condition is a
step) and for barriers, and cheap insurance for vanillas. It is exposed as a
config knob (`rannacher_steps`), defaulted **on** for the 2-D config and for
digital/barrier recipes.

## 9. Validation strategy

Testing mirrors the source layout (`valax/pricing/pde/X.py` ↔
`tests/test_pde/test_X.py`) and follows the tiered tolerance convention from
`AGENTS.md` and the existing PDE tests:

- **Cross-checks against existing pricers.** European↔analytic BS (rel
  `< 5e-3` on a fine grid); American↔binomial (`american=True`);
  spread↔Margrabe/Kirk; callable/puttable↔Hull–White tree; digital↔tight
  call-spread proxy; Heston↔COS; otherwise MC within `2·stderr`.
- **QuantLib comparisons** under `tests/test_quantlib_comparison/`, driving
  both engines from the shared `effective_market` in `_ql_adapters.py`
  (extended with Heston/basket adapters as needed): VALAX FD vs
  `FdBlackScholesVanillaEngine` / `FdHestonVanillaEngine` (rel `< 1e-2`);
  American premium vs `BinomialVanillaEngine` CRR (abs `< 0.05`).
- **Convergence-ordering test** (`fine < coarse`) as the PDE-specific accuracy
  idiom, rather than pinning an absolute error.
- **`@eqx.filter_jit` smoke test** for every new pricing function (the mandated
  test that is currently missing from `test_crank_nicolson.py` — we do not copy
  that gap).
- **Autodiff Greeks** taken with `jax.grad` *through* the solver, sanity-checked
  vs FD at `rtol ≈ 1e-4`.
- **Golden values** only for canonical configs lacking an analytic/QL reference,
  registered via `assert_matches_golden` with tolerances loosened toward the PDE
  floor.

## 10. Delivery phases

The layers build strictly bottom-up, so the work sequences into reviewable
slices:

1. **PR-1 (1-D foundation).** `config`, `grids`, `linalg`, `schemes`,
   `operators` (1-D), `boundary`, `terminal`, `exercise`, `coefficients` (BS
   only), `dispatch`, `recipes` (American/Digital/Barrier under BS), and the
   `solvers.py` refactor. High value, no ADI. Cross-checks: binomial, analytic,
   call-spread proxy.
2. **PR-2 (2-D / stochastic vol).** `operators` (2-D cross-term), `adi`, and
   Heston/SLV/local-vol recipes. Cross-checks: COS, MC, QuantLib
   `FdHestonVanillaEngine`.
3. **PR-3 (rates).** Hull–White short-rate PDE for callable/puttable bonds and
   Bermudan swaptions. Cross-checks: Hull–White tree, LSM.

Each PR adds Google-style docstrings and switches on its
[`api/pricing.md`](../api/pricing.md) entries, keeping `mkdocs build --strict`
green throughout.

## 11. Relationship to existing docs

- **Theory:** the PDE math (operators, ADI, penalty method, Rannacher, per-model
  PDEs) lives in [theory §5.2](../theory.md#52-pde-finite-differences).
- **User guide:** usage, the model×instrument matrix, config knobs, and
  Greeks-through-solver are in the [PDE guide](../guide/pde.md).
- **API reference:** [`api/pricing.md`](../api/pricing.md) — populated
  incrementally as symbols land.
