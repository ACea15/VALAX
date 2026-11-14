# PDE Solvers — Design

!!! note "Status: as-built architecture"
    This page describes how the `valax/pricing/pde/` package is **factored and
    why**: its layers, the seams between them, and the numerical decisions
    behind them. It began as a pre-implementation specification; it now
    documents the code as shipped, and records where the build departed from
    the original spec ([§10](#10-delivery-phases)).

    It deliberately does **not** state which instruments the engine covers:
    that is the *PDE* column of the
    [coverage matrix](../applications/instrument-pricing.md#coverage-matrix)
    (checked against the registry), and `registered_recipes()` lists the
    registered pairs at runtime. Usage is in the [PDE guide](../guide/pde.md);
    the maths in [theory §5.2](../theory/pricing-methods.md#52-pde-finite-differences),
    [Hull-White Finite Differences](../theory/hull-white-pde.md),
    [G2++ §8](../theory/g2pp.md#8-pde-2-d-finite-differences) and
    [Cheyette §8](../theory/cheyette.md#8-pde-finite-differences); public
    symbols on [`api/pricing.md`](../api/pricing.md).

## 1. Purpose

The PDE engine is one of VALAX's four numerical routes (analytic, Monte Carlo,
PDE, lattice). It earns its place where the others are weakest:

- **Early exercise.** American and Bermudan exercise are free-boundary
  problems that a backward grid solves natively, with deterministic
  (noise-free) prices and Greeks — a callable bond or Bermudan swaption needs
  no regression, unlike Longstaff–Schwartz.
- **Barriers and digitals.** Discontinuous payoffs and absorbing boundaries
  are handled exactly on the mesh, where MC and CRR trees converge slowly and
  erratically near the barrier/strike.
- **Independent cross-checks.** Heston against COS, Hull-White against the
  trinomial tree and Jamshidian, G2++ against the Brigo–Mercurio quadrature,
  Cheyette against its own exact-`y` oracle: a price that agrees across
  engines is a price worth trusting.

The package started as a single 1-D Crank–Nicolson function for a European
under Black–Scholes. It is now a layered subsystem: reusable numerics (grids,
operators, steppers, boundaries, payoff smoothing) underneath per-model recipe
modules registered on a dispatcher, reachable both directly and through
`valax.price(..., method="pde")`.

## 2. Design goals and non-goals

**Goals.**

1. **Multivariate.** First-class 1-D *and* 2-D solvers. The 2-D layer is
   general enough for Heston `(ln S, v)`, two-factor short rates (G2++,
   Cheyette 2F) and the exact-`y` one-factor Cheyette `(x, y)`.
2. **Composable numerics.** Grids, linear algebra, time-stepping, spatial
   operators, boundary conditions, terminal conditions, payoff smoothing and
   early exercise are **separate, reusable modules**, not inlined per pricer.
3. **Autodiff-clean.** The whole solve is differentiable end-to-end so Greeks
   come from `jax.grad`/`jax.jacobian`, never finite differences (per
   `AGENTS.md`). This constrains the early-exercise method (§7), the read-off
   (§4.2) and the payoff sampling (§8).
4. **Convention-consistent.** Configs are `eqx.Module`s with `static=True`
   grid sizes/flags; recipes are pure functions of `(instrument, model,
   config, **market_args)`; dispatch mirrors `mc/dispatch.py`; schedules are
   handled with **traced** indices so every recipe is `filter_jit`-able.
5. **Backward-compatible.** `pde_price(option, spot, vol, rate, dividend,
   config)` keeps its original signature and behaviour, re-expressed as a thin
   façade over the layered core (§4.13).

**Non-goals.**

- No general n-D (d > 2) solver — the curse of dimensionality makes MC the
  right tool there. The two-factor exact-`y` Cheyette PDE (5-D) is
  deliberately not built; see
  [research ideas](../research-ideas.md) for the multi-dimensional discussion.
- No adaptive/AMR meshing. Grids are structured — uniform, or non-uniform via
  a fixed analytic `sinh` stretch — never adaptively refined.
- No sparse/algebraic multigrid. Multi-D implicit solves are done by
  **operator splitting (ADI)** into per-axis tridiagonal solves (§6).

## 3. Package layout

```
valax/pricing/pde/
├── __init__.py         # public exports; imports the recipe modules for their side effects
├── config.py           # Scheme / Exercise enums, PDEConfig (1-D), PDEConfig2D
├── grids.py            # Grid1D / Grid2D builders, boundary_coords, cubic read-offs
├── linalg.py           # tridiagonal solve (lineax) + matvec
├── operators.py        # Operator1D: non-uniform 3-point stencil from drift/diffusion/discount
├── operators2d.py      # Operator2D: ADI split A0 (mixed) + A1 + A2
├── boundary.py         # Dirichlet callables, linearity folds, Heston variance-axis rows
├── schemes.py          # 1-D theta-scheme + Rannacher; solver_fn / event_fn seams
├── schemes2d.py        # 2-D ADI: Douglas / Craig–Sneyd / Hundsdorfer–Verwer
├── terminal.py         # equity terminal payoffs on the log-spot mesh
├── payoff_smoothing.py # C¹ cell-averaged positive part + smoothed exercise projections
├── exercise.py         # American penalty solver; plain explicit projection
├── coefficients.py     # model → operator: BS, local vol, Heston, Hull-White, G2++
├── rates_payoffs.py    # model-agnostic short-rate payoffs (swap tails, coupons, caplets, CMS, exercise)
├── dispatch.py         # (instrument_cls, model_cls) registry, PDEResult, pde_price_dispatch
├── recipes.py          # equity recipes (BS, local vol, Heston)
├── hull_white.py       # Hull-White 1-D recipes
├── g2pp.py             # G2++ 2-D recipes
├── cheyette.py         # Cheyette 1F/2F frozen-y recipes, operators, exact-y oracle
└── solvers.py          # back-compat pde_price façade
```

The dependency graph is layered: numerics never import recipes, and the
recipe modules depend only on the numerics (plus `dispatch.register`):

```
config ─┐
grids ──┼─► operators ─► operators2d ─► boundary ─► schemes / schemes2d
linalg ─┘                                               │
                     terminal · payoff_smoothing · exercise
                                     │
            coefficients ────────────┼──────── rates_payoffs
                                     ▼
       recipes · hull_white · g2pp · cheyette ──register──► dispatch
                                     │
                                  solvers (pde_price façade)
```

Two pragmatic exceptions to strict layering: `cheyette.py` reuses
`g2pp._phi_step_discounts`, and builds its own operators (rather than in
`coefficients.py`) because they depend on Cheyette-specific state helpers.

## 4. Layer by layer

Small pure functions, `eqx.Module` configs with static grid fields, and JIT
applied by the *caller* (no `@jax.jit` baked into pricing functions —
consistent with the rest of `valax/pricing/`).

### 4.1 `config.py` — configuration

```python
class Scheme(enum.Enum):
    IMPLICIT = "implicit"        # backward Euler (θ = 1), 1-D
    CRANK_NICOLSON = "cn"        # θ = 1/2, 1-D
    DOUGLAS = "douglas"          # 2-D ADI
    CRAIG_SNEYD = "craig_sneyd"  # 2-D ADI, 2nd order in the mixed term
    HV = "hv"                    # 2-D ADI, Hundsdorfer–Verwer
    def is_adi(self) -> bool: ...

class Exercise(enum.Enum):
    EUROPEAN = "european"; AMERICAN = "american"; BERMUDAN = "bermudan"

class PDEConfig(eqx.Module):     # all fields static
    n_spot: int = 200            # interior nodes (log-spot, or the short-rate state)
    n_time: int = 200
    spot_range: float = 4.0      # half-width in std devs
    scheme: Scheme = Scheme.CRANK_NICOLSON
    exercise: Exercise = Exercise.EUROPEAN
    rannacher_steps: int = 0
    penalty_rho: float = 1.0e6   # American penalty coefficient
    penalty_iters: int = 5       # fixed penalty iterations per step
    smooth_payoff: bool = True   # cell-averaged kinks (short-rate recipes)

class PDEConfig2D(eqx.Module):   # all fields static
    n_x: int = 128; n_y: int = 64; n_time: int = 200
    x_range: float = 4.0         # half-width in std devs (both G2++ / Cheyette-2F axes)
    y_max: float = 1.0           # Heston v_max; exact-y Cheyette: multiple of ȳ
    y_scale: float = 0.1         # sinh concentration of the Heston variance axis
    scheme: Scheme = Scheme.CRAIG_SNEYD
    exercise: Exercise = Exercise.EUROPEAN
    theta: float = 0.5
    rannacher_steps: int = 2
    smooth_payoff: bool = True
```

`PDEConfig` keeps its three original fields first and defaulted identically,
so pre-existing call sites are untouched. Fields are reused across recipe
families rather than duplicated: the short-rate 1-D recipes size their
centred-state mesh with `n_spot` / `spot_range`.

Two caveats worth knowing:

- **`exercise` is not consulted by any recipe.** Exercise style follows the
  *instrument type* — `AmericanOption` always takes the penalty path,
  `CallableBond` / `PuttableBond` / `BermudanSwaption` always project at their
  exercise dates. The field is carried for API stability only.
- **Recipes floor `rannacher_steps`.** Digital, barrier and every short-rate
  recipe use at least 2 fully-implicit start-up steps regardless of the config
  (`_MIN_RANNACHER = 2` in the short-rate modules).

Through `price(..., method="pde")` the config may be `None`: the default per
pair comes from `valax.core.policy` (see the
[PDE guide § 5.1](../guide/pde.md#51-the-dispatcher)).

### 4.2 `grids.py` — meshes

Meshes are `eqx.Module`s holding **interior** nodes only; the exterior
(boundary) coordinates are one edge spacing beyond the first/last node
(`boundary_coords`), shared by the operator stencils and the Dirichlet
boundaries so both agree on the boundary placement.

```python
class Grid1D(eqx.Module):
    nodes: Float[Array, " n"]
    n: int = eqx.field(static=True)

class Grid2D(eqx.Module):          # tensor product of two Grid1D axes
    x: Grid1D
    y: Grid1D                      # .n_x, .n_y, .shape properties

def uniform_log_spot_grid(spot, vol, expiry, *, n, half_width) -> Grid1D
def uniform_linear_grid(lo, hi, *, n) -> Grid1D            # barrier domains, exact-y axis
def centred_state_grid(std_dev, *, n, half_width) -> Grid1D # short-rate factors, centred at 0
def sinh_concentrated_grid(lo, hi, center, *, n, scale) -> Grid1D  # Tavella–Randall stretch
def log_spot_variance_grid(spot, expiry, v0, v_max, *, n_x, n_y,
                           x_half_width, v_scale, vol_ref=None) -> Grid2D  # Heston
def read_off_1d(grid, values, query) -> Float[Array, ""]
def read_off_2d(grid, values, query_x, query_y) -> Float[Array, ""]
```

The equity meshes are centred on the spot; the short-rate meshes are centred
on the origin of a zero-mean state, so their read-off point is fixed
(§5). The Heston variance axis is `sinh`-concentrated around `v0`, where the
value function has the most curvature.

#### Read-off, grid detachment, and second-order Greeks (gamma)

Neither read-off uses `jnp.interp`. A subtle interaction between
piecewise-linear interpolation and the *spot-centred* grid silently collapses
gamma to `~0`, so two coupled choices are required to get correct second-order
spot Greeks through the ordinary autodiff Greek engine.

**The trap: a co-moving grid freezes the read-off.** `uniform_log_spot_grid`
places nodes at `nodes = ln(spot) + a`, where the offset vector `a` (spacing and
width) is independent of `spot`. The query is `ln(spot)`. So under
`jax.grad(·, spot)` the query **and every node translate together** — the query
sits at a fixed *fractional* cell position for all spot. The read-off is then a
linear combination of node values with **frozen weights**, *regardless of
interpolation order*. Worse, each node value is a piecewise-linear function of
spot (the solve is linear in the terminal data, and the sampled payoff
`max(S·e^{aₖ} − K, 0)` is piecewise-linear in `S`). Hence the priced `V(S)` is
itself **piecewise-linear in `S`**: delta (the slope) is exact, but the
pointwise second derivative — what autodiff returns — is identically zero. The
option's convexity lives entirely in the kinks *between* nodes, spaced one grid
cell apart; a finite-difference bump only recovers gamma once `h ≳ dx`.

**The fix — two independent requirements:**

1. **Detach the grid from the differentiated spot.** `uniform_log_spot_grid`
   builds its nodes on `lax.stop_gradient(spot)`. The forward value is unchanged
   (still centred on spot), but the *only* differentiable spot dependence is now
   the read-off query `ln(spot)`. The grid is a numerical scaffold; its
   placement gradient is a discretization artifact, and dropping it is the
   correct definition of a PDE Greek. `vol`/`expiry` are left **live** so
   vega/theta keep the grid-width sensitivity (detaching them regresses vega).
   European/American/digital recipes inherit this from the shared builder;
   **barriers** (which mesh with `uniform_linear_grid`, one edge pinned to the
   barrier) apply the same detachment at the recipe level in `_barrier_bs` —
   grid bounds on `ln(stop_gradient(spot))`, read-off query left live. Barrier
   gamma then matches a QuantLib FD reference and obeys in/out gamma parity
   `Γ(KO) + Γ(KI) = Γ(vanilla)`. The local-vol recipe also detaches its
   grid-width reference vol, and Heston detaches `v0` / `theta` for mesh
   placement while keeping them live in the operator and read-off.
2. **A curvature-carrying read-off.** `read_off_1d` is a 4-point **cubic
   Hermite** (a non-uniform Catmull-Rom: C¹, tangents from the second-order
   three-point derivative, so it reproduces quadratics exactly; flat
   extrapolation preserved), so `V_g'' ≠ 0`. A linear read-off — *even
   detached* — still fails, yielding the spurious `Γ = −V_g'/S² = −Δ/S`.
   `read_off_2d` applies the same kernel tensor-wise (separable bicubic), which
   is why it is used instead of `surfaces._interp.bilinear_2d`: bilinear has
   no within-cell curvature and would collapse Heston gamma and the mixed
   `∂²/∂x∂v` Greek.

With both in place the query slides across a static value field and autodiff
recovers, through the **unified** `greeks()` engine (no PDE-specific Greek
path):

$$
\Delta = \frac{V_g'}{S}, \qquad \Gamma = \frac{V_g'' - V_g'}{S^2}, \qquad x = \ln S.
$$

Measured against Black–Scholes at ATM (`n_spot=400`): delta error `~7e-5`, gamma
error `~1e-5` (vs `≡0` before), and price accuracy *improves* (the cubic read-off
is closer to analytic than the linear one). The cubic's second derivative and
an on-grid central second difference are the *same stencil* on a uniform grid —
reading `V''` off the value field directly is an equivalent route; we prefer
the read-off form because it keeps every Greek on the single autodiff path.
Regression coverage: `tests/test_pde/test_grids.py`,
`tests/test_pde/test_crank_nicolson.py::TestPDESecondOrderGreeks`,
`tests/test_pde/test_barrier_pde.py::TestBarrierGamma`, the QuantLib
cross-checks `test_pde_pr1_ql.py::{test_european_gamma_matches_quantlib,
test_barrier_gamma_matches_quantlib}`, and the `--benchmark-only` convergence
profile in `test_gamma_benchmark.py`.

### 4.3 `linalg.py` — the tridiagonal workhorse

```python
def tridiagonal_solve(lower, diag, upper, rhs) -> Float[Array, " n"]   # lineax.Tridiagonal
def tridiagonal_matvec(lower, diag, upper, v) -> Float[Array, " n"]
```

Bands follow the solver convention (`lower`/`upper` of length `n − 1`). This is
the only module in `valax/pricing/` that touches `lineax`; every implicit
stage of every scheme — 1-D steps, the American penalty iterations, and the
`vmap`-batched per-axis ADI solves — goes through it.

### 4.4 `operators.py` / `operators2d.py` — spatial operators

`build_operator_1d(grid, drift, diffusion, discount)` assembles
$\mathcal{L}V = \tfrac12\sigma^2 V_{xx} + \mu V_x - rV$ with the
**non-uniform** three-point central stencil (exact for quadratics; reduces to
the classic uniform form). Each coefficient may be a scalar or a per-node
array — the per-node *discount* is what short-rate models need, since there
the discount rate is the state itself. `Operator1D` stores the bands **per
row** (length `n` each) so the edge rows' couplings to the exterior nodes are
available to the boundary layer; the interior solve uses `lower[1:]` /
`upper[:-1]`.

`Operator2D` is the three-way ADI split $L = A_0 + A_1 + A_2$ (in't Hout &
Foulon 2010):

```python
class Operator2D(eqx.Module):
    a1_lower, a1_diag, a1_upper   # (n_x, n_y): tridiagonal along axis 0
    a2_lower, a2_diag, a2_upper   # (n_x, n_y): tridiagonal along axis 1
    c0                            # (n_x, n_y): mixed coefficient, A0 V = c0 · D_x(D_y V)
    sx_*, sv_*                    # per-axis first-derivative weights for A0
    # apply_a0 / apply_a1 / apply_a2 / apply

def build_operator_2d(grid, ...) -> Operator2D
```

The discount is split evenly between `A1` and `A2` (each implicit per-axis
solve stays well posed); `A0` is applied **only explicitly** and never
inverted, so it is stored as a coefficient plus separable stencil weights.

### 4.5 `schemes.py` — 1-D time-stepping

```python
def solve_backward_1d(operator, boundary, terminal, *, expiry, n_time, theta,
                      rannacher_steps=0, solver_fn=None, event_fn=None) -> Float[Array, " n"]
def theta_for_scheme(scheme) -> float   # IMPLICIT → 1.0, CN → 0.5; raises on an ADI scheme
```

A θ-scheme march from the terminal condition to `t = 0` in a single
`jax.lax.scan`, with **Rannacher start-up** (the first `rannacher_steps` steps
fully implicit). Three extension points cover every recipe:

- **Stacked operators.** `operator` may carry bands of shape `(n,)` (constant
  in time) or `(n_time, n)` (row `m` used at backward step `m`, nearest expiry
  first). Local vol (`lv_operator_stack`), Hull-White (`hw_operator_stack`)
  and frozen-`y` Cheyette 1F (`cheyette_operator_stack`) use the stacked form.
- **`solver_fn`** replaces the per-step tridiagonal solve — the American
  penalty method plugs in here.
- **`event_fn(level, values)`** post-processes the field after each step —
  coupon injection and Bermudan / callable / puttable exercise projection.

> **Time-index convention — get this wrong and every time-dependent boundary is
> silently mispriced.** The scan marches `m = 0 … n_time-1` starting from the
> *terminal payoff*, so the level entering step `m` has time-remaining `m*dt`
> (`m = 0` is expiry, `tau = 0`) and the level being solved sits at `(m+1)*dt`.
> `event_fn` is handed the **forward** level index of the level just solved,
> `n_time - m - 1`, counting `0` at `t = 0` up to `n_time` at expiry.
> The stepper originally mirrored the boundary `tau`; see entry 2 of
> [Numerical Pitfalls](numerical-pitfalls.md) for why that survived 1298
> QuantLib comparisons, and `TestBoundaryTimeDirection` /
> `TestEventSeam` (`tests/test_pde/test_schemes.py`) for the guards that now
> pin both conventions.

### 4.6 `schemes2d.py` — 2-D ADI

```python
def solve_backward_2d(operator, boundary, terminal, *, expiry, n_time, scheme,
                      theta, rannacher_steps, event_fn=None) -> Float[Array, "n_x n_y"]
```

Douglas, Craig–Sneyd and Hundsdorfer–Verwer (§6) over the `Operator2D` split.
Each implicit stage inverts one direction only, as a batch of independent
tridiagonal solves (`tridiagonal_solve` under `jax.vmap`); the mixed term is
explicit. Rannacher steps run plain Douglas with θ = 1. With `A0 = A2 = 0` every
scheme collapses exactly to the 1-D θ-method — a structural test the suite
relies on.

The seams mirror the 1-D stepper, with one difference in form:

- **Time-dependent operators** are passed as a *function* of the backward step
  index, `operator(m) -> Operator2D`, assembled inside the loop rather than
  stored as `n_time` copies of the 2-D bands. Piecewise-constant σ(t) uses this
  (`g2pp_operator_rows`, `cheyette_operator_rows_2d`,
  `cheyette_exact_operator_rows`).
- **`event_fn`** has the same forward-level convention as in 1-D; the
  short-rate recipes use it for the deterministic per-step discount, coupon
  injection and Bermudan projection.

### 4.7 `boundary.py` — boundary conditions

Two mechanisms, chosen by whether the problem has far-field data.

**Dirichlet data (equity).** `Boundary1D(lower_fn, upper_fn)` and
`Boundary2D(x_lower_fn, x_upper_fn)` hold callables `tau -> value`. They are
**plain Python objects, not pytrees**: they are closed over by the `scan` body
and produce traced values from the traced `tau`. Factories:
`bs_european_boundary` (deep-ITM/OTM asymptotics), `american_boundary`
(intrinsic), `digital_boundary`, `knockout_boundary` (wraps another boundary,
zeroing the barrier edge), `heston_boundary` (the 1-D asymptotics reused per
variance slice), and the inert `zero_boundary`. The boundary value enters as
an affine ghost term weighted `(1-θ)`/`θ` across the step.

**Operator folds (no far-field formula).** Short-rate claims have no
closed-form far-field value — a callable's deep-ITM/OTM value depends on the
whole remaining exercise schedule — so the edge condition is a *shape*, baked
into the operator rows and paired with `zero_boundary`:

- `apply_linearity_bc_1d` / `apply_linearity_bc_2d` impose $V_{xx} = 0$ (and
  $V_{yy} = 0$ in 2-D, per axis independently) by folding a
  linearly-extrapolated ghost into the edge rows. Exact on affine fields; the
  edge convection degrades to one-sided (first order), harmless once the
  domain is wide enough that the edges carry negligible probability — worth
  *verifying* by sweeping the width rather than assuming (entry 2 of
  [Numerical Pitfalls](numerical-pitfalls.md)). The exact-`y` Cheyette folds
  only the `x` axis (its `y` axis is hyperbolic).
- `apply_heston_variance_bc` rewrites the first/last rows of `A2`:
    - **`v = 0` (degenerate).** The variance diffusion `½ξ²v` vanishes and the
      drift `κ(θ-v) → κθ > 0` points *into* the domain, so the row is a
      **drift-only, one-sided upwind** difference with no sub-zero ghost —
      well posed whether or not the Feller condition `2κθ ≥ ξ²` holds.
    - **`v = v_max`.** Linearity `V_vv = 0` via ghost extrapolation.

**Mixed term at the boundary.** `A0` is zero on all four edges in every 2-D
operator: a one-sided cross stencil there injects spurious flux (entry 1 of
[Numerical Pitfalls](numerical-pitfalls.md)). Being explicit-only, it needs no
boundary data of its own.

### 4.8 `terminal.py` and `payoff_smoothing.py` — payoffs on the mesh

`terminal.py` holds the equity terminal conditions on the log-spot mesh:
`vanilla_terminal`, `intrinsic_payoff` (the American obstacle) and
`digital_terminal`. Heston broadcasts the vanilla payoff across the variance
axis inside its recipe; barriers use the vanilla payoff plus an absorbing
boundary (§4.7), the correct treatment for continuous monitoring. Short-rate
payoffs live in `rates_payoffs.py` (§4.11).

`payoff_smoothing.py` replaces a pointwise `max(f, 0)` by a C¹ function of `f`
and its half-variation over the node's cell (Pooley–Vetzal–Forsyth 2003):
`smooth_positive_part` (exact cell average, 1-D), `bias_neutral_positive_part`
(zero integrated bias, 2-D), `cell_half_variation`, the rank-dispatching
`positive_part`, and the exercise projections `project_max` / `project_min`.
The short-rate recipes route every kinked payoff and projection through these
when `config.smooth_payoff` is set; the equity and Heston recipes ignore the
flag. Rationale in §8.

### 4.9 `exercise.py` — early exercise

```python
def penalty_solver(payoff, rho, iters) -> TridiagonalSolver   # plugs into solver_fn
def explicit_project(values, exercise_value, *, is_min) -> Float[Array, " n"]
```

`penalty_solver` returns a drop-in replacement for `tridiagonal_solve` that
iterates $(A + \rho\,\mathrm{diag}(m^k))V^{k+1} = b + \rho\, m^k \odot g$ a
**fixed** `iters` times (a `lax.scan`), so the graph is static and
differentiable. `explicit_project` is the plain pointwise `min`/`max`; the
production short-rate recipes use the smoothed `project_min` / `project_max`
of §4.8 instead (identical to it when `smooth_payoff=False`).

Discrete exercise lands through `event_fn` with **traced** level indices
(`snap_to_levels`, `.at[levels].set(...)`), so schedules need not be concrete
and the pricers stay `filter_jit`-able — unlike the trinomial tree, whose
half-width is an array shape. Snapping is used for *exercise* dates only;
cashflow dates get an exact analytic correction instead (§5).

### 4.10 `coefficients.py` — model → operator

| Model | Axes | Builder | Notes |
|---|---|---|---|
| `BlackScholesModel` | `ln S` | `bs_operator` | constant drift `r−q−½σ²`, diffusion `σ²`, discount `r` |
| `LocalVolModel` | `ln S` | `lv_operator_stack` | per-node, per-step Dupire `σ²_loc(x, t)` — a stacked operator |
| `HestonModel` | `(ln S, v)` | `heston_operator_2d` | `½v ∂ₓₓ`, `½ξ²v ∂ᵥᵥ`, mixed `ρξv ∂ₓᵥ` |
| `HullWhiteModel` | centred `x` | `hw_operator_stack` | drift `−a x`, discount `x + ᾱ_k` with `α` *integrated* per step |
| `G2PPModel` | centred `(x, y)` | `g2pp_operator_2d`, `g2pp_operator_rows` | state-only discount `x + y`; `φ(t)` factored out (§5); rows for σ(t) |

The Cheyette operators live in `cheyette.py` (they need the model's
Gaussian-limit mean and `ȳ(t)`): `cheyette_operator_stack` (1F frozen-`y`,
state-dependent diffusion `½σ²(1 + β(x̃ + m))²`), `cheyette_operator_2d` /
`cheyette_operator_rows_2d` (2F frozen-`y`), and `cheyette_exact_operator_2d` /
`cheyette_exact_operator_rows` (1F exact-`y`, with first-order **upwinded**
advection on the hyperbolic `y` axis).

SLV and two-asset operators are not built; see
[roadmap §5.4](../roadmap.md#54-multi-dimensional-pde-solvers).

### 4.11 `rates_payoffs.py` — shared short-rate payoffs

Hull-White, G2++ and Cheyette price the same family of rates payoffs, and the
payoff arithmetic is identical across them once the model can value
zero-coupon bonds at its grid nodes. Recipes therefore hand this module two
closures over `(model, grid)`:

- `bond_portfolio(t, payment_times, cashflows) -> node values` — $\sum_i c_i P(t, T_i)$;
- `discount_factor(t, payment_time) -> node values` — a single $P(t, T)$;

and it supplies, for 1-D or 2-D node arrays alike: `snap_to_levels`,
`swap_tail_values`, `coupon_values_by_level` (timing-exact coupon injection),
`caplet_fixing_values` / `caplet_values_by_level` (a caplet as a put on the
accrual-end bond, injected at its fixing level), `cms_spread_values_by_level`,
`exercise_obstacle_by_level`, and the terminal-plus-event builders
`exercisable_terminal_and_event` (callable/puttable) and
`bermudan_terminal_and_event`. Everything model-specific — affine
coefficients, the exact-fit shift, frozen- vs exact-`y` — stays inside the
closures.

### 4.12 `dispatch.py` and the recipe modules

A mirror of `mc/dispatch.py`: a module-level registry keyed on
`(type(instrument), type(model))`, a `register` decorator (raises on a
duplicate unless `overwrite=True`), `registered_recipes()`, and
`pde_price_dispatch(instrument, model, config, **market_args) -> PDEResult`,
which raises a `ValueError` listing the registered pairs on a miss.
`PDEResult` is an `eqx.Module` with a scalar `price` (and `__float__`).

Recipes are spread over four modules, all imported for their side effects by
`pde/__init__.py`:

| Module | Model family | Engine |
|---|---|---|
| `recipes.py` | Black–Scholes, local vol, Heston | 1-D CN (+ penalty / Rannacher / absorbing BC); 2-D ADI for Heston |
| `hull_white.py` | Hull-White | 1-D, centred state, linearity BC |
| `g2pp.py` | G2++ | 2-D ADI, linearity BC on both axes |
| `cheyette.py` | Cheyette 1F / 2F (frozen-`y`) | 1-D / 2-D, reusing the Hull-White / G2++ substrate |

The short-rate modules were split out of `recipes.py` because they need their
own grid, boundary and discrete-event machinery. Each module docstring
records the oracle used to validate each recipe.

**Off-registry entry point.** `cheyette_exact_price(instrument, model,
config)` prices a one-factor `FixedRateBond`, `Swaption` or `Caplet` with the
exact-`y` 2-D PDE. It is not registered because it would share the
`(instrument, CheyetteModel)` key with the frozen-`y` recipe while taking a
`PDEConfig2D`; the Cheyette calibrator selects it with `engine="exact"`.

**Market arguments and `price()`.** Equity recipes take `spot`. Short-rate
recipes may additionally name `discount_curve` and `forward_curve` (and, for
multi-index trades, `curves` / `spots`). `valax.core.pricing.price` bridges
the registry directly — no adapter per recipe — and supplies those arguments
**only to a recipe that names them explicitly** in its signature: a
`**market_args` catch-all does not count, so an un-migrated recipe can never
silently drop a projection curve, and `price()` refuses a trade touching
curves its recipe cannot take. Hull-White and G2++ honour both curves
(re-fit the exact-fit model to the discount curve; frozen tenor basis for
projection — [Multi-Curve Short Rates](../theory/multi-curve-short-rate.md));
Cheyette is single-curve. Both default to `None`, leaving single-curve prices
bit-for-bit unchanged.

### 4.13 `solvers.py` — the back-compat façade

```python
def pde_price(option, spot, vol, rate, dividend, config=PDEConfig()) -> Float[Array, ""]
```

Builds a `BlackScholesModel` and runs grid → `bs_operator` →
`bs_european_boundary` → `vanilla_terminal` → `solve_backward_1d` →
`read_off_1d`. Same signature as the original monolithic solver; it is a
public kernel in its own right, not legacy.

## 5. Coordinate and discretisation conventions

- **Equity spot axis:** log-spot `x = ln S` (uniform in moneyness; removes the
  `S²` coefficient). Grid placement detached from autodiff (§4.2).
- **Variance axis (Heston):** `v` directly on `[0, y_max]`, `sinh`-concentrated
  at `v0`, with the boundary rows of §4.7.
- **Short-rate axes:** the **centred** state of `r(t) = x(t) + α(t)`
  (Hull-White), `r = x + y + φ(t)` (G2++), or the centred Cheyette factor
  `x̃ = x − m(t)` (frozen-`y`), on `centred_state_grid` sized from the
  state's terminal standard deviation. Drift and diffusion are then
  time-independent (Hull-White, G2++ with constant σ), the state starts at the
  origin — so the read-off point is fixed and needs no `stop_gradient`
  scaffolding — and the curve dependence collapses into the deterministic
  shift.
- **Deterministic discount shifts.** Hull-White and Cheyette 1F put `α(t)`
  into a per-step operator stack. G2++ and the 2-D Cheyette recipes instead
  **factor out** the spatially-uniform `φ(t)`: with `V = D(t) W`, `W` solves a
  PDE with the state-only discount, and `exp(−∫φ)` re-enters as a per-step
  scalar applied through `event_fn`.
- **Time:** backward from the terminal condition, `dt = T / n_time`, via
  `jax.lax.scan`. See the time-index convention box in §4.5.
- **Curve-derived quantities are *integrated* across a step, not sampled at its
  midpoint.** A log-linear discount curve has a piecewise-constant instantaneous
  forward that jumps at every pillar, so midpoint sampling is only first-order
  on the steps that straddle one — enough to stall an otherwise second-order
  scheme (entry 3 of [Numerical Pitfalls](numerical-pitfalls.md)). The same
  discipline applies to piecewise-constant σ(t): each step uses exact step
  averages of the variance rates.
- **Contractual dates are not snapped to time levels** where an exact correction
  exists. Snapping displaces a cashflow by up to `dt/2`, an `O(dt)` error that
  dominates a second-order scheme; the short-rate recipes instead scale each
  coupon by the model's analytic bond price from its snapped level to its true
  payment date (entry 4 of [Numerical Pitfalls](numerical-pitfalls.md)).
  Exercise dates *are* snapped — a decision has to happen at a level — but that
  error is second order.
- **Exercise ordering.** Exercise is decided on the *ex-coupon* continuation
  value and the coupon added afterwards (call/put prices are quoted ex-coupon),
  matching the tree; callable/puttable exercise compares a *clean* call price
  against a clean continuation value.
- **Read-off:** cubic Hermite in 1-D; separable bicubic in 2-D (§4.2).

## 6. Multi-dimensional scheme: why ADI

The central numerical decision. A 2-D implicit step in principle inverts a
large block-banded operator; we instead **split** it into per-axis tridiagonal
solves.

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **ADI (Douglas / Craig–Sneyd / HV)** | Each step = a few **1-D tridiagonal solves per axis** → reuses `linalg.py`; unconditionally stable; the industry standard for Heston/basket FD | Cross term treated explicitly → needs CS/HV to recover 2nd order | **Chosen** |
| Full 2-D implicit (block/sparse) | Cross term implicit; conceptually uniform | No sparse/block operator in VALAX; large systems; harder to keep autodiff-clean | Rejected |
| Explicit (FTCS) | Trivial, no solves | CFL-restricted `Δt`; unstable/slow for realistic grids | Rejected |

ADI factors cleanly into the tridiagonal building blocks, and Craig–Sneyd/HV
correctly handle the mixed term — Heston's spot–vol correlation `ρξv ∂ₓᵥ` and
G2++'s factor correlation `ρση ∂ₓᵧ` alike. Douglas is robust for weakly
correlated problems; Craig–Sneyd is the accuracy default (`PDEConfig2D`); HV is
available for strongly correlated / convection-dominated regimes. The exact-`y`
Cheyette oracle uses Douglas (no mixed term; upwinded `y`).

## 7. Early exercise: method rationale

| Method | Autodiff | Smoothness | Used for |
|---|---|---|---|
| **Penalty method** | Clean (fixed iteration of smooth ops) | Smooth Greeks | **American (continuous)** |
| **Projection at event levels** | Clean (`min`/`max`, smoothed by default) | C¹ with `smooth_payoff` | **Bermudan / callable / puttable** |
| PSOR | Awkward (data-dependent iteration count) | Non-smooth | Rejected |

- **Penalty method** for continuous American exercise: a forcing term
  $\rho\,\max(g - V, 0)$ pushes the solution above the payoff, solved with a
  fixed number of iterations per step. Fixed iteration count + smooth
  operations ⇒ clean `jax.grad`.
- **Projection** for discrete-date exercise: `V ← max/min(V_cont, exercise)`
  only at snapped exercise levels — the lattice pattern of
  `hull_white_tree.py`. Because the projection is itself a kink, the
  short-rate recipes cell-average it (§8).
- **PSOR is rejected**: its data-dependent iteration count and non-smooth
  updates are hard to make cleanly differentiable in JAX.

American exercise is 1-D only; projection works on both the 1-D and 2-D
steppers. American / exotic payoffs on the 2-D grid are planned
([roadmap §5.4](../roadmap.md#54-multi-dimensional-pde-solvers)).

## 8. Non-smooth payoffs: Rannacher start-up and cell averaging

Two complementary remedies, one in time and one in space.

**Rannacher start-up (time).** Crank–Nicolson and Craig–Sneyd oscillate on
discontinuous terminal data (digitals) and kinks (vanillas at the strike,
swaption terminals, projections), polluting gamma/vega. Replacing the first
`rannacher_steps` steps with fully-implicit steps damps the high-frequency
modes. Defaulted on for the 2-D config, and floored at 2 by the digital,
barrier and short-rate recipes.

**Cell averaging (space).** Rannacher does nothing about where the kink sits
relative to the *nodes*. Sampled pointwise, the discrete price is only
piecewise smooth in the model inputs — it kinks each time a node crosses the
exercise boundary — so finite-difference risk depends on the step and 1-D
convergence is erratic. `payoff_smoothing` makes the price C¹ at an `O(h²)`
bias: the exact cell average on 1-D grids, and a bias-neutral kernel on 2-D
grids, where the cell average's one-signed bias measured larger than the
error it removed. The switch is `smooth_payoff` (default on; `False` restores
pointwise sampling bit-for-bit, and is part of the config that calibration
provenance records). The kernels, the measured evidence and the 2-D
decision are in
[Hull-White Finite Differences § 4](../theory/hull-white-pde.md#kinks-cell-averaged-payoff-and-exercise)
and the [PDE guide § 5.5](../guide/pde.md#55-non-smooth-payoffs-rannacher-start-up-and-cell-averaging).

## 9. Validation strategy

Tests mirror the source layout (`valax/pricing/pde/X.py` ↔
`tests/test_pde/test_X.py`, plus per-model and per-feature files such as
`test_heston_pde.py`, `test_g2pp.py`, `test_cheyette.py`,
`test_*_piecewise.py`, `test_short_rate_caps.py`, `test_cms_spread_pde.py`,
`test_multicurve.py`, `test_payoff_smoothing.py`), following the tiered
tolerances of `AGENTS.md`:

- **Cross-checks against other engines.** European ↔ analytic BS;
  American ↔ binomial; digital ↔ analytic; barrier ↔ structural invariants
  (in/out parity, including gamma parity) and QuantLib FD gamma — VALAX has
  no closed-form barrier pricer yet; local vol ↔ Dupire MC; Heston ↔ COS (`rel < 1e-3`) and Andersen-QE MC; Hull-White ↔ analytic
  bond, Jamshidian, the trinomial tree; G2++ ↔ the Brigo–Mercurio quadrature;
  Cheyette ↔ its β = 0 reduction to Hull-White / G2++ and the frozen-`y` ↔
  exact-`y` bias measurement; Bermudans ↔ co-terminal European bounds.
- **QuantLib comparisons** under `tests/test_quantlib_comparison/`:
  `test_pde_pr1_ql.py`, `test_pde_lattice_ql.py`, `test_local_vol_pde_ql.py`,
  `test_heston_pde_ql.py`, `test_hull_white_pde_ql.py`, `test_g2pp_pde_ql.py`,
  `test_cheyette_pde_ql.py` (via the β = 0 reduction; QuantLib has no Cheyette
  engine).
- **Measured convergence order** and **domain-width independence**, not just
  an absolute error at the default grid — the lesson of entries 2–4 of
  [Numerical Pitfalls](numerical-pitfalls.md)
  (e.g. `TestPDEDomainWidthIndependence`, `TestBoundaryTimeDirection`).
- **Structural tests:** Heston with frozen variance reproduces
  Black–Scholes under every ADI scheme (`test_schemes2d.py`); with smoothed
  payoffs, full-revaluation FD risk is step-independent and equals `jax.grad`,
  and the Hull-White European swaption error falls by 4 per refinement
  (`test_payoff_smoothing.py`).
- **`@eqx.filter_jit` smoke tests** for the recipes, and the route contract
  (`tests/test_core/test_route_contract.py`) for every `price(..., method="pde")`
  pair.
- **Autodiff Greeks** through the solver, sanity-checked against FD.

## 10. Delivery phases

The layers were built bottom-up in reviewable slices. This section is the
history; current coverage is the coverage matrix's *PDE* column, open work is
[roadmap §5.4](../roadmap.md#54-multi-dimensional-pde-solvers).

1. **PR-1 (1-D foundation).** `config`, `grids`, `linalg`, `schemes`,
   `operators`, `boundary`, `terminal`, `exercise`, `coefficients` (BS),
   `dispatch`, `recipes` (European / American / digital / barrier under BS),
   and the `solvers.py` refactor. Local vol followed as a stacked-operator
   recipe.
2. **PR-2 (2-D / stochastic vol).** European Heston: `operators2d`,
   `schemes2d` (Douglas / Craig–Sneyd / HV), `heston_operator_2d`, the
   variance-axis boundary rows. Validated against COS, Andersen-QE MC and
   QuantLib `AnalyticHestonEngine` / `FdHestonVanillaEngine`. The mixed-term
   boundary bug found here is entry 1 of
   [Numerical Pitfalls](numerical-pitfalls.md).
3. **PR-3 (rates).** The Hull-White short-rate PDE (`hull_white.py`): bonds,
   callable/puttable bonds, European and Bermudan swaptions, then caplets and
   caps. Three discretisation bugs surfaced and were fixed — time-reversed
   boundary sampling in *both* steppers, midpoint-sampled `α`, and cashflow
   snapping (entries 2–4 of [Numerical Pitfalls](numerical-pitfalls.md)).
4. **Two-factor and skewed short rates.** G2++ (`g2pp.py`) on the 2-D ADI
   substrate, including CMS-spread swaps and callable/puttable bonds; Cheyette
   1F/2F frozen-`y` and the 1F exact-`y` oracle (`cheyette.py`); the shared
   payoff layer `rates_payoffs.py` factored out of the three.
5. **Cross-cutting upgrades.** Multi-curve discounting/projection for the
   Hull-White and G2++ recipes; piecewise-constant σ(t) for all three
   short-rate models (stacked rows in 1-D, per-step operator functions in 2-D);
   cell-averaged payoffs and projections (`payoff_smoothing.py`).

**Where the build departed from the original spec**, in each case for
autodiff correctness or simplicity:

- The 2-D stepper is **`schemes2d.py`** + **`operators2d.py`**, not a
  standalone `adi.py`.
- The 2-D read-off is a **separable bicubic Hermite**, not `bilinear_2d`
  (§4.2).
- Boundaries are **callables plus operator folds**, not
  `Dirichlet` / `Neumann` / `Linearity` classes; no Neumann condition was
  needed.
- Discrete exercise and coupons use the **`event_fn`** seam and American
  exercise the **`solver_fn`** seam, rather than a single `project` hook.
- The exercise style follows the **instrument type**; `config.exercise` is not
  consulted (§4.1).
- Short-rate payoffs live in **`rates_payoffs.py`** and the Cheyette operators
  in **`cheyette.py`**, not in `terminal.py` / `coefficients.py`.
- The `SLVModel` and `MultiAssetGBMModel` coefficient adapters, and the
  `spread_terminal` / 2-D American paths, were not built — they are the open
  items of roadmap §5.4.

## 11. Relationship to other docs

- **Coverage (what is priced by PDE):**
  [coverage matrix](../applications/instrument-pricing.md#coverage-matrix),
  *PDE* column.
- **Usage, configs, recipe-level engine details:** [PDE guide](../guide/pde.md).
- **Maths:** [theory §5.2](../theory/pricing-methods.md#52-pde-finite-differences)
  (operators, ADI, penalty, Rannacher),
  [Hull-White Finite Differences](../theory/hull-white-pde.md),
  [G2++ §8](../theory/g2pp.md#8-pde-2-d-finite-differences),
  [Cheyette §8](../theory/cheyette.md#8-pde-finite-differences),
  [Time-Dependent Volatility](../theory/time-dependent-volatility.md),
  [Multi-Curve Short Rates](../theory/multi-curve-short-rate.md).
- **Bugs found building it:** [Numerical Pitfalls](numerical-pitfalls.md).
- **API reference:** [`api/pricing.md`](../api/pricing.md#pde).
