# Volatility Surfaces

A volatility surface maps (strike, expiry) pairs to implied volatilities. Every option pricing function needs one — a flat scalar vol is a toy assumption that ignores the smile (wings trade at higher vol than ATM) and the term structure (short-dated vs long-dated options have different vol levels).

VALAX provides three vol surface implementations, all as callable `eqx.Module` pytrees that are JIT-compilable, differentiable, and vmappable.

## Grid Surface

The simplest construction: store implied vols on a discrete (strike, expiry) grid and bilinearly interpolate. No model assumptions.

```python
from valax.surfaces import GridVolSurface
import jax.numpy as jnp

strikes = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0])
expiries = jnp.array([0.25, 0.5, 1.0])
vols = jnp.array([
    [0.30, 0.24, 0.20, 0.22, 0.28],  # T=0.25: steep smile
    [0.28, 0.23, 0.19, 0.21, 0.26],  # T=0.5: moderate smile
    [0.26, 0.22, 0.18, 0.20, 0.24],  # T=1.0: flatter smile
])

surface = GridVolSurface(strikes=strikes, expiries=expiries, vols=vols)

# Query at any (strike, expiry) — bilinear interpolation
vol = surface(jnp.array(95.0), jnp.array(0.75))

# Vectorize across strikes
import jax
smile = jax.vmap(lambda K: surface(K, jnp.array(0.5)))(strikes)
```

The `vols` grid is a differentiable leaf. `jax.grad` through a pricing function that queries this surface gives **vega bucketed by (strike, expiry) node** — the sensitivity to each grid point individually.

**When to use:** Direct construction from broker quotes or when you have a dense grid of market-observed implied vols and don't want to impose parametric assumptions.

## SABR Surface

Industry standard for rates (swaption cubes) and common for equity. Fits a SABR model per expiry slice, interpolates parameters across expiries.

```python
from valax.surfaces import SABRVolSurface, calibrate_sabr_surface

# Market data: strikes and vols at each expiry
strikes_3m = jnp.linspace(80.0, 120.0, 11)
strikes_1y = jnp.linspace(80.0, 120.0, 11)
vols_3m = ...  # observed smile at 3M
vols_1y = ...  # observed smile at 1Y

result = calibrate_sabr_surface(
    strikes_per_expiry=[strikes_3m, strikes_1y],
    market_vols_per_expiry=[vols_3m, vols_1y],
    forwards=jnp.array([100.0, 100.0]),
    expiries=jnp.array([0.25, 1.0]),
    fixed_beta=jnp.array(0.5),  # common: 0.5 for equity, 0.0 for rates
)
result.check()             # raises CalibrationError naming any failed slice
surface = result.value     # SABRVolSurface

# Query: SABR params are interpolated to the expiry, then Hagan formula applied
vol = surface(jnp.array(95.0), jnp.array(0.5))
```

Like every surface builder, `calibrate_sabr_surface` returns a
[`CalibrationResult`](../api/calibration.md#valax.calibration.result.CalibrationResult)
and is a **batched level**: each expiry slice is solved independently in one
`vmap`, with per-slice diagnostics aligned with `result.labels` (`"3M"`,
`"1Y"`, …), so a failed slice is reported by name (`result.failed_labels()`)
rather than poisoning the others. Ragged strike grids are padded with
zero-weight points. Solver choice is the keyword-only
`settings=SolverSettings(...)` (default
[`FIT_SETTINGS`](../api/calibration.md#valax.calibration.settings.FIT_SETTINGS),
Levenberg–Marquardt).

You can also construct directly from known parameters:

```python
surface = SABRVolSurface(
    expiries=jnp.array([0.25, 1.0]),
    forwards=jnp.array([100.0, 100.0]),
    alphas=jnp.array([0.30, 0.25]),
    betas=jnp.array([0.5, 0.5]),
    rhos=jnp.array([-0.2, -0.3]),
    nus=jnp.array([0.4, 0.35]),
)
```

All SABR parameters are differentiable. `jax.grad` through the surface gives sensitivities to alpha, rho, nu at each expiry — useful for SABR vega risk.

**When to use:** Rates desks (swaption/cap vol cubes), or when you want a parsimonious parametric fit that extrapolates sensibly beyond the observed strike range.

## SVI Surface

Gatheral's Stochastic Volatility Inspired parameterization. Five parameters per expiry define total implied variance as a function of log-moneyness:

$$w(k) = a + b \left(\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right)$$

where $k = \log(K/F)$ and implied vol $= \sqrt{w / T}$.

```python
from valax.surfaces import SVIVolSurface, calibrate_svi_slice, calibrate_svi_surface

# Calibrate a single slice — .value is the fitted SVISlice
params = calibrate_svi_slice(
    strikes=jnp.linspace(80.0, 120.0, 15),
    market_vols=observed_vols,
    forward=jnp.array(100.0),
    expiry=jnp.array(1.0),
).value

# Or calibrate an entire surface (a batched level: one fit per expiry)
surface = calibrate_svi_surface(
    strikes_per_expiry=[strikes_3m, strikes_6m, strikes_1y],
    market_vols_per_expiry=[vols_3m, vols_6m, vols_1y],
    forwards=jnp.array([100.0, 100.0, 100.0]),
    expiries=jnp.array([0.25, 0.5, 1.0]),
).check().value
```

SVI interpolates **total variance** (not implied vol) across expiries. This preserves the calendar spread no-arbitrage condition: total variance must be non-decreasing in expiry.

| Parameter | Meaning | Constraint |
|---|---|---|
| $a$ | Overall variance level | Real |
| $b$ | Slope of the wings | $b \geq 0$ |
| $\rho$ | Left/right asymmetry | $-1 < \rho < 1$ |
| $m$ | Horizontal shift of the smile vertex | Real |
| $\sigma$ | Smoothing at the vertex | $\sigma > 0$ |

**When to use:** Equity options desks. SVI fits market smiles well with only 5 parameters, extrapolates cleanly, and can be constrained for arbitrage-freeness. The linear wings match the Roger Lee moment formula for extreme strikes.

## Cap → caplet (optionlet) stripping (rates)

Rates vol arrives in a form no pricer can use. A cap is quoted with **one flat volatility per maturity** — the single number that, applied to every optionlet in the strip, reproduces the market premium. A 6M optionlet and a 9Y optionlet do not share a volatility, so the first step of any rates-vol pipeline is to *strip* the quotes into a per-expiry term structure.

This is also **the canonical dual-curve operation**. A cap has two curve dependencies that move independently:

- each period's forward rate $F_i$ is **projected** off the tenor (IBOR / term) curve, and
- the payoff at $T_i$ is **discounted** off the OIS / collateral curve.

So the stripped vols are a function of *both* curves: the same quotes strip to different optionlet vols under a different tenor basis. Every entry point takes `curve` (discounting) plus an optional `forward_curve` (projection), matching the convention of the cap/caplet pricers.

### The bootstrap

Caps on a common schedule are **nested** — the 2Y cap's optionlets are the 1Y cap's plus the incremental ones — so the strip is solved forward in maturity, one scalar unknown at a time:

$$
\text{Cap}^{\text{mkt}}_i
  = \underbrace{\sum_{j \in \mathcal{C}_{i-1}} \text{Optionlet}_j(\sigma_j)}_{\text{already stripped}}
  + \sum_{j \in \mathcal{C}_i \setminus \mathcal{C}_{i-1}} \text{Optionlet}_j(\sigma_i),
$$

where $\mathcal{C}_i$ is the $i$-th quoted cap's optionlet index set and $\sigma_i$ — the vol shared by the incremental bucket — is the only unknown. Each step is a scalar Newton root-find (bucket by bucket, labelled `"cap 1Y"`, `"cap 2Y"`, …), and the solved buckets are packaged as **one lower-triangular root level**, so the whole strip is differentiable by a single **implicit-function adjoint**: `jax.grad` of a stripped vol with respect to a quoted cap vol, a discount pillar, or a projection pillar works without unrolling the solver.

```python
import jax.numpy as jnp
import equinox as eqx
from valax.surfaces import strip_caplet_vols, cap_price_from_strip

# caps: nested quarterly cap strip (1Y / 2Y / 3Y / 4Y), same strike;
# ois: discount curve;  sofr3m: projection curve.
quotes = jnp.array([0.30, 0.34, 0.36, 0.37])   # flat quoted cap vols

result = strip_caplet_vols(
    caps, ois,
    cap_vols=quotes,          # or cap_premiums=... if the desk quotes price
    forward_curve=sofr3m,     # omit for the single-curve strip
)
result.check()                # names a failed bucket, e.g. "cap 3Y"
strip = result.value          # CapletVolCurve

strip.vols        # one vol per optionlet, constant within each quoted bucket
strip.expiries    # optionlet expiries (year fractions)
strip.forwards    # projected forwards, read off sofr3m
strip(jnp.array(2.5))          # vol lookup at any expiry

# The defining property: every quoted cap is repriced exactly.
cap_price_from_strip(caps[2], strip, ois, sofr3m)
```

`strip_caplet_vols` is JIT-safe (`eqx.filter_jit` over the quotes) and reprices each quoted cap to ~1e-10 relative — including when the repricing is done by QuantLib's `BlackCapFloorEngine` off the adopted strip, which is how the routine is validated against `ql.OptionletStripper1` ([validation pyramid § 5.2](../architecture/quantlib-validation-pyramid.md#52-files)). Pass `is_normal=True` to strip Bachelier-quoted vols, which is the branch to use at or below zero rates. An optionlet that has already fixed has zero vega and nothing to strip — drop it from the quoted strip, as the market does, or the call raises.

### From a strip to a surface

`strip_caplet_vols` gives one *column*: the term structure at a single strike. `strip_optionlet_surface` does the whole quotes → surface leg — strip each strike column, transpose into one smile per optionlet expiry, read the ATM anchor (the projected forwards) off the curves, and fit one SABR model per expiry:

```python
from valax.surfaces import strip_optionlet_surface

strikes = jnp.array([0.020, 0.028, 0.035, 0.045, 0.060])
cap_vol_matrix = ...   # shape (n_strikes, n_caps): flat quoted vols

result = strip_optionlet_surface(
    caps, strikes, cap_vol_matrix, ois,
    forward_curve=sofr3m,
)
result.check()                    # checks the strip level (children[0]) too
surface = result.value            # OptionletVolSurface
strip_level = result.children[0]  # the column strips, as one root level

surface(jnp.array(0.035), jnp.array(2.5))   # smile-consistent optionlet vol
```

The result is the optionlet SABR level with the strip level as
`result.children[0]`; `settings=` drives the per-expiry SABR fits and
`strip_settings=` the strip
([`STRIP_SETTINGS`](../api/surfaces.md#valax.surfaces.caplet_stripping.STRIP_SETTINGS)
by default).

The surface is an [`OptionletVolSurface`](../api/surfaces.md#optionlet-surface-rates) ready for the caplet / cap / rates-exotics pricers — see [Interest Rate Exotics](rates-exotics.md) for the instruments that consume it.

Two points worth knowing:

- **The forwards are curve-derived.** `build_sabr_caplet_surface` takes a caller-supplied `forwards` array; `strip_optionlet_surface` reads them off the projection curve with `optionlet_forwards`, so the smile is anchored on the same forwards the pricer will use. That is what puts the quotes → surface leg inside the framework rather than in the caller's spreadsheet.
- **Beta is fixed by default** (`fixed_beta=0.5`). Alpha and beta share the ATM backbone $\sigma_{ATM} \approx \alpha F^{\beta - 1}$ and are not jointly identifiable from a single cap smile — a free-beta Levenberg-Marquardt fit stalls rather than converging (see [Calibration § Solvers](calibration.md#solvers)). Rates desks fix beta by convention; pass `fixed_beta=None` to free it anyway.

**When to use:** any rates workflow that starts from broker cap/floor quotes — pricing caps and floors off a smile rather than a flat vol, feeding an optionlet surface to the rates exotics, or building the vol input for LMM calibration.

**Inside a calibration chain**, the same two levels are the nodes
[`caplet_strip_node`](../api/calibration.md#valax.calibration.nodes.caplet_strip_node)
(curves → strip) and
[`optionlet_sabr_node`](../api/calibration.md#valax.calibration.nodes.optionlet_sabr_node)
(strip → optionlet SABR surface), so cap-vol quote risk flows through the strip
and the curves in one adjoint sweep; see
[From Quotes to Risk](calibration-to-risk.md).

## Smile dynamics: what moves when the curve moves

A fitted smile is anchored on a forward. When the curve moves, the forward
moves — and the convention you choose for what happens to the *vol* is a
modelling decision, not an implementation detail. It determines the delta of
every option priced off the surface, so it has to match the desk's risk policy
and it has to be stated.

Write $F_0$ for the forward the node was calibrated at, $F$ for the forward the
curve implies now, $K$ for the absolute strike and $\delta = K - F_0$ for the
quoted offset. The three standard conventions differ in exactly one place —
which forward enters the Hagan evaluation:

| Convention | Vol read | Forward used | Needs $F_0$? |
|---|---|---|---|
| **Sticky absolute strike** | $\sigma_{SABR}(K;\, F_0, \alpha, \rho, \nu, \beta)$ | $F_0$ — the stored, calibration-time forward | yes |
| **Sticky moneyness** | $\sigma_{SABR}(K - (F - F_0);\, F_0, \ldots)$ | both $F_0$ and $F$ | yes |
| **SABR backbone** | $\sigma_{SABR}(K;\, F, \alpha, \rho, \nu, \beta)$ | $F$ only — **computed, never interpolated** | no |

Under the backbone convention the parameters are held fixed and the smile is
re-evaluated at the new forward, with $\beta$ setting the backbone slope
($\beta \to 0$ a normal/flat backbone, $\beta \to 1$ lognormal). It is the one
convention that consults no stored forward at query time; the forward should be
the exact par rate for the traded schedule, since interpolating a forward is
strictly worse than computing one.

### Declaring the convention

The convention is a **static field**, so it is a declared choice rather than a
property of how the code happens to be wired:

```python
from valax.surfaces import SwaptionCube, SMILE_DYNAMICS, QUOTE_FORMS

SMILE_DYNAMICS   # ('sticky_strike', 'sticky_moneyness', 'sabr_backbone')
QUOTE_FORMS      # ('absolute_strike', 'atm_offset')

cube = SwaptionCube(..., dynamics="sabr_backbone", quote_form="absolute_strike")
```

- **`dynamics`** selects the row of the table above, on both `SwaptionCube` and
  `OptionletVolSurface`. It **defaults to `"sticky_strike"`**, which is what
  VALAX has always done, so no existing number changes.
- **`quote_form`** records how the calibrating quote's strike was expressed.
  `__call__` always takes an *absolute* strike; `quote_form` carries no
  query-time behaviour of its own.

The pair is **validated at construction**, because the dynamics is a
*consequence* of what the quote holds fixed: an offset-to-ATM quote moves with
the forward (sticky moneyness), an absolute-strike quote does not (sticky
strike), and `"sabr_backbone"` holds the *parameters* fixed so it contradicts
neither.

| `quote_form` | admissible `dynamics` |
|---|---|
| `"absolute_strike"` | `"sticky_strike"`, `"sabr_backbone"` |
| `"atm_offset"` | `"sticky_moneyness"`, `"sabr_backbone"` |

Anything else raises. The live forward is supplied by the pricers at query
time — `src(strike, expiry[, tenor], forward=F)` — and the two
forward-dependent conventions **raise rather than silently fall back** if they
are queried without one. `forwards` is retained on the objects but means
exactly "the forward this node was fitted at", $F_0$; `forward_at` returns that,
not the live rate.

The practical consequence, which is worth stating because it is what a risk
report shows: under the default `"sticky_strike"`, $\partial\sigma/\partial
\theta_c$ is **exactly zero** and the SABR backbone contributes nothing to
delta. Under `"sabr_backbone"` it does. Both are asserted in
`tests/test_surfaces/test_smile_dynamics.py`.

### What the convention does to delta

The choice is **invisible in the price**. Evaluated at the fitted forward
$F = F_0$, all three conventions return the same volatility — they differ only
in how it *responds* to a forward move. So no repricing test can detect a wrong
convention, and delta is the first quantity that can see it:

$$
\frac{dV}{dF} = \underbrace{\frac{\partial V}{\partial F}}_{\text{frozen vol}}
  + \underbrace{\frac{\partial V}{\partial\sigma}}_{\text{vega}}\cdot\frac{d\sigma}{dF}.
$$

| Convention | $d\sigma/dF$ at $F = F_0$ | Delta on the OTM payer below |
|---|---|---|
| **Sticky absolute strike** | $0$ exactly | 1,556,635 |
| **Sticky moneyness** | $-\,\partial\sigma/\partial K$ | 1,692,384 (+8.7%) |
| **SABR backbone** | $\partial\sigma/\partial F$ | 1,475,398 (−5.2%) |

Measured on a 1y×5y payer struck at 3.5% against a 3.0% forward, $\beta = 0.5$,
$\rho = -0.25$, $\nu = 0.4$, unit annuity 4.5, notional 1e6 — the fixture of
the T-12 gate. The ordering is not an accident of that fixture: with
$\rho < 0$ the smile falls in the strike, so sticky moneyness picks up
$-\partial\sigma/\partial K > 0$ and reports the **largest** delta, while the
backbone derivative is negative at this strike and reports the smallest. A
desk choosing between them is choosing a hedge ratio that differs by some 14%
across the two extremes, on an identically-priced trade.

Two things follow. First, this is why V-5 is a *documentation* requirement:
there is no correct answer to test against, only consistency with a stated
policy, so the gate checks the code against this page rather than against
economics. Second, the distinction vanishes on a flat smile — with $\nu = 0$,
$\beta = 1$, $\rho = 0$ the expansion is constant in both strike and forward
and every convention gives $d\sigma/dF = 0$. Any check of this behaviour has
to be run on a skewed, convex smile at an off-ATM strike, or it passes while
testing nothing.

Two caveats come with the non-default conventions:

- **Off-node queries are mixed-order.** At an intermediate (expiry, tenor) the
  cube bilinearly interpolates $\alpha, \rho, \nu, \beta$ but receives an
  *exact* forward. Since $\alpha$ is the volatility of $F^\beta$, interpolating
  it across nodes with differing forwards and then evaluating at an exact
  forward is not the same as interpolating the ATM vols. A second-order effect,
  but a real one: an off-node ATM query under the backbone convention will not
  equal the bilinear interpolation of neighbouring ATM vols.
- **`sabr_backbone` has a prerequisite.** It reproduces the calibrated market
  vol at a node only if $F = F_0$ there, which holds only when the cube was
  calibrated with forwards derived from the *same* curve and schedule
  conventions the pricer uses. That is what the curve-aware calibration route
  below guarantees — and why `sabr_backbone` is never selected implicitly, even
  when the quote form would admit it. See the
  [audit conformance map](../architecture/aad-audit-conformance.md) for how this
  scores against requirement V-5.

## Swaption cube: curve-aware calibration

A swaption cube's at-the-money anchor is the forward **par swap rate**, and the
premium it is quoted against is the **annuity** — both functions of the curves.
A forward handed in from outside would be a constant as far as the fit is
concerned, so $\partial\theta_v/\partial\theta_c = 0$ and no curve risk could
ever reach the volatility layer — which is why the cube is calibrated **only**
from the curves.

`calibrate_swaption_cube_from_curves` takes the **schedules** — one
representative `Swaption` per node — and the curves, and derives the grid axes
and the per-node $(S_x, A_d)$ itself, through the same
`swaption_forward_and_annuity` the pricer uses:

```python
import jax.numpy as jnp
from valax.dates.schedule import generate_schedule
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Swaption
from valax.surfaces import calibrate_swaption_cube_from_curves

# One representative swaption per (expiry, tenor) node. Only the schedule and
# day count are read — the `strike` field is ignored, since a node spans a
# whole smile.
def node(expiry_year, tenor):
    return Swaption(
        expiry_date=ymd_to_ordinal(expiry_year, 1, 1),
        fixed_dates=generate_schedule(
            expiry_year, 1, 1, expiry_year + tenor, 1, 1, frequency=1,
        ).payment_dates,
        strike=jnp.array(0.03), notional=jnp.array(1e6), is_payer=True,
    )

nodes = [[node(2026, 5), node(2026, 10)],      # 1y x {5y, 10y}
         [node(2027, 5), node(2027, 10)]]      # 2y x {5y, 10y}

offsets = jnp.array([-0.010, -0.005, 0.0, 0.005, 0.010])   # ATM offsets
vols = ...   # [i][j] array of quoted vols at those offsets

result = calibrate_swaption_cube_from_curves(
    nodes,
    [[offsets] * 2] * 2,
    vols,
    ois,                      # discounting
    sofr3m,                   # projection (omit for single-curve)
    quote_form="atm_offset",  # strikes are offsets to each node's own forward
    fixed_beta=jnp.array(0.5),
)
result.labels     # one "<expiry>x<tenor>" label per cell, e.g. "1Yx5Y"
result.check()    # a failed cell is named, e.g. "2Yx10Y"
cube = result.value

cube.forwards     # curve-derived F_0 per node, not caller-supplied
cube.expiries     # year fractions, exactly what the pricer will query at
cube.dynamics     # 'sticky_moneyness' — implied by the atm_offset quote form
```

Three things this buys:

- **Quote form is representable.** Pass `quote_form="atm_offset"` and the
  offsets are converted to absolute strikes per node using that node's derived
  $F_0$ (spec Q-3). `dynamics` then defaults to the convention that quote form
  implies, so the pair is consistent by construction.
- **The grid coordinate is the pricer's coordinate.** Expiries and tenors come
  from `swaption_node_coordinates`, which computes
  `year_fraction(reference_date, expiry_date)` and
  `year_fraction(expiry_date, fixed_dates[-1])` — the very expressions
  `swaption_price` evaluates. A node therefore reprices *exactly*, rather than
  a hair off its own grid point.
- **Curve risk reaches the vols.** `jax.grad` of a fitted parameter with respect
  to a discount or projection pillar is non-zero, because the forward really is
  an input to the least squares (asserted in
  `tests/test_greeks/test_vol_adjoint.py`).

Each (expiry, tenor) cell is solved independently as one slice of a batched
level (one `vmap`), so a cell that fails to fit is reported by its label rather
than failing the whole cube. Use
`calibrate_swaption_cube_from_graph(..., graph, discount_id, forward_id)` to
resolve the curves off a bootstrapped [`CurveGraph`](curves.md) instead, or
[`swaption_cube_node`](../api/calibration.md#valax.calibration.nodes.swaption_cube_node)
to make the cube a node of a calibration chain (curves → cube), so swaption-vol
quote risk and the curve risk of the forwards come out of one adjoint sweep —
the starting point of per-trade model calibration in
[example 14](../examples.md#per-trade-model-calibration-for-bermudans).

### Premium quotes

When the matrix is quoted in **premium** rather than vol, the conversion is also
curve-dependent: the premium is quoted against a numeraire, and that numeraire
is $A_d(\theta_c)$ (physical settlement) or $P_d(0,T^p)\,A_{\text{cash}}(S_x)$
(cash settlement). `swaption_implied_normal_vol` solves

$$
R(\sigma_N;\, \theta_c, q)
  = N \cdot \mathcal{N}(\theta_c)\,
    \mathrm{Bachelier}\!\left(S_x(\theta_c), K, \sigma_N, T\right) - q = 0
$$

as a scalar Newton root level, implicitly differentiated — so the *implied
vol* carries curve risk, which it cannot if the numeraire is frozen:

```python
from valax.surfaces import swaption_implied_normal_vol

sigma_n = swaption_implied_normal_vol(
    swaption, premium, ois, sofr3m,
    settlement="physical",     # or "cash"
).value                        # a CalibrationResult; .value is the vol
```

The two settlement conventions are not interchangeable and imply different
vols from the same premium. The cash branch is the street-standard
*approximation* — the flat-yield annuity is evaluated at the forward rather than
inside the expectation, so the cash-settled convexity correction is not applied;
that deviation is recorded in the
[audit conformance map](../architecture/aad-audit-conformance.md). Under
`eqx.filter_jit` the cash branch needs an explicit `freq=`, since the fixed-leg
frequency is a convention inferred from the schedule and is never guessed from a
tracer.

### Vega in quote space

Once a cube is fitted, the vega a desk hedges with is the sensitivity to the
**quoted vols**, not to $(\alpha, \rho, \nu)$. That is one more application of
the same implicit-function-theorem step the curve layer uses; the mechanism and
the API live in
[Greeks § Quote-space (hedge-space) risk](../api/greeks.md#quote-space-hedge-space-risk).

## Common Patterns

### Pricing with a surface

All surfaces share the same callable interface — `surface(strike, expiry) -> vol`:

```python
from valax.pricing.analytic import black_scholes_price
from valax.instruments import EuropeanOption

option = EuropeanOption(strike=jnp.array(95.0), expiry=jnp.array(0.5), is_call=True)

# Look up the smile-consistent vol for this option
vol = surface(option.strike, option.expiry)
price = black_scholes_price(option, spot, vol, rate, dividend)
```

### Surface Greeks via autodiff

Because surfaces are differentiable pytrees, `jax.grad` flows through them:

```python
# Vega: sensitivity to parallel vol shift
def price_with_surface(vol_grid):
    s = GridVolSurface(strikes=..., expiries=..., vols=vol_grid)
    vol = s(option.strike, option.expiry)
    return black_scholes_price(option, spot, vol, rate, dividend)

vega_bucketed = jax.grad(price_with_surface)(surface.vols)
# Shape: (n_expiries, n_strikes) — sensitivity to each grid node
```

For SABR surfaces, you get sensitivities to each SABR parameter at each expiry:

```python
def price_with_alphas(alphas):
    s = SABRVolSurface(expiries=..., forwards=..., alphas=alphas, ...)
    vol = s(option.strike, option.expiry)
    return black_scholes_price(option, spot, vol, rate, dividend)

d_price_d_alpha = jax.grad(price_with_alphas)(surface.alphas)
# Shape: (n_expiries,) — SABR alpha sensitivity per expiry bucket
```

### Vectorization

`jax.vmap` across strikes to compute the full smile:

```python
strikes = jnp.linspace(80.0, 120.0, 50)
smile = jax.vmap(lambda K: surface(K, jnp.array(0.5)))(strikes)
```

## Using a Surface as a Dupire Input

Beyond `surface(K, T) → σ_IV`, all three surface types expose

```python
surface.total_variance(log_moneyness, expiry) -> Float[Array, ""]
```

returning total variance $w(k, T) = \sigma_{\text{IV}}^2 \cdot T$ directly.
This is the duck-typed input expected by
[`dupire_local_vol`](../api/pricing.md#valax.pricing.analytic.dupire_local_vol) and by SLV's
leverage-function calibration ([SLV guide](slv.md),
[`calibrate_slv_leverage`](../api/calibration.md#valax.calibration.calibrate_slv_leverage)).
The consistency identity

```python
surface.total_variance(jnp.log(K / F_T), T) == surface(K, T) ** 2 * T
```

holds to machine precision.

### Which surface to pick

| Use case | Recommended surface | Why |
|---|---|---|
| Dupire local vol extraction | **`SVIVolSurface`** | $w(k)$ is closed-form $C^\infty$ in $k$; `jax.grad` gives exact $\partial w / \partial k$ and $\partial^2 w / \partial k^2$. |
| SLV (supported, see [SLV guide](slv.md)) | `SVIVolSurface` | Same reasons; differentiability also flows through to leverage-grid calibration via [`calibrate_slv_leverage`](../api/calibration.md#valax.calibration.calibrate_slv_leverage). |
| Quick prototyping on broker quotes | `GridVolSurface` (in log-moneyness) | No calibration needed; bilinear interpolation. |
| Rates desks | `SABRVolSurface` | Industry standard for swaption cubes. |

`SABRVolSurface.total_variance` works but uses Hagan's asymptotic IV
formula internally — at extreme strikes the resulting "local vol" is a
measurement of the expansion's residual rather than a clean market signal.

### End-to-end: SVI → Dupire → Local-vol MC

The Dupire-consistency loop is the canonical sanity check that the
extraction is plumbed correctly. Calibrate SVI to a market smile, wrap it
in a `LocalVolModel`, run LV MC, and reprice the input smile — the result
should match the input to within a few tens of basis points at typical
MC budgets.

```python
import jax, jax.numpy as jnp
from valax.surfaces import calibrate_svi_surface
from valax.models import LocalVolModel
from valax.pricing.mc import generate_local_vol_paths
from valax.instruments import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_implied_vol

# 1. Calibrate SVI to a market smile (or surface)
svi = calibrate_svi_surface(
    strikes_per_expiry=[strikes_3m, strikes_6m, strikes_1y],
    market_vols_per_expiry=[vols_3m, vols_6m, vols_1y],
    forwards=jnp.array([100.0, 100.5, 101.0]),
    expiries=jnp.array([0.25, 0.5, 1.0]),
).check().value

# 2. Wrap in a local-vol model
model = LocalVolModel.from_flat_rate(svi, rate=0.03, dividend=0.01)

# 3. Simulate
paths = generate_local_vol_paths(
    model, spot=jnp.array(100.0),
    T=1.0, n_steps=500, n_paths=100_000,
    key=jax.random.PRNGKey(20260101),
)

# 4. Reprice the input smile via MC → implied vol → compare
strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])
df = jnp.exp(-0.03 * 1.0)
for K in strikes:
    payoff = jnp.maximum(paths[:, -1] - K, 0.0)
    mc_price = df * jnp.mean(payoff)
    opt = EuropeanOption(strike=K, expiry=jnp.array(1.0), is_call=True)
    iv_mc = black_scholes_implied_vol(
        opt, jnp.array(100.0),
        jnp.array(0.03), jnp.array(0.01), mc_price,
    )
    iv_market = svi(K, jnp.array(1.0))
    diff_bp = float(jnp.abs(iv_mc - iv_market)) * 1e4
    print(f"K={float(K):5.0f}  market={float(iv_market):.4f}  mc={float(iv_mc):.4f}  diff={diff_bp:.1f} bp")
```

Single-seed runs at this size show ~15–30 bp worst-case noise; the
[Dupire-consistency gate](monte-carlo.md#local-volatility) averages 4
seeds to hit < 20 bp. See [theory §4.4](../theory/volatility.md#44-local-volatility-dupire)
for the full discussion of MC bias floors and the Milstein scheme as
the path to sub-5-bp accuracy.

### Vega-bucketed Greeks through the LV pipeline

Because every leaf of the surface pytree is differentiable, `jax.grad`
of a LV MC price w.r.t. SVI parameters gives a vector of per-slice,
per-parameter sensitivities for free:

```python
def lv_mc_price(svi_params: SVIVolSurface, key):
    model = LocalVolModel.from_flat_rate(svi_params, rate=0.03, dividend=0.01)
    paths = generate_local_vol_paths(
        model, jnp.array(100.0), 1.0, 500, 100_000, key,
    )
    payoff = jnp.maximum(paths[:, -1] - 100.0, 0.0)
    return jnp.exp(-0.03 * 1.0) * jnp.mean(payoff)

# Sensitivity to the variance level (a) at each calibrated expiry
sens_a = jax.grad(lambda a: lv_mc_price(
    eqx.tree_at(lambda s: s.a_vec, svi, a),
    jax.random.PRNGKey(42),
))(svi.a_vec)
# shape (n_expiries,) — vega-bucketed by expiry slice
```
