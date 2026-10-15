# Cheyette: Quasi-Gaussian Skew and a Low-Dimensional HJM

How a state-dependent short-rate volatility buys a genuine rate **skew** while
keeping the term structure a finite-dimensional Markov process, why the
zero-coupon bond stays affine regardless of that volatility, and how VALAX
simulates and prices the one- and two-factor Cheyette models.

Model: `valax/models/cheyette.py`.
Monte Carlo: `valax/pricing/mc/cheyette_paths.py`.
PDE: `valax/pricing/pde/cheyette.py`.
Calibration: `valax/calibration/cheyette.py`.
Recipes: `valax/pricing/mc/recipes.py`.

---

## 1 · The model

The Cheyette (quasi-Gaussian) model is the Markovian representation of the
Heath–Jarrow–Morton framework under a **separable** forward-rate volatility
structure. Hull–White and G2++ are its *Gaussian* special cases: there the
volatility is deterministic, rates are normally distributed, and the implied
smile is flat. Cheyette lets the short-rate volatility depend on the **state**
— a local-volatility function $\sigma_r(t, x)$ — and so produces a volatility
**skew** on rates while staying low-dimensional enough for a lattice, PDE, or
Monte-Carlo scheme.

The short rate is the initial forward plus a single mean-reverting state
variable, made Markovian by an auxiliary accumulator $y$:

$$
r(t) = f^M(0, t) + x(t)
$$

$$
dx(t) = \bigl(y(t) - \kappa\,x(t)\bigr)\,dt + \sigma_r(t, x(t))\,dW(t), \quad x(0) = 0
$$

$$
dy(t) = \bigl(\sigma_r(t, x(t))^2 - 2\kappa\,y(t)\bigr)\,dt, \quad y(0) = 0
$$

$\kappa$ is the mean-reversion speed and $\sigma_r(t, x)$ the instantaneous
short-rate volatility. VALAX uses a **displaced-diffusion / linear** local
volatility (`cheyette_local_vol`),

$$
\sigma_r(t, x) = \sigma(t)\,(1 + \beta\,x),
$$

where $\sigma(t)$ sets the level and the skew parameter $\beta$ tilts the
volatility with the state. The level is piecewise constant in time, one value
per bucket, and a constant level is a single bucket (§10). Formulas on this
page are written for the constant case, $\sigma(t) \equiv \sigma$. Negative $\beta$ lowers volatility as rates rise (a
normal-to-lognormal interpolation); $\beta = 0$ recovers the constant Gaussian
volatility.

!!! success "Exact fit for free"
    Because $r = f^M(0, t) + x$ with $x(0) = 0$ and the convexity is generated
    *dynamically* by the $y$-drift rather than a static shift $\varphi$, the
    model reprices the initial curve by construction — no calibration step for
    the drift. At $t = 0$, $x = y = 0$, `cheyette_bond_price` returns
    $P^M(0, T)$ to machine precision.

---

## 2 · The affine bond — skew without losing tractability

The defining property of the whole quasi-Gaussian class is that the zero-coupon
bond is **affine in the state regardless of the local-volatility form**. The
function $\sigma_r$ enters only the *dynamics* of $(x, y)$, never the bond
formula (`cheyette_bond_price`):

$$
P(t, T \mid x, y)
    = \frac{P^M(0, T)}{P^M(0, t)}
      \exp\!\Bigl[-x\,G(t, T) - \tfrac{1}{2}\,y\,G(t, T)^2\Bigr],
\qquad
G(t, T) = \frac{1 - e^{-\kappa (T - t)}}{\kappa}.
$$

$G$ is the quasi-Gaussian analogue of Hull–White's $B$ (`cheyette_G`). The
accumulator $y$ enters *quadratically*, which is exactly the term that a
constant-volatility Gaussian model carries as a deterministic convexity — here
it is a genuine state variable that fluctuates with the realised volatility.
The bond's autodiff Greeks are closed-form: $\partial P/\partial x = -G\,P$ and
$\partial P/\partial y = -\tfrac12 G^2 P$.

!!! note "Cheyette *is* local-vol G2++ — made tractable"
    A natural question is why not simply replace the constant $\sigma$ in
    [G2++](g2pp.md) with a state-dependent $\sigma(t, x)$. That substitution *is*
    the (two-factor) Cheyette model — there is no third thing. But done naively
    it breaks the two properties that make a Gaussian short-rate model usable:
    the exact-fit shift $\varphi(t)$ and the affine bond both rely on the
    integrated variance $\int \sigma^2\,ds$ being a *deterministic* constant, and
    a state-dependent $\sigma$ makes it random and path-dependent.

    The accumulator $y$ is precisely the fix: it promotes that integrated
    variance to a live Markov state, which keeps the model finite-dimensional
    *and* restores the affine bond for **any** local-volatility form. G2++ is the
    special case where $y$ is frozen to its deterministic Gaussian value
    (`cheyette_y_gaussian_2f`) and folded into $\varphi$; Cheyette lets it
    fluctuate. That is exactly why the $\beta = 0$ reduction of §4–5 is exact.

---

## 3 · Where the skew comes from

For a lognormal-style displacement ($\beta < 0$) the volatility of the short
rate falls as the rate rises, so out-of-the-money payer options (struck above
the forward) see *less* volatility than receivers struck symmetrically below —
the price of a payer/receiver pair is no longer symmetric in the strike, and the
Black-implied volatility acquires a **downward slope in strike**. That is the
skew a flat Gaussian model cannot produce, and it is the reason Cheyette exists
alongside the lognormal-only LMM and the *static* SABR smile: it delivers a
**dynamic, low-dimensional** skew that a PDE or tree can roll backward for
callable exotics.

The natural showcase instruments are the skew-sensitive vanillas — **caplets**
and **European swaptions** — and, once the second factor is added, the
skew-and-decorrelation exotics (Bermudan swaptions with skew, callable CMS,
TARNs). VALAX registers `Caplet`, `Cap` and `Swaption` recipes for the
one-factor model.

!!! warning "Local volatility, not stochastic volatility"
    $\sigma_r(t, x)$ is a **local** volatility in the state — one source of
    randomness per factor, not a separate volatility process. So Cheyette
    reproduces a *skew* well, but its smile **dynamics** are the Dupire-like
    local-vol kind: the backbone moves the "wrong" way as rates move, and a
    single factor struggles to fit strong curvature and both wings at once. When
    the smile *dynamics* (not just today's skew) drive the risk — long-dated
    vega-convexity products, forward-starting exotics — the next rung is to add a
    stochastic-volatility driver on top (quasi-Gaussian with stochastic vol),
    which is beyond the current implementation.

---

## 4 · The Gaussian limit — reduction to Hull–White

At $\beta = 0$ the local volatility is constant, the state is Gaussian, and the
model **is** Hull–White. The accumulator becomes deterministic
(`cheyette_y_gaussian`),

$$
y(t) = \frac{\sigma^2}{2\kappa}\bigl(1 - e^{-2\kappa t}\bigr) = \mathrm{Var}[x(t)],
$$

and — crucially — the Cheyette factor is **not centred**: its $y$-drift gives it
a non-zero mean (`cheyette_gaussian_mean`)

$$
m(t) = \mathbb{E}[x(t)] = \frac{\sigma^2}{2\kappa^2}\bigl(1 - e^{-\kappa t}\bigr)^2,
$$

which is exactly the Hull–White convexity shift $\alpha(t) - f^M(0, t)$.  A
centred Hull–White state $x_{\mathrm{HW}}$ therefore corresponds to the Cheyette
factor $x = x_{\mathrm{HW}} + m(t)$, and under that mapping the affine bond of §2
equals `hw_bond_price` to machine precision. This is the primary validation
oracle (there is no QuantLib Cheyette engine).

---

## 5 · Two factors — decorrelation on top of skew

A single factor forces every tenor rate to move in lock-step. Adding a second
mean-reverting factor lets rates of different maturities **decorrelate** — the
low-dimensional, skew-capable analogue of [G2++](g2pp.md). The short rate is the
sum of two states on top of the initial forward (`CheyetteModel2F`):

$$
r(t) = f^M(0, t) + x_1(t) + x_2(t),
$$

$$
dx_i = \Bigl(\textstyle\sum_j y_{ij} - \kappa_i x_i\Bigr)dt
        + \sigma_{r,i}(t, x_i)\,dW_i, \qquad
dy_{ij} = \bigl(\sigma_{r,i}\sigma_{r,j}\rho_{ij} - (\kappa_i + \kappa_j)y_{ij}\bigr)dt,
$$

with per-factor local volatilities $\sigma_{r,i} = \sigma_i(1 + \beta_i x_i)$,
instantaneous Brownian correlation $\rho$ ($\rho_{12} = \rho$, $\rho_{ii} = 1$),
and a symmetric auxiliary matrix $y_{ij}$. The bond stays affine in the enlarged
state (`cheyette_bond_price_2f`):

$$
P(t, T \mid x, y)
    = \frac{P^M(0, T)}{P^M(0, t)}
      \exp\!\Bigl[-\textstyle\sum_i x_i G_i
                  - \tfrac{1}{2}\sum_{i,j} y_{ij} G_i G_j\Bigr],
\qquad
G_i = \frac{1 - e^{-\kappa_i (T - t)}}{\kappa_i}.
$$

As with G2++, the tenor-rate decorrelation is a model *output* bounded by
$\rho$: it is largest when $\rho$ is strongly negative and $\kappa_1 \neq
\kappa_2$, and vanishes as $|\rho| \to 1$. When both skews vanish the factors are
Gaussian and the model **reduces to G2++** — the two-factor accumulators become
deterministic and reproduce the G2++ covariance term
(`cheyette_y_gaussian_2f`), so the affine bond matches `g2pp_bond_price` to
machine precision under the same non-centred state map (`cheyette_gaussian_mean_2f`).

`CheyetteModel2F` parametrises the second level as a ratio of the first, as
G2++ does. Its fields are `volatility_1` ($\sigma_1$) and `volatility_ratio`
($\lambda = \sigma_2/\sigma_1$), and $\sigma_2$ is the read-only property
`volatility_2`.

The decorrelation-sensitive showcase is the `CMSSpreadSwap` steepener /
flattener, registered for the two-factor model.

---

## 6 · Monte Carlo (integrating-factor scheme)

Once $\sigma_r$ depends on the state the factor is **non-Gaussian**, so the
exact Cholesky-of-covariance sampling used for Hull–White and G2++ does not
transfer. `generate_cheyette_paths` advances each step with an
**integrating-factor** (exponential-Euler) update that integrates the linear
mean-reversion drift exactly and freezes the local volatility over the step,

$$
x(t+\Delta t) = x(t)\,e^{-\kappa\Delta t}
    + \frac{y(t)}{\kappa}\bigl(1 - e^{-\kappa\Delta t}\bigr)
    + \sigma_r(t, x)\,\sqrt{\tfrac{1 - e^{-2\kappa\Delta t}}{2\kappa}}\;z,
$$

with the accumulator advanced by the exact solution of its linear ODE,

$$
y(t+\Delta t) = y(t)\,e^{-2\kappa\Delta t}
    + \frac{\sigma_r(t, x)^2}{2\kappa}\bigl(1 - e^{-2\kappa\Delta t}\bigr).
$$

In the Gaussian limit the accumulator step is *exact* and the factor step is
exact given the frozen deterministic drift, so the scheme collapses onto the
exact Gaussian simulation up to a smooth $O(\Delta t^2)$ drift-freezing error.

**Exact-forward discounting.** The Cheyette short rate is simply $r = f^M(0, t)
+ \sum_i x_i$, with the convexity carried dynamically by the $y$-drift — there is
no separate shift $\varphi$ to integrate. The deterministic forward part of the
discount is integrated *exactly*: over each step it telescopes to the market
discount-factor ratio $\ln P^M(0, t_i)/P^M(0, t_{i+1})$, so a curved initial
forward introduces no repricing bias. Only the smooth stochastic $\sum_i x_i$
part uses the trapezoidal rule. The martingale identity
$\mathbb{E}[D(0, T)] = P^M(0, T)$ then holds within Monte-Carlo error on both
flat and steep curves, with or without a skew.

The two-factor generator `generate_cheyette_paths_2f` draws correlated shocks by
a $2\times2$ Cholesky of the $[[1,\rho],[\rho,1]]$ correlation and advances the
symmetric accumulator $y_{ij}$ analogously.

---

## 7 · Validation

| Oracle | Agreement | What it tests |
|---|---|---|
| Exact-fit ZCB at $t = 0$ | `< 1e-10` abs (flat) | $r = f^M(0,t) + x$ reprices the curve |
| Hull–White reduction ($\beta = 0$) | `< 1e-9` rel | 1F affine bond collapses to HW under $x = x_{\mathrm{HW}} + m(t)$ |
| G2++ reduction ($\beta_i = 0$) | `< 1e-9` rel | 2F affine bond collapses to G2++ |
| $\partial P/\partial x = -G P$, $\partial P/\partial y = -\tfrac12 G^2 P$ | `< 1e-9` rel | Autodiff Greeks vs closed form |
| $\mathbb{E}[D(0,T)] = P^M(0,T)$ (MC) | $\lvert z\rvert < 3$ | Integrating-factor scheme + exact-forward discount, flat & steep |
| Fixed bond MC vs curve | $\lvert z\rvert < 3$ | Recipe discounting |
| Caplet $-$ floorlet parity | $\lvert z\rvert < 3$ | Model-free cap/floor identity |
| Swaption MC vs `hw_swaption_price` ($\beta = 0$) | $\lvert z\rvert < 3.5$ | Independent analytic reduction oracle |
| Swaption price vs skew | monotone, measurable | The smile the Gaussian model cannot make |
| CMS steepener $+$ flattener | `= 0` | Sign / discounting conventions |
| CMS steepener vs $\rho$ | monotone | Decorrelation sensitivity of the second factor |

There is **no QuantLib Cheyette engine**, so validation triangulates the
exact-fit and Gaussian-limit reductions (which pin the model against the already
QuantLib-validated Hull–White and G2++), the internal MC↔analytic agreement, and
model-free parity relations, in the spirit of the Andersen–Piterbarg
quasi-Gaussian benchmarks.

---

## 8 · PDE (finite differences)

The headline products — Bermudan swaptions, callables — have no closed form, so
they are priced by solving the Cheyette backward PDE. The obstacle is
dimensionality: the state is $(x, y)$ in one factor and
$(x_1, x_2, y_{11}, y_{22}, y_{12})$ in two, i.e. a naive PDE is 2-D (1F) or 5-D
(2F). The auxiliary $y$ is **advection-only** (a pure-drift accumulator, no
Brownian term — see §6), which is what makes lower-dimensional treatments
possible. VALAX ships two tiers.

Source: `valax/pricing/pde/cheyette.py`.

### Tier 1 — frozen-$y$ (the production engine)

Replace the stochastic accumulator by its deterministic Gaussian-limit value
$\bar y(t)$ (`cheyette_y_gaussian`), so **one factor becomes a genuine 1-D PDE**
and **two factors a 2-D ADI PDE** — reusing the [Hull–White](hull-white-pde.md)
and [G2++](g2pp.md) substrates almost verbatim. Working in the *centred* factor
$\tilde x_i = x_i - m_i(t)$ with $m = $ `cheyette_gaussian_mean` recovers exactly
the Gaussian centred dynamics $d\tilde x_i = -\kappa_i \tilde x_i\,dt +
\sigma_{r,i}\,dW_i$, so the centred grid (`centred_state_grid`), the
zero-curvature boundary (`apply_linearity_bc_1d` / `_2d`) and the exact-fit
discount factoring all carry over. The one-factor backward equation is

$$
\frac{\partial W}{\partial t}
    + \tfrac12\,\sigma_r(t, \tilde x + m)^2\,\frac{\partial^2 W}{\partial \tilde x^2}
    - \kappa \tilde x\,\frac{\partial W}{\partial \tilde x}
    - \bigl(\tilde x + \alpha_{\mathrm{Ch}}(t)\bigr) W = 0,
$$

with $\alpha_{\mathrm{Ch}}(t) = f^M(0,t) + m(t)$ (baked into the discount and
exactly step-averaged, the Hull–White `hw_alpha_average` discipline) and, in two
factors, the ADI cross term $\rho\,\sigma_{r,1}\sigma_{r,2}\,V_{\tilde x_1
\tilde x_2}$ with $\varphi_{\mathrm{Ch}}(t) = f^M(0,t) + m_1(t) + m_2(t)$
factored out per step (the G2++ $\varphi$ discipline). Exercise, coupon
injection and tail-swap values plug into the same `event_fn` seam and are
computed analytically from `cheyette_bond_price[_2f]` with the frozen $y = \bar
y$.

**The one new ingredient** is the diffusion coefficient $\tfrac12\sigma_r(t,
\tilde x + m)^2 = \tfrac12\sigma^2(1 + \beta(\tilde x + m))^2$ — *spatially
varying*, where Hull–White and G2++ carry a constant $\tfrac12\sigma^2$. The
operator builders already accept a per-node diffusion field, so this is a new
coefficient adapter (`cheyette_operator_stack`, `cheyette_operator_2d`), not new
solver machinery. In two factors the cross term becomes node-dependent too.

!!! note "Frozen-$y$ is a slightly different model for $\beta \neq 0$"
    Freezing $y$ decouples the accumulator from the state, so for $\beta \neq 0$
    the frozen-$y$ PDE and the true-$y$ Monte Carlo (§6) are *slightly different
    models*: their cross-check measures the frozen-$y$ error, not machine
    agreement. At $\beta = 0$ the accumulator is already deterministic and the
    two coincide — which is the primary reduction oracle below.

### Tier 2 — exact-$y$ (one-factor validation oracle)

Carrying $y$ as a real axis gives a genuine **2-D** PDE in $(x, y)$ — the true
model, no frozen-$y$ approximation:

$$
\frac{\partial V}{\partial t}
    + \tfrac12\,\sigma_r(x)^2\,\frac{\partial^2 V}{\partial x^2}
    + (y - \kappa x)\,\frac{\partial V}{\partial x}
    + \bigl(\sigma_r(x)^2 - 2\kappa y\bigr)\,\frac{\partial V}{\partial y}
    - x\,V = 0.
$$

The $x$ axis is parabolic (central differences); the $y$ axis is **pure
advection** and uses first-order **upwinding** for stability. The deterministic
$f^M(0,t)$ discount is factored out per step; the accumulator convexity is now
generated *natively* by the $y$ dynamics, so there is no shift to add. Its value
is diagnostic: `cheyette_exact_price(instrument, model, config)` gives the
exact-$y$ price with **zero Monte-Carlo noise**, so the difference from Tier 1
is a clean, deterministic measurement of the frozen-$y$ bias. Empirically that
bias is a small fraction of the price for realistic rate volatilities — the
quantitative justification for using the frozen-$y$ engine in production.

The **two-factor** exact-$y$ PDE is 5-D (2 diffusive + 3 advective directions)
and is intentionally *not* implemented: it is logged as a research spike in
[`research-ideas.md`](../research-ideas.md), since carrying it through a
differentiable, reverse-mode calibration loop is dominated by the
scan-checkpoint memory cost, not the forward FLOPs.

### PDE validation

| Oracle | Agreement | What it tests |
|---|---|---|
| Hull–White PDE reduction ($\beta = 0$, same mesh) | `< 1e-10` abs | 1F operator + discount collapse to HW |
| G2++ PDE reduction ($\beta_i = 0$, same mesh) | `< 1e-9` abs | 2F ADI operator collapses to G2++ |
| QuantLib `FdHullWhite`/`Jamshidian` ($\beta = 0$) | `< 5e-4` rel | 1F, closing the chain to QuantLib |
| QuantLib `FdG2SwaptionEngine` ($\beta_i = 0$) | `< 1.5e-2` rel | 2F FD vs FD |
| Coupon bond reprices curve | `< 5e-5` abs (frozen), `< 5e-5` (exact-$y$) | Exact fit / scheme calibration |
| Frozen-$y$ vs exact-$y$ ($\beta = 0$) | grid tolerance | The two tiers are the same model at zero skew |
| MC ↔ frozen-$y$ PDE | within MC error $+$ frozen-$y$ bias | §6 engine vs §8 engine |
| $\partial \text{price}/\partial(\sigma, \beta)$ finite | — | Autodiff flows through the backward sweep (for calibration) |

Registered recipes (via `pde_price_dispatch`): `FixedRateBond` (the option-free
calibration of the scheme), `Swaption`, `BermudanSwaption`, and — one factor —
`CallableBond` / `PuttableBond`.

---

## 9 · Calibration

Cheyette is calibrated *through the PDE* rather than an approximate-analytic
swaption formula, because the PDE covers the full instrument set — caps,
European and Bermudan swaptions, callables — on one consistent engine. Model
prices come from the frozen-$y$ recipes (production) or, for one factor, the
exact-$y$ solver (bias-free); both are cleanly reverse-mode differentiable — the
backward sweep is `lineax` tridiagonal solves inside a `lax.scan` with nothing
detached from autodiff — so the least-squares Jacobian is **exact autodiff
through the PDE**. No finite-difference Greeks and no implicit-function-theorem
adjoint are needed.

Source: `valax/calibration/cheyette.py`.

**Identification.** One factor $(\kappa, \sigma, \beta)$: fix $\kappa$ (weakly
identified by the ATM level); $\sigma$ then sets the ATM level and $\beta$ the
skew, so calibrating $\beta$ needs a **strike slice** of caplets or swaptions —
the skew is invisible at a single strike. Two factors add $\rho$ for
decorrelation: fix $\kappa_1, \kappa_2$ (well separated), start $\rho$ from a
prior, and add a decorrelation-sensitive instrument (e.g. a CMS-spread option)
to pin it.

**Engine choice (one factor).** `calibrate_cheyette` selects the engine:
`"frozen"` (default, fast) or `"exact"` (bias-free, more expensive per solve).
`calibrate_cheyette_2f` is always frozen-$y$ (the exact-$y$ 2F PDE is the 5-D
research spike above).

**Solver.** A line-searched BFGS on the relative-residual sum of squares
(default
[`CHEYETTE_SETTINGS`](../api/calibration.md#valax.calibration.cheyette.CHEYETTE_SETTINGS),
BFGS to `1e-10` — looser than the closed-form fits because the PDE has a
discretisation floor), with `fixed_params` support and the curve pinned
(exact-fitted, not a degree of freedom: it is the level's upstream input, and
the returned model is curve-free). The result is a `CalibrationResult` whose
`.provenance` records the PDE engine (`cheyette_provenance`) — load-bearing
for the frozen-$y$ engine, whose parameters are valid only on that engine
(`check_engine`). Any least-squares method can be selected with
`settings=SolverSettings(...)`. Because
the PDE sits inside the optimiser loop, use a moderate grid and a small
instrument set, reusing the mesh across iterations.

---

## 10 · Time-dependent volatility

The level $\sigma(t)$ (one factor) and the pair $(\sigma_1(t), \lambda(t))$
(two factors, $\sigma_2 = \lambda\sigma_1$) are piecewise constant on buckets,
using the same convention as Hull–White and G2++. What changes:

- **Frozen $y$.** $\bar y(t) = \int_0^t \sigma(u)^2 e^{-2\kappa(t-u)}\,du$ is
  exactly the Hull–White state variance with $a = \kappa$, and $m(t)$ is the
  Hull–White convexity. Both come from a Gaussian twin (Hull–White in one
  factor, G2++ in two) built on the same buckets. The frozen-$y$ discount and its
  exact step average are therefore the twin's.
- **Diffusion.** Each PDE step and each Monte-Carlo step uses the exact time
  average of $\sigma^2$ over the step, times the state factor
  $(1 + \beta x)^2$ (`cheyette_mean_variance_rate`, and
  `cheyette_mean_variance_rates_2f` in two factors).
- **Exact-$y$ PDE.** The operator becomes time dependent. Use
  `cheyette_exact_operator_rows`, because the constant-only
  `cheyette_exact_operator_2d` raises for more than one bucket. The $y$ axis is
  scaled by the peak of $\bar y$ over the horizon rather than by $\bar y(T)$,
  since $\bar y$ is no longer monotone.

The formulas are in
[Time-Dependent Volatility §6](time-dependent-volatility.md#6-cheyette-with-time-dependent-volatility).
A one-bucket model runs the constant formulas bit for bit.

**Validation.**

| Oracle | Agreement | Test |
|---|---|---|
| 1F, $\beta = 0$ vs piecewise Hull–White | round-off | `tests/test_models/test_cheyette_piecewise.py` |
| 1F exact-$y$ PDE refinement | about first order (upwinded $y$ axis) | `tests/test_pde/test_cheyette_piecewise.py` |
| 1F frozen vs exact $y$ at $\beta = 0.5$ | gap below the exact grid's own error | same |
| 1F Monte Carlo | within MC error | `tests/test_mc/test_cheyette_piecewise_paths.py` |
| 2F, $\beta_i = 0$ vs piecewise G2++ | round-off | `tests/test_models/test_cheyette2f_piecewise.py`, `tests/test_pde/test_cheyette2f_piecewise.py`, `tests/test_mc/test_cheyette2f_piecewise_paths.py` |

In two factors, the Gaussian-limit means at very short $t$ are limited to about
$2.4\times10^{-13}$ relative by cancellation. The constant formula has the same
limit.

---

## References

- Cheyette, O. (1992). "Markov Representation of the Heath–Jarrow–Morton Model".
- Ritchken, P. & Sankarasubramanian, L. (1995). "Volatility Structures of Forward Rates and the Dynamics of the Term Structure". *Mathematical Finance* 5(1).
- Andersen, L. & Piterbarg, V. (2010). *Interest Rate Modeling*, Volume 2, ch. 13 (quasi-Gaussian models).
