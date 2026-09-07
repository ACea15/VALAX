# Cheyette: Quasi-Gaussian Skew and a Low-Dimensional HJM

How a state-dependent short-rate volatility buys a genuine rate **skew** while
keeping the term structure a finite-dimensional Markov process, why the
zero-coupon bond stays affine regardless of that volatility, and how VALAX
simulates and prices the one- and two-factor Cheyette models.

Model: `valax/models/cheyette.py`.
Monte Carlo: `valax/pricing/mc/cheyette_paths.py`.
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
\sigma_r(t, x) = \sigma\,(1 + \beta\,x),
$$

where $\sigma$ sets the level and the skew parameter $\beta$ tilts the
volatility with the state. Negative $\beta$ lowers volatility as rates rise (a
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

## References

- Cheyette, O. (1992). "Markov Representation of the Heath–Jarrow–Morton Model".
- Ritchken, P. & Sankarasubramanian, L. (1995). "Volatility Structures of Forward Rates and the Dynamics of the Term Structure". *Mathematical Finance* 5(1).
- Andersen, L. & Piterbarg, V. (2010). *Interest Rate Modeling*, Volume 2, ch. 13 (quasi-Gaussian models).
