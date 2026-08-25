# Hull-White Finite Differences: Bermudans and Callables

## Overview

The Hull-White trinomial tree and the exact-OU Monte-Carlo engine each have a
blind spot. The tree handles early exercise naturally but is awkward to
differentiate (its width $j_{\max}$ is a *shape*, so it must be pinned outside
the trace) and its accuracy is capped by date snapping. Monte Carlo is
differentiable and flexible but needs a regression proxy for the exercise
decision, which turns an exact policy into an estimated one.

The backward PDE has neither problem. Early exercise is a **pointwise
projection on the value function** — no regression, no policy error — and the
whole solve is a fixed-shape sequence of tridiagonal solves, so `jax.grad`
flows through it directly. That makes it the natural engine for **Bermudan
swaptions** and **callable/puttable bonds**, and the third independent route to
prices the other two engines already produce.

---

## 1 · Choosing the State Variable

Written directly in the short rate, the pricing PDE has a drift $\theta(t) - ar$
whose time dependence would force the mesh to move. The standard remedy is the
same change of variable the tree and the MC engine use — split off a
deterministic shift:

$$
r(t) = x(t) + \alpha(t), \qquad dx(t) = -a\,x(t)\,dt + \sigma\,dW(t), \quad x(0) = 0
$$

$$
\alpha(t) = f^M(0,t) + \frac{\sigma^2}{2a^2}\bigl(1 - e^{-at}\bigr)^2
$$

(See [Hull-White MC](hull-white-mc.md) §1 for the derivation; the identity that
makes the drift *exactly* $-ax$ with no residual term is
$\theta(t) = \alpha'(t) + a\,\alpha(t)$.)

Three things fall out, all of which the numerics want:

- the drift and diffusion are **time-independent**, so the mesh is built once;
- the state starts exactly at the **origin**, so the price is always read off at
  $x = 0$ — a fixed point, unlike the equity solvers whose read-off slides with
  spot and needs `stop_gradient` scaffolding to keep gamma well defined;
- all dependence on the initial curve is compressed into the scalar
  $\alpha(t)$.

For a claim with value $V(x,\tau)$, $\tau = T - t$, the backward equation is

$$
\frac{\partial V}{\partial \tau}
  = \tfrac{1}{2}\sigma^2 \frac{\partial^2 V}{\partial x^2}
  - a x \frac{\partial V}{\partial x}
  - \bigl(x + \alpha(t)\bigr) V
$$

Only the **discount** coefficient varies — and it varies in both space and
time, because it *is* the short rate. This is why the operator is assembled as
a stack of per-step bands rather than once.

---

## 2 · Why $\alpha$ Must Be Integrated, Not Sampled

The obvious discretisation evaluates $\alpha$ at each step's midpoint, which is
second-order accurate for smooth $\alpha$. It is not smooth. A log-linear
discount curve — VALAX's interpolation, and the market standard — has a
**piecewise-constant instantaneous forward** $f^M(0,t)$ that jumps at every
pillar. Midpoint sampling then leaves an $O(\Delta t)$ error on each step
straddling a pillar.

The symptom is unmistakable: on a flat curve the scheme converges cleanly at
second order, while on a sloped curve it **stalls**, plateauing at a
zero-coupon-bond repricing error of $\sim 4\times10^{-6}$ no matter how finely
time is refined. That is a violation of Hull-White's defining property — it is
supposed to fit the initial curve *exactly*. (Written up as entry 3 of
[Numerical Pitfalls](../architecture/numerical-pitfalls.md) — the flat-curve
fixture is what hid it.)

The fix is to average $\alpha$ exactly across each step. Both halves integrate
in closed form, and the market-forward half telescopes into a discount ratio:

$$
\int_{t_0}^{t_1} f^M(0,s)\,ds = \ln\frac{P^M(0,t_0)}{P^M(0,t_1)}
$$

so each step discounts by precisely the market forward discount factor across
it, for any curve shape. With `hw_alpha_average` in place, convergence is clean
second order on flat, sloped and humped curves alike:

| mesh $n_x = n_t$ | flat | sloped | humped |
|---|---|---|---|
| 100 | 1.31e-05 | 1.56e-05 | 4.68e-05 |
| 200 | 3.29e-06 | 3.91e-06 | 1.17e-05 |
| 400 | 8.23e-07 | 9.78e-07 | 2.94e-06 |
| 800 | 2.06e-07 | 2.45e-07 | 7.36e-07 |

---

## 3 · Boundaries Without a Far-Field Formula

Every Dirichlet boundary in the equity solvers relies on knowing the value
function's asymptotics in closed form. No such formula exists for a callable
bond: its far-field value depends on the entire remaining exercise schedule.

So instead of imposing a *value*, impose a *shape* — zero curvature at both
edges:

$$
\left.\frac{\partial^2 V}{\partial x^2}\right|_{x_{\min}}
= \left.\frac{\partial^2 V}{\partial x^2}\right|_{x_{\max}} = 0
$$

Linearly extrapolating the exterior ghost and folding it back into the stencil
zeroes the first row's sub-diagonal and the last row's super-diagonal, so no
boundary data is consumed at all. This is the 1-D analogue of the
$v = v_{\max}$ treatment in the Heston ADI operator, and it matches how
QuantLib's short-rate FD operators behave.

Two properties bound what it buys:

- it is **exact on affine fields** — where $V$ is linear in $x$ the
  extrapolated ghost *is* the true value, so the folded rows reproduce the
  operator to machine precision;
- the convection term at the edges degrades to a **one-sided** difference
  (forward at the lower edge, backward at the upper), hence first order there.

Both are harmless once the domain is wide enough that the edges carry
negligible probability, and empirically prices are flat in the domain width
from about four standard deviations outward.

---

## 4 · Discrete Events: Cashflows and Exercise

Coupons and exercise decisions enter through a per-step hook on the backward
sweep. Two details carry real money:

**Exercise is decided ex-coupon.** Call and put prices are quoted ex-coupon, so
a holder redeemed on a coupon date still collects that coupon. The projection
therefore runs *before* the coupon is added. Reversing the order undervalues a
callable bond by up to a full coupon — a bug that really occurred in the tree
implementation and was caught by the QuantLib harness.

**Cashflow dates are not snapped.** Snapping a coupon to the nearest time level
displaces it by up to $\Delta t/2$, an $O(\Delta t)$ error that dominates
everything else (it costs $\sim 2\times10^{-3}$ on a five-year bullet, three
orders of magnitude worse than the scheme's own error — entry 4 of
[Numerical Pitfalls](../architecture/numerical-pitfalls.md)). Instead a coupon due at
$t_c$ is attached to the nearest level $t_k$ and scaled by the **analytic**
Hull-White bond price $P(t_k, t_c \mid x)$, exact at every node and for either
sign of $t_c - t_k$. Cashflow timing then contributes no discretisation error
at all.

Exercise dates *are* snapped — a decision has to happen at a time level — but
that error is second order, since the exercise boundary is smooth in time.

For a Bermudan swaption the exercise value is likewise analytic: exercising at
$T_e$ enters the tail swap on the remaining fixed dates, worth

$$
\pm N\Bigl(1 - \sum_{i \ge e} c_i P(T_e, T_i \mid x)\Bigr),
\qquad c_i = K\tau_i + \delta_{in}
$$

at every node, straight from the affine bond price. So the *only* numerical
error in the whole scheme is in the continuation value.

---

## 5 · Validation

| Instrument | Oracle | Agreement |
|---|---|---|
| Fixed-rate bond | analytic curve price | 8e-6, second order |
| European swaption | Jamshidian (VALAX and QuantLib) | ~3e-5 relative |
| European swaption | `ql.FdHullWhiteSwaptionEngine` | ~3e-5 relative |
| Bermudan swaption | `ql.FdHullWhiteSwaptionEngine` | ~2e-4 relative |
| Bermudan swaption | `ql.TreeSwaptionEngine` | ~1.2e-3 relative |
| Callable / puttable bond | VALAX HW tree, `ql.TreeCallableFixedRateBondEngine` | < 5e-3 |

Two structural checks complement the numerical ones, and are sharper because
they hold *exactly* rather than to tolerance:

- a callable bond whose call is struck far out of the money reproduces the
  bullet bond to machine precision, isolating the exercise machinery from the
  diffusion;
- a Bermudan swaption with a single exercise date reproduces the European.

A note on the tree comparison tolerance: it is set by the **tree's** accuracy,
not the PDE's. Pricing an effectively option-free bond *on the tree* leaves
$-2.2\times10^{-3}$ to $+3.4\times10^{-5}$ of error depending on the step count —
non-monotone, because the count decides where dates land — against $8\times10^{-6}$
and cleanly second-order for the PDE. Likewise QuantLib's own FD and tree
engines disagree with each other by more than VALAX's PDE disagrees with
QuantLib's FD.

---

## 6 · Differentiability

The solve is a fixed-shape `lax.scan` of tridiagonal solves, so it composes
with `eqx.filter_jit` and `eqx.filter_grad` with nothing special required.
Unlike the trinomial tree — whose half-width $j_{\max}$ determines array shapes
and so must be computed outside the trace — the PDE recipes have no concrete
inputs at all: schedules are scattered with traced indices, so dates stay
traceable.

Autodiff sensitivities to both $a$ and $\sigma$ match central differences to
$10^{-8}$–$10^{-11}$ across callable bonds, European swaptions and Bermudan
swaptions — including through the exercise projection.

---

## References

- Hull & White (1990), "Pricing Interest-Rate-Derivative Securities."
- Brigo & Mercurio (2006), *Interest Rate Models — Theory and Practice*, ch. 3.
- Andersen & Piterbarg (2010), *Interest Rate Modeling*, vol. 3.
- in't Hout & Foulon (2010), "ADI finite difference schemes for option pricing
  in the Heston model with correlation."
