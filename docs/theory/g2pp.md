# G2++: Two-Factor Gaussian and Decorrelation

Why a second stochastic factor is what lets rates of different maturities move
imperfectly together, how far that decorrelation actually goes, and how VALAX
prices, simulates and calibrates the G2++ model.

Model: `valax/models/g2pp.py`.
Analytic swaption: `valax/pricing/analytic/g2pp_swaptions.py`.
Monte Carlo: `valax/pricing/mc/g2pp_paths.py`.
Calibration: `valax/calibration/g2pp.py`.

---

## 1 · The model

G2++ is the two-additive-factor Gaussian short-rate model — equivalent to the
two-factor Hull–White (HW-2F). The short rate is the sum of two mean-reverting
factors plus a deterministic shift that exact-fits the initial curve:

$$
r(t) = x(t) + y(t) + \varphi(t)
$$

$$
dx(t) = -a\,x(t)\,dt + \sigma\,dW_1(t), \quad x(0) = 0
$$

$$
dy(t) = -b\,y(t)\,dt + \eta\,dW_2(t), \quad y(0) = 0,
\qquad dW_1(t)\,dW_2(t) = \rho\,dt
$$

The five free parameters are the mean-reversion speeds \(a, b\), the factor
volatilities \(\sigma, \eta\), and the correlation \(\rho \in (-1, 1)\). The
shift \(\varphi\) is **not** free — it is pinned by the exact-fit condition,

$$
\varphi(t) = f^M(0, t)
    + \frac{\sigma^2}{2a^2}\bigl(1 - e^{-at}\bigr)^2
    + \frac{\eta^2}{2b^2}\bigl(1 - e^{-bt}\bigr)^2
    + \rho\frac{\sigma\eta}{ab}\bigl(1 - e^{-at}\bigr)\bigl(1 - e^{-bt}\bigr),
$$

so that the model reprices the initial discount curve \(P^M(0, \cdot)\) by
construction. The factors \(x, y\) are then zero-mean Ornstein–Uhlenbeck
processes with time-independent coefficients — the parameterisation every
numerical scheme wants (`g2pp_phi`).

The model stays affine/Gaussian, so the zero-coupon bond is closed-form given
the two factors (`g2pp_bond_price`):

$$
P(t, T \mid x, y)
    = \frac{P^M(0, T)}{P^M(0, t)}
      \exp\!\Bigl[\tfrac{1}{2}\bigl(V(t, T) - V(0, T) + V(0, t)\bigr)
                    - B(a, t, T)\,x - B(b, t, T)\,y\Bigr],
$$

with \(B(z, t, T) = (1 - e^{-z(T-t)})/z\) and the Gaussian variance term
\(V(t, T)\) (variance of \(\int_t^T r\,du\), `g2pp_V`, Brigo–Mercurio §4.2):

$$
V(t, T)
= \frac{\sigma^2}{a^2}\Bigl[(T-t) + \tfrac{2}{a}e^{-a(T-t)}
    - \tfrac{1}{2a}e^{-2a(T-t)} - \tfrac{3}{2a}\Bigr]
+ \frac{\eta^2}{b^2}\Bigl[\cdots b \cdots\Bigr]
+ 2\rho\frac{\sigma\eta}{ab}\Bigl[(T-t)
    + \tfrac{e^{-a(T-t)} - 1}{a}
    + \tfrac{e^{-b(T-t)} - 1}{b}
    - \tfrac{e^{-(a+b)(T-t)} - 1}{a+b}\Bigr].
$$

At \(t = 0\), \(x = y = 0\) the bracket vanishes and \(P = P^M(0, T)\) to
machine precision.

---

## 2 · Decorrelation — and to what extent

The word "decorrelate" hides a subtlety, because **two different correlations**
live in G2++ and are easy to conflate:

1. **\(\rho\)** — the instantaneous correlation between the two driving Brownian
   motions \(W_1, W_2\). A single scalar, a *model input*.
2. **The tenor-rate correlation** — the correlation between forward/zero rates
   of *different* maturities. A *model output* that HW-1F forces to be exactly
   1, and that the second factor relaxes below 1.

These are not the same object. The instantaneous forward rate loads on the two
factors as

$$
df(t, T) \;\longleftarrow\; \sigma\,e^{-a(T-t)}\,dW_1 + \eta\,e^{-b(T-t)}\,dW_2,
$$

so for two maturities \(T_1, T_2\) the instantaneous correlation is

$$
\mathrm{Corr}\bigl(df(t,T_1), df(t,T_2)\bigr) =
\frac{\sigma^2 e^{-a(u_1+u_2)} + \eta^2 e^{-b(u_1+u_2)}
      + \rho\sigma\eta\bigl(e^{-a u_1 - b u_2} + e^{-b u_1 - a u_2}\bigr)}
     {\sqrt{\sigma^2 e^{-2a u_1} + \eta^2 e^{-2b u_1} + 2\rho\sigma\eta\,e^{-(a+b)u_1}}
      \;\sqrt{\;\cdots u_2 \cdots\;}},
$$

with \(u_i = T_i - t\). Because \(a \neq b\), the two exponentials decay at
different rates, so different maturities receive **differently-shaped** shocks
and the correlation drops below 1. That is the decorrelation.

**How far does it go?** The decorrelation is only partial and is *bounded by*
\(\rho\):

- At \(\rho = \pm 1\), \(W_1\) and \(W_2\) are the same Brownian motion. The
  diffusion collapses to a single shock
  \((\sigma e^{-a(T-t)} \pm \eta e^{-b(T-t)})\,dW\), every forward rate is
  driven by one source of randomness, and **all tenor correlations return to
  1** — the model degenerates to a one-factor world.
- Genuine decorrelation therefore requires \(|\rho| < 1\) **and** \(a \neq b\).
  The further \(\rho\) sits from \(\pm 1\) (and the wider the split between
  \(a\) and \(b\)), the more the two factors act as independent "level" and
  "slope" shapes, and the lower the short/long correlation can go.
- In practice \(\rho\) is calibrated **strongly negative** (\(\approx -0.7\) to
  \(-0.95\)); that is what buys the low long/short correlation the
  decorrelation-sensitive instruments need.

!!! warning "\(\rho\) is never 'removed'"
    Decorrelation does **not** mean \(W_1\) and \(W_2\) are made independent.
    \(\rho\) is a preserved input; what becomes imperfectly correlated is the
    set of *tenor rates*, to an extent that shrinks back toward zero as
    \(|\rho| \to 1\). A third, purely numerical sense of "decorrelate" appears
    in the Monte-Carlo scheme (§4), where the correlated pair is Cholesky-split
    into independent draws — a change of basis that preserves \(\rho\) exactly.

---

## 3 · European swaptions (Brigo–Mercurio integral)

Under **one** factor, every bond price at expiry is monotone in the single
state variable, so Jamshidian's trick collapses the coupon-bond option to a
finite sum of closed forms (see
[Hull–White Swaptions](hull-white-swaptions.md)). With **two** factors the
exercise region is a curve in \((x, y)\)-space, the bond-price inequalities no
longer flip together, and Jamshidian fails.

Brigo & Mercurio (2006), §4.2.4, give the exact price as a **one-dimensional
integral** over the first factor with a Jamshidian-style critical boundary on
the second. For a payer swaption (a put on the coupon bond),

$$
\text{PS} = N\,P(0, T)\int_{-\infty}^{\infty}
    \frac{e^{-\frac12\left(\frac{x-\mu_x}{\sigma_x}\right)^2}}{\sigma_x\sqrt{2\pi}}
    \Bigl[\Phi(-h_1(x)) - \sum_{i} \lambda_i(x)\,e^{\kappa_i(x)}\,\Phi(-h_2^i(x))\Bigr]\,dx,
$$

where \((\mu_x, \mu_y, \sigma_x, \sigma_y, \rho_{xy})\) are the \(T\)-forward
measure moments of \((x(T), y(T))\), and for each \(x\) the exercise boundary
\(\bar y(x)\) solves \(\sum_i c_i A_i(x)\,e^{-B(b,T,t_i)\,y} = 1\). VALAX
evaluates the outer integral with **64-node Gauss–Hermite quadrature** and
finds \(\bar y(x)\) with an `optimistix` Newton root-find at each node. Because
the root-find is implicitly differentiable, `jax.grad` flows through the
boundary without unrolling the iterations.

Discounting comes from `model.initial_curve`, so prices are automatically
consistent with the exact-fitted curve.

!!! note "Not the decorrelation payoff"
    A *single* ATM swaption's value **rises** with \(\rho\) (the factors
    reinforce, lifting the forward swap-rate variance). Decorrelation
    (\(\rho < 0\)) instead *lowers* a lone swaption — its benefit shows up in
    genuinely spread-sensitive payoffs like the CMS-spread swap of §5.

---

## 4 · Monte Carlo (exact two-factor scheme)

Conditional on the factors at \(t\), the pair \((x(t+\Delta t), y(t+\Delta t))\)
is jointly Gaussian with means \(x(t)e^{-a\Delta t}, y(t)e^{-b\Delta t}\) and a
time-homogeneous \(2\times2\) covariance (`g2pp_factor_covariance`):

$$
\mathrm{Var}[x] = \tfrac{\sigma^2}{2a}(1 - e^{-2a\Delta t}), \quad
\mathrm{Var}[y] = \tfrac{\eta^2}{2b}(1 - e^{-2b\Delta t}), \quad
\mathrm{Cov}[x, y] = \rho\tfrac{\sigma\eta}{a+b}(1 - e^{-(a+b)\Delta t}).
$$

`generate_g2pp_paths` draws the increment by Cholesky-factoring that matrix once
and applying it to standard normals — the two-factor analogue of the exact
Hull–White scheme. The scheme is unbiased in the factor marginals at any step
size.

**Exact-forward discounting.** The deterministic forward part of the drift is
integrated *exactly* over each step — it telescopes to the market
discount-factor ratio \(\ln P^M(0, t_i)/P^M(0, t_{i+1})\) — so a curved initial
forward introduces no repricing bias (the two-factor analogue of Hull–White's
exact step-average). Only the small, smooth convexity and stochastic
\(x + y\) parts use the trapezoidal rule. The martingale check
\(\mathbb{E}[D(0, T)] = P^M(0, T)\) then holds within Monte-Carlo error on both
flat and steep curves.

Registered recipes (via `mc_price_dispatch`): `FixedRateBond`,
`FloatingRateBond`, `Swaption`, and the decorrelation-sensitive `CMSSpreadSwap`.

---

## 5 · CMS-spread swaps (the decorrelation payoff)

The `CMSSpreadSwap` is a steepener / flattener: each period pays the spread
between two swap rates of different tenors (e.g. 10Y − 2Y) net of a fixed
strike. Under G2++ both CMS rates are computed **analytically** from each path's
factors,

$$
S^{\text{tenor}}(t) = \frac{1 - P(t, t + \text{tenor} \mid x, y)}
                           {\sum_{j} P(t, t + j \mid x, y)},
$$

so the payoff depends on the *joint* distribution of short- and long-tenor
rates — exactly what the second factor controls. Empirically the steepener PV
is **monotone in \(\rho\)** (well outside Monte-Carlo error), and a steepener
and flattener on shared paths sum to zero. A one-factor model cannot produce
this sensitivity at all.

---

## 6 · Calibration

G2++ has five free parameters, so it is fitted jointly to an ATM /
co-terminal swaption surface with `calibrate_g2pp`, using the semi-analytic
integral of §3 (implicitly differentiable — exact autodiff Jacobian). The
default solver is a line-searched `optimistix.BFGS` minimiser of the sum of
squares; as for Hull–White, Levenberg–Marquardt is deliberately not offered
because the residual closes over a *sequence* of instrument pytrees, which
trips `optimistix` 0.1.0.

!!! note "Weak identification and degenerate basins"
    The two mean reversions \(a, b\) are only weakly pinned by an ATM surface
    (the classic mean-reversion-vs-vol degeneracy), and the objective has
    degenerate basins at \(\sigma \to 0\) with \(\rho \to \pm 1\). Desk practice
    is to **fix \(a, b\)** (historical or co-terminal) and start the vols and
    \(\rho\) from a sensible prior. Pass `fixed_params` to pin any subset.

---

## 7 · Validation

| Oracle | Agreement | What it tests |
|---|---|---|
| `ql.G2SwaptionEngine` | `< 1e-4` rel | Same B–M integral — pins conventions |
| `ql.FdG2SwaptionEngine` (50×100×100) | `< 1.5e-2` rel | **Independent** 2-D finite-difference method |
| Payer − receiver parity | `< 1e-9` abs | Exact-fit: parity is pure curve, model-free |
| Hull–White reduction (\(\eta \to 0, \rho = 0\)) | qualitative | ZCB and \(\varphi\) collapse to HW-1F |
| MC ↔ analytic swaption | \(|z| < 3.5\) | Independent scheme sharing only the ZCB |
| \(V(t,T)\) vs covariance quadrature | `< 1e-4` rel | Variance term derived independently |
| Synthetic calibration round-trip | vols/\(\rho\) recovered | Pricer + transforms + residual assembly |
| CMS steepener \(+\) flattener | `= 0` | Sign / discounting conventions |

---

## References

- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models — Theory and Practice*, ch. 4 (§4.2).
- Hull, J. & White, A. (1994). "Numerical Procedures for Implementing Term Structure Models II: Two-Factor Models".
