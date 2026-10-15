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
volatilities \(\sigma, \eta\), and the correlation \(\rho \in (-1, 1)\).
`G2PPModel` does not take \(\eta\) directly. Its fields are `volatility_x`
(\(\sigma\)) and `volatility_ratio` (\(\lambda = \eta/\sigma\)), and \(\eta\) is
the read-only property `volatility_y`. Both volatilities may be piecewise
constant in time on shared buckets (§9). The formulas on this page are written
for the constant case. The
shift \(\varphi\) is **not** free — it is pinned by the exact-fit condition,

$$
\varphi(t) = f^M(0, t)
    + \frac{\sigma^2}{2a^2}\bigl(1 - e^{-at}\bigr)^2
    + \frac{\eta^2}{2b^2}\bigl(1 - e^{-bt}\bigr)^2
    + \rho\frac{\sigma\eta}{ab}\bigl(1 - e^{-at}\bigr)\bigl(1 - e^{-bt}\bigr),
$$

so that the model reprices the initial discount curve \(P^M(0, \cdot)\) by
construction. The factors \(x, y\) are then zero-mean Ornstein–Uhlenbeck
processes whose coefficients carry no curve dependence (and are constant in
time for constant volatilities) — the parameterisation every numerical scheme
wants (`g2pp_phi`).

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
boundary without unrolling the iterations. The root-find is guarded in the same
way as the Hull–White critical rate, so a failed boundary gives a NaN price and
NaN gradient rather than a runtime error inside `jit` / `vmap`
([Hull–White Swaptions §4](hull-white-swaptions.md#4-finding-rstar)). With a
piecewise-constant \(\sigma(t), \eta(t)\) the formula is unchanged. Only the
moments \((\mu_x, \mu_y, \sigma_x, \sigma_y, \rho_{xy})\) and the bond
coefficients (through \(V\)) change (§9).

Discounting comes from the model's `initial_curve` — bound from the
`MarketState` by `valax.price` (or explicitly with `bind_curves` for a kernel
call) — so prices are automatically consistent with the exact-fitted curve.

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
size. With piecewise-constant volatilities the step covariance depends on where
the step starts (`g2pp_state_covariance`), so it is factored once **per step**.
It stays exact across a bucket breakpoint, and the whole deterministic drift
\(\int\varphi\) is then integrated exactly (§9).

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
integral of §3 (implicitly differentiable — exact autodiff Jacobian). It
returns a `CalibrationResult` whose `.value` is a curve-free `G2PPModel`
(`valax.price` binds the market curve at pricing time). The default settings
([`BASKET_SETTINGS`](../api/calibration.md#valax.calibration.basket.BASKET_SETTINGS))
are a line-searched BFGS minimiser of the sum of squares, as for Hull–White:
the Brigo–Mercurio pricer nests a root-find (the critical boundary
\(\bar y(x)\)) whose implicit derivative is non-finite far from the solution,
where an undamped Levenberg–Marquardt step can land; BFGS's line search keeps
the iterates where the pricer is defined. Levenberg–Marquardt is still
available via `settings=SolverSettings(method="levenberg_marquardt", ...)`.

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

Validation with time-dependent volatilities is listed in §9.

---

## 8 · PDE (2-D finite differences)

The headline decorrelation-sensitive exotic — the **Bermudan swaption** — has no
closed form. VALAX prices it (and its European sibling) by solving the G2++
backward PDE on the same 2-D ADI substrate built for Heston. For a claim value
\(V(t, x, y)\) in the two centred factors,

$$
\frac{\partial V}{\partial t}
  + \tfrac12\sigma^2 V_{xx} + \tfrac12\eta^2 V_{yy} + \rho\sigma\eta\, V_{xy}
  - a x\, V_x - b y\, V_y
  - \bigl(x + y + \varphi(t)\bigr) V = 0 ,
$$

read off at the origin \((x, y) = (0, 0)\) since both factors start at zero.

**The cross term.** \(\rho\sigma\eta\, V_{xy}\) is the whole point of a two-factor
model — it is what lets tenor rates decorrelate — and it is exactly the term the
Heston recipe already handles. It enters as a *constant* mixed coefficient into
`g2pp_operator_2d`, is stored as the explicit-only operator \(A_0\), and is
zeroed on the four domain edges (the in't Hout–Foulon treatment). The
Craig–Sneyd / Hundsdorfer–Verwer correctors recover second order in it.

**Factoring \(\varphi(t)\) (the exact-fit discipline).** The deterministic shift
is spatially uniform, so a discount \(-\varphi(t)V\) *commutes* with the spatial
operator. Writing \(V = D(t)\,W\) with \(D'/D = \varphi\) factors it out: \(W\)
solves the same PDE with the **state-only** discount \(x + y\) — a
*time-independent* operator, built once, no per-step stacking — and the
deterministic part re-enters as a per-step scalar factor
\(\exp\!\big(-\!\int_{t_k}^{t_{k+1}}\varphi\,ds\big)\) applied along the sweep.
The step integral is evaluated **exactly**: the market-forward part telescopes
into a discount-factor ratio \(\ln P^M(0,t_k)/P^M(0,t_{k+1})\) and the
convexity/cross parts have closed-form antiderivatives. This is the two-factor
analogue of Hull–White's exact step-average; a midpoint sample would inject a
curved-forward bias and break the model's exact fit to the initial curve. As a
result the PDE reprices a plain coupon bond off its own input curve to ~1e-6.

**Boundaries.** Neither factor axis has a closed-form far-field value once the
instrument carries optionality, so both impose zero curvature
(`apply_linearity_bc_2d`, the 2-D analogue of the Hull–White edge fold). The
\((x, y)\) domain is sized from the factors' terminal covariance
(`g2pp_factor_covariance`); the factors are zero-mean, so the grid is centred at
the origin.

**Exercise.** The European terminal payoff is \(\max(\text{swap PV}, 0)\); the
Bermudan projects \(\max(\text{continuation}, \text{exercise})\) at each snapped
exercise level through the stepper's `event_fn` seam. The exercise (tail-swap)
value is computed *analytically* at every node from the affine bond price
\(P(t, T\mid x, y) = A\,e^{-B_a x - B_b y}\), so exercise decisions carry no
discretisation error — the only numerical error is in the continuation value.

Registered recipes (via `pde_price_dispatch`): `FixedRateBond` (the option-free
*calibration* of the scheme), `Swaption`, and `BermudanSwaption`. The European
PDE agrees with the analytic §3 price to a few \(10^{-3}\), and both the
European and Bermudan agree with QuantLib's independent `ql.FdG2SwaptionEngine`
to within its ~1% grid tolerance.

With piecewise-constant volatilities the diffusion block is no longer constant.
`g2pp_operator_rows` builds one operator per backward step from the exact step
averages of \(\sigma^2\), \(\eta^2\) and \(\rho\sigma\eta\)
(`g2pp_diffusion_rates`), and assembles it inside the ADI loop. The constant
`g2pp_operator_2d` raises for more than one bucket. The \(\varphi\) factoring
above is unchanged, and its step integral now comes from the exact-fit identity
\(\int_0^t\varphi = -\ln P^M(0,t) + \tfrac12 V(0,t)\) (§9).

---

## 9 · Time-dependent volatility

\(\sigma(t)\) and the ratio \(\lambda(t)\) are piecewise constant on shared
buckets, with \(\eta(t) = \lambda(t)\,\sigma(t)\). The ratio parametrisation
lets a per-trade co-terminal bootstrap fit one bucketed field (\(\sigma\)) while
\(\lambda\), \(\rho\), \(a\) and \(b\) stay fixed. Every Gaussian quantity on
this page then becomes a closed-form sum of decayed integrals: \(V(t,T)\)
(evaluated bucket by bucket, with a cancellation-free series on short windows),
\(\varphi\), its integral, the conditional covariance, and the forward-measure
moments. The derivations live in one place,
[Time-Dependent Volatility §5](time-dependent-volatility.md#5-g2-with-time-dependent-volatility).
A one-bucket model runs the constant formulas above bit for bit.

**Validation.** No QuantLib oracle exists for a time-dependent two-factor G2
model. The piecewise model is therefore checked against:

- Gauss-Legendre quadrature of \(V\), \(\varphi\) and the covariances;
- the exact-fit identity \(P^M(0,t) = e^{-\int_0^t\varphi + \frac12 V(0,t)}\);
- the tower property;
- Monte Carlo (within \(2\,\)stderr) and the PDE;
- its **one-factor limit** (\(\lambda = 0\), \(\rho = 0\)), which reproduces
  piecewise Hull–White.

The tests are `tests/test_models/test_g2pp_piecewise.py`,
`tests/test_pricing/test_g2pp_piecewise_swaptions.py`,
`tests/test_pde/test_g2pp_piecewise.py` and `tests/test_mc/test_g2pp_paths.py`.
The one-factor limit ties the piecewise model back to the QuantLib-validated
Hull–White.

---

## References

- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models — Theory and Practice*, ch. 4 (§4.2).
- Hull, J. & White, A. (1994). "Numerical Procedures for Implementing Term Structure Models II: Two-Factor Models".
