# Time-Dependent Volatility in the Short-Rate Models

How VALAX makes the volatility of its exact-fit short-rate models (Hull-White,
G2++, one- and two-factor Cheyette) piecewise constant in time, the closed-form
integrals every engine is built on, and how the closed form, the PDE, the Monte
Carlo and the trinomial tree consume them.

This page owns the shared mathematics. The model pages carry only what changes
for them and link back here:
[Hull-White swaptions](hull-white-swaptions.md),
[Hull-White PDE](hull-white-pde.md), [Hull-White MC](hull-white-mc.md),
[G2++](g2pp.md) and [Cheyette](cheyette.md).

Buckets: `valax/models/short_rate.py`.
Models: `valax/models/hull_white.py`, `valax/models/g2pp.py`,
`valax/models/cheyette.py`.
General tree: `valax/pricing/lattice/hull_white_tree.py`.

---

## 1 · The bucket convention

A volatility is a vector of **buckets**. With interior breakpoints
$0 < \tau_1 < \dots < \tau_{n-1}$, in year fractions from the curve's
reference date,

$$
\sigma(t) = \sigma_j \quad\text{for } t \in [\tau_j, \tau_{j+1}),
\qquad \tau_0 = 0,\ \tau_n = \infty,
$$

so the last value extends flat past the last breakpoint. Bucket lookup is
right-continuous: at $t = \tau_j$ the value is $\sigma_j$ (`bucket_index`, a
`searchsorted(..., side="right")`).

| Field | Shape | Nature |
|---|---|---|
| `volatility` (and G2++ / Cheyette-2F `volatility_ratio`) | `Float[Array, " n_buckets"]` | calibrated data |
| `volatility_times` (keyword-only) | `Float[Array, " n_buckets_minus_1"]` | **traced** data, not static |

Four facts follow from the design:

- **A constant volatility is one bucket.** Constructors promote a scalar to
  shape `(1,)` (`as_buckets`). The default breakpoints are an empty array,
  which is valid for exactly one bucket; several buckets without breakpoints
  raise at construction (`default_bucket_times`, `check_buckets`, which check
  shapes only and never read a traced value).
- **The bucket count is static, the breakpoints are not.** The count is an
  array shape, so it is fixed at trace time. The breakpoints are an ordinary
  pytree leaf. Two models for trades with different exercise dates therefore
  have the same tree structure and stack under `eqx.filter_vmap`. Only the
  count must agree.
- **All bucketed fields of a model share one set of breakpoints.** The
  `volatility_bucket_fields` registry names them: `volatility` for Hull-White
  and Cheyette, `volatility_x` and `volatility_ratio` for G2++, `volatility_1`
  and `volatility_ratio` for Cheyette-2F.
- **Breakpoints are an input, not a calibrated parameter.** In the per-trade
  co-terminal bootstrap they are the co-terminal expiries except the last,
  $\tau = (T_1, \dots, T_{n-1})$. The swaption expiring at $T_k$ then sees
  only buckets $0, \dots, k-1$ (bucket $k-1$ is $[T_{k-1}, T_k)$), so the fit
  is triangular. How that level enters the calibration chain is covered in
  [Calibration Chains](calibration-chains.md).

---

## 2 · The one primitive: decayed integrals

Every Gaussian quantity of these models is an integral of a piecewise-constant
function $v$ (for example $\sigma^2$, $\eta^2$ or $\sigma\eta$) against a
decaying exponential. For a rate $k > 0$ define (`decayed_integral`)

$$
I_k(t) = \int_0^t v(u)\,e^{-k(t-u)}\,du, \qquad
J_k(t) = \int_0^t I_k(s)\,ds .
$$

Inside bucket $j$, $I$ solves the linear ODE $I' = v_j - kI$. With
$\delta = t - \tau_j$,

$$
I_k(t) = I_k(\tau_j)\,e^{-k\delta} + \frac{v_j}{k}\bigl(1 - e^{-k\delta}\bigr),
$$

$$
J_k(t) = J_k(\tau_j) + \frac{I_k(\tau_j)}{k}\bigl(1 - e^{-k\delta}\bigr)
       + \frac{v_j}{k^2}\bigl(k\delta - 1 + e^{-k\delta}\bigr).
$$

The same recurrence, evaluated at $\delta = \tau_{j+1} - \tau_j$, tabulates the
breakpoint values $I_k(\tau_j)$ and $J_k(\tau_j)$. So everything is closed form,
there is no quadrature, and the cost is one short loop over the (static) bucket
count. Two numerical details keep it accurate for small $k\delta$:
$1 - e^{-x}$ is evaluated as `-expm1(-x)`, and $x - (1 - e^{-x})$ switches to its
Taylor series below $x = 10^{-3}$, where the closed form loses about
$2\varepsilon/x$ to cancellation.

Two derived quantities recur below:

- the **window** over $[s, t]$ follows from the semigroup property,

  $$
  K_k(s, t) = \int_s^t v(u)\,e^{-k(t-u)}\,du = I_k(t) - e^{-k(t-s)}\,I_k(s);
  $$

- the undecayed integral $\int_0^t v(u)\,du$ (`piecewise_integral`) is a
  cumulative sum over whole buckets plus a partial bucket. It gives the
  **exact step average** of a variance rate,

  $$
  \overline{v}(t_0, t_1) = \frac{1}{t_1 - t_0}\int_{t_0}^{t_1} v(u)\,du ,
  $$

  which is the right coefficient for a solver step that straddles a breakpoint.

---

## 3 · Hull-White with piecewise-constant volatility

### 3.1 State variance and the exact-fit shift

Hull-White is written, as in every VALAX engine, as a centred
Ornstein-Uhlenbeck state plus a deterministic shift:

$$
r(t) = x(t) + \alpha(t), \qquad
dx(t) = -a\,x(t)\,dt + \sigma(t)\,dW(t), \quad x(0) = 0 .
$$

The state is Gaussian with mean zero. Its conditional variance over $[s, t]$
is a window of §2 with $v = \sigma^2$ and $k = 2a$ (`hw_state_variance`):

$$
V(s, t) = \int_s^t \sigma(u)^2 e^{-2a(t-u)}\,du = I_{2a}(t) - e^{-2a(t-s)}\,I_{2a}(s),
$$

and the unconditional variance is $\mathrm{Var}[x(t)] = I_{2a}(t)$
(`hw_short_rate_variance`). For a constant $\sigma$ both reduce to the familiar
$\frac{\sigma^2}{2a}\bigl(1 - e^{-2a(t-s)}\bigr)$.

The exact-fit shift is the market forward plus a convexity term. With
$B(u, t) = (1 - e^{-a(t-u)})/a$ and
$e^{-a(t-u)}B(u, t) = (e^{-a(t-u)} - e^{-2a(t-u)})/a$, the convexity splits into
two decayed integrals (`hw_alpha`):

$$
\alpha(t) = f^M(0, t) + \int_0^t \sigma(u)^2 e^{-a(t-u)} B(u, t)\,du
          = f^M(0, t) + \frac{I_a(t) - I_{2a}(t)}{a}.
$$

For a constant $\sigma$ this is the textbook
$f^M(0,t) + \frac{\sigma^2}{2a^2}(1 - e^{-at})^2$. The exact time average of
$\alpha$ over a step, which the PDE and tree discount need
(`hw_alpha_average`), integrates both halves in closed form. The forward half
telescopes into $\ln P^M(0, t_0)/P^M(0, t_1)$, and the convexity half is
$\bigl(J_a - J_{2a}\bigr)/a$ differenced across the step. The reason for
averaging rather than sampling is in
[Hull-White PDE §2](hull-white-pde.md#2-why-alpha-must-be-integrated-not-sampled).

### 3.2 Bond price and zero-coupon bond option

Conditional on $r$ at time $t$, the bond is still affine (`hw_bond_price`):

$$
P(t, T \mid r) = A(t, T)\,e^{-B(t, T)\,r}, \qquad
\ln A(t, T) = \ln\frac{P^M(0, T)}{P^M(0, t)} + B(t, T)\,f^M(0, t)
            - \tfrac12\,B(t, T)^2\,I_{2a}(t).
$$

$B$ does not depend on $\sigma$, and the volatility enters only through the
state variance $I_{2a}(t)$. Two consequences hold for any $\sigma(t)$:

- $\ln P(T, S)$ is Gaussian with standard deviation
  $\sigma_p = B(T, S)\sqrt{I_{2a}(T)}$. That is the only change to the
  zero-coupon bond option (`hw_zcb_option_price`).
- Every bond is still strictly decreasing in $r$, so Jamshidian's
  decomposition carries over unchanged (§4.1).

---

## 4 · How each engine consumes $\sigma(t)$

The rule is the same in every engine: replace each place where the constant
formula used $\sigma^2\,\Delta t$ or $\frac{\sigma^2}{2a}(1 - e^{-2a\Delta t})$
by the **exact** integral over that interval. A step or a level that straddles a
breakpoint then carries exactly the variance it should. No engine needs its time
grid aligned to the breakpoints.

### 4.1 Closed form (Jamshidian)

The swaption is the sum of zero-coupon bond options struck at
$X_i = P(T_0, T_i \mid r^\star)$ (see
[Hull-White Swaptions](hull-white-swaptions.md)), with $\sigma_p$ from §3.2 and
the critical rate $r^\star$ solved against the piecewise bond of §3.2. The nested
Newton solve is guarded, so a failed solve gives a NaN price rather than a
runtime error ([Hull-White Swaptions §4](hull-white-swaptions.md#4-finding-rstar)).

### 4.2 PDE: exact step averages

The backward equation keeps the drift $-ax$ and the discount
$x + \alpha(t)$, and its diffusion becomes time dependent. On a uniform grid
$t_k = k\,\Delta t$, the operator of the step $[t_k, t_{k+1}]$ uses

$$
\tfrac12\,\overline{\sigma^2}_k\,V_{xx}, \qquad
\overline{\sigma^2}_k = \frac{1}{\Delta t}\int_{t_k}^{t_{k+1}} \sigma(u)^2\,du
$$

(`hw_mean_variance_rate`), together with the exact step average
$\bar\alpha_k$ of §3.1 in the discount.
Hull-White already assembles one operator per step, because the discount varies
in time, so the piecewise diffusion just adds one scalar per row
(`hw_operator_stack`). The mesh is still built once. It is centred at the
origin with half-width set by the state's standard deviation at the horizon,
$\sqrt{I_{2a}(T_h)}$.

The two-factor engines use the same rule. The G2++ rows (`g2pp_operator_rows`)
use the exact step averages of $\sigma^2$, $\eta^2$ and $\rho\sigma\eta$
(`g2pp_diffusion_rates`). The Cheyette rows (§6) use the step average of the
level, multiplied by the state-dependent factor. The 2-D rows are returned as a
function of the step index and assembled inside the ADI time loop, so memory
stays that of a single operator.

### 4.3 Monte Carlo: exact step variance

The centred state still has an exact Gaussian transition:

$$
x(t_{i+1}) \mid x(t_i) \sim \mathcal{N}\!\Bigl(x(t_i)\,e^{-a\Delta t},\;
V(t_i, t_{i+1})\Bigr),
$$

with the window $V$ of §3.1 computed per step (it is no longer the same for
every step). The short rate is $x + \alpha(t)$ with the piecewise $\alpha$, and
the numeraire is accumulated as in [Hull-White MC](hull-white-mc.md). The
scheme stays exact in distribution for any step size and any breakpoint
placement. G2++ factors the per-step $2\times2$ conditional covariance of §5.3
with one Cholesky decomposition per step. Cheyette freezes the step's local
volatility at $\sqrt{\overline{\sigma^2}}\,(1 + \beta x)$ (§6).

### 4.4 The general Hull-White trinomial tree

The classical Hull & White (1994) tree has a constant spacing
$\Delta x = \sigma\sqrt{3\Delta t}$ and branches on a first-order (Euler) mean
$-a\,j\,\Delta t$. Neither survives a time-dependent $\sigma$. VALAX builds the
general tree of Hull & White (2001) on the same centred state $x$
(`build_general_hull_white_tree`, `GeneralHullWhiteTree`), with exact moments.
The time grid is uniform, $t_i = i\,\Delta t$ with $\Delta t = T/n$.

**Level spacing.** Level $i$ has nodes $x_{i,j} = j\,\Delta x_i$ for
$j = -J, \dots, J$, and the spacing of the next level follows the exact step
variance:

$$
\Delta x_{i+1} = \sqrt{3\,V(t_i, t_{i+1})},
$$

so the spacing tracks $\sigma(t)$. Level 0 holds only the root and reuses the
first step's spacing.

**Branching.** The exact OU conditional mean from node $(i, j)$ is
$M = x_{i,j}\,e^{-a\Delta t}$. The middle branch targets the **nearest** node
$k = \operatorname{round}(M/\Delta x_{i+1})$ of the next level, and the other
two branches go to $k \pm 1$. With $\varepsilon = M/\Delta x_{i+1} - k$,

$$
p_u = \tfrac16 + \tfrac{\varepsilon^2 + \varepsilon}{2}, \qquad
p_m = \tfrac23 - \varepsilon^2, \qquad
p_d = \tfrac16 + \tfrac{\varepsilon^2 - \varepsilon}{2}.
$$

These match the first two conditional moments exactly. In units of
$\Delta x_{i+1}$ the mean displacement from $k$ is $p_u - p_d = \varepsilon$, and
the second moment is $p_u + p_d = \tfrac13 + \varepsilon^2$, so the variance is
$\tfrac13\Delta x_{i+1}^2 = V(t_i, t_{i+1})$. $p_u$ and $p_d$ are positive for
every $\varepsilon$ (their quadratics have no real roots), and $p_m \ge 0$ for
$|\varepsilon| \le \sqrt{2/3}$. Nearest-node branching keeps
$|\varepsilon| \le \tfrac12$, so all three probabilities are positive
($p_m \ge \tfrac{5}{12}$).

**Edges and width.** Mean reversion keeps the tree finite without a separate
edge rule. The reachable half-width propagates as

$$
w_{i+1} = \max_{|j| \le w_i}\Bigl|\operatorname{round}\Bigl(
  j\,\frac{\Delta x_i\,e^{-a\Delta t}}{\Delta x_{i+1}}\Bigr)\Bigr| + 1 ,
$$

and `hw_tree_width` returns $J = \max_i w_i$. When the ratio
$q_i = \Delta x_i e^{-a\Delta t}/\Delta x_{i+1}$ is below one, the width
stays bounded. A rough estimate is $1/(2(1 - q_i))$, about $1/(2a\Delta t)$
for a near-constant $\sigma$, but the rounding in the recursion moves it
noticeably. Measured with `hw_tree_width` on a 3-bucket σ(t):

| $a$ | $T$, `n_steps` | $J$ | classical $\lceil 0.1835/(a\Delta t)\rceil$ | $1/(2a\Delta t)$ |
|---|---|---|---|---|
| 0.03 | 10, 100 | 112 | 62 | 167 (not reached by $T$) |
| 0.03 | 30, 300 | 167 | 62 | 167 |
| 0.1 | 10, 100 | 51 | 19 | 50 |
| 0.3 | 10, 100 | 26 | 7 | 17 |

So the general tree is about 2.7–3.7× wider than the classical one, which
uses non-standard edge branching to stop earlier. A drop in $\sigma(t)$ makes
the next level finer ($q_i > 1$) and widens the tree. $J$ is an array shape, so
it is static. It is computed eagerly from a concrete model and must be passed in
as `j_max` to `jit` or differentiate a tree price. Targets are clipped to
$\pm(J-1)$, so a node whose nearest target would fall outside the tree branches
inward at the edge, with a larger $|\varepsilon|$. If that drives a probability
negative at a node that carries Arrow-Debreu mass, the tree is marked
`valid=False` and its prices are **NaN**. An undersized $J$ never produces a
wrong number.

**Exact fit.** The shifts $\alpha_i$ are fitted by Arrow-Debreu forward
induction, as in the constant tree. With $Q_{0,0} = 1$,

$$
\alpha_i = \frac{1}{\Delta t}\ln\frac{\sum_j Q_{i,j}\,e^{-x_{i,j}\Delta t}}{P^M(0, t_{i+1})},
\qquad
Q_{i+1,k} = \sum_j Q_{i,j}\,p_{(i,j)\to k}\,e^{-(x_{i,j} + \alpha_i)\Delta t},
$$

so rolling back a unit payoff from any level reprices $P^M(0, t_i)$ exactly.

**Products.** The swaption pricers (`swaption_tree_price`,
`bermudan_swaption_tree_price`) value the underlying tail swap analytically at
each node of the exercise level, using the affine bond of §3.2 at
$r = x + \alpha(t)$. The swap is valued at the true exercise date, and the level
is the one that date snaps to. By default the kinked payoff and exercise
projection are cell-averaged (`smooth_payoff=True`), as on the PDE
([Hull-White PDE §4](hull-white-pde.md#kinks-cell-averaged-payoff-and-exercise)).
Pointwise sampling makes a lattice converge in an oscillating way; with cell
averaging the tree converges cleanly at first order.

**Dispatch.** `build_hull_white_tree` sends a model with more than one bucket
to the general tree. A one-bucket model keeps the Hull & White (1994) builder,
bit for bit (§7).

---

## 5 · G2++ with time-dependent volatility

### 5.1 The volatility-ratio parametrisation

$$
r(t) = x(t) + y(t) + \varphi(t), \quad
dx = -a\,x\,dt + \sigma(t)\,dW_1, \quad
dy = -b\,y\,dt + \eta(t)\,dW_2, \quad
dW_1\,dW_2 = \rho\,dt,
$$

with the second volatility parametrised as a **ratio** of the first:

$$
\eta(t) = \lambda(t)\,\sigma(t).
$$

`G2PPModel` takes `volatility_x` ($\sigma$) and `volatility_ratio` ($\lambda$,
written $r(t)$ in the code docstrings) on shared buckets. $\eta$ is the
read-only property `volatility_y`, and a one-bucket ratio is broadcast to the
bucket count of $\sigma$. The reason for this choice is calibration: a per-trade
co-terminal bootstrap can fit **one** bucketed field ($\sigma$) while the ratio,
the correlation and the mean reversions stay fixed. The shape of the second
factor relative to the first is then held across buckets, and only the overall
level is bootstrapped. Cheyette-2F uses the same parametrisation (§6).

### 5.2 The variance term, bucket by bucket

The variance of $\int_t^T (x + y)\,du$ given the factors at $t$ (`g2pp_V`) is

$$
V(t, T) = \int_t^T \Bigl[\sigma^2 B_a(u, T)^2 + \eta^2 B_b(u, T)^2
          + 2\rho\sigma\eta\,B_a(u, T)\,B_b(u, T)\Bigr]du ,
\qquad B_z(u, T) = \frac{1 - e^{-z(T-u)}}{z}.
$$

In terms of §2 windows $K^v_k = K^v_k(t, T)$ at rates
$k \in \{0, a, 2a, b, 2b, a+b\}$ this is

$$
V = \frac{K^{\sigma^2}_0 - 2K^{\sigma^2}_a + K^{\sigma^2}_{2a}}{a^2}
  + \frac{K^{\eta^2}_0 - 2K^{\eta^2}_b + K^{\eta^2}_{2b}}{b^2}
  + \frac{2\rho}{ab}\bigl(K^{\sigma\eta}_0 - K^{\sigma\eta}_a - K^{\sigma\eta}_b
  + K^{\sigma\eta}_{a+b}\bigr),
$$

but that combination cancels catastrophically on a short window, with relative
error about $\varepsilon/(k\tau)^2$. VALAX evaluates it **bucket by bucket**
instead. On the part $[u_0, u_1]$ of bucket $j$ inside $[t, T]$, substitute
$s = T - u$:

$$
\int_{u_0}^{u_1}\bigl(1 - e^{-p(T-u)}\bigr)\bigl(1 - e^{-q(T-u)}\bigr)du
= H_{pq}(T - u_0) - H_{pq}(T - u_1),
\qquad
H_{pq}(s) = \int_0^s (1 - e^{-pu})(1 - e^{-qu})\,du ,
$$

so $V = \sum_j \bigl[\sigma_j^2\,\Delta H_{aa}/a^2 + \eta_j^2\,\Delta H_{bb}/b^2
+ 2\rho\,\sigma_j\eta_j\,\Delta H_{ab}/(ab)\bigr]$.

For $(p+q)s \ge 1$, $H$ uses its closed form
$s - \frac{1 - e^{-ps}}{p} - \frac{1 - e^{-qs}}{q} + \frac{1 - e^{-(p+q)s}}{p+q}$.
That form loses about $\varepsilon/(ps)^2$ on short windows, so below
$(p+q)s = 1$ it switches to the series

$$
H_{pq}(s) = s\sum_{n \ge 3}\frac{(-1)^{n+1}}{n!}\,A_{n-1},
\qquad A_k = (x + y)^k - x^k - y^k, \quad x = ps,\ y = qs,
$$

truncated at 24 terms (truncation below $10^{-20}$ relative at the switch).
Each $A_k$ is a sum of positive binomial terms, so it has no cancellation. It is
built by the recurrence

$$
A_{k+1} = (x + y)\,A_k + xy\,\bigl(x^{k-1} + y^{k-1}\bigr), \qquad A_1 = 0,
$$

whose terms are non-negative too. The three rate pairs $(a,a)$, $(b,b)$, $(a,b)$
at both window ends are evaluated in one broadcast call. Why the series is
written this way, rather than as the expanded double sum, is a compile-time
story told in
[Numerical Pitfalls §6](../architecture/numerical-pitfalls.md#6-graph-size-is-a-cost-an-unrolled-series-made-a-gradient-compile-for-seven-minutes).

### 5.3 Shift, moments and engines

Everything else is decayed integrals of §2 (with $I^v_k = I^v_k(t)$):

- **Exact-fit shift** (`g2pp_phi`):
  $\varphi(t) = f^M(0,t) + \frac{I^{\sigma^2}_a - I^{\sigma^2}_{2a}}{a}
  + \frac{I^{\eta^2}_b - I^{\eta^2}_{2b}}{b}
  + \rho\Bigl[\frac{I^{\sigma\eta}_b - I^{\sigma\eta}_{a+b}}{a}
  + \frac{I^{\sigma\eta}_a - I^{\sigma\eta}_{a+b}}{b}\Bigr]$.
- **Integrated shift**, from the exact-fit condition alone
  (`g2pp_phi_integral`):

  $$
  \int_0^t \varphi(s)\,ds = -\ln P^M(0, t) + \tfrac12\,V(0, t).
  $$

  Differenced across a step, this gives the exact per-step discount that the
  PDE applies and the Monte-Carlo numeraire integrates.
- **Conditional covariance** over $[s, t]$ (`g2pp_state_covariance`):
  windows at rates $2a$, $2b$ and $a+b$ of $\sigma^2$, $\eta^2$ and
  $\rho\sigma\eta$, with conditional means $x(s)e^{-a(t-s)}$ and
  $y(s)e^{-b(t-s)}$. This is the exact Monte-Carlo step covariance and, from
  $s = 0$, the PDE grid's sizing covariance.
- **$T$-forward moments** for the swaption (`g2pp_forward_measure_moments`):

  $$
  \mu_x = -\Bigl[\frac{I^{\sigma^2}_a - I^{\sigma^2}_{2a}}{a}
          + \rho\,\frac{I^{\sigma\eta}_a - I^{\sigma\eta}_{a+b}}{b}\Bigr],
  \qquad
  \mu_y = -\Bigl[\frac{I^{\eta^2}_b - I^{\eta^2}_{2b}}{b}
          + \rho\,\frac{I^{\sigma\eta}_b - I^{\sigma\eta}_{a+b}}{a}\Bigr],
  $$

  with $\sigma_x^2 = I^{\sigma^2}_{2a}(T)$, $\sigma_y^2 = I^{\eta^2}_{2b}(T)$ and
  $\rho_{xy} = \rho\,I^{\sigma\eta}_{a+b}(T)/(\sigma_x\sigma_y)$.

The Brigo-Mercurio swaption integral is **unchanged**. Only these moments and
the bond coefficients (through $V$) change
([G2++ §3](g2pp.md#3-european-swaptions-brigomercurio-integral)).

**One-factor limit.** With $\lambda = 0$ and $\rho = 0$ only the $\sigma^2$
terms survive. $V$ is then the Hull-White integrated variance and
$\varphi(t) = f^M(0,t) + (I_a - I_{2a})/a = \alpha(t)$, so G2++ is
piecewise Hull-White (§3).

---

## 6 · Cheyette with time-dependent volatility

The one-factor local volatility is $\sigma_r(t, x) = \sigma(t)\,(1 + \beta x)$,
with $\sigma(t)$ on buckets and the bucket chosen by $t$ (`cheyette_local_vol`).
The Cheyette engines get their Gaussian quantities from a **Gaussian twin**
built on the same buckets (`volatility_times` is copied).

- **Frozen-$y$ accumulator.** The deterministic accumulator solves
  $\dot{\bar y} = \sigma(t)^2 - 2\kappa\bar y$, so

  $$
  \bar y(t) = I_{2\kappa}(t) = \int_0^t \sigma(u)^2 e^{-2\kappa(t-u)}\,du ,
  $$

  which is exactly the Hull-White state variance of §3.1 with $a = \kappa$
  (`cheyette_y_gaussian`).
- **Gaussian-limit mean.** $m(t) = \int_0^t e^{-\kappa(t-s)}\bar y(s)\,ds
  = (I_\kappa(t) - I_{2\kappa}(t))/\kappa$ (`cheyette_gaussian_mean`). This is
  the Hull-White convexity, so $\alpha_{\mathrm{Ch}} = f^M + m$ is the HW twin's
  $\alpha$, and its exact step average is that twin's `hw_alpha_average`.
- **Engines.** The frozen-$y$ PDE row of step $k$ diffuses at
  $\overline{\sigma^2}_k\,\max(1 + \beta(\tilde x + m), 0)^2$
  (`cheyette_mean_variance_rate`), and the Monte Carlo freezes
  $\sigma_r = \sqrt{\overline{\sigma^2}_k}\,(1 + \beta x)$ over the step. The
  exact-$y$ PDE becomes time dependent. `cheyette_exact_operator_rows` builds
  one operator per step, with $\overline{\sigma^2}_k$ in both the $x$ diffusion
  and the $y$ drift $\sigma_r^2 - 2\kappa y$. The constant-only
  `cheyette_exact_operator_2d` raises for more than one bucket. The $y$ axis is
  scaled by the **peak** of $\bar y$ over $[0, T]$ rather than by $\bar y(T)$.
  Within bucket $j$, $\bar y$ relaxes monotonically towards
  $\sigma_j^2/2\kappa$, so it is no longer monotone overall, and the peak sits
  at a breakpoint or at $T$.

**Two factors.** `CheyetteModel2F` takes `volatility_1` ($\sigma_1$) and
`volatility_ratio` ($\lambda$, so $\sigma_2 = \lambda\sigma_1$, the read-only
`volatility_2`), exactly as G2++ does. The accumulator matrix and means are the
G2++ ones (`cheyette_y_gaussian_2f`, `cheyette_gaussian_mean_2f`):

$$
\bar y_{ij}(t) = \rho_{ij}\,I^{\sigma_i\sigma_j}_{\kappa_i + \kappa_j}(t),
\qquad
m_1 = \frac{I^{\sigma_1^2}_{\kappa_1} - I^{\sigma_1^2}_{2\kappa_1}}{\kappa_1}
    + \rho\,\frac{I^{\sigma_1\sigma_2}_{\kappa_1} - I^{\sigma_1\sigma_2}_{\kappa_1+\kappa_2}}{\kappa_2},
$$

and $m_2$ is the same with the factor indices swapped. They use the identity
$\int_0^t e^{-a(t-s)} I_k(s)\,ds = (I_a(t) - I_k(t))/(k - a)$. The frozen-$y$
discount $\varphi_{\mathrm{Ch}} = f^M + m_1 + m_2$ is the G2++ twin's
$\varphi$, so the exact step discount is the identity of §5.3. The 2-D rows
(`cheyette_operator_rows_2d`) use the step averages of $\sigma_1^2$,
$\sigma_2^2$ and $\rho\sigma_1\sigma_2$ (`cheyette_mean_variance_rates_2f`)
times the state factors. At $\beta_i = 0$ they are the G2++ rows.

---

## 7 · The constant case is bit-identical

Generalising a formula usually changes its rounding even when its value is
mathematically the same. Routing a one-bucket model through
`decayed_integral` would reproduce the constant closed form only to about one
ulp. For a library whose risk is compared run against run, that is a
regression. So every kernel branches on `is_constant(volatility)`, a check of
the array **shape** resolved at trace time. For one bucket it runs the original
constant formula **in its original evaluation order**, and the piecewise branch
is never traced.

Even the order matters. In `hw_bond_price`, reordering the terms of $\ln A$
left the value unchanged but reordered the reverse-mode accumulation, and a
gradient moved by one ulp. The constant branch therefore keeps the order
verbatim.

The guarantee is checked against a golden file captured **before** the change
(`tests/test_models/test_phase4_golden.py`, data in
`tests/data/phase4_golden.json`). Every engine's prices, gradients and paths are
recomputed and compared bit for bit, in a fast Hull-White sibling and a full
slow recomputation. The only exceptions are the deliberately re-parametrised
G2++ and Cheyette-2F entries, whose second volatility became a ratio (§5.1).
Their values stayed bit-identical, and they are held to $10^{-14}$ relative on
values and $10^{-12}$ on gradients. The largest difference measured at
migration was $1.4\times10^{-15}$, on one gradient.

---

## 8 · Validation

| Model / engine | Oracle | Agreement | Test |
|---|---|---|---|
| HW primitives ($V$, $\alpha$, $\bar\alpha$, bond) | Gauss-Legendre quadrature, incl. steps straddling breakpoints | $\le 10^{-12}$ rel | `test_models/test_hull_white_piecewise.py` |
| HW Jamshidian, $\sigma(t)$ | QuantLib `Gsr` + `Gaussian1dSwaptionEngine(gsr, 2048, 12.0, True, False, disc)` | $< 5\times10^{-6}$ rel (oracle's own error $1.8$–$2.5\times10^{-6}$) | `test_quantlib_comparison/test_hull_white_piecewise_ql.py` |
| Same, breakpoints shifted (anti-vacuity) | must **disagree** | $> 10^{-3}$ rel | same |
| HW Bermudan, $\sigma(t)$ | tree vs PDE vs QuantLib | $\sim 10^{-5}$ | same |
| HW PDE | Jamshidian, incl. a 5× steep $\sigma(t)$ | second order, finest $1.4\times10^{-5}$ | `test_pde/test_hull_white_piecewise.py` |
| HW MC | ZCB and Jamshidian | within $2\,$stderr | `test_mc/test_hull_white_piecewise_paths.py` |
| General HW tree | exact ZCB repricing (Arrow-Debreu fit) | $10^{-12}$ rel | `test_lattice/test_hull_white_tree_piecewise.py` |
| General HW tree, European | Jamshidian, Richardson $2V(2n) - V(n)$ | $\le 3\times10^{-6}$ | same |
| General HW tree, steep $\sigma(t)$ 3Y receiver | Jamshidian | irregular: $1.8\times10^{-5}$ (400/800), $3.5\times10^{-6}$ (800/1600) | same |
| General HW tree, undersized $J$ | must not price | NaN | same |
| G2++ $V$, $\varphi$, covariance, moments | quadrature, exact-fit identity, tower property | round-off | `test_models/test_g2pp_piecewise.py` |
| G2++ swaption, $\sigma(t)$ | MC ($2\,$stderr) and PDE | consistent | `test_pricing/test_g2pp_piecewise_swaptions.py`, `test_pde/test_g2pp_piecewise.py`, `test_mc/test_g2pp_paths.py` |
| G2++ one-factor limit ($\lambda = 0$, $\rho = 0$) | piecewise HW | round-off | `test_models/test_g2pp_piecewise.py` |
| Cheyette 1F, $\beta = 0$ | piecewise HW | round-off | `test_models/test_cheyette_piecewise.py` |
| Cheyette 1F exact-$y$ PDE | its own refinement | about first order (upwinded $y$) | `test_pde/test_cheyette_piecewise.py` |
| Cheyette 1F frozen vs exact $y$, $\beta = 0.5$ | exact-$y$ grid | gap below the exact grid's own error | same |
| Cheyette 1F MC | analytic / PDE | within MC error | `test_mc/test_cheyette_piecewise_paths.py` |
| Cheyette 2F, $\beta_i = 0$ | piecewise G2++ | round-off | `test_models/test_cheyette2f_piecewise.py`, `test_pde/test_cheyette2f_piecewise.py`, `test_mc/test_cheyette2f_piecewise_paths.py` |
| Nested root guards ($r^\star$, $\bar y$) | healthy path vs unguarded solve | bit-identical value and gradient; failure → NaN | `test_pricing/test_root_guard.py` |
| One-bucket models, all engines | golden file captured before the change | bit-identical (§7) | `test_models/test_phase4_golden.py` |

Notes on reading the table:

- **QuantLib's Jamshidian is not the oracle.**
  `Gaussian1dJamshidianSwaptionEngine` gets its bond options by fixed-grid
  numerical integration, and even at constant $\sigma$ it is $1$–$5\times10^{-4}$
  away from QuantLib's own exact `JamshidianSwaptionEngine`. The dense
  `Gaussian1dSwaptionEngine` above is the tight oracle, and its own error is
  measured against the exact engine on the constant case.
- **No QuantLib oracle exists for time-dependent two-factor G2.** G2++ is
  therefore triangulated through quadrature, the exact-fit identity, the tower
  property, Monte Carlo and the PDE, and pinned to QuantLib through its
  one-factor limit.
- **Short-time Cheyette-2F means.** The Gaussian means at very short $t$ are
  limited to about $2.4\times10^{-13}$ relative by cancellation. The constant
  formula has the same limit, so this is not a regression.

---

## References

- Hull, J. & White, A. (1990). "Pricing Interest-Rate-Derivative Securities". *Review of Financial Studies* 3(4).
- Hull, J. & White, A. (1994). "Numerical Procedures for Implementing Term Structure Models I: Single-Factor Models". *Journal of Derivatives* 2(1).
- Hull, J. & White, A. (2001). "The General Hull-White Model and Supercalibration". *Financial Analysts Journal* 57(6).
- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models — Theory and Practice*, §3.3 (Hull-White) and §4.2 (G2++).
- Andersen, L. & Piterbarg, V. (2010). *Interest Rate Modeling*, Volume 2 (Gaussian and quasi-Gaussian short-rate models with time-dependent parameters).
- Jamshidian, F. (1989). "An Exact Bond Option Formula". *Journal of Finance* 44(1).
