# Hull-White Monte Carlo: Exact Conditional Simulation

## Overview

The Hull-White one-factor model specifies the following risk-neutral
dynamics for the instantaneous short rate $r(t)$:

$$
dr(t) = [\theta(t) - a\,r(t)]\,dt + \sigma\,dW(t)
$$

where $a > 0$ is the mean-reversion speed, $\sigma > 0$ is the
short-rate volatility, and $\theta(t)$ is a deterministic time-dependent
drift calibrated to exactly fit the initial term structure.  The process
$\{r(t)\}$ is an **Ornstein-Uhlenbeck (OU) process** with a
time-varying level $\theta(t)/a$.

Because the SDE is linear and driven by a single Brownian motion,
$r(t)$ is **Gaussian at every future time**, with analytically available
conditional mean and variance.  VALAX exploits this to sample paths
*exactly* — drawing directly from the true conditional distribution —
rather than approximating the SDE with an Euler-Maruyama finite
difference.

---

## 1 · Transforming to a Centred OU Process

Define the **shifted process**

$$
x(t) \;=\; r(t) - \alpha(t)
$$

where $\alpha(t)$ is chosen so that $x$ is a zero-drift OU:

$$
dx(t) = -a\,x(t)\,dt + \sigma\,dW(t)
$$

The function $\alpha(t)$ absorbs all of the time-dependent drift and is
related to $\theta$ and the initial forward curve by the exact-fit condition
(see §3.3 of Brigo-Mercurio (2006)):

$$
\alpha(t) = f^M(0,t) + \frac{\sigma^2}{2a^2}\bigl(1 - e^{-at}\bigr)^2
$$

Note the denominator is $2a^2$, not $2a$ — the $1/a$ from $B(t)=(1-e^{-at})/a$
is squared along with the exponential. (Contrast the *conditional variance*
below, which genuinely carries $\sigma^2/2a$.)

where $f^M(0,t) = -\partial_t \ln P^M(0,t)$ is the market instantaneous
forward rate.

---

## 2 · Exact Conditional Distribution

For the centred OU process $x(t)$, the transition distribution over an
interval $\Delta t$ is:

$$
x(t + \Delta t) \;\big|\; x(t) \;\sim\; \mathcal{N}\!\left(
  x(t)\,e^{-a\Delta t},\;
  \frac{\sigma^2}{2a}\bigl(1 - e^{-2a\Delta t}\bigr)
\right)
$$

This is exact for **any** step size $\Delta t$ — there is no
$O(\Delta t)$ or $O(\sqrt{\Delta t})$ discretisation error.

Translating back to $r(t) = x(t) + \alpha(t)$ and letting
$t_i, t_{i+1} = t_i + \Delta t$ be consecutive grid points, the
conditional distribution of $r(t_{i+1})$ given $r(t_i)$ is:

$$
\boxed{
r(t_{i+1}) \;\big|\; r(t_i) \;\sim\; \mathcal{N}\!\left(
  \mu_i,\; v
\right)
}
$$

with

$$
\mu_i \;=\; \alpha(t_{i+1}) + \bigl[r(t_i) - \alpha(t_i)\bigr]\,e^{-a\Delta t}
$$

$$
v \;=\; \frac{\sigma^2}{2a}\bigl(1 - e^{-2a\,\Delta t}\bigr)
$$

The conditional mean $\mu_i$ has a transparent interpretation:

- $\alpha(t_{i+1})$ is the **long-run level** of $r$ at the next time
  point — the rate toward which $r$ mean-reverts.
- $[r(t_i) - \alpha(t_i)]$ is the **deviation** of the current rate from
  its long-run level; this decays exponentially at rate $a$.
- The formula ensures the exact-fit property: the model ZCB price
  $P(t, T | r)$ recovers $P^M(0, T)$ when the short rate equals the
  initial forward $f^M(0, 0)$.

---

## 3 · Why Exact Sampling Matters

The naive Euler-Maruyama discretisation approximates the transition as

$$
r(t_{i+1}) \approx r(t_i) + [\theta(t_i) - a\,r(t_i)]\,\Delta t
             + \sigma\,\sqrt{\Delta t}\,Z_i,\quad Z_i \sim \mathcal{N}(0,1)
$$

This introduces a **strong-order $\tfrac{1}{2}$** discretisation error:
the conditional variance matches the exact value $v$ to leading order,
but higher moments and path statistics only converge at rate
$\sqrt{\Delta t}$.  Pricing instruments that depend on the path integral
of $r$ (money-market account) or on the full path (callable bonds,
Bermudan swaptions) thus carry a systematic bias that cannot be removed
by increasing $n_\text{paths}$ — only by increasing $n_\text{steps}$.

Exact sampling removes this bias entirely.  The conditional distribution
at each step is drawn from the true Gaussian, so:

- **Zero strong-order discretisation error** in the short-rate marginal
  at each grid point.
- The **money-market numeraire** accumulated over the path converges
  faster with step count, since the only remaining error is in the
  numerical integration of $r$ between grid points (trapezoidal rule,
  $O(\Delta t^2)$ per interval).
- **Fewer steps are needed** to achieve a given pricing accuracy, saving
  computation without sacrificing correctness.

---

## 4 · Money-Market Numeraire Accumulation

The stochastic discount factor from time 0 to time $t_i$ is

$$
D(0, t_i) = \exp\!\left(-\int_0^{t_i} r(s)\,ds\right)
$$

Within each interval $[t_i, t_{i+1}]$, we approximate the integral
by the **trapezoidal rule**:

$$
\int_{t_i}^{t_{i+1}} r(s)\,ds \;\approx\;
\frac{\Delta t}{2}\bigl(r(t_i) + r(t_{i+1})\bigr)
$$

This is consistent with the exact conditional mean trajectory of the OU
bridge between $r(t_i)$ and $r(t_{i+1})$ and is unbiased to
$O(\Delta t^2)$.  The log-discount factor accumulates as:

$$
\ln D(0, t_{i+1}) = \ln D(0, t_i)
                  - \frac{\Delta t}{2}\bigl(r(t_i) + r(t_{i+1})\bigr)
$$

In the code, `log_discount_factors[:, i]` stores $\ln D(0, t_i)$ for
each path, initialised to 0 at $t_0 = 0$.

---

## 5 · Initial Condition

Under exact fit, the initial short rate is

$$
r(0) = f^M(0, 0)
$$

the market instantaneous forward rate at time zero.  This is the unique
starting point consistent with $P(0, T \mid r(0)) = P^M(0, T)$ for all
$T$, which can be verified from the affine ZCB formula in
`valax.models.hull_white.hw_bond_price`.

---

## 6 · ZCB Consistency Check

A fundamental model-validation test is:

$$
E^Q\!\left[\exp\!\left(-\int_0^T r(s)\,ds\right)\right]
= P^M(0, T)
$$

That is, the MC average of the money-market discount factors should
recover the market zero-coupon bond price.  Equivalently,

$$
E^Q\!\left[\exp\!\bigl(\text{log\_discount\_factors}[:, -1]\bigr)\right]
\;\approx\; P^M(0, T)
$$

This identity is tested in `tests/test_mc/test_hull_white_paths.py ::
TestZCBConsistency` against `hw_bond_price(model, r0, 0, T)`.

---

## 7 · Algorithm Summary

```
Input: HullWhiteModel(a, σ, P^M), T, n_steps, n_paths, key
───────────────────────────────────────────────────────────
dt    = T / n_steps
v     = σ²/(2a) · (1 − exp(−2a·dt))        # constant conditional variance
e_adt = exp(−a·dt)

# Precompute alpha grid (note σ²/(2a²) — see §1)
For i = 0 … n_steps:
  alpha[i] = f^M(0, t_i) + σ²/(2a²)·(1 − exp(−a·t_i))²

r[:, 0]   = alpha[0]  =  f^M(0, 0)         # initial condition (all paths)
ldf[:, 0] = 0

For i = 0 … n_steps − 1:
  Z ~ N(0, I_{n_paths})                      # independent standard normals
  mu = alpha[i+1] + (r[:, i] − alpha[i]) · e_adt
  r[:, i+1]   = mu + sqrt(v) · Z
  ldf[:, i+1] = ldf[:, i] − dt/2 · (r[:, i] + r[:, i+1])

Return HullWhitePathResult(short_rates=r, log_discount_factors=ldf)
```

The inner loop is compiled with `jax.lax.scan` for efficiency and full
JIT compatibility.

---

## 8 · References

- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models — Theory and
  Practice*, 2nd ed. Springer. §3.3 (Hull-White model), §3.3.2 (exact
  simulation of the OU bridge).
- Hull, J. & White, A. (1990). "Pricing Interest-Rate-Derivative
  Securities." *Review of Financial Studies*, 3(4), 573-592.
- Hull, J. & White, A. (1994). "Numerical Procedures for Implementing
  Term Structure Models: Single-Factor Models." *Journal of Derivatives*,
  2(1), 7-16.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
  Springer. §3.4 (exact simulation of mean-reverting processes).
