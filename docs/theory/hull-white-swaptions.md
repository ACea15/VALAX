# Hull–White Swaptions: Jamshidian Decomposition

How VALAX prices European swaptions in closed form under the one-factor
Hull–White model, and why that closed form is what makes the model
*calibratable*.

Implementation: `valax/pricing/analytic/hull_white_swaptions.py`.
Calibration: `valax/calibration/hull_white.py`.

---

## 1 · The problem

A European payer swaption expiring at \(T_0\) on a fixed leg paying \(K\tau_i\)
at dates \(T_1 < \dots < T_n\) has time-\(T_0\) payoff

$$
N\Bigl(1 - \sum_{i=1}^{n} c_i\,P(T_0, T_i)\Bigr)^{+},
\qquad
c_i = K\tau_i + \mathbf{1}_{\{i=n\}}
$$

The quantity in brackets is one minus the value of a **coupon bond** with cash
flows \(c_i\). So a payer swaption is a *put* on a coupon bond struck at par,
and a receiver swaption is a *call*.

Options on coupon bonds have no general closed form, because the payoff depends
on a whole vector of correlated bond prices. Valuing it directly means an
\(n\)-dimensional integral.

---

## 2 · Jamshidian's observation

In a **one-factor** model every bond price at \(T_0\) is a deterministic
function of the single state variable \(r(T_0)\). Under Hull–White,

$$
P(T_0, T_i \mid r) = A(T_0,T_i)\,e^{-B(T_0,T_i)\,r},
\qquad B(t,T) = \frac{1 - e^{-a(T-t)}}{a} > 0
$$

so each \(P(T_0,T_i\mid r)\) is **strictly decreasing** in \(r\), and therefore
so is the coupon bond

$$
CB(r) = \sum_{i=1}^{n} c_i\,P(T_0,T_i \mid r)
$$

Strict monotonicity means there is a **unique** critical rate \(r^\star\) with
\(CB(r^\star) = 1\). Define the corresponding bond prices
\(X_i = P(T_0, T_i \mid r^\star)\).

Now the key step. Because *every* \(P_i\) decreases in \(r\),

$$
CB(r) < 1 \iff r > r^\star \iff P(T_0,T_i\mid r) < X_i \;\;\text{for every } i
$$

All \(n\) inequalities flip **at the same instant**. The single joint exercise
event therefore factorises into \(n\) individual ones, and

$$
\Bigl(1 - CB(r)\Bigr)^{+}
= \sum_{i=1}^{n} c_i\,\Bigl(X_i - P(T_0,T_i\mid r)\Bigr)^{+}
$$

This is an *exact pathwise identity*, not an approximation. The coupon-bond
option has become a portfolio of zero-coupon bond options — and those do have a
closed form.

!!! note "Why this needs one factor"
    With two or more factors the exercise region is a curve or surface in state
    space rather than a single threshold, the inequalities no longer flip
    together, and the decomposition fails. This is exactly why G2++ swaptions
    need a numerical integration instead.

---

## 3 · The zero-coupon bond option

Under Hull–White, \(\ln P(T,S)\) is Gaussian under the \(T\)-forward measure
with standard deviation

$$
\sigma_p = \sigma\,\sqrt{\frac{1 - e^{-2aT}}{2a}}\;B(T,S)
$$

Writing \(h = \dfrac{1}{\sigma_p}\ln\dfrac{P(0,S)}{P(0,T)X} + \dfrac{\sigma_p}{2}\),

$$
ZBC = P(0,S)\,\Phi(h) - X\,P(0,T)\,\Phi(h - \sigma_p)
$$

$$
ZBP = X\,P(0,T)\,\Phi(-h + \sigma_p) - P(0,S)\,\Phi(-h)
$$

These are Black formulas with the *bond* as underlying.

!!! note "Time-dependent volatility"
    With a piecewise-constant \(\sigma(t)\) only \(\sigma_p\) changes. It
    becomes \(\sigma_p = B(T,S)\sqrt{I_{2a}(T)}\), where
    \(I_{2a}(T) = \int_0^T \sigma(u)^2 e^{-2a(T-u)}\,du\) is the variance of the
    state at expiry. \(B\) does not involve \(\sigma\), so every bond is still
    monotone in \(r\) and the decomposition of §2 holds unchanged. The derivation
    is in
    [Time-Dependent Volatility §3](time-dependent-volatility.md#3-hull-white-with-piecewise-constant-volatility).

Summing gives the swaption:

$$
\text{payer} = N\sum_{i=1}^{n} c_i\,ZBP(T_0, T_i, X_i),
\qquad
\text{receiver} = N\sum_{i=1}^{n} c_i\,ZBC(T_0, T_i, X_i)
$$

---

## 4 · Finding \(r^\star\)

\(CB(r) = 1\) is a smooth, strictly monotone scalar equation, solved with an
`optimistix` Newton iteration started at the market instantaneous forward
\(f^M(0,T_0)\). Typically 3–4 iterations reach machine precision.

Because `optimistix` root-finds are **implicitly differentiable**, `jax.grad`
does not unroll the Newton iterations. It applies the implicit function theorem
to the converged solution:

$$
\frac{\partial r^\star}{\partial \theta}
= -\left(\frac{\partial CB}{\partial r}\right)^{-1}\frac{\partial CB}{\partial \theta}
$$

This matters — \(r^\star\) genuinely depends on the model parameters, and the
price is *not* stationary with respect to it. Differentiating through the
solver iterations would be both slower and less accurate. VALAX's autodiff
sensitivities agree with central differences to ~1e-11.

**A failed solve is NaN, not an exception.** Differentiating through a
root-find that did not converge makes the implicit-derivative linear solve
non-finite. Inside `jit` or `vmap` that raises at runtime, so one absurd iterate
of a calibration would abort a whole batch. `hw_critical_rate` therefore runs a
detached probe solve first to decide whether the solve succeeded. The
differentiable solve then runs on `where(ok, real input, detached stand-in)`,
and a failure returns NaN, so the price and its gradient are NaN. On the
healthy path every input is the real one, so the value and the gradient are
bit-identical to the unguarded solve (`tests/test_pricing/test_root_guard.py`).
The [G2++ boundary](g2pp.md#3-european-swaptions-brigomercurio-integral)
\(\bar y(x)\) uses the same guard.

---

## 5 · Why this makes Hull–White calibratable

With a constant volatility, Hull–White has exactly **two** free parameters:
\(a\) and \(\sigma\). The drift \(\theta(t)\) is not free — it is pinned by the
exact-fit condition to the initial curve. So calibration means fitting two
numbers to a swaption surface. A piecewise-constant \(\sigma(t)\) adds one
volatility per bucket. The bucket breakpoints are inputs, not fitted
parameters (see
[Time-Dependent Volatility §1](time-dependent-volatility.md#1-the-bucket-convention)).

That is only practical if each model price is cheap. A lattice or Monte-Carlo
price inside a least-squares loop is expensive and introduces discretisation
noise into the objective, which wrecks gradient-based optimisation. Jamshidian
gives an exact, smooth, differentiable price, so the least-squares Jacobian is
exact autodiff rather than a bumped approximation.

### The two parameters are strongly correlated

\(a\) controls how fast forward-rate volatility **decays** with expiry;
\(\sigma\) sets its **level**. Over a narrow expiry range a change in one is
nearly offset by a change in the other, so the ATM surface pins down their
combination far better than either individually. Fitting a single quote is
genuinely under-determined — any \((a,\sigma)\) on a level set reproduces it.

Desk practice is therefore to **fix \(a\)** (from a historical estimate or a
co-terminal fit) and let \(\sigma\) absorb the level. Pass
`fixed_mean_reversion` to work that way.

`calibrate_hull_white` returns a `CalibrationResult` whose `.value` is a
curve-free `HullWhiteModel` (`valax.price` binds the market curve at pricing
time). Its default settings
([`BASKET_SETTINGS`](../api/calibration.md#valax.calibration.basket.BASKET_SETTINGS))
are line-searched BFGS rather than Levenberg–Marquardt: the critical-rate
root-find of §4 has an implicit derivative that is non-finite far from the
solution, where an undamped LM step can land, while BFGS's line search keeps
the iterates where the pricer is defined. Per-trade calibration to a
co-terminal basket read off a calibrated swaption cube is the
`swaption_basket_node` of [example 14](../examples.md#per-trade-model-calibration-for-bermudans).

### What it cannot do

A one-factor Gaussian model cannot reproduce a smile at all, because volatility
does not depend on strike. If you need smile, you need SABR on top or a
different model.

With a **constant** \(\sigma\) it also produces a *normal* volatility term
structure that decays with expiry, so it cannot match an arbitrary ATM surface
shape. Fitting a flat Black-76 surface leaves a visible residual, around 8 %
rms in the test suite. That is the correct outcome for a two-parameter model,
not a bug.

A piecewise-constant \(\sigma(t)\) removes the term-structure limitation for
**one strip** of options. With one bucket per co-terminal expiry, the
co-terminal swaptions of a Bermudan can be matched exactly, one bucket after
another (see [Calibration Chains](calibration-chains.md)). It still does not
fit a whole surface. One \(\sigma(t)\) can fit one strip (for example, the
co-terminals of one trade), but not every expiry × tenor cell at once, because
a given expiry has a single \(\sigma\) history for all tenors.

---

## 6 · Validation

| Oracle | Agreement | What it tests |
|---|---|---|
| `ql.JamshidianSwaptionEngine` | `< 1e-4` rel | Same closed form — pins conventions |
| `ql.TreeSwaptionEngine` (400 steps) | `< 5e-3` rel | **Independent** numerical method |
| Payer − receiver parity | `< 1e-9` rel | Exact-fit: parity is pure curve, model-free |
| Autodiff vs central differences | `~1e-11` | Implicit differentiation of \(r^\star\) |
| Synthetic round-trip | `1e-5` on \((a,\sigma)\) | Calibration recovers generating parameters |
| QuantLib `Gsr` + dense `Gaussian1dSwaptionEngine`, \(\sigma(t)\) | `< 5e-6` rel | Piecewise \(\sigma_p\) against an independent implementation |
| Same, breakpoints shifted | `> 1e-3` rel (must disagree) | Anti-vacuity: the buckets are really read |
| Root guard (`test_root_guard.py`) | bit-identical when healthy, NaN on failure | §4 guard |

The \(\sigma(t)\) rows are explained in
[Time-Dependent Volatility §8](time-dependent-volatility.md#8-validation),
including why QuantLib's `Gaussian1dJamshidianSwaptionEngine` is not used as the
oracle.

---

## References

- Jamshidian, F. (1989). "An Exact Bond Option Formula". *Journal of Finance* 44(1).
- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models — Theory and Practice*, §3.3.
- Hull, J. & White, A. (1990). "Pricing Interest-Rate-Derivative Securities".
