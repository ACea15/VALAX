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

These are Black formulas with the *bond* as underlying. Summing gives the
swaption:

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

---

## 5 · Why this makes Hull–White calibratable

Hull–White has exactly **two** free parameters: \(a\) and \(\sigma\). The drift
\(\theta(t)\) is not free — it is pinned by the exact-fit condition to the
initial curve. So calibration means fitting two numbers to a swaption surface.

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

### What it cannot do

A one-factor Gaussian model produces a *normal* volatility term structure that
decays with expiry. It cannot reproduce a smile at all (volatility does not
depend on strike), and it cannot match an arbitrary ATM surface shape. Fitting
a flat Black-76 surface leaves a visible residual — around 8 % rms in the test
suite — and that is the correct outcome, not a bug. If you need smile, you need
SABR on top or a different model.

---

## 6 · Validation

| Oracle | Agreement | What it tests |
|---|---|---|
| `ql.JamshidianSwaptionEngine` | `< 1e-4` rel | Same closed form — pins conventions |
| `ql.TreeSwaptionEngine` (400 steps) | `< 5e-3` rel | **Independent** numerical method |
| Payer − receiver parity | `< 1e-9` rel | Exact-fit: parity is pure curve, model-free |
| Autodiff vs central differences | `~1e-11` | Implicit differentiation of \(r^\star\) |
| Synthetic round-trip | `1e-5` on \((a,\sigma)\) | Calibration recovers generating parameters |

---

## References

- Jamshidian, F. (1989). "An Exact Bond Option Formula". *Journal of Finance* 44(1).
- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models — Theory and Practice*, §3.3.
- Hull, J. & White, A. (1990). "Pricing Interest-Rate-Derivative Securities".
