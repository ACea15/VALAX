# Multi-Curve Short-Rate Models: the Deterministic Tenor Basis

**Implementation:** `valax/pricing/multicurve.py`, consumed by
`valax/pricing/pde/{hull_white,g2pp}.py` and the Hull-White / G2++ recipes in
`valax/pricing/mc/recipes.py`.

This page explains the one modelling assumption that lets an
**exact-fit-to-one-curve** short-rate model price in the multi-curve world:
OIS discounting with a distinct forward (projection) curve, where the basis
between them is frozen at its time-zero value.

For *why* the market is multi-curve at all, see
[Why the Multi-Curve Framework](../guide/why-multicurve.md) and
[Curve Framework §3.2](curves.md#32-single-curve-vs-multi-curve-framework).
For how to *call* it, see [PDE Solvers §5.7](../guide/pde.md) and
[Monte Carlo §2.4](../guide/monte-carlo.md).

---

## 1. The problem: one model, two curves

Hull-White and G2++ describe a single short rate $r(t)$, and their exact-fit
shift ($\alpha(t)$, $\varphi(t)$) is constructed so the model reprices **one**
input term structure:

$$
\mathbb{E}\!\left[e^{-\int_0^T r(s)\,ds}\right] = P_0(T).
$$

That is the whole model: one stochastic driver, one curve. Post-2008 pricing
needs two — a collateral/OIS curve for discounting and a tenor curve for
projecting IBOR-style fixings — and there is nothing in a one-factor short-rate
model for the second curve to attach to. Simply handing a recipe a second
`DiscountCurve` is therefore not enough; a *modelling decision* is required
about how the two curves move relative to one another.

## 2. The assumption: a frozen multiplicative basis

Define the projection-to-discount bond ratio

$$
\beta(t, T) \;=\; \frac{P^{\text{fwd}}(t, T)}{P^{\text{disc}}(t, T)} .
$$

The **deterministic-basis** (or frozen-basis) assumption is that this ratio is
not stochastic — it equals its observable time-zero value for all $t$:

$$
\boxed{\;
\beta(t, T) \;=\;
\frac{P^{\text{fwd}}_0(T) \big/ P^{\text{fwd}}_0(t)}
     {P^{\text{disc}}_0(T) \big/ P^{\text{disc}}_0(t)}
\;}
$$

so that

$$
P^{\text{fwd}}(t, T) \;=\; P^{\text{disc}}(t, T)\,\beta(t, T),
$$

with $P^{\text{disc}}$ produced by the short-rate model (fitted to the discount
curve) and $\beta$ read straight off the two market curves. Cash flows are
discounted with the model; floating rates are projected off the model's bond
scaled by $\beta$.

### 2.1 The only quantity the payoffs need

Every payoff in VALAX's short-rate solvers uses $\beta$ through its **ratio
over a single accrual period**, which is independent of the valuation time $t$:

$$
b(T_s, T_e)
\;=\; \frac{\beta(t, T_s)}{\beta(t, T_e)}
\;=\; \frac{P^{\text{fwd}}_0(T_s) \big/ P^{\text{fwd}}_0(T_e)}
           {P^{\text{disc}}_0(T_s) \big/ P^{\text{disc}}_0(T_e)}
\;=\; \frac{1 + \tau F^{\text{fwd}}(T_s, T_e)}{1 + \tau F^{\text{disc}}(T_s, T_e)} ,
$$

the ratio of the two curves' simply-compounded growth factors over the period.
This is `valax.pricing.multicurve.tenor_basis`. Because $b$ is a *constant
multiplier* at every grid node and on every path, no affine formula inside the
solvers has to change: the overlay costs nothing.

$b = 1$ is exactly the single-curve case, and the implementation encodes it as
`basis=None` so the single-curve arithmetic is executed unchanged — the
non-regression guarantee is structural, not numerical.

## 3. What the assumption buys, and what it costs

### 3.1 Linear products are priced *exactly*

A swap or FRN present value depends only on the two curves, never on the
dynamics. Under the overlay the floating leg is

$$
\sum_j \bigl(b_j\,P(t, T_{j-1}) - P(t, T_j)\bigr)
\;\xrightarrow[\ t = 0\ ]{}\;
\sum_j \left(\frac{P^{\text{fwd}}_0(T_{j-1})}{P^{\text{fwd}}_0(T_j)} - 1\right) P^{\text{disc}}_0(T_j),
$$

which *is* the textbook dual-curve floating leg. So a swap priced by the
Hull-White PDE with OIS discounting and a distinct projection curve agrees with
`ois_swap_price(swap, ois, projection)` to discretisation tolerance, and today's
forward curve is reproduced by construction. This is the property the test
suites assert (`tests/test_pde/test_multicurve.py`,
`tests/test_mc/test_multicurve.py`).

### 3.2 Optionality picks up the basis, correctly in level

A caplet on the projected forward becomes, at its fixing date,

$$
N\bigl(b - P(T_0, T_1)(1 + K\tau)\bigr)^+ ,
$$

i.e. the basis displaces the strike side of the bond option. The vanilla
Jamshidian/Gauss-Hermite decomposition and the PDE terminal condition both
absorb it without further change.

### 3.3 The cost: no basis volatility

Because $\beta$ is deterministic, **the OIS/IBOR basis has zero volatility
under this model.** Concretely:

- An option *on the basis* (a basis swaption, a spread option between two
  tenor curves) is worth nothing — the model cannot price it, and should not
  be asked to.
- Convexity corrections arising from stochastic basis are absent. For vanilla
  caps and swaptions these are second-order and well below calibration noise;
  for basis-contingent exotics they are the entire product.
- Basis *delta* is still available (bump the projection curve and reprice, or
  differentiate — the basis is a differentiable function of the curve
  pillars). It is basis *vega* that does not exist.

This is the standard market treatment and matches the deterministic-basis
convention already used by VALAX's multi-curve bootstrap. A genuinely
stochastic-basis construction requires a multi-curve HJM / Mercurio-style
extension with its own driver per tenor curve; it is catalogued in
[Research Ideas](../research-ideas.md) and deliberately out of scope here.

## 4. Discounting: binding rather than re-weighting

The second half of "multi-curve" is *which* curve a trade discounts on. For an
exact-fit short-rate model that is not an overlay at all — the model's
`initial_curve` **is** its discount curve, because $\alpha$ / $\varphi$ is
constructed to reproduce it. The curve is therefore market data the model is
*bound to*, not one of its parameters: models (including every calibrator's
output) are curve-free, and `valax.price` binds `initial_curve` from the
`MarketState` — the curve `model.curve_id` if set, else the instrument's
`discount_id` — via `valax.core.binding.curve_requirements`, refusing a model
that arrives pre-bound. Kernel-level callers bind explicitly with
`valax.core.binding.bind_curves`; a recipe called directly fills an *unbound*
model from its `discount_curve` argument
(`valax.pricing.multicurve.with_discount_curve`) and never re-fits a bound one.

The calibrated dynamics — mean reversion, volatility, correlation — are
intrinsic and carry over unchanged. Note the consistency invariant of the
unified pricing contract
([tutorial § 7](../guide/tutorial-unified-pricing.md#7-models-that-reference-the-market)):
a model is only valid against the market it was calibrated to, or a
scenario-shocked version of it.

## 5. Scope in VALAX today

<!-- coverage-map -->

| Engine / model | Discount curve | Projection curve |
|---|---|---|
| Analytic rates pricers | ✅ | ✅ |
| Hull-White analytic swaption (Jamshidian) | ✅ | ✅ |
| G2++ analytic swaption (eq. 4.31 quadrature) | ✅ | ✅ |
| Hull-White PDE (bonds, swaptions, Bermudans, caps) | ✅ | ✅ |
| G2++ PDE (bonds, swaptions, Bermudans, caps) | ✅ | ✅ |
| Hull-White MC (bonds, FRN, callables) | ✅ | ✅ (FRN) |
| G2++ MC (bonds, FRN, swaptions) | ✅ | ✅ |
| G2++ CMS-spread (PDE and MC) | ✅ | ❌ — **rejected, not ignored** |
| Cheyette, LMM | — | — |
| **Calibration** — `calibrate_hull_white`, `calibrate_g2pp` | ✅ | ✅ |
| **Calibration** — `calibrate_cheyette`, `calibrate_cheyette_2f` | ✅ | ❌ — **rejected, not ignored** |

The CMS-spread recipes rebuild their par swap rates from the model's own bond
strip, so a distinct projection curve would be silently dropped. They raise
instead: a wrong price with no signal is the one outcome worse than an
unsupported feature. The Cheyette calibrators refuse for the same reason —
their PDE engine has no discount/projection split to honour.

Whether the deterministic overlay applies is declared **per model type** by
`valax.pricing.multicurve.basis_overlay` (Hull–White and G2++: yes; Cheyette:
no), and the recipes read the per-period basis through
`model_tenor_basis` / `model_swap_leg_basis`, which raise for a model type
without the overlay. Consequently `valax.price` raises whenever a trade
resolves a distinct projection curve that the recipe/model cannot use —
previously such a trade could be priced single-curve without a signal. A
future stochastic-basis model registers `basis_overlay = False` and projects
off its own index curves instead.

### 5.1 Why the closed forms extend so cheaply

Because the basis is *deterministic*, a dual-curve swap is still an affine
combination of discount bonds: the floating leg is worth ``float_start``
instead of par at expiry, and each fixed cash flow picks up
:math:`1 - b_{i+1}` (:func:`valax.pricing.multicurve.apply_swap_basis`). So

- **Hull-White** keeps Jamshidian exactly: the critical rate now solves
  :math:`CB(r^\star) = \text{float\_start}` rather than :math:`= 1`, and the
  decomposition into zero-coupon bond options is otherwise untouched.
- **G2++** keeps the eq. (4.31) quadrature: the :math:`\bar y(x)` boundary
  solves to ``float_start`` and the strike term :math:`\Phi(\mp h_1)` is scaled
  by it.

That is the whole extension — no new integrals, no loss of closed form, and the
same basis algebra as the MC/PDE recipes. Consequently **calibration stays
cheap and its Jacobian stays exact autodiff**, which is what makes dual-curve
fitting practical rather than merely possible. A stochastic basis would break
this affine structure and is a different modelling exercise (§3.3).

## References

- Henrard, M. (2014). *Interest Rate Modelling in the Multi-Curve Framework*.
  Palgrave Macmillan. — the book-length treatment; deterministic-basis and
  stochastic-basis constructions side by side.
- Bianchetti, M. (2010). "Two Curves, One Price." *Risk*, 23(8).
- Mercurio, F. (2010). "Interest Rates and the Credit Crunch: New Formulas and
  Market Models." Bloomberg Portfolio Research Paper.
- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models — Theory and
  Practice*, 2nd ed., ch. 3–4 (Hull-White and G2++ exact fit).
