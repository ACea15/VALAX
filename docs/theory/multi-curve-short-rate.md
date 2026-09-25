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

## 4. Discounting: re-fitting rather than re-weighting

The second half of "multi-curve" is *which* curve a trade discounts on. For an
exact-fit short-rate model that is not an overlay at all — the model's
`initial_curve` **is** its discount curve, because $\alpha$ / $\varphi$ is
constructed to reproduce it. Honouring an instrument's `discount_id` therefore
means re-fitting the model to that curve, which is what
`valax.pricing.multicurve.with_discount_curve` does (the price-time analogue of
`HullWhiteModel.with_market_curve`).

The calibrated dynamics — mean reversion, volatility, correlation — are
intrinsic and carry over unchanged. Note the consistency invariant of the
[pricing ontology](../architecture/pricing-ontology.md#32-consistency-invariant-state-it-in-code-docs):
a model is only valid against the market it was calibrated to, or a
scenario-shocked version of it.

## 5. Scope in VALAX today

| Engine / model | Discount curve | Projection curve |
|---|---|---|
| Analytic rates pricers | ✅ | ✅ |
| Hull-White PDE (bonds, swaptions, Bermudans, caps) | ✅ | ✅ |
| G2++ PDE (bonds, swaptions, Bermudans, caps) | ✅ | ✅ |
| Hull-White MC (bonds, FRN, callables) | ✅ | ✅ (FRN) |
| G2++ MC (bonds, FRN, swaptions) | ✅ | ✅ |
| G2++ CMS-spread (PDE and MC) | ✅ | ❌ — **rejected, not ignored** |
| Cheyette, LMM | — | — |

The CMS-spread recipes rebuild their par swap rates from the model's own bond
strip, so a distinct projection curve would be silently dropped. They raise
instead: a wrong price with no signal is the one outcome worse than an
unsupported feature.

## References

- Henrard, M. (2014). *Interest Rate Modelling in the Multi-Curve Framework*.
  Palgrave Macmillan. — the book-length treatment; deterministic-basis and
  stochastic-basis constructions side by side.
- Bianchetti, M. (2010). "Two Curves, One Price." *Risk*, 23(8).
- Mercurio, F. (2010). "Interest Rates and the Credit Crunch: New Formulas and
  Market Models." Bloomberg Portfolio Research Paper.
- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models — Theory and
  Practice*, 2nd ed., ch. 3–4 (Hull-White and G2++ exact fit).
