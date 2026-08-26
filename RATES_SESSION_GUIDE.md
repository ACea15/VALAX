# Rates Session Guide (TEMPORARY — delete or promote when the session lands)

> **Status: temporary working document.** This is a scratch plan to open a
> focused interest-rates development session cold. It is deliberately **not**
> wired into the mkdocs site. When the work it describes is done, either delete
> this file or promote the durable parts into `docs/roadmap.md` /
> `docs/architecture/`.

Companion reading (durable docs): the interest-rate *theory* lives in
`docs/theory.md` (§2.6 LMM, §2.8 Hull-White, §3 Curve Framework, and the new
Interest-Rate Derivatives section); the roadmap rates tiers are in
`docs/roadmap.md`.

---

## 1. Where rates stands today (map)

Maturity: **production** (validated, differentiable, tested) / **partial** /
**stub** / **missing**.

| Area | State | Where |
|---|---|---|
| **Multi-curve bootstrap** (joint Newton solver, OIS/tenor-basis/XCCY/FX/futures, quote Jacobians) | **production** | `valax/curves/bootstrap_graph.py`, `graph.py`, `instruments.py` |
| Single-curve bootstrap | production | `valax/curves/bootstrap.py` |
| `bootstrap_multi_curve` (sequential) | **deprecated** | `valax/curves/multi_curve.py` |
| **Analytic caplets/caps** (Black-76 + Bachelier, dual-curve) | **production** | `valax/pricing/analytic/caplets.py` |
| **Analytic swaptions** (Black-76 + Bachelier, dual-curve annuity) | **production** | `valax/pricing/analytic/swaptions.py` |
| **Bonds** (YTM, duration, convexity, KRDs, floating, OIS) | production | `valax/pricing/analytic/{bonds,floating}.py` |
| **Hull-White 1F** analytic exact-fit ZCB | production | `valax/models/hull_white.py` |
| **Hull-White trinomial tree** (callable/puttable bonds) | **production** (QL-validated) | `valax/pricing/lattice/hull_white_tree.py` |
| **Hull-White short-rate MC** (exact OU; fixed/floating/callable/puttable) | **production** | `valax/pricing/mc/hull_white_paths.py`, `recipes.py` |
| **Hull-White swaptions** (Jamshidian) | **production** (QL-validated) | `valax/pricing/analytic/hull_white_swaptions.py` |
| **Hull-White calibration** to swaption surface | **production** | `valax/calibration/hull_white.py` |
| **Hull-White short-rate PDE** (bonds, callable/puttable, European **and Bermudan** swaptions) | **production** (QL-validated) | `valax/pricing/pde/hull_white.py` |
| **LMM/BGM** MC (PCA, Rebonato vol) | partial (MC only) | `valax/models/lmm.py`, `valax/pricing/mc/lmm_paths.py` |
| Bermudan swaption (LSM on LMM) | partial | `valax/pricing/mc/bermudan.py` |
| CMS / range-accrual analytic | **partial (no convexity adj)** | `valax/pricing/analytic/rates_exotics.py` |
| Inflation (ZC/YoY swaps, caps) | partial | `valax/pricing/analytic/inflation.py`, `valax/curves/inflation.py` |
| **SABR** smile: Hagan implied vol + single-smile & per-expiry **calibration** | **production** (QL-validated) | `valax/pricing/analytic/sabr.py`, `valax/calibration/sabr.py`, `valax/surfaces/sabr_surface.py` |
| **G2++/HW-2F** (multi-factor Gaussian short rate) | **missing** | planned `valax/models/g2pp.py` (workstream 6) |
| **Cheyette / quasi-Gaussian** (Markovian HJM w/ skew) | **missing** | planned `valax/models/cheyette.py` (workstream 7, deferred) |

## 2. The gaps that define the session

1. **No short-rate / market-model calibration.** `valax/calibration/` has
   SABR/Heston/SLV; the generic **SABR smile calibrator works for rate smiles
   too** (`fixed_beta=0.0`, LM/BFGS/Adam, QL-validated). ~~Hull-White→swaptions
   (Jamshidian)~~ is now ✅ **delivered** (`valax/calibration/hull_white.py`).
   Still missing: LMM caplet/swaption stripping, and G2++/HW-2F.
2. ~~**No Hull-White MC**~~ — ✅ **delivered.** `generate_hull_white_paths`
   (exact conditional OU sampling) plus MC recipes for fixed/floating/callable/
   puttable bonds. Roadmap MC-B1 closed.
3. ~~**No rates PDE.**~~ — ✅ **delivered.** Roadmap PR-3 is closed:
   `valax/pricing/pde/hull_white.py` prices fixed-rate, callable and puttable
   bonds, European swaptions and **Bermudan swaptions** on the existing 1-D
   substrate. Bermudans no longer need a regression proxy — under Hull-White
   the tail-swap exercise value is analytic at every node, so the projection is
   exact.
3b. **No OAS solver.** Now that callable bonds price on both the tree and the
   PDE, option-adjusted spread is a short step: add a constant shift to the
   model's discount curve and root-find it against a market price. Autodiff
   then gives effective duration/convexity directly. Roadmap 3.2.

4. **No multi-factor short rate and no skew-in-dynamics model.** Single-factor
   Hull-White is the only short-rate model, so tenor rates move in lockstep
   (perfect correlation) and rate skew lives only in the *static* SABR smile, not
   the model dynamics (LMM is lognormal-only). This blocks decorrelation products
   (CMS-spread, steepeners) and skew-sensitive callables. Addressed by
   **workstream 6 (G2++/HW-2F)** and **workstream 7 (Cheyette, deferred)**. No
   Vasicek/CIR either (not currently scoped).
5. ~~**Thin QuantLib validation.**~~ — ✅ **delivered** for swaptions, caps/floors
   and the HW callable/puttable tree, in
   `tests/test_quantlib_comparison/test_rates_pricers_ql.py`.
   `test_cap_strip_on_caplet_vols_ql.py` remains a skipped placeholder (needs
   workstream 5a).

   **This paid for itself immediately.** On its first execution the tree
   comparison caught a real pricing bug: `callable_bond_price` compared the
   *cum-coupon* continuation value against the call price, undervaluing callable
   bonds by up to one coupon (2.16 price points on a 5 % 5Y par-callable). Call
   prices are quoted ex-coupon. The bug had been invisible because the QL fixture
   raised a constructor `TypeError` under QuantLib >= 1.35 and never ran.
6. **SABR is not joined to the curves, and there is no normal-vol variant.** The
   SABR calibrator is production-grade but disjoint from the rate pricers:
   (a) `sabr_price` discounts with a **flat scalar rate**, not the multi-curve
   forwards/annuity; (b) only the **lognormal Black** Hagan expansion exists —
   no **normal/Bachelier or shifted** SABR for post-2008 / negative-rate quoting;
   (c) there are **no rates vol-surface objects** (swaption cube, cap/floor
   optionlet surface) fed from market rate quotes. See workstream 5.
7. **CMS/range-accrual ship without convexity adjustment** (documented caveats).

**Cross-cutting production prerequisite (not required for prototypes):** business
calendars + a stub/compounding-aware cashflow engine (roadmap P1.1/P1.2).

## 3. Recommended session sequence

Ordered to keep the **clean-oracle discipline** that caught the 33% Heston bug:
pin what exists to a reference *before* building on it.

1. ✅ **DONE — QuantLib validation harness for the existing rate pricers.**
   `tests/test_quantlib_comparison/test_rates_pricers_ql.py` covers swaptions
   (Black-76 + Bachelier, payer *and* receiver), caps/floors, and the HW
   callable/puttable tree. Two harness bugs and one pricer bug were found and
   fixed; see gap 5 above.

2. ✅ **DONE — Hull-White short-rate MC.** `generate_hull_white_paths` uses exact
   conditional OU sampling (no Euler bias in `r` at any step size). All four bond
   recipes registered. Triangulation holds: on a single-call bond MC and the tree
   agree to `2.4e-4` (well inside MC error); with several call dates the myopic
   MC policy is a documented upper bound on the tree price.

3. ✅ **DONE — Hull-White swaptions (Jamshidian) + calibration.**
   `hw_swaption_price` matches `ql.JamshidianSwaptionEngine` to `< 1e-4` and the
   *independent* `ql.TreeSwaptionEngine` to `< 5e-3`. `calibrate_hull_white`
   recovers generating parameters from a synthetic surface to `1e-5`.
   Gap 1 above ("no model calibration") is closed for Hull-White.

   Two findings worth carrying forward: (a) `optimistix` 0.1.0's
   Levenberg-Marquardt raises `List arity mismatch` when the residual closes
   over a sequence of instrument pytrees, so the calibrator defaults to
   line-searched BFGS — plain Gauss-Newton is undamped and overshoots from
   distant starts; (b) a one-factor Gaussian model leaves ~8 % rms residual
   against a flat Black-76 surface, which is the model's honest limit and is
   asserted as such in the tests.

4. ✅ **DONE — PR-3: Hull-White short-rate PDE.**
   `valax/pricing/pde/hull_white.py` covers fixed-rate/callable/puttable bonds,
   European swaptions and **Bermudan swaptions**, solved in the centred state
   `x` of `r = x + alpha(t)`. Agreement: `8e-6` (second-order) vs the analytic
   curve price on an option-free bond; `~3e-5` relative vs Jamshidian;
   `~2e-4` vs `ql.FdHullWhiteSwaptionEngine` and `~1.2e-3` vs
   `ql.TreeSwaptionEngine` on the Bermudan; `< 5e-3` vs the HW tree and
   `ql.TreeCallableFixedRateBondEngine` on callables. Fully `filter_jit` /
   `filter_grad` compatible, with autodiff-vs-FD agreement at `1e-8`–`1e-11`.
   Theory write-up: `docs/theory/hull-white-pde.md`.

   **Four findings worth carrying forward.**

   (a) **The shared 1-D *and* 2-D steppers sampled Dirichlet boundaries at the
   mirrored time level** — `(n_time - m)*dt` instead of `m*dt`. Every
   time-dependent boundary was therefore evaluated with the wrong discount
   factor. It was masked by the default `spot_range=4.0`: an ATM BS European
   call was off by `1.4e-1` at half-width 2, `1.4e-2` at 3, but only `4.5e-4` at
   4 — small enough to slip through 1298 QuantLib comparisons. Post-fix every
   width sits at `~1e-5`. *Lesson: a tolerance that passes at the default
   configuration is not evidence; sweep the numerical knobs.*

   (b) **`alpha(t)` must be integrated across each step, not sampled at the
   midpoint.** A log-linear curve's instantaneous forward is piecewise constant
   with jumps at pillars, so midpoint sampling is first-order there. The tell
   was a scheme that converged cleanly at second order on a *flat* curve but
   stalled at `4e-6` on a sloped one — i.e. it broke Hull-White's defining
   exact fit. `hw_alpha_average` closes it in closed form. *Lesson: test
   convergence on a non-flat curve; flat curves hide interpolation artefacts.*

   (c) **Don't snap cashflow dates.** Snapping displaces each coupon by up to
   `dt/2`, an `O(dt)` error worth `~2e-3` on a 5Y bullet — 1000x the scheme's
   own error, and enough to hide its convergence order. Scaling each coupon by
   the analytic `P(t_k, t_c | x)` removes it exactly. This makes the **PDE more
   accurate than the tree**, which still snaps (`-2.2e-3` to `+3.4e-5` on an
   option-free bond, non-monotone in the step count). Tree-comparison
   tolerances are therefore set by the *tree's* error.

   (d) The PDE has no concrete-input requirement, unlike the tree (whose
   `j_max` is an array shape). Schedules scatter with traced indices, so the
   pricers jit and differentiate as-is — which matters for putting a Bermudan
   inside a calibration objective.

**→ Next up: OAS (small, closes the callable-bond story), then workstream 5 (SABR ↔ multi-curve seam), then 6 (G2++).**

---

### 3b. OAS / Z-spread for callable and puttable bonds *(small, ~1 day)*

**What it is.** The option-adjusted spread (OAS) is the constant parallel shift
`s` on the model's discount curve such that the model price equals the market
dirty price. For a plain bond it collapses to the Z-spread (no option to adjust
for); for a callable bond it isolates the pure credit/liquidity component by
stripping out the call's value.

**Why now.** The PDE callable pricer is now differentiable with no concrete
inputs — `jax.grad` flows through the exercise projection directly. That's what
makes OAS cheap: it's a one-parameter root-find whose sensitivities come for
free. None of that would have been possible before PR-3.

**Location.** `valax/risk/oas.py`. It's a risk metric, sits next to
`shocks.py`, and borrows `parallel_shift` directly from there.

**The design insight that makes this clean.** Under Hull-White, a parallel shift
sends every pillar zero rate `r_i → r_i + s`, and since `−s·t` is exactly
linear the shift holds at *every* continuous time `t`:

```
f^M(0,t) → f^M(0,t) + s   =>   alpha(t) → alpha(t) + s
```

The convexity term of `alpha` (which depends only on `a`, `σ`) is untouched.
The PDE therefore sees `r = x + alpha(t) + s` — a pure constant shift to the
discount coefficient, with the x-dynamics completely unchanged. The two
conventional definitions of OAS (shift discounting only vs re-fit the whole
model) **coincide exactly** under HW + parallel shift. Worth asserting as a
unit test: `hw_alpha(shifted_model, t) − hw_alpha(original, t) == s` to machine
precision.

**Implementation.** ~80 lines. Root-find `s` with `optimistix.Newton`, passing
`parallel_shift(model.initial_curve, s)` into a fresh `HullWhiteModel` at each
step. Implicitly differentiable, so `jax.grad` on the solved `s` gives
effective duration and convexity automatically without unrolling iterations.

```python
# sketch
def callable_bond_oas(bond, model, market_price, config):
    def residual(s, _):
        shifted = HullWhiteModel(..., initial_curve=parallel_shift(curve, s))
        return pde_price_dispatch(bond, shifted, config).price - market_price
    return optx.root_find(residual, optx.Newton(...), x0=jnp.zeros(()))
```

**Tests — in order, each acting as oracle for the next.**

1. **`hw_alpha` shift identity** — assert `hw_alpha_average(shifted, t0, t1) −
   hw_alpha_average(original, t0, t1) == s` to `1e-14`. Free, exact, catches any
   mistake in how the shift is threaded into the model.
2. **Round-trip to zero** — price the bond with the model; feed that back as
   "market price"; recover OAS = 0 to solver tolerance (`< 1e-10`).
3. **Option-free bond: OAS ≡ Z-spread exactly.** Compute Z-spread through
   `fixed_rate_bond_price` (completely separate code path), compare to OAS from
   the PDE. Exact equality to solver tolerance — not a tolerance-band, because
   there's genuinely no option component to adjust for.
4. **Z-spread > OAS for a callable.** The call is valuable to the issuer, so
   z-spread overestimates the credit/liquidity component. Model-free directional
   invariant; any sign flip is a bug.
5. **Effective duration(callable) < duration(bullet)** — the call truncates the
   price upside, compressing duration. Another directional invariant.
6. **Negative effective convexity near the call boundary.** This is the money
   shot — the entire economic point of the callable-bond story, and nothing but
   a real model reproduces it. If convexity is positive everywhere, something is
   wrong upstream. Check via `jax.grad(jax.grad(oas_price_fn))(spread)`;
   sanity-check the value against central differences (test device only, per
   `AGENTS.md`).
7. **QuantLib `CallableFixedRateBond.OAS(...)` comparison** — this is the
   external oracle. ⚠️ **Compounding-convention trap**: QL's OAS is quoted on a
   *compounded* convention (takes `frequency` and `compounding` args), while our
   `parallel_shift` is continuously compounded. Convert explicitly with
   `log(1 + s/freq) * freq ≈ s` for small `s`, or just pick a regime where the
   difference is sub-bp and document it. Confusing the two would produce a
   plausible-looking ~5bp systematic error that's easy to mistake for model
   difference.

**Engine consistency note.** `dP/ds ≈ −Duration × P ≈ −4.5 × 95 ≈ −430` per
unit spread on a 5Y bond. The ~1e-3 price gap between the tree and PDE engines
(set by the tree's coupon-snapping error) maps to **< 0.05bp of OAS**. Engine
choice is irrelevant for OAS — use the PDE, it's faster to differentiate and
more accurate.

---

5. **SABR ↔ multi-curve integration seam** *(high value, medium readiness — both
   halves already exist)*. Marry the mature-but-disjoint multi-curve framework
   and SABR calibrator into a single curve-aware smile-pricing path. SABR sits
   *between* two curve touchpoints (see theory §9.2–9.3): the **forward** it
   models comes from the forwarding curve (caplet forward / forward swap rate),
   and **discounting** after `SABR → vol → Black-76/Bachelier` uses the OIS
   annuity `A(0) = Σ τ_i P_OIS(0,T_i)`. Deliverables:
   - **(a) Normal/Bachelier (and shifted) SABR** implied-vol variant — the
     `sabr_implied_vol` today is lognormal Black only; rates quote normal vol and
     need it for negative rates.
   - **(b) Rates vol-surface objects** — a **swaption cube** and a **cap/floor
     optionlet surface**, each carrying a normal-vs-lognormal quoting flag and
     built by stripping/calibrating market quotes with the existing SABR fitter.
   - **(c) A curve-aware pricing path** `(CurveGraph + SABR cube) → cap/swaption
     price`, pulling forwards from the forwarding curve and the annuity from OIS,
     rather than `sabr_price`'s flat-rate discounting.
   **Deps:** multi-curve framework (done) + SABR calibrator (done) + (1)'s QL
   harness. **Oracle:** QuantLib SABR/`OptionletStripper1` + the dual-curve
   Black-76/Bachelier pricers already in `caplets.py`/`swaptions.py`; unblocks
   the skipped `test_cap_strip_on_caplet_vols_ql.py`. This is also the
   prerequisite for CMS convexity below.

   ### 5·bis — state of the code, verified (read this before starting)

   Checked against the source at the close of the PR-3 session, so a cold start
   does not have to rediscover it:

   - **`sabr_price(option, forward, rate, model)` takes a flat scalar `rate`
     and a hand-passed `forward`.** No curve, no annuity. This is the whole of
     the (c) gap — there is no curve-aware path to extend, only one to write.
   - **`sabr_implied_vol` is Hagan lognormal only** — one function, no
     `is_normal` / shift parameter to thread. (a) is a genuinely new expansion,
     not a flag.
   - **`SABRVolSurface` is strike × expiry with no tenor axis.** It therefore
     *cannot* become a swaption cube by extension — (b) needs a new object
     (expiry × tenor × strike) rather than a field added to the existing one.
     `SABRVolSurface` also feeds Dupire/SLV via `total_variance`, so it has
     equity consumers: don't repurpose it, leave it alone.
   - **`bachelier.py` has `bachelier_price` but no implied-vol inverter.**
     A `bachelier_implied_vol` is a prerequisite for any lognormal↔normal
     quote conversion, and is not currently anywhere in the repo.
   - **`generate_sabr_paths` already exists** (`valax/pricing/mc/sabr_paths.py`)
     — the natural *internal* oracle for the normal expansion, in the
     triangulation style that has now caught five bugs in this codebase.

   **Suggested first slice (5a), with its oracles.** Self-contained and fully
   validatable before any surface plumbing exists:

   - Pin the **exact degenerate reductions** first — these hold to machine
     precision and need no external reference: `β=0, ν→0` ⟹ normal vol `= α`
     exactly (arithmetic Brownian motion); `β=1, ν→0` ⟹ lognormal vol `= α`
     exactly.
   - Then the **MC cross-check**: both expansions must land within MC error of
     `generate_sabr_paths` on the same option.
   - Then the **capability test** that is the point of the exercise: negative
     and zero strikes must price finitely, where the lognormal path cannot run
     at all.
   - ⚠️ **Trap:** the lognormal and normal Hagan expansions are *different*
     approximations to the same SDE and agree only to `O(ν²T)`. A consistency
     test must assert that the ATM gap **scales** correctly, *not* that the two
     are equal. An equality assertion here is exactly the kind that gets
     loosened until it means nothing.
   - ⚠️ **Check before assuming:** QuantLib 1.41's API for normal / shifted
     SABR smile sections has *not* been verified. `ql.sabrVolatility` is the
     lognormal one. Confirm what exists before writing a QL test around it —
     the reductions plus the MC carry the validation regardless.

   **Also relevant from workstream 3:** `optimistix` 0.1.0's
   Levenberg-Marquardt raises `List arity mismatch` when the residual closes
   over a sequence of instrument pytrees. Any new calibrator (cube stripping,
   per-expiry fits) will hit this; `calibrate_hull_white` works around it with
   line-searched BFGS.

6. **G2++ / HW-2F (multi-factor Gaussian short rate)** *(medium readiness — the
   whole affine/Markovian toolchain from HW-1F generalizes)*. Adds a second
   stochastic factor so tenor rates **decorrelate**, unlocking instruments HW-1F
   structurally cannot price: **CMS-spread options, steepeners/flatteners, spread
   range-accruals**, and decorrelation-sensitive **Bermudan swaptions**. (Also
   closes the tf-quant-finance "multi-factor short rate" capability gap.)

   G2++ is the two-additive-factor Gaussian model (equivalent to HW-2F):

   ```
   r(t) = x(t) + y(t) + φ(t)
   dx = -a·x dt + σ dW₁ ,  dy = -b·y dt + η dW₂ ,  dW₁·dW₂ = ρ dt
   ```

   with `φ(t)` chosen to fit the initial curve exactly. It stays affine/Gaussian,
   so ZCB prices are closed-form (Brigo–Mercurio) and swaptions are semi-analytic.

   Deliverables:
   - **(a) Model** `valax/models/g2pp.py` (mirror `hull_white.py`):
     `G2PPModel(eqx.Module)` carrying `a, b, σ, η, ρ` + curve; the analytic ZCB
     `P(t,T)` and its Gaussian variance term `V(t,T)` (Brigo–Mercurio §4.2);
     `φ` fitting via the initial curve. Static fields per AGENTS.md;
     `a,b,σ,η` via the `positive` transform and `ρ` via the existing
     `correlation` transform in `valax/calibration/transforms.py`.
   - **(b) Analytic swaption** `valax/pricing/analytic/g2pp_swaptions.py`:
     Brigo–Mercurio semi-analytic — 1-D Gaussian quadrature over one factor with
     a Jamshidian-style decomposition on the other. Reuse the dual-curve annuity
     from `swaptions.py`.
   - **(c) Monte Carlo** `valax/pricing/mc/g2pp_paths.py`: `generate_g2pp_paths`
     — **exact** two-factor correlated-OU scheme (Gaussian conditional mean/cov,
     no discretization bias). Register `(FixedRateBond/FloatingRateBond/Swaption/
     CMSSwap-spread, G2PPModel)` recipes in `valax/pricing/mc/recipes.py`.
   - **(d) Calibration** `valax/calibration/g2pp.py`: `calibrate_g2pp` to the ATM
     / co-terminal swaption surface via `optimistix`, reusing `loss.py` +
     `transforms.py`.
   - **(e) PDE (optional, later):** 2-D short-rate PDE on the existing 2-D ADI
     substrate (`schemes2d.py`, `operators2d.py`) for decorrelation-aware
     Bermudan swaptions.

   **Oracles:** QuantLib `ql.G2` + `G2SwaptionEngine` / `FdG2SwaptionEngine`;
   internal **MC ↔ analytic** triangulation; and a **reduction check** — driving
   `ρ`/one factor to a degenerate limit must recover the HW-1F pricer already in
   the repo. **Deps:** workstream 1 (QL harness) + a swaption-vol quoting object
   from workstream 5b.

7. **Cheyette / quasi-Gaussian (deferred, lower priority)** — a Markovian HJM
   model that adds **skew to the rate dynamics** (local vol on rates) while
   staying low-dimensional, so it remains PDE/tree-friendly. Complements
   workstream 6: G2++ adds *decorrelation*, Cheyette adds *skew* (recall LMM here
   is lognormal-only and SABR skew is static). Target instruments:
   **skew-sensitive callables** — Bermudan swaptions with skew, callable CMS,
   TARNs/snowballs.

   Markovian state `(x, y)` with

   ```
   r(t) = f(0,t) + x(t)
   dx = (y − a·x) dt + σ(t,x) dW ,  dy = (σ(t,x)² − 2a·y) dt
   ```

   where the local-vol `σ(t,x)` carries the skew and `y` is the auxiliary
   variance accumulator. Single- or multi-factor.

   Deliverables (larger build, sequence after 6):
   - **Model** `valax/models/cheyette.py`.
   - **MC** carrying the `y` accumulator; **1-D/2-D PDE** on the existing
     substrate for callables; **calibration** to the swaption smile.

   **Oracles:** **reduction to HW-1F** when `σ` is constant; QuantLib has no
   direct Cheyette engine, so cross-check against internal MC/PDE agreement and
   Andersen–Piterbarg literature benchmarks.

Later / lower priority: **CMS convexity** (Hagan replication over the workstream-5
swaption smile; oracle QL `AnalyticHaganPricer`).

## 4. Reusable machinery already in place

- **Curves + Jacobians:** `bootstrap_curve_graph`, `quote_jacobian` — discount
  factors, forwards, and their sensitivities are differentiable and QL-validated.
- **PDE substrate:** `valax/pricing/pde/{grids,operators,schemes,boundary,`
  `terminal,exercise,dispatch,recipes}.py` — 1-D theta-scheme + Rannacher +
  penalty/projection exercise seam, plus the 2-D ADI layer from PR-2.
- **Short-rate PDE pieces (new in PR-3, all model-agnostic — G2++ wants exactly
  these):** `centred_state_grid` (mesh for a mean-reverting state starting at
  the origin), `apply_linearity_bc_1d` + `zero_boundary` (zero-curvature edges
  for problems with no closed-form far field), the `event_fn` seam on
  `solve_backward_1d` (discrete coupons and exercise projection), and
  `hw_operator_stack` as the template for a per-step operator whose discount
  coefficient *is* the state.
- **Calibration substrate:** `valax/calibration/{loss,transforms}.py` +
  `optimistix` least-squares (used by SABR/Heston) — directly reusable for HW/LMM.
- **QL harness:** `tests/test_quantlib_comparison/_ql_adapters.py` (date snapping,
  flat curves, Act/365, `NullCalendar`).
- **Numerical debugging discipline:** `docs/architecture/numerical-pitfalls.md`.
