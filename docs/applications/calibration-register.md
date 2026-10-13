# Calibration Methodology & Assumptions Register

Every calibration in VALAX makes choices: what it fits and what it holds
fixed, which instruments it fits to, in which space it measures the misfit,
which engine prices the targets, and what it silently assumes about the
market. This page is the **single register of those choices** — the
model-validation view of the calibration layer. For each calibration it
states the methodology, the assumptions, the known limitations (with the
evidence that measured them) and the improvement path.

**What this page owns, and what it does not.** It owns *assumptions,
limitations and improvement paths*. It does not repeat how-tos (they live on
the [guides](../guide/calibration.md)), derivations (the
[theory](../theory/calibration.md) pages) or status (the
[coverage matrix](instrument-pricing.md) and the [roadmap](../roadmap.md)):
each entry links to those instead. A planned improvement is a roadmap line;
the entry here links to it.

**Keeping it true.** A calibration entry point without an entry here fails
`tests/test_docs/test_calibration_register.py`, and adding or changing a
calibration is not done until its entry is updated (AGENTS.md, *Definition of
done*). Defaults quoted here are read from the code; if you change one,
change it here.

## How to read an entry

| Field | Meaning |
|---|---|
| **Entry points** | The public functions / node factories it covers |
| **Fits / holds fixed** | Free parameters (and their defaults and transforms); parameters held fixed, and whether as reported fixed inputs `p` |
| **Targets & objective** | What the residual compares, in which space (price / vol), absolute or relative, default weights; the level kind (`root`, `least_squares`, `explicit`, `fixed_point`) |
| **Engine & provenance** | What prices the targets inside the fit, and the `CalibrationProvenance` recorded |
| **Settings** | The default `SolverSettings` |
| **Market inputs** | Curves (single / dual) or loose scalars |
| **Assumptions** | Modelling and market assumptions the fit relies on |
| **Limitations** | What it cannot do, with the evidence (test, example or measurement) |
| **Validation** | The tests that gate it and their tolerances |
| **Improve** | The improvement path, linked to the roadmap |

## Cross-cutting assumptions

These apply to every entry below unless the entry says otherwise.

| # | Assumption | Consequence | Improve |
|---|---|---|---|
| X-1 | Every calibration returns a `CalibrationResult` with a traceable `converged` flag; failures are reported, never thrown inside `jit`, and any non-finite residual counts as not converged. Batched levels name the failing node, including a sub-node axis (`sub_labels`, e.g. strike × bucket `"K=3.00% cap 2Y"`); sequential solves name their **root cause** (`root_cause_labels()`); joint root levels localise a failure to non-finite quotes, then to the worst residual cluster, then to the starting-point residuals. Trade data travels in `LevelInputs.data`, so results returned from `jit` / `vmap` stay usable eagerly (`condition_number()`, `check()`). | A caller that ignores `converged` / `check()` can consume a failed fit — but failed fits install NaN (cube cells and strip buckets alike), so they cannot pass silently as plausible numbers. Joint-level localisation is a heuristic: it isolates a single gross outlier, not a mutual inconsistency spread over several quotes. | Leave-one-out / damped re-solve on failure; a-priori quote outlier screening (X-8) |
| X-2 | Least-squares levels are differentiated through their first-order condition $\nabla_\theta \tfrac12\lVert r\rVert^2 = 0$ (exact, second-order terms included). | On a noisy fit with non-zero residuals the adjoint and autodiff paths agree to ~1e-8 of the largest sensitivity, not to an absolute 1e-10 — the normal equations square the condition number (example 15). | — |
| X-3 | Weights default to **ones**: no vega weighting, no bid/ask weighting. | Price-space fits over-weight expensive options; vol-space fits weight every quote equally regardless of liquidity. | Bid/ask pairs as fit weights — [roadmap §5.6](../roadmap.md#unified-calibration-pricing-and-risk) ("Richer synthetic noise") |
| X-4 | Parameters are solved in an unconstrained space through transforms (`positive` = softplus, `correlation` = tanh, `bounded` = sigmoid). | Bounds are respected by construction, but no *joint* constraints exist (e.g. Heston's Feller condition, SVI no-arbitrage). | Constrained (KKT) levels — [roadmap §5.6](../roadmap.md#unified-calibration-pricing-and-risk) |
| X-5 | The engine a fit was priced on is recorded as provenance: `exact` (closed form), `discretised` (PDE / MC, config recorded) or `modelling_bias` (frozen-`y` Cheyette, engine-bound). `price()` does not consult it; `check_engine` is an eager opt-in. | A modelling-bias fit used on another engine is only caught if the caller checks. | — |
| X-6 | Equity calibrators take **loose scalars** (spot, rate, dividend, forward), not curves. | Their parameters carry no curve risk unless the chain node computes the forward from the market (`sabr_smile_node` with `equity_forward`). | Curve-aware equity calibration — [roadmap §5.5](../roadmap.md#unified-calibration-pricing-and-risk) |
| X-7 | Short-rate models are **exact-fitted to the discount curve** (the curve is upstream input, bound by `price`), and their parameters are **constant in time** (scalar σ, a, …). | A per-trade fit to *n* co-terminals with fewer than *n* free parameters is a least-squares compromise, not an exact repricing (~1–2% misses measured; examples 14, 18). | Piecewise-constant σ(t) and the co-terminal bootstrap — [roadmap §5.6](../roadmap.md#unified-calibration-pricing-and-risk) |
| X-8 | Quotes are taken as given: no staleness / outlier screening, no quote-time alignment. | A stale but feasible quote passes its own level and surfaces downstream, if at all (example 16, scene 3). | — |

## Curves

### Joint curve bootstrap

| | |
|---|---|
| **Entry points** | `bootstrap_curve_graph`, `curve_graph_node`, `bootstrap_simultaneous` (single-curve wrapper), `fitted_curve_quotes`, `quote_jacobian` |
| **Fits / holds fixed** | Pillar log-discount-factors of every curve in the graph; seed flat 4%. The system must be square (one instrument per pillar) and closed (every curve an instrument touches is in the graph). |
| **Targets & objective** | `root`. Each instrument's own residual (rate identity or PV identity — units are not uniform across instrument types, and are not normalised). Labels name each instrument (`"3:SwapRate"`, or the quote key in the node). |
| **Engine & provenance** | Instrument identities, no model; no provenance. |
| **Settings** | `EXACT_SETTINGS` (Newton, 1e-10). |
| **Market inputs** | Any number of curves (multi-curve, cross-currency via FX instruments); fixings history. |
| **Assumptions** | Log-linear interpolation in discount factor (`log_linear_df` — the only mode); values held flat beyond the last pillar; futures convexity through a pluggable hook (zero by default). |
| **Limitations** | No alternative interpolation, no turn-of-year step, no Hull-White futures convexity, no CSA discounting, no over-determined (least-squares) variant, fixings not seasoned into OIS / futures residuals ([curves guide § 8](../guide/curves.md#8-what-is-not-yet-implemented)). Failure localisation (`failed_labels()`): a non-finite quote is named exactly; otherwise the nodes within `FAILURE_CLUSTER_FRACTION` = 0.5 of the worst residual; when Newton diverges to NaN everywhere (an infeasible quote) the cluster is taken on the residuals at the flat starting guess (`extras["initial_residuals"]`). On the 14-quote synthetic build this names a single grossly off quote (20%, −50%, 60% OIS) alone and `max_steps=1` names the long-end cluster (4 of 14), but it does not localise a subtle inconsistency spread over several quotes (example 16, scene 4; `test_bootstrap_graph.py::TestFailureLocalisation`). |
| **Validation** | `tests/test_curves/test_bootstrap_graph.py` (repricing residual < 1e-10–1e-12 for 2/3/4-curve builds; equals `bootstrap_simultaneous` to 1e-12); `test_quote_jacobian.py` (autodiff vs FD rtol 1e-4); QuantLib comparison for the single-curve build. |
| **Improve** | Over-determined bootstrap, alternative interpolation, turns — [roadmap Tier 1.1](../roadmap.md#11-multi-curve-bootstrapping); leave-one-out / damped re-solve on failure for multi-quote inconsistencies, and an a-priori outlier check against neighbouring tenors (open). How-to: [curves guide](../guide/curves.md); maths: [theory § 3.8](../theory/curves.md). |

### Sequential bootstrap

| | |
|---|---|
| **Entry points** | `bootstrap_sequential` |
| **Fits / holds fixed** | One pillar at a time, closed form, from `DepositRate` / `FRA` / `SwapRate` only. |
| **Targets & objective** | Explicit pillar-by-pillar algebra; returns a plain `DiscountCurve`, **not** a `CalibrationResult` (no diagnostics). |
| **Engine & provenance** | — |
| **Settings** | None (no solver). |
| **Market inputs** | Single curve; ignores fixings. |
| **Assumptions** | Instruments non-overlapping and ordered so each adds exactly one pillar. |
| **Limitations** | Not JIT-able as a whole; other instrument types raise `TypeError`; no quote risk path (use the joint bootstrap). |
| **Validation** | `tests/test_curves/test_bootstrap.py` (deposits 1e-10, FRA forwards 1e-8); QuantLib DFs abs < 1e-10. |
| **Improve** | Superseded by the joint bootstrap for anything beyond teaching / cross-checks. |

### Survival curve from CDS spreads

| | |
|---|---|
| **Entry points** | `from_cds_spreads` (with `from_hazard_rates`) |
| **Fits / holds fixed** | Hazard rates $h_i = s_i / (1-R)$, recovery $R = 0.4$ by default. |
| **Targets & objective** | `explicit` — the closed-form **credit triangle**, not a repricing bootstrap; returns a curve, not a `CalibrationResult`. |
| **Engine & provenance** | — |
| **Settings** | None. |
| **Market inputs** | Spreads, recovery; no discount curve. |
| **Assumptions** | Flat-spread / continuous-premium approximation; piecewise-constant hazard (log-linear survival). |
| **Limitations** | Does not reprice the CDS quotes; first spread ignored; no CDS pricer to bootstrap against yet. |
| **Validation** | `tests/test_curves/test_survival.py` (triangle survival atol 5e-4). |
| **Improve** | Repricing hazard bootstrap once the CDS pricer lands — [roadmap Tier 3.4](../roadmap.md#34-credit-derivatives). |

### Inflation curve from zero-coupon rates

| | |
|---|---|
| **Entry points** | `from_zc_rates` |
| **Fits / holds fixed** | CPI pillars $\mathrm{CPI}(T_i) = \mathrm{CPI}_0 (1+z_i)^{T_i}$. |
| **Targets & objective** | `explicit` algebra; returns an `InflationCurve`, not a `CalibrationResult`. |
| **Engine & provenance** | — |
| **Settings** | None. |
| **Market inputs** | ZC swap rates, base CPI. |
| **Assumptions** | Log-linear CPI interpolation, flat beyond the range; no indexation lag beyond the instrument's own. |
| **Limitations** | No seasonality; no YoY-quote bootstrap; no dedicated test module. |
| **Validation** | Exercised through the inflation pricer tests only. |
| **Improve** | Seasonality — [roadmap](../roadmap.md); how-to: [calibration guide § 2.6](../guide/calibration.md#26-inflation). |

## Equity / FX smiles and stochastic-vol models

### SABR smile

| | |
|---|---|
| **Entry points** | `calibrate_sabr`, `sabr_smile_node` |
| **Fits / holds fixed** | α, ρ, ν always; β **held fixed at 0.5 by default** in every SABR entry point (`calibrate_sabr`, `calibrate_sabr_surface`, `build_sabr_caplet_surface`, the swaption cube and the chain nodes; the node can also read it from a `beta_key` fixed input); `fixed_beta=None` calibrates it, `0.0` for normal-backbone rates smiles. Seeds: α from the median vol, ρ −0.2, ν 0.3 (`calibrate_sabr`) / 0.4 (node). |
| **Targets & objective** | `least_squares`, absolute implied-vol residual, weights one. The node fits all expiries in one joint solve. |
| **Engine & provenance** | Hagan lognormal (or normal via `vol_fn`); no provenance. |
| **Settings** | `FIT_SETTINGS` (Levenberg-Marquardt, 1e-8). |
| **Market inputs** | Scalar forward and expiry (`calibrate_sabr`); the node computes forwards from the market (`equity_forward`). |
| **Assumptions** | Hagan's asymptotic formula is accurate at the quoted strikes/expiries; the quote convention matches `vol_fn` (lognormal vs normal). |
| **Limitations** | β and α are poorly jointly identified: freeing β costs ~260× in conditioning ([calibration guide § 8](../guide/calibration.md#8-is-the-fit-trustworthy-conditioning-at-every-level)). Fitting normal quotes with the lognormal formula gives a small residual and wrong parameters ([pitfall 01](https://github.com/ACea15/VALAX/blob/main/examples/pitfalls/01_sabr_vol_convention_mismatch.py)). Fixing β at 0.5 is a modelling choice (the backbone); a smile whose backbone is clearly different (e.g. normal rates, β ≈ 0) must pass its own β. |
| **Validation** | `tests/test_calibration/test_sabr_calibration.py` (fixed-β α atol 1e-4, ρ/ν 1e-3); QuantLib smile ≤ 5 bp. |
| **Improve** | How-to: [calibration guide § 3](../guide/calibration.md#3-sabr-smile-calibration). |

### SABR surface

| | |
|---|---|
| **Entry points** | `calibrate_sabr_surface` |
| **Fits / holds fixed** | One SABR slice per expiry (β fixed at 0.5 by default; `fixed_beta=None` fits it per slice). |
| **Targets & objective** | Batched `least_squares` (labels `"1Y"`, `"3M"`, …), vol residual, weights one; short rows padded with zero weight. |
| **Engine & provenance** | Hagan lognormal; no provenance. |
| **Settings** | `FIT_SETTINGS`. |
| **Market inputs** | Scalar forwards per expiry. |
| **Assumptions** | Between expiries, α, β, ρ, ν and the forward are **interpolated linearly in expiry** and held flat outside the grid. |
| **Limitations** | Parameter interpolation is not arbitrage-aware; slices are independent (no term-structure smoothing). |
| **Validation** | `tests/test_surfaces/test_sabr_surface.py` (batched = per-slice fits to 1e-12; padding neutral). |
| **Improve** | How-to: [vol surfaces § SABR surface](../guide/vol-surfaces.md#sabr-surface). |

### SVI slice and surface

| | |
|---|---|
| **Entry points** | `calibrate_svi_slice`, `calibrate_svi_surface` |
| **Fits / holds fixed** | a, b, ρ, m, σ per expiry (b, σ positive, ρ tanh). |
| **Targets & objective** | `least_squares` (surface batched by expiry), absolute vol residual, weights one; total variance floored at 1e-10. |
| **Engine & provenance** | SVI closed form; no provenance. |
| **Settings** | `FIT_SETTINGS`. |
| **Market inputs** | Scalar forwards. |
| **Assumptions** | Total variance interpolated **linearly in expiry**; below the first slice extrapolated linearly through the origin, flat above the last. |
| **Limitations** | **No arbitrage constraints are imposed** — neither butterfly within a slice nor calendar across slices; linear total-variance interpolation preserves calendar ordering only between slices that already satisfy it. |
| **Validation** | `tests/test_surfaces/test_svi.py` (round trip atol 1e-4); batched vs per-slice 1e-10. |
| **Improve** | Constrained (KKT) arbitrage-free SVI — [roadmap §5.6](../roadmap.md#unified-calibration-pricing-and-risk). How-to: [vol surfaces § SVI](../guide/vol-surfaces.md#svi-surface). |

### Heston

| | |
|---|---|
| **Entry points** | `calibrate_heston` (and any `price_fit_node` with a Heston template) |
| **Fits / holds fixed** | v0, κ, θ, ξ, ρ (positive / tanh); any subset held fixed via `fixed_params` (reported as fixed inputs); rate and dividend fixed. Seeds v0 0.04, κ 2, θ 0.04, ξ 0.5, ρ −0.7. |
| **Targets & objective** | `least_squares`, **absolute price** residual, weights one; `expiry` a scalar or one per quote (**multi-expiry**), the pricer vmapped per (strike, expiry). |
| **Engine & provenance** | Injected `pricing_fn` (COS by default in the tests); provenance only if passed. |
| **Settings** | `FIT_SETTINGS`. |
| **Market inputs** | Loose scalars (X-6). |
| **Assumptions** | The basket's expiries pin κ, ξ, v0 jointly; no Feller constraint. |
| **Limitations** | On a single smile κ, ξ and v0 lie in a flat valley: on example 17's SABR market the five-parameter 1Y-only fit does not converge in 512 steps (κ 8.87, ξ 4.29, 0.5Y IV error 1243 bp), whereas two expiries converge in 27 steps (κ 1.39, ξ 1.06; 23 / 13 bp) and two expiries with κ fixed in 11. Unweighted price residuals over-weight expensive options (X-3). |
| **Validation** | `tests/test_calibration/test_heston_calibration.py` (COS round trip v0/θ 1e-6, κ/ξ 1e-4; non-convergence reported; fixed parameters respected and reported; multi-expiry converges where single-expiry does not). |
| **Improve** | Vol-space (vega-weighted) residuals and an optional Feller penalty (open). How-to: [calibration guide § 4](../guide/calibration.md#4-heston-stochastic-vol-calibration). |

## Rates volatility

### Caplet strip

| | |
|---|---|
| **Entry points** | `strip_caplet_vols`, `caplet_strip_node`, `caplet_strip_level`, `strip_columns_level` |
| **Fits / holds fixed** | One vol per quoted cap bucket, **piecewise constant** across the bucket's incremental optionlets. |
| **Targets & objective** | `root`, solved bucket by bucket (scalar Newton) and differentiated as one lower-triangular system; residual = cap price per unit notional (flat-vol quote → target by the same pricer). Buckets labelled by the cap's **maturity** from the valuation date (`"cap 2Y"`, matching the `USD.CAP.2Y.*` quote keys); in the chain, strike columns are batched and nodes named per (strike, bucket), `"K=3.00% cap 2Y"`. |
| **Engine & provenance** | Black-76 or Bachelier caplets. |
| **Settings** | `STRIP_SETTINGS` (Newton, 1e-12, 100 steps). |
| **Market inputs** | Discount + projection curve (dual-curve). |
| **Assumptions** | Caps nested and ascending, one strike / direction / day count per column; the spot-starting (already fixed) period is excluded. |
| **Limitations** | A failed bucket installs **NaN**, so the later buckets of its column fail too (the strip is triangular); `root_cause_labels()` and the `check()` message name the first failure, and the downstream optionlet SABR reports every NaN-fed expiry (example 16, scene 3). The bucket root-find is **unconstrained in sign**: an infeasible quote can "converge" to a negative bucket vol (e.g. −4.8% for a 2Y lognormal cap vol of 3.2% against a 1Y of 30%) with no failure reported. A stale-but-feasible quote strips into jagged vols and fails only downstream. |
| **Validation** | `tests/test_surfaces/test_caplet_stripping.py` (every cap repriced < 1e-10, single- and dual-curve; `TestBucketLabels`, `TestFailedBucketIsNaN`, `TestStripColumnsNaming`); QuantLib cap-on-strip < 1e-6. |
| **Improve** | Constrain bucket vols ≥ 0 (positive transform, or report σ < 0 as not converged); a jaggedness / monotonicity check on the stripped term structure (open). How-to: [vol surfaces § stripping](../guide/vol-surfaces.md#cap-caplet-optionlet-stripping-rates). |

### Optionlet SABR surface

| | |
|---|---|
| **Entry points** | `build_sabr_caplet_surface`, `optionlet_sabr_node`, `strip_optionlet_surface` |
| **Fits / holds fixed** | SABR per optionlet expiry; β **fitted by default in `build_sabr_caplet_surface`**, **fixed at 0.5 in `optionlet_sabr_node` / `strip_optionlet_surface`** (α and β are not jointly identifiable from one cap smile). |
| **Targets & objective** | Batched `least_squares` by expiry, vol residual on the stripped vols; the node has no quotes of its own (its adjoint flows to the cap quotes and curves). |
| **Engine & provenance** | Hagan lognormal or shifted-normal. |
| **Settings** | `FIT_SETTINGS` (strip: `STRIP_SETTINGS`). |
| **Market inputs** | The upstream strip and curves. |
| **Assumptions** | A **smoothing** of the strip, not a recoverable truth; parameters interpolated linearly in expiry; sticky-strike dynamics by default. |
| **Limitations** | With few noisy strikes per expiry, some draws push ρ to −1 (κ ~ 1e12) and the strict fit fails, named per expiry (example 15). `price` has **no built-in (`Cap`, smile) route**: caps are priced off the surface by `cap_price`, or through a user-registered pricer (example 15). |
| **Validation** | `tests/test_surfaces/test_optionlet_surface.py`; `tests/test_calibration/test_rates_vol_nodes.py` (adjoint = autodiff rtol 1e-6, = full-recalibration FD rtol 2e-4). |
| **Improve** | Register caps / swaptions off the vol surfaces in `price` — [roadmap §5.6](../roadmap.md#unified-calibration-pricing-and-risk) ("Every pricer reachable from `price`"). |

### Swaption cube

| | |
|---|---|
| **Entry points** | `calibrate_swaption_cube_from_curves`, `calibrate_swaption_cube_from_graph`, `swaption_cube_level`, `swaption_cube_node` |
| **Fits / holds fixed** | SABR per (expiry, tenor) cell; β fixed at 0.5 by default (`_from_curves`, `_from_graph`, `swaption_cube_node`); pass `0.0` for normal-backbone quotes, `None` to fit it. |
| **Targets & objective** | Batched `least_squares` (labels `"5Yx10Y"`), absolute vol residual, weights one; ATM-offset quotes become strikes $F + \Delta$. |
| **Engine & provenance** | Hagan lognormal or shifted-normal (synthetic universe: normal vols, shift 3%, β 0). |
| **Settings** | `FIT_SETTINGS`. |
| **Market inputs** | Discount + projection curves; each cell's forward and annuity are **recomputed from the curves** inside the level, so the cube carries curve risk. |
| **Assumptions** | Parameters and forwards interpolated bilinearly in (expiry, tenor), flat outside the grid; dynamics as declared by `quote_form` (sticky strike / moneyness; SABR backbone never implicit). |
| **Limitations** | Strike is never tabulated (the smile is the SABR formula); interpolation is not arbitrage-aware. |
| **Validation** | `tests/test_surfaces/test_swaption_cube.py` (node round trip < 1e-8); `test_rates_vol_nodes.py` (noise-free α ≤ 1e-10, ρ/ν ≤ 1e-8). |
| **Improve** | How-to: [vol surfaces § swaption cube](../guide/vol-surfaces.md#swaption-cube-curve-aware-calibration). |

### Swaption premium → normal vol

| | |
|---|---|
| **Entry points** | `swaption_implied_normal_vol`, `swaption_implied_normal_vol_from_graph` |
| **Fits / holds fixed** | One normal vol per premium. |
| **Targets & objective** | `root`, Bachelier premium − quoted premium (currency units); seed from the straddle bound. |
| **Engine & provenance** | Bachelier with physical or cash settlement. |
| **Settings** | `INVERSION_SETTINGS` (Newton, 1e-12, 100 steps). |
| **Market inputs** | Discount + projection curves. |
| **Assumptions** | Premium above intrinsic (else no solution). |
| **Limitations** | — |
| **Validation** | `tests/test_surfaces/test_swaption_premium.py` (round trip rtol 1e-10; dσ/dp = 1/vega rtol 1e-6). |
| **Improve** | How-to: [vol surfaces § premium quotes](../guide/vol-surfaces.md#premium-quotes). |

## Short-rate models

### Hull-White

| | |
|---|---|
| **Entry points** | `calibrate_hull_white` |
| **Fits / holds fixed** | **Both a and σ fitted by default** (`fixed_mean_reversion=None`), although fixing a is recommended for a co-terminal basket; seeds a 0.05, σ 0.01; both constant in time (X-7). |
| **Targets & objective** | `least_squares`, **relative** price residual $(P_k - q_k)/\max(\lvert q_k\rvert, 10^{-8})$, weights one; vol quotes → prices with `swaption_prices_from_vols` on the same curve pair. |
| **Engine & provenance** | Jamshidian (`CLOSED_FORM_PROVENANCE`, exact). |
| **Settings** | `BASKET_SETTINGS` (BFGS, 1e-12). |
| **Market inputs** | Discount curve (exact fit) + optional projection curve through the frozen tenor-basis overlay. |
| **Assumptions** | One Gaussian factor; deterministic tenor basis. |
| **Limitations** | Cannot reproduce a smile or a whole surface (5.9% max error on a surface, [calibration guide § 6](../guide/calibration.md#6-dual-curve-rates-calibration-ois-discounting)); constant σ cannot match a co-terminal strip exactly (~1.2%, example 14). |
| **Validation** | `tests/test_calibration/test_hull_white_calibration.py` (parameters recovered rel 1e-5); `test_dual_curve_calibration.py`. No QuantLib cross-check of the calibrator. |
| **Improve** | Piecewise-constant σ(t) and the co-terminal bootstrap — [roadmap §5.6](../roadmap.md#unified-calibration-pricing-and-risk). Maths: [Hull-White swaptions](../theory/hull-white-swaptions.md). |

### G2++

| | |
|---|---|
| **Entry points** | `calibrate_g2pp` |
| **Fits / holds fixed** | All five (a, b, σ, η, ρ) free by default; seeds a 0.5, b 0.1, σ 0.01, η 0.008, ρ −0.7; constant in time. |
| **Targets & objective** | As Hull-White (relative price residual). |
| **Engine & provenance** | Brigo-Mercurio quadrature (64 Gauss-Hermite nodes; `CLOSED_FORM_PROVENANCE`). |
| **Settings** | `BASKET_SETTINGS` with 512 steps. |
| **Market inputs** | As Hull-White (dual-curve via frozen basis). |
| **Assumptions** | Two Gaussian factors, constant correlation. |
| **Limitations** | (a, b) are weakly identified by an ATM surface; a co-terminal basket alone leaves the two factors unidentified (condition number ~1e16; example 18) — add a short-tenor option. |
| **Validation** | `tests/test_calibration/test_g2pp_calibration.py` (vols rel 2e-2 with mean reversions pinned; repricing < 1e-3). |
| **Improve** | Piecewise σ(t) with a fixed η/σ ratio — [roadmap §5.6](../roadmap.md#unified-calibration-pricing-and-risk). Maths: [G2++ § 6](../theory/g2pp.md). |

### Cheyette (1F / 2F)

| | |
|---|---|
| **Entry points** | `calibrate_cheyette`, `calibrate_cheyette_2f` |
| **Fits / holds fixed** | 1F κ, σ, β (skew) free by default; 2F κ₁, κ₂, σ₁, σ₂, β₁, β₂, ρ; pinning κ is recommended. |
| **Targets & objective** | As Hull-White (relative price residual). |
| **Engine & provenance** | 1F `engine="frozen"` (default; 121×60 Crank-Nicolson PDE) → `modelling_bias`, **engine-bound**; `engine="exact"` (exact-`y` 2-D PDE) → `discretised`; 2F always frozen. |
| **Settings** | `CHEYETTE_SETTINGS` (BFGS, 1e-10). |
| **Market inputs** | **Single curve only** (a distinct projection curve raises). |
| **Assumptions** | Frozen-`y` replaces the stochastic accumulator by its deterministic limit — a bias that does not converge away under refinement. |
| **Limitations** | Parameters valid only on the calibration engine and grid (`check_engine`); no dual-curve; β needs a strike slice in the basket. |
| **Validation** | `tests/test_calibration/test_cheyette_calibration.py` (σ rel 1–2e-2, β abs 1–2e-2; engine binding). |
| **Improve** | Dual-curve Cheyette recipes — [roadmap §5.5](../roadmap.md#unified-calibration-pricing-and-risk). Maths: [Cheyette § 9](../theory/cheyette.md). |

### Per-trade short-rate basket

| | |
|---|---|
| **Entry points** | `swaption_basket_node`, `coterminal_swaptions`, `basket_fit_level` |
| **Fits / holds fixed** | The template's `transforms` fields; `fixed_fields` held at reported fixed inputs; optional zero **target shifts** (∂V/∂T_k = the vega profile). Validated set-ups: HW σ (a fixed); G2++ σ, η (a, b, ρ fixed; basket + the short-tenor option at the first exercise); Cheyette σ, β (κ fixed; basket + one off-strike option). |
| **Targets & objective** | `least_squares`, relative price residual against the basket **priced off the calibrated cube and curves**; weights one. |
| **Engine & provenance** | `price(..., method=...)`, default `"analytic"` (Jamshidian / Brigo-Mercurio); provenance from `engine_provenance` (`exact`, `discretised` with the grid, or engine-bound Cheyette). |
| **Settings** | `BASKET_SETTINGS` (Cheyette: pass `CHEYETTE_SETTINGS`). |
| **Market inputs** | The calibrated market (curves bound by `price`; the trade's discount / projection ids). |
| **Assumptions** | The basket is the trade's hedge universe; constant model parameters (X-7). |
| **Limitations** | Over-determined, so the basket is not repriced exactly (1–2%, examples 14 / 18). Calibrating on the closed form leaves the Bermudan's grid error uncancelled; fitting the basket on the Bermudan's own grid makes the Bermudan several times more accurate on that grid (41×41×40: −0.55 vs −0.07 bp) at the cost of grid-bound parameters (example 18). With `smooth_payoff=True` (the default on `PDEConfig` / `PDEConfig2D`) the PDE samples kinked payoffs and exercise projections smoothed over each node's cell — the exact cell average in 1-D (HW, Cheyette; Pooley–Vetzal–Forsyth 2003), a bias-neutral C¹ variant in 2-D (G2++) — so the discrete Bermudan price is C¹ in quotes and parameters and FD at ordinary steps (1e-5 on quotes / targets, 1e-4 on mean reversions) matches the adjoint to ≤ 5e-6 relative; before, steps ≥ 1e-6 could straddle a node crossing (2–4% errors) and pointwise autodiff on a kink was ~6% off the smooth slope. The smoothing bias is O(h²); a grid-bound calibration is bound to the flag too (it is part of the recorded config). Basket, template and weights travel in `LevelInputs.data`, so `condition_number()` works on results returned from `filter_jit` and per trade inside `filter_vmap` (`TestJittedResultDiagnostics`). |
| **Validation** | `tests/test_calibration/test_per_trade_basket.py` (adjoint = autodiff rtol 1e-6; full-recalibration FD rtol 5e-5; `filter_vmap` = per-trade fits, HW 1e-10, G2++ 5e-8); `tests/test_pde/test_payoff_smoothing.py` (Bermudan FD = `jax.grad` across steps, HW and G2++). |
| **Improve** | Co-terminal **bootstrap** with piecewise σ(t): a square, triangular `root` level that reprices every co-terminal exactly — [roadmap §5.6](../roadmap.md#unified-calibration-pricing-and-risk). How-to: [From Quotes to Risk § rates](../guide/calibration-to-risk.md#rates-per-trade-model-calibration); maths: [Calibration Chains § I.10](../theory/calibration-chains.md#i10-per-trade-model-levels-and-the-vega-profile). |

### LMM

No calibrator exists (`LMMModel` is priced uncalibrated; it also still embeds its
curve). Tracked on the [roadmap](../roadmap.md#unified-calibration-pricing-and-risk).

## Generic and market-input levels

### Price fit (any model through `price`)

| | |
|---|---|
| **Entry points** | `price_fit_node` |
| **Fits / holds fixed** | The template's `transforms` fields; `fixed_fields` as reported fixed inputs. |
| **Targets & objective** | `least_squares`, **absolute price** residual; `quote_form` `"price"`, `"lognormal_vol"` (Black-Scholes at the quoted vol, on the bound rate/dividend) or a callable; weights one. |
| **Engine & provenance** | `price(..., method)` (default `"auto"`, resolved per instrument); provenance from `engine_provenance` (the most restrictive across the basket). |
| **Settings** | `FIT_SETTINGS`. |
| **Market inputs** | The upstream market. |
| **Assumptions** | Whatever the chosen model and engine assume. |
| **Limitations** | Absolute price residuals weight expensive options more (X-3) — pass `weights=1/vega` for vol-like residuals. |
| **Validation** | `tests/test_greeks/test_quote_risk.py::TestModelLevel` (Heston adjoint = autodiff). |
| **Improve** | How-to: [From Quotes to Risk § Tier 2](../guide/calibration-to-risk.md#tier-2-a-calibrated-model). |

### Explicit market inputs

| | |
|---|---|
| **Entry points** | `market_inputs_node` |
| **Fits / holds fixed** | Nothing: quotes map directly to spots, dividends, flat curves (`flat_curve`, act/365) and constant vols. |
| **Targets & objective** | `explicit`; differentiated by plain AD. |
| **Assumptions** | Constant parameters (Tier 0). |
| **Limitations** | — |
| **Validation** | `tests/test_calibration/test_chain.py`; example 10 (textbook Black-Scholes Greeks). |
| **Improve** | — |

## Stochastic-local volatility (SLV)

| | |
|---|---|
| **Entry points** | `calibrate_slv`, `calibrate_slv_leverage` |
| **Fits / holds fixed** | The leverage grid L(t, k), given a Heston backbone — fitted inside `calibrate_slv` (`fixed_params`, multi-expiry) or supplied with `heston=` (a model or a result, recorded as the child). Defaults: `method="particle"`, `n_iterations=2`, `ridge=30` pseudo-particles (kernel only), Silverman bandwidth, L clipped to [0.05, 5], `n_substeps=1` particle steps per leverage row; the Dupire target goes through the default local-vol guard. |
| **Targets & objective** | `fixed_point`: an explicit level. Each pass is a **Guyon–Henry-Labordère sweep**: row *i* = σ_Dupire / √Ê[V \| k] from particles simulated under rows 0…i−1, with the pricer's midpoint-in-time leverage query (the unbuilt end-of-step row is predicted). The residual is $L_K - L_{K-1}$ — the Monte Carlo noise of the estimator across re-simulated passes; converged when $\max\lvert\Delta L\rvert \le$ `atol`; `n_iterations=1` is never converged; `clip_fraction` reported. |
| **Engine & provenance** | Particle Monte Carlo (QE variance, midpoint log-spot); no provenance. |
| **Settings** | `SLV_SETTINGS` (only `atol = 5e-2` is read). |
| **Market inputs** | The target implied-vol surface (Dupire, guarded), loose scalars. |
| **Assumptions** | The Markovian projection reprices the smile; the kernel estimate of E[V \| k] is shrunk toward the unconditional mean by `ridge` pseudo-particles. |
| **Limitations** | Vanilla error at example 17's budget (50k particles, 48 rows, full ξ ≈ 1.1): 9–24 bp RMS per expiry across passes; ≈ 25–37 bp at 20k particles, 24 rows, ξ ≈ 1.3 ([SLV guide § limitations](../guide/slv.md#limitations-and-known-approximations)). The max\|ΔL\| noise floor is 0.11–0.14 on a 48×17 grid (sparse early / wing cells), so the default `atol = 5e-2` needs roughly 10× more particles (example 17 uses 0.2). A swarm step coarser than the pricer's costs tens of bp at high ξ (use `n_substeps`). AD through the Heston parameters is only pathwise — Andersen-QE switches branch, so a bump FD through the Heston refit does not converge; with the backbone held, AD matches FD to 2e-4. Local-vol paths are evaluated through the guard (|k| ≤ 8·√(ATM total variance), implied-variance fallback, variance in [1e-8, 25 × ATM]); `guard=None` gives raw Dupire. |
| **Validation** | `tests/test_calibration/test_slv_calibration.py` (flat-surface one-pass exactness; high-ξ repricing at least as good as the old two-pass scheme; ridge semantics; `heston=` child; jit = eager); `tests/test_pricing/test_local_vol_paths.py` (guarded NaN repro finite, gradients finite, guard = raw inside the domain); example 17 (AD vs FD with the backbone held, 1.8e-4). |
| **Improve** | A smooth / continuous QE variant for parameter risk; Fokker–Planck PDE calibration (SLV-2); a norm-weighted convergence metric; averaging passes (open). Strategy: [SLV § fixed point](../guide/slv.md#how-the-fixed-point-is-reported), [Calibration Chains § I.8](../theory/calibration-chains.md#i8-fixed-point-levels). |

## Not calibrations

- **Dupire local volatility** (`dupire_local_vol`, `LocalVolModel`) is a
  pricing-time map from an implied-vol surface, not a fitted level; its
  quality is the quality of the surface it reads (see
  [theory § 4.4](../theory/volatility.md)). Monte Carlo and the SLV target
  query it through `guarded_local_vol` (`DEFAULT_LOCAL_VOL_GUARD`): exact
  Dupire inside the domain of an arbitrage-free surface; outside it, the
  moneyness is clamped, a non-positive Gatheral denominator falls back to the
  implied variance, and the variance is banded — so an extreme path can no
  longer turn a price into NaN.
- **Quote conversions and helpers** (`swaption_prices_from_vols`,
  `curve_quote_vector`, `period_label`) transform inputs; they make no
  modelling choice of their own beyond the conventions they document.
