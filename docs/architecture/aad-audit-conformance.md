# Multi-Curve / AAD Risk: Audit Conformance Map

This page is the scoreboard for VALAX against the *Multi-Curve Calibration and
AAD Risk: Audit Specification* — the requirement-by-requirement standard for a
multi-curve interest-rate pricing stack whose quote sensitivities are computed
by adjoint algorithmic differentiation, with every calibration differentiated
through the implicit function theorem (IFT).

It exists so that nobody has to infer conformance from memory or from reading
the code. Each requirement gets one row: the implementing symbol, a status, and
— where the status is not ✅ — a one-line statement of exactly what is missing.

!!! info "Scope and framing"

    The specification works a single market through as its illustration. VALAX
    is **currency- and index-agnostic**: curves are addressed by identifier
    (`CurveSpec.curve_id`, the frozen `<CCY>.<INDEX>.<TENOR>[.<QUALIFIER>]`
    alphabet), so nothing on this page is specific to one jurisdiction's
    conventions. Where the specification names a market's default convention,
    the corresponding requirement here is read as "the convention is explicit,
    consistent and stated" rather than "it equals that particular default".

    Out of scope in the specification, and therefore here: XVA, inflation,
    stochastic basis models, credit.

**Related reading.** The two canonical feature maps remain
[the coverage matrix](../applications/instrument-pricing.md) and
[the roadmap](../roadmap.md); this page does not duplicate them, it scores a
different axis. Engine mechanics live in the solver guides; the quote-space
risk API is documented in [API → Greeks](../api/greeks.md#quote-space-hedge-space-risk).

## Status legend

| Status | Meaning |
|---|---|
| ✅ | Implemented and covered by a test |
| 🟡 | Partially implemented — the row's note states the gap precisely |
| ❌ | Not implemented |
| n-a | Not applicable to VALAX's current modelling choices |

The specification's own grading applies to the requirements themselves:
**MUST** means a defect if violated; **SHOULD** means a deviation that must be
documented and justified. The MUST/SHOULD column below reproduces that.

## The three calibration levels

The specification layers the stack; VALAX's implementation status differs
sharply by layer, so it is worth stating up front.

| Level | Unknowns | VALAX status |
|---|---|---|
| **1a — Curves** | Curve node values | ✅ Joint Newton bootstrap, implicit-diff quote Jacobian, **full adjoint chain to quote space** |
| **1b — Volatility objects** | Optionlet vols, SABR cube parameters | 🟡 Cap/optionlet stripping is curve-coupled and differentiable; the **swaption cube is not** — it takes precomputed forwards, so `dθ_v/dθ_c = 0` identically |
| **2 — Model** | `σ(t)`, skew parameters | 🟡 Dual-curve Hull-White / G2++ calibration ships; no `λ_m` adjoint level, so model calibration is not differentiated by IFT |

Because the levels compose, the Layer-1b gap is load-bearing: it blocks three
of the six paths by which curves enter Layer 2 (see **M-3** below).

---

## N — Notation and conventions

| ID | Requirement | MUST/SHOULD | Implementing symbol | Status | Gap |
|---|---|---|---|---|---|
| N-1 | Accrual fractions in the leg's own day count; one stack-wide model day count; never mixed | MUST | `valax.dates.year_fraction`; per-leg static `day_count` fields on every quote type | ✅ | — |
| N-2 | Every rate annualised and paired with its own day count | MUST | Per-leg `fixed_day_count` / `float_day_count` on `IborSwapRate`, `TenorBasisSwap`, `CrossCurrencyBasisSwap` | ✅ | — |
| N-3 | Discount cash flows at the payment date, not accrual end, where a payment lag exists | MUST | `payment_dates` / `payment_lag` on `SwapRate`, `OISSwapRate`, `IborSwapRate` (per leg), `TenorBasisSwap` (per leg), `CrossCurrencyBasisSwap` (per leg); `_floating_leg_pv(accrual_end_dates=…, payment_dates=…)`; `_compounded_float_leg_pv` | ✅ | Accrual and payment are now separate roles: projection and `τ` come from the accrual schedule, discounting from the payment schedule. Either an explicit business-day-adjusted payment array or a uniform calendar-day `payment_lag`; supplying both raises. Default is `T^p = T^e`, which preserves every pre-existing number bit for bit. Gated by `tests/test_curves/test_instrument_residuals.py::TestPaymentLagTelescoping` (T-5) and `…::TestPaymentLagPerLeg` |
| N-4 | Projection (pseudo-discount) factors never discount | MUST | `_floating_leg_pv` discounts on `curves_touched[0]` and projects on `curves_touched[1]` | ✅ | — |
| N-5 | The index period is generated from the index definition, not assumed equal to the coupon accrual period; stubs explicit | MUST | `IborSwapRate.fixing_dates` (explicit per-coupon fixing dates); `valax.dates.generate_schedule` (stubs, EOM rolls) | 🟡 | Fixing dates are supplied by the caller rather than derived from a versioned index definition; there is no index registry |
| N-6 | Schedules, calendars and conventions from versioned reference data | SHOULD | — | ❌ | No business-day calendars at all; conventions are per-instrument static fields |

## Q — Market instruments and quotes

| ID | Requirement | MUST/SHOULD | Implementing symbol | Status | Gap |
|---|---|---|---|---|---|
| Q-1 | Each quote stored with definition, type, units, source and timestamp under one snapshot ID | MUST | — | ❌ | `MarketState` carries no snapshot identifier, as-of date or per-quote provenance |
| Q-2 | Each quote enters exactly one residual; no quote twice, no residual without a quote | MUST | `bootstrap_graph._validate_square`, `_validate_curves_touched` | ✅ | Enforced eagerly: the build raises unless `n_instruments == n_pillars` |
| Q-3 | Strikes keep their quoted form (an ATM offset stays an offset; the absolute strike is recomputed from the curve) | MUST | `SwaptionCube.quote_form` / `OptionletVolSurface.quote_form`; `calibrate_swaption_cube_from_curves(quote_form="atm_offset")` | ✅ | The quote form is a declared static field, and the curve-aware fit converts offsets to absolute strikes per node using that node's **curve-derived** `F_0`. Under `"atm_offset"` the adjoint's strike grid is rebuilt from the forward too, so `dR_v/dF_0` carries the grid-shift term — gated in `tests/test_greeks/test_vol_adjoint.py` |
| Q-4 | Vol quotes carry their convention (normal, or shifted lognormal with the shift) | MUST | `SwaptionCube.is_normal` / `.shift` / `.quote_form`; `OptionletVolSurface`; `valax.surfaces.shifted` | 🟡 | Strike form is now carried and validated (Q-3). The *vol* convention (`is_normal`, `shift`) still lives on the fitted object rather than on the quote, so a normal/lognormal mismatch is caught only by non-convergence |
| Q-5 | Quasi-market inputs live in `p`, versioned like quotes, with sensitivities reported separately | MUST | — | ❌ | No `p` container; `fixed_beta`, `shift`, convexity adjustments and turn spreads are loose arguments |
| Q-6 | Stale or overridden quotes flagged and excluded or down-weighted by an explicit rule | SHOULD | — | ❌ | — |

## R — Residual framework

| ID | Requirement | MUST/SHOULD | Implementing symbol | Status | Gap |
|---|---|---|---|---|---|
| R-1 | Each calibration exposes residual, unknowns and inputs explicitly so the partials are evaluable at the solution | MUST | `valax.curves.BootstrapInstrument.residual`; `valax.curves.curve_residual_fn`; `valax.greeks.CalibrationLevel`; `valax.greeks.sabr_slice_level` / `cube_node_level`; `valax.surfaces.swaption_premium_residual` | ✅ | `CalibrationLevel` is the generic carrier. Layers 1a (curves) and 1b (smile fits, premium→vol) are wired; Layer 2 (model) is not yet |
| R-2 | Exact-calibration residuals in quote units, or PV units with `dR/dq` scaled accordingly | MUST | `curve_residual_fn` (documents the mixed units); `quote_deltas_from_cotangent` | ✅ | VALAX residuals are deliberately mixed — rate-unit for futures, PV-unit for swaps. The adjoint **differentiates** `dR/dq` rather than assuming `-I`, so scaling is handled per instrument with no special-casing |
| R-3 | `W`, `Λ`, `θ₀` documented; `θ₀` constant for sensitivities | MUST | `valax.calibration.loss.weighted_sse` (weights `W`) | 🟡 | Weights exist; there is no regularisation matrix `Λ` and no prior `θ₀` anywhere |
| R-4 | Convergence tolerances at least two orders below the smallest validation bump | MUST | `bootstrap_curve_graph` (Newton `rtol=atol=1e-10`); asserted in `tests/test_greeks/test_quote_adjoint.py::_build` | ✅ | Smallest validation bump is 0.1 bp = 1e-5; residuals run at 1e-16 |
| R-5 | State full Hessian vs Gauss-Newton in `dR/dθ`; monitor the neglected term | SHOULD | `valax.greeks.sabr_slice_level` (residual = `jax.grad` of the weighted SSE, so `dR/dθ` is the **full** Hessian); `SliceVolAdjoint.stationarity_norm` | ✅ | **Decided: full Hessian**, obtained for free because the residual *is* the SSE gradient and `jax.jacobian` of it is exact. Nothing is neglected, so there is no neglected term to monitor; what is surfaced instead is `stationarity_norm = max|R_v|`, the first-order-condition violation that would invalidate the theorem. `tests/test_greeks/test_vol_adjoint.py::TestLevelIsStationary::test_hessian_is_the_full_one_not_gauss_newton` shows the two differ on an imperfect fit |
| R-6 | Discrete choices (pillar dates, basket membership, PDE grid) frozen within a valuation and its sensitivity run | MUST | Pillars are `CurveSpec` fields; baskets and `PDEConfig` are caller-supplied | 🟡 | Frozen in practice because they are inputs, but nothing *enforces* that a bump does not re-select them |

## C — Curve construction (Layer 1a)

| ID | Requirement | MUST/SHOULD | Implementing symbol | Status | Gap |
|---|---|---|---|---|---|
| C-1 | Each residual matches its specified formula, including payment lag, leg day counts, frequencies | MUST | The eleven quote types in `valax.curves.instruments` ([inventory](#instrument-residual-inventory)) | 🟡 | Day counts, frequencies and **payment lag** (N-3) are all correct, and the OIS floating leg is now the per-period sum `Σ_i P_d(0,T_i^p)(P_d(0,T_i^s)/P_d(0,T_i^e) − 1)`, with the telescope kept only as a zero-lag fast path. **Averaged** money-market futures are implemented too — `MoneyMarketFuture.averaging` ∈ `{"compounded", "arithmetic"}`, gated by `…::TestAveragedFutures`. One § 4.2 variant remains unimplemented: the tenor-basis convention that compounds the short leg up to the long leg's payment frequency |
| C-2 | The discount curve is the collateral curve of the trade's CSA; projection curves never discount | MUST | `CurveSpec.currency`; `discount_id` / `forward_id` selectors on rates and bond instruments | 🟡 | The discount/projection *split* is enforced (N-4), but there is no CSA model: `CurveSpec.currency` is declared and never consumed, and a trade cannot name its collateral agreement |
| C-3 | Coupons fixing on or before the valuation date use the stored fixing; partly-accrued RFR coupons blend realised compounding with the curve forward | MUST | `valax.curves.FixingHistory`; `_floating_leg_pv` | 🟡 | Fixing lookup works and is all-or-nothing per period; a period *straddling* today is treated as fully projected |
| C-4 | The interpolation operator is differentiable, or piecewise differentiable with kinks only at nodes; limiter-based schemes flagged | MUST | `DiscountCurve` (log-linear in DF ⇒ piecewise-constant instantaneous forwards, kinks only at pillars) | 🟡 | Compliant as far as it goes, and no limiter-based scheme exists to flag. But it is the *only* mode, and `CurveSpec.interp` is validated without being consumed — see [deferred work](#deferred-with-rationale) |
| C-5 | One pillar per instrument, non-singular Jacobian, condition number monitored | MUST | `_validate_square`; `CurveBuildDiagnostics.jacobian_condition_number`; `CurveAdjoint.jacobian_condition_number` | ✅ | Squareness is a hard eager check; the condition number is reported at both build and adjoint time |
| C-6 | When blocks couple, the solve and the IFT treat them jointly | MUST | `bootstrap_curve_graph` (one Newton over the concatenated log-DFs of every curve); `curve_adjoint` | ✅ | Joint by construction — there is no sequential path to get wrong. Gated by `TestT3ParallelShiftIdentity` and `TestT15…::test_discount_quote_delta_carries_the_indirect_part` |
| C-7 | Futures, FRA and OIS payment-lag convexity either modelled or explicitly neglected, and the choice recorded | SHOULD | `valax.curves.convexity` (`ConvexityAdjFn`, `no_convexity_adj`, `constant_convexity_adj`); `OISSwapRate` docstring (payment-lag timing convexity) | 🟡 | A plug-in point exists for futures and the default is explicitly "none". With N-3 closed, the OIS payment-lag item is now *narrowly* scoped and recorded where it bites: the discount factor moves to `T^p`, but the compounded coupon is not re-expectation-corrected for no longer being a martingale under the `T^p`-forward measure. See [documented choices](#documented-choices). No model-based adjustment ships |
| C-8 | Jump spreads live in `p` and are applied identically in the solve and in every sensitivity run | MUST | `valax.curves.TurnInstrument` | 🟡 | A turn is representable as a window-forward residual, which the adjoint differentiates like any other quote. But it relies on the caller placing pillars tightly around the turn, and the spread is not held in a `p` container (Q-5) |

## V — Volatility objects (Layer 1b)

| ID | Requirement | MUST/SHOULD | Implementing symbol | Status | Gap |
|---|---|---|---|---|---|
| V-1 | Vol objects take forwards, annuities and discount factors from the same snapshot's curves through differentiable calls | MUST | `strip_caplet_vols`, `strip_optionlet_surface`; `calibrate_swaption_cube_from_curves` / `_from_graph`, `swaption_node_forwards_and_annuities`, `swaption_node_coordinates`; the `forward=` leg of the vol-source protocol in `valax.pricing.analytic.rates_smile` | ✅ | Both sides now comply and are gated. Caps: `tests/test_surfaces/test_caplet_stripping.py` differentiates a stripped vol w.r.t. projection-curve pillars. Swaptions: the curve-aware fit derives `(S_x, A_d)` and the grid axes from the node schedules via the *pricer's own* `swaption_forward_and_annuity`, so `d(fitted α)/d(DF) ≠ 0` — asserted against the legacy frozen-forward route, which is exactly `0.0`, in `tests/test_greeks/test_vol_adjoint.py::…::test_the_coupling_is_absent_from_a_frozen_forward_fit`. The legacy `calibrate_swaption_cube` survives as an explicit opt-out |
| V-2 | Premium-to-vol conversion uses the instrument's settlement convention and the discount annuity | MUST | `strip_caplet_vols(cap_premiums=...)`; `valax.surfaces.swaption_implied_normal_vol` / `swaption_premium_residual` / `swaption_forward_and_numeraire` | ✅ | Caps: DF-weighted, which is correct — a cap strip has no annuity. Swaptions: `R = N·𝒩(θ_c)·Bachelier(S_x(θ_c), K, σ_N, T) − q = 0` solved by scalar `optimistix.Newton` with implicit diff, where 𝒩 is the physical annuity `A_d` or, under cash settlement, `P_d(0,T^p)·A_cash(S_x)`. The two settlements imply different vols from the same premium and both round-trip to 1e-10. Cash-settled *convexity* is a documented deviation, below |
| V-3 | Optionlet stripping uses the same caplet schedule, index and first-caplet exclusion as the quoted caps | MUST | `strip_caplet_vols` reads schedules off the `Cap` pytrees it is given; `caplet_stripping._validate_strip` | ✅ | Schedule and index are inherited from the quoted instrument, so they cannot drift. First-caplet exclusion **is** enforced: `_validate_strip` counts optionlets fixing on or before the curve's reference date and raises, because an already-fixed optionlet has identically zero vega and the bootstrap would be solving for an unidentifiable parameter |
| V-4 | RFR optionlets use accrual-period variance, not variance to the start date | MUST | — | ❌ | Optionlet pricers are term-rate Black-76 / Bachelier; no in-arrears accrual-period variance treatment |
| V-5 | The smile dynamics implied by holding the vol quotes fixed is documented and matches risk policy | MUST | `valax.surfaces.dynamics` (`SMILE_DYNAMICS`, `QUOTE_FORMS`, `validate_smile_dynamics`, `resolve_smile_query`, `default_dynamics_for`); `SwaptionCube.dynamics`, `OptionletVolSurface.dynamics`; [Volatility Surfaces § Smile dynamics](../guide/vol-surfaces.md#smile-dynamics-what-moves-when-the-curve-moves) | ✅ | **Documented and selectable.** All three conventions are implemented and the `(dynamics, quote_form)` pair is validated at construction, because the dynamics is a consequence of what the quote holds fixed. Default `"sticky_strike"` preserves every pre-existing number; the forward-dependent conventions raise rather than silently fall back when queried without a live forward. `tests/test_surfaces/test_smile_dynamics.py` asserts `∂σ/∂θ_c` is exactly `0.0` under the default and non-zero under `"sabr_backbone"` — so the convention is now a fact of record rather than a side effect |
| V-6 | SABR beta, shifts, weights and wing settings in `p`, frozen within a sensitivity run | MUST | `fixed_beta`, `shift` arguments on the cube and surface fits | 🟡 | Frozen by virtue of being arguments, but not held in a `p` container and carrying no adjoints (Q-5, A-8) |
| V-7 | Butterfly and calendar arbitrage checked after each fit | SHOULD | `valax.ButterflyArbError`, `CalendarArbError`, `NonConvexSmileError` (reserved types) | ❌ | The exception types are declared and never raised; no post-fit check runs |
| V-8 | If an arbitrage-free SABR PDE is used, its grid is frozen | SHOULD | — | n-a | Only the Hagan expansion ships; there is no SABR PDE whose grid could move |

## M — Model calibration (Layer 2)

| ID | Requirement | MUST/SHOULD | Implementing symbol | Status | Gap |
|---|---|---|---|---|---|
| M-1 | The model short rate is the collateral rate; the model reproduces the initial discount curve to solver precision | MUST | `hw_market_df`, `g2pp_market_df`, `cheyette_market_df` (initial curve entering the reconstitution formula by construction) | ✅ | The initial curve is fitted exactly by construction, not by calibration |
| M-2 | Every projected fixing uses a basis from the same snapshot's curves, with the coupon's own index period | MUST | `valax.pricing.multicurve` (`tenor_basis`, `swap_leg_basis`, `apply_swap_basis`, `with_discount_curve`) | 🟡 | Deterministic tenor basis ships for Hull-White and G2++ (MC and PDE). `calibrate_cheyette` / `calibrate_cheyette_2f` **raise** on a distinct projection curve |
| M-3 | All six paths by which curves enter Layer 2 are differentiable functions in code, none a cached constant | MUST | Paths 1, 2 and 6 via `multicurve` + `hw_market_df`; paths 3 and 4 via `calibrate_swaption_cube_from_curves` + `SwaptionCube.dynamics` | 🟡 | Paths 1–4 and 6 are now differentiable *end to end*: the cube's forwards and annuities are curve-derived (V-1) and the query-time forward reaches the smile through the vol-source protocol, so `dR_v/dθ_c` exists and is FD-validated. Path 5 (basket strikes) still runs off caller-supplied strikes, and the Layer 2 *model* calibrators have not yet been given a `CalibrationLevel`, so `dR_m/dθ_c` remains absent |
| M-4 | Basket targets use the cube at the basket strike with the basket's actual underlying | MUST | `swaption_prices_from_vols(..., forward_curve=...)`; `swaption_node_forwards_and_annuities` | 🟡 | The curve-derived forwards and annuities a basket target needs now exist and are differentiable, but `swaption_prices_from_vols` has not been rewired to consume them — it still takes precomputed forwards |
| M-5 | Fixed model inputs (mean reversion, correlation, factor count) in `p` and reported as their own sensitivities | MUST | — | ❌ | Mean reversion and correlation are ordinary model leaves; no separate `p` reporting |
| M-6 | The PDE grid is frozen across calibration iterations, bumps and the adjoint pass | MUST | `PDEConfig` (caller-supplied, static) | 🟡 | Frozen because it is an input; not enforced, and no test scans for grid-induced steps |
| M-7 | Payoff kinks smoothed and the first time steps damped, so the calibration Jacobian and gamma are not noisy | MUST | `valax.pricing.mc.payoffs` (sigmoid smoothing); `pde.schemes.theta_scheme` / `schemes2d` (`rannacher_steps`), forced to ≥ 2 by `pde.hull_white`, `pde.g2pp`, `pde.cheyette` (`_MIN_RANNACHER = 2`) | ✅ | Resolved. Smoothing exists on the Monte Carlo payoff path, and every short-rate PDE engine applies **Rannacher start-up unconditionally**: each takes `rannacher_steps=max(config.rannacher_steps, _MIN_RANNACHER)`, so a caller cannot switch the damping off even by passing `PDEConfig(rannacher_steps=0)`. No test yet scans a PDE-priced calibration Jacobian for grid-induced steps — that is the T-8 smoothness gate |
| M-8 | After calibration, reprice the basket with the production engine and report residuals | SHOULD | — | ❌ | Calibration round-trip tests exist, but no production-engine repricing report |

## A — Adjoint chain

| ID | Requirement | MUST/SHOULD | Implementing symbol | Status | Gap |
|---|---|---|---|---|---|
| A-1 | No tape records optimiser iterations; each calibration is differentiated only through its residual at the converged solution | MUST | `bootstrap_curve_graph` (`optimistix.root_find`, implicit adjoint); `level_adjoint` | 🟡 | **Layer 1a complies and is gated** by T-2. The model calibrators inherit `optimistix`'s implicit default rather than passing `adjoint=` explicitly, and no test differentiates through any `calibrate_*` except caplet stripping. `calibrate_slv_leverage` is a hand-rolled loop and *would* be taped |
| A-2 | The residual in the adjoint is the same code path as the one solved | MUST | `bootstrap_graph._graph_residual` — the single implementation, called by both `bootstrap_curve_graph` and `curve_residual_fn` | ✅ | Structurally guaranteed: there is one residual function and both consumers call it |
| A-3 | Every term of the chain present, including `dR_m/dθ_c`, `dR_v/dθ_c`, `dR_v/dq_v` | MUST | `curve_quote_deltas` (λ_c); `valax.greeks.vol_adjoint` — `sabr_slice_level`, `vol_quote_vegas`, `SliceVolAdjoint.vol_quote_vegas` (`dR_v/dq_v`) and `.forward_bar` (`dR_v/dθ_c`) | 🟡 | λ_c and λ_v are complete. Both `dR_v/dq_v` and `dR_v/dθ_c` are validated against full-rebuild central finite differences at two bump sizes, under both quote forms; the `optimistix` `ImplicitAdjoint` comparison is retained as a consistency check, not an independent one. λ_m does not exist, so `dR_m/dθ_c` is still absent |
| A-4 | Signs follow the adjoint form | MUST | `quote_deltas_from_cotangent` (returns `-(dR/dq)^T λ`) | ✅ | Validated by sign and magnitude against full-rebuild FD in both directions |
| A-5 | For least-squares levels, `dR/dθ` and `dR/dq` are the full derivatives of the first-order condition including `Λ`, or Gauss-Newton is documented and validated | MUST | — | ❌ | No least-squares level is differentiated yet, and no `Λ` exists (R-3) |
| A-6 | The condition number of each `dR/dθ` is monitored | MUST | `CurveBuildDiagnostics.jacobian_condition_number`; `CurveAdjoint.jacobian_condition_number` | ✅ | Reported for Layer 1a at build and adjoint time; gated by `TestFactorisationReuse::test_condition_number_is_reported` |
| A-7 | Layer 1 Jacobian factorisations computed once per snapshot and reused across trades | SHOULD | `CurveAdjoint.factorisation` (a `lineax.LU` state built once in `curve_adjoint`) | ✅ | Each additional trade is one back-substitution plus one reverse sweep; gated by `TestFactorisationReuse::test_reused_handle_matches_fresh_handles` |
| A-8 | Fixed inputs `p` have adjoints accumulated and reported, not silently dropped | MUST | — | ❌ | No `p` container exists to accumulate into |

## T — Verification tests

Run the whole suite with `pytest -m audit`. Every gate names the requirement it
verifies and states its own tolerance in the test docstring.

| ID | Test | Status | Where | Tolerance committed to |
|---|---|---|---|---|
| T-1 | Reprice every calibration instrument from the calibrated objects | 🟡 | `CurveBuildDiagnostics.fitted_quotes` / `.quoted_values` (exact quote-unit fit error, because every residual is linear in its primary quote scalar) | Curves: `max_abs_residual < 1e-10`. **Vol objects: no gate** |
| T-2 | AAD quote sensitivities vs full-rebuild central finite differences, every quote, ≥2 bump sizes | ✅ | `tests/test_greeks/test_quote_adjoint.py::TestT2AgainstFiniteDifferences` | Bumps 0.1 bp and 1 bp; `rtol = 1e-5` where `|FD| > 1`, `atol = 1e-4` on FD-noise-dominated entries. Observed agreement is 1e-9 to 1e-11 relative |
| T-3 | Parallel-shift identity: summed quote deltas vs an all-quotes-bumped rebuild | ✅ | `…::TestT3ParallelShiftIdentity` (single- and dual-curve) | `rtol = 1e-6`; observed 1e-11 |
| T-4 | Single-curve limit (unit basis) matches an independent single-curve pricer | 🟡 | `tests/test_curves/test_bootstrap_graph.py::TestSingleCurveDegeneracy`, `tests/test_calibration/test_dual_curve_calibration.py::TestSingleCurveNonRegression` | Covered at the curve and calibration layers; not yet an `audit`-marked end-to-end gate |
| T-5 | OIS telescoping with zero payment lag | ✅ | `tests/test_curves/test_instrument_residuals.py::TestPaymentLagTelescoping` | `atol = 1e-14` on a unit-notional floating-leg PV — a round-off budget, not a modelling tolerance, since the per-period sum and the closed telescope differ only by summation order. The gate drives the **per-period** branch (explicit payment dates equal to the accrual ends) against a two-read reference `P_d(0,T_0) − P_d(0,T_n)`, so it is no longer vacuous; `test_telescope_does_not_survive_a_lag` pins the converse |
| T-6 | Put-call parity (caplet−floorlet = FRA; payer−receiver = annuity × (forward−strike)) | ❌ | — | — |
| T-7 | Bachelier ATM straddle identity | ❌ | — | — |
| T-8 | Smoothness scan: price and delta over a fine grid of bumps show no steps | ❌ | — | — |
| T-9 | Basket repriced by the production engine after calibration | ❌ | — | — |
| T-10 | Day count and schedule spot checks on known dates | 🟡 | `tests/test_dates/` | Exact match asserted for day counts and schedule generation; not `audit`-marked, and there is no reference-data fixture to check against (N-6) |
| T-11 | Fixing handling across a fixing date; sensitivities to stored fixings exactly zero | ❌ | — | Fixings are non-differentiable by construction (integer-keyed lookup), but this is not asserted |
| T-12 | Smile-dynamics check with vol quotes held as defined | ❌ | — | The convention to check against is now documented (V-5), so the gate is writable. Today it would be near-tautological — only one convention exists — so it lands with the `dynamics` field |
| T-13 | Condition numbers of `dR/dθ` reported at every level | 🟡 | `CurveBuildDiagnostics`, `CurveAdjoint`; `…::TestFactorisationReuse::test_condition_number_is_reported` | Alert threshold **1e10** for Layer 1a. Levels 1b and 2 have no Jacobian to report |
| T-14 | Fixed-input sensitivities vs finite differences | ❌ | — | Blocked on A-8 / Q-5 |
| T-15 | Discount-quote delta with projection *quotes* fixed vs projection *nodes* fixed; the former is reported | ✅ | `…::TestT15QuotesFixedVsNodesFixed` | Asserts the two differ materially, that the reported figure is the quotes-fixed one (FD-validated), and that the indirect dependence of projection nodes on discount quotes is non-zero |

---

## Specification § 8.3 deliverables

### Identifier map

| Specification symbol | VALAX identifier |
|---|---|
| `t = 0`, valuation date | `CurveSpec` / `DiscountCurve.reference_date`, `MarketState.reference_date` (integer day ordinal) |
| `P_d(0, T)` — collateral discount factor | `graph[discount_id](T)`, i.e. `DiscountCurve.__call__` |
| `P_x(0, T)` — projection pseudo-discount factor | `graph[forward_id](T)`; never used to discount |
| `F_x`, `L_x` — forward / fixing | `valax.curves.forward_rate`; realised values via `FixingHistory.lookup_many` |
| `A_d` — discount annuity | `valax.pricing.analytic.swaptions.annuity`; per swaption via `swaption_forward_and_annuity`; per cube node via `swaption_node_forwards_and_annuities` |
| `𝒩` — premium numeraire (settlement-dependent) | `valax.surfaces.swaption_forward_and_numeraire`; cash branch via `valax.pricing.analytic.cash_settled_annuity` |
| `S_x` — forward par swap rate | `valax.pricing.analytic.swaptions.swap_rate`; `swaption_forward_and_annuity` |
| `q` — market quote vector | `valax.curves.curve_quote_vector(instruments)` |
| `θ_c` — curve unknowns | `valax.curves.log_dfs_from_graph(graph, specs)` (concatenated log-DFs, reference pillar excluded) |
| `θ_v` — vol-object unknowns | `CapletVolCurve` vols; `SwaptionCube` SABR parameter grids — packed flat by `valax.greeks.sabr_theta` in `sabr_theta_names` order |
| `θ_m` — model unknowns | Model pytree leaves (`HullWhiteModel.sigma`, …) |
| `p` — fixed non-solved inputs | **No container** (Q-5) |
| `R` — residual | `BootstrapInstrument.residual`; `curve_residual_fn`; `CalibrationLevel.residual_fn` |
| `∂R/∂θ` | `CurveAdjoint.residual_jacobian` |
| `λ` — adjoint multiplier | `LevelAdjoint.lam`; `SliceVolAdjoint.lam` (λ_v) |
| `q̄` — quote sensitivity | return of `curve_quote_deltas` / `quote_deltas_from_cotangent`; `SliceVolAdjoint.vol_quote_vegas` for vol quotes |

### Instrument-residual inventory

Every Layer-1a quote type, its primary quote scalar (the column it owns in
`q`), and the curves it constrains. All eleven implement
`BootstrapInstrument.residual`; the primary-quote dispatch lives in
`bootstrap_graph._QUOTE_FIELD`.

| Quote type | Primary quote | Constrains |
|---|---|---|
| `DepositRate` | `.rate` | one curve, short end |
| `FRA` | `.rate` | projection curve, short end |
| `SwapRate` | `.rate` | one curve (single-curve compounded float leg) |
| `OISSwapRate` | `.rate` | discount curve |
| `IborSwapRate` | `.rate` | discount + projection (coupled) |
| `MoneyMarketFuture` | `.futures_rate` | projection curve, short end; `averaging` selects the compounded (product of daily ratios) or arithmetic-average (sum of them) settlement functional |
| `TenorBasisSwap` | `.spread` | two projection curves + discount |
| `CrossCurrencyBasisSwap` | `.spread` | two discount curves + projection; `variant` selects constant-notional or mark-to-market reset |
| `FXForward` | `.quoted_forward` | two short-end curves (covered interest parity) |
| `FXSwap` | `.far_rate` | foreign-collateral discount curve, short end |
| `TurnInstrument` | `.jump_size` | window forward across a turn date |

Layer 1b has three residuals, all differentiated at the solution rather than
through the solver:

| Residual | Unknown `θ_v` | Inputs | Implementation |
|---|---|---|---|
| Cap premium − quoted premium (one scalar root-find per quoted maturity) | bucket vol `σ_i` | cap quotes, discount + projection curves | `strip_caplet_vols` (`optimistix.Newton`, implicit diff) |
| Stationarity of the weighted SSE of a SABR smile | `(α, ρ, ν)` or `(α, β, ρ, ν)` | quoted vols, node forward `F_0` | `calibrate_swaption_cube_from_curves` solves; `valax.greeks.sabr_slice_level` + `vol_quote_vegas` differentiate |
| `N·𝒩(θ_c)·Bachelier(S_x, K, σ_N, T) − q` | `σ_N` | premium quote, discount + projection curves, settlement convention | `valax.surfaces.swaption_premium_residual` / `swaption_implied_normal_vol` |

Layer 2: `calibrate_hull_white`, `calibrate_g2pp`, `calibrate_cheyette(_2f)` —
not yet expressed as `CalibrationLevel`s.

### Stated tolerances

| Gate | Tolerance VALAX commits to |
|---|---|
| T-1 (curves) | `max_abs_residual < 1e-10`; per-instrument quote-unit fit error via `fitted_quotes - quoted_values` |
| T-1 (vol objects) | Swaption cube: every fitted quote reprices from the calibrated object at its own node coordinate to `atol < 1e-8` in vol units (`tests/test_surfaces/test_swaption_cube.py::TestCurveAwareCalibration`). Caps: quoted caps reprice to ~1e-10 relative by construction |
| T-2 (curve quotes) | Bumps 0.1 bp and 1 bp; `rtol = 1e-5` where `|FD| > 1`, else `atol = 1e-4` |
| T-2 (vol quotes) | Vol-quote vegas: bumps 1e-4 and 1e-3 in vol units. Forward coupling: bumps 0.1 bp and 1 bp, both quote forms. Both at `rtol = 1e-4` relative to the largest component of the full-rebuild finite-difference vector. The `optimistix` `ImplicitAdjoint` comparison is also run at `rtol = 1e-9`, but is recorded as a **consistency** check rather than an independent one — it shares `sabr_implied_vol` with the adjoint under test, and that shared primitive is exactly what once let both agree while both were wrong (see [Numerical Pitfalls entry 5](numerical-pitfalls.md#5-a-value-correct-guard-with-no-derivative-jnpwhere-on-a-removable-singularity)) |
| T-9 | **Not yet gated** |
| T-13 | Condition-number alert threshold **1e10** (Layer 1a). Layer 1b surfaces `SliceVolAdjoint.stationarity_norm` rather than a condition number; smile-fit Hessians are routinely ~1e6-conditioned by the known α/ρ/ν identifiability, so a threshold alert there would fire on healthy fits |

### Documented choices

The four choices the specification requires to be stated explicitly:

| Choice | VALAX position |
|---|---|
| **Smile dynamics** (V-5) | **Default sticky absolute strike, now a declared and selectable choice.** `dynamics` ∈ `{"sticky_strike", "sticky_moneyness", "sabr_backbone"}` is a static field on both rates vol objects, validated against `quote_form` because the dynamics is a consequence of what the quote holds fixed. The default preserves every pre-existing number; `"sabr_backbone"` is never selected implicitly, because it reproduces a node's market vol only when `F = F_0` there — which holds only on the curve-aware calibration route. The three conventions, the admissibility table, the off-node mixed-order caveat and the backbone prerequisite are stated once in [Volatility Surfaces § Smile dynamics](../guide/vol-surfaces.md#smile-dynamics-what-moves-when-the-curve-moves) |
| **Neglected convexities** (C-7) | Futures convexity is **explicitly neglected by default** (`no_convexity_adj`) with a plug-in point (`ConvexityAdjFn`) for a desk-supplied or constant adjustment; no model-based adjustment ships. FRA settlement convexity and OIS payment-lag **timing** convexity are **neglected without being modelled**. The latter is now a genuinely second-order omission rather than a consequence of ignoring the lag outright: since N-3 closed, the lagged coupon is discounted at its true `T^p`, and what remains neglected is only the drift correction from the compounded rate no longer being a martingale under the `T^p`-forward measure. **Cash-settled swaption convexity** is likewise neglected: `swaption_forward_and_numeraire(settlement="cash")` evaluates the street-standard flat-yield annuity *at the forward* rather than inside the expectation, so the smile-integral correction (Mercurio 2008) is not applied. Second order in vol, and stated in the module's own docstring |
| **Gauss-Newton vs full Hessian** (R-5) | **Full Hessian.** The smile level's residual *is* the gradient of the weighted SSE (`valax.greeks.sabr_slice_level`), so `dR_v/dθ_v` obtained by `jax.jacobian` is exactly `J^T W J + Σ_k r_k ∇²r_k` — the Gauss-Newton term plus the curvature term, with nothing dropped. Chosen over Gauss-Newton because the two coincide to the size of the residual on a tight fit, so the approximation only ever matters when the fit is *loose* — precisely when a sensitivity is most likely to be trusted wrongly. There is no neglected term to monitor; `SliceVolAdjoint.stationarity_norm` surfaces the first-order-condition violation instead, which is the assumption that *can* fail |
| **Multiplicative vs additive basis** (§ 6.2) | **Additive**: `valax.pricing.multicurve.tenor_basis` computes a deterministic spread `s = F_x − F_d` from today's curves and adds it to the model-implied rate |

---

## Deferred, with rationale

These are known ❌ / 🟡 rows that are *not* scheduled next, so that the gap list
stays honest rather than aspirational.

| Item | Requirements | Why deferred |
|---|---|---|
| Genuine step / instantaneous-forward interpolation, calendar-aware turns | C-4, C-8 | A curve-representation project in its own right (tracked as the interpolation workstream on the [roadmap](../roadmap.md)). The shipped log-linear-DF mode is already piecewise differentiable with kinks only at pillars, so it does not *violate* C-4 |
| Full CSA / collateralised discounting | C-2 | Requires a collateral-agreement model on the instrument, not just a curve identifier. The discount/projection split that C-2 mainly guards is already enforced |
| RFR in-arrears optionlet variance | V-4 | Needs a compounded-in-arrears optionlet pricer; the current optionlets are term-rate by construction |
| Dual-curve Cheyette calibration | M-2 | `calibrate_cheyette` raises rather than silently single-curving, which is the safe failure mode. Hull-White and G2++ cover the dual-curve route |
| Business-day calendars and versioned reference data | N-6, N-5, T-10 | Large, orthogonal, and already tracked on the roadmap |
