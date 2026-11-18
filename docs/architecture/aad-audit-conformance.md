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

<!-- coverage-map -->

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
| **1a — Curves** | Curve node values | ✅ Joint Newton bootstrap, implicit-diff quote Jacobian, **full adjoint chain to quote space** (`λ_c`) |
| **1b — Volatility objects** | Optionlet vols, SABR cube parameters | ✅ Cap/optionlet stripping and the swaption cube are both curve-coupled, differentiable **chain levels**: `swaption_cube_node` recomputes each node's forward and annuity from the upstream curves, `caplet_strip_node` → `optionlet_sabr_node` strip and smooth on them, so `dθ_v/dθ_c` is non-zero and `chain_adjoint` carries curve → vol → price in one sweep (`tests/test_calibration/test_rates_vol_nodes.py`) |
| **2 — Model** | `σ(t)`, skew parameters | ✅ Per underlying (`price_fit_node` + `calibrate_model`, Heston) **and per trade** (`swaption_basket_node`: Hull-White fitted to a Bermudan's co-terminal basket off the calibrated cube). `chain_adjoint` folds the trade's `λ_m` through the cube and curve levels onto the quotes; the basket target shifts report `λ_m` as the vega profile across the basket. Gated by `tests/test_calibration/test_per_trade_basket.py` (adjoint = autodiff to 1e-6; = full-recalibration FD to 5e-4 on a curve quote, a cube vol quote, the held-fixed mean reversion and a target shift) |
| **Fixed inputs `p`** | Not solved for | ✅ `FixedInputs` / `fixed_input_adjoints` report `p̄` separately from `q̄`, with snapshot provenance to label both |

The `λ_m` level now exists per trade, so every term of the adjoint chain is
present (**A-3**). Of the six paths by which curves enter Layer 2, five are
differentiable end to end; path 5 (basket strikes *defined off the curves*,
e.g. ATM baskets) has no implementation — `coterminal_swaptions` uses the
trade's own strike. See **M-3** below.

## Verification scoreboard

| Gate | State |
|---|---|
| T-suite (T-1 … T-15) | **15 / 15** implemented and `audit`-marked |
| `pytest -m audit` | 331 passed |
| Full `pytest` | 4601 passed, 1 skipped, 6 xfailed |
| Requirement rows (N/Q/R/C/V/M/A) | 50 total — **31 ✅ · 16 🟡 · 2 ❌ · 1 n-a** |

Every 🟡 states its gap in its own row. The two remaining ❌ are whole
features rather than partial ones, which is why they are not 🟡:

- **V-4** — RFR optionlets priced on accrual-period variance instead of
  variance to the period start.
- **V-7** — butterfly and calendar arbitrage checked after each vol fit.

**N-6** left the ❌ set with Tier 1.3: `valax.dates.calendars` supplies
business-day calendars for six markets from a version-stamped, cross-checked
artifact. It is **🟡 rather than ✅ on purpose.** The requirement has two
clauses — calendars *and* conventions from versioned reference data — and
only the first is closed. Conventions are still per-instrument static
strings, with no index definition to resolve them from, and no curve-build
code consults a calendar. Marking the row ✅ would claim the registry half
that N-5 is still waiting on.

The 🟡 set is dominated by the genuinely deferred workstreams — curve
interpolation modes (C-4, C-8), full CSA discounting (C-2) and curve-defined
basket strikes (M-3, path 5) — plus clauses that are *unimplementable*
rather than unimplemented, such as V-6's wing settings, for which VALAX has no
parameter to freeze.

---

## N — Notation and conventions

| ID | Requirement | MUST/SHOULD | Implementing symbol | Status | Gap |
|---|---|---|---|---|---|
| N-1 | Accrual fractions in the leg's own day count; one stack-wide model day count; never mixed | MUST | `valax.dates.year_fraction`; per-leg static `day_count` fields on every quote type | ✅ | — |
| N-2 | Every rate annualised and paired with its own day count | MUST | Per-leg `fixed_day_count` / `float_day_count` on `IborSwapRate`, `TenorBasisSwap`, `CrossCurrencyBasisSwap` | ✅ | — |
| N-3 | Discount cash flows at the payment date, not accrual end, where a payment lag exists | MUST | `payment_dates` / `payment_lag` on `SwapRate`, `OISSwapRate`, `IborSwapRate` (per leg), `TenorBasisSwap` (per leg), `CrossCurrencyBasisSwap` (per leg); `_floating_leg_pv(accrual_end_dates=…, payment_dates=…)`; `_compounded_float_leg_pv` | ✅ | Accrual and payment are now separate roles: projection and `τ` come from the accrual schedule, discounting from the payment schedule. Either an explicit business-day-adjusted payment array or a uniform calendar-day `payment_lag`; supplying both raises. Default is `T^p = T^e`, which preserves every pre-existing number bit for bit. Gated by `tests/test_curves/test_instrument_residuals.py::TestPaymentLagTelescoping` (T-5) and `…::TestPaymentLagPerLeg` |
| N-4 | Projection (pseudo-discount) factors never discount | MUST | `_floating_leg_pv` discounts on `curves_touched[0]` and projects on `curves_touched[1]` | ✅ | — |
| N-5 | The index period is generated from the index definition, not assumed equal to the coupon accrual period; stubs explicit | MUST | `IborSwapRate.fixing_dates` (explicit per-coupon fixing dates); `valax.instruments.FloatingLeg.index_id` (per-leg index descriptor); `valax.dates.generate_schedule` (stubs, EOM rolls, business-day adjustment) | 🟡 | Fixing dates are supplied by the caller rather than derived from a versioned index definition; there is no index registry. Tier 1.3 supplied the missing *mechanism* (a schedule can be rolled on a named market calendar) and Tier 1.4 made the index an explicit per-leg field, so a trade now *names* the index each leg fixes against and `MultiLegSwap.indices_touched` enumerates them. But the row's first clause is still open, and the leg engine does not close it: the projection window **is** the accrual period, by construction. That is precisely why LIBOR-in-arrears is absent rather than approximated — decoupling the index period from the accrual period is what this row asks for, and it needs the registry to say what that period is |
| N-6 | Schedules, calendars and conventions from versioned reference data | SHOULD | `valax.dates.calendars` (`HolidayCalendar`, six market factories, `adjust`, `union`/`intersection`, `bus_252`); `valax.dates._holiday_data` (generated artifact carrying `CALENDAR_DATA_VERSION` and per-market `PROVENANCE`) | 🟡 | Calendars now exist and come from version-stamped reference data, cross-checked against a second independent source across each table's whole range. The row stays 🟡 because its second clause is unchanged: *conventions* are still per-instrument static fields, not reference data — nothing resolves `day_count`/`business_day_convention` from an index definition, and nothing in `valax/curves/` consults a calendar (see N-5) |

## Q — Market instruments and quotes

| ID | Requirement | MUST/SHOULD | Implementing symbol | Status | Gap |
|---|---|---|---|---|---|
| Q-1 | Each quote stored with definition, type, units, source and timestamp under one snapshot ID | MUST | `valax.market.SnapshotProvenance`, `QuoteProvenance` | ✅ | `from_instruments` derives each quote's type and units from the bootstrap instrument in solver order, so the record cannot drift from the quote vector; `label(q_bar)` turns an anonymous sensitivity vector into a named report and raises on a length mismatch rather than zipping short |
| Q-2 | Each quote enters exactly one residual; no quote twice, no residual without a quote | MUST | `bootstrap_graph._validate_square`, `_validate_curves_touched` | ✅ | Enforced eagerly: the build raises unless `n_instruments == n_pillars` |
| Q-3 | Strikes keep their quoted form (an ATM offset stays an offset; the absolute strike is recomputed from the curve) | MUST | `SwaptionCube.quote_form` / `OptionletVolSurface.quote_form`; `calibrate_swaption_cube_from_curves(quote_form="atm_offset")` | ✅ | The quote form is a declared static field, and the curve-aware fit converts offsets to absolute strikes per node using that node's **curve-derived** `F_0`. Under `"atm_offset"` the adjoint's strike grid is rebuilt from the forward too, so `dR_v/dF_0` carries the grid-shift term — gated in `tests/test_greeks/test_vol_adjoint.py` |
| Q-4 | Vol quotes carry their convention (normal, or shifted lognormal with the shift) | MUST | `SwaptionCube.is_normal` / `.shift` / `.quote_form`; `OptionletVolSurface`; `valax.surfaces.shifted` | 🟡 | Strike form is now carried and validated (Q-3). The *vol* convention (`is_normal`, `shift`) still lives on the fitted object rather than on the quote, so a normal/lognormal mismatch is caught only by non-convergence |
| Q-5 | Quasi-market inputs live in `p`, versioned like quotes, with sensitivities reported separately | MUST | `valax.greeks.FixedInputs`, `fixed_input_adjoints`; `valax.market.SnapshotProvenance` | ✅ | `p̄` is reported as its own named, united table, never folded into `q̄`. Provenance is a sibling of `MarketState`, not a field: all-static metadata would still enter JIT cache keys, and provenance changes every snapshot. **Gap:** the futures convexity adjustment is a closure over a constant (`constant_convexity_adj`), not a pytree leaf, so it has no lens and no `p̄` today — the carrier supports it the moment it becomes a leaf |
| Q-6 | Stale or overridden quotes flagged and excluded or down-weighted by an explicit rule | SHOULD | `QuoteProvenance.stale`, `SnapshotProvenance.stale_quote_ids` | 🟡 | Flagging and reporting are in place. *Excluding or down-weighting* is deliberately not: that is a calibration policy, and it stays with the caller who assembles the instrument list rather than being hidden in a metadata container |

## R — Residual framework

| ID | Requirement | MUST/SHOULD | Implementing symbol | Status | Gap |
|---|---|---|---|---|---|
| R-1 | Each calibration exposes residual, unknowns and inputs explicitly so the partials are evaluable at the solution | MUST | `valax.calibration.CalibrationResult` (`theta`, `inputs`, `residual_fn`, `.level`) returned by every calibrator; `valax.curves.BootstrapInstrument.residual`; `valax.curves.curve_residual_fn`; `valax.greeks.sabr_slice_level` / `cube_node_level`; `valax.surfaces.swaption_premium_residual` | ✅ | Every calibrator — curves, smiles, strips, cubes and the short-rate model baskets (Layer 2) — returns a `CalibrationResult` whose `residual_fn` is the system the solver drove to zero, built by `solve_level` / `solve_batched_level`. Explicit and fixed-point levels (`theta=None`) carry no residual by construction and are differentiated by plain AAD |
| R-2 | Exact-calibration residuals in quote units, or PV units with `dR/dq` scaled accordingly | MUST | `curve_residual_fn` (documents the mixed units); `quote_deltas_from_cotangent` | ✅ | VALAX residuals are deliberately mixed — rate-unit for futures, PV-unit for swaps. The adjoint **differentiates** `dR/dq` rather than assuming `-I`, so scaling is handled per instrument with no special-casing |
| R-3 | `W`, `Λ`, `θ₀` documented; `θ₀` constant for sensitivities | MUST | Weights live in each level's own residual — e.g. `valax.calibration.sabr_slice_residual_fn`, `basket_fit_level(weights=...)`, `stack_slices` (zero weight on padding) | 🟡 | Weights `W` exist and are part of the solved (and differentiated) residual, so the two cannot disagree; `basket_fit_level` holds them as a fixed input. There is no regularisation matrix `Λ` and no prior `θ₀` anywhere |
| R-4 | Convergence tolerances at least two orders below the smallest validation bump | MUST | `bootstrap_curve_graph` (Newton `rtol=atol=1e-10`); asserted in `tests/test_greeks/test_quote_adjoint.py::_build` | ✅ | Smallest validation bump is 0.1 bp = 1e-5; residuals run at 1e-16 |
| R-5 | State full Hessian vs Gauss-Newton in `dR/dθ`; monitor the neglected term | SHOULD | `valax.greeks.sabr_slice_level` (residual = `jax.grad` of the weighted SSE, so `dR/dθ` is the **full** Hessian); `SliceVolAdjoint.stationarity_norm` | ✅ | **Decided: full Hessian**, obtained for free because the residual *is* the SSE gradient and `jax.jacobian` of it is exact. Nothing is neglected, so there is no neglected term to monitor; what is surfaced instead is `stationarity_norm = max|R_v|`, the first-order-condition violation that would invalidate the theorem. `tests/test_greeks/test_vol_adjoint.py::TestLevelIsStationary::test_hessian_is_the_full_one_not_gauss_newton` shows the two differ on an imperfect fit |
| R-6 | Discrete choices (pillar dates, basket membership, PDE grid) frozen within a valuation and its sensitivity run | MUST | Pillars are static `CurveSpec` fields; baskets and `PDEConfig` are caller-supplied | 🟡 | Frozen in practice because they are inputs, and T-8 now exercises the condition directly: every point of its bump scan rebuilds the graph, the adjoint and the factorisation from the quotes alone, so nothing is carried between bumps and a re-selection would show as a step. Still 🟡 because nothing *enforces* it — a caller who re-derived pillar dates or basket membership from bumped quotes would violate R-6 silently |

## C — Curve construction (Layer 1a)

| ID | Requirement | MUST/SHOULD | Implementing symbol | Status | Gap |
|---|---|---|---|---|---|
| C-1 | Each residual matches its specified formula, including payment lag, leg day counts, frequencies | MUST | The eleven quote types in `valax.curves.instruments` ([inventory](#instrument-residual-inventory)) | 🟡 | Day counts, frequencies and **payment lag** (N-3) are all correct, and the OIS floating leg is now the per-period sum `Σ_i P_d(0,T_i^p)(P_d(0,T_i^s)/P_d(0,T_i^e) − 1)`, with the telescope kept only as a zero-lag fast path. **Averaged** money-market futures are implemented too — `MoneyMarketFuture.averaging` ∈ `{"compounded", "arithmetic"}`, gated by `…::TestAveragedFutures`. One § 4.2 variant remains unimplemented: the tenor-basis convention that compounds the short leg up to the long leg's payment frequency |
| C-2 | The discount curve is the collateral curve of the trade's CSA; projection curves never discount | MUST | `CurveSpec.currency`; `discount_id` / `forward_id` selectors on rates and bond instruments | 🟡 | The discount/projection *split* is enforced (N-4), but there is no CSA model: `CurveSpec.currency` is declared and never consumed, and a trade cannot name its collateral agreement |
| C-3 | Coupons fixing on or before the valuation date use the stored fixing; partly-accrued RFR coupons blend realised compounding with the curve forward | MUST | `valax.curves.FixingHistory`; `FixingSeries.compounded_factor`; `_floating_leg_pv(ref_date=…, accrued_compounding=…)`; `accrued_compounding` on `IborSwapRate` and per leg on `TenorBasisSwap` / `CrossCurrencyBasisSwap` | ✅ | Three sources in strict precedence: recorded fixing → partly-accrued blend `1 + τ_j F_j = A · P_x(0,t_0)/P_x(0,T_j^e)` → projection. The straddling period is located by the half-open test `T^s ≤ t_0 < T^e`, which makes it unique. Both degeneracies hold *exactly* (`A = 1` ⇒ pure projection; `A = P_x(T^s)/P_x(t_0)` ⇒ no-op), gated by `tests/test_curves/test_instrument_residuals.py::TestPartlyAccruedFixings`. A fully-*ended* period is deliberately out of scope and must come from a recorded fixing — see the note in `_floating_leg_pv` |
| C-4 | The interpolation operator is differentiable, or piecewise differentiable with kinks only at nodes; limiter-based schemes flagged | MUST | `DiscountCurve` (log-linear in DF ⇒ piecewise-constant instantaneous forwards, kinks only at pillars); `DISCOUNT_CURVE_INTERP`; `graph._SUPPORTED_INTERP` | 🟡 | Compliant as far as it goes, and no limiter-based scheme exists to flag. It remains the *only* mode — see [deferred work](#deferred-with-rationale). `CurveSpec.interp` is declared and checked but not forwarded, which is now **guarded rather than latent**: `_SUPPORTED_INTERP` is *derived from* `DISCOUNT_CURVE_INTERP` rather than spelled out, and `_build_graph_from_log_dfs` re-checks the mode at the one place it would have to be forwarded, raising `NotImplementedError`. Widening the declarable set without teaching `DiscountCurve` to dispatch fails loudly; `tests/test_curves/test_graph.py::TestCurveSpecInterp` is the tripwire |
| C-5 | One pillar per instrument, non-singular Jacobian, condition number monitored | MUST | `_validate_square`; `bootstrap_curve_graph(...).condition_number()`; `CurveAdjoint.jacobian_condition_number` | ✅ | Squareness is a hard eager check; the condition number is reported at both build and adjoint time |
| C-6 | When blocks couple, the solve and the IFT treat them jointly | MUST | `bootstrap_curve_graph` (one Newton over the concatenated log-DFs of every curve); `curve_adjoint` | ✅ | Joint by construction — there is no sequential path to get wrong. Gated by `TestT3ParallelShiftIdentity` and `TestT15…::test_discount_quote_delta_carries_the_indirect_part` |
| C-7 | Futures, FRA and OIS payment-lag convexity either modelled or explicitly neglected, and the choice recorded | SHOULD | `valax.curves.convexity` (`ConvexityAdjFn`, `no_convexity_adj`, `constant_convexity_adj`); `OISSwapRate` docstring (payment-lag timing convexity) | 🟡 | A plug-in point exists for futures and the default is explicitly "none". With N-3 closed, the OIS payment-lag item is now *narrowly* scoped and recorded where it bites: the discount factor moves to `T^p`, but the compounded coupon is not re-expectation-corrected for no longer being a martingale under the `T^p`-forward measure. See [documented choices](#documented-choices). No model-based adjustment ships |
| C-8 | Jump spreads live in `p` and are applied identically in the solve and in every sensitivity run | MUST | `valax.curves.TurnInstrument` | 🟡 | A turn is representable as a window-forward residual, which the adjoint differentiates like any other quote. But it relies on the caller placing pillars tightly around the turn, and the spread is not held in a `p` container (Q-5) |

## V — Volatility objects (Layer 1b)

| ID | Requirement | MUST/SHOULD | Implementing symbol | Status | Gap |
|---|---|---|---|---|---|
| V-1 | Vol objects take forwards, annuities and discount factors from the same snapshot's curves through differentiable calls | MUST | `strip_caplet_vols`, `strip_optionlet_surface`, `caplet_strip_node` → `optionlet_sabr_node`; `calibrate_swaption_cube_from_curves` / `_from_graph`, `swaption_cube_node`, `swaption_node_forwards_and_annuities`, `swaption_node_coordinates`; the `forward=` leg of the vol-source protocol in `valax.pricing.analytic.rates_smile` | ✅ | Both sides comply and are gated, standalone and as chain levels. Caps: `tests/test_surfaces/test_caplet_stripping.py` differentiates a stripped vol w.r.t. projection-curve pillars. Swaptions: the curve-aware fit derives `(S_x, A_d)` and the grid axes from the node schedules via the *pricer's own* `swaption_forward_and_annuity`, so `d(fitted α)/d(DF) ≠ 0` — asserted against a fit whose forward is `stop_gradient`-ed, which is exactly `0.0`, in `tests/test_greeks/test_vol_adjoint.py::…::test_the_coupling_is_absent_from_a_frozen_forward_fit`. In a chain the forwards are recomputed inside the level from the upstream curves, and `tests/test_calibration/test_rates_vol_nodes.py` gates the coupled curve → vol → price risk against autodiff and full-recalibration FD. There is no frozen-forward swaption-cube route any more |
| V-2 | Premium-to-vol conversion uses the instrument's settlement convention and the discount annuity | MUST | `strip_caplet_vols(cap_premiums=...)`; `valax.surfaces.swaption_implied_normal_vol` / `swaption_premium_residual` / `swaption_forward_and_numeraire` | ✅ | Caps: DF-weighted, which is correct — a cap strip has no annuity. Swaptions: `R = N·𝒩(θ_c)·Bachelier(S_x(θ_c), K, σ_N, T) − q = 0` solved by scalar `optimistix.Newton` with implicit diff, where 𝒩 is the physical annuity `A_d` or, under cash settlement, `P_d(0,T^p)·A_cash(S_x)`. The two settlements imply different vols from the same premium and both round-trip to 1e-10. Cash-settled *convexity* is a documented deviation, below |
| V-3 | Optionlet stripping uses the same caplet schedule, index and first-caplet exclusion as the quoted caps | MUST | `strip_caplet_vols` reads schedules off the `Cap` pytrees it is given; `caplet_stripping._validate_strip` | ✅ | Schedule and index are inherited from the quoted instrument, so they cannot drift. First-caplet exclusion **is** enforced: `_validate_strip` counts optionlets fixing on or before the curve's reference date and raises, because an already-fixed optionlet has identically zero vega and the bootstrap would be solving for an unidentifiable parameter |
| V-4 | RFR optionlets use accrual-period variance, not variance to the start date | MUST | — | ❌ | Optionlet pricers are term-rate Black-76 / Bachelier; no in-arrears accrual-period variance treatment |
| V-5 | The smile dynamics implied by holding the vol quotes fixed is documented and matches risk policy | MUST | `valax.surfaces.dynamics` (`SMILE_DYNAMICS`, `QUOTE_FORMS`, `validate_smile_dynamics`, `resolve_smile_query`, `default_dynamics_for`); `SwaptionCube.dynamics`, `OptionletVolSurface.dynamics`; [Volatility Surfaces § Smile dynamics](../guide/vol-surfaces.md#smile-dynamics-what-moves-when-the-curve-moves) | ✅ | **Documented and selectable.** All three conventions are implemented and the `(dynamics, quote_form)` pair is validated at construction, because the dynamics is a consequence of what the quote holds fixed. Default `"sticky_strike"` preserves every pre-existing number; the forward-dependent conventions raise rather than silently fall back when queried without a live forward. `tests/test_surfaces/test_smile_dynamics.py` asserts `∂σ/∂θ_c` is exactly `0.0` under the default and non-zero under `"sabr_backbone"` — so the convention is now a fact of record rather than a side effect |
| V-6 | SABR beta, shifts, weights and wing settings in `p`, frozen within a sensitivity run | MUST | `fixed_beta` / `shift` on the cube and surface fits; `valax.greeks.FixedInputs` + `fixed_input_adjoints` for their adjoints | 🟡 | **Beta and the shift are now in `p` with reported adjoints** and are gated by T-14, including the shift's zero adjoint on a lognormal cube (it enters only the Bachelier branch) with the normal-cube complement. Still partial for the other two: least-squares **weights** are frozen as a fit argument but carry no adjoint, and **wing settings** are not a VALAX concept at all — there is no wing-extrapolation parameter to freeze, so that clause is unimplementable rather than unimplemented |
| V-7 | Butterfly and calendar arbitrage checked after each fit | SHOULD | `valax.ButterflyArbError`, `CalendarArbError`, `NonConvexSmileError` (reserved types) | ❌ | The exception types are declared and never raised; no post-fit check runs |
| V-8 | If an arbitrage-free SABR PDE is used, its grid is frozen | SHOULD | — | n-a | Only the Hagan expansion ships; there is no SABR PDE whose grid could move |

## M — Model calibration (Layer 2)

| ID | Requirement | MUST/SHOULD | Implementing symbol | Status | Gap |
|---|---|---|---|---|---|
| M-1 | The model short rate is the collateral rate; the model reproduces the initial discount curve to solver precision | MUST | `hw_market_df`, `g2pp_market_df`, `cheyette_market_df` (initial curve entering the reconstitution formula by construction) | ✅ | The initial curve is fitted exactly by construction, not by calibration |
| M-2 | Every projected fixing uses a basis from the same snapshot's curves, with the coupon's own index period | MUST | `valax.pricing.multicurve` (`tenor_basis`, `swap_leg_basis`, `apply_swap_basis`, `with_discount_curve`) | 🟡 | Deterministic tenor basis ships for Hull-White and G2++ (MC and PDE). `calibrate_cheyette` / `calibrate_cheyette_2f` **raise** on a distinct projection curve |
| M-3 | All six paths by which curves enter Layer 2 are differentiable functions in code, none a cached constant | MUST | Paths 1, 2 and 6 via `price()` binding the model's curve (`curve_requirements`) + `multicurve`; paths 3 and 4 via `swaption_cube_node` + `SwaptionCube.dynamics`; the model level itself via `swaption_basket_node` | 🟡 | Paths 1–4 and 6 are differentiable *end to end* and `dR_m/dθ_c` now exists: the per-trade basket level reads the curves (model binding, target annuities and forwards) and the cube from the market, so `chain_adjoint` carries `λ_m` onto the curve quotes, FD-validated in `tests/test_calibration/test_per_trade_basket.py`. Path 5 (basket strikes defined off the curves, e.g. ATM baskets) is not implemented: `coterminal_swaptions` uses the trade's own strike |
| M-4 | Basket targets use the cube at the basket strike with the basket's actual underlying | MUST | `swaption_basket_node` (targets = `price(swaption, ImpliedVolModel(cube_id), market)`, i.e. the `rates_smile.swaption_price` kernel, of each basket swaption off the calibrated cube and curves) | ✅ | Each target is the basket swaption itself priced off the cube at its own strike, expiry and tenor, with its forward and annuity from the snapshot curves (`swaption_forward_and_annuity`), so the target moves with both cube and curves. The standalone `calibrate_hull_white` still takes caller-supplied target prices (`swaption_prices_from_vols`) and is not the M-4 route |
| M-5 | Fixed model inputs (mean reversion, correlation, factor count) in `p` and reported as their own sensitivities | MUST | `valax.greeks.FixedInputs`, `fixed_input_adjoints` | ✅ | Reported separately from `q̄` with names and units. Factor count is structural (a different model class), so it has no sensitivity by construction rather than by omission |
| M-6 | The PDE grid is frozen across calibration iterations, bumps and the adjoint pass | MUST | `PDEConfig` (caller-supplied, all-static fields) | 🟡 | Frozen because every field is static, so the grid cannot vary with a model parameter or a curve. Two gates now bear on it: T-9 requires repricing from a **reused** `PDEConfig` to be bit-identical, and T-8 scans a quote over a fine grid of bumps and requires no steps. The gap is narrower but real — **T-8's scan is at the curve and analytic layers, not through the PDE**, so a grid-induced step in a PDE price under a bump would not be caught; and nothing *enforces* that a caller does not rebuild the config inside a bump loop |
| M-7 | Payoff kinks smoothed and the first time steps damped, so the calibration Jacobian and gamma are not noisy | MUST | `valax.pricing.mc.payoffs` (sigmoid smoothing); `valax.pricing.pde.payoff_smoothing` (`PDEConfig(2D).smooth_payoff`, default on); `pde.schemes.theta_scheme` / `schemes2d` (`rannacher_steps`), forced to ≥ 2 by `pde.hull_white`, `pde.g2pp`, `pde.cheyette` (`_MIN_RANNACHER = 2`) | ✅ | Resolved. Smoothing exists on the Monte Carlo payoff path **and on the short-rate PDE path**: terminal payoffs and exercise / call / put projections are sampled as cell averages of the positive part (exact for a locally linear argument in 1-D; a bias-neutral C¹ kernel in 2-D), so Hull-White / G2++ / Cheyette prices are C¹ in quotes and parameters — a G2++ σ-scan shows 100 first-difference jumps pointwise and none smoothed, and Bermudan FD matches `jax.grad` across steps 1e-3…1e-6 (`tests/test_pde/test_payoff_smoothing.py`). Every short-rate PDE engine also applies **Rannacher start-up unconditionally**: each takes `rannacher_steps=max(config.rannacher_steps, _MIN_RANNACHER)`, so a caller cannot switch the damping off even by passing `PDEConfig(rannacher_steps=0)`. The equity PDEs (and the equity Bermudan projection) do not yet smooth their payoffs |
| M-8 | After calibration, reprice the basket with the production engine and report residuals | SHOULD | `tests/test_calibration/test_audit_conformance.py::TestBasketRepricedByTheProductionPDE` | ✅ | The calibrated Hull-White model reprices its own basket through `pde_price_dispatch` — a different discretisation of the same model, where repricing with the analytic Jamshidian formula the fit drove to target would be near-tautological (2e-15). Residuals are reported per member, max 1.47e-04 at a 200×200 grid, and **attributed**: doubling the grid drops them to 3.67e-05, so they are second-order discretisation error rather than misfit. See T-9 |

## A — Adjoint chain

| ID | Requirement | MUST/SHOULD | Implementing symbol | Status | Gap |
|---|---|---|---|---|---|
| A-1 | No tape records optimiser iterations; each calibration is differentiated only through its residual at the converged solution | MUST | `bootstrap_curve_graph` (`optimistix.root_find`, implicit adjoint); `level_adjoint`; `fixed_input_adjoints` | 🟡 | **Layer 1a complies and is gated** by T-2. **A model calibrator is now differentiated through too**: T-14's re-calibrated reading differentiates `calibrate_hull_white` end to end and matches finite differences that each re-run the fit, which is positive evidence that `optimistix`'s implicit default is in force there — the earlier note that no test differentiated through any `calibrate_*` except caplet stripping is superseded. Every calibrator now solves through the single `run_solver`, so the implicit default is in force in one place — but it is inherited rather than passed as `adjoint=` explicitly, so nothing *enforces* it. `calibrate_slv_leverage` is a deliberate, documented exception: a Monte Carlo fixed point with no residual driven to zero, reported as a `fixed_point` level and differentiated by plain AAD through its unrolled passes (see [Calibration Chains § I.8](../theory/calibration-chains.md#i8-fixed-point-levels)) |
| A-2 | The residual in the adjoint is the same code path as the one solved | MUST | `bootstrap_graph._graph_residual` — the single implementation, called by both `bootstrap_curve_graph` and `curve_residual_fn` | ✅ | Structurally guaranteed: there is one residual function and both consumers call it |
| A-3 | Every term of the chain present, including `dR_m/dθ_c`, `dR_v/dθ_c`, `dR_v/dq_v` | MUST | `valax.greeks.quote_risk.chain_adjoint` over `curve_graph_node`, `swaption_cube_node` / `caplet_strip_node` / `optionlet_sabr_node` and `swaption_basket_node` (via `model_levels`); `curve_quote_deltas` (λ_c); `valax.greeks.vol_adjoint` (`SliceVolAdjoint.vol_quote_vegas`, `.forward_bar`) | ✅ | λ_c, λ_v and λ_m are all present. `dR_v/dq_v` and `dR_v/dθ_c` are validated against full-rebuild FD at two bump sizes under both quote forms (`test_vol_adjoint.py`) and, as chain levels, against autodiff and full-recalibration FD (`test_rates_vol_nodes.py`); `dR_m/dθ_c` and `dR_m/dθ_v` through the per-trade basket are validated the same way (`test_per_trade_basket.py`: adjoint = autodiff to 1e-6, = FD to 5e-4) |
| A-4 | Signs follow the adjoint form | MUST | `quote_deltas_from_cotangent` (returns `-(dR/dq)^T λ`) | ✅ | Validated by sign and magnitude against full-rebuild FD in both directions |
| A-5 | For least-squares levels, `dR/dθ` and `dR/dq` are the full derivatives of the first-order condition including `Λ`, or Gauss-Newton is documented and validated | MUST | `valax.greeks.sabr_slice_level` + `level_adjoint` (`dR/dθ`); `vol_quote_vegas` (`dR/dq_v`) | ✅ | Satisfied by the same mechanism as R-5: the smile level's residual *is* the weighted-SSE gradient, so `jax.jacobian` of it is the exact full Hessian and nothing is approximated away. `dR_v/dq_v` is FD-validated at two bump sizes. The `Λ` clause is **vacuous rather than skipped** — VALAX carries no regulariser (R-3), so there is no damping term that could be omitted from the derivative. This row previously read ❌ on the grounds that no least-squares level was differentiated, which contradicted R-5 in this same table |
| A-6 | The condition number of each `dR/dθ` is monitored | MUST | `CalibrationResult.condition_number()` (every calibrator, including `bootstrap_curve_graph`); `CurveAdjoint.jacobian_condition_number`; `level_condition_number` / `condition_alert` / `ILL_CONDITIONED_THRESHOLD` | ✅ | "Each" is literal: every level reports through the one result type (a batched level over all its blocks), Layer 1a also at adjoint time, all against one shared threshold. Gated by `tests/test_greeks/test_level_conditioning.py` (T-13) and, for the rates vol and basket levels, `test_rates_vol_nodes.py::TestConditioning` / `test_per_trade_basket.py` |
| A-7 | Layer 1 Jacobian factorisations computed once per snapshot and reused across trades | SHOULD | `CurveAdjoint.factorisation` (a `lineax.LU` state built once in `curve_adjoint`) | ✅ | Each additional trade is one back-substitution plus one reverse sweep; gated by `TestFactorisationReuse::test_reused_handle_matches_fresh_handles` |
| A-8 | Fixed inputs `p` have adjoints accumulated and reported, not silently dropped | MUST | `valax.greeks.fixed_inputs` (`FixedInput`, `FixedInputs`, `fixed_input_adjoints`, `FixedInputAdjoint`) | ✅ | A `FixedInput` stores *where* the value lives, not the value: the adjoint is taken by re-injecting the leaf into the live pytree and differentiating through the injection, so the reported number belongs by construction to the leaf that was priced. A container holding copies would be a second source of truth and would reproduce A-8's failure in a new place. Both the **direct** (`∂V/∂p`, objects frozen) and **total** (recalibrate-then-price) readings are supported and gated; they differ by 6.6× on mean reversion |

## T — Verification tests

Run the whole suite with `pytest -m audit`. Every gate names the requirement it
verifies and states its own tolerance in the test docstring.

| ID | Test | Status | Where | Tolerance committed to |
|---|---|---|---|---|
| T-1 | Reprice every calibration instrument from the calibrated objects | ✅ | **Curves** — `tests/test_curves/test_bootstrap_graph.py::…::test_fitted_quotes_near_quoted_at_convergence`, off `fitted_curve_quotes(...)` against `result.inputs.quotes` (an exact quote-unit fit error, because every residual is linear in its primary quote scalar). **Caps** — `tests/test_surfaces/test_caplet_stripping.py::TestExactRepricing`. **Swaption cube** — `tests/test_surfaces/test_swaption_cube.py::TestCurveAwareCalibration::test_t1_every_fitted_quote_reprices`. **Basket** — `tests/test_calibration/test_audit_conformance.py::TestBasketRepricedByTheProductionPDE` (shared with T-9) | All four levels the requirement names are now covered and `audit`-marked, so `pytest -m audit` collects them. Tolerances per level in the table below. The earlier "vol objects: no gate" note was **stale**: the cube gate already existed but was neither marked nor recorded here |
| T-2 | AAD quote sensitivities vs full-rebuild central finite differences, every quote, ≥2 bump sizes | ✅ | `tests/test_greeks/test_quote_adjoint.py::TestT2AgainstFiniteDifferences` | Bumps 0.1 bp and 1 bp; `rtol = 1e-5` where `|FD| > 1`, `atol = 1e-4` on FD-noise-dominated entries. Observed agreement is 1e-9 to 1e-11 relative |
| T-3 | Parallel-shift identity: summed quote deltas vs an all-quotes-bumped rebuild | ✅ | `…::TestT3ParallelShiftIdentity` (single- and dual-curve) | `rtol = 1e-6`; observed 1e-11 |
| T-4 | Single-curve limit (unit basis) matches an independent single-curve pricer | ✅ | `tests/test_calibration/test_audit_conformance.py::TestSingleCurveLimitAtThePricers`, `::TestSingleCurveLimitAtTheBootstrap`, `::TestSingleCurveLimitAtTheModel` | `rtol = 1e-15` at the pricers (observed **exactly 0.0** for the float leg, par rate, swap price, annuity and caplet forward; 1.4e-17 on the swaption forward — it is a telescoping identity, not an approximation); `rtol = 1e-10` on the bootstrap's two curves; `rtol = 1e-8` on the refitted model parameter. The sharp form: `forward_curve=curve` is passed **explicitly**, so the dual-curve code path runs and must land on the single-curve answer. The two pre-existing tests cannot do this — one compares two *solvers*, the other compares `forward_curve=None` against the default, which is the same branch. The complement is gated too: a 25 bp basis must *not* degenerate |
| T-5 | OIS telescoping with zero payment lag | ✅ | `tests/test_curves/test_instrument_residuals.py::TestPaymentLagTelescoping` | `atol = 1e-14` on a unit-notional floating-leg PV — a round-off budget, not a modelling tolerance, since the per-period sum and the closed telescope differ only by summation order. The gate drives the **per-period** branch (explicit payment dates equal to the accrual ends) against a two-read reference `P_d(0,T_0) − P_d(0,T_n)`, so it is no longer vacuous; `test_telescope_does_not_survive_a_lag` pins the converse |
| T-6 | Put-call parity (caplet−floorlet = FRA; payer−receiver = annuity × (forward−strike)) | ✅ | `tests/test_surfaces/test_audit_parities.py::TestCapletFloorletParityIsTheFRA`, `::TestPayerReceiverParityIsTheAnnuity`, plus the per-leg gates `::TestCapletLegsMatchTheirOwnClosedForms` / `::TestSwaptionLegsMatchTheirOwnClosedForms` | `rtol = 1e-12`, `atol = 1e-9` on a 1e6 notional. **Stated naively this gate is vacuous**: VALAX derives the floorlet and the receiver *by* parity (`caplets.py`, `swaptions.py`), so the difference is exact by construction. The gate therefore checks each leg against an independently written closed form (the put as `K Φ(−d₂) − F Φ(−d₁)`, never as `call − (F−K)`) and rebuilds `τ`, `A_d`, `S_x` and `P_d(T_e)` by hand, which is what catches a swapped projection/discount curve or a mis-anchored first accrual period. Run single- *and* dual-curve, since a `P_d`/`P_x` swap is invisible when the curves are one object |
| T-7 | Bachelier ATM straddle identity | ✅ | `…::TestBachelierATMStraddle` (swaption and caplet forms) | `rtol = 1e-10`. Pins the *scale* of the normal-vol convention — a missing `√(2π)`, a doubled `√T`, or a variance/vol mix-up shows up here and nowhere in T-6, which is vol-insensitive. Not weakened by the receiver coming from parity: the parity term is zero at the money, so the identity constrains the payer formula alone. Corollaries gated too: the ATM straddle is exactly linear in σ, and the Black-76 ATM straddle converges to the Bachelier one at the measured `O((σ√T)²)` rate (asserted as a convergence *rate*, not a fixed tolerance) |
| T-8 | Smoothness scan: price and delta over a fine grid of bumps show no steps | ✅ | `tests/test_calibration/test_audit_conformance.py::TestSmoothnessUnderGridRefinement` | Phrased as a **convergence rate**, not a tolerance: "no steps" is unassertable at any tolerance, and asserting near-linearity over an interval wide enough to show real curvature is the mistake of numerical-pitfalls entry 3. Halving the scan spacing must shrink `max|Δ²price|` 4× and `max|Δdelta|` 2×; a single discontinuity pins both ratios near 1. Ratios required within **15%** of 4 and 2; observed 3.99 and 2.00. Scan: ±2 bp on the 5y OIS quote, 17 then 33 points, every point a full rebuild (graph, adjoint and factorisation) so nothing is carried between bumps — the R-6/M-6 condition |
| T-9 | Basket repriced by the production engine after calibration | ✅ | `…::TestBasketRepricedByTheProductionPDE` | `< 5e-4` relative at a 200×200 grid; observed max 1.47e-04, mean 5.2e-05. Refining to 400×400 must cut the residual by **at least 2×** (second order predicts 4; observed 4.0), which is what attributes it to discretisation rather than misfit — a single reported residual cannot distinguish those. The analytic (Jamshidian) refit is recorded separately at 2e-15 to show what the PDE comparison adds: a genuinely different discretisation of the same model. M-6 gated alongside by requiring bit-identical repricing from a reused `PDEConfig` |
| T-10 | Day count and schedule spot checks on known dates | ✅ | `tests/test_dates/test_audit_conventions.py` | Every expected value is derived from the published convention using the **standard library only** (`datetime`, `calendar`) or integer arithmetic transcribed from the rule, so nothing is checked against VALAX's own ordinal algebra. Dates exact (integer equality); `act_360` / `act_365` / `30_360` exact (bit-for-bit); `act_act` `atol = 1e-14` — see the tolerance table for why that one cannot be exact |
| T-11 | Fixing handling across a fixing date; a fixed coupon's curve sensitivities exactly zero | ✅ | `tests/test_curves/test_instrument_residuals.py::TestPartlyAccruedFixings` (three `audit`-marked gates) | Tolerance: **exact equality to `0.0`**, not a threshold. A recorded coupon's projection-curve adjoint is identically zero because the selection is a `jnp.where` over a NaN-free replacement, so the unselected branch receives a zero cotangent. The complement is asserted too — sensitivity to the stored fixing itself, and to the discount curve, must be non-zero — so the gate cannot pass by nothing being connected. A third gate shows a straddling period depends on *both* the realised factor and the curve |
| T-12 | Smile-dynamics check with vol quotes held as defined | ✅ | `tests/test_surfaces/test_smile_dynamics.py::TestT12SmileDynamicsDelta` | `rtol = 1e-12`; `sticky_strike`'s zero is asserted **exact**. Checked in delta, because at `F = F₀` all three conventions return the same vol and therefore the same price — **no repricing test can detect a wrong convention**. `dσ/dF` is checked against partials of `sabr_implied_vol` taken directly, so the cube does not define its own right answer. Gated against vacuity too: the three deltas must differ by >2%, and a companion test pins that a flat smile (ν=0, β=1, ρ=0) collapses the whole distinction |
| T-13 | Condition numbers of `dR/dθ` reported at every level | ✅ | `valax.greeks.level_condition_number`, `condition_alert`, `ILL_CONDITIONED_THRESHOLD`; `tests/test_greeks/test_level_conditioning.py` | Alert threshold **1e10**, now shared by all levels from one constant. Layer 1a gated on a real curve build (and the three reports of the same Jacobian asserted to agree to `rtol = 1e-10`), Layer 1b on a real SABR fit, Layer 2 on a real Hull-White fit in both its square and least-squares forms. The ill-conditioned-but-finite branch of the alert is driven by a synthetic Hilbert level — a documented choice, since no realistic VALAX level reaches 1e10 |
| T-14 | Fixed-input sensitivities vs finite differences | ✅ | `tests/test_greeks/test_fixed_inputs.py` | The gate's three clauses are asserted separately. *Present*: a named, united row per input. *Non-zero where expected* **and exactly `0.0` where not** — an unused model is placed in the tree and its adjoint required to be identically zero, so a carrier fabricating numbers or reading the wrong leaf would fail. *Matching*: central finite differences at two bump sizes, `rtol = 1e-6`/`1e-5` direct (observed 6.3e-10, 6.2e-8) and `1e-5`/`1e-3` re-calibrated (observed 2.1e-8, 4.6e-6), each FD re-running the whole fit. Covers mean reversion (model layer) and the SABR exponent and shift (vol layer); the shift's zero adjoint on a lognormal cube is pinned *as* the expectation, with the normal-cube complement |
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
| `θ_m` — model unknowns | Model pytree leaves (`HullWhiteModel.volatility`, …), unconstrained via `TransformSpec`; `CalibrationResult.theta` |
| `p` — fixed non-solved inputs | `QuoteSet` fixed inputs → `LevelInputs.fixed`; `FixedInputs` for paths into fitted objects (Q-5) |
| `R` — residual | `BootstrapInstrument.residual`; `curve_residual_fn`; `CalibrationResult.residual_fn` / `CalibrationLevel.residual_fn` |
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
| Cap premium − quoted premium (solved bucket by bucket, one convergence flag per bucket) | bucket vol `σ_i` | cap quotes, discount + projection curves | `caplet_strip_level` — one lower-triangular root level behind `strip_caplet_vols`; strike columns batched by `strip_columns_level` (`caplet_strip_node`) |
| Stationarity of the weighted SSE of a SABR smile | `(α, ρ, ν)` or `(α, β, ρ, ν)` | quoted vols, node forward `F_0` | `swaption_cube_level` (behind `calibrate_swaption_cube_from_curves` and `swaption_cube_node`), `sabr_batch_level` (optionlet surface); the level residual is the stacked per-node first-order condition, so `chain_adjoint` differentiates it directly. `valax.greeks.sabr_slice_level` + `vol_quote_vegas` re-derive the same level from a fitted object |
| `N·𝒩(θ_c)·Bachelier(S_x, K, σ_N, T) − q` | `σ_N` | premium quote, discount + projection curves, settlement convention | `valax.surfaces.swaption_premium_residual` / `swaption_implied_normal_vol` |

Layer 2: `calibrate_hull_white`, `calibrate_g2pp`, `calibrate_cheyette(_2f)`
are weighted least-squares levels (`basket_fit_level`: quotes = target prices,
upstream = curves, held-fixed parameters as fixed inputs) returning a
`CalibrationResult` with a curve-free model; the per-trade rates level is
`swaption_basket_node`, whose targets are read off the calibrated cube and
curves and whose model prices go through `price()`. Every one carries its
residual (the first-order condition of the fit), so the IFT step applies.

### Stated tolerances

| Gate | Tolerance VALAX commits to |
|---|---|
| T-1 (curves) | Two distinct numbers, because the residual and the quote are not in the same units: `max_abs_residual < 1e-10` in *residual* units, and `max|fitted_quotes − quoted_values| < 1e-8` in each instrument's *native quote* units. The second is the looser of the two by construction — a deposit or futures residual is ~`O(DF·τ·Δr)`, so the quote-unit fit is the residual scaled by ~`1/(DF·τ)` |
| T-1 (vol objects) | Swaption cube: every fitted quote reprices from the calibrated object at its own node coordinate to `atol < 1e-8` in vol units, under both quote forms. Caps: `< 1e-10` relative — stripping is a per-maturity root-find, so repricing is exact up to solver tolerance by construction rather than by approximation |
| T-1 (basket) | Analytic refit `< 1e-12` relative; production PDE `< 5e-4` at a 200×200 grid. Shared with T-9, which adds the grid-refinement attribution |
| T-2 (curve quotes) | Bumps 0.1 bp and 1 bp; `rtol = 1e-5` where `|FD| > 1`, else `atol = 1e-4` |
| T-2 (vol quotes) | Vol-quote vegas: bumps 1e-4 and 1e-3 in vol units. Forward coupling: bumps 0.1 bp and 1 bp, both quote forms. Both at `rtol = 1e-4` relative to the largest component of the full-rebuild finite-difference vector. The `optimistix` `ImplicitAdjoint` comparison is also run at `rtol = 1e-9`, but is recorded as a **consistency** check rather than an independent one — it shares `sabr_implied_vol` with the adjoint under test, and that shared primitive is exactly what once let both agree while both were wrong (see [Numerical Pitfalls entry 5](numerical-pitfalls.md#5-a-value-correct-guard-with-no-derivative-jnpwhere-on-a-removable-singularity)) |
| T-9 | **Not yet gated** |
| T-14 | Direct `p̄`: `rtol = 1e-6` at a 1e-5 bump, `1e-5` at 1e-4 (observed 6.3e-10, 6.2e-8). Re-calibrated `p̄`: `rtol = 1e-5` at 1e-4, `1e-3` at 1e-3 (observed 2.1e-8, 4.6e-6) — looser because every finite difference re-runs a full Hull-White fit, so the solver's own tolerance enters the difference twice. An input the price does not read: **exactly `0.0`** |
| T-4 | `rtol = 1e-15` at the pricers (one ulp; observed exactly 0.0 on five of six observables), `1e-10` on the bootstrap's two curves, `1e-8` on the refitted model parameter |
| T-8 | Refinement ratios within **15%** of their theoretical 4 (price, `O(h²)`) and 2 (delta, `O(h)`); observed 3.99 and 2.00. No tolerance is placed on the differences themselves, deliberately — their magnitude is meaningless, only their scaling is |
| T-9 | PDE basket residual **< 5e-4** relative at 200×200 (observed max 1.47e-04); refinement ratio to 400×400 **> 2** (observed 4.0); analytic refit < 1e-12 |
| T-12 | `rtol = 1e-12` on the chain-rule decomposition and on each convention's `dσ/dF` against the Hagan partials; `sticky_strike`'s `dσ/dF == 0.0` asserted as **exact equality** (the branch does not consult the live forward, so the cotangent is structurally zero, and a tolerance would mask it starting to). Observed: the convention identities hold to exactly zero relative error, the chain rule to ~1e-16 |
| T-6 | `rtol = 1e-12` per leg against its own closed form, and for the parity term against an independently reconstructed forward value; `atol = 1e-9` currency units on a 1e6 notional, because the parity term passes through zero at the money where a relative tolerance is meaningless |
| T-7 | `rtol = 1e-10` — two orders looser than T-6 deliberately: the strike is set to a *separately computed* forward, so the contract is at the money only to the agreement of the two forward reconstructions, and the residual `(F − K)` enters the straddle quadratically |
| T-10 | Dates **exact** (integer equality — a date is a whole number of days, so approximate equality is never the right assertion). `act_360`, `act_365`, `30_360`: **exact**, bit-for-bit, each being integer day arithmetic plus one division. `act_act`: `atol = 1e-14`, observed worst 2.9e-15. It cannot be exact and the asymmetry is deliberate — `valax.dates.daycounts.act_act` evaluates a *cumulative* function and subtracts (`F(end) − F(start)`), while ISDA 4.16(b) *sums* per-year contributions. Equal in exact arithmetic, different floating-point operation sequences. The closed form is the better-conditioned of the two, so the implementation is not the one being corrected here |
| T-13 | Condition-number alert threshold **1e10**, in one place: `valax.greeks.adjoint.ILL_CONDITIONED_THRESHOLD`. Agreement between the three reports of the same Layer-1a Jacobian: `rtol = 1e-10`. Every level must report a finite κ ≥ 1. **Measured levels:** curve build ≈ 1.1 (structurally near-identity — a telescoping OIS leg makes each quote pin its own maturity pillar, and crowding pillars one day apart does not spoil it); SABR slice ≈ 2.2e4 with β fixed, ≈ 5.6e6 with β free; Hull-White (a, σ) ≈ 5.6e3 against ≈ 1 for σ alone. All four orders of magnitude inside the threshold, so **the earlier concern that a 1e10 alert would fire on healthy smile fits is measurably unfounded** — it was based on the ~1e6 Hessian conditioning, which is 1e4 short of the threshold |

### Documented choices

The four choices the specification requires to be stated explicitly:

| Choice | VALAX position |
|---|---|
| **Smile dynamics** (V-5) | **Default sticky absolute strike, now a declared and selectable choice.** `dynamics` ∈ `{"sticky_strike", "sticky_moneyness", "sabr_backbone"}` is a static field on both rates vol objects, validated against `quote_form` because the dynamics is a consequence of what the quote holds fixed. The default preserves every pre-existing number; `"sabr_backbone"` is never selected implicitly, because it reproduces a node's market vol only when `F = F_0` there — which holds only on the curve-aware calibration route. The three conventions, the admissibility table, the off-node mixed-order caveat and the backbone prerequisite are stated once in [Volatility Surfaces § Smile dynamics](../guide/vol-surfaces.md#smile-dynamics-what-moves-when-the-curve-moves) |
| **Neglected convexities** (C-7) | Futures convexity is **explicitly neglected by default** (`no_convexity_adj`) with a plug-in point (`ConvexityAdjFn`) for a desk-supplied or constant adjustment; no model-based adjustment ships. FRA settlement convexity and OIS payment-lag **timing** convexity are **neglected without being modelled**. The latter is now a genuinely second-order omission rather than a consequence of ignoring the lag outright: since N-3 closed, the lagged coupon is discounted at its true `T^p`, and what remains neglected is only the drift correction from the compounded rate no longer being a martingale under the `T^p`-forward measure. **Cash-settled swaption convexity** is likewise neglected: `swaption_forward_and_numeraire(settlement="cash")` evaluates the street-standard flat-yield annuity *at the forward* rather than inside the expectation, so the smile-integral correction (Mercurio 2008) is not applied. Second order in vol, and stated in the module's own docstring |
| **Gauss-Newton vs full Hessian** (R-5) | **Full Hessian.** The smile level's residual *is* the gradient of the weighted SSE (`valax.greeks.sabr_slice_level`), so `dR_v/dθ_v` obtained by `jax.jacobian` is exactly `J^T W J + Σ_k r_k ∇²r_k` — the Gauss-Newton term plus the curvature term, with nothing dropped. Chosen over Gauss-Newton because the two coincide to the size of the residual on a tight fit, so the approximation only ever matters when the fit is *loose* — precisely when a sensitivity is most likely to be trusted wrongly. There is no neglected term to monitor; `SliceVolAdjoint.stationarity_norm` surfaces the first-order-condition violation instead, which is the assumption that *can* fail |
| **Multiplicative vs additive basis** (§ 6.2) | **Additive**: `valax.pricing.multicurve.tenor_basis` computes a deterministic spread `s = F_x − F_d` from today's curves and adds it to the model-implied rate |

## Asset-class chain requirements (RT, EQ, FX, XA)

The cross-asset requirements of [Calibration Chains](../theory/calibration-chains.md)
(Part III), against the generic chain layer (`valax.calibration.chain`,
`valax.greeks.quote_risk`).

| Req. | Requirement (short) | Status |
|---|---|---|
| RT-1 | Rates meet the Audit in full | 🟡 per the rows above |
| RT-2 | Equities / FX use the rates OIS and xccy curves from the same snapshot | 🟡 a `CalibrationChain` builds each curve once and every later level reads it from the same `MarketState`; no FX / xccy chain is assembled yet |
| EQ-1 | Forward with discrete dividends discounted at `r_s`, documented repo convention | ❌ continuous dividend yield only (`equity_forward`) |
| EQ-2 | American quotes: joint forward + vol fit per expiry | ❌ |
| EQ-3 | Constrained fits on the KKT active set, changes flagged | ❌ SVI is not a chain level |
| EQ-4 | Dupire inputs smooth; local variance positive | ❌ Dupire is not a chain level |
| EQ-5 | Level 2 calibrated once per underlying, shared | ✅ `calibrate_model` on the snapshot market; the result is passed to every trade's `quote_risk_adjoint(model_result=...)` |
| EQ-6 | MC triggers smoothed, regression frozen, duality gap reported | 🟡 smoothing ships (`valax.pricing.mc.payoffs`); duality gap not reported |
| EQ-7 | Correlations / quanto inputs are fixed inputs with sensitivities | 🟡 the mechanism ships (`fixed_fields` → `p̄`); no correlation level is wired |
| FX-1 … FX-8 | Spot-date forwards, delta system, conventions, touch level, hybrids | ❌ no FX chain levels yet |
| XA-1 | Shared objects built once per snapshot | ✅ by construction of `calibrate_market` |
| XA-2 | Quote risk spans every asset class a trade depends on | ✅ mechanism: `chain_adjoint` accumulates upstream cotangents across levels (curve → spot → smile → model validated); cross-asset levels pending |
| XA-3 | A rates quote change rebuilds every dependent object | ✅ `calibrate_market` recalibrates the whole chain; the autodiff path differentiates through every dependency |
| XA-4 | Collateral currency selects the discount curve uniformly | 🟡 via instrument selectors (`discount_id`), not a CSA model (see C-2) |
| XA-5 | Keys unique across asset classes | 🟡 `QuoteSet` keys are free-form; uniqueness is by convention |

---

## Deferred, with rationale

These are known ❌ / 🟡 rows that are *not* scheduled next, so that the gap list
stays honest rather than aspirational.

| Item | Requirements | Why deferred |
|---|---|---|
| Genuine step / instantaneous-forward interpolation, calendar-aware turns | C-4, C-8 | A curve-representation project in its own right (tracked as the interpolation workstream on the [roadmap](../roadmap.md)). The shipped log-linear-DF mode is already piecewise differentiable with kinks only at pillars, so it does not *violate* C-4. Deferring it no longer leaves a silently-ignored field behind: the one-mode invariant is asserted in both directions, so the first new mode must arrive with its dispatch |
| Full CSA / collateralised discounting | C-2 | Requires a collateral-agreement model on the instrument, not just a curve identifier. The discount/projection split that C-2 mainly guards is already enforced |
| RFR in-arrears optionlet variance | V-4 | Needs a compounded-in-arrears optionlet pricer; the current optionlets are term-rate by construction |
| Dual-curve Cheyette calibration | M-2 | `calibrate_cheyette` raises rather than silently single-curving, which is the safe failure mode. Hull-White and G2++ cover the dual-curve route |
| Versioned **index registry** (conventions resolved from an index definition) | N-5, N-6 | Narrower than it was, twice over: Tier 1.3 shipped the calendar half (`valax.dates.calendars`, cross-checked and version-stamped) and Tier 1.4 gave every leg an explicit `index_id`, so an index is now *named* per leg even though nothing resolves it. What is still deferred is the registry itself — a mapping from an index name (`USD.SOFR`, `EUR.EURIBOR.6M`) to its calendar, day count, roll and lag — plus the wiring to consult one, in both the curve build and the leg engine, whose projection window is still the accrual period by construction. Day-count and schedule reference data are already gated by T-10, and adjustment by `test_dates_ql.py` |
