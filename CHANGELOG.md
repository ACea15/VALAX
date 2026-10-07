# Changelog

All notable changes to VALAX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

VALAX is currently pre-1.0. Everything below was developed under the `0.1.0`
version tag in `pyproject.toml`. The first tagged release will compress the
history below into a single `[0.1.0]` entry; until then, all changes accumulate
under `[Unreleased]` and are grouped by feature area for discoverability.

### Added — rates option parity and Bachelier ATM gates (T-6, T-7)

`tests/test_surfaces/test_audit_parities.py` gates the two option identities
the audit specification asks for. The interesting part is why the obvious
version of T-6 would have been worthless.

VALAX does not price a floorlet or a receiver from its own formula — it prices
the caplet/payer and subtracts the forward value (`caplets.py`,
`swaptions.py`). So "caplet − floorlet = FRA" and "payer − receiver =
A_d(S_x − K)" are true *by construction*, to the last bit, and asserting them
tests only that subtraction is subtraction. That is the failure already
recorded as entry 4 of `docs/architecture/numerical-pitfalls.md`:
corroboration from something that shares your approximation is worthless.

The gate therefore builds two independent references. Each leg is checked
against its own closed form, with the put written as `K Φ(−d₂) − F Φ(−d₁)`
rather than as `call − (F − K)`, which is what makes the out-of-the-money leg
genuinely checkable. And `τ`, `A_d`, `S_x` and `P_d(T_e)` are reconstructed by
hand from the definitions, by explicit loop over curve reads — because that is
the part that actually bites: the annuity's first period accrues from *expiry*,
the dual-curve floating leg does not telescope, and the discount factor is read
at the payment date. Mis-wire any of those and parity still holds while the
reconstruction fails. Everything runs single- *and* dual-curve, since a
swapped `P_d`/`P_x` is invisible when the two curves are the same object.

T-7 (ATM straddle = `2 A_d σ √(T/2π)`) is the only gate here that constrains
the volatility scale at all — T-6 is vol-insensitive by construction — so it
is what would catch a missing `√(2π)`, a doubled `√T`, or a variance/vol
mix-up. It is not weakened by the receiver coming from parity, because the
parity term vanishes at the money. Two corollaries are gated with it: the ATM
straddle is exactly linear in σ, and the Black-76 ATM straddle converges to
the Bachelier one — asserted as a convergence *rate* (halving σ must shrink
the relative gap fourfold, the leading correction being `O((σ√T)²)`) rather
than at a fixed tolerance, which would pass equally well against a wrong
formula that happened to be close.

`docs/theory/rate-derivatives.md` now records that the out-of-the-money leg is
derived from parity: it guarantees a floor inherits exactly the smile
treatment its cap received, and it is the reason the gate is built the way it
is.

### Added — conditioning reported at every calibration level, with a shared alert threshold

`dR/dθ`'s condition number decides how much of the double-precision mantissa
survives the transposed solve that every quote sensitivity goes through. It was
reported only for Layer 1a (`CurveBuildDiagnostics`, `CurveAdjoint`), and the
`1e10` threshold everyone referred to existed only in prose — as a bare literal
in two docstrings and one doctest, with no constant and no alert anywhere in
code.

- `valax.greeks.adjoint.ILL_CONDITIONED_THRESHOLD` (`1e10`) is now the single
  source of truth; the docstrings and the doctest refer to it instead of
  restating it.
- `level_condition_number(level)` reports `κ(dR/dθ)` for **any**
  `CalibrationLevel` — curve, smile or model — accepting a pytree `θ` and
  rectangular (least-squares) levels as well as square ones. It deliberately
  *does* materialise `dR/dθ`, since a condition number is a property of the
  whole matrix; it is a per-snapshot diagnostic and the sensitivity path
  itself stays Jacobian-free.
- `condition_alert(κ, level_name=..., threshold=...)` returns a message or
  `None`, rather than warning or raising, so the caller picks the channel. A
  non-finite `κ` gets a distinct message: that is a singular system where the
  implicit function theorem does not apply at all, so the sensitivities are
  meaningless rather than imprecise.

Measured across the three layers (all four or more orders of magnitude inside
the threshold): curve build ≈ 1.1, SABR slice ≈ 2.2e4 with β fixed and
≈ 5.6e6 with β free, Hull-White (a, σ) ≈ 5.6e3 against ≈ 1 for σ alone. The
diagnostic therefore puts numbers on two pieces of standing advice — fix β,
fix mean reversion — that were previously folklore. It also retires the
concern recorded in the conformance map that a `1e10` alert would fire on
healthy smile fits: the worst healthy fit measured is 1e4 short of it.

Two findings worth recording, both measured rather than assumed. A VALAX curve
build is **structurally** near-perfectly conditioned: an OIS floating leg
telescopes, so in log-DF space the `P(T_n)` term dominates the annuity term and
each quote very nearly pins its own maturity pillar — crowding two pillars to
one day apart leaves κ ≈ 1.15, so ill-conditioning cannot be provoked from
pillar spacing. The failure mode that *is* real is a **duplicated quote**: it
keeps the system square, so no shape check rejects it, but it makes two
residual rows identical and leaves a pillar unconstrained.

Gated by `tests/test_greeks/test_level_conditioning.py` (**T-13**,
`pytest -m audit`), which asserts the three reports of the same Layer-1a
Jacobian agree to `rtol = 1e-10` and drives the alert's finite branch from a
synthetic Hilbert level — a documented choice, since no realistic level
reaches the threshold.

### Added — day-count and schedule reference-data gate (T-10)

`tests/test_dates/test_audit_conventions.py` pins τ and generated schedule
dates against references derived from the published conventions using the
**standard library only** (`datetime`, `calendar`) or integer arithmetic
transcribed from the rule. The independence is the point: `ymd_to_ordinal` and
`thirty_360` both route through VALAX's own Hinnant ordinal/calendar
conversion, so checking one against the other would prove only that the
algorithm is used twice.

Dates match exactly (integer equality). `act_360`, `act_365` and `30_360` match
**bit-for-bit**. `act_act` matches to `atol = 1e-14` (observed worst 2.9e-15)
and cannot be exact: `valax.dates.daycounts.act_act` evaluates a cumulative
function and subtracts, while ISDA 4.16(b) sums per-year contributions — equal
in exact arithmetic, different floating-point operation sequences. The closed
form is the better-conditioned of the two, so the implementation is not the
thing being corrected by the tolerance.

### Changed — `CurveSpec.interp` is no longer a field that could be silently ignored

`CurveSpec.interp` was validated against `graph._SUPPORTED_INTERP` and then
never forwarded anywhere, because `DiscountCurve` has no `interp` field — it
hard-codes log-linear-in-DF. That was not a live bug: the supported tuple held
exactly one entry, and that entry is precisely what the curve implements, so
no caller could ask for something they did not get. It was a *latent* one.
The day a second mode joined the tuple, every curve declaring it would have
quietly carried on interpolating log-linearly.

Rather than implement the dispatch (a curve-representation project, still
deferred) or delete a field the interpolation workstream needs, the invariant
is now enforced from both ends:

- New `valax.curves.DISCOUNT_CURVE_INTERP` names the mode
  `DiscountCurve.__call__` actually implements.
- `graph._SUPPORTED_INTERP` is **derived from** it rather than spelled out, so
  the declarable set cannot drift from the deliverable one by accident.
- `bootstrap_graph._build_graph_from_log_dfs` re-checks the mode at the single
  place it would have to be forwarded, raising `NotImplementedError` naming
  exactly what a future implementer must do first.
- `tests/test_curves/test_graph.py::TestCurveSpecInterp` asserts the one-mode
  invariant directly, so widening the tuple fails the suite until the dispatch
  lands.

Behaviour is unchanged for every existing caller. Narrows spec C-4.

### Added — Partly-accrued overnight coupons blend realised and projected

`_floating_leg_pv` treated every coupon as all-or-nothing: either a recorded
fixing existed for its fixing date, or the whole period was projected off the
curve. For a term rate that is right. For an *overnight-compounded* coupon it
is not — such a coupon has no single fixing date, it accrues daily, so a
period that has begun and not ended is partly realised and partly
forward-looking, and projecting all of it discards weeks of known data.

- **New `FixingSeries.compounded_factor(start, end, day_count)`** reduces a
  daily history to the realised compounding factor
  `A = Π(1 + r̂_j δ_j)`, plus `FixingHistory.compounded_factor(index_id, …)`
  which returns exactly `1.0` for an unrecorded index (a missing *history*
  legitimately means nothing has accrued, unlike a missing *fixing* for a
  coupon that needs one, which stays a `KeyError`). Weekends need no calendar:
  an absent entry leaves the preceding rate applying across the gap weighted
  by calendar days, which is the market convention.
- **New `accrued_compounding`** on `IborSwapRate`, and per leg as
  `leg_a_/leg_b_accrued_compounding` on `TenorBasisSwap` and
  `dom_/for_accrued_compounding` on `CrossCurrencyBasisSwap`. The blend is
  exact rather than an approximation, because the compounding factor over the
  period *is* the product of the realised and remaining factors:
  `1 + τ_j F_j = A · P_x(0,t_0)/P_x(0,T_j^e)`.
- **`ref_date` is finally used.** Every floating-leg residual used to `del` it;
  it now locates the straddling period, by the half-open test
  `T^s ≤ t_0 < T^e`. Closing both ends would match *two* periods whenever
  `t_0` lands on an accrual boundary — the one ending there and the one
  starting there — which cannot share one realised factor.
- **Precedence is explicit:** recorded fixing → blend → projection. A recorded
  value outranks a partial accrual because for a tenor index it *is* the
  compounded coupon rate for the whole period.
- **A fully-ended period is deliberately out of scope**, since its rate is
  entirely known. Where a payment lag leaves one ended but unpaid, record its
  compounded rate under `index_id`; reaching for `accrued_compounding` there
  would silently leave it projected. Documented on `_floating_leg_pv`.
- **Default is off**, so nothing changes for anyone who does not set the field.
- Closes spec requirement C-3 and the T-11 gate. See
  [Bootstrap instruments § Partly-accrued periods](docs/guide/curves.md#partly-accrued-periods).

### Added — Averaged money-market futures

`MoneyMarketFuture` had one forward: the compounded, two-discount-factor one.
Three-month RFR futures do settle on that, but one-month RFR futures settle on
the *arithmetic average* of the daily index values, which is a different
functional of the same curve.

- **New static field** `averaging` ∈ `("compounded", "arithmetic")`,
  defaulting to `"compounded"`; `AVERAGING_CONVENTIONS` is exported from
  `valax.curves`. The compounded branch is bit-for-bit what it always was.
- **The two are a product and a sum of the same daily ratios.** With
  `ρ_j = DF(t_j)/DF(t_{j+1})`, compounded is `(Π ρ_j - 1)/τ` and averaged is
  `Σ(ρ_j - 1)/τ`. The day-count-weighted average collapses to that sum
  because `r_j δ_j = ρ_j - 1` by definition, and under `act_*` the daily
  accrual fractions sum to the period's exactly.
- **The gap does not vanish on a flat curve.** It is the cross terms — the
  interest-on-interest averaging discards — of order `r²τ/2`: measured 0.628
  bp on a 1M contract at 4% against 0.667 bp predicted, and 1.93 bp on a 3M.
  So reading a 1M contract as compounded mis-places the short end by a real
  amount rather than by something that averages out.
- **Only the averaged contract has exposure inside its own period.** The
  compounded forward telescopes, so interior pillars have *identically* zero
  sensitivity; the sum does not telescope, so the averaged forward picks up
  genuine sensitivity to the shape of the curve between its own start and end
  dates. That shape exposure vanishes on a flat curve, where adjacent `ρ_j`
  cancel pairwise — which is why the gate checks it on a sloped one.
- **No calendars required.** The market carries the preceding business day's
  rate across weekends weighted by *calendar* days, which is what a
  calendar-daily subdivision of a smooth curve reproduces. A concrete day span
  *is* required; traced dates raise a `ValueError` naming the way out instead
  of averaging over the wrong number of days.
- Narrows spec requirement C-1. See
  [Bootstrap instruments § Compounded vs averaged futures](docs/guide/curves.md#compounded-vs-averaged-futures).

### Added — Payment lag: accrual end and payment date are now separate roles

A coupon has two dates that matter, and until now VALAX's curve quote types
carried only one. `_floating_leg_pv` used a single array as both the accrual
end (what the coupon is worth) and the payment date (when you get it, and
therefore how it is discounted), which is correct only when the two coincide.
Real overnight-index swaps pay one or two business days after accrual ends,
so every OIS-bootstrapped discount factor was very slightly misplaced.

- **New fields.** `payment_dates` and `payment_lag` on `SwapRate` and
  `OISSwapRate`; `fixed_payment_dates` / `float_payment_dates` on
  `IborSwapRate`; `leg_a_payment_dates` / `leg_b_payment_dates` on
  `TenorBasisSwap`; `dom_payment_dates` / `for_payment_dates` on
  `CrossCurrencyBasisSwap`. `payment_lag` is a uniform shift in **calendar**
  days; an explicit array is the route for a business-day-adjusted schedule,
  since VALAX still has no holiday calendars. Supplying both for one leg
  raises, as does a negative lag or a schedule of the wrong length.
- **OIS floating leg generalised.** The telescoping identity
  `DF(T_0) - DF(T_n)` holds only at zero lag. The leg is now the per-period
  sum `Σ_i P_d(0,T_i^p)(P_d(0,T_i^s)/P_d(0,T_i^e) - 1)`
  (`_compounded_float_leg_pv`), with the closed telescope retained as an
  explicit fast path when no lag is configured.
- **Numbers are unchanged by default.** With neither field set the fast path
  is taken and every pre-existing curve, residual and bootstrap reproduces bit
  for bit. The two forms agree to 8e-17 — pure summation order.
- **Under the constant-notional cross-currency variant** the final notional
  re-exchange is paid with the last coupon, so it now discounts at the last
  payment date rather than the last accrual end.
- **Direction of the effect.** A uniform lag always *lowers* the par rate,
  whatever the shape of the curve, and by exactly nothing on a flat one: the
  par-rate weights pick up a factor `P(0,T^p)/P(0,T^e) ≈ exp(-f_i ℓ)` that
  decreases in that period's own forward, so the lag shifts weight away from
  the high-forward periods. The magnitude tracks the *dispersion* of the
  forwards, not their level.
- **Still neglected,** and now recorded as such rather than implied by the
  missing feature: the payment-lag *timing convexity*. The discount factor
  moves to `T^p` but the compounded coupon gets no drift correction for
  ceasing to be a martingale under the `T^p`-forward measure.
- Closes spec requirement N-3 and the T-5 gate; narrows C-1. See
  [Bootstrap instruments § Payment lag](docs/guide/curves.md#payment-lag-accrual-end-vs-payment-date).

### Changed (**breaking**) — Vol sources are now handed the live forward

A smile is a function of moneyness, so *which* forward it is anchored to decides
whether a curve move reprices the smile or merely slides along it. Until now a
vol source was never told the forward the pricer was about to use: it was queried
at `src(strike, expiry[, tenor])` and (for the SABR objects) read a forward frozen
at calibration time, while the pricer computed the swap rate and annuity from the
live curve. Vol and price could therefore see different forwards, and
`∂σ/∂θ_curve` was structurally zero.

- **Protocol.** A vol source is now called as
  `src(strike, expiry[, tenor], forward=F)`, with `forward` keyword-only and `F`
  the live curve-derived forward for that very instrument (forward par swap rate
  for a swaption, projected period forward for an optionlet). Every source must
  accept the keyword; flat sources ignore it. Affects
  `valax.pricing.analytic.rates_smile` (`swaption_price`, `caplet_price`,
  `cap_price`), `cms_convexity`, and the `cms_cap_floor` / `range_accrual`
  pricers in `rates_exotics`.
- **Numbers are unchanged.** `SwaptionCube.__call__` and
  `OptionletVolSurface.__call__` accept `forward` and currently *ignore* it —
  the sticky-absolute-strike convention — so every existing price is preserved
  bit-for-bit. What the convention is, and what the alternatives would do, is
  documented once in
  [Smile dynamics](docs/guide/vol-surfaces.md#smile-dynamics-what-moves-when-the-curve-moves).
- **Forward and annuity computed once.** New public observables
  `valax.pricing.analytic.swaptions.annuity` (the promotion of the former private
  `_annuity`; the old name remains as an alias),
  `swaption_forward_and_annuity`, `caplets.caplet_forward` and
  `caplets.cap_forwards`. `rates_smile.swaption_price` computes `(S, A_d)` once
  and prices through the new `swaption_black76_from_forward` /
  `swaption_bachelier_from_forward` cores; the caplet/cap pricers gained a
  `forward=` / `forwards=` keyword so a caller can hand back the forward it read
  its vol at. Vol and payoff now see the same forward *structurally*, not by
  coincidence. `surfaces.caplet_stripping.optionlet_forwards` delegates to
  `cap_forwards` instead of re-deriving the projection.
- **Migration.** Custom vol sources must add `forward=None` to their
  `__call__` signature (variadic sources: `def __call__(self, *coords,
  forward=None)`). Callers of the surfaces' `__call__` are unaffected —
  `forward` is optional.

### Added — Smile dynamics are now a declared choice, not an accident

Sticky-absolute-strike was previously the *only* expressible convention, and it
was arrived at structurally (the surface was never handed a forward) rather than
chosen. It is now named, validated and selectable.

- **`valax.surfaces.dynamics`** (new module) owns the convention mechanics:
  `SMILE_DYNAMICS`, `QUOTE_FORMS`, `CONSISTENT_DYNAMICS`,
  `validate_smile_dynamics`, `default_dynamics_for`, `resolve_smile_query`.
  One implementation, shared by both rates vol objects.
- **`SwaptionCube` and `OptionletVolSurface`** gained two static fields:
  - `dynamics` ∈ `{"sticky_strike", "sticky_moneyness", "sabr_backbone"}`,
    default `"sticky_strike"` — so every existing price is unchanged.
    `"sticky_strike"` evaluates Hagan at the fitted $F_0$;
    `"sticky_moneyness"` at $(K - (F - F_0);\, F_0)$; `"sabr_backbone"` at the
    live $F$ with the parameters held fixed.
  - `quote_form` ∈ `{"absolute_strike", "atm_offset"}` (spec Q-3).
  The pair is **validated at construction**: the dynamics is a consequence of
  what the quote holds fixed, so absolute-strike + sticky-moneyness (and
  atm-offset + sticky-strike) are configuration errors, not options. A
  forward-dependent dynamics queried with no forward raises rather than
  silently falling back.
- **`forwards` is now documented as $F_0$** — "the forward this node was fitted
  at" — on both objects and on `forward_at`, with the live forward supplied at
  query time.
- Under `"sticky_strike"`, `∂σ/∂θ_curve` is **exactly** zero and a new test
  asserts it; under `"sabr_backbone"` it is non-zero. Both are now facts of
  record rather than side effects.

### Added — Curve-aware swaption cubes, premium quotes, and vega in quote space

Closes the load-bearing gap in the adjoint chain. `calibrate_swaption_cube`
takes a caller-supplied `forwards` array, and a forward handed in from outside
is a constant as far as the fit is concerned — so `∂θ_v/∂θ_c` was *identically
zero* and no curve risk could reach the volatility layer. That blocked three of
the six paths by which curves enter model calibration, and a missing
`dR_v/dθ_c` is the single most common cause of wrong deltas.

- **`calibrate_swaption_cube_from_curves`** / **`_from_graph`** take the node
  *schedules* — one representative `Swaption` per `(expiry, tenor)` — plus a
  discount/projection pair, and derive everything else:
  - `swaption_node_coordinates` builds the grid axes from the same expressions
    `swaption_price` evaluates, so a node reprices at *its own* grid point
    rather than a hair off it;
  - `swaption_node_forwards_and_annuities` computes each node's `(S_x, A_d)`
    through the pricer's own `swaption_forward_and_annuity`.
  `quote_form="atm_offset"` converts offsets to absolute strikes per node using
  that node's derived `F_0` (spec Q-3), and `dynamics` then defaults to the
  convention that quote form implies. `calibrate_swaption_cube` survives as the
  explicit opt-out. T-1 gate: every fitted quote reprices to `< 1e-8` in vol
  units, both quote forms.
- **`valax.surfaces.swaption_premium`** (new module) converts premium quotes:
  `swaption_implied_normal_vol` solves
  `N·𝒩(θ_c)·Bachelier(S_x(θ_c), K, σ_N, T) − q = 0` by scalar
  `optimistix.Newton` with implicit differentiation, so the *implied vol*
  carries curve risk. `settlement="physical"` uses the annuity `A_d`;
  `"cash"` uses `P_d(0,T^p)·A_cash(S_x)` via the newly public
  `valax.pricing.analytic.cash_settled_annuity` (the promotion of
  `cms_convexity._standard_annuity`, now shared by Hagan and the cash branch).
  The two settlements imply different vols from the same premium. Cash-settled
  *convexity* is an explicitly documented deviation. The initial guess inverts
  the **straddle** asymptote rather than the at-the-money one, which is a
  guaranteed upper bound on `σ√T` and stays conditioned far from the money
  where the naive seed collapses into underflowed vega.
- **`valax.greeks.vol_adjoint`** (new module) adds λ_v. A smile fit is
  over-determined, so the square system the implicit function theorem is applied
  to is the fit's **first-order condition** — the stationarity of the weighted
  SSE. `sabr_slice_level` states it; `vol_quote_vegas` returns both input
  blocks: vega to the **quoted vols**, and `forward_bar = ∂V/∂F_0`, the
  `dR_v/dθ_c` term (spec A-3). `cube_node_level` / `optionlet_slice_level` /
  `cube_node_cotangent` wire it to the surface objects. Differentiating the SSE
  *gradient* gives the **full** Hessian, not Gauss-Newton — the documented
  resolution of spec R-5, chosen because the two coincide on a tight fit and so
  the approximation only matters when the fit is loose.
- **Bug fix — `bilinear_2d` returned NaN on a length-1 axis.** It divided by
  `grid[0] - grid[0]`, so every query against a single-expiry or single-tenor
  cube (and any `GridVolSurface` with one strike or one expiry) came back NaN. A
  degenerate axis is now flat along that direction. Surfaced by calibrating a
  one-node cube, which is the smallest curve-aware fit and a legitimate thing to
  do.
- **Verification.** Both input blocks are gated against full-rebuild central
  finite differences at two bump sizes, under both quote forms. The
  `optimistix` `ImplicitAdjoint` comparison is kept as a *consistency* check
  and labelled as such — it shares `sabr_implied_vol` with the adjoint under
  test, so it is not independent evidence.

### Fixed — `sabr_implied_vol` lost its ATM forward derivative

Hagan's expansions contain `z / x(z)`, which is `0/0` at the money. It was
guarded by pinning the ratio to the constant `1.0` — the correct *value*, and
zero *slope*, so automatic differentiation discarded the whole `z/x`
contribution. Since `z ∝ log(F/K)`, that corrupted `∂σ/∂F` at the
at-the-money strike by **18%** in the lognormal branch and to **identically
zero** in the normal branch. The `α`, `ρ`, `ν` derivatives were always correct
(at `F = K` their `∂z/∂·` terms carry a vanishing `log(F/K)` factor), and
**no price was ever affected** — which is why it survived.

- `_z_over_x` now hands over to the Taylor series
  `1 − (ρ/2)z + (1/6 − ρ²/4)z² + O(z³)` instead of to a constant, so both
  branches are smooth and carry the right slope. Extracted once and shared by
  `sabr_implied_vol` and `sabr_normal_implied_vol`; the outer double-`jnp.where`
  is retained, since that part genuinely prevents NaN gradients leaking from the
  inactive branch.
- The crossover moved from `1e-7` to `_ATM_Z_TOL = 1e-4`, chosen by balancing
  series truncation `O(z³)` against the closed form's cancellation `O(ε/z)`:
  `z³ ~ ε/z` gives `z ~ ε^(1/4)`, where both branches are accurate to ~1e-12.
  The old `1e-7` was *too small* — it kept the ill-conditioned closed form in
  play across a decade where the series was already better.
- At-the-money values are **bit-identical** before and after (the series is
  exactly `1` at `z = 0`).
- **Who was affected:** `dynamics="sabr_backbone"` deltas at ATM, and the
  `forward_bar` leg of `vol_quote_vegas` whenever an ATM strike was in the fit.
  Both shipped in this same release cycle; no tagged release contained the
  corrupted adjoint.
- Regression: `tests/test_models/test_sabr.py::TestATMDerivatives` checks AD
  against finite differences *at* the money for both expansions, asserts the ATM
  forward derivative is non-zero, and checks the two branches agree at the
  crossover with no jump or kink.

Found while building the λ_v gate, and the way it hid is the point: a
full-rebuild finite difference disagreed and was *dismissed* because a second
"independent" adjoint — `optimistix`'s `ImplicitAdjoint`, different code,
different parametrisation — agreed with the first bit-for-bit. It shared
`sabr_implied_vol` underneath, so it shared the defect: two votes that were
really one. An ill-conditioned least-squares Hessian (κ ≈ 2e6) then supplied a
plausible but false mechanism for distrusting the finite difference. Written up
as entry 5 of `docs/architecture/numerical-pitfalls.md`, with a runnable
reproduction in `examples/pitfalls/02_where_guard_kills_the_derivative.py`.

### Added — Quote-space (hedge-space) adjoint risk

Closes the `J_build` leg of the Jacobian stack. `curve_pillar_deltas` already
answered "how does the price move if a pillar discount factor moves", but a desk
hedges in instruments, not pillars — and the two are not a rescaling of each
other. Holding the other *quotes* fixed forces the whole curve graph to
re-solve, so an OIS quote picks up its indirect effect on every projection curve
calibrated against it; a bump that held projection *nodes* fixed misses that
term entirely. On a dual-curve book the two conventions differ in magnitude
*and* sign.

- **`valax.greeks.adjoint`** (new module).
  - `curve_adjoint(graph, ref, specs, instruments)` builds a per-snapshot
    handle: the residual Jacobian `∂R_c/∂θ_c` at the solution, its condition
    number, and a `lineax.LU` factorisation computed **once**.
  - `curve_quote_deltas(instrument, model, market, adjoint=...)` returns
    `∂price/∂quote` for every bootstrap instrument via the implicit function
    theorem: one reverse sweep through the pricer, one transposed solve, one
    reverse sweep through the residual. The quote Jacobian is never formed and
    the Newton iterations are never taped, so cost is flat in the number of
    quotes and each additional trade is a back-substitution.
  - `CalibrationLevel` / `level_adjoint` / `LevelAdjoint` generalise the same
    step to any calibration written as `R(θ; inputs) = 0` — the hook the
    volatility and model layers will plug into. The generic path reproduces the
    curve-specialised one bit-for-bit.
  - `curve_log_df_cotangent` handles the change of variables from
    `∂V/∂DF` (all pillars) to the solver's `log(DF)` space (reference pillar
    excluded), so callers no longer hand-slice `[1:]`.
- **Residual single-sourcing.** `valax.curves.curve_residual_fn` exposes the
  residual as `R(θ_c, q_c)`, and both it and `bootstrap_curve_graph` now call
  one extracted `_graph_residual`. The solved system and the differentiated
  system are the same code path by construction, not by convention. Also added:
  `curve_quote_vector`, `log_dfs_from_graph`.
- **Mixed residual units handled by construction.** VALAX residuals are
  deliberately not uniform — a futures residual is a rate difference, a swap
  residual is `rate × annuity − float_pv`. The adjoint differentiates `∂R/∂q`
  rather than assuming `−I`, so both scale correctly with no per-instrument
  special-casing.
- **Conformance gates.** New `audit` pytest marker (`pytest -m audit`) and
  `tests/test_greeks/test_quote_adjoint.py`: AAD vs full-rebuild central finite
  differences for every quote at 0.1 bp and 1 bp (single- and dual-curve), the
  parallel-shift identity, and the quotes-fixed vs nodes-fixed distinction.
  Observed agreement 1e-9 to 1e-11 relative.
- **Docs.** New
  [Audit Conformance Map](docs/architecture/aad-audit-conformance.md) scoring
  VALAX requirement-by-requirement against the multi-curve / AAD risk audit
  specification, including the identifier map, the instrument→residual
  inventory, committed tolerances and the explicitly documented (and explicitly
  *undecided*) convention choices.

### Added — Dual-curve rates calibration (OIS discounting + tenor projection)

Closes the consistency hole between pricing and calibration: since P5.1 the
MC/PDE short-rate engines priced dual-curve, but *every* rates calibrator fitted
single-curve, so the tenor basis leaked into the fitted volatility. Measured on
a 50 bp basis, the old single-curve residual mis-fits Hull-White `σ` by **62 %**
(0.016661 vs 0.010261) — a model that reprices its own calibration set and is
wrong everywhere else. This violated the model/market consistency invariant that
`price(instrument, model, market)` documents.

- **Analytic dual-curve swaptions (the prerequisite).** Because the basis is
  deterministic, a dual-curve swap stays an affine combination of discount
  bonds, so both closed forms extend without losing their closed form:
  `hw_swaption_price(swaption, model, forward_curve=None)` — Jamshidian's
  critical rate now solves `CB(r*) = float_start` rather than `= 1`, via a new
  `par_value` argument on `hw_critical_rate`; and
  `g2pp_swaption_price(..., forward_curve=...)`, which previously **raised
  `NotImplementedError`**, now solves its `ȳ(x)` boundary to `float_start` and
  scales the eq.-(4.31) strike term by it. Both reuse
  `valax.pricing.multicurve.{swap_leg_basis,apply_swap_basis}`, so there remains
  exactly one implementation of the basis algebra shared with the MC/PDE
  recipes.
- **Calibration.** `calibrate_hull_white` and `calibrate_g2pp` take
  `forward_curve=` (projection) and `curve_id=` (records the discount curve's
  `MarketState` identifier on the fitted model, so `with_market_curve` can
  re-bind it under a shocked market). `curve` remains the discount curve the
  model is exact-fitted to. `swaption_prices_from_vols` takes `forward_curve=`
  too — both legs of the residual must see both curves, since the same quoted
  vol implies a different price once the forward swap rate is projected off a
  different curve.
- **Honest refusal where the engine cannot comply.** `calibrate_cheyette` and
  `calibrate_cheyette_2f` accept `forward_curve` only to **raise** via
  `require_single_curve`: the Cheyette PDE has no discount/projection split, and
  a silently single-curve fit would be a wrong calibration with no signal.
- **Fixed — `cap_price_black76` silently ignored `forward_curve`.** It accepted
  the argument (its Bachelier sibling and both caplet pricers handle it) but
  projected off the discount curve unconditionally, returning a single-curve
  price for a dual-curve request. Regression-tested against the caplet strip.
- **Single-curve is unchanged, bit-for-bit.** The basis is a *trace-time*
  branch (`None` ⇒ `float_start = 1.0` and untouched cash flows), verified by
  exact equality against the pre-change implementations across 18 payer/receiver
  × strike × expiry/tenor configurations per model — not a tolerance. Multi-curve
  is an upgrade, not a replacement.
- **Validation** (`tests/test_pricing/test_dual_curve_swaptions.py`,
  `tests/test_calibration/test_dual_curve_calibration.py`): payer-minus-receiver
  parity equals the dual-curve forward swap value to ~2e-14 (model-independent,
  so it isolates the discount/projection split); the closed forms agree with the
  independent dual-curve PDE recipes to 1.5e-5 (HW) and 5.6e-5 (G2++); prices
  are monotone in the basis and collapse exactly onto single-curve at zero
  basis; dual-curve round-trips recover the generating parameters; and a
  single-curve fit of dual-curve targets is asserted to *miss*, pinning the bug
  this closes. Gradients flow to the projection curve's pillars, so basis risk
  is autodiff rather than bumped.
- Docs: new [calibration guide § 6](docs/guide/calibration.md), theory § 5.1 on
  why the closed forms extend cheaply, coverage-matrix and roadmap updates, and
  a roadmap item for the missing **QuantLib** cross-check of the rates
  calibrators (none exists in either mode; current gates prove self-consistency,
  not convention).

### Changed — Pricing ontology: tutorial in the docs, ledger out of them

- **New page `docs/guide/tutorial-unified-pricing.md`** (nav: *Examples →
  Tutorial — The Unified Pricing Contract*) — the user-facing contract, taught
  rather than specified: the four arguments and what each owns, one trade
  through all four engines, method selection, `model=None` curve discounting,
  `discount_id` / `forward_id` selectors turning a single-curve trade
  multi-curve, keyed multi-asset markets and the surface-id alphabet,
  `from_market` / `with_market_curve` model binding, the three sensitivity
  families (including the theta autodiff-vs-bump policy and pillars → buckets →
  quote-space), `register_pricer`, and a table of the failure modes with their
  actual messages. Every snippet was executed in document order; the numbers in
  the prose are its real output.
- **`docs/architecture/pricing-ontology.md` → `PRICING_ONTOLOGY.md`** (repo
  root, outside the MkDocs build). The migration report was half contract, half
  phase-by-phase ledger; the contract half is now the tutorial, and what remains
  is a contributor artefact — so it reads as one, led by a new **§0 Outstanding
  work** backlog consolidating the old §12.8 / §13 / §14.2 items into a single
  checklist.
- Inbound references rewired: `mkdocs.yml` nav, `docs/api/core.md` (which now
  defers the how-to material to the tutorial instead of restating it),
  `docs/api/greeks.md`, `docs/roadmap.md`, `docs/theory/multi-curve-short-rate.md`,
  and the docstring path references in `valax/core/`, `valax/greeks/`,
  `valax/pricing/multicurve.py`. No code behaviour changed.

### Added — Unified pricing signature and `MarketState` (pricing ontology, P2)

- `valax/core/market_state.py`: **`MarketState`**, the calibrated market
  container — a keyed `CurveGraph`, identifier-keyed vol surfaces, spots,
  dividends, fixings, and the credit / inflation / FX corners in one
  differentiable pytree. It *generalises* `MarketData` (one curve → keyed
  graph; one scalar vol per asset → full surfaces) rather than renaming it:
  `from_market_data` / `to_market_data` are a lossless two-way view, so the
  `risk/var.py` scenario framework is untouched for one deprecation cycle.
- `valax/core/pricing.py`: **`price(instrument, model, market, *, method,
  config, key) -> PricingResult`** — one call convention replacing three
  (loose analytic scalars, `(instrument, model, config, **market_args)`,
  and `(instrument, market)`). **No pricer was rewritten.** Analytic and
  lattice pricers are reached through thin adapters
  (`valax/core/adapters.py`); the MC and PDE recipe registries are *bridged*,
  not duplicated, so every recipe registered there is reachable through the
  unified signature immediately and the registries cannot drift apart.
- `method="auto"` keeps *availability* (the registry) separate from
  *preference* (a policy table), falling back to `analytic → lattice → pde →
  mc`. A subclass that only adds market selectors inherits its base class's
  pricers (MRO resolution), which is the seam P3 will use.
- `valax/core/resolve.py`: **`resolve`** binds an instrument's contractual
  selectors (discount curve, projection curve, underlying) and a model's
  surface reference against the market. Selectors are static strings with
  sentinel defaults, so **single-curve pricing needs zero configuration** and
  `forward_id=None` keeps the existing "project off the discount curve"
  semantics bit-for-bit.
- `model=None` is a first-class case for instruments priced by curve
  discounting alone (fixed-rate bonds, vanilla and OIS swaps, FRNs).
- Gates met (`tests/test_core/`): single-curve parity with the legacy calls to
  ≤ 1e-10 with no selector configuration; MC/PDE prices bit-identical to the
  dispatchers; `registered_pricers()` covers every pair in both legacy
  registries; `jax.grad` through `price` yields market delta and model vega
  together.
- Design record: the five open questions in
  `PRICING_ONTOLOGY.md` are now resolved (one
  `PricingResult` container; a separate auto-policy table; the surface-id
  alphabet `<UNDERLYING>.<KIND>[.<QUALIFIER>]`; a one-release `MarketData`
  view; `theta` + `dual_delta` promoted). New page: `docs/api/core.md`.

### Changed — MC-Curves-2b: `bootstrap_multi_curve` runs on the joint solver

- `valax/curves/multi_curve.py` no longer contains a solver. The deprecated
  two-tier entry point translates its input into `CurveSpec`s plus graph
  instruments and makes a single `bootstrap_curve_graph` call; the sequential
  OIS-then-tenor pipeline (`_bootstrap_forward_curve`, `_SwapData`,
  `_pack_swap_data`, `_dual_curve_swap_residuals`) is deleted.
- The semantic question that deferred this work is resolved by *translation*:
  a forward-leg `SwapRate` — which the sequential code read as "project on the
  tenor curve, discount on OIS" — becomes an `IborSwapRate` with
  `float_dates == fixed_dates` and fixing dates at the period starts, which is
  exactly the residual the old code computed. Discount-leg and forward-leg
  deposit/FRA quotes keep their single-curve residual.
- `MultiCurveSet` is now a **view** over `CurveGraph`: `from_graph` /
  `to_graph` convert between the shapes without copying or re-solving, so
  legacy containers can use the graph-aware pricers and shock primitives.
- New behaviour the sequential path never had: one Newton solve for every
  curve, and `jax.grad` through the build via `ImplicitAdjoint` — including
  w.r.t. swap-leg quotes. Curves match an explicit joint-solver build to
  ≤ 1e-12 (`tests/test_curves/test_multi_curve.py`); all pre-existing
  multi-curve tests pass unchanged.

### Added — Hull-White short-rate PDE: Bermudan swaptions and callables (roadmap PR-3)

- `valax/pricing/pde/hull_white.py`: finite-difference pricing under
  `HullWhiteModel`, registered on `pde_price_dispatch` for `FixedRateBond`,
  `CallableBond`, `PuttableBond`, `Swaption` and **`BermudanSwaption`**. This is
  the third independent numerical route to the model, joining the trinomial tree
  and the exact-OU Monte Carlo.
- **Bermudan swaptions no longer need a regression proxy.** On a backward grid
  early exercise is a pointwise projection on the value function, and under
  Hull-White the tail-swap exercise value is analytic at every node — so the
  policy carries no error at all and the only numerical error is in the
  continuation value. Previously these were reachable only via LSM on LMM paths.
- Solved in the centred state \(x\) of \(r(t) = x(t) + \alpha(t)\): drift and
  diffusion are then time-independent, the state starts at the origin, and the
  read-off point is fixed (so, unlike the equity recipes, no `stop_gradient`
  grid scaffolding is required).
- Fully traceable: schedules are scattered with *traced* indices, so unlike the
  trinomial tree — whose half-width \(j_{\max}\) is an array shape and forces
  `mean_reversion` to be concrete — the PDE pricers take no concrete inputs and
  compose with `eqx.filter_jit` / `eqx.filter_grad` directly. Sensitivities to
  \(a\) and \(\sigma\) match central differences to `1e-8`–`1e-11`, including
  through the exercise projection.
- Validation: analytic curve price for option-free bonds (`8e-6`, cleanly second
  order); Jamshidian for European swaptions (`~3e-5` relative, VALAX and
  QuantLib); `ql.FdHullWhiteSwaptionEngine` for Bermudans (`~2e-4`) and
  `ql.TreeSwaptionEngine` (`~1.2e-3`); the HW tree and
  `ql.TreeCallableFixedRateBondEngine` for callables/puttables (`< 5e-3`).
  Two exact structural checks complement these: an out-of-the-money call
  reproduces the bullet bond to machine precision, and a single-exercise
  Bermudan reproduces the European.
- New substrate, all model-agnostic and reusable (G2++ will want the same
  pieces): `pde/grids.centred_state_grid`,
  `pde/boundary.apply_linearity_bc_1d` (zero-curvature ghost fold — a callable
  bond has no closed-form far-field value to pin with Dirichlet data) and
  `pde/boundary.zero_boundary`, `pde/coefficients.hw_operator_stack`, and an
  `event_fn` seam on `solve_backward_1d` for discrete coupons and exercise.
- `valax/models/hull_white.py`: `hw_alpha` (the exact-fit shift, now public and
  shared with the MC path generator instead of duplicated) and
  `hw_alpha_average`.
- Theory: `docs/theory/hull-white-pde.md`.

### Fixed — cashflow-timing error in short-rate backward induction

- Snapping a coupon to the nearest time level displaces it by up to `dt/2`, an
  `O(dt)` error that dominated everything else — `~2e-3` on a five-year bullet,
  three orders of magnitude worse than the scheme's own error, and enough to
  mask its second-order convergence entirely. The PDE recipes instead scale each
  coupon by the analytic Hull-White bond price `P(t_k, t_c | x)` from its
  snapped level to its true payment date: exact at every node, for either sign
  of `t_c - t_k`. The trinomial tree still snaps, and is correspondingly the
  less accurate engine (`-2.2e-3` to `+3.4e-5` on an option-free bond, depending
  on the step count).
- `hw_alpha` must be **integrated** across each time step, not sampled at its
  midpoint. A log-linear discount curve has a piecewise-constant instantaneous
  forward that jumps at every pillar, so midpoint sampling is only first-order
  on steps straddling a pillar. The symptom was a scheme that converged cleanly
  at second order on a flat curve but *stalled* at a `~4e-6` bond-repricing
  error on a sloped one — a violation of Hull-White's defining exact fit.
  `hw_alpha_average` integrates both halves in closed form (the market-forward
  half telescopes into a discount ratio), restoring second order on flat,
  sloped and humped curves alike.

### Fixed — time-reversed boundary sampling in the PDE steppers

- `solve_backward_1d` and `solve_backward_2d` sampled the Dirichlet boundary
  callables at `(n_time - m) * dt` instead of `m * dt`. Since `lax.scan` marches
  backward from the terminal payoff, the two are exactly mirrored, so every
  time-dependent boundary was evaluated with the wrong discount factor — the
  far-field value was off by `K(1 - exp(-rT))` and the residual leaked inward.
- **This changes prices**, most visibly on narrow grids. For an ATM
  Black-Scholes European call (`n_spot = n_time = 400`) the error against the
  analytic price was `1.4e-1` at `spot_range=2`, `1.4e-2` at `3` and `4.5e-4` at
  the default `4`; after the fix all widths sit at `~1e-5`, i.e. pure
  discretisation error. The default `spot_range=4.0` masked it, which is why it
  survived the QuantLib comparison suite.
- Guarded by two new regression tests that fail on the old code: a
  spatially-constant probe in `tests/test_pde/test_schemes.py` (the solution
  must stay constant in `x` to machine precision, which only holds if the edge
  ghosts are taken at the interior's time level), and a domain-width
  independence check in `tests/test_pde/test_crank_nicolson.py`.

### Added — Hull-White swaptions (Jamshidian) and swaption-surface calibration

- `valax/pricing/analytic/hull_white_swaptions.py`: `hw_swaption_price` prices
  European payer/receiver swaptions in closed form. Because every bond price at
  expiry is monotone in the single state variable, the option on the coupon
  bond decomposes *exactly* into a portfolio of zero-coupon bond options — no
  lattice, no simulation, no numerical integration. Also exposes
  `hw_critical_rate` and `hw_zcb_option_price`.
- The critical rate is an `optimistix` Newton solve, so it is implicitly
  differentiable: `jax.grad` applies the implicit function theorem to the
  converged root rather than unrolling iterations. Autodiff sensitivities match
  central differences to `~1e-11`.
- `valax/calibration/hull_white.py`: `calibrate_hull_white` fits \((a, \sigma)\)
  to a swaption surface, with `fixed_mean_reversion` for the usual desk
  workflow of pinning `a` and fitting `sigma`. `swaption_prices_from_vols`
  converts Black-76/Bachelier quotes to the price targets.
- `hw_market_df` / `hw_instantaneous_forward` are now public accessors on
  `valax/models/hull_white.py` (previously private, but needed across modules).
- Validation: `< 1e-4` relative vs `ql.JamshidianSwaptionEngine`, and `< 5e-3`
  vs `ql.TreeSwaptionEngine` — the latter an *independent* numerical method.
  Synthetic round-trip recovers the generating \((a, \sigma)\) to `1e-5`.
- Theory: `docs/theory/hull-white-swaptions.md`.

### Added — Hull-White short-rate Monte Carlo (roadmap MC-B1)

- `valax/pricing/mc/hull_white_paths.py`: `generate_hull_white_paths` and
  `HullWhitePathResult`. Samples the **exact** conditional Ornstein-Uhlenbeck
  law rather than discretising the SDE, so `r(t)` is unbiased at any step size;
  the only discretisation error is the trapezoidal accumulation of
  \(\int r\,dt\) into the log discount factor. Derivation in
  `docs/theory/hull-white-mc.md`.
- Four MC dispatcher recipes registered against `HullWhiteModel`:
  `FixedRateBond`, `FloatingRateBond`, `CallableBond`, `PuttableBond`.
  Callable/puttable use the analytic Hull-White affine continuation value at
  each exercise date, which is exact for the bullet remainder and so needs no
  Longstaff-Schwartz regression basis. Exercise indicators are sigmoid-smoothed
  so pathwise Greeks stay well-defined.
- `hw_tree_j_max` exposes the Hull & White (1994) lattice half-width so callers
  can hoist the shape-determining `j_max` out of the trace and differentiate
  through the trinomial tree with `eqx.filter_grad`.
- `tests/test_quantlib_comparison/test_rates_pricers_ql.py`: QuantLib
  cross-validation for swaptions (Black-76 + Bachelier, payer and receiver),
  caps/floors, and the Hull-White callable/puttable tree.

### Fixed — Callable/puttable bonds exercised one coupon too cheaply

- `valax/pricing/lattice/hull_white_tree.py` compared the **cum-coupon**
  continuation value against the call/put price. Call prices are quoted
  *ex-coupon*: a holder called on a coupon date receives the redemption amount
  *plus* that date's coupon. The exercise decision now precedes the coupon
  accrual in both `callable_bond_price` and `puttable_bond_price`.
- Impact: callable bonds were undervalued by up to one coupon (2.16 price
  points on a 5 % 5Y par-callable). The error was invisible because the
  QuantLib tree comparison had never executed — its fixture raised a
  constructor `TypeError` under QuantLib >= 1.35. With both fixed, VALAX now
  matches `ql.TreeCallableFixedRateBondEngine` to `< 5e-3` relative, and MC
  agrees with the tree to within MC error on a single-call bond.

### Fixed — Hull-White forwards collapsed to zero before the first curve pillar

- `_instantaneous_forward` differentiates a `jnp.interp` of log discount
  factors. `DiscountCurve` extrapolates flat, so a curve whose first pillar sat
  strictly after the reference date had zero log-DF slope on the leading stub
  and returned `f(0, t) = 0` there. `_log_df_grid` now anchors the
  interpolation at the no-arbitrage point \(\ln P(0,0) = 0\); curves that
  already carry a `t = 0` pillar are unaffected.
- Two call sites had been working around this by injecting a `t = 0` pillar.

### Fixed — PDE second-order spot Greek (gamma) via curvature-carrying read-off

- `valax/pricing/pde/grids.py::read_off_1d` now uses a **Catmull-Rom cubic**
  (C¹ Hermite on the uniform grid, flat extrapolation preserved) instead of
  piecewise-linear `jnp.interp`. Linear interpolation has no within-cell
  curvature, so `jax.grad(jax.grad(pde_price))` collapsed gamma to `~0`. The
  cubic carries genuine curvature and also improves price/delta accuracy.
- `uniform_log_spot_grid` builds its nodes on `lax.stop_gradient(spot)`. The
  grid still centres on spot (unchanged forward price) but no longer *co-moves*
  with it under autodiff — previously the query `ln(spot)` and every node
  translated together, freezing the read-off at a fixed fractional cell
  position and making `V(S)` piecewise-linear in `S` (exact delta, zero
  pointwise gamma). Detaching leaves the query as the sole differentiable spot
  dependence; `vol`/`expiry` stay live so vega/theta are unaffected. European,
  American and digital recipes inherit this automatically.
- `_barrier_bs` gets the same treatment at the recipe level (barriers build
  their mesh with `uniform_linear_grid`, one edge pinned to the barrier): the
  grid bounds use `ln(stop_gradient(spot))` while the read-off query stays live.
  Barrier gamma now matches a QuantLib finite-difference reference and satisfies
  in/out gamma parity `Γ(KO) + Γ(KI) = Γ(vanilla)` (`~1e-5`).
- Net effect: gamma (and other second-order spot Greeks) now flow through the
  **unified** `greeks()` autodiff engine — no PDE-specific Greek path. ATM
  Black-Scholes cross-check at `n_spot=400`: gamma error `~1e-5` (was `≡0`),
  delta error `~7e-5`, price closer to analytic than before.
- Tests: `tests/test_pde/test_grids.py` (read-off unit + grid-detachment),
  `TestPDESecondOrderGreeks` in `test_crank_nicolson.py` (gamma vs BS across
  moneyness/expiry, call/put parity, convergence, `filter_jit` smoke),
  `TestBarrierGamma` in `test_barrier_pde.py` (finite/nonzero, sign, in/out
  gamma parity, `filter_jit`), the QuantLib cross-checks
  `test_pde_pr1_ql.py::{test_european_gamma_matches_quantlib,
  test_barrier_gamma_matches_quantlib}`, and `test_gamma_benchmark.py`
  (`--benchmark-only` throughput + convergence profile). Full suite green.

### Changed — Andersen QE Heston path generator (HE-1 closure)

- Replaced the diffrax Euler-with-reflection scheme in
  ``valax/pricing/mc/paths.py::generate_heston_paths`` with
  Andersen's (2008) Quadratic-Exponential algorithm, implemented
  directly on ``jax.lax.scan``. The variance is sampled by exact
  two-moment matching with a quadratic-vs-exponential switch at
  ``ψ_c = 1.5``; the log-spot uses Andersen's central discretisation
  with trapezoidal weights ``γ₁ = γ₂ = 0.5``. The function signature
  is unchanged, so all 16 MC recipes and downstream callers pick up
  the improvement transparently.
- Eliminates the ``O(1/sqrt(n_steps))`` bias the old scheme exhibited
  when the variance process spent time at the absorbing boundary —
  i.e. whenever Feller's condition ``2·κ·θ > ξ²`` was violated, which
  is the common case for single-expiry SABR-style Heston
  calibrations. ``TestHestonQLCalibratesVALAXReprices`` flipped from
  ``xfail @ 6 SE / n_steps=500`` to ``pass @ 3 SE / n_steps=100``.
- Unblocked the Stage-3 chain test
  ``tests/test_quantlib_comparison/test_exotic_on_heston_surface_ql.py``
  (previously ``pytest.mark.skip`` with HE-1 reason). It is now a
  shared-surface Heston Asian chain test: QL calibrates Heston to a
  SABR-generated smile, VALAX adopts the fitted parameters, both
  engines MC-price an arithmetic Asian call on the *same* calibrated
  surface, and the means agree to ``3 · combined_stderr`` across 3
  seeds × 3 moneynesses (9 tests, all green on first run).
- Net suite delta vs. Sprint 4: **2009 → 2021 passing**,
  ``xfailed`` 7 → 6, ``skipped`` 3 → 2, total runtime down ≈28%.

### Added — QuantLib Validation Pyramid (three-stage parametric sweep)

A three-stage validation campaign that converts the existing
fixed-scenario `tests/test_quantlib_comparison/` files into a
parametric sweep, then adds calibration-agreement and chain-validation
stages. Every assertion is driven by a synthetic-market sample
(``valax.market.sample_scalar_market`` / ``sample_nss_curve`` /
``sample_sabr_params`` / ``sample_heston_params``) under a per-test
``SeedRegistry``, so each test runs across N seeds rather than one
hardcoded case.

- **Shared adapter module** ``tests/test_quantlib_comparison/_ql_adapters.py``
  centralises every VALAX→QuantLib convention translation. Notably
  ``snap_expiry_to_days`` aligns the continuous year-fraction VALAX
  expiry to QuantLib's integer-day expiry — without this, even the
  trivial BS comparison fails at ``1e-10``.
- **Stage 1 — Pricer parametric sweep (836 tests).** All seven
  existing comparison files wrapped in 10–20-seed loops. Files
  touched: ``test_european_options_ql.py`` (140), ``test_pde_lattice_ql.py``
  (100), ``test_sabr_ql.py`` (26), ``test_fixed_income_ql.py`` (220),
  ``test_heston_ql.py`` (30), ``test_monte_carlo_ql.py`` (60),
  ``test_risk_greeks_ql.py`` (240).
- **Stage 2 — Calibration agreement (200 tests).** New:
  - ``test_curve_bootstrap_ql.py`` — VALAX ``bootstrap_sequential`` and
    QL ``PiecewiseLogLinearDiscount`` agree to ``abs<1e-10`` on
    discount factors at off-pillar dates, across 20 seeds × 8
    checked dates.
  - ``test_sabr_calibration_ql.py`` — fitted smiles agree to ``0.0 bp``
    (max disagreement empirically) on a dense extrapolated strike
    grid. Required QL flags: ``vegaWeighted=False`` and
    ``allowExtrapolation=True`` — convention drifts that the single-
    scenario tests had hidden.
  - ``test_heston_ql.py::TestHestonQLCalibratesVALAXReprices`` — QL
    calibrates Heston, VALAX MC reprices. Surfaces a real bias under
    Feller-violating calibrated parameters; flipped to ``xfail`` with
    a roadmap reference (**HE-1**) rather than fudged tolerance.
- **Stage 3 — Chain validation (60 tests + 2 placeholders).** New:
  - ``test_exotics_on_sabr_surface_ql.py`` — shared-surface chain
    pattern: QL calibrates SABR, VALAX adopts parameters, both
    engines BS-price European calls reading vol from the *same*
    surface. Vols agree to ``1e-12``, prices to ``1e-10``.
  - ``test_exotic_on_heston_surface_ql.py`` — skipped, blocked by
    HE-1. Re-enables once Andersen QE lands.
  - ``test_cap_strip_on_caplet_vols_ql.py`` — skipped, needs a
    ``build_sabr_caplet_surface`` convenience helper.
- **Persistent plan document**
  ``docs/architecture/quantlib-validation-pyramid.md`` — full
  three-stage plan with per-sprint session log, tolerance policy,
  triage rules, and acceptance criteria. Survives the session in
  which it was written.
- **mkdocs nav** updated with the plan doc under Architecture.
- **roadmap.md** updated:
  - New row in *Current State* for the validation pyramid.
  - New **HE-series backlog** with HE-1 (Andersen QE / full-
    truncation Heston discretisation) — the single new roadmap item
    surfaced by the sweep.
- **Test count**: **+1096 passing tests** (913 → 2009), one
  documented xfail (HE-1), two xpasses (HE-1 only fires on
  Feller-violating draws), two new documented skips (Stage 3
  placeholders).

### Added — Synthetic market data, reproducibility & arbitrage stress tests

- **`valax.market.synthetic`** subpackage: end-to-end generators that
  drive every component of the library without any external data
  source. Six-stage workflow (ground-truth world → noisy observations
  → calibration → portfolio → pricing → risk) with one module per
  stage:
  - `config.py` — `SyntheticMarketConfig` (`eqx.Module`, fully static).
  - `seeds.py` — `SeedRegistry(master_seed, library_version)` with
    deterministic `key(name, version)` derivation via SHA-256-stable
    folds. Replaces ad-hoc `jax.random.PRNGKey(42)` sprinkled across
    tests.
  - `curves.py` — `flat_discount_curve`, `sample_flat_curve`,
    `sample_nss_curve`, `sample_discount_curve`. Fills the gap that
    `valax.curves` has no flat-curve helper; NSS sampler clips DFs to
    `(0, 1]` for safety.
  - `correlations.py` — `sample_correlation`, `block_correlation`,
    `sample_correlation_from_config` with PSD reprojection.
  - `scalars.py` — `sample_scalar_market` (single-asset / single-option
    draw matching the inputs to `examples/comparisons/01_european_options.py`).
  - `snapshots.py` — `sample_market_data` and
    `sample_market_with_correlation` producing canonical `MarketData`.
  - `model_params.py` — ground-truth samplers for `BlackScholesModel`,
    `HestonModel` (Feller-respecting), `SABRModel`, `HullWhiteModel`,
    `MultiAssetGBMModel`. Used to test **calibrators**, not pricers.
  - `observations.py` — `synthesize_sabr_smile`,
    `synthesize_price_strip`, `synthesize_curve_quotes` (turn clean
    truth into noisy desk-style quotes).
  - `portfolio.py` — `sample_option_portfolio` (split call/put
    stacked-pytree returns to respect the static `is_call` field),
    `sample_swap_portfolio`.
  - `paths.py` — `evolve_market` producing a stacked `MarketData`
    tape via correlated GBM.
  - `scenarios.py` — `sample_scenario_set` matching the existing
    `ScenarioSet` shape.
  - `arbitrage.py` — eight injectors (`inject_non_psd_correlation`,
    `inject_butterfly_arb`, `inject_non_convex_smile`,
    `inject_calendar_arb`, `inject_pcp_violation`,
    `inject_negative_density`, `inject_inconsistent_bootstrap_quotes`,
    `inject_basket_variance_violation`) each returning
    `(perturbed, ArbDiagnosis)`.
- **Reserved exception types** in `valax/core/diagnostics.py`:
  `ArbitrageError` (base) plus `NonPSDCorrelationError`,
  `ButterflyArbError`, `CalendarArbError`, `PutCallParityError`,
  `NonConvexSmileError`, `InconsistentQuotesError`. Not raised by the
  library yet; the test suite's `xfail` set is the public backlog of
  detectors to add.
- **Golden-dataset harness** in `tests/golden/`: versioned `.npz`
  artifacts indexed by `tests/golden/golden_manifest.json`,
  `assert_matches_golden(name, value, *, version, rtol, atol)`
  helper, `REGEN_GOLDEN=1` regeneration switch, and
  `scripts/regen_goldens.py` entry point.
- **Test infrastructure**:
  - `tests/conftest.py` with `master_seed`, `seed_registry`,
    `default_synth_cfg`, `library_version` fixtures
    (`VALAX_MASTER_SEED` env var override).
  - 93 new tests under `tests/test_market/` — structural (curves,
    correlations, snapshots, paths, model params, observations,
    portfolio), arbitrage injection + detect-or-regularise handling,
    end-to-end non-tautological patterns, calibration-residual
    closed loops, and the golden harness itself.
  - 5 `@pytest.mark.xfail(strict=True)` items in
    `test_arbitrage_handling.py` are the public, machine-readable
    backlog of missing safety checks.
- **Closed-loop validation patterns** (deliberately
  non-tautological; replaces "calibrate X on data from X, recover
  X's parameters"):
  - Pricer × implied-vol round-trip.
  - Analytic vs Monte Carlo cross-engine consistency within 3·stderr.
  - Autodiff vs central-finite-difference Greeks.
  - NSS curve self-consistency at off-pillar dates.
  - SABR smile-residual within 1.5× injected observation noise.
  - Bootstrap non-self-roundtrip: NSS truth → quotes at *non-NSS*
    tenors → bootstrap → off-pillar zero-rate recovery within an
    interpolation tolerance.
- **Examples**:
  - `examples/07_synthetic_market.py` — snapshot, basket pricing,
    5-step market tape.
  - `examples/08_end_to_end_workflow.py` — full Stages 1 → 6
    walkthrough ending with a deliberate calendar-arb injection.
- **Public surface** re-exported from `valax.market` so user code
  reads `from valax.market import sample_market_data, SeedRegistry`
  without reaching into the submodule.
- **Pytest markers** registered: `arbitrage`, `golden`, `detects`.
- **Documentation**: new guide pages
  `docs/guide/synthetic_market.md` and
  `docs/guide/reproducibility_and_arbitrage_tests.md`; new API
  reference `docs/api/market.md`; mkdocs nav updated; org-mode
  mirrors regenerated via `scripts/md2org.sh`.

### Added — Multi-asset Monte Carlo

- **`MultiAssetGBMModel`** (`valax/models/multi_asset.py`): $N$-asset
  correlated GBM under a single risk-neutral measure. Carries
  per-asset `vols` and `dividends`, a scalar `rate`, and an
  $n \times n$ `correlation` matrix.
- **`validate_correlation(C, tol)`** helper: returns the minimum
  eigenvalue of a candidate correlation matrix (symmetric + unit-diag
  + PSD check) for pre-construction sanity.
- **`generate_correlated_gbm_paths`** (`valax/pricing/mc/multi_asset_paths.py`):
  exact log-Euler correlated GBM path generator using the Cholesky
  factor of the correlation matrix. No discretization bias for pure
  GBM regardless of `n_steps`.
- **`spread_option_mc_payoff`** and **`worst_of_basket_payoff`**
  (`valax/pricing/mc/payoffs.py`): per-path payoffs for spread options
  and worst-of baskets on multi-asset paths.
- **Two new dispatcher recipes**:
  - `(SpreadOption, MultiAssetGBMModel)` — validates against the
    Margrabe closed form at $K=0$ (exact within 3 SE across
    correlations) and Kirk's approximation at $K \neq 0$.
  - `(WorstOfBasketOption, MultiAssetGBMModel)` — correlation-sensitive
    basket payoffs; `jax.grad` through `correlation` gives the
    correlation Greeks.
- Test coverage: 27 new tests (12 path-generator + 15 recipe tests)
  covering shape, statistical convergence of empirical correlation,
  risk-neutral drift, analytical-reference agreement, dispatcher
  error handling, and autodiff flow.

### Added — Monte Carlo dispatcher

- **Unified MC entry point** (`valax/pricing/mc/dispatch.py`):
  `mc_price_dispatch(instrument, model, config, key, **market_args)`
  looks up a recipe keyed on `(type(instrument), type(model))` and runs
  the appropriate path generation + payoff + discounting. Replaces the
  hand-assembly of path generators + payoff functions for every
  instrument / model pair.
- **`MCResult` container** (price + stderr + n_paths) as a frozen
  `equinox.Module`; `float(result)` shortcut for scalar use.
- **`register()` decorator** for contributors to add new
  (instrument, model) recipes without modifying engine code. Duplicate
  registration raises unless `overwrite=True`.
- **`registered_recipes()`** introspection helper.
- **14 built-in recipes** shipped in `valax/pricing/mc/recipes.py`:
  - Equity × (`BlackScholesModel`, `HestonModel`): `EuropeanOption`,
    `AsianOption`, `EquityBarrierOption`, `LookbackOption`,
    `VarianceSwap`.
  - Rates × `LMMModel`: `Caplet`, `Cap`, `Swaption` (European),
    `BermudanSwaption` (Longstaff-Schwartz).
- **Helper** `discounted_mean_and_stderr(cashflows, df, n_paths)` for
  consistent discounting across recipes.
- Legacy `mc_price` / `mc_price_with_stderr` kept unchanged for
  backward compatibility; new code should use `mc_price_dispatch`.

### Added — Models

- **Hull-White one-factor short-rate model** (`valax/models/hull_white.py`):
  mean-reverting `dr = (θ(t) - a·r) dt + σ dW` with $\theta(t)$ implicitly
  calibrated to the initial discount curve. Includes analytic ZCB price
  (`hw_bond_price`), short-rate variance (`hw_short_rate_variance`), and the
  $B(\tau)$ helper.
- **Hull-White trinomial tree** (`valax/pricing/lattice/hull_white_tree.py`):
  Hull-White (1994) construction with Arrow-Debreu forward sweep for exact
  curve fit; `callable_bond_price` and `puttable_bond_price` via backward
  induction. Fully `jax.jit` / `jax.grad` compatible.
- **SABR stochastic volatility model** (`valax/models/sabr.py`,
  `valax/pricing/analytic/sabr.py`, `valax/pricing/mc/sabr_paths.py`): Hagan's
  asymptotic expansion for implied vol, plus diffrax-based MC simulation.
- **LIBOR Market Model** (`valax/models/lmm.py`): piecewise-constant and
  Rebonato volatility specs; exponential and two-parameter correlation; PCA
  factor loading; spot-measure drift correction. MC simulation in
  `valax/pricing/mc/lmm_paths.py`.

### Added — Instruments (35+ data-only pytrees)

- **Equity**: `EuropeanOption`, `AmericanOption`, `EquityBarrierOption`,
  `AsianOption`, `LookbackOption`, `VarianceSwap`, `SpreadOption`.
- **Fixed income**: `ZeroCouponBond`, `FixedRateBond`, `FloatingRateBond`,
  `CallableBond`, `PuttableBond`, `ConvertibleBond` (data-only — pricer
  planned).
- **Rates**: `Caplet`, `Cap`, `Floor`, `InterestRateSwap`, `Swaption`,
  `OISSwap`, `CrossCurrencySwap`, `TotalReturnSwap`, `CMSSwap`, `CMSCapFloor`,
  `RangeAccrual`, `BermudanSwaption`.
- **Inflation**: `ZeroCouponInflationSwap`, `YearOnYearInflationSwap`,
  `InflationCapFloor`.
- **FX**: `FXForward`, `FXVanillaOption`, `FXBarrierOption`.
- **Credit** (data-only — pricer planned): `CDS`, `CDOTranche`.

### Added — Pricing engines

- **Analytical**: Black-Scholes, Black-76, Bachelier, Garman-Kohlhagen (FX),
  Hagan-SABR, bond pricing (`bonds.py`), caplet/cap/swaption analytics,
  inflation pricers (`inflation.py`), rates exotics (`rates_exotics.py`:
  XCCY, TRS, CMS, range accrual), spread options (Margrabe + Kirk),
  variance swap.
- **Monte Carlo**: GBM, Heston, SABR, LMM path generation via diffrax.
  Engine in `valax/pricing/mc/engine.py` with deterministic seeding and
  standard-error reporting. Bermudan swaptions via Longstaff-Schwartz
  on LMM paths (`valax/pricing/mc/bermudan.py`).
- **PDE**: Crank-Nicolson finite-difference solver on the log-spot
  Black-Scholes PDE, using `lineax` for tridiagonal solves
  (`valax/pricing/pde/solvers.py`).
- **Lattice**: CRR binomial tree (European + American) and Hull-White
  trinomial tree (callable/puttable bonds).

### Added — Curves and dates

- `DiscountCurve` with log-linear interpolation and flat extrapolation.
- `MultiCurveSet`: OIS discount curve plus tenor-specific forward curves
  for dual-curve swap pricing.
- Bootstrapping in `valax/curves/bootstrap.py`: sequential analytic solve
  per pillar; simultaneous Newton solve in log-DF space via `optimistix`.
- Bootstrap instruments: `DepositRate`, `FRA`, `SwapRate`.
- `InflationCurve` with log-CPI interpolation, ZCIS breakeven, and YoY
  forward rate helpers.
- Day count conventions: Act/360, Act/365, Act/Act, 30/360 — all
  JIT-compatible integer-ordinal arithmetic.
- Schedule generation in `valax/dates/schedule.py`.

### Added — Calibration

- SABR per-expiry calibration (`valax/calibration/sabr.py`) via
  `optimistix.least_squares` (Levenberg-Marquardt) and BFGS.
- Heston calibration (`valax/calibration/heston.py`) supporting LM, BFGS,
  and Optax solvers.
- Parameter constraint transforms (`valax/calibration/transforms.py`):
  softplus, tanh, sigmoid for positivity, $(-1, 1)$, and $[0, 1]$
  parameters respectively.

### Added — Volatility surfaces

- `GridVolSurface` with bilinear interpolation on the (strike, expiry) grid.
- `SABRVolSurface` with per-expiry SABR calibration and parameter
  interpolation across expiries.
- `SVIVolSurface` with Gatheral's SVI parameterization per expiry,
  calibrated by Levenberg-Marquardt.
- All surfaces are differentiable pytrees — `jax.grad` through the surface
  gives smile-aware vega.

### Added — Risk and scenarios

- `MarketData` container and `MarketScenario` / `ScenarioSet` pytrees in
  `valax/market/`.
- Curve shocks: parallel, steepener, flattener, butterfly, key-rate
  (`valax/risk/shocks.py`).
- Scenario generation: parametric (normal and t), historical replay,
  Monte Carlo (`valax/risk/scenarios.py`).
- Value at Risk and Expected Shortfall via vmapped repricing
  (`valax/risk/var.py`); parametric (delta-normal) VaR using autodiff
  gradients; P&L attribution via second-order Taylor expansion.
- **Sensitivity ladders** (`valax/risk/ladders.py`): bucketed pytree
  carrying first-order (delta, vega, rho) and second-order (gamma, vanna,
  volga, cross-gammas) sensitivities. Single Jacobian + single Hessian
  pass via autodiff; 10-rung waterfall P&L decomposition with
  unexplained residual.

### Added — Greeks framework

- `valax/greeks/autodiff.py` provides `greeks()` (all first-order Greeks at
  once via reverse-mode autodiff) and `greek()` (single named Greek).
  Works on any pricing function transparently because every VALAX data
  structure is a JAX pytree — `jax.grad` flows through curves, surfaces,
  and model parameters.

### Added — Portfolio

- `batch_price` and `batch_greeks` (`valax/portfolio/batch.py`) for
  vmap-vectorized pricing across thousands of instruments in a single
  call.

### Added — Examples and validation

- 6 runnable example scripts in `examples/` covering equity options
  (BSM + Greeks + portfolio vmap), SABR smile + calibration, fixed income
  (curves + duration + KRD), rates derivatives (caps/swaps/swaptions),
  Monte Carlo (GBM/Heston/SABR/exotics), and PDE/lattice methods.
- 8 QuantLib comparison test files in `tests/test_quantlib_comparison/`
  validating European options, fixed income, Heston, SABR, MC, PDE/lattice,
  risk Greeks, and parametric VaR against QuantLib reference values.

### Added — Documentation

- mkdocs Material site published to GitHub Pages via the workflow added
  in `0ceec5e`.
- Full theory document (`docs/theory.md`) covering risk-neutral pricing,
  every implemented model (BSM, Black-76, Bachelier, Heston, SABR, LMM,
  Garman-Kohlhagen, Hull-White, two-asset BSM/spread options), curves,
  bootstrapping, day counts, inflation, volatility surfaces, all four
  pricing methods, autodiff Greeks, risk measures, and calibration.
- User guides under `docs/guide/`: instruments overview, analytical
  pricing, fixed income, callable/puttable/FRN/convertible bonds,
  short-rate (Hull-White) workflow, FX derivatives, inflation,
  interest rate exotics, equity exotics, credit, Greeks, Monte Carlo,
  PDE, lattice, calibration, vol surfaces, risk.
- API reference under `docs/api/` for every module.
- `CONTRIBUTING.md` documenting architectural constraints, code style,
  test expectations, and PR checklist.
- This `CHANGELOG.md`.
- Production roadmap (`docs/roadmap.md`) with tier-based feature inventory
  plus a five-priority execution plan.

### Changed

- Risk engine upgraded to support multi-asset repricing and parametric
  VaR alongside the existing full-revaluation VaR (`15e6c01`).
- API reference brought up to date with all newer pricers and models
  (`6a88702`).
- Hull-White documentation added and callable/puttable bonds promoted to
  "Implemented" in the roadmap (`484dc45`).
- Rates exotics, FRN, OIS pricing documented and instruments promoted to
  "Implemented" in the roadmap (`23592a6`, `84c11e6`).

### Infrastructure

- `pyproject.toml` with hatchling build backend; core deps `jax[cpu]`,
  `equinox`, `diffrax`, `optimistix`, `optax`, `lineax`, `jaxtyping`.
  Dev deps: `pytest`, `pytest-benchmark`, `pytest-cov`, `hypothesis`.
  Docs deps: `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`.
- GitHub Pages deployment workflow.

## [0.0.0] — Initial commit

- Phase 1 MVP: analytical pricing with autodiff Greeks (`d900420`).
- Project skeleton, Black-Scholes pricer, autodiff Greeks framework.

[Unreleased]: https://github.com/ACea15/VALAX/compare/v0.0.0...HEAD
[0.0.0]: https://github.com/ACea15/VALAX/releases/tag/v0.0.0
