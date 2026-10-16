# Pricing functions

All pricing functions are pure functions with no side effects: they
take an instrument and market inputs and return a scalar JAX array.
This is the shape that `jax.grad`, `jax.jit`, and `jax.vmap` expect —
Greeks, batch pricing, and calibration all compose out of the box.

!!! tip "One call convention over all of them"
    The pricers on this page are also reachable through the unified
    [`price(instrument, model, market, method=...)`](core.md) entry point,
    which binds an instrument's curve/underlying selectors against a
    [`MarketState`](core.md#marketstate) and picks a method for you — except
    the Hull–White trinomial-tree pricers under [Lattice](#lattice), which
    are direct calls only for now. The
    explicit signatures below remain the primary API; the unified one is
    what portfolio, risk, and multi-method comparison code should use.

## Analytical (closed-form)

### Equity vanillas

::: valax.pricing.analytic.black_scholes_price

::: valax.pricing.analytic.black76_price

::: valax.pricing.analytic.bachelier_price

::: valax.pricing.analytic.black_scholes.black_scholes_implied_vol

### Digital options

Closed-form cash-or-nothing digital: a call pays ``payout`` if
\(S_T > K\) with price \(\text{payout}\, e^{-rT} N(d_2)\); the put pays if
\(S_T < K\). Used as the analytic reference for the PDE digital recipe.

::: valax.pricing.analytic.digital_option_price

### SABR (Hagan formula)

::: valax.pricing.analytic.sabr_implied_vol

::: valax.pricing.analytic.sabr_price

### Heston (COS method)

Semi-analytic Heston European option price via the Fang–Oosterlee
(2008) COS expansion with the Lord–Kahl "Little Trap" characteristic
function. Call and put payoff coefficients are computed directly (not
via put-call parity) for sharper accuracy at deep OTM strikes; the
truncation interval is set from closed-form Heston cumulants.

Defaults `N=160, L=12` give < 1e-7 absolute error in moneyness range
0.85–1.15; deep wings benefit from `L=18, N=256`. Agrees with
QuantLib's `AnalyticHestonEngine` to < 5e-7 across the validation
grid. `rate` and `dividend` default to `model.rate` / `model.dividend`
when `None` — pass explicit values to reprice the same fitted model
under a stressed discount curve. See
[theory §2.4](../theory/stochastic-models.md#24-heston-stochastic-volatility).

::: valax.pricing.analytic.heston_cos_price

### Dupire local volatility

Local volatility \(\sigma_\text{loc}(k, T)\) extracted from an
implied-vol surface via Gatheral's IV-space form of the Dupire
formula. The `surface` argument is duck-typed — any object exposing
`total_variance(k, T) -> Float[Array, ""]` works (all three VALAX
surfaces qualify). All three partial derivatives of total variance are
evaluated via `jax.grad` directly on the surface — no finite
differences. Requires `jax_enable_x64=True` (enforced; enabled by
default in `valax/__init__.py`). See
[theory §4.4](../theory/volatility.md#44-local-volatility-dupire).

::: valax.pricing.analytic.dupire_local_vol

::: valax.pricing.analytic.dupire_local_vol_from_strike

Path simulation (the local-vol Monte Carlo and the SLV leverage target)
evaluates Dupire through a guard that makes it total: the query is clamped to
a trusted log-moneyness domain (flat local vol beyond), a non-positive
Gatheral denominator falls back to the implied variance, and the local
variance is clipped to a band around the ATM variance — with finite
gradients throughout. See the
[Monte Carlo guide § local volatility](../guide/monte-carlo.md#local-volatility).

::: valax.pricing.analytic.LocalVolGuard

::: valax.pricing.analytic.guarded_local_vol

::: valax.pricing.analytic.guarded_local_variance

### Bonds

::: valax.pricing.analytic.zero_coupon_bond_price

::: valax.pricing.analytic.fixed_rate_bond_price

::: valax.pricing.analytic.fixed_rate_bond_price_from_yield

::: valax.pricing.analytic.yield_to_maturity

::: valax.pricing.analytic.modified_duration

::: valax.pricing.analytic.convexity

::: valax.pricing.analytic.key_rate_durations

### Caplets / caps

::: valax.pricing.analytic.caplet_price_black76

::: valax.pricing.analytic.caplet_price_bachelier

::: valax.pricing.analytic.cap_price_black76

::: valax.pricing.analytic.cap_price_bachelier

### Swaptions

::: valax.pricing.analytic.swap_rate

::: valax.pricing.analytic.swap_price

::: valax.pricing.analytic.swaption_price_black76

::: valax.pricing.analytic.swaption_price_bachelier

#### Hull–White (Jamshidian)

Model-consistent swaption pricing under the one-factor Hull–White short-rate
model. Because every bond price at expiry is monotone in the single state
variable, the option on the coupon bond decomposes **exactly** into a portfolio
of zero-coupon bond options — no lattice, no simulation. This is the pricer
that makes Hull–White calibratable; see
[Jamshidian Decomposition](../theory/hull-white-swaptions.md).

Discounting uses the model's `initial_curve`, which
[`price`](core.md#the-unified-signature) binds from the `MarketState` (this
is the auto-default `"analytic"` route for `Swaption` × `HullWhiteModel`);
called directly as a kernel, the model must be bound first with
[`bind_curves`](core.md#valax.core.binding.bind_curves).

Both Jamshidian pricers accept a piecewise-constant \(\sigma(t)\) (only the
bond-option variance changes). The critical-rate Newton solve is guarded: an
absurd or non-finite parameter yields a NaN price and NaN gradient instead of
a runtime raise inside `jit` / `vmap`. Measured overhead and an example:
[Analytical guide § short-rate swaptions](../guide/analytical.md#short-rate-swaptions-jamshidian-brigomercurio).

::: valax.pricing.analytic.hw_swaption_price

::: valax.pricing.analytic.hw_critical_rate

::: valax.pricing.analytic.hw_zcb_option_price

#### G2++ (Brigo–Mercurio semi-analytic)

Model-consistent swaption pricing under the two-factor G2++ model. With two
state variables Jamshidian's one-dimensional decomposition no longer applies, so
the price is an exact **one-dimensional integral** over the first factor with a
Jamshidian-style critical boundary on the second, evaluated by Gauss–Hermite
quadrature. See [G2++: Two-Factor Gaussian](../theory/g2pp.md#3-european-swaptions-brigomercurio-integral).

Discounting uses the model's `initial_curve`, bound from the market by
`price` (or explicitly with `bind_curves` for a kernel call). The `forward_curve` argument on
this *analytic* pricer is reserved for a future extension and must be `None`;
for a dual-curve G2++ swaption today, use the PDE or MC recipe, which implement
the deterministic tenor basis of
[`valax.pricing.multicurve`](#multi-curve-overlay-for-the-short-rate-solvers).

The price accepts piecewise-constant \(\sigma(t)\) and \(\eta(t)\) (via
`volatility_x` / `volatility_ratio`), and the exercise-boundary Newton solve
carries the same NaN-not-raise guard as Jamshidian's.

::: valax.pricing.analytic.g2pp_swaption_price

### FX pricing

::: valax.pricing.analytic.fx_forward_rate

::: valax.pricing.analytic.fx_forward_price

::: valax.pricing.analytic.garman_kohlhagen_price

::: valax.pricing.analytic.fx_implied_vol

FX delta in one of three market conventions: `"spot"` — standard spot
delta \(e^{-r_f T} \Phi(d_1)\); `"forward"` — forward delta
\(\Phi(d_1)\); `"premium_adjusted"` — premium-adjusted spot delta
\(e^{-r_f T} \Phi(d_1) - P/S\).

::: valax.pricing.analytic.fx_delta

::: valax.pricing.analytic.strike_to_delta

::: valax.pricing.analytic.delta_to_strike

### Variance swaps

Under Black–Scholes the fair variance is the squared implied vol.
`variance_swap_price_seasoned` blends realized variance over the
elapsed period with implied variance over the remaining period.

::: valax.pricing.analytic.variance_swap_fair_strike

::: valax.pricing.analytic.variance_swap_price

::: valax.pricing.analytic.variance_swap_price_seasoned

### Floating-rate instruments

Pricing for FRNs and OIS swaps. In the default single-curve setup the
OIS float leg uses the telescoping identity
\(N \cdot (DF(T_0) - DF(T_n))\) and FRNs satisfy the par-at-reset
invariant; pass the optional `forward_curve` argument to project the
floating leg off a distinct forecasting curve.

::: valax.pricing.analytic.floating_rate_bond_price

::: valax.pricing.analytic.ois_swap_price

::: valax.pricing.analytic.ois_swap_rate

### Swap legs

The generalisation of the annuity and float-leg expressions above to
amortising notionals, realised fixings and a projection curve per leg.
Each function reproduces the narrower pricer it generalises bit for bit
on its own path, which is asserted with bare `==` in
`tests/test_pricing/test_legs.py`. Conventions and worked examples:
[Fixed Income → Swap legs and multi-leg trades](
../guide/fixed-income.md#swap-legs-and-multi-leg-trades).

::: valax.pricing.analytic.fixed_leg_pv

::: valax.pricing.analytic.floating_leg_pv

::: valax.pricing.analytic.leg_pv

::: valax.pricing.analytic.swap_pv

::: valax.pricing.analytic.swap_fair_rate

### Rates exotics

::: valax.pricing.analytic.cross_currency_swap_price

::: valax.pricing.analytic.cross_currency_basis_spread

::: valax.pricing.analytic.total_return_swap_price

CMS pricers use per-period forward par swap rates on a synthetic
annual underlying swap. **No convexity adjustment** — see the
[rates-exotics guide](../guide/rates-exotics.md) for caveats.

::: valax.pricing.analytic.cms_swap_price

::: valax.pricing.analytic.cms_cap_floor_price_black76

::: valax.pricing.analytic.range_accrual_price_black76

### Curve-graph pricing

Thin wrappers that resolve curve identifiers on a
[`CurveGraph`](curves.md#valax.curves.CurveGraph) and delegate to the
`DiscountCurve`-based pricers above. Passing `forward_id=None`
reproduces single-curve behaviour exactly; see
[the curves guide §8](../guide/curves.md#8-what-is-not-yet-implemented).

::: valax.pricing.analytic.swap_price_from_graph

::: valax.pricing.analytic.swap_rate_from_graph

::: valax.pricing.analytic.swaption_price_black76_from_graph

::: valax.pricing.analytic.swaption_price_bachelier_from_graph

::: valax.pricing.analytic.floating_rate_bond_price_from_graph

::: valax.pricing.analytic.ois_swap_price_from_graph

::: valax.pricing.analytic.ois_swap_rate_from_graph

::: valax.pricing.analytic.caplet_price_black76_from_graph

::: valax.pricing.analytic.caplet_price_bachelier_from_graph

::: valax.pricing.analytic.cap_price_black76_from_graph

::: valax.pricing.analytic.cap_price_bachelier_from_graph

::: valax.pricing.analytic.total_return_swap_price_from_graph

::: valax.pricing.analytic.cms_swap_price_from_graph

::: valax.pricing.analytic.cms_cap_floor_price_black76_from_graph

::: valax.pricing.analytic.range_accrual_price_black76_from_graph

### Inflation derivatives

::: valax.pricing.analytic.zcis_price

::: valax.pricing.analytic.zcis_breakeven_rate

::: valax.pricing.analytic.yyis_price

::: valax.pricing.analytic.inflation_cap_floor_price_black76

### Spread options

Margrabe's exact formula for exchange options (\(K = 0\)) is
independent of the risk-free rate. Kirk's approximation treats
\(S_2 + K\) as a single asset with adjusted vol and degenerates to
Margrabe when \(K = 0\).

::: valax.pricing.analytic.margrabe_price

::: valax.pricing.analytic.kirk_price

::: valax.pricing.analytic.spread_option_price

## Monte Carlo

See the [Monte Carlo guide](../guide/monte-carlo.md) for the full
coverage map and contributor cookbook.

### Unified dispatcher (preferred)

`mc_price_dispatch` looks up a recipe keyed on
`(type(instrument), type(model))` and runs the appropriate path
generation + payoff + discounting. Raises `ValueError` with the list
of available recipes if the pair is not registered.

Typical `market_args`:

- Equity recipes: `spot` (required).
- LMM rate recipes: `forward_index` / `forward_indices` + `taus` (or
  `exercise_indices` for Bermudan).
- Optional per-recipe knobs: e.g. `annual_factor` for variance-swap
  realization, `n_steps_per_period` for LMM, `lsm_config` for
  Bermudan.
- Short-rate recipes (Hull–White, G2++, Cheyette): `T` (horizon) and
  `n_steps` (default 100). These recipes **ignore `MCConfig.n_steps`**; see
  the [Monte Carlo guide § time-dependent volatility](../guide/monte-carlo.md#time-dependent-volatility).

::: valax.pricing.mc.mc_price_dispatch

::: valax.pricing.mc.MCResult

::: valax.pricing.mc.MCConfig

::: valax.pricing.mc.register

::: valax.pricing.mc.registered_recipes

#### Built-in recipes

Single-asset equity (each combo with `BlackScholesModel`,
`HestonModel`, `LocalVolModel`, and `SLVModel`):

- `(EuropeanOption, ...)`
- `(AsianOption, ...)`
- `(EquityBarrierOption, ...)`
- `(LookbackOption, ...)`
- `(VarianceSwap, ...)`

Multi-asset equity (`MultiAssetGBMModel`):

- `(SpreadOption, MultiAssetGBMModel)` — payoff
  \(\max(S_1 - S_2 - K, 0)\); validates Margrabe at \(K = 0\) and
  Kirk at \(K \neq 0\).
- `(WorstOfBasketOption, MultiAssetGBMModel)` — payoff on
  \(\min_i S_i(T)/S_i(0)\); correlation-sensitive.

Rates (LMM):

- `(Caplet, LMMModel)`
- `(Cap, LMMModel)`
- `(Swaption, LMMModel)`
- `(BermudanSwaption, LMMModel)`

Rates (short-rate, exact conditional simulation):

- `(FixedRateBond, HullWhiteModel)`, `(FloatingRateBond, HullWhiteModel)`,
  `(CallableBond, HullWhiteModel)`, `(PuttableBond, HullWhiteModel)`
- `(FixedRateBond, G2PPModel)`, `(FloatingRateBond, G2PPModel)`,
  `(Swaption, G2PPModel)`
- `(CMSSpreadSwap, G2PPModel)` — decorrelation-sensitive steepener/flattener;
  both CMS rates are computed analytically per path from the G2++ affine ZCB.
- `(FixedRateBond, CheyetteModel)`, `(FloatingRateBond, CheyetteModel)`,
  `(Caplet, CheyetteModel)`, `(Cap, CheyetteModel)`, `(Swaption, CheyetteModel)`
  — the skew-sensitive vanillas that motivate the Cheyette local volatility.
- `(FixedRateBond, CheyetteModel2F)`, `(FloatingRateBond, CheyetteModel2F)`,
  `(CMSSpreadSwap, CheyetteModel2F)`.

### Legacy entry points

Still exported for backward compatibility; new code should use
`mc_price_dispatch`.

::: valax.pricing.mc.mc_price

::: valax.pricing.mc.mc_price_with_stderr

### Path generators (low-level)

::: valax.pricing.mc.generate_gbm_paths

::: valax.pricing.mc.generate_heston_paths

::: valax.pricing.mc.generate_sabr_paths

::: valax.pricing.mc.generate_correlated_gbm_paths

Local-vol path generator uses `lax.scan` + log-Euler with
**midpoint-in-time \(\sigma\)**. `scheme="midpoint_euler"` (default)
or `scheme="milstein"` (opt-in, ~2× cost — helps strong-order accuracy
for path-statistics-sensitive analyses; no measurable benefit on
vanilla repricing).

::: valax.pricing.mc.generate_local_vol_paths

SLV path generator combines an Andersen-QE variance leg with a
log-Euler / Milstein log-spot leg using midpoint-in-time \(L\).
Spot/variance correlation via
\(Z_1 = \rho Z_v + \sqrt{1 - \rho^2} Z_\perp\) — exact on the QE
quadratic branch (typical case), approximate on the exponential
branch.

::: valax.pricing.mc.generate_slv_paths

::: valax.pricing.mc.LMMPathResult

::: valax.pricing.mc.generate_lmm_paths

The Hull–White short-rate generator samples the **exact** conditional
Ornstein–Uhlenbeck law rather than discretising the SDE, so it is unbiased in
\(r(t)\) at any step size (the only discretisation error is the trapezoidal
accumulation of \(\int r\,dt\) into the log discount factor). With a
piecewise-constant \(\sigma(t)\) each step uses the exact step variance
(`hw_state_variance`). See
[Hull–White Monte Carlo](../theory/hull-white-mc.md) for the derivation.

::: valax.pricing.mc.HullWhitePathResult

::: valax.pricing.mc.generate_hull_white_paths

The G2++ generator samples the **exact** two-factor conditional Gaussian law
(Cholesky of the \(2\times2\) conditional covariance) and integrates the
deterministic forward part of the discount exactly, so it is unbiased in the
factors and reprices the initial curve within Monte-Carlo error on flat and
steep curves alike. With piecewise-constant volatilities the covariance is
factored per step (`g2pp_state_covariance`). See [G2++ Monte Carlo](../theory/g2pp.md#4-monte-carlo-exact-two-factor-scheme).

::: valax.pricing.mc.G2PPPathResult

::: valax.pricing.mc.generate_g2pp_paths

### Payoff functions (low-level)

Equity payoffs on single-asset paths:

::: valax.pricing.mc.european_payoff

::: valax.pricing.mc.asian_option_payoff

::: valax.pricing.mc.asian_payoff

::: valax.pricing.mc.equity_barrier_payoff

::: valax.pricing.mc.barrier_payoff

::: valax.pricing.mc.lookback_payoff

::: valax.pricing.mc.variance_swap_payoff

Multi-asset payoffs on `generate_correlated_gbm_paths` output:

::: valax.pricing.mc.spread_option_mc_payoff

::: valax.pricing.mc.worst_of_basket_payoff

Rate payoffs on `LMMPathResult`:

::: valax.pricing.mc.caplet_mc_payoff

::: valax.pricing.mc.cap_mc_payoff

::: valax.pricing.mc.swaption_mc_payoff

### Bermudan (Longstaff–Schwartz)

::: valax.pricing.mc.bermudan_swaption_lsm

::: valax.pricing.mc.LSMConfig

## PDE

Finite-difference pricing. The 1-D Crank–Nicolson solver `pde_price` prices
European options under Black–Scholes; the dispatcher `pde_price_dispatch` routes
`(instrument, model)` pairs to the appropriate recipe — European, American (via
the penalty-method free boundary), digital (with Rannacher start-up), and
barrier (absorbing boundary, knock-ins via in/out parity). Interest-rate
instruments under `HullWhiteModel` — including **Bermudan swaptions** and
callable/puttable bonds — are covered by the
[Hull–White short-rate PDE](#hullwhite-short-rate-pde) recipes below. See the
[PDE design doc](../architecture/pde-design.md) for the architecture.

### Direct solver (1-D European)

::: valax.pricing.pde.pde_price

::: valax.pricing.pde.PDEConfig

### Dispatcher

`pde_price_dispatch` looks up a recipe keyed on
`(type(instrument), type(model))` and raises `ValueError` (listing the available
recipes) if the pair is unregistered.

::: valax.pricing.pde.pde_price_dispatch

::: valax.pricing.pde.PDEResult

::: valax.pricing.pde.register

::: valax.pricing.pde.registered_recipes

### Configuration enums

::: valax.pricing.pde.Scheme

::: valax.pricing.pde.Exercise

::: valax.pricing.pde.PDEConfig2D

### Hull–White short-rate PDE

Interest-rate instruments under `HullWhiteModel`, solved in the centred state
variable \(x\) of \(r(t) = x(t) + \alpha(t)\). Recipes are registered for
`FixedRateBond`, `CallableBond`, `PuttableBond`, `Caplet`, `Cap` (the `is_cap`
flag selects cap vs floor), `Swaption` and `BermudanSwaption`, and are reached
through `pde_price_dispatch` like any other pair. A caplet prices as a European
option on the accrual-end zero-coupon bond; a cap/floor prices in a single
backward sweep with each caplet's fixing-date value injected at its fixing
level (see the [shared payoff layer](#shared-rates-payoff-layer)). See the
theory note
[Hull-White Finite Differences](../theory/hull-white-pde.md) for the
discretisation, boundary treatment and validation.

Unlike the trinomial tree, nothing here has to be concrete: schedules are
scattered with traced indices, so the pricers compose with `eqx.filter_jit` and
`eqx.filter_grad` directly.

With a piecewise-constant \(\sigma(t)\) each stacked row diffuses at the exact
step average of \(\sigma^2\); the per-model \(\sigma(t)\) coverage of all
short-rate PDEs (which builders take it, which raise) is in the
[PDE guide §5.8](../guide/pde.md#58-time-dependent-volatility).

::: valax.pricing.pde.coefficients.hw_operator_stack

::: valax.pricing.pde.grids.centred_state_grid

::: valax.pricing.pde.boundary.apply_linearity_bc_1d

::: valax.pricing.pde.boundary.zero_boundary

### G2++ two-factor short-rate PDE

Interest-rate instruments under `G2PPModel`, solved on the 2-D ADI substrate in
the two centred Gaussian factors \(x, y\) of \(r(t) = x(t) + y(t) + \varphi(t)\).
The cross term \(\rho\sigma\eta\, V_{xy}\) — the source of tenor decorrelation —
is handled as a constant mixed operator exactly as Heston's is; the deterministic
shift \(\varphi(t)\) is factored out and applied as an exact per-step scalar
discount so the scheme preserves the initial-curve fit. Recipes are registered
for `FixedRateBond`, `Caplet`, `Cap`, `CallableBond`, `PuttableBond`,
`Swaption` and the headline `BermudanSwaption`, reached through
`pde_price_dispatch` like any other pair — the callable/puttable and caplet/cap
payoffs reuse the [shared payoff layer](#shared-rates-payoff-layer) on the 2-D
ADI grid. See the theory note
[G2++ PDE](../theory/g2pp.md#8-pde-2-d-finite-differences) for the
discretisation, boundary treatment and validation.

The constant operator above is for a one-bucket model and raises
`ValueError` for more than one volatility bucket; a piecewise-constant model
is solved with one operator per backward step, assembled inside the ADI loop
by `g2pp_operator_rows` (the recipes pick the builder themselves).

::: valax.pricing.pde.coefficients.g2pp_operator_2d

::: valax.pricing.pde.coefficients.g2pp_operator_rows

::: valax.pricing.pde.boundary.apply_linearity_bc_2d

### Cheyette quasi-Gaussian short-rate PDE

The Cheyette (quasi-Gaussian) model priced on the same short-rate substrate, in
two tiers. **Tier 1 (frozen-\(y\), production)** freezes the accumulator to its
Gaussian-limit value \(\bar y(t)\), so one factor is a 1-D PDE and two factors a
2-D ADI PDE reusing the Hull–White / G2++ machinery — the only new ingredient is
a *spatially-varying* diffusion \(\tfrac12\sigma^2(1+\beta(\tilde x + m))^2\)
carrying the skew. Recipes are registered for `FixedRateBond`, `Caplet`, `Cap`,
`Swaption`, `BermudanSwaption`, `CallableBond` and `PuttableBond` on **both**
factors, reached through `pde_price_dispatch`. A caplet strike slice is the
natural skew-\(\beta\) identification instrument for the PDE-based calibrator.
**Tier 2 (exact-\(y\), one factor)** carries the accumulator as a real
advection axis (`cheyette_exact_price`, which also supports a `Caplet`) for a
bias-free, zero-Monte-Carlo-noise measurement of the frozen-\(y\) error. See
the theory note [Cheyette §8 PDE](../theory/cheyette.md#8-pde-finite-differences).

All three tiers accept a piecewise-constant volatility level. The 1-F
`cheyette_operator_stack` handles both cases; the constant 2-D builders
`cheyette_operator_2d` (frozen-\(y\) 2F) and `cheyette_exact_operator_2d`
(exact-\(y\)) raise `ValueError` for more than one bucket, and their
per-step counterparts `cheyette_operator_rows_2d` and
`cheyette_exact_operator_rows` take over.

::: valax.pricing.pde.cheyette.cheyette_operator_stack

::: valax.pricing.pde.cheyette.cheyette_operator_2d

::: valax.pricing.pde.cheyette.cheyette_operator_rows_2d

::: valax.pricing.pde.cheyette.cheyette_exact_operator_2d

::: valax.pricing.pde.cheyette.cheyette_exact_operator_rows

::: valax.pricing.pde.cheyette.cheyette_exact_price

### Shared rates payoff layer

The Hull–White, G2++ and Cheyette short-rate PDE recipes all price the same
family of vanilla-rates payoffs — coupon bonds, swaps, caplets/caps, callable
exercise — and the payoff arithmetic is model-agnostic: it needs only the
model's affine bond value at the grid nodes plus the instrument schedule.
`valax.pricing.pde.rates_payoffs` holds that backbone once; each recipe passes
two thin closures (a coupon-portfolio valuer and a single-`P(t,T)` valuer over
its grid) and reuses the shared swap-tail, coupon-injection, caplet-payoff and
exercise-projection builders. This is what keeps the three modules DRY and made
the caplet/cap/floor and two-factor callable/puttable recipes cheap to add.

::: valax.pricing.pde.rates_payoffs.swap_tail_values

::: valax.pricing.pde.rates_payoffs.coupon_values_by_level

::: valax.pricing.pde.rates_payoffs.caplet_fixing_values

::: valax.pricing.pde.rates_payoffs.caplet_values_by_level

::: valax.pricing.pde.rates_payoffs.exercise_obstacle_by_level

::: valax.pricing.pde.rates_payoffs.exercisable_terminal_and_event

::: valax.pricing.pde.rates_payoffs.bermudan_terminal_and_event

### Cell-averaged payoffs

Every kinked payoff and exercise projection above is sampled through
`valax.pricing.pde.payoff_smoothing` when `config.smooth_payoff` is set (the
default). The positive part is replaced by its exact cell average on 1-D grids
and by a bias-neutral \(C^1\) kernel on 2-D grids. Either way the discrete
price is \(C^1\) in its inputs, at an \(O(h^2)\) bias. See
[PDE guide §5.5](../guide/pde.md#55-non-smooth-payoffs-rannacher-start-up-and-cell-averaging).

::: valax.pricing.pde.payoff_smoothing.smooth_positive_part

::: valax.pricing.pde.payoff_smoothing.bias_neutral_positive_part

::: valax.pricing.pde.payoff_smoothing.cell_half_variation

::: valax.pricing.pde.payoff_smoothing.positive_part

::: valax.pricing.pde.payoff_smoothing.project_max

::: valax.pricing.pde.payoff_smoothing.project_min

## Multi-curve overlay for the short-rate solvers

Hull–White and G2++ are exact-fit-to-*one*-curve models, so multi-curve pricing
(OIS discounting with a distinct projection curve) needs a modelling decision
rather than plumbing. `valax.pricing.multicurve` holds that decision once: the
discount curve is the model's `initial_curve`, **bound from the market** by
[`price`](core.md#the-unified-signature) (see
[curve fields](core.md#curve-fields-exact-fit-short-rate-models));
`with_discount_curve` only fills a model that arrives *unbound* (a recipe
called directly with `discount_curve=...`) and never re-fits a bound one.
Projection uses a **frozen multiplicative tenor basis** read off the two
curves, supplied as the optional `forward_curve` market argument
(automatically from an instrument's `forward_id` selector under `price`).

Whether the overlay applies is a property of the **model type**, declared once
by `basis_overlay`: Hull–White and G2++ use it; Cheyette does not, and
`model_tenor_basis` / `model_swap_leg_basis` raise when handed a distinct
projection curve for such a model — so `price()` now refuses a trade whose
resolved projection curve the recipe/model cannot use, instead of silently
pricing it single-curve.

`None` is the single-curve case throughout, and is a *trace-time* branch rather
than a multiplication by one, so single-curve prices are unchanged
bit-for-bit. See [Multi-Curve Short Rates](../theory/multi-curve-short-rate.md)
for the derivation and the limits of the approximation (no basis volatility).

::: valax.pricing.multicurve.with_discount_curve

::: valax.pricing.multicurve.basis_overlay

::: valax.pricing.multicurve.model_tenor_basis

::: valax.pricing.multicurve.model_swap_leg_basis

::: valax.pricing.multicurve.tenor_basis

::: valax.pricing.multicurve.swap_leg_basis

::: valax.pricing.multicurve.apply_swap_basis

::: valax.pricing.multicurve.require_single_curve

## Cross-engine reconciliation

`price_all_engines` runs whichever of the analytic / PDE / MC / QuantLib
engines has a recipe for a given `(instrument, model)` and returns them in one
table with absolute/relative differences against a reference engine. Engines
are best-effort: a missing recipe (e.g. Hull–White caplet MC) or an absent
optional dependency (QuantLib) is omitted rather than raising, so the same call
spans the whole short-rate instrument matrix. It underpins the validation
triangulation used across the tests and the `examples/comparisons/` scripts.

::: valax.pricing.reconcile.price_all_engines

::: valax.pricing.reconcile.ReconciliationTable

::: valax.pricing.reconcile.EngineQuote

## Short-rate engine coverage matrix

Which numerical engines can price each vanilla-rates instrument under each
short-rate model. **A** = analytic (closed form), **P** = finite-difference
PDE, **M** = Monte Carlo, **T** = Hull–White trinomial tree (direct call, not
yet through `price()`), **Q** = QuantLib comparison available. Every engine
listed accepts piecewise-constant volatility \(\sigma(t)\).

| Instrument                       | Hull–White | G2++      | Cheyette 1F | Cheyette 2F |
|----------------------------------|------------|-----------|-------------|-------------|
| Fixed-rate bond                  | A P M      | A P M     | P M         | P M         |
| Floating-rate note               | A M        | A M       | M           | M           |
| Caplet / floorlet                | A P (Q)    | P         | P M         | P           |
| Cap / floor                      | A P Q      | P         | P M         | P           |
| European swaption                | A P T Q    | A P M Q   | P M         | P           |
| Bermudan swaption                | P T Q      | P Q       | P           | P           |
| Callable / puttable bond         | P M T Q    | P         | P           | P           |
| CMS-spread swap                  | —          | P M       | —           | P M         |

**Multi-curve.** The Hull–White and G2++ engines accept a trade's own
`discount_curve` and `forward_curve` (see
[the overlay above](#multi-curve-overlay-for-the-short-rate-solvers)); the
Cheyette engines and the CMS-spread recipes are single-curve and reject a
distinct projection curve rather than ignoring it.

**Monte Carlo** here means a registered short-rate `mc_price_dispatch` recipe
(see [Built-in recipes](#built-in-recipes)); there is none for Hull–White
swaptions or for Bermudan swaptions under the short-rate models — the MC
Bermudan route is Longstaff–Schwartz on `LMMModel`.

Notes: the Cheyette PDE is the frozen-\(y\) production engine; the one-factor
exact-\(y\) PDE (`cheyette_exact_price`) additionally prices the fixed-rate
bond, European swaption and caplet for a bias-free cross-check. There is no
QuantLib Cheyette engine, so Cheyette is validated by reduction to Hull–White /
G2++ at \(\beta = 0\) and against its own Monte-Carlo engine.

### Why some instruments are MC-only

A handful of instruments are deliberately **not** on the PDE (they stay
Monte-Carlo-only), each for a structural reason rather than an omission:

- **LMM (LIBOR Market Model)** — the state is the whole vector of forward
  rates (one dimension per tenor), so there is no low-dimensional PDE; LMM is
  intrinsically a simulation model.
- **Range accruals** — the coupon depends on the *fraction of days* the index
  spends in a band, a path-functional that a low-dimensional backward PDE
  cannot carry without an extra occupation-time state.
- **CMS in arrears / averaging** — the payoff observes a swap rate off its
  natural measure with a path-average or timing shift; the convexity/timing
  adjustment is naturally a simulation (or static-replication) quantity.
- **Total-return and cross-currency swaps** — these carry an equity/FX (and,
  for XCCY, a second-currency curve) factor on top of rates, pushing the PDE
  dimension past what the short-rate substrate targets.
- **Five-dimensional exact-\(y\) two-factor Cheyette** — the exact accumulator
  is a \(2\times2\) symmetric matrix, giving a 5-D PDE; logged as out of scope
  in `docs/research-ideas.md`. The frozen-\(y\) 2-D ADI PDE covers the
  two-factor case in production.

## Lattice

### CRR binomial tree

Supports both European and American exercise.

::: valax.pricing.lattice.binomial.binomial_price

::: valax.pricing.lattice.binomial.BinomialConfig

### Callable / puttable bonds

Backward induction on a Hull–White recombining trinomial tree.
Per-step \(\alpha\) calibration matches market discount factors
exactly. Callable price is bounded above by the equivalent straight
bond; puttable price is bounded below.

The lattice half-width \(j_{\max}\) determines array shapes, so it must be a
concrete Python `int`. It is computed automatically from the concrete model
parameters, which forces them to be concrete — pass `j_max=` explicitly (see
`hw_tree_j_max` for a constant \(\sigma\), `hw_tree_width` for \(\sigma(t)\))
to differentiate through the tree with `eqx.filter_grad`.

::: valax.pricing.lattice.hull_white_tree.hw_tree_j_max

::: valax.pricing.lattice.hull_white_tree.build_hull_white_tree

::: valax.pricing.lattice.hull_white_tree.HullWhiteTree

::: valax.pricing.lattice.hull_white_tree.callable_bond_price

::: valax.pricing.lattice.hull_white_tree.puttable_bond_price

### General tree for piecewise-constant σ(t)

`build_hull_white_tree` dispatches a model with more than one volatility
bucket to `build_general_hull_white_tree` (Hull & White 2001), whose level
spacing follows \(\sigma(t)\); a one-bucket model keeps the Hull–White 1994
tree above, bit-identically. The general tree's half-width depends on the
whole volatility path, so size it eagerly with `hw_tree_width` (not
`hw_tree_j_max`) before `jit` / `grad`; an undersized width gives NaN prices,
never wrong ones. The callable / puttable pricers above run on it unchanged.
See [Lattice guide § general tree](../guide/lattice.md#general-tree-for-piecewise-constant-volatility).

::: valax.pricing.lattice.hull_white_tree.hw_tree_width

::: valax.pricing.lattice.hull_white_tree.build_general_hull_white_tree

::: valax.pricing.lattice.hull_white_tree.GeneralHullWhiteTree

### Swaptions on the Hull–White tree

European and Bermudan swaptions on either tree, with cell-averaged payoffs
(`smooth_payoff=True`, default) giving monotone first-order convergence in
`n_steps`. They are independent checks of Jamshidian and the PDE. Usage,
accuracy evidence and step-count guidance:
[Lattice guide § swaptions on the tree](../guide/lattice.md#swaptions-on-the-tree).

::: valax.pricing.lattice.hull_white_tree.swaption_tree_price

::: valax.pricing.lattice.hull_white_tree.bermudan_swaption_tree_price

### Through `price()`: `TreeConfig`

The four tree pricers above are routed through `valax.price(...,
method="lattice")` for `CallableBond`, `PuttableBond`, `Swaption` and
`BermudanSwaption` under `HullWhiteModel`, always on the general tree, with
the half-width sized from the config alone (`⌈width_factor · n_steps⌉`) so
the routes run on a traced model under `jit` / `grad`. `method="auto"` stays
the PDE (bonds, Bermudan) or Jamshidian (European swaption). See
[Lattice guide § through `price()`](../guide/lattice.md#hull-white-tree-through-price).

::: valax.pricing.lattice.TreeConfig
