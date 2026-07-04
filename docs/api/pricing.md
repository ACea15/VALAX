# Pricing functions

All pricing functions are pure functions with no side effects: they
take an instrument and market inputs and return a scalar JAX array.
This is the shape that `jax.grad`, `jax.jit`, and `jax.vmap` expect —
Greeks, batch pricing, and calibration all compose out of the box.

## Analytical (closed-form)

### Equity vanillas

::: valax.pricing.analytic.black_scholes_price

::: valax.pricing.analytic.black76_price

::: valax.pricing.analytic.bachelier_price

::: valax.pricing.analytic.black_scholes.black_scholes_implied_vol

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
[theory §2.4](../theory.md#24-heston-stochastic-volatility).

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
[theory §4.4](../theory.md#44-local-volatility-dupire).

::: valax.pricing.analytic.dupire_local_vol

::: valax.pricing.analytic.dupire_local_vol_from_strike

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

Single-curve pricing for FRNs and OIS swaps. The OIS float leg uses
the telescoping identity \(N \cdot (DF(T_0) - DF(T_n))\); FRNs satisfy
the par-at-reset invariant.

::: valax.pricing.analytic.floating_rate_bond_price

::: valax.pricing.analytic.ois_swap_price

::: valax.pricing.analytic.ois_swap_rate

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

Crank–Nicolson finite-difference solver in log-spot space.

::: valax.pricing.pde.solvers.pde_price

::: valax.pricing.pde.solvers.PDEConfig

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

::: valax.pricing.lattice.hull_white_tree.build_hull_white_tree

::: valax.pricing.lattice.hull_white_tree.HullWhiteTree

::: valax.pricing.lattice.hull_white_tree.callable_bond_price

::: valax.pricing.lattice.hull_white_tree.puttable_bond_price
