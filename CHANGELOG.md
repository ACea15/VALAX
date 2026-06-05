# Changelog

All notable changes to VALAX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

VALAX is currently pre-1.0. Everything below was developed under the `0.1.0`
version tag in `pyproject.toml`. The first tagged release will compress the
history below into a single `[0.1.0]` entry; until then, all changes accumulate
under `[Unreleased]` and are grouped by feature area for discoverability.

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
  `equinox`, `diffrax`, `optimistix`, `optax`, `lineax`, `jaxtyping`,
  `beartype`. Dev deps: `pytest`, `pytest-benchmark`, `hypothesis`.
  Docs deps: `mkdocs`, `mkdocs-material`.
- GitHub Pages deployment workflow.

## [0.0.0] — Initial commit

- Phase 1 MVP: analytical pricing with autodiff Greeks (`d900420`).
- Project skeleton, Black-Scholes pricer, autodiff Greeks framework.

[Unreleased]: https://github.com/acea/valax/compare/v0.0.0...HEAD
[0.0.0]: https://github.com/acea/valax/releases/tag/v0.0.0
