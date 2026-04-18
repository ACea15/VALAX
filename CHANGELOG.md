# Changelog

All notable changes to VALAX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

VALAX is currently pre-1.0. Everything below was developed under the `0.1.0`
version tag in `pyproject.toml`. The first tagged release will compress the
history below into a single `[0.1.0]` entry; until then, all changes accumulate
under `[Unreleased]` and are grouped by feature area for discoverability.

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
