# Risk Methodology & Assumptions Register

<!-- coverage-map -->

Every risk number in VALAX rests on choices: which coordinates it is
measured in, which order of Taylor expansion it truncates at, which
estimator turns a P&L vector into a quantile, which approximation turns a
test statistic into a p-value. This page is the **single register of those
choices**, the model-validation view of the risk layer. For each risk
measure it states the methodology, the defaults, the assumptions, the known
limitations (with the evidence that measured them) and the improvement path.

**What this page owns, and what it does not.** It owns *assumptions,
limitations and improvement paths*. It does not repeat how-tos (the risk
guides, starting at [Bucketed Sensitivities](../risk-sensitivities.md)),
derivations ([Theory § 7](../theory/risk-measures.md#7-risk-measures)) or
status (the [FRTB capability matrix](regulatory.md#frtb-capability-matrix),
[Risk Overview § 7](../risk-overview.md#7-the-state-of-the-risk-section-and-whats-next)
and the [roadmap](../roadmap.md)): each entry links to those instead. A
planned improvement is a roadmap line; the entry links to it. The sibling
page for calibrations is the
[Calibration Assumptions Register](calibration-register.md).

**Keeping it true.** `tests/test_docs/test_risk_register.py` fails if a
public risk-measure entry point has no entry here or if a named gate test
does not exist. Defaults and tolerances quoted here are read from the code
and tests; if you change one, change it here.

## How to read an entry

| Field | Meaning |
|---|---|
| **Entry points** | The public functions and classes it covers |
| **Definition & defaults** | What is computed, in which coordinates, with the default arguments |
| **Assumptions** | Modelling and market assumptions the number relies on |
| **Limitations** | What it cannot do or gets wrong, with the evidence (code line, test or measurement); *inference* marks a limitation read from the code but not yet measured by a test |
| **Validation** | The tests that gate it and their tolerances |
| **Improve** | The improvement path, linked to the roadmap |

## Cross-cutting assumptions

These apply to every entry below unless the entry says otherwise.

| # | Assumption | Consequence | Improve |
|---|---|---|---|
| R-X1 | **Unit prices, no positions.** `price` returns a unit price; nothing in `valax/risk/state_var.py` carries a quantity, sign or trade id. A "group" is a homogeneous batch of instruments under one model. | Book-level numbers are sums the caller weights by hand; there is no heterogeneous book aggregation, so desk-level P&L vectors, PLA and per-trade contributions cannot be produced by the library alone. | Position layer — [roadmap § 4.4 W1a](../roadmap.md#44-frtb-and-desk-risk-programme) |
| R-X2 | **Risk measures belong at the top.** A quantile of a sum is not a sum of quantiles; the group primitives return additive quantities (PV, P&L vectors, sensitivities) and the module docstring of `state_var.py` forbids summing per-group VaRs. | A caller who sums group VaRs overstates (or, for hedged books, misstates) risk; nothing prevents it. | Position layer and book aggregation — [W1a](../roadmap.md#44-frtb-and-desk-risk-programme) |
| R-X3 | **Factor coverage.** The keyed factor kinds are `spot`, `vol` (one parallel shift per surface), `curve` (zero-rate per pillar) and `dividend` (`FACTOR_KINDS`, `valax/risk/scenarios.py`). Rates surfaces are labelled `EQ.VOL`. | Credit spreads, FX spots, inflation, fixings and node-level vol are not keyed factors: they are absent from keyed sensitivities, scenarios, covariances and VaR (see [R-10](#r-10-keyed-scenario-shocks)). | Factor kinds — [W1b](../roadmap.md#44-frtb-and-desk-risk-programme) |
| R-X4 | **Determinism and common random numbers.** Every function is pure; MC groups reuse one PRNG key across base and shocked valuations. | Risk replays bit-for-bit. With a fresh key per revaluation the MC noise dominates small P&Ls: measured −0.207 spurious P&L against a +1.16 signal (`tests/test_risk/test_state_var.py::TestCommonRandomNumbers`). | — |
| R-X5 | **Two layouts.** The keyed path (`MarketState`, `price()`) is current; the flat path (`MarketData`, opaque `pricing_fn`, one rate block) is legacy, kept for one deprecation cycle. | Entries R-2, R-3 (flat part) and R-6 inherit every flat-path limitation until ported. | Measurement-layer port — [W1c](../roadmap.md#44-frtb-and-desk-risk-programme) |

## Sensitivities

### R-1 Keyed sensitivities

| | |
|---|---|
| **Entry points** | `keyed_sensitivities`, `KeyedSensitivities` (`key_rate_dv01`, `dv01`, `basis_dv01`, `vega`, `spot_delta`, `dividend_delta`, `block`, `gamma_block`, `block_totals`, `to_dict`), `BASIS_POINT` |
| **Definition & defaults** | Derivatives of $x \mapsto \mathrm{PV}(\mathrm{shock}(\text{market}, \mathrm{unflatten}(x)))$ at $x = 0$, where $x$ is the factor vector of a `KeyedFactorLayout` (default: `from_market_state(market)`, every curve, surface and asset it holds). One `jax.value_and_grad`; the full `jax.hessian` only if `second_order=True` (default `False`). `method="auto"`, `config=None`, `key` threaded unchanged into every evaluation. Quoting bumps: `key_rate_dv01` / `dv01` / `dividend_delta` per `BASIS_POINT = 1e-4`, `vega` per `0.01`, `spot_delta` per unit of the shock convention. |
| **Assumptions** | The rate convention is whatever `bump_graph_curve` means, to all orders: a continuously compounded additive zero-rate bump at each pillar, $DF_i \leftarrow DF_i\, e^{-\Delta r_i t_i}$, with $t_i$ on the curve's own day count; between pillars the curve's log-linear interpolation spreads the bump. Vol is a parallel shift (`shift_surface`), defined for every surface type. |
| **Limitations** | Parallel vega only: one scalar per surface, no smile or term-structure buckets (`shift_surface` takes a scalar). A model carrying its own scalar `rate` / `vol` / `dividend` shadows the market object, so those sensitivities are exactly zero, with a `UserWarning`; the risk is on `group_model_gradient` (`TestSensitivitiesContract`). Hessian cost grows with the square of the factor count. *Inference:* a pillar at $t = 0$ has a structurally zero key-rate sensitivity ($e^{-\Delta r \cdot 0} = 1$). |
| **Validation** | `tests/test_risk/test_state_sensitivities.py`: central FD of the same shock map — curve pillars `h=1e-6`, `rel=1e-4, abs=1e-6`; vega `h=1e-5`, `rel=1e-4`; spot delta `h=1e-4`, `rel=1e-4`; PV equals `group_pv`, `rel=1e-12`. Hessian symmetric `atol=1e-8`. Per-curve DV01s add to the joint move `rel=1e-10`; key-rate DV01s sum to the parallel DV01 `rel=1e-12`; an unused projection curve carries `atol=1e-12` zero risk; basis DV01 equals the projection DV01 `rel=1e-12`. `filter_jit` equals eager `rtol=1e-12`. |
| **Improve** | Node-level vega and further factor kinds — [W1b](../roadmap.md#44-frtb-and-desk-risk-programme); all-orders P&L along a direction via revaluation grids — [R-11](#r-11-revaluation-grids). How-to: [Bucketed Sensitivities § Keyed](../risk-sensitivities.md#keyed-sensitivities-on-a-marketstate); maths: [Theory § 7.4](../theory/risk-measures.md#74-bucketed-sensitivities). |

### R-2 Flat sensitivities (legacy)

| | |
|---|---|
| **Entry points** | `compute_flat_sensitivities`, `FlatSensitivities` |
| **Definition & defaults** | `jax.grad` in discount-factor space, converted by hand: $\partial V/\partial r_i = \partial V/\partial DF_i \cdot(-t_i DF_i)$, and $\partial^2 V/\partial r_i^2 = H^{DF}_{ii}(t_i DF_i)^2 + g^{DF}_i\, t_i^2 DF_i$. Separate Hessian blocks, each contracted to its diagonal (`gamma_spot`, `volga`, `gamma_rate`, and `vanna` as asset $i$'s spot × asset $i$'s vol); full `cross_spot_rate` and `cross_vol_rate` blocks. No options. |
| **Assumptions** | One discount curve is the only rate factor; vols are one scalar per asset. |
| **Limitations** | Diagonal second order only: no rate × rate off-diagonals, no cross-asset spot or vanna terms, no volga off-diagonals, no dividend second order or dividend cross terms (the `FlatSensitivities` fields hold none). The pricing closure builds `MarketData` **without `curve_graph`** (`valax/risk/sensitivities.py`), while full repricing (`reprice_under_scenario`) keeps it — *inference:* for a pricer that reads the graph, sensitivities and HPL are taken on different markets. The test pricer collapses the curve to one short rate with a fixed `/365` day count. |
| **Validation** | `tests/test_risk/test_sensitivities.py`: shapes and signs; agreement with `pnl_attribution` (delta `rtol=1e-4`, gamma `rtol=1e-3`). No finite-difference comparison in this file. |
| **Improve** | Superseded by [R-1](#r-1-keyed-sensitivities); removal with `MarketData` — [measurement-layer port](../roadmap.md#1-risk-engine-keyed-end-to-end-bar-portfolio-aggregation). |

### R-3 Waterfall P&L explain

| | |
|---|---|
| **Entry points** | `keyed_waterfall_pnl`, `keyed_waterfall_report`, `KeyedWaterfall`; legacy `waterfall_pnl`, `waterfall_pnl_report`, `WaterfallPnL`, `pnl_attribution` |
| **Definition & defaults** | Second-order Taylor explain $\Delta V \approx \delta^\top x + \tfrac12 x^\top H x$ in the sensitivities' own coordinates. Keyed: per-block first order $\delta_b^\top x_b$, per-block own gamma $\tfrac12 x_b^\top H_{bb} x_b$, and `cross_pnl` = everything else, one scalar. `keyed_waterfall_report` defaults to `second_order=True` and reprices with the same key. Flat: ten fixed terms (four deltas, spot and rate gamma, vanna, volga, spot × rate, vol × rate). |
| **Assumptions** | The move is small enough for a second-order expansion; the scenario is expressed in the sensitivities' coordinates (keyed: a scenario id outside the layout raises `ValueError`). |
| **Limitations** | Truncation at second order by construction: the `unexplained` residual grows with the move and is the signal that a revaluation grid is needed. Keyed cross terms are not attributed per pair (one scalar). Flat explain omits every term [R-2](#r-2-flat-sensitivities-legacy) omits. `pnl_attribution` has spot gamma only; it computes a second Hessian it never uses (`valax/risk/var.py`, the line commented "not quite right"), wasted work; its docstring term list leaves out `delta_div`, which is included in the total. |
| **Validation** | `tests/test_risk/test_state_sensitivities.py::TestWaterfall`: 25 bp basis widening on a two-swap book, actual P&L 9,817.3 (`rel=1e-3`) with `|unexplained| / |actual| < 1e-5` at second order; terms close (`rel=1e-10`, `rel=1e-8`, `rel=1e-12`); a pure basis move puts exactly zero on the OIS term. `tests/test_risk/test_sensitivities.py`: terms sum to predicted `atol=1e-10`; small scenario relative error `< 0.01`; second order beats first on a large scenario. `tests/test_risk/test_var.py`: `pnl_attribution` unexplained `< 0.1·|actual| + 0.01`. |
| **Improve** | Per-pair cross attribution (open); all-orders explain along a direction — [R-11 revaluation grids](#r-11-revaluation-grids). How-to: [Bucketed Sensitivities § Waterfall](../risk-sensitivities.md#waterfall-pl-decomposition). |

## VaR and Expected Shortfall

### R-4 Delta-normal VaR and component VaR

| | |
|---|---|
| **Entry points** | `keyed_parametric_var`, `keyed_component_var`, `block_covariance`, `historical_block_covariance`, `psd_factor`; legacy `parametric_var` |
| **Definition & defaults** | $\mathrm{VaR} = z_\alpha \sqrt{\max(\delta^\top\Sigma\delta, 0)\, h}$ with $z_\alpha$ = `-norm.ppf(1 - confidence)`; `confidence=0.99`, `horizon=1.0` (square-root-of-time in units of the covariance's period). Component VaR: Euler weights $\delta \odot \Sigma\delta / \max(\delta^\top\Sigma\delta, 10^{-300})$ summed per block, adding to the total. `block_covariance`: per-block covariances plus one scalar correlation per block pair (unstated pairs uncorrelated; absent blocks deterministic), symmetrised, optional `jitter=0.0`. `historical_block_covariance`: sample covariance, $n-1$ denominator. |
| **Assumptions** | P&L linear in the factors and factors jointly normal; static portfolio over the horizon; i.i.d. returns for square-root-of-time. |
| **Limitations** | First order: wrong for convex books (the gap to full-revaluation VaR is the convexity diagnostic). `_checked_cov` checks shape only — no symmetry or PSD check; the clamp acts on the scalar variance, so *inference:* an indefinite covariance can give a finite but wrong VaR, or zero, silently. Component VaR is reported as 0 at zero total variance. Cross-block correlation is one scalar per pair, no per-pillar structure. Legacy `parametric_var`: one rate block (named curves "contribute no delta and their risk is not in the number", per its docstring) and no horizon argument. |
| **Validation** | `tests/test_risk/test_state_sensitivities.py::TestKeyedVaR`: closed form `rel=1e-12`; 10-day equals $\sqrt{10}$ × 1-day `rel=1e-10`; components sum to the total `rel=1e-10`; a zero-variance basis block contributes `abs=1e-8` zero; delta-normal within `rel=0.05` of the empirical VaR of 20,000 `keyed_parametric_scenarios`. |
| **Improve** | PSD validation of user covariances (open); marginal / incremental VaR per trade — [roadmap § 4.1](../roadmap.md#41-var-and-expected-shortfall). Maths: [Theory § 7.1](../theory/risk-measures.md#71-value-at-risk-var). |

### R-5 Full-revaluation VaR and ES, and the scenario generators

| | |
|---|---|
| **Entry points** | `value_at_risk`, `expected_shortfall`, `group_scenario_pnl`, `keyed_parametric_scenarios`, `keyed_historical_scenarios`; legacy `portfolio_pnl`, `parametric_scenarios`, `historical_scenarios` |
| **Definition & defaults** | `value_at_risk(pnl, confidence=0.99)` $= -$`jnp.percentile(pnl, (1 - confidence)·100)`, a positive loss threshold; *inference:* NumPy's default linear interpolation between order statistics. `expected_shortfall(pnl, confidence=0.99)` = minus the mean of P&Ls `<= -VaR` (ties at VaR included). `group_scenario_pnl` computes the base PV once and `vmap`s shock-and-reprice over the scenario axis with one key. `keyed_parametric_scenarios(key, cov, layout, n_scenarios, distribution="normal", df=5.0)` factors `cov` with `psd_factor` (eigen-decomposition, negative eigenvalues clamped to 0); `"t"` scales the normal draw by $\sqrt{\nu/\chi^2_\nu}$, one mixing variable per scenario. `keyed_historical_scenarios(returns, layout)` reshapes given returns. |
| **Assumptions** | The scenario set represents the horizon's factor moves (no horizon argument: the caller scales returns or covariance); full revaluation is exact for the factors that are shocked. |
| **Limitations** | No test pins the quantile convention or an ES value against a known number. ES is the plain empirical tail mean, no fractional-observation weighting at the VaR point. `distribution="t"` is **not rescaled** to unit variance, so *inference:* the scenario covariance is $\nu/(\nu-2)\,\Sigma$ (×1.67 at the default $\nu = 5$), not $\Sigma$. Legacy `parametric_scenarios` factors with Cholesky (NaN on a singular covariance). Factors outside [R-X3](#cross-cutting-assumptions) are not shocked and contribute no P&L. |
| **Validation** | `tests/test_risk/test_var.py`: VaR > 0, ES ≥ VaR, monotone in confidence (to `1e-6`). `tests/test_risk/test_state_var.py`: bit-exact against the legacy path on single-curve books; `group_scenario_pnl` equals one-by-one repricing exactly; 25 bp basis P&L 9,817.3 `rel=1e-3`; common random numbers reprice identically. `tests/test_risk/test_keyed_scenarios.py`: `t` tails fatter than normal (`df=3`); shape checks. |
| **Improve** | Rescale the `t` generator to the target covariance (open); liquidity-horizon ES and stressed calibration — [W4](../roadmap.md#44-frtb-and-desk-risk-programme). Maths: [Theory § 7.2](../theory/risk-measures.md#72-expected-shortfall-cvar). |

## FRTB P&L vectors and tests

### R-6 HPL / RTPL vectors

| | |
|---|---|
| **Entry points** | `risk_theoretical_pnl_vector`, `hypothetical_pnl_vector`, `explained_unexplained_vector` |
| **Definition & defaults** | RTPL: the ten-term flat waterfall of [R-3](#r-3-waterfall-pl-explain) batched over a `ScenarioSet` (multiplicative spots honoured). HPL: alias of `portfolio_pnl` (full revaluation). Unexplained = HPL − RTPL. |
| **Assumptions** | Those of [R-2](#r-2-flat-sensitivities-legacy) and [R-3](#r-3-waterfall-pl-explain). |
| **Limitations** | Flat `MarketData` only, one rate block, diagonal gammas; RTPL is built without the `curve_graph` that HPL keeps (see R-2). No position weighting, so not a desk-level PLA input ([R-X1](#cross-cutting-assumptions)). Keyed full-revaluation HPL exists (`group_scenario_pnl`); keyed RTPL does not. |
| **Validation** | `tests/test_risk/test_pnl_vectors.py`: RTPL equals per-scenario `waterfall_pnl` `atol=1e-8`; HPL equals `portfolio_pnl` `atol=1e-10`; small scenarios relative error `< 0.05`. |
| **Improve** | Keyed RTPL / HPL — [W1c](../roadmap.md#44-frtb-and-desk-risk-programme); desk-level vectors — [W1a, W4](../roadmap.md#44-frtb-and-desk-risk-programme). Maths: [Theory § 7.5](../theory/risk-measures.md#75-pl-vectors-hypothetical-risk-theoretical-actual). |

### R-7 PLA test

| | |
|---|---|
| **Entry points** | `pla_spearman`, `ks_statistic`, `pla_ks`, `pla_traffic_light` |
| **Definition & defaults** | Spearman: Pearson correlation of ranks (denominator floored at `1e-12`). KS: $D = \max|F_x - F_y|$ over right-continuous ECDFs on the joint sample. p-value: asymptotic Kolmogorov series (100 terms) at $\lambda = (\sqrt{n_e} + 0.12 + 0.11/\sqrt{n_e})D$, $n_e = nm/(n+m)$. Zones (BCBS d558): `spearman_green=0.80`, `spearman_amber=0.70`, `ks_green_p=0.264`, `ks_amber_p=0.055`, `n_obs=250`; the zone is the worse of the two tests. |
| **Assumptions** | RTPL and HPL have the same length `n_obs` and are aligned by date. |
| **Limitations** | Ranks are **ordinal — ties are not averaged**, so Spearman is biased when the series contain ties (e.g. zero-P&L days); the `_ranks` docstring contradicts itself on this. The KS p-value is asymptotic, unvalidated at small $n$; `pla_traffic_light` does not read the series lengths. No comparison against SciPy. The zone function returns strings, so it is not JIT-compatible. |
| **Validation** | `tests/test_risk/test_backtesting.py`: Spearman of monotone / reversed series `> 0.999` / `< -0.999`, independent `|ρ| < 0.1`; KS identical `< 1e-6`, shifted `> 0.4`; zone logic including worst-of-two; end to end, identical series are green. |
| **Improve** | Average ranks over ties and a SciPy reference test (open); desk-level PLA — [W4](../roadmap.md#44-frtb-and-desk-risk-programme). Maths: [Theory § 7.7](../theory/risk-measures.md#77-frtb-pl-attribution-test). |

### R-8 VaR backtesting

| | |
|---|---|
| **Entry points** | `var_breaches`, `kupiec_pof`, `christoffersen_independence`, `christoffersen_conditional_coverage`, `basel_traffic_light` |
| **Definition & defaults** | Breach: $-\text{P\&L} > \text{VaR}$ (strict). Kupiec POF likelihood ratio (`confidence=0.99`), $\chi^2_1$ p-value; Christoffersen first-order Markov independence, $\chi^2_1$; conditional coverage = sum, $\chi^2_2$. Basel zones: green ≤ 4, yellow ≤ 9 breaches for 250 days at 99%; other $(n, \alpha)$ from the binomial CDF (green while CDF < 0.95, yellow while < 0.9999), which reproduces 4 / 9 at the standard setting. |
| **Assumptions** | Breaches i.i.d. under the null (Kupiec); first-order Markov alternative (Christoffersen). |
| **Limitations** | Asymptotic $\chi^2$ p-values only, no exact binomial test. No comparison against SciPy or published worked examples. The zone function returns strings (not JIT-compatible). |
| **Validation** | `tests/test_risk/test_backtesting.py`: Kupiec LR `< 1e-6` at the expected rate, `> 50` at 25%; Christoffersen clustered `> 10`, spaced `< 5`, zero with no breaches; conditional coverage equals the sum `atol=1e-6`; Basel boundaries 4 / 5 / 9 / 10 and the 500-day variant. |
| **Improve** | Exact binomial p-values and SciPy reference tests (open); desk-level backtesting — [W4](../roadmap.md#44-frtb-and-desk-risk-programme). Maths: [Theory § 7.6](../theory/risk-measures.md#76-var-backtesting). |

## Coordinates

### R-9 Bucketing and factor reparameterisation

| | |
|---|---|
| **Entry points** | `BucketMap`, `aggregate`, `pushforward_scenario`, `aggregate_covariance`, `aggregate_matrix`, `pushforward_sensitivities`, `pullback_shocks`, `reparameterize_covariance`, `jacobian_from_fn`, `tenor_bucket_map`, `equal_weight_bucket_map`, `level_slope_curvature_jacobian`, `pca_jacobian`, `bucket_sensitivities`, `BucketedSensitivities` |
| **Definition & defaults** | Linear aggregation $\delta_b = A\delta_x$, $\Delta x = A^\top \Delta b$, $\Sigma_b = A\Sigma_x A^\top$ (P&L-invariant by construction); Jacobian reparameterisation $\delta_b = J^\top\delta_x$, $\Delta x = J\Delta b$ (first order for nonlinear maps). `tenor_bucket_map(..., weight="indicator")` assigns each pillar to its nearest vertex; `"linear"` splits between neighbouring vertices, everything below the first / beyond the last going to that vertex. `pca_jacobian(returns, n_components, center=True)` by SVD, eigenvalues $s^2/(n-1)$. `level_slope_curvature_jacobian`: columns $[1,\ t - \bar t,\ t^2 - \overline{t^2}]$. |
| **Assumptions** | A bucket is a linear (or locally linear) image of the raw factors. |
| **Limitations** | `bucket_sensitivities` buckets diagonal gammas as $A\,\gamma_{\text{diag}}$, not $AHA^\top$, so cross-pillar gamma within a bucket is lost; vanna is bucketed by `spot_bucket` only; it reads `FlatSensitivities`, not keyed sensitivities. PCA loadings have arbitrary sign (no normalisation). Level / slope / curvature columns are not orthogonal. *Inference:* an equidistant pillar goes to the lower vertex under `"indicator"`. |
| **Validation** | `tests/test_risk/test_bucketing.py`: P&L invariance $\delta_b\cdot\Delta b = \delta_x\cdot\Delta x$ `rtol=1e-10`; Jacobian duality `rtol=1e-10`; linear-weight column sums `atol=1e-10`; PCA orthonormal `atol=1e-8`, low-rank reconstruction `atol=1e-6`; `bucket_sensitivities` preserves total P&L `rtol=1e-10`. |
| **Improve** | Keyed bucketing of first and second order ($AHA^\top$) — [W1c](../roadmap.md#44-frtb-and-desk-risk-programme); FRTB vertex and bucket tables — [W3](../roadmap.md#44-frtb-and-desk-risk-programme). How-to: [Risk Bucketing](../risk-bucketing.md); maths: [Theory § 7.8](../theory/risk-measures.md#78-risk-bucketing-linear-and-jacobian-transformations). |

### R-10 Keyed scenario shocks

| | |
|---|---|
| **Entry points** | `shock`, `shift_surface`, `KeyedFactorLayout`, `keyed_stress_scenario`, `keyed_curve_stress`, `keyed_basis_shift` |
| **Definition & defaults** | `shock(market, scenario)` applies the four `KeyedScenario` fields: spots `level·(1 + δ)` if `multiplicative` else `level + δ`; dividends additive; vols by `shift_surface` (a `ConstantVol` shifted in place, an existing `ShiftedVolSurface` accumulated, any other surface wrapped); curves by `bump_graph_curve` (the [R-1](#r-1-keyed-sensitivities) zero-rate convention). Unknown identifiers raise `KeyError`; `"_default_"` resolves only when the table has one entry. |
| **Assumptions** | A scenario is an additive (or, for spots, multiplicative) move in named risk factors. |
| **Limitations** | `MarketState.credit_curves`, `inflation_curves`, `fx_spots` and `fixings` have no branch in `shock`: they cannot be shocked through a `KeyedScenario` (the `SurvivalCurve` primitives in `valax/risk/shocks.py` exist but are unwired). No test asserts this. Vol shocks are parallel only. |
| **Validation** | `tests/test_risk/test_state_shocks.py`: unknown identifiers raise; purity; only the named curve moves; vol shifts accumulate without nesting; round trip equals legacy `apply_scenario` exactly; gradients flow to shock magnitudes and to the base market. |
| **Improve** | Keyed shocks for credit / inflation / FX and node-level vol — [W1b](../roadmap.md#44-frtb-and-desk-risk-programme). How-to: [Scenarios & Shocks](../risk-scenarios.md). |

### R-11 Revaluation grids

| | |
|---|---|
| **Entry points** | `revaluation_grid`, `RevaluationGrid` (`ladder`, `delta_ladder`, `index`), `revaluation_grid_pnl`, `GridPnL`, `key_rate_directions`, `parallel_direction`, `INTERPOLATIONS` |
| **Definition & defaults** | For each direction $d_k$ (a vector in `KeyedFactorLayout` coordinates, or a `KeyedScenario`) and each level $h_j$ of a shared, strictly increasing grid containing `0.0`: the PV of `shock(market, h_j d_k)` and its exact derivative in $h$ (one forward-mode `jax.jvp`), `vmap`-ped over directions × levels through `group_pv` / `price()`. `method="auto"`, `config=None`, `key` reused at every node. `revaluation_grid_pnl(..., interpolation="hermite")` reads the P&L of each move by cubic Hermite on (PV, delta); `"delta"` integrates the piecewise-linear delta. Beyond the outer rung: linear extrapolation from that rung, flagged in `out_of_range`. Optional `sensitivities=` adds the second-order cross-direction term $\tfrac12 x^\top Hx - \sum_k \tfrac12 m_k^2 d_k^\top H d_k$. |
| **Assumptions** | The P&L of a move along a direction is the integral of the delta along it; between rungs the value function is smooth enough for the interpolant (four continuous derivatives for the Hermite bound, three for delta integration). Directions with disjoint support add up. |
| **Limitations** | **All orders along each direction, second order across directions**: mixed third- and higher-order terms between directions are outside the number (a 2-D grid is [planned](../roadmap.md#1-risk-engine-keyed-end-to-end-bar-portfolio-aggregation)). Outside the rungs only first order (linear extrapolation). A payoff kink between rungs (a barrier, a digital near expiry) violates the smoothness the error bounds need, so the interpolation error is no longer $O(w^4)$ there — refine the rungs around it. Overlapping directions give valid ladders but no total (NaN, with a `UserWarning`). Cost is `n_dirs × n_levels` priced-with-tangent evaluations, against one gradient for sensitivities. Factor coverage is that of [R-X3](#cross-cutting-assumptions). Levels and directions must be concrete (validated before tracing). |
| **Validation** | `tests/test_risk/test_revaluation_grids.py`: rung PVs equal full revaluation `rel=1e-12`; node deltas equal central FD `rel=1e-6`; Hermite P&L exact at the rungs `rtol=1e-12`; on an option book's +23.7 spot move the grid error is below 1/1000 of the second-order Taylor error (measured 5.9e-4 against 1.76); halving the spacing cuts the error by more than 8× (Hermite) and 3.5× (delta), measured 25× / 9× and 3.8× / 4.1×; on a dual-curve swap book (+100 bp on both curves) the cross terms bring the error from > 100 (measured 966) to < 0.1 (measured 0.027) and below a tenth of the Taylor error; out-of-range extrapolation exact; `filter_jit` equals eager; the guide example against full revaluation `rel=1e-5`. |
| **Improve** | Two-dimensional product grids — [roadmap § 1](../roadmap.md#1-risk-engine-keyed-end-to-end-bar-portfolio-aggregation); FRTB curvature on top of three-rung grids — [W3](../roadmap.md#44-frtb-and-desk-risk-programme). How-to: [Revaluation Grids](../risk-revaluation-grids.md); maths: [Theory § 7.4](../theory/risk-measures.md#revaluation-grids-pl-beyond-second-order). |

## Numbers quoted elsewhere

Figures that the risk docs quote, and whether a test pins them.

| Figure | Where quoted | Pinned by |
|---|---|---|
| +9,817 P&L for a 25 bp basis widening (two-swap book) | roadmap, Risk: VaR & P&L | `approx(9817.3, rel=1e-3)` in `tests/test_risk/test_state_var.py`, `test_state_sensitivities.py`, `test_keyed_scenarios.py` |
| +0.92 for +100 bp on a 1Y two-option equity book (market-sourced rate) | roadmap | `approx(0.9216, abs=1e-3)` in `tests/test_core/test_binding.py` |
| Component VaR 5,922 OIS + 494 basis | roadmap, Bucketed Sensitivities | **Not pinned**: `TestKeyedVaR` asserts only that components add up and the basis share is positive |
| VaR(99%) 9,596.95 on a three-scenario example | Risk: VaR & P&L | **Not pinned** (consistent with linear-interpolation quantiles) |
