# Model Calibration

VALAX calibrates model parameters to market data using gradient-based optimization. Because all pricing functions are differentiable via JAX, the optimizer gets exact Jacobians for free — no finite-difference bumping.

This page covers the **end-to-end calibration workflow**: which liquid market instruments are taken as inputs, what calibrated object each one produces (a curve, a vol surface, a set of model parameters), and which exotic / structured instruments are then priced consistently against those calibrated objects.

!!! tip "Looking for a hands-on walk-through?"
    The [Rates End-to-End Tutorial](tutorial-rates.md) drives every stage on this
    page — synthetic ground-truth curve, noisy quotes, bootstrap, validation,
    pricing, and autodiff DV01 — in a single runnable example with the
    modelling assumptions called out at each step.

!!! tip "Chaining calibrations"
    This page covers the individual fits. To chain them — quotes → curves →
    smiles → model — with one convergence contract (`SolverSettings`,
    `CalibrationResult`) and quote-space risk through the whole chain, see
    [From Quotes to Risk](calibration-to-risk.md).

!!! info "Assumptions and limitations"
    This page is the how-to. What each calibrator fits and holds fixed, its
    objective, engine and defaults, its known limitations and its
    improvement path are recorded once, in the
    [Calibration Assumptions Register](../applications/calibration-register.md).

## 1. End-to-End Workflow at a Glance

Every VALAX calibration follows the same three-stage pipeline:

```
   ┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
   │   Market quotes          │    │   Calibration            │    │   Calibrated artefact    │
   │   (liquid hedges)        │──▶ │   (least-squares /       │──▶ │   (curve, surface,       │
   │                          │    │    Newton / LM / BFGS)   │    │    model parameters)     │
   └──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
                                                                              │
                                                                              ▼
                                                                  ┌──────────────────────────┐
                                                                  │   Pricing of exotics /   │
                                                                  │   structured products    │
                                                                  │   (consistent w/ hedges) │
                                                                  └──────────────────────────┘
```

The market instruments on the left are the **liquid hedging universe** — the things a trading desk can actually trade in size with tight bid-offers. The exotics on the right are the **illiquid book** — bespoke structures whose risk is decomposed into the calibrated artefacts and ultimately hedged with the liquid set on the left.

### Calibration matrix

The table below summarises the full pipeline VALAX supports today (✅) or plans to support (🟡). Click any entry for the in-depth guide.

| Market | Calibration inputs (liquid) | Calibration routine | Calibrated artefact | Exotics priced against it |
|--------|-----------------------------|---------------------|---------------------|---------------------------|
| **Rates — discount** | OIS deposits, OIS par swaps, FX forwards / FX swaps (cross-currency discount) | [`bootstrap_sequential` / `bootstrap_simultaneous`](curves.md#3-sequential-bootstrap) | `DiscountCurve` (OIS / €STR / SOFR) | All cashflow discounting; bonds; every swap PV |
| **Rates — projection** | Money-market deposits, FRAs, IMM futures (with convexity adj.), IBOR par swaps, tenor-basis swaps | [`bootstrap_curve_graph`](curves.md#5-multi-curve-bootstrap) ✅ | `CurveGraph` (one `DiscountCurve` per tenor, keyed by identifier) | IRS, FRN, OIS swap, CMS, range accrual, Bermudan swaption ✅ |
| **Cross-currency** | FX spot, FX forward points, cross-currency basis swaps (constant-notional and MTM variants) | [`bootstrap_curve_graph`](curves.md#5-multi-curve-bootstrap) ✅ | `CurveGraph` (dom + for OIS + tenor curves, all keyed by identifier) | Cross-currency swaps, quanto adjustments ✅ |
| **Inflation** | Zero-coupon inflation swaps (ZCIS), year-on-year inflation swaps (YYIS) | [`from_zc_rates`](curves.md#7-inflation-curves) | `InflationCurve` (forward CPI) | ZCIS, YYIS, inflation caps/floors ✅ |
| **Equity smile (per slice)** | Listed vanilla calls/puts on a single expiry (broker smile / OPRA chain) | [`calibrate_sabr`](#3-sabr-smile-calibration), [`calibrate_svi_slice`](vol-surfaces.md#svi-surface) | `SABRModel` or `SVIParams` for one expiry | European, American, digital, barrier, compound, chooser, Asian, lookback options |
| **Equity surface** | Vanilla calls/puts across an (expiry × strike) grid | [`calibrate_sabr_surface`](vol-surfaces.md#sabr-surface), [`calibrate_svi_surface`](vol-surfaces.md#svi-surface), [`GridVolSurface`](vol-surfaces.md#grid-surface) | `SABRVolSurface`, `SVIVolSurface`, `GridVolSurface` | Autocallables, worst-of, cliquet, variance swap, all path-dependent exotics |
| **Equity stochastic vol** | Listed vanilla call prices (mid) across expiries and strikes | [`calibrate_heston`](#4-heston-stochastic-vol-calibration) | `HestonModel` ($v_0, \kappa, \theta, \xi, \rho$) | Forward-starting options, cliquets, variance swaps, VIX-style products (MC) |
| **Rates volatility** | ATM swaption straddles, swaption cube ($K \times T_\text{exp} \times T_\text{tail}$), caps/floors | SABR cube on the curve forwards (`calibrate_swaption_cube_from_curves`, chain: `swaption_cube_node`) ✅; cap strip → optionlet SABR (`strip_optionlet_surface`, chain: `caplet_strip_node` → `optionlet_sabr_node`) ✅; [`calibrate_hull_white`](#6-dual-curve-rates-calibration-ois-discounting) (Jamshidian) and `calibrate_g2pp` ✅, single- **or** dual-curve; Hull-White per trade off the cube ([`swaption_basket_node`](calibration-to-risk.md#rates-per-trade-model-calibration)) ✅ | SABR swaption cube; `OptionletVolSurface`; `HullWhiteModel` $(a, \sigma)$; `G2PPModel` | Callable / puttable bonds ✅, Bermudan swaptions ✅, CMS spread options, range accruals |
| **FX volatility** | ATM straddles, 25Δ and 10Δ risk reversals + butterflies (broker strangles) | [`calibrate_sabr`](#3-sabr-smile-calibration) on the strike grid implied by the broker convention | FX `SABRVolSurface` / `SVIVolSurface` | FX vanilla, FX barrier, quanto, TARF (planned) |
| **Credit** | Par CDS spreads at standard IMM tenors (6M, 1Y, 3Y, 5Y, 7Y, 10Y) | Survival-curve bootstrap 🟡 | `SurvivalCurve` (piece-wise constant hazard) | Off-market CDS, index tranches (CDO), CVA |

The *Calibration inputs* column is what a desk actually pulls from Bloomberg / Refinitiv / broker runs every morning. The *Calibrated artefact* column is the JAX pytree VALAX returns. The *Exotics* column lists what you can then price with that pytree as an input — consistently with the original hedging instruments by construction.

## 2. Typical Market Instruments by Asset Class

This section spells out, for each calibration target, **what the inputs look like in real life** and which VALAX type represents them.

### 2.1 Rates curves (discount + projection)

The interest-rate calibration universe is the most heterogeneous; it combines money-market, futures, and swap quotes into a multi-curve set. Every quote type below is a `BootstrapInstrument` pytree in `valax/curves/instruments.py`.

| Market instrument | Tenor range | What it constrains | VALAX type |
|-------------------|-------------|--------------------|------------|
| Overnight deposit (O/N, T/N) | < 1 week | Front of the OIS curve | `DepositRate` |
| OIS / SOFR / €STR par swap | 1W – 50Y | OIS discount curve | `OISSwapRate` |
| LIBOR-replacement deposit (term SOFR, EURIBOR fixing) | 1M – 12M | Anchor of tenor forward curve | `DepositRate` |
| Forward Rate Agreement (FRA) | 1×4, 3×6, 6×9, 9×12, … | 3M/6M forward rate at a future date | `FRA` |
| Money-market future (Eurodollar / SOFR / EURIBOR) | Quarterly IMM out to ~3Y | 3M forward rate at IMM dates (with convexity adj.) | `MoneyMarketFuture` |
| IBOR par swap (3M, 6M floating leg) | 2Y – 50Y | Long-end 3M/6M forward curve | `IborSwapRate` |
| Tenor-basis swap (3M-vs-6M, 1M-vs-3M) | 1Y – 30Y | Spread between two tenor curves | `TenorBasisSwap` |
| FX forward / FX swap | O/N – 2Y | Front end of foreign discount curve via CIP | `FXForward`, `FXSwap` |
| Cross-currency basis swap (e.g. EUR/USD MTM) | 2Y – 30Y | Long end of foreign discount curve | `CrossCurrencyBasisSwap` |
| Turn-of-year / quarter-end | Specific dates | Calendar liquidity dislocations | `TurnInstrument` |

**Calibration output.** A `CurveGraph` — a dict of `DiscountCurve` values keyed by validated identifiers of the form `<CCY>.<INDEX>.<TENOR>[.<QUALIFIER>]` (e.g. `USD.SOFR.OIS`, `USD.SOFR.3M`, `EUR.EURIBOR.6M`) — returned as `bootstrap_curve_graph(...).value` on a `CalibrationResult` that carries the build's convergence, residuals, RMSE and Jacobian condition number (see [§3 Solvers](#curve-builds-report-through-the-same-result)). Any number of currencies and tenors can coexist in a single graph. The legacy single-currency `MultiCurveSet` (OIS + tenor-keyed forwards) remains as a view over a `CurveGraph`; see [Curves and Bootstrapping](curves.md) for the full bootstrap mechanics, the joint Newton residual system, and the runnable USD dual-curve example in §5.2.

**What you can price with it.** Every fixed-income instrument in `valax.instruments`: `ZeroCouponBond`, `FixedRateBond`, `FloatingRateBond`, `InterestRateSwap`, `OISSwap`, `CrossCurrencySwap`, `TotalReturnSwap`, `CMSSwap`, `RangeAccrual`, plus the discount step of every other pricer in the library.

### 2.2 Equity volatility surface

| Market instrument | What you observe | VALAX calibration input |
|-------------------|------------------|--------------------------|
| Listed vanilla European calls and puts | Implied volatility for each (strike, expiry) on the OPRA / EOD chain | `strikes`, `market_vols`, `forward`, `expiry` arrays |
| Broker smile snaps (sell-side composite) | A clean per-expiry smile with explicit ATM, 25Δ, 10Δ wings | Same — one slice per expiry |
| Variance swap fair strikes | Model-free at-the-money variance | Used as a cross-check, not as a primary input |

**Calibration routines.**

| Routine | Use when |
|---------|----------|
| [`calibrate_sabr`](#3-sabr-smile-calibration) | Single-expiry parametric fit; standard for rates and FX |
| [`calibrate_sabr_surface`](vol-surfaces.md#sabr-surface) | Multi-expiry SABR cube; smooth across expiries |
| [`calibrate_svi_slice`](vol-surfaces.md#svi-surface) | Single-expiry SVI (Gatheral); better wings for equity |
| [`calibrate_svi_surface`](vol-surfaces.md#svi-surface) | Multi-expiry SVI, one slice per expiry with total variance interpolated linearly in expiry. **No arbitrage constraints are imposed** (neither butterfly nor calendar) — see the [calibration register](../applications/calibration-register.md#svi-slice-and-surface) |
| `GridVolSurface(...)` | Direct construction when broker quotes are already smoothed |

**Calibration output.** A `SABRVolSurface`, `SVIVolSurface`, or `GridVolSurface` — all JAX pytrees, all callable as `surface(strike, expiry) -> vol`.

**What you can price with it.**

- *Vanillas* repriced consistently: European, American (PDE / lattice), digital, barrier, Asian, lookback.
- *Path-dependent exotics:* autocallable, worst-of basket, cliquet, variance swap, compound and chooser options — see [Equity Exotics](equity-exotics.md).
- *Vega risk:* `jax.grad` flows through the surface to give bucketed vega per (strike, expiry) node for grid surfaces, or per SABR / SVI parameter per expiry for parametric ones.

### 2.3 FX volatility

FX vol is quoted by brokers in a non-standard convention: ATM straddles plus 25Δ / 10Δ risk reversals (RR) and butterflies (BF / strangles). The conversion to the (strike, vol) form VALAX needs is mechanical but convention-sensitive (premium-adjusted vs unadjusted delta, spot vs forward delta). Once converted, calibration uses the same `calibrate_sabr` / `calibrate_svi_*` routines as equity.

| Broker quote | What it encodes |
|--------------|-----------------|
| ATM straddle vol $\sigma_\text{ATM}$ | Volatility at the delta-neutral straddle strike |
| 25Δ risk reversal $\text{RR}_{25}$ | $\sigma_{25\text{C}} - \sigma_{25\text{P}}$ — smile asymmetry |
| 25Δ butterfly $\text{BF}_{25}$ | $\frac{1}{2}(\sigma_{25\text{C}} + \sigma_{25\text{P}}) - \sigma_\text{ATM}$ — wing convexity |
| 10Δ RR and BF | Tail asymmetry and convexity |

See [Analytical Pricing § FX Options](analytical.md#fx-options-garman-kohlhagen) for the three delta conventions and the strike↔delta inverter.

**What you can price with it.** `FXVanillaOption`, `FXBarrierOption`, quanto adjustments, TARF (planned). See [FX Derivatives](fx.md).

### 2.4 Interest-rate volatility

| Market instrument | What it constrains | Calibration target |
|-------------------|--------------------|--------------------|
| ATM swaption straddles (per $T_\text{exp}, T_\text{tail}$) | Black/Bachelier ATM vol surface | Hull-White $(a, \sigma)$ via Jamshidian ✅ |
| Swaption cube ($K \times T_\text{exp} \times T_\text{tail}$) | Full swaption smile | SABR per (expiry, tail) slice |
| Cap / floor straddles | Caplet vol stripping → caplet term structure | Black-76 caplet vols |
| Eurodollar / SOFR options | Short-end caplet vol | Caplet vol stripping |

**Calibration output.** A SABR swaption cube (`calibrate_swaption_cube_from_curves`, or `swaption_cube_node` in a chain) and a calibrated `HullWhiteModel` via `calibrate_hull_white` — or, per trade off the calibrated cube, via `swaption_basket_node` (see [From Quotes to Risk](calibration-to-risk.md#rates-per-trade-model-calibration)). The Jamshidian decomposition turns each swaption into a portfolio of options on individual ZCBs, each priced by Black-76 on an integrated short-rate variance — fast enough to drive a least-squares fit of $(a, \sigma)$; see [API: Calibration](../api/calibration.md).

**What you can price with it.** `CallableBond`, `PuttableBond` ✅; `BermudanSwaption` (Hull-White PDE through `price`, or LSM on LMM paths) ✅; CMS caps/floors, range accruals, callable structured notes 🟡.

### 2.5 Credit (planned)

| Market instrument | What it constrains | Calibration target |
|-------------------|--------------------|--------------------|
| Par CDS spreads at standard IMM tenors (6M, 1Y, 3Y, 5Y, 7Y, 10Y) | Hazard-rate term structure $h(t)$ | `SurvivalCurve` (piece-wise constant hazard) |
| CDS index quotes (CDX IG/HY, iTraxx Main/Crossover) | Portfolio default expectation | Index hazard curve |
| Tranche base correlations (CDX 0–3, 3–7, …) | Gaussian-copula correlation skew | Base-correlation curve |
| Bond–CDS basis | Funding-adjusted hazard | Cross-check / spread basis |

**Calibration output.** A bootstrapped `SurvivalCurve`, recovery rate $R$ (typically 40 % for senior unsecured), and base-correlation curves for tranches. See [Credit Derivatives](credit.md).

**What you can price with it.** Off-market `CDS`, `CDOTranche`, and the credit leg of CVA / wrong-way risk calculations.

### 2.6 Inflation

| Market instrument | What it constrains | Calibration target |
|-------------------|--------------------|--------------------|
| Zero-coupon inflation swap (ZCIS) breakevens at 1Y, 2Y, 5Y, 10Y, 20Y, 30Y | Forward CPI level $I(T) / I(0)$ | `InflationCurve` pillars |
| Year-on-year inflation swap (YYIS) | YoY forward rate (with small convexity adj.) | `InflationCurve` short end + convexity |
| Inflation caps / floors (YoY) | YoY inflation vol | YoY Black-76 vol surface |

See [Inflation §7 in Curves](curves.md#7-inflation-curves) for curve construction — which is algebraic (`from_zc_rates`), not a bootstrap — and [Inflation Derivatives](inflation.md) for pricing.

## 3. SABR Smile Calibration

Fit SABR parameters ($\alpha$, $\rho$, $\nu$) to an observed volatility smile. Beta is typically fixed.

```python
import jax.numpy as jnp
from valax.calibration import calibrate_sabr

# Market data: strikes and observed implied vols
strikes = jnp.array([80., 85., 90., 95., 100., 105., 110., 115., 120.])
market_vols = jnp.array([0.28, 0.26, 0.24, 0.22, 0.21, 0.205, 0.21, 0.22, 0.235])
forward = jnp.array(100.0)
expiry = jnp.array(1.0)

# Calibrate with beta fixed at 0.5
result = calibrate_sabr(
    strikes, market_vols, forward, expiry,
    fixed_beta=jnp.array(0.5),
).check()                       # raises CalibrationError if the fit failed
fitted = result.value           # the SABRModel

print(f"alpha={float(fitted.alpha):.4f}")
print(f"rho={float(fitted.rho):.4f}")
print(f"nu={float(fitted.nu):.4f}")
```

Every calibrator in VALAX returns a `CalibrationResult` like this one: the
calibrated object is `.value`, and the same object carries the solved level
(`theta`, `inputs`, `residual_fn`) and its diagnostics.

### Solvers

Solver choice is data, not code: every calibrator takes
`settings: SolverSettings` and routes through the single factory
`run_solver`. A `SolverSettings` names the algorithm (`method`), `rtol`,
`atol`, `max_steps` and `throw`; all fields are static, so changing a
tolerance changes the compiled program but never the calibrator.

| Problem | `method` options (`"auto"` = first) |
|---|---|
| `"root"` (square systems: curves, strips, quote conversions) | `newton`, `chord`, `levenberg_marquardt`, `dogleg` |
| `"least_squares"` (smiles, Heston, model baskets) | `levenberg_marquardt`, `gauss_newton`, `dogleg`, `bfgs`, `optax_adam` |

**Defaults per calibrator family.**

| Family | Default settings | Why |
|---|---|---|
| Curve bootstraps (`bootstrap_curve_graph`, `curve_graph_node`) | `EXACT_SETTINGS` — Newton to 1e-10 | Square systems with an exact solution |
| Cap strips (`strip_caplet_vols`, `caplet_strip_node`), premium → vol (`swaption_implied_normal_vol`) | `STRIP_SETTINGS` / `INVERSION_SETTINGS` — scalar Newton to 1e-12, 100 steps | Square, one unknown per bucket / quote, well conditioned: solved to machine precision |
| Smiles and surfaces (`calibrate_sabr`, `calibrate_svi_*`, `calibrate_sabr_surface`, cube, optionlet surface), `calibrate_heston` | `FIT_SETTINGS` — Levenberg-Marquardt to 1e-8 | A fit residual does not vanish, and price-based fits have a noise floor near 1e-10 that a tighter step tolerance cannot get under |
| `calibrate_hull_white`, `calibrate_g2pp`, `swaption_basket_node` | `BASKET_SETTINGS` — line-searched BFGS to 1e-12 | The Jamshidian / Brigo-Mercurio closed forms nest a root-find whose implicit derivative is non-finite far from the solution, where an undamped LM step can land; BFGS's line search keeps the iterates where the pricer is defined. Tight, because the engines have no noise floor |
| `calibrate_cheyette`, `calibrate_cheyette_2f` | `CHEYETTE_SETTINGS` — BFGS to 1e-10 | As above, with the PDE's discretisation floor |
| `calibrate_slv_leverage` | `SLV_SETTINGS` — only `atol = 5e-2` is read | A Monte Carlo fixed point, not a solve; see [SLV](slv.md#how-the-fixed-point-is-reported) |

```python
from valax.calibration import SolverSettings

# BFGS instead of the default Levenberg-Marquardt
result = calibrate_sabr(
    strikes, market_vols, forward, expiry,
    fixed_beta=jnp.array(0.5),
    settings=SolverSettings(method="bfgs", rtol=1e-10, atol=1e-10),
)
```

**Convergence is traceable.** `throw` defaults to `False`, so a failed solve
never raises mid-trace: it is *reported* as a JAX boolean,
`result.diagnostics.converged` (or `result.converged`, which also covers any
child results), alongside `residuals`, `max_abs_residual`, `rmse` and
`n_steps`. That survives `jit` and `vmap` — a batch of per-trade calibrations
returns a batch of flags rather than failing the whole trace. Outside a trace,
`result.check()` raises `CalibrationError` on failure (optionally also on
`max_residual=` or `max_condition_number=`), and returns the result so it
chains; `calibrate_market(..., strict=True)` does the same for a whole chain.

**Failures are named.** Grids of independent fits — a smile per expiry, SABR
per cube cell, SVI per slice, a strip per strike column — are solved as one
*batched* level: each node is its own solve with its own `converged` flag,
all inside one `vmap`, and `labels` names the nodes. So a bad cell is reported
by name rather than as an index:

```python
cube = calibrate_swaption_cube_from_curves(...)   # one SABR fit per cell
cube.failed_labels()                              # e.g. ('5Yx10Y',)
cube.check()   # CalibrationError: ... did not converge at node(s) ['5Yx10Y'] ...
```

Labels follow the market's own tenors: `"2Y"` for a smile expiry, `"5Yx10Y"`
for a cube cell, `"cap 3Y"` for a strip bucket — the cap's **maturity** from
the valuation date, so it matches the quote key `USD.CAP.3Y.<bp>` — and
`"K=3.00% cap 3Y"` for a bucket of a cap-matrix strip (one label per
`(strike, bucket)`, row-major over the `(n_strikes, n_buckets)` diagnostics,
via `solve_batched_level(..., sub_labels=...)`). A curve build labels each
quote by its key (`curve_graph_node`) or `"<i>:<Type>"`
(`bootstrap_curve_graph`).

**Sequential solves name their first failure.** A cap strip is triangular:
bucket *i* reads the vols of buckets < *i*. A failed bucket installs **NaN**
(as a failed cube cell does — never a plausible finite vol), so every later
bucket of its column fails too. `failed_labels()` lists all of them;
`root_cause_labels()` — and the `check()` message, as "first failure(s)" —
names the one whose own solve failed:

```python
strip.failed_labels()      # ('K=-0.86% cap 2Y', 'K=-0.86% cap 3Y')
strip.root_cause_labels()  # ('K=-0.86% cap 2Y',)
```

**Joint solves are localised, not blanket-blamed.** A curve build is one
Newton solve, so a failure has no per-node flag; `failed_labels()` attributes
it as follows (see `CalibrationResult.failed_labels`):

1. a non-finite **quote** names exactly that quote (it poisons every
   residual, so residuals cannot localise it);
2. an explicit `max_residual=` names every node above it (absolute semantics);
3. otherwise the **worst cluster** — nodes within a factor
   `FAILURE_CLUSTER_FRACTION` (0.5) of the largest residual; if the solve
   diverged to NaN everywhere (an infeasible quote, e.g. a −50% OIS rate),
   the cluster is taken on the residuals at the starting point
   (`diagnostics.extras["initial_residuals"]`), where the off-market quote
   stands out. That last step is a heuristic: treat it as "start here".

In example 16 a NaN or infeasible curve quote is named alone, and
`max_steps=1` names the four long-end quotes rather than all 14.

**Results from `jit` / `vmap` keep their diagnostics.** Trade data a residual
reads — a per-trade basket, the model template, weights — travels in
`LevelInputs.data` rather than in the static residual's closure, so
`result.condition_number()` works on a fit returned from
`eqx.filter_jit` (and inside `eqx.filter_vmap`). `data` is not a risk input:
`chain_adjoint` and the quote-risk paths ignore its cotangent.

**Engine provenance (Cheyette).** A fitted model is a property of the model
only if the pricing engine it was fitted on is exact or converges under
refinement. Frozen-`y` Cheyette carries a *modelling* bias, so its fitted
parameters are only valid on that engine and grid. Each result records its
engine as `result.provenance`, and `check_engine` refuses a mismatch where it
matters:

```python
from valax.calibration import calibrate_cheyette, check_engine

fit = calibrate_cheyette(basket, targets, curve, engine="frozen", config=cfg).check()
check_engine(fit, "pde", cfg)        # OK: same engine, same grid
check_engine(fit, "pde", finer_cfg)  # CalibrationError: modelling-bias engine
```

Exact-`y` fits (`engine="exact"`) are recorded as `discretised` and Hull-White
/ G2++ fits as `exact`, for which `check_engine` accepts any engine.
`price()` never consults provenance — the check is an explicit, eager opt-in.

#### Curve builds report through the same result

`bootstrap_curve_graph` returns the same root level as the chain's
`curve_graph_node`. The desk diagnostics of a build map onto it as:

| What you want | Where it is |
|---|---|
| Calibrated `CurveGraph` | `result.value` |
| Converged? residuals, RMSE, iterations | `result.diagnostics` (`result.check()` raises) |
| Jacobian condition number | `result.condition_number()` |
| Quoted values | `result.inputs.quotes` |
| Fitted quotes, native units | `fitted_curve_quotes(instruments, result.value, ref)` |
| Which quote broke the build | `result.failed_labels()` (NaN quote → that quote; otherwise the worst cluster) |

`quote_jacobian` raises `CalibrationError` on an unconverged build rather than
returning sensitivities of a system nobody solved.

### Weighted Calibration

Emphasize ATM strikes by passing per-strike weights:

```python
weights = jnp.exp(-0.5 * ((strikes - forward) / 10.0) ** 2)
fitted = calibrate_sabr(
    strikes, market_vols, forward, expiry,
    fixed_beta=jnp.array(0.5),
    weights=weights,
).value
```

## 4. Heston Stochastic-Vol Calibration

Fit Heston parameters $(v_0, \kappa, \theta, \xi, \rho)$ to **option prices** (not implied vols — Heston's natural output is a price via the characteristic-function pricer). Requires a pricing function to be injected — this allows using any Heston pricer (semi-analytic, Monte Carlo, or a neural surrogate).

```python
from valax.calibration import calibrate_heston

result = calibrate_heston(
    strikes, market_prices, spot, rate, dividend, expiry,
    pricing_fn=my_heston_pricer,
).check()
fitted = result.value     # HestonModel
```

Spot, rate, dividend, expiry and the strikes are held-fixed inputs of the
level, so its adjoint reports their sensitivities alongside the quote ones.

**Inputs.** A strip of vanilla European call (or put) prices observed across strikes for a single expiry, or stacked across multiple expiries (vectorised). Typically taken from the listed equity-index option chain (mid prices) or from a broker-cleaned mid surface.

**Output.** A calibrated `HestonModel` pytree — five parameters that jointly fit the smile and term structure.

**What you can price with it.** Anything whose payoff depends on the realised path of variance: forward-starting options, cliquets, variance and volatility swaps (analytic Heston VS strike), VIX-style payoffs, and any structured equity note whose risk is sensitive to vol-of-vol or the spot-vol correlation $\rho$.

## 5. Calibrated Outputs as Pricing Inputs

The whole point of calibration is that the outputs become the **inputs to the pricing of exotics**. The table below maps each calibrated artefact to the pricing functions / instruments that consume it.

| Calibrated artefact | Consumed by | Example exotics priced |
|---------------------|-------------|------------------------|
| `DiscountCurve` (OIS) | Every cashflow PV in the library | All bonds, swaps, options' discounting |
| `CurveGraph` (OIS + one `DiscountCurve` per tenor per currency) | `valax.pricing.analytic.rates`, `valax.pricing.mc.lmm` *(consumption via a single `DiscountCurve` argument today; see [`curves.md` §8](curves.md#8-what-is-not-yet-implemented) for the pricing-side gap)* | IRS, OIS swap, Bermudan swaption (LSM on LMM), CMS, range accrual |
| `InflationCurve` | `valax.pricing.analytic.inflation` | ZCIS, YYIS, inflation caps/floors |
| `SABRVolSurface` / `SVIVolSurface` / `GridVolSurface` | All Black-Scholes / Black-76 pricers, `valax.pricing.mc`, `valax.pricing.pde`, `valax.pricing.lattice` | European, American, digital, barrier, Asian, lookback, autocallable, worst-of, cliquet, variance swap |
| `SABRModel` (single slice) | `sabr_implied_vol`, `calibrate_sabr_surface` (as initial guess) | One-expiry vanilla repricing, smile-consistent digital replication, CMS convexity adjustment |
| `HestonModel` | Heston semi-analytic pricer, `valax.pricing.mc.heston` | Forward-start options, cliquets, variance swaps, VIX-style products |
| `HullWhiteModel` + trinomial tree | `callable_bond_price`, `puttable_bond_price`, `bermudan_swaption_price` 🟡 | Callable / puttable bonds, Bermudan swaptions, callable structured notes |
| `SurvivalCurve` 🟡 | `valax.instruments.credit` pricers | Off-market CDS, CDOTranche, CVA |

**Differentiability all the way through.** Because every calibrated artefact is a JAX pytree with the calibrated parameters as dynamic leaves, `jax.grad` of an exotic price w.r.t. the *original market quotes* flows in one reverse-mode pass:

```
market quote  ──▶  bootstrap / calibrate  ──▶  artefact  ──▶  exotic price
     │                                                              │
     └──────────────── jax.grad / optimistix.ImplicitAdjoint ────────┘
```

This gives you, for free:

- **Bucketed DV01** of an exotic to each input swap quote (via `bootstrap_simultaneous`'s `ImplicitAdjoint`).
- **Bucketed vega** of an exotic to each (strike, expiry) node of the calibrated vol surface.
- **SABR-parameter vega** per expiry on a SABR cube.
- **Hull-White $(a, \sigma)$ sensitivity** of a callable bond.

See [Greeks & Automatic Differentiation](../theory/greeks.md#6-greeks-and-automatic-differentiation) and the [end-to-end example](../examples.md) (`examples/08_end_to_end_workflow.py`) for a full walk-through from synthetic market quotes to calibrated SABR fit to portfolio-level Greeks.

## 6. Dual-Curve Rates Calibration (OIS Discounting)

Post-2008 a swaption is discounted on OIS and its underlying swap projects a
*tenor* curve carrying a basis. The short-rate calibrators support that
directly.

!!! info "Multi-curve is an upgrade, not a replacement"
    Every calibrator defaults to the single-curve fit it always performed, and
    the single-curve code path is reached by a **trace-time** branch, so its
    numbers are unchanged bit-for-bit. Dual-curve is opt-in per call, one
    keyword: `forward_curve=`. The same principle holds across the library —
    see [Why Multi-Curve?](why-multicurve.md) and the
    [pricing contract](tutorial-unified-pricing.md#5-selectors-single-curve-for-free-multi-curve-on-purpose).

### Why it has to be a calibration feature

Hull-White and G2++ price dual-curve through a frozen deterministic tenor basis
([theory](../theory/multi-curve-short-rate.md)). If you *price* dual-curve but
*calibrate* single-curve, the basis has nowhere to go but into the fitted
volatility — you get a model that reprices your calibration set and is wrong
everywhere else. That violates the
[model/market consistency invariant](tutorial-unified-pricing.md#7-models-that-reference-the-market).

Both legs of the residual must see both curves:

| Leg | Function | What the projection curve changes |
|---|---|---|
| **Market targets** | `swaption_prices_from_vols(..., forward_curve=...)` | The forward swap rate $S$; the annuity still discounts on OIS |
| **Model prices** | `calibrate_hull_white(..., forward_curve=...)` | The floating leg's value at expiry, hence the exercise boundary |

### Recipe

??? note "Setup — the two curves and a small swaption grid"

    ```python
    import jax
    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    from valax.curves import DiscountCurve
    from valax.dates import ymd_to_ordinal
    from valax.instruments.rates import Swaption

    REF = ymd_to_ordinal(2025, 1, 1)


    def flat_curve(rate, n_years=21):
        pillars = jnp.array(
            [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(n_years)],
            dtype=jnp.int32,
        )
        times = (pillars - int(REF)).astype(jnp.float64) / 365.0
        return DiscountCurve(
            pillar_dates=pillars,
            discount_factors=jnp.exp(-rate * times),
            reference_date=REF,
            day_count="act_365",
        )


    def swaption(expiry_years, tail_years, strike=0.032):
        return Swaption(
            expiry_date=jnp.int32(int(ymd_to_ordinal(2025 + expiry_years, 1, 1))),
            fixed_dates=jnp.array(
                [int(ymd_to_ordinal(2025 + expiry_years + i, 1, 1))
                 for i in range(1, tail_years + 1)], dtype=jnp.int32),
            strike=jnp.array(strike),
            notional=jnp.array(1_000_000.0),
            is_payer=True,
            day_count="act_365",
        )
    ```

    `act_365` on both the curves and the trades is deliberate: the short-rate
    machinery builds its time axis from the *instrument's* day count while the
    curve uses its own, so matching them keeps the arithmetic clean.

```python
import jax.numpy as jnp
from valax.calibration import (
    calibrate_hull_white, hw_swaption_prices, swaption_prices_from_vols,
)

ois = flat_curve(0.030)         # discounting
projection = flat_curve(0.035)  # 3M projection — a 50 bp basis

surface = [swaption(e, t) for e, t in ((1, 5), (2, 5), (3, 5), (2, 10), (5, 10))]
market_vols = jnp.array([0.24, 0.23, 0.22, 0.22, 0.20])

# 1. Quoted vols -> target prices, using BOTH curves.
targets = swaption_prices_from_vols(
    surface, ois, market_vols, forward_curve=projection
)

# 2. Fit: discount on OIS, project off the tenor curve.
result = calibrate_hull_white(
    surface, targets, ois,
    fixed_mean_reversion=jnp.array(0.08),
    forward_curve=projection,
    curve_id="USD.SOFR.OIS",     # the market curve price() will bind
).check()
model = result.value

float(model.volatility)   # 0.010261
model.curve_id            # 'USD.SOFR.OIS'
model.initial_curve       # None — the fitted model is curve-free

# 3. Fit quality on its own targets (a kernel call: bind the curve explicitly).
from valax.core.binding import bind_curves

model_prices = hw_swaption_prices(bind_curves(model, ois), surface, projection)
# max relative error 5.9% — one factor cannot fit a whole surface exactly
```

`calibrate_g2pp` takes the same two keywords. `curve` is always the **discount**
curve (the model stays exact-fitted to it); `forward_curve` only ever projects.

**The fitted model is curve-free, and BFGS is the default.** The curves are
the level's *upstream* inputs — so the result's adjoint carries their risk —
but they are not stored on the returned model: `price()` binds the curve from
the `MarketState` at pricing time, which is what lets curve-free models of the
same type stack under `vmap`. The default solver is `BASKET_SETTINGS`
(line-searched BFGS to 1e-12, see [Solvers](#solvers)); a fixed mean reversion
is a held-fixed input whose sensitivity the level reports. To fit a model per
trade to a basket read off a calibrated cube instead of to caller-supplied
target prices, see
[From Quotes to Risk § Rates](calibration-to-risk.md#rates-per-trade-model-calibration).

### How much does ignoring the basis cost?

Running the *old* single-curve residual against those same dual-curve targets,
for the 50 bp basis above:

| | $\sigma$ |
|---|---|
| Dual-curve fit (correct) | 0.010261 |
| Single-curve residual on dual-curve targets | 0.016661 |

A **62 % error in the fitted volatility** — the basis absorbed into the vol,
exactly the failure this closes. (The effect is large here because a flat 50 bp
basis moves a 3.2 %-strike payer swaption a long way in the money; the direction
is general even where the magnitude is smaller.) A round-trip on
model-generated targets recovers $\sigma = 0.012000$ from a truth of 0.012.

### Recording the market linkage

Passing `curve_id` writes the discount curve's identifier onto the fitted model
as a static field. `price()` binds `initial_curve` from whatever
[`MarketState`](../api/core.md#marketstate) it is given — the curve named by
`curve_id`, else the instrument's discount curve — so pricing under a shocked
market needs no refit and no re-binding step:

```python
from valax.core import MarketState
from valax.core.pricing import price
from valax.curves.graph import CurveGraph

shocked = MarketState(
    curves=CurveGraph(curves={"USD.SOFR.OIS": flat_curve(0.040)})
)
pv = price(surface[0], model, shocked).price   # same params, shocked curve
```

The model's own gradient therefore has no curve leaf: curve risk is reported
once, on the market. `price()` refuses a model whose `initial_curve` arrives
pre-bound (that would shadow the market curve); kernel-level callers bind
explicitly with `bind_curves`, as above.

### Scope

| Calibrator | Dual-curve |
|---|---|
| `calibrate_hull_white` | ✅ frozen tenor basis |
| `calibrate_g2pp` | ✅ frozen tenor basis |
| `calibrate_cheyette`, `calibrate_cheyette_2f` | ❌ — **raises**, does not ignore |

The Cheyette PDE engine has no discount/projection split at all, so passing
`forward_curve` raises rather than silently returning a single-curve fit. A
wrong calibration with no signal is the one outcome worse than an unsupported
feature; dual-curve Cheyette is tracked on the
[roadmap](../roadmap.md#7-modelling).

Because the basis enters analytically, the least-squares Jacobian remains exact
autodiff — including sensitivity to the *projection* curve's pillars, which is
what makes basis risk differentiable rather than bumped.

## 7. How It Works

### Parameter Transforms

Model parameters have natural constraints (e.g., $\alpha > 0$, $-1 < \rho < 1$). Rather than using constrained optimization, VALAX reparametrizes to unconstrained space:

| Constraint | Transform | Inverse |
|-----------|-----------|---------|
| $x > 0$ | softplus | inverse softplus |
| $a < x < b$ | scaled sigmoid | logit |
| $-1 < x < 1$ | tanh | arctanh |

The optimizer works in unconstrained $\mathbb{R}^n$; results are mapped back to valid parameters automatically.

```python
from valax.calibration import positive, correlation, bounded

# Custom transforms
pos = positive()                    # x > 0
corr = correlation()                # -1 < x < 1
frac = bounded(0.0, 1.0)           # 0 < x < 1
```

### Loss Functions

Calibration minimizes weighted residuals in implied-vol space (default) or price space:

$$\min_\theta \sum_i w_i \left( \sigma^\text{model}(K_i; \theta) - \sigma^\text{market}(K_i) \right)^2$$

Vol-space calibration is more stable than price-space for SABR because it removes the option's moneyness dependence from the residuals.

## 8. Is the Fit Trustworthy? Conditioning at Every Level

A converged calibration is not automatically a usable one. Every VALAX
calibration is a residual driven to zero, $R(\theta; \text{inputs}) = 0$, and
every sensitivity taken through it (see
[quote-space risk](../api/greeks.md#quote-space-hedge-space-risk)) solves a
system built from $\partial R / \partial \theta$. How invertible that matrix is
decides how much of the double-precision mantissa survives: a level at
condition number $\kappa$ retains roughly $16 - \log_{10}\kappa$ significant
digits.

This is the failure mode the residual norm cannot see. A fit can land on a tiny
residual — solver happy, repricing exact — and still hand back quote deltas
whose leading digits are noise, because two parameters were nearly redundant
and the solver had no way to tell you so.

`level_condition_number` reports $\kappa$ for **any** calibration level, so the
curve bootstrap, the smile fit and a model fit are all measured on one scale
(every calibrator's result exposes it as `result.condition_number()`);
`condition_alert` tests it against the shared threshold
`ILL_CONDITIONED_THRESHOLD` (`1e10`) and returns a message, or `None` when the
level is healthy. It returns text rather than warning or raising, so the caller
picks the channel — log line, risk-report row, or a hard build failure.

```python
import jax
import jax.numpy as jnp

from valax.curves import CurveSpec, DepositRate, OISSwapRate, bootstrap_curve_graph
from valax.dates import ymd_to_ordinal
from valax.greeks import condition_alert, level_condition_number
from valax.greeks.vol_adjoint import sabr_slice_level
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol

ref = ymd_to_ordinal(2025, 1, 1)
ois = "USD.SOFR.OIS"
pillars = [int(ymd_to_ordinal(y, 1, 1)) for y in (2026, 2027, 2030)]
spec = CurveSpec(
    curve_id=ois, currency="USD",
    pillar_dates=jnp.array(pillars, jnp.int32), day_count="act_365",
)
quotes = [
    DepositRate(start_date=ref, end_date=pillars[0], rate=jnp.array(0.045),
                day_count="act_365", curves_touched=(ois,)),
    OISSwapRate(start_date=ref, fixed_dates=jnp.array(pillars[:2], jnp.int32),
                rate=jnp.array(0.047), day_count="act_365", curves_touched=(ois,)),
    OISSwapRate(start_date=ref, fixed_dates=jnp.array(pillars, jnp.int32),
                rate=jnp.array(0.050), day_count="act_365", curves_touched=(ois,)),
]
build = bootstrap_curve_graph(ref, [spec], quotes).check()

# Layer 1a — the build reports its own conditioning.
kappa = build.condition_number()
print(f"curve kappa      = {float(kappa):.3e}")
print("curve alert      =", condition_alert(kappa, level_name="USD.SOFR.OIS"))

# Layer 1b — any level, through the generic helper.
model = SABRModel(alpha=jnp.array(0.012), beta=jnp.array(0.5),
                  rho=jnp.array(-0.25), nu=jnp.array(0.45))
strikes = jnp.array([0.02, 0.03, 0.04, 0.05, 0.06])
forward, expiry = jnp.array(0.04), jnp.array(5.0)
vols = jax.vmap(lambda k: sabr_implied_vol(model, forward, k, expiry))(strikes)

fixed_beta = sabr_slice_level(model, strikes, vols, forward, expiry)
free_beta = sabr_slice_level(model, strikes, vols, forward, expiry, free_beta=True)
print(f"smile kappa      = {float(level_condition_number(fixed_beta)):.3e}")
print(f"  with beta free = {float(level_condition_number(free_beta)):.3e}")
```

```
curve kappa      = 1.137e+00
curve alert      = None
smile kappa      = 2.160e+04
  with beta free = 5.585e+06
```

Reading those numbers:

- **The curve is near-perfectly conditioned** ($\kappa \approx 1.1$). That is
  structural, not luck. An OIS floating leg telescopes, so each swap residual
  is $r \cdot A - (P(T_0) - P(T_n))$, and in $\log DF$ space the
  $P(T_n)$ term (order $0.9$) dominates the annuity term (order
  $r\tau P$, so order $0.05$). Each quote therefore very nearly pins its own
  maturity pillar. Crowding two pillars one day apart does *not* spoil this —
  measured, and gated, in `tests/test_greeks/test_level_conditioning.py`.
- **Freeing $\beta$ costs 260× of conditioning.** A SABR smile has three shape
  degrees of freedom and four parameters, so $\alpha$ and $\beta$ trade off
  against each other almost perfectly. This is the quantitative form of the
  advice in [§3](#3-sabr-smile-calibration) to fix $\beta$; both numbers are
  still far inside the threshold, so this is a *reported cost*, not a failure.

The same reading applies to a model fit: holding Hull-White's mean reversion
fixed (`fixed_mean_reversion=`) improves its conditioning by orders of
magnitude, which is why
[`calibrate_hull_white`](../api/calibration.md#valax.calibration.calibrate_hull_white)
recommends it.

!!! warning "This diagnostic forms a matrix; the sensitivity path does not"
    A condition number is a statement about a whole matrix, so there is no
    vector-Jacobian shortcut and `level_condition_number` materialises
    $\partial R/\partial\theta$. Call it **once per snapshot**, not per trade.
    `level_adjoint` and `curve_quote_deltas` remain Jacobian-free.

The one genuinely singular case worth knowing: quoting the **same instrument
twice** keeps the system square, so no shape check rejects it, but it makes two
residual rows identical and leaves one pillar unconstrained. The build fails to
converge and $\kappa$ comes back non-finite, which `condition_alert` reports
distinctly — the implicit function theorem does not apply at all there, so the
sensitivities are meaningless rather than merely imprecise.

## 9. What a Risk Report Has to Contain

A calibration chain produces three outputs, not one, and the two that are not
delta are the ones that go missing:

| Output | What it is | Where |
|---|---|---|
| $\bar q_c$ | Delta to each **curve quote** | [`curve_quote_deltas`](../api/greeks.md#quote-space-hedge-space-risk) |
| $\bar q_v$ | Vega to each **vol quote** | [`vol_quote_vegas`](../api/greeks.md#vega-to-volatility-quotes) |
| $\bar p$ | Sensitivity to each **fixed input** | `fixed_input_adjoints`, below |

### 9.1 Fixed inputs: the numbers nobody quotes

The fixed inputs $p$ are what the calibration holds still rather than solves
for: mean reversion, the SABR exponent $\beta$ and any displacement shift,
futures convexity adjustments, turn spreads, wing settings. They are neither
market quotes nor calibrated parameters, which is precisely why they tend to
appear in no report at all — and a number that moves the price but is
monitored nowhere is an unhedged position.

`FixedInput` records **where** a value lives, not the value. Those numbers
already sit on the model and surface pytrees; a container that copied them
would be a second source of truth, and the day it disagreed the report would
describe a model nobody priced with. Differentiating through the re-injection
makes the number belong, by construction, to the leaf that was priced.

```python
import jax.numpy as jnp

from valax.calibration.hull_white import calibrate_hull_white, hw_swaption_prices
from valax.core.binding import bind_curves
from valax.greeks import fixed_input_adjoints, fixed_inputs_from_paths
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price

# `curve`, `trade` and `basket` as built in the Hull-White recipe of §6, with
# the fitted model bound for kernel calls: `model = bind_curves(result.value,
# curve)`; `targets = hw_swaption_prices(model, basket)`.

p = fixed_inputs_from_paths(
    [("hull_white.mean_reversion", lambda m: m.mean_reversion)],
    units=("1/year",),
)

# (a) Direct: the model frozen at its calibrated parameters.
direct = fixed_input_adjoints(lambda m: hw_swaption_price(trade, m), model, p)

# (b) Total: mean reversion moves, and the model is re-fitted to the
#     same basket before pricing.
def recalibrate_and_price(m):
    fitted = calibrate_hull_white(
        basket, targets, curve, fixed_mean_reversion=m.mean_reversion
    ).value
    return hw_swaption_price(trade, bind_curves(fitted, curve))

total = fixed_input_adjoints(recalibrate_and_price, model, p)

print(f"p                        = {float(direct.values[0]):.3f} /year")
print(f"dV/dp, model frozen      = {float(direct.adjoints[0]):,.0f}")
print(f"dV/dp, model re-fitted   = {float(total.adjoints[0]):,.0f}")
```

```
p                        = 0.080 /year
dV/dp, model frozen      = -133,734
dV/dp, model re-fitted   = -20,258
```

**The two differ by a factor of 6.6, and choosing wrongly is worse than not
reporting.** Moving mean reversion changes what the calibration lands on:
$\sigma$ re-fits and offsets most of the direct effect. Quoting the frozen
number as the exposure overstates it more than six times over. Which one you
get is decided entirely by the function you hand to `fixed_input_adjoints` —
a plain pricer gives (a), a recalibrate-then-price gives (b) — so the choice
is explicit rather than accidental.

No implicit-function-theorem step is needed for either. A fixed input is held
constant *through* the solve, so differentiating the recalibration directly is
correct, and it is cheap because `optimistix` differentiates its own implicit
solution rather than taping the iterations.

!!! note "A fixed input must be a scalar leaf"
    `FixedInput.where` must select a whole, scalar leaf. Indexing into one
    (`lambda c: c.betas[0, 0]`) is not a lens `equinox.tree_at` accepts, and a
    grid is a *set* of fixed inputs rather than one. Both are refused with a
    message saying so. The per-node $\beta$ grid on a `SwaptionCube` is
    therefore reported through a scalar-$\beta$ `SABRModel`, not through the
    cube.

### 9.2 Provenance: which quote was which

`SnapshotProvenance` records, per quote, its type, units, source and
timestamp under one snapshot id, and labels a sensitivity vector with them:

```python
from valax.market import SnapshotProvenance

provenance = SnapshotProvenance.from_instruments(
    "USD.2025-01-01.EOD", ref, quotes,
    source="INTERNAL-EOD", timestamp="2025-01-01T17:00:00Z",
    quote_ids=("USD.ON.1Y", "USD.OIS.2Y", "USD.OIS.5Y"),
)
provenance.label(q_bar)       # (("USD.ON.1Y", -12.3), ...)
provenance.stale_quote_ids()  # quotes flagged stale or overridden
```

Type and units are derived from the instruments in solver order, so the names
and the numbers come from the same sequence and a reordered quote list cannot
mislabel the report; a length mismatch raises rather than zipping short.

The record is kept **alongside** `MarketState`, never inside it. Every field
is a string or an integer, so it is not a differentiated leaf — but it could
have been carried as a static field without being differentiated, and that
would still have been wrong: static fields participate in JIT cache keys, and
provenance changes on every snapshot by definition, so every jitted pricer
would recompile whenever a timestamp moved.
