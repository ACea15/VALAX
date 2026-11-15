# Risk Factors

This page is the canonical catalogue of every **risk factor** in VALAX: the inputs that, when they move, change portfolio value, and against which the engine measures Greeks and runs P&L attribution / VaR / PLA.

For the *workflow* (scenarios, shocks, VaR, backtests) see [Scenarios & Shocks](risk-scenarios.md). For the underlying maths see [Models & Theory §7](theory/risk-measures.md#7-risk-measures). This page focuses on **what the factors are**, **where they live in the code**, and **what is and isn't yet implemented**.

---

## 1. Conceptual Framework

### 1.1 What counts as a risk factor

A **risk factor** is any market observable that:

1. Is consumed by at least one pricing function in `valax/pricing/`, and
2. Can be moved independently of the other factors without violating no-arbitrage.

The fundamental unit is the factor's **change** ($\Delta x$), not its level — both because (a) most pricing functions are nonlinear in the levels but locally linear in the changes, and (b) covariance matrices, scenarios, and historical returns are naturally expressed as changes.

### 1.2 Factor identifier convention

Every factor gets a hierarchical string identifier of the form

```
<CATEGORY>.<SUBTYPE>.<CURRENCY?>.<UNDERLIER?>.<TENOR?>
```

For example:

| ID | Meaning |
|---|---|
| `EQ.SPOT.AAPL` | AAPL equity spot |
| `EQ.VOL.AAPL.ATM.1Y` | AAPL implied vol, ATM, 1Y expiry |
| `IR.OIS.USD.5Y` | USD OIS zero rate, 5Y pillar |
| `IR.SOFR.USD.3M.5Y` | USD 3M SOFR forward curve, 5Y pillar |
| `IR.BASIS.USD.SOFR3M.OIS.5Y` | 3M SOFR vs OIS basis, 5Y |
| `CRED.HAZARD.IBM.5Y` | IBM hazard rate, 5Y pillar |
| `FX.SPOT.EURUSD` | EUR/USD spot |
| `INF.BREAKEVEN.USD.10Y` | USD CPI breakeven, 10Y |
| `MOD.HESTON.AAPL.KAPPA` | Heston mean-reversion for AAPL |
| `CORR.EQ.AAPL.MSFT` | AAPL–MSFT spot correlation |

The identifier is what links a factor across **scenarios** (coordinates of a stacked `KeyedScenario`), **covariance matrices** (rows / columns of $\Sigma$), **sensitivities** (entries of `KeyedSensitivities.to_dict()`), and **PLA reports** (per-factor RTPL contributions — still positional, see below).

!!! note "Current code"
    Two layouts coexist. The **legacy** one is flat and *positional*
    (`[spots, vols, rates, dividends]`) inside `MarketData` /
    `MarketScenario`; it has one undivided `rates` block and therefore
    cannot name a curve. It is kept for one deprecation cycle, and the
    FRTB PLA vectors (`risk_theoretical_pnl_vector`) still run on it.

    The **current** one is identifier-keyed:
    [`KeyedFactorLayout`](api/risk.md#keyed-per-curve-factor-layout) holds
    one `FactorBlock` per named market object (every curve on the
    `CurveGraph`, every surface, every asset) and emits exactly the
    identifiers below from `layout.factor_ids()`. The same layout indexes
    scenarios ([`keyed_parametric_scenarios`](risk-scenarios.md#keyed-generators-and-the-per-curve-covariance-layout)),
    covariance matrices (`block_covariance`) and sensitivities
    ([`KeyedSensitivities`](risk-sensitivities.md#keyed-sensitivities-on-a-marketstate)), so
    the three line up by **name** rather than by position. Identifier
    coverage is still partial — the IR and equity families below are keyed;
    FX, credit, inflation and the model-parameter families are not yet
    (see [§4 Roadmap](#4-roadmap)).

### 1.3 Shock conventions

| Shock | Definition | Use cases |
|---|---|---|
| **Additive** | $x_{\text{new}} = x_{\text{old}} + \Delta x$ | Rates, vols, dividend yields, spreads, hazard rates (all natural in absolute units). |
| **Multiplicative / log** | $x_{\text{new}} = x_{\text{old}} \cdot (1 + r)$ | Spot prices, FX rates, equity indices (where daily returns are the natural primitive). |
| **Multiplicative** on shape, additive on level | $x_{\text{new}} = x_{\text{old}}(1 + r)$ for spot, additive on related vol/curve | Mixed-mode parametric MC where spot returns are log-normal but curves shift in absolute bp. |

VALAX defaults to additive shocks; multiplicative spot shocks are opted in via `KeyedScenario(multiplicative=True)` (or `MarketScenario(multiplicative=True)` on the legacy flat path).

### 1.4 Factor categories

| Category | Symbol | Examples |
|---|---|---|
| **Equity / Index price** | $S$ | AAPL spot, SPX index |
| **Interest rate (zero curve)** | $r$ | OIS, SOFR forward, term curves |
| **Interest-rate basis** | $b$ | SOFR vs OIS, 3M vs 6M, XCCY |
| **Credit spread / hazard** | $h$, $s$ | CDS spread, default intensity |
| **FX spot / forward** | $X$, $F^{FX}$ | EUR/USD, FX swap points |
| **Inflation / breakeven** | $\pi$ | CPI level, breakeven inflation |
| **Implied volatility** | $\sigma$ | ATM vol, vol surface points |
| **Vol-surface parameters** | $\alpha,\beta,\rho,\nu$ | SABR, SVI per-slice params |
| **Stochastic model parameters** | $\theta$ | Heston $(\kappa,\theta,\xi,\rho)$, Hull-White $(a,\sigma)$ |
| **Correlation** | $\rho_{ij}$ | Equity–equity, equity–FX, IR–IR |
| **Recovery rate** | $R$ | Per-name credit recovery |
| **Dividend yield** | $q$ | Continuous yield per asset |

The rest of this document inventories every factor by category.

### 1.5 How many factors — and why you reduce them

A real bank VaR system tracks thousands to tens of thousands of risk factors. The exact number depends on the desk and portfolio:

| Factor type | Typical count | Details |
|---|---|---|
| **IR curve pillars** (per curve) | 15-20 | Standard tenors: ON, 1W, 1M, 2M, 3M, 6M, 9M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y |
| **IR curves** (per currency) | 2-5 | OIS discount, SOFR/ESTR forward, basis, cross-currency basis |
| **Currencies covered** | 10-30 | G10 + major EM |
| **IR subtotal** | ~1,000-2,000 | 20 pillars x 3 curves x 20 currencies |
| **Equity spots** | 100-5,000+ | Individual names or sector indices |
| **Equity vol surfaces** | 500-10,000+ | 5-10 strikes x 5-8 expiries per underlier |
| **FX spots** | 20-30 | Against USD or EUR |
| **FX vol surfaces** | ~500 | 5 deltas x 8 expiries x 12 major pairs |
| **Credit spreads** | 500-5,000 | By issuer x tenor bucket |
| **Commodities** | 100-500 | Forward curves per commodity |
| **Inflation** | 50-100 | Breakeven curves per currency |

**Rates-only desk:** ~1,000-3,000 factors. **Multi-asset bank-wide VaR:** 5,000-50,000 factors. JP Morgan's firm-wide RiskMetrics system famously tracked 10,000+ factors.

Nobody runs VaR on 10,000 raw factors with a full covariance matrix — it would be rank-deficient (you rarely have 10,000 independent daily observations). Industry uses dimensionality reduction:

- **PCA on yield curves.** The first 3 principal components (level, slope, curvature) explain ~95-99% of yield-curve variance. Instead of shocking 20 pillar rates independently, shock 3-5 PC scores and reconstruct pillar moves — both more stable and more interpretable. See the [PCA Curve Shocks (Rates)](guide/pca-rates.md) guide for the end-to-end VALAX workflow, and [Risk Bucketing](risk-bucketing.md#jacobian-reparameterization) for the `pca_jacobian` reparameterization.
- **Factor models for equities.** Decompose returns into systematic factors (market, sector, style) plus idiosyncratic. Shock the systematic factors jointly, add independent idiosyncratic noise (the Barra/Axioma approach).
- **Filtered historical simulation.** Weight recent observations more heavily (EWMA), or fit a GARCH model per factor and standardize returns before drawing scenarios — captures volatility clustering without inflating the factor count.

The bucketing and Jacobian machinery that turns this fine-grained factor space into a coarser standardized one is documented in [Risk Bucketing](risk-bucketing.md#risk-bucketing); see also §7 below.

---

## 2. Implemented Factors

Legend: ✅ implemented, 🚧 partially implemented (factor structure exists but shock primitive missing or only via autodiff), 📋 planned, blocked by listed dependency.

### 2.1 Equity

| Factor ID | Description | Where it lives | Shock primitive | Sensitivity | Instruments | Pricing engines |
|---|---|---|---|---|---|---|
| `EQ.SPOT.<asset>` ✅ | Equity / index spot per asset. | `MarketState.spots[asset_id]` (legacy: `MarketData.spots[a]`) | `KeyedScenario.spot_shocks` (add. or mult.; legacy `MarketScenario.spot_shocks`) | `delta_spot`, `gamma_spot`, `vanna` | All equity options, equity barriers, Asian, lookback, autocallable, basket, spread, variance swap, TRS leg | `pricing/analytic/black_scholes.py`, `pricing/mc/*`, `pricing/pde/*`, `pricing/lattice/binomial.py` |
| `EQ.VOL.<asset>` ✅ | One vol number per surface / asset: on a `MarketState` the **parallel shift** of a whole surface (any surface type), on the legacy path a single per-asset ATM-like scalar. *Not* a node-level smile factor (see [§3.5](#35-volatility-surfaces)). | `MarketState.surfaces[surface_id]` (legacy: `MarketData.vols[a]`) | `KeyedScenario.vol_shocks` via `shift_surface` (legacy `MarketScenario.vol_shocks`, additive) | `delta_vol` (vega), `volga`, `vanna` | Same as above | Same as above |
| `EQ.DIV.<asset>` ✅ | Continuous dividend yield per asset. | `MarketState.dividends[asset_id]` (legacy: `MarketData.dividends[a]`) | `KeyedScenario.dividend_shocks` (legacy `MarketScenario.dividend_shocks`) | `delta_div` | Equity options, autocallables, TRS | `pricing/analytic/black_scholes.py`, `pricing/mc/*` |
| `EQ.CORR.<a>.<b>` 🚧 | Pairwise correlation between equity log-returns. Used by multi-asset MC. | `MultiAssetGBMModel.correlation` (a model parameter, not in `MarketState`) | Autodiff only (`jax.grad` w.r.t. correlation entry) | `corr_<a,b>` (via grad) | Spread options, worst-of basket, multi-asset MC | `pricing/mc/multi_asset_paths.py`, `pricing/analytic/spread.py` |

### 2.2 Interest Rates — Single Curve

| Factor ID | Description | Where it lives | Shock primitive | Sensitivity | Instruments | Pricing engines |
|---|---|---|---|---|---|---|
| `IR.ZERO.<ccy>.<pillar>` ✅ | Continuously-compounded zero rate at each curve pillar of a single-curve market — the `"_default_"` curve of a `MarketState` (e.g. `MarketState.flat(...)`), equivalently the legacy `MarketData.discount_curve`. For several named curves see [§3.2](#32-interest-rates-multi-curve). | `DiscountCurve.discount_factors` (one per pillar, log-DF storage) | `bump_curve_zero_rates`, `parallel_shift`, `key_rate_bump`; `KeyedScenario.curve_shocks["_default_"]`, `keyed_stress_scenario`, `keyed_steepener` / `keyed_flattener` / `keyed_butterfly`; legacy `MarketScenario.rate_shocks`, `steepener`, `flattener`, `butterfly` | `delta_rate` (per pillar = key-rate DV01s / KRD), `gamma_rate`, `cross_spot_rate`, `cross_vol_rate` | Bonds (ZC, fixed, FRN, callable, puttable), swaps, swaptions, caplets/caps, equity-option discounting, every other instrument | `pricing/analytic/bonds.py`, `pricing/analytic/swaptions.py`, `pricing/analytic/caplets.py`, `pricing/analytic/floating.py`, all other engines via the curve `price()` resolves from the `MarketState` |

### 2.3 Volatility (scalar form)

A `MarketState` holds full vol surfaces from `valax/surfaces/` (`GridVolSurface`, `SABRVolSurface`, `SVIVolSurface`, `ConstantVol`, …) keyed by surface id, and the keyed risk layer moves each one by a single **parallel shift** (`shift_surface`), so the vol factor is one number per surface — the [design choice](risk-scenarios.md#vol-shocks-parallel-shift-not-node-shock) that keeps it defined for every surface type. The legacy `MarketData.vols` stores one scalar per equity asset. Node-level (strike × expiry) and surface-parameter factors are **not yet** risk factors — see [§3.5](#35-volatility-surfaces).

### 2.4 Sensitivity-side coverage

Every factor in this section is automatically picked up by `keyed_sensitivities()` on a `MarketState` — one `KeyedSensitivities` block per named curve, surface and asset, read back by factor id ([Bucketed Sensitivities § Keyed sensitivities](risk-sensitivities.md#keyed-sensitivities-on-a-marketstate)). The legacy flat path, `compute_flat_sensitivities()` and `pnl_attribution()` over `MarketData`, produces the positional fields:

- `sens.delta_spot` ∈ ℝ^`n_assets`
- `sens.delta_vol` ∈ ℝ^`n_assets`
- `sens.delta_rate` ∈ ℝ^`n_pillars` (already converted to zero-rate space)
- `sens.delta_div` ∈ ℝ^`n_assets`
- second-order diagonal and cross blocks: `gamma_spot`, `gamma_rate`, `volga`, `vanna`, `cross_spot_rate`, `cross_vol_rate`

See [Bucketed Sensitivities](risk-sensitivities.md#bucketed-sensitivities-and-waterfall-pl) for the full schema.

---

## 3. Planned Factors

These factors have **either an instrument or a curve type in the codebase that needs them**, but the risk engine cannot yet shock them or attribute P&L to them. Ordered roughly by priority / unlocking power.

### 3.1 Credit

| Factor ID | Description | What is needed | Pricing engines that will consume it |
|---|---|---|---|
| `CRED.HAZARD.<name>.<pillar>` ✅ *(curve & shocks now in place; pricing engine TBD)* | Per-issuer hazard rate at each pillar. Survival probabilities $S(t)=\exp(-\int_0^t h)$. | `SurvivalCurve` ✅ in `valax/curves/survival.py`. `bump_hazard_rates`, `key_rate_hazard_bump`, `parallel_credit_spread_shift` ✅ in `valax/risk/shocks.py`. **Still needed:** CDS pricer that ingests a `SurvivalCurve`, and keyed shocks for `MarketState.credit_curves` (the primitives above are not yet wired into `KeyedScenario` / `shock`). | (planned) `pricing/analytic/cds.py` for `CDS`, `pricing/lattice/*` for `ConvertibleBond` with credit |
| `CRED.SPREAD.<name>.<pillar>` ✅ *(via `parallel_credit_spread_shift`)* | Par CDS spread per pillar; converted to hazard via $h\approx s/(1-R)$. | Implemented via the credit-triangle conversion in shocks; bootstrap-from-spreads available via `SurvivalCurve.from_cds_spreads`. | Same as above |
| `CRED.RECOVERY.<name>` 📋 | Per-name recovery rate (typically 40% for senior unsecured). Affects loss-given-default in CDS and convertibles. | Recovery as a scalar leaf on `CreditEntity` (TBD) or a static parameter on each `CDS`. | CDS, CDO |
| `CRED.BASE_CORR.<bucket>` 📋 | Base correlation per CDO tranche bucket. | Gaussian copula simulator and base-correlation lookup table. | `CDOTranche` |

### 3.2 Interest Rates — Multi-Curve

The modern curve container is `CurveGraph` (`valax/curves/graph.py`),
a flat dict of `DiscountCurve` values keyed by validated identifiers
of the form `<CCY>.<INDEX>.<TENOR>[.<QUALIFIER>]` (e.g.
`USD.SOFR.OIS`, `USD.SOFR.3M`, `EUR.EURIBOR.6M`).  The pedagogical
single-currency container `MultiCurveSet`
(`valax/curves/multi_curve.py`) — one OIS discount curve plus a
tenor-keyed dict of forward curves — is retained as a legacy view.
The delta-bucketing engine for multi-curve builds is
`quote_jacobian` (`valax/curves/bootstrap_graph.py`), which returns
a `(n_pillars_total, n_quotes)` sensitivity matrix through
`optimistix.ImplicitAdjoint` at the cost of one linear solve per
quote — this is what a production KRD / basis DV01 / xccy DV01
pipeline stands on.  The `CurveGraph` is what a `MarketState` carries
in `MarketState.curves`, so every curve on it is a market input to
`price(instrument, model, market)`.  On the legacy flat path,
`apply_scenario` shocks only the single
`MarketData.discount_curve` (plus the `"_default_"` entry of an
attached graph, which is the same curve); the shock primitives below
close the gap for the `MultiCurveSet` view, and the `CurveGraph`-aware
variants (`bump_graph_curve`, `parallel_graph_shift`,
`key_rate_graph_bump` in `valax/risk/shocks.py`) bump any named
curve on a graph — including one carried on the optional
`MarketData.curve_graph` field.  Since keyed scenarios landed, a
*named* curve is addressable from a scenario too:
`KeyedScenario(curve_shocks={curve_id: bumps})` applied with
`valax.risk.shock` to a `MarketState`
([guide](risk-scenarios.md#keyed-scenarios-on-a-marketstate)), and a whole
*distribution* of such moves is addressable through
`KeyedFactorLayout` — one block per named curve in the scenario
generators, the covariance matrix and the sensitivities alike
([guide](risk-scenarios.md#keyed-generators-and-the-per-curve-covariance-layout)).

| Factor ID | Description | What is needed | Pricing engines that will consume it |
|---|---|---|---|
| `IR.OIS.<ccy>.<pillar>` ✅ *(via `bump_discount_curve` on `MultiCurveSet`, or `bump_graph_curve` on `CurveGraph`)* | OIS / collateral discount-curve zero rate. | `bump_discount_curve` ✅ and `bump_graph_curve` ✅ in `valax/risk/shocks.py`; `MarketState.curves` ✅ holds the `CurveGraph` (the legacy `MarketData.curve_graph` carries one optionally); `KeyedScenario.curve_shocks` + `valax.risk.shock` ✅ address any graph curve from a scenario; `KeyedFactorLayout` ✅ gives each curve its own block in a scenario *set*, a covariance matrix (`block_covariance`) and a sensitivity set (`KeyedSensitivities`), so a projection curve carries its own VaR (`keyed_component_var`) and P&L term (`keyed_waterfall_pnl`). | All analytic rates pricers (dual-curve via the optional `forward_curve` argument; see [`guide/curves.md` §8](guide/curves.md#8-what-is-not-yet-implemented)). |
| `IR.FWD.<ccy>.<tenor>.<pillar>` ✅ *(via `bump_forward_curve` or `bump_graph_curve`)* | Tenor-specific forward curve (e.g. 3M SOFR forwards). | `bump_forward_curve(mcs, tenor, rate_bumps)` and `parallel_basis_shift` ✅ on `MultiCurveSet`; `bump_graph_curve(graph, curve_id, rate_bumps)` / `parallel_graph_shift` / `key_rate_graph_bump` ✅ on `CurveGraph`. | `pricing/analytic/floating.py` (FRN, OIS swap), `pricing/analytic/swaptions.py`, `pricing/analytic/caplets.py` — all with optional `forward_curve` |
| `IR.BASIS.<ccy>.<tenorA>.<tenorB>.<pillar>` ✅ | Forward-curve basis (e.g. 3M vs 6M, SOFR vs OIS): the spread $b$ in $r^{\text{proj}} = r^{\text{disc}} + b$. | `keyed_basis_shift(layout, projection_id, bump, over=discount_id)` ✅ takes a **basis** as input rather than a raw zero-rate bump. `as_basis_layout` ✅ makes it a *coordinate*, not just a scenario: it reparameterises the projection curve's block into a spread and names it with this identifier (`basis_factor_prefix`), so the basis gets its own variance in `block_covariance`, its own term in `keyed_waterfall_pnl`, its own `KeyedSensitivities.basis_dv01`, and its own `keyed_component_var` contribution — independently of discounting, which is how the market trades it. Gate: `tests/test_risk/test_state_sensitivities.py::TestBasisDV01`, `tests/test_risk/test_keyed_scenarios.py::TestBasisLayout`. With `quote_jacobian` the delta w.r.t. a `TenorBasisSwap.spread` is additionally available through the calibrated `CurveGraph`. | Basis swaps, dual-curve XCCY |
| `IR.XCCY.<ccy>.<pillar>` 🚧 | Cross-currency basis between two currencies' funding curves. | The same machinery as `IR.BASIS` now applies — `as_basis_layout` is currency-agnostic and `basis_factor_prefix` falls back to both full curve ids when the currencies differ — but the *pricer* is the gap: `cross_currency_swap_price` is single-curve-per-currency (each leg projects off its own discount curve). `price()` now resolves the second-currency curve (`CrossCurrencySwap.foreign_discount_id`) and the FX spot, so both curves' DV01 and the FX delta come out of one gradient, but no basis between a currency's projection and discount curve enters the price yet. `quote_jacobian` on a four-curve `CurveGraph` closed by a `CrossCurrencyBasisSwap` produces the xccy DV01 in one solve. | `CrossCurrencySwap` (pricer exists; per-leg curve selection TBD) |
| `IR.CONVEXITY.<ccy>` 📋 | Hull-White / quasi-Gaussian convexity adjustment for CMS, futures, in-arrears caps. | Reuse `valax/curves/convexity.py`; expose its parameters as risk factors.  The `MoneyMarketFuture.convexity_adj_fn` plug-in accepts `no_convexity_adj` / `constant_convexity_adj` today; `hull_white_convexity_adj(model)` queued for MC-Curves-3. | `CMSSwap`, `CMSCapFloor`, money-market futures |

### 3.3 FX

| Factor ID | Description | What is needed | Pricing engines |
|---|---|---|---|
| `FX.SPOT.<pair>` 🚧 | FX spot (domestic per foreign). On a `MarketState` it is its own pair-keyed leaf, `MarketState.fx_spots`, read by `price()` for `FXForward`, `FXVanillaOption` and `CrossCurrencySwap`, and its single-trade delta comes out of `valax.greeks.market_risk`. (The legacy `MarketData` has no FX field and piggybacks on `spots`.) | Keyed scenario shocks: `KeyedScenario` / `shock` do not address `fx_spots` yet, and `KeyedFactorLayout` has no FX block. | `pricing/analytic/fx.py`, `pricing/analytic/black76.py`, all FX options & forwards |
| `IR.OIS.<foreign_ccy>` 🚧 | Foreign / second-currency discount curve. Acts as the "dividend yield" in Garman-Kohlhagen. | The per-currency registry is the `CurveGraph` on `MarketState` (e.g. the EUR/USD four-curve `bootstrap_curve_graph` build), and FX trades name it via `foreign_discount_id`, so it is shocked and gets key-rate DV01s like any named curve. What is missing is the FX side of the pair — see `FX.SPOT` above. | All FX options (Garman-Kohlhagen), XCCY swaps |
| `FX.VOL.<pair>.<delta>.<expiry>` 📋 | FX vol surface (delta-strike convention). | A `FXVolSurface` type with delta-strike + expiry pillars, plus shocks. | `FXVanillaOption`, `FXBarrierOption`, `QuantoOption` |
| `FX.SMILE.<pair>.<param>` 📋 | Parametric FX-smile factors (risk reversal, butterfly, strangle) per delta/expiry. | Smile parametrisation (Vanna-Volga / SABR-FX) layered on top of the vol surface. | Same as above |

### 3.4 Inflation

| Factor ID | Description | What is needed | Pricing engines |
|---|---|---|---|
| `INF.CPI.<index>.<pillar>` 🚧 | Forward CPI level. `InflationCurve` (`valax/curves/inflation.py`) lives in the identifier-keyed `MarketState.inflation_curves`, is read by `price()` for the inflation swaps and caps/floors, and its single-trade sensitivities come out of `valax.greeks.market_risk`; it is not yet in the keyed risk scenarios. | Keyed shocks (`KeyedScenario` / `shock` do not address `inflation_curves` yet) and a `KeyedFactorLayout` block. Add `bump_cpi_levels`, `parallel_breakeven_shift`. | `pricing/analytic/inflation.py` for ZC/YoY inflation swaps and inflation caps/floors |
| `INF.BREAKEVEN.<index>.<pillar>` 🚧 | Zero-coupon breakeven inflation rate per pillar (the "inflation DV01" axis). | Shock primitive `parallel_breakeven_shift` and `key_rate_breakeven_bump`. | Same as above |
| `INF.SEASONALITY.<index>` 📋 | Monthly seasonality adjustments for CPI. | Seasonality vector + bumping primitive. | YoY swaps that span sub-annual periods |

### 3.5 Volatility Surfaces

A `MarketState` already carries full surfaces (`MarketState.surfaces`), but the keyed risk layer moves each by one parallel shift (§2.3), and the legacy `MarketData.vols` is a scalar per asset. Real equity / FX / rates desks shock a full smile / term structure.

| Factor ID | Description | What is needed | Pricing engines |
|---|---|---|---|
| `EQ.VOL.<asset>.<strike>.<expiry>` 📋 | Implied vol at every node of a `GridVolSurface`. | The surface is already in `MarketState.surfaces`. Needed: a node-level keyed shock (per-node bump and parametric ATM-level / skew / convexity / term-structure) and node-level, surface-keyed vega buckets — open on the [roadmap](roadmap.md#1-risk-engine-keyed-end-to-end-bar-portfolio-aggregation). | Equity options priced from a surface; future Dupire local-vol PDE; smile-aware MC. |
| `EQ.SABR.<asset>.<expiry>.{ALPHA,RHO,NU,BETA}` 📋 | SABR slice parameters per expiry. | A `SABRVolSurface` already sits in `MarketState.surfaces`; needed is a parameter-level shock (bump each parameter and reconstruct vol) wired into the keyed layout. | `pricing/analytic/sabr.py`, MC paths in `pricing/mc/sabr_paths.py`. |
| `EQ.SVI.<asset>.<expiry>.{A,B,RHO,M,SIGMA}` 📋 | SVI raw parameters per slice. | Same pattern as SABR. | Equity options on SVI-fitted surfaces. |
| `RATES.SWAPTION_VOL.<ccy>.<expiry>.<tenor>` 📋 | Black / Bachelier vol cube on the swaption grid. | `SwaptionVolCube` type + cube-bumping primitives. | `pricing/analytic/swaptions.py`. |

### 3.6 Model Parameters

These are second-tier factors: they are **inputs to the stochastic model** rather than directly market-quoted, but P&L attribution often allocates a residual to "model drift".

| Factor ID | Description | What is needed | Pricing engines |
|---|---|---|---|
| `MOD.HW.<ccy>.{A,SIGMA}` 📋 | Hull-White mean-reversion and volatility. | Model parameters as leaves of a `HullWhiteModel` pytree (already true); add a registry mapping them into the scenario set. | `pricing/lattice/hull_white_tree.py`, callable/puttable bonds. |
| `MOD.HESTON.<asset>.{KAPPA,THETA,XI,RHO,V0}` 📋 | Heston parameters. | Same as HW. | `models/heston.py`, `pricing/mc/paths.py` (Heston paths). |
| `MOD.LMM.<ccy>.{VOL_TS,CORR}` 📋 | LMM forward-rate volatility term structure and correlation. | Existing `LMMModel` already pytree-friendly. | `models/lmm.py`, `pricing/mc/lmm_paths.py`, Bermudan swaptions. |

### 3.7 Correlations

| Factor ID | Description | What is needed | Pricing engines |
|---|---|---|---|
| `CORR.EQ.<a>.<b>` 🚧 | Equity–equity correlation. Already differentiable inside `MultiAssetGBMModel.correlation`. | Expose as a named, registry-managed risk factor with PSD-preserving shocks. | Multi-asset MC, spread / basket options. |
| `CORR.EQ_FX.<asset>.<pair>` 📋 | Equity–FX correlation (used for quanto adjustment). | Cross-asset correlation matrix as a named factor (on the model or the `MarketState`). | `QuantoOption`. |
| `CORR.IR.<ccy_a>.<ccy_b>` 📋 | IR–IR cross-currency correlation. | Multi-curve correlation block. | XCCY structured products, hybrid notes. |

---

## 4. Roadmap

This is the dependency-ordered build-out plan that turns 🚧 / 📋 entries into ✅.

1. **Survival curve + credit shocks** *(this iteration: completed for the curve and shock primitives)*
    - Unlocks: CDS pricer, OAS for callable bonds, convertible-bond credit dimension.
2. **Multi-curve market** — *(shipped, bar the quote-space leg)* `MarketState.curves` ✅ holds the jointly bootstrapped `CurveGraph` (`bootstrap_curve_graph`; dict of `DiscountCurve` keyed by identifier, supports any number of currencies × tenors + basis + xccy) that `price()` and the dual-curve pricers read, the legacy `MarketData.curve_graph` ✅ carries one optionally, and the `CurveGraph`-aware `bump_*` primitives ✅ mirror the `MultiCurveSet` ones.  Scenario-level addressing shipped with `KeyedScenario` + `valax.risk.shock` on a `MarketState` ✅, and the VaR path no longer drops the graph ✅.  Keyed scenario *generators* and the per-curve covariance layout shipped with `KeyedFactorLayout` + `block_covariance` + `keyed_parametric_scenarios` ✅, and the keyed sensitivities / per-curve VaR decomposition with `KeyedSensitivities` + `keyed_component_var` ✅, so a projection curve and a basis each carry their own P&L and VaR.  **Remaining:** integration with `quote_jacobian` for quote-space delta bucketing, porting the FRTB PLA vector pair onto the keyed sensitivities, and retiring the legacy flat `valax/risk/var.py` engine (kept for one deprecation cycle).
    - Unlocks: real OIS-vs-SOFR basis risk, dual-curve KRDs, dual-curve PLA decomposition (separate `delta_ois` and `delta_sofr` key-rate DV01s), full cross-currency delta chains, closed-form basis DV01 and xccy DV01 via the implicit-adjoint quote-Jacobian.
3. **FX risk factors** — *(container shipped)* the per-currency curve registry is the `CurveGraph`, and `MarketState.fx_spots` ✅ already separates `FX.SPOT.<pair>` from `EQ.SPOT.<asset>` for pricing and single-trade risk. **Remaining:** keyed shocks and a `KeyedFactorLayout` block for `fx_spots`, and FX vol.
    - Unlocks: full Garman-Kohlhagen risk, XCCY swap KRDs in both currencies, multi-currency PLA.
4. **Vol-surface risk factors** — *(partially shipped)* `MarketState.surfaces` ✅ holds a surface (grid, SABR, SVI, …) per id, and the keyed layer gives each one a parallel-shift vega (`KeyedSensitivities.vega`) ✅. **Remaining:** node-level, surface-keyed vega buckets by strike × expiry.
    - Unlocks: realistic vol PLA, smile-aware stress scenarios, FRTB vega capital requirements.
5. **Inflation factor wiring** — `MarketState.inflation_curves` ✅ exists and is priced from; **remaining:** keyed shocks and a layout block for it, and `INF.CPI` / `INF.BREAKEVEN` sensitivities.
    - Unlocks: inflation desk PLA, inflation-linked bond IE01.
6. **Model-parameter sensitivities** — generic infrastructure that lets any pytree model (Heston, HW, LMM, SVI) auto-register its parameters as named risk factors.
    - Unlocks: model-drift attribution, "Greeks-of-Greeks" diagnostics.
7. **Hierarchical factor IDs and PSD-preserving correlation shocks** — *(partially shipped)* the identifier-based layout exists and is the current one for curves, surfaces, spots and dividends (`KeyedFactorLayout.factor_ids`, `FACTOR_CATEGORIES`), and `psd_factor` already projects a non-PSD covariance onto its nearest PSD counterpart when sampling.  **Remaining:** identifiers for the FX / credit / inflation / model-parameter families, PSD-preserving *correlation* shocks (as opposed to covariance sampling), and retiring the legacy positional path.
    - Unlocks: bank-scale (~10⁴ factor) deployments, FRTB factor-eligibility audit trails.

Each tier deliberately keeps the autodiff invariant intact: every new factor is just another leaf of a pytree that pricing functions consume, so adding the factor automatically gives both the bucketed deltas (one reverse-mode pass) and the second-order block (one Hessian pass) for free.

---

## 5. Instrument → Factor Matrix

<!-- coverage-map -->

Quick reverse lookup: which factors does each instrument load on? `*` indicates a factor that is consumed but not yet exposed as an independent risk factor in the keyed risk layer (`KeyedFactorLayout` on a `MarketState`).

| Instrument | Spot | Vol | Rate (single) | Rate (OIS) | Rate (Fwd) | Basis | Credit | FX | Inflation | Vol surface | Model params |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `EuropeanOption` (BSM) | ✅ | ✅ | ✅ | — | — | — | — | — | — | * | — |
| `EuropeanOption` (Heston MC) | ✅ | * | ✅ | — | — | — | — | — | — | — | * |
| `AmericanOption` | ✅ | ✅ | ✅ | — | — | — | — | — | — | * | — |
| `EquityBarrierOption` | ✅ | ✅ | ✅ | — | — | — | — | — | — | * | — |
| `AsianOption` / `LookbackOption` | ✅ | ✅ | ✅ | — | — | — | — | — | — | * | — |
| `Autocallable` (planned MC) | ✅ | * | ✅ | — | — | — | — | — | — | * | * |
| `WorstOfBasketOption` | ✅ | ✅ | ✅ | — | — | — | — | — | — | * | * (corr) |
| `SpreadOption` | ✅ | ✅ | ✅ | — | — | — | — | — | — | — | * (corr) |
| `VarianceSwap` | ✅ | ✅ | ✅ | — | — | — | — | — | — | * | — |
| `ZeroCouponBond` / `FixedRateBond` | — | — | ✅ | * | — | — | * | — | — | — | — |
| `FloatingRateBond` | — | — | ✅ | * | * | * | * | — | — | — | — |
| `CallableBond` / `PuttableBond` | — | — | ✅ | * | — | — | * | — | — | — | * (HW $a,\sigma$) |
| `ConvertibleBond` | * | * | ✅ | * | — | — | * | — | — | * | * |
| `InterestRateSwap` | — | — | ✅ | * | * | * | — | — | — | — | — |
| `OISSwap` | — | — | ✅ | * | — | — | — | — | — | — | — |
| `CrossCurrencySwap` | — | — | ✅ | * | * | * | — | * | — | — | — |
| `Swaption` / `BermudanSwaption` | — | — | ✅ | * | * | — | — | — | — | * | * (LMM) |
| `Caplet` / `Cap` | — | — | ✅ | * | * | — | — | — | — | * | — |
| `CMSSwap` / `CMSCapFloor` | — | — | ✅ | * | * | — | — | — | — | * | * (convexity) |
| `CDS` | — | — | ✅ | * | — | — | ✅ (planned engine) | — | — | — | — |
| `CDOTranche` | — | — | ✅ | * | — | — | * | — | — | — | * (base corr) |
| `FXForward` | — | — | ✅ | * | — | — | — | * | — | — | — |
| `FXVanillaOption` / `FXBarrierOption` | — | — | ✅ | * | — | — | — | * | — | * | — |
| `QuantoOption` | * | * | ✅ | * | — | — | — | * | — | * | * (corr) |
| `TARF` | — | — | ✅ | * | — | — | — | * | — | * | — |
| `ZeroCouponInflationSwap` / `YoYInflationSwap` | — | — | ✅ | * | — | — | — | — | * | — | — |
| `InflationCapFloor` | — | — | ✅ | * | — | — | — | — | * | * | — |

A `*` entry is the highest-priority area for the next iteration: the engine *uses* the factor but does not yet shock it as a named risk-factor input.

---

## 6. Quick API Reference

See the [Risk API page](api/risk.md) for full signatures; the table below maps factor categories to the primitive functions.

| Category | Bump (per-pillar) | Parallel shift | Key-rate / pointwise | Notes |
|---|---|---|---|---|
| Equity spot | `KeyedScenario.spot_shocks` | — | — | Set `multiplicative=True` for returns. Legacy flat path: `MarketScenario.spot_shocks`. |
| Equity vol (per surface) | `KeyedScenario.vol_shocks` | `shift_surface` | — | One parallel shift per surface. Legacy flat path: `MarketScenario.vol_shocks`, one scalar per asset. |
| Dividend yield | `KeyedScenario.dividend_shocks` | — | — | Additive. Legacy flat path: `MarketScenario.dividend_shocks`. |
| IR (named curve on a `CurveGraph`) | `bump_graph_curve` | `parallel_graph_shift` | `key_rate_graph_bump` | Applied via `KeyedScenario.curve_shocks[curve_id]` inside `shock`; generators in [Scenarios § Keyed generators](risk-scenarios.md#keyed-generators-and-the-per-curve-covariance-layout). |
| IR (single curve) | `bump_curve_zero_rates` | `parallel_shift` | `key_rate_bump` | On a bare `DiscountCurve`. Legacy flat path: applied via `MarketScenario.rate_shocks` inside `apply_scenario`. |
| IR (multi-curve forward) | `bump_forward_curve` | `parallel_basis_shift` | (use `bump_forward_curve` with one nonzero pillar) | Targets a specific tenor key in a `MultiCurveSet`. |
| IR (multi-curve discount) | `bump_discount_curve` | (pass uniform array) | (single nonzero pillar) | Leaves all forward curves untouched. |
| Credit hazard | `bump_hazard_rates` | `parallel_credit_spread_shift` | `key_rate_hazard_bump` | Spread→hazard via $\Delta h=\Delta s/(1-R)$. On a bare `SurvivalCurve`; not yet reachable from `KeyedScenario`. |
| Inflation CPI | 📋 `bump_cpi_levels` | 📋 `parallel_breakeven_shift` | 📋 `key_rate_breakeven_bump` | Planned; see roadmap §4 item 5. |
| FX spot / vol | 📋 | 📋 | 📋 | Planned; see roadmap §4 item 3. |
| Vol surface (node-level) | 📋 | ✅ `shift_surface` (parallel) | 📋 | Node bumps planned; see roadmap §4 item 4. |

For end-to-end use of these primitives in a P&L explain or backtest, see the [VaR & P&L](risk-var.md) and [Backtesting & FRTB PLA](risk-backtesting.md) guides.

---

## 7. Bucketing

Every factor in this registry can be re-expressed in a **coarser coordinate system** via one of two transformations:

| Transformation | Operator | Use cases |
|---|---|---|
| **Linear aggregation** (`BucketMap`) | $\delta_b = A\,\delta_x$, $\Delta x = A^{\!\top}\Delta b$, $\Sigma_b = A\,\Sigma_x A^{\!\top}$ | FRTB tenor vertices, ISDA SIMM buckets, sector / currency / rating aggregation. |
| **Jacobian reparameterization** | $\delta_b = J^{\!\top}\delta_x$, $\Delta x = J\,\Delta b$, $\Sigma_b = J^{\!\top}\Sigma_x J$ | PCA factor reduction, level / slope / curvature, SVI / SABR parameter sensitivities. |

Linear aggregation is the special case $J = A^{\!\top}$. The full implementation lives in `valax/risk/bucketing.py`; see [Theory § 7.8](theory/risk-measures.md#78-risk-bucketing-linear-and-jacobian-transformations) and the [Risk Bucketing guide](risk-bucketing.md#risk-bucketing).

### Standard bucket schemes

| Scheme | Factor category | Implementation |
|---|---|---|
| FRTB IR tenor vertices `{0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30}` | `IR.ZERO.*`, `IR.OIS.*`, `IR.FWD.*`, `IR.BASIS.*` | `tenor_bucket_map(pillar_times, vertices, weight="indicator")` |
| Smooth tenor re-binning | Same as above | `tenor_bucket_map(..., weight="linear")` |
| Equity / credit sector buckets | `EQ.SPOT.*`, `CRED.HAZARD.*` | `equal_weight_bucket_map(group_membership, n_buckets)` |
| Yield-curve PCA factors | Any IR pillar grid | `pca_jacobian(returns, n_components)` |
| Litterman-Scheinkman L/S/C | Same | `level_slope_curvature_jacobian(pillar_times)` |
| SABR / SVI parameter Greeks | `EQ.VOL.*`, future `RATES.SWAPTION_VOL.*` | `jacobian_from_fn(svi_or_sabr_fn, params_base)` |

### Bucketing the full sensitivity set

`bucket_sensitivities(sens, rate_bucket=..., spot_bucket=..., vol_bucket=..., div_bucket=...)` applies independent bucket maps to every component of a `FlatSensitivities` — including bilateral aggregation of `cross_spot_rate` and `cross_vol_rate` — and returns a `BucketedSensitivities` pytree carrying the original bucket labels for human-readable reports.
