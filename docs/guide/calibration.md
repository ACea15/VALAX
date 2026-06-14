# Model Calibration

VALAX calibrates model parameters to market data using gradient-based optimization. Because all pricing functions are differentiable via JAX, the optimizer gets exact Jacobians for free — no finite-difference bumping.

This page covers the **end-to-end calibration workflow**: which liquid market instruments are taken as inputs, what calibrated object each one produces (a curve, a vol surface, a set of model parameters), and which exotic / structured instruments are then priced consistently against those calibrated objects.

!!! tip "Looking for a hands-on walk-through?"
    The [Rates End-to-End Tutorial](tutorial-rates.md) drives every stage on this
    page — synthetic ground-truth curve, noisy quotes, bootstrap, validation,
    pricing, and autodiff DV01 — in a single runnable example with the
    modelling assumptions called out at each step.

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
| **Rates — projection** | Money-market deposits, FRAs, IMM futures (with convexity adj.), IBOR par swaps, tenor-basis swaps | [`bootstrap_multi_curve`](curves.md#5-multi-curve-bootstrap) | `MultiCurveSet` (one forward curve per tenor) | IRS, FRN, OIS swap, CMS, range accrual, Bermudan swaption ✅ |
| **Cross-currency** | FX spot, FX forward points, cross-currency basis swaps | [`bootstrap_curve_graph`](curves.md#5-multi-curve-bootstrap) 🟡 | Foreign-leg `DiscountCurve` with XCCY basis | Cross-currency swaps, quanto adjustments ✅ |
| **Inflation** | Zero-coupon inflation swaps (ZCIS), year-on-year inflation swaps (YYIS) | [`from_zc_rates`](curves.md#7-inflation-curves) | `InflationCurve` (forward CPI) | ZCIS, YYIS, inflation caps/floors ✅ |
| **Equity smile (per slice)** | Listed vanilla calls/puts on a single expiry (broker smile / OPRA chain) | [`calibrate_sabr`](#3-sabr-smile-calibration), [`calibrate_svi_slice`](vol-surfaces.md#svi-surface) | `SABRModel` or `SVIParams` for one expiry | European, American, digital, barrier, compound, chooser, Asian, lookback options |
| **Equity surface** | Vanilla calls/puts across an (expiry × strike) grid | [`calibrate_sabr_surface`](vol-surfaces.md#sabr-surface), [`calibrate_svi_surface`](vol-surfaces.md#svi-surface), [`GridVolSurface`](vol-surfaces.md#grid-surface) | `SABRVolSurface`, `SVIVolSurface`, `GridVolSurface` | Autocallables, worst-of, cliquet, variance swap, all path-dependent exotics |
| **Equity stochastic vol** | Listed vanilla call prices (mid) across expiries and strikes | [`calibrate_heston`](#4-heston-stochastic-vol-calibration) | `HestonModel` ($v_0, \kappa, \theta, \xi, \rho$) | Forward-starting options, cliquets, variance swaps, VIX-style products (MC) |
| **Rates volatility** | ATM swaption straddles, swaption cube ($K \times T_\text{exp} \times T_\text{tail}$), caps/floors | SABR per smile ✅; Hull-White Jamshidian fit 🟡 | SABR swaption cube; `HullWhiteModel` $(a, \sigma)$ | Callable / puttable bonds ✅, Bermudan swaptions, CMS spread options, range accruals |
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

**Calibration output.** A `MultiCurveSet` containing one OIS `DiscountCurve` plus one forward `DiscountCurve` per tenor label. See [Curves and Bootstrapping](curves.md) for the full bootstrap mechanics and the joint Newton residual system.

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
| [`calibrate_svi_surface`](vol-surfaces.md#svi-surface) | Multi-expiry SVI; calendar-spread no-arbitrage by construction |
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
| ATM swaption straddles (per $T_\text{exp}, T_\text{tail}$) | Black/Bachelier ATM vol surface | Hull-White $(a, \sigma)$ via Jamshidian 🟡 |
| Swaption cube ($K \times T_\text{exp} \times T_\text{tail}$) | Full swaption smile | SABR per (expiry, tail) slice |
| Cap / floor straddles | Caplet vol stripping → caplet term structure | Black-76 caplet vols |
| Eurodollar / SOFR options | Short-end caplet vol | Caplet vol stripping |

**Calibration output.** A SABR swaption cube (today) and a calibrated `HullWhiteModel` (roadmap — see [Short-Rate Models §1](short-rate.md#2-build-the-model)). The Jamshidian decomposition turns each swaption into a portfolio of options on individual ZCBs, each priced by Black-76 on an integrated short-rate variance — fast enough to drive a least-squares fit of $(a, \sigma)$.

**What you can price with it.** `CallableBond`, `PuttableBond` ✅; `BermudanSwaption` (currently via LSM on LMM paths) ✅; CMS caps/floors, range accruals, callable structured notes 🟡.

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

See [Inflation §7 in Curves](curves.md#7-inflation-curves) for the bootstrap and [Inflation Derivatives](inflation.md) for pricing.

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
fitted, sol = calibrate_sabr(
    strikes, market_vols, forward, expiry,
    fixed_beta=jnp.array(0.5),
)

print(f"alpha={float(fitted.alpha):.4f}")
print(f"rho={float(fitted.rho):.4f}")
print(f"nu={float(fitted.nu):.4f}")
```

### Solvers

Three backends are available:

| Solver | Method | Best for |
|--------|--------|----------|
| `"levenberg_marquardt"` | Least-squares (optimistix) | Default — fast, exploits Jacobian |
| `"bfgs"` | Quasi-Newton (optimistix) | Fallback when LM struggles |
| `"optax_adam"` | Gradient descent (optax) | Research / experimentation |

```python
# Use BFGS instead
fitted, sol = calibrate_sabr(
    strikes, market_vols, forward, expiry,
    fixed_beta=jnp.array(0.5),
    solver="bfgs",
)
```

### Weighted Calibration

Emphasize ATM strikes by passing per-strike weights:

```python
weights = jnp.exp(-0.5 * ((strikes - forward) / 10.0) ** 2)
fitted, sol = calibrate_sabr(
    strikes, market_vols, forward, expiry,
    fixed_beta=jnp.array(0.5),
    weights=weights,
)
```

## 4. Heston Stochastic-Vol Calibration

Fit Heston parameters $(v_0, \kappa, \theta, \xi, \rho)$ to **option prices** (not implied vols — Heston's natural output is a price via the characteristic-function pricer). Requires a pricing function to be injected — this allows using any Heston pricer (semi-analytic, Monte Carlo, or a neural surrogate).

```python
from valax.calibration import calibrate_heston

fitted, sol = calibrate_heston(
    strikes, market_prices, spot, rate, dividend, expiry,
    pricing_fn=my_heston_pricer,
)
```

**Inputs.** A strip of vanilla European call (or put) prices observed across strikes for a single expiry, or stacked across multiple expiries (vectorised). Typically taken from the listed equity-index option chain (mid prices) or from a broker-cleaned mid surface.

**Output.** A calibrated `HestonModel` pytree — five parameters that jointly fit the smile and term structure.

**What you can price with it.** Anything whose payoff depends on the realised path of variance: forward-starting options, cliquets, variance and volatility swaps (analytic Heston VS strike), VIX-style payoffs, and any structured equity note whose risk is sensitive to vol-of-vol or the spot-vol correlation $\rho$.

## 5. Calibrated Outputs as Pricing Inputs

The whole point of calibration is that the outputs become the **inputs to the pricing of exotics**. The table below maps each calibrated artefact to the pricing functions / instruments that consume it.

| Calibrated artefact | Consumed by | Example exotics priced |
|---------------------|-------------|------------------------|
| `DiscountCurve` (OIS) | Every cashflow PV in the library | All bonds, swaps, options' discounting |
| `MultiCurveSet` (OIS + 3M/6M forward) | `valax.pricing.analytic.rates`, `valax.pricing.mc.lmm` | IRS, OIS swap, Bermudan swaption (LSM on LMM), CMS, range accrual |
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

See [Greeks & Risk Sensitivities](greeks.md) and the [end-to-end example](../examples.md) (`examples/08_end_to_end_workflow.py`) for a full walk-through from synthetic market quotes to calibrated SABR fit to portfolio-level Greeks.

## 6. How It Works

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
