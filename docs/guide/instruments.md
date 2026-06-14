# Instruments Guide

VALAX covers **35+ instruments** across **6 asset classes**: equity options and
exotics, fixed income, interest rate derivatives, FX derivatives, credit derivatives,
and inflation derivatives. Every instrument is an `equinox.Module` subclass — a
frozen dataclass that is automatically a valid JAX pytree. Instruments carry **no
pricing logic**; they are pure data containers describing the contractual terms of a
trade. Pricing is performed by separate pure functions (see
[Analytical Pricing](analytical.md), [Monte Carlo](monte-carlo.md),
[PDE Methods](pde.md), and [Lattice Methods](lattice.md)).

!!! tip
    All instruments are immutable pytrees. To modify a field, use `equinox.tree_at`
    or create a new instance. This guarantees safe `jax.jit`, `jax.vmap`, and
    `jax.grad` usage.

## Navigation

This page is an **index**. Detailed documentation for each asset class lives on
its own page:

| Page | Instruments | Status |
|------|-------------|--------|
| [Fixed Income](fixed-income.md) | `ZeroCouponBond`, `FixedRateBond`, discount curves, YTM, duration, key-rate durations | ✅ Implemented |
| [Callable, Puttable, FRN, Convertible](callable-bonds.md) | `FloatingRateBond`, `CallableBond`, `PuttableBond`, `ConvertibleBond` | FRN/Callable/Puttable ✅; Convertible planned |
| [FX Derivatives](fx.md) | `FXForward`, `FXVanillaOption`, `FXBarrierOption`, `QuantoOption`, `TARF`, `FXSwap` | Forward/Vanilla/Barrier ✅; Quanto/TARF/FXSwap planned |
| [Inflation Derivatives](inflation.md) | `ZeroCouponInflationSwap`, `YearOnYearInflationSwap`, `InflationCapFloor` | ✅ Implemented |
| [Interest Rate Exotics](rates-exotics.md) | `OISSwap`, `CrossCurrencySwap`, `TotalReturnSwap`, `CMSSwap`, `CMSCapFloor`, `RangeAccrual` | ✅ Implemented |
| [Equity Exotics](equity-exotics.md) | `SpreadOption`, `DigitalOption`, `CompoundOption`, `ChooserOption`, `Autocallable`, `WorstOfBasketOption`, `Cliquet` | Spread ✅; others planned |
| [Credit Derivatives](credit.md) | `CDS`, `CDOTranche` | Planned (prerequisites for XVA) |

Vanilla instruments (`EuropeanOption`, `AmericanOption`, basic barriers, Asian and
lookback options, `Caplet`/`Cap`/`Floor`, `InterestRateSwap`, `Swaption`) are
documented on the corresponding pricing pages:

- [Analytical Pricing](analytical.md) — Black-Scholes, Black-76, Bachelier, SABR,
  bond pricing, caplet/cap/swaption analytics, FX Garman-Kohlhagen + delta
  conventions, variance swap.
- [Monte Carlo](monte-carlo.md) — path-dependent equity exotics (Asian, barrier,
  lookback), variance swap realized-vol payoffs, SABR/Heston/LMM simulation,
  Bermudan swaption via Longstaff-Schwartz.
- [Lattice Methods](lattice.md) — CRR binomial (European + American), Hull-White
  trinomial (callable/puttable bonds — see [Short-Rate Models](short-rate.md)).
- [PDE Methods](pde.md) — Crank-Nicolson on the log-spot Black-Scholes PDE.

## Summary Table

The table below summarises every instrument covered across the VALAX docs.

**Legend.** ✅ Implemented end-to-end (instrument + pricer + tests). 🟡 Instrument
pytree is defined and importable from `valax.instruments`, but the corresponding
pricing function is on the roadmap. Every row's *instrument data class* exists
today — the *Status / Notes* column tracks the **pricer**.

| Asset Class | Instrument | Pricing Method | Module | Status / Notes |
|-------------|-----------|----------------|--------|----------------|
| **Credit** | `CDS` | Survival curve + discounting | `valax.instruments.credit` | 🟡 Instrument ✅; pricer planned (roadmap P2.4 — prerequisite for CVA) |
| **Credit** | `CDOTranche` | Gaussian copula (base correlation) | `valax.instruments.credit` | 🟡 Instrument ✅; pricer planned (P2.4) |
| **Fixed Income** | `ZeroCouponBond`, `FixedRateBond` | Discounted cashflows + YTM / duration / convexity | `valax.instruments.bonds` | ✅ `pricing.analytic.bonds` |
| **Fixed Income** | `FloatingRateBond` | Forward-curve projection + discounting | `valax.instruments.bonds` | ✅ `pricing.analytic.floating` |
| **Fixed Income** | `CallableBond` | Hull-White trinomial tree + OAS | `valax.instruments.bonds` | ✅ HW tree pricer (`pricing.lattice.hull_white_tree`); 🟡 OAS curve-shift solver on roadmap |
| **Fixed Income** | `PuttableBond` | Hull-White trinomial tree | `valax.instruments.bonds` | ✅ `pricing.lattice.hull_white_tree` |
| **Fixed Income** | `ConvertibleBond` | Equity–credit PDE | `valax.instruments.bonds` | 🟡 Instrument ✅; pricer planned |
| **Inflation** | `ZeroCouponInflationSwap` (ZCIS) | Inflation forward curve | `valax.instruments.inflation` | ✅ `pricing.analytic.inflation` |
| **Inflation** | `YearOnYearInflationSwap` (YYIS) | Forward-ratio baseline | `valax.instruments.inflation` | ✅ Baseline implemented; 🟡 convexity adjustment planned (currently treated as zero) |
| **Inflation** | `InflationCapFloor` | Black-76 on YoY forward | `valax.instruments.inflation` | ✅ `pricing.analytic.inflation` |
| **Equity** | `EuropeanOption` | Black-Scholes-Merton (closed form) | `valax.instruments.options` | ✅ `pricing.analytic.black_scholes` |
| **Equity** | `AmericanOption` | CRR binomial / Crank-Nicolson PDE | `valax.instruments.options` | ✅ `pricing.lattice.binomial`, `pricing.pde.solvers` |
| **Equity** | `EquityBarrierOption`, `AsianOption`, `LookbackOption` | Monte Carlo on GBM paths | `valax.instruments.options` | ✅ `pricing.mc.payoffs` + `pricing.mc.recipes` |
| **Equity** | `VarianceSwap` | Analytic BSM fair strike + MC realised variance | `valax.instruments.options` | ✅ `pricing.analytic.variance_swap` + MC |
| **Equity** | `SpreadOption` | Margrabe ($K=0$) / Kirk ($K\neq 0$) | `valax.instruments.options` | ✅ `pricing.analytic.spread` |
| **Equity** | `WorstOfBasketOption` | Monte Carlo (multi-asset) | `valax.instruments.options` | ✅ `pricing.mc.recipes._worst_of_basket_multi_asset` |
| **Equity** | `DigitalOption` | Black-Scholes closed form (cash- or asset-or-nothing) | `valax.instruments.options` | 🟡 Instrument ✅; pricer planned |
| **Equity** | `CompoundOption` | Geske closed form / BSM extension | `valax.instruments.options` | 🟡 Instrument ✅; pricer planned |
| **Equity** | `ChooserOption` | Put-call symmetry on a synthetic option | `valax.instruments.options` | 🟡 Instrument ✅; pricer planned |
| **Equity** | `Autocallable` | MC on local-vol / SLV paths | `valax.instruments.options` | 🟡 Instrument ✅; pricer planned |
| **Equity** | `Cliquet` | MC on forward-starting BSM / SLV paths | `valax.instruments.options` | 🟡 Instrument ✅; pricer planned |
| **FX** | `FXForward` | Covered interest-rate parity (CIP) | `valax.instruments.fx` | ✅ `pricing.analytic.fx` |
| **FX** | `FXVanillaOption` | Garman-Kohlhagen + 3 delta conventions | `valax.instruments.fx` | ✅ `pricing.analytic.fx` (incl. strike↔delta inverter) |
| **FX** | `FXBarrierOption` | Reiner-Rubinstein closed form (analytic) | `valax.instruments.fx` | 🟡 Instrument ✅; analytic pricer planned (MC works today via generic GBM recipe) |
| **FX** | `QuantoOption` | Modified Garman-Kohlhagen (quanto drift adj.) | `valax.instruments.fx` | 🟡 Instrument ✅; pricer planned |
| **FX** | `TARF` | Monte Carlo path simulation | `valax.instruments.fx` | 🟡 Instrument ✅; pricer planned |
| **FX** | `FXSwap` | Discounted near + far leg cashflows | `valax.instruments.fx` | 🟡 Instrument ✅; pricer planned |
| **Rates** | `Caplet`, `Cap` (Floor via `is_cap=False` flag) | Black-76 / Bachelier | `valax.instruments.rates` | ✅ `pricing.analytic.caplets` (floors share the same pricer) |
| **Rates** | `Swaption` (European) | Black-76 / Bachelier on forward swap rate | `valax.instruments.rates` | ✅ `pricing.analytic.swaptions` |
| **Rates** | `InterestRateSwap` (IRS) | Discounted fixed leg + projected floating leg | `valax.instruments.rates` | ✅ `pricing.analytic.swaptions` (`swap_price`) |
| **Rates** | `BermudanSwaption` | Longstaff-Schwartz on LMM paths | `valax.instruments.rates` | ✅ `pricing.mc.bermudan` |
| **Rates** | `OISSwap` | Telescoping single-curve identity | `valax.instruments.rates` | ✅ `pricing.analytic.floating` |
| **Rates** | `CrossCurrencySwap` | Two-curve + FX, par-basis solver | `valax.instruments.rates` | ✅ `pricing.analytic.rates_exotics` |
| **Rates** | `TotalReturnSwap` | Self-financing reduction to underlying | `valax.instruments.rates` | ✅ `pricing.analytic.rates_exotics` |
| **Rates** | `CMSSwap` | Forward par swap rate | `valax.instruments.rates` | ✅ Baseline implemented; 🟡 convexity adjustment planned |
| **Rates** | `CMSCapFloor` | Black-76 on forward CMS rate | `valax.instruments.rates` | ✅ `pricing.analytic.rates_exotics` |
| **Rates** | `RangeAccrual` | Black-76 digital replication (snapshot) | `valax.instruments.rates` | ✅ `pricing.analytic.rates_exotics` (single-fixing snapshot; term-structure version planned) |

!!! info "How to read the status column"
    The 🟡 rows mean you can `import` the instrument, construct it as a pytree,
    and use it as data in your own pipelines — but calling the corresponding
    `*_price()` function will raise `NotImplementedError` until the pricer
    lands. See the [Roadmap](../roadmap.md) for delivery order.

!!! tip
    All instruments in this guide are **fully compatible** with JAX transformations.
    Use `jax.jit` for speed, `jax.grad` for Greeks, `jax.vmap` for batch pricing
    across strikes/maturities/scenarios, and `jax.pmap` for multi-GPU distribution.
    See [Greeks & Risk Sensitivities](greeks.md) and [Risk Management](risk.md)
    for details.
