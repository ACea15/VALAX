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

The table below summarizes every instrument covered across the VALAX docs.

| Asset Class | Instrument | Pricing Method | Module |
|-------------|-----------|---------------|--------|
| **Credit** | CDS | Survival curve + discounting | `valax.instruments.credit` (planned) |
| **Credit** | CDO Tranche | Gaussian copula (base correlation) | `valax.instruments.credit` (planned) |
| **Fixed Income** | ZeroCouponBond, FixedRateBond | Discounted cashflows + YTM/duration/convexity | `valax.instruments.bonds` |
| **Fixed Income** | Floating Rate Bond | Forward curve projection | `valax.instruments.bonds` |
| **Fixed Income** | Callable Bond | Hull-White trinomial tree + OAS | `valax.instruments.bonds` |
| **Fixed Income** | Puttable Bond | Hull-White trinomial tree | `valax.instruments.bonds` |
| **Fixed Income** | Convertible Bond | Equity-credit PDE (planned) | `valax.instruments.bonds` |
| **Inflation** | ZCIS | Inflation forward curve | `valax.instruments.inflation` |
| **Inflation** | YYIS | Forward-ratio baseline (convexity adj. planned) | `valax.instruments.inflation` |
| **Inflation** | Inflation Cap/Floor | Black-76 on YoY forward | `valax.instruments.inflation` |
| **Equity** | European, American, Barrier, Asian, Lookback | BSM / PDE / Lattice / MC | `valax.instruments.options` |
| **Equity** | Variance Swap | Analytic BSM + MC realized variance | `valax.instruments.options` |
| **Equity** | Spread Option | Margrabe (K=0) / Kirk (K≠0) | `valax.instruments.options` |
| **Equity exotics (planned)** | Digital, Compound, Chooser, Autocallable, Worst-of, Cliquet | BSM ext. / MC / SLV MC | `valax.instruments.options` |
| **FX** | Forward | Covered interest rate parity | `valax.instruments.fx` |
| **FX** | Vanilla Option | Garman-Kohlhagen + 3 delta conventions | `valax.instruments.fx` |
| **FX** | Barrier | Instrument defined; analytic pricing planned | `valax.instruments.fx` |
| **FX (planned)** | Quanto, TARF, FX Swap | Modified GK / MC / Discounted cashflows | `valax.instruments.fx` |
| **Rates** | Caplet, Cap, Floor, Swaption | Black-76 / Bachelier | `valax.instruments.rates` |
| **Rates** | IRS, Bermudan Swaption | Analytic / LSM on LMM paths | `valax.instruments.rates` |
| **Rates** | OIS Swap | Telescoping single-curve | `valax.instruments.rates` |
| **Rates** | Cross-Currency Swap | Two-curve + FX, par basis solver | `valax.instruments.rates` |
| **Rates** | Total Return Swap | Self-financing reduction | `valax.instruments.rates` |
| **Rates** | CMS Swap | Forward par swap rate (convexity adj. planned) | `valax.instruments.rates` |
| **Rates** | CMS Cap/Floor | Black-76 on forward CMS rate | `valax.instruments.rates` |
| **Rates** | Range Accrual | Black-76 digital replication (snapshot) | `valax.instruments.rates` |

!!! tip
    All instruments in this guide are **fully compatible** with JAX transformations.
    Use `jax.jit` for speed, `jax.grad` for Greeks, `jax.vmap` for batch pricing
    across strikes/maturities/scenarios, and `jax.pmap` for multi-GPU distribution.
    See [Greeks & Risk Sensitivities](greeks.md) and [Risk Management](risk.md)
    for details.
