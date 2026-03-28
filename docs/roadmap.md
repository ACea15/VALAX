# Roadmap: Path to Production

VALAX today covers standard equity options, vanilla rates derivatives, four pricing methods, and three stochastic models — all with autodiff Greeks and JIT compilation. This is roughly **5-10% of feature parity** with production quant libraries at major banks (JPM Athena, Goldman SecDB, etc.), which have been built over 20-30 years by hundreds of quants.

The good news: the architecture (pure-functional JAX, autodiff-first, pytree data model) is arguably **more modern** than most bank stacks. The gap is breadth, not depth.

This roadmap organizes every missing piece into prioritized tiers. Each tier unlocks the next.

---

## Current State

| Area | What We Have |
|------|-------------|
| **Instruments** | `EuropeanOption`, `ZeroCouponBond`, `FixedRateBond`, `Caplet`, `Cap`, `InterestRateSwap`, `Swaption` |
| **Models** | Black-Scholes, Heston (MC), SABR (analytic + MC), LMM (MC with PCA factors) |
| **Pricing** | Black-Scholes, Black-76, Bachelier analytic; Monte Carlo (GBM, Heston, SABR, LMM); Crank-Nicolson PDE; CRR binomial tree |
| **Greeks** | 1st order (delta, vega, rho) and 2nd order (gamma, vanna, volga) via autodiff; key-rate durations via curve pytree differentiation |
| **Curves** | `DiscountCurve` with log-linear interpolation, flat extrapolation |
| **Dates** | Act/360, Act/365, Act/Act, 30/360; ordinal-based date arithmetic; coupon schedule generation |
| **Calibration** | SABR and Heston calibration with LM, BFGS, and Optax solvers; parameter constraint transforms |
| **Portfolio** | `batch_price()` and `batch_greeks()` via vmap |
| **Market Data** | `MarketData` container, `MarketScenario` / `ScenarioSet` for risk factor shocks |
| **Risk** | Curve shocks (parallel, steepener, butterfly, key-rate), parametric/historical/stress scenarios, VaR, ES |

---

## Tier 1: Critical Infrastructure

These are foundational pieces that block almost everything else. They should be built first.

### 1.1 Multi-Curve Bootstrapping

- [x] **Sequential bootstrap** from deposits, FRAs, and par swap rates — analytic solve per pillar
- [x] **Simultaneous bootstrap** via `optimistix.root_find` (Newton) in log-DF space — handles overlapping instruments
- [x] **Multi-curve framework** — `MultiCurveSet` with OIS discount curve + tenor-specific forward curves, dual-curve swap bootstrap
- [x] **Jacobian of discount factors w.r.t. input rates** via autodiff (free with JAX, verified in tests)
- [ ] **Basis curve** support (e.g., 1M vs 3M SOFR basis)
- [ ] **Turn-of-year effects** and stub handling

**Why:** Every rates product needs proper forward curves. Manual pillar specification doesn't scale.

**Approach:** Root-finding via `optimistix.root_find` on the system of bootstrapping equations. Each instrument (deposit, swap) provides one equation. The curve is a pytree — autodiff gives the Jacobian for free.

**Status:** Core bootstrapping implemented in `valax/curves/bootstrap.py` and `valax/curves/multi_curve.py`. Bootstrap instrument pytrees (`DepositRate`, `FRA`, `SwapRate`) in `valax/curves/instruments.py`. Both sequential and simultaneous methods produce `DiscountCurve` objects compatible with all existing pricing functions. Dual-curve bootstrap supports OIS discounting with separate forward projection curves.

### 1.2 Volatility Surface Construction

- [x] **Vol surface object** — `GridVolSurface` with bilinear interpolation on (strike, expiry) grid
- [x] **SABR smile per expiry** — `SABRVolSurface` with per-expiry SABR calibration and parameter interpolation
- [x] **SVI parametric fitting** — `SVIVolSurface` with Gatheral's SVI parameterization and LM calibration
- [ ] **SSVI** — global arbitrage-free parameterization (Gatheral-Jacquier)
- [ ] **Sticky-strike vs sticky-delta** conventions
- [ ] **Local vol extraction** from implied vol surface (Dupire formula)

**Why:** Every option product needs a vol surface. Scalar vol is a toy assumption.

**Approach:** Store the surface as an `eqx.Module` pytree with pillar data. Interpolation via cubic splines (or SABR per expiry). SVI fitting via `optimistix.least_squares`. The surface must be differentiable for Greeks.

**Status:** Three surface types implemented in `valax/surfaces/`: `GridVolSurface` (interpolated grid), `SABRVolSurface` (SABR per expiry with calibration), `SVIVolSurface` (SVI per expiry with calibration). All are callable, JIT-able, differentiable pytrees. See the [Vol Surfaces guide](guide/vol-surfaces.md).

### 1.3 Business Calendars and Holiday Schedules

- [ ] **Holiday calendar objects** for major markets: TARGET, NYSE, SOFR (Fed), London, Tokyo
- [ ] **Date adjustment conventions**: modified following, modified preceding, end-of-month
- [ ] **Business day counting** (e.g., Bus/252 for Brazil)
- [ ] **Calendar combination** (union/intersection for cross-currency)

**Why:** Every real trade settles on business days. Wrong dates = wrong cashflows = wrong prices.

**Approach:** Precomputed boolean arrays of holidays (JIT-compatible, no runtime lookups). Calendar as a pytree with a `jnp.array` of holiday ordinals. Adjustment functions operate on integer ordinals.

### 1.4 Trade Representation and Cashflow Engine

- [ ] **Leg abstraction** — fixed leg, floating leg with compounding conventions (in arrears, in advance, compounded, averaged)
- [ ] **Cashflow schedule generation** with proper date adjustment, stub periods (short/long front/back stubs)
- [ ] **Fixing histories** — realized fixings for partially-settled floating legs
- [ ] **Amortization schedules** — notional step-down for amortizing swaps
- [ ] **Accrued interest** calculation

**Why:** Real swaps and bonds have complex cashflow structures. The current simplified schedule generation can't handle production trades.

**Approach:** `FixedLeg` and `FloatingLeg` as `eqx.Module` pytrees. Schedule generation takes a calendar + conventions and produces arrays of (accrual_start, accrual_end, payment_date, notional). Cashflow projection is a pure function over these arrays.

---

## Tier 2: Models

New stochastic models that unlock entire product categories.

### 2.1 Short-Rate Models

- [ ] **Hull-White (one-factor)** — mean-reverting short rate; analytic bond prices and swaption prices
- [ ] **G2++ (two-factor)** — two correlated Hull-White factors; richer term structure dynamics
- [ ] **CIR** — square-root diffusion, ensures positive rates
- [ ] **Vasicek** — Ornstein-Uhlenbeck for rates (allows negative rates)
- [ ] **Calibration to swaption surface** for Hull-White / G2++

**Why:** Short-rate models are the backbone of rates desks for Bermudan swaptions, callable bonds, and path-dependent IR exotics. LMM alone is not enough.

**Approach:** All as `eqx.Module` with drift/diffusion for diffrax. Hull-White has closed-form bond prices (Jamshidian) — implement both analytic and MC pricing. Calibrate to ATM swaption vols via `optimistix`.

### 2.2 Heston Semi-Analytic Pricing

- [ ] **Characteristic function** for Heston model
- [ ] **COS method** (Fang-Oosterlee) for option pricing from characteristic function
- [ ] **Fourier inversion** (Carr-Madan / Lewis) as alternative
- [ ] **Calibration to vol surface** using semi-analytic pricing (fast enough for optimizer loop)

**Why:** MC-only Heston is too slow for calibration and real-time pricing. Semi-analytic methods give prices in microseconds.

**Approach:** Implement the Heston characteristic function as a pure JAX function. COS method is matrix operations — naturally JAX-friendly. `vmap` over strikes for the full smile in one call.

### 2.3 Local Volatility

- [ ] **Dupire local vol** extraction from implied vol surface
- [ ] **Local vol MC simulation** via diffrax
- [ ] **Local vol PDE** pricing (Fokker-Planck or backward Kolmogorov)

**Why:** Local vol is the standard model for exotic equity derivatives. It matches the entire vol surface by construction.

**Approach:** Dupire formula involves derivatives of the implied vol surface — autodiff through the surface pytree. Simulation via diffrax with state-dependent diffusion.

### 2.4 Stochastic-Local Volatility (SLV)

- [ ] **Leverage function** calibration (ratio of local vol to conditional expectation of stochastic vol)
- [ ] **SLV MC simulation** — Heston dynamics with local vol overlay
- [ ] **Particle method** or **kernel regression** for leverage function estimation

**Why:** SLV is the industry standard for exotic equity pricing. It combines the smile-matching of local vol with realistic dynamics of stochastic vol.

**Approach:** Two-pass calibration: (1) calibrate Heston to vanillas, (2) compute leverage function via MC particle method. Leverage function stored as a 2D grid pytree.

### 2.5 Jump-Diffusion Models

- [ ] **Merton jump-diffusion** — log-normal jumps added to GBM
- [ ] **Kou double-exponential** — asymmetric jump sizes
- [ ] **Bates** — Heston + Merton jumps (stochastic vol with jumps)
- [ ] **Semi-analytic pricing** via characteristic functions for all three

**Why:** Jumps explain short-dated smile steepness that diffusion models can't capture. Critical for short-dated FX and equity options.

**Approach:** Characteristic functions are straightforward extensions. MC simulation via diffrax with compound Poisson jump component.

---

## Tier 3: Products and Asset Classes

New instrument types and entire asset classes.

### 3.1 Bermudan Swaptions

- [ ] **Longstaff-Schwartz (LSM)** regression for early exercise on LMM paths
- [ ] **American Monte Carlo** with basis function selection (polynomials, neural network regressors)
- [ ] **Lower/upper bound** estimation (Andersen-Broadie)

**Why:** Bermudan swaptions are among the most traded IR derivatives. Every rates desk prices them daily.

**Approach:** Generate LMM paths (already have this). At each exercise date, regress continuation value on state variables. Use `jnp.polyval` or a small neural network for regression. The exercise boundary is differentiable for Greeks.

### 3.2 Callable and Puttable Bonds

- [ ] **Callable bond** pricing via backward induction on Hull-White tree/PDE
- [ ] **OAS (option-adjusted spread)** calculation
- [ ] **Effective duration and convexity** via autodiff on OAS

**Why:** Callable bonds are a massive market (most corporate bonds are callable). OAS is the standard risk metric.

**Approach:** Requires Hull-White (Tier 2.1). Build a trinomial tree or use PDE with early exercise boundary. OAS is a shift to the discount curve — autodiff gives sensitivities.

### 3.3 FX Derivatives

- [ ] **FX forward** pricing with domestic/foreign discount curves
- [ ] **FX vanilla options** — Garman-Kohlhagen (modified Black-Scholes)
- [ ] **FX barrier options** — continuous and discrete monitoring
- [ ] **FX smile conventions** — delta-based quoting (10D put, 25D put, ATM DNS, 25D call, 10D call)
- [ ] **Quanto options** — correlation between FX and underlying
- [ ] **TARFs (Target Accrual Range Forwards)** — path-dependent FX exotics via MC

**Why:** FX is one of the largest derivatives markets. FX options have unique smile conventions (delta-space, not strike-space).

**Approach:** Garman-Kohlhagen is a minor extension of Black-Scholes. FX smile needs delta-strike conversion utilities. TARFs via MC with diffrax.

### 3.4 Credit Derivatives

- [ ] **Survival curve** construction from CDS spreads
- [ ] **CDS pricing** — protection and premium leg valuation
- [ ] **Hazard rate bootstrapping** from CDS term structure
- [ ] **CDO tranche** pricing (Gaussian copula, base correlation)

**Why:** Credit derivatives are essential for bank trading books and CVA computation.

**Approach:** Survival curves as `eqx.Module` pytrees (analogous to discount curves). CDS pricing is cashflow discounting with survival probabilities. CDO requires copula simulation — JAX's PRNG system handles this well.

### 3.5 Inflation Derivatives

- [ ] **Zero-coupon inflation swap** pricing
- [ ] **Year-on-year inflation swap** pricing
- [ ] **Inflation cap/floor** — Black-76 on forward CPI ratio
- [ ] **Seasonality adjustment** for monthly CPI

**Why:** Inflation markets have grown significantly. Real-money investors use inflation swaps for liability hedging.

**Approach:** Inflation curve as a pytree mapping dates to forward CPI levels. Pricing is standard cashflow discounting with CPI ratios.

### 3.6 Equity Exotics

- [ ] **Autocallable / phoenix notes** — path-dependent with early redemption barriers, coupon barriers, knock-in puts
- [ ] **Worst-of basket options** — multi-asset with correlation
- [ ] **Lookback options** — floating and fixed strike
- [ ] **Touch / no-touch / one-touch** — digital barriers
- [ ] **Range accruals** — accrue coupon while spot is in range
- [ ] **Compound options** — option on an option
- [ ] **Variance / volatility swaps** — realized vol payoffs

**Why:** Structured products desks sell these daily. Autocallables alone are a multi-hundred-billion dollar market.

**Approach:** All priced via MC with appropriate payoff functions. Multi-asset requires correlated path generation (Cholesky on correlation matrix). Autocallables need path-wise barrier monitoring. SLV model (Tier 2.4) is the standard for pricing.

### 3.7 CMS and Spread Options

- [ ] **CMS rate convexity adjustment** (replication or SABR-based)
- [ ] **CMS cap/floor** pricing
- [ ] **CMS spread options** — option on spread between two CMS rates
- [ ] **Range accrual on CMS** — structured coupon products

**Why:** CMS products are actively traded in rates markets. The convexity adjustment is non-trivial and a classic quant problem.

**Approach:** CMS convexity adjustment via static replication (Hagan) or SABR integration. Spread options via 2D copula or Margrabe-like extensions.

### 3.8 Mortgage-Backed Securities and Convertible Bonds

- [ ] **MBS** — prepayment models (PSA, CPR), OAS, effective duration
- [ ] **Convertible bonds** — equity-credit hybrid with call/put features, PDE or tree pricing

**Why:** Large markets but specialized. Lower priority unless targeting specific desks.

**Approach:** MBS requires a prepayment model (empirical) and MC simulation on interest rate paths. Convertibles need a 1D PDE with credit and equity components.

---

## Tier 4: Risk and XVA

Portfolio-level risk management and valuation adjustments.

### 4.1 VaR and Expected Shortfall

- [x] **Historical VaR** — P&L distribution from historical scenarios
- [x] **Parametric VaR** — delta-normal using autodiff sensitivities and covariance matrix
- [x] **Monte Carlo VaR** — full revaluation under simulated scenarios (parametric + t-distribution)
- [x] **Expected Shortfall (CVaR)** — tail risk measure
- [ ] **Marginal / component / incremental VaR**

**Why:** Regulatory requirement (Basel III/IV). Every bank risk system computes these daily.

**Approach:** Scenario generation + batch repricing via `vmap`. Autodiff gives portfolio Greeks for parametric VaR. Full reval VaR benefits from JIT compilation and GPU acceleration — this is where JAX shines.

**Status:** Core framework implemented in `valax/risk/`. Parametric (normal/t) and historical scenario generation, curve shocks (parallel, steepener, flattener, butterfly, key-rate), full-revaluation VaR/ES via vmapped repricing. Parametric VaR via delta-normal with autodiff gradients. See the [Risk guide](guide/risk.md).

### 4.2 Stress Testing and Scenario Analysis

- [x] **Scenario definition framework** — shifts to curves, vols, spots, correlations
- [ ] **Historical stress scenarios** — replay historical crises (2008, COVID, SVB)
- [ ] **Reverse stress testing** — find scenarios that cause a target loss (optimization problem)
- [x] **P&L attribution** — second-order Taylor decomposition into delta (spot, vol, rate, div), gamma, and unexplained via autodiff

**Why:** Regulatory requirement and core risk management practice.

**Approach:** Scenarios as pytree diffs applied to `MarketData`. Batch repricing via `vmap`. P&L attribution via Taylor expansion using autodiff Greeks. Reverse stress testing via `optimistix.minimize`.

**Status:** Scenario framework implemented. `MarketScenario` and `ScenarioSet` pytrees, `apply_scenario` for full market state shocks, named stress builders (steepener, flattener, butterfly). P&L attribution via `pnl_attribution()` with second-order Taylor expansion. Historical crisis replay and reverse stress testing are next.

### 4.3 XVA Suite

- [ ] **CVA (Credit Valuation Adjustment)** — expected loss from counterparty default
- [ ] **DVA (Debit Valuation Adjustment)** — symmetric credit charge
- [ ] **FVA (Funding Valuation Adjustment)** — cost of funding uncollateralized exposure
- [ ] **MVA (Margin Valuation Adjustment)** — cost of posting initial margin
- [ ] **KVA (Capital Valuation Adjustment)** — cost of regulatory capital
- [ ] **Exposure simulation** — expected exposure (EE), potential future exposure (PFE), exposure at default (EAD)
- [ ] **Netting and collateral** — netting set aggregation, CSA modeling, margin period of risk

**Why:** XVA is now a first-class P&L item at every bank. XVA desks are among the most compute-intensive in banking.

**Approach:** Exposure simulation via MC (reuse existing path generators). At each simulation date, reprice the netting set (this is where GPU acceleration via JAX gives massive speedup). CVA = integral of discounted expected exposure * default probability. Autodiff gives XVA sensitivities.

---

## Tier 5: ML and Performance

Machine learning integration and computational scaling.

### 5.1 Neural Surrogate Pricers

- [ ] **Train neural networks** to approximate slow pricing functions (e.g., Bermudan swaption price as function of market state)
- [ ] **Differential ML** — use autodiff Greeks as training targets alongside prices
- [ ] **Online learning** — update surrogate as market moves

**Why:** XVA requires millions of nested pricings. Neural surrogates reduce compute by 100-1000x while preserving Greeks via autodiff through the network.

**Approach:** Standard neural network in equinox (already a JAX neural network library). Train on (market_state, price, greeks) triples. The surrogate is a pytree — `jax.grad` gives Greeks automatically.

### 5.2 Learned Volatility Surfaces

- [ ] **Neural SVI** — neural network parameterization of the vol surface that is arbitrage-free by construction
- [ ] **Variational autoencoder** for vol surface dynamics (generate realistic vol surface moves for risk simulation)

**Why:** Vol surface fitting is a daily challenge. Neural approaches can enforce arbitrage constraints more naturally than traditional parametric methods.

### 5.3 Reinforcement Learning for Hedging

- [ ] **Deep hedging** — RL agent learns hedging strategy that minimizes a risk measure (CVaR of P&L)
- [ ] **Transaction cost-aware** hedging
- [ ] **Model-free** — learns from historical data, no assumed dynamics

**Why:** Traditional delta hedging assumes continuous rebalancing and no transaction costs. Deep hedging handles realistic frictions.

**Approach:** Environment = simulated market paths. Agent = neural network policy. Train with JAX + optax. The entire training loop is differentiable.

### 5.4 Multi-Dimensional PDE Solvers

- [ ] **ADI (Alternating Direction Implicit)** for 2D PDEs (e.g., Heston PDE in (S, v) space)
- [ ] **Sparse grid** methods for higher dimensions
- [ ] **Neural PDE solvers** (PINNs) for high-dimensional problems

**Why:** Current PDE solver is 1D only. Many important models (Heston, local-stochastic vol, multi-asset) require 2D+ PDE solvers.

**Approach:** ADI scheme decomposes 2D operator into 1D sweeps — reuse existing tridiagonal solver via lineax. Sparse grids reduce curse of dimensionality.

---

## Tier 6: Operations and Integration

Production deployment concerns.

### 6.1 Market Data Integration

- [ ] **Market data adapters** — Bloomberg, Refinitiv, ICE, exchange feeds
- [ ] **Real-time curve and surface building** from streaming quotes
- [ ] **Market data snapping** — end-of-day snapshots for risk

### 6.2 Trade Lifecycle

- [ ] **Trade booking** — create, amend, cancel trades
- [ ] **Event processing** — fixings, exercises, expirations, settlements
- [ ] **Position management** — aggregation, netting

### 6.3 Real-Time Risk

- [ ] **Incremental risk** — update portfolio risk when a single trade changes (avoid full recomputation)
- [ ] **Streaming Greeks** — real-time sensitivity updates as market data ticks
- [ ] **Risk limits and alerts**

**Why:** These are necessary for production deployment but are more about systems engineering than quantitative finance. They become relevant once the pricing and risk engine is mature.

---

## VALAX Advantages to Leverage

Throughout this roadmap, VALAX's JAX-native architecture provides structural advantages:

| Capability | Traditional Libraries | VALAX |
|---|---|---|
| **Greeks** | Finite differences (bump-and-reprice) | Exact via `jax.grad` — faster, no numerical noise |
| **Calibration** | Numerical Jacobians | Analytic Jacobians via autodiff — faster convergence |
| **Batch pricing** | Serial loops or manual parallelism | `vmap` — one line, automatic vectorization |
| **GPU/TPU** | Requires rewrite (CUDA kernels) | Free via JAX backend swap |
| **XVA nested MC** | Cluster computing, days of runtime | GPU-accelerated, potentially real-time |
| **Neural surrogates** | Separate ML stack (PyTorch/TF) | Same framework — train and deploy in JAX |
| **Risk sensitivities** | Separate Greek engine | Any pricing function is automatically differentiable |

These advantages compound as the library grows. Each new pricing function automatically supports Greeks, GPU acceleration, and batch evaluation with zero additional code.
