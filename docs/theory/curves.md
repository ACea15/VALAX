[← Models & Theory index](index.md)

# 3. Curve Framework

The yield curve is the most fundamental piece of market data. It determines the time value of money and drives the pricing of every fixed income product.

### 3.1 Discount Factors, Zero Rates, and Forward Rates

Three equivalent representations of the term structure:

**Discount factor** $DF(T)$: the price today of $1 received at time $T$.

$$
DF(T) = e^{-r(T) \cdot T}
$$

**Continuously-compounded zero rate** $r(T)$: the constant rate that, compounded continuously, gives the discount factor.

$$
r(T) = -\frac{\ln DF(T)}{T}
$$

**Simply-compounded forward rate** $F(T_1, T_2)$: the rate implied by the curve for the period $[T_1, T_2]$.

$$
F(T_1, T_2) = \frac{1}{\tau(T_1, T_2)}\left(\frac{DF(T_1)}{DF(T_2)} - 1\right)
$$

where $\tau$ is the day count fraction (see Section 3.4).

**VALAX implementation:** `DiscountCurve` in `valax/curves/discount.py` stores $(T_i, DF_i)$ pairs and interpolates. `forward_rate()` and `zero_rate()` compute the derived quantities. All three representations are linked via autodiff — differentiating a price w.r.t. `discount_factors` gives key-rate durations automatically.

### 3.2 Single-Curve vs. Multi-Curve Framework

> **Motivation, in prose.** For a narrative walkthrough of *why* the
> multi-curve framework exists and what breaks under the single-curve
> assumption, see [`why-multicurve.md`](../guide/why-multicurve.md).
> This section is the arbitrage-theoretic derivation of the same story.

**Pre-2008 (single-curve):** One curve did everything. A 3M LIBOR swap curve was bootstrapped from deposits and swaps, then used for both discounting and forward rate projection. The implicit assumption: interbank lending is risk-free.

**Post-2008 (multi-curve):** The credit crisis revealed that LIBOR tenors carry different credit/liquidity premia. A 3M rate is *not* three compounded 1M rates — the basis spread between them can be 20+ bps.

The modern framework uses **separate curves for separate roles**:

| Curve | Role | Typical index |
|-------|------|---------------|
| **Discount curve** | Present-value discounting of all cashflows | OIS (SOFR in USD, €STR in EUR) |
| **Forward curve (3M)** | Projection of 3M floating rates | 3M SOFR / 3M EURIBOR |
| **Forward curve (6M)** | Projection of 6M floating rates | 6M EURIBOR |

**Why OIS for discounting?** The choice of discount rate is a *consequence* of the collateral arrangement, not a property of the trade. Under a Credit Support Annex (CSA), the receiver of a positive-MTM derivative posts collateral to the payer; the collateral earns a specified rate (the **collateral remuneration rate**) until the next mark. A no-arbitrage argument — replicating an uncollateralised payoff using a collateralised position plus a funded short cash position — shows that the discount rate must equal the collateral remuneration rate, because that is the rate at which the cash leg of the replicating portfolio is funded (Piterbarg 2010). For a single-currency CSA paying overnight rates with daily margining, that rate is OIS (SOFR in USD, €STR in EUR, SONIA in GBP, TONA in JPY). Hence "OIS discounting" under standard CSAs.

**Multi-currency CSAs and cheapest-to-deliver.** Many CSAs allow collateral to be posted in any of several eligible currencies. If the poster pays the collateral rate of whichever currency is cheapest to fund, the effective discount curve becomes:

$$
DF^{\text{CSA}}(T) = \min_{c \in \mathcal{C}}\;DF_c^{\text{converted}}(T)
$$

where each $DF_c^{\text{converted}}$ is the OIS curve of currency $c$ converted into the trade currency via the FX forward and the cross-currency basis (see §3.7). The collateral poster's optionality has economic value, especially when XCCY bases widen — a real driver of pricing and hedging on long-dated trades. Implementing cheapest-to-deliver requires the joint multi-currency curve graph of §3.8 plus an outer max/min projection at lookup time; it is on the production roadmap but not yet implemented.

**Uncollateralised trades.** When no CSA is in place, the cashflows must be funded at the institution's *own* funding rate, which is typically OIS plus a credit-and-liquidity spread. Pricing this case correctly leads to **Funding Valuation Adjustment (FVA)** — a separate adjustment to the OIS-discounted MTM rather than a different discount curve. FVA is roadmap P3.1.

**Dual-curve pricing of a swap:**

For a swap where the floating leg pays 3M SOFR:

1. **Project floating cashflows** using the 3M forward curve: $F_i = \frac{1}{\tau_i}\left(\frac{DF_{\text{3M}}(T_{i-1})}{DF_{\text{3M}}(T_i)} - 1\right)$

2. **Discount all cashflows** (both fixed and floating) using the OIS curve: $PV = \sum_i \text{amount}_i \times DF_{\text{OIS}}(T_i^{\text{pay}})$

In the single-curve world, the floating leg of a par swap has a clean identity: $PV_{\text{float}} = N \times (DF_{\text{start}} - DF_{\text{end}})$. This breaks under dual-curve — you must project each forward rate individually from the forward curve and discount with the OIS curve.

**VALAX implementation:** The modern joint solver `bootstrap_curve_graph` (`valax/curves/bootstrap_graph.py`, MC-Curves-2) is the primary path: it takes a set of `CurveSpec` declarations, a flat list of `BootstrapInstrument` quotes (`IborSwapRate` for the dual-curve residual, plus `OISSwapRate`, `TenorBasisSwap`, `CrossCurrencyBasisSwap`, `FXForward`, `FXSwap`, `MoneyMarketFuture`, `TurnInstrument`), and returns a `CurveGraph` (dict of `DiscountCurve` keyed by identifier) plus a `CurveBuildDiagnostics` report. The pedagogical single-currency container `MultiCurveSet` in `valax/curves/multi_curve.py` (OIS `discount_curve` + tenor-keyed `forward_curves` dict) is retained as a legacy view; its `bootstrap_multi_curve()` sequential pipeline is deprecated in favour of the joint solver. See the [Multi-Curve guide](../guide/curves.md) and [Why Multi-Curve?](../guide/why-multicurve.md) for the narrative treatment.

### 3.3 Curve Bootstrapping

**The bootstrap problem:** Given $N$ market quotes (deposit rates, FRA rates, par swap rates), find the $N$ discount factors at the corresponding pillar dates such that all instruments reprice exactly.

**Sequential bootstrap** (implemented in `valax/curves/bootstrap.py`, `bootstrap_sequential`):

Process instruments in maturity order. Each instrument provides one equation with one unknown (the discount factor at its maturity), assuming all shorter-maturity DFs are already known.

- **Deposit:** $DF(T) = \frac{DF(T_{\text{start}})}{1 + r \cdot \tau}$
- **FRA:** Same formula as deposit, but $T_{\text{start}} > t_0$, so $DF(T_{\text{start}})$ comes from a prior instrument
- **Swap:** $DF(T_n) = \frac{DF(T_{\text{start}}) - r_{\text{swap}} \sum_{i=1}^{n-1} \tau_i \cdot DF(T_i)}{1 + r_{\text{swap}} \cdot \tau_n}$

This works when instruments are non-overlapping and ordered by maturity.

**Simultaneous bootstrap** (implemented in `valax/curves/bootstrap.py`, `bootstrap_simultaneous`):

When instruments overlap (e.g., swaps sharing intermediate payment dates), the sequential method is insufficient. Instead, solve the full system simultaneously:

$$
\mathbf{R}(\mathbf{x}) = \mathbf{0}
$$

where $\mathbf{x} = (\ln DF_1, \ldots, \ln DF_N)$ (log-space ensures positivity) and $R_i$ is the repricing residual for instrument $i$. VALAX uses `optimistix.root_find` with Newton's method. The Jacobian $\partial R_i / \partial x_j$ is computed automatically via `jax.jacobian` — no finite differences needed.

### 3.4 Interpolation Methods

Between bootstrap pillars, the curve must be interpolated. The choice of interpolation determines the smoothness of forward rates.

**Log-linear on discount factors** (VALAX's current method in `DiscountCurve.__call__`):

$$
\ln DF(t) = \text{linear\_interp}(t; t_i, \ln DF_i)
$$

This is equivalent to **piecewise-constant continuously-compounded forward rates** between pillars. It is simple, monotone (DFs are always decreasing), and stable. The drawback: forward rates have jumps at pillar dates, which can cause unrealistic hedging behavior for instruments sensitive to the forward rate curve shape.

**Alternatives not yet implemented:**

| Method | Forward rate behavior | Pros | Cons |
|--------|----------------------|------|------|
| Linear on zero rates | Piecewise linear forwards | Simple | Non-monotone DFs possible |
| Cubic spline on log-DF | Smooth forwards | Beautiful curves | Can oscillate, negative forwards |
| Monotone convex | Positive, continuous forwards | Market standard | Complex implementation |
| Tension splines | Controllable smoothness | Flexible | Extra parameter |

**Why interpolation matters:** A cap prices each caplet using the forward rate over its accrual period. If the forward curve has artificial jumps at pillar dates, caplet prices will have artifacts where accrual periods straddle pillars. Smooth forward rates produce smoother caplet prices and more stable hedging.

### 3.5 Day Count Conventions

Day count conventions determine the year fraction $\tau(T_1, T_2)$ used in rate calculations. Different conventions exist for historical and market-practice reasons.

| Convention | Formula | Typical use |
|------------|---------|-------------|
| **Act/360** | $\frac{d}{360}$ | USD money market, SOFR, EURIBOR swaps |
| **Act/365 Fixed** | $\frac{d}{365}$ | GBP swaps, AUD swaps, many curve internals |
| **Act/Act (ISDA)** | $\frac{d_1}{D_1} + \frac{d_2}{D_2}$ | US Treasuries, government bonds |
| **30/360** | $\frac{360(Y_2-Y_1)+30(M_2-M_1)+(D_2^*-D_1^*)}{360}$ | US corporate bonds, EUR fixed legs |

where $d$ is the actual number of days and $D$ is the denominator (360 or 365).

**VALAX implementation:** `valax/dates/daycounts.py` implements all four conventions as pure functions on integer ordinals. The `year_fraction(start, end, convention)` dispatcher selects the appropriate function. All are JIT-compatible.

**Practical impact:** Using Act/360 instead of Act/365 on a $100M 5Y swap changes the PV by thousands of dollars. Getting the day count wrong is one of the most common sources of pricing discrepancies between systems.

### 3.6 Inflation Curves and Breakeven Pricing

Inflation derivatives link two different economies: the **nominal** economy (where cashflows are paid and discounted) and the **real** economy (where index-linked payoffs are determined). Pricing any inflation instrument requires two curves:

- The **nominal discount curve** $P^N(0, T)$ — same object used for every other fixed income product.
- The **inflation curve** — a term structure of **forward CPI levels** $\text{CPI}(T)$ derived from inflation swap quotes.

#### Forward CPI vs. Real Rates

There are two equivalent representations of the inflation curve:

**Forward CPI representation** (used in VALAX):

$$
\text{CPI}(T) = \mathbb{E}^{\mathbb{Q}_N}\!\left[\text{CPI}_T\right]
$$

i.e. the expected future CPI index under the nominal risk-neutral measure. From the forward CPI curve, two derived rates are used in pricing:

**Zero-coupon (breakeven) inflation rate** $z(T)$ — the annually-compounded rate such that:

$$
\text{CPI}(T) = \text{CPI}(0) \cdot (1 + z(T))^T \quad\Longleftrightarrow\quad z(T) = \left(\frac{\text{CPI}(T)}{\text{CPI}(0)}\right)^{1/T} - 1
$$

**Year-on-year forward inflation rate** — the single-period rate between two pillars:

$$
\text{YoY}(T_{i-1}, T_i) = \frac{\text{CPI}(T_i)}{\text{CPI}(T_{i-1})} - 1
$$

**Real-rate representation** (equivalent, not directly stored in VALAX):

$$
\text{CPI}(T) = \text{CPI}(0) \cdot \frac{P^R(0, T)}{P^N(0, T)}
$$

where $P^R(0, T)$ is the real-curve discount factor. This is a **Fisher-identity-in-expectation** statement: the forward CPI ratio equals the ratio of real to nominal discount factors. VALAX stores the forward CPI levels directly, which is more numerically stable (no division of two small discount factors) and maps cleanly to the quoted ZCIS breakeven market.

**Interpolation:** VALAX interpolates the inflation curve in **log-CPI space**, giving piecewise-constant instantaneous forward inflation rates between pillars — the direct analogue of log-linear discount factor interpolation. This keeps forward CPI levels positive and produces smooth year-on-year forward rates.

#### Zero-Coupon Inflation Swaps (ZCIS)

The ZCIS is the liquid benchmark of the inflation market — it defines the curve. One party pays a fixed rate compounded over the maturity; the counterparty pays the realized CPI ratio. Both settle as a **single cashflow at maturity**:

$$
\text{PV}^{\text{fix}} = N\,DF^N(T)\left[(1 + K)^T - 1\right]  
$$

$$
\text{PV}^{\text{inf}} = N\,DF^N(T)\left[\frac{\text{CPI}(T)}{\text{CPI}(0)} - 1\right]
$$

The **breakeven rate** $K^*$ is the fixed rate that makes the NPV zero. Because both legs discount at the same $DF^N(T)$, the discount factor cancels and the breakeven is a pure statement about the forward CPI curve:

$$
K^*(T) = \left(\frac{\text{CPI}(T)}{\text{CPI}(0)}\right)^{1/T} - 1
$$

This is algebraically identical to the zero-coupon inflation rate $z(T)$ — the ZCIS breakeven *is* the quoted point on the inflation curve. Implemented in `valax/pricing/analytic/inflation.py` as `zcis_price` and `zcis_breakeven_rate`.

#### Year-on-Year Inflation Swaps (YYIS)

The YYIS pays the one-period YoY CPI ratio at each coupon date rather than the cumulative ratio at maturity:

$$
\text{PV}^{\text{inf}} = N \sum_{i=1}^{n}\left(\frac{\text{CPI}(t_i)}{\text{CPI}(t_{i-1})} - 1\right)DF^N(t_i)
$$

$$
\text{PV}^{\text{fix}} = N\,K \sum_{i=1}^{n} \tau_i\,DF^N(t_i)
$$

**Convexity adjustment** (not applied in VALAX's baseline pricer — documented as a known limitation):

The true expected YoY CPI ratio under the $t_i$-forward measure differs from the ratio of forward CPIs:

$$
\mathbb{E}^{\mathbb{Q}_{t_i}}\!\left[\frac{\text{CPI}(t_i)}{\text{CPI}(t_{i-1})}\right] \neq \frac{\text{CPI}(t_i)}{\text{CPI}(t_{i-1})}
$$

The difference is a **convexity correction** that depends on:

- The volatility of the forward CPI ratio (inflation vol)
- The correlation between real and nominal short rates

Under the Jarrow-Yildirim model (a three-factor Heath-Jarrow-Morton framework with nominal rates, real rates, and CPI), the convexity adjustment has a closed form involving the covariance of the three Brownian motions. VALAX's YYIS pricer (`yyis_price`) uses the forward ratio directly — this is the standard **baseline** practice and is accurate to within a few basis points for typical inflation volatilities and rate correlations. A Jarrow-Yildirim extension with convexity adjustment is on the roadmap.

#### Inflation Caps and Floors

An inflation cap is a strip of caplets, each paying:

$$
\text{Caplet}_i = N \cdot \max\!\left(\text{YoY}_i - K,\; 0\right)
$$

where $\text{YoY}_i$ is the realized year-on-year CPI ratio. Market practice is to price each caplet via **Black-76** on the forward YoY rate $F_i$ (treated as lognormal) with a market-quoted inflation volatility:

$$
\text{Caplet}_i = N\,DF^N(t_i)\left[F_i\,\Phi(d_1) - K\,\Phi(d_2)\right]
$$

$$
d_1 = \frac{\ln(F_i/K) + \tfrac{1}{2}\sigma^2 T_i}{\sigma\sqrt{T_i}}, \qquad d_2 = d_1 - \sigma\sqrt{T_i}
$$

Floors follow by put-call parity. Implemented in `valax/pricing/analytic/inflation.py` as `inflation_cap_floor_price_black76`. The same caveat on the convexity adjustment applies: using the forward YoY rate directly is the baseline; a full Jarrow-Yildirim treatment would add a convexity term.

#### Seasonality

Published CPI indices exhibit strong **monthly seasonality** — energy in winter, retail around holidays. Market practice is to bootstrap a seasonally-adjusted CPI curve from ZCIS quotes (which span annual periods so seasonality averages out), then overlay a monthly seasonality factor pattern to produce month-end CPI projections. VALAX does not yet model seasonality — the `InflationCurve` stores forward CPI levels at whatever pillar dates are provided and interpolates smoothly between them. For instruments that settle on ZCIS anniversary dates, this is exact; for month-end indexed instruments, a seasonality overlay is a roadmap item.

**VALAX implementation:** Curve in `valax/curves/inflation.py` (`InflationCurve`, `forward_cpi`, `zc_inflation_rate`, `yoy_forward_rate`, `from_zc_rates`). Pricers in `valax/pricing/analytic/inflation.py` (`zcis_price`, `zcis_breakeven_rate`, `yyis_price`, `inflation_cap_floor_price_black76`). Instruments in `valax/instruments/inflation.py`. Because the inflation curve is an `equinox.Module` pytree, `jax.grad` of any price w.r.t. the curve's `forward_cpis` gives **inflation key-rate sensitivities (IE01)** for free.

### 3.7 No-Arbitrage Relations Across Curves

The multi-curve framework introduces several curves per currency plus FX-implied relations across currencies. Three no-arbitrage identities tie them together; each one is the calibration anchor for one class of bootstrap instrument and is necessary background for the joint solver of §3.8.

#### Covered Interest Rate Parity (CIP)

Let $S(0)$ be the FX spot (units of domestic per unit of foreign), $DF_d(T)$ the domestic OIS discount curve, $DF_f(T)$ the foreign OIS discount curve, and $F^{FX}(0, T)$ the quoted FX forward rate for delivery $T$. CIP states:

$$
F^{FX}(0, T) = S(0) \cdot \frac{DF_f(T)}{DF_d(T)}
$$

The argument is replication: borrowing one unit of foreign currency at $r_f$, converting at $S(0)$, investing at $r_d$, and entering an offsetting forward sale must produce zero net P&L. CIP holds *exactly* under perfect collateralisation in both currencies, because both sides of the replicating portfolio are funded at their respective OIS rates. It is the no-arb constraint that **FX forwards** and **FX swaps** impose on the curve graph.

#### Tenor Basis

Pre-2008, the implicit assumption was that 6M LIBOR equalled two compounded 3M LIBORs:

$$
\left(1 + L_{6M}\,\tau_{6M}\right) \stackrel{?}{=} \left(1 + L_{3M}^{(1)}\,\tau_{3M}\right)\!\left(1 + L_{3M}^{(2)}\,\tau_{3M}\right)
$$

Post-2008 this identity fails: longer tenors carry more credit and liquidity risk per unit time, so the right-hand side is systematically lower than the left. The market quotes the discrepancy as a **tenor basis spread** $b_{3M, 6M}$ — the spread on a basis swap that exchanges the floating leg of one tenor against the floating leg of the other:

$$
\sum_i \tau^{6M}_i\,F^{6M}_i\,DF_{\text{OIS}}(T^{6M}_i)
\;=\; \sum_j \tau^{3M}_j \left[F^{3M}_j + b_{3M, 6M}\right] DF_{\text{OIS}}(T^{3M}_j)
$$

Solving for $b$ given the two forward curves and OIS discounting determines the basis — or, equivalently, observing $b$ in the market and treating it as a calibration equation determines one of the unknown forward curves given the other. This is the residual imposed by `TenorBasisSwap` in §11.5.

The basis is **not** a small correction: 3M-vs-6M EURIBOR basis sat near 50 bp in late 2008 and remains 5–20 bp in normal markets.

#### Cross-Currency Basis

Combining CIP with collateral discounting in two currencies produces a deviation from naive CIP. Under a single-currency-collateral CSA — say, USD collateral — a EUR cashflow must be discounted using a USD-OIS curve converted into EUR via the FX forward, *not* the EUR-OIS curve. The market quotes this deviation as the **cross-currency basis spread** $s$ on a cross-currency basis swap (CCBS):

$$
\sum_i \tau^{USD}_i \left[F^{USD}_i + s\right] DF^{USD}_{\text{OIS}}(T^{USD}_i)
\;=\;
S(0)\,\sum_j \tau^{EUR}_j\,F^{EUR}_j\,DF^{EUR}_{\text{OIS}}(T^{EUR}_j),
$$

where the spread is conventionally added to the funding-stressed currency leg (sign and side conventions vary by pair). The MTM and constant-notional variants of CCBS differ in whether the notional is reset at each fixing to the prevailing FX rate; both are quoted in the market and both are implemented as separate `CrossCurrencyBasisSwap` variants in §11.5.

The XCCY basis is the calibration instrument that ties two currencies' curves together. It is also a primary indicator of cross-currency funding stress: EUR-USD basis hit roughly $-150$ bp during the Eurozone crisis and the COVID dollar squeeze, and remains the largest persistent deviation from naive CIP in the rates market.

**Why these matter for the bootstrap.** Each identity above has a *quoted* market instrument — FX forward, tenor basis swap, cross-currency basis swap — whose residual involves *more than one curve*. No sequential single-curve bootstrap can satisfy them. The next subsection extends the bootstrap framework to a joint multi-curve solve.

### 3.8 Joint Multi-Curve Calibration

> **See also.** [`why-multicurve.md`](../guide/why-multicurve.md) tells
> the same story without the linear-algebra: which instruments
> couple which curves, and why no ordering of one-curve solves can
> honour a tenor-basis or cross-currency quote.

Section 3.3 described the single-curve bootstrap as a square root-finding system $\mathbf{R}(\mathbf{x}) = \mathbf{0}$, where $\mathbf{x} = (\ln DF_1, \ldots, \ln DF_N)$ and each residual $R_i$ is the repricing error of one instrument. The multi-curve generalisation stacks all per-curve log-DF vectors into one global state vector and runs one Newton iteration over the lot.

#### The joint residual system

Let the curve graph contain $K$ curves with $n_1, n_2, \ldots, n_K$ pillars respectively. The global state vector is

$$
\mathbf{x} = \big(\ln DF^{(1)}_1, \ldots, \ln DF^{(1)}_{n_1},\;\ldots,\;\ln DF^{(K)}_1, \ldots, \ln DF^{(K)}_{n_K}\big) \in \mathbb{R}^{N},\qquad N = \sum_{k=1}^{K} n_k.
$$

Each calibration instrument $\mathcal{I}_i$ contributes one residual $R_i$ that depends only on the curves it touches:

$$
R_i\big(\mathbf{x}\big) = \text{Pricing}_i\big(\{C^{(k)}\}_{k \in \mathcal{T}(i)}\big) - \text{Quote}_i,
$$

where $\mathcal{T}(i)$ is the set of curves involved in instrument $i$. The system is square — $\dim(\mathbf{R}) = \dim(\mathbf{x}) = N$ — when the number of instruments equals the total pillar count. VALAX's `bootstrap_curve_graph` (production design §11.4) solves this with `optimistix.Newton` in log-DF space, exactly as in the single-curve case but with a much larger Jacobian.

#### Why the joint solve is mathematically forced

A *sequential* multi-curve approach — bootstrap OIS, then 3M, then 6M, then the cross-currency curves one at a time — works **only if** every instrument touches a single currently-unknown curve and any other curves it touches are already calibrated. The multi-curve framework breaks this assumption in three concrete ways:

1. **Tenor basis swaps** touch two unknown forward curves simultaneously (the 3M and 6M curves of the same currency). The basis residual fixes a relation between them but determines neither alone — without the joint solve, the system is underdetermined in one curve and overdetermined in the other.
2. **Cross-currency basis swaps** touch one OIS curve plus one forward curve in *each* of two currencies, and FX spot. No ordering of single-currency bootstraps satisfies them, because the residual mixes EUR-side and USD-side projection and discounting in a single equation.
3. **Cheapest-to-deliver multi-currency CSAs** (§3.2) make the discount curve of one currency a function of OIS curves in several currencies. The discount curve and the foreign OIS curves cannot be solved independently.

The joint Newton solve handles all three uniformly: each instrument owns its residual function and declares which curves it touches; the solver does not branch on instrument type.

#### Worked example: counting equations

Consider a minimal EUR-USD multi-curve graph with four curves:

- `USD.SOFR.OIS` — 8 pillars
- `USD.SOFR.3M` — 6 pillars
- `EUR.ESTR.OIS` — 8 pillars
- `EUR.EURIBOR.6M` — 6 pillars

Total unknowns: $N = 28$ log-DFs. To produce a square system, the calibration set must contain 28 instruments, partitioned across curve-touch patterns:

| Instrument class                | Count | Curves touched |
|---------------------------------|------:|----------------|
| USD OIS deposits + swaps        |     8 | `USD.SOFR.OIS` |
| USD 3M futures + IBOR swaps     |     6 | `USD.SOFR.OIS`, `USD.SOFR.3M` |
| EUR OIS deposits + swaps        |     8 | `EUR.ESTR.OIS` |
| EUR 6M futures + IBOR swaps     |     5 | `EUR.ESTR.OIS`, `EUR.EURIBOR.6M` |
| EUR-USD cross-currency basis    |     1 | All four + FX spot |

Total: 28 instruments, 28 unknowns. The Jacobian is $28 \times 28$, sparse with a block structure determined by the curve-touch matrix. The single CCBS instrument is what couples the two-currency blocks; without it, the EUR and USD subsystems decouple and the FX-implied basis cannot be calibrated.

#### Calibration health: Jacobian condition number

The Newton iteration's stability and the post-calibration sensitivity of the curves to input quotes are both governed by the condition number $\kappa(\mathbf{J})$ of $\mathbf{J} = \partial \mathbf{R}/\partial \mathbf{x}$. A poorly-conditioned Jacobian — typically caused by overlapping instruments that don't add information (e.g., two swaps with nearly identical maturities and offsetting basis quotes) — produces a curve that is technically calibrated but fragile to small input perturbations. VALAX's `CurveBuildDiagnostics` (production design §11.6) records $\kappa(\mathbf{J})$ for every build; pathological conditioning is the most common single cause of "the curve calibrates but Greeks blow up."

#### Implicit differentiation through the calibrated curve

Because each pricing function takes the calibrated curve as input, downstream Greeks must propagate sensitivities back through the bootstrap:

$$
\frac{\partial \text{Price}}{\partial \text{Quote}_i} \;=\; \frac{\partial \text{Price}}{\partial \mathbf{x}^*} \cdot \frac{\partial \mathbf{x}^*}{\partial \text{Quote}_i}
$$

where $\mathbf{x}^*$ is the calibrated state vector. The implicit function theorem applied to the Newton fixed point $\mathbf{R}(\mathbf{x}^*; \text{Quotes}) = \mathbf{0}$ gives:

$$
\frac{\partial \mathbf{x}^*}{\partial \text{Quote}_i} \;=\; -\,\mathbf{J}^{-1}\,\frac{\partial \mathbf{R}}{\partial \text{Quote}_i}
$$

— one linear solve, regardless of how many Newton iterations the bootstrap took. `optimistix.ImplicitAdjoint` performs this automatically when `jax.grad` traces through `bootstrap_curve_graph`. No unrolling. This is the mechanism by which a 28-quote curve build delivers a $28 \times 28$ "DF-sensitivities-to-quotes" matrix at the cost of one extra linear solve, rather than 28 finite-difference re-bootstraps.

**VALAX implementation:** Shipped in MC-Curves-2. The joint solver `bootstrap_curve_graph` (`valax/curves/bootstrap_graph.py`) concatenates the log-DFs of every curve into a single flat state vector and drives one Newton solve over the concatenated residual system via `optimistix.Newton`. It handles every joint constraint listed above — tenor basis, cross-currency basis, FX forward — through the plug-in [`BootstrapInstrument`](../api/curves.md) protocol, and delivers the implicit-adjoint quote-Jacobian described in the next paragraph via `quote_jacobian`. See the [Multi-Curve guide](../guide/curves.md#5-multi-curve-bootstrap) for a runnable USD dual-curve example. The sequential `bootstrap_multi_curve` (`valax/curves/multi_curve.py`) is retained one deprecation cycle for the single-currency OIS + tenor case; it emits `DeprecationWarning` and points callers at `bootstrap_curve_graph`.

### 3.9 Futures, Convexity Adjustment, and Fixings

Two implementation details with theoretical content that the bootstrap cannot ignore.

#### Why futures and FRAs differ

A money-market future (Eurodollar, SOFR future) and an FRA over the same accrual period $[T_1, T_2]$ are economically similar but pay differently:

- The **FRA** pays $\tau\,(L - K)$ at $T_2$ in a single cashflow.
- The **future** is daily margined: the position holder receives or posts cash equal to the daily change in the futures rate, and that cash earns the prevailing short rate.

Under the $T_2$-forward measure, the FRA rate is a martingale: $F^{FRA} = \mathbb{E}^{Q^{T_2}}[L]$. Under the rolling money-market measure $Q$ (the natural measure for daily-margined contracts), the *futures rate* is a martingale: $F^{fut} = \mathbb{E}^{Q}[L]$. These two expectations are taken under different measures, so they differ.

The change of numéraire from $Q^{T_2}$ to $Q$ produces a Radon-Nikodym derivative whose form depends on the rates dynamics. The size of $F^{fut} - F^{FRA}$ — the **convexity adjustment** — is therefore a function of the term-structure model, not a model-free quantity.

#### Hull-White convexity adjustment

Under the Hull-White one-factor model (§2.8), $r_t$ is Gaussian and the bond reconstruction formula $P(t, T) = A(t, T)\,e^{-B(t, T)\,r_t}$ holds. The forward rate $F^{FRA}$ over $[T_1, T_2]$ is

$$
F^{FRA} = \frac{1}{\tau}\!\left[\frac{P(0, T_1)}{P(0, T_2)} - 1\right].
$$

Under the rolling-money-market measure $Q$, the simply-compounded rate $L = (1/\tau)\big(1/P(T_1, T_2) - 1\big)$ has expectation

$$
F^{fut} = \mathbb{E}^Q[L] = \frac{1}{\tau}\left[\frac{1}{A(T_1, T_2)} \cdot \mathbb{E}^Q\!\left[e^{B(T_1, T_2)\,r_{T_1}}\right] - 1\right].
$$

Because $r_{T_1}$ is normal under $Q$ with computable mean $\mu_{T_1}$ and variance $V_{T_1}$, the moment-generating-function evaluation gives

$$
\mathbb{E}^Q\!\left[e^{B(T_1, T_2)\,r_{T_1}}\right] = \exp\!\left(B(T_1, T_2)\,\mu_{T_1} + \tfrac{1}{2}\,B(T_1, T_2)^2\,V_{T_1}\right).
$$

Carrying through the algebra and subtracting the FRA rate yields the closed-form HW1F convexity adjustment

$$
F^{fut} - F^{FRA} = \frac{1}{\tau}\Big[(1 + \tau\,F^{FRA})\big(e^{\gamma(T_1, T_2)} - 1\big)\Big],
$$

with

$$
\gamma(T_1, T_2) = B(T_1, T_2)\,\sigma^2 \left[B(T_1, T_2)\,\frac{1 - e^{-2aT_1}}{4a} + \frac{(1 - e^{-aT_1})^2}{2a^2}\right].
$$

For small mean reversion $a \to 0$, this reduces to the textbook approximation

$$
F^{fut} - F^{FRA} \;\approx\; \tfrac{1}{2}\,\sigma^2\,T_1\,(T_2 - T_1).
$$

Brigo & Mercurio (2006) §3.6.2 contains the full derivation and the G2++ extension.

#### Magnitude

Rough scaling at typical USD parameters ($\sigma \approx 1\%$, $a \approx 3\%$):

| Future maturity $T_1$ | Convexity adjustment |
|----------------------:|---------------------:|
| 1Y                    | ~0.5 bp              |
| 3Y                    | ~5 bp                |
| 5Y                    | ~15 bp               |
| 10Y                   | ~60 bp               |

The adjustment grows roughly quadratically in $T_1$. It is small enough at the front end that desks often quote a constant bps adjustment; large enough at the long end that ignoring it produces visible mispricing of the curve shape. VALAX's `MoneyMarketFuture` (§11.5) accepts a pluggable `convexity_adj_fn`: a constant approximation for desk-supplied bps, and the HW1F formula above once the short-rate model is wired into the curve build.

#### Fixings on partially-seasoned curves

A floating leg is a strip of forward rates $F_i$ over accrual periods $[T_{i-1}, T_i]$. Each rate is *fixed* at $T_{i-1}$ (the fixing date) and *paid* at $T_i$. Once the fixing date has passed, the rate is no longer forward-looking — it is a known realised value $\hat{L}_i$ stored in market data, not a function of any curve.

For a calibration instrument whose first reset has already occurred at the time of the build, the residual must use the realised fixing instead of projecting from the curve:

$$
\text{coupon}_1 = \tau_1 \cdot \hat{L}_1,\qquad \text{coupon}_i = \tau_i \cdot F_i\quad\text{for}\quad i \geq 2.
$$

Ignoring fixings causes a sneaky mis-bootstrap: the first coupon of every seasoned swap is replaced by a forward projection, and the curve absorbs the error by twisting at the short end to make the par condition hold. The error is often small for long-dated swaps but is the **first-order** cause of mis-pricing on instruments where the first coupon is a large fraction of total PV — recently-started forward-starting swaps, near-maturity FRNs, or any instrument inside its first accrual period.

VALAX's `FixingHistory` (§11.8) is the data structure that carries realised fixings into the bootstrap; each `BootstrapInstrument` reads from it before falling back to forward projection. This is mandatory infrastructure for production curve builds, not an optimisation.

**VALAX implementation:** Shipped in MC-Curves-1 and MC-Curves-2. `MoneyMarketFuture` in `valax/curves/instruments.py` carries a pluggable `convexity_adj_fn: ConvexityAdjFn` static field with `no_convexity_adj` and `constant_convexity_adj` factories in `valax/curves/convexity.py`. `FixingHistory` and `FixingSeries` in `valax/curves/fixings.py` are the JIT-friendly registries every floating-leg residual (`IborSwapRate`, `TenorBasisSwap`, both legs of `CrossCurrencyBasisSwap`) consults before falling back to forward projection. The Hull-White-derived `hull_white_convexity_adj(model)` factory is still pending (queued for MC-Curves-3) because it depends on threading the calibrated short-rate model through the curve build; the plug-in point is in place.
