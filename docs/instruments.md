# Instruments — Coverage & Definitions

This is the **reference map** of every instrument VALAX prices or intends to
price. It answers three questions in one place:

1. **What is covered?** — a coverage matrix cross-tabulating each instrument
   against its numerical method(s), volatility / model input, and calibration
   route.
2. **What is the instrument?** — a short, definition-focused write-up per
   instrument (underlying, cash flows, payoff), *without* re-deriving the
   pricing mathematics.
3. **Where is the maths?** — every write-up links to the relevant
   [Models & Theory](theory/index.md) chapter for the numeraire choice,
   stochastic process, and closed-form / numerical machinery.

It complements, rather than repeats, the two sibling pages:

- [Instruments Guide](guide/instruments.md) — task-oriented routing/status index
  with usage recipes per asset class.
- [API: Instruments](api/instruments.md) — the autodoc field-level reference for
  each `equinox.Module`.

!!! note "Instruments are data, pricing is a function"
    Every instrument is a frozen, data-only `equinox.Module` pytree carrying the
    contractual terms only — no `.npv()` method. Pricing lives in separate pure
    functions `price(instrument, *market_args)`, so the *same* instrument can be
    routed to an analytic, Monte-Carlo, PDE, or lattice engine depending on the
    model you pair it with. See [Design Rationale](design-rationale.md).

**Status legend.** ✅ implemented and tested · 🟡 instrument pytree defined,
pricer on the [roadmap](roadmap.md) · "n-a" = not applicable (the instrument is
priced by curve discounting/projection, with no volatility input).

---

## Coverage matrix

### Equity options & exotics

| Instrument | Method(s) | Vol / model input | Calibration | Key assumptions | Status |
|---|---|---|---|---|---|
| [European](#european-option) | Analytic · MC · PDE · Lattice | constant · Heston · Dupire local vol · SLV | SABR / Heston / SLV / Dupire | GBM (or chosen model); continuous hedging | ✅ |
| [American](#american-option) | PDE · Lattice | constant | — | early exercise; GBM | ✅ |
| [Equity barrier](#equity-barrier-option) | MC · PDE | constant · Heston · local · SLV | as per model | continuous monitoring; sigmoid-smoothed | ✅ |
| [Asian](#asian-option) | MC | constant · Heston · local · SLV | as per model | discrete averaging (arith/geo) | ✅ |
| [Lookback](#lookback-option) | MC | constant · Heston · local · SLV | as per model | discrete extremum monitoring | ✅ |
| [Variance swap](#variance-swap) | Analytic · MC | vol smile/surface | from surface | static replication; continuous sampling | ✅ |
| [Digital](#digital-option) | Analytic · PDE | constant | — | GBM; Rannacher damping (PDE) | ✅ |
| [Spread](#spread-option) | Analytic · MC | per-asset vols + correlation | per-asset SABR | 2-asset correlated GBM | ✅ |
| [Worst-of basket](#worst-of-basket-option) | MC | per-asset vols + correlation | per-asset | n-asset correlated GBM | ✅ |
| [Compound](#compound-option) | — | constant | — | Geske option-on-option | 🟡 |
| [Chooser](#chooser-option) | — | constant | — | choice at intermediate date | 🟡 |
| [Autocallable](#autocallable) | — | SLV (paths exist) | SLV | path-dependent autocall/KI | 🟡 |
| [Cliquet](#cliquet) | — | constant/local | — | forward-start ratchet | 🟡 |

### Interest-rate derivatives

| Instrument | Method(s) | Vol / model input | Calibration | Key assumptions | Status |
|---|---|---|---|---|---|
| [Caplet](#caplet-floorlet) | Analytic · MC | flat Black-76/Bachelier · `OptionletVolSurface` · LMM | SABR caplet strip | single forward per period | ✅ |
| [Cap / Floor](#cap-floor) | Analytic · MC | flat · optionlet surface · LMM | SABR caplet strip | strip of independent caplets | ✅ |
| [Interest-rate swap](#interest-rate-swap) | Analytic | n-a (curves) | curve bootstrap | replication of floating leg | ✅ |
| [Swaption](#swaption) | Analytic · MC · PDE | flat · `SwaptionCube` · Hull-White · LMM | SABR cube / HW→surface | annuity numeraire | ✅ |
| [CMS swap](#cms-swap) | Analytic | `SwaptionCube` / scalar (convexity) | SABR cube | Hagan convexity adjustment | ✅ |
| [CMS cap / floor](#cms-cap-floor) | Analytic | `SwaptionCube` / scalar (+ convexity) | SABR cube | Black-76/Bachelier + convexity | ✅ |
| [Range accrual](#range-accrual) | Analytic | `OptionletVolSurface` / scalar | SABR | snapshot digital replication | ✅ |
| [OIS swap](#ois-swap) | Analytic | n-a (curves) | curve bootstrap | compounded overnight leg | ✅ |
| [Cross-currency swap](#cross-currency-swap) | Analytic | n-a (curves + FX spot) | joint curve bootstrap | telescoping under CIP | ✅ |
| [Total return swap](#total-return-swap) | Analytic | n-a (curves) | curve bootstrap | self-financing asset | ✅ |
| [Bermudan swaption](#bermudan-swaption) | MC · PDE | Hull-White · LMM | HW→surface / LMM | early exercise | ✅ |

### Bonds & fixed income

| Instrument | Method(s) | Vol / model input | Calibration | Key assumptions | Status |
|---|---|---|---|---|---|
| [Zero-coupon bond](#zero-coupon-bond) | Analytic | n-a (curve) | curve bootstrap | single cash flow | ✅ |
| [Fixed-rate bond](#fixed-rate-bond) | Analytic · MC · PDE | n-a · Hull-White | curve / HW | deterministic coupons | ✅ |
| [Floating-rate bond](#floating-rate-note) | Analytic · MC | n-a (curves) · Hull-White | curve / HW | par-reset replication | ✅ |
| [Callable bond](#callable-bond) | Lattice · MC · PDE | Hull-White | HW→swaptions | issuer-optimal call | ✅ |
| [Puttable bond](#puttable-bond) | Lattice · MC · PDE | Hull-White | HW→swaptions | holder-optimal put | ✅ |
| [Convertible bond](#convertible-bond) | — | equity-credit | — | equity conversion option | 🟡 |

### FX derivatives

| Instrument | Method(s) | Vol / model input | Calibration | Key assumptions | Status |
|---|---|---|---|---|---|
| [FX forward](#fx-forward) | Analytic | n-a (curves) | curve bootstrap | covered interest parity | ✅ |
| [FX vanilla option](#fx-vanilla-option) | Analytic | constant (delta-quoted) | delta-smile → strike | Garman-Kohlhagen | ✅ |
| [FX barrier option](#fx-barrier-option) | — | constant | — | Reiner-Rubinstein (planned) | 🟡 |
| [Quanto option](#quanto-option) | — | constant + FX corr | — | fixed-FX domestic payout | 🟡 |
| [TARF](#tarf) | — | constant | — | path-dependent target accrual | 🟡 |
| [FX swap](#fx-swap) | — | n-a (curves) | curve bootstrap | near + far leg pair | 🟡 |

### Credit derivatives

| Instrument | Method(s) | Vol / model input | Calibration | Key assumptions | Status |
|---|---|---|---|---|---|
| [CDS](#credit-default-swap-cds) | — (curve bootstrap ✅) | n-a (hazard curve) | `SurvivalCurve` from spreads | credit triangle; constant recovery | 🟡 |
| [CDO tranche](#cdo-tranche) | — | Gaussian copula | base correlation | one-factor copula | 🟡 |

### Inflation derivatives

| Instrument | Method(s) | Vol / model input | Calibration | Key assumptions | Status |
|---|---|---|---|---|---|
| [Zero-coupon inflation swap](#zero-coupon-inflation-swap-zcis) | Analytic | n-a (inflation curve) | inflation bootstrap | index lag; single exchange | ✅ |
| [Year-on-year inflation swap](#year-on-year-inflation-swap-yyis) | Analytic | n-a (inflation curve) | inflation bootstrap | YoY convexity ignored | ✅ |
| [Inflation cap / floor](#inflation-cap-floor) | Analytic | flat Black-76 | — | Black-76 on forward YoY | ✅ |

---

## Equity options & exotics

Underlying: an equity spot $S_t$ under [Black-Scholes / GBM](theory/stochastic-models.md#21-black-scholes-geometric-brownian-motion), or a
richer model — [Heston](theory/stochastic-models.md#24-heston-stochastic-volatility),
[local volatility (Dupire)](theory/volatility.md#44-local-volatility-dupire), or
[stochastic-local vol](theory/volatility.md#45-stochastic-local-volatility). Usage recipes:
[Equity exotics guide](guide/equity-exotics.md). Fields: [API](api/instruments.md).

### European option
Right to buy (call) or sell (put) $S$ at strike $K$ and expiry $T$. A single terminal cash flow

$$ V_T = (S_T - K)^+ \ \text{(call)}, \qquad (K - S_T)^+ \ \text{(put)}. $$

The reference contract of the library — priced four independent ways (closed form, tree, PDE, MC) as the cross-method benchmark.
See [Theory §2.1](theory/stochastic-models.md#21-black-scholes-geometric-brownian-motion) and pricing methods
[§5.1](theory/pricing-methods.md#51-analytical-closed-form) · [§5.2](theory/pricing-methods.md#52-pde-finite-differences) · [§5.3](theory/pricing-methods.md#53-monte-carlo-simulation) · [§5.4](theory/pricing-methods.md#54-lattice-binomial-trees).

### American option
Same payoff as the European, but exercisable at *any* time $\tau \le T$; value is the optimal-stopping supremum $\sup_{\tau\le T}\mathbb{E}^{\mathbb{Q}}[e^{-r\tau}(S_\tau-K)^+]$. Solved by a free-boundary [PDE (penalty method)](theory/pricing-methods.md#52-pde-finite-differences) or a [binomial tree](theory/pricing-methods.md#54-lattice-binomial-trees) with a per-node exercise test.

### Equity barrier option
A vanilla payoff that is activated (knock-in) or extinguished (knock-out) when $S$ first crosses a `barrier` (up/down). Path-dependent; priced by [MC](theory/pricing-methods.md#53-monte-carlo-simulation) or an absorbing-boundary [PDE](theory/pricing-methods.md#52-pde-finite-differences). The discontinuous knock is [sigmoid-smoothed](guide/equity-exotics.md) so pathwise `jax.grad` Greeks stay finite.

### Asian option
Payoff on the **average** of $S$ over observation dates, $V_T=(\bar S - K)^+$ with $\bar S$ arithmetic or geometric. Averaging damps volatility, so Asians are cheaper than vanillas. MC-priced across all equity models; the geometric case has a closed-form control variate. See [MC guide](guide/monte-carlo.md).

### Lookback option
Payoff on the path extremum: floating-strike pays $S_T - \min_t S_t$ (call), fixed-strike pays $(\max_t S_t - K)^+$. Strongly path-dependent; MC-priced with discrete monitoring.

### Variance swap
Swaps realised variance for a strike: $V_T = N_{\text{var}}\,(\sigma^2_{\text{realised}} - K_{\text{var}})$. Model-independently **statically replicated** by a strip of OTM options across the [volatility surface](theory/volatility.md#42-the-volatility-surface) (the log-contract), plus an MC cross-check.

### Digital option
Pays a fixed `payout` if in-the-money at $T$: $V_T = \text{payout}\cdot\mathbf 1\{S_T > K\}$ (call). Closed-form under GBM; the PDE route uses Rannacher damping to tame the discontinuous terminal condition. See [Theory §2.1](theory/stochastic-models.md#21-black-scholes-geometric-brownian-motion).

### Spread option
Pays on the difference of two assets, $V_T=(S_1(T)-S_2(T)-K)^+$. Closed form via **Margrabe** (exact for $K=0$, an exchange option) or **Kirk's approximation** ($K\neq0$), plus correlated two-asset MC. See [Theory §2.9](theory/stochastic-models.md#29-two-asset-correlated-bsm-and-spread-options).

### Worst-of basket option
Payoff on the worst-performing of $n$ correlated assets. No closed form; priced by [MC](theory/pricing-methods.md#53-monte-carlo-simulation) under a multi-asset correlated-GBM model. See [Theory §2.9](theory/stochastic-models.md#29-two-asset-correlated-bsm-and-spread-options).

### Compound option
An option whose underlying is itself a vanilla European option (option-on-option). Priced by Geske's decomposition — *pricer on the roadmap*. 🟡

### Chooser option
The holder decides at a `choose_date` whether the contract becomes a call or a put with common strike/expiry — a straddle-like structure. *Pricer on the roadmap.* 🟡

### Autocallable
A structured note observed on a schedule: auto-redeems early with a coupon if spot is above the autocall barrier, pays conditional (optionally memory) coupons above the coupon barrier, and embeds a down-and-in put struck via `ki_barrier`. SLV path generation exists; the structured-payoff recipe is *on the roadmap*. 🟡

### Cliquet
A ratchet of forward-starting options on capped/floored **periodic returns** with a global floor. *Forward-start recipe on the roadmap.* 🟡

---

## Interest-rate derivatives

Underlying: forward and swap rates off a multi-curve framework
([Theory §3](theory/curves.md#3-curve-framework)), priced by choosing the
numeraire that makes the relevant rate a martingale
([Theory §9.1](theory/rate-derivatives.md#91-change-of-numeraire-and-the-forward-measure)).
Usage: [Rates exotics guide](guide/rates-exotics.md). Fields: [API](api/instruments.md).

### Caplet / floorlet
A single-period call/put on a simply-compounded forward rate $F_i$: pays $\tau_i\,(F_i(T_{i-1})-K)^+$ at $T_i$ (caplet). Priced by Black-76 or Bachelier under the $T_i$-forward measure; a per-expiry smile is supplied by an `OptionletVolSurface`. See [Theory §9.2](theory/rate-derivatives.md#92-caps-floors-and-the-caplet-formula) and [§2.2](theory/stochastic-models.md#22-black-76-futures-forwards)/[§2.3](theory/stochastic-models.md#23-bachelier-normal-model).

### Cap / floor
A strip of caplets/floorlets on a common strike over a payment schedule; the price is the sum of the constituent optionlet prices, each reading its own vol from the surface. See [Theory §9.2](theory/rate-derivatives.md#92-caps-floors-and-the-caplet-formula).

### Interest-rate swap
Exchange of a fixed leg for a floating leg. The floating leg values by the replication identity $\text{PV}=N\,(P(t_0)-P(t_N))$ (single-curve) or by projected forwards (multi-curve). See [Theory §3.2](theory/curves.md#32-single-curve-vs-multi-curve-framework).

### Swaption
European option to enter a payer/receiver swap at `expiry_date`. Under the **annuity (level) measure** the swap rate is a martingale and the price collapses to Black-76/Bachelier on the forward swap rate, times the annuity; a `SwaptionCube` supplies the strike/expiry/tenor smile. Alternatively priced in a one-factor [Hull-White](theory/stochastic-models.md#28-hull-white-one-factor-short-rate-model) model via [Jamshidian decomposition](theory/hull-white-swaptions.md). See [Theory §9.3](theory/rate-derivatives.md#93-swaptions-and-the-annuity-measure).

### CMS swap
One leg pays a **constant-maturity swap rate** (e.g. the 10y rate) fixed each period. Because a swap rate is a martingale only under its own annuity measure, its expectation under the payment measure needs a **convexity adjustment** (Hagan analytic or static replication), consuming the `SwaptionCube` smile. See [Theory §9.5](theory/rate-derivatives.md#95-cms-and-convexity-adjustments).

### CMS cap / floor
A strip of options on the CMS rate, $ (\text{CMS}_i - K)^+ $, priced Black-76/Bachelier on the **convexity-adjusted** forward CMS rate with a per-period smile query. See [Theory §9.5](theory/rate-derivatives.md#95-cms-and-convexity-adjustments).

### Range accrual
A note whose coupon accrues in proportion to the fraction of the period the reference rate spends inside $[L,U]$. Priced by a snapshot digital replication — the probability the forward lands in-range, with per-barrier smile vols (skew between $L$ and $U$). See [Rates exotics guide](guide/rates-exotics.md).

### OIS swap
Fixed vs. daily-**compounded overnight** index. The market-standard discounting instrument post-LIBOR; valued off the OIS discount curve. See [Theory §3.2](theory/curves.md#32-single-curve-vs-multi-curve-framework).

### Cross-currency swap
Exchange of floating legs in two currencies plus (optionally) notional exchange. Values by telescoping each currency's float leg and converting at spot under covered interest parity; the residual is the basis. See [Theory §3.7](theory/curves.md#37-no-arbitrage-relations-across-curves).

### Total return swap
Exchanges the total return of a reference asset for a funding rate + spread. Under a self-financing assumption the asset leg telescopes on the discount curve; the NPV reduces to the funding spread annuity. See [Rates exotics guide](guide/rates-exotics.md).

### Bermudan swaption
Right to enter the underlying swap on one of several `exercise_dates` — the canonical early-exercise rate exotic. Priced by Longstaff-Schwartz on [LMM](theory/stochastic-models.md#26-libor-market-model-lmm-bgm) paths, or backward induction on the [Hull-White tree/PDE](theory/hull-white-pde.md). See [Theory §9.4](theory/rate-derivatives.md#94-bermudan-swaptions-and-early-exercise).

---

## Bonds & fixed income

Underlying: the discount curve (and, for optionality, the short rate under
[Hull-White](theory/stochastic-models.md#28-hull-white-one-factor-short-rate-model)).
Usage: [Fixed income guide](guide/fixed-income.md) · [Callable bonds guide](guide/callable-bonds.md).

### Zero-coupon bond
A single `face_value` cash flow at maturity; price $=$ face $\times$ discount factor. The atom of curve pricing. See [Theory §3.1](theory/curves.md#31-discount-factors-zero-rates-and-forward-rates).

### Fixed-rate bond
Periodic fixed coupons plus face at maturity. Ships with yield-to-maturity, modified duration, convexity, and key-rate durations. Also priced in Hull-White (MC/PDE) as a building block for callables. See [Fixed income guide](guide/fixed-income.md).

### Floating-rate note
Coupons reset periodically off a reference rate plus a `spread`; values by par-reset replication (a floater prices near par at reset, plus the spread annuity). See [Theory §3.2](theory/curves.md#32-single-curve-vs-multi-curve-framework).

### Callable bond
A fixed-rate bond the *issuer* may redeem early at `call_prices` on `call_dates` — long a bond, short a call on the bond. Requires an exact-curve-fit short-rate model: priced on the [Hull-White trinomial tree](theory/hull-white-pde.md), the HW PDE, or HW MC, with an issuer-optimal $\min(\text{continuation},\text{call})$ at each call date. See [Callable bonds guide](guide/callable-bonds.md).

### Puttable bond
As above but the *holder* may sell back at `put_prices` — long a bond, long a put — with a holder-optimal $\max(\text{continuation},\text{put})$ node rule. See [Callable bonds guide](guide/callable-bonds.md).

### Convertible bond
A fixed-rate bond carrying an embedded option to convert into `conversion_ratio` shares — an equity-credit hybrid needing a joint equity/rates (and ideally default) model. *Equity-credit PDE on the roadmap.* 🟡

---

## FX derivatives

Underlying: a spot FX rate $S$ (domestic per foreign) with the foreign rate
acting as a dividend yield — [Garman-Kohlhagen](theory/stochastic-models.md#27-garman-kohlhagen-fx-options).
Usage: [FX guide](guide/fx.md).

### FX forward
Agreement to exchange currencies at a delivery rate $K$ at maturity; value $=(F-K)\,P_d(T)$ with $F=S\,e^{(r_d-r_f)T}$ from covered interest parity. See [Theory §2.7](theory/stochastic-models.md#27-garman-kohlhagen-fx-options).

### FX vanilla option
European call/put on the FX rate, priced by Garman-Kohlhagen (Black-Scholes with $q=r_f$). FX-specific quoting (delta space, DNS ATM, premium-adjusted delta, premium currency) is handled by the FX utilities. See [Theory §2.7](theory/stochastic-models.md#27-garman-kohlhagen-fx-options) and [FX guide](guide/fx.md).

### FX barrier option
Single-barrier knock-in/knock-out on the FX spot. Closed-form Reiner-Rubinstein pricer *on the roadmap*; the pytree and FOR/DOM conventions are defined. 🟡

### Quanto option
An option on a **foreign** underlying paid in **domestic** currency at a fixed FX conversion — the price picks up a correlation-driven quanto drift adjustment. *Pricer on the roadmap.* 🟡

### TARF
Target Accrual Redemption Forward: a schedule of leveraged FX forwards that **terminates early** once accumulated gains reach a `target`. Strongly path-dependent. *Pricer on the roadmap.* 🟡

### FX swap
A near-leg spot and far-leg forward transacted together (an FX-implied funding instrument). The pytree is defined; the trade-level pricer is *on the roadmap* (the curve-building `FXSwap` quote is already used in bootstrapping). 🟡

---

## Credit derivatives

Underlying: a survival/hazard-rate curve bootstrapped from CDS spreads via the
credit triangle. Usage: [Credit guide](guide/credit.md).

### Credit default swap (CDS)
Protection against default of a reference entity: the **protection leg** pays $(1-R)$ of notional on default, funded by a **premium leg** of periodic spread payments until default or maturity. VALAX bootstraps the `SurvivalCurve` from market spreads today; the standalone CDS *pricer is on the roadmap*. 🟡

### CDO tranche
Exposure to portfolio losses in an $[\text{attachment},\text{detachment}]$ band, paying a spread on the surviving tranche notional. Priced by a one-factor Gaussian copula over `n_names`. *Pricer on the roadmap.* 🟡

---

## Inflation derivatives

Underlying: a CPI index projected off a real/nominal (breakeven) inflation curve
with a fixing `index_lag`. Theory:
[Inflation curves & breakeven pricing](theory/curves.md#36-inflation-curves-and-breakeven-pricing).
Usage: [Inflation guide](guide/inflation.md).

### Zero-coupon inflation swap (ZCIS)
A single maturity exchange of cumulative inflation for a fixed compounded rate: $\big(\text{CPI}_T/\text{CPI}_0 - (1+k)^{T}\big)\,N$. The quoted fixed rate is the breakeven inflation rate. See [Theory §3.6](theory/curves.md#36-inflation-curves-and-breakeven-pricing).

### Year-on-year inflation swap (YYIS)
Periodic exchange of realised **annual** inflation $\text{CPI}_i/\text{CPI}_{i-1}-1$ for a fixed rate. A small convexity/timing adjustment (currently ignored) separates it from a strip of ZCIS. See [Theory §3.6](theory/curves.md#year-on-year-inflation-swaps-yyis).

### Inflation cap / floor
A strip of options on year-on-year CPI returns, $(\text{YoY}_i - K)^+$, priced Black-76 on the forward YoY rate. See [Theory §3.6](theory/curves.md#36-inflation-curves-and-breakeven-pricing).

---

## See also

- [Instruments Guide](guide/instruments.md) — usage recipes and per-asset-class routing.
- [API: Instruments](api/instruments.md) — field-level reference for every pytree.
- [Models & Theory](theory/index.md) — the mathematical foundations linked throughout.
- [Roadmap](roadmap.md) — sequencing of the 🟡 pricers.
