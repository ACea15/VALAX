# Instrument Pricing

This is the **reference map** of every instrument VALAX prices or intends to
price — the canonical instruments page. It answers three questions in one place:

1. **What is covered?** — a coverage matrix cross-tabulating each instrument
   against its numerical method(s), volatility / model input, and calibration
   route.
2. **What is the instrument?** — a short, definition-focused write-up per
   instrument (underlying, cash flows, payoff), *without* re-deriving the
   pricing mathematics.
3. **Where is the maths?** — every write-up links to the relevant
   [Models & Theory](../theory/index.md) chapter for the numeraire choice,
   stochastic process, and closed-form / numerical machinery.

For field-level detail (constructor arguments, shapes, defaults) this page
defers to its one sibling, [API: Instruments](../api/instruments.md) — the autodoc
reference for each `equinox.Module`. Task-oriented *how-to* recipes live on the
per-asset-class guide pages (see [Usage guides](#usage-guides) below).

!!! note "Instruments are data, pricing is a function"
    Every instrument is a frozen, data-only `equinox.Module` pytree carrying the
    contractual terms only — no `.npv()` method. Pricing lives in separate pure
    functions `price(instrument, *market_args)`, so the *same* instrument can be
    routed to an analytic, Monte-Carlo, PDE, or lattice engine depending on the
    model you pair it with. See [Design Rationale](../design-rationale.md).

**Status legend.** ✅ implemented and tested · 🟡 instrument pytree defined,
pricer on the [roadmap](../roadmap.md) · "n-a" = not applicable (the instrument is
priced by curve discounting/projection, with no volatility input).

---

## Usage guides

The coverage matrix and definitions below are the *what*; each asset class has a
task-oriented *how-to* guide with runnable examples:

| Asset class | How-to guide |
|---|---|
| Bonds & fixed income | [Fixed Income](../guide/fixed-income.md) · [Callable / Puttable / FRN / Convertible](../guide/callable-bonds.md) |
| Interest-rate derivatives | [Rate Exotics](../guide/rates-exotics.md) |
| Equity options & exotics | [Equity Exotics](../guide/equity-exotics.md) |
| FX derivatives | [FX](../guide/fx.md) |
| Inflation derivatives | [Inflation](../guide/inflation.md) |
| Credit derivatives | [Credit](../guide/credit.md) |

Vanilla contracts are documented on the pricing-method pages:
[Analytical](../guide/analytical.md) · [Monte Carlo](../guide/monte-carlo.md) ·
[PDE](../guide/pde.md) · [Lattice](../guide/lattice.md).

---

## Coverage matrix

The **Calibration** column names the route that produces each input; what
each route fits, holds fixed and assumes, and its known limitations, are in
the [Calibration Assumptions Register](calibration-register.md). Every
route can also be expressed as a level of a calibration chain (quotes →
curves → vol objects → model) with quote-space risk through the whole chain —
see [From Quotes to Risk](../guide/calibration-to-risk.md); which levels exist
as chain nodes today is tracked on the
[roadmap](../roadmap.md#unified-calibration-pricing-and-risk).

The **`price()` routes** column lists what
[`price(instrument, model, market, method=...)`](../guide/tutorial-unified-pricing.md)
reaches, per model: `Model: methods`, with the `method="auto"` choice in
**bold** (the explicit policy of `valax.core.policy`; every listed method has a
default config, so `config=None` works). A **—** means the instrument is not
routed through `price()` yet and is priced by calling its pricer directly
(the *Method(s)* column); † marks a route that is registered but not yet
priceable through `price()` (LMM: the adapter lands with the LMM rework —
see the [roadmap](../roadmap.md)). Which curves, spots, FX spots and
inflation curves a routed trade reads is decided by its static selectors
(`discount_id`, `forward_id`, each leg's ids, `foreign_discount_id`,
`inflation_id`, `underlying_ids`), so every route works under `jit` / `grad`
with the trade as a traced argument. `tests/test_docs/test_price_routes.py` checks this
column against the registry and the auto policy in both directions.
Model labels: `curves` = `model=None` (curve discounting) · `BS`
`BlackScholesModel` · `Heston` `HestonModel` · `LV` `LocalVolModel` · `SLV`
`SLVModel` · `MA-GBM` `MultiAssetGBMModel` · `IV` `ImpliedVolModel` (market
vol surface by id) · `HW` `HullWhiteModel` · `G2++` `G2PPModel` · `Ch-1F`
`CheyetteModel` · `Ch-2F` `CheyetteModel2F` · `LMM` `LMMModel`.

### Equity options & exotics

| Instrument | Method(s) | `price()` routes | Vol / model input | Calibration | Key assumptions | Status |
|---|---|---|---|---|---|---|
| [European](#european-option) | Analytic · MC · PDE · Lattice | BS: **analytic**, lattice, pde, mc<br>Heston: **analytic**, pde, mc<br>LV: **pde**, mc<br>SLV: **mc** | constant · Heston · Dupire local vol · SLV | SABR / Heston / SLV / Dupire; as a chain: curves → SABR smile → Heston (`calibrate_market` / `calibrate_model`, quote risk via `chain_adjoint`) | GBM (or chosen model); continuous hedging | ✅ |
| [American](#american-option) | PDE · Lattice | BS: **lattice**, pde | constant | — | early exercise; GBM | ✅ |
| [Equity barrier](#equity-barrier-option) | MC · PDE | BS: **pde**, mc<br>Heston: **mc**<br>LV: **mc**<br>SLV: **mc** | constant · Heston · local · SLV | as per model | continuous monitoring; sigmoid-smoothed | ✅ |
| [Asian](#asian-option) | MC | BS: **mc**<br>Heston: **mc**<br>LV: **mc**<br>SLV: **mc** | constant · Heston · local · SLV | as per model | discrete averaging (arith/geo) | ✅ |
| [Lookback](#lookback-option) | MC | BS: **mc**<br>Heston: **mc**<br>LV: **mc**<br>SLV: **mc** | constant · Heston · local · SLV | as per model | discrete extremum monitoring | ✅ |
| [Variance swap](#variance-swap) | Analytic · MC | BS: **mc**<br>Heston: **mc**<br>LV: **mc**<br>SLV: **mc** (replication: direct call) | vol smile/surface | from surface | static replication; continuous sampling | ✅ |
| [Digital](#digital-option) | Analytic · PDE | BS: **analytic**, pde | constant | — | GBM; Rannacher damping (PDE) | ✅ |
| [Spread](#spread-option) | Analytic · MC | MA-GBM: **mc** (spots by `underlying_ids`; Margrabe / Kirk: direct call) | per-asset vols + correlation | per-asset SABR | 2-asset correlated GBM | ✅ |
| [Worst-of basket](#worst-of-basket-option) | MC | MA-GBM: **mc** (spots by `underlying_ids`) | per-asset vols + correlation | per-asset | n-asset correlated GBM | ✅ |
| [Compound](#compound-option) | — | — | constant | — | Geske option-on-option | 🟡 |
| [Chooser](#chooser-option) | — | — | constant | — | choice at intermediate date | 🟡 |
| [Autocallable](#autocallable) | — | — | SLV (paths exist) | SLV | path-dependent autocall/KI | 🟡 |
| [Cliquet](#cliquet) | — | — | constant/local | — | forward-start ratchet | 🟡 |

### Interest-rate derivatives

| Instrument | Method(s) | `price()` routes | Vol / model input | Calibration | Key assumptions | Status |
|---|---|---|---|---|---|---|
| [Caplet](#caplet-floorlet) | Analytic · MC · PDE | IV: **analytic**<br>HW: **pde**<br>G2++: **pde**<br>Ch-1F: **pde**, mc<br>Ch-2F: **pde**<br>LMM: **mc**† | flat Black-76/Bachelier · `OptionletVolSurface` · LMM · short-rate (HW / G2++ / Cheyette) | `strip_caplet_vols` / `strip_optionlet_surface` (dual-curve cap→caplet strip + per-expiry SABR); as a chain: `caplet_strip_node` → `optionlet_sabr_node` on the upstream curves | single forward per period | ✅ |
| [Cap / Floor](#cap-floor) | Analytic · MC · PDE | IV: **analytic**<br>HW: **pde**<br>G2++: **pde**<br>Ch-1F: **pde**, mc<br>Ch-2F: **pde**<br>LMM: **mc**† | flat · optionlet surface · LMM · short-rate (HW / G2++ / Cheyette) | `strip_caplet_vols` / `strip_optionlet_surface` (dual-curve cap→caplet strip + per-expiry SABR); as a chain: `caplet_strip_node` → `optionlet_sabr_node` on the upstream curves | strip of independent caplets | ✅ |
| [Interest-rate swap](#interest-rate-swap) | Analytic | curves: **analytic** | n-a (curves) | curve bootstrap | replication of floating leg | ✅ |
| [Swaption](#swaption) | Analytic · MC · PDE · Lattice (Hull-White analytic = Jamshidian, G2++ analytic = Brigo-Mercurio; Hull-White tree = `swaption_tree_price` on the general tree) | IV: **analytic**<br>HW: **analytic**, lattice, pde<br>G2++: **analytic**, pde, mc<br>Ch-1F: **pde**, mc<br>Ch-2F: **pde**<br>LMM: **mc**† | flat · `SwaptionCube` · Hull-White · G2++ · Cheyette (constant or [piecewise-constant σ(t)](../theory/time-dependent-volatility.md)) · LMM | `calibrate_swaption_cube_from_curves` / `_from_graph` (curve-aware SABR cube); as a chain: `swaption_cube_node` on the upstream curves · `swaption_implied_normal_vol` (premium quotes) · `calibrate_hull_white` / `calibrate_g2pp` (single- or [dual-curve](../guide/calibration.md#6-dual-curve-rates-calibration-ois-discounting)); per trade off the cube: `swaption_basket_node` | annuity numeraire; dual-curve via frozen tenor basis | ✅ |
| [CMS swap](#cms-swap) | Analytic | — | `SwaptionCube` / scalar (convexity) | SABR cube | Hagan convexity adjustment | ✅ |
| [CMS-spread swap](#cms-spread-swap) | MC · PDE | G2++: **pde**, mc<br>Ch-2F: **pde**, mc | G2++ · Cheyette-2F | HW → swaption surface | 2-factor decorrelation (steepener / flattener) | ✅ |
| [CMS cap / floor](#cms-cap-floor) | Analytic | — | `SwaptionCube` / scalar (+ convexity) | SABR cube | Black-76/Bachelier + convexity | ✅ |
| [Range accrual](#range-accrual) | Analytic | — | `OptionletVolSurface` / scalar | SABR | snapshot digital replication | ✅ |
| [OIS swap](#ois-swap) | Analytic | curves: **analytic** | n-a (curves) | curve bootstrap | compounded overnight leg | ✅ |
| [Amortising swap](#amortising-swap) | Analytic | curves: **analytic** (`MultiLegSwap`) | n-a (curves) | curve bootstrap | per-period notional factors | ✅ |
| [Seasoned (part-accrued) swap](#seasoned-part-accrued-swap) | Analytic | curves: **analytic** (`MultiLegSwap`; fixings from the market) | n-a (curves + fixing history) | curve bootstrap | realised/projected cut-over at the valuation date | ✅ |
| [Tenor basis swap](#tenor-basis-swap) | Analytic | curves: **analytic** (`MultiLegSwap`, every leg's curve) | n-a (curves) | curve bootstrap · `TenorBasisSwap` quote | one projection curve per leg | ✅ |
| [Cross-currency swap](#cross-currency-swap) | Analytic | curves: **analytic** (domestic + foreign curve, keyed FX spot) | n-a (curves + FX spot) | joint curve bootstrap | telescoping under CIP | ✅ |
| [Total return swap](#total-return-swap) | Analytic | — | n-a (curves) | curve bootstrap | self-financing asset | ✅ |
| [Bermudan swaption](#bermudan-swaption) | MC · PDE · Lattice (Hull-White tree: `bermudan_swaption_tree_price` on the general tree) | HW: **pde**, lattice<br>G2++: **pde**<br>Ch-1F: **pde**<br>Ch-2F: **pde**<br>LMM: **mc**† | Hull-White · G2++ · Cheyette (constant or [piecewise-constant σ(t)](../theory/time-dependent-volatility.md)) · LMM | per trade (Hull-White, G2++, Cheyette 1F/2F; Cheyette single-curve): `coterminal_swaptions` + `swaption_basket_node` off the calibrated cube — `mode="coterminal_bootstrap"` gives one σ bucket per exercise date and reprices every co-terminal exactly, `mode="least_squares"` fits constant parameters — priced through `price(..., method="pde")` with quote risk via `chain_adjoint` ([how-to](../guide/calibration-to-risk.md#rates-per-trade-model-calibration)); LMM uncalibrated | early exercise | ✅ |

### Bonds & fixed income

| Instrument | Method(s) | `price()` routes | Vol / model input | Calibration | Key assumptions | Status |
|---|---|---|---|---|---|---|
| [Zero-coupon bond](#zero-coupon-bond) | Analytic | — | n-a (curve) | curve bootstrap | single cash flow | ✅ |
| [Fixed-rate bond](#fixed-rate-bond) | Analytic · MC · PDE | curves: **analytic**<br>HW: **pde**, mc<br>G2++: **pde**, mc<br>Ch-1F: **pde**, mc<br>Ch-2F: **pde**, mc | n-a · Hull-White · G2++ · Cheyette | curve / HW | day-count accrual per period | ✅ |
| [Floating-rate bond](#floating-rate-note) | Analytic · MC | curves: **analytic**<br>HW: **mc**<br>G2++: **mc**<br>Ch-1F: **mc**<br>Ch-2F: **mc** | n-a (curves) · Hull-White · G2++ · Cheyette | curve / HW | par-reset replication | ✅ |
| [Callable bond](#callable-bond) | Lattice · MC · PDE | HW: **pde**, lattice, mc<br>G2++: **pde**<br>Ch-1F: **pde**<br>Ch-2F: **pde** | Hull-White (tree / MC / PDE; σ(t) on the general tree) · G2++ · Cheyette (PDE) | HW→swaptions | issuer-optimal call, clean call price | ✅ |
| [Puttable bond](#puttable-bond) | Lattice · MC · PDE | HW: **pde**, lattice, mc<br>G2++: **pde**<br>Ch-1F: **pde**<br>Ch-2F: **pde** | Hull-White (tree / MC / PDE; σ(t) on the general tree) · G2++ · Cheyette (PDE) | HW→swaptions | holder-optimal put, clean put price | ✅ |
| [Convertible bond](#convertible-bond) | — | — | equity-credit | — | equity conversion option | 🟡 |

!!! note "Conventions that apply to every row above"
    Coupons are sized from each period's **day-count year fraction**, never
    from `coupon_rate / frequency`, so unequal periods (stubs, and the
    181/184-day alternation of any semiannual actual/N bond) pay unequal
    amounts. Every pricer returns a **dirty** price; call and put prices are
    quoted **clean**. Schedules support day-of-month preservation,
    end-of-month rolls, short/long front/back stubs and **business-day
    adjustment** against a holiday calendar (TARGET, SOFR/SIFMA, NYSE,
    London, Tokyo, Brazil) under following, modified-following, preceding or
    modified-preceding. Adjustment is **opt-in**: the default convention is
    `"unadjusted"`, so every row above is priced on an unadjusted grid unless
    a calendar is configured. See
    [Dates and Conventions](../guide/dates-and-conventions.md).

    **Notional is flat** on the single-trade pytrees in
    `valax.instruments.{rates,bonds}`, and those pricers hold no fixing
    history. An **amortising**, **part-accrued** or **multi-index** trade is
    expressed instead as a tuple of legs — see the three rows above and the
    [leg engine](../guide/fixed-income.md#swap-legs-and-multi-leg-trades).

### FX derivatives

| Instrument | Method(s) | `price()` routes | Vol / model input | Calibration | Key assumptions | Status |
|---|---|---|---|---|---|---|
| [FX forward](#fx-forward) | Analytic | — | n-a (curves) | curve bootstrap | covered interest parity | ✅ |
| [FX vanilla option](#fx-vanilla-option) | Analytic | — | constant (delta-quoted) | delta-smile → strike | Garman-Kohlhagen | ✅ |
| [FX barrier option](#fx-barrier-option) | — | — | constant | — | Reiner-Rubinstein (planned) | 🟡 |
| [Quanto option](#quanto-option) | — | — | constant + FX corr | — | fixed-FX domestic payout | 🟡 |
| [TARF](#tarf) | — | — | constant | — | path-dependent target accrual | 🟡 |
| [FX swap](#fx-swap) | — | — | n-a (curves) | curve bootstrap | near + far leg pair | 🟡 |

### Credit derivatives

| Instrument | Method(s) | `price()` routes | Vol / model input | Calibration | Key assumptions | Status |
|---|---|---|---|---|---|---|
| [CDS](#credit-default-swap-cds) | — (hazard curve ✅) | — | n-a (hazard curve) | `SurvivalCurve` from spreads (closed-form credit triangle, not a repricing bootstrap) | credit triangle; constant recovery | 🟡 |
| [CDO tranche](#cdo-tranche) | — | — | Gaussian copula | base correlation | one-factor copula | 🟡 |

### Inflation derivatives

| Instrument | Method(s) | `price()` routes | Vol / model input | Calibration | Key assumptions | Status |
|---|---|---|---|---|---|---|
| [Zero-coupon inflation swap](#zero-coupon-inflation-swap-zcis) | Analytic | curves: **analytic** (keyed inflation curve) | n-a (inflation curve) | algebraic (`from_zc_rates`) | index lag; single exchange | ✅ |
| [Year-on-year inflation swap](#year-on-year-inflation-swap-yyis) | Analytic | curves: **analytic** (keyed inflation curve) | n-a (inflation curve) | algebraic (`from_zc_rates`) | YoY convexity ignored | ✅ |
| [Inflation cap / floor](#inflation-cap-floor) | Analytic | IV: **analytic** (lognormal vol per caplet fixing) | flat Black-76 | — | Black-76 on forward YoY | ✅ |

---

## Equity options & exotics

Underlying: an equity spot $S_t$ under [Black-Scholes / GBM](../theory/stochastic-models.md#21-black-scholes-geometric-brownian-motion), or a
richer model — [Heston](../theory/stochastic-models.md#24-heston-stochastic-volatility),
[local volatility (Dupire)](../theory/volatility.md#44-local-volatility-dupire), or
[stochastic-local vol](../theory/volatility.md#45-stochastic-local-volatility). Usage recipes:
[Equity exotics guide](../guide/equity-exotics.md). Fields: [API](../api/instruments.md).

### European option
Right to buy (call) or sell (put) $S$ at strike $K$ and expiry $T$. A single terminal cash flow

$$ V_T = (S_T - K)^+ \ \text{(call)}, \qquad (K - S_T)^+ \ \text{(put)}. $$

The reference contract of the library — priced four independent ways (closed form, tree, PDE, MC) as the cross-method benchmark.
See [Theory §2.1](../theory/stochastic-models.md#21-black-scholes-geometric-brownian-motion) and pricing methods
[§5.1](../theory/pricing-methods.md#51-analytical-closed-form) · [§5.2](../theory/pricing-methods.md#52-pde-finite-differences) · [§5.3](../theory/pricing-methods.md#53-monte-carlo-simulation) · [§5.4](../theory/pricing-methods.md#54-lattice-binomial-trees).

### American option
Same payoff as the European, but exercisable at *any* time $\tau \le T$; value is the optimal-stopping supremum $\sup_{\tau\le T}\mathbb{E}^{\mathbb{Q}}[e^{-r\tau}(S_\tau-K)^+]$. Solved by a free-boundary [PDE (penalty method)](../theory/pricing-methods.md#52-pde-finite-differences) or a [binomial tree](../theory/pricing-methods.md#54-lattice-binomial-trees) with a per-node exercise test.

### Equity barrier option
A vanilla payoff that is activated (knock-in) or extinguished (knock-out) when $S$ first crosses a `barrier` (up/down). Path-dependent; priced by [MC](../theory/pricing-methods.md#53-monte-carlo-simulation) or an absorbing-boundary [PDE](../theory/pricing-methods.md#52-pde-finite-differences). The discontinuous knock is [sigmoid-smoothed](../guide/equity-exotics.md) so pathwise `jax.grad` Greeks stay finite.

### Asian option
Payoff on the **average** of $S$ over observation dates, $V_T=(\bar S - K)^+$ with $\bar S$ arithmetic or geometric. Averaging damps volatility, so Asians are cheaper than vanillas. MC-priced across all equity models; the geometric case has a closed-form control variate. See [MC guide](../guide/monte-carlo.md).

### Lookback option
Payoff on the path extremum: floating-strike pays $S_T - \min_t S_t$ (call), fixed-strike pays $(\max_t S_t - K)^+$. Strongly path-dependent; MC-priced with discrete monitoring.

### Variance swap
Swaps realised variance for a strike: $V_T = N_{\text{var}}\,(\sigma^2_{\text{realised}} - K_{\text{var}})$. Model-independently **statically replicated** by a strip of OTM options across the [volatility surface](../theory/volatility.md#42-the-volatility-surface) (the log-contract), plus an MC cross-check.

### Digital option
Pays a fixed `payout` if in-the-money at $T$: $V_T = \text{payout}\cdot\mathbf 1\{S_T > K\}$ (call). Closed-form under GBM; the PDE route uses Rannacher damping to tame the discontinuous terminal condition. See [Theory §2.1](../theory/stochastic-models.md#21-black-scholes-geometric-brownian-motion).

### Spread option
Pays on the difference of two assets, $V_T=(S_1(T)-S_2(T)-K)^+$. Closed form via **Margrabe** (exact for $K=0$, an exchange option) or **Kirk's approximation** ($K\neq0$), plus correlated two-asset MC. See [Theory §2.9](../theory/stochastic-models.md#29-two-asset-correlated-bsm-and-spread-options).

### Worst-of basket option
Payoff on the worst-performing of $n$ correlated assets. No closed form; priced by [MC](../theory/pricing-methods.md#53-monte-carlo-simulation) under a multi-asset correlated-GBM model. See [Theory §2.9](../theory/stochastic-models.md#29-two-asset-correlated-bsm-and-spread-options).

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
([Theory §3](../theory/curves.md#3-curve-framework)), priced by choosing the
numeraire that makes the relevant rate a martingale
([Theory §9.1](../theory/rate-derivatives.md#91-change-of-numeraire-and-the-forward-measure)).
Usage: [Rates exotics guide](../guide/rates-exotics.md). Fields: [API](../api/instruments.md).

### Caplet / floorlet
A single-period call/put on a simply-compounded forward rate $F_i$: pays $\tau_i\,(F_i(T_{i-1})-K)^+$ at $T_i$ (caplet). Priced by Black-76 or Bachelier under the $T_i$-forward measure; a per-expiry smile is supplied by an `OptionletVolSurface`. See [Theory §9.2](../theory/rate-derivatives.md#92-caps-floors-and-the-caplet-formula) and [§2.2](../theory/stochastic-models.md#22-black-76-futures-forwards)/[§2.3](../theory/stochastic-models.md#23-bachelier-normal-model).

### Cap / floor
A strip of caplets/floorlets on a common strike over a payment schedule; the price is the sum of the constituent optionlet prices, each reading its own vol from the surface. Market quotes are *flat* per maturity and are converted into that per-expiry surface by the cap→caplet strip — see [Volatility Surfaces § Cap → caplet stripping](../guide/vol-surfaces.md#cap-caplet-optionlet-stripping-rates) and [Theory §9.2](../theory/rate-derivatives.md#92-caps-floors-and-the-caplet-formula).

### Interest-rate swap
Exchange of a fixed leg for a floating leg. The floating leg values by the replication identity $\text{PV}=N\,(P(t_0)-P(t_N))$ (single-curve) or by projected forwards (multi-curve). See [Theory §3.2](../theory/curves.md#32-single-curve-vs-multi-curve-framework).

### Swaption
European option to enter a payer/receiver swap at `expiry_date`. Under the **annuity (level) measure** the swap rate is a martingale and the price collapses to Black-76/Bachelier on the forward swap rate, times the annuity; a `SwaptionCube` supplies the strike/expiry/tenor smile, anchored on forwards derived from the curves themselves and read under a declared [smile-dynamics convention](../guide/vol-surfaces.md#smile-dynamics-what-moves-when-the-curve-moves). Alternatively priced in a one-factor [Hull-White](../theory/stochastic-models.md#28-hull-white-one-factor-short-rate-model) model via [Jamshidian decomposition](../theory/hull-white-swaptions.md), or under G2++ by the Brigo-Mercurio quadrature — both of which also accept a distinct projection curve under the [deterministic tenor basis](../theory/multi-curve-short-rate.md#51-why-the-closed-forms-extend-so-cheaply). See [Theory §9.3](../theory/rate-derivatives.md#93-swaptions-and-the-annuity-measure).

### CMS swap
One leg pays a **constant-maturity swap rate** (e.g. the 10y rate) fixed each period. Because a swap rate is a martingale only under its own annuity measure, its expectation under the payment measure needs a **convexity adjustment** (Hagan analytic or static replication), consuming the `SwaptionCube` smile. See [Theory §9.5](../theory/rate-derivatives.md#95-cms-and-convexity-adjustments).

### CMS-spread swap
A steepener / flattener whose coupon pays the spread between two constant-maturity swap rates of *different* tenors (e.g. 10y − 2y) net of a fixed strike, $C_i = N\,(S^{\text{long}}_i - S^{\text{short}}_i - K)\,\tau_i$. Its value depends on the **joint** distribution of short- and long-tenor rates — a decorrelation payoff a one-factor model structurally cannot express — so it is priced in a **two-factor** model (G2++ or two-factor Cheyette), never analytically as a single instrument. Two independent numerical routes agree within Monte-Carlo error: exact-conditional **Monte Carlo** (each CMS rate rebuilt analytically per path from the affine ZCB) and a **2-D ADI PDE** (each period's coupon injected at its fixing level as a linear function of the node state and discounted home in one sweep). See [Theory §9.5](../theory/rate-derivatives.md#95-cms-and-convexity-adjustments) and the [G2++ chapter](../theory/g2pp.md).

### CMS cap / floor
A strip of options on the CMS rate, $ (\text{CMS}_i - K)^+ $, priced Black-76/Bachelier on the **convexity-adjusted** forward CMS rate with a per-period smile query. See [Theory §9.5](../theory/rate-derivatives.md#95-cms-and-convexity-adjustments).

### Range accrual
A note whose coupon accrues in proportion to the fraction of the period the reference rate spends inside $[L,U]$. Priced by a snapshot digital replication — the probability the forward lands in-range, with per-barrier smile vols (skew between $L$ and $U$). See [Rates exotics guide](../guide/rates-exotics.md).

### OIS swap
Fixed vs. daily-**compounded overnight** index. The market-standard discounting instrument post-LIBOR; valued off the OIS discount curve. See [Theory §3.2](../theory/curves.md#32-single-curve-vs-multi-curve-framework).

### Amortising swap
A swap whose notional steps down over its life. Represented as a `MultiLegSwap` of `FixedLeg` / `FloatingLeg` pytrees carrying `amortisation`, an array of **dimensionless outstanding-notional factors** against a scalar reference notional. Expressing the schedule as factors rather than absolute notionals keeps the notional outside the coupon summation, which is what makes a non-amortising leg reproduce the flat pricers bit for bit. Priced by `valax.pricing.analytic.legs` (`fixed_leg_pv`, `floating_leg_pv`, `swap_pv`). See the [leg engine guide](../guide/fixed-income.md#amortisation-is-a-factor-not-a-notional).

### Seasoned (part-accrued) swap
A trade valued mid-period, where part of the current coupon has already been earned. Realised fixings come from a `FixingHistory` supplied at pricing time (not stored on the instrument), and each period is settled, straddling or unborn according to a single **cut-over date** obtained by clipping the valuation date into the period — so the three cases are one expression rather than three branches. See the [leg engine guide](../guide/fixed-income.md#seasoning).

### Tenor basis swap
Two floating legs on the *same* currency but different index tenors (e.g. compounded SOFR vs 3M term rate), discounted on a third, OIS curve. The instrument that motivates **per-leg index descriptors**: each `FloatingLeg` names its own `forward_id`, so the trade reads three curves, which a singular `forward_id` on a trade pytree cannot express. The matching curve-build quote type is `valax.curves.instruments.TenorBasisSwap`. See the [leg engine guide](../guide/fixed-income.md#per-leg-index-descriptors).

### Cross-currency swap
Exchange of floating legs in two currencies plus (optionally) notional exchange. Values by telescoping each currency's float leg and converting at spot under covered interest parity; the residual is the basis. See [Theory §3.7](../theory/curves.md#37-no-arbitrage-relations-across-curves).

### Total return swap
Exchanges the total return of a reference asset for a funding rate + spread. Under a self-financing assumption the asset leg telescopes on the discount curve; the NPV reduces to the funding spread annuity. See [Rates exotics guide](../guide/rates-exotics.md).

### Bermudan swaption
Right to enter the underlying swap on one of several `exercise_dates` — the canonical early-exercise rate exotic. Priced by Longstaff-Schwartz on [LMM](../theory/stochastic-models.md#26-libor-market-model-lmm-bgm) paths, or backward induction on the [Hull-White PDE](../theory/hull-white-pde.md) or [tree](../theory/time-dependent-volatility.md#44-the-general-hull-white-trinomial-tree), the G2++ 2-D PDE or the Cheyette PDE. Its model is typically calibrated per trade by a co-terminal bootstrap: a [piecewise-constant σ(t)](../theory/time-dependent-volatility.md) with one bucket per exercise date, repricing each co-terminal European exactly ([Calibration Chains § I.10](../theory/calibration-chains.md#i10-per-trade-model-levels-and-the-vega-profile)). See [Theory §9.4](../theory/rate-derivatives.md#94-bermudan-swaptions-and-early-exercise).

---

## Bonds & fixed income

Underlying: the discount curve (and, for optionality, the short rate under
[Hull-White](../theory/stochastic-models.md#28-hull-white-one-factor-short-rate-model)).
Usage: [Fixed income guide](../guide/fixed-income.md) · [Callable bonds guide](../guide/callable-bonds.md).

### Zero-coupon bond
A single `face_value` cash flow at maturity; price $=$ face $\times$ discount factor. The atom of curve pricing. See [Theory §3.1](../theory/curves.md#31-discount-factors-zero-rates-and-forward-rates).

### Fixed-rate bond
Periodic fixed coupons plus face at maturity. Ships with yield-to-maturity, modified duration, convexity, and key-rate durations. Also priced in Hull-White (MC/PDE) as a building block for callables. See [Fixed income guide](../guide/fixed-income.md).

### Floating-rate note
Coupons reset periodically off a reference rate plus a `spread`; values by par-reset replication (a floater prices near par at reset, plus the spread annuity). See [Theory §3.2](../theory/curves.md#32-single-curve-vs-multi-curve-framework).

### Callable bond
A fixed-rate bond the *issuer* may redeem early at `call_prices` on `call_dates` — long a bond, short a call on the bond. Requires an exact-curve-fit short-rate model: priced on the [Hull-White trinomial tree](../theory/hull-white-pde.md), the HW PDE, or HW MC, with an issuer-optimal $\min(\text{continuation},\text{call} + \text{accrued})$ at each call date — the call price is quoted clean, so accrued interest is added to put both sides of the comparison on the same footing. See [Callable bonds guide](../guide/callable-bonds.md).

### Puttable bond
As above but the *holder* may sell back at `put_prices` — long a bond, long a put — with a holder-optimal $\max(\text{continuation},\text{put} + \text{accrued})$ node rule. See [Callable bonds guide](../guide/callable-bonds.md).

### Convertible bond
A fixed-rate bond carrying an embedded option to convert into `conversion_ratio` shares — an equity-credit hybrid needing a joint equity/rates (and ideally default) model. *Equity-credit PDE on the roadmap.* 🟡

---

## FX derivatives

Underlying: a spot FX rate $S$ (domestic per foreign) with the foreign rate
acting as a dividend yield — [Garman-Kohlhagen](../theory/stochastic-models.md#27-garman-kohlhagen-fx-options).
Usage: [FX guide](../guide/fx.md).

### FX forward
Agreement to exchange currencies at a delivery rate $K$ at maturity; value $=(F-K)\,P_d(T)$ with $F=S\,e^{(r_d-r_f)T}$ from covered interest parity. See [Theory §2.7](../theory/stochastic-models.md#27-garman-kohlhagen-fx-options).

### FX vanilla option
European call/put on the FX rate, priced by Garman-Kohlhagen (Black-Scholes with $q=r_f$). FX-specific quoting (delta space, DNS ATM, premium-adjusted delta, premium currency) is handled by the FX utilities. See [Theory §2.7](../theory/stochastic-models.md#27-garman-kohlhagen-fx-options) and [FX guide](../guide/fx.md).

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
credit triangle. Usage: [Credit guide](../guide/credit.md).

### Credit default swap (CDS)
Protection against default of a reference entity: the **protection leg** pays $(1-R)$ of notional on default, funded by a **premium leg** of periodic spread payments until default or maturity. VALAX builds the `SurvivalCurve` from market spreads today via the closed-form credit triangle (no root-find, no leg repricing); both the standalone CDS *pricer* and a true repricing hazard bootstrap *are on the roadmap*. 🟡

### CDO tranche
Exposure to portfolio losses in an $[\text{attachment},\text{detachment}]$ band, paying a spread on the surviving tranche notional. Priced by a one-factor Gaussian copula over `n_names`. *Pricer on the roadmap.* 🟡

---

## Inflation derivatives

Underlying: a CPI index projected off a real/nominal (breakeven) inflation curve
with a fixing `index_lag`. Theory:
[Inflation curves & breakeven pricing](../theory/curves.md#36-inflation-curves-and-breakeven-pricing).
Usage: [Inflation guide](../guide/inflation.md).

### Zero-coupon inflation swap (ZCIS)
A single maturity exchange of cumulative inflation for a fixed compounded rate: $\big(\text{CPI}_T/\text{CPI}_0 - (1+k)^{T}\big)\,N$. The quoted fixed rate is the breakeven inflation rate. See [Theory §3.6](../theory/curves.md#36-inflation-curves-and-breakeven-pricing).

### Year-on-year inflation swap (YYIS)
Periodic exchange of realised **annual** inflation $\text{CPI}_i/\text{CPI}_{i-1}-1$ for a fixed rate. A small convexity/timing adjustment (currently ignored) separates it from a strip of ZCIS. See [Theory §3.6](../theory/curves.md#year-on-year-inflation-swaps-yyis).

### Inflation cap / floor
A strip of options on year-on-year CPI returns, $(\text{YoY}_i - K)^+$, priced Black-76 on the forward YoY rate. See [Theory §3.6](../theory/curves.md#36-inflation-curves-and-breakeven-pricing).

---

## See also

- [API: Instruments](../api/instruments.md) — field-level reference for every pytree.
- [Usage guides](#usage-guides) — per-asset-class how-to pages with runnable examples.
- [Models & Theory](../theory/index.md) — the mathematical foundations linked throughout.
- [Roadmap](../roadmap.md) — sequencing of the 🟡 pricers.
