# Instruments

All instruments are `equinox.Module` subclasses — frozen dataclasses
registered as JAX pytrees. **They carry no pricing logic** (pricing
lives in [`valax.pricing`](pricing.md); Greeks in
[`valax.greeks`](greeks.md)). Instruments are data-only, which is what
makes `jax.grad` through a price w.r.t. any instrument field work
without special-casing per contract type.

!!! note "Static vs. differentiable fields"
    Fields marked with `eqx.field(static=True)` (e.g. `is_call`,
    `day_count`, `frequency`) control code paths or metadata and are
    **not** differentiable — JAX traces separate code paths per static
    value. Every other field is a `Float`/`Int` JAX array leaf that
    `jax.grad` will flow through. Batch pricing via `jax.vmap` builds a
    single instance with batched arrays for the non-static fields.

## Equity options and exotics

### European

::: valax.instruments.EuropeanOption

### American

Exercisable at any time up to expiry. Pricing requires methods that
handle early exercise (binomial trees, PDE with free boundary, or
Longstaff–Schwartz MC). For calls on non-dividend-paying stocks, the
American price equals the European price (early exercise is never
optimal). For puts, or when dividends are present, the early exercise
premium is positive.

::: valax.instruments.AmericanOption

### Barrier

European option that activates (knock-in) or deactivates (knock-out)
when spot breaches the barrier. Knock-in / knock-out parity:
`knock_in_price + knock_out_price = vanilla_price` for the same
strike, barrier, and type. Barrier monitoring is continuous for
analytical pricing, discrete (per time step) for Monte Carlo.
`smoothing > 0` replaces the barrier indicator with a sigmoid so
pathwise Greeks stay well-defined near the barrier.

::: valax.instruments.EquityBarrierOption

### Asian

Payoff depends on the average spot price over observation dates
rather than terminal spot. No closed-form price exists for arithmetic
averages under BSM; geometric averages do have a closed form. Lower
vega and gamma than equivalent vanillas because averaging reduces
effective volatility.

::: valax.instruments.AsianOption

### Lookback

Floating-strike: call payoff is \(S_T - \min(S_t)\), put payoff is
\(\max(S_t) - S_T\). Fixed-strike: call payoff is
\(\max(\max(S_t) - K, 0)\), put payoff is \(\max(K - \min(S_t), 0)\).
Floating-strike lookbacks are always in the money (non-negative
payoff by construction).

::: valax.instruments.LookbackOption

### Variance swap

::: valax.instruments.VarianceSwap

### Compound option

Option on an option — a European call/put whose underlying is another
European option. Priced via nested Black–Scholes.

::: valax.instruments.CompoundOption

### Chooser option

At the choice date, the holder chooses whether the option becomes a
call or a put with the same strike and expiry.

::: valax.instruments.ChooserOption

### Autocallable

Structured note that automatically redeems if the underlying is above
an autocall barrier on an observation date.

::: valax.instruments.Autocallable

### Worst-of basket

Payoff on the worst-performing of a basket of assets. Highly
correlation-sensitive.

::: valax.instruments.WorstOfBasketOption

### Cliquet

Series of forward-starting options with periodic strike resets. Pays
the sum of per-period returns subject to local and/or global caps and
floors.

::: valax.instruments.Cliquet

### Digital

Pays a fixed amount if the underlying finishes in the money; zero
otherwise. Priced as the derivative of a European price w.r.t.
strike.

::: valax.instruments.DigitalOption

### Spread option

Payoff on the difference of two asset prices. Analytic pricers are
Margrabe (\(K = 0\)) and Kirk (\(K \neq 0\)); see
[`spread_option_price`](pricing.md#spread-options).

::: valax.instruments.SpreadOption

## Fixed income

### Zero-coupon bond

::: valax.instruments.ZeroCouponBond

### Fixed-rate bond

Use `generate_schedule()` from
[`valax.dates`](dates.md#schedule-generation) to build
`payment_dates`. Only future cash flows
(`payment_date > settlement_date`) are included in pricing.

::: valax.instruments.FixedRateBond

### Floating-rate note

Coupon resets periodically off a reference rate. Priced under the
single-curve assumption via
[`floating_rate_bond_price`](pricing.md#floating-rate-instruments);
satisfies the par-at-reset invariant (a zero-spread FRN on its first
reset date prices to face value).

::: valax.instruments.FloatingRateBond

### Callable / puttable / convertible bonds

Priced via backward induction on a Hull–White trinomial tree; see
[Callable / puttable bonds](pricing.md#callable--puttable-bonds).
Callable price is bounded above by the equivalent straight bond;
puttable price is bounded below.

::: valax.instruments.CallableBond

::: valax.instruments.PuttableBond

::: valax.instruments.ConvertibleBond

## Interest rate derivatives

### Caplet / cap

Single-period rate options. `Cap` is a strip of `Caplet`s over a
payment schedule.

::: valax.instruments.Caplet

::: valax.instruments.Cap

### Swap and swaption

::: valax.instruments.InterestRateSwap

::: valax.instruments.Swaption

### Bermudan swaption

Swaption exercisable at multiple discrete dates. Priced via
Longstaff–Schwartz on LMM paths; see
[`bermudan_swaption_lsm`](pricing.md#bermudan-longstaffschwartz).

::: valax.instruments.BermudanSwaption

### OIS swap

Overnight-index swap. Float leg uses the telescoping identity
\(N \cdot (DF(T_0) - DF(T_n))\); see
[`ois_swap_price`](pricing.md#floating-rate-instruments).

::: valax.instruments.OISSwap

### Cross-currency and total-return swaps

::: valax.instruments.CrossCurrencySwap

::: valax.instruments.TotalReturnSwap

### CMS instruments and range accrual

CMS pricers use per-period forward par swap rates on a synthetic
annual underlying swap. **No convexity adjustment** — see the
[rates-exotics guide](../guide/rates-exotics.md) for caveats.

::: valax.instruments.CMSSwap

::: valax.instruments.CMSCapFloor

::: valax.instruments.RangeAccrual

## FX derivatives

FX instruments are quoted in terms of a **currency pair** `FOR/DOM`
(e.g. EUR/USD means 1 EUR = X USD). The **foreign** currency is the
asset (numeraire of the option payoff), and the **domestic** currency
is the pricing currency. In Garman–Kohlhagen terms: `spot` is the
price of 1 unit of foreign currency in domestic terms, `r_domestic`
is the domestic risk-free rate, and `r_foreign` is the foreign
risk-free rate (which acts like a dividend yield).

::: valax.instruments.FXForward

::: valax.instruments.FXVanillaOption

::: valax.instruments.FXBarrierOption

::: valax.instruments.QuantoOption

### TARF (Target Redemption Forward)

::: valax.instruments.TARF

### FX swap

::: valax.instruments.FXSwap

## Credit derivatives

Credit derivatives transfer credit risk between counterparties. The
fundamental building block is the CDS, which provides insurance
against default of a reference entity. See
[`SurvivalCurve`](curves.md#survival-credit-curve) for the
underlying term structure.

::: valax.instruments.CDS

::: valax.instruments.CDOTranche

## Inflation derivatives

Inflation derivatives are linked to a Consumer Price Index (CPI) or
similar inflation index. Inflation indices are published with a lag
(typically 2–3 months). CPI ratios are used
(\(CPI(T) / CPI(0)\)), not absolute index levels; seasonality
adjustments may be required for monthly indices.

### Zero-coupon inflation swap (ZCIS)

Single exchange at maturity of the cumulative inflation return vs. a
fixed rate.

::: valax.instruments.ZeroCouponInflationSwap

### Year-on-year inflation swap (YYIS)

Periodic exchanges of annual inflation vs. a fixed rate.

::: valax.instruments.YearOnYearInflationSwap

### Inflation cap / floor

Option overlays on year-on-year inflation.

::: valax.instruments.InflationCapFloor
