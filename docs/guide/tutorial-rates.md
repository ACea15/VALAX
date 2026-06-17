# Tutorial — Rates Products End-to-End on Synthetic Data

This tutorial walks the full quant pipeline on a **single, self-contained
example**: we generate synthetic market data with the VALAX engine, calibrate a
yield curve to noisy par-swap quotes, and then use the calibrated curve to price
an off-market interest-rate swap and a European swaption. Every stage spells
out the modelling assumptions that go into it.

The intent is pedagogical: you should be able to copy each block, run it, and
get the same printout as the text. Nothing here depends on external market data.

| Stage | What we build | Module |
|-------|---------------|--------|
| 1 | Ground-truth NSS USD curve (the "real world") | `valax.market.synthetic.curves` |
| 2 | Noisy par-rate quotes from liquid hedging instruments | `valax.market.synthetic.observations` |
| 3 | Bootstrapped `DiscountCurve` calibrated to those quotes | `valax.curves.bootstrap` |
| 4 | Validation: residual ≈ observation noise | this page |
| 5 | Pricing: 5Y IRS + 1Y×5Y European swaption | `valax.pricing.analytic.swaptions` |
| 6 | Risk: bucketed DV01 and per-quote DV01 via autodiff | `jax.grad` through the bootstrap |

## 0. Setup

We only need the JAX numerics imports and the VALAX subpackages we'll touch in
each stage. A single `SeedRegistry` drives every random draw so the run is
bit-for-bit reproducible.

```python
import jax
import jax.numpy as jnp
import equinox as eqx

import valax
from valax.dates import ymd_to_ordinal, year_fraction, generate_schedule

# Stage 1 — synthetic ground truth.
from valax.market.synthetic import (
    SeedRegistry,
    SyntheticMarketConfig,
    sample_nss_curve,
    synthesize_curve_quotes,
)

# Stage 3 — bootstrap.
from valax.curves import (
    DiscountCurve,
    DepositRate,
    SwapRate,
    bootstrap_simultaneous,
    zero_rate,
)

# Stage 5 — pricing.
from valax.instruments import InterestRateSwap, Swaption
from valax.pricing.analytic.swaptions import (
    swap_rate,
    swap_price,
    swaption_price_black76,
)

registry = SeedRegistry(master_seed=20260101, library_version=valax.__version__)
```

!!! tip "Why JAX float64?"
    Rates work — small DFs, log-DF interpolation, tiny par-rate residuals — is
    intolerant of single precision. The VALAX test suite runs under
    `jax.config.update("jax_enable_x64", True)`; you should too. Put this at
    the top of any production script that touches curves.

## 1. Ground-Truth World: a Synthetic NSS Curve

A real desk pulls the day's quotes from Bloomberg / Refinitiv. In a tutorial we
need the opposite direction: pick a **truth** that we know exactly, then derive
the quotes a trading desk would see *if* the truth happened to be the day's
market. We use the Nelson-Siegel-Svensson family because it is the workhorse
parametric shape for term-structure modelling at central banks and at the BIS.

```python
cfg = SyntheticMarketConfig(
    curve_kind="nss",
    reference_date=ymd_to_ordinal(2026, 3, 16),
    day_count="act_365",
)
truth_curve = sample_nss_curve(registry, cfg)

ref = int(cfg.reference_date)
print(f"reference date  : {ref}")
print(f"pillar count    : {truth_curve.discount_factors.shape[0]}")
print(f"DF at 1Y / 5Y / 10Y / 30Y:")
for years in (1.0, 5.0, 10.0, 30.0):
    df = float(truth_curve(jnp.int32(ref + int(round(years * 365)))))
    z = float(zero_rate(truth_curve, jnp.int32(ref + int(round(years * 365)))))
    print(f"  {years:>4.0f}Y  DF={df:.6f}  zero={z * 100:.3f}%")
```

### Assumptions baked into Stage 1

- **Parametric truth.** We are *defining* the truth to be a six-parameter NSS
  curve. Real curves do not have to be NSS-shaped; using NSS here means we
  cannot test the bootstrap's ability to recover non-parametric kinks (e.g. a
  turn-of-year jump). The synthetic generator has a `TurnInstrument` quote
  type for that — out of scope for this tutorial.
- **Log-linear interpolation off the pillars.** `DiscountCurve` interpolates
  discount factors log-linearly between pillars, which gives piece-wise
  constant continuously-compounded forward rates. The NSS truth has *smooth*
  forward rates, so even a perfect bootstrap will show step-shaped forwards.
  The curve values **at** the pillars are exact; only between pillars do we
  see interpolation artefacts.
- **Single-curve world.** We are not yet distinguishing OIS discounting from
  3M / 6M projection. Post-2008 a real desk runs a `MultiCurveSet` — see the
  [Curves guide §5](curves.md#5-multi-curve-bootstrap). The principles below
  generalise unchanged; only the instrument list grows.
- **No bid-offer, no holiday calendar, ACT/365 throughout.** Real
  instruments use ACT/360 for money-market legs, business-day-adjusted
  schedules, and a quoted bid-offer spread. We collapse those into the
  zero-mean Gaussian noise we apply in Stage 2.

## 2. Observation Layer: Noisy Quotes a Desk Would See

A trader looks at par rates, not at discount factors. We pick the standard
USD swap calibration grid (3M and 6M deposits at the short end, then 1Y / 2Y /
3Y / 5Y / 7Y / 10Y par swaps) and derive what each one's *fair* quote would be
under our truth curve, then add Gaussian basis-point noise.

```python
# ── 2.1 Standard calibration grid ──────────────────────────────────
# Short end: two money-market deposits.
deposit_tenors_months = (3, 6)
# Swap belly: semi-annual fixed legs from 1Y to 10Y.
swap_tenors_years = (1, 2, 3, 5, 7, 10)

ref_y, ref_m, ref_d = 2026, 3, 16

def add_months(year: int, month: int, n: int) -> tuple[int, int]:
    m_total = month - 1 + n
    return year + m_total // 12, (m_total % 12) + 1

# Build the (start, end) date pairs for each deposit.
deposit_dates = [
    (
        jnp.asarray(ref, dtype=jnp.int32),
        jnp.asarray(
            ymd_to_ordinal(*add_months(ref_y, ref_m, m), ref_d),
            dtype=jnp.int32,
        ),
    )
    for m in deposit_tenors_months
]

# Build the semi-annual fixed-leg schedule for each swap.
swap_schedules = [
    generate_schedule(
        ref_y, ref_m, ref_d,
        ref_y + n, ref_m, ref_d,
        frequency=2,
    )
    for n in swap_tenors_years
]
```

### 2.2 Clean par rates implied by the truth curve

For deposits, the clean simply-compounded rate is

$$r_\text{depo} = \frac{1}{\tau}\!\left(\frac{1}{DF(\text{end})} - 1\right),$$

and for par swaps it is the standard ratio of the floating-leg replication to
the fixed-leg annuity:

$$r_\text{swap} = \frac{DF(\text{start}) - DF(T_n)}{\sum_i \tau_i\, DF(T_i)}.$$

We use VALAX's `swap_rate` helper for the second one, keeping our code aligned
with the pricer.

```python
def clean_deposit_rate(start, end, curve, day_count="act_360"):
    df_end = curve(end)
    tau = year_fraction(start, end, day_count)
    return (1.0 / df_end - 1.0) / tau

def clean_swap_rate(start, fixed_dates, curve, day_count="act_360"):
    # Use the same primitive the pricer uses, so calibration and pricing
    # cannot drift apart.
    template = InterestRateSwap(
        start_date=start,
        fixed_dates=fixed_dates.astype(jnp.int32),
        fixed_rate=jnp.asarray(0.0),
        notional=jnp.asarray(1.0),
        pay_fixed=True,
        day_count=day_count,
    )
    return swap_rate(template, curve)

ref_arr = jnp.asarray(ref, dtype=jnp.int32)

clean_par_rates = jnp.stack(
    [clean_deposit_rate(s, e, truth_curve) for (s, e) in deposit_dates]
    + [clean_swap_rate(ref_arr, sch, truth_curve) for sch in swap_schedules]
)
```

### 2.3 Sprinkle observation noise

`synthesize_curve_quotes` adds zero-mean Gaussian noise with a 1-σ width in
basis points. Two bp is on the optimistic end of broker spreads for major
liquid swaps; we use it here so the calibration residual stays clearly visible
above floating-point error.

```python
NOISE_BP = 2.0  # 1-σ broker bid-offer-like noise

noisy_par_rates = synthesize_curve_quotes(
    registry, clean_par_rates, bp_noise=NOISE_BP,
)

labels = ([f"{m}M depo" for m in deposit_tenors_months]
          + [f"{n}Y swap" for n in swap_tenors_years])
print("\n     instrument |   clean   |   noisy   |  Δ (bp)")
for lab, clean, noisy in zip(labels, clean_par_rates, noisy_par_rates):
    delta_bp = float(noisy - clean) * 1e4
    print(f"  {lab:>11}  | {float(clean) * 100:7.4f}% | "
          f"{float(noisy) * 100:7.4f}% | {delta_bp:+6.2f}")
```

### Assumptions baked into Stage 2

- **Zero-mean Gaussian noise.** Real quote noise is **bid-offer**, which is
  asymmetric (the mid-mark is bias-prone), and heavier-tailed than Gaussian
  (think CDS / inflation crisis days). For a back-test of calibration
  *stability* this is fine; for stress testing tail behaviour you would want
  bootstrap resampling or a Student-t kernel.
- **One quote per instrument.** A real desk has dozens of overlapping quotes
  (Eurodollar / SOFR futures plus FRAs plus swaps in the same window). The
  cleaner the over-determined system, the more important the **simultaneous**
  solver (next stage) becomes.

## 3. Calibration: Bootstrap the Curve from Noisy Quotes

Now we step into a trading desk's shoes: we **forget** the truth curve and
recover a curve from the noisy quotes alone. Each quote becomes a
`BootstrapInstrument` — a pytree whose `residual(graph, fixings, ref) == 0`
encodes the instrument's no-arbitrage relation on the curve.

```python
# ── 3.1 Pack the quotes into BootstrapInstrument pytrees ──────────
quote_instruments = []

# Deposits: end-of-deposit pillar.
for (start, end), rate in zip(deposit_dates, noisy_par_rates[:2]):
    quote_instruments.append(
        DepositRate(
            start_date=start,
            end_date=end,
            rate=jnp.asarray(rate),
            day_count="act_360",
        )
    )

# Swaps: maturity pillar.
for fixed_dates, rate in zip(swap_schedules, noisy_par_rates[2:]):
    quote_instruments.append(
        SwapRate(
            start_date=ref_arr,
            fixed_dates=fixed_dates.astype(jnp.int32),
            rate=jnp.asarray(rate),
            day_count="act_360",
        )
    )

# ── 3.2 Pillar grid: one pillar per quote ──────────────────────────
pillar_dates = jnp.stack(
    [inst.end_date for inst in quote_instruments[:2]]
    + [inst.fixed_dates[-1] for inst in quote_instruments[2:]]
).astype(jnp.int32)

# ── 3.3 Run the simultaneous bootstrap ────────────────────────────
calibrated_curve = bootstrap_simultaneous(
    reference_date=ref_arr,
    pillar_dates=pillar_dates,
    instruments=quote_instruments,
    day_count="act_365",
)

print("\n--- Calibrated curve vs truth at the pillars ---")
print("  tenor |  zero (truth) |  zero (cal)   |  Δ (bp)")
for years in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0):
    d = jnp.int32(ref + int(round(years * 365)))
    z_truth = float(zero_rate(truth_curve, d))
    z_cal = float(zero_rate(calibrated_curve, d))
    print(f"  {years:>5.2f}Y | {z_truth * 100:9.4f}% | "
          f"{z_cal * 100:9.4f}% | {(z_cal - z_truth) * 1e4:+6.2f}")
```

### Assumptions baked into Stage 3

- **Square system.** `bootstrap_simultaneous` requires exactly one
  instrument per pillar. With over-determined quote sets you must either
  drop instruments or move to a least-squares fit (roadmap; see
  [MC-Curves-2](../architecture/mc-curves-2.md)).
- **Log-linear interpolation between pillars** (built into `DiscountCurve`).
  The Newton solve happens in **log-DF space**, guaranteeing positive DFs
  at every iterate. Forward rates between pillars are constant; if your
  product is sensitive to a specific forward-rate shape (e.g. a forward
  starting on a non-pillar date) you may want a monotone-convex interpolator
  — roadmapped, not yet shipped.
- **Single-curve discounting.** OIS discount + tenor-specific forward
  projection is the production setup. The math here generalises one-for-one;
  see [`bootstrap_multi_curve`](curves.md#5-multi-curve-bootstrap).
- **Exact analytic Jacobian.** Newton's $\partial R_i / \partial x_j$ is
  computed by `jax.jacobian`, **not** a finite-difference bump. That's why
  the solver converges in a handful of iterations and why differentiating
  *through* the bootstrap (Stage 6) is essentially free via `optimistix`'s
  `ImplicitAdjoint`.

## 4. Validation: Are Residuals Inside the Noise Floor?

A calibration is "good" when the **per-instrument repricing residual** is at
or below the **observation noise** you fed it. Anything materially smaller
means you have overfit; anything materially larger means your model is missing
a feature the market is pricing.

```python
def fitted_par_rates(curve):
    return jnp.stack(
        [clean_deposit_rate(s, e, curve) for (s, e) in deposit_dates]
        + [clean_swap_rate(ref_arr, sch, curve) for sch in swap_schedules]
    )

residuals_bp = (fitted_par_rates(calibrated_curve) - noisy_par_rates) * 1e4
print("\n     instrument |  residual (bp)")
for lab, r in zip(labels, residuals_bp):
    print(f"  {lab:>11}  | {float(r):+8.4f}")

rms_bp = float(jnp.sqrt(jnp.mean(residuals_bp ** 2)))
print(f"\n  RMS residual : {rms_bp:.4f} bp   "
      f"(observation noise 1-σ was {NOISE_BP:.1f} bp)")
```

For a square Newton system the residual should be at machine precision (the
bootstrap *exactly* reprices the noisy quotes by construction). The "noise"
shows up instead as the **gap between calibrated and truth curves** that you
printed at the end of Stage 3.

## 5. Pricing: A 5Y IRS and a 1Y×5Y Swaption

We can now treat the calibrated curve as our valuation curve. Two illustrative
exotics:

1. An **off-market 5Y payer IRS** paying a fixed coupon **25 bp above par**.
   Its NPV is non-zero precisely because the fixed rate differs from the par
   rate — a payer of an above-par fixed coupon is *out of the money*.
2. A **1Y × 5Y European payer swaption** struck **at the forward swap rate**
   (i.e. at-the-money-forward). For the swaption we need a **swaption
   volatility**; in real life this comes from a separate calibration of the
   swaption cube (SABR per (expiry, tail)). Here we plug in a flat 30 % Black
   vol — exactly the kind of stand-in input the downstream desk would replace
   with a calibrated cube.

We *derive* both the strikes from the calibrated curve so the tutorial stays
sensible regardless of which random NSS draw we got in Stage 1.

```python
# ── 5.1 5Y payer IRS at +25 bp over par ───────────────────────────
five_y_sched = swap_schedules[swap_tenors_years.index(5)].astype(jnp.int32)

# First, ask the curve what the par 5Y rate is.
par_template = InterestRateSwap(
    start_date=ref_arr,
    fixed_dates=five_y_sched,
    fixed_rate=jnp.asarray(0.0),         # placeholder; not used by swap_rate
    notional=jnp.asarray(1.0),
    pay_fixed=True,
    day_count="act_360",
)
par_5y = swap_rate(par_template, calibrated_curve)

irs = InterestRateSwap(
    start_date=ref_arr,
    fixed_dates=five_y_sched,
    fixed_rate=par_5y + 0.0025,          # 25 bp above market
    notional=jnp.asarray(50_000_000.0),  # USD 50 M
    pay_fixed=True,
    day_count="act_360",
)

npv_irs = float(swap_price(irs, calibrated_curve))
print("\n--- 5Y payer IRS @ par + 25 bp ---")
print(f"  par swap rate (cal): {float(par_5y) * 100:.4f} %")
print(f"  contract fixed rate: {float(irs.fixed_rate) * 100:.4f} %")
print(f"  NPV                : USD {npv_irs:,.0f}")
print(f"  sign check         : payer of an above-par coupon is "
      f"{'ITM' if npv_irs > 0 else 'OTM'} ✓")

# ── 5.2 1Y × 5Y European payer swaption struck ATM-forward ────────
# Underlying: 5Y semi-annual swap starting in 1Y.
expiry = jnp.asarray(ymd_to_ordinal(ref_y + 1, ref_m, ref_d), dtype=jnp.int32)
und_schedule = generate_schedule(
    ref_y + 1, ref_m, ref_d, ref_y + 6, ref_m, ref_d, frequency=2,
).astype(jnp.int32)

# Forward 1Y×5Y par swap rate, via the same primitive the pricer uses.
fwd_template = InterestRateSwap(
    start_date=expiry,
    fixed_dates=und_schedule,
    fixed_rate=jnp.asarray(0.0),
    notional=jnp.asarray(1.0),
    pay_fixed=True,
    day_count="act_360",
)
forward_swap_rate = swap_rate(fwd_template, calibrated_curve)

swaption = Swaption(
    expiry_date=expiry,
    fixed_dates=und_schedule,
    strike=forward_swap_rate,           # ATM-forward
    notional=jnp.asarray(50_000_000.0),
    is_payer=True,
    day_count="act_360",
)

black_vol = jnp.asarray(0.30)  # placeholder — would come from a swaption cube
swaption_npv = float(
    swaption_price_black76(swaption, calibrated_curve, black_vol)
)
print("\n--- 1Y × 5Y payer swaption (ATM-forward) ---")
print(f"  fwd 1Y×5Y swap rate: {float(forward_swap_rate) * 100:.4f} %")
print(f"  Black vol (assumed): {float(black_vol) * 100:.1f} %")
print(f"  NPV                : USD {swaption_npv:,.0f}")
print(f"  running premium    : {swaption_npv / 50e6 * 1e4:.1f} bp of notional")
```

The ATM-forward strike is the natural choice for a tutorial because it makes
the swaption value depend only on the **vol times sqrt-T** combination — the
curve-dependent intrinsic value drops out. In a production book most quoted
swaptions cluster near ATM-forward for exactly this reason: it's where
vol-traders prefer to express risk.

### Assumptions baked into Stage 5

- **Same curve for projection and discounting.** As in Stage 1, our IRS uses
  the single calibrated curve for both forward-rate projection (via the
  replication identity $PV_\text{float} = DF(\text{start}) - DF(\text{end})$)
  and for discounting. A multi-curve setup splits these.
- **Black-76 lognormality.** `swaption_price_black76` assumes the forward
  swap rate is **lognormal** under the annuity measure. For low-rate
  regimes (eurozone 2015-21 territory) you should switch to
  `swaption_price_bachelier`, which assumes a normal forward rate — both
  pricers live in `valax.pricing.analytic.swaptions`.
- **Constant Black vol.** A single `black_vol` scalar implicitly says the
  vol surface is flat in strike and expiry. In production this is one node
  of a SABR-fitted swaption cube — see [Volatility Surfaces](vol-surfaces.md).

## 6. Risk: Two Flavours of DV01 via Autodiff

The reason we built everything as pure JAX pytrees is that `jax.grad` now does
all our risk work. We compute the IRS DV01 in two complementary ways.

### 6.1 Per-pillar DV01 (bucketed)

Differentiating the IRS NPV w.r.t. the **discount factors at each pillar**
gives a vector of bucket sensitivities — these are the curve-level "Greeks".

```python
def irs_npv_from_dfs(dfs):
    new_curve = eqx.tree_at(
        lambda c: c.discount_factors, calibrated_curve, dfs
    )
    return swap_price(irs, new_curve)

dP_dDF = jax.grad(irs_npv_from_dfs)(calibrated_curve.discount_factors)

# Convert ∂NPV/∂DF_i into a 1 bp parallel-pillar bump effect.
pillar_times = year_fraction(ref_arr, pillar_dates, "act_365")
# A 1bp upward shift in the zero rate at pillar i shifts DF_i by
# (-tau_i * 1bp * DF_i); summing all such shifts gives parallel DV01.
parallel_dv01 = float(
    jnp.sum(-pillar_times * calibrated_curve.discount_factors * 1e-4 * dP_dDF)
)
print("\n--- IRS Greeks via autodiff ---")
print(f"  parallel DV01           : USD {parallel_dv01:,.2f} per 1 bp shift")

# Per-pillar contribution (bucketed): ∂NPV / ∂(zero rate at pillar i)
bucketed_dv01 = -pillar_times * calibrated_curve.discount_factors * dP_dDF * 1e-4
print("  bucketed DV01 per pillar:")
for lab, sens in zip(labels, bucketed_dv01):
    print(f"    {lab:>11}: USD {float(sens):,.2f}")
```

### 6.2 Per-quote DV01 (through the bootstrap)

The bucketed DV01 above is the desk's *internal* representation. The hedging
desk lives in **quote space** — it can only trade the original deposits and
swaps. Differentiating end-to-end from `noisy_par_rates → curve → IRS price`
gives the sensitivity to each *quoted instrument* directly:

```python
def irs_npv_from_quotes(quotes):
    # Rebuild the BootstrapInstrument list with the new quotes.
    new_insts = [
        DepositRate(
            start_date=inst.start_date,
            end_date=inst.end_date,
            rate=quotes[i],
            day_count=inst.day_count,
        )
        if isinstance(inst, DepositRate)
        else SwapRate(
            start_date=inst.start_date,
            fixed_dates=inst.fixed_dates,
            rate=quotes[i],
            day_count=inst.day_count,
        )
        for i, inst in enumerate(quote_instruments)
    ]
    new_curve = bootstrap_simultaneous(
        reference_date=ref_arr,
        pillar_dates=pillar_dates,
        instruments=new_insts,
        day_count="act_365",
    )
    return swap_price(irs, new_curve)

quote_dv01 = jax.grad(irs_npv_from_quotes)(noisy_par_rates) * 1e-4
print("\n  quote DV01 per market instrument (USD per 1 bp):")
for lab, sens in zip(labels, quote_dv01):
    print(f"    {lab:>11}: USD {float(sens):,.2f}")
```

These two vectors are related by the Jacobian of the bootstrap
$\partial DF / \partial \text{quote}$. `jax.grad` composes them in one
reverse-mode pass via `optimistix.ImplicitAdjoint` — in a bump-and-reprice
system this would cost you *N* bootstrap re-runs plus *N* re-pricings.

## 7. What We Glossed Over

| Topic | Where to read more |
|-------|--------------------|
| Multi-curve (OIS discount + tenor projection) | [Curves §5](curves.md#5-multi-curve-bootstrap) |
| Non-NSS curve shapes, monotone-convex interpolation | [Theory §3](../theory.md#3-curve-framework) |
| Calibrating the swaption *cube* (SABR per slice) | [Volatility Surfaces](vol-surfaces.md), [Calibration §3](calibration.md#3-sabr-smile-calibration) |
| Hull-White $(a, \sigma)$ from swaptions (Jamshidian) | [Short-Rate Models](short-rate.md) |
| Pricing exotics — callable bonds, CMS, range accruals | [Callable Bonds](callable-bonds.md), [Interest Rate Exotics](rates-exotics.md) |
| Reproducibility contract, arbitrage stress tests | [Reproducibility & Arbitrage Tests](reproducibility_and_arbitrage_tests.md) |
| The full six-stage synthetic pipeline | [Synthetic Market Data](synthetic_market.md), `examples/08_end_to_end_workflow.py` |

The same six-stage skeleton — **truth → noisy quotes → calibration →
validation → pricing → autodiff risk** — applies unchanged to the equity,
FX, inflation, and credit tutorials that follow this one. The only thing that
changes is the calibration object (SABR/SVI surface, Heston, hazard curve)
and the list of liquid instruments. See the [Calibration matrix](calibration.md#calibration-matrix)
for the full picture.
