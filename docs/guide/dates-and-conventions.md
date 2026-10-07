# Dates and Conventions

This page owns **schedules, day counts, accrual and the clean/dirty
distinction**. Instrument payoffs live on their asset-class guides
([Fixed Income](fixed-income.md), [Callable Bonds](callable-bonds.md),
[Rates Exotics](rates-exotics.md)); engine mechanics live on the solver
guides. Here we cover only how a pair of dates becomes a set of cashflows.

## Why a schedule is three arrays, not one

A coupon is defined by two different things: the **period it accrues over**,
which sets its size, and the **date it pays**, which sets its discounting.
`generate_schedule` therefore returns both:

```python
from valax.dates import generate_schedule

accrual_start, accrual_end, payment_dates = generate_schedule(
    2025, 1, 31, 2030, 1, 31, frequency=2
)
```

Period `i` accrues over `[accrual_start[i], accrual_end[i])` and pays on
`payment_dates[i]`. The three arrays are contiguous —
`accrual_start[i + 1] == accrual_end[i]` — so accrual partitions the tenor
exactly, with no gaps and no double-counting.

`payment_dates` currently always equals `accrual_end`. It is a separate array
because business-day adjustment moves a payment without moving the accrual it
pays for. That is [Tier 1.3](../roadmap.md#13-business-calendars-and-holiday-schedules);
keeping the two distinct now means nothing downstream has to change when it
lands.

The return value is a `Schedule` pytree, so `sched.payment_dates` works as
well as tuple unpacking.

## Day counts

| Name | Convention | Notes |
|---|---|---|
| `"act_360"` | Actual/360 | Money-market default |
| `"act_365"` | Actual/365 Fixed | |
| `"act_act"` | Actual/Actual (ISDA) | Each day weighted by its own year's length |
| `"30_360"` | 30/360 US bond basis | All periods equal by construction |

```python
from valax.dates import year_fraction, ymd_to_ordinal

start, end = ymd_to_ordinal(2003, 11, 1), ymd_to_ordinal(2004, 5, 1)
year_fraction(start, end, "act_act")   # 61/365 + 121/366 = 0.4977...
```

ACT/ACT ISDA splits the period at calendar-year boundaries and divides the days
in each year by that year's own length. It is computed in closed form, so it is
a single vectorised expression and safe inside JIT.

### The conventions matter — measurably

Conventions are not cosmetic. For a 5% semiannual bond over a 181-day period,
Act/360 pays 2.5139 per 100 of face and Act/365 pays 2.4795 — a 1.4%
difference **per coupon**, in the same direction on every coupon, which is how
it reaches basis points of yield. Picking the wrong basis is a mispricing, not
a rounding choice.

30/360 is the convention under which every period is exactly `1 / frequency`.
Under any actual/N convention the 181- and 184-day half-years of a semiannual
bond pay **different amounts**, which is why coupon sizing must go through the
day count:

```python
from valax.dates import fixed_coupon_amounts

fixed_coupon_amounts(bond)   # e.g. [2.4794, 2.5205, 2.4794, 2.5205]
```

This has a visible consequence: the textbook par identity — a bond priced at a
yield equal to its coupon trades at exactly 100 — holds only under 30/360.
Under Act/365 the same 6% bond prices at 100.010, and under Act/360 at 100.366.

## Conventions: stubs, rolls and the calendar slot

`ScheduleConventions` is a frozen pytree with all-static fields:

```python
from valax.dates import ScheduleConventions, generate_schedule

conv = ScheduleConventions(
    day_count="act_365",
    stub="short_front",     # 'long_front', 'short_back', 'long_back', 'none'
    end_of_month=False,
    roll_day=None,          # None infers from the generation anchor
)
sched = generate_schedule(2025, 4, 20, 2026, 10, 15, frequency=4, conventions=conv)
```

### Day-of-month preservation

The roll day is re-applied from the anchor each period, so it is clamped only
where the target month is too short and never ratchets down:
31 Jan → 28 Feb → **31 Mar**, not 31 Jan → 28 Feb → 28 Mar.

### End-of-month

With `end_of_month=True` every generated date takes the last day of its own
month, so 29 Feb → 31 Aug → 28 Feb. With `end_of_month=False` the
day-of-month is preserved instead, giving 29 Feb → 28 Aug → 28 Feb.

### Stubs

When the tenor is not a whole number of coupon periods, the odd period is a
**stub**. Front stubs roll backward from maturity (so maturity lands on its
stated day); back stubs roll forward from the dated date.

| `stub` | Effect |
|---|---|
| `"short_front"` | A period shorter than regular, first |
| `"long_front"` | The odd days merged into the first regular period |
| `"short_back"` | A short period at maturity |
| `"long_back"` | The odd days merged into the last regular period |
| `"none"` | Assert the tenor divides evenly; raise if it does not |

A long stub is simply a short stub with the boundary between it and its
neighbour removed, so it produces one fewer period.

!!! warning "Supply `accrual_start_dates` for stubbed schedules"
    When an instrument carries only `payment_dates`, accrual periods are
    reconstructed as running from the previous payment date. That is exact for
    a regular schedule but **cannot see a stub**. Pass the generator's
    `accrual_start` to the instrument's `accrual_start_dates` field whenever
    the schedule is irregular.

### The Tier 1.3 slot

`calendar` and `business_day_convention` exist and are validated, but are not
yet consumed. Supplying either raises `NotImplementedError`:

```python
ScheduleConventions(business_day_convention="modified_following")
# NotImplementedError: ... not implemented yet (roadmap Tier 1.3)
```

Accepting a calendar and ignoring it would reproduce exactly the failure this
module exists to prevent — a convention that is configured but never applied —
so it fails loudly instead.

## Accrued interest, clean and dirty

Every price VALAX produces is a **dirty** price: the present value of all
remaining cashflows, including interest accrued since the last coupon date.
Market quotes for bonds, and the strike of an embedded call, are **clean**.

```python
from valax.dates import bond_accrued_interest, clean_from_dirty
from valax.pricing.analytic.bonds import fixed_rate_bond_price

dirty = fixed_rate_bond_price(bond, curve)
accrued = bond_accrued_interest(bond)
clean = clean_from_dirty(dirty, accrued)
```

Accrued interest is zero **on** a coupon date, by construction: that date ends
one accrual period and starts the next. It grows monotonically within a period
and resets across each coupon. It is also JIT-safe and differentiable — the
period search is a masked reduction, not a Python branch.

### Where the distinction bites

**Option-adjusted spread.** The OAS solvers invert a dirty repricing map, so
they take an explicit `price_type` rather than guessing:

```python
from valax.risk.oas import bond_z_spread

bond_z_spread(bond, curve, market_quote, price_type="clean")
```

Feeding a clean quote to the dirty map without converting biases the recovered
spread by the accrued interest. For a 5% bond three months into its period that
is **28 basis points** — one-directional, not noise.

**Callable and puttable exercise.** A call price is quoted clean; the
continuation value it is compared against is dirty. An issuer calling
mid-period must pay the redemption amount *plus* accrued interest, so the
exercise obstacle is raised by that accrual in both the
[lattice](lattice.md) and [PDE](pde.md) engines. Accrual is measured on the
true call date rather than at the grid step it snaps to, which is what lets a
tree and a PDE with different time discretisations agree.

Because accrual is zero when exercise and coupon dates coincide, this makes no
difference to the common textbook case — which is precisely why the
discrepancy went unnoticed.

## Floating-rate accrual

For an FRN, `fixing_dates` say when the rate is *observed* and
`accrual_start_dates` say what it is *applied to*. The two coincide for a plain
fixing-in-advance note and diverge under a fixing lag, an in-arrears
convention, or a stub. One convention applies across every FRN engine: the
coupon accrues over `[accrual_start, payment_date)` and the projected rate is
the forward over that same period.

If `accrual_start_dates` is not supplied, the fixing dates stand in for the
accrual starts. Supply them explicitly when a lag is present, or the accrual
period silently stretches by the length of the lag and over-pays every coupon.

## Boundary, not traced code

`generate_schedule` runs **eagerly, before tracing**. It depends on calendar
structure — month lengths, leap years, stub placement — which is Python-level
configuration rather than differentiable data, and it validates its inputs with
real exceptions:

```python
generate_schedule(2025, 2, 30, 2027, 1, 1)   # ValueError: date does not exist
generate_schedule(2025, 1, 15, 2030, 1, 15, frequency=5)   # ValueError
generate_schedule(2027, 1, 15, 2025, 1, 15)  # ValueError: not strictly before
```

The arrays it returns are ordinary ordinal arrays and are safe to close over
inside `eqx.filter_jit`.
