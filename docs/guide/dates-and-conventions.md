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

`payment_dates` equals `accrual_end` — both of them adjusted, once a
business-day convention is configured. The array is separate because a
payment **lag** moves a payment without moving the accrual it pays for;
business-day adjustment moves the two together. `generate_schedule` takes no
lag, so the separation is exercised by the instruments
(`SwapRate.payment_lag`, and [Tier 1.4](../roadmap.md#14-trade-representation-and-cashflow-engine)
for legs), not here.

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

## Conventions: stubs, rolls and calendars

`ScheduleConventions` is a frozen pytree. Every field is static except
`calendar`, whose holiday ordinals are a JAX array and therefore a leaf:

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

## Business-day calendars and adjustment

A period end that lands on a Sunday is not paid on the Sunday. Which working
day it moves to is a convention, and it changes the cashflow: a different
payment date is a different discount factor, and a different accrual end is a
different coupon.

### Calendars

A `HolidayCalendar` is a pytree of **integer day ordinals** — the weekday
closures of one market. Weekends are not stored; they are arithmetic. Six
markets ship with the library, each cached:

| Factory | Market | Covers |
|---|---|---|
| `target_calendar()` | EUR settlement (TARGET2) | 1999–2099 |
| `sofr_calendar()` | USD rates (SIFMA government securities) | 1978–2099 |
| `nyse_calendar()` | NYSE trading days | 1971–2099 |
| `london_calendar()` | GBP settlement (England & Wales) | 1978–2099 |
| `tokyo_calendar()` | JPY settlement (national + bank holidays) | 1989–2098 |
| `brazil_calendar()` | BRL (B3 / ANBIMA), for Bus/252 | 1970–2099 |

`nyse_calendar()` and `sofr_calendar()` are **not** interchangeable, and
neither contains the other: the exchange closes on Good Friday, while the
bond market closes on Columbus Day and Veterans Day. Use `sofr_calendar()`
for USD rates.

Each range is bounded by a documented fact rather than by convenience —
TARGET opened with the euro in 1999; Tokyo stops at 2098 because the equinox
holidays are astronomical and fixed by cabinet order only about a year
ahead. **A date outside a calendar's range raises.** An uncovered year is
unknown, not holiday-free.

### Adjusting a schedule

```python
from valax.dates import ScheduleConventions, generate_schedule, target_calendar

conv = ScheduleConventions(
    day_count="act_360",
    business_day_convention="modified_following",
    calendar=target_calendar(),
)

plain = generate_schedule(2025, 1, 31, 2027, 1, 31, frequency=2)
rolled = generate_schedule(2025, 1, 31, 2027, 1, 31, frequency=2, conventions=conv)

print(plain.accrual_end.tolist())    # [20300, 20484, 20665, 20849]
print(rolled.accrual_end.tolist())   # [20300, 20483, 20665, 20847]
```

Two of the four period ends move, because 31 January 2026 is a Saturday and
31 January 2027 a Sunday:

| Period end | Unadjusted | Adjusted |
|---|---|---|
| 1 | 2025-07-31 (Thu) | 2025-07-31 (Thu) |
| 2 | 2026-01-31 (**Sat**) | 2026-01-30 (Fri) |
| 3 | 2026-07-31 (Fri) | 2026-07-31 (Fri) |
| 4 | 2027-01-31 (**Sun**) | 2027-01-29 (Fri) |

Both moved backwards, not forwards, because `modified_following` may not
cross a month boundary — the point of the `modified_` prefix:

```python
from valax.dates import target_calendar, ymd_to_ordinal

cal = target_calendar()
saturday = ymd_to_ordinal(2025, 5, 31)            # 31 May 2025 is a Saturday

print(int(cal.adjust(saturday, "following")))           # 20241 -> 2 June
print(int(cal.adjust(saturday, "modified_following")))  # 20238 -> 30 May
```

Adjustment is applied to the assembled boundary grid, never during the roll,
which is also QuantLib's order: the roll-day, end-of-month and stub
arithmetic define the *unadjusted* grid, and the convention then moves each
boundary onto a working day. Both accrual legs are sliced from one adjusted
list, so `accrual_start[i + 1] == accrual_end[i]` still holds.

With `end_of_month=True` an adjusted month-end date stays month-end: the last
*business* day of its own month.

### The default has not moved

`business_day_convention` defaults to `"unadjusted"`, and that path is the
pre-adjustment code reached by a static branch — the dates it produces are
bit for bit what they were before calendars existed. Adjustment is opt-in.

### The pairing is validated

A calendar and an adjustment rule are only meaningful together, so
`ScheduleConventions` validates them as a pair. Each half alone is a
convention that *looks* configured and is not:

```python
ScheduleConventions(business_day_convention="modified_following")
# ValueError: ... needs a calendar to adjust against, but calendar is None.

ScheduleConventions(calendar=target_calendar())
# ValueError: ... was supplied but nothing would consult it ...
```

These guards predate adjustment, when they raised `NotImplementedError`.
They did not go away when it landed; only their reason changed, from "not
implemented" to "incoherent".

### Bus/252

Bus/252 counts **business** days rather than elapsed ones, so a calendar is
part of the convention and not an optional refinement — two calendars give
two different year fractions for the same dates. It therefore lives in a
second registry, `CALENDAR_DAY_COUNT_FNS`, and requires its calendar
explicitly:

```python
from valax.dates import brazil_calendar, year_fraction, ymd_to_ordinal

brl = brazil_calendar()
start, end = ymd_to_ordinal(2025, 1, 1), ymd_to_ordinal(2026, 1, 1)

print(int(brl.business_days_between(start, end)))               # 250
print(float(year_fraction(start, end, "bus_252", calendar=brl)))  # 0.9920634920634921
print(float(year_fraction(start, end, "act_360")))                # 1.0138888888888888

year_fraction(start, end, "bus_252")
# ValueError: Day count 'bus_252' counts business days and so needs a calendar ...
```

The denominator is the market constant 252, not a count derived from the
calendar, so a full year is only approximately 1.0.

Unlike adjustment, Bus/252 runs **inside** traced code — `year_fraction` is
called on JAX arrays by the accrual layer, once per coupon. The count is a
difference of two gathers, so it is JIT-compatible and its cost grows with
neither the span nor the size of the holiday table.

??? note "Why two tables, and why not a `frozenset`"
    A calendar carries the same information twice, separated by role.

    The **sparse** array of holiday ordinals is the canonical definition. It
    is single-digit kilobytes, it is what `_holiday_data.py` stores, and it
    is what `union`/`intersection` operate on. Keeping the stored form sparse
    is what makes a data change reviewable — one line per changed cashflow
    date in `git diff`.

    The **dense** array is a prefix sum of business days, and it is what
    every query actually reads. A Python `dict` or `frozenset` cannot be
    consulted with a *traced* key at all, and comparing against a baked-in
    set lowers to an `O(n)` compare-against-all — worse than a binary
    search. The JAX-shaped equivalent of a hash table is a dense array
    indexed by `ordinal - first_ordinal`, so that is what is used. One
    prefix sum answers both questions: a difference of *adjacent* entries
    says whether a day is a business day, and a difference of *distant*
    entries counts the business days between two dates.

    The trade is ~185 KB per calendar against ~6 KB, in exchange for a
    measured **27.8×** reduction in marginal cost per cashflow (15.56 ns →
    0.56 ns). On a single 80-coupon swap the end-to-end gain is small,
    because ~24 µs of Python dispatch dominates a lone call; the figure that
    matters is the marginal one, since inside a real pricing trace that
    fixed cost is paid once for the whole function.

    The two tables cannot silently diverge: `TestTheTwoRepresentationsAgree`
    checks the dense form against the sparse one at *every* covered ordinal,
    not a sample.

### Cross-currency: union and intersection

A cross-currency payment needs both centres open, so the joint calendar is
the **union** of their closures. The covered range narrows to the overlap,
because outside it one of the two has no data:

```python
from valax.dates import sofr_calendar, target_calendar

joint = target_calendar().union(sofr_calendar())
print(joint.name, joint.first_year, joint.last_year, int(joint.holidays.size))
# TARGET+SOFR 1999 2099 1392
```

`intersection` gives the complement question — when both centres are shut at
once.

### Where the data comes from

The tables are a **generated, version-stamped artifact**
(`valax/dates/_holiday_data.py`), not a runtime lookup. Holiday dates are
reference data, and deriving them at runtime from a third-party package would
make the dates a trade pays on depend on which version of that package
happened to be installed — a `pip install -U` could move a swap's cashflows
with nothing to show in the diff.

So the upstreams (`holidays`, `pandas_market_calendars`) are consulted
*offline* by `scripts/generate_holiday_tables.py` and are **not runtime
dependencies**. Each table had to pass a cross-check against a second,
independently maintained source — or, for TARGET, against its own published
six-day rule — across the whole of its declared range before being emitted;
the generator exits non-zero on any disagreement. See `CONTRIBUTING.md` at
the repository root for how to regenerate it.

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

That split is deliberate and general. Dates stay **dynamic** `jnp` integer
arrays rather than static Python data, so one compilation serves every
schedule of the same length and a book of trades can be stacked and `vmap`ped
— which static dates would make impossible. The cost is that any computation
needing a date's *value* at trace time (a daily subdivision, say) cannot be
traced, and must instead promote a static *bound*. Making the dates NumPy does
not help: `equinox` treats `np.ndarray` as dynamic too. The rule, the measured
comparison of all four representations, and the three sanctioned ways out are
in [JAX Patterns § 2.5](../architecture/jax-patterns.md#25-when-you-genuinely-need-a-value-at-trace-time-data-dependent-shapes).
