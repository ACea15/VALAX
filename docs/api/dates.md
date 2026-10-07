# Dates

Date utilities for JIT-compatible date arithmetic. All dates are
**integer ordinals** (days since 1970-01-01) stored as `jnp.int32`
arrays. Python `datetime` is intentionally avoided inside traced code
— convert at the user-facing boundary with
[`ymd_to_ordinal`][valax.dates.daycounts.ymd_to_ordinal] and work with
ordinals thereafter.

## Date conversion

::: valax.dates.daycounts.ymd_to_ordinal

## Day count conventions

The elapsed-day conventions share the signature
`(start, end) -> year_fraction` and support batched inputs via
broadcasting. `"bus_252"` counts **business** days instead, so it takes a
calendar as a third argument and lives in a separate registry,
`CALENDAR_DAY_COUNT_FNS`; pass the calendar through
[`year_fraction`][valax.dates.daycounts.year_fraction]'s `calendar`
parameter, which raises rather than defaulting to a calendar the caller
never chose. Valid convention names are `"act_365"`, `"act_360"`,
`"act_act"`, `"30_360"` and `"bus_252"`.

::: valax.dates.daycounts.year_fraction

::: valax.dates.daycounts.act_365

::: valax.dates.daycounts.act_360

::: valax.dates.daycounts.act_act

::: valax.dates.daycounts.thirty_360

::: valax.dates.daycounts.bus_252

::: valax.dates.daycounts.ordinal_from_year_fraction

## Business-day calendars and adjustment

A calendar holds its market's weekday closures as a sorted array of integer
ordinals; weekends are arithmetic rather than data. Lookups are
`searchsorted`, so adjustment vectorises and
[`bus_252`][valax.dates.daycounts.bus_252] is usable inside traced code.
Every calendar declares the year range its data covers and **raises**
outside it — an uncovered year is unknown, not holiday-free. See the
[Dates and Conventions guide](../guide/dates-and-conventions.md#business-day-calendars-and-adjustment).

::: valax.dates.calendars.HolidayCalendar

::: valax.dates.calendars.adjust

::: valax.dates.calendars.market_calendar

::: valax.dates.calendars.target_calendar

::: valax.dates.calendars.sofr_calendar

::: valax.dates.calendars.nyse_calendar

::: valax.dates.calendars.london_calendar

::: valax.dates.calendars.tokyo_calendar

::: valax.dates.calendars.brazil_calendar

::: valax.dates.calendars.weekday

::: valax.dates.calendars.is_weekend

## Schedule conventions

::: valax.dates.conventions.ScheduleConventions

## Schedule generation

::: valax.dates.schedule.Schedule

::: valax.dates.schedule.generate_schedule

## Accrual and clean/dirty prices

::: valax.dates.accrual.accrual_periods

::: valax.dates.accrual.floating_accrual_periods

::: valax.dates.accrual.fixed_coupon_amounts

::: valax.dates.accrual.accrued_interest

::: valax.dates.accrual.accrued_interest_at_dates

::: valax.dates.accrual.accrued_interest_by_time

::: valax.dates.accrual.bond_accrued_interest

::: valax.dates.accrual.bond_accrued_interest_at_dates

::: valax.dates.accrual.clean_from_dirty

::: valax.dates.accrual.dirty_from_clean
