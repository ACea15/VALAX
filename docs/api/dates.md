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

All day-count functions share the signature
`(start, end) -> year_fraction` and support batched inputs via
broadcasting. Valid convention names for
[`year_fraction`][valax.dates.daycounts.year_fraction] are
`"act_365"`, `"act_360"`, `"act_act"`, and `"30_360"`.

::: valax.dates.daycounts.year_fraction

::: valax.dates.daycounts.act_365

::: valax.dates.daycounts.act_360

::: valax.dates.daycounts.act_act

::: valax.dates.daycounts.thirty_360

::: valax.dates.daycounts.ordinal_from_year_fraction

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
