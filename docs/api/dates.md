# Dates

Date utilities for JIT-compatible date arithmetic. All dates are **integer ordinals** (days since 1970-01-01) stored as `jnp.int32` arrays.

## Date Conversion

### `ymd_to_ordinal`

```python
ymd_to_ordinal(year: int, month: int, day: int) -> Int[Array, ""]
```

Convert a calendar date to an ordinal. Use at the user-facing boundary — inside JIT-traced code, work with ordinals directly.

## Day Count Conventions

All day count functions have the signature:

```python
fn(start: Int[Array, "..."], end: Int[Array, "..."]) -> Float[Array, "..."]
```

They support batched inputs via broadcasting.

### `act_365`

Actual/365 Fixed. Year fraction = actual days / 365.

### `act_360`

Actual/360. Year fraction = actual days / 360. Common for money market instruments.

### `act_act`

Actual/Actual (ISDA simplified). Year fraction = actual days / 365.25.

### `thirty_360`

30/360 Bond Basis (US). Adjusts day counts to assume 30-day months.

### `year_fraction`

```python
year_fraction(start, end, convention="act_365") -> Float[Array, "..."]
```

Dispatch to any convention by name. Valid names: `"act_365"`, `"act_360"`, `"act_act"`, `"30_360"`.

## Schedule Generation

### `generate_schedule`

```python
generate_schedule(
    start_year, start_month, start_day,
    end_year, end_month, end_day,
    frequency=2,
) -> Int[Array, "n_dates"]
```

Generate coupon payment dates backward from maturity. The start date is excluded; the end date (maturity) is always included.

**Arguments**:

| Parameter | Description |
|---|---|
| `start_*` | Issue/settlement date (excluded from schedule) |
| `end_*` | Maturity date (included in schedule) |
| `frequency` | Payments per year: 1 (annual), 2 (semi-annual), 4 (quarterly) |
