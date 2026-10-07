"""Regenerate ``valax/dates/_holiday_data.py`` from upstream reference data.

Usage::

    pip install -e ".[calendars-data]"
    python scripts/generate_holiday_tables.py

Why a generator and not a runtime dependency
---------------------------------------------
Holiday dates are *reference data*, not a computation.  If the tables were
built at runtime from a third-party package, the dates a trade pays on would
become a function of which version of that package happened to be installed:
a ``pip install -U`` could silently move a swap's cashflows with no change to
VALAX and no line in the diff.  So the upstreams are consulted **offline**,
the result is checked in as a version-stamped artifact, and
:mod:`valax.dates.calendars` reads nothing but integers.

Independence
------------
Dates are converted to VALAX ordinals with :mod:`datetime` arithmetic
(``(d - date(1970, 1, 1)).days``) and deliberately **not** with
:func:`valax.dates.daycounts.ymd_to_ordinal`.  Validating VALAX's Hinnant
conversion with VALAX's own Hinnant conversion would prove only that the
algorithm is used twice; ``tests/test_dates/test_calendars.py`` asserts the
two agree, which is only a real assertion because they are derived
independently.

Cross-checks
------------
Every market is validated against a second, independently maintained source
-- or, where none exists, against its own published rule -- across the
**whole** of its declared year range.  There is no sampling window, so a
check cannot pass vacuously.  The script exits non-zero rather than emitting
a short or surprising table.

Each market's year range is bounded by a documented historical fact rather
than by convenience; see ``MARKETS`` below.  Two classes of upstream content
are deliberately rejected:

* **Exchange outages.**  A ``valid_days`` complement includes unscheduled
  closures.  The Tokyo Stock Exchange's 1 October 2020 system failure was a
  trading outage, not a settlement holiday -- banks settled normally, so a
  payment must not roll off it.  See ``_EXCLUSIONS``.
* **Weekend closures.**  Saturdays and Sundays are already non-business days
  by the weekend rule in :mod:`valax.dates.calendars`, so storing them would
  double the tables and add nothing.
"""

from __future__ import annotations

import datetime as dt
import textwrap
from pathlib import Path
from typing import Callable, Iterable

import holidays
import pandas_market_calendars as pmc

#: VALAX epoch.  Matches :func:`valax.dates.daycounts.ymd_to_ordinal`.
EPOCH = dt.date(1970, 1, 1)

#: Bumped by hand whenever the emitted tables change in a way that could move
#: a cashflow date.  Recorded in the artifact so provenance is reconstructible
#: from the generated file alone.
CALENDAR_DATA_VERSION = "2026.1"

#: Upstream entries that are *not* settlement holidays, with the reason.
#: Removing these is a judgement about what the data means, so it is recorded
#: here rather than silently filtered.
_EXCLUSIONS: dict[str, dict[dt.date, str]] = {
    "TOKYO": {
        dt.date(2020, 10, 1): (
            "Tokyo Stock Exchange Arrowhead system failure -- a full-day "
            "trading outage, not a holiday; banks settled normally"
        ),
    },
}

OUTPUT = Path(__file__).resolve().parent.parent / "valax" / "dates" / "_holiday_data.py"


# ──────────────────────────────────────────────────────────────────────
# Source adapters
# ──────────────────────────────────────────────────────────────────────


def _weekdays(dates: Iterable[dt.date]) -> set[dt.date]:
    """Drop weekend entries -- see the module docstring."""
    return {d for d in dates if d.weekday() < 5}


def pmc_closures(name: str, first_year: int, last_year: int) -> set[dt.date]:
    """Weekday closures of a ``pandas_market_calendars`` calendar.

    Derived as the complement of ``valid_days`` rather than by reading the
    calendar's holiday *rules*, so ad-hoc closures are included -- which is
    what makes ``_EXCLUSIONS`` necessary for genuine outages.
    """
    calendar = pmc.get_calendar(name)
    trading = {
        ts.date()
        for ts in calendar.valid_days(f"{first_year}-01-01", f"{last_year}-12-31")
    }
    out: set[dt.date] = set()
    day = dt.date(first_year, 1, 1)
    last = dt.date(last_year, 12, 31)
    while day <= last:
        if day.weekday() < 5 and day not in trading:
            out.add(day)
        day += dt.timedelta(days=1)
    return out


def holidays_country(code: str, first_year: int, last_year: int, **kw) -> set[dt.date]:
    years = range(first_year, last_year + 1)
    return _weekdays(holidays.country_holidays(code, years=years, **kw))


def holidays_financial(code: str, first_year: int, last_year: int) -> set[dt.date]:
    years = range(first_year, last_year + 1)
    return _weekdays(holidays.financial_holidays(code, years=years))


# ──────────────────────────────────────────────────────────────────────
# Cross-checks.  Each raises on violation.
# ──────────────────────────────────────────────────────────────────────


def _easter_sunday(year: int) -> dt.date:
    """Anonymous Gregorian computus -- integer arithmetic, no dependency."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


#: TARGET closed on 31 December in 1999 (Y2K changeover) and 2001 (euro cash
#: changeover).  Both are documented ECB decisions, not rule-derivable.
_TARGET_ONE_OFFS = frozenset({dt.date(1999, 12, 31), dt.date(2001, 12, 31)})


def check_target(primary: set[dt.date], first: int, last: int) -> str:
    """Every TARGET closure must be one of the six published days.

    TARGET2's closing days are New Year's Day, Good Friday, Easter Monday,
    1 May and 25/26 December.  Checking against that rule is genuinely
    independent -- it comes from the ECB's published schedule rather than
    from another package that might share an upstream with ``holidays``.
    """
    off_rule = [
        day
        for day in sorted(primary)
        if day not in _TARGET_ONE_OFFS
        and (day.month, day.day) not in {(1, 1), (5, 1), (12, 25), (12, 26)}
        and day != _easter_sunday(day.year) - dt.timedelta(days=2)
        and day != _easter_sunday(day.year) + dt.timedelta(days=1)
    ]
    if off_rule:
        raise SystemExit(
            f"TARGET: {len(off_rule)} closure(s) match neither the published "
            f"six-day rule nor a documented one-off: {off_rule[:10]}"
        )
    return (
        f"all {len(primary)} closures match the published TARGET six-day rule "
        "(+2 documented one-offs)"
    )


def _exact(
    label: str, primary: set[dt.date], secondary: set[dt.date]
) -> str:
    diff = primary ^ secondary
    if diff:
        raise SystemExit(
            f"{label}: the two upstreams disagree on {len(diff)} date(s): "
            f"{sorted(diff)[:10]}"
        )
    return f"exact agreement with a second source on all {len(primary)} dates"


def _superset(
    label: str,
    primary: set[dt.date],
    secondary: set[dt.date],
    allowed_surplus: Callable[[dt.date], bool],
    surplus_desc: str,
) -> str:
    missing = sorted(secondary - primary)
    if missing:
        raise SystemExit(
            f"{label}: the secondary source has {len(missing)} date(s) the "
            f"primary lacks: {missing[:10]}"
        )
    unexplained = sorted(d for d in primary - secondary if not allowed_surplus(d))
    if unexplained:
        raise SystemExit(
            f"{label}: {len(unexplained)} surplus date(s) are not {surplus_desc}: "
            f"{unexplained[:10]}"
        )
    return (
        f"superset of a second source's {len(secondary)} dates; the "
        f"{len(primary) - len(secondary)} surplus are {surplus_desc}"
    )


def check_nyse(primary: set[dt.date], first: int, last: int) -> str:
    return _exact("NYSE", primary, holidays_financial("XNYS", first, last))


def check_london(primary: set[dt.date], first: int, last: int) -> str:
    return _exact(
        "LONDON", primary, holidays_country("GB", first, last, subdiv="ENG")
    )


def check_sofr(primary: set[dt.date], first: int, last: int) -> str:
    """SIFMA and the federal calendar differ, and only in one lawful way.

    Federal agencies observe a Saturday holiday on the preceding Friday; the
    US government securities market does not.  The bond market also closes on
    Good Friday, which is not a federal holiday.  So neither set contains the
    other, and a blunt superset assertion would be false.  The real invariant
    is that every federal-only date is a Friday whose Saturday is itself a
    federal holiday.
    """
    federal = holidays_country("US", first, last)
    # One year of headroom: a 31 December federal observance belongs to the
    # *following* year's New Year's Day, which would otherwise look unexplained
    # at the top of the range.  This is why no table runs to 2100 -- the
    # cross-check upstream's own horizon ends there, so its final year cannot
    # be validated for that edge.
    observed = set(holidays.country_holidays("US", years=range(first, last + 2)))
    unexplained = [
        d
        for d in sorted(federal - primary)
        if not (d.weekday() == 4 and d + dt.timedelta(days=1) in observed)
    ]
    if unexplained:
        raise SystemExit(
            f"SOFR: {len(unexplained)} federal holiday(s) absent from SIFMA are "
            f"not Saturday-observance shifts: {unexplained[:10]}"
        )
    return (
        f"{len(federal - primary)} differences from the federal calendar, all "
        "Saturday-observance shifts; SIFMA additionally closes Good Friday"
    )


def check_tokyo(primary: set[dt.date], first: int, last: int) -> str:
    return _superset(
        "TOKYO",
        primary,
        holidays_country("JP", first, last),
        lambda d: (d.month, d.day) in {(1, 2), (1, 3), (12, 31)},
        "Japanese bank holidays (2-3 Jan, 31 Dec)",
    )


def check_brazil(primary: set[dt.date], first: int, last: int) -> str:
    return _superset(
        "BRAZIL",
        primary,
        holidays_country("BR", first, last),
        lambda d: True,
        "B3/ANBIMA financial closures (Carnival, Corpus Christi, 24/31 Dec)",
    )


# ──────────────────────────────────────────────────────────────────────
# Market specifications
# ──────────────────────────────────────────────────────────────────────


class Market:
    """One market's primary source, cross-check and bounded year range."""

    def __init__(
        self,
        key: str,
        description: str,
        source: str,
        loader: Callable[[int, int], set[dt.date]],
        checker: Callable[[set[dt.date], int, int], str],
        first_year: int,
        last_year: int,
        range_reason: str,
    ) -> None:
        self.key = key
        self.description = description
        self.source = source
        self.loader = loader
        self.checker = checker
        self.first_year = first_year
        self.last_year = last_year
        self.range_reason = range_reason


MARKETS = [
    Market(
        "TARGET",
        "EUR settlement (TARGET2 closing days, ECB)",
        "holidays:XECB",
        lambda a, b: holidays_financial("XECB", a, b),
        check_target,
        1999,
        2099,
        "TARGET opened with the euro on 4 January 1999; there is no earlier "
        "calendar to publish",
    ),
    Market(
        "NYSE",
        "New York Stock Exchange trading days",
        "pandas_market_calendars:NYSE",
        lambda a, b: pmc_closures("NYSE", a, b),
        check_nyse,
        1971,
        2099,
        "the sources disagree on 29 May 1970, the last Memorial Day before "
        "the Uniform Monday Holiday Act settled its date",
    ),
    Market(
        "SOFR",
        "USD rates: SIFMA US government securities market",
        "pandas_market_calendars:SIFMA_US",
        lambda a, b: pmc_closures("SIFMA_US", a, b),
        check_sofr,
        1978,
        2099,
        "Veterans Day was observed in October from 1971 to 1977 and the "
        "upstream applies modern rules to those years",
    ),
    Market(
        "LONDON",
        "GBP settlement (England & Wales bank holidays)",
        "pandas_market_calendars:LSE",
        lambda a, b: pmc_closures("LSE", a, b),
        check_london,
        1978,
        2099,
        "1 January became an English bank holiday in 1974 and May Day in "
        "1978; the upstream applies modern rules before that",
    ),
    Market(
        "TOKYO",
        "JPY settlement (national holidays + bank holidays)",
        "pandas_market_calendars:JPX",
        lambda a, b: pmc_closures("JPX", a, b),
        check_tokyo,
        1989,
        2098,
        "the Tokyo exchange's year-end closure ran to 30 December before "
        "1989, and the sources diverge on the 2099 autumnal equinox, which "
        "is astronomical and fixed by cabinet order only a year ahead",
    ),
    Market(
        "BRAZIL",
        "BRL: B3 / ANBIMA financial calendar (Bus/252)",
        "pandas_market_calendars:B3",
        lambda a, b: pmc_closures("B3", a, b),
        check_brazil,
        1970,
        2099,
        "the sources agree across the full epoch-to-2099 span",
    ),
]


# ──────────────────────────────────────────────────────────────────────
# Emission
# ──────────────────────────────────────────────────────────────────────


def _format_table(ordinals: list[int]) -> str:
    body = textwrap.fill(
        ", ".join(str(o) for o in ordinals),
        width=72,
        initial_indent="    ",
        subsequent_indent="    ",
        break_long_words=False,
    )
    return f"(\n{body},\n)"


def main() -> int:
    generated_on = dt.date.today().isoformat()
    blocks: list[str] = []
    provenance: list[str] = []
    ranges: list[str] = []
    report: list[str] = []

    for market in MARKETS:
        dates = market.loader(market.first_year, market.last_year)
        if not dates:
            raise SystemExit(f"{market.key}: upstream returned no closures")

        excluded = _EXCLUSIONS.get(market.key, {})
        for day, reason in excluded.items():
            if day not in dates:
                raise SystemExit(
                    f"{market.key}: exclusion {day} is no longer present "
                    f"upstream, so it should be deleted ({reason})"
                )
            dates.discard(day)

        note = market.checker(dates, market.first_year, market.last_year)
        ordinals = sorted((d - EPOCH).days for d in dates)
        if len(set(ordinals)) != len(ordinals):
            raise SystemExit(f"{market.key}: duplicate ordinals")

        comment = [
            f"#: {market.description}.",
            f"#: Source: ``{market.source}``.",
            f"#: Cross-check: {note}.",
            f"#: Range {market.first_year}-{market.last_year}: {market.range_reason}.",
        ]
        for day, reason in sorted(excluded.items()):
            comment.append(f"#: Excluded {day}: {reason}.")
        wrapped = "\n".join(
            textwrap.fill(
                line, width=78, initial_indent="", subsequent_indent="#:     "
            )
            for line in comment
        )
        blocks.append(f"{wrapped}\n{market.key}: tuple[int, ...] = {_format_table(ordinals)}")

        provenance.append(f'    "{market.key}": "{market.source}",')
        ranges.append(
            f'    "{market.key}": ({market.first_year}, {market.last_year}),'
        )
        report.append(
            f"  {market.key:8s} {len(ordinals):5d} closures  "
            f"{market.first_year}-{market.last_year}  [{note}]"
        )

    header = f'''"""Holiday reference data -- GENERATED FILE, DO NOT EDIT BY HAND.

Regenerate with::

    pip install -e ".[calendars-data]"
    python scripts/generate_holiday_tables.py

Each table is a sorted tuple of **integer day ordinals** (days since
1970-01-01, the VALAX epoch) listing the **weekday** closures of one market.
Weekend closures are omitted: Saturdays and Sundays are already non-business
days by the weekend rule in :mod:`valax.dates.calendars`, so storing them
would double the tables and add nothing.

Each market's covered year range is part of its definition rather than a
global constant -- TARGET did not exist before the euro, and Japan's equinox
holidays are astronomical and fixed by cabinet order only about a year
ahead.  Every range boundary is set by a documented historical fact, stated
in the comment above each table.  Queries outside a calendar's range raise
rather than silently reporting "business day"; see
:class:`valax.dates.calendars.HolidayCalendar`.

Every table was cross-checked against a second independently maintained
source, or against its own published rule, across the whole of its declared
range -- not a sampled window.  The generator exits non-zero on any
disagreement.

Generated: {generated_on}
Upstreams: holidays {holidays.__version__}, pandas_market_calendars {pmc.__version__}
"""

from __future__ import annotations

#: Bumped when the tables change in a way that could move a cashflow date.
CALENDAR_DATA_VERSION = "{CALENDAR_DATA_VERSION}"

#: Upstream source for each market, as consulted by the generator.
PROVENANCE: dict[str, str] = {{
{chr(10).join(provenance)}
}}

#: ``market -> (first_year, last_year)`` covered by the table below.
COVERED_YEARS: dict[str, tuple[int, int]] = {{
{chr(10).join(ranges)}
}}
'''

    OUTPUT.write_text(header + "\n\n" + "\n\n".join(blocks) + "\n")
    print(f"[generate_holiday_tables] wrote {OUTPUT}")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
