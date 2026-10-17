"""The coverage matrix's ``price()`` routes column matches the registry.

``docs/applications/instrument-pricing.md`` states, per instrument and model,
which methods ``price()`` reaches and which one ``method="auto"`` picks. This
test parses that column and compares it, in both directions, with what the
library registers (``registered_pricers()``) and with the auto policy — so a
new route, a removed one or a changed ``auto`` fails here until the matrix
says so.

Cell syntax: ``Label: **auto**, other, ...`` per model, models separated by
``<br>``; ``—`` = not routed (priced by a direct call); ``†`` = registered but
not yet priceable through ``price()`` (:data:`KNOWN_UNPRICEABLE`);
parenthesised text is commentary.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from valax.core import pricing as P
from valax.instruments.bonds import (
    CallableBond,
    FixedRateBond,
    FloatingRateBond,
    PuttableBond,
)
from valax.instruments.fx import FXForward, FXVanillaOption
from valax.instruments.inflation import (
    InflationCapFloor,
    YearOnYearInflationSwap,
    ZeroCouponInflationSwap,
)
from valax.instruments.legs import MultiLegSwap
from valax.instruments.options import (
    AmericanOption,
    AsianOption,
    DigitalOption,
    EquityBarrierOption,
    EuropeanOption,
    LookbackOption,
    SpreadOption,
    VarianceSwap,
    WorstOfBasketOption,
)
from valax.instruments.rates import (
    BermudanSwaption,
    CMSSpreadSwap,
    Cap,
    Caplet,
    CrossCurrencySwap,
    InterestRateSwap,
    OISSwap,
    Swaption,
)
from valax.models import (
    BlackScholesModel,
    CheyetteModel,
    CheyetteModel2F,
    G2PPModel,
    HestonModel,
    HullWhiteModel,
    ImpliedVolModel,
    LMMModel,
    LocalVolModel,
    MultiAssetGBMModel,
    SLVModel,
)

MATRIX = (
    Path(__file__).resolve().parents[2]
    / "docs" / "applications" / "instrument-pricing.md"
)
COLUMN = "`price()` routes"

#: Model labels of the column (the legend above the matrix).
LABELS: dict[str, type] = {
    "curves": type(None),
    "BS": BlackScholesModel,
    "Heston": HestonModel,
    "LV": LocalVolModel,
    "SLV": SLVModel,
    "MA-GBM": MultiAssetGBMModel,
    "IV": ImpliedVolModel,
    "HW": HullWhiteModel,
    "G2++": G2PPModel,
    "Ch-1F": CheyetteModel,
    "Ch-2F": CheyetteModel2F,
    "LMM": LMMModel,
}

#: Matrix row anchor -> instrument class, for every routed instrument.
ROWS: dict[str, type] = {
    "european-option": EuropeanOption,
    "american-option": AmericanOption,
    "equity-barrier-option": EquityBarrierOption,
    "asian-option": AsianOption,
    "lookback-option": LookbackOption,
    "variance-swap": VarianceSwap,
    "digital-option": DigitalOption,
    "spread-option": SpreadOption,
    "worst-of-basket-option": WorstOfBasketOption,
    "caplet-floorlet": Caplet,
    "cap-floor": Cap,
    "interest-rate-swap": InterestRateSwap,
    "swaption": Swaption,
    "cms-spread-swap": CMSSpreadSwap,
    "ois-swap": OISSwap,
    "amortising-swap": MultiLegSwap,
    "seasoned-part-accrued-swap": MultiLegSwap,
    "tenor-basis-swap": MultiLegSwap,
    "cross-currency-swap": CrossCurrencySwap,
    "zero-coupon-inflation-swap-zcis": ZeroCouponInflationSwap,
    "year-on-year-inflation-swap-yyis": YearOnYearInflationSwap,
    "inflation-cap-floor": InflationCapFloor,
    "fx-forward": FXForward,
    "fx-vanilla-option": FXVanillaOption,
    "bermudan-swaption": BermudanSwaption,
    "fixed-rate-bond": FixedRateBond,
    "floating-rate-note": FloatingRateBond,
    "callable-bond": CallableBond,
    "puttable-bond": PuttableBond,
}

#: Registered (instrument, model, method) routes ``price()`` cannot run yet,
#: named so the matrix marks them (†) instead of advertising them as working.
#: Empty: the multi-asset recipes left it with the keyed spots of D16 and the
#: four LMM routes with the LMM adapter of D18.
KNOWN_UNPRICEABLE: set[tuple[type, type, str]] = set()


def _library(cls: type) -> bool:
    return cls is type(None) or cls.__module__.startswith("valax.")


def _registry() -> dict[tuple[type, type], tuple[str, ...]]:
    """``(instrument, model) -> available methods`` of the library itself."""
    P._ensure_backends()
    pairs = {(i, m) for (i, m, _) in P._REGISTRY}
    for method in ("pde", "mc"):
        pairs |= set(P._legacy_registry(method))
    return {(i, m): P.available_methods(i, m) for (i, m) in pairs
            if _library(i) and _library(m)}


def _matrix_rows() -> list[tuple[str, str]]:
    """``(anchor, routes cell)`` of every row of every coverage-matrix table."""
    text = MATRIX.read_text()
    section = text.split("## Coverage matrix", 1)[1].split("\n---\n", 1)[0]
    rows, col = [], None
    for line in section.splitlines():
        if not line.startswith("|"):
            col = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "Instrument":
            assert COLUMN in cells, f"matrix table without a {COLUMN} column: {line}"
            col, width = cells.index(COLUMN), len(cells)
            continue
        if set(cells[0]) <= {"-"} or col is None:
            continue
        assert len(cells) == width, f"row has {len(cells)} cells, header {width}: {line}"
        anchor = re.search(r"\]\(#([^)]+)\)", cells[0])
        assert anchor, f"row without an anchor link: {line}"
        rows.append((anchor.group(1), cells[col]))
    return rows


def _parse(cell: str) -> dict[type, tuple[str, tuple[str, ...], set[str]]]:
    """``model -> (auto, methods, unpriceable methods)`` of one cell."""
    if cell.startswith("—"):
        return {}
    out = {}
    for entry in re.sub(r"\([^)]*\)", "", cell).split("<br>"):
        label, _, methods = entry.partition(":")
        label = label.strip()
        assert label in LABELS, f"unknown model label {label!r} in {cell!r}"
        names, auto, dagger = [], None, set()
        for raw in methods.split(","):
            raw = raw.strip()
            name = raw.replace("*", "").replace("†", "").strip()
            if raw.startswith("**"):
                assert auto is None, f"two auto methods in {entry!r}"
                auto = name
            if "†" in raw:
                dagger.add(name)
            names.append(name)
        assert auto == names[0], f"auto method must come first, in bold: {entry!r}"
        out[LABELS[label]] = (auto, tuple(names), dagger)
    return out


@pytest.fixture(scope="module")
def documented():
    rows = _matrix_rows()
    anchors = [a for a, _ in rows]
    assert len(anchors) == len(set(anchors)), "duplicate matrix rows"
    return {a: _parse(cell) for a, cell in rows}


def test_unrouted_rows_say_so(documented):
    stray = {a: r for a, r in documented.items() if a not in ROWS and r}
    assert not stray, f"routes documented for an unmapped row: {stray}"
    for anchor in ROWS:
        assert anchor in documented, f"routed instrument row missing: {anchor}"


def test_documented_routes_are_registered(documented):
    registry = _registry()
    for anchor, cls in ROWS.items():
        for model, (auto, methods, _) in documented[anchor].items():
            key = f"{cls.__name__} x {model.__name__}"
            assert (cls, model) in registry, f"documented but not registered: {key}"
            assert set(methods) == set(registry[(cls, model)]), (
                key, methods, registry[(cls, model)])
            assert auto == P.auto_method(cls, model), (key, auto)


def test_every_registered_route_is_documented(documented):
    by_class = {cls: anchor for anchor, cls in ROWS.items()}
    for (cls, model), methods in _registry().items():
        key = f"{cls.__name__} x {model.__name__}"
        assert cls in by_class, f"registered but no matrix row mapped: {key}"
        assert model in documented[by_class[cls]], f"registered, undocumented: {key}"


def test_unpriceable_routes_are_marked(documented):
    marked = {
        (ROWS[a], model, m)
        for a in ROWS for model, (_, _, dagger) in documented[a].items()
        for m in dagger
    }
    assert marked == KNOWN_UNPRICEABLE


def test_public_listing_agrees():
    """``registered_pricers()`` (by name) covers the same library routes."""
    names = set(P.registered_pricers())
    for (cls, model), methods in _registry().items():
        for m in methods:
            assert (cls.__name__, model.__name__, m) in names
