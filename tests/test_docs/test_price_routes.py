"""The coverage matrix matches the code.

``docs/applications/instrument-pricing.md`` is the single instrument × solver
coverage map. This test checks, in both directions:

* the ``price()`` routes column — per instrument and model, which methods
  ``price()`` reaches and which one ``method="auto"`` picks — against what the
  library registers (``registered_pricers()``) and the auto policy;
* the four solver columns (*Analytic*, *MC*, *PDE*, *Lattice*): ✅ iff a
  method of that solver is registered for the instrument, ◇ iff a kernel is
  named in :data:`KERNEL_ONLY` (and imports), 🟡 links to a roadmap section
  with an open ``- [ ]`` item;
* each row's *Status* against its solver cells.

So a new route, a removed one, a changed ``auto``, a new kernel or a shipped
roadmap item fails here until the matrix says so.

Routes-cell syntax: ``Label: **auto**, other, ...`` per model, models
separated by ``<br>``; ``—`` = not routed (priced by a direct call); ``†`` =
registered but not yet priceable through ``price()``
(:data:`KNOWN_UNPRICEABLE`); parenthesised text is commentary. The solver-cell
grammar is in ``coverage_matrix.py``.
"""

from __future__ import annotations

import functools
import importlib
import re

import pytest

from coverage_matrix import (
    KERNEL,
    METHOD_OF,
    NONE,
    PLANNED,
    ROUTED,
    SOLVERS,
    roadmap_sections,
    rows as matrix_rows,
)
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

#: ``(row anchor, solver column) -> kernels`` behind every ◇ cell: tested
#: pricers called directly, not (yet) routed through ``price()``.
KERNEL_ONLY: dict[tuple[str, str], tuple[str, ...]] = {
    ("variance-swap", "Analytic"): (
        "valax.pricing.analytic.variance_swap.variance_swap_price",),
    ("spread-option", "Analytic"): (
        "valax.pricing.analytic.spread.margrabe_price",
        "valax.pricing.analytic.spread.kirk_price"),
    ("cms-swap", "Analytic"): (
        "valax.pricing.analytic.rates_exotics.cms_swap_price",),
    ("cms-cap-floor", "Analytic"): (
        "valax.pricing.analytic.rates_exotics.cms_cap_floor_price_black76",),
    ("range-accrual", "Analytic"): (
        "valax.pricing.analytic.rates_exotics.range_accrual_price_black76",),
    ("total-return-swap", "Analytic"): (
        "valax.pricing.analytic.rates_exotics.total_return_swap_price",),
    ("zero-coupon-bond", "Analytic"): (
        "valax.pricing.analytic.bonds.zero_coupon_bond_price",),
}


def _library(cls: type) -> bool:
    return cls is type(None) or cls.__module__.startswith("valax.")


@functools.cache
def _registry() -> dict[tuple[type, type], tuple[str, ...]]:
    """``(instrument, model) -> available methods`` of the library itself."""
    P._ensure_backends()
    pairs = {(i, m) for (i, m, _) in P._REGISTRY}
    for method in ("pde", "mc"):
        pairs |= set(P._legacy_registry(method))
    return {(i, m): P.available_methods(i, m) for (i, m) in pairs
            if _library(i) and _library(m)}


def _registered_methods(cls: type | None) -> set[str]:
    """Every ``price()`` method registered for ``cls``, over all models."""
    if cls is None:
        return set()
    return {m for (i, _), ms in _registry().items() if i is cls for m in ms}


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
def rows():
    rows = matrix_rows()
    anchors = [r.anchor for r in rows]
    assert len(anchors) == len(set(anchors)), "duplicate matrix rows"
    return rows


@pytest.fixture(scope="module")
def documented(rows):
    return {r.anchor: _parse(r.routes) for r in rows}


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


def test_every_registered_method_has_a_solver_column():
    methods = {m for ms in _registry().values() for m in ms}
    assert methods <= set(METHOD_OF.values()), methods - set(METHOD_OF.values())


def test_routed_cells_match_the_registry(rows):
    """✅ in a solver column iff that solver is registered for the instrument."""
    for row in rows:
        registered = _registered_methods(ROWS.get(row.anchor))
        for solver, cell in row.solvers.items():
            routed = METHOD_OF[solver] in registered
            assert (cell.kind == ROUTED) == routed, (
                f"{row.anchor} / {solver}: cell is {cell.kind}, "
                f"registered methods {sorted(registered)}")


def test_kernel_cells_match_kernel_only(rows):
    """◇ iff the (row, solver) names a kernel in :data:`KERNEL_ONLY`."""
    marked = {(r.anchor, s) for r in rows
              for s, c in r.solvers.items() if c.kind == KERNEL}
    assert marked == set(KERNEL_ONLY), (
        f"◇ without a KERNEL_ONLY entry: {marked - set(KERNEL_ONLY)}; "
        f"KERNEL_ONLY entry not marked ◇: {set(KERNEL_ONLY) - marked}")


@pytest.mark.parametrize("path", sorted(
    p for paths in KERNEL_ONLY.values() for p in paths))
def test_kernel_only_symbols_import(path):
    module, _, name = path.rpartition(".")
    assert callable(getattr(importlib.import_module(module), name))


def test_planned_cells_point_at_open_roadmap_items(rows):
    """Every 🟡 links to a roadmap section that still has an open ``- [ ]``."""
    sections = roadmap_sections()
    for row in rows:
        for solver, cell in row.solvers.items():
            if cell.roadmap_anchor is None:
                continue
            where = f"{row.anchor} / {solver} -> #{cell.roadmap_anchor}"
            assert cell.roadmap_anchor in sections, f"no such roadmap section: {where}"
            assert re.search(r"^\s*- \[ \]", sections[cell.roadmap_anchor], re.M), (
                f"roadmap section has no open item: {where}")


def test_status_agrees_with_solver_cells(rows):
    """✅ status iff something prices it; 🟡 status names the planned solver."""
    for row in rows:
        kinds = {c.kind for c in row.solvers.values()}
        priced = bool(kinds & {ROUTED, KERNEL})
        assert row.status in {"✅", "🟡"}, f"{row.anchor}: status {row.status!r}"
        assert (row.status == "✅") == priced, (row.anchor, row.status, kinds)
        if row.status == "🟡":
            assert PLANNED in kinds, f"{row.anchor}: 🟡 with no planned solver"
        assert kinds <= {ROUTED, KERNEL, PLANNED, NONE}


def test_solver_columns_are_the_four_families():
    assert SOLVERS == tuple(METHOD_OF)


def test_public_listing_agrees():
    """``registered_pricers()`` (by name) covers the same library routes."""
    names = set(P.registered_pricers())
    for (cls, model), methods in _registry().items():
        for m in methods:
            assert (cls.__name__, model.__name__, m) in names
