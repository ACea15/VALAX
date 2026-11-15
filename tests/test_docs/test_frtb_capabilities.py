"""The FRTB capability matrix (``docs/applications/regulatory.md`` § 2.7) is true.

The matrix is the one place FRTB component status lives. This test keeps it
honest against the code and the roadmap:

* every status is one of ✅ / 🟡 / 📋;
* every backticked symbol in the *VALAX today* column imports from a public
  ``valax`` package (``Class.method`` resolved attribute by attribute);
* a ✅ row names at least one symbol and an existing test file as its gate;
* a 🟡 row names at least one symbol;
* a 🟡 or 📋 row links to a roadmap section that still has an open item.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from coverage_matrix import roadmap_sections

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "applications" / "regulatory.md"
MARKER = "<!-- coverage-map -->"
STATUSES = {"✅", "🟡", "📋"}
COLUMNS = ("Component", "Approach", "Status", "VALAX today", "Gate", "Gap and roadmap")

#: Public packages a symbol in the matrix may be exported from.
PUBLIC_PACKAGES = ("valax.risk", "valax.greeks", "valax.core", "valax.market")

_ROADMAP_LINK = re.compile(r"\]\(\.\./roadmap\.md#([^)]+)\)")
_SYMBOL = re.compile(r"`([A-Za-z_][\w.]*)`")


def _rows() -> list[dict[str, str]]:
    text = PAGE.read_text()
    section = text.split("{#frtb-capability-matrix}", 1)[1].split("\n## ", 1)[0]
    section = section.split("\n### ", 1)[0]
    assert MARKER in section, "the FRTB matrix lost its coverage-map marker"
    lines = [ln for ln in section.splitlines() if ln.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    assert tuple(header) == COLUMNS, header
    rows = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == len(COLUMNS), line
        rows.append(dict(zip(COLUMNS, cells)))
    return rows


ROWS = _rows()


def _resolve(dotted: str) -> object:
    head, *rest = dotted.split(".")
    for package in PUBLIC_PACKAGES:
        obj = getattr(importlib.import_module(package), head, None)
        if obj is not None:
            break
    else:
        raise AssertionError(f"{head!r} is not exported by {PUBLIC_PACKAGES}")
    for attr in rest:
        obj = getattr(obj, attr)
    return obj


def _ids() -> list[str]:
    return [r["Component"] for r in ROWS]


def test_matrix_is_not_empty():
    assert len(ROWS) >= 10


@pytest.mark.parametrize("row", ROWS, ids=_ids())
def test_status_is_known(row):
    assert row["Status"] in STATUSES, row["Status"]


@pytest.mark.parametrize("row", ROWS, ids=_ids())
def test_named_symbols_import(row):
    for name in _SYMBOL.findall(row["VALAX today"]):
        _resolve(name)


@pytest.mark.parametrize("row", [r for r in ROWS if r["Status"] in {"✅", "🟡"}],
                         ids=lambda r: r["Component"])
def test_shipped_or_partial_rows_name_a_symbol(row):
    assert _SYMBOL.findall(row["VALAX today"]), (
        f"{row['Component']}: status {row['Status']} but no implementing symbol")


@pytest.mark.parametrize("row", [r for r in ROWS if r["Status"] == "✅"],
                         ids=lambda r: r["Component"])
def test_shipped_rows_have_an_existing_gate(row):
    gates = re.findall(r"`(tests/[^`]+\.py)`", row["Gate"])
    assert gates, f"{row['Component']}: ✅ without a gate"
    for gate in gates:
        assert (ROOT / gate).is_file(), gate


@pytest.mark.parametrize("row", [r for r in ROWS if r["Status"] in {"🟡", "📋"}],
                         ids=lambda r: r["Component"])
def test_planned_rows_link_to_an_open_roadmap_item(row):
    anchors = _ROADMAP_LINK.findall(row["Gap and roadmap"])
    assert anchors, f"{row['Component']}: {row['Status']} without a roadmap link"
    sections = roadmap_sections()
    assert any(
        a in sections and re.search(r"^\s*- \[ \]", sections[a], re.M)
        for a in anchors
    ), f"{row['Component']}: no linked roadmap section has an open item ({anchors})"
