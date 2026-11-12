"""Parser for the coverage matrix of ``docs/applications/instrument-pricing.md``.

Shared by ``tests/test_docs/test_price_routes.py`` (which checks the matrix
against the code) and ``scripts/coverage_report.py`` (which prints the gaps).
Parsing only — no ``valax`` import, so the report runs without JAX.

Solver-cell grammar (the legend above the matrix):

* ``✅`` — routed through ``price()`` for at least one model;
* ``◇`` — kernel only (a tested pricer called directly);
* ``[🟡](../roadmap.md#section)`` — planned, tracked in that roadmap section;
* ``✅ [+🟡](../roadmap.md#section)`` — routed, with more models planned;
* ``—`` — not offered and not planned.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "docs" / "applications" / "instrument-pricing.md"
ROADMAP = ROOT / "docs" / "roadmap.md"

ROUTES = "`price()` routes"
SOLVERS: tuple[str, ...] = ("Analytic", "MC", "PDE", "Lattice")
#: Solver column -> the ``method=`` name ``price()`` registers it under.
METHOD_OF: dict[str, str] = {
    "Analytic": "analytic", "MC": "mc", "PDE": "pde", "Lattice": "lattice",
}

ROUTED, KERNEL, PLANNED, NONE = "routed", "kernel", "planned", "none"

_PLANNED = re.compile(r"^\[🟡\]\(\.\./roadmap\.md#([^)\s]+)\)$")
_EXTENDED = re.compile(r"^✅ \[\+🟡\]\(\.\./roadmap\.md#([^)\s]+)\)$")


@dataclass(frozen=True)
class SolverCell:
    """One solver cell: its kind and, if planned work is linked, the anchor."""

    kind: str
    roadmap_anchor: str | None = None


@dataclass(frozen=True)
class Row:
    """One instrument row of the coverage matrix."""

    anchor: str
    name: str
    asset_class: str
    routes: str
    solvers: dict[str, SolverCell]
    status: str
    line: str


def parse_solver_cell(cell: str) -> SolverCell:
    """Parse one solver cell; raises ``ValueError`` outside the grammar."""
    cell = cell.strip()
    if cell == "✅":
        return SolverCell(ROUTED)
    if cell == "◇":
        return SolverCell(KERNEL)
    if cell == "—":
        return SolverCell(NONE)
    if m := _PLANNED.match(cell):
        return SolverCell(PLANNED, m.group(1))
    if m := _EXTENDED.match(cell):
        return SolverCell(ROUTED, m.group(1))
    raise ValueError(f"solver cell outside the legend: {cell!r}")


def rows(path: Path = MATRIX) -> list[Row]:
    """Every instrument row of every asset-class table of the matrix."""
    text = path.read_text()
    section = text.split("## Coverage matrix", 1)[1].split("\n---\n", 1)[0]
    out: list[Row] = []
    header: list[str] | None = None
    asset_class = ""
    for line in section.splitlines():
        if line.startswith("### "):
            asset_class = line[4:].strip()
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "Instrument":
            missing = [c for c in (*SOLVERS, ROUTES, "Status") if c not in cells]
            if missing:
                raise ValueError(f"matrix table without columns {missing}: {line}")
            header = cells
            continue
        if header is None or set(cells[0]) <= {"-"}:
            continue
        if len(cells) != len(header):
            raise ValueError(
                f"row has {len(cells)} cells, header {len(header)}: {line}")
        link = re.search(r"\[([^\]]+)\]\(#([^)]+)\)", cells[0])
        if not link:
            raise ValueError(f"row without an anchor link: {line}")
        col = dict(zip(header, cells))
        out.append(Row(
            anchor=link.group(2),
            name=link.group(1),
            asset_class=asset_class,
            routes=col[ROUTES],
            solvers={s: parse_solver_cell(col[s]) for s in SOLVERS},
            status=col["Status"],
            line=line,
        ))
    return out


def slugify(title: str) -> str:
    """The heading id MkDocs (Python-Markdown ``toc``) generates."""
    title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    title = re.sub(r"[^\w\s-]", "", title).strip().lower()
    return re.sub(r"[-\s]+", "-", title)


def roadmap_sections(path: Path = ROADMAP) -> dict[str, str]:
    """``heading id -> section body`` (up to the next heading of the same or a
    higher level) for every heading of the roadmap."""
    lines = path.read_text().splitlines()
    heads = []
    fenced = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if fenced:
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
        if m:
            title = m.group(2)
            explicit = re.search(r"\{#([^}]+)\}\s*$", title)
            ident = explicit.group(1) if explicit else slugify(title)
            heads.append((i, len(m.group(1)), ident))
    out = {}
    for k, (i, level, ident) in enumerate(heads):
        end = next((j for j, lv, _ in heads[k + 1:] if lv <= level), len(lines))
        out[ident] = "\n".join(lines[i + 1:end])
    return out
