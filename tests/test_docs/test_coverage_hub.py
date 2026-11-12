"""The coverage hub (``docs/coverage.md``) indexes every coverage map.

A page that owns a coverage map carries a ``<!-- coverage-map -->`` marker;
the hub's table must list exactly the marked pages, and the hub must be in the
MkDocs nav. Instrument × solver status lives only in the coverage matrix of
``docs/applications/instrument-pricing.md``, so the solver guides and the
pricing API page must not carry status marks in their tables.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
HUB = DOCS / "coverage.md"
MARKER = "<!-- coverage-map -->"

#: Pages that document engines, not coverage: status lives in the matrix.
ENGINE_PAGES = (
    "guide/analytical.md",
    "guide/monte-carlo.md",
    "guide/pde.md",
    "guide/lattice.md",
    "api/pricing.md",
)
STATUS_MARKS = ("✅", "🟡", "✓", "🟠", "🔴", "◇")


def _marked_pages() -> set[str]:
    return {
        p.relative_to(DOCS).as_posix()
        for p in DOCS.rglob("*.md")
        if p != HUB and MARKER in p.read_text()
    }


def _hub_pages() -> set[str]:
    """Pages linked from the hub's "The maps" table."""
    text = HUB.read_text()
    table = text.split("## The maps", 1)[1].split("\n## ", 1)[0]
    pages = set()
    for line in table.splitlines():
        if line.startswith("|"):
            pages |= set(re.findall(r"\]\(([^)#]+\.md)(?:#[^)]*)?\)", line))
    return pages


def test_every_marked_page_is_on_the_hub():
    missing = _marked_pages() - _hub_pages()
    assert not missing, f"coverage maps not listed on docs/coverage.md: {missing}"


def test_every_hub_page_is_marked():
    unmarked = _hub_pages() - _marked_pages()
    assert not unmarked, f"hub lists pages without a {MARKER} marker: {unmarked}"


def test_hub_links_resolve():
    for page in _hub_pages():
        assert (DOCS / page).is_file(), page


def test_hub_is_in_the_nav():
    assert re.search(r":\s*coverage\.md\s*$", (ROOT / "mkdocs.yml").read_text(), re.M)


@pytest.mark.parametrize("page", ENGINE_PAGES)
def test_engine_pages_carry_no_status_marks(page):
    """Status per instrument × solver is the matrix's job, not the guide's."""
    offending = [
        line for line in (DOCS / page).read_text().splitlines()
        if line.lstrip().startswith("|") and any(m in line for m in STATUS_MARKS)
    ]
    assert not offending, (
        f"{page} has status marks in a table — state status in the coverage "
        f"matrix (applications/instrument-pricing.md) and link to it: {offending}")
