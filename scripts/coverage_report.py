"""Print the gaps of the instrument × solver coverage matrix.

Reads ``docs/applications/instrument-pricing.md`` through the parser the docs
test uses (``tests/test_docs/coverage_matrix.py``) and lists, per solver
family, the planned (🟡) cells — with the roadmap section that tracks each —
and, unless ``--planned``, the absent (—) ones. Also prints a per-solver
summary. Parsing only: runs without JAX.

Usage::

    python scripts/coverage_report.py              # planned + absent, by solver
    python scripts/coverage_report.py --planned    # planned only
    python scripts/coverage_report.py --markdown   # as a Markdown table
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "test_docs"))

from coverage_matrix import KERNEL, NONE, PLANNED, ROUTED, SOLVERS, rows  # noqa: E402

LABEL = {ROUTED: "✅", KERNEL: "◇", PLANNED: "🟡", NONE: "—"}


def main(argv: list[str] | None = None) -> int:
    """Print the report.

    Args:
        argv: command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--planned", action="store_true",
                        help="only planned cells (and routed cells with planned extensions)")
    parser.add_argument("--markdown", action="store_true", help="Markdown output")
    args = parser.parse_args(argv)

    matrix = rows()
    summary = {s: Counter() for s in SOLVERS}
    gaps = []
    for row in matrix:
        for solver, cell in row.solvers.items():
            summary[solver][cell.kind] += 1
            planned = cell.roadmap_anchor is not None
            if planned or (cell.kind == NONE and not args.planned):
                what = ("planned (more models)" if cell.kind == ROUTED
                        else "planned" if planned else "absent")
                where = f"roadmap.md#{cell.roadmap_anchor}" if planned else ""
                gaps.append((solver, row.asset_class, row.name, what, where))

    if args.markdown:
        print("| Solver | Asset class | Instrument | Gap | Tracked at |")
        print("|---|---|---|---|---|")
        for g in gaps:
            print("| " + " | ".join(g) + " |")
    else:
        for solver in SOLVERS:
            mine = [g for g in gaps if g[0] == solver]
            print(f"\n== {solver} ({len(mine)} gaps)")
            for _, asset, name, what, where in mine:
                print(f"  {asset[:28]:<28} {name:<32} {what:<22} {where}")

    print("\nSummary (cells per solver): "
          + " · ".join(f"{k} {LABEL[k]}" for k in (ROUTED, KERNEL, PLANNED, NONE)))
    for solver in SOLVERS:
        c = summary[solver]
        print(f"  {solver:<9} " + "  ".join(
            f"{LABEL[k]} {c[k]:>2}" for k in (ROUTED, KERNEL, PLANNED, NONE)))
    print(f"  {len(matrix)} instruments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
