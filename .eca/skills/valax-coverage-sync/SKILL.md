---
name: valax-coverage-sync
description: Cross-check VALAX's coverage/progress documentation against the actual code and tests, then update the lists to match. Edits docs (canonical maps first, then derived lists) and verifies with pytest + mkdocs build --strict. Use when asked to reconcile, sync, or update the roadmap / coverage matrices after code changes, or to make the docs truthful again.
---

# valax-coverage-sync

Make the coverage/progress documentation match the code. Establish ground truth from `valax/` + `tests/`, update the two canonical maps, propagate to the derived lists, and verify the build. For a read-only proposal of what to build next from the same maps, use `valax-next-steps`.

## Ground truth = code + tests (never memory)

Status is derived only from what the code actually provides:

- **Implemented surface:** enumerate public symbols from the package `__init__.py` files and modules:
  `valax/pricing/{analytic,mc,pde,lattice}/`, `valax/instruments/`, `valax/models/`, `valax/calibration/`, `valax/curves/`, `valax/surfaces/`, `valax/risk/`, `valax/greeks/`, `valax/portfolio/`, `valax/market/`.
- **✅ vs 🟡:** a feature is ✅ *implemented & tested* only if there is a pricer/function **and** a passing test exercising it (look in the mirrored `tests/` package, and where a QuantLib counterpart exists, `tests/test_quantlib_comparison/`). A pytree/instrument defined but with no pricer is 🟡. Anything else is 📋 (planned).
- Cite the implementing `valax.*` symbol (module + name) for every ✅ you write.

## Sources of truth and propagation order

Edit in this order so no new contradictions are created:

1. `docs/applications/instrument-pricing.md` — the coverage matrix: status legend ✅/🟡/n-a, plus the four solver columns (`Analytic | MC | PDE | Lattice`; cells ✅ routed · ◇ kernel only · `[🟡](../roadmap.md#section)` planned · `✅ [+🟡](…)` routed with more models planned · —). A ◇ needs a `KERNEL_ONLY` entry in `tests/test_docs/test_price_routes.py`; a 🟡 needs an open `- [ ]` item in the linked roadmap section. That test is the mechanical check — run it first; it pinpoints most drift.
2. `docs/roadmap.md` — flip `[ ]`→`[x]` (or add the line) naming the symbol; update the Current-State tables; record the blocker on every open item a 🟡 cell points to. (The roadmap no longer carries its own instrument matrix — it links to the coverage matrix.)
3. Propagate to the other coverage maps — all listed on `docs/coverage.md` (pages carry a `<!-- coverage-map -->` marker; `tests/test_docs/test_coverage_hub.py` keeps the list complete): `docs/guide/calibration.md`, `docs/applications/calibration-register.md`, `docs/theory/calibration-chains.md`, `docs/architecture/aad-audit-conformance.md`, `docs/risk-factors.md`, `docs/risk-overview.md`, `docs/applications/index.md` (section 4), `docs/theory/multi-curve-short-rate.md`, `docs/architecture/production.md`, `docs/scaling/index.md`.
4. Solver guides (`docs/guide/{analytical,monte-carlo,pde,lattice}.md`) and `docs/api/pricing.md` hold **no status** — only engine details (scheme, dimension, oracle, recipe notes) and a link to the matrix; `test_coverage_hub.py` fails on status marks in their tables. Fix stale engine prose there, never add a status table.

## Procedure

1. **Scan the code** to build the implemented-symbol inventory (read the relevant `__init__.py` files and module signatures).
2. **Scan tests** to mark which of those are tested (✅) vs merely present (🟡).
3. **Diff against the docs:** for every doc status entry, compare claimed vs actual; build a change set.
4. **Apply edits**, canonical files first (order above), each ✅ citing its symbol. Respect the AGENTS.md "definition of done": when you mark something implemented, also ensure/flag the owning guide, API export, and theory page — add the row/link where trivial, otherwise list it as follow-up.
5. **Never fabricate.** If code presence is ambiguous, leave the status unchanged and add it to the follow-up list rather than guessing.
6. **Verify:** run `pytest tests/test_docs` (matrix ↔ registry/kernels/roadmap, hub completeness, calibration register), then `pytest` (or a targeted subset if the change is scoped), `mkdocs build --strict` and `python scripts/check_doc_anchors.py`; all must be green.
7. **Report** what changed (file:line, old→new, citing symbol/test) and what still needs human authoring (theory derivations, new guide sections).

## Rules

- Code and tests are the only source of truth for status; docs conform to them, never the reverse.
- Make the minimal edits that restore truth; do not rewrite prose beyond the status/symbol being corrected.
- Keep the two canonical maps (`instrument-pricing.md`, `roadmap.md`) as the drivers; derived lists follow.
- Always finish with a green `mkdocs build --strict`.
