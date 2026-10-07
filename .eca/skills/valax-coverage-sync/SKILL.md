---
name: valax-coverage-sync
description: Cross-check VALAX's coverage/progress documentation against the actual code and tests, then update the lists to match. Edits docs (canonical maps first, then derived lists) and verifies with pytest + mkdocs build --strict. Use when asked to reconcile, sync, or update the roadmap / coverage matrices after code changes, or to make the docs truthful again.
---

# valax-coverage-sync

Make the coverage/progress documentation match the code. Establish ground truth from `valax/` + `tests/`, update the two canonical maps, propagate to the derived lists, and verify the build. For a read-only audit that only *suggests* changes, use `valax-coverage-review`.

## Ground truth = code + tests (never memory)

Status is derived only from what the code actually provides:

- **Implemented surface:** enumerate public symbols from the package `__init__.py` files and modules:
  `valax/pricing/{analytic,mc,pde,lattice}/`, `valax/instruments/`, `valax/models/`, `valax/calibration/`, `valax/curves/`, `valax/surfaces/`, `valax/risk/`, `valax/greeks/`, `valax/portfolio/`, `valax/market/`.
- **✅ vs 🟡:** a feature is ✅ *implemented & tested* only if there is a pricer/function **and** a passing test exercising it (look in the mirrored `tests/` package, and where a QuantLib counterpart exists, `tests/test_quantlib_comparison/`). A pytree/instrument defined but with no pricer is 🟡. Anything else is 📋 (planned).
- Cite the implementing `valax.*` symbol (module + name) for every ✅ you write.

## Sources of truth and propagation order

Edit in this order so no new contradictions are created:

1. `docs/applications/instrument-pricing.md` — the coverage matrix (legend ✅/🟡/n-a).
2. `docs/roadmap.md` — flip `[ ]`→`[x]` (or add the line) naming the symbol; update Current-State tables and the Instrument Coverage Matrix.
3. Propagate to derived lists so they agree: `docs/applications/index.md` (section 5), `docs/risk-factors.md`, `docs/risk-overview.md`, `docs/guide/{pde,lattice,monte-carlo,calibration}.md`, `docs/api/pricing.md`, `docs/architecture/production.md`.

## Procedure

1. **Scan the code** to build the implemented-symbol inventory (read the relevant `__init__.py` files and module signatures).
2. **Scan tests** to mark which of those are tested (✅) vs merely present (🟡).
3. **Diff against the docs:** for every doc status entry, compare claimed vs actual; build a change set.
4. **Apply edits**, canonical files first (order above), each ✅ citing its symbol. Respect the AGENTS.md "definition of done": when you mark something implemented, also ensure/flag the owning guide, API export, and theory page — add the row/link where trivial, otherwise list it as follow-up.
5. **Never fabricate.** If code presence is ambiguous, leave the status unchanged and add it to the follow-up list rather than guessing.
6. **Verify:** run `pytest` (or a targeted subset if the change is scoped) and `mkdocs build --strict`; both must be green.
7. **Report** what changed (file:line, old→new, citing symbol/test) and what still needs human authoring (theory derivations, new guide sections).

## Rules

- Code and tests are the only source of truth for status; docs conform to them, never the reverse.
- Make the minimal edits that restore truth; do not rewrite prose beyond the status/symbol being corrected.
- Keep the two canonical maps (`instrument-pricing.md`, `roadmap.md`) as the drivers; derived lists follow.
- Always finish with a green `mkdocs build --strict`.
