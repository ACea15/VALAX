---
name: valax-next-steps
description: Propose the next steps of engineering work for VALAX, grounded in the source of truth of what is already implemented (docs/roadmap.md + docs/applications/instrument-pricing.md, corroborated by the code). Read-only — produces a prioritized, actionable backlog with rationale, prerequisites, and implementing targets. Use when asked "what should we build next", to plan the next sprint or PR, or to turn the roadmap into concrete next actions.
---

# valax-next-steps

Turn the current state of VALAX into a prioritized list of *what to build next*. Read what is already implemented from the source of truth, find the frontier of unbuilt work, and propose concrete, sequenced next steps. **Read-only: it proposes work; it does not implement it or edit docs.** To make the status docs themselves truthful against the code, use `valax-coverage-sync`.

## Source of truth for "what is implemented"

Per AGENTS.md, status is defined by two canonical files, corroborated by the code so you never propose something already done:

1. `docs/roadmap.md` — Current-State tables, the Instrument Coverage Matrix, the Priority 1–5 snapshot with its **Recommended Execution Order** and **Assessment**, and the Tier 1–6 `[x]`/`[ ]` checklists (each implemented item names a `valax.*` symbol).
2. `docs/applications/instrument-pricing.md` — coverage matrix (instrument x method x model input x calibration x status). Legend: ✅ implemented & tested · 🟡 pytree defined but pricer not implemented · n-a.
3. **Code check (to avoid stale proposals):** confirm a sample of ✅ claims by reading `valax/` exports + the mirrored `tests/`; treat 🟡 (defined, no pricer), 📋 (planned), and empty matrix cells as candidate work.

## What counts as a "next step"

Candidate work items, gathered from the source of truth:

- Every roadmap `[ ]` line (unshipped tier / priority item).
- Every 🟡 in the coverage matrix (instrument pytree exists, pricer missing) — usually a small, high-value step.
- Empty/missing cells in the instrument x method matrix (e.g. an instrument with an analytic pricer but no MC/PDE route where one is expected).
- "Definition-of-done" gaps for things that ARE built but under-documented (missing owning guide, API export, or theory page) — cheap wins.
- Follow-ups explicitly noted in roadmap prose (e.g. "Carr-Madan remains as a follow-up", "CDS / CDO pricers pending").

## Ranking heuristics (in order)

1. **Dependency readiness** — prefer unblocked work. Honor roadmap-stated blockers (e.g. `MarketState` / multi-curve `MarketData` blocks a large fraction of downstream items); never propose a dependent before its prerequisite.
2. **Stated priority** — P1 > P2 > ... and the roadmap's own **Recommended Execution Order** is a strong prior.
3. **Leverage / impact** — items that unlock many others or close a whole audience gap (cross-reference `docs/applications/index.md` section 5 and `docs/risk-factors.md`).
4. **Effort / risk** — call out quick wins (a 🟡 pricer, a missing API export) separately from multi-quarter efforts.

## Procedure

1. Read the two canonical maps; build the done-vs-not frontier. Spot-check a sample of ✅ against `valax/` + `tests/` so you trust the "done" side.
2. Enumerate candidate next steps from the sources above.
3. Rank them with the heuristics; note prerequisites and what each unblocks.
4. For each proposed step give: **what** (concrete target — module/symbol to add, e.g. `valax/pricing/analytic/...` + mirrored test), **why now** (priority + dependency rationale), **prerequisites**, **definition-of-done deliverables** (matrix row, owning guide, roadmap `[x]`, API export, theory page — per AGENTS.md), and **acceptance criteria** (tests to pass, QuantLib cross-check if a counterpart exists).
5. Present a short **"next steps" plan**: a Top-N ordered list plus a separate **Quick wins** bucket. Flag where the code reality diverges from the roadmap's stated ordering.

## Rules

- **Read-only.** Propose work; do not implement or edit docs. If the roadmap/matrix themselves look wrong vs the code, say so and refer to `valax-coverage-sync`.
- Ground every proposal in the source of truth; never invent an item that is already ✅ (verify against code), and never propose skipping a stated prerequisite.
- Prefer a small number of high-confidence, well-sequenced steps over an exhaustive dump.
