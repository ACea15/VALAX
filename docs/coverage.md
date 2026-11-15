# Coverage Maps

This page lists every coverage map in the docs: what each one answers, which
page owns it, and whether a test keeps it in step with the code. Start here to
check what VALAX covers today, or to find the gaps when deciding what to build
next.

Each map has exactly **one** owning page. Other pages link to it and do not
repeat its status marks. Any page that carries a coverage map has a
`<!-- coverage-map -->` marker. `tests/test_docs/test_coverage_hub.py` fails if
a marked page is missing from the table below, or if a page in the table has no
marker.

## The maps

| Map | Question it answers | Dimensions | Owning page | Kept in step with the code by |
|---|---|---|---|---|
| **Instrument × solver** | What is priced, by which solver family, under which model, through `price()` or by direct call, and what is planned? | instrument × {Analytic, MC, PDE, Lattice} × model; status | [Instrument Pricing — coverage matrix](applications/instrument-pricing.md#coverage-matrix) | `tests/test_docs/test_price_routes.py`, which checks routes and `auto` against the registry, ✅ cells against registered methods, ◇ cells against named kernels, 🟡 cells against open roadmap items, and status against cells |
| **Roadmap** | What is built (`[x]`) and what is planned (`[ ]`), and what blocks each item? | tier × item | [Roadmap](roadmap.md) | Partly: every 🟡 matrix cell must link to a roadmap section that still has an open item |
| **Calibration pipeline** | Which liquid quotes calibrate which artefact, and with which routine? | market × routine × artefact | [Calibration — calibration matrix](guide/calibration.md#calibration-matrix) | Manual |
| **Calibration register** | What each calibration fits, assumes and gets wrong, with evidence | calibration entry point × fields | [Calibration Assumptions Register](applications/calibration-register.md) | `tests/test_docs/test_calibration_register.py`, which requires every exported entry point to have an entry |
| **Risk register** | What each risk measure assumes, gets wrong (with evidence) and how it improves | risk measure × fields | [Risk Methodology & Assumptions Register](applications/risk-register.md) | `tests/test_docs/test_risk_register.py`, which requires every public risk-measure entry point to have an entry and every named gate to exist |
| **Calibration chain levels** | Which chain levels and nodes exist (quotes → curves → vols → model) | chain level × node | [Calibration Chains — Part II](theory/calibration-chains.md#part-ii-how-valax-implements-the-chain) | Manual |
| **AAD audit conformance** | Which audit requirements of the calibration / adjoint chain are met | requirement × status | [AAD Audit Conformance](architecture/aad-audit-conformance.md) | Manual (its own scoreboard) |
| **Risk factors** | Which risk factors exist, how each is shocked, and which instruments are exposed to them | factor × primitive; instrument × factor | [Risk Factors](risk-factors.md) (instrument × factor: [§ 5](risk-factors.md#5-instrument-factor-matrix)) | Manual |
| **Risk pipeline** | Which stages of the risk pipeline ship | stage × status | [Risk Overview § 7](risk-overview.md#7-the-state-of-the-risk-section-and-whats-next) | Manual |
| **FRTB capabilities** | Which FRTB SA / IMA (and SIMM) components ship, which are partial, which are planned | component × approach × status | [Regulatory § 2.7 — FRTB capability matrix](applications/regulatory.md#frtb-capability-matrix) | `tests/test_docs/test_frtb_capabilities.py`, which checks that named symbols import, gates exist, and 🟡 / 📋 rows link to an open roadmap item |
| **Business capabilities** | Which bank capabilities VALAX serves today, for which business | capability × business × status | [Applications Overview § 4](applications/index.md#4-coverage-what-ships-today-vs-roadmap) | Manual |
| **Multi-curve short-rate scope** | Which short-rate engines honour a dual-curve market | engine × capability | [Multi-Curve Short Rates § 5](theory/multi-curve-short-rate.md#5-scope-in-valax-today) | Manual |
| **Production readiness** | Which productionisation layers exist | layer × status | [Productionisation § 11.1](architecture/production.md#111-status-today) | Manual |
| **Scaling ladder** | Which scaling rungs (vmap, sharding, multi-device) ship | rung × status | [Scaling Up](scaling/index.md#the-scaling-ladder) | Manual |

## Rules

- **Instrument × solver status lives only in the matrix.** The solver guides
  ([Analytical](guide/analytical.md), [Monte Carlo](guide/monte-carlo.md),
  [PDE](guide/pde.md), [Lattice](guide/lattice.md)) and
  [API: Pricing](api/pricing.md) document engine mechanics only: scheme,
  dimension, cross-check oracle and recipe notes. They link to the matrix for
  status. `test_coverage_hub.py` fails if a table in one of those pages
  carries a status mark.
- **Planned work has a home on the roadmap.** A 🟡 matrix cell links to the
  roadmap section that tracks it, and that section must have an open item that
  records the blocker.
- **A new coverage table** gets the marker and a row here, in the same change.
  If it overlaps an existing map, link to that map instead of adding a table.

## Finding what is missing

```bash
python scripts/coverage_report.py              # planned (🟡) and absent (—) cells, by solver
python scripts/coverage_report.py --planned    # only the planned ones, with their roadmap section
python scripts/coverage_report.py --markdown   # the same as a Markdown table
```

The report reads the instrument × solver matrix through the same parser the
test uses. For a ranked backlog of what to build next, read it together with
the roadmap's
[Production Priority Roadmap](roadmap.md#production-priority-roadmap-snapshot-reconciled-against-the-codebase).
