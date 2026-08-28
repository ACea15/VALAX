# Pitfalls — things that do *not* work (yet), and why

This folder is a lab bench for **known failure modes**: numerical traps, solver
fragilities, and modelling gotchas discovered while building VALAX. Each script
is a self-contained, runnable reproduction you can open and *poke at* — flip a
solver, tighten a tolerance, rescale the problem — to build intuition about
what's actually going wrong.

These are **not** examples of correct usage (see `examples/` for that). They are
deliberately red: they demonstrate something broken or subtle, isolate it to the
smallest reproduction, and hand you the knobs to experiment.

## Conventions

- Cell-based (`# %%`) like the rest of `examples/` — run top-to-bottom, or
  step through cells in an interactive window (VS Code / Jupytext).
- Each file starts with a **SYMPTOM / DIAGNOSIS / KNOBS** header.
- Experiments are wrapped so the script runs end-to-end without crashing, even
  when the underlying call diverges or raises — the point is to *observe* it.
- Where a fix is known, it's named at the bottom under **WAY OUT**, but the
  fix is intentionally *not* applied here.

## Index

| File | Pitfall |
|------|---------|
| `01_sabr_vol_convention_mismatch.py` | Calibrating a **normal**-quoted SABR smile with the **lognormal** Hagan formula (a vol-convention mismatch). The fit reports a small residual but returns nonsense parameters, so anything re-priced in the correct convention is badly wrong. Shows how residual-at-truth + wrapper-vs-direct diagnostics tell a *model mismatch* apart from a *solver* problem. (Historically a real `calibrate_sabr` bug — a dropped `vol_fn` — that this playground caught.) |
