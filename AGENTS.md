# AGENTS.md

VALAX is a JAX-native quantitative finance valuation engine (Python ≥3.11). See `CONTRIBUTING.md` for full detail.

**Before assuming a feature's status, check the two canonical maps — never guess from memory:** the coverage matrix (`docs/applications/instrument-pricing.md`) and the roadmap (`docs/roadmap.md`). Keeping those two truthful is a hard requirement of every change (see *Definition of done* below).

## Build / lint / test
- Install: `pip install -e ".[dev]"` (optional: `pip install -e ".[docs]"`, `pip install QuantLib` for `tests/test_quantlib_comparison/`).
- Full suite: `pytest`. Benchmarks only: `pytest --benchmark-only`. Marker filter: `pytest -m "golden and detects"` (markers: `arbitrage`, `golden`, `detects`).
- Single file: `pytest tests/test_pricing/test_black_scholes.py`. Single test: `pytest tests/test_pricing/test_black_scholes.py::test_call_price -v`.
- Docs (strict, must pass pre-PR): `mkdocs build --strict`. Serve: `mkdocs serve`.
- No configured linter/formatter; match surrounding style and keep `mkdocs build --strict` + `pytest` green.

## Code style
- **Imports**: stdlib → third-party (`jax`, `jax.numpy as jnp`, `equinox as eqx`, `diffrax`, `optimistix`, `optax`, `lineax`) → `jaxtyping` (`Float`, `Int`) and `from jax import Array` → `valax.*` absolute imports. No relative imports; no `import *`.
- **Deps (use the JAX-ecosystem tool, never hand-roll)**: `diffrax` for SDE path simulation, `optimistix` for root-finding / least-squares, `optax` for gradient-based optimization, `lineax` for structured linear solvers, `equinox` for pytree data + `filter_jit`/`filter_grad`.
- **Types**: annotate every array arg with `jaxtyping` and *named* shape dims (`Float[Array, "n_paths n_steps"]`, scalars `Float[Array, ""]`). Pricing fns return a scalar JAX array.
- **Data structures**: all structured data subclasses `equinox.Module` (frozen, pytree). Mark non-differentiable fields with `eqx.field(static=True)` (e.g. `is_call`, day-counts, `n_steps`).
- **JAX rules**: pure functions only — no mutable state, no `self.cache`, no mutated globals/lists/dicts, no `print` in JIT-traced code. Use `@eqx.filter_jit` / `eqx.filter_grad` (not raw `jax.jit`) when modules carry static fields. Greeks via `jax.grad`/`jax.jacobian`, never finite differences (except as test sanity checks). Smooth discontinuous payoffs with sigmoids (see `valax/pricing/mc/payoffs.py`).
- **No-go inside traced code**: `scipy.*`, Python `datetime` (use integer day ordinals via `valax/dates/`), class hierarchies for dispatch (use plain functions or `functools.singledispatch`). Instruments are data-only — pricing lives in separate functions (`price(instrument, *market_args)`), never `instrument.npv()`.
- **Naming**: `snake_case` for functions/modules/variables, `PascalCase` for `eqx.Module` classes, `UPPER_SNAKE` for module-level constants (assigned exactly once). Tests mirror source: `valax/X/Y.py` ↔ `tests/test_X/test_Y.py`; exception: pricing sub-packages map to top-level test packages (`valax/pricing/analytic/` ↔ `tests/test_pricing/`, `valax/pricing/mc/` ↔ `tests/test_mc/`, `valax/pricing/pde/` ↔ `tests/test_pde/`, `valax/pricing/lattice/` ↔ `tests/test_lattice/`).
- **Docstrings**: Google-style (`Args:` / `Returns:` / `Raises:` / `References:`), never NumPy-style underlined sections. Document `Args:` and `Returns:` on every public function/class. Use LaTeX math in `.. math::` blocks. Do **not** repeat `jaxtyping` type/shape annotations inside the docstring (the signature is the source of truth) — describe semantics only.
- **Error handling**: prefer JAX-friendly numerical guards (`jnp.maximum(x, eps)`, `jnp.where(...)`) over Python `raise` inside traced code; raise `ValueError`/`TypeError` only at the user-boundary (pre-trace) for invalid configuration.
- **Tests**: closed-form to `atol≈1e-10`; autodiff Greeks vs FD `rtol≈1e-4`; MC within `2*stderr` of analytic; use `hypothesis` for invariants (parity, monotonicity); add a `@eqx.filter_jit` smoke test for every new pricing fn; add a QuantLib comparison when a QL counterpart exists.

## Layout
- **Source:** `valax/{core,dates,curves,surfaces,instruments,models,calibration,market,greeks,risk,portfolio}/` (domain packages) and `valax/pricing/{analytic,mc,pde,lattice}/` (pricing impls grouped by numerical method).
- **Docs:** `docs/` served by MkDocs (nav lives in `mkdocs.yml`), grouped by audience — `applications/` (management-facing), `guide/` (how-to), `theory/` (the maths), `api/` (mkdocstrings autodoc), `architecture/` (design + validation), plus `roadmap.md`.

## What exists — sources of truth
Do not infer a feature's presence/absence from memory or from a single guide page. The authoritative maps are:
- **Coverage matrix** — `docs/applications/instrument-pricing.md`: every instrument × pricing method × model input × calibration route × status. The single authoritative instrument→pricer map.
- **Roadmap** — `docs/roadmap.md`: implemented (`[x]`) vs planned (`[ ]`) across tiers, each with the implementing symbol.
- **Status legend** (used by both): ✅ implemented & tested · 🟡 instrument pytree defined but pricer not yet implemented · n-a (no vol input; priced by curve discounting).

## Documentation (a first-class deliverable, not an afterthought)
Docs go stale when code lands without doc updates — the failure mode this project has already paid for. Docs are governed by a **single-source-of-truth ownership model**; respect it so nothing is documented twice:
- **Coverage matrix** (`applications/instrument-pricing.md`) — instrument *definitions* + the coverage matrix + links to theory. **No pricing derivations here.**
- **Instrument guides** (`guide/{fixed-income,callable-bonds,rates-exotics,fx,inflation,equity-exotics,credit}.md`) — each instrument's payoff/cashflow maths and how-to recipe live on **exactly one** asset-class guide.
- **Solver guides** (`guide/{analytical,monte-carlo,pde,lattice}.md`) — engine mechanics + that engine's model-coverage matrix + the vanilla contract for that engine. Do **not** re-derive an instrument's exotic payoff here; link to its instrument guide.
- **Calibration** (`guide/{calibration,curves,why-multicurve,vol-surfaces,slv}.md`), **Theory** (`theory/`, the maths), **API** (`api/`, mkdocstrings autodoc of public symbols).
- **No duplication.** Document each instrument, derivation, or selling point once and cross-link. If you catch yourself copying a formula, example, or status table between pages, replace the copy with a link.

## Definition of done — adding/changing a pricer, instrument, model, calibration, or surface
In the **same change** (not "later"), and stated with clear intent in the commit/PR body:
1. **Coverage matrix** — add or adjust the row in `applications/instrument-pricing.md` so the method(s), model input, and ✅/🟡 status match the code exactly.
2. **Owning guide** — document it on the one correct page (instrument guide for a new instrument; solver guide for a new engine/model recipe) with a runnable, JIT-safe example; cross-link from elsewhere rather than copy.
3. **Roadmap** — flip `[ ]`→`[x]` (or add the line) naming the implementing symbol/module. Never leave an implemented feature unchecked; never mark an unbuilt one done. A change that only defines an instrument pytree with no pricer is 🟡, not ✅.
4. **API** — export new public symbols (`__init__.py`) and confirm they render on the relevant `docs/api/*.md` autodoc page.
5. **Theory** — add/extend the maths page when introducing new modelling.
6. **Verify** — both `pytest` and `mkdocs build --strict` must be green (strict catches broken doc links and orphaned/missing nav entries).

When **removing or renaming**, grep the docs for inbound links to the old path/anchor and fix them before deleting; `mkdocs build --strict` guards page-level links but not intra-page anchors. The coverage matrix and roadmap are the two files you must never forget to touch.
