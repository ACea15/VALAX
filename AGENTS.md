# AGENTS.md

VALAX is a JAX-native quantitative finance valuation engine (Python ≥3.11). See `CONTRIBUTING.md` for full detail.

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
- **Naming**: `snake_case` for functions/modules/variables, `PascalCase` for `eqx.Module` classes, `UPPER_SNAKE` for module-level constants (assigned exactly once). Tests mirror source: `valax/X/Y.py` ↔ `tests/test_X/test_Y.py`.
- **Docstrings**: Google-style (`Args:` / `Returns:` / `Raises:` / `References:`), never NumPy-style underlined sections. Document `Args:` and `Returns:` on every public function/class. Use LaTeX math in `.. math::` blocks. Do **not** repeat `jaxtyping` type/shape annotations inside the docstring (the signature is the source of truth) — describe semantics only.
- **Error handling**: prefer JAX-friendly numerical guards (`jnp.maximum(x, eps)`, `jnp.where(...)`) over Python `raise` inside traced code; raise `ValueError`/`TypeError` only at the user-boundary (pre-trace) for invalid configuration.
- **Tests**: closed-form to `atol≈1e-10`; autodiff Greeks vs FD `rtol≈1e-4`; MC within `2*stderr` of analytic; use `hypothesis` for invariants (parity, monotonicity); add a `@eqx.filter_jit` smoke test for every new pricing fn; add a QuantLib comparison when a QL counterpart exists.

## Layout
- Source: `valax/{core,dates,curves,surfaces,instruments,models,calibration,market,greeks,risk,portfolio}/` (domain packages) and `valax/pricing/{analytic,mc,pde,lattice}/` (pricing impls grouped by numerical method).
