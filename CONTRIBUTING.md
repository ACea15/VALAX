# Contributing to VALAX

Thanks for your interest! VALAX is a JAX-native quantitative finance valuation
engine. This guide explains how to set up a development environment, the
**hard architectural constraints** imposed by JAX, the test expectations every
PR must meet, and the review checklist.

## Table of contents

1. [Development setup](#1-development-setup)
2. [Architectural rules](#2-architectural-rules)
3. [Code style](#3-code-style)
4. [Testing requirements](#4-testing-requirements)
5. [Documentation](#5-documentation)
6. [Pull request checklist](#6-pull-request-checklist)
7. [Reporting bugs and proposing features](#7-reporting-bugs-and-proposing-features)

---

## 1. Development setup

VALAX is developed against Python 3.13.12 in a `pyenv` virtualenv called `valax`.
Any Python ≥ 3.11 should work for development.

```bash
# Activate the dev environment.
pyenv activate valax            # or: python -m venv .venv && source .venv/bin/activate

# Install in editable mode with dev dependencies.
pip install -e ".[dev]"

# Optional: docs dependencies.
pip install -e ".[docs]"
```

### Common commands

```bash
# Full test suite (≈ a few seconds; the QL comparison tests may take longer).
pytest

# Single test file.
pytest tests/test_pricing/test_black_scholes.py

# Single test by name.
pytest tests/test_pricing/test_black_scholes.py::test_call_price -v

# Performance benchmarks.
pytest --benchmark-only

# Build docs (strict mode catches broken links and unrendered references).
mkdocs build --strict

# Serve docs locally (http://127.0.0.1:8000).
mkdocs serve
```

### QuantLib comparison tests

`tests/test_quantlib_comparison/` requires the optional `QuantLib` Python package.
It is **not** in `[dev]` because we do not want a hard dependency on it for the
core test suite. Install it separately if you want to run the comparison tests:

```bash
pip install QuantLib
```

---

## 2. Architectural rules

These are **hard constraints** imposed by JAX. Code that violates them cannot be
`jit`-compiled, `grad`-differentiated, or `vmap`-vectorized — and therefore
cannot be merged.

### 2.1 No mutable state

JAX requires pure functions. That means:

- **No** `self.cache`, no `__setattr__`, no instance attributes that change
  after construction.
- **No** Python lists, dicts, or sets that are mutated during computation.
- **No** module-level variables that pricing functions depend on.

If you need shared state, pass it explicitly through the function signature.
If you need a constant, make it a module-level constant assigned exactly once.

### 2.2 Every data structure is an `equinox.Module`

All structured data — instruments, curves, models, market data, risk ladders —
must subclass `equinox.Module`. This makes them frozen dataclasses that are
automatically registered as JAX pytrees.

```python
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array


class EuropeanOption(eqx.Module):
    strike: Float[Array, ""]
    expiry: Float[Array, ""]
    is_call: bool = eqx.field(static=True)  # not differentiable → static
```

Mark non-differentiable fields with `eqx.field(static=True)`. Typical examples:
`is_call`, `interp_method`, day-count strings, integer counts (`frequency`,
`n_steps`).

### 2.3 Instruments are data-only

Instruments **describe a contract** — strike, expiry, notional, schedules. They
**carry no pricing logic**. Pricing is a separate pure function:

```python
# Yes
price = black_scholes_price(option, spot, vol, rate, dividend)

# No (this is the QuantLib pattern, not the VALAX pattern)
option.set_pricing_engine(engine)
price = option.npv()
```

Composability over inheritance: any pricing function automatically works with
`jax.grad`, `jax.vmap`, and `jax.jit` without needing engine subclasses or
visitor patterns.

### 2.4 Greeks come from autodiff, not finite differences

If you find yourself writing `(price(x + h) - price(x - h)) / (2 * h)`, stop.
Use `jax.grad` or `jax.jacobian`. The only exception is sanity-check tests
(see §4.2).

For instruments with discontinuous payoffs (digitals, hard barriers), smooth the
payoff with a sigmoid (see `valax/pricing/mc/payoffs.py`) so autodiff still
gives meaningful sensitivities. The likelihood-ratio method is an alternative
but is not yet implemented.

### 2.5 Dates are integer ordinals inside JAX-traced code

JAX cannot trace through `datetime` objects. Inside any function that may be
`jit`/`grad`/`vmap`'d, dates must be integer ordinals (days since epoch) stored
in JAX arrays. Use the helpers in `valax/dates/` for conversion at the user
boundary.

### 2.6 No scipy inside JIT-traced code

`scipy.optimize`, `scipy.linalg`, etc. are not JAX-traceable. Use the
JAX-native equivalents:

| Need | Use |
|------|-----|
| Root finding | `optimistix.root_find` (Newton) |
| Least squares | `optimistix.least_squares` (Levenberg-Marquardt) |
| Minimization | `optimistix.minimise` (BFGS, etc.) or `optax` for SGD-style |
| Linear solves | `lineax` (tridiagonal, dense, structured) |
| SDE simulation | `diffrax` (do not hand-roll Euler-Maruyama) |
| Distributions / special functions | `jax.scipy.stats` and `jax.scipy.special` |

### 2.7 No class hierarchies for dispatch

Don't create `PricingEngine` base classes. If you need polymorphic behaviour:

- Prefer plain functions with explicit arguments.
- Use `functools.singledispatch` if dispatch on Python type is genuinely needed.
- Pattern-match on the module type if you must.

---

## 3. Code style

### 3.1 Type annotations with `jaxtyping`

Annotate every array argument with shape and dtype:

```python
from jaxtyping import Float, Int
from jax import Array

def gbm_paths(
    spot: Float[Array, ""],
    vol: Float[Array, ""],
    rate: Float[Array, ""],
    n_paths: int,
    n_steps: int,
) -> Float[Array, "n_paths n_steps"]: ...
```

Use **named** dimensions whenever possible: `Float[Array, "n_paths n_steps"]`,
not `Float[Array, "a b"]`. Scalars are `Float[Array, ""]`.

### 3.2 Use `equinox.filter_jit` and `equinox.filter_grad`

When operating on `eqx.Module` instances that contain static fields, use the
`filter_*` variants — they handle the static/dynamic split automatically:

```python
import equinox as eqx

@eqx.filter_jit
def fast_price(option, spot, vol, rate, dividend): ...

# rather than @jax.jit, which will choke on static fields.
```

### 3.3 Pricing function signature

```python
def price(instrument, *market_args) -> Float[Array, ""]: ...
```

- Pure (no side effects, no globals, no `print` inside JIT-traced code).
- Differentiable (smooth approximations for discontinuous payoffs).
- Returns a scalar JAX array.

### 3.4 Tests mirror source layout

Every module under `valax/X/Y.py` has a corresponding test file at
`tests/test_X/test_Y.py`. New modules need new test files in the right place.

### 3.5 Docstrings

Use Google-style docstrings. Math goes in LaTeX inside reStructuredText
`.. math::` blocks (these render correctly in mkdocs and in editor hover):

```python
def hw_bond_price(model, r, t, T):
    """Zero-coupon bond price under Hull-White given short rate *r* at time *t*.

    .. math::

        P(t, T \\mid r) = A(t, T)\\,e^{-B(t, T)\\,r}

    Args:
        model: Hull-White model.
        r: Current short rate.
        t: Current time in year fractions.
        T: Bond maturity time in year fractions.

    Returns:
        Zero-coupon bond price :math:`P(t, T)`.
    """
```

---

## 4. Testing requirements

Every PR must include tests. The expectations vary by feature type.

### 4.1 Closed-form solutions

Validate against a known closed-form benchmark to **machine precision**
(absolute tolerance `1e-10` or better):

```python
def test_call_price_at_atm():
    # Hand-computed reference value from a reliable source.
    expected = 10.4506
    actual = black_scholes_price(option, spot, vol, rate, dividend)
    assert jnp.isclose(actual, expected, atol=1e-4)
```

### 4.2 Greeks

Validate autodiff Greeks against:

1. **Closed-form formulas** when they exist (e.g., Black-Scholes delta = $\Phi(d_1)$).
2. **Finite differences** as a sanity check, with relative tolerance `1e-4` or
   tighter — the autodiff and the FD value should agree.

```python
delta_ad = jax.grad(price_fn, argnums=1)(option, spot, vol, rate, div)
delta_fd = (price_fn(option, spot + 1e-4, vol, rate, div) -
            price_fn(option, spot - 1e-4, vol, rate, div)) / 2e-4
assert jnp.isclose(delta_ad, delta_fd, rtol=1e-4)
```

### 4.3 Monte Carlo

Assert convergence within **2 standard errors** of the analytical solution:

```python
mc_price, mc_se = mc_price_with_stderr(option, spot, model, config, key)
assert abs(mc_price - analytical_price) < 2 * mc_se
```

### 4.4 Property-based tests with `hypothesis`

For invariants that should hold across all valid inputs (put-call parity,
positivity, monotonicity), use `hypothesis`:

```python
from hypothesis import given, strategies as st

@given(spot=st.floats(50, 150), vol=st.floats(0.05, 0.8))
def test_put_call_parity(spot, vol):
    # C - P = S*exp(-q*T) - K*exp(-r*T)
    ...
```

### 4.5 QuantLib comparison

For any new pricing function that has a QuantLib counterpart, add a comparison
test in `tests/test_quantlib_comparison/`. The tolerance depends on the method
but is typically `1e-6` (analytic vs. analytic) to a few bps (MC, PDE).

### 4.6 JIT-compatibility

Every pricing function should be tested under `jax.jit` to catch tracing
violations. A simple smoke test:

```python
@eqx.filter_jit
def jit_price(option, spot, vol, rate, div):
    return black_scholes_price(option, spot, vol, rate, div)

# If this works, the function is JIT-compatible.
jit_price(option, spot, vol, rate, div)
```

---

## 5. Documentation

### 5.1 What goes where

| Type of change | Doc to update |
|----------------|---------------|
| New theory / formula | `docs/theory.md` |
| New instrument | `docs/guide/<asset-class>.md` and `docs/api/instruments.md` |
| New pricing function | `docs/guide/<method>.md` and `docs/api/pricing.md` |
| New model | `docs/guide/<area>.md` and `docs/api/models.md` |
| New runnable demo | `examples/` and `docs/examples.md` |
| Roadmap status change | `docs/roadmap.md` (move from 🟠/🟡 to ✅) |
| Release-worthy feature | `CHANGELOG.md` under `[Unreleased]` |

### 5.2 Build docs locally before opening a PR

```bash
mkdocs build --strict
```

`--strict` fails on broken cross-references and missing files. Always run it
before pushing.

### 5.3 Cite implementation paths

In theory and guide pages, cite the actual VALAX module path so readers can
trace each formula to its implementation:

> ```
> **VALAX implementation:** `valax/pricing/analytic/spread.py`
> (`margrabe_price`, `kirk_price`).
> ```

---

## 6. Pull request checklist

Before requesting review, confirm:

- [ ] **Code follows §2 architectural rules** (no mutable state, equinox modules,
  pure functions, autodiff Greeks).
- [ ] **Type annotations** with `jaxtyping` on all array arguments.
- [ ] **Docstrings** on every public function and class.
- [ ] **Tests** added under `tests/` mirroring the source layout, covering:
  closed-form (if applicable), Greeks vs FD, JIT-compat, and edge cases.
- [ ] **QuantLib comparison test** added if there's a QL counterpart.
- [ ] **Docs updated** per §5.1.
- [ ] **`mkdocs build --strict`** passes locally.
- [ ] **`pytest`** passes locally.
- [ ] **`CHANGELOG.md`** updated under `[Unreleased]`.
- [ ] **Roadmap entry** updated if this completes a planned item.
- [ ] **Commit messages** follow the existing convention: imperative, descriptive,
  e.g. `Add spread option pricing: Margrabe and Kirk's approximation`.

---

## 7. Reporting bugs and proposing features

- **Bugs**: open a GitHub issue with a minimal reproducer (instrument
  construction + pricing call + observed vs. expected output). If the bug
  involves a JAX transform (`jit`/`grad`/`vmap`), include the failing transform
  call.
- **Features**: open a GitHub issue describing the use case. Cross-reference
  the [Roadmap](docs/roadmap.md) — if the feature is already planned, comment
  on the existing tracking item; if not, propose a new one with an indication
  of which Tier / Priority it falls under.

Thanks for contributing to VALAX!
