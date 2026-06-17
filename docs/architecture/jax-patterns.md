# JAX Patterns in VALAX

VALAX is JAX-native end to end. Every model, instrument, curve, surface, payoff,
and pricer is a `jax.jit`-compilable, `jax.grad`-differentiable,
`jax.vmap`-vectorisable function. This document is the canonical reference for
the JAX idioms that recur throughout the codebase — what they are, why they
work, and which file:line shows each one in production use.

It is deliberately opinionated. JAX has many ways to express the same thing,
and VALAX picks one per situation and uses it consistently so that contributors
can read across modules without re-deriving the conventions every time.

The intended audience is anyone touching VALAX source: contributors writing new
pricers, reviewers checking that a PR doesn't accidentally break autodiff, and
quants who want to understand why the code is shaped the way it is.

---

## §1 — Trace time vs runtime: the foundational distinction

The single mental model that unlocks everything else.

When you call a JAX-transformed function (`jax.jit`, `jax.grad`, `jax.vmap`),
JAX runs your Python code **once** with abstract placeholders standing in for
the array arguments. This is *tracing*. The result of tracing is a *jaxpr* — a
small, typed IR — that gets compiled to XLA and executed at *runtime*.

Two kinds of values flow through your code:

| Value kind | Resolved | Visible to JAX | Examples |
|---|---|---|---|
| **Python values** | At trace time | Used to *select* control flow | strings, Python ints, Python bools, `eqx.field(static=True)` leaves, dict keys, tuple structures, dataclass type identities |
| **JAX abstract values** | Inside the jaxpr at runtime | Become tensors in XLA | `jnp.ndarray`, `Float[Array, ""]`, anything coming out of `jnp.` |

The rule is short:

> Python `if`, `for`, `isinstance` are fine on Python values.
> Python `if`, `for`, `isinstance` on JAX abstract values raise
> `ConcretizationTypeError`.

When you see a Python `if` or `isinstance` in VALAX, you are looking at a
*trace-time* branch — it runs once, in Python, before any JAX value exists.
The unselected branch is not compiled. The selected branch becomes part of the
jaxpr. JAX never sees the condition.

When the branch genuinely depends on a runtime value (a comparison of two
arrays, a stopping criterion, an early-exit), the right tool is `lax.cond`,
`lax.switch`, `lax.scan`, `lax.fori_loop`, or `lax.while_loop`. These get
compiled into XLA control flow primitives.

The rest of this document is a catalogue of patterns that exploit this
distinction.

---

## §2 — Static dispatch patterns

VALAX uses three forms of trace-time static dispatch. All three are
mechanically equivalent — Python decides at trace time which branch enters the
jaxpr. They differ in *what gets dispatched on* and where the choice lives in
the API.

### 2.1 Python `if` on a kwarg literal

The most direct form. A keyword argument is a Python string (or Python int /
bool); the Python interpreter runs the `if` before tracing starts.

**Example.** `valax/pricing/mc/local_vol_paths.py:179,197`:

```python
def generate_local_vol_paths(model, spot, T, n_steps, n_paths, key,
                              *, scheme: Literal["midpoint_euler", "milstein"] = "midpoint_euler"):
    ...
    if scheme == "midpoint_euler":
        def step(carry, scan_input):
            ...  # cheaper, weak-order-1 step
    else:  # scheme == "milstein"
        _sigma_and_grad_scalar = jax.value_and_grad(
            lambda kk, tt: dupire_local_vol(surface, kk, tt), argnums=0,
        )
        def step(carry, scan_input):
            ...  # strong-order-1 step using value_and_grad
    ...
    _, log_S_seq = jax.lax.scan(step, log_S0, (times, keys))
```

Only the selected `step` closure enters the `lax.scan` body. The other branch
is never traced. Under `jax.jit`, each distinct `scheme` value gets its own
compilation cache entry — Python strings are hashable and JAX treats them as
static cache keys by default.

**Why not `jnp.where` or `lax.cond` here?** Both alternatives would *compile
both* branches and select between their results at runtime — defeating the
entire point of having a "cheap default scheme" and an "opt-in expensive
scheme." Python `if` produces strictly less XLA code.

### 2.2 `isinstance(model, …)` on a pytree type

Same mechanism, but the trace-time information is the **Python type** of the
input model rather than a kwarg literal.

**Example.** `valax/pricing/mc/engine.py:55-65`:

```python
def mc_price(option, spot, model, config, key, payoff_fn=european_payoff):
    T = option.expiry
    if isinstance(model, HestonModel):
        paths, _ = generate_heston_paths(model, spot, T, config.n_steps, config.n_paths, key)
        rate = model.rate
    elif isinstance(model, LocalVolModel):
        paths = generate_local_vol_paths(model, spot, T, config.n_steps, config.n_paths, key)
        rate = model.rate
    else:
        paths = generate_gbm_paths(model, spot, T, config.n_steps, config.n_paths, key)
        rate = model.rate
    ...
```

`type(model)` is a Python type object — fully concrete at trace time. The
`isinstance` check runs in pure Python; JAX never sees it. The selected
generator function gets traced into the jaxpr, the others don't.

The same pattern appears in `valax/pricing/mc/recipes.py:126,130` inside
`_equity_paths`, which is the helper the unified MC dispatcher uses for every
single-asset equity recipe.

**Why this works for autodiff.** `jax.grad(mc_price)(model)` will traverse only
the differentiable leaves of `model` (the parameters); the *type* of the
container pytree is structural metadata, not a differentiable input. JAX's
pytree machinery (via `equinox`) treats the type identity as compilation-cache
metadata exactly like the static fields in §2.3.

### 2.3 `eqx.field(static=True)` for compilation-cache metadata

When a leaf of a pytree is conceptually a configuration choice rather than a
differentiable value, marking it static moves it from "JAX abstract value" to
"compilation cache key." It is then available to Python branching inside
methods of the pytree.

**Examples** (`grep -rn 'static=True' valax/` finds 21 files; the most
characteristic):

```python
# valax/pricing/mc/engine.py:23-27 — MC configuration
class MCConfig(eqx.Module):
    n_paths: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)

# valax/pricing/mc/bermudan.py:21-29 — LSM regression configuration
class LSMConfig(eqx.Module):
    poly_degree: int = eqx.field(static=True, default=3)

# valax/instruments/options.py:19 — payoff direction
class EuropeanOption(eqx.Module):
    strike: Float[Array, ""]
    expiry: Float[Array, ""]
    is_call: bool = eqx.field(static=True, default=True)
```

`n_paths` and `n_steps` are static because they determine the *shape* of the
sampled path tensor — shapes must be known at trace time to allocate XLA
buffers. `is_call` is static because the payoff branches on it via a Python
`if`. `poly_degree` is static because it determines the polynomial basis
dimension in the LSM regression.

These fields are still part of the pytree (so they round-trip cleanly through
`jax.tree_util` operations), but JAX treats them like keyword arguments: they
are part of the compilation cache key, not part of the traced computation. A
different `MCConfig(n_paths=10_000, n_steps=50)` triggers a different XLA
compilation; a different `n_paths=20_000` triggers another. For a given static
config, every call with the same shapes/dtypes hits the cache.

**Anti-pattern.** Marking a *differentiable* value static (e.g.
`spot: float = eqx.field(static=True)`) silently breaks `jax.grad` — the
gradient w.r.t. that field is meaningless because JAX never traced through it.
The discipline is: parameters → traced leaves; configuration → static fields.

### 2.4 When you genuinely need runtime control flow

If the branch condition is a JAX abstract value (a comparison of two arrays, a
flag computed inside the function), Python `if` is wrong and will raise
`ConcretizationTypeError` inside a transformed function. Pick the smallest
`jax.lax` primitive that fits:

| Primitive | Use when |
|---|---|
| `lax.cond(pred, true_fn, false_fn, *operands)` | Two-way branch on a traced bool — both branches compile, runtime picks one |
| `lax.switch(index, branches, *operands)` | Multi-way branch on a traced integer index |
| `jnp.where(cond, x, y)` | Elementwise selection — **both** `x` and `y` are computed everywhere, then masked. See §3 for the autodiff trap |
| `lax.scan(f, init, xs)` | Bounded loop with carry state — the workhorse, see `valax/pricing/mc/paths.py:223` (Heston QE), `valax/pricing/mc/local_vol_paths.py:217` (LV stepper) |
| `lax.fori_loop(lower, upper, body, init)` | Bounded loop without inputs — see `valax/pricing/mc/bermudan.py` (LSM backward induction) |
| `lax.while_loop(cond, body, init)` | Unbounded loop — **no autodiff support**; use only when convergence depends on a runtime condition |

The cost ordering, lowest-to-highest XLA footprint, is roughly: Python `if` (one
branch in the graph) < `lax.cond` (both branches in the graph, runtime select)
< `jnp.where` (both expressions computed everywhere, masked) ≈ `lax.switch`.
Prefer Python `if` whenever the condition resolves at trace time.

---

## §3 — The "double where" autodiff-safety pattern

The most subtle JAX issue, and the source of most "my gradient is NaN"
debugging sessions. VALAX uses the double-where pattern in at least four
places; understanding it once unlocks all of them.

### 3.1 The trap

`jnp.where(cond, safe, expr)` looks like it should mean "compute `expr` only
where `cond` is True, otherwise return `safe`." For the *value*, that is
correct. For the *gradient*, it is not.

JAX computes gradients by reverse-mode autodiff over the jaxpr. `jnp.where`
desugars into a primitive that computes *both* `safe` and `expr` everywhere
and then masks. If `expr` contains a divide-by-zero or `log(0)` or
`sqrt(negative)` anywhere — even at points where `cond` is False and the value
would be masked out — the gradient at those points is `NaN` or `Inf`, and that
propagates back through the entire backward pass.

A canonical broken pattern:

```python
# BROKEN: gradient at z=0 is NaN even though the value is "safe"
z_over_x = jnp.where(jnp.abs(z) < 1e-7, 1.0, z / x_z)
```

At `z = 0`, the value is `1.0` (fine). The gradient backward through `z / x_z`
sees a division by `x_z = 0` (because `x_z` is also zero at the ATM SABR
limit), produces `Inf`/`NaN`, and even though `jnp.where` masks the value, it
does *not* mask the gradient flowing into the unselected branch.

### 3.2 The fix: double-where

Substitute safe inputs *into* the problematic expression before the operation,
so the unselected branch evaluates to a benign value with benign gradient. Then
the outer `jnp.where` only picks between two healthy quantities.

**Real example — SABR's `z / chi(z)` ratio at the ATM limit**
(`valax/pricing/analytic/sabr.py:62-65`):

```python
is_small = jnp.abs(z) < 1e-7
safe_z = jnp.where(is_small, 1.0, z)        # inner: replace z with 1 in the divide
safe_x = jnp.where(is_small, 1.0, x_z)      # inner: replace x_z with 1 in the divide
z_over_x = jnp.where(is_small, 1.0, safe_z / safe_x)  # outer: pick the answer
```

The inner `safe_z` and `safe_x` substitute `1.0` *before* the division runs.
The division `safe_z / safe_x = 1.0 / 1.0` is finite with finite gradient. The
outer `jnp.where` then picks between the analytic ATM limit (`1.0`) and the
genuine ratio. The gradient backward through `safe_z / safe_x` is well-defined
everywhere; the outer `jnp.where` masks the value cleanly.

**Real example — Heston COS payoff coefficients at `k = 0`**
(`valax/pricing/analytic/heston.py:163,186-187`):

The `psi_k` integral has a removable singularity at `k = 0`. The fix:

```python
# valax/pricing/analytic/heston.py:186-187
safe_k = jnp.where(k == 0, 1.0, k)
psi_nonzero = (b - a) / (safe_k * jnp.pi) * (sin_d - sin_c)
# ... outer where selects between psi_nonzero and the k=0 analytic limit
```

The same idiom: substitute a benign `safe_k` into the divisor before the
divide, so the unselected branch is never an `Inf` waiting to poison the
gradient.

**Real example — `kappa·dt → 0` Taylor fallback in Andersen QE**
(`valax/pricing/mc/paths.py:143-147`):

```python
kdt_safe = jnp.where(jnp.abs(kdt) > 1e-8, kdt, 1.0)
taylor = 1.0 - 0.5 * kdt + kdt * kdt / 6.0
one_minus_E_over_kappa = dt * jnp.where(
    jnp.abs(kdt) > 1e-8,
    one_minus_E / kdt_safe,    # safe denominator
    taylor,                    # Taylor expansion near zero
)
```

Same pattern: `kdt_safe` ensures the division `one_minus_E / kdt_safe` is
always finite, then the outer `where` picks between the exact ratio (when
`kdt` is non-trivial) and the truncated Taylor series (when `kdt` is small).

### 3.3 The rule

> If a `jnp.where` branch contains a divide, `log`, `sqrt`, `pow`, or any other
> operation that can produce a non-finite value at the unselected points,
> substitute the inputs to that operation with their safe values *first*. The
> outer `jnp.where` should only pick between two values that are both
> mathematically well-defined and have well-defined gradients.

The mnemonic: *the gradient sees everywhere `where` doesn't*.

---

## §4 — Numerical guards that don't poison autodiff

Companion to §3. The double-where pattern handles divide-by-zero-style issues;
this section is about the lighter-weight guards that show up in MC steppers and
calibration routines.

### 4.1 `jnp.maximum(x, tiny)` is smooth and safe

A simple lower clamp:

```python
xi_safe = jnp.where(jnp.abs(xi) > tiny, xi, tiny)   # paths.py:155
# or equivalently
w_safe = jnp.maximum(w, 1e-10)                        # dupire.py:139
```

Both produce the same value. `jnp.maximum` is mildly preferable because it has
a single, locally smooth gradient (1 on the unclamped side, 0 on the clamped
side); `jnp.where` produces a *discontinuous* selection that is technically
non-differentiable at the boundary but JAX defines the gradient by branch
selection so it works in practice.

The danger is the same as §3: if the clamp's argument was *itself* derived
from a problematic computation, you still get NaN gradients. The clamp guards
the **output**, not the intermediate steps. If `w = something / kappa` and
`kappa = 0` is in scope, `jnp.maximum(w, 1e-10)` does not save you — the
division already produced an `Inf`.

### 4.2 The deliberate non-guard: Dupire's denominator

The Dupire local-vol formula has a g-function in the denominator whose sign
detects butterfly arbitrage. VALAX deliberately does **not** clamp it
(`valax/pricing/analytic/dupire.py:40-41,155`):

```python
# numerator clamped (calendar-arb floating-point noise)
numerator = jnp.maximum(dw_dT, 0.0)

# denominator left UNCLAMPED — non-positive denominator is an arb signal
g = 1.0 - (log_moneyness / w_safe) * dw_dk + ...

sigma2 = numerator / g
return jnp.sqrt(sigma2)   # NaN if g <= 0 — intentional diagnostic
```

A NaN at the output here is not a bug. It is the formula telling the caller
that their input vol surface contains a butterfly-arbitrage violation at the
queried `(k, T)`. Clamping `g` would silently produce a wrong answer and lose
the diagnostic. The unit test `TestArbitrageDetection` in `test_dupire.py`
verifies that the NaN does in fact propagate when fed a pathological SVI
slice.

The lesson is that not every numerical issue deserves a guard. A guard should
exist to handle *floating-point noise* in a region where the math is
analytically fine. It should not exist to mask *genuine* numerical problems
that signal upstream issues.

### 4.3 Avoid Python `if` on traced values for "numerical safety"

A very common anti-pattern from quant code ported from C++ or NumPy:

```python
# BROKEN inside any JAX-transformed function
if jnp.any(v < 0):
    v = jnp.maximum(v, 0)
```

The Python `if` on a traced array is `ConcretizationTypeError`. The fix is
simpler than people expect — just always clamp:

```python
v = jnp.maximum(v, 0)   # unconditional, zero overhead, autodiff-clean
```

`jnp.maximum(v, 0)` is a single XLA op. It is faster than the broken pattern
even if it were legal, and it works under `jit`, `grad`, and `vmap` without
qualification.

---

## §5 — Pytrees and equinox

VALAX uses `equinox` as its pytree library. The contract is:

1. Every model, instrument, curve, surface, payoff config, etc. is an
   `eqx.Module`.
2. Differentiable leaves are typed `Float[Array, ""]` or
   `Float[Array, " n_something"]` via `jaxtyping`.
3. Configuration / shape-determining leaves are marked
   `eqx.field(static=True)`.
4. Field declaration order is the pytree flatten order; it is documented in
   docstrings and treated as a stable contract for `eqx.tree_at` consumers.

### 5.1 The field-order contract

`equinox` pytrees flatten in the order fields are declared. Downstream code
that uses `eqx.tree_at(lambda m: m.<field>, model, new_value)` relies on the
field position being stable. VALAX commits to this in docstrings:

```python
# valax/models/heston.py
class HestonModel(eqx.Module):
    """Heston stochastic volatility model.

    Field order (stable contract for ``eqx.tree_at`` consumers):
        v0, kappa, theta, xi, rho, rate, dividend
    """
    v0: Float[Array, ""]
    kappa: Float[Array, ""]
    theta: Float[Array, ""]
    xi: Float[Array, ""]
    rho: Float[Array, ""]
    rate: Float[Array, ""]
    dividend: Float[Array, ""]
```

Same convention for `LocalVolModel`, `SVISlice`, `SVIVolSurface`,
`SABRVolSurface`, `BlackScholesModel`, `LMMModel`, etc. — every Module
in `valax/models/` and `valax/surfaces/` documents its field order
explicitly.

### 5.2 Functional updates via `eqx.tree_at`

Pytrees are immutable. To "change" a field, build a new pytree with the change
applied:

```python
# Bump only v0, keep everything else
bumped_model = eqx.tree_at(lambda m: m.v0, heston_model, heston_model.v0 + 0.001)
```

This is the idiom used throughout calibration (`valax/calibration/heston.py`,
`valax/surfaces/svi.py::calibrate_svi_slice`) and bumping for finite-difference
Greeks (which exist as the QuantLib-comparison sanity check against autodiff
Greeks — production code uses `jax.grad` directly).

### 5.3 `jax.grad` traverses only the differentiable leaves

When you call `jax.grad(f)(model)`, JAX returns a *pytree of the same
structure* whose leaves are the gradients at the differentiable positions.
Static fields are passed through unchanged. Non-array leaves (like a `bool`
that wasn't marked static) raise an error.

Concretely, for `jax.grad(price_fn)(heston_model)` you get back a
`HestonModel` whose `v0`, `kappa`, ..., `dividend` are the seven first-order
parameter sensitivities of the price. No keyword juggling, no manual
unpacking — the gradient pytree has the same shape as the input pytree, which
is why every Greek in VALAX is just one `jax.grad` call.

---

## §6 — JIT cache semantics and autodiff-through-autodiff

### 6.1 The compilation cache key

`jax.jit(fn)` caches compiled XLA programs keyed on:

1. The function identity (`id(fn)`).
2. The *abstract* shape and dtype of every traced input — concrete values are
   not part of the key (that's the whole point of jit).
3. The *concrete* values of every static input — kwarg literals, Python ints
   marked static, `eqx.field(static=True)` leaves.
4. The pytree structure of every input (which fields are present, in what
   order).

A new entry in the cache means a new XLA compilation, which takes
milliseconds-to-seconds. A cache hit is sub-microsecond. The practical
implication:

> Vary your traced inputs all you want (spot, vol, rate — JAX caches once and
> reuses). Vary your static inputs sparingly (every distinct `n_steps`
> recompiles).

For functions where the user might toggle a configuration kwarg in a hot loop
(e.g. the `scheme="midpoint_euler" | "milstein"` choice on
`generate_local_vol_paths`), make the staticness explicit with
`static_argnames` so the recompilation behaviour is documented and
predictable:

```python
jit_fn = jax.jit(generate_local_vol_paths,
                 static_argnames=("scheme", "n_steps", "n_paths"))
```

Without `static_argnames`, JAX's auto-static-detection still works for hashable
types (strings, ints) but the behaviour becomes implicit. The explicit form is
self-documenting and protects against the function signature changing in a way
that flips a kwarg from static-by-default to traced.

### 6.2 Autodiff is composable: nested `jax.grad`, `jax.value_and_grad`, and friends

JAX autodiff is built around composable transformations. Anything that works
with `jax.grad` can be combined arbitrarily with `jax.jit`, `jax.vmap`,
`jax.value_and_grad`, and other transforms — including being differentiated
again.

**Concrete example: Milstein step uses autodiff inside autodiff inside scan.**
The Milstein scheme on `generate_local_vol_paths`
(`valax/pricing/mc/local_vol_paths.py:200-205`) needs `∂σ_loc / ∂k` at every
time step:

```python
_sigma_and_grad_scalar = jax.value_and_grad(
    lambda kk, tt: dupire_local_vol(surface, kk, tt),
    argnums=0,
)

def step(carry, scan_input):
    ...
    sigma, dsigma_dk = jax.vmap(lambda kk: _sigma_and_grad_scalar(kk, t_n))(k)
    ...
    return log_S_next, log_S_next

_, log_S_seq = jax.lax.scan(step, log_S0, (times, keys))
```

This composes three things:

1. `dupire_local_vol` itself uses `jax.grad` internally to compute
   `∂w/∂k`, `∂²w/∂k²`, `∂w/∂T` from the surface.
2. `jax.value_and_grad(dupire_local_vol, argnums=0)` differentiates *through*
   that — so we're taking a `jax.grad` of a function that already contains
   `jax.grad`. JAX handles this cleanly.
3. The whole thing sits inside `lax.scan`, which sits inside `jax.lax.scan`,
   and may itself be wrapped in `jax.grad(price_fn)` by the caller for vega
   sensitivities.

The contract is: every JAX transform is a pure function on functions, with no
hidden state. They commute (`jit ∘ grad = grad ∘ jit`), they nest, and they
expose the same API at every nesting depth. This is the property that makes
`jax.grad(monte_carlo_price)(svi_surface)` work as a single expression and
return a vega-bucketed gradient pytree — every layer is autodiff-clean by
construction.

### 6.3 `jax.value_and_grad` vs `jax.grad`

When you need both the value and its gradient (Newton solvers, MC steppers
that need `σ` and `∂σ/∂k`), `jax.value_and_grad` is strictly preferable to
calling the function twice or computing both separately. JAX shares the
forward pass between the two outputs at zero extra cost.

VALAX uses `jax.value_and_grad`:

- In the Milstein step (above) — one combined call per path per step.
- In Newton-style implied-vol inversion
  (`valax/pricing/analytic/black_scholes.py::black_scholes_implied_vol`) —
  one call per Newton iteration gets you the price and the vega.

---

## §7 — Common breakage modes (quick reference)

The most-failure-prone JAX patterns and their fixes. Bookmark this section; it
catches 80% of real bugs.

| Error / symptom | What you probably wrote | Fix |
|---|---|---|
| `ConcretizationTypeError` | `if some_jax_array > 0:` inside a jit/grad'd function | Use `lax.cond` if the branch must happen at runtime, or make the value static via `static_argnames` / `eqx.field(static=True)` if it should resolve at trace time |
| `TracerArrayConversionError` | `np.asarray(some_traced_value)` or `float(traced)` inside a transformed function | Stay inside `jnp.` for arithmetic; convert to host arrays only *after* the outermost transform returns |
| Silent NaN coming out of `jax.grad` | `jnp.where(cond, safe, problem_expr)` where `problem_expr` is non-finite at the masked points | Apply the double-where pattern from §3 — substitute safe values into `problem_expr`'s inputs first |
| Function recompiles on every call | `jax.jit(fn)`, then varying a Python kwarg in a hot loop | Add `static_argnames=("the_kwarg",)` to make the staticness explicit; vary only the JAX array inputs to hit the cache |
| `TypeError: ... is not a valid JAX type` from a pytree leaf | A non-array Python value (bool, int, str) sitting as a regular `eqx.Module` field without `static=True` | Mark it `eqx.field(static=True)` |
| `jax.grad` returns zero w.r.t. a parameter you expected it to depend on | The parameter was marked `static=True` — JAX never traced through it | Remove the `static=True` marker; if shape determinism is needed, factor the shape out into a separate static config |
| `jax.grad` fails with "input has integer dtype" | `jax.grad` requires floating-point inputs | Cast to float explicitly: `jnp.asarray(x, dtype=jnp.float64)` |
| `jax.vmap` returns wrong shape | `in_axes` defaults to `0` (leading axis); your batch dim is somewhere else | Pass `in_axes=...` explicitly. For pytree inputs, `in_axes=HestonModel(0, None, None, ...)` or use `None` to broadcast |
| Gradient explodes through a `lax.scan` of many steps | Recursive forward-mode accumulation; vanishing/exploding gradient through long sequences | Use `jax.checkpoint` (also known as `jax.remat`) inside the scan body to trade memory for stability |
| `jnp.einsum` returns the wrong values | Subscripts use the wrong axis order, or `jnp.einsum` was called on a list (gets cast to array of objects) | Pass arrays explicitly with `,`-separated subscripts; if in doubt, write it out as an explicit `jnp.sum(a * b, axis=...)` first |

---

## §8 — Further reading

- **The official JAX documentation** is the canonical reference for primitive
  semantics. Particularly worth reading:
  [How to think in JAX](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html),
  [JAX Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html),
  and the [Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html).
- **Equinox documentation** —
  [`eqx.Module`](https://docs.kidger.site/equinox/api/module/module/) and
  [`eqx.field`](https://docs.kidger.site/equinox/api/module/advanced_fields/)
  cover the static-field machinery.
- **The double-where idiom** is folklore-level JAX wisdom; for a careful
  treatment see the
  [Common Gotchas page](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#NaN-in-grad).

In-tree references that walk through specific applications:

- **MC architecture overview** — [`architecture/overview.md`](overview.md)
- **MC dispatcher design** — [`architecture/mc-curves-2.md`](mc-curves-2.md)
- **Why VALAX is JAX-native** — [`design-rationale.md`](../design-rationale.md)
  §3 and §4 connect the architectural commitment to JAX with the specific
  pain points it lets VALAX skip.
