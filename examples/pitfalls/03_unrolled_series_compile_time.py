# %% [markdown]
# # Pitfall: a correct, fast formula whose *traced graph* is enormous
#
# **SYMPTOM.** A per-trade G2++ Bermudan with a piecewise-constant σ(t) priced
# in milliseconds and every value and gradient test passed — but the first
# quote-risk call through the 2-D PDE sat for **seven minutes** in XLA
# compilation (427 s, against 2 s for the same trade at constant vol). Nothing
# was wrong with any number. The run-time benchmark could not see it: compiled,
# the piecewise price ran in 2.8 ms against 2.6 ms.
#
# **ROOT CAUSE.** JAX traces Python. A `for` loop over a static range is
# *unrolled* into the graph, so the cost of a formula at trace time is its
# number of operations as written — not its arithmetic cost at run time. The
# piecewise variance term V(t, T) needs
#
# ```
# H(s) = ∫_0^s (1 - e^{-pu})(1 - e^{-qu}) du
# ```
#
# on short windows, where the closed form cancels catastrophically, so it uses
# the series `s Σ_n (-1)^{n+1}/n! Σ_j C(n-1, j) (ps)^j (qs)^{n-1-j}`. Written as
# that **double** sum it is ~1,000 operations per call; V called it **six** times
# (three rate pairs × two window ends), and the bond price calls V three times
# per node. The 3-bucket price lowered to **94,812** HLO lines against 4,645 at
# constant vol, and XLA's compile time grows faster than linearly in graph size
# — worst of all for the reverse-mode graph.
#
# **THE LESSON.** In traced code, *graph size is a cost metric of its own*,
# invisible to value tests and run-time benchmarks. Write series as recurrences
# (Horner / a running term), not as expanded sums; evaluate repeated
# elementwise calls as one broadcast call; and guard the size of hot primitives
# with a cheap lowering-only test.
#
# **KNOBS (search `KNOB`):** `N_SERIES`, `N_CALLS`, `TIME_GRAD_COMPILE`.

# %% Imports
import math
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ============================================================================
# 0. KNOBS
# ============================================================================
# %%
N_SERIES = 24            # KNOB: series terms (24 = VALAX's choice: 1e-20 at the switch)
N_CALLS = 48             # KNOB: independent call sites in the "model" below
TIME_GRAD_COMPILE = True  # KNOB: also time the reverse-mode compile (slower)

P, Q = 0.8, 0.03         # G2++ mean reversions a, b: a fast and a slow factor


# ============================================================================
# 1. TWO WRITINGS OF THE SAME SERIES
# ============================================================================
# %% Both sum the same non-negative terms; only the *shape of the program*
# differs.
def h_double_sum(p, q, s):
    """The original: the inner binomial sum written out term by term."""
    small = (p + q) * s < 1.0
    ss = jnp.where(small, s, 0.0)
    ps, qs = p * ss, q * ss
    p_pow, q_pow = [jnp.ones_like(ss)], [jnp.ones_like(ss)]
    for _ in range(1, N_SERIES):
        p_pow.append(p_pow[-1] * ps)
        q_pow.append(q_pow[-1] * qs)
    series = jnp.zeros_like(ss)
    for n in range(3, N_SERIES + 1):
        k = n - 1
        bracket = sum(math.comb(k, j) * p_pow[j] * q_pow[k - j] for j in range(1, k))
        series = series + ((1.0 if n % 2 else -1.0) / math.factorial(n)) * bracket
    closed = s + jnp.expm1(-p * s) / p + jnp.expm1(-q * s) / q - jnp.expm1(-(p + q) * s) / (p + q)
    return jnp.where(small, ss * series, closed)


def h_recurrence(p, q, s):
    """The fix: A_k = (x+y)^k - x^k - y^k by A_{k+1} = (x+y)A_k + xy(x^{k-1} + y^{k-1}).

    Every term is non-negative (no cancellation), and each series term costs a
    handful of operations instead of k.
    """
    small = (p + q) * s < 1.0
    ss = jnp.where(small, s, 0.0)
    x, y = p * ss, q * ss
    a_k, x_pow, y_pow = jnp.zeros_like(ss), jnp.ones_like(ss), jnp.ones_like(ss)
    series = jnp.zeros_like(ss)
    for n in range(3, N_SERIES + 1):
        a_k = (x + y) * a_k + x * y * (x_pow + y_pow)
        x_pow, y_pow = x_pow * x, y_pow * y
        series = series + ((1.0 if n % 2 else -1.0) / math.factorial(n)) * a_k
    closed = s + jnp.expm1(-p * s) / p + jnp.expm1(-q * s) / q - jnp.expm1(-(p + q) * s) / (p + q)
    return jnp.where(small, ss * series, closed)


s = jnp.geomspace(1e-6, 30.0, 200)
a, b = h_double_sum(P, Q, s), h_recurrence(P, Q, s)
print(f"max |recurrence / double sum - 1| over s in [1e-6, 30]: "
      f"{float(jnp.max(jnp.abs(b / a - 1))):.1e}")
print(">>> Same numbers, to round-off. A value test cannot tell them apart.")


# ============================================================================
# 2. WHAT A VALUE TEST AND A BENCHMARK SEE: NOTHING
# ============================================================================
# %%
def bench(f, *args, reps=200):
    jf = jax.jit(f)
    jax.block_until_ready(jf(*args))
    t0 = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(jf(*args))
    return (time.perf_counter() - t0) / reps * 1e6


print(f"\n  compiled run time per call: double sum {bench(lambda v: h_double_sum(P, Q, v), s):6.1f} µs,"
      f"  recurrence {bench(lambda v: h_recurrence(P, Q, v), s):6.1f} µs")
print(">>> Both cheap once compiled: XLA fuses either into a few elementwise")
print("    kernels. The difference is all at trace / compile time.")


# ============================================================================
# 3. WHAT THE GRAPH SEES
# ============================================================================
# %% Lowering is cheap (no XLA compile), so HLO line counts are a fast census.
def hlo_lines(f, *args):
    return len(jax.jit(f).lower(*args).as_text().splitlines())


print("\n  HLO lines for ONE call:")
print(f"    double sum  {hlo_lines(lambda v: h_double_sum(P, Q, v), s):6d}")
print(f"    recurrence  {hlo_lines(lambda v: h_recurrence(P, Q, v), s):6d}")


# %% A "model" with many call sites — V(t, T) made six calls per evaluation and
# was evaluated at many dates, so call sites multiply. Differentiated with
# respect to the mean reversion p, which runs through every series term (as a
# quote-risk sweep does).
def model_value(h, p):
    total = 0.0
    for i in range(N_CALLS):
        total = total + jnp.sum(h(p, Q, s * (1.0 + 0.01 * i)))
    return total


p0 = jnp.asarray(P)
for name, h in (("double sum", h_double_sum), ("recurrence", h_recurrence)):
    v_lines = hlo_lines(lambda pp: model_value(h, pp), p0)
    g_lines = hlo_lines(jax.grad(lambda pp: model_value(h, pp)), p0)
    line = f"  {N_CALLS} call sites, {name:10s}: value HLO {v_lines:7d} lines, grad HLO {g_lines:7d}"
    for label, f in (("value", lambda pp: model_value(h, pp)),
                     ("grad", jax.grad(lambda pp: model_value(h, pp)))):
        if label == "grad" and not TIME_GRAD_COMPILE:
            continue
        t0 = time.perf_counter()
        jax.block_until_ready(jax.jit(f)(p0))
        line += f"   {label} compile {time.perf_counter() - t0:6.2f} s"
    print(line)
print(">>> Graph size, and with it compile time, scale with the *program as")
print("    written*. Run time does not. In isolation the compile gap is ~4-5x")
print("    and grows linearly with N_CALLS (measured: 96 call sites, grad 17 s vs")
print("    5 s). Inside the G2++ PDE's reverse-mode graph — loops, exercise")
print("    events, a market-wide cotangent — the same bloat cost 427 s vs 4 s.")


# ============================================================================
# 4. THE SECOND HALF OF THE FIX: ONE BROADCAST CALL, NOT SIX
# ============================================================================
# %% V(t, T) needs H for three rate pairs (a,a), (b,b), (a,b) at two window
# ends. Six calls trace the series six times; one broadcast call over a
# (pair, end, ...) array traces it once — and computes identical elements.
s0, s1 = s, s * 0.5
pairs = ((P, P), (Q, Q), (P, Q))


def six_calls(s0, s1):
    return jnp.stack([h_recurrence(p, q, s0) - h_recurrence(p, q, s1) for p, q in pairs])


def one_call(s0, s1):
    pp = jnp.reshape(jnp.array([p for p, _ in pairs]), (3, 1, 1))
    qq = jnp.reshape(jnp.array([q for _, q in pairs]), (3, 1, 1))
    hh = h_recurrence(pp, qq, jnp.stack([s0, s1])[jnp.newaxis])
    return hh[:, 0] - hh[:, 1]


print(f"\n  six calls: {hlo_lines(six_calls, s0, s1):5d} HLO lines;  one broadcast call: "
      f"{hlo_lines(one_call, s0, s1):5d};  max |diff| = "
      f"{float(jnp.max(jnp.abs(six_calls(s0, s1) - one_call(s0, s1)))):.1e}")


# ============================================================================
# 5. THE PRODUCTION CENSUS (how it was localised)
# ============================================================================
# %% Same technique on the real primitive: lower g2pp_V at constant vol and with
# three buckets. The guard test (below) bounds the ratio.
from valax.models.g2pp import G2PPModel, g2pp_V  # noqa: E402


def _model(vols, ratios, times):
    return G2PPModel(mean_reversion_x=jnp.asarray(P), mean_reversion_y=jnp.asarray(Q),
                     volatility_x=jnp.asarray(vols), volatility_ratio=jnp.asarray(ratios),
                     correlation=jnp.asarray(-0.7), volatility_times=jnp.asarray(times))


t = jnp.linspace(0.0, 4.0, 41)
census = {
    name: len(eqx.filter_jit(lambda m: g2pp_V(m, t, t + 1.0)).lower(m).as_text().splitlines())
    for name, m in (("constant", _model((0.01,), (0.8,), ())),
                    ("3 buckets", _model((0.012, 0.008, 0.015), (0.8, 0.6, 0.9), (1.0, 2.0))))
}
print(f"\n  production g2pp_V HLO lines: constant {census['constant']}, 3 buckets "
      f"{census['3 buckets']} ({census['3 buckets'] / census['constant']:.1f}x; was ~12x)")


# %% [markdown]
# ## WAY OUT
#
# 1. **Treat graph size as a cost.** In JAX, a Python loop over a static range
#    is unrolled; a formula costs its operation count *as written* at trace and
#    compile time. Write series as recurrences — a running term, Horner, or
#    here `A_{k+1} = (x+y)A_k + xy(x^{k-1} + y^{k-1})` — never as an expanded
#    double sum.
# 2. **Batch repeated elementwise calls.** Six calls of one elementwise
#    function trace it six times; one call over a stacked, broadcast array
#    traces it once and computes the same elements.
# 3. **Use `lax.scan` / `fori_loop` when the trip count is large** (the PDE
#    time loop already does): a rolled loop is traced once regardless of length.
#    For a ~20-term series the unrolled recurrence is fine and fuses better.
# 4. **Measure compile time, and census the graph.** `jit(f).lower(x).as_text()`
#    costs no compilation; counting its lines per primitive localised this in
#    minutes, after a stack dump (`faulthandler.dump_traceback_later`) had shown
#    the process was inside `backend_compile`, not computing.
# 5. **Guard it.** `tests/test_models/test_g2pp_piecewise.py::
#    test_piecewise_variance_term_graph_stays_small` lowers the piecewise V
#    (no compile) and bounds its graph against the constant one.
#
# Entry 6 of [Numerical Pitfalls](../../docs/architecture/numerical-pitfalls.md)
# has the full write-up.
