# %% [markdown]
# # Pitfall: a `jnp.where` guard with the right value and no derivative
#
# **SYMPTOM.** An adjoint sensitivity disagrees with a full-rebuild finite
# difference by ~80%. The FD is **stable across three decades of bump size**,
# which is the textbook evidence that it is converged. A *second, independently
# written* adjoint — `optimistix`'s own `ImplicitAdjoint`, in a different
# parametrisation, through different code — agrees with the first adjoint
# **bit-for-bit**. Two against one. The FD must be the broken one.
#
# It was not. Both adjoints were wrong, for the same reason, because they share
# a primitive.
#
# **ROOT CAUSE.** Hagan's SABR expansion contains the ratio
#
# ```
# z / x(z),    x(z) = log((sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
# ```
#
# which is `0/0` at the money, where `z ∝ log(F/K)` vanishes. The obvious guard
# is the one every JAX codebase writes:
#
# ```python
# is_small = jnp.abs(z) < 1e-7
# z_over_x = jnp.where(is_small, 1.0, safe_z / safe_x)   # limit is 1
# ```
#
# The **value** is right: `z/x(z) -> 1` as `z -> 0`. The **derivative** is not.
# A constant has zero slope, so AD propagates *nothing* through this term at the
# money — discarding the whole `z/x` contribution to `dsigma/dF`. The true
# limiting slope is `d(z/x)/dz -> -rho/2`, which is not zero.
#
# Prices were never affected. Only gradients, only at one strike, and only the
# derivative with respect to the **forward** (the `alpha`/`rho`/`nu`
# derivatives survive, because at `z = 0` their `dz/d(param)` terms are
# multiplied by `log(F/K) = 0` and genuinely vanish — `dz/dF` is the one that
# does not).
#
# **THE LESSON.** Two implementations agreeing is worth nothing when they share
# the thing that is broken. And `jnp.where`-guarding a removable singularity
# with a *constant* is a correctness bug in the derivative even when the value
# is exact — switch to the **series**, so both branches are smooth functions
# that agree to high order.
#
# **KNOBS (search `KNOB`):** `BROKEN_GUARD`, `IS_NORMAL`, `RHO`, `OFFSET`.

# %% Imports
import jax
import jax.numpy as jnp
import numpy as np

from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import _ATM_Z_TOL

jax.config.update("jax_enable_x64", True)


# ============================================================================
# 0. KNOBS
# ============================================================================
# %%
BROKEN_GUARD = True   # KNOB: True = reproduce the old constant-1.0 guard
IS_NORMAL = False     # KNOB: the normal branch failed *harder* (grad -> 0.0)
RHO = -0.25           # KNOB: the error is proportional to rho; try 0.0
OFFSET = 0.0          # KNOB: strike offset from the forward; 0.0 = ATM

FORWARD = jnp.array(0.03)
EXPIRY = jnp.array(1.0)
MODEL = SABRModel(
    alpha=jnp.array(0.0075 if IS_NORMAL else 0.22),
    beta=jnp.array(0.0 if IS_NORMAL else 0.5),
    rho=jnp.array(RHO),
    nu=jnp.array(0.40),
)
STRIKE = FORWARD + OFFSET


# ============================================================================
# 1. THE TWO GUARDS, SIDE BY SIDE
# ============================================================================
# %% Both are "correct" as functions. Only one is correct as a *differentiable*
# function.
def z_over_x_broken(z, rho):
    """The old guard: pin the removable singularity to its limiting value."""
    s = jnp.sqrt(1.0 - 2.0 * rho * z + z**2)
    x = jnp.log((s + z - rho) / (1.0 - rho))
    is_small = jnp.abs(z) < 1e-7
    safe_z = jnp.where(is_small, 1.0, z)
    safe_x = jnp.where(is_small, 1.0, x)
    return jnp.where(is_small, 1.0, safe_z / safe_x)   # <-- constant


def z_over_x_fixed(z, rho):
    """The fix: hand over to the Taylor series, not to a constant."""
    s = jnp.sqrt(1.0 - 2.0 * rho * z + z**2)
    x = jnp.log((s + z - rho) / (1.0 - rho))
    is_small = jnp.abs(z) < _ATM_Z_TOL
    safe_z = jnp.where(is_small, 1.0, z)
    safe_x = jnp.where(is_small, 1.0, x)
    series = 1.0 - 0.5 * rho * z + (1.0 / 6.0 - 0.25 * rho**2) * z**2
    return jnp.where(is_small, series, safe_z / safe_x)


rho = jnp.array(RHO)
print(f"rho = {RHO}      true limiting slope d(z/x)/dz at z=0 is -rho/2 = "
      f"{-RHO / 2:+.6f}")
print(f"  value  at z=0:  broken={float(z_over_x_broken(jnp.array(0.0), rho)):.10f}"
      f"   fixed={float(z_over_x_fixed(jnp.array(0.0), rho)):.10f}"
      "   <- both right")
print(f"  d/dz   at z=0:  broken="
      f"{float(jax.grad(z_over_x_broken)(jnp.array(0.0), rho)):+.6f}"
      f"   fixed={float(jax.grad(z_over_x_fixed)(jnp.array(0.0), rho)):+.6f}"
      "   <- only one right")
print(">>> Identical values. One of them carries no derivative at all.")


# ============================================================================
# 2. THE CONSEQUENCE: dsigma/dF AT THE MONEY
# ============================================================================
# %% A minimal re-implementation of Hagan's lognormal expansion, parametrised by
# which guard it uses, so the two can be compared on the same code path.
def hagan_vol(model, forward, strike, expiry, guard):
    alpha, beta, rho_, nu = model.alpha, model.beta, model.rho, model.nu
    omb = 1.0 - beta
    FK = forward * strike
    log_FK = jnp.log(forward / strike)
    FK_beta = jnp.sqrt(FK) ** omb

    z = (nu / alpha) * FK_beta * log_FK
    z_over_x = guard(z, rho_)

    lsq = log_FK**2
    d1 = 1.0 + omb**2 / 24.0 * lsq + omb**4 / 1920.0 * lsq**2
    numerator = alpha / (FK_beta * d1)

    FK_full = FK**omb
    n1 = omb**2 / 24.0 * alpha**2 / FK_full
    n2 = 0.25 * rho_ * beta * nu * alpha / FK_beta
    n3 = (2.0 - 3.0 * rho_**2) / 24.0 * nu**2
    return numerator * z_over_x * (1.0 + (n1 + n2 + n3) * expiry)


guard = z_over_x_broken if BROKEN_GUARD else z_over_x_fixed


def vol_of_forward(f):
    return hagan_vol(MODEL, f, STRIKE, EXPIRY, guard)


ad = float(jax.grad(vol_of_forward)(FORWARD))
print(f"\nstrike offset from forward = {OFFSET:+g}   guard = "
      f"{'BROKEN (constant)' if BROKEN_GUARD else 'fixed (series)'}")
print(f"  sigma = {float(vol_of_forward(FORWARD)):.10f}")
print("\n   h          central FD       AD            disagreement")
print("   " + "-" * 56)
for h in (1e-5, 1e-6, 1e-7, 1e-8):
    fd = (float(vol_of_forward(FORWARD + h))
          - float(vol_of_forward(FORWARD - h))) / (2.0 * h)
    print(f"   {h:<10.0e} {fd:>13.6f} {ad:>13.6f}    "
          f"{abs(ad - fd) / max(abs(fd), 1e-12):>9.2e}")
print("   " + "-" * 56)
print(">>> The FD is stable over four decades of h. If the AD disagrees with")
print("    THAT, the AD is the thing to doubt -- stability in h rules out")
print("    truncation and round-off, which are the FD's only failure modes here.")


# ============================================================================
# 3. THE TRAP: A SECOND 'INDEPENDENT' ADJOINT AGREES WITH THE FIRST
# ============================================================================
# %% Both of these differentiate the *same* hagan_vol. Different modes,
# different code paths inside JAX -- and identically wrong, because the defect
# is upstream of both.
fwd_mode = float(jax.jacfwd(vol_of_forward)(FORWARD))
rev_mode = float(jax.jacrev(vol_of_forward)(FORWARD))
print(f"\n  forward-mode AD = {fwd_mode:.6f}")
print(f"  reverse-mode AD = {rev_mode:.6f}")
print(f"  difference      = {abs(fwd_mode - rev_mode):.3e}")
print(">>> Perfect agreement, both wrong. In the real incident the 'second")
print("    opinion' was optimistix's ImplicitAdjoint through a least-squares")
print("    calibration -- genuinely different code, genuinely different")
print("    parametrisation, and still sharing sabr_implied_vol underneath.")
print("    Independence has to hold at the level of ASSUMPTIONS and PRIMITIVES,")
print("    not implementations.")


# ============================================================================
# 4. WHY ONLY THE FORWARD DERIVATIVE BROKE
# ============================================================================
# %% z = (nu/alpha) * (FK)^((1-beta)/2) * log(F/K). At F == K, every parameter
# derivative of z carries a factor log(F/K) = 0 and genuinely vanishes. The F
# derivative does not: d/dF log(F/K) = 1/F.
print("\n  at F == K, which derivatives of z survive?")
for name, fn in [
    ("dz/dF    ", lambda: jax.grad(lambda f: (MODEL.nu / MODEL.alpha)
                                  * jnp.sqrt(f * STRIKE) ** (1 - MODEL.beta)
                                  * jnp.log(f / STRIKE))(FORWARD)),
    ("dz/dalpha", lambda: jax.grad(lambda a: (MODEL.nu / a)
                                   * jnp.sqrt(FORWARD * STRIKE) ** (1 - MODEL.beta)
                                   * jnp.log(FORWARD / STRIKE))(MODEL.alpha)),
    ("dz/dnu   ", lambda: jax.grad(lambda n: (n / MODEL.alpha)
                                   * jnp.sqrt(FORWARD * STRIKE) ** (1 - MODEL.beta)
                                   * jnp.log(FORWARD / STRIKE))(MODEL.nu)),
]:
    print(f"    {name} = {float(fn()):+.6f}")
print(">>> Only dz/dF is non-zero, so only dsigma/dF lost a term. A bug that")
print("    corrupts exactly one entry of one row of the Jacobian is the kind")
print("    that survives every aggregate check.")


# ============================================================================
# 5. THE FIX, AND WHY THE SERIES IS THE RIGHT SHAPE OF FIX
# ============================================================================
# %%
print("\n  z/x(z) series: 1 - (rho/2) z + (1/6 - rho^2/4) z^2 + O(z^3)")
print("   z          closed form          series            rel. err")
print("   " + "-" * 60)
for z in (1e-2, 1e-3, 1e-4, 1e-5):
    zz = jnp.array(z)
    s = jnp.sqrt(1.0 - 2.0 * rho * zz + zz**2)
    closed = float(zz / jnp.log((s + zz - rho) / (1.0 - rho)))
    ser = float(1.0 - 0.5 * rho * zz + (1.0 / 6.0 - 0.25 * rho**2) * zz**2)
    print(f"   {z:<10.0e} {closed:.16f}  {ser:.16f}  "
          f"{abs(closed - ser) / abs(closed):.2e}")
print("   " + "-" * 60)
print(f"  VALAX switches at |z| < _ATM_Z_TOL = {_ATM_Z_TOL:g}, chosen by balancing")
print("  the series truncation O(z^3) against the closed form's cancellation")
print("  O(eps/z): z^3 ~ eps/z  =>  z ~ eps^(1/4) ~ 1e-4. Both sides ~1e-12 there.")

# %% Confirm the production function is fixed.
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_normal_implied_vol

print("\n  production sabr_implied_vol / sabr_normal_implied_vol, AD vs FD at ATM:")
for label, fn, m in [
    ("lognormal", lambda f: sabr_implied_vol(MODEL, f, FORWARD, EXPIRY), MODEL),
    ("normal   ", lambda f: sabr_normal_implied_vol(
        SABRModel(alpha=jnp.array(0.0075), beta=jnp.array(0.0),
                  rho=rho, nu=jnp.array(0.40)),
        f, FORWARD, EXPIRY, jnp.array(0.0)), None),
]:
    a = float(jax.grad(fn)(FORWARD))
    h = 1e-7
    f_ = (float(fn(FORWARD + h)) - float(fn(FORWARD - h))) / (2 * h)
    print(f"    {label}  AD={a:>12.6f}  FD={f_:>12.6f}  "
          f"rel={abs(a - f_) / max(abs(f_), 1e-12):.2e}")


# %% [markdown]
# ## WAY OUT
#
# 1. **Never `jnp.where` a removable singularity onto a constant.** Use the
#    series expansion of the limit instead. Both branches are then smooth, they
#    agree to `O(z^3)`, and the derivative is right on both sides. Keep the
#    outer double-`where` — that part is doing real work, stopping NaN
#    gradients leaking out of the inactive branch.
# 2. **Pick the crossover by balancing the two errors**, not by picking a
#    small-looking number. Series truncation grows like `z**3`; the closed
#    form's cancellation grows like `eps/z`. They cross at `eps**0.25 ~ 1e-4`.
#    The original `1e-7` was *too small* — it kept the ill-conditioned closed
#    form in play over a decade where the series was already better.
# 3. **Test derivatives at the singular point specifically.** Every value test
#    passed. The guard is in `tests/test_models/test_sabr.py::TestATMDerivatives`,
#    which checks AD against FD *at* the money and asserts the ATM forward
#    derivative is non-zero.
# 4. **When a reference disagrees, rank your evidence by independence.** An FD
#    that is stable across four decades of `h` shares *no* code with the
#    adjoint. A second adjoint through the same primitive shares everything
#    that matters. The agreement of two dependent methods is not two votes.
#
# Entry 4 of [Numerical Pitfalls](../../docs/architecture/numerical-pitfalls.md)
# already recorded this lesson — "corroboration from an engine that shares your
# approximation is worthless" — about QuantLib and VALAX both snapping cashflow
# dates. It was then walked into again here, which is the whole reason this
# folder exists.
