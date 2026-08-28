# %% [markdown]
# # Pitfall: calibrating a smile with the *wrong vol convention*
#
# **SYMPTOM.** A normal-quoted SABR swaption smile is calibrated, the solver
# reports success with a small residual — and yet the resulting parameters are
# nonsense (alpha ~40x too small), so any price/vol you read back with the
# *correct* (normal) convention is wildly wrong (~1e-2 in vol).
#
# **ROOT CAUSE.** The residual was built with the **lognormal** Hagan formula
# while the quotes are **normal** (Bachelier) vols — a model/convention mismatch.
# The lognormal formula *can* approximately reproduce the numeric vol values, so
# the least-squares residual looks fine, but it does so with meaningless
# parameters. Downstream, `SwaptionCube` queries with the normal formula and the
# mismatch surfaces as a large error.
#
# This was a real bug: `calibrate_sabr` accepted a `vol_fn` argument but a stale
# line dropped it, always using `sabr_implied_vol` (lognormal). This very
# playground is what caught it — by contrasting the wrapper against a direct
# `optimistix` call (Section 4). It is fixed now; this file keeps the *mismatch*
# reproducible on purpose (flip `WRONG_MODEL`) as a permanent lesson.
#
# **THE LESSON.** A small calibration residual is necessary but **not
# sufficient**. Always (a) make the residual model match the quote convention,
# and (b) sanity-check fitted parameters (for beta=0 normal SABR, alpha should be
# about the ATM normal vol).
#
# **KNOBS (search `KNOB`):** `WRONG_MODEL`, `SCALE`, `START_AT_TRUTH`.

# %% Imports
import functools

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_normal_implied_vol
from valax.calibration.sabr import calibrate_sabr, _default_sabr_guess
from valax.calibration.transforms import (
    model_to_unconstrained, unconstrained_to_model, positive, correlation,
)
from valax.calibration.loss import vol_residuals, weighted_sse

jax.config.update("jax_enable_x64", True)


# ============================================================================
# 0. KNOBS
# ============================================================================
# %%
WRONG_MODEL = True     # KNOB: True = fit normal quotes with the LOGNORMAL formula
SCALE = "rates"        # KNOB: "rates" (F~0.025) | "equity" (F~100)
START_AT_TRUTH = False # KNOB: seed the optimiser at the exact solution


# %% Problem setup: a self-consistent *normal* smile
if SCALE == "rates":
    FORWARD = jnp.array(0.025)
    TRUE = SABRModel(alpha=jnp.array(0.01), beta=jnp.array(0.0),
                     rho=jnp.array(-0.20), nu=jnp.array(0.30))
    STRIKES = jnp.linspace(0.015, 0.035, 5)
else:
    FORWARD = jnp.array(100.0)
    TRUE = SABRModel(alpha=jnp.array(30.0), beta=jnp.array(0.0),
                     rho=jnp.array(-0.20), nu=jnp.array(0.30))
    STRIKES = jnp.linspace(60.0, 140.0, 5)

EXPIRY = jnp.array(2.0)
NORMAL_VOL = functools.partial(sabr_normal_implied_vol, shift=jnp.asarray(0.0))
MARKET = jax.vmap(lambda K: NORMAL_VOL(TRUE, FORWARD, K, EXPIRY))(STRIKES)

# The residual model: deliberately wrong (lognormal) or right (normal).
FIT_VOL = sabr_implied_vol if WRONG_MODEL else NORMAL_VOL
print(f"scale={SCALE}  residual model = "
      f"{'LOGNORMAL (WRONG)' if WRONG_MODEL else 'normal (right)'}")
print("market (normal) vols :", np.round(np.asarray(MARKET), 6))


# ============================================================================
# 1. THE FIT 'SUCCEEDS' BUT THE PARAMETERS ARE NONSENSE
# ============================================================================
# %%
guess = TRUE if START_AT_TRUTH else None
fitted, sol = calibrate_sabr(
    strikes=STRIKES, market_vols=MARKET, forward=FORWARD, expiry=EXPIRY,
    initial_guess=guess, fixed_beta=jnp.array(0.0),
    vol_fn=FIT_VOL, is_normal=not WRONG_MODEL, max_steps=1024,
)
# Residual in the model that was fit (may look fine even when wrong):
resid_fit = jax.vmap(lambda K: FIT_VOL(fitted, FORWARD, K, EXPIRY))(STRIKES) - MARKET
# Error you actually care about: re-price with the true (normal) convention.
normal_err = jax.vmap(lambda K: NORMAL_VOL(fitted, FORWARD, K, EXPIRY))(STRIKES) - MARKET

print(f"\nfitted alpha = {float(fitted.alpha):.5f}   (true = {float(TRUE.alpha):.5f}"
      f";  ATM sanity: alpha should ~ ATM normal vol = {float(MARKET[2]):.5f})")
print(f"residual in the FITTED model  = {float(jnp.max(jnp.abs(resid_fit))):.2e}"
      "   <- can look 'converged'")
print(f"error re-priced in NORMAL vol = {float(jnp.max(jnp.abs(normal_err))):.2e}"
      "   <- what actually bites")
if WRONG_MODEL:
    print(">>> Small fit residual, absurd alpha, large normal-vol error: "
          "the classic convention-mismatch signature.")


# ============================================================================
# 2. TELL: 'TRUTH' IS NOT A ZERO OF THE WRONG-MODEL RESIDUAL
# ============================================================================
# %% If the true params do not zero the residual, your model != your quotes.
transforms = {"alpha": positive(), "rho": correlation(), "nu": positive()}
y_true = model_to_unconstrained(TRUE, transforms)
args_fit = (transforms, TRUE, FIT_VOL, STRIKES, MARKET, FORWARD, EXPIRY,
            jnp.ones_like(STRIKES))
args_true = (transforms, TRUE, NORMAL_VOL, STRIKES, MARKET, FORWARD, EXPIRY,
             jnp.ones_like(STRIKES))
print(f"\nSSE at truth, FITTED model = {float(weighted_sse(y_true, args_fit)):.3e}"
      f"   ({'nonzero => wrong model' if WRONG_MODEL else 'zero => right model'})")
print(f"SSE at truth, NORMAL model = {float(weighted_sse(y_true, args_true)):.3e}"
      "   (zero => truth fits the normal quotes)")


# ============================================================================
# 3. CONDITIONING ASIDE (why the wrong fit lands on absurd params)
# ============================================================================
# %% Condition number of JᵀJ for the fitted-model residual at truth.
def residual_vec(pv):
    a, r, n = pv
    m = SABRModel(alpha=a, beta=jnp.array(0.0), rho=r, nu=n)
    return jax.vmap(lambda K: FIT_VOL(m, FORWARD, K, EXPIRY))(STRIKES) - MARKET

J = jax.jacobian(residual_vec)(jnp.array([float(TRUE.alpha), float(TRUE.rho), float(TRUE.nu)]))
sv = jnp.linalg.svd(J.T @ J, compute_uv=False)
print(f"\nJᵀJ singular values = {np.array(sv)}")
print(f"condition number    = {float(sv[0] / sv[-1]):.3e}")


# ============================================================================
# 4. YOUR PLAYGROUND: wrapper vs direct optimistix, and the fix
# ============================================================================
# %% Contrast the calibrator against a direct optimistix call, and show the
# RIGHT model recovering truth to machine precision.
def run_optx(vol_fn, y0):
    args = (transforms, TRUE, vol_fn, STRIKES, MARKET, FORWARD, EXPIRY, jnp.ones_like(STRIKES))
    sol = optx.least_squares(vol_residuals, optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8),
                             y0, args=args, max_steps=1024, throw=False)
    m = unconstrained_to_model(sol.value, transforms, TRUE)
    normal_err = jax.vmap(lambda K: NORMAL_VOL(m, FORWARD, K, EXPIRY))(STRIKES) - MARKET
    return f"alpha={float(m.alpha):.5f}  normal-vol err={float(jnp.max(jnp.abs(normal_err))):.2e}"

g = _default_sabr_guess(FORWARD, MARKET, jnp.array(0.0), is_normal=True)
import equinox as eqx
g = eqx.tree_at(lambda m: m.beta, g, jnp.array(0.0))
y_guess = model_to_unconstrained(g, transforms)

print("\n--- direct optimistix LM(1e-8), start = guess ---")
print("WRONG (lognormal) model:", run_optx(sabr_implied_vol, y_guess))
print("RIGHT (normal)    model:", run_optx(NORMAL_VOL, y_guess))
print("\n=> Same solver, same data, same start. Only the residual *model* differs."
      "\n   The fix was one line: make calibrate_sabr use the passed vol_fn.")
