# %% [markdown]
# # VALAX vs QuantLib: SABR Volatility Smile
#
# Side-by-side comparison of:
# - SABR implied volatility (Hagan formula)
# - Smile generation across strikes
# - Model parameter sensitivities
# - Calibration to market data
#
# Validated by: tests/test_quantlib_comparison/test_sabr_ql.py

# %% Imports
import jax
import jax.numpy as jnp
import numpy as np
import QuantLib as ql
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_price
from valax.instruments.options import EuropeanOption

# ============================================================================
# 1. COMMON PARAMETERS
# ============================================================================

print("=" * 70)
print("VALAX vs QuantLib: SABR Volatility Smile")
print("=" * 70)

# SABR parameters (typical equity/rates smile)
alpha = 0.04    # initial vol level
beta = 0.5      # CEV exponent
rho = -0.25     # spot-vol correlation (negative skew)
nu = 0.4        # vol of vol

forward = 0.03  # 3% forward rate
expiry = 2.0    # 2 years

print(f"\nSABR parameters: α={alpha}, β={beta}, ρ={rho}, ν={nu}")
print(f"Forward: {forward*100:.1f}%, Expiry: {expiry}Y")

# Strike range: 50bp to 550bp (around 3% forward)
strikes = np.linspace(0.005, 0.055, 21)

# ============================================================================
# 2. VALAX — SABR SMILE
# ============================================================================

print(f"\n{'-' * 70}")
print("VALAX: SABR Implied Vols (Hagan formula)")
print("-" * 70)

sabr_model = SABRModel(
    alpha=jnp.array(alpha),
    beta=jnp.array(beta),
    rho=jnp.array(rho),
    nu=jnp.array(nu),
)

valax_vols = []
for K in strikes:
    iv = float(sabr_implied_vol(sabr_model, jnp.array(forward), jnp.array(K), jnp.array(expiry)))
    valax_vols.append(iv)

# ============================================================================
# 3. QUANTLIB — SABR SMILE
# ============================================================================

print(f"\n{'-' * 70}")
print("QuantLib: SABR Implied Vols")
print("-" * 70)

ql_vols = []
for K in strikes:
    try:
        iv = ql.sabrVolatility(float(K), forward, expiry, alpha, beta, nu, rho)
        ql_vols.append(iv)
    except RuntimeError:
        ql_vols.append(float('nan'))

# ============================================================================
# 4. COMPARISON
# ============================================================================

print(f"\n{'=' * 70}")
print("SMILE COMPARISON")
print("=" * 70)

print(f"\n{'Strike':>8} {'VALAX IV':>12} {'QL IV':>12} {'Diff (bp)':>12}")
print("-" * 46)
for i, K in enumerate(strikes):
    if not np.isnan(ql_vols[i]):
        diff_bp = abs(valax_vols[i] - ql_vols[i]) * 10000
        print(f"{K*100:7.2f}% {valax_vols[i]*100:11.4f}% {ql_vols[i]*100:11.4f}% {diff_bp:11.2f}")
    else:
        print(f"{K*100:7.2f}% {valax_vols[i]*100:11.4f}% {'N/A':>12}")

# ============================================================================
# 5. VALAX AUTODIFF ADVANTAGE: MODEL SENSITIVITIES
# ============================================================================

print(f"\n{'=' * 70}")
print("AUTODIFF ADVANTAGE: SABR Parameter Sensitivities")
print("=" * 70)

print("\nSensitivities of implied vol to SABR parameters (at forward = strike):")
print("VALAX computes these via jax.grad — no finite differences needed.\n")

atm_strike = jnp.array(forward)

# d(IV)/d(alpha)
div_dalpha = jax.grad(
    lambda m: sabr_implied_vol(m, jnp.array(forward), atm_strike, jnp.array(expiry)),
    allow_int=True,
)(sabr_model)
print(f"  d(IV)/d(α):  α sensitivity via pytree grad")

# Scalar sensitivities at ATM
def atm_vol(a, b, r, n):
    m = SABRModel(alpha=a, beta=b, rho=r, nu=n)
    return sabr_implied_vol(m, jnp.array(forward), atm_strike, jnp.array(expiry))

grad_alpha = float(jax.grad(atm_vol, argnums=0)(jnp.array(alpha), jnp.array(beta), jnp.array(rho), jnp.array(nu)))
grad_rho = float(jax.grad(atm_vol, argnums=2)(jnp.array(alpha), jnp.array(beta), jnp.array(rho), jnp.array(nu)))
grad_nu = float(jax.grad(atm_vol, argnums=3)(jnp.array(alpha), jnp.array(beta), jnp.array(rho), jnp.array(nu)))

print(f"\nATM implied vol sensitivities:")
print(f"  d(IV)/d(α) = {grad_alpha:.6f}   (vol level sensitivity)")
print(f"  d(IV)/d(ρ) = {grad_rho:.6f}   (skew sensitivity)")
print(f"  d(IV)/d(ν) = {grad_nu:.6f}   (smile curvature sensitivity)")

print("""
In QuantLib, computing these sensitivities requires:
  → Finite differences: bump each parameter, recompute smile
  → 2N+1 evaluations for N parameters (central differences)
  → Approximation error from bump size choice

In VALAX:
  → One jax.grad call per parameter (exact)
  → Or differentiate through the entire pricing chain:
    d(price)/d(SABR_params) in a single backward pass
""")

# ============================================================================
# 6. PRICE SENSITIVITIES THROUGH SABR
# ============================================================================

print(f"{'=' * 70}")
print("END-TO-END: d(Price)/d(SABR params)")
print("=" * 70)

# Price an option through SABR → Black-76 → price
# Then differentiate the entire chain w.r.t. SABR parameters
option = EuropeanOption(
    strike=jnp.array(forward),
    expiry=jnp.array(expiry),
    is_call=True,
)

rate = jnp.array(0.03)

def price_through_sabr(a, r_param, n):
    m = SABRModel(alpha=a, beta=jnp.array(beta), rho=r_param, nu=n)
    return sabr_price(option, jnp.array(forward), rate, m)

price_val = float(price_through_sabr(jnp.array(alpha), jnp.array(rho), jnp.array(nu)))
dp_dalpha = float(jax.grad(price_through_sabr, argnums=0)(jnp.array(alpha), jnp.array(rho), jnp.array(nu)))
dp_drho = float(jax.grad(price_through_sabr, argnums=1)(jnp.array(alpha), jnp.array(rho), jnp.array(nu)))
dp_dnu = float(jax.grad(price_through_sabr, argnums=2)(jnp.array(alpha), jnp.array(rho), jnp.array(nu)))

print(f"\nATM call price: {price_val:.6f}")
print(f"\nPrice sensitivities to SABR parameters:")
print(f"  d(Price)/d(α) = {dp_dalpha:.6f}")
print(f"  d(Price)/d(ρ) = {dp_drho:.6f}")
print(f"  d(Price)/d(ν) = {dp_dnu:.6f}")

print("""
This differentiates through:
  SABR params → Hagan formula → implied vol → Black-76 → price
The entire chain is traced by JAX — no manual chain rule needed.
""")

# ============================================================================
# 7. API COMPARISON
# ============================================================================

print(f"{'=' * 70}")
print("API DESIGN: SABR")
print("=" * 70)

print("""
┌────────────────────────┬────────────────────────────────┬──────────────────────────────────┐
│ Aspect                 │ VALAX                          │ QuantLib                         │
├────────────────────────┼────────────────────────────────┼──────────────────────────────────┤
│ SABR vol               │ sabr_implied_vol(model, F,K,T) │ ql.sabrVolatility(K,F,T,α,β,ν,ρ)│
│ Model representation   │ SABRModel pytree (4 params)    │ Separate scalar arguments        │
│ Param sensitivities    │ jax.grad (exact, automatic)    │ Finite differences (manual)      │
│ End-to-end d(P)/d(θ)  │ One backward pass              │ Not directly supported           │
│ Calibration            │ JAX optimizer (optax/optimistix)│ ql.EndCriteria + Simplex/LM     │
│ Differentiable calib   │ Yes (unroll optimizer)         │ No                               │
│ GPU acceleration       │ Automatic (JAX)                │ CPU only                         │
└────────────────────────┴────────────────────────────────┴──────────────────────────────────┘
""")
