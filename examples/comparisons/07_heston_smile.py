# %% [markdown]
# # VALAX vs QuantLib: Heston Stochastic Volatility Model
#
# Side-by-side comparison of:
# - Heston MC path generation
# - Implied volatility smile under Heston
# - QuantLib's analytic Heston (Fourier transform) as reference
# - Model parameter sensitivities via autodiff
#
# Validated by: tests/test_quantlib_comparison/test_heston_ql.py

# %% Imports
import jax
import jax.numpy as jnp
import QuantLib as ql
from valax.instruments.options import EuropeanOption
from valax.models.heston import HestonModel
from valax.pricing.mc.paths import generate_heston_paths
from valax.pricing.analytic.black_scholes import black_scholes_price, black_scholes_implied_vol

# ============================================================================
# 1. COMMON PARAMETERS
# ============================================================================

print("=" * 70)
print("VALAX vs QuantLib: Heston Stochastic Volatility")
print("=" * 70)

S0 = 100.0
T = 1.0
r = 0.04
q = 0.01

# Heston parameters — typical equity calibration
v0 = 0.04        # initial variance (vol ≈ 20%)
kappa = 2.0      # mean reversion speed
theta = 0.04     # long-run variance
xi = 0.5         # vol of vol
rho = -0.7       # spot-vol correlation (leverage effect)

print(f"\nSpot: {S0}, T: {T}Y, r: {r}, q: {q}")
print(f"Heston: v0={v0}, κ={kappa}, θ={theta}, ξ={xi}, ρ={rho}")

# QuantLib setup
today_ql = ql.Date(26, 3, 2026)
ql.Settings.instance().evaluationDate = today_ql
maturity_ql = today_ql + ql.Period(1, ql.Years)

spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(today_ql, r, ql.Actual365Fixed()))
div_handle = ql.YieldTermStructureHandle(ql.FlatForward(today_ql, q, ql.Actual365Fixed()))

heston_process = ql.HestonProcess(rate_handle, div_handle, spot_handle, v0, kappa, theta, xi, rho)
heston_model_ql = ql.HestonModel(heston_process)
analytic_engine = ql.AnalyticHestonEngine(heston_model_ql)

# ============================================================================
# 2. IMPLIED VOLATILITY SMILE
# ============================================================================

print(f"\n{'=' * 70}")
print("IMPLIED VOLATILITY SMILE: Heston MC vs QuantLib Analytic")
print("=" * 70)

# VALAX: generate Heston MC paths
heston_v = HestonModel(
    v0=jnp.array(v0), kappa=jnp.array(kappa), theta=jnp.array(theta),
    xi=jnp.array(xi), rho=jnp.array(rho),
    rate=jnp.array(r), dividend=jnp.array(q),
)

key = jax.random.PRNGKey(42)
n_paths = 200_000
spot_paths, var_paths = generate_heston_paths(
    heston_v, jnp.array(S0), T, 252, n_paths, key
)

# Compare implied vols at various strikes
test_strikes = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]

print(f"\n{'Strike':>8} {'QL Analytic':>12} {'VALAX MC IV':>12} {'Diff (bp)':>10} {'MC SE':>8}")
print("-" * 52)

for K_val in test_strikes:
    # QuantLib analytic Heston
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K_val)
    exercise = ql.EuropeanExercise(maturity_ql)
    option_ql = ql.VanillaOption(payoff, exercise)
    option_ql.setPricingEngine(analytic_engine)
    ql_price = option_ql.NPV()
    ql_iv = option_ql.impliedVolatility(ql_price,
        ql.BlackScholesMertonProcess(spot_handle, div_handle, rate_handle,
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today_ql, ql.NullCalendar(), 0.2, ql.Actual365Fixed()))),
        1e-8, 1000, 0.001, 4.0)

    # VALAX MC
    mc_payoffs = jnp.maximum(spot_paths[:, -1] - K_val, 0.0)
    mc_price = float(jnp.exp(-jnp.array(r) * T) * jnp.mean(mc_payoffs))
    mc_se = float(jnp.exp(-jnp.array(r) * T) * jnp.std(mc_payoffs) / jnp.sqrt(float(n_paths)))

    opt = EuropeanOption(strike=jnp.array(K_val), expiry=jnp.array(T), is_call=True)
    mc_iv = float(black_scholes_implied_vol(opt, jnp.array(S0), jnp.array(r), jnp.array(q), jnp.array(mc_price)))

    diff_bp = abs(mc_iv - ql_iv) * 10000
    print(f"{K_val:8.0f} {ql_iv*100:11.2f}% {mc_iv*100:11.2f}% {diff_bp:9.1f} ${mc_se:>7.4f}")

# ============================================================================
# 3. OPTION PRICES COMPARISON
# ============================================================================

print(f"\n{'=' * 70}")
print("HESTON OPTION PRICES")
print("=" * 70)

print(f"\n{'Strike':>8} {'QL Analytic':>14} {'VALAX MC':>14} {'SE':>8} {'# SE':>8}")
print("-" * 54)
for K_val in test_strikes:
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K_val)
    exercise = ql.EuropeanExercise(maturity_ql)
    option_ql = ql.VanillaOption(payoff, exercise)
    option_ql.setPricingEngine(analytic_engine)
    ql_price = option_ql.NPV()

    mc_payoffs = jnp.maximum(spot_paths[:, -1] - K_val, 0.0)
    mc_price = float(jnp.exp(-jnp.array(r) * T) * jnp.mean(mc_payoffs))
    mc_se = float(jnp.exp(-jnp.array(r) * T) * jnp.std(mc_payoffs) / jnp.sqrt(float(n_paths)))
    n_se = abs(mc_price - ql_price) / mc_se if mc_se > 0 else 0

    print(f"{K_val:8.0f} ${ql_price:>13.4f} ${mc_price:>13.4f} ${mc_se:>7.4f} {n_se:>7.1f}")

# ============================================================================
# 4. SMILE CHARACTERISTICS
# ============================================================================

print(f"\n{'=' * 70}")
print("HESTON SMILE CHARACTERISTICS")
print("=" * 70)

print(f"""
The Heston model produces:
  → Negative skew (ρ = {rho}): low strikes have higher implied vol
  → Fat tails (ξ = {xi}): wings are elevated vs Black-Scholes
  → Mean-reverting variance (κ = {kappa}, θ = {theta})

Observed in both VALAX MC and QuantLib analytic — the smile shape
is model-inherent, not an artifact of either implementation.
""")

# ============================================================================
# 5. AUTODIFF ADVANTAGE: HESTON MODEL SENSITIVITIES
# ============================================================================

print(f"{'=' * 70}")
print("AUTODIFF ADVANTAGE: Heston Parameter Sensitivities")
print("=" * 70)

# Differentiate MC price w.r.t. Heston parameters
# Note: MC gradient is biased but gives directional sensitivity
K_test = 100.0
opt_test = EuropeanOption(strike=jnp.array(K_test), expiry=jnp.array(T), is_call=True)

def heston_mc_price(v0_p, kappa_p, theta_p, xi_p, rho_p):
    """MC price as a function of Heston parameters (small path count for gradient)."""
    model = HestonModel(
        v0=v0_p, kappa=kappa_p, theta=theta_p, xi=xi_p, rho=rho_p,
        rate=jnp.array(r), dividend=jnp.array(q),
    )
    spots, _ = generate_heston_paths(model, jnp.array(S0), T, 50, 10_000, key)
    payoffs = jnp.maximum(spots[:, -1] - K_test, 0.0)
    return jnp.exp(-jnp.array(r) * T) * jnp.mean(payoffs)

# Compute sensitivities
price_base = float(heston_mc_price(jnp.array(v0), jnp.array(kappa), jnp.array(theta), jnp.array(xi), jnp.array(rho)))

param_names = ["v0", "κ", "θ", "ξ", "ρ"]
print(f"\nATM call price (MC, 10K paths): ${price_base:.4f}")

# Compute sensitivities one at a time (some may be numerically unstable)
print(f"\nd(Price)/d(param) via jax.grad through MC simulation:")
print(f"{'Param':<10} {'Sensitivity':>14}")
print("-" * 26)
for i, name in enumerate(param_names):
    try:
        g = jax.grad(heston_mc_price, argnums=i)(
            jnp.array(v0), jnp.array(kappa), jnp.array(theta), jnp.array(xi), jnp.array(rho)
        )
        val = float(g)
        if jnp.isnan(g):
            print(f"{name:<10} {'NaN (*)':>14}")
        else:
            print(f"{name:<10} {val:>14.6f}")
    except Exception:
        print(f"{name:<10} {'error':>14}")

print(f"\n(*) NaN gradients indicate numerical instability in the SDE solver")
print(f"    at this discretization. Finer steps or variance-preserving schemes")
print(f"    (e.g., log-variance transform) improve gradient stability.")

print("""
VALAX differentiates through:
  Heston params → SDE dynamics → diffrax solver → terminal payoff → price

This gives approximate sensitivities to ALL model parameters
in a single backward pass. In QuantLib, you'd need to:
  → Bump each parameter separately
  → Rerun MC (or re-evaluate Fourier integral)
  → 2N+1 evaluations for N parameters
""")

# ============================================================================
# 6. API COMPARISON
# ============================================================================

print(f"{'=' * 70}")
print("API DESIGN: Heston Model")
print("=" * 70)

print("""
┌───────────────────────┬────────────────────────────────┬────────────────────────────────────┐
│ Aspect                │ VALAX                          │ QuantLib                           │
├───────────────────────┼────────────────────────────────┼────────────────────────────────────┤
│ Model definition      │ HestonModel(v0,κ,θ,ξ,ρ,r,q)  │ HestonProcess(YTS,YTS,spot,params) │
│ MC paths              │ generate_heston_paths()        │ Implicit in MCEngine               │
│ Analytic pricing      │ Not yet (planned: Fourier)     │ AnalyticHestonEngine (FFT)         │
│ MC pricing            │ payoff(paths) → mean           │ MCEuropeanHestonEngine             │
│ Implied vol smile     │ MC prices → BS IV inversion    │ option.impliedVolatility()         │
│ Model sensitivities   │ jax.grad through MC paths      │ Finite differences (manual)        │
│ Calibration           │ JAX optimizer + autodiff grads  │ ql.HestonModelHelper + LM         │
│ GPU paths             │ Automatic (JAX)                │ CPU only                           │
└───────────────────────┴────────────────────────────────┴────────────────────────────────────┘
""")
