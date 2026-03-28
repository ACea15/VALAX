# %% [markdown]
# # VALAX vs QuantLib: Monte Carlo Pricing
#
# Side-by-side comparison of:
# - GBM path generation and European option MC pricing
# - Heston stochastic volatility MC
# - Implied volatility smile from MC
# - Exotic payoffs (Asian, barrier)
#
# Validated by: tests/test_quantlib_comparison/test_monte_carlo_ql.py

# %% Imports
import jax
import jax.numpy as jnp
import QuantLib as ql
from valax.instruments.options import EuropeanOption
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.pricing.mc.paths import generate_gbm_paths, generate_heston_paths
from valax.pricing.mc.engine import mc_price_with_stderr, MCConfig
from valax.pricing.mc.payoffs import european_payoff, asian_payoff, barrier_payoff
from valax.pricing.analytic.black_scholes import black_scholes_price

# ============================================================================
# 1. COMMON PARAMETERS
# ============================================================================

print("=" * 70)
print("VALAX vs QuantLib: Monte Carlo Pricing")
print("=" * 70)

S0 = 100.0
K = 105.0
T = 1.0
sigma = 0.25
r = 0.04
q = 0.01
n_paths = 100_000
n_steps = 252

print(f"\nParameters: S={S0}, K={K}, T={T}, σ={sigma}, r={r}, q={q}")
print(f"MC config:  {n_paths:,} paths, {n_steps} steps")

# Analytical reference
call = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=True)
bs_ref = float(black_scholes_price(call, jnp.array(S0), jnp.array(sigma), jnp.array(r), jnp.array(q)))
print(f"BS analytical reference: ${bs_ref:.4f}")

# ============================================================================
# 2. VALAX GBM MONTE CARLO
# ============================================================================

print(f"\n{'-' * 70}")
print("VALAX: GBM Monte Carlo")
print("-" * 70)

bs_model = BlackScholesModel(
    vol=jnp.array(sigma), rate=jnp.array(r), dividend=jnp.array(q),
)
key = jax.random.PRNGKey(42)
config = MCConfig(n_paths=n_paths, n_steps=n_steps)

mc_p, mc_se = mc_price_with_stderr(call, jnp.array(S0), bs_model, config, key)
valax_mc = float(mc_p)
valax_se = float(mc_se)

print(f"MC price: ${valax_mc:.4f}  ±${valax_se:.4f}")
print(f"Error vs BS: ${abs(valax_mc - bs_ref):.4f}  ({abs(valax_mc - bs_ref)/valax_se:.1f} SE)")

# ============================================================================
# 3. QUANTLIB GBM MONTE CARLO
# ============================================================================

print(f"\n{'-' * 70}")
print("QuantLib: GBM Monte Carlo")
print("-" * 70)

today_ql = ql.Date(26, 3, 2026)
ql.Settings.instance().evaluationDate = today_ql

spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
rate_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today_ql, r, ql.Actual365Fixed())
)
div_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today_ql, q, ql.Actual365Fixed())
)
vol_handle = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(today_ql, ql.NullCalendar(), sigma, ql.Actual365Fixed())
)

bsm_process = ql.BlackScholesMertonProcess(
    spot_handle, div_handle, rate_handle, vol_handle
)

maturity = today_ql + ql.Period(1, ql.Years)
payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
exercise = ql.EuropeanExercise(maturity)
option_ql = ql.VanillaOption(payoff, exercise)

# MC engine with pseudorandom paths
rng = "pseudorandom"
mc_engine = ql.MCEuropeanEngine(bsm_process, rng,
                                 timeSteps=n_steps,
                                 requiredSamples=n_paths,
                                 seed=42)
option_ql.setPricingEngine(mc_engine)
ql_mc = option_ql.NPV()
ql_mc_err = option_ql.errorEstimate()

print(f"MC price: ${ql_mc:.4f}  ±${ql_mc_err:.4f}")
print(f"Error vs BS: ${abs(ql_mc - bs_ref):.4f}")

# ============================================================================
# 4. HEAD-TO-HEAD
# ============================================================================

print(f"\n{'=' * 70}")
print("GBM MC COMPARISON")
print("=" * 70)

print(f"\n{'Metric':<20} {'VALAX':>14} {'QuantLib':>14} {'BS Exact':>14}")
print("-" * 64)
print(f"{'MC Price':<20} ${valax_mc:>13.4f} ${ql_mc:>13.4f} ${bs_ref:>13.4f}")
print(f"{'Std Error':<20} ${valax_se:>13.4f} ${ql_mc_err:>13.4f} {'—':>14}")
print(f"{'Error vs BS':<20} ${abs(valax_mc-bs_ref):>13.4f} ${abs(ql_mc-bs_ref):>13.4f} {'—':>14}")

# ============================================================================
# 5. HESTON STOCHASTIC VOLATILITY
# ============================================================================

print(f"\n{'=' * 70}")
print("HESTON STOCHASTIC VOLATILITY")
print("=" * 70)

# Common Heston parameters
v0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.5, -0.7

print(f"\nHeston: v0={v0}, κ={kappa}, θ={theta}, ξ={xi}, ρ={rho}")

# --- VALAX Heston MC ---
heston_v = HestonModel(
    v0=jnp.array(v0), kappa=jnp.array(kappa), theta=jnp.array(theta),
    xi=jnp.array(xi), rho=jnp.array(rho),
    rate=jnp.array(r), dividend=jnp.array(q),
)

key2 = jax.random.PRNGKey(123)
spot_paths, var_paths = generate_heston_paths(
    heston_v, jnp.array(S0), T, n_steps, n_paths, key2
)

# Price calls at multiple strikes
test_strikes = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]

print(f"\n--- Heston MC Call Prices ---")
print(f"{'Strike':>8} {'VALAX MC':>12} {'SE':>8}")
valax_heston_prices = {}
for K_val in test_strikes:
    payoffs = jnp.maximum(spot_paths[:, -1] - K_val, 0.0)
    mc_h = float(jnp.exp(-jnp.array(r) * T) * jnp.mean(payoffs))
    se_h = float(jnp.exp(-jnp.array(r) * T) * jnp.std(payoffs) / jnp.sqrt(float(n_paths)))
    valax_heston_prices[K_val] = (mc_h, se_h)
    print(f"{K_val:8.0f} ${mc_h:>11.4f} ${se_h:>7.4f}")

# --- QuantLib Heston MC ---
heston_process = ql.HestonProcess(
    rate_handle, div_handle, spot_handle,
    v0, kappa, theta, xi, rho
)

print(f"\n--- QuantLib Heston: Analytic (semi-closed form) ---")
print(f"{'Strike':>8} {'QL Analytic':>12} {'vs VALAX MC':>12}")
for K_val in test_strikes:
    payoff_h = ql.PlainVanillaPayoff(ql.Option.Call, K_val)
    exercise_h = ql.EuropeanExercise(maturity)
    option_h = ql.VanillaOption(payoff_h, exercise_h)

    # Use Heston analytic engine as reference
    heston_model_ql = ql.HestonModel(heston_process)
    analytic_engine = ql.AnalyticHestonEngine(heston_model_ql)
    option_h.setPricingEngine(analytic_engine)

    ql_h = option_h.NPV()
    v_mc, v_se = valax_heston_prices[K_val]
    diff_se = abs(v_mc - ql_h) / v_se if v_se > 0 else 0
    print(f"{K_val:8.0f} ${ql_h:>11.4f}  {diff_se:>10.1f} SE")

# ============================================================================
# 6. EXOTIC PAYOFFS — VALAX ADVANTAGE
# ============================================================================

print(f"\n{'=' * 70}")
print("EXOTIC PAYOFFS (VALAX)")
print("=" * 70)

# Generate paths for exotics
paths = generate_gbm_paths(bs_model, jnp.array(S0), T, 252, 100_000, key)

# Asian call
asian_opt = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(T), is_call=True)
asian_pays = asian_payoff(paths, asian_opt)
asian_p = float(jnp.exp(-jnp.array(r) * T) * jnp.mean(asian_pays))
asian_se = float(jnp.exp(-jnp.array(r) * T) * jnp.std(asian_pays) / jnp.sqrt(100_000.0))

# Barrier (up-and-out) call
barrier_opt = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(T), is_call=True)
barrier_pays = barrier_payoff(paths, barrier_opt, barrier=jnp.array(130.0), is_up=True, is_knock_in=False)
barrier_p = float(jnp.exp(-jnp.array(r) * T) * jnp.mean(barrier_pays))
barrier_se = float(jnp.exp(-jnp.array(r) * T) * jnp.std(barrier_pays) / jnp.sqrt(100_000.0))

# European reference
euro_pays = european_payoff(paths, asian_opt)
euro_p = float(jnp.exp(-jnp.array(r) * T) * jnp.mean(euro_pays))

print(f"\n{'Payoff Type':<25} {'Price':>10} {'SE':>8}")
print("-" * 45)
print(f"{'European (K=100)':<25} ${euro_p:>9.4f} ${float(jnp.exp(-jnp.array(r)*T)*jnp.std(euro_pays)/jnp.sqrt(100000.0)):>7.4f}")
print(f"{'Asian arith. avg (K=100)':<25} ${asian_p:>9.4f} ${asian_se:>7.4f}")
print(f"{'Up-out barrier (B=130)':<25} ${barrier_p:>9.4f} ${barrier_se:>7.4f}")

print("""
VALAX exotic pricing pattern:
  payoffs = asian_payoff(paths, option)       # pure function on path array
  price = exp(-rT) * mean(payoffs)            # discount and average

Adding a new payoff = writing a single pure function.
No new engine class, no inheritance hierarchy.
""")

# ============================================================================
# 7. API COMPARISON
# ============================================================================

print(f"{'=' * 70}")
print("API DESIGN: Monte Carlo")
print("=" * 70)

print("""
┌────────────────────┬───────────────────────────────────┬──────────────────────────────────┐
│ Aspect             │ VALAX                             │ QuantLib                         │
├────────────────────┼───────────────────────────────────┼──────────────────────────────────┤
│ Path generation    │ generate_gbm_paths(model, S, T,   │ Implicit inside MC engine        │
│                    │   n_steps, n_paths, key)          │                                  │
│ Pricing            │ mc_price(option, S, model, cfg)   │ option.setPricingEngine(MCEngine)│
│ Payoff definition  │ Pure function on path array       │ Payoff class + Engine subclass   │
│ Exotic payoffs     │ Write a function, pass to MC      │ Subclass MCPayoff or PathPayoff  │
│ Heston             │ generate_heston_paths() + payoff  │ HestonProcess + MCEngine         │
│ Standard error     │ mc_price_with_stderr()            │ option.errorEstimate()           │
│ Reproducibility    │ JAX PRNG key (deterministic)      │ Seed parameter                   │
│ Differentiability  │ jax.grad through MC (biased est.) │ Not supported                    │
│ GPU                │ Automatic (JAX backend)           │ CPU only                         │
└────────────────────┴───────────────────────────────────┴──────────────────────────────────┘
""")
