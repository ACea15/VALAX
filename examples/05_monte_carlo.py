# %% [markdown]
# # Monte Carlo Pricing: GBM, Heston, and SABR Paths
#
# This example covers:
# - GBM path generation and European option pricing
# - Heston stochastic volatility paths
# - SABR forward dynamics
# - Convergence analysis: MC vs analytical
# - Exotic payoffs (Asian, barrier)
# - MC standard error estimation

# %% Imports
import jax
import jax.numpy as jnp
from valax.instruments.options import EuropeanOption
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.models.sabr import SABRModel
from valax.pricing.mc.paths import generate_gbm_paths, generate_heston_paths
from valax.pricing.mc.sabr_paths import generate_sabr_paths
from valax.pricing.mc.engine import mc_price, mc_price_with_stderr, MCConfig
from valax.pricing.mc.payoffs import european_payoff, asian_payoff, barrier_payoff
from valax.pricing.analytic.black_scholes import black_scholes_price

# ============================================================================
# 1. GBM PATHS AND EUROPEAN OPTION PRICING
# ============================================================================

# %% Generate GBM paths
bs_model = BlackScholesModel(
    vol=jnp.array(0.25),
    rate=jnp.array(0.04),
    dividend=jnp.array(0.01),
)

key = jax.random.PRNGKey(42)
spot = jnp.array(100.0)
T = 1.0
n_steps = 252    # daily steps
n_paths = 50_000

paths = generate_gbm_paths(bs_model, spot, T, n_steps, n_paths, key)
print(f"GBM paths shape: {paths.shape}  (n_paths x n_steps+1)")
print(f"Initial spot: {float(paths[0, 0]):.2f}")
print(f"Mean terminal: {float(jnp.mean(paths[:, -1])):.2f}")
print(f"Std terminal:  {float(jnp.std(paths[:, -1])):.2f}")

# %% Path statistics
# Under risk-neutral GBM, E[S_T] = S_0 * exp((r-q)*T)
expected_terminal = float(spot) * jnp.exp((bs_model.rate - bs_model.dividend) * T)
print(f"\nRisk-neutral E[S_T]: {float(expected_terminal):.2f}")
print(f"MC mean S_T:         {float(jnp.mean(paths[:, -1])):.2f}")

# %% Price a European call via MC
option = EuropeanOption(strike=jnp.array(105.0), expiry=jnp.array(T), is_call=True)
config = MCConfig(n_paths=50_000, n_steps=252)

mc_p, mc_se = mc_price_with_stderr(option, spot, bs_model, config, key)
bs_p = black_scholes_price(option, spot, bs_model.vol, bs_model.rate, bs_model.dividend)

print(f"\n--- European Call (K=105, T=1Y) ---")
print(f"MC price:     ${float(mc_p):.4f}  ±${float(mc_se):.4f}")
print(f"Analytic BS:  ${float(bs_p):.4f}")
print(f"Error:        ${abs(float(mc_p - bs_p)):.4f}  ({abs(float(mc_p - bs_p))/float(mc_se):.1f} SE)")

# ============================================================================
# 2. CONVERGENCE ANALYSIS
# ============================================================================

# %% MC price converges as n_paths increases
print(f"\n--- Convergence Analysis ---")
print(f"{'n_paths':>10} {'MC price':>10} {'SE':>8} {'Error':>8} {'#SE':>6}")
for n in [1000, 5000, 10_000, 50_000, 100_000]:
    cfg = MCConfig(n_paths=n, n_steps=100)
    p, se = mc_price_with_stderr(option, spot, bs_model, cfg, key)
    err = abs(float(p - bs_p))
    n_se = err / float(se) if float(se) > 0 else 0
    print(f"{n:>10,} {float(p):>10.4f} {float(se):>8.4f} {err:>8.4f} {n_se:>6.1f}")

# ============================================================================
# 3. EXOTIC PAYOFFS
# ============================================================================

# %% Asian option (arithmetic average price)
# The payoff is max(avg(S_path) - K, 0) for a call
asian_call = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(T), is_call=True)

# Manual pricing since mc_price uses european_payoff by default
paths_for_asian = generate_gbm_paths(bs_model, spot, T, 252, 100_000, key)
asian_payoffs = asian_payoff(paths_for_asian, asian_call)
asian_price = float(jnp.exp(-bs_model.rate * T) * jnp.mean(asian_payoffs))
asian_se = float(jnp.exp(-bs_model.rate * T) * jnp.std(asian_payoffs) / jnp.sqrt(100_000.0))

print(f"\n--- Asian Call (arithmetic avg, K=100) ---")
print(f"MC price: ${asian_price:.4f}  ±${asian_se:.4f}")
print(f"(Asian < European since avg reduces vol)")

# European for comparison
euro_payoffs = european_payoff(paths_for_asian, asian_call)
euro_price = float(jnp.exp(-bs_model.rate * T) * jnp.mean(euro_payoffs))
print(f"European: ${euro_price:.4f}")

# %% Up-and-out barrier call
# The option knocks out (becomes worthless) if S ever crosses the barrier
barrier_call = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(T), is_call=True)

# barrier_payoff(paths, option, barrier, is_up, is_out)
barrier_payoffs = barrier_payoff(
    paths_for_asian, barrier_call,
    barrier=jnp.array(130.0),  # knock-out at 130
    is_up=True,                # up barrier
    is_knock_in=False,         # knock-out (not knock-in)
)
barrier_price = float(jnp.exp(-bs_model.rate * T) * jnp.mean(barrier_payoffs))
barrier_se = float(jnp.exp(-bs_model.rate * T) * jnp.std(barrier_payoffs) / jnp.sqrt(100_000.0))

print(f"\n--- Up-and-Out Barrier Call (K=100, B=130) ---")
print(f"MC price: ${barrier_price:.4f}  ±${barrier_se:.4f}")
print(f"(Barrier < European since paths that cross 130 are knocked out)")

# ============================================================================
# 4. HESTON STOCHASTIC VOLATILITY
# ============================================================================

# %% Generate Heston paths
heston = HestonModel(
    v0=jnp.array(0.04),       # initial variance (vol = 20%)
    kappa=jnp.array(2.0),     # mean reversion speed
    theta=jnp.array(0.04),    # long-run variance
    xi=jnp.array(0.5),        # vol of vol
    rho=jnp.array(-0.7),      # spot-vol correlation (negative = leverage effect)
    rate=jnp.array(0.04),
    dividend=jnp.array(0.01),
)

key2 = jax.random.PRNGKey(123)
spot_paths, var_paths = generate_heston_paths(heston, spot, T, 252, 50_000, key2)

print(f"\n--- Heston Paths ---")
print(f"Spot paths: {spot_paths.shape}")
print(f"Var paths:  {var_paths.shape}")
print(f"Mean terminal spot: {float(jnp.mean(spot_paths[:, -1])):.2f}")
print(f"Mean terminal var:  {float(jnp.mean(var_paths[:, -1])):.4f}  (theta={float(heston.theta):.4f})")

# %% Heston MC call price
heston_payoffs = jnp.maximum(spot_paths[:, -1] - 105.0, 0.0)
heston_price = float(jnp.exp(-heston.rate * T) * jnp.mean(heston_payoffs))
heston_se = float(jnp.exp(-heston.rate * T) * jnp.std(heston_payoffs) / jnp.sqrt(50_000.0))

print(f"\nHeston MC call (K=105): ${heston_price:.4f}  ±${heston_se:.4f}")

# %% Compare Heston implied vol smile vs flat BS
# Price calls at different strikes under Heston, then invert to BS implied vol
test_strikes = jnp.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
print(f"\n--- Heston Implied Vol Smile ---")
print(f"{'Strike':>8} {'MC Price':>10} {'Implied Vol':>12}")
from valax.pricing.analytic.black_scholes import black_scholes_implied_vol
for K in test_strikes:
    payoffs = jnp.maximum(spot_paths[:, -1] - K, 0.0)
    mc_call = jnp.exp(-heston.rate * T) * jnp.mean(payoffs)
    opt = EuropeanOption(strike=K, expiry=jnp.array(T), is_call=True)
    iv = black_scholes_implied_vol(opt, spot, heston.rate, heston.dividend, mc_call)
    print(f"{float(K):8.0f} {float(mc_call):10.4f} {float(iv)*100:11.2f}%")

# ============================================================================
# 5. SABR FORWARD DYNAMICS
# ============================================================================

# %% Generate SABR paths (forward + stochastic vol)
sabr = SABRModel(
    alpha=jnp.array(0.25),
    beta=jnp.array(0.5),
    rho=jnp.array(-0.3),
    nu=jnp.array(0.4),
)

forward = jnp.array(100.0)
key3 = jax.random.PRNGKey(456)
fwd_paths, vol_paths = generate_sabr_paths(sabr, forward, T, 200, 50_000, key3)

print(f"\n--- SABR Paths ---")
print(f"Forward paths: {fwd_paths.shape}")
print(f"Vol paths:     {vol_paths.shape}")
print(f"Mean terminal forward: {float(jnp.mean(fwd_paths[:, -1])):.2f}  (should ≈ {float(forward):.0f})")
print(f"Mean terminal alpha:   {float(jnp.mean(vol_paths[:, -1])):.4f}  (init={float(sabr.alpha):.4f})")

# %% SABR MC vs analytic comparison
from valax.pricing.analytic.sabr import sabr_price
r = jnp.array(0.04)
K = jnp.array(105.0)
opt = EuropeanOption(strike=K, expiry=jnp.array(T), is_call=True)

analytic_sabr = sabr_price(opt, forward, r, sabr)
mc_sabr_payoffs = jnp.maximum(fwd_paths[:, -1] - K, 0.0) * jnp.exp(-r * T)
mc_sabr = float(jnp.mean(mc_sabr_payoffs))
mc_sabr_se = float(jnp.std(mc_sabr_payoffs) / jnp.sqrt(50_000.0))

print(f"\n--- SABR: MC vs Analytic (K=105) ---")
print(f"Analytic (Hagan): ${float(analytic_sabr):.4f}")
print(f"MC:               ${mc_sabr:.4f}  ±${mc_sabr_se:.4f}")
print(f"Error:            ${abs(mc_sabr - float(analytic_sabr)):.4f}  ({abs(mc_sabr - float(analytic_sabr))/mc_sabr_se:.1f} SE)")
