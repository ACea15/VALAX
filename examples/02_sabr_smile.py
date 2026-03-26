# %% [markdown]
# # SABR Volatility Smile and Calibration
#
# This example covers:
# - The SABR stochastic volatility model
# - Generating implied vol smiles from SABR parameters
# - Effect of each parameter (alpha, beta, rho, nu) on the smile
# - Calibrating SABR to market data
# - Greeks through the full SABR -> Black-76 chain

# %% Imports
import jax
import jax.numpy as jnp
from valax.models.sabr import SABRModel
from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_price
from valax.calibration.sabr import calibrate_sabr

# ============================================================================
# 1. SABR MODEL BASICS
# ============================================================================

# %% Define a SABR model
# The SABR model has 4 parameters:
#   alpha: initial volatility level
#   beta:  CEV backbone (0 = normal, 0.5 = typical equity, 1 = lognormal)
#   rho:   correlation between forward and vol Brownians (negative = skew)
#   nu:    vol-of-vol (controls smile curvature / wings)

model = SABRModel(
    alpha=jnp.array(0.25),
    beta=jnp.array(0.5),
    rho=jnp.array(-0.30),
    nu=jnp.array(0.40),
)

forward = jnp.array(100.0)
expiry = jnp.array(1.0)

# %% Compute implied vol at a single strike
vol_atm = sabr_implied_vol(model, forward, jnp.array(100.0), expiry)
vol_otm = sabr_implied_vol(model, forward, jnp.array(120.0), expiry)
print(f"ATM vol:      {float(vol_atm)*100:.2f}%")
print(f"OTM call vol: {float(vol_otm)*100:.2f}%")

# %% Generate a full smile across strikes
strikes = jnp.linspace(70.0, 130.0, 25)
smile = jax.vmap(lambda K: sabr_implied_vol(model, forward, K, expiry))(strikes)

print("\n--- SABR Implied Vol Smile ---")
print(f"{'Strike':>8} {'IV (%)':>8}")
for i in range(0, len(strikes), 4):  # print every 4th
    print(f"{float(strikes[i]):8.1f} {float(smile[i])*100:8.2f}")

# ============================================================================
# 2. PARAMETER SENSITIVITY
# ============================================================================

# %% Effect of rho (skew parameter)
# Negative rho => higher vol at low strikes (equity-like downside skew)
# Positive rho => higher vol at high strikes
print("\n--- Effect of rho on the smile ---")
for rho_val in [-0.5, -0.25, 0.0, 0.25, 0.5]:
    m = SABRModel(alpha=jnp.array(0.25), beta=jnp.array(0.5),
                  rho=jnp.array(rho_val), nu=jnp.array(0.4))
    vol_low = sabr_implied_vol(m, forward, jnp.array(80.0), expiry)
    vol_atm = sabr_implied_vol(m, forward, jnp.array(100.0), expiry)
    vol_high = sabr_implied_vol(m, forward, jnp.array(120.0), expiry)
    print(f"  rho={rho_val:+.2f}:  K=80 → {float(vol_low)*100:.1f}%  "
          f"K=100 → {float(vol_atm)*100:.1f}%  "
          f"K=120 → {float(vol_high)*100:.1f}%")

# %% Effect of nu (vol-of-vol)
# Higher nu => wider smile, fatter tails
print("\n--- Effect of nu on OTM wing vols ---")
for nu_val in [0.1, 0.3, 0.5, 0.8]:
    m = SABRModel(alpha=jnp.array(0.25), beta=jnp.array(0.5),
                  rho=jnp.array(-0.3), nu=jnp.array(nu_val))
    vol_80 = sabr_implied_vol(m, forward, jnp.array(80.0), expiry)
    vol_120 = sabr_implied_vol(m, forward, jnp.array(120.0), expiry)
    print(f"  nu={nu_val:.1f}:  K=80 → {float(vol_80)*100:.1f}%  K=120 → {float(vol_120)*100:.1f}%")

# %% Effect of beta (CEV backbone)
# beta=0: normal model (vol quoted in absolute terms)
# beta=0.5: square-root model (common for equity)
# beta=1: lognormal model (pure stochastic vol)
print("\n--- Effect of beta on ATM vol level ---")
for beta_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
    # Adjust alpha so ATM vol is roughly comparable
    alpha_adj = 0.25 * forward ** (0.5 - beta_val) if beta_val != 0.5 else 0.25
    m = SABRModel(alpha=jnp.array(alpha_adj), beta=jnp.array(beta_val),
                  rho=jnp.array(-0.3), nu=jnp.array(0.4))
    vol_atm = sabr_implied_vol(m, forward, forward, expiry)
    print(f"  beta={beta_val:.2f}:  ATM vol → {float(vol_atm)*100:.2f}%")

# ============================================================================
# 3. SABR PRICING AND GREEKS
# ============================================================================

# %% Price an option under SABR
# Under the hood: SABR implied vol → Black-76 → option price
option = EuropeanOption(strike=jnp.array(105.0), expiry=expiry, is_call=True)
price = sabr_price(option, forward, jnp.array(0.04), model)
print(f"\nSABR call price (K=105): ${float(price):.4f}")

# %% Greeks through the full SABR chain via jax.grad
# Delta: sensitivity to forward price
delta_fn = jax.grad(lambda f: sabr_price(option, f, jnp.array(0.04), model))
delta = delta_fn(forward)
print(f"SABR delta: {float(delta):.4f}")

# Gamma
gamma_fn = jax.grad(jax.grad(lambda f: sabr_price(option, f, jnp.array(0.04), model)))
gamma = gamma_fn(forward)
print(f"SABR gamma: {float(gamma):.6f}")

# %% Sensitivity to SABR parameters (model risk / calibration risk)
# d(price)/d(alpha) — how much does the price change if alpha is wrong?
def price_from_alpha(alpha):
    m = SABRModel(alpha=alpha, beta=model.beta, rho=model.rho, nu=model.nu)
    return sabr_price(option, forward, jnp.array(0.04), m)

dalpha = jax.grad(price_from_alpha)(model.alpha)

# d(price)/d(rho)
def price_from_rho(rho):
    m = SABRModel(alpha=model.alpha, beta=model.beta, rho=rho, nu=model.nu)
    return sabr_price(option, forward, jnp.array(0.04), m)

drho = jax.grad(price_from_rho)(model.rho)

print(f"\nModel sensitivities:")
print(f"  d(price)/d(alpha) = {float(dalpha):.4f}")
print(f"  d(price)/d(rho)   = {float(drho):.4f}")

# ============================================================================
# 4. SABR CALIBRATION
# ============================================================================

# %% Generate synthetic "market" data from a known model
true_model = SABRModel(
    alpha=jnp.array(0.22),
    beta=jnp.array(0.5),      # we'll fix beta during calibration
    rho=jnp.array(-0.28),
    nu=jnp.array(0.38),
)

cal_strikes = jnp.linspace(75.0, 125.0, 11)
market_vols = jax.vmap(
    lambda K: sabr_implied_vol(true_model, forward, K, expiry)
)(cal_strikes)

print("\n--- Synthetic Market Data ---")
print(f"{'Strike':>8} {'Market IV (%)':>12}")
for i in range(len(cal_strikes)):
    print(f"{float(cal_strikes[i]):8.1f} {float(market_vols[i])*100:12.2f}")

# %% Calibrate SABR — fix beta=0.5, fit alpha/rho/nu
fitted, sol = calibrate_sabr(
    cal_strikes, market_vols, forward, expiry,
    fixed_beta=jnp.array(0.5),
    solver="levenberg_marquardt",
)

print(f"\n--- Calibration Results (Levenberg-Marquardt) ---")
print(f"  alpha: true={float(true_model.alpha):.4f}  fitted={float(fitted.alpha):.4f}")
print(f"  rho:   true={float(true_model.rho):.4f}  fitted={float(fitted.rho):.4f}")
print(f"  nu:    true={float(true_model.nu):.4f}  fitted={float(fitted.nu):.4f}")
print(f"  beta:  fixed={float(fitted.beta):.1f}")

# %% Verify the fit by comparing vols
fitted_vols = jax.vmap(
    lambda K: sabr_implied_vol(fitted, forward, K, expiry)
)(cal_strikes)

max_error = float(jnp.max(jnp.abs(fitted_vols - market_vols)))
print(f"\nMax vol error: {max_error*10000:.2f} bps")

# %% Calibrate with BFGS for comparison
fitted_bfgs, _ = calibrate_sabr(
    cal_strikes, market_vols, forward, expiry,
    fixed_beta=jnp.array(0.5),
    solver="bfgs",
)
print(f"\nBFGS result:  alpha={float(fitted_bfgs.alpha):.4f}  "
      f"rho={float(fitted_bfgs.rho):.4f}  nu={float(fitted_bfgs.nu):.4f}")

# %% Weighted calibration — emphasize ATM strikes
weights = jnp.exp(-0.5 * ((cal_strikes - forward) / 10.0) ** 2)
fitted_w, _ = calibrate_sabr(
    cal_strikes, market_vols, forward, expiry,
    fixed_beta=jnp.array(0.5),
    weights=weights,
)

# Compare ATM fit quality
atm_idx = jnp.argmin(jnp.abs(cal_strikes - forward))
vol_uniform = sabr_implied_vol(fitted, forward, cal_strikes[atm_idx], expiry)
vol_weighted = sabr_implied_vol(fitted_w, forward, cal_strikes[atm_idx], expiry)
print(f"\nATM fit comparison:")
print(f"  Market:   {float(market_vols[atm_idx])*10000:.2f} bps")
print(f"  Uniform:  {float(vol_uniform)*10000:.2f} bps  (error: {abs(float(vol_uniform - market_vols[atm_idx]))*10000:.4f} bps)")
print(f"  Weighted: {float(vol_weighted)*10000:.2f} bps  (error: {abs(float(vol_weighted - market_vols[atm_idx]))*10000:.4f} bps)")
