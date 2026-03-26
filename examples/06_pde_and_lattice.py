# %% [markdown]
# # PDE and Lattice Methods: Finite Differences and Binomial Trees
#
# This example covers:
# - Crank-Nicolson PDE pricing for European options
# - CRR binomial tree for European and American options
# - American vs European put pricing (early exercise premium)
# - Convergence to Black-Scholes as grid/steps increase
# - Greeks via autodiff through numerical solvers

# %% Imports
import jax
import jax.numpy as jnp
from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.pde.solvers import pde_price, PDEConfig
from valax.pricing.lattice.binomial import binomial_price, BinomialConfig

# Market data: a stock at $100
spot = jnp.array(100.0)
vol = jnp.array(0.25)
rate = jnp.array(0.05)
dividend = jnp.array(0.02)

# ============================================================================
# 1. CRANK-NICOLSON PDE
# ============================================================================

# %% Price a European call via Crank-Nicolson finite differences
call = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)

# Default grid: 200 spatial x 200 time steps
pde_call = pde_price(call, spot, vol, rate, dividend)
bs_call = black_scholes_price(call, spot, vol, rate, dividend)

print("--- Crank-Nicolson PDE ---")
print(f"PDE price:       ${float(pde_call):.4f}")
print(f"BS analytical:   ${float(bs_call):.4f}")
print(f"Error:           ${abs(float(pde_call - bs_call)):.4f}")

# %% European put
put = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)
pde_put = pde_price(put, spot, vol, rate, dividend)
bs_put = black_scholes_price(put, spot, vol, rate, dividend)
print(f"\nPDE put:         ${float(pde_put):.4f}")
print(f"BS put:          ${float(bs_put):.4f}")
print(f"Error:           ${abs(float(pde_put - bs_put)):.4f}")

# %% Grid convergence: finer grids give better accuracy
print(f"\n--- PDE Grid Convergence (European Call) ---")
print(f"{'N_spot':>8} {'N_time':>8} {'PDE Price':>10} {'Error':>10}")
for n in [50, 100, 200, 400]:
    cfg = PDEConfig(n_spot=n, n_time=n)
    p = pde_price(call, spot, vol, rate, dividend, config=cfg)
    err = abs(float(p - bs_call))
    print(f"{n:>8} {n:>8} {float(p):>10.6f} {err:>10.6f}")

# %% PDE Greeks via autodiff
# Yes — autodiff works through the entire Crank-Nicolson solver!
pde_delta = jax.grad(lambda s: pde_price(call, s, vol, rate, dividend))(spot)
pde_gamma = jax.grad(jax.grad(lambda s: pde_price(call, s, vol, rate, dividend)))(spot)
pde_vega = jax.grad(lambda v: pde_price(call, spot, v, rate, dividend))(vol)

bs_delta = jax.grad(lambda s: black_scholes_price(call, s, vol, rate, dividend))(spot)
bs_gamma = jax.grad(jax.grad(lambda s: black_scholes_price(call, s, vol, rate, dividend)))(spot)
bs_vega = jax.grad(lambda v: black_scholes_price(call, spot, v, rate, dividend))(vol)

print(f"\n--- PDE Greeks vs Analytical ---")
print(f"{'Greek':>8} {'PDE':>10} {'BS':>10} {'Error':>10}")
print(f"{'Delta':>8} {float(pde_delta):>10.6f} {float(bs_delta):>10.6f} {abs(float(pde_delta - bs_delta)):>10.6f}")
print(f"{'Gamma':>8} {float(pde_gamma):>10.6f} {float(bs_gamma):>10.6f} {abs(float(pde_gamma - bs_gamma)):>10.6f}")
print(f"{'Vega':>8} {float(pde_vega):>10.4f} {float(bs_vega):>10.4f} {abs(float(pde_vega - bs_vega)):>10.4f}")

# ============================================================================
# 2. CRR BINOMIAL TREE
# ============================================================================

# %% Price European options with the binomial tree
euro_config = BinomialConfig(n_steps=500, american=False)

tree_call = binomial_price(call, spot, vol, rate, dividend, config=euro_config)
tree_put = binomial_price(put, spot, vol, rate, dividend, config=euro_config)

print(f"\n--- CRR Binomial Tree (500 steps) ---")
print(f"Tree call:       ${float(tree_call):.4f}  (BS: ${float(bs_call):.4f})")
print(f"Tree put:        ${float(tree_put):.4f}  (BS: ${float(bs_put):.4f})")

# %% Convergence of binomial tree
print(f"\n--- Binomial Tree Convergence ---")
print(f"{'Steps':>8} {'Tree Price':>10} {'Error':>10}")
for n in [50, 100, 200, 500, 1000]:
    cfg = BinomialConfig(n_steps=n, american=False)
    p = binomial_price(call, spot, vol, rate, dividend, config=cfg)
    err = abs(float(p - bs_call))
    print(f"{n:>8} {float(p):>10.6f} {err:>10.6f}")

# ============================================================================
# 3. AMERICAN OPTIONS (EARLY EXERCISE)
# ============================================================================

# %% American put — can be exercised early, so worth more than European put
american_config = BinomialConfig(n_steps=500, american=True)

american_put = binomial_price(put, spot, vol, rate, dividend, config=american_config)
european_put = binomial_price(put, spot, vol, rate, dividend, config=euro_config)

print(f"\n--- American vs European Put (ATM, K=100) ---")
print(f"European put: ${float(european_put):.4f}")
print(f"American put: ${float(american_put):.4f}")
print(f"Early exercise premium: ${float(american_put - european_put):.4f}")

# %% American call on a stock with dividends
# American call is worth more than European only if dividends > 0
american_call = binomial_price(call, spot, vol, rate, dividend, config=american_config)
european_call = binomial_price(call, spot, vol, rate, dividend, config=euro_config)

print(f"\n--- American vs European Call (with 2% dividend) ---")
print(f"European call: ${float(european_call):.4f}")
print(f"American call: ${float(american_call):.4f}")
print(f"Early exercise premium: ${float(american_call - european_call):.4f}")

# American call with zero dividend — should equal European
call_no_div = binomial_price(call, spot, vol, rate, jnp.array(0.0), config=american_config)
euro_no_div = binomial_price(call, spot, vol, rate, jnp.array(0.0), config=euro_config)
print(f"\nWith 0% dividend:")
print(f"  European: ${float(euro_no_div):.4f}  American: ${float(call_no_div):.4f}  "
      f"Premium: ${abs(float(call_no_div - euro_no_div)):.4f}")

# %% American put: early exercise value across strikes
# Deep ITM American puts have the most early exercise premium
print(f"\n--- Early Exercise Premium vs Strike ---")
print(f"{'Strike':>8} {'Euro Put':>10} {'Amer Put':>10} {'Premium':>10} {'Premium %':>10}")
for K_val in [80.0, 90.0, 100.0, 110.0, 120.0]:
    opt = EuropeanOption(strike=jnp.array(K_val), expiry=jnp.array(1.0), is_call=False)
    euro_p = binomial_price(opt, spot, vol, rate, dividend, config=euro_config)
    amer_p = binomial_price(opt, spot, vol, rate, dividend, config=american_config)
    prem = float(amer_p - euro_p)
    prem_pct = prem / float(euro_p) * 100 if float(euro_p) > 0.01 else 0.0
    print(f"{K_val:>8.0f} {float(euro_p):>10.4f} {float(amer_p):>10.4f} {prem:>10.4f} {prem_pct:>9.2f}%")

# %% Binomial tree Greeks via autodiff
# Autodiff works through the entire tree rollback via jax.lax.scan!
tree_delta = jax.grad(lambda s: binomial_price(put, s, vol, rate, dividend, config=american_config))(spot)
tree_gamma = jax.grad(jax.grad(lambda s: binomial_price(put, s, vol, rate, dividend, config=american_config)))(spot)

print(f"\n--- American Put Greeks (autodiff through binomial tree) ---")
print(f"Delta: {float(tree_delta):.4f}  (negative for put)")
print(f"Gamma: {float(tree_gamma):.6f}")

# ============================================================================
# 4. METHOD COMPARISON
# ============================================================================

# %% Compare all three methods for a European call
print(f"\n--- Method Comparison: European Call (K=100, T=1Y) ---")
print(f"{'Method':>20} {'Price':>10} {'Delta':>10} {'Gamma':>10}")

methods = {
    "Black-Scholes": lambda s, v: black_scholes_price(call, s, v, rate, dividend),
    "Crank-Nicolson": lambda s, v: pde_price(call, s, v, rate, dividend, PDEConfig(n_spot=300, n_time=300)),
    "Binomial (500)": lambda s, v: binomial_price(call, s, v, rate, dividend, BinomialConfig(n_steps=500)),
}

for name, fn in methods.items():
    p = fn(spot, vol)
    d = jax.grad(fn, argnums=0)(spot, vol)
    g = jax.grad(jax.grad(fn, argnums=0), argnums=0)(spot, vol)
    print(f"{name:>20} {float(p):>10.4f} {float(d):>10.6f} {float(g):>10.6f}")
