# %% [markdown]
# # VALAX vs QuantLib: PDE and Lattice Methods
#
# Side-by-side comparison of:
# - Crank-Nicolson finite differences vs QuantLib FD engine
# - CRR binomial tree vs QuantLib binomial engine
# - American option pricing
# - Convergence analysis
# - Greeks via autodiff through numerical solvers (VALAX advantage)
#
# Validated by: tests/test_quantlib_comparison/test_pde_lattice_ql.py

# %% Imports
import jax
import jax.numpy as jnp
import QuantLib as ql
from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.pde.solvers import pde_price, PDEConfig
from valax.pricing.lattice.binomial import binomial_price, BinomialConfig

# ============================================================================
# 1. COMMON PARAMETERS
# ============================================================================

print("=" * 70)
print("VALAX vs QuantLib: PDE and Lattice Methods")
print("=" * 70)

S0 = 100.0
K = 100.0
T = 1.0
sigma = 0.25
r = 0.05
q = 0.02

spot = jnp.array(S0)
vol = jnp.array(sigma)
rate = jnp.array(r)
dividend = jnp.array(q)

call = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=True)
put = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=False)

bs_call = float(black_scholes_price(call, spot, vol, rate, dividend))
bs_put = float(black_scholes_price(put, spot, vol, rate, dividend))

print(f"\nParameters: S={S0}, K={K}, T={T}, σ={sigma}, r={r}, q={q}")
print(f"BS reference: call=${bs_call:.6f}, put=${bs_put:.6f}")

# QuantLib setup
today_ql = ql.Date(26, 3, 2026)
ql.Settings.instance().evaluationDate = today_ql
maturity_ql = today_ql + ql.Period(1, ql.Years)

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

# ============================================================================
# 2. CRANK-NICOLSON PDE
# ============================================================================

print(f"\n{'=' * 70}")
print("CRANK-NICOLSON PDE: European Call")
print("=" * 70)

# --- VALAX PDE ---
print(f"\n--- Grid Convergence ---")
print(f"{'Grid':>10} {'VALAX PDE':>12} {'Error':>10}")
for n in [50, 100, 200, 400]:
    cfg = PDEConfig(n_spot=n, n_time=n)
    p = float(pde_price(call, spot, vol, rate, dividend, config=cfg))
    print(f"{f'{n}x{n}':>10} {p:>12.6f} {abs(p - bs_call):>10.6f}")

# --- QuantLib FD ---
call_payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
exercise = ql.EuropeanExercise(maturity_ql)
option_ql = ql.VanillaOption(call_payoff, exercise)

fd_engine = ql.FdBlackScholesVanillaEngine(bsm_process, 200, 200)
option_ql.setPricingEngine(fd_engine)
ql_pde = option_ql.NPV()

valax_pde = float(pde_price(call, spot, vol, rate, dividend, config=PDEConfig(n_spot=200, n_time=200)))

print(f"\n{'Method':<18} {'Price':>12} {'Error vs BS':>12}")
print("-" * 44)
print(f"{'VALAX (200x200)':<18} {valax_pde:>12.6f} {abs(valax_pde - bs_call):>12.6f}")
print(f"{'QL FD (200x200)':<18} {ql_pde:>12.6f} {abs(ql_pde - bs_call):>12.6f}")
print(f"{'BS Analytical':<18} {bs_call:>12.6f} {'—':>12}")

# ============================================================================
# 3. PDE GREEKS — VALAX AUTODIFF ADVANTAGE
# ============================================================================

print(f"\n{'=' * 70}")
print("PDE GREEKS: Autodiff vs QuantLib")
print("=" * 70)

# VALAX: autodiff through the entire Crank-Nicolson solver
pde_delta_v = float(jax.grad(lambda s: pde_price(call, s, vol, rate, dividend))(spot))
pde_gamma_v = float(jax.grad(jax.grad(lambda s: pde_price(call, s, vol, rate, dividend)))(spot))
pde_vega_v = float(jax.grad(lambda v: pde_price(call, spot, v, rate, dividend))(vol))

# QuantLib: FD engine provides Greeks via finite differences on the grid
ql_delta = option_ql.delta()
ql_gamma = option_ql.gamma()
# Note: QL FD engine doesn't always provide vega; skip it
ql_vega = None

# BS reference
bs_delta = float(jax.grad(lambda s: black_scholes_price(call, s, vol, rate, dividend))(spot))
bs_gamma = float(jax.grad(jax.grad(lambda s: black_scholes_price(call, s, vol, rate, dividend)))(spot))
bs_vega = float(jax.grad(lambda v: black_scholes_price(call, spot, v, rate, dividend))(vol))

print(f"\n{'Greek':<10} {'VALAX PDE':>12} {'QL FD':>12} {'BS Exact':>12}")
print("-" * 48)
print(f"{'Delta':<10} {pde_delta_v:>12.6f} {ql_delta:>12.6f} {bs_delta:>12.6f}")
print(f"{'Gamma':<10} {pde_gamma_v:>12.6f} {ql_gamma:>12.6f} {bs_gamma:>12.6f}")
vega_ql_str = f"{ql_vega:>12.4f}" if ql_vega is not None else f"{'N/A':>12}"
print(f"{'Vega':<10} {pde_vega_v:>12.4f} {vega_ql_str} {bs_vega:>12.4f}")

print("""
VALAX computes PDE Greeks via jax.grad through the solver.
QuantLib FD engine computes them from the grid solution.
Both converge to BS analytical as the grid refines.
""")

# ============================================================================
# 4. CRR BINOMIAL TREE — EUROPEAN OPTIONS
# ============================================================================

print(f"{'=' * 70}")
print("CRR BINOMIAL TREE: European Call")
print("=" * 70)

# --- VALAX ---
print(f"\n--- Convergence ---")
print(f"{'Steps':>8} {'VALAX':>12} {'Error':>10}")
for n in [50, 100, 200, 500]:
    cfg = BinomialConfig(n_steps=n, american=False)
    p = float(binomial_price(call, spot, vol, rate, dividend, config=cfg))
    print(f"{n:>8} {p:>12.6f} {abs(p - bs_call):>10.6f}")

# --- QuantLib ---
tree_engine_ql = ql.BinomialVanillaEngine(bsm_process, "crr", 500)
option_ql.setPricingEngine(tree_engine_ql)
ql_tree = option_ql.NPV()

valax_tree = float(binomial_price(call, spot, vol, rate, dividend, config=BinomialConfig(n_steps=500)))

print(f"\n{'Method':<18} {'Price':>12} {'Error vs BS':>12}")
print("-" * 44)
print(f"{'VALAX (500)':<18} {valax_tree:>12.6f} {abs(valax_tree - bs_call):>12.6f}")
print(f"{'QL CRR (500)':<18} {ql_tree:>12.6f} {abs(ql_tree - bs_call):>12.6f}")
print(f"{'BS Analytical':<18} {bs_call:>12.6f} {'—':>12}")

# ============================================================================
# 5. AMERICAN PUT — EARLY EXERCISE
# ============================================================================

print(f"\n{'=' * 70}")
print("AMERICAN PUT: Early Exercise Premium")
print("=" * 70)

# VALAX
euro_put_v = float(binomial_price(put, spot, vol, rate, dividend,
                                    config=BinomialConfig(n_steps=500, american=False)))
amer_put_v = float(binomial_price(put, spot, vol, rate, dividend,
                                    config=BinomialConfig(n_steps=500, american=True)))

# QuantLib
put_payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
amer_exercise = ql.AmericanExercise(today_ql, maturity_ql)
amer_put_ql = ql.VanillaOption(put_payoff, amer_exercise)
amer_engine = ql.BinomialVanillaEngine(bsm_process, "crr", 500)
amer_put_ql.setPricingEngine(amer_engine)
ql_amer_put = amer_put_ql.NPV()

euro_put_ql = ql.VanillaOption(put_payoff, exercise)
euro_put_ql.setPricingEngine(tree_engine_ql)
ql_euro_put = euro_put_ql.NPV()

print(f"\n{'Metric':<25} {'VALAX':>12} {'QuantLib':>12} {'Diff':>12}")
print("-" * 63)
print(f"{'European Put':<25} {euro_put_v:>12.4f} {ql_euro_put:>12.4f} {abs(euro_put_v-ql_euro_put):>12.6f}")
print(f"{'American Put':<25} {amer_put_v:>12.4f} {ql_amer_put:>12.4f} {abs(amer_put_v-ql_amer_put):>12.6f}")
print(f"{'Early Exercise Premium':<25} {amer_put_v-euro_put_v:>12.4f} {ql_amer_put-ql_euro_put:>12.4f} "
      f"{abs((amer_put_v-euro_put_v)-(ql_amer_put-ql_euro_put)):>12.6f}")

# --- Across strikes ---
print(f"\n--- Early Exercise Premium Across Strikes ---")
print(f"{'Strike':>8} {'VALAX Prem.':>14} {'QL Prem.':>14} {'Diff':>10}")
print("-" * 48)
for K_val in [80.0, 90.0, 100.0, 110.0, 120.0]:
    opt = EuropeanOption(strike=jnp.array(K_val), expiry=jnp.array(T), is_call=False)
    v_euro = float(binomial_price(opt, spot, vol, rate, dividend, config=BinomialConfig(n_steps=500, american=False)))
    v_amer = float(binomial_price(opt, spot, vol, rate, dividend, config=BinomialConfig(n_steps=500, american=True)))

    ql_payoff = ql.PlainVanillaPayoff(ql.Option.Put, K_val)
    ql_euro = ql.VanillaOption(ql_payoff, exercise)
    ql_euro.setPricingEngine(tree_engine_ql)
    ql_amer = ql.VanillaOption(ql_payoff, amer_exercise)
    ql_amer.setPricingEngine(amer_engine)

    v_prem = v_amer - v_euro
    q_prem = ql_amer.NPV() - ql_euro.NPV()
    print(f"{K_val:8.0f} {v_prem:>14.4f} {q_prem:>14.4f} {abs(v_prem - q_prem):>10.4f}")

# ============================================================================
# 6. AUTODIFF THROUGH BINOMIAL TREE
# ============================================================================

print(f"\n{'=' * 70}")
print("AUTODIFF THROUGH BINOMIAL TREE: American Put Greeks")
print("=" * 70)

# VALAX: autodiff through jax.lax.scan rollback
tree_delta = float(jax.grad(
    lambda s: binomial_price(put, s, vol, rate, dividend, config=BinomialConfig(n_steps=500, american=True))
)(spot))
tree_gamma = float(jax.grad(jax.grad(
    lambda s: binomial_price(put, s, vol, rate, dividend, config=BinomialConfig(n_steps=500, american=True))
))(spot))
tree_vega = float(jax.grad(
    lambda v: binomial_price(put, spot, v, rate, dividend, config=BinomialConfig(n_steps=500, american=True))
)(vol))

# QuantLib American put Greeks
ql_amer_delta = amer_put_ql.delta()
ql_amer_gamma = amer_put_ql.gamma()
# QL binomial engine doesn't provide vega for American options
try:
    ql_amer_vega = amer_put_ql.vega()
except RuntimeError:
    ql_amer_vega = None

print(f"\n{'Greek':<10} {'VALAX':>12} {'QuantLib':>12}")
print("-" * 36)
print(f"{'Delta':<10} {tree_delta:>12.6f} {ql_amer_delta:>12.6f}")
print(f"{'Gamma':<10} {tree_gamma:>12.6f} {ql_amer_gamma:>12.6f}")
vega_ql_str = f"{ql_amer_vega:>12.4f}" if ql_amer_vega is not None else f"{'N/A':>12}"
print(f"{'Vega':<10} {tree_vega:>12.4f} {vega_ql_str}")

print("""
VALAX advantage:
  → American option Greeks via autodiff through the entire tree rollback
  → jax.grad differentiates through jax.lax.scan (backward induction)
  → Works for ANY payoff — no need for special-case formulas
  → Higher-order Greeks (gamma, speed, etc.) are just nested jax.grad

QuantLib:
  → Greeks from the tree grid solution (finite differences on tree)
  → Works well but limited to what the engine computes
""")

# ============================================================================
# 7. API COMPARISON
# ============================================================================

print(f"{'=' * 70}")
print("API DESIGN: PDE & Lattice")
print("=" * 70)

print("""
┌──────────────────┬──────────────────────────────────┬──────────────────────────────────────┐
│ Aspect           │ VALAX                            │ QuantLib                             │
├──────────────────┼──────────────────────────────────┼──────────────────────────────────────┤
│ PDE pricing      │ pde_price(option, S, σ, r, q)    │ FdBlackScholesVanillaEngine(process) │
│ PDE config       │ PDEConfig(n_spot=200, n_time=200)│ FdEngine(process, tGrid, xGrid)      │
│ Binomial pricing │ binomial_price(opt, S, σ, r, q)  │ BinomialVanillaEngine(process, type) │
│ American options │ BinomialConfig(american=True)    │ AmericanExercise(start, end)          │
│ PDE Greeks       │ jax.grad through solver (exact)  │ From grid solution (FD approx)        │
│ Tree Greeks      │ jax.grad through rollback (exact)│ From tree (FD approx)                 │
│ Higher Greeks    │ Nested jax.grad (any order)      │ Not available for American             │
│ Custom payoffs   │ Pass any pure function           │ Subclass engine/instrument             │
└──────────────────┴──────────────────────────────────┴──────────────────────────────────────┘
""")
