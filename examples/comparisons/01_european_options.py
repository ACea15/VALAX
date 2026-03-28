# %% [markdown]
# # VALAX vs QuantLib: European Options
#
# Side-by-side comparison of:
# - European call/put pricing (Black-Scholes)
# - Greeks computation (autodiff vs analytic formulas)
# - Implied volatility inversion
# - Performance (JIT-compiled JAX vs QuantLib C++)
#
# Validated by: tests/test_quantlib_comparison/test_european_options_ql.py

# %% Imports
import time
import jax
import jax.numpy as jnp
import QuantLib as ql
from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price, black_scholes_implied_vol
from valax.greeks.autodiff import greeks

# ============================================================================
# 1. SETUP — IDENTICAL MARKET DATA
# ============================================================================

# Market parameters
S0 = 105.0       # spot price
K = 100.0        # strike
T = 1.0          # 1 year to expiry
sigma = 0.25     # 25% vol
r = 0.04         # 4% risk-free rate
q = 0.01         # 1% continuous dividend yield

print("=" * 70)
print("VALAX vs QuantLib: European Options Comparison")
print("=" * 70)
print(f"\nMarket data: S={S0}, K={K}, T={T}, σ={sigma}, r={r}, q={q}\n")

# ============================================================================
# 2. VALAX PRICING
# ============================================================================

print("-" * 70)
print("VALAX (JAX autodiff)")
print("-" * 70)

# Define instrument — data-only pytree, no pricing engine
call_valax = EuropeanOption(
    strike=jnp.array(K),
    expiry=jnp.array(T),
    is_call=True,
)
put_valax = EuropeanOption(
    strike=jnp.array(K),
    expiry=jnp.array(T),
    is_call=False,
)

# Pure function call: price = f(instrument, market_data)
spot = jnp.array(S0)
vol = jnp.array(sigma)
rate = jnp.array(r)
dividend = jnp.array(q)

valax_call = float(black_scholes_price(call_valax, spot, vol, rate, dividend))
valax_put = float(black_scholes_price(put_valax, spot, vol, rate, dividend))
print(f"Call price: ${valax_call:.6f}")
print(f"Put price:  ${valax_put:.6f}")

# Greeks via autodiff — exact, not finite differences
g = greeks(black_scholes_price, call_valax, spot, vol, rate, dividend)
print(f"\nGreeks (call):")
print(f"  Delta:  {float(g['delta']):.6f}")
print(f"  Gamma:  {float(g['gamma']):.6f}")
print(f"  Vega:   {float(g['vega']):.6f}")
print(f"  Theta:  {float(g['theta']):.6f}  (per year, i.e. d(P)/d(T))")
print(f"  Rho:    {float(g['rho']):.6f}")

# Higher-order Greeks — just compose jax.grad
vanna_fn = jax.grad(jax.grad(
    lambda s, v: black_scholes_price(call_valax, s, v, rate, dividend)
))
valax_vanna = float(vanna_fn(spot, vol))
print(f"  Vanna:  {valax_vanna:.6f}")

volga_fn = jax.grad(jax.grad(
    lambda v: black_scholes_price(call_valax, spot, v, rate, dividend)
))
valax_volga = float(volga_fn(vol))
print(f"  Volga:  {valax_volga:.6f}")

# ============================================================================
# 3. QUANTLIB PRICING
# ============================================================================

print(f"\n{'-' * 70}")
print("QuantLib (C++ engine)")
print("-" * 70)

# QuantLib requires building objects: calendar, dates, handles, engine
today_ql = ql.Date(26, 3, 2026)
ql.Settings.instance().evaluationDate = today_ql

# Market data handles
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

# Black-Scholes-Merton process
bsm_process = ql.BlackScholesMertonProcess(
    spot_handle, div_handle, rate_handle, vol_handle
)

# Build instruments + attach pricing engine
maturity_date = today_ql + ql.Period(1, ql.Years)
exercise = ql.EuropeanExercise(maturity_date)

call_payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
put_payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)

call_ql = ql.VanillaOption(call_payoff, exercise)
put_ql = ql.VanillaOption(put_payoff, exercise)

engine = ql.AnalyticEuropeanEngine(bsm_process)
call_ql.setPricingEngine(engine)
put_ql.setPricingEngine(engine)

ql_call = call_ql.NPV()
ql_put = put_ql.NPV()
print(f"Call price: ${ql_call:.6f}")
print(f"Put price:  ${ql_put:.6f}")

# Greeks — QuantLib computes them via analytic formulas (not autodiff)
# Note: QuantLib vega = per 1% vol shift, rho = per 1% rate shift
# To compare with VALAX (derivatives w.r.t. decimal), multiply QL by 100
ql_delta = call_ql.delta()
ql_gamma = call_ql.gamma()
ql_vega = call_ql.vega()          # per 1% → multiply by 100 for per-decimal
ql_theta_day = call_ql.thetaPerDay()
ql_rho = call_ql.rho()            # per 1% → multiply by 100 for per-decimal

print(f"\nGreeks (call) — QuantLib native units:")
print(f"  Delta:       {ql_delta:.6f}")
print(f"  Gamma:       {ql_gamma:.6f}")
print(f"  Vega:        {ql_vega:.6f}  (per 1% vol shift)")
print(f"  Theta/day:   {ql_theta_day:.6f}")
print(f"  Rho:         {ql_rho:.6f}  (per 1% rate shift)")

# ============================================================================
# 4. HEAD-TO-HEAD COMPARISON
# ============================================================================

print(f"\n{'=' * 70}")
print("HEAD-TO-HEAD COMPARISON")
print("=" * 70)
# Both libraries use the same units for delta/gamma/vega/rho (per-decimal).
# Only theta differs: VALAX = d(P)/d(T) per year, QL = per calendar day.
v_delta, v_gamma = float(g['delta']), float(g['gamma'])
v_vega, v_theta, v_rho = float(g['vega']), float(g['theta']), float(g['rho'])
v_theta_day = v_theta / 365.0  # convert annual to daily for comparison

print(f"\n{'Metric':<20} {'VALAX':>14} {'QuantLib':>14} {'Diff':>14}")
print("-" * 62)
print(f"{'Call Price':<20} {valax_call:>14.6f} {ql_call:>14.6f} {abs(valax_call - ql_call):>14.2e}")
print(f"{'Put Price':<20} {valax_put:>14.6f} {ql_put:>14.6f} {abs(valax_put - ql_put):>14.2e}")
print(f"{'Delta':<20} {v_delta:>14.6f} {ql_delta:>14.6f} {abs(v_delta - ql_delta):>14.2e}")
print(f"{'Gamma':<20} {v_gamma:>14.6f} {ql_gamma:>14.6f} {abs(v_gamma - ql_gamma):>14.2e}")
print(f"{'Vega':<20} {v_vega:>14.6f} {ql_vega:>14.6f} {abs(v_vega - ql_vega):>14.2e}")
print(f"{'Theta (daily)':<20} {v_theta_day:>14.6f} {ql_theta_day:>14.6f} {abs(v_theta_day - ql_theta_day):>14.2e}")
print(f"{'Rho':<20} {v_rho:>14.6f} {ql_rho:>14.6f} {abs(v_rho - ql_rho):>14.2e}")

# ============================================================================
# 5. WHAT VALAX CAN DO THAT QUANTLIB CAN'T (EASILY)
# ============================================================================

print(f"\n{'=' * 70}")
print("AUTODIFF ADVANTAGE: Higher-Order Greeks")
print("=" * 70)

# Third-order Greek: speed = d³P/dS³
speed_fn = jax.grad(jax.grad(jax.grad(
    lambda s: black_scholes_price(call_valax, s, vol, rate, dividend)
)))
speed = float(speed_fn(spot))
print(f"\nSpeed (d³P/dS³):  {speed:.8f}")
print(f"Vanna (d²P/dSdσ): {valax_vanna:.6f}")
print(f"Volga (d²P/dσ²):  {valax_volga:.6f}")
print("\n→ These require no formulas — just compose jax.grad.")
print("→ In QuantLib, you'd need to implement finite differences or")
print("  derive and code each formula by hand.")

# ============================================================================
# 6. IMPLIED VOLATILITY
# ============================================================================

print(f"\n{'=' * 70}")
print("IMPLIED VOLATILITY")
print("=" * 70)

# VALAX: Newton-Raphson with autodiff Jacobian
recovered_vol_valax = float(black_scholes_implied_vol(
    call_valax, spot, rate, dividend, jnp.array(valax_call)
))

# QuantLib: built-in implied vol
recovered_vol_ql = call_ql.impliedVolatility(ql_call, bsm_process, 1e-8, 1000, 0.001, 4.0)

print(f"\nInput vol:                  {sigma:.6f}")
print(f"VALAX recovered vol:        {recovered_vol_valax:.6f}  (error: {abs(recovered_vol_valax - sigma):.2e})")
print(f"QuantLib recovered vol:     {recovered_vol_ql:.6f}  (error: {abs(recovered_vol_ql - sigma):.2e})")

# ============================================================================
# 7. PERFORMANCE
# ============================================================================

print(f"\n{'=' * 70}")
print("PERFORMANCE: JIT-Compiled JAX vs QuantLib C++")
print("=" * 70)

n_iters = 50_000

# VALAX: JIT compile then benchmark
jit_price = jax.jit(black_scholes_price, static_argnames=[])
_ = jit_price(call_valax, spot, vol, rate, dividend)  # warm-up

start = time.perf_counter()
for _ in range(n_iters):
    result = jit_price(call_valax, spot, vol, rate, dividend)
result.block_until_ready()
valax_time = time.perf_counter() - start

# QuantLib: already compiled C++
start = time.perf_counter()
for _ in range(n_iters):
    call_ql.NPV()
ql_time = time.perf_counter() - start

print(f"\n{n_iters:,} single-option prices:")
print(f"  VALAX:    {valax_time:.3f}s  ({valax_time/n_iters*1e6:.1f} µs/call)")
print(f"  QuantLib: {ql_time:.3f}s  ({ql_time/n_iters*1e6:.1f} µs/call)")

# ============================================================================
# 8. API DESIGN COMPARISON
# ============================================================================

print(f"\n{'=' * 70}")
print("API DESIGN COMPARISON")
print("=" * 70)

print("""
┌─────────────────────┬───────────────────────────────────┬────────────────────────────────────┐
│ Aspect              │ VALAX                             │ QuantLib                           │
├─────────────────────┼───────────────────────────────────┼────────────────────────────────────┤
│ Instrument          │ Data-only pytree (3 fields)       │ Payoff + Exercise + Option objects  │
│ Pricing             │ Pure function call                │ setPricingEngine() + NPV()          │
│ Greeks              │ jax.grad (automatic)              │ Analytic formulas (per-model)       │
│ Higher Greeks       │ Compose jax.grad (any order)      │ Manual finite differences           │
│ Implied Vol         │ Newton + autodiff Jacobian        │ Built-in solver (Brent)             │
│ Vectorization       │ vmap over portfolios              │ Loop over instruments               │
│ GPU/TPU             │ Automatic (JAX backend)           │ CPU only                            │
│ Compilation         │ JIT on first call                 │ Pre-compiled C++                    │
└─────────────────────┴───────────────────────────────────┴────────────────────────────────────┘
""")
