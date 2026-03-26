# %% [markdown]
# # Equity Options: Pricing and Greeks with VALAX
#
# This example covers:
# - Defining European option contracts
# - Black-Scholes pricing
# - Computing Greeks via autodiff (no finite differences!)
# - Implied volatility inversion
# - Portfolio-level vectorized pricing with vmap

# %% Imports
import jax
import jax.numpy as jnp
from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price, black_scholes_implied_vol
from valax.greeks.autodiff import greeks, greek
from valax.portfolio.batch import batch_price, batch_greeks

# ============================================================================
# 1. DEFINING INSTRUMENTS
# ============================================================================

# %% Define a single European call option
# Instruments are data-only pytrees — no pricing logic, just contract terms.
# This is a deliberate departure from QuantLib's instrument.setPricingEngine() pattern.

call = EuropeanOption(
    strike=jnp.array(100.0),
    expiry=jnp.array(1.0),    # 1 year to expiry
    is_call=True,
)
print(f"Call option: K={float(call.strike)}, T={float(call.expiry)}")

# %% Define a put with the same terms
put = EuropeanOption(
    strike=jnp.array(100.0),
    expiry=jnp.array(1.0),
    is_call=False,
)

# ============================================================================
# 2. BLACK-SCHOLES PRICING
# ============================================================================

# %% Synthetic market data for a tech stock
# Pricing functions are pure: price = f(instrument, market_data)
spot = jnp.array(105.0)      # current stock price
vol = jnp.array(0.25)        # 25% annualized implied vol
rate = jnp.array(0.04)       # 4% risk-free rate
dividend = jnp.array(0.01)   # 1% continuous dividend yield

# %% Price the call
call_price = black_scholes_price(call, spot, vol, rate, dividend)
print(f"Call price: ${float(call_price):.4f}")

# %% Price the put
put_price = black_scholes_price(put, spot, vol, rate, dividend)
print(f"Put price:  ${float(put_price):.4f}")

# %% Verify put-call parity: C - P = S*exp(-qT) - K*exp(-rT)
parity_lhs = call_price - put_price
parity_rhs = spot * jnp.exp(-dividend * call.expiry) - call.strike * jnp.exp(-rate * call.expiry)
print(f"Put-call parity check: {float(parity_lhs):.6f} == {float(parity_rhs):.6f}")
print(f"Error: {abs(float(parity_lhs - parity_rhs)):.2e}")

# ============================================================================
# 3. GREEKS VIA AUTODIFF
# ============================================================================

# %% Compute all Greeks at once
# One function call gives price + all first and second order Greeks.
# Under the hood this uses jax.grad (NOT finite differences).
g = greeks(black_scholes_price, call, spot, vol, rate, dividend)

print("\n--- Greeks (ATM-ish call, S=105, K=100) ---")
print(f"Price:        {float(g['price']):>10.4f}")
print(f"Delta:        {float(g['delta']):>10.4f}")
print(f"Gamma:        {float(g['gamma']):>10.6f}")
print(f"Vega:         {float(g['vega']):>10.4f}")
print(f"Theta:        {float(g['theta']):>10.4f}  (per day)")
print(f"Rho:          {float(g['rho']):>10.4f}")
print(f"Vanna:        {float(g['vanna']):>10.6f}")
print(f"Volga:        {float(g['volga']):>10.4f}")

# %% Compute a single Greek
delta = greek(black_scholes_price, "delta", call, spot, vol, rate, dividend)
gamma = greek(black_scholes_price, "gamma", call, spot, vol, rate, dividend)
print(f"\nSingle Greek: delta = {float(delta):.4f}, gamma = {float(gamma):.6f}")

# %% Higher order Greeks via raw jax.grad
# VALAX pricing functions are pure, so you can always compose jax.grad directly.
# Example: d^3(price)/d(spot)^3 = "speed"
speed_fn = jax.grad(jax.grad(jax.grad(
    lambda s: black_scholes_price(call, s, vol, rate, dividend)
)))
speed = speed_fn(spot)
print(f"Speed (d³P/dS³): {float(speed):.8f}")

# ============================================================================
# 4. IMPLIED VOLATILITY
# ============================================================================

# %% Round-trip: price -> implied vol -> should recover original vol
market_price = black_scholes_price(call, spot, vol, rate, dividend)
recovered_vol = black_scholes_implied_vol(call, spot, rate, dividend, market_price)
print(f"\nImplied vol round-trip: input={float(vol):.4f}, recovered={float(recovered_vol):.4f}")
print(f"Error: {abs(float(vol - recovered_vol)):.2e}")

# %% Implied vol from a "market" price
# Suppose we observe the call trading at $15.50
observed_price = jnp.array(15.50)
iv = black_scholes_implied_vol(call, spot, rate, dividend, observed_price)
print(f"Market price ${float(observed_price):.2f} => IV = {float(iv)*100:.2f}%")

# ============================================================================
# 5. PORTFOLIO-LEVEL PRICING WITH VMAP
# ============================================================================

# %% Define a portfolio of 5 call options with different strikes
n = 5
portfolio = EuropeanOption(
    strike=jnp.array([90.0, 95.0, 100.0, 105.0, 110.0]),
    expiry=jnp.full(n, 1.0),
    is_call=True,
)

# Market data for each — same underlying but position-specific
spots = jnp.full(n, 105.0)
vols = jnp.array([0.28, 0.26, 0.25, 0.24, 0.23])    # skew: higher vol at lower strikes
rates = jnp.full(n, 0.04)
dividends = jnp.full(n, 0.01)

# %% Price all 5 options in one vectorized call
prices = batch_price(black_scholes_price, portfolio, spots, vols, rates, dividends)
print("\n--- Portfolio Prices ---")
for i in range(n):
    print(f"  K={float(portfolio.strike[i]):6.1f}  vol={float(vols[i])*100:.0f}%  price=${float(prices[i]):.4f}")

# %% Compute Greeks for the entire portfolio in one call
portfolio_greeks = batch_greeks(black_scholes_price, portfolio, spots, vols, rates, dividends)
print("\n--- Portfolio Greeks ---")
print(f"{'Strike':>8} {'Delta':>8} {'Gamma':>10} {'Vega':>8} {'Theta':>8}")
for i in range(n):
    print(f"{float(portfolio.strike[i]):8.1f} "
          f"{float(portfolio_greeks['delta'][i]):8.4f} "
          f"{float(portfolio_greeks['gamma'][i]):10.6f} "
          f"{float(portfolio_greeks['vega'][i]):8.4f} "
          f"{float(portfolio_greeks['theta'][i]):8.4f}")

# %% Portfolio-level risk aggregation
# With all Greeks as arrays, aggregation is just array ops
notionals = jnp.array([100.0, 200.0, -150.0, 100.0, -50.0])  # long/short positions
portfolio_delta = jnp.sum(notionals * portfolio_greeks["delta"])
portfolio_gamma = jnp.sum(notionals * portfolio_greeks["gamma"])
portfolio_vega = jnp.sum(notionals * portfolio_greeks["vega"])
print(f"\nPortfolio aggregates (notional-weighted):")
print(f"  Net delta: {float(portfolio_delta):.2f}")
print(f"  Net gamma: {float(portfolio_gamma):.4f}")
print(f"  Net vega:  {float(portfolio_vega):.2f}")

# ============================================================================
# 6. JIT COMPILATION
# ============================================================================

# %% JIT-compile the pricing function for speed
# First call compiles; subsequent calls are near-instantaneous.
jit_price = jax.jit(black_scholes_price, static_argnames=[])

# Warm up
_ = jit_price(call, spot, vol, rate, dividend)

# Time it
import time
start_time = time.perf_counter()
for _ in range(10_000):
    _ = jit_price(call, spot, vol, rate, dividend)
_.block_until_ready()
elapsed = time.perf_counter() - start_time
print(f"\n10,000 JIT-compiled BS prices in {elapsed:.3f}s ({elapsed/10_000*1e6:.1f} µs/call)")
