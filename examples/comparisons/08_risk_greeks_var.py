# %% [markdown]
# # VALAX vs QuantLib: Risk Building Blocks
#
# Side-by-side comparison of:
# - Greeks computation (autodiff vs analytic) — building blocks for parametric VaR
# - Repricing under market shocks — building blocks for full-revaluation VaR
# - P&L attribution (Taylor decomposition vs QuantLib Greeks x shocks)
# - Parametric VaR (delta-normal) from autodiff vs manual QL-Greeks formula
# - Performance: JIT-compiled full-reval VaR vs QuantLib loop repricing
#
# Validated by: tests/test_quantlib_comparison/test_risk_greeks_ql.py

# %% Imports
import time
import jax
import jax.numpy as jnp
import QuantLib as ql
import numpy as np

from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.greeks.autodiff import greeks
from valax.curves.discount import DiscountCurve
from valax.market.data import MarketData
from valax.market.scenario import MarketScenario
from valax.dates.daycounts import ymd_to_ordinal
from valax.risk.var import (
    reprice_under_scenario,
    pnl_attribution,
    parametric_var,
    portfolio_pnl,
    _extract_short_rate,
)
from valax.risk.scenarios import stress_scenario, stack_scenarios
from valax.market.scenario import ScenarioSet

# ============================================================================
# 0. COMMON MARKET DATA
# ============================================================================

# Market parameters
S0 = 105.0       # spot price
K = 100.0        # strike
T = 1.0          # 1 year to expiry
sigma = 0.25     # 25% vol
r = 0.04         # 4% risk-free rate
q = 0.01         # 1% continuous dividend yield

print("=" * 70)
print("VALAX vs QuantLib: Risk Building Blocks")
print("=" * 70)
print(f"\nMarket data: S={S0}, K={K}, T={T}, sigma={sigma}, r={r}, q={q}\n")

# --- VALAX setup ---
call_valax = EuropeanOption(
    strike=jnp.array(K),
    expiry=jnp.array(T),
    is_call=True,
)
spot = jnp.array(S0)
vol = jnp.array(sigma)
rate = jnp.array(r)
dividend = jnp.array(q)

# --- QuantLib setup ---
today_ql = ql.Date(26, 3, 2026)
ql.Settings.instance().evaluationDate = today_ql

spot_quote = ql.SimpleQuote(S0)
spot_handle = ql.QuoteHandle(spot_quote)
rate_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today_ql, r, ql.Actual365Fixed())
)
div_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today_ql, q, ql.Actual365Fixed())
)
vol_quote = ql.SimpleQuote(sigma)
vol_handle = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(today_ql, ql.NullCalendar(), ql.QuoteHandle(vol_quote), ql.Actual365Fixed())
)

bsm_process = ql.BlackScholesMertonProcess(
    spot_handle, div_handle, rate_handle, vol_handle
)

maturity_date = today_ql + ql.Period(1, ql.Years)
exercise = ql.EuropeanExercise(maturity_date)
call_payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
call_ql = ql.VanillaOption(call_payoff, exercise)
call_ql.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))

# ============================================================================
# SECTION A: GREEKS COMPARISON
# ============================================================================

print("=" * 70)
print("SECTION A: Greeks Comparison")
print("  (These are the building blocks of parametric VaR and P&L attribution)")
print("=" * 70)

# VALAX: autodiff Greeks
g = greeks(black_scholes_price, call_valax, spot, vol, rate, dividend)
v_delta = float(g["delta"])
v_gamma = float(g["gamma"])
v_vega = float(g["vega"])
v_rho = float(g["rho"])
v_theta = float(g["theta"]) / 365.0  # per year via bump, convert to per day

# QuantLib: analytic Greeks
ql_delta = call_ql.delta()
ql_gamma = call_ql.gamma()
ql_vega = call_ql.vega()           # QL convention: per-decimal (same as VALAX)
ql_rho = call_ql.rho()             # QL convention: per-decimal (same as VALAX)
ql_theta_day = call_ql.thetaPerDay()

print(f"\n{'Greek':<16} {'VALAX':>14} {'QuantLib':>14} {'Diff':>14}")
print("-" * 58)
print(f"{'Delta':<16} {v_delta:>14.6f} {ql_delta:>14.6f} {abs(v_delta - ql_delta):>14.2e}")
print(f"{'Gamma':<16} {v_gamma:>14.6f} {ql_gamma:>14.6f} {abs(v_gamma - ql_gamma):>14.2e}")
print(f"{'Vega':<16} {v_vega:>14.6f} {ql_vega:>14.6f} {abs(v_vega - ql_vega):>14.2e}")
print(f"{'Rho':<16} {v_rho:>14.6f} {ql_rho:>14.6f} {abs(v_rho - ql_rho):>14.2e}")
print(f"{'Theta (daily)':<16} {v_theta:>14.6f} {ql_theta_day:>14.6f} {abs(v_theta - ql_theta_day):>14.2e}")

print("\n-> VALAX uses jax.grad (exact autodiff); QuantLib uses analytic formulas.")
print("   Both give the same result — but VALAX extends trivially to any model.")

# ============================================================================
# SECTION B: REPRICING UNDER SHOCKS
# ============================================================================

print(f"\n{'=' * 70}")
print("SECTION B: Repricing Under Market Shocks")
print("  (Full-revaluation VaR does this 1000s of times)")
print("=" * 70)

# Define shock sizes
spot_bump_pct = 0.05    # +5% spot
vol_bump_abs = 0.01     # +1pp vol
rate_bump_abs = 0.005   # +50bp rate

# --- VALAX repricing via risk engine ---
# Build a batched single-instrument portfolio for reprice_under_scenario
ref = ymd_to_ordinal(2026, 3, 26)
pillars = jnp.array([ref, ymd_to_ordinal(2027, 3, 26)])
pillar_times = (pillars - ref).astype(jnp.float64) / 365.0
dfs = jnp.exp(-r * pillar_times)
curve = DiscountCurve(pillar_dates=pillars, discount_factors=dfs, reference_date=ref)

base_market = MarketData(
    spots=jnp.array([S0]),
    vols=jnp.array([sigma]),
    dividends=jnp.array([q]),
    discount_curve=curve,
)

instruments_batched = EuropeanOption(
    strike=jnp.array([K]),
    expiry=jnp.array([T]),
    is_call=True,
)

def _bs_market_fn(option, market: MarketData):
    r_extracted = _extract_short_rate(market.discount_curve)
    return black_scholes_price(option, market.spots, market.vols, r_extracted, market.dividends)

# Base price
zero_scenario = MarketScenario(
    spot_shocks=jnp.zeros(1),
    vol_shocks=jnp.zeros(1),
    rate_shocks=jnp.zeros(2),
    dividend_shocks=jnp.zeros(1),
)
valax_base = float(reprice_under_scenario(_bs_market_fn, instruments_batched, base_market, zero_scenario))

# Spot bump (+5%)
spot_up = MarketScenario(
    spot_shocks=jnp.array([S0 * spot_bump_pct]),
    vol_shocks=jnp.zeros(1),
    rate_shocks=jnp.zeros(2),
    dividend_shocks=jnp.zeros(1),
)
valax_spot_up = float(reprice_under_scenario(_bs_market_fn, instruments_batched, base_market, spot_up))

# Vol bump (+1pp)
vol_up = MarketScenario(
    spot_shocks=jnp.zeros(1),
    vol_shocks=jnp.array([vol_bump_abs]),
    rate_shocks=jnp.zeros(2),
    dividend_shocks=jnp.zeros(1),
)
valax_vol_up = float(reprice_under_scenario(_bs_market_fn, instruments_batched, base_market, vol_up))

# Rate bump (+50bp)
rate_up = MarketScenario(
    spot_shocks=jnp.zeros(1),
    vol_shocks=jnp.zeros(1),
    rate_shocks=jnp.full(2, rate_bump_abs),
    dividend_shocks=jnp.zeros(1),
)
valax_rate_up = float(reprice_under_scenario(_bs_market_fn, instruments_batched, base_market, rate_up))

# --- QuantLib repricing under shocks ---
# We rebuild QL instruments under each shock to get exact repriced values.

def ql_reprice(s, v, r_new, q_new):
    """Reprice QL European call with given parameters."""
    today = ql.Date(26, 3, 2026)
    spot_h = ql.QuoteHandle(ql.SimpleQuote(s))
    rate_h = ql.YieldTermStructureHandle(ql.FlatForward(today, r_new, ql.Actual365Fixed()))
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(today, q_new, ql.Actual365Fixed()))
    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), v, ql.Actual365Fixed())
    )
    proc = ql.BlackScholesMertonProcess(spot_h, div_h, rate_h, vol_h)
    mat = today + ql.Period(1, ql.Years)
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(mat))
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
    return opt.NPV()

ql_base = ql_reprice(S0, sigma, r, q)
ql_spot_up = ql_reprice(S0 * (1 + spot_bump_pct), sigma, r, q)
ql_vol_up = ql_reprice(S0, sigma + vol_bump_abs, r, q)
ql_rate_up = ql_reprice(S0, sigma, r + rate_bump_abs, q)

print(f"\n{'Scenario':<22} {'VALAX':>14} {'QuantLib':>14} {'Diff':>14}")
print("-" * 64)
print(f"{'Base':<22} {valax_base:>14.6f} {ql_base:>14.6f} {abs(valax_base - ql_base):>14.2e}")
print(f"{'Spot +5%':<22} {valax_spot_up:>14.6f} {ql_spot_up:>14.6f} {abs(valax_spot_up - ql_spot_up):>14.2e}")
print(f"{'Vol +1pp':<22} {valax_vol_up:>14.6f} {ql_vol_up:>14.6f} {abs(valax_vol_up - ql_vol_up):>14.2e}")
print(f"{'Rate +50bp':<22} {valax_rate_up:>14.6f} {ql_rate_up:>14.6f} {abs(valax_rate_up - ql_rate_up):>14.2e}")

print("\n-> VALAX `reprice_under_scenario` matches QuantLib manual repricing.")
print("   Both produce identical shocked prices for the same shocks.")

# ============================================================================
# SECTION C: P&L ATTRIBUTION
# ============================================================================

print(f"\n{'=' * 70}")
print("SECTION C: P&L Attribution")
print("  (Decompose a combined shock into risk factor contributions)")
print("=" * 70)

# Combined shock: spot +5%, vol +1pp, rate +50bp
combined_scenario = MarketScenario(
    spot_shocks=jnp.array([S0 * spot_bump_pct]),
    vol_shocks=jnp.array([vol_bump_abs]),
    rate_shocks=jnp.full(2, rate_bump_abs),
    dividend_shocks=jnp.zeros(1),
)

attr = pnl_attribution(_bs_market_fn, instruments_batched, base_market, combined_scenario)

# QuantLib first-order P&L attribution: Greek x shock
dS = S0 * spot_bump_pct
dvol = vol_bump_abs
drate = rate_bump_abs

ql_delta_pnl = ql_delta * dS
ql_gamma_pnl = 0.5 * ql_gamma * dS**2
ql_vega_pnl = ql_vega * dvol
ql_rho_pnl = ql_rho * drate

ql_first_order = ql_delta_pnl + ql_vega_pnl + ql_rho_pnl
ql_second_order = ql_first_order + ql_gamma_pnl

# Actual P&L (QL)
ql_combined = ql_reprice(S0 + dS, sigma + dvol, r + drate, q)
ql_actual_pnl = ql_combined - ql_base

print(f"\n{'Component':<24} {'VALAX':>14} {'QL Greeks':>14} {'Diff':>14}")
print("-" * 66)
print(f"{'Delta (spot)':<24} {float(attr['delta_spot']):>14.6f} {ql_delta_pnl:>14.6f} {abs(float(attr['delta_spot']) - ql_delta_pnl):>14.2e}")
print(f"{'Gamma (spot)':<24} {float(attr['gamma_spot']):>14.6f} {ql_gamma_pnl:>14.6f} {abs(float(attr['gamma_spot']) - ql_gamma_pnl):>14.2e}")
print(f"{'Vega (vol)':<24} {float(attr['delta_vol']):>14.6f} {ql_vega_pnl:>14.6f} {abs(float(attr['delta_vol']) - ql_vega_pnl):>14.2e}")
print(f"{'Rho (rate)':<24} {float(attr['delta_rate']):>14.6f} {ql_rho_pnl:>14.6f} {abs(float(attr['delta_rate']) - ql_rho_pnl):>14.2e}")
print(f"{'':<24} {'':>14} {'':>14} {'':>14}")
print(f"{'1st-order total':<24} {float(attr['total_first_order']):>14.6f} {ql_first_order:>14.6f} {abs(float(attr['total_first_order']) - ql_first_order):>14.2e}")
print(f"{'2nd-order total':<24} {float(attr['total_second_order']):>14.6f} {ql_second_order:>14.6f} {abs(float(attr['total_second_order']) - ql_second_order):>14.2e}")
print(f"{'Actual P&L':<24} {float(attr['actual']):>14.6f} {ql_actual_pnl:>14.6f} {abs(float(attr['actual']) - ql_actual_pnl):>14.2e}")
print(f"{'Unexplained':<24} {float(attr['unexplained']):>14.6f} {ql_actual_pnl - ql_second_order:>14.6f}")

# Show improvement from 1st to 2nd order
err_1st = abs(float(attr['actual']) - float(attr['total_first_order']))
err_2nd = abs(float(attr['unexplained']))
print(f"\n-> 1st-order error: {err_1st:.6f}")
print(f"-> 2nd-order error: {err_2nd:.6f}  ({err_2nd/err_1st*100:.1f}% of 1st-order)")
print("   The gamma correction captures most of the nonlinearity.")

# ============================================================================
# SECTION D: PARAMETRIC VaR
# ============================================================================

print(f"\n{'=' * 70}")
print("SECTION D: Parametric (Delta-Normal) VaR")
print("  (1-asset covariance matrix, autodiff Greeks vs manual QL formula)")
print("=" * 70)

confidence = 0.99

# Covariance matrix for risk factors: [spot, vol, rate_0, rate_1, dividend]
# Simple diagonal: only spot has meaningful variance
# spot_std ~ 2% of S0 per day => variance = (S0*0.02)^2
spot_daily_std = S0 * 0.02
vol_daily_std = 0.002
rate_daily_std = 0.001
div_daily_std = 0.0001

n_factors = 1 + 1 + 2 + 1  # spots, vols, rate pillars, dividends
cov = jnp.diag(jnp.array([
    spot_daily_std**2,
    vol_daily_std**2,
    rate_daily_std**2,
    rate_daily_std**2,
    div_daily_std**2,
]))

# VALAX parametric VaR (uses autodiff internally)
valax_pvar = float(parametric_var(_bs_market_fn, instruments_batched, base_market, cov, confidence))

# Manual calculation using QuantLib Greeks + same formula
# delta vector: [dP/dS, dP/dvol, dP/dr_0, dP/dr_1, dP/dq]
# For a flat curve the rate sensitivity at each pillar needs the chain rule:
# dP/dr_i = dP/dDF_i * (-t_i * DF_i)
# But for simplicity with a single option under flat rate, the total rho gives:
# We compute the delta vector the same way VALAX does internally.

# QuantLib delta vector (same factor ordering as covariance)
# For the rate pillars: we approximate using ql_rho split equally
# Since pillar 0 is t=0 (DF=1, no sensitivity) and pillar 1 is t=1:
# dP/dr at pillar 1 ~ rho (since it's 1-year expiry on a 1-year pillar)
ql_delta_vec = np.array([
    ql_delta,             # spot
    ql_vega,              # vol
    0.0,                  # rate pillar 0 (t=0, no sensitivity)
    ql_rho,               # rate pillar 1 (t=1)
    call_ql.dividendRho() if hasattr(call_ql, 'dividendRho') else 0.0,  # dividend
])

# QL manual parametric VaR
import scipy.stats
z_alpha = scipy.stats.norm.ppf(confidence)
cov_np = np.array(cov)
portfolio_var = ql_delta_vec @ cov_np @ ql_delta_vec
ql_pvar = z_alpha * np.sqrt(max(portfolio_var, 0.0))

print(f"\nConfidence level: {confidence*100:.0f}%")
print(f"\n{'Method':<35} {'VaR':>14}")
print("-" * 49)
print(f"{'VALAX parametric_var (autodiff)':<35} {valax_pvar:>14.6f}")
print(f"{'Manual (QL Greeks + formula)':<35} {ql_pvar:>14.6f}")
print(f"{'Difference':<35} {abs(valax_pvar - ql_pvar):>14.2e}")

print(f"\nDelta vectors:")
print(f"  {'Factor':<16} {'VALAX (internal)':>16} {'QL manual':>16}")
print(f"  {'-'*48}")
print(f"  {'Spot':<16} {'(autodiff)':>16} {ql_delta_vec[0]:>16.6f}")
print(f"  {'Vol':<16} {'(autodiff)':>16} {ql_delta_vec[1]:>16.6f}")
print(f"  {'Rate p0':<16} {'(autodiff)':>16} {ql_delta_vec[2]:>16.6f}")
print(f"  {'Rate p1':<16} {'(autodiff)':>16} {ql_delta_vec[3]:>16.6f}")
print(f"  {'Dividend':<16} {'(autodiff)':>16} {ql_delta_vec[4]:>16.6f}")

print("\n-> Both methods use the same delta-normal formula:")
print("   VaR = z_alpha * sqrt(delta^T @ cov @ delta)")
print("   VALAX computes the delta vector via autodiff; QL uses analytic Greeks.")

# ============================================================================
# SECTION E: PERFORMANCE — Full-Reval VaR
# ============================================================================

print(f"\n{'=' * 70}")
print("SECTION E: Performance — Full-Revaluation VaR (1000 scenarios)")
print("=" * 70)

n_scenarios = 1000

# Generate random scenarios for both
key = jax.random.PRNGKey(42)
# Simple: just spot shocks from normal distribution
z = jax.random.normal(key, shape=(n_scenarios,))
spot_shocks_arr = z * spot_daily_std  # daily spot shock

# --- VALAX: vmap over scenarios (JIT-compiled) ---
scenario_set = ScenarioSet(
    spot_shocks=spot_shocks_arr[:, None],  # (n_scenarios, 1)
    vol_shocks=jnp.zeros((n_scenarios, 1)),
    rate_shocks=jnp.zeros((n_scenarios, 2)),
    dividend_shocks=jnp.zeros((n_scenarios, 1)),
)

# Warm up JIT
_ = portfolio_pnl(_bs_market_fn, instruments_batched, base_market, scenario_set)

start = time.perf_counter()
for _ in range(10):
    pnl_valax = portfolio_pnl(_bs_market_fn, instruments_batched, base_market, scenario_set)
    pnl_valax.block_until_ready()
valax_time = (time.perf_counter() - start) / 10

# --- QuantLib: loop over scenarios ---
spot_shocks_np = np.array(spot_shocks_arr)

start = time.perf_counter()
for _ in range(10):
    pnl_ql = np.zeros(n_scenarios)
    for i in range(n_scenarios):
        shocked_spot = S0 + float(spot_shocks_np[i])
        pnl_ql[i] = ql_reprice(shocked_spot, sigma, r, q) - ql_base
ql_time = (time.perf_counter() - start) / 10

print(f"\n{n_scenarios} scenario full-revaluation VaR (avg of 10 runs):")
print(f"  VALAX (vmap+JIT):  {valax_time*1000:.1f} ms  ({valax_time/n_scenarios*1e6:.1f} us/scenario)")
print(f"  QuantLib (loop):   {ql_time*1000:.1f} ms  ({ql_time/n_scenarios*1e6:.1f} us/scenario)")
if ql_time > 0:
    print(f"  Speedup:           {ql_time/valax_time:.1f}x")

# Verify P&L distributions match
valax_var_99 = float(-jnp.percentile(pnl_valax, 1.0))
ql_var_99 = float(-np.percentile(pnl_ql, 1.0))
print(f"\n  VaR(99%) from scenarios:")
print(f"    VALAX:   {valax_var_99:.4f}")
print(f"    QuantLib: {ql_var_99:.4f}")
print(f"    Diff:     {abs(valax_var_99 - ql_var_99):.2e}")

print(f"\n-> VALAX uses jax.vmap to price all {n_scenarios} scenarios in parallel.")
print("   QuantLib must loop over scenarios in Python.")
print("   The gap widens further on GPU and with larger portfolios.")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'=' * 70}")
print("SUMMARY: Risk Building Blocks")
print("=" * 70)
print("""
+---------------------+----------------------------------+----------------------------------+
| Risk Task           | VALAX                            | QuantLib                         |
+---------------------+----------------------------------+----------------------------------+
| Greeks              | jax.grad (any model, any order)  | Analytic formulas (per model)    |
| Repricing           | reprice_under_scenario (vmap)    | Manual rebuild per scenario      |
| P&L Attribution     | pnl_attribution (autodiff Taylor)| Manual: Greek x shock            |
| Parametric VaR      | parametric_var (autodiff delta)  | Manual: delta^T @ cov @ delta    |
| Full-Reval VaR      | portfolio_pnl + vmap (parallel)  | Loop over scenarios              |
| GPU/TPU             | Automatic                        | Not supported                    |
+---------------------+----------------------------------+----------------------------------+

Key advantage: VALAX computes all sensitivities via autodiff, so adding a new
model or instrument requires zero new Greek formulas. The risk engine works
unchanged for any differentiable pricing function.
""")
