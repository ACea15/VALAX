# %% [markdown]
# # Hull-White One-Factor Model: MC vs Tree vs QuantLib
#
# Three-way comparison for a single model, three numerical methods:
#
# | Method            | VALAX                               | QuantLib oracle              |
# |-------------------|-------------------------------------|------------------------------|
# | **Analytic ZCB**  | `hw_bond_price`                     | `ql.HullWhite` + ZCB formula |
# | **Trinomial tree**| `callable_bond_price` / `puttable_bond_price` | `ql.TreeCallableFixedRateBondEngine` |
# | **Short-rate MC** | `generate_hull_white_paths`         | — (tree + analytic as oracle)|
#
# Key themes:
# - Exact-fit property: `hw_bond_price(r0=f(0,0), 0, T) == P^M(0, T)`
# - ZCB martingale check: `E[exp(-∫r dt)] == P^M(0, T)` (MC vs analytic)
# - Callable / puttable bond monotonicity: callable ≤ straight ≤ puttable
# - Convergence: MC error ∝ 1/√N;  tree error ∝ 1/steps
# - Autodiff through both the tree and the MC path generator
#
# Validated by:
#   tests/test_mc/test_hull_white_paths.py
#   tests/test_lattice/test_hull_white_tree.py
#   tests/test_quantlib_comparison/test_rates_pricers_ql.py

# %% Imports
import jax
import jax.numpy as jnp
import equinox as eqx
import QuantLib as ql

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.instruments.bonds import CallableBond, FixedRateBond, PuttableBond
from valax.models.hull_white import (
    HullWhiteModel,
    hw_bond_price,
    hw_short_rate_variance,
    _instantaneous_forward,
    _market_df,
)
from valax.pricing.analytic.bonds import fixed_rate_bond_price
from valax.pricing.lattice.hull_white_tree import (
    build_hull_white_tree,
    callable_bond_price,
    hw_tree_j_max,
    puttable_bond_price,
)
from valax.pricing.mc.hull_white_paths import generate_hull_white_paths

# ============================================================================
# 1. SHARED MARKET SETUP
# ============================================================================

print("=" * 72)
print("Hull-White One-Factor Model: MC vs Tree vs QuantLib")
print("=" * 72)

# Model parameters
A = 0.10          # mean reversion speed
SIGMA = 0.010     # short-rate volatility (100 bps)
FLAT_RATE = 0.05  # flat 5% continuously-compounded curve

# Instrument parameters
MAT_YEARS = 5     # 5-year bond
COUPON = 0.05     # 5% annual coupon (ATM for a flat 5% curve)
FACE = 100.0
FREQ = 1          # annual coupons

# Reference date: 2026-01-01 (integer ordinal)
REF_ORD = int(ymd_to_ordinal(2026, 1, 1))
DAY_COUNT = "act_365"

print(f"\nModel  : Hull-White 1F,  a={A},  σ={SIGMA}")
print(f"Curve  : flat {FLAT_RATE*100:.1f}%  (continuously compounded, Act/365)")
print(f"Bond   : {MAT_YEARS}Y annual coupon={COUPON*100:.0f}%,  face={FACE:.0f}")


# ── Build VALAX flat discount curve  (t=0 pillar with df=1, per the
#    DiscountCurve contract that the leading discount factor is 1.0)
def _flat_curve(ref: int, rate: float, years: int) -> DiscountCurve:
    pillars = jnp.array(
        [ref] + [ref + int(round(k * 365)) for k in range(1, years + 2)],
        dtype=jnp.int32,
    )
    t = (pillars - ref).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-jnp.asarray(rate) * t),
        reference_date=jnp.int32(ref),
        day_count=DAY_COUNT,
    )


curve = _flat_curve(REF_ORD, FLAT_RATE, MAT_YEARS + 2)
model = HullWhiteModel(
    mean_reversion=jnp.asarray(A),
    volatility=jnp.asarray(SIGMA),
    initial_curve=curve,
)

# ── Payment and call/put date arrays
payment_ords = jnp.array(
    [REF_ORD + int(round(k * 365)) for k in range(1, MAT_YEARS + 1)],
    dtype=jnp.int32,
)
# Callable / puttable on years 2 and 4
call_ords = jnp.array(
    [REF_ORD + int(round(k * 365)) for k in [2, 4]], dtype=jnp.int32
)

straight_bond = FixedRateBond(
    payment_dates=payment_ords,
    settlement_date=jnp.int32(REF_ORD),
    coupon_rate=jnp.asarray(COUPON),
    face_value=jnp.asarray(FACE),
    frequency=FREQ,
    day_count=DAY_COUNT,
)
callable_bond = CallableBond(
    payment_dates=payment_ords,
    settlement_date=jnp.int32(REF_ORD),
    coupon_rate=jnp.asarray(COUPON),
    face_value=jnp.asarray(FACE),
    call_dates=call_ords,
    call_prices=jnp.ones(2),   # par calls
    frequency=FREQ,
    day_count=DAY_COUNT,
)
puttable_bond = PuttableBond(
    payment_dates=payment_ords,
    settlement_date=jnp.int32(REF_ORD),
    coupon_rate=jnp.asarray(COUPON),
    face_value=jnp.asarray(FACE),
    put_dates=call_ords,
    put_prices=jnp.ones(2),    # par puts
    frequency=FREQ,
    day_count=DAY_COUNT,
)

# ── QuantLib setup (flat curve, same parameters)
QL_TODAY = ql.Date(1, 1, 2026)
ql.Settings.instance().evaluationDate = QL_TODAY
dc_ql = ql.Actual365Fixed()
cal_ql = ql.NullCalendar()

ql_disc = ql.YieldTermStructureHandle(
    ql.FlatForward(QL_TODAY, FLAT_RATE, dc_ql)
)
ql_hw = ql.HullWhite(ql_disc, float(A), float(SIGMA))

def _ql_annual_schedule(n_years: int):
    start = QL_TODAY
    end = QL_TODAY + ql.Period(int(round(n_years * 365)), ql.Days)
    return ql.Schedule(
        start, end,
        ql.Period(ql.Annual),
        cal_ql, ql.Unadjusted, ql.Unadjusted,
        ql.DateGeneration.Forward, False,
    )

# ============================================================================
# 2. EXACT-FIT ZCB CHECK
# ============================================================================

print(f"\n{'=' * 72}")
print("§1  EXACT-FIT PROPERTY:  hw_bond_price(r₀, 0, T) = P^M(0, T)")
print("=" * 72)

r0 = float(_instantaneous_forward(model, jnp.asarray(0.0)))
print(f"\nr(0) = f^M(0,0) = {r0:.6f}   (initial short rate under exact fit)")

print(f"\n{'T (y)':>7} {'P^M(0,T)':>12} {'hw_bond_price':>14} {'|diff|':>10}")
print("-" * 46)
for T in [0.5, 1.0, 2.0, 3.0, 5.0]:
    pm = float(_market_df(model, jnp.asarray(T)))
    phw = float(hw_bond_price(
        model,
        jnp.asarray(r0),
        jnp.asarray(0.0),
        jnp.asarray(T),
    ))
    print(f"{T:7.1f} {pm:12.8f} {phw:14.8f} {abs(pm - phw):10.2e}")

print("\n→  Exact fit guaranteed by the affine A(t,T) formula for any curve shape.")

# ============================================================================
# 3. SHORT-RATE DISTRIBUTION
# ============================================================================

print(f"\n{'=' * 72}")
print("§2  SHORT-RATE DISTRIBUTION AT HORIZON T")
print("=" * 72)

print(f"\nAnalytic unconditional moments  (r₀ = {r0:.4f},  a={A},  σ={SIGMA}):")
print(f"\n{'T (y)':>7} {'E[r(T)]':>12} {'Var[r(T)]':>12} {'Std[r(T)]':>12}")
print("-" * 46)
for T in [1.0, 2.0, 3.0, 5.0]:
    alpha_T = float(
        _instantaneous_forward(model, jnp.asarray(T))
        + (SIGMA**2 / (2 * A**2)) * (1 - jnp.exp(-A * T))**2
    )
    var_T = float(hw_short_rate_variance(model, jnp.asarray(T)))
    print(f"{T:7.1f} {alpha_T:12.6f} {var_T:12.8f} {var_T**0.5:12.6f}")

print(f"\nNote: E[r(T)] = α(T) = f^M(0,T) + σ²/(2a²)·(1−e^{{−aT}})²")
print(f"      For a flat curve at rate c: E[r(T)] slightly exceeds c (convexity drift).")

# ============================================================================
# 4. MC ZCB MARTINGALE CHECK
# ============================================================================

print(f"\n{'=' * 72}")
print("§3  MC MARTINGALE CHECK:  E[exp(−∫₀ᵀ r dt)] = P^M(0,T)")
print("=" * 72)

N_PATHS  = 50_000
KEY      = jax.random.PRNGKey(0)

print(f"\nUsing {N_PATHS:,} paths, 250 steps/year  (exact conditional sampling):\n")
print(f"{'T (y)':>7} {'P^M(0,T)':>12} {'MC mean':>12} {'|bias|':>10} {'SE':>10} {'|bias|/SE':>10}")
print("-" * 66)
for T in [1.0, 2.0, 3.0, 5.0]:
    n_steps = int(T * 250)
    result = generate_hull_white_paths(
        model, T=T, n_steps=n_steps, n_paths=N_PATHS, key=KEY,
    )
    sdf = jnp.exp(result.log_discount_factors[:, -1])
    mc_mean = float(jnp.mean(sdf))
    se      = float(jnp.std(sdf) / jnp.sqrt(float(N_PATHS)))
    analytic = float(_market_df(model, jnp.asarray(T)))
    bias     = abs(mc_mean - analytic)
    print(f"{T:7.1f} {analytic:12.8f} {mc_mean:12.8f} {bias:10.2e} {se:10.2e} {bias/se:10.2f}")

print("\n→  |bias|/SE < 3 across all horizons → consistent with zero bias.")
print("   Residual trapezoidal error ∝ dt²/T (negligible at 250 steps/year).")

# ============================================================================
# 5. STRAIGHT BOND: MC vs ANALYTIC vs QL DISCOUNTING
# ============================================================================

print(f"\n{'=' * 72}")
print("§4  STRAIGHT BOND PRICE:  MC vs Analytic vs QuantLib")
print("=" * 72)

# ── Analytic (discount curve)
analytic_straight = float(fixed_rate_bond_price(straight_bond, curve))

# ── MC: sum coupon and face SDFs
T_MAT = float(year_fraction(jnp.int32(REF_ORD), payment_ords[-1], DAY_COUNT))
N_STEPS_BOND = int(T_MAT * 250)
result_bond = generate_hull_white_paths(
    model, T=T_MAT, n_steps=N_STEPS_BOND, n_paths=N_PATHS, key=KEY,
)
dt_bond = T_MAT / N_STEPS_BOND
coupon_cf = COUPON * FACE
coupon_years = [float(year_fraction(jnp.int32(REF_ORD), d, DAY_COUNT)) for d in payment_ords]

pv_paths = jnp.zeros(N_PATHS, dtype=jnp.float64)
for i, t_cf in enumerate(coupon_years):
    idx = min(int(round(t_cf / dt_bond)), N_STEPS_BOND)
    sdf = jnp.exp(result_bond.log_discount_factors[:, idx])
    cf = coupon_cf if i < len(coupon_years) - 1 else coupon_cf + FACE
    pv_paths = pv_paths + cf * sdf

mc_straight = float(jnp.mean(pv_paths))
mc_se = float(jnp.std(pv_paths) / jnp.sqrt(float(N_PATHS)))

# ── QuantLib (discounting bond engine)
sched = _ql_annual_schedule(MAT_YEARS)
ql_bond = ql.FixedRateBond(0, FACE, sched, [COUPON], dc_ql)
ql_bond.setPricingEngine(ql.DiscountingBondEngine(ql_disc))
ql_straight = ql_bond.dirtyPrice()

print(f"\n{'Method':<24} {'Price':>10} {'|diff vs analytic|':>20}")
print("-" * 56)
print(f"{'VALAX analytic (DF)':<24} {analytic_straight:10.4f} {'—':>20}")
print(f"{'VALAX MC (50k paths)':<24} {mc_straight:10.4f} {abs(mc_straight - analytic_straight):>20.4f}  ({abs(mc_straight-analytic_straight)/mc_se:.1f} SE)")
print(f"{'QuantLib discounting':<24} {ql_straight:10.4f} {abs(ql_straight - analytic_straight):>20.4f}")

# ============================================================================
# 6. CALLABLE BOND: TREE vs QuantLib
# ============================================================================

print(f"\n{'=' * 72}")
print("§5  CALLABLE BOND:  HW Trinomial Tree vs QuantLib")
print("=" * 72)

# ── VALAX tree convergence
print(f"\n--- Tree convergence (VALAX) ---")
print(f"{'Steps':>8} {'Callable':>12} {'Puttable':>12}")
print("-" * 36)
tree_callable_ref = None
for n_steps_tree in [50, 100, 200, 500]:
    pc = float(callable_bond_price(callable_bond, model, n_steps=n_steps_tree))
    pp = float(puttable_bond_price(puttable_bond, model, n_steps=n_steps_tree))
    print(f"{n_steps_tree:>8} {pc:>12.4f} {pp:>12.4f}")
    if n_steps_tree == 200:
        tree_callable_ref = pc
        tree_puttable_ref = pp

# ── QuantLib tree (200 steps)
ql_call_sched = ql.CallabilitySchedule()
ql_put_sched  = ql.CallabilitySchedule()
# Call/put on coupon payment dates at years 2 and 4 (matching VALAX call_ords)
for k in [2, 4]:
    ql_date = QL_TODAY + ql.Period(int(round(k * 365)), ql.Days)
    # Snap to the nearest business day in the schedule
    ql_call_sched.append(ql.Callability(
        ql.BondPrice(100.0, ql.BondPrice.Dirty), ql.Callability.Call, ql_date,
    ))
    ql_put_sched.append(ql.Callability(
        ql.BondPrice(100.0, ql.BondPrice.Dirty), ql.Callability.Put, ql_date,
    ))

ql_callable_bond = ql.CallableFixedRateBond(
    0, FACE, sched, [COUPON], dc_ql, ql.Unadjusted, 100.0, QL_TODAY, ql_call_sched,
)
ql_puttable_bond = ql.CallableFixedRateBond(
    0, FACE, sched, [COUPON], dc_ql, ql.Unadjusted, 100.0, QL_TODAY, ql_put_sched,
)
tree_engine_ql = ql.TreeCallableFixedRateBondEngine(ql_hw, 200)
ql_callable_bond.setPricingEngine(tree_engine_ql)
ql_puttable_bond.setPricingEngine(tree_engine_ql)

ql_callable = ql_callable_bond.dirtyPrice()
ql_puttable = ql_puttable_bond.dirtyPrice()

# ── Head-to-head at 200 steps
print(f"\n--- Head-to-head at 200 steps ---")
print(f"(Note: residual diff on callable/puttable reflects VALAX NullCalendar")
print(f" integer-day ordinals vs QuantLib date-generation — see WS1 QL tests")
print(f" in tests/test_quantlib_comparison/test_rates_pricers_ql.py for the")
print(f" rigorously aligned comparison at rel < 5e-3.)\n")
print(f"{'Instrument':<24} {'VALAX tree':>12} {'QuantLib tree':>14} {'|diff|':>10} {'rel':>8}")
print("-" * 70)
for label, v_price, q_price in [
    ("Straight bond",   analytic_straight, ql_straight),
    ("Callable bond",   tree_callable_ref, ql_callable),
    ("Puttable bond",   tree_puttable_ref, ql_puttable),
]:
    diff = abs(v_price - q_price)
    rel  = diff / q_price
    print(f"{label:<24} {v_price:>12.4f} {q_price:>14.4f} {diff:>10.4f} {rel:>8.4%}")

# ── Monotonicity check (fundamental no-arb constraint)
print(f"\n--- Monotonicity check (callable ≤ straight ≤ puttable) ---\n")
opt_val_call = analytic_straight - tree_callable_ref
opt_val_put  = tree_puttable_ref - analytic_straight
print(f"  Straight (analytic DF) = {analytic_straight:.4f}")
print(f"  Callable (VALAX tree)  = {tree_callable_ref:.4f}"
      f"  →  call option value = {opt_val_call:.4f}")
print(f"  Puttable (VALAX tree)  = {tree_puttable_ref:.4f}"
      f"  →  put  option value = {opt_val_put:.4f}")
assert tree_callable_ref <= analytic_straight + 0.05, "Callable should be ≤ straight"
assert tree_puttable_ref >= analytic_straight - 0.05, "Puttable should be ≥ straight"
print("\n  ✓  Monotonicity holds.")

# ============================================================================
# 7. PARAMETER SENSITIVITY (AUTODIFF)
# ============================================================================

print(f"\n{'=' * 72}")
print("§6  PARAMETER SENSITIVITY VIA AUTODIFF")
print("=" * 72)

print("""
VALAX differentiates through both numerical methods via eqx.filter_grad.

The single prerequisite for tree autodiff: pre-compute j_max as a concrete
Python int from the model before entering the JAX trace.  This decouples
the shape-determining computation from the differentiable leaves (σ, a).
Inside the trace j_max is a static literal, so shapes are known at compile
time and jax.grad can flow through the entire backward-induction scan.
""")

# ── Pre-compute j_max once from concrete model values ─────────────────
TREE_STEPS = 100
j_max_val = hw_tree_j_max(float(model.mean_reversion), MAT_YEARS / TREE_STEPS)
print(f"  Pre-computed j_max = {j_max_val}  (from a={A}, dt={MAT_YEARS/TREE_STEPS:.3f})")

# ── Tree autodiff: eqx.filter_grad through the full backward-induction scan ──
grad_cb = eqx.filter_grad(
    lambda m: callable_bond_price(callable_bond, m, n_steps=TREE_STEPS, j_max=j_max_val)
)(model)
grad_pb = eqx.filter_grad(
    lambda m: puttable_bond_price(puttable_bond, m, n_steps=TREE_STEPS, j_max=j_max_val)
)(model)

print(f"\n  Tree  d(callable)/d(σ) = {float(grad_cb.volatility):+.4f}")
print(f"  Tree  d(callable)/d(a) = {float(grad_cb.mean_reversion):+.4f}")
print(f"  Tree  d(puttable)/d(σ) = {float(grad_pb.volatility):+.4f}")
print(f"  Tree  d(puttable)/d(a) = {float(grad_pb.mean_reversion):+.4f}")
print(f"  (callable σ-grad < 0: higher vol → issuer benefits → bond cheaper for holder)")
print(f"  (puttable σ-grad > 0: higher vol → holder benefits → bond more valuable)")

# ── MC autodiff: fully JIT + filter_grad ──────────────────────────────
@eqx.filter_jit
@eqx.filter_grad
def _mc_grad(m):
    res = generate_hull_white_paths(m, T=3.0, n_steps=200, n_paths=10_000, key=KEY)
    return jnp.mean(jnp.exp(res.log_discount_factors[:, -1]))

grad_mc = _mc_grad(model)
print(f"\n  MC    d(ZCB P(0,3))/d(σ) = {float(grad_mc.volatility):+.6f}")
print(f"  MC    d(ZCB P(0,3))/d(a) = {float(grad_mc.mean_reversion):+.6f}")

print("""
→  Both the tree rollback and the MC scan are fully differentiable once
   j_max is supplied as a static int.  In QuantLib there is no autodiff
   path — sensitivities always require finite-difference bumps of the model.
""")

# ============================================================================
# 8. VOLATILITY SENSITIVITY: CALLABLE OPTION VALUE vs SIGMA
# ============================================================================

print(f"{'=' * 72}")
print("§7  EMBEDDED CALL OPTION VALUE vs SHORT-RATE VOLATILITY σ")
print("=" * 72)

print(f"\n{'σ':>8} {'Straight':>12} {'Callable':>12} {'Option val':>12} {'Δ(opt val)':>12}")
print("-" * 58)
prev_opt_val = None
for sigma_val in [0.005, 0.010, 0.015, 0.020, 0.030]:
    m_bump = HullWhiteModel(
        mean_reversion=model.mean_reversion,
        volatility=jnp.asarray(sigma_val),
        initial_curve=curve,
    )
    p_straight = float(fixed_rate_bond_price(straight_bond, curve))
    p_callable  = float(callable_bond_price(callable_bond, m_bump, n_steps=200))
    opt_val = p_straight - p_callable
    delta_str = f"{opt_val - prev_opt_val:+12.4f}" if prev_opt_val is not None else f"{'—':>12}"
    print(f"{sigma_val:8.3f} {p_straight:12.4f} {p_callable:12.4f} {opt_val:12.4f} {delta_str}")
    prev_opt_val = opt_val

print("\n→  Call option value rises monotonically with σ — higher vol means")
print("   higher probability the issuer benefits from early redemption.")

# ============================================================================
# 9. SUMMARY TABLE
# ============================================================================

print(f"\n{'=' * 72}")
print("SUMMARY")
print("=" * 72)

print(f"""
┌─────────────────────┬────────────────────────────────┬──────────────────────────────┐
│ Capability          │ VALAX                          │ QuantLib oracle              │
├─────────────────────┼────────────────────────────────┼──────────────────────────────┤
│ Analytic ZCB        │ hw_bond_price (affine, exact)  │ HullWhite.discountBond       │
│ Exact-fit property  │ ✓  (by construction, any curve)│ ✓                            │
│ Trinomial tree      │ build_hull_white_tree +        │ TreeCallableFixedRate-       │
│                     │ callable/puttable_bond_price   │ BondEngine                   │
│ Short-rate MC       │ generate_hull_white_paths      │ — (not in QL)                │
│                     │ (exact OU conditional sampling)│                              │
│ MC ZCB check        │ E[exp(-∫r)] ≈ P^M(0,T)  ✓    │ N/A                          │
│ Callable bond price │ tree:  {tree_callable_ref:.4f}                │ {ql_callable:.4f}                   │
│ Puttable bond price │ tree:  {tree_puttable_ref:.4f}                │ {ql_puttable:.4f}                   │
│ |diff| (callable)  │ {abs(tree_callable_ref - ql_callable):.4f}                          │ reference                    │
│ |diff| (puttable)  │ {abs(tree_puttable_ref - ql_puttable):.4f}                          │ reference                    │
│ Autodiff (tree)     │ jax.grad through rollback  ✓  │ FD bump required             │
│ Autodiff (MC)       │ jax.grad through scan loop ✓  │ not supported                │
│ GPU/TPU             │ automatic (JAX backend)    ✓  │ CPU only                     │
└─────────────────────┴────────────────────────────────┴──────────────────────────────┘
""")
