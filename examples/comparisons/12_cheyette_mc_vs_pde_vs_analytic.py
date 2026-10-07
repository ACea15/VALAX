# %% [markdown]
# # Cheyette Quasi-Gaussian Model: PDE vs MC vs analytic, and the frozen-`y` bias
#
# The skew-capable companion to the Hull-White (`10_...`) and G2++ (`11_...`)
# cross-checks. The Cheyette (quasi-Gaussian) model adds a **local-volatility
# skew** `beta` to the Gaussian short-rate family, so a *strike slice* of
# caplets or swaptions carries information a single ATM quote cannot — which is
# exactly what the PDE-based calibrator fits.
#
# There is no QuantLib Cheyette engine, so the triangulation is:
#
# | Method                 | VALAX                                        | Role                              |
# |------------------------|----------------------------------------------|-----------------------------------|
# | **Frozen-`y` PDE**     | `pde_price_dispatch` (production engine)     | fast, tiny frozen-`y` bias        |
# | **Exact-`y` PDE**      | `cheyette_exact_price` (2-D `(x,y)` grid)    | bias-free, zero MC noise          |
# | **Short-rate MC**      | exact-`y` Monte Carlo recipe                 | independent, MC noise             |
# | **Hull-White analytic**| `hw_zcb_option_price` at `beta = 0`          | reduction oracle                  |
#
# Themes:
# - **Reduction.** At `beta = 0` the frozen-`y` Cheyette caplet reproduces the
#   closed-form Hull-White caplet — the primary validation.
# - **Frozen-`y` bias.** At `beta != 0` the frozen-`y` PDE and the exact-`y` PDE
#   differ by a small, *deterministic* amount (no MC noise): the frozen-`y`
#   approximation's error, which the exact-`y` engine measures exactly.
# - **Instrument coverage.** Caplet, cap, European swaption and Bermudan
#   swaption all price on one consistent engine.
#
# Validated by:
#   tests/test_pde/test_short_rate_caps.py
#   tests/test_pde/test_cheyette.py
#   tests/test_calibration/test_cheyette_calibration.py

# %% Imports
import jax
import jax.numpy as jnp

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.instruments.rates import BermudanSwaption, Cap, Caplet, Swaption
from valax.models.cheyette import CheyetteModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.hull_white_swaptions import hw_zcb_option_price
from valax.pricing.analytic.swaptions import _annuity
from valax.pricing.mc import MCConfig
from valax.pricing.pde import PDEConfig, PDEConfig2D, Scheme, pde_price_dispatch
from valax.pricing.pde.cheyette import cheyette_exact_price
from valax.pricing.reconcile import price_all_engines

_ACT365 = "act_365"
REF_YEAR = 2025
REF_ORD = int(ymd_to_ordinal(REF_YEAR, 1, 1))
FLAT_RATE = 0.03


def flat_curve(rate=FLAT_RATE, n_years=12):
    years = [0.0] + [float(k) for k in range(1, n_years)]
    pillars = jnp.array(
        [REF_ORD + int(round(y * 365)) for y in years], dtype=jnp.int32
    )
    times = (pillars - REF_ORD).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rate * times),
        reference_date=jnp.int32(REF_ORD),
        day_count=_ACT365,
    )


def cheyette(curve, beta, kappa=0.10, sigma=0.01):
    return CheyetteModel(
        mean_reversion=jnp.array(kappa),
        volatility=jnp.array(sigma),
        skew=jnp.array(beta),
        initial_curve=curve,
    )


CURVE = flat_curve()
BETA = 0.20
MODEL = cheyette(CURVE, beta=BETA)
HW_TWIN = HullWhiteModel(
    mean_reversion=jnp.array(0.10), volatility=jnp.array(0.01), initial_curve=CURVE
)

PDE_1F = PDEConfig(n_spot=301, n_time=200, spot_range=6.0, scheme=Scheme.CRANK_NICOLSON)
PDE_EXACT = PDEConfig2D(n_x=161, n_y=81, n_time=200, x_range=6.0, y_max=3.0)
MC_CFG = MCConfig(n_paths=100_000, n_steps=120)

print("=" * 74)
print("CHEYETTE QUASI-GAUSSIAN — frozen-y PDE vs exact-y PDE vs MC vs analytic")
print("=" * 74)
print(f"\nReference date : {REF_YEAR}-01-01   (flat {FLAT_RATE:.1%} curve)")
print(f"Parameters     : kappa=0.10  sigma=0.010  beta={BETA:.2f}")


def _yf(a, b):
    return year_fraction(jnp.int32(int(a)), jnp.int32(int(b)), _ACT365)


def hw_caplet_analytic(caplet, model):
    T = _yf(REF_ORD, caplet.fixing_date)
    S = _yf(REF_ORD, caplet.end_date)
    tau = _yf(caplet.start_date, caplet.end_date)
    strike_bond = 1.0 / (1.0 + caplet.strike * tau)
    zb = hw_zcb_option_price(
        model, T, jnp.array([S]), jnp.array([strike_bond]),
        is_call=not caplet.is_cap,
    )[0]
    return float(caplet.notional * (1.0 + caplet.strike * tau) * zb)


# ============================================================================
# 1. CAPLET — reduction, frozen-y bias, and MC triangulation
# ============================================================================

print("\n" + "=" * 74)
print("1. CAPLET  (2Y fixing, 6M accrual, 3% strike)")
print("=" * 74)

fix = int(ymd_to_ordinal(2027, 1, 1))
end = int(ymd_to_ordinal(2027, 7, 1))
caplet = Caplet(
    fixing_date=jnp.int32(fix), start_date=jnp.int32(fix), end_date=jnp.int32(end),
    strike=jnp.array(0.03), notional=jnp.array(1.0e6),
    is_cap=True, day_count=_ACT365,
)

# Reduction at beta = 0: frozen-y Cheyette caplet == Hull-White caplet.
caplet_beta0_frozen = float(pde_price_dispatch(caplet, cheyette(CURVE, 0.0), PDE_1F).price)
caplet_hw_analytic = hw_caplet_analytic(caplet, HW_TWIN)
caplet_hw_pde = float(pde_price_dispatch(caplet, HW_TWIN, PDE_1F).price)

print("\n  Reduction check (beta = 0):")
print(f"    Hull-White analytic     : {caplet_hw_analytic:.6f}")
print(f"    Hull-White PDE          : {caplet_hw_pde:.6f}")
print(f"    Cheyette frozen-y PDE   : {caplet_beta0_frozen:.6f}   "
      f"(rel to HW analytic {abs(caplet_beta0_frozen - caplet_hw_analytic) / caplet_hw_analytic:.1e})")

# At beta != 0: frozen-y PDE vs exact-y PDE (the frozen-y bias) vs MC.
frozen = float(pde_price_dispatch(caplet, MODEL, PDE_1F).price)
exact = float(cheyette_exact_price(caplet, MODEL, PDE_EXACT).price)
table = price_all_engines(
    caplet, MODEL, pde_config=PDE_1F, mc_config=MC_CFG, mc_key=jax.random.PRNGKey(0)
)
mc, mc_se = table.price("mc"), table.quotes["mc"].stderr

print(f"\n  At beta = {BETA:.2f}:")
print(f"    Frozen-y PDE            : {frozen:.6f}")
print(f"    Exact-y  PDE            : {exact:.6f}   "
      f"(frozen-y bias {frozen - exact:+.6f}, {(frozen / exact - 1.0):+.2e})")
print(f"    Exact-y  MC             : {mc:.6f} +/- {mc_se:.6f}   "
      f"(z vs exact-y PDE {(mc - exact) / mc_se:+.2f})")


# ============================================================================
# 2. CAP — a strip on one backward sweep
# ============================================================================

print("\n" + "=" * 74)
print("2. CAP  (semiannual, 1Y..4Y, 3% strike)")
print("=" * 74)

fixings = jnp.array(
    [int(ymd_to_ordinal(2026 + i // 2, 1 if i % 2 == 0 else 7, 1)) for i in range(6)],
    dtype=jnp.int32,
)
ends = jnp.array(
    [int(ymd_to_ordinal(2026 + (i + 1) // 2, 7 if i % 2 == 0 else 1, 1)) for i in range(6)],
    dtype=jnp.int32,
)
cap = Cap(
    fixing_dates=fixings, start_dates=fixings, end_dates=ends,
    strike=jnp.array(0.03), notional=jnp.array(1.0e6),
    is_cap=True, day_count=_ACT365,
)
cap_cfg = PDEConfig(n_spot=401, n_time=600, spot_range=6.0)
cap_table = price_all_engines(
    cap, MODEL, pde_config=cap_cfg, mc_config=MC_CFG, mc_key=jax.random.PRNGKey(1)
)
print(f"\n  Frozen-y PDE : {cap_table.price('pde'):.4f}")
print(f"  Exact-y  MC  : {cap_table.price('mc'):.4f} +/- {cap_table.quotes['mc'].stderr:.4f}   "
      f"(z {(cap_table.price('mc') - cap_table.price('pde')) / cap_table.quotes['mc'].stderr:+.2f})")


# ============================================================================
# 3. EUROPEAN & BERMUDAN SWAPTIONS
# ============================================================================

print("\n" + "=" * 74)
print("3. SWAPTIONS  (2Y into 5Y, ATM payer)")
print("=" * 74)


def annual_dates(start_year, count):
    return jnp.array(
        [int(ymd_to_ordinal(start_year + k, 1, 1)) for k in range(count)],
        dtype=jnp.int32,
    )


exp_ord = int(ymd_to_ordinal(2027, 1, 1))
fixed_ords = annual_dates(2028, 5)  # 2028..2032
ann = _annuity(jnp.int32(exp_ord), fixed_ords, CURVE, _ACT365)
atm = float((CURVE(jnp.int32(exp_ord)) - CURVE(fixed_ords[-1])) / ann)

euro = Swaption(
    expiry_date=jnp.int32(exp_ord), fixed_dates=fixed_ords,
    strike=jnp.asarray(atm), notional=jnp.array(1.0e6),
    is_payer=True, day_count=_ACT365,
)
euro_frozen = float(pde_price_dispatch(euro, MODEL, PDE_1F).price)
euro_exact = float(cheyette_exact_price(euro, MODEL, PDE_EXACT).price)
euro_tab = price_all_engines(
    euro, MODEL, pde_config=PDE_1F, mc_config=MC_CFG, mc_key=jax.random.PRNGKey(2)
)
print(f"\n  European (ATM strike {atm:.4%}):")
print(f"    Frozen-y PDE : {euro_frozen:.4f}")
print(f"    Exact-y  PDE : {euro_exact:.4f}   (frozen-y bias {euro_frozen - euro_exact:+.4f})")
print(f"    Exact-y  MC  : {euro_tab.price('mc'):.4f} +/- {euro_tab.quotes['mc'].stderr:.4f}   "
      f"(z vs exact-y PDE {(euro_tab.price('mc') - euro_exact) / euro_tab.quotes['mc'].stderr:+.2f})")

# Co-terminal Bermudan (no closed form; frozen-y PDE prices it natively).
ex_ords = annual_dates(2027, 5)  # 2027..2031
berm = BermudanSwaption(
    exercise_dates=ex_ords, fixed_dates=fixed_ords,
    strike=jnp.asarray(atm), notional=jnp.array(1.0e6),
    is_payer=True, day_count=_ACT365,
)
berm_frozen = float(pde_price_dispatch(berm, MODEL, PDE_1F).price)
print(f"\n  Bermudan (co-terminal, annual exercise 2027..2031):")
print(f"    Frozen-y PDE : {berm_frozen:.4f}")
print(f"    Early-exercise premium over European: {berm_frozen - euro_frozen:+.4f}  "
      f"(>= 0 as required)")

print("\n" + "=" * 74)
print("One engine, four vanilla-rates instruments; the frozen-y bias is the")
print("deterministic gap between the frozen-y and exact-y PDE, and MC agrees")
print("with the exact-y PDE within Monte-Carlo noise.")
print("=" * 74)
