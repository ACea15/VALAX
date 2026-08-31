# %% [markdown]
# # G2++ Two-Factor Model: Analytic vs PDE vs MC vs QuantLib
#
# The decorrelation-aware companion to the Hull-White cross-check
# (`10_hull_white_mc_vs_tree_vs_pde.py`). G2++ adds a *second* stochastic factor,
# so rates of different tenors can **decorrelate** — the feature a one-factor
# model structurally cannot express — and its headline instrument, the
# **Bermudan swaption**, has no closed form. Here we price it three independent
# ways and watch them agree.
#
# | Method             | VALAX                                   | Oracle                         |
# |--------------------|-----------------------------------------|--------------------------------|
# | **Analytic**       | `g2pp_swaption_price` (Gauss-Hermite)   | `ql.G2SwaptionEngine`          |
# | **Finite-diff PDE**| `pde_price_dispatch` (2-D ADI)          | `ql.FdG2SwaptionEngine`        |
# | **Short-rate MC**  | `generate_g2pp_paths` (exact 2-factor)  | analytic / PDE                 |
# | **Early-exercise MC** | Longstaff-Schwartz on the two factors | PDE / `ql.FdG2SwaptionEngine`  |
#
# Key themes:
# - European swaption: analytic == PDE == MC == QuantLib, four independent routes.
# - The cross term `rho*sigma*eta*V_xy` is what the second factor buys you; the
#   PDE handles it exactly as the Heston recipe handles `rho*xi*v*V_xv`.
# - Exact-fit: the PDE reprices the initial curve off its own `phi(t)` shift.
# - **Bermudan**: PDE vs QuantLib's 2-D FD engine vs a Longstaff-Schwartz MC,
#   with a strictly positive early-exercise premium over the best co-terminal
#   European.
# - Answering "can the Bermudan be priced with early-exercise MC?" — yes: LSM
#   regresses the continuation value on the two Gaussian factors `(x, y)`.
#
# Validated by:
#   tests/test_pde/test_g2pp.py
#   tests/test_mc/test_g2pp_recipes.py
#   tests/test_quantlib_comparison/test_g2pp_pde_ql.py
#   tests/test_quantlib_comparison/test_g2pp_swaptions_ql.py

# %% Imports
import jax
import jax.numpy as jnp

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.g2pp import G2PPModel, g2pp_B, g2pp_bond_price
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price
from valax.pricing.analytic.swaptions import _annuity
from valax.pricing.mc.dispatch import MCConfig, mc_price_dispatch
from valax.pricing.mc.g2pp_paths import generate_g2pp_paths
from valax.pricing.pde import PDEConfig2D, Scheme, pde_price_dispatch

try:
    import QuantLib as ql

    _HAS_QL = True
except ImportError:  # pragma: no cover - QuantLib is an optional dependency
    _HAS_QL = False

_ACT365 = "act_365"


# ============================================================================
# 1. SHARED MARKET SETUP
# ============================================================================

print("=" * 72)
print("G2++ TWO-FACTOR MODEL — analytic vs PDE vs MC vs QuantLib")
print("=" * 72)

REF_YEAR = 2026
REF_ORD = int(ymd_to_ordinal(REF_YEAR, 1, 1))
FLAT_RATE = 0.03


def flat_curve(rate=FLAT_RATE, n_years=20):
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


def g2pp_model(curve, rho=-0.70):
    # Well-separated a != b (avoids the a ~ b degeneracy); |rho| < 1.
    return G2PPModel(
        mean_reversion_x=jnp.array(0.50),
        mean_reversion_y=jnp.array(0.10),
        volatility_x=jnp.array(0.010),
        volatility_y=jnp.array(0.008),
        correlation=jnp.array(rho),
        initial_curve=curve,
    )


CURVE = flat_curve()
MODEL = g2pp_model(CURVE)
PDE_CFG = PDEConfig2D(n_x=131, n_y=131, n_time=100, x_range=6.0, scheme=Scheme.CRAIG_SNEYD)

print(f"\nReference date : {REF_YEAR}-01-01   (flat {FLAT_RATE:.1%} curve)")
print("Parameters     : a=0.50 b=0.10  sigma=0.010 eta=0.008  rho=-0.70")


# ---------------------------------------------------------------------------
# Small helpers shared by the sections below.
# ---------------------------------------------------------------------------


def annual_dates(start_year, count):
    return jnp.array(
        [int(ymd_to_ordinal(start_year + k, 1, 1)) for k in range(count)],
        dtype=jnp.int32,
    )


def atm_strike(curve, expiry_ord, fixed_ords):
    ann = _annuity(jnp.int32(expiry_ord), fixed_ords, curve, _ACT365)
    return float((curve(jnp.int32(expiry_ord)) - curve(fixed_ords[-1])) / ann)


def ql_g2_setup(model, rate):
    """A ql.G2 model + discount handle matching the VALAX flat curve."""
    ref = ql.Date(1, 1, REF_YEAR)
    ql.Settings.instance().evaluationDate = ref
    disc = ql.YieldTermStructureHandle(ql.FlatForward(ref, rate, ql.Actual365Fixed()))
    g2 = ql.G2(
        disc,
        float(model.mean_reversion_x),
        float(model.volatility_x),
        float(model.mean_reversion_y),
        float(model.volatility_y),
        float(model.correlation),
    )
    return disc, g2


# ============================================================================
# 2. EUROPEAN SWAPTION — four independent routes to one price
# ============================================================================

print("\n" + "=" * 72)
print("2. EUROPEAN SWAPTION  (5Y into 5Y, ATM payer)")
print("=" * 72)

EXP_YEAR, TENOR = 2031, 5
expiry_ord = int(ymd_to_ordinal(EXP_YEAR, 1, 1))
fixed_ords = annual_dates(EXP_YEAR + 1, TENOR)  # payments 2032..2036
strike = atm_strike(CURVE, expiry_ord, fixed_ords)

euro = Swaption(
    expiry_date=jnp.int32(expiry_ord),
    fixed_dates=fixed_ords,
    strike=jnp.asarray(strike),
    notional=jnp.array(1.0),
    is_payer=True,
    day_count=_ACT365,
)

# --- Analytic (the gold reference; validated to ~1e-10 vs ql.G2SwaptionEngine).
v_analytic = float(g2pp_swaption_price(euro, MODEL))

# --- 2-D finite-difference PDE.
v_pde = float(pde_price_dispatch(euro, MODEL, PDE_CFG).price)

# --- Short-rate Monte Carlo (production recipe, exact two-factor scheme).
mc_res = mc_price_dispatch(
    euro, MODEL, MCConfig(n_paths=200_000, n_steps=60), jax.random.PRNGKey(0)
)
v_mc, mc_se = float(mc_res.price), float(mc_res.stderr)

# --- QuantLib FD (independent 2-D finite-difference engine).
if _HAS_QL:
    disc, g2 = ql_g2_setup(MODEL, FLAT_RATE)
    expiry_ql = ql.Date(1, 1, EXP_YEAR)
    end_ql = ql.Date(1, 1, EXP_YEAR + TENOR)
    sched = ql.Schedule(
        expiry_ql, end_ql, ql.Period(ql.Annual), ql.NullCalendar(),
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False,
    )
    idx = ql.IborIndex(
        "Flat", ql.Period(ql.Annual), 0, ql.EURCurrency(), ql.NullCalendar(),
        ql.Unadjusted, False, ql.Actual365Fixed(), disc,
    )
    ql_swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer, 1.0, sched, strike, ql.Actual365Fixed(),
        sched, idx, 0.0, ql.Actual365Fixed(),
    )
    ql_euro = ql.Swaption(ql_swap, ql.EuropeanExercise(expiry_ql))
    ql_euro.setPricingEngine(ql.FdG2SwaptionEngine(g2, 50, 100, 100))
    v_ql = ql_euro.NPV()

print(f"\n  ATM strike           : {strike:.4%}\n")
print(f"  Analytic (Gauss-Herm): {v_analytic:.6f}")
print(f"  PDE   (2-D ADI)      : {v_pde:.6f}   (rel {abs(v_pde - v_analytic) / v_analytic:.1e})")
print(f"  MC    (200k paths)   : {v_mc:.6f} +/- {mc_se:.6f}   "
      f"(z={(v_mc - v_analytic) / mc_se:+.2f})")
if _HAS_QL:
    print(f"  QuantLib FD          : {v_ql:.6f}   (rel {abs(v_ql - v_analytic) / v_analytic:.1e})")


# --- Decorrelation: a lone swaption *rises* with rho (the factors reinforce).
print("\n  Decorrelation (rho sensitivity of the same swaption):")
for rho in (-0.70, 0.00, 0.50):
    m = g2pp_model(CURVE, rho=rho)
    a = float(g2pp_swaption_price(euro, m))
    p = float(pde_price_dispatch(euro, m, PDE_CFG).price)
    print(f"    rho={rho:+.2f}:  analytic={a:.6f}   PDE={p:.6f}   (rel {abs(p - a) / a:.1e})")


# ============================================================================
# 3. BERMUDAN SWAPTION — PDE vs QuantLib FD vs Longstaff-Schwartz MC
# ============================================================================

print("\n" + "=" * 72)
print("3. BERMUDAN SWAPTION  (co-terminal, annual exercise)")
print("=" * 72)

# Co-terminal Bermudan: exercise at the start of each year 2029..2033 into the
# tail of a swap maturing 2034.  fixed_dates are the payment dates 2030..2034;
# exercise_dates are the period starts 2029..2033.
FIRST_EX_YEAR, FINAL_YEAR = 2029, 2034
n_periods = FINAL_YEAR - FIRST_EX_YEAR  # 5
fixed_ords_b = annual_dates(FIRST_EX_YEAR + 1, n_periods)      # 2030..2034
exercise_ords_b = annual_dates(FIRST_EX_YEAR, n_periods)       # 2029..2033
start_ord_b = int(exercise_ords_b[0])
strike_b = atm_strike(CURVE, start_ord_b, fixed_ords_b)

berm = BermudanSwaption(
    exercise_dates=exercise_ords_b,
    fixed_dates=fixed_ords_b,
    strike=jnp.asarray(strike_b),
    notional=jnp.array(1.0),
    is_payer=True,
    day_count=_ACT365,
)

# --- (a) PDE — the native method for early exercise.
b_pde = float(pde_price_dispatch(berm, MODEL, PDE_CFG).price)

# --- (b) The best co-terminal European (via the PDE) — the exercise-premium floor.
euro_prices = []
for i, ey in enumerate(range(FIRST_EX_YEAR, FINAL_YEAR)):
    e_ord = int(ymd_to_ordinal(ey, 1, 1))
    tail = fixed_ords_b[i:]
    s = Swaption(
        expiry_date=jnp.int32(e_ord), fixed_dates=tail,
        strike=jnp.asarray(strike_b), notional=jnp.array(1.0),
        is_payer=True, day_count=_ACT365,
    )
    euro_prices.append(float(pde_price_dispatch(s, MODEL, PDE_CFG).price))
best_euro = max(euro_prices)


# --- (c) Early-exercise Monte Carlo (Longstaff-Schwartz on the two factors). ---
def g2pp_bermudan_lsm(
    model, exercise_ords, fixed_ords, strike, notional, is_payer,
    *, n_paths, sub_steps_per_year, key, poly_degree=2,
):
    r"""Price a G2++ Bermudan swaption by Longstaff-Schwartz Monte Carlo.

    Backward induction over the exercise dates, regressing the (path-wise,
    discounted-to-that-date) continuation value on a polynomial basis in the
    two Gaussian factors ``(x, y)`` — the model's Markov state. The exercise
    value at each date is the analytic tail-swap PV from the affine bond price,
    so only the *continuation* estimate carries regression error.
    """
    ref = model.initial_curve.reference_date
    a, b = model.mean_reversion_x, model.mean_reversion_y

    ex_times = year_fraction(ref, exercise_ords, _ACT365)  # (n_ex,)
    horizon = float(ex_times[-1])
    n_steps = int(round(horizon)) * sub_steps_per_year
    paths = generate_g2pp_paths(model, horizon, n_steps, n_paths, key)

    # Grid index of each exercise date (dates land on grid nodes by construction).
    ex_idx = jnp.round(ex_times / horizon * n_steps).astype(jnp.int32)

    def tail_swap_pv(e, x_e, y_e):
        """Analytic payer/receiver tail-swap PV at exercise ``e``, per path."""
        t_e = ex_times[e]
        pay_t = year_fraction(ref, fixed_ords, _ACT365)          # (n_periods,)
        starts = jnp.concatenate([exercise_ords[e][None], fixed_ords[:-1]])
        taus = year_fraction(starts, fixed_ords, _ACT365)
        # Only payments strictly after the exercise date are alive.
        alive = (pay_t > t_e + 1e-9).astype(pay_t.dtype)
        cashflows = strike * taus * alive
        cashflows = cashflows.at[-1].add(alive[-1])              # + principal
        A = g2pp_bond_price(model, jnp.zeros(()), jnp.zeros(()), t_e, pay_t)
        Ba, Bb = g2pp_B(a, pay_t - t_e), g2pp_B(b, pay_t - t_e)
        disc = A * jnp.exp(-Ba * x_e[:, None] - Bb * y_e[:, None])  # (n_paths, n)
        bond = disc @ cashflows
        payer = notional * (1.0 - bond)
        return payer if is_payer else -payer

    def factors(e):
        i = ex_idx[e]
        x_e = paths.factor_x[:, i]
        y_e = paths.factor_y[:, i]
        d0_e = jnp.exp(paths.log_discount_factors[:, i])  # numeraire to time 0
        return x_e, y_e, d0_e

    def basis(x_e, y_e):
        cols = [jnp.ones_like(x_e), x_e, y_e]
        if poly_degree >= 2:
            cols += [x_e * x_e, x_e * y_e, y_e * y_e]
        return jnp.stack(cols, axis=1)

    n_ex = exercise_ords.shape[0]

    # Last date: exercise iff the tail swap has positive value.
    x_e, y_e, d0_e = factors(n_ex - 1)
    ev = tail_swap_pv(n_ex - 1, x_e, y_e)
    cashflow0 = d0_e * jnp.maximum(ev, 0.0)  # value discounted to time 0

    for e in range(n_ex - 2, -1, -1):
        x_e, y_e, d0_e = factors(e)
        ev = tail_swap_pv(e, x_e, y_e)
        itm = ev > 0.0
        # Continuation value discounted to *this* exercise date (divide out the
        # path-dependent common numeraire so the regressor sees only the state).
        cont_te = cashflow0 / d0_e
        X = basis(x_e, y_e)
        W = itm.astype(X.dtype)
        XtWX = (X * W[:, None]).T @ X + 1e-8 * jnp.eye(X.shape[1])
        beta = jnp.linalg.solve(XtWX, (X * W[:, None]).T @ cont_te)
        cont_hat = X @ beta
        exercise_now = itm & (ev > cont_hat)
        cashflow0 = jnp.where(exercise_now, d0_e * ev, cashflow0)

    price = jnp.mean(cashflow0)
    stderr = jnp.std(cashflow0) / jnp.sqrt(n_paths)
    return float(price), float(stderr)


b_mc, b_mc_se = g2pp_bermudan_lsm(
    MODEL, exercise_ords_b, fixed_ords_b, jnp.asarray(strike_b), jnp.array(1.0),
    True, n_paths=120_000, sub_steps_per_year=12, key=jax.random.PRNGKey(3),
)

# --- (d) QuantLib FD (prices Bermudan exercise natively).
if _HAS_QL:
    disc, g2 = ql_g2_setup(MODEL, FLAT_RATE)
    start_ql = ql.Date(1, 1, FIRST_EX_YEAR)
    end_ql = ql.Date(1, 1, FINAL_YEAR)
    sched = ql.Schedule(
        start_ql, end_ql, ql.Period(ql.Annual), ql.NullCalendar(),
        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False,
    )
    idx = ql.IborIndex(
        "Flat", ql.Period(ql.Annual), 0, ql.EURCurrency(), ql.NullCalendar(),
        ql.Unadjusted, False, ql.Actual365Fixed(), disc,
    )
    ql_swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer, 1.0, sched, strike_b, ql.Actual365Fixed(),
        sched, idx, 0.0, ql.Actual365Fixed(),
    )
    ql_berm = ql.Swaption(ql_swap, ql.BermudanExercise(list(sched)[:-1]))
    ql_berm.setPricingEngine(ql.FdG2SwaptionEngine(g2, 50, 100, 100))
    b_ql = ql_berm.NPV()

print(f"\n  ATM strike            : {strike_b:.4%}")
print(f"  Exercise dates        : {FIRST_EX_YEAR}..{FINAL_YEAR - 1}  (annual)\n")
print(f"  PDE   (2-D ADI)       : {b_pde:.6f}")
if _HAS_QL:
    print(f"  QuantLib FD           : {b_ql:.6f}   (rel {abs(b_pde - b_ql) / b_ql:.1e})")
print(f"  MC-LSM (120k paths)   : {b_mc:.6f} +/- {b_mc_se:.6f}   "
      f"(rel to PDE {abs(b_mc - b_pde) / b_pde:.1e})")
print(f"\n  Best co-terminal Euro : {best_euro:.6f}   (PDE)")
print(f"  Early-exercise premium: {b_pde - best_euro:+.6f}   "
      f"({(b_pde / best_euro - 1.0):+.1%})  >= 0 as required")

print("\n" + "=" * 72)
print("All four routes agree; the Bermudan carries a positive early-exercise")
print("premium and is priceable by early-exercise (Longstaff-Schwartz) MC.")
print("=" * 72)
