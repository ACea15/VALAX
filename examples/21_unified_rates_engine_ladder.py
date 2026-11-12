"""Example 21 — The rates engine ladder: one trade, many models and engines, one call.

A European and a Bermudan swaption on the same underlying swap, priced
through ``price()`` under Hull-White, G2++, Cheyette (1F and 2F) and the LMM
by every numerical method each pair registers. Between rows only ``model=``,
``method=`` and ``config=`` change — the instrument and the ``MarketState``
are the same objects throughout. Models are curve-free: ``price()`` binds the
market curve.

  1. Market and trades: a single-curve market (Cheyette refuses a distinct
     projection curve, so one curve discounts and projects), a 2Y-into-5Y
     European payer at the money, and the Bermudan on the same swap with
     annual exercise 2Y…6Y (European expiry = first exercise date).
  2. The European ladder: ``available_methods`` / ``auto_method`` /
     ``effective_config`` and the price of every engine. Within each model the
     engines agree: Hull-White tree and PDE against Jamshidian (the tree's
     error is first order — Richardson extrapolation removes it), G2++ PDE
     and MC against Brigo-Mercurio, Cheyette MC against its PDE. No
     cross-model equality is claimed.
  3. The Bermudan ladder: Bermudan ≥ European under every model and engine;
     Hull-White tree vs PDE; the LMM's Longstaff-Schwartz price (a low-biased
     estimator with no path-wise stderr) checked by its dispersion across
     seeds.
  4. Policy: ``auto_method`` for every pair, ``method="auto"`` is the
     explicit method, and the errors for an unregistered method, an unknown
     method name and an MC call without a key.
  5. Risk through the same call: ``market_risk`` of the European under
     Hull-White analytic / PDE / tree — the curve pillar deltas agree across
     engines; a parallel DV01 from the report vs a ±1 bp ``shock``; one LMM
     report with a fixed key (common random numbers), checked against a
     same-key bump.
  6. JIT: ``eqx.filter_jit`` of ``price()`` for the Bermudan G2++ PDE, and an
     ``eqx.filter_vmap`` strike ladder under Hull-White analytic, both equal
     to the eager calls.

Run::

    python examples/21_unified_rates_engine_ladder.py
"""

# %% Imports and setup
from __future__ import annotations

import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import valax
from valax.core import (
    MarketState,
    auto_method,
    available_methods,
    effective_config,
    price,
)
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.greeks import market_risk
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.market import KeyedScenario, SeedRegistry
from valax.models import (
    CheyetteModel,
    CheyetteModel2F,
    G2PPModel,
    HullWhiteModel,
    LMMModel,
)
from valax.models.lmm import ExponentialCorrelation, PiecewiseConstantVol
from valax.pricing.analytic.swaptions import swaption_forward_and_annuity
from valax.pricing.lattice import TreeConfig
from valax.pricing.mc import MCConfig
from valax.pricing.pde.config import PDEConfig, PDEConfig2D
from valax.risk.state_shocks import shock

FAST = os.environ.get("VALAX_EXAMPLES_FAST") == "1"
T0 = time.perf_counter()

# Engine settings. Fast mode runs small explicit grids / trees / path sets;
# full mode leaves the PDE grids to the pricing policy (config=None, the
# pair's default — shown by effective_config) and refines the tree and MC.
TREE_N = 50 if FAST else 200
TREE = TreeConfig(n_steps=TREE_N)
TREE_2N = TreeConfig(n_steps=2 * TREE_N)         # for the Richardson check
PDE_1D = PDEConfig(n_spot=100, n_time=50) if FAST else None
PDE_2D = PDEConfig2D(n_x=41, n_y=41, n_time=30) if FAST else None
MC = MCConfig(n_paths=8_192, n_steps=24) if FAST else MCConfig(n_paths=65_536, n_steps=48)
N_SEEDS = 4                                       # LSM seed dispersion

# Tolerances, each from measured errors on this trade (see the comments in
# sections 2, 3 and 5); full mode's are tighter because its grids are.
HW_PDE_TOL = 2e-4 if FAST else 2e-5     # measured 7.4e-5 (100x50), 4.5e-6 (200x200)
G2_PDE_TOL = 4e-3 if FAST else 6e-4     # measured 1.9e-3 (41²x30), 2.6e-4 (128x64x200)
TREE_EURO_C = 0.25                      # rel. error ≤ C / n_steps; measured 0.17 / n
TREE_BERM_C = 0.6                       # vs PDE; measured 0.41 / n
HW_PDE_BERM_TOL = 1e-3 if FAST else 3e-4   # measured 5.4e-4, 1.5e-4 vs a 800x800 grid
RICHARDSON_TOL = 1e-4                   # measured 1.4e-5 (n=50/100), 1.4e-6 (200/400)
N_SIGMA = 3.0


def header(s: str) -> None:
    print(f"\n{'=' * 72}\n  {s}\n{'=' * 72}")


def cfg_str(cfg) -> str:
    """A compact rendering of an engine config."""
    if cfg is None:
        return "—"
    if isinstance(cfg, MCConfig):
        return f"MC {cfg.n_paths}p×{cfg.n_steps}s"
    if isinstance(cfg, PDEConfig2D):
        return f"ADI {cfg.n_x}×{cfg.n_y}×{cfg.n_time}"
    if isinstance(cfg, PDEConfig):
        return f"CN {cfg.n_spot}×{cfg.n_time}"
    if isinstance(cfg, TreeConfig):
        return f"tree {cfg.n_steps}"
    return type(cfg).__name__


registry = SeedRegistry(master_seed=20260321, library_version=valax.__version__)

# %% 1. Market and trades
header("1. One single-curve market, one European and one Bermudan swaption")

REF = int(ymd_to_ordinal(2026, 1, 1))
DC = "act_365"
PILLAR_YEARS = (0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30)
pillars = jnp.array([REF + 365 * y for y in PILLAR_YEARS], dtype=jnp.int32)
t_pillar = year_fraction(jnp.int32(REF), pillars, DC)
# Nelson-Siegel zero curve: a mildly inverted front end (3.6% → 3.5% at 2Y)
# rising to ~3.9% at 30Y.
x = jnp.maximum(t_pillar / 2.0, 1e-12)
ns_slope = jnp.where(t_pillar > 0, (1.0 - jnp.exp(-x)) / x, 1.0)
zero = 0.040 - 0.004 * ns_slope - 0.010 * (ns_slope - jnp.exp(-x))
curve = DiscountCurve(pillar_dates=pillars, discount_factors=jnp.exp(-zero * t_pillar),
                      reference_date=jnp.int32(REF), day_count=DC)
market = MarketState.single_curve(curve)
N_PILLARS = len(PILLAR_YEARS)
print("  zero rates: " + "  ".join(f"{y}Y {float(z):.2%}" for y, z in
                                   zip(PILLAR_YEARS[1:], zero[1:])))


def years(*ys: int) -> jax.Array:
    return jnp.array([REF + 365 * y for y in ys], dtype=jnp.int32)


NOTIONAL = 1e6
probe = Swaption(expiry_date=years(2)[0], fixed_dates=years(3, 4, 5, 6, 7),
                 strike=jnp.asarray(0.0), notional=jnp.asarray(NOTIONAL), day_count=DC)
fwd, ann = swaption_forward_and_annuity(probe, curve)
STRIKE = jnp.round(fwd * 1e4) / 1e4            # ATM, rounded to 1 bp
european = eqx.tree_at(lambda s: s.strike, probe, STRIKE)
bermudan = BermudanSwaption(exercise_dates=years(2, 3, 4, 5, 6),
                            fixed_dates=years(3, 4, 5, 6, 7), strike=STRIKE,
                            notional=jnp.asarray(NOTIONAL), day_count=DC)
assert int(european.expiry_date) == int(bermudan.exercise_dates[0])
assert bool(jnp.all(european.fixed_dates == bermudan.fixed_dates))
print(f"  2Yx5Y forward swap rate {float(fwd):.4%}, annuity {float(ann):.4f}; "
      f"strike {float(STRIKE):.2%} (payer, notional {NOTIONAL:,.0f})")
print("  European: expiry 2Y into 3Y…7Y annual fixed; Bermudan: same swap, "
      "exercise 2Y…6Y")
print("  The market is single-curve on purpose: Cheyette's recipes take no")
print("  projection curve and price() refuses a trade naming a distinct one.")
print("  ✓ European expiry = first Bermudan exercise; identical fixed schedules")

# Models: curve-free (initial_curve=None) — price() binds the market curve.
# Parameters are chosen to give comparable vol levels; the models are
# different, so their prices differ and only within-model engines are compared.
MODELS = {
    "Hull-White": HullWhiteModel(mean_reversion=jnp.asarray(0.03),
                                 volatility=jnp.asarray(0.009)),
    "G2++": G2PPModel(mean_reversion_x=jnp.asarray(0.5), mean_reversion_y=jnp.asarray(0.05),
                      volatility_x=jnp.asarray(0.0075), volatility_ratio=jnp.asarray(1.2),
                      correlation=jnp.asarray(-0.6)),
    "Cheyette": CheyetteModel(mean_reversion=jnp.asarray(0.03),
                              volatility=jnp.asarray(0.009), skew=jnp.asarray(4.0)),
    "Cheyette2F": CheyetteModel2F(
        mean_reversion_1=jnp.asarray(0.5), mean_reversion_2=jnp.asarray(0.05),
        volatility_1=jnp.asarray(0.0075), volatility_ratio=jnp.asarray(1.2),
        skew_1=jnp.asarray(3.0), skew_2=jnp.asarray(1.0), correlation=jnp.asarray(-0.6)),
    # The LMM's annual tenor grid T_k = 1Y…7Y carries both trades' periods
    # (the recipes map the trade onto the grid from its dates).
    "LMM": LMMModel(tenor_dates=years(1, 2, 3, 4, 5, 6, 7),
                    vol_structure=PiecewiseConstantVol(vols=jnp.full((6, 6), 0.22)),
                    corr_structure=ExponentialCorrelation(beta=jnp.asarray(0.1))),
}
assert all(getattr(m, "initial_curve", None) is None for m in MODELS.values())
print(f"  models (curve-free): {', '.join(MODELS)}")
print("  ✓ every model is curve-free; price() binds the market curve")


def run_config(name: str, method: str):
    """The config this example passes for one (model, method)."""
    two_d = name in ("G2++", "Cheyette2F")
    return {"lattice": TREE, "pde": PDE_2D if two_d else PDE_1D, "mc": MC}.get(method)


def mc_key(inst, name: str) -> jax.Array:
    return registry.key(f"example21.{type(inst).__name__}.{name}")


def ladder(inst) -> dict:
    """Price ``inst`` under every model by every registered method."""
    out = {}
    print(f"  {'model':<11}{'methods':<22}{'auto':<9}{'method':<9}{'config':<18}"
          f"{'price':>11}{'± stderr':>10}")
    for name, model in MODELS.items():
        methods = available_methods(inst, model)
        auto = auto_method(inst, model)
        for i, method in enumerate(methods):
            cfg = run_config(name, method)
            res = price(inst, model, market, method=method, config=cfg,
                        key=mc_key(inst, name) if method == "mc" else None)
            assert res.method == method
            assert res.config == effective_config(inst, model, method, cfg)
            out[name, method] = res
            se = "" if res.stderr is None else f"{float(res.stderr):10.2f}"
            print(f"  {name if i == 0 else '':<11}{','.join(methods) if i == 0 else '':<22}"
                  f"{auto if i == 0 else '':<9}{method:<9}{cfg_str(res.config):<18}"
                  f"{float(res.price):>11.2f}{se}")
    return out


def rel(a, b) -> float:
    return abs(float(a) / float(b) - 1.0)


# %% 2. The European ladder
header("2. European ladder: switch only model= and method=")

E = ladder(european)
print()

# Hull-White: Jamshidian is the reference.
hw_ref = E["Hull-White", "analytic"].price
err_pde = rel(E["Hull-White", "pde"].price, hw_ref)
err_tree = rel(E["Hull-White", "lattice"].price, hw_ref)
print(f"  Hull-White  PDE vs Jamshidian {err_pde:.1e} (tol {HW_PDE_TOL:.0e});  "
      f"tree {err_tree:.1e} (tol {TREE_EURO_C}/n = {TREE_EURO_C / TREE_N:.1e})")
assert err_pde < HW_PDE_TOL
assert err_tree < TREE_EURO_C / TREE_N
# The tree's error is first order in n_steps (measured 136 / 69 / 34 / 17 /
# 8.6 currency units at n = 25 / 50 / 100 / 200 / 400), so one Richardson
# step 2·P(2n) − P(n) removes it.
tree_2n = price(european, MODELS["Hull-White"], market, method="lattice", config=TREE_2N).price
richardson = 2.0 * tree_2n - E["Hull-White", "lattice"].price
err_rich = rel(richardson, hw_ref)
print(f"  Hull-White  tree n={TREE_N}/{2 * TREE_N}: Richardson 2P(2n)−P(n) = "
      f"{float(richardson):.2f}, vs Jamshidian {err_rich:.1e} (tol {RICHARDSON_TOL:.0e})")
assert err_rich < RICHARDSON_TOL
print("  ✓ Hull-White: PDE and tree agree with Jamshidian; the tree error is first order")

# G2++: Brigo-Mercurio closed form is the reference; MC within 3 stderr.
g2_ref = E["G2++", "analytic"].price
err_g2 = rel(E["G2++", "pde"].price, g2_ref)
g2_mc = E["G2++", "mc"]
z_g2 = (float(g2_mc.price) - float(g2_ref)) / float(g2_mc.stderr)
print(f"  G2++        PDE vs Brigo-Mercurio {err_g2:.1e} (tol {G2_PDE_TOL:.0e});  "
      f"MC {z_g2:+.2f} stderr")
assert err_g2 < G2_PDE_TOL
assert abs(z_g2) < N_SIGMA
print("  ✓ G2++: PDE and MC agree with the closed form")

# Cheyette: no closed form. MC (Euler, local vol frozen per step) vs the
# frozen-y PDE. Both carry an approximation at β ≠ 0; measured at this β = 4
# with 10×100k paths, MC − PDE(481×240) = +34 ± 30 (48 steps) and +42 ± 41
# (24 steps): any bias is ≲ 0.3%, under this run's MC noise, so 3 stderr is
# honest.
ch_pde, ch_mc = E["Cheyette", "pde"], E["Cheyette", "mc"]
z_ch = (float(ch_mc.price) - float(ch_pde.price)) / float(ch_mc.stderr)
print(f"  Cheyette    MC vs frozen-y PDE {z_ch:+.2f} stderr")
assert abs(z_ch) < N_SIGMA
print("  ✓ Cheyette: MC and PDE agree within MC error")
print("  Cheyette2F (PDE) and LMM (MC) register one engine each for the European:")
print("  nothing to cross-check within the model; their rows are reported only.")
assert float(E["LMM", "mc"].stderr) > 0.0
assert all(np.isfinite(float(r.price)) and float(r.price) > 0.0 for r in E.values())

# %% 3. The Bermudan ladder
header("3. Bermudan ladder: same models, same market, the early-exercise trade")

B = ladder(bermudan)
print()

# Each model's European reference: its exact / most accurate engine.
euro_ref = {"Hull-White": E["Hull-White", "analytic"], "G2++": E["G2++", "analytic"],
            "Cheyette": E["Cheyette", "pde"], "Cheyette2F": E["Cheyette2F", "pde"],
            "LMM": E["LMM", "mc"]}

# LMM: Longstaff-Schwartz reports stderr 0 (the regression makes a path-wise
# estimate unreliable), so the sampling error comes from N_SEEDS independent
# seeds, priced by one compiled program.
lmm = MODELS["LMM"]
lsm = eqx.filter_jit(lambda k: price(bermudan, lmm, market, method="mc", config=MC, key=k).price)
seeds = jax.random.split(registry.key("example21.lsm_seeds"), N_SEEDS)
lsm_prices = np.array([float(lsm(k)) for k in seeds])
lsm_mean = float(lsm_prices.mean())
lsm_se = float(lsm_prices.std(ddof=1) / np.sqrt(N_SEEDS))
assert float(B["LMM", "mc"].stderr) == 0.0
print(f"  LMM LSM over {N_SEEDS} seeds: {lsm_mean:.2f} ± {lsm_se:.2f} "
      f"(per-seed {', '.join(f'{p:.0f}' for p in lsm_prices)})")

for (name, method), res in B.items():
    e = euro_ref[name]
    if name == "LMM":
        # LSM is low-biased: only require it to clear the European less noise.
        floor = float(e.price) - N_SIGMA * float(np.hypot(float(e.stderr), lsm_se))
        value = lsm_mean
    else:
        # Engine error of the comparison: the tree's first-order error, the
        # 1-D CN grid's (HW, Cheyette) or the 2-D ADI grid's (G2++, Cheyette2F).
        if method == "lattice":
            tol = TREE_BERM_C / TREE_N
        elif name in ("Hull-White", "Cheyette"):
            tol = HW_PDE_BERM_TOL
        else:
            tol = G2_PDE_TOL
        floor = float(e.price) * (1.0 - tol)
        value = float(res.price)
    print(f"  {name:<11}{method:<9} Bermudan {value:10.2f} ≥ European {float(e.price):10.2f}"
          f"  (exercise premium {value / float(e.price) - 1.0:+.1%})")
    assert np.isfinite(value) and value >= floor, (name, method, value, floor)
print("  ✓ Bermudan ≥ European under every model and engine (up to engine error)")

# Hull-White tree vs PDE: the tree's Bermudan error is first order too
# (measured 0.40–0.41 / n_steps relative at n = 50 / 200 vs an 800×800 PDE).
err_b = rel(B["Hull-White", "lattice"].price, B["Hull-White", "pde"].price)
tol_b = TREE_BERM_C / TREE_N + HW_PDE_BERM_TOL
print(f"  Hull-White  tree vs PDE {err_b:.1e} (tol {TREE_BERM_C}/n + PDE error = {tol_b:.1e})")
assert err_b < tol_b
print("  ✓ Hull-White Bermudan: tree and PDE agree")
print("  G2++, Cheyette and Cheyette2F price the Bermudan on their PDE only.")

# %% 4. Policy
header("4. Policy: auto_method, method='auto', and what an unregistered method says")

print(f"  {'instrument':<18}" + "".join(f"{n:>12}" for n in MODELS))
for inst in (european, bermudan):
    print(f"  {type(inst).__name__:<18}"
          + "".join(f"{auto_method(inst, m):>12}" for m in MODELS.values()))

# method="auto" resolves to the policy's method: same config (and key) ⇒
# bit-identical price.
for inst, name, done in ((european, "Hull-White", E), (bermudan, "G2++", B),
                         (european, "LMM", E)):
    method = auto_method(inst, MODELS[name])
    auto = price(inst, MODELS[name], market, config=run_config(name, method),
                 key=mc_key(inst, name) if method == "mc" else None)
    assert auto.method == method
    assert float(auto.price) == float(done[name, method].price)
    print(f"  price({type(inst).__name__}, {name}, method='auto') → {auto.method}: "
          f"{float(auto.price):.2f} == explicit")
print("  ✓ method='auto' is exactly the explicit policy method")


def refused(call, must_contain: str) -> str:
    try:
        call()
    except ValueError as exc:
        msg = str(exc)
        assert must_contain in msg, msg
        return msg
    raise AssertionError("expected a ValueError")


msg = refused(lambda: price(bermudan, MODELS["Hull-White"], market, method="analytic"),
              "No analytic pricer registered for (BermudanSwaption, HullWhiteModel)")
listed = [line.strip() for line in msg.splitlines()[2:]]
assert "(BermudanSwaption, HullWhiteModel, pde)" in listed
print(f"  analytic Bermudan under HW → ValueError: {msg.splitlines()[0]}")
print(f"    …then lists all {len(listed)} registered (instrument, model, method), e.g.")
for line in listed:
    if line.startswith("(BermudanSwaption, HullWhiteModel"):
        print(f"      {line}")
msg = refused(lambda: price(european, MODELS["Hull-White"], market, method="fft"),
              "Unknown pricing method")
print(f"  method='fft' → ValueError: {msg}")
msg = refused(lambda: price(european, lmm, market, method="mc", config=MC),
              "needs a PRNG key")
print(f"  LMM mc without key → ValueError: {msg}")
print("  ✓ unregistered / unknown / keyless requests fail early with a useful message")

# %% 5. Risk through the same call
header("5. market_risk: curve pillar deltas across engines, DV01 vs a ±1 bp shock")

hw = MODELS["Hull-White"]
reports = {
    "analytic": market_risk(european, hw, market, method="analytic"),
    "pde": market_risk(european, hw, market, method="pde", config=PDE_1D),
    "lattice": market_risk(european, hw, market, method="lattice", config=TREE),
}
deltas = {m: r.curves["_default_"] for m, r in reports.items()}
assert all(r.declared.curves == ("_default_",) for r in reports.values())
df = curve.discount_factors


def dv01_of(g: jax.Array) -> float:
    """Parallel zero-rate DV01 (+1 bp) from ∂V/∂DF: Σ g_i · (−t_i DF_i) · 1e-4."""
    return float(jnp.sum(g * (-t_pillar * df)) * 1e-4)


print(f"  ∂V/∂DF per pillar (Hull-White European):")
print(f"  {'pillar':>8}" + "".join(f"{m:>14}" for m in deltas))
for i, y in enumerate(PILLAR_YEARS[:8]):
    print(f"  {y:>7}Y" + "".join(f"{float(g[i]):>14,.1f}" for g in deltas.values()))
print(f"  {'DV01':>8}" + "".join(f"{dv01_of(g):>14,.4f}" for g in deltas.values()))

# The t = 0 pillar (DF ≡ 1) is a gauge direction: the analytic and tree
# routes give it 0, the PDE route ≈ −V (it reads the curve relative to
# DF(0)). A zero-rate shock never moves it, so the DV01s agree; the
# comparison below is on the t > 0 pillars.
v_pde = float(reports["pde"].price)
assert float(deltas["analytic"][0]) == 0.0
assert abs(float(deltas["pde"][0]) + v_pde) < 1e-3 * v_pde
print(f"  t=0 pillar: analytic {float(deltas['analytic'][0]):.1f}, PDE "
      f"{float(deltas['pde'][0]):,.1f} ≈ −V = {-v_pde:,.1f} (a gauge: DF(0) ≡ 1)")

scale = float(jnp.max(jnp.abs(deltas["analytic"])))
# Measured max |Δ| / max |g| on t > 0: PDE 4e-5 (100×50), 1.7e-5 (200×200);
# tree 2.5e-4 (n=50), 9e-5 (n=200) — about 0.013–0.018 / n_steps.
risk_tol = {"pde": 1e-4 if FAST else 4e-5, "lattice": 0.03 / TREE_N}
for m in ("pde", "lattice"):
    gap = float(jnp.max(jnp.abs(deltas[m][1:] - deltas["analytic"][1:]))) / scale
    print(f"  {m:<8} vs analytic: max |Δ ∂V/∂DF| / max |∂V/∂DF| = {gap:.1e} "
          f"(tol {risk_tol[m]:.1e})")
    assert gap < risk_tol[m]
    assert rel(dv01_of(deltas[m]), dv01_of(deltas["analytic"])) < risk_tol[m]
print("  ✓ pillar deltas and DV01 agree across analytic / PDE / tree")

up = shock(market, KeyedScenario(curve_shocks={"_default_": jnp.full(N_PILLARS, 1e-4)}))
dn = shock(market, KeyedScenario(curve_shocks={"_default_": jnp.full(N_PILLARS, -1e-4)}))
fd = (float(price(european, hw, up, method="analytic").price)
      - float(price(european, hw, dn, method="analytic").price)) / 2.0
ad = dv01_of(deltas["analytic"])
print(f"  Hull-White analytic DV01: market_risk {ad:.6f}   ±1 bp shock {fd:.6f}"
      f"   (rel {ad / fd - 1:+.1e})")
assert rel(ad, fd) < 1e-5     # central difference: O(h²) convexity, measured 1.5e-6
print("  ✓ the report's parallel DV01 is what full revaluation measures")

# LMM: a fixed key makes the MC price a deterministic, differentiable function
# of the curve (common random numbers), so one backward pass gives path-wise
# curve deltas and a same-key bump reproduces them.
lmm_key = registry.key("example21.lmm_risk")
lmm_risk = market_risk(european, lmm, market, method="mc", config=MC, key=lmm_key)
g_lmm = lmm_risk.curves["_default_"]
lmm_pv_jit = eqx.filter_jit(
    lambda m: price(european, lmm, m, method="mc", config=MC, key=lmm_key).price)


def lmm_pv(m: MarketState) -> float:
    return float(lmm_pv_jit(m))


assert rel(lmm_risk.price, lmm_pv(market)) < 1e-12   # same key ⇒ same paths
fd_lmm = (lmm_pv(up) - lmm_pv(dn)) / 2.0
ad_lmm = dv01_of(g_lmm)
print(f"  LMM (MC, fixed key) price {float(lmm_risk.price):,.2f}; ∂V/∂DF on "
      f"2Y…7Y: " + ", ".join(f"{float(v):,.0f}" for v in g_lmm[2:7]))
print(f"  LMM DV01: market_risk {ad_lmm:.4f}   same-key ±1 bp shock {fd_lmm:.4f}"
      f"   (rel {ad_lmm / fd_lmm - 1:+.1e})")
assert bool(jnp.all(jnp.isfinite(g_lmm)))
assert float(jnp.max(jnp.abs(g_lmm))) > 0.0 and ad_lmm != 0.0
# Path-wise AD vs a same-key bump of a kinked payoff: measured 2.5e-4 (fast),
# 1.6e-4 (full).
assert rel(ad_lmm, fd_lmm) < 1e-3
print("  ✓ LMM curve deltas are finite, non-zero, and match a common-random-number bump")

# %% 6. JIT and vmap
header("6. filter_jit of price(), filter_vmap over a strike ladder")

g2 = MODELS["G2++"]
berm_g2 = eqx.filter_jit(lambda inst, mkt: price(inst, g2, mkt, method="pde",
                                                  config=PDE_2D).price)
jitted = float(berm_g2(bermudan, market))
eager = float(B["G2++", "pde"].price)
print(f"  Bermudan G2++ PDE: filter_jit {jitted:.6f}   eager {eager:.6f}")
assert abs(jitted - eager) <= 1e-9 * eager
print("  ✓ price() compiles end to end with instrument and market as traced arguments")

strikes = STRIKE + jnp.linspace(-0.01, 0.01, 9)


@eqx.filter_jit
def strike_ladder(ks: jax.Array) -> jax.Array:
    def one(k):
        return price(eqx.tree_at(lambda s: s.strike, european, k), hw, market,
                     method="analytic").price
    return eqx.filter_vmap(one)(ks)


vmapped = strike_ladder(strikes)
loop = jnp.array([float(price(eqx.tree_at(lambda s: s.strike, european, k), hw, market,
                                    method="analytic").price) for k in strikes])
print("  strike   " + "".join(f"{float(k):>9.2%}" for k in strikes))
print("  price    " + "".join(f"{float(v):>9.0f}" for v in vmapped))
np.testing.assert_allclose(np.asarray(vmapped), np.asarray(loop), rtol=1e-12, atol=1e-8)
assert bool(jnp.all(jnp.diff(vmapped) < 0.0))   # a payer loses value as the strike rises
print("  ✓ one vmapped, compiled call == the eager loop; prices fall with the strike")

print(f"\n  All checks passed ({'fast' if FAST else 'full'} mode, "
      f"{time.perf_counter() - T0:.1f}s).")
