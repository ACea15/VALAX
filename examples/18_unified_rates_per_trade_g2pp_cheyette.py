"""Example 18 — Rates: per-trade G2++ and Cheyette on the calibrated cube → Bermudan risk.

The two-factor and skewed siblings of example 14's per-trade Hull-White:

  1. Sample a true rates market (OIS + 3M projection curves, a SABR swaption
     cube in normal vols) and quote it with Gaussian noise: 0.5 bp on rates,
     0.5 normal-vol bp on the cube; calibrate curves → cube.
  2. Calibrate G2++ **per trade**: each Bermudan gets the two factor vols
     (σ, η) fitted to its own basket off the cube, with the mean reversions
     (a, b) and the correlation ρ held fixed as fixed inputs. A co-terminal
     basket alone cannot separate the two factors (the fit is singular), so
     the basket adds the 1Y-tenor swaption at the first exercise date (short
     vs. long rates is what the second factor moves). G2++ has no analytic
     swaption route in
     ``price``; the basket is priced through ``price(..., method="pde")``
     (2-D ADI), and same-shaped baskets are fitted in one ``eqx.filter_vmap``.
  3. Price the Bermudans through ``price`` (G2++ PDE) and check the basket
     PDE against the Brigo-Mercurio closed form.
  4. Quote risk of one Bermudan by the adjoint sweep (λ_m through cube and
     curves) and independently by autodiff through every calibration and
     every PDE; the basket target shifts give the vega profile; one target
     shift and one cube quote are checked by full recalibration.
  5. Cheyette (frozen-y PDE, σ and skew β fitted to a two-strike co-terminal
     basket). The Cheyette PDE has no discount/projection split, so it runs
     on a **single-curve** market (OIS discounts and projects) and ``price``
     refuses a trade that names a distinct projection curve. Adjoint vs.
     autodiff, one FD check.
  6. Model risk: Hull-White, G2++ and Cheyette, each fitted per trade to the
     same Bermudan's basket on the same single-curve market.

Run::

    python examples/18_unified_rates_per_trade_g2pp_cheyette.py
"""

# %% Imports and setup
from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import valax
from valax.calibration import (
    CalibrationChain,
    calibrate_market,
    calibrate_model,
    coterminal_swaptions,
    positive,
    swaption_basket_node,
)
from valax.calibration.cheyette import CHEYETTE_SETTINGS, cheyette_provenance
from valax.calibration.transforms import free
from valax.core.binding import bind_curves
from valax.core.pricing import price
from valax.greeks import quote_risk_adjoint, quote_risk_autodiff, reconcile_quote_risk
from valax.instruments.rates import BermudanSwaption
from valax.market import (
    QuoteSet,
    RatesVolSpec,
    RatesWorldSpec,
    SeedRegistry,
    default_config,
    rates_vol_snapshot,
)
from valax.models.cheyette import CheyetteModel
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price
from valax.pricing.analytic.rates_smile import swaption_price
from valax.pricing.analytic.swaptions import swaption_forward_and_annuity
from valax.pricing.pde.config import PDEConfig, PDEConfig2D
from valax.surfaces.constant import ConstantVol

jax.config.update("jax_enable_x64", True)

FAST = os.environ.get("VALAX_EXAMPLES_FAST") == "1"
VOL_NOISE_BP = 0.5
# G2++ (two-factor ADI) and Cheyette (frozen-y, one-factor) grids. The fits
# are *on-engine*: basket and Bermudan share the grid, so the grid only sets
# how close the engine is to the continuous model, not the calibration.
PDE_G2 = (PDEConfig2D(n_x=41, n_y=41, n_time=40, x_range=5.0) if FAST
          else PDEConfig2D(n_x=81, n_y=81, n_time=100, x_range=5.0))
PDE_CH = (PDEConfig(n_spot=101, n_time=60, spot_range=5.0) if FAST
          else PDEConfig(n_spot=201, n_time=150, spot_range=5.0))
PDE_HW = PDEConfig(n_spot=100, n_time=60) if FAST else PDEConfig(n_spot=200, n_time=150)
# Fixed inputs: G2++ mean reversions and correlation, Cheyette and HW kappa.
A_KEY, B_KEY, RHO_KEY = "USD.G2.A", "USD.G2.B", "USD.G2.RHO"
KAPPA_KEY, HW_A_KEY = "USD.CHEY.KAPPA", "USD.HW.MEAN_REVERSION"
G2_FIXED = {A_KEY: 0.8, B_KEY: 0.03, RHO_KEY: -0.7}


def header(s: str) -> None:
    print(f"\n{'=' * 72}\n  {s}\n{'=' * 72}")


registry = SeedRegistry(master_seed=20260214, library_version=valax.__version__)
vol_spec = RatesVolSpec(
    expiry_years=(1, 2, 5) if FAST else (1, 2, 3, 5),
    tenor_years=(2, 5) if FAST else (2, 3, 5, 7),
    cap_years=(1,),
)


def market_chain(rates_spec: RatesWorldSpec, extra_fixed: dict):
    """Noisy snapshot → calibrated curves → cube (the market levels)."""
    snap, world = rates_vol_snapshot(
        registry, default_config(), rates_spec, vol_spec,
        bp_noise=0.5, vol_bp_noise=VOL_NOISE_BP,
    )
    chain = CalibrationChain(nodes=snap.chain.nodes[:2])  # curves → cube
    quotes = snap.quotes.select(chain.quote_keys)
    fixed = snap.fixed.merge(QuoteSet.from_dict(extra_fixed))
    calibrated = calibrate_market(chain, quotes, fixed, strict=True)
    return snap, world, chain, quotes, fixed, calibrated


def bermudan(world, strike: float, *, multi_curve: bool = True) -> BermudanSwaption:
    """2Y-into-5Y Bermudan payer, annual exercise at 2Y, 3Y, 4Y."""
    ref = int(world.rates.reference_date)
    return BermudanSwaption(
        exercise_dates=jnp.array([ref + 365 * k for k in (2, 3, 4)], dtype=jnp.int32),
        fixed_dates=jnp.array([ref + 365 * k for k in range(3, 8)], dtype=jnp.int32),
        strike=jnp.asarray(strike),
        notional=jnp.asarray(1.0),
        day_count=world.rates.day_count,
        discount_id=world.discount_id,
        forward_id=world.forward_id if multi_curve else None,
    )


def g2_basket(trade) -> tuple:
    """Co-terminal swaptions + the 1Y-tenor swaption at the first exercise.

    The co-terminal set pins the overall vol level; the short-tenor option
    (one coupon of the first exercise's underlying) pins the split between the
    fast (a) and slow (b) factor, which the co-terminals alone cannot.
    """
    co = coterminal_swaptions(trade)
    short = eqx.tree_at(lambda s: s.fixed_dates, co[0], co[0].fixed_dates[:1])
    return co + (short,)


def target_keys(name: str, n: int) -> tuple[str, ...]:
    return tuple(f"{name}.TARGET.{i}" for i in range(n))


def g2_node(basket, cube_id, name="g2"):
    keys = target_keys(name, len(basket))
    node = swaption_basket_node(
        name,
        template=G2PPModel(
            mean_reversion_x=jnp.asarray(0.8), mean_reversion_y=jnp.asarray(0.03),
            volatility_x=jnp.asarray(0.006), volatility_y=jnp.asarray(0.006),
            correlation=jnp.asarray(-0.7),
        ),
        basket=basket,
        cube_id=cube_id,
        transforms={"volatility_x": positive(), "volatility_y": positive()},
        fixed_fields={"mean_reversion_x": A_KEY, "mean_reversion_y": B_KEY,
                      "correlation": RHO_KEY},
        target_keys=keys,
        method="pde",
        config=PDE_G2,
    )
    return node, keys


def tree_index(tree, i):
    return jax.tree.map(lambda x: x[i], tree)


# %% 1. Multi-curve market
header("1. Noisy quotes → curves → swaption cube (multi-curve: OIS + 3M)")

rates_spec = RatesWorldSpec(pillar_years=(1, 2, 3, 5, 7, 10))
n_basket = 4  # 3 co-terminal + 1 short-tenor
g2_fixed = {**G2_FIXED, **{k: 0.0 for k in target_keys("g2", n_basket)}}
snap, world, chain, quotes, fixed, calibrated = market_chain(rates_spec, g2_fixed)
vols = snap.parts["vols"]
cube = calibrated.market.surfaces[vol_spec.cube_id]
cube_fit = calibrated.result(vols.cube_node.name)
rmse_bp = float(jnp.sqrt(jnp.mean(cube_fit.diagnostics.rmse ** 2))) * 1e4
print(f"  {len(snap.parts['curves'].keys)} curve quotes, cube cells {cube_fit.labels}")
print(f"  cube fit RMSE {rmse_bp:.2f} normal-vol bp (quote noise {VOL_NOISE_BP} bp)")
assert rmse_bp < 1.5 * VOL_NOISE_BP

# %% 2. Per-trade G2++, batched
header("2. G2++ (σ, η) fitted per trade, a/b/ρ fixed, PDE basket (filter_vmap)")

ref = int(world.rates.reference_date)
atm = float(jnp.mean(cube.forwards))
trades = [bermudan(world, atm), bermudan(world, atm + 0.004)]
baskets = [g2_basket(t) for t in trades]
assert all(len(b) == n_basket for b in baskets)


def g2_fit(basket):
    node, _ = g2_node(basket, vol_spec.cube_id)
    return node.calibrate(QuoteSet(), calibrated.market, fixed)


stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *baskets)
batched = eqx.filter_vmap(g2_fit)(stacked)
assert bool(jnp.all(batched.diagnostics.converged))
assert batched.value.initial_curve is None  # curve-free: price binds the curve
# The basket is a traced argument, so one compiled fit serves every trade.
# The batched and single fits run the same BFGS on PDE prices whose
# floating-point reduction order differs under vmap; they stop at points
# ~1e-9..3e-8 apart (relative) along the level's flattest direction, inside
# the solver tolerance — hence 1e-7, not the 1e-10 of the closed-form
# Hull-White fit.
g2_fit_one = eqx.filter_jit(g2_fit)
for i, (t, b) in enumerate(zip(trades, baskets)):
    one = g2_fit_one(b)
    for f in ("volatility_x", "volatility_y"):
        np.testing.assert_allclose(getattr(batched.value, f)[i], getattr(one.value, f),
                                   rtol=1e-7)
    print(f"  trade {i}: K={float(t.strike):.3%}  σ={float(batched.value.volatility_x[i]):.5f}"
          f"  η={float(batched.value.volatility_y[i]):.5f}  basket RMSE "
          f"{float(batched.diagnostics.rmse[i]):.1e} (relative)")
print("  filter_vmap fits == trade-by-trade fits (rtol 1e-7)")

# %% 3. Price the book
header("3. Bermudans through price() (G2++ 2-D PDE); basket PDE vs closed form")

models = [tree_index(batched.value, i) for i in range(len(trades))]
disc = calibrated.market.curve(world.discount_id)
fwd = calibrated.market.curve(world.forward_id)
for i, (t, m, b) in enumerate(zip(trades, models, baskets)):
    pv = float(price(t, m, calibrated.market, method="pde", config=PDE_G2).price)
    euro = max(float(price(sw, m, calibrated.market, method="pde", config=PDE_G2).price)
               for sw in b[:3])
    print(f"  trade {i}: Bermudan {pv * 1e4:8.2f} bp  ≥ max co-terminal European {euro * 1e4:8.2f} bp")
    assert pv >= euro - 1e-6
worst = 0.0
for sw in baskets[0]:
    mkt = float(swaption_price(sw, disc, cube, fwd))
    pde = float(price(sw, models[0], calibrated.market, method="pde", config=PDE_G2).price)
    exact = float(g2pp_swaption_price(sw, bind_curves(models[0], disc), fwd))
    worst = max(worst, abs(pde / exact - 1.0))
    tenor = (int(sw.fixed_dates[-1]) - int(sw.expiry_date)) / 365
    print(f"    {(int(sw.expiry_date) - ref) / 365:.0f}Yx{tenor:.0f}Y: cube {mkt * 1e4:7.2f} bp"
          f"  G2++ PDE {pde * 1e4:7.2f} bp  Brigo-Mercurio {exact * 1e4:7.2f} bp")
print(f"  max |PDE / closed form - 1| = {worst:.1e} (grid error of the basket engine)")
assert worst < (1e-2 if FAST else 2e-3)

# %% 4. Quote risk of trade 0
header("4. Quote risk of trade 0: adjoint (λ_m) vs autodiff, vega profile, FD")

trade, basket = trades[0], baskets[0]
node, keys = g2_node(basket, vol_spec.cube_id)
model_result = calibrate_model(node, quotes, calibrated.market, fixed, strict=True)
if not FAST:  # eager Hessian through the PDE basket: skipped in fast mode
    g2_cond = float(model_result.condition_number())
    print(f"  per-trade G2++ level: cond(∂R/∂θ) = {g2_cond:.1e} (well posed with 2Yx1Y)")
    assert g2_cond < 1e5
adj = quote_risk_adjoint(
    trade, None, calibrated, model_result=model_result, method="pde", config=PDE_G2
)
ad = quote_risk_autodiff(
    trade, None, chain, quotes, fixed, model_node=node, method="pde", config=PDE_G2
)
rec = reconcile_quote_risk(adj, ad, rtol=1e-6, atol=1e-10)
print(f"  adjoint vs autodiff: max |Δ| = {rec.max_abs_diff:.1e}  ok={rec.ok}")
assert rec.ok, rec
print("  level stationarity: "
      + ", ".join(f"{n}={float(v):.0e}" for n, v in adj.level_residuals.items()))

q = adj.quotes
curve_keys = snap.parts["curves"].keys
print(f"  rho: Σ ∂V/∂(curve quotes) per bp {sum(float(q[k]) for k in curve_keys) * 1e4 * 1e-4:+.4f} bp")
print(f"  fixed inputs: ∂V/∂a {float(adj.fixed[A_KEY]):+.2e}  ∂V/∂b {float(adj.fixed[B_KEY]):+.2e}"
      f"  ∂V/∂ρ {float(adj.fixed[RHO_KEY]):+.2e}")

print("  λ_m vega profile across the basket (∂V per 1 normal-vol bp of each basket option):")
profile = []
for k, sw in zip(keys, basket):
    def basket_price(v, _sw=sw):
        return swaption_price(_sw, disc, ConstantVol(vol=v, is_normal=True), fwd)

    s_k, _ = swaption_forward_and_annuity(sw, disc, fwd)
    expiry = (int(sw.expiry_date) - ref) / 365.0
    tenor = (int(sw.fixed_dates[-1]) - int(sw.expiry_date)) / 365.0
    vega_k = jax.grad(basket_price)(cube(sw.strike, expiry, tenor, forward=s_k))
    profile.append(float(adj.fixed[k]) * float(vega_k) * 1e-4)
    print(f"    {expiry:.0f}Yx{tenor:.0f}Y:  ∂V/∂T = {float(adj.fixed[k]):+.4f}"
          f"   → {profile[-1] * 1e4:+.4f} bp per vol bp")
cube_total = sum(float(q[k]) for row in vols.swaption_keys for ks in row for k in ks) * 1e-4
print(f"  total vega: cube quotes {cube_total * 1e4:+.4f} bp  vs  Σ basket profile "
      f"{sum(profile) * 1e4:+.4f} bp per vol bp")
np.testing.assert_allclose(cube_total, sum(profile), rtol=2e-2)
assert cube_total > 0.0  # long a payer Bermudan: long vol

@eqx.filter_jit
def pv_fn(qq, pp):
    """Full recalibration: curves → cube → per-trade G2++ → Bermudan PDE."""
    market = calibrate_market(chain, qq, pp).market
    return price(trade, node.calibrate(qq, market, pp).value, market,
                 method="pde", config=PDE_G2).price



vol_key = vols.swaption_keys[vol_spec.expiry_years.index(2)][vol_spec.tenor_years.index(5)][
    vol_spec.strike_offsets_bp.index(0.0)]  # 2Yx5Y ATM: the cell the 2Yx5Y basket option reads
# The discrete 2-D PDE price is piecewise smooth in its inputs (max(swap, 0)
# sampled at nodes: a node crossing the exercise boundary is a kink), and each
# recalibration stops at BFGS stationarity (~1e-11 noise in price). The step
# sits between the two floors: ≥ 1e-6 can straddle a grid kink, ≤ 1e-7 starts
# to see the solver noise. Adjoint and autodiff are the exact derivative of the
# discrete map, so inside a smooth piece they agree with FD to ~1e-6.
for kind, key, h in (("p", keys[1], 3e-7), ("q", vol_key, 3e-7)):
    if kind == "p":
        up = pv_fn(quotes, fixed.with_values({key: fixed[key] + h}))
        dn = pv_fn(quotes, fixed.with_values({key: fixed[key] - h}))
        got = float(adj.fixed[key])
    else:
        up = pv_fn(quotes.with_values({key: quotes[key] + h}), fixed)
        dn = pv_fn(quotes.with_values({key: quotes[key] - h}), fixed)
        got = float(adj.quotes[key])
    fd = float((up - dn) / (2 * h))
    print(f"  full-recalibration FD ∂V/∂{key}: {fd:+.6f}  vs adjoint {got:+.6f}")
    assert abs(fd) > 1e-3, key  # a live sensitivity, not 0 == 0
    np.testing.assert_allclose(got, fd, rtol=5e-4)

# %% 5. Cheyette on a single-curve market
header("5. Cheyette (σ, β) per trade on a single-curve market (frozen-y PDE)")

# Cheyette's PDE recipes take no projection curve: on the multi-curve market
# price() refuses rather than silently dropping the 3M basis.
probe = CheyetteModel(mean_reversion=jnp.asarray(0.03), volatility=jnp.asarray(0.008),
                      skew=jnp.asarray(0.0))
try:
    price(trade, probe, calibrated.market, method="pde", config=PDE_CH)
    raise AssertionError("Cheyette priced a multi-curve trade")
except ValueError as exc:
    print(f"  multi-curve trade refused: {str(exc).splitlines()[0][:76]}...")

n_ch = 4  # co-terminal at K + the first one at K + 100bp (the strike slice identifies β)
sc_fixed = {
    KAPPA_KEY: 0.03, HW_A_KEY: 0.03, **G2_FIXED,
    **{k: 0.0 for k in target_keys("ch", n_ch)},
    **{k: 0.0 for k in target_keys("g2", n_basket)},
    **{k: 0.0 for k in target_keys("hw", 3)},
}
sc_snap, sc_world, sc_chain, sc_quotes, sc_fixed_q, sc_cal = market_chain(
    RatesWorldSpec(pillar_years=(1, 2, 3, 5, 7, 10), projection_tenors=()), sc_fixed,
)
assert sc_world.forward_id == sc_world.discount_id  # OIS discounts and projects
sc_cube = sc_cal.market.surfaces[vol_spec.cube_id]
sc_trade = bermudan(sc_world, float(jnp.mean(sc_cube.forwards)), multi_curve=False)
ch_basket = coterminal_swaptions(sc_trade) + coterminal_swaptions(
    sc_trade, strike=sc_trade.strike + 0.01)[:1]
assert len(ch_basket) == n_ch
ch_keys = target_keys("ch", n_ch)
ch_node = swaption_basket_node(
    "ch",
    template=CheyetteModel(mean_reversion=jnp.asarray(0.03), volatility=jnp.asarray(0.008),
                           skew=jnp.asarray(0.0)),
    basket=ch_basket,
    cube_id=vol_spec.cube_id,
    transforms={"volatility": positive(), "skew": free()},
    fixed_fields={"mean_reversion": KAPPA_KEY},
    target_keys=ch_keys,
    method="pde",
    config=PDE_CH,
    settings=CHEYETTE_SETTINGS,
)
ch_res = calibrate_model(ch_node, sc_quotes, sc_cal.market, sc_fixed_q, strict=True)
ch = ch_res.value
assert ch.initial_curve is None
prov = cheyette_provenance("frozen", PDE_CH)
print(f"  σ={float(ch.volatility):.5f}  β={float(ch.skew):+.3f}  basket RMSE "
      f"{float(ch_res.diagnostics.rmse):.1e}"
      + ("" if FAST else f"  cond {float(ch_res.condition_number()):.1e}"))
print(f"  engine provenance: {prov.method} / {prov.engine_kind} (frozen-y parameters are"
      " engine-bound)")
ch_pv = float(price(sc_trade, ch, sc_cal.market, method="pde", config=PDE_CH).price)
ch_euro = max(float(price(sw, ch, sc_cal.market, method="pde", config=PDE_CH).price)
              for sw in ch_basket[:3])
print(f"  Bermudan {ch_pv * 1e4:.2f} bp ≥ max co-terminal European {ch_euro * 1e4:.2f} bp")
assert ch_pv >= ch_euro - 1e-6

ch_adj = quote_risk_adjoint(sc_trade, None, sc_cal, model_result=ch_res,
                            method="pde", config=PDE_CH)
ch_ad = quote_risk_autodiff(sc_trade, None, sc_chain, sc_quotes, sc_fixed_q,
                            model_node=ch_node, method="pde", config=PDE_CH)
ch_rec = reconcile_quote_risk(ch_adj, ch_ad, rtol=1e-6, atol=1e-10)
print(f"  adjoint vs autodiff: max |Δ| = {ch_rec.max_abs_diff:.1e}  ok={ch_rec.ok}"
      f"  (stationarity {float(ch_adj.level_residuals['ch']):.0e})")
assert ch_rec.ok, ch_rec
ch_pv_fn = eqx.filter_jit(lambda pp: price(
    sc_trade, ch_node.calibrate(sc_quotes, sc_cal.market, pp).value,
    sc_cal.market, method="pde", config=PDE_CH,
).price)
k1, h = ch_keys[1], 1e-6
fd = float((ch_pv_fn(sc_fixed_q.with_values({k1: h}))
            - ch_pv_fn(sc_fixed_q.with_values({k1: -h}))) / (2 * h))
print(f"  FD ∂V/∂T_1 {fd:+.6f} vs adjoint {float(ch_adj.fixed[k1]):+.6f};"
      f"  ∂V/∂κ {float(ch_adj.fixed[KAPPA_KEY]):+.2e}")
np.testing.assert_allclose(float(ch_adj.fixed[k1]), fd, rtol=5e-4)

# %% 6. Model risk on one trade
header("6. Model risk: HW vs G2++ vs Cheyette, per trade, same market and trade")

hw_basket = coterminal_swaptions(sc_trade)
hw_node = swaption_basket_node(
    "hw",
    template=HullWhiteModel(mean_reversion=jnp.asarray(0.03), volatility=jnp.asarray(0.008)),
    basket=hw_basket, cube_id=vol_spec.cube_id,
    transforms={"volatility": positive()}, fixed_fields={"mean_reversion": HW_A_KEY},
    target_keys=target_keys("hw", 3),
)
sc_g2_node, _ = g2_node(g2_basket(sc_trade), vol_spec.cube_id)
fits = {
    "Hull-White": (calibrate_model(hw_node, sc_quotes, sc_cal.market, sc_fixed_q, strict=True),
                   PDE_HW, "co-terminal (3)"),
    "G2++": (calibrate_model(sc_g2_node, sc_quotes, sc_cal.market, sc_fixed_q, strict=True),
             PDE_G2, "co-terminal + 2Yx1Y (4)"),
    "Cheyette": (ch_res, PDE_CH, "co-terminal + 2Yx5Y @ K+100bp (4)"),
}
sc_disc = sc_cal.market.curve(sc_world.discount_id)
pvs, misses = {}, {}
for name, (res, cfg, label) in fits.items():
    m = res.value
    pvs[name] = float(price(sc_trade, m, sc_cal.market, method="pde", config=cfg).price)
    misses[name] = max(
        abs(float(price(sw, m, sc_cal.market, method="pde", config=cfg).price)
            / float(swaption_price(sw, sc_disc, sc_cube, None)) - 1.0)
        for sw in hw_basket
    )
    print(f"  {name:10s} Bermudan {pvs[name] * 1e4:8.2f} bp   basket {label:34s}"
          f" max co-terminal miss {misses[name]:.1%}")
spread = (max(pvs.values()) - min(pvs.values())) / pvs["Hull-White"]
print(f"  model spread {spread:.2%} of the HW price. Every model reprices the co-terminal")
print("  Europeans to within a few percent (fit residual, not exact), so the spread mixes")
print("  fit quality with the exercise-value model risk the Europeans do not pin down.")
assert all(v < 0.05 for v in misses.values()), misses
assert spread < 0.10
print("\n  All checks passed.")
