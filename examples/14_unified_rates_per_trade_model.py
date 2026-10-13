"""Example 14 — Rates: quotes → curves → swaption cube → per-trade Hull-White → Bermudan risk.

  1. Sample a true rates market (OIS + 3M projection curves, a SABR swaption
     cube in normal vols) and quote it with Gaussian noise: 0.5 bp on rates,
     0.5 normal-vol bp on the cube.
  2. Calibrate the market chain: the curves, then the cube on the
     curve-implied forwards (every cell its own SABR fit, named on failure).
  3. Calibrate Hull-White **per trade**: each Bermudan gets σ fitted to its own
     co-terminal European basket, priced off the calibrated cube, with mean
     reversion held fixed. Same-shaped baskets are calibrated in one
     ``eqx.filter_vmap``; the fitted models are curve-free (``price`` binds the
     market curve), so they stack.
  4. Price the Bermudans through ``price`` (Hull-White PDE) and check the model
     reprices its basket (Jamshidian through ``price`` vs. the cube).
  5. Quote risk of one Bermudan by the adjoint sweep — the model level's
     multiplier λ_m pushed through the cube and curve levels — and
     independently by autodiff through every calibration and the PDE. The
     basket target shifts report λ_m as the **vega profile across the
     basket**, checked against full recalibration.

Run::

    python examples/14_unified_rates_per_trade_model.py
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
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.rates_smile import swaption_price
from valax.pricing.analytic.swaptions import swaption_forward_and_annuity
from valax.pricing.pde.config import PDEConfig

jax.config.update("jax_enable_x64", True)

FAST = os.environ.get("VALAX_EXAMPLES_FAST") == "1"
VOL_NOISE_BP = 0.5
A_KEY = "USD.HW.MEAN_REVERSION"
PDE = PDEConfig(n_spot=100, n_time=60) if FAST else PDEConfig(n_spot=200, n_time=150)


def header(s: str) -> None:
    print(f"\n{'=' * 72}\n  {s}\n{'=' * 72}")


registry = SeedRegistry(master_seed=20260214, library_version=valax.__version__)
rates_spec = RatesWorldSpec(pillar_years=(1, 2, 3, 5, 7, 10))
vol_spec = RatesVolSpec(
    expiry_years=(1, 2, 5) if FAST else (1, 2, 3, 5),
    tenor_years=(2, 5) if FAST else (2, 3, 5, 7),
    cap_years=(1,),
)

# %% 1. Noisy synthetic quotes
header("1. Synthetic rates quotes (0.5 bp curves, 0.5 normal-vol bp cube)")

snap, world = rates_vol_snapshot(
    registry, default_config(), rates_spec, vol_spec,
    bp_noise=0.5, vol_bp_noise=VOL_NOISE_BP,
)
vols = snap.parts["vols"]
chain = CalibrationChain(nodes=snap.chain.nodes[:2])  # curves → cube
quotes = snap.quotes.select(chain.quote_keys)
n_vol = sum(len(k) for row in vols.swaption_keys for k in row)
print(f"  {len(snap.parts['curves'].keys)} curve quotes, {n_vol} swaption vol quotes "
      f"({len(vol_spec.expiry_years)}x{len(vol_spec.tenor_years)} cells x "
      f"{len(vol_spec.strike_offsets_bp)} strikes)")

# %% 2. Market chain
header("2. Calibrate curves → swaption cube (on the curve-implied forwards)")

fixed = snap.fixed.merge(QuoteSet.from_dict({A_KEY: 0.03}))
calibrated = calibrate_market(chain, quotes, fixed, strict=True)
cube_fit = calibrated.result(vols.cube_node.name)
rmse_bp = float(jnp.sqrt(jnp.mean(cube_fit.diagnostics.rmse ** 2))) * 1e4
print(f"  cube cells {cube_fit.labels}")
print(f"  cube fit RMSE {rmse_bp:.2f} normal-vol bp (quote noise {VOL_NOISE_BP} bp), "
      f"condition number {float(cube_fit.condition_number()):.1e}")
assert rmse_bp < 1.5 * VOL_NOISE_BP
cube = calibrated.market.surfaces[vol_spec.cube_id]
err = float(jnp.max(jnp.abs(cube.forwards - world.cube.forwards))) * 1e4
print(f"  max forward error vs truth {err:.2f} bp (from 0.5 bp curve noise)")

# %% 3. Per-trade Hull-White, batched
header("3. Hull-White σ fitted per trade to its co-terminal basket (filter_vmap)")

ref = int(world.rates.reference_date)


def bermudan(strike: float) -> BermudanSwaption:
    """2Y-into-5Y Bermudan payer, annual exercise, on the market's curves."""
    return BermudanSwaption(
        exercise_dates=jnp.array([ref + 365 * k for k in (2, 3, 4, 5)], dtype=jnp.int32),
        fixed_dates=jnp.array([ref + 365 * k for k in range(3, 8)], dtype=jnp.int32),
        strike=jnp.asarray(strike),
        notional=jnp.asarray(1.0),
        day_count=world.rates.day_count,
        discount_id=world.discount_id,
        forward_id=world.forward_id,
    )


TRANSFORMS = {"volatility": positive()}
FIXED_FIELDS = {"mean_reversion": A_KEY}


def basket_node(basket, name="hw"):
    keys = tuple(f"{name}.TARGET.{i}" for i in range(len(basket)))
    node = swaption_basket_node(
        name,
        template=HullWhiteModel(mean_reversion=jnp.asarray(0.03),
                                volatility=jnp.asarray(0.008)),
        basket=basket,
        cube_id=vol_spec.cube_id,
        transforms=TRANSFORMS,
        fixed_fields=FIXED_FIELDS,
        target_keys=keys,
    )
    return node, keys


atm = float(jnp.mean(cube.forwards))
trades = [bermudan(atm), bermudan(atm + 0.005)]
baskets = [coterminal_swaptions(t) for t in trades]
_, target_keys = basket_node(baskets[0])
fixed = fixed.merge(QuoteSet.from_dict({k: 0.0 for k in target_keys}))


def fit(basket):
    node, _ = basket_node(basket)
    return node.calibrate(QuoteSet(), calibrated.market, fixed)


stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *baskets)
batched = eqx.filter_vmap(fit)(stacked)
assert bool(jnp.all(batched.diagnostics.converged))
assert batched.value.initial_curve is None  # curve-free: price binds the curve
for i, t in enumerate(trades):
    print(f"  trade {i}: K={float(t.strike):.3%}  σ_HW={float(batched.value.volatility[i]):.5f}"
          f"  basket fit RMSE {float(batched.diagnostics.rmse[i]):.1e} (relative)")

# %% 4. Price the book
header("4. Bermudans through price() (PDE); the model reprices its basket")

models = [jax.tree.map(lambda x: x[i], batched.value) for i in range(len(trades))]
for i, (t, m, b) in enumerate(zip(trades, models, baskets)):
    pv = price(t, m, calibrated.market, method="pde", config=PDE).price
    euro = max(
        float(price(sw, m, calibrated.market).price) for sw in b  # Jamshidian
    )
    print(f"  trade {i}: Bermudan {float(pv) * 1e4:8.2f} bp  ≥ max European {euro * 1e4:8.2f} bp")
    assert float(pv) >= euro - 1e-6
disc = calibrated.market.curve(world.discount_id)
fwd = calibrated.market.curve(world.forward_id)
for sw in baskets[0]:
    mkt = float(swaption_price(sw, disc, cube, fwd))
    mdl = float(price(sw, models[0], calibrated.market).price)
    print(f"    basket expiry {(int(sw.expiry_date) - ref) / 365:.0f}Y: cube {mkt * 1e4:7.2f} bp"
          f"  HW {mdl * 1e4:7.2f} bp")

# %% 5. Quote risk of trade 0
header("5. Quote risk of trade 0: adjoint (λ_m) vs autodiff, vega profile")

trade, (node, _) = trades[0], basket_node(baskets[0])
model_result = calibrate_model(node, quotes, calibrated.market, fixed, strict=True)
adj = quote_risk_adjoint(
    trade, None, calibrated, model_result=model_result, method="pde", config=PDE
)
ad = quote_risk_autodiff(
    trade, None, chain, quotes, fixed, model_node=node, method="pde", config=PDE
)
rec = reconcile_quote_risk(adj, ad, rtol=1e-6, atol=1e-10)
print(f"  adjoint vs autodiff: max |Δ| = {rec.max_abs_diff:.1e}  ok={rec.ok}")
assert rec.ok, rec

q = adj.quotes
curve_keys = snap.parts["curves"].keys
print(f"  rho: Σ ∂V/∂(curve quotes) per bp {sum(float(q[k]) for k in curve_keys) * 1e-4 * 1e4:+.4f} bp")
print("  vega to the cube, per normal-vol bp, by cell (summed over strikes):")
for e, row in zip(vol_spec.expiry_years, vols.swaption_keys):
    cells = [sum(float(q[k]) for k in keys) * 1e-4 * 1e4 for keys in row]
    print(f"    {e}Y expiry: " + "  ".join(
        f"{t}Y:{c:+.4f}" for t, c in zip(vol_spec.tenor_years, cells)) + "  (bp)")

# λ_m as the vega profile across the basket: ∂V/∂T_k (target price), and in
# normal-vol bp via each basket instrument's own vega.
print("  λ_m vega profile across the basket (∂V per 1 normal-vol bp of the basket option):")
profile = []
for k, sw in zip(target_keys, baskets[0]):
    def basket_price(v, _sw=sw):
        from valax.surfaces.constant import ConstantVol

        return swaption_price(_sw, disc, ConstantVol(vol=v, is_normal=True), fwd)

    s_k, _ = swaption_forward_and_annuity(sw, disc, fwd)
    vol_k = cube(sw.strike, (int(sw.expiry_date) - ref) / 365.0,
                 (int(sw.fixed_dates[-1]) - int(sw.expiry_date)) / 365.0,
                 forward=s_k)
    vega_k = jax.grad(basket_price)(vol_k)
    profile.append(float(adj.fixed[k]) * float(vega_k) * 1e-4)
    print(f"    expiry {(int(sw.expiry_date) - ref) / 365:.0f}Y:  ∂V/∂T = {float(adj.fixed[k]):+.4f}"
          f"   → {profile[-1] * 1e4:+.4f} bp per vol bp")
assert all(p > 0 for p in profile), profile  # long payer Bermudan: long every co-terminal

# Two views of the same vega: shifting every cube quote by 1 bp moves each
# basket vol by ~1 bp, so the total cube vega equals the summed basket profile.
cube_total = sum(float(q[k]) for row in vols.swaption_keys for keys in row for k in keys) * 1e-4
print(f"  total vega: cube quotes {cube_total * 1e4:+.4f} bp  vs  Σ basket profile "
      f"{sum(profile) * 1e4:+.4f} bp per vol bp")
np.testing.assert_allclose(cube_total, sum(profile), rtol=2e-2)

# Check one entry of λ_m by full recalibration.
pv_fn = eqx.filter_jit(lambda p: price(
    trade, node.calibrate(quotes, calibrated.market, p).value,
    calibrated.market, method="pde", config=PDE,
).price)
k1, h = target_keys[1], 1e-6
fd = float((pv_fn(fixed.with_values({k1: h})) - pv_fn(fixed.with_values({k1: -h}))) / (2 * h))
print(f"  full-recalibration FD of ∂V/∂T_1: {fd:+.6f}  vs adjoint {float(adj.fixed[k1]):+.6f}")
np.testing.assert_allclose(float(adj.fixed[k1]), fd, rtol=1e-3)
print(f"  ∂V/∂a (held-fixed mean reversion): {float(adj.fixed[A_KEY]):+.5f}")
print(f"  level stationarity: "
      + ", ".join(f"{n}={float(v):.0e}" for n, v in adj.level_residuals.items()))
print("\n  The Bermudan's vega sits on the cube cells its co-terminal basket reads,")
print("  and its curve risk includes the basket's forwards: per-trade calibration")
print("  routes the model sensitivity λ_m through the cube and the curves to quotes.")
