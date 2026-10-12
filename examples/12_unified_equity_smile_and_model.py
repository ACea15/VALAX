"""Example 12 — Equity: noisy quotes → curve, smile, Heston → exotic risk.

  1. Sample a true equity market (OIS curve, spot, dividend, SABR smile term
     structure) and quote it with Gaussian noise: 0.5 bp on rates, 15 vol bp
     on the smile.
  2. Calibrate the market chain (curve → spot/dividend → SABR smiles on the
     curve-implied forwards). With noisy quotes the fit cannot recover the
     truth exactly; its residual sits at the noise level, which is checked.
  3. Calibrate Heston to the smile quotes, once for the underlying, as a
     separate input to pricing. Mean reversion is a held-fixed input.
  4. Price a small book — European, up-and-out barrier, Asian — under the
     smile (Black-Scholes at the strike's vol) and under Heston (analytic for
     the European, Monte Carlo through ``price`` for the exotics).
  5. Quote risk of the Heston-priced barrier: spot delta, curve rho, vega to
     every smile quote, and the fixed-input sensitivities (κ, SABR β), by the
     adjoint sweep and independently by autodiff through every calibration
     and the Monte Carlo pricer.

Run::

    python examples/12_unified_equity_smile_and_model.py
"""

# %% Imports and setup
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

import valax
from valax.calibration import calibrate_market, calibrate_model, price_fit_node
from valax.calibration.transforms import HESTON_TRANSFORMS
from valax.core.pricing import price
from valax.greeks import quote_risk_adjoint, quote_risk_autodiff, reconcile_quote_risk
from valax.instruments.options import EquityBarrierOption, EuropeanOption
from valax.market import (
    EquityWorldSpec,
    QuoteSet,
    SeedRegistry,
    default_config,
    equity_snapshot,
    sample_equity_trades,
)
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.pricing.mc.engine import MCConfig

FAST = os.environ.get("VALAX_EXAMPLES_FAST") == "1"
VOL_NOISE_BP = 15.0


def header(s: str) -> None:
    print(f"\n{'=' * 72}\n  {s}\n{'=' * 72}")


registry = SeedRegistry(master_seed=20260101, library_version=valax.__version__)
cfg = default_config()
spec = EquityWorldSpec(expiries=(0.5, 1.0) if FAST else (0.25, 0.5, 1.0, 2.0))
mc = MCConfig(n_paths=2_000 if FAST else 20_000, n_steps=50)
key = jax.random.PRNGKey(7)

# %% 1. Noisy synthetic quotes
header("1. Synthetic equity quotes (0.5 bp curve noise, 15 vol-bp smile noise)")

snap, world = equity_snapshot(registry, cfg, spec, bp_noise=0.5, vol_bp_noise=VOL_NOISE_BP)
smile_q = snap.parts["smile"]
print(f"  spot {float(world.spot):.2f}, dividend {float(world.dividend):.3%}, "
      f"{sum(len(r) for r in smile_q.vol_keys)} smile quotes")
for row in smile_q.vol_keys:
    print("   " + "  ".join(f"{k.split('.', 2)[2]:>7}:{float(snap.quotes[k]):6.2%}" for k in row))

# %% 2. Market chain
header("2. Calibrate curve → spot/dividend → SABR smiles")

calibrated = calibrate_market(snap.chain, snap.quotes, snap.fixed, strict=True)
smile_fit = calibrated.result("spx_smile")
rmse_bp = float(smile_fit.diagnostics.rmse) * 1e4
print(f"  smile fit RMSE {rmse_bp:.1f} vol bp (quote noise {VOL_NOISE_BP:.0f} vol bp)")
assert rmse_bp < 1.5 * VOL_NOISE_BP
surface = calibrated.market.surfaces["_default_"]
for i, t in enumerate(spec.expiries):
    print(f"  T={t:<5} alpha {float(surface.alphas[i]):.4f} (true {float(world.smile.alphas[i]):.4f})"
          f"  rho {float(surface.rhos[i]):+.3f} (true {float(world.smile.rhos[i]):+.3f})"
          f"  nu {float(surface.nus[i]):.3f} (true {float(world.smile.nus[i]):.3f})")

# %% 3. Heston per underlying
header("3. Heston calibrated to the smile quotes (κ held fixed)")

# Basket: the longest expiry (FAST) or the two longest.
n_basket = 1 if FAST else 2
basket_rows = range(len(spec.expiries) - n_basket, len(spec.expiries))
basket = [
    EuropeanOption(strike=k, expiry=jnp.asarray(spec.expiries[i]))
    for i in basket_rows for k in world.strikes[i]
]
basket_keys = [k for i in basket_rows for k in smile_q.vol_keys[i]]
heston_node = price_fit_node(
    "spx_heston",
    template=HestonModel(
        v0=jnp.asarray(0.04), kappa=jnp.asarray(1.5), theta=jnp.asarray(0.04),
        xi=jnp.asarray(0.5), rho=jnp.asarray(-0.6),
    ),
    transforms={k: HESTON_TRANSFORMS[k] for k in ("v0", "theta", "xi", "rho")},
    basket=basket,
    quote_keys=basket_keys,
    quote_form="lognormal_vol",
    fixed_fields={"kappa": "SPX.HESTON.KAPPA"},
)
fixed = snap.fixed.merge(QuoteSet.from_dict({"SPX.HESTON.KAPPA": 1.5}))
heston = calibrate_model(heston_node, snap.quotes, calibrated.market, fixed, strict=True)
h = heston.value
print(f"  v0 {float(h.v0):.4f}  theta {float(h.theta):.4f}  xi {float(h.xi):.3f}  "
      f"rho {float(h.rho):+.3f}  (κ = {float(h.kappa)} fixed)")
print(f"  basket fit RMSE {float(heston.diagnostics.rmse):.3e} (price units), "
      f"{int(heston.diagnostics.n_steps)} LM steps")

# %% 4. Book under both models
header("4. Price a small book under the smile and under Heston")

trades = sample_equity_trades(
    registry, world, n_per_kind=1 if FAST else 2, expiry_range=(0.5, 1.0)
)
print(f"  {'trade':<24}{'smile (BS@K)':>14}{'Heston':>12}")
for t in trades:
    name = f"{type(t).__name__[:-6]} K={float(t.strike):.1f} T={float(t.expiry):.2f}"
    method = "analytic" if isinstance(t, EuropeanOption) else "mc"
    cfg_kw = {} if method == "analytic" else {"config": mc, "key": key}
    pv_bs = price(t, BlackScholesModel(), calibrated.market, method=method, **cfg_kw).price
    pv_h = price(t, h, calibrated.market, method=method, **cfg_kw).price
    print(f"  {name:<24}{float(pv_bs):14.4f}{float(pv_h):12.4f}")

# %% 5. Quote risk of the Heston-priced barrier
header("5. Quote risk: up-and-out barrier under Heston (Monte Carlo)")

barrier = next(t for t in trades if isinstance(t, EquityBarrierOption))
adj = quote_risk_adjoint(
    barrier, None, calibrated, model_result=heston, method="mc", config=mc, key=key
)
ad = quote_risk_autodiff(
    barrier, None, snap.chain, snap.quotes, fixed, model_node=heston_node,
    method="mc", config=mc, key=key,
)
rec = reconcile_quote_risk(adj, ad, rtol=1e-5, atol=1e-8)
print(f"  adjoint vs autodiff (same paths): max |Δ| = {rec.max_abs_diff:.1e}  ok={rec.ok}")
assert rec.ok, rec

q = adj.quotes
print(f"  delta  ∂V/∂spot          {float(q[smile_q.spot_key]):10.4f}")
print(f"  rho    Σ ∂V/∂OIS (per bp) {sum(float(q[k]) for k in snap.parts['curves'].keys) * 1e-4:10.6f}")
print("  vega to smile quotes (per vol point), by expiry:")
for t, row in zip(spec.expiries, smile_q.vol_keys):
    print(f"    T={t:<5} " + "  ".join(f"{float(q[k]) * 1e-2:+.4f}" for k in row))
print(f"  fixed inputs: ∂V/∂κ {float(adj.fixed['SPX.HESTON.KAPPA']):+.5f}   "
      f"∂V/∂β_SABR {float(adj.fixed['SPX.SABR.BETA']):+.5f}")
print(f"  level stationarity: "
      + ", ".join(f"{n}={float(v):.0e}" for n, v in adj.level_residuals.items()))
print("\n  Only the basket expiries carry vega: under Heston the barrier is hedged")
print("  with the options the model was calibrated to. The SABR β sensitivity is")
print("  exactly zero: the Heston targets are the vol *quotes*, not the fitted smile,")
print("  so nothing this trade depends on reads the SABR surface.")
assert all(
    abs(float(q[k])) < 1e-10
    for i, row in enumerate(smile_q.vol_keys) if i not in basket_rows for k in row
)
assert abs(float(adj.fixed["SPX.SABR.BETA"])) < 1e-12
