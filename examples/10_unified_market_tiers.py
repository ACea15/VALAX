"""Example 10 — One option, three market tiers, one workflow.

The unified framework treats every market as a calibration chain from
quotes, so the same five calls price and risk an option whether the market
is a flat Black-Scholes world or a bootstrapped curve with a calibrated smile
and a calibrated model:

    quotes ─▶ calibrate_market ─▶ (calibrate_model) ─▶ price ─▶ quote risk

  Tier 0  flat quotes (spot, rate, vol, dividend) — quote risk must equal the
          textbook Black-Scholes delta, rho, vega and dividend sensitivity.
  Tier 1  synthetic OIS strip + spot + SABR smiles on the curve forwards.
  Tier 2  Tier 1 plus a Heston model calibrated to the smile quotes.

At every tier the explicit adjoint sweep and autodiff through the solves are
computed independently and reconciled.

Run::

    python examples/10_unified_market_tiers.py
"""

# %% Imports and setup
from __future__ import annotations

import os

import jax
import jax.numpy as jnp

import valax
from valax.calibration import (
    CalibrationChain,
    calibrate_market,
    calibrate_model,
    market_inputs_node,
    price_fit_node,
)
from valax.calibration.transforms import HESTON_TRANSFORMS
from valax.core.pricing import price
from valax.greeks import quote_risk_adjoint, quote_risk_autodiff, reconcile_quote_risk
from valax.instruments.options import EuropeanOption
from valax.market import (
    EquityWorldSpec,
    QuoteSet,
    SeedRegistry,
    default_config,
    equity_snapshot,
)
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.pricing.analytic.black_scholes import black_scholes_price

FAST = os.environ.get("VALAX_EXAMPLES_FAST") == "1"


def header(s: str) -> None:
    print(f"\n{'=' * 72}\n  {s}\n{'=' * 72}")


def summarise(label, pv, risk, vol_prefix, curve_prefix, spot_key):
    vega = sum(float(v) for k, v in risk.quotes.values.items() if k.startswith(vol_prefix))
    rho = sum(float(v) for k, v in risk.quotes.values.items() if k.startswith(curve_prefix))
    print(
        f"{label:<28} PV {float(pv):9.4f}   delta {float(risk.quotes[spot_key]):8.4f}"
        f"   Σvega {vega:9.4f}   Σrho {rho:9.4f}"
    )


registry = SeedRegistry(master_seed=20260101, library_version=valax.__version__)
cfg = default_config()

# %% Tier 0 — the constant-parameter market as a one-node chain
header("Tier 0: flat Black-Scholes quotes")

flat_chain = CalibrationChain(nodes=(
    market_inputs_node(
        spots={"_default_": "SPX.SPOT"},
        dividends={"_default_": "SPX.DIV"},
        flat_rates={"_default_": "USD.RATE"},
        flat_vols={"_default_": "SPX.VOL"},
    ),
))
flat_quotes = QuoteSet.from_dict(
    {"SPX.SPOT": 100.0, "SPX.DIV": 0.01, "USD.RATE": 0.04, "SPX.VOL": 0.22}
)
option = EuropeanOption(strike=jnp.asarray(105.0), expiry=jnp.asarray(1.0))
model = BlackScholesModel()  # every field bound from the market

flat = calibrate_market(flat_chain, flat_quotes)
pv0 = price(option, model, flat.market).price
risk0 = quote_risk_adjoint(option, model, flat)
risk0_ad = quote_risk_autodiff(option, model, flat_chain, flat_quotes)

textbook = dict(zip(
    ("SPX.SPOT", "SPX.VOL", "USD.RATE", "SPX.DIV"),
    jax.grad(black_scholes_price, argnums=(1, 2, 3, 4))(option, 100.0, 0.22, 0.04, 0.01),
))
for key, value in textbook.items():
    print(f"  ∂V/∂{key:<9} chain {float(risk0.quotes[key]):10.6f}   textbook {float(value):10.6f}")
    assert abs(float(risk0.quotes[key]) - float(value)) < 1e-10
assert reconcile_quote_risk(risk0, risk0_ad, rtol=1e-10).ok
print("  ✓ quote risk of the degenerate chain is exactly the textbook Greeks")

# %% Tier 1 — synthetic quotes: OIS strip, spot, SABR smiles
header("Tier 1: OIS strip + spot + SABR smiles (synthetic, noise-free)")

spec = EquityWorldSpec(expiries=(0.5, 1.0) if FAST else (0.25, 0.5, 1.0, 2.0))
snap, world = equity_snapshot(registry, cfg, spec)
print(f"  {len(snap.quotes)} quotes, chain levels: {[n.name for n in snap.chain.nodes]}")

calibrated = calibrate_market(snap.chain, snap.quotes, snap.fixed, strict=True)
for node, res in zip(snap.chain.nodes, calibrated.results):
    print(
        f"  level {node.name:<12} converged={bool(res.diagnostics.converged)}"
        f"  max|r|={float(res.diagnostics.max_abs_residual):.1e}"
    )
fitted = calibrated.market.surfaces["_default_"]
err = float(jnp.max(jnp.abs(fitted.alphas - world.smile.alphas)))
print(f"  max |alpha_fit - alpha_true| = {err:.1e}  (noise-free quotes → exact recovery)")
assert err < 1e-8

option1 = EuropeanOption(strike=world.spot * 1.05, expiry=jnp.asarray(spec.expiries[-1]))
pv1 = price(option1, model, calibrated.market).price
risk1 = quote_risk_adjoint(option1, model, calibrated)
risk1_ad = quote_risk_autodiff(option1, model, snap.chain, snap.quotes, snap.fixed)
rec1 = reconcile_quote_risk(risk1, risk1_ad, rtol=1e-6, atol=1e-9)
print(f"  adjoint vs autodiff: max |Δ| = {rec1.max_abs_diff:.1e}  ok={rec1.ok}")
assert rec1.ok

# %% Tier 2 — a Heston model calibrated on the market, as a separate input
header("Tier 2: Heston calibrated to the smile quotes (per underlying)")

last_expiry = spec.expiries[-1]
smile_keys = snap.parts["smile"].vol_keys[-1]
basket = [EuropeanOption(strike=k, expiry=jnp.asarray(last_expiry)) for k in world.strikes[-1]]
heston_node = price_fit_node(
    "spx_heston",
    template=HestonModel(
        v0=jnp.asarray(0.04), kappa=jnp.asarray(1.5), theta=jnp.asarray(0.04),
        xi=jnp.asarray(0.5), rho=jnp.asarray(-0.5),
    ),
    transforms={k: HESTON_TRANSFORMS[k] for k in ("v0", "theta", "xi", "rho")},
    basket=basket,
    quote_keys=smile_keys,
    quote_form="lognormal_vol",
    fixed_fields={"kappa": "SPX.HESTON.KAPPA"},  # held fixed: reported as p̄
)
fixed2 = snap.fixed.merge(QuoteSet.from_dict({"SPX.HESTON.KAPPA": 1.5}))
heston = calibrate_model(heston_node, snap.quotes, calibrated.market, fixed2, strict=True)
print(f"  fit RMSE (price units) {float(heston.diagnostics.rmse):.2e} — a 4-parameter model"
      f" does not reproduce a SABR smile exactly")

pv2 = price(option1, heston.value, calibrated.market).price
risk2 = quote_risk_adjoint(option1, None, calibrated, model_result=heston)
risk2_ad = quote_risk_autodiff(
    option1, None, snap.chain, snap.quotes, fixed2, model_node=heston_node
)
rec2 = reconcile_quote_risk(risk2, risk2_ad, rtol=1e-5, atol=1e-8)
print(f"  adjoint vs autodiff: max |Δ| = {rec2.max_abs_diff:.1e}  ok={rec2.ok}")
assert rec2.ok
print(f"  ∂V/∂κ (fixed input) = {float(risk2.fixed['SPX.HESTON.KAPPA']):.4f}")

# %% Side by side
header("Same option, three markets")
summarise("Tier 0 flat", pv0, risk0, "SPX.VOL", "USD.RATE", "SPX.SPOT")
summarise("Tier 1 curve + SABR (BS)", pv1, risk1, "SPX.VOL", "USD.OIS", "SPX.SPOT")
summarise("Tier 2 curve + SABR → Heston", pv2, risk2, "SPX.VOL", "USD.OIS", "SPX.SPOT")
print("\n  In Tier 1 the vega sits on the quotes near the strike; in Tier 2 it is")
print("  spread over the basket the model was calibrated to — same API, different")
print("  hedge, decided only by how the market and model were calibrated.")
