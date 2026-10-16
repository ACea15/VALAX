"""Example 15 — Rates: quotes → curves → cap strip → optionlet SABR → caps → quote risk.

  1. Sample a true rates market (OIS + 3M projection curves, a cap matrix of
     flat normal vols generated from per-bucket caplet smiles) and quote it
     with Gaussian noise: 0.5 bp on rates, 0.5 normal-vol bp on the caps.
  2. Calibrate the market chain curves → cap strip → optionlet SABR: the
     strip bootstraps each strike column into per-optionlet vols on the
     curve-projected forwards; the optionlet level fits one SABR smile per
     expiry on the strip and has no quotes of its own. Report each level's
     fit and conditioning, the strip error against the truth, and the exact
     recovery from noise-free quotes.
  3. Price a book of caps and floors — on and off the quoted strike grid, at a
     quoted and an unquoted maturity — through ``price`` off the calibrated
     optionlet surface. Check cap − floor parity against the forward-rate
     strip, and that the smoothed surface reprices the quoted caps within the
     quote noise.
  4. Quote risk of one cap by the adjoint sweep (strip and optionlet levels
     pulled back onto the cap quotes and the curves) and independently by
     autodiff through every calibration; both checked against
     full-recalibration finite differences. Vega lands on the cap quotes the
     caplets read — exactly zero on the longer maturities — and the curve
     risk includes the strip's own dependence on the forwards.

Caps are priced off the optionlet surface under
:class:`~valax.models.ImpliedVolModel` — the market-quoted "model" that names
the vol source by its id — so ``price`` and the quote-risk functions built on
it see an ordinary (instrument, model, market) triple; the route is the
curve-aware :func:`~valax.pricing.analytic.rates_smile.cap_price`.

Run::

    python examples/15_unified_rates_caps_optionlet_chain.py
"""

# %% Imports and setup
from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import valax
from valax.calibration import CalibrationChain, calibrate_market
from valax.core.pricing import price
from valax.dates.daycounts import year_fraction
from valax.greeks import quote_risk_adjoint, quote_risk_autodiff, reconcile_quote_risk
from valax.instruments.rates import Cap
from valax.market import (
    RatesVolSpec,
    RatesWorldSpec,
    SeedRegistry,
    default_config,
    rates_vol_snapshot,
)
from valax.models import ImpliedVolModel
from valax.pricing.analytic.caplets import cap_forwards, cap_price_bachelier
from valax.pricing.analytic.rates_smile import cap_price
from valax.surfaces.shifted import ShiftedVolSurface

jax.config.update("jax_enable_x64", True)

FAST = os.environ.get("VALAX_EXAMPLES_FAST") == "1"
RATE_NOISE_BP = 0.5
VOL_NOISE_BP = 0.5
NOTIONAL = 1_000_000.0


def header(s: str) -> None:
    print(f"\n{'=' * 72}\n  {s}\n{'=' * 72}")


# Seed note: with only five noisy strikes per optionlet expiry, some draws push
# a bucket's SABR ρ to −1 and the strict fit fails (named per expiry); this
# seed gives an interior fit.
registry = SeedRegistry(master_seed=20260303, library_version=valax.__version__)
rates_spec = RatesWorldSpec(pillar_years=(1, 2, 3, 5, 7))
vol_spec = RatesVolSpec(
    expiry_years=(1,),            # the cube is not used here: keep it minimal
    tenor_years=(2,),
    cap_years=(1, 2, 3, 5),
    cap_strike_offsets_bp=(-100.0, -50.0, 0.0, 50.0, 100.0),
)
SMILE = ImpliedVolModel(vol_spec.optionlet_id)  # caps off the optionlet surface

# %% 1. Noisy synthetic quotes
header("1. Synthetic rates + cap quotes (0.5 bp curves, 0.5 normal-vol bp caps)")

snap, world = rates_vol_snapshot(
    registry, default_config(), rates_spec, vol_spec,
    bp_noise=RATE_NOISE_BP, vol_bp_noise=VOL_NOISE_BP,
)
vols = snap.parts["vols"]
curve_node = snap.chain.nodes[0]
chain = CalibrationChain(nodes=(curve_node, vols.strip_node, vols.optionlet_node))
quotes = snap.quotes.select(chain.quote_keys)
fixed = snap.fixed
curve_keys = snap.parts["curves"].keys
cap_flat = [k for row in vols.cap_keys for k in row]
strikes = [float(k) for k in world.cap_strikes]
print(f"  {len(curve_keys)} curve quotes, {len(cap_flat)} cap vol quotes "
      f"({len(vol_spec.cap_years)} maturities x {len(strikes)} strikes)")
print(f"  {'strike':>8}" + "".join(f"{f'cap {y}Y':>11}" for y in vol_spec.cap_years)
      + "   (flat normal vol, bp)")
for k, row in zip(strikes, vols.cap_keys):
    print(f"  {k:8.2%}" + "".join(f"{float(quotes[q]) * 1e4:11.2f}" for q in row))

# %% 2. Market chain
header("2. Calibrate curves → cap strip → optionlet SABR")

calibrated = calibrate_market(chain, quotes, fixed, strict=True)
strip_fit = calibrated.result(vols.strip_node.name)
opt_fit = calibrated.result(vols.optionlet_node.name)
for res in calibrated.results:
    print(f"  {res.name:<20} converged={bool(jnp.all(res.diagnostics.converged))}  "
          f"max|R|={float(jnp.max(res.diagnostics.max_abs_residual)):.1e}  "
          f"cond={float(res.condition_number()):.1e}")
print(f"  strip: one lower-triangular bootstrap per strike column, nodes named "
      f"(strike, bucket): {strip_fit.labels[0]!r} … {strip_fit.labels[-1]!r}")
print(f"         buckets ending at optionlets {tuple(int(e) for e in world.strip.bucket_ends)} "
      f"= caps {tuple(f'{y}Y' for y in vol_spec.cap_years)}")

strip = calibrated.market.surfaces[vol_spec.strip_id]
strip_err_bp = float(jnp.max(jnp.abs(strip.vols - world.strip.vols))) * 1e4
print(f"  strip vs truth: max |σ_fit − σ_true| = {strip_err_bp:.2f} normal-vol bp "
      f"(bootstrap amplifies the {VOL_NOISE_BP} bp quote noise in later buckets)")
assert strip_err_bp < 6 * VOL_NOISE_BP  # 1.1–2.4 bp across seeds at 0.5 bp noise

opt = calibrated.market.surfaces[vol_spec.optionlet_id]
print(f"  optionlet SABR (β={vol_spec.beta}, shifted normal), one smile per expiry:")
for i, lab in enumerate(opt_fit.labels):
    if i % 4 == 3 or i == len(opt_fit.labels) - 1:  # one line per bucket end
        print(f"    {lab:>6}: α={float(opt.alphas[i]):.5f}  ρ={float(opt.rhos[i]):+.3f}  "
              f"ν={float(opt.nus[i]):.3f}  fit RMSE {float(opt_fit.diagnostics.rmse[i]) * 1e4:.2f} bp")
opt_rmse_bp = float(jnp.max(opt_fit.diagnostics.rmse)) * 1e4
assert opt_rmse_bp < 3 * VOL_NOISE_BP

# Noise-free quotes: the curves and the strip are recovered to solver tolerance
# (the optionlet SABR is a smoothing of the strip, not a recoverable truth).
clean_chain = CalibrationChain(nodes=(curve_node, vols.strip_node))
clean = calibrate_market(clean_chain, snap.clean.select(clean_chain.quote_keys), fixed,
                         strict=True)
clean_err = float(jnp.max(jnp.abs(
    clean.market.surfaces[vol_spec.strip_id].vols - world.strip.vols)))
clean_df = max(
    float(jnp.max(jnp.abs(clean.market.curve(c).discount_factors
                          - world.rates.truth[c].discount_factors)))
    for c in world.rates.curve_ids
)
print(f"  noise-free quotes: max strip error {clean_err:.1e}, max DF error {clean_df:.1e}")
assert clean_err < 1e-10 and clean_df < 1e-10

# %% 3. Price a cap/floor book
header("3. Caps and floors through price() off the optionlet surface")

ref = int(world.rates.reference_date)
day_count = world.rates.day_count
edges = ref + (jnp.arange(1, 4 * max(vol_spec.cap_years) + 2, dtype=jnp.int32) * 365) // 4
disc = calibrated.market.curve(world.discount_id)
fwd = calibrated.market.curve(world.forward_id)


def cap(years: int, strike: float, is_cap: bool = True) -> Cap:
    """Quarterly cap/floor, first fixing in 3M (the quoted-cap convention)."""
    n = 4 * years - 1
    return Cap(
        fixing_dates=edges[:n], start_dates=edges[:n], end_dates=edges[1:n + 1],
        strike=jnp.asarray(strike), notional=jnp.asarray(NOTIONAL),
        is_cap=is_cap, day_count=day_count,
        discount_id=world.discount_id, forward_id=world.forward_id,
    )


off_grid = strikes[2] + 0.0017            # between two quoted strikes
book = [
    ("cap   2Y on-grid ", cap(2, strikes[2])),
    ("cap   3Y off-grid", cap(3, off_grid)),
    ("cap   4Y off-grid", cap(4, strikes[1] + 0.0031)),   # unquoted maturity
    ("floor 5Y on-grid ", cap(5, strikes[1], is_cap=False)),
    ("floor 3Y off-grid", cap(3, off_grid, is_cap=False)),
]
pv_jit = eqx.filter_jit(lambda inst, m: price(inst, SMILE, m).price)
pvs = {}
for name, inst in book:
    pv = float(pv_jit(inst, calibrated.market))
    direct = float(cap_price(inst, disc, opt, fwd))
    pvs[name] = pv
    print(f"  {name}  K={float(inst.strike):.3%}  PV = {pv:12,.2f}")
    np.testing.assert_allclose(pv, direct, rtol=1e-12)  # price() is the smile cap pricer
    assert pv > 0.0

# Cap − floor = the forward-rate strip Σ N τ_i P(T_i) (F_i − K): model-free.
print("  cap − floor parity vs the forward strip Σ N·τ·P(T_pay)·(F − K):")
parity_err = 0.0
for years, k in ((2, strikes[0]), (3, off_grid), (5, strikes[4])):
    c, f = cap(years, k), cap(years, k, is_cap=False)
    diff = float(pv_jit(c, calibrated.market) - pv_jit(f, calibrated.market))
    tau = year_fraction(c.start_dates, c.end_dates, c.day_count)
    strip_pv = float(jnp.sum(NOTIONAL * tau * disc(c.end_dates)
                             * (cap_forwards(c, disc, fwd) - k)))
    parity_err = max(parity_err, abs(diff - strip_pv))
    print(f"    {years}Y K={k:.3%}: cap−floor {diff:12,.4f}  forward strip {strip_pv:12,.4f}")
print(f"  max parity error {parity_err:.1e} (notional {NOTIONAL:,.0f})")
assert parity_err < 1e-6

# Repricing the quoted caps: the strip reprices them exactly by construction;
# the SABR surface smooths it, so the misfit, expressed as a flat-vol error
# (premium gap / flat vega), sits at the quote-noise level.
errs = []
for k, row in zip(strikes, vols.cap_keys):
    for y, q in zip(vol_spec.cap_years, row):
        c = cap(y, k)
        flat = quotes[q]
        quoted_pv = cap_price_bachelier(c, disc, flat, fwd)
        flat_vega = jax.grad(lambda v: cap_price_bachelier(c, disc, v, fwd))(flat)
        errs.append(float((pv_jit(c, calibrated.market) - quoted_pv) / flat_vega) * 1e4)
errs = np.asarray(errs)
print(f"  quoted caps repriced off the SABR surface: RMS {np.sqrt(np.mean(errs ** 2)):.2f}, "
      f"max {np.max(np.abs(errs)):.2f} flat-vol bp (quote noise {VOL_NOISE_BP} bp)")
assert np.sqrt(np.mean(errs ** 2)) < 2 * VOL_NOISE_BP
assert np.max(np.abs(errs)) < 4 * VOL_NOISE_BP

# %% 4. Quote risk of the 3Y off-grid cap
header("4. Quote risk of the 3Y off-grid cap: adjoint vs autodiff vs FD")

trade = book[1][1]
adj = quote_risk_adjoint(trade, SMILE, calibrated)
ad = quote_risk_autodiff(trade, SMILE, chain, quotes, fixed)
# The optionlet level is a least-squares fit with non-zero residuals (noisy
# strip): the two paths agree to ~1e-8 of the largest sensitivity, while a few
# entries ~1e-3 times smaller differ at ~4e-6 relative — so the absolute
# tolerance is scaled to the risk, as in tests/test_calibration.
scale = max(abs(float(v)) for v in ad.quotes.values.values())
rec = reconcile_quote_risk(adj, ad, rtol=1e-6, atol=1e-8 * scale)
print(f"  adjoint vs autodiff: max |Δ| = {rec.max_abs_diff:.1e} "
      f"(largest ∂PV/∂q = {scale:.2e}, atol 1e-8 x that)  ok={rec.ok}")
assert rec.ok, rec
rec_curves = reconcile_quote_risk(
    eqx.tree_at(lambda r: r.quotes, adj, adj.quotes.select(curve_keys)),
    eqx.tree_at(lambda r: r.quotes, ad, ad.quotes.select(curve_keys)),
    rtol=1e-6, atol=1e-10,
)
print(f"  curve quotes alone: max |Δ| = {rec_curves.max_abs_diff:.1e}  ok={rec_curves.ok} "
      "(rtol 1e-6, atol 1e-10)")
assert rec_curves.ok, rec_curves
q = adj.quotes

print("  vega by cap quote (∂PV per +1 normal-vol bp):")
print(f"  {'strike':>8}" + "".join(f"{f'cap {y}Y':>11}" for y in vol_spec.cap_years))
for k, row in zip(strikes, vols.cap_keys):
    print(f"  {k:8.2%}" + "".join(f"{float(q[key]) * 1e-4:11.2f}" for key in row))
vega_total = sum(float(q[k]) for k in cap_flat) * 1e-4
beyond = max(abs(float(q[row[-1]])) for row in vols.cap_keys) * 1e-4
print(f"  total vega {vega_total:+,.2f} per vol bp; on the 5Y quotes {beyond:.1e} "
      "(the 3Y cap's caplets fix before the 5Y bucket starts)")
assert beyond < 1e-10 * abs(vega_total)  # structurally zero, up to round-off
# A 1 bp shift of every cap quote moves the whole surface by ~1 bp: compare
# with the trade's parallel-surface vega.
surf_vega = float(jax.grad(
    lambda e: cap_price(trade, disc, ShiftedVolSurface(base=opt, shift=e), fwd)
)(jnp.asarray(0.0))) * 1e-4
print(f"  parallel-surface vega {surf_vega:+,.2f} per vol bp (vs Σ quote vegas {vega_total:+,.2f})")
np.testing.assert_allclose(vega_total, surf_vega, rtol=1e-2)  # ~1 bp, not exactly

rho_total = sum(float(q[k]) for k in curve_keys) * 1e-4
ois = sum(float(q[k]) for k in curve_keys if ".OIS." in k) * 1e-4
proj = rho_total - ois
print(f"  rho: Σ ∂PV per +1 bp on curve quotes {rho_total:+,.2f} "
      f"(OIS {ois:+,.2f}, 3M {proj:+,.2f})")

# Sticky-surface rho: the same curve bump with the calibrated surface frozen
# (curves re-bootstrapped onto the calibrated market). The gap is the strip's
# and the smile's own dependence on the forwards — risk the chain carries.
frozen = quote_risk_autodiff(
    trade, SMILE, CalibrationChain(nodes=(curve_node,)),
    quotes.select(curve_node.quote_keys), base=calibrated.market,
)
rho_frozen = sum(float(frozen.quotes[k]) for k in curve_keys) * 1e-4
print(f"  rho with the vol surface frozen {rho_frozen:+,.2f}  → strip/smile forward "
      f"dependence contributes {rho_total - rho_frozen:+,.2f} per bp")
assert abs(rho_total - rho_frozen) > 1e-3 * abs(rho_total)

# Full-recalibration central differences.
pv_of_quotes = eqx.filter_jit(
    lambda qs: price(trade, SMILE, calibrate_market(chain, qs, fixed).market).price
)
checks = [(vols.cap_keys[2][2], 1e-6), ("USD.IRS3M.3Y", 1e-5)]
if not FAST:
    checks += [(vols.cap_keys[3][1], 1e-6), ("USD.OIS.2Y", 1e-5)]
for key, h in checks:
    fd = float((pv_of_quotes(quotes.with_values({key: quotes[key] + h}))
                - pv_of_quotes(quotes.with_values({key: quotes[key] - h}))) / (2 * h))
    print(f"  {key:<18} adjoint {float(q[key]):+14.4f}  full-recalibration FD {fd:+14.4f}")
    assert abs(fd) > 0.0
    np.testing.assert_allclose(float(q[key]), fd, rtol=2e-4)

print("\n  The cap's vega sits on the cap quotes its caplets read — the 1Y–3Y")
print("  maturities, spread across the strikes the optionlet smiles are fitted to —")
print("  and none on the 5Y quotes. Its curve risk includes the strip: moving a")
print("  curve quote moves the forwards the cap matrix was stripped and smiled on.")
