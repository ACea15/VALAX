"""Example 16 — Diagnosing a failed calibration: named nodes, conditioning, provenance.

Every VALAX calibrator returns a ``CalibrationResult`` whose ``converged`` flag
is traceable and whose eager helpers — ``check()``, ``failed_labels()``,
``node_max_abs_residuals()``, ``condition_number()`` — report a failure *by
name*. This example breaks a rates market on purpose, one way at a time, and
reads the diagnosis back.

  1. Clean baseline: a synthetic rates + rates-vol market (curves → swaption
     cube; curves → cap strip → SABR optionlets), calibrated with
     ``calibrate_market(strict=True)``; per-level labels, max residual and
     condition number.
  2. A NaN swaption vol in the ``2Yx10Y`` cell: ``strict=True`` raises naming
     the cell, ``failed_labels()`` returns exactly that cell, and every other
     cell — an independent per-node solve — equals the baseline fit.
  3. A stale cap quote: far off-market it makes the strip bucket infeasible
     — named by strike *and* cap maturity (``"K=-0.86% cap 2Y"``, matching
     the quote key ``USD.CAP.2Y.-86``), installed as NaN so the later buckets
     of the column fail too, with ``root_cause_labels()`` naming the first;
     merely stale it strips "fine" and surfaces downstream as unfittable
     optionlet smiles.
  4. Too-tight solver settings (``max_steps=1``): the curve level and a SABR
     smile report ``converged=False`` and are named — the curve by its worst
     cluster of quotes, not all 14; the downstream cube still converges — on
     the wrong forwards. An infeasible or NaN curve quote is named alone.
  5. Conditioning: a SABR smile quoted at nearly-redundant strikes against
     ``ILL_CONDITIONED_THRESHOLD``, vs. the healthy chain levels.
  6. Engine provenance: a frozen-``y`` Cheyette fit is engine-bound
     (``check_engine`` refuses another PDE grid), a Hull-White Jamshidian fit
     (``exact``) is portable.
  7. Fix the quotes and recalibrate: every level converges and the market
     equals the baseline.

Run::

    python examples/16_unified_calibration_diagnostics.py
"""

# %% Imports and setup
from __future__ import annotations

import dataclasses
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import valax
from valax.calibration import (
    ILL_CONDITIONED_THRESHOLD,
    CalibrationChain,
    CalibrationError,
    SolverSettings,
    calibrate_cheyette,
    calibrate_hull_white,
    calibrate_market,
    calibrate_sabr,
    cheyette_pde_prices,
    cheyette_provenance,
    check_engine,
    condition_alert,
    curve_graph_node,
    sabr_vol_fn,
    swaption_prices_from_vols,
)
from valax.core.binding import bind_curves
from valax.market import (
    RatesVolSpec,
    RatesWorldSpec,
    SeedRegistry,
    default_config,
    rates_vol_snapshot,
)
from valax.models.cheyette import CheyetteModel
from valax.models.sabr import SABRModel
from valax.pricing.analytic.swaptions import swaption_forward_and_annuity
from valax.pricing.pde import PDEConfig, Scheme
from valax.surfaces.caplet_stripping import cap_strip_labels, strip_caplet_vols

jax.config.update("jax_enable_x64", True)

FAST = os.environ.get("VALAX_EXAMPLES_FAST") == "1"


def header(s: str) -> None:
    print(f"\n{'=' * 72}\n  {s}\n{'=' * 72}")


def max_abs_diff(a, b) -> float:
    """Largest absolute difference over every float leaf of two pytrees."""
    la, lb = jax.tree.leaves(eqx.filter(a, eqx.is_inexact_array)), jax.tree.leaves(
        eqx.filter(b, eqx.is_inexact_array))
    return max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(la, lb))


def summary(res) -> str:
    worst = float(np.nanmax(np.asarray(res.diagnostics.max_abs_residual)))
    return (f"{res.name:<22} {res.problem:<13} {len(res.labels):>3} labels  "
            f"max|r| {worst:8.1e}  converged={bool(res.converged)}")


registry = SeedRegistry(master_seed=20260216, library_version=valax.__version__)
rates_spec = RatesWorldSpec(
    pillar_years=(1, 2, 3, 5, 7, 10, 12) if FAST else (1, 2, 3, 5, 7, 10, 12, 15),
)
vol_spec = RatesVolSpec(
    expiry_years=(1, 2) if FAST else (1, 2, 5),
    tenor_years=(2, 5, 10),
    cap_years=(1, 2, 3) if FAST else (1, 2, 3, 5),
)

# %% 1. Clean baseline
header("1. Clean baseline: curves → cube, curves → cap strip → optionlet SABR")

snap, world = rates_vol_snapshot(
    registry, default_config(), rates_spec, vol_spec, bp_noise=0.5, vol_bp_noise=0.5,
)
curves, vols = snap.parts["curves"], snap.parts["vols"]
CUBE, STRIP, OPT = vols.cube_node.name, vols.strip_node.name, vols.optionlet_node.name
base = calibrate_market(snap.chain, snap.quotes, snap.fixed, strict=True)
assert bool(base.converged)
kappas = {}
for res in base.results:
    kappas[res.name] = float(res.condition_number())
    print(f"  {summary(res)}  κ {kappas[res.name]:8.1e}")
    assert bool(res.converged) and res.failed_labels() == (), res.name
    assert np.isfinite(kappas[res.name]) and kappas[res.name] < ILL_CONDITIONED_THRESHOLD
print(f"  cube cells   {base.result(CUBE).labels}")
strip_labels = base.result(STRIP).labels
print(f"  strip buckets {strip_labels[:3]} … ({len(strip_labels)} = strikes × cap "
      f"maturities; one bootstrap per strike, each bucket named)")

# %% 2. A NaN swaption quote in one cube cell
header("2. NaN swaption vol in the 2Yx10Y cell: named, and isolated")

e_idx, t_idx = vol_spec.expiry_years.index(2), vol_spec.tenor_years.index(10)
nan_key = vols.swaption_keys[e_idx][t_idx][2]
nan_quotes = snap.quotes.with_values({nan_key: jnp.nan})
print(f"  quote {nan_key} := NaN")

try:
    calibrate_market(snap.chain, nan_quotes, snap.fixed, strict=True)
    raise AssertionError("strict=True should have raised")
except CalibrationError as exc:
    msg = str(exc)
    print(f"  strict=True raised: {msg}")
    assert "2Yx10Y" in msg and CUBE in msg

nan_cal = calibrate_market(snap.chain, nan_quotes, snap.fixed)  # strict=False: traceable
assert not bool(nan_cal.converged)
cube_res = nan_cal.result(CUBE)
print(f"  failed_labels() = {cube_res.failed_labels()}")
assert cube_res.failed_labels() == ("2Yx10Y",)
node_res = cube_res.node_max_abs_residuals()
for lab, r, ok in zip(cube_res.labels, node_res, np.asarray(cube_res.diagnostics.converged)):
    print(f"    {lab:>7}: max|r| {r:9.2e}  converged={bool(ok)}")
assert int(np.sum(np.asarray(cube_res.diagnostics.converged))) == len(cube_res.labels) - 1
assert not np.isfinite(node_res[cube_res.labels.index("2Yx10Y")])  # NaN counts as failed

healthy = np.ones(len(cube_res.labels), dtype=bool)
healthy[cube_res.labels.index("2Yx10Y")] = False
base_cube, nan_cube = base.market.surfaces[vol_spec.cube_id], nan_cal.market.surfaces[vol_spec.cube_id]
cell_diff = max(
    float(np.max(np.abs(np.asarray(getattr(nan_cube, f)).reshape(-1)[healthy]
                        - np.asarray(getattr(base_cube, f)).reshape(-1)[healthy])))
    for f in ("alphas", "rhos", "nus", "forwards")
)
print(f"  healthy cells vs baseline: max |Δ(alpha, rho, nu, F)| = {cell_diff:.1e}")
assert cell_diff <= 1e-12
print(f"  the failed cell's alpha in the (non-strict) market: "
      f"{float(nan_cube.alphas[e_idx, t_idx])} — visible, never silent")
# The other levels never read the cube: unaffected.
for name in (curves.node.name, STRIP, OPT):
    assert bool(nan_cal.result(name).converged)
    assert max_abs_diff(nan_cal.result(name).value, base.result(name).value) <= 1e-12

# %% 3. A stale / off-market cap quote
header("3. Off-market cap quote: infeasible strip bucket, or a stale but strippable one")

col = 1
bucket_labels = cap_strip_labels(world.caps, world.rates.reference_date)
n_buckets = len(bucket_labels)
strike_label = strip_labels[col * n_buckets].split(" ")[0]  # e.g. "K=-0.86%"
cap_key = vols.cap_keys[col][1]  # second cap maturity at this strike
bad_node = f"{strike_label} {bucket_labels[1]}"
print(f"  strip buckets {bucket_labels} — named by cap maturity, like the quote keys; "
      f"bad quote {cap_key}")
assert cap_key.split(".")[2] == bucket_labels[1].removeprefix("cap ")  # "2Y" ↔ "cap 2Y"
base_strip = base.market.surfaces[vol_spec.strip_id]
disc_id, fwd_id = world.discount_id, world.forward_id


def corrupt_cap(factor: float):
    q = snap.quotes.with_values({cap_key: snap.quotes[cap_key] * factor})
    return q, calibrate_market(snap.chain, q, snap.fixed)


# 3a. Far off-market (a tenth of its value): the bucket's incremental premium is
# out of reach of any non-negative caplet vol, so its Newton root-find fails.
# The failed bucket installs NaN — like a failed cube cell — so the later
# buckets of the column (which read it: the strip is triangular) fail too;
# root_cause_labels() names the first one, the bad quote's bucket.
bad_q, bad_cal = corrupt_cap(0.1)
strip_res, opt_res = bad_cal.result(STRIP), bad_cal.result(OPT)
conv = np.asarray(strip_res.diagnostics.converged)  # (n_strikes, n_buckets)
print(f"  [quote × 0.1] strip failed_labels()     = {strip_res.failed_labels()}")
print(f"                strip root_cause_labels() = {strip_res.root_cause_labels()}")
bad_vols = np.asarray(bad_cal.market.surfaces[vol_spec.strip_id].vols[col])
print(f"    stripped vols at {strike_label}: {np.round(bad_vols * 1e4, 2)} bp")
assert strip_res.failed_labels() == tuple(
    f"{strike_label} {b}" for b in bucket_labels[1:])
assert strip_res.root_cause_labels() == (bad_node,)
n_first = world.strip.bucket_ends[0]
assert np.isfinite(bad_vols[:n_first]).all() and np.isnan(bad_vols[n_first:]).all()
other = [k for k in range(conv.shape[0]) if k != col]
assert conv[other].all() and conv[col, 0]  # other columns, and the earlier bucket, are fine
col_diff = float(jnp.max(jnp.abs(bad_cal.market.surfaces[vol_spec.strip_id].vols[jnp.array(other)]
                                 - base_strip.vols[jnp.array(other)])))
print(f"    other strike columns vs baseline: max |Δvol| = {col_diff:.1e}")
assert col_diff <= 1e-12
try:
    bad_cal.check()
    raise AssertionError("check() should have raised")
except CalibrationError as exc:
    print(f"    market.check() raised: {exc}")
    assert bad_node in str(exc) and STRIP in str(exc) and "first failure" in str(exc)
print(f"    downstream optionlet SABR failed_labels() = {opt_res.failed_labels()}")
# Every optionlet expiry fed a NaN strike fails; the first bucket's expiries do not.
opt_conv = np.asarray(opt_res.diagnostics.converged)
assert opt_conv[:n_first].all() and not opt_conv[n_first:].any()

# The standalone strip of the same column names the cap bucket itself.
caps_k = [eqx.tree_at(lambda c: c.strike, cap, world.cap_strikes[col]) for cap in world.caps]
one = strip_caplet_vols(
    caps_k, base.market.curve(disc_id),
    cap_vols=bad_q.vector(vols.cap_keys[col]),
    forward_curve=base.market.curve(fwd_id), is_normal=True,
    name=f"cap strip {strike_label}",
)
print(f"    standalone strip_caplet_vols failed_labels() = {one.failed_labels()}, "
      f"root cause {one.root_cause_labels()}")
assert one.failed_labels() == bucket_labels[1:]
assert one.root_cause_labels() == (bucket_labels[1],)
try:
    one.check()
    raise AssertionError("check() should have raised")
except CalibrationError as exc:
    print(f"    standalone check() raised: {exc}")
    assert f"first failure(s) {[bucket_labels[1]]}" in str(exc)

# 3b. Merely stale (half its value): the bootstrap still has a root; the damage
# shows up as a jagged caplet term structure that SABR cannot fit downstream.
_, stale_cal = corrupt_cap(0.5)
stale_strip, stale_opt = stale_cal.result(STRIP), stale_cal.result(OPT)
print(f"  [quote × 0.5] strip converged={bool(stale_strip.converged)}; stripped vols at "
      f"{strike_label}: {np.round(np.asarray(stale_cal.market.surfaces[vol_spec.strip_id].vols[col]) * 1e4, 2)} bp")
print(f"    downstream optionlet SABR failed_labels() = {stale_opt.failed_labels()}")
assert bool(stale_strip.converged)
assert len(stale_opt.failed_labels()) > 0
try:
    stale_cal.check()
    raise AssertionError("check() should have raised")
except CalibrationError as exc:
    print(f"    market.check() raised: {exc}")
    assert OPT in str(exc) and stale_opt.failed_labels()[0] in str(exc)
# Expiries of the untouched first bucket are fitted exactly as in the baseline.
first = float(jnp.max(jnp.abs(stale_cal.market.surfaces[vol_spec.optionlet_id].alphas[:n_first]
                              - base.market.surfaces[vol_spec.optionlet_id].alphas[:n_first])))
print(f"    first-bucket optionlet alphas vs baseline: max |Δ| = {first:.1e}")
assert first <= 1e-12

# %% 4. Too-tight solver settings
header("4. Unconvergeable by settings: max_steps=1 on the curves and on a SABR smile")

TIGHT = SolverSettings(method="newton", max_steps=1)
tight_curves = curve_graph_node(
    curves.node.name, world.rates.reference_date, curves.specs, curves.instruments,
    curves.keys, settings=TIGHT,
)
tight_chain = CalibrationChain(nodes=(tight_curves,) + snap.chain.nodes[1:2])  # curves → cube
try:
    calibrate_market(tight_chain, snap.quotes, snap.fixed, strict=True)
    raise AssertionError("strict=True should have raised")
except CalibrationError as exc:
    print(f"  strict=True raised: {str(exc)[:200]}...")
    assert curves.node.name in str(exc) and "did not converge" in str(exc)
tight_cal = calibrate_market(tight_chain, snap.quotes, snap.fixed)
tc = tight_cal.result(curves.node.name)
print(f"  {summary(tc)}")
print(f"  failed_labels() = {tc.failed_labels()}  (the worst cluster, not all "
      f"{len(tc.labels)} quotes)")
# One Newton step from a flat guess leaves the long end furthest off: the
# cluster within a factor two of the worst node is named, nothing else.
tc_res = tc.node_max_abs_residuals()
long_end = tuple(l for l, r in zip(tc.labels, tc_res) if r >= 0.5 * tc_res.max())
assert not bool(tc.converged) and tc.failed_labels() == long_end
assert 0 < len(long_end) < len(tc.labels)
assert int(tc.diagnostics.n_steps) <= 1

# One infeasible curve quote (a 5Y OIS at -50%: no positive discount factor
# reprices it) diverges the joint Newton solve to NaN everywhere; the bad
# quote is still named alone, read off the residuals at the starting point.
neg_key = "USD.OIS.5Y"
neg_res = curves.node.calibrate(snap.quotes.with_values({neg_key: -0.5}))
print(f"  {neg_key} := -50%: converged={bool(neg_res.converged)}, all node residuals "
      f"NaN={bool(np.isnan(neg_res.node_max_abs_residuals()).all())}, "
      f"failed_labels() = {neg_res.failed_labels()}")
assert neg_res.failed_labels() == (neg_key,)
nan_curve = curves.node.calibrate(snap.quotes.with_values({"USD.IRS3M.7Y": jnp.nan}))
print(f"  USD.IRS3M.7Y := NaN: failed_labels() = {nan_curve.failed_labels()}")
assert nan_curve.failed_labels() == ("USD.IRS3M.7Y",)
fwd_err = float(jnp.max(jnp.abs(tight_cal.market.surfaces[vol_spec.cube_id].forwards
                                - base_cube.forwards))) * 1e4
print(f"  downstream cube converged={bool(tight_cal.result(CUBE).converged)} — but its "
      f"forwards are off by up to {fwd_err:.2f} bp: check the upstream level first")
assert fwd_err > 0.01

i, j = vol_spec.expiry_years.index(2), vol_spec.tenor_years.index(5)
f0, t0 = base_cube.forwards[i, j], base_cube.expiries[i]
offsets = jnp.asarray(vol_spec.strike_offsets_bp) * 1e-4
smile_vols = snap.quotes.vector(vols.swaption_keys[i][j])
vol_fn = sabr_vol_fn(True, vol_spec.shift)
sabr_kwargs = dict(fixed_beta=jnp.asarray(0.0), vol_fn=vol_fn, is_normal=True, name="sabr 2Yx5Y")
loose = calibrate_sabr(f0 + offsets, smile_vols, f0, t0, **sabr_kwargs).check()
tight = calibrate_sabr(f0 + offsets, smile_vols, f0, t0,
                       settings=SolverSettings(method="levenberg_marquardt", max_steps=1),
                       **sabr_kwargs)
print(f"  SABR default settings: converged={bool(loose.converged)} "
      f"steps={int(loose.diagnostics.n_steps)} rmse={float(loose.diagnostics.rmse) * 1e4:.2f} bp")
print(f"  SABR max_steps=1     : converged={bool(tight.converged)} "
      f"steps={int(tight.diagnostics.n_steps)} rmse={float(tight.diagnostics.rmse) * 1e4:.2f} bp")
assert bool(loose.converged) and not bool(tight.converged)
try:
    tight.check()
    raise AssertionError("check() should have raised")
except CalibrationError as exc:
    print(f"  check() raised: {exc}")
    assert "sabr 2Yx5Y" in str(exc)

# %% 5. Conditioning
header("5. Conditioning: nearly-redundant strikes vs ILL_CONDITIONED_THRESHOLD")

print(f"  threshold ILL_CONDITIONED_THRESHOLD = {ILL_CONDITIONED_THRESHOLD:.0e}")
for name, kappa in kappas.items():
    print(f"    {name:<22} κ = {kappa:8.1e}  (~{np.log10(kappa):.0f} of 16 digits lost)")
true_smile = SABRModel(alpha=world.cube.alphas[i, j], beta=world.cube.betas[i, j],
                       rho=world.cube.rhos[i, j], nu=world.cube.nus[i, j])
f_true = world.cube.forwards[i, j]
# Four noise-free quotes within 0.15 bp of the forward: they all measure the
# ATM level, so (rho, nu) are barely identified — nearly-redundant quotes.
clustered = f_true + jnp.asarray([-0.075, -0.025, 0.025, 0.075]) * 1e-4
cl_vols = jax.vmap(lambda k: vol_fn(true_smile, f_true, k, t0))(clustered)
cl = calibrate_sabr(clustered, cl_vols, f_true, t0, **sabr_kwargs)
kappa_cl = float(cl.condition_number())
print(f"  SABR on {len(clustered)} strikes 0.05 bp apart: converged={bool(cl.converged)}, "
      f"max|r| {float(cl.diagnostics.max_abs_residual):.1e}, κ = {kappa_cl:.1e}")
assert bool(cl.converged)  # the solver is happy: only the condition number objects
alert = condition_alert(cl.condition_number(), level_name=cl.name)
print(f"  condition_alert: {alert}")
assert kappa_cl > ILL_CONDITIONED_THRESHOLD and alert is not None and cl.name in alert
assert condition_alert(loose.condition_number(), level_name=loose.name) is None
try:
    cl.check(max_condition_number=ILL_CONDITIONED_THRESHOLD)
    raise AssertionError("check() should have raised")
except CalibrationError as exc:
    print("  check(max_condition_number=...) raised the same alert")
    assert str(exc) == alert
print(f"  the well-spread smile (±100 bp): κ = {float(loose.condition_number()):.1e} — "
      f"re-specify the quotes, not the solver")

# %% 6. Engine provenance
header("6. Provenance: a frozen-y Cheyette fit is engine-bound; Jamshidian is exact")

ois = base.market.curve(disc_id)
sw0 = world.swaptions[i][j]  # 2Y x 5Y, single-curve on the OIS curve
atm, _ = swaption_forward_and_annuity(sw0, ois)
basket = [eqx.tree_at(lambda s: s.strike, sw0, atm + d) for d in (-0.005, 0.0, 0.005)]
CFG = PDEConfig(n_spot=61, n_time=24 if FAST else 40, spot_range=6.0,
                scheme=Scheme.CRANK_NICOLSON)
true_ch = CheyetteModel(mean_reversion=jnp.asarray(0.05), volatility=jnp.asarray(0.009),
                        skew=jnp.asarray(0.1), initial_curve=ois)
targets = cheyette_pde_prices(true_ch, basket, "frozen", CFG)
ch = calibrate_cheyette(
    basket, targets, ois, fixed_params={"mean_reversion": 0.05}, engine="frozen", config=CFG,
    settings=SolverSettings(method="bfgs", rtol=1e-10, atol=1e-10, max_steps=60 if FAST else 200),
)
print(f"  Cheyette fit: σ={float(ch.value.volatility):.5f} (true 0.009), "
      f"β={float(ch.value.skew):.3f} (true 0.1), converged={bool(ch.converged)}")
print(f"  provenance: {ch.provenance.method!r}, {ch.provenance.engine_kind!r}")
assert ch.provenance == cheyette_provenance("frozen", CFG)
assert ch.provenance.engine_kind == "modelling_bias"
assert check_engine(ch, "pde", CFG) is ch
reprice = cheyette_pde_prices(bind_curves(ch.value, ois), basket, "frozen", CFG)
rel = float(jnp.max(jnp.abs(reprice / targets - 1.0)))
print(f"  check_engine('pde', same grid) accepted; basket repriced to {rel:.1e} (relative)")
assert rel < 1e-3
for method, cfg in (("pde", dataclasses.replace(CFG, n_spot=121)), ("mc", None)):
    try:
        check_engine(ch, method, cfg)
        raise AssertionError("check_engine should have refused")
    except CalibrationError as exc:
        print(f"  check_engine({method!r}, {'finer grid' if cfg else None}) refused: "
              f"{str(exc)[:110]}...")
        assert "modelling bias" in str(exc) and "cheyette" in str(exc)

hw = calibrate_hull_white(
    basket, swaption_prices_from_vols(basket, ois, jnp.full(3, 0.008), normal=True), ois,
    fixed_mean_reversion=jnp.asarray(0.03),
).check()
print(f"  Hull-White (Jamshidian) provenance {hw.provenance.engine_kind!r}: "
      f"σ={float(hw.value.volatility):.5f}")
assert hw.provenance.engine_kind == "exact"
for method, cfg in (("pde", CFG), ("mc", None), ("lattice", None)):
    assert check_engine(hw, method, cfg) is hw
disc_res = dataclasses.replace(ch, provenance=cheyette_provenance("exact", CFG))
assert disc_res.provenance.engine_kind == "discretised"
assert check_engine(disc_res, "pde", dataclasses.replace(CFG, n_spot=121)) is disc_res
print("  exact and discretised engines are audit metadata: any engine is accepted")

# %% 7. Fix and recalibrate
header("7. Replace the bad quotes, recalibrate: everything converges, matches baseline")

broken = nan_quotes.with_values({cap_key: snap.quotes[cap_key] * 0.1})
fixed_quotes = broken.with_values({k: snap.quotes[k] for k in (nan_key, cap_key)})
assert sum(not bool(jnp.isfinite(v)) for v in broken.values.values()) == 1
fixed_cal = calibrate_market(snap.chain, fixed_quotes, snap.fixed, strict=True)
for res in fixed_cal.results:
    print(f"  {summary(res)}")
    assert bool(res.converged) and res.failed_labels() == ()
diff = max_abs_diff(fixed_cal.market, base.market)
print(f"  recalibrated market vs baseline: max |Δ| over every leaf = {diff:.1e}")
assert diff <= 1e-12
print("\n  Diagnosis order: market.check() names the level and node; failed_labels(),")
print("  root_cause_labels() and node_max_abs_residuals() localise it; condition_number() flags an")
print("  ill-posed quote set; check_engine() guards engine-bound parameters.")
