"""Example 17 — Equity SLV: the leverage function as a reported fixed point.

  1. Sample a true equity market (OIS curve, spot, dividend, SABR smile term
     structure), quote it with noise, and calibrate the market chain (curve →
     spot/dividend → SABR smiles). Calibrate the Heston backbone to *both*
     smiles (κ held fixed) — pass 1 of the two-pass SLV calibration. A
     one-expiry, five-parameter Heston fit on this market does not converge;
     the two-expiry fit does.
  2. ``calibrate_slv(heston=…)`` builds the leverage ``L(k, t)`` (pass 2) on
     top of that backbone for ``n_iterations = 1, 2, 3, …``. Each pass is a
     full Guyon–Henry-Labordère forward sweep, so one pass is already the
     self-consistent calibration; later passes re-simulate it, and the result
     reports the last increment ``L_K − L_{K−1}`` as the residual (a Monte
     Carlo stability check), ``converged`` as ``max|ΔL| <= atol`` (never for a
     single pass), the clip fraction, and the Heston backbone as its child.
     Each SLV model reprices the vanilla smile by Monte Carlo.
  3. Price an up-and-out call under local vol, Heston and SLV — three models
     that agree on the 1Y vanillas, three different barrier prices.
  4. Risk of the SLV barrier to every market quote by plain reverse-mode AD
     through the whole chain: curve and smile fits (implicit adjoints), the
     Heston fit (implicit adjoint), the *unrolled* leverage passes, and the
     Monte Carlo pricer. One entry is checked against a central finite
     difference that reruns every calibration with the same random numbers.

The local-vol model prices on the raw SABR surface: the default
``LocalVolGuard`` of the path generator keeps a path that wanders deep into
the wing (where Hagan's expansion returns NaN) finite.

Run::

    python examples/17_unified_slv_fixed_point.py
"""

# %% Imports and setup
from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

import valax
from valax.calibration import (
    SLV_SETTINGS,
    CalibrationError,
    SolverSettings,
    calibrate_heston,
    calibrate_market,
    calibrate_model,
    calibrate_slv,
    price_fit_node,
)
from valax.calibration.transforms import HESTON_TRANSFORMS
from valax.core.binding import project_rate
from valax.core.pricing import price
from valax.instruments.options import EquityBarrierOption, EuropeanOption
from valax.market import (
    EquityWorldSpec,
    QuoteSet,
    SeedRegistry,
    default_config,
    equity_snapshot,
)
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.models.local_vol import LocalVolModel
from valax.pricing.analytic.black_scholes import (
    black_scholes_implied_vol,
    black_scholes_price,
)
from valax.pricing.analytic.heston import heston_cos_price
from valax.pricing.mc.engine import MCConfig

jax.config.update("jax_enable_x64", True)

FAST = os.environ.get("VALAX_EXAMPLES_FAST") == "1"
VOL_NOISE_BP = 10.0
T_EXO = 1.0  # leverage horizon, exotic expiry
KAPPA = 1.5  # Heston mean reversion (held fixed)
ITERATIONS = (1, 2, 3) if FAST else (1, 2, 3, 4, 5)
# Leverage calibration: cheap (one particle step per time row), same in both modes.
N_CAL = 50_000
K_GRID = jnp.linspace(-0.4, 0.4, 17)  # log-moneyness grid of L
# One leverage row per pricing step: the swarm and the pricer then take the
# same time steps and query L at the same midpoints.
N_STEPS = 48
T_GRID = jnp.linspace(T_EXO / N_STEPS, T_EXO, N_STEPS)
# Fixed-point tolerance on max|ΔL| over the whole grid. Measured noise floor of
# this budget (two independent 50k-particle sweeps): ≈ 0.1 — the sparse
# early-time wing cells dominate the max; ATM cells move by ~0.01.
SLV_ATOL = 0.2
MC = MCConfig(n_paths=20_000 if FAST else 100_000, n_steps=N_STEPS)
# Section 4 differentiates through every particle, so it runs a smaller budget.
N_CAL_RISK = 5_000 if FAST else 10_000
MC_RISK = MCConfig(n_paths=5_000 if FAST else 20_000, n_steps=24)
CAL_KEY = jax.random.PRNGKey(17)  # particle swarm of the fixed point
PRICE_KEY = jax.random.PRNGKey(2017)  # pricing paths (common random numbers)
KAPPA_KEY = "SPX.HESTON.KAPPA"


def header(s: str) -> None:
    print(f"\n{'=' * 72}\n  {s}\n{'=' * 72}")


registry = SeedRegistry(master_seed=20260317, library_version=valax.__version__)
cfg = default_config()
# Same market in both modes (full mode only adds passes and Monte Carlo budget).
spec = EquityWorldSpec(expiries=(0.5, 1.0))
I_EXO = spec.expiries.index(T_EXO)

# %% 1. Market chain and the Heston backbone
header("1. Quotes → curve → SABR smiles; Heston backbone on both smiles (pass 1)")

snap, world = equity_snapshot(registry, cfg, spec, bp_noise=0.5, vol_bp_noise=VOL_NOISE_BP)
smile_q = snap.parts["smile"]
calibrated = calibrate_market(snap.chain, snap.quotes, snap.fixed, strict=True)
market = calibrated.market
surface = market.surface()
smile_rmse_bp = float(calibrated.result("spx_smile").diagnostics.rmse) * 1e4
print(f"  spot {float(market.spot()):.2f}, dividend {float(market.dividend()):.3%}, "
      f"smile fit RMSE {smile_rmse_bp:.1f} vol bp (quote noise {VOL_NOISE_BP:.0f})")
assert smile_rmse_bp < 1.5 * VOL_NOISE_BP

# The flat carry every model below shares: the 1Y curve rate and the dividend.
# The SLV particle method and path generator drift at a flat r − q; at 1Y this
# is exactly the curve's forward, which is what the leverage horizon and the
# exotic see. All models share it so their prices differ only through dynamics.
RATE_1Y = project_rate(market.curve(), jnp.asarray(T_EXO))


def flat_carry(model, mkt):
    """Pin a model's rate/dividend at the 1Y curve rate and the dividend."""
    rate = project_rate(mkt.curve(), jnp.asarray(T_EXO))
    return eqx.tree_at(
        lambda m: (m.rate, m.dividend), model, (rate, mkt.dividend()),
        is_leaf=lambda x: x is None,
    )


# Why two expiries: on one SABR smile the five Heston parameters lie in a flat
# valley (κ, ξ and v0 trade off). ``calibrate_heston`` takes one expiry per
# quote, so the same call on both smiles is well posed.
def cos_call(model, k, s, r, q, t):
    return heston_cos_price(EuropeanOption(strike=k, expiry=t), s, r, q, model)


strikes = jnp.concatenate([jnp.asarray(world.strikes[i]) for i in range(I_EXO + 1)])
expiries = jnp.concatenate([
    jnp.full(len(world.strikes[i]), spec.expiries[i]) for i in range(I_EXO + 1)
])
smile_prices = jax.vmap(lambda k, t: black_scholes_price(
    EuropeanOption(strike=k, expiry=t), market.spot(), surface(k, t), RATE_1Y,
    market.dividend()))(strikes, expiries)
one_y = np.asarray(expiries) == T_EXO
for label, sel, t in (("1Y only ", one_y, jnp.asarray(T_EXO)),
                      ("both    ", slice(None), expiries)):
    r = calibrate_heston(strikes[sel], smile_prices[sel], market.spot(), RATE_1Y,
                         market.dividend(), t, cos_call)
    m = r.value
    print(f"  calibrate_heston, 5 free, {label}: converged {str(bool(r.converged)):5}  "
          f"{int(r.diagnostics.n_steps):3d} steps  κ {float(m.kappa):5.2f}  ξ {float(m.xi):5.2f}")
    if label.startswith("1Y"):
        assert not bool(r.converged)
    else:
        assert bool(r.converged)

# The chain node used below (and differentiated in section 4) is the same
# two-expiry fit through the pricer registry, with κ held fixed and reported.
heston_node = price_fit_node(
    "spx_heston",
    template=HestonModel(
        v0=jnp.asarray(0.04), kappa=jnp.asarray(KAPPA), theta=jnp.asarray(0.04),
        xi=jnp.asarray(0.5), rho=jnp.asarray(-0.6),
    ),
    transforms={k: HESTON_TRANSFORMS[k] for k in ("v0", "theta", "xi", "rho")},
    basket=[
        EuropeanOption(strike=k, expiry=jnp.asarray(spec.expiries[i]))
        for i in range(I_EXO + 1) for k in world.strikes[i]
    ],
    quote_keys=[k for i in range(I_EXO + 1) for k in smile_q.vol_keys[i]],
    quote_form="lognormal_vol",
    fixed_fields={"kappa": KAPPA_KEY},
)
fixed = snap.fixed.merge(QuoteSet.from_dict({KAPPA_KEY: KAPPA}))
heston_result = calibrate_model(heston_node, snap.quotes, market, fixed, strict=True)
heston = flat_carry(heston_result.value, market)
print(f"  backbone node: v0 {float(heston.v0):.4f}  θ {float(heston.theta):.4f}  "
      f"ξ {float(heston.xi):.3f}  ρ {float(heston.rho):+.3f}  (κ = {KAPPA} fixed), "
      f"{int(heston_result.diagnostics.n_steps)} LM steps")

# The vanilla grid every model must reprice: all quoted strikes up to 1Y.
vanillas = [
    (i, EuropeanOption(strike=k, expiry=jnp.asarray(spec.expiries[i])))
    for i in range(I_EXO + 1) for k in world.strikes[i]
]
rows = np.array([i for i, _ in vanillas])
target_iv = np.array([float(surface(o.strike, o.expiry)) for _, o in vanillas])


def rms_by_expiry(err_bp: np.ndarray) -> list[float]:
    return [float(np.sqrt(np.mean(err_bp[rows == i] ** 2))) for i in range(I_EXO + 1)]


@eqx.filter_jit
def analytic_ivs(model, mkt):
    """Implied vols of the vanilla grid priced analytically (Heston: COS)."""
    return jnp.stack([
        black_scholes_implied_vol(
            o, mkt.spot(), model.rate, model.dividend,
            price(o, model, mkt, method="analytic").price,
        )
        for _, o in vanillas
    ])


@eqx.filter_jit
def mc_vanilla_ivs(model, mkt):
    """Implied vols (and their MC standard errors) of the vanilla grid."""
    ivs, sds = [], []
    for _, o in vanillas:
        res = price(o, model, mkt, method="mc", config=MC, key=PRICE_KEY)
        vega = jax.grad(lambda v, _o=o: black_scholes_price(
            _o, mkt.spot(), v, model.rate, model.dividend))(surface(o.strike, o.expiry))
        ivs.append(black_scholes_implied_vol(o, mkt.spot(), model.rate, model.dividend, res.price))
        sds.append(res.stderr / vega)
    return jnp.stack(ivs), jnp.stack(sds)


heston_err_bp = (np.asarray(analytic_ivs(heston, market)) - target_iv) * 1e4
heston_rms = rms_by_expiry(heston_err_bp)
print("  Heston (COS) vs the SABR smile, RMS IV error: " + ", ".join(
    f"{spec.expiries[i]}Y {e:.0f} bp" for i, e in enumerate(heston_rms)))


def slv_fit(hst_result, mkt, n_iterations, *, n_paths=N_CAL, atol=SLV_ATOL):
    """Pass 2 on top of the fitted backbone: ``calibrate_slv(heston=…)``.

    The node result is recorded as the child; its model leaves rate/dividend
    to the market, so the flat carry is passed explicitly.
    """
    return calibrate_slv(
        spot=mkt.spot(), rate=project_rate(mkt.curve(), jnp.asarray(T_EXO)),
        dividend=mkt.dividend(), surface=mkt.surface(),
        log_moneyness_grid=K_GRID, time_grid=T_GRID, n_paths=n_paths, key=CAL_KEY,
        heston=hst_result, method="kernel", n_iterations=n_iterations,
        settings=SolverSettings(atol=atol, rtol=0.0),
    )


# %% 2. The leverage fixed point
header("2. SLV leverage: Guyon–Henry-Labordère passes, n_iterations sweep (pass 2)")

print("  backbone: the full Heston fit (no vol-of-vol mixing); kernel estimator,")
print(f"  default ridge; {N_CAL} particles on a {len(T_GRID)}t × {len(K_GRID)}k grid; "
      f"vanillas repriced with {MC.n_paths} paths × {MC.n_steps} steps")
print(f"  convergence: max|L_K − L_(K−1)| <= atol = {SLV_ATOL} (above the noise floor of "
      f"this budget; SLV_SETTINGS default {SLV_SETTINGS.atol})\n")
fits, increments, slv_err_bp, slv_sd_bp = {}, {}, {}, {}
print(f"  {'n':>2} {'max|ΔL|':>9} {'rms ΔL':>8} {'conv':>6} {'clip':>6}"
      + "".join(f"{f'RMS IV err {t}Y':>17}" for t in spec.expiries[:I_EXO + 1])
      + f"{'max|err|/σ_MC':>15}")
for n in ITERATIONS:
    fit = fits[n] = slv_fit(heston_result, market, n)
    d = fit.diagnostics
    increments[n] = float(d.max_abs_residual)
    iv, sd = mc_vanilla_ivs(fit.value, market)
    slv_err_bp[n] = (np.asarray(iv) - target_iv) * 1e4
    slv_sd_bp[n] = np.asarray(sd) * 1e4
    print(f"  {n:>2} {increments[n]:9.4f} {float(d.rmse):8.4f} {str(bool(d.converged)):>6} "
          f"{float(d.extras['clip_fraction']):6.1%}"
          + "".join(f"{e:14.0f} bp" for e in rms_by_expiry(slv_err_bp[n]))
          + f"{float(np.max(np.abs(slv_err_bp[n]) / slv_sd_bp[n])):15.1f}")
print("  (n = 1 reports the distance from L ≡ 1 — there is nothing to compare it with)")

# The report is exactly the documented contract.
for n, fit in fits.items():
    d = fit.diagnostics
    assert fit.theta is None and fit.problem == "fixed_point" and int(d.n_steps) == n
    assert bool(d.converged) == (n >= 2 and increments[n] <= SLV_ATOL)
    assert 0.0 <= float(d.extras["clip_fraction"]) <= 1.0
    assert len(fit.children) == 1 and fit.children[0].name == "spx_heston"
for n in ITERATIONS[1:]:
    # Passes use deterministic per-pass keys, so the n-pass grid extends the
    # (n−1)-pass one and the residual is literally their difference.
    step = fits[n].value.leverage.values - fits[n - 1].value.leverage.values
    assert jnp.allclose(fits[n].diagnostics.residuals, step.ravel(), atol=1e-12)
assert not bool(fits[1].converged)
# One GHL pass is already self-consistent: the vanillas are repriced as well
# after one pass as after several (within the pricing noise).
rms_1 = max(rms_by_expiry(slv_err_bp[1]))
rms_best = min(max(rms_by_expiry(slv_err_bp[n])) for n in ITERATIONS)
assert rms_1 < rms_best + 10.0, (rms_1, rms_best)
floor = max(increments[n] for n in ITERATIONS[1:])
print("\n  max|ΔL| by pass: " + " → ".join(f"{increments[n]:.3f}" for n in ITERATIONS) + ".")
print("  Every pass is a full self-consistent sweep with fresh randomness, so from")
print(f"  n = 2 on the increment is the Monte Carlo noise of the estimator, bounded")
print(f"  by {floor:.3f} and not monotone: a stability check, not a contraction. More")
print("  passes cannot push it down; more particles (or a larger ridge) can.")
assert floor <= SLV_ATOL

try:
    fits[1].check()
    raise AssertionError("a single pass must not pass check()")
except CalibrationError as e:
    print(f"  n=1 check(): CalibrationError — {str(e)[:70]}…")
n_conv = next(n for n in ITERATIONS if bool(fits[n].converged))
slv_result = fits[n_conv].check()  # covers the Heston child as well
slv = slv_result.value
print(f"  n={n_conv} check(): passes (fixed point and Heston child)")
tight = slv_fit(heston_result, market, n_conv, atol=1e-3)
try:
    tight.check()
    raise AssertionError("atol below the MC noise floor must fail")
except CalibrationError:
    print(f"  n={n_conv} with atol=1e-3 (below the noise floor): CalibrationError, as it must")

slv_rms = rms_by_expiry(slv_err_bp[n_conv])
print(f"\n  RMS IV error vs the smile: Heston {heston_rms[0]:.0f} bp at {spec.expiries[0]}Y "
      f"→ SLV {slv_rms[0]:.0f} bp; 1Y {heston_rms[I_EXO]:.0f} → {slv_rms[I_EXO]:.0f} bp")
z_max = float(np.max(np.abs(slv_err_bp[n_conv]) / slv_sd_bp[n_conv]))
rms_range = (min(max(rms_by_expiry(slv_err_bp[n])) for n in ITERATIONS),
             max(max(rms_by_expiry(slv_err_bp[n])) for n in ITERATIONS))
print(f"  worst SLV vanilla {z_max:.1f} σ_MC from the smile. σ_MC is the pricing noise only;")
print("  the leverage carries its own estimator noise — the worst-expiry RMS error ranges")
print(f"  {rms_range[0]:.0f}–{rms_range[1]:.0f} bp across the independent passes above. "
      "Heston's error is systematic.")
assert z_max < 4.0  # pricing noise + leverage-estimator noise, no systematic error
assert max(slv_rms) < 40.0  # particle-method accuracy at this budget

# %% 3. Exotic: LV vs Heston vs SLV
header("3. Up-and-out call under local vol, Heston and SLV (same vanillas)")

spot = market.spot()
barrier = EquityBarrierOption(
    strike=spot, expiry=jnp.asarray(T_EXO), barrier=1.3 * spot,
    is_call=True, is_up=True, is_knock_in=False, smoothing=0.5,
)
vanilla = EuropeanOption(strike=spot, expiry=jnp.asarray(T_EXO))
# Raw SABR surface: the path generator's default LocalVolGuard clamps the
# Dupire query to |k| <= 8 ATM standard deviations, so a path deep in the wing
# (where Hagan's expansion returns NaN) stays finite.
local_vol = flat_carry(LocalVolModel(surface=surface), market)
smile_pv = float(price(vanilla, BlackScholesModel(), market).price)


@eqx.filter_jit
def mc_pv(inst, model, mkt):
    res = price(inst, model, mkt, method="mc", config=MC, key=PRICE_KEY)
    return res.price, res.stderr


print(f"  K = spot = {float(spot):.2f}, barrier {float(barrier.barrier):.2f}, T = {T_EXO}Y; "
      f"smile (BS@K) ATM call {smile_pv:.4f}")
print(f"  {'model':<10}{'ATM call':>12}{'± σ_MC':>9}{'barrier':>12}{'± σ_MC':>9}")
book = {}
for name, model in (("local vol", local_vol), ("Heston", heston), ("SLV", slv)):
    v, v_sd = mc_pv(vanilla, model, market)
    b, b_sd = mc_pv(barrier, model, market)
    book[name] = (float(v), float(v_sd), float(b), float(b_sd))
    print(f"  {name:<10}{float(v):12.4f}{float(v_sd):9.4f}{float(b):12.4f}{float(b_sd):9.4f}")
for name, (v, v_sd, b, b_sd) in book.items():
    assert 0.0 < b < v, name  # knock-out can only remove value
    # All three are calibrated to the 1Y smile: the ATM call agrees with it to
    # MC noise plus the model's own vanilla-fit error (≤ 1% of premium).
    assert abs(v - smile_pv) < 4 * v_sd + 0.01 * smile_pv, (name, v, smile_pv)
lv_b, h_b, s_b = (book[k][2] for k in ("local vol", "Heston", "SLV"))
print(f"\n  barrier spread across models: {max(lv_b, h_b, s_b) - min(lv_b, h_b, s_b):.4f} "
      f"(vs MC σ ≈ {book['SLV'][3]:.4f}) — the vanillas pin the marginals, not the dynamics")
assert max(lv_b, h_b, s_b) - min(lv_b, h_b, s_b) > 5 * book["SLV"][3]
between = min(lv_b, h_b) <= s_b <= max(lv_b, h_b)
print(f"  SLV between LV and Heston: {between}")

# %% 4. Risk by AD through the whole chain
header("4. Quote risk of the SLV barrier: AD through calibrations + fixed point + MC")


def slv_barrier_pv(quotes: QuoteSet, backbone=None) -> Float[Array, ""]:
    """Quotes → curve/smile → Heston → leverage (unrolled) → SLV barrier price.

    ``backbone`` (a fitted Heston result) holds pass 1 fixed; ``None``
    recalibrates it from the quotes like everything else.
    """
    mkt = calibrate_market(snap.chain, quotes, fixed).market
    hst = heston_node.calibrate(quotes, mkt, fixed) if backbone is None else backbone
    model = slv_fit(hst, mkt, 2, n_paths=N_CAL_RISK).value
    return price(barrier, model, mkt, method="mc", config=MC_RISK, key=PRICE_KEY).price


pv_fn = eqx.filter_jit(slv_barrier_pv)
grad_fn = eqx.filter_jit(jax.grad(slv_barrier_pv))
pv0 = float(pv_fn(snap.quotes))
g = grad_fn(snap.quotes)
print(f"  barrier {pv0:.4f} on the smaller risk budget ({N_CAL_RISK} particles, "
      f"{MC_RISK.n_paths} paths × {MC_RISK.n_steps} steps,")
print("  coarser barrier monitoring — risk is exact for *this* computation)")
print(f"  delta  ∂V/∂spot            {float(g[smile_q.spot_key]):+10.4f}")
print(f"  rho    Σ ∂V/∂OIS (per bp)  {sum(float(g[k]) for k in snap.parts['curves'].keys) * 1e-4:+10.6f}")
print("  vega to smile quotes (per vol point), by expiry:")
for t, row in zip(spec.expiries, smile_q.vol_keys):
    print(f"    T={t:<5} " + "  ".join(f"{float(g[k]) * 1e-2:+.4f}" for k in row))
assert np.all(np.isfinite(np.asarray([float(v) for v in g.values.values()])))


def central_fd(fn, key, h):
    up = float(fn(snap.quotes.with_values({key: snap.quotes[key] + h})))
    dn = float(fn(snap.quotes.with_values({key: snap.quotes[key] - h})))
    return (up - dn) / (2 * h)


# Spot-check 1: everything downstream of the Heston backbone — the smile fit
# (implicit adjoint), the unrolled GHL passes and the MC pricer — against a
# central FD that reruns them with the same random numbers. This part of the
# Monte Carlo sample is a smooth function of the quotes.
pv_held = eqx.filter_jit(lambda q: slv_barrier_pv(q, heston_result))
g_held = eqx.filter_jit(jax.grad(lambda q: slv_barrier_pv(q, heston_result)))(snap.quotes)
vol_keys = [k for row in smile_q.vol_keys for k in row]
k_star = max(vol_keys, key=lambda k: abs(float(g_held[k])))
fd = central_fd(pv_held, k_star, 1e-4)  # one vol bp
rel = abs(float(g_held[k_star]) - fd) / abs(fd)
print(f"\n  spot-check {k_star}, Heston backbone held: AD {float(g_held[k_star]):+.6f}  vs  "
      f"central FD {fd:+.6f}  (rel. diff {rel:.1e})")
assert rel < 1e-2

# Spot-check 2: through the backbone as well. Each Andersen-QE variance step
# picks its branch by ψ = s²/m² ≷ 1.5, and ψ depends on (κ, θ, ξ). At this
# vol-of-vol some particle-steps sit near the switch, so moving the Heston fit
# flips branches: the sample is only *piecewise* smooth in the quotes. AD is
# the pathwise derivative of the smooth pieces; a bump FD picks up the flips
# and does not converge as h → 0 — so it cannot verify AD here.
fds = {h: central_fd(pv_fn, k_star, h) for h in (1e-4, 1e-5)}
print(f"  full chain incl. the Heston refit: AD {float(g[k_star]):+.4f}; central FD "
      + ", ".join(f"{fd_h:+.1f} (h={h:g})" for h, fd_h in fds.items()))
print("  — the FD is dominated by QE branch flips (it does not settle as h shrinks).")

# %% 5. What the fixed point is, and is not
header("5. The strategy")
print("  • The leverage is an explicit level (theta=None, problem='fixed_point'): its")
print("    value is the output of n_iterations unrolled Monte Carlo passes, each a")
print("    self-consistent Guyon–Henry-Labordère sweep — not the root of a residual.")
print("    Nothing is solved, so there is no IFT level to invert.")
print("  • Its residual is the last increment L_K − L_(K−1): the Monte Carlo noise")
print("    of the estimator. converged means that increment is below atol, and atol")
print("    must sit above the noise floor — one pass never counts, and extra passes")
print("    cannot push the increment down; more particles can.")
print("  • Its risk is plain reverse-mode AD through the passes actually run (and")
print("    through the Monte Carlo pricer), exact for the grid returned wherever the")
print("    sample is smooth: the held-backbone FD check above reruns the same passes")
print("    with the same random numbers. Through the Heston parameters the QE")
print("    variance scheme makes the sample piecewise smooth; AD is then pathwise.")
