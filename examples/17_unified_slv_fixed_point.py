"""Example 17 — Equity SLV: the leverage function as a reported fixed point.

  1. Sample a true equity market (OIS curve, spot, dividend, SABR smile term
     structure), quote it with noise, and calibrate the market chain (curve →
     spot/dividend → SABR smiles). Calibrate Heston to the 1Y smile quotes
     (κ held fixed) — pass 1 of the two-pass SLV calibration.
  2. Calibrate the SLV leverage ``L(k, t)`` (pass 2) for
     ``n_iterations = 1, 2, 3, …``. The leverage is the output of a Monte Carlo
     fixed point ``L_{k+1} = G(L_k)``, and the result says so: the last
     increment ``L_K − L_{K−1}`` is the residual, ``converged`` is
     ``max|ΔL| <= atol`` (never for a single pass), the clip fraction is
     reported, and ``check()`` covers the Heston child too. Each SLV model
     reprices the vanilla smile by Monte Carlo, including the 6M expiry the
     Heston backbone was *not* fitted to.
  3. Price an up-and-out call under local vol, Heston and SLV — three models
     that agree on the 1Y vanillas, three different barrier prices.
  4. Risk of the SLV barrier to every market quote by plain reverse-mode AD
     through the whole chain: curve and smile fits (implicit adjoints), the
     Heston fit (implicit adjoint), the *unrolled* leverage passes, and the
     Monte Carlo pricer. One entry is checked against a central finite
     difference that reruns every calibration with the same random numbers.

Run::

    python examples/17_unified_slv_fixed_point.py
"""

# %% Imports and setup
from __future__ import annotations

import dataclasses
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

import valax
from valax.calibration import (
    CalibrationError,
    SolverSettings,
    calibrate_market,
    calibrate_model,
    calibrate_slv_leverage,
    price_fit_node,
)
from valax.calibration.slv import SLV_SETTINGS
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
from valax.models import SLVModel
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.models.local_vol import LocalVolModel
from valax.pricing.analytic.black_scholes import (
    black_scholes_implied_vol,
    black_scholes_price,
)
from valax.pricing.mc.engine import MCConfig

jax.config.update("jax_enable_x64", True)

FAST = os.environ.get("VALAX_EXAMPLES_FAST") == "1"
VOL_NOISE_BP = 10.0
T_EXO = 1.0  # Heston basket expiry, leverage horizon, exotic expiry
KAPPA = 1.5  # Heston mean reversion (held fixed)
MIX = 0.5  # SLV mixing fraction: backbone vol-of-vol = MIX × Heston ξ
ITERATIONS = (1, 2, 3) if FAST else (1, 2, 3, 4, 5)
# Leverage calibration: cheap (one particle step per time row), same in both modes.
N_CAL = 50_000
K_GRID = jnp.linspace(-0.4, 0.4, 17)  # log-moneyness grid of L
T_GRID = jnp.linspace(T_EXO / 16, T_EXO, 16)
RIDGE = 200.0  # kernel-estimator prior weight (see the note in section 2)
MC = MCConfig(n_paths=20_000 if FAST else 100_000, n_steps=48)
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
header("1. Quotes → curve → SABR smiles; Heston fitted to the 1Y smile (pass 1)")

snap, world = equity_snapshot(registry, cfg, spec, bp_noise=0.5, vol_bp_noise=VOL_NOISE_BP)
smile_q = snap.parts["smile"]
calibrated = calibrate_market(snap.chain, snap.quotes, snap.fixed, strict=True)
market = calibrated.market
surface = market.surface()
smile_rmse_bp = float(calibrated.result("spx_smile").diagnostics.rmse) * 1e4
print(f"  spot {float(market.spot()):.2f}, dividend {float(market.dividend()):.3%}, "
      f"smile fit RMSE {smile_rmse_bp:.1f} vol bp (quote noise {VOL_NOISE_BP:.0f})")
assert smile_rmse_bp < 1.5 * VOL_NOISE_BP

heston_node = price_fit_node(
    "spx_heston",
    template=HestonModel(
        v0=jnp.asarray(0.04), kappa=jnp.asarray(KAPPA), theta=jnp.asarray(0.04),
        xi=jnp.asarray(0.5), rho=jnp.asarray(-0.6),
    ),
    transforms={k: HESTON_TRANSFORMS[k] for k in ("v0", "theta", "xi", "rho")},
    basket=[EuropeanOption(strike=k, expiry=jnp.asarray(T_EXO)) for k in world.strikes[I_EXO]],
    quote_keys=smile_q.vol_keys[I_EXO],
    quote_form="lognormal_vol",
    fixed_fields={"kappa": KAPPA_KEY},
)
fixed = snap.fixed.merge(QuoteSet.from_dict({KAPPA_KEY: KAPPA}))


def flat_carry(model, mkt):
    """Pin a model's rate/dividend at the 1Y curve rate and the dividend.

    The SLV particle method and path generator drift at a flat ``r − q``; at
    1Y this is exactly the curve's forward, which is what the 1Y Heston
    basket, the leverage horizon and the exotic all see. All models below
    share it so their prices differ only through the dynamics.
    """
    rate = project_rate(mkt.curve(), jnp.asarray(T_EXO))
    return eqx.tree_at(
        lambda m: (m.rate, m.dividend), model, (rate, mkt.dividend()),
        is_leaf=lambda x: x is None,
    )


heston_result = calibrate_model(heston_node, snap.quotes, market, fixed, strict=True)
heston = flat_carry(heston_result.value, market)
print(f"  Heston: v0 {float(heston.v0):.4f}  θ {float(heston.theta):.4f}  "
      f"ξ {float(heston.xi):.3f}  ρ {float(heston.rho):+.3f}  (κ = {KAPPA} fixed)")
print(f"  1Y basket fit RMSE {float(heston_result.diagnostics.rmse):.2e} (price units), "
      f"{int(heston_result.diagnostics.n_steps)} LM steps")


def slv_backbone(hst: HestonModel) -> HestonModel:
    """The SLV stochastic-vol backbone: Heston with its vol-of-vol scaled by MIX."""
    return eqx.tree_at(lambda m: m.xi, hst, MIX * hst.xi)


def leverage_fit(hst, mkt, n_iterations, *, n_paths=N_CAL, settings=None):
    """Pass 2: the leverage fixed point against the market's SABR surface."""
    return calibrate_slv_leverage(
        slv_backbone(hst), mkt.surface(), mkt.spot(), K_GRID, T_GRID, n_paths, CAL_KEY,
        method="kernel", ridge=RIDGE, n_iterations=n_iterations, settings=settings,
    )


def two_pass(lev, hst_result, hst, mkt):
    """The composite ``calibrate_slv`` returns: pass 2 carrying pass 1 as a child.

    ``calibrate_slv`` runs pass 1 itself with ``calibrate_heston`` — all five
    Heston parameters free on one expiry, which on this smile wanders down a
    flat valley and never converges. Pass 1 here is the κ-fixed basket node
    above, so the two results are composed the same way by hand.
    """
    slv = SLVModel.from_heston_and_leverage(slv_backbone(hst), mkt.surface(), lev.value)
    return dataclasses.replace(lev, value=slv, children=(hst_result,))


class ClampedWings(eqx.Module):
    """The SABR surface with flat total variance beyond ``|k| > k_max``.

    Hagan's expansion is not usable at ``|log(K/F)| ≫ 1``; a local-vol path
    that wanders there would return NaN. Flat wings keep Dupire finite and
    leave every quoted strike (|k| < 0.25) untouched.
    """

    surface: eqx.Module
    k_max: float = eqx.field(static=True, default=1.0)

    def total_variance(
        self, log_moneyness: Float[Array, ""], expiry: Float[Array, ""]
    ) -> Float[Array, ""]:
        """Total variance at the clamped log-moneyness.

        Args:
            log_moneyness: ``k = log(K / F_T)``.
            expiry: Year fraction.

        Returns:
            ``w(clip(k, -k_max, k_max), T)``.
        """
        k = jnp.clip(log_moneyness, -self.k_max, self.k_max)
        return self.surface.total_variance(k, expiry)


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
    f"{spec.expiries[i]}Y {e:.0f} bp" for i, e in enumerate(heston_rms))
    + "  (only 1Y is in the basket)")

# %% 2. The leverage fixed point
header("2. SLV leverage as a fixed point: n_iterations sweep (pass 2)")

print(f"  backbone: Heston with ξ × {MIX} (mixing fraction); kernel estimator, ridge {RIDGE:.0f};")
print(f"  {N_CAL} particles on a {len(T_GRID)}t × {len(K_GRID)}k grid; "
      f"vanillas repriced with {MC.n_paths} paths × {MC.n_steps} steps")
print(f"  convergence: max|L_K − L_(K−1)| <= atol = {SLV_SETTINGS.atol} (SLV_SETTINGS)\n")
fits, increments, slv_err_bp, slv_sd_bp = {}, {}, {}, {}
print(f"  {'n':>2} {'max|ΔL|':>9} {'rms ΔL':>8} {'conv':>6} {'clip':>6}"
      + "".join(f"{f'RMS IV err {t}Y':>17}" for t in spec.expiries[:I_EXO + 1])
      + f"{'max|err|/σ_MC':>15}")
for n in ITERATIONS:
    fit = fits[n] = two_pass(leverage_fit(heston, market, n), heston_result, heston, market)
    d = fit.diagnostics
    increments[n] = float(d.max_abs_residual)
    iv, sd = mc_vanilla_ivs(fit.value, market)
    slv_err_bp[n] = (np.asarray(iv) - target_iv) * 1e4
    slv_sd_bp[n] = np.asarray(sd) * 1e4
    print(f"  {n:>2} {increments[n]:9.4f} {float(d.rmse):8.4f} {str(bool(d.converged)):>6} "
          f"{float(d.extras['clip_fraction']):6.1%}"
          + "".join(f"{e:14.0f} bp" for e in rms_by_expiry(slv_err_bp[n]))
          + f"{float(np.max(np.abs(slv_err_bp[n]) / slv_sd_bp[n])):15.1f}")
print("  (n = 1 measures the distance from the L ≡ 1 warm start, not convergence)")

# The report is exactly the documented contract.
for n, fit in fits.items():
    d = fit.diagnostics
    assert fit.theta is None and fit.problem == "fixed_point" and int(d.n_steps) == n
    assert bool(d.converged) == (n >= 2 and increments[n] <= SLV_SETTINGS.atol)
    assert 0.0 <= float(d.extras["clip_fraction"]) <= 1.0
    assert len(fit.children) == 1 and fit.children[0].name == "spx_heston"
for n in ITERATIONS[1:]:
    # Passes use deterministic per-pass keys, so the n-pass grid extends the
    # (n−1)-pass one and the residual is literally their difference.
    step = fits[n].value.leverage.values - fits[n - 1].value.leverage.values
    assert jnp.allclose(fits[n].diagnostics.residuals, step.ravel(), atol=1e-12)
assert not bool(fits[1].converged)
assert increments[2] < 0.25 * increments[1]  # the second pass removes the warm-start error
floor = max(increments[n] for n in ITERATIONS[1:])
print("\n  max|ΔL| by pass: " + " → ".join(f"{increments[n]:.3f}" for n in ITERATIONS) + ".")
print("  Once the warm-start error is gone the increment is Monte Carlo noise (every")
print(f"  pass re-simulates the swarm with fresh randomness): bounded by {floor:.3f} for")
print("  n >= 2 but not monotone — more passes cannot push it below the noise floor;")
print("  more particles can.")
assert floor < 3 * SLV_SETTINGS.atol

try:
    fits[1].check()
    raise AssertionError("a single pass must not pass check()")
except CalibrationError as e:
    print(f"  n=1 check(): CalibrationError — {str(e)[:70]}…")
n_conv = next(n for n in ITERATIONS if bool(fits[n].converged))
slv_fit = fits[n_conv].check()  # covers the Heston child as well
slv = slv_fit.value
print(f"  n={n_conv} check(): passes (fixed point and Heston child)")
tight = two_pass(
    leverage_fit(heston, market, n_conv, settings=SolverSettings(atol=1e-3, rtol=0.0)),
    heston_result, heston, market,
)
try:
    tight.check()
    raise AssertionError("atol below the MC noise floor must fail")
except CalibrationError:
    print(f"  n={n_conv} with atol=1e-3 (below the noise floor): CalibrationError, as it must")

slv_rms = rms_by_expiry(slv_err_bp[n_conv])
print(f"\n  RMS IV error vs the smile: Heston {heston_rms[0]:.0f} bp at {spec.expiries[0]}Y "
      f"→ SLV {slv_rms[0]:.0f} bp; 1Y {heston_rms[I_EXO]:.0f} → {slv_rms[I_EXO]:.0f} bp")
assert slv_rms[0] < heston_rms[0]  # SLV fixes the expiry Heston was not fitted to
assert max(slv_rms) < 40.0  # particle-method accuracy at this budget (docs: 100–200 bp at 10k)

# %% 3. Exotic: LV vs Heston vs SLV
header("3. Up-and-out call under local vol, Heston and SLV (same vanillas)")

spot = market.spot()
barrier = EquityBarrierOption(
    strike=spot, expiry=jnp.asarray(T_EXO), barrier=1.3 * spot,
    is_call=True, is_up=True, is_knock_in=False, smoothing=0.5,
)
vanilla = EuropeanOption(strike=spot, expiry=jnp.asarray(T_EXO))
local_vol = flat_carry(LocalVolModel(surface=ClampedWings(surface)), market)
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


def slv_barrier_pv(quotes: QuoteSet) -> Float[Array, ""]:
    """Quotes → curve/smile → Heston → leverage (unrolled) → SLV barrier price."""
    mkt = calibrate_market(snap.chain, quotes, fixed).market
    hst = flat_carry(heston_node.calibrate(quotes, mkt, fixed).value, mkt)
    lev = leverage_fit(hst, mkt, 2, n_paths=N_CAL_RISK).value
    model = SLVModel.from_heston_and_leverage(slv_backbone(hst), mkt.surface(), lev)
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

vol_keys = [k for row in smile_q.vol_keys for k in row]
k_star = max(vol_keys, key=lambda k: abs(float(g[k])))
h = 1e-4  # one vol bp
fd = (float(pv_fn(snap.quotes.with_values({k_star: snap.quotes[k_star] + h})))
      - float(pv_fn(snap.quotes.with_values({k_star: snap.quotes[k_star] - h})))) / (2 * h)
rel = abs(float(g[k_star]) - fd) / abs(fd)
print(f"\n  spot-check {k_star}: AD {float(g[k_star]):+.6f}  vs  full-recalibration "
      f"central FD {fd:+.6f}  (rel. diff {rel:.1e})")
assert rel < 1e-2
assert np.all(np.isfinite(np.asarray([float(v) for v in g.values.values()])))

# %% 5. What the fixed point is, and is not
header("5. The strategy")
print("  • The leverage is an explicit level (theta=None, problem='fixed_point'): its")
print("    value is the output of n_iterations unrolled Monte Carlo passes, not the")
print("    root of a residual. Nothing is solved, so there is no IFT level to invert.")
print("  • Its residual is the last increment L_K − L_(K−1); converged means that")
print("    increment is below atol, and atol must sit above the MC noise floor —")
print("    one pass never counts, and extra passes stop helping once at the floor.")
print("  • Its risk is plain reverse-mode AD through the passes actually run (and")
print("    through the Monte Carlo pricer), exact for the grid returned: the FD")
print("    spot-check above reruns the same passes with the same random numbers.")
