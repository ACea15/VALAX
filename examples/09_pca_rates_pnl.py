# %% [markdown]
# # PCA Curve Shocks → Bond P&L
#
# Risk-management workflow:
#
# 1. Generate a synthetic time series of yield curves (250 days, NSS-drawn).
# 2. Extract zero-rate returns on a fixed pillar grid.
# 3. Fit a 3-component :class:`~valax.risk.factors.RatesFactorModel`.
#    Inspect explained variance and per-pillar R².
# 4. Shock a base curve with named "level / slope / curvature" PC moves and
#    reprice a small bond ladder under each scenario.
# 5. Run the same shocks through
#    :func:`~valax.risk.shocks.apply_scenario` + bond pricing to confirm the
#    typed ``model.scenario(...)`` path matches.

# %% Imports
import jax
import jax.numpy as jnp

from valax.curves.discount import DiscountCurve, zero_rate
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.market.data import MarketData
from valax.market.synthetic import SeedRegistry, default_config
from valax.market.synthetic.config import SyntheticMarketConfig
from valax.market.synthetic.curves import sample_nss_curve
from valax.risk.factors import fit_rates_pca, zero_rate_returns_from_snapshots
from valax.risk.shocks import apply_scenario

# ============================================================================
# 1. Generate a synthetic curve time series
# ============================================================================

# %% A 250-day NSS-curve history seeded for reproducibility.
N_DAYS = 250
REFERENCE_DATE = ymd_to_ordinal(2026, 1, 1)

base_cfg = default_config(n_assets=1)
base_cfg = SyntheticMarketConfig(
    **{**base_cfg.__dict__, "reference_date": REFERENCE_DATE, "curve_kind": "nss"},
)

curves: list[DiscountCurve] = []
for day in range(N_DAYS):
    registry = SeedRegistry(
        master_seed=20260101 + day,
        library_version="example-09",
    )
    curves.append(sample_nss_curve(registry, base_cfg))

print(f"Generated {len(curves)} synthetic curve snapshots.")

# ============================================================================
# 2. Build the zero-rate returns matrix on a fixed pillar grid
# ============================================================================

# %% Standard rates buckets in years: 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y.
PILLAR_TENORS = jnp.array(
    [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0],
    dtype=jnp.float64,
)
query_dates = jnp.array(
    [REFERENCE_DATE + int(round(t * 365.0)) for t in PILLAR_TENORS],
    dtype=jnp.int32,
)

returns = zero_rate_returns_from_snapshots(curves, query_dates)
print(f"\nReturns matrix shape: {returns.shape}  (n_obs - 1, n_pillars)")
print(f"Mean daily change (bp): {jnp.mean(returns) * 1e4:.2f}")
print(f"Std  daily change (bp): {jnp.std(returns) * 1e4:.2f}")

# ============================================================================
# 3. Fit a 3-component PCA factor model
# ============================================================================

# %% Three components is the textbook level / slope / curvature triple.
model = fit_rates_pca(returns, PILLAR_TENORS, n_components=3)

print("\n--- Rates PCA Factor Model ---")
print(f"Fraction explained:        {float(model.fraction_explained) * 100:.2f}%")
print(f"Eigenvalues (variance):    {[f'{float(v):.3e}' for v in model.eigenvalues]}")
print(f"One-sigma scores (bp/day): "
      f"{[f'{float(jnp.sqrt(v)) * 1e4:.2f}' for v in model.eigenvalues]}")

# %% Per-pillar R²: how much of each pillar's variance the top 3 PCs explain.
r2 = model.r_squared_per_pillar(returns)
print("\nPer-pillar R² (level + slope + curvature):")
for t, r in zip(PILLAR_TENORS.tolist(), r2.tolist(), strict=True):
    print(f"  T={t:>5.2f}Y   R² = {r * 100:5.2f}%")

# %% Inspect the loadings. PC1 should be nearly flat (level), PC2 monotone
# from short to long (slope), PC3 a "smile" (curvature).
print("\nLoading matrix J (rows = pillars, cols = PC1 / PC2 / PC3):")
for t, row in zip(PILLAR_TENORS.tolist(), model.jacobian.tolist(), strict=True):
    print(f"  T={t:>5.2f}Y   "
          f"PC1={row[0]:+.3f}  PC2={row[1]:+.3f}  PC3={row[2]:+.3f}")

# ============================================================================
# 4. Build a base curve and a small bond ladder
# ============================================================================

# %% Mildly upward-sloping base curve aligned with the PCA pillar grid.
base_zero_rates = jnp.array(
    [0.045, 0.044, 0.042, 0.040, 0.039, 0.038, 0.038, 0.038, 0.040, 0.042],
)
base_curve = DiscountCurve(
    pillar_dates=query_dates,
    discount_factors=jnp.exp(-base_zero_rates * PILLAR_TENORS),
    reference_date=jnp.int32(REFERENCE_DATE),
    day_count="act_365",
)

# %% Three zero-coupon bonds at 2Y / 5Y / 10Y with $1m notional each.
bond_maturities = jnp.array(
    [REFERENCE_DATE + 2 * 365, REFERENCE_DATE + 5 * 365, REFERENCE_DATE + 10 * 365],
    dtype=jnp.int32,
)
bond_notionals = jnp.array([1_000_000.0, 1_000_000.0, 1_000_000.0])


def bond_ladder_pv(curve: DiscountCurve) -> jnp.ndarray:
    """Total present value of the zero-coupon bond ladder."""
    return jnp.sum(bond_notionals * curve(bond_maturities))


base_pv = bond_ladder_pv(base_curve)
print(f"\nBase ladder PV: ${float(base_pv):,.2f}")

# ============================================================================
# 5. Reprice under named PC shocks
# ============================================================================

# %% Use one-sigma score units so the scenarios are comparable.
sigmas = jnp.sqrt(model.eigenvalues)

named_scenarios = {
    "+1σ Level (PC1)":     jnp.array([+1.0, 0.0, 0.0]) * sigmas,
    "-1σ Level (PC1)":     jnp.array([-1.0, 0.0, 0.0]) * sigmas,
    "+1σ Slope (PC2)":     jnp.array([0.0, +1.0, 0.0]) * sigmas,
    "+1σ Curvature (PC3)": jnp.array([0.0, 0.0, +1.0]) * sigmas,
    "Steepener (+L,+S)":   jnp.array([+1.0, +1.0, 0.0]) * sigmas,
}

print("\n--- P&L under named PC scenarios ---")
for name, scores in named_scenarios.items():
    shocked_curve = model.shock_curve(base_curve, scores)
    pv = bond_ladder_pv(shocked_curve)
    pnl = pv - base_pv
    print(f"  {name:<24s}  ΔPV = ${float(pnl):>+14,.2f}")

# ============================================================================
# 6. Cross-check via MarketScenario / apply_scenario
# ============================================================================

# %% The typed-scenario path: build a MarketScenario whose ``rate_shocks``
# are the reconstructed PC moves, then run it through the standard
# apply_scenario hook used by the rest of the risk stack.
base_market = MarketData(
    spots=jnp.zeros(0),
    vols=jnp.zeros(0),
    dividends=jnp.zeros(0),
    discount_curve=base_curve,
)

scores = jnp.array([+1.0, +1.0, 0.0]) * sigmas
scen = model.scenario(scores, n_assets=0)
shocked_market = apply_scenario(base_market, scen)

manual_curve = model.shock_curve(base_curve, scores)
delta = jnp.max(
    jnp.abs(
        shocked_market.discount_curve.discount_factors
        - manual_curve.discount_factors,
    ),
)
print(f"\nScenario-path vs manual-path DF difference: {float(delta):.2e}")
print("(should be zero up to floating point noise)")

# %% Differentiate through the whole shock-and-price pipeline. This is the
# main reason VALAX is JAX-native: PC-score sensitivities of the ladder
# come from a single ``jax.grad`` call.
def pnl_of_scores(s: jnp.ndarray) -> jnp.ndarray:
    return bond_ladder_pv(model.shock_curve(base_curve, s)) - base_pv


pc_sensitivities = jax.grad(pnl_of_scores)(jnp.zeros(3))
print("\nPC-score sensitivities at the base curve (ΔPV per unit score):")
for i, g in enumerate(pc_sensitivities.tolist(), start=1):
    print(f"  PC{i}: {g:+.4e}")
