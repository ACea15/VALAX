# Market Data

The `valax.market` namespace carries everything VALAX considers "the
state of the world": spots, vols, dividends, the discount curve,
correlations, scenario shocks, and — since the synthetic submodule
landed — the generators, observation layer, and arbitrage-injection
stress tools that drive the library end-to-end without any external
data source.

The page is organised around the six-stage workflow:

```
Stage 1  Ground truth     →  Stage 2  Observations  →  Stage 3  Calibration
                                                            ↓
Stage 6  Risk          ←  Stage 5  Pricing & Greeks  ←  Stage 4  Portfolio
```

| Stage | Module | What it produces |
|---|---|---|
| 1 | `valax.market.synthetic.snapshots`, `curves`, `correlations`, `model_params` | `MarketData`, `DiscountCurve`, correlation matrix, ground-truth `SABRModel` / `HestonModel` / `HullWhiteModel` / `MultiAssetGBMModel`. |
| 2 | `valax.market.synthetic.observations` | Noisy implied-vol smile, price strip, par-rate quotes. |
| 3 | [`valax.calibration`](calibration.md) | Fitted model parameters. |
| 4 | `valax.market.synthetic.portfolio` | Stacked `EuropeanOption` / `InterestRateSwap` pytrees. |
| 5 | [`valax.portfolio`](pricing.md), [`valax.greeks`](greeks.md) | Vectorised prices and Greeks. |
| 6 | `valax.market.synthetic.paths`, `scenarios` | `MarketData` tapes and `ScenarioSet` batches. |

A user-facing tour of the same workflow lives in [User Guide → Synthetic Market Data](../guide/synthetic_market.md); the runnable counterpart is `examples/08_end_to_end_workflow.py`.

---

## Core market state

### `MarketData`

```python
class MarketData(eqx.Module):
    spots: Float[Array, "n_assets"]
    vols: Float[Array, "n_assets"]
    dividends: Float[Array, "n_assets"]
    discount_curve: DiscountCurve
```

Canonical container of the market state at a single valuation date.
All four leaves are differentiable, so `jax.grad` through a pricing
function that consumes a `MarketData` returns sensitivities to every
spot, vol, dividend, and curve pillar simultaneously.

| Field | Type | Differentiable | Description |
|---|---|---|---|
| `spots` | `Float[Array, "n_assets"]` | Yes | Per-asset spot prices. |
| `vols` | `Float[Array, "n_assets"]` | Yes | Per-asset implied vols (scalar — not a surface; see [non-goals](#non-goals)). |
| `dividends` | `Float[Array, "n_assets"]` | Yes | Per-asset continuous dividend yields. |
| `discount_curve` | [`DiscountCurve`](curves.md#discountcurve) | Yes (curve pillars) | Term structure of discount factors. |

### `MarketScenario`

```python
class MarketScenario(eqx.Module):
    spot_shocks: Float[Array, "n_assets"]
    vol_shocks: Float[Array, "n_assets"]
    rate_shocks: Float[Array, "n_pillars"]
    dividend_shocks: Float[Array, "n_assets"]
    multiplicative: bool = False  # static
```

Additive (or, when `multiplicative=True`, multiplicative for spots
only) shocks to apply to a `MarketData`. Consumed by
[`valax.risk.apply_scenario`](risk.md#apply_scenario).

### `ScenarioSet`

Same fields as `MarketScenario` with a leading `n_scenarios` axis on
every leaf. Designed for `jax.vmap` over the scenario axis. Created
either by `stack_scenarios(list_of_scenarios)` or directly via
[`sample_scenario_set`](#sample_scenario_set).

### `zero_scenario`

```python
zero_scenario(n_assets, n_pillars) -> MarketScenario
```

No-op scenario (all shocks zero). Useful as a baseline pivot.

### `stack_scenarios`

```python
stack_scenarios(scenarios: list[MarketScenario]) -> ScenarioSet
```

Stack a list of single-scenario pytrees into a batched `ScenarioSet`.

---

## Synthetic data — configuration & seeds

### `SyntheticMarketConfig`

```python
class SyntheticMarketConfig(eqx.Module):
    n_assets: int = 3
    reference_date: Int[Array, ""]                                 # default 2026-01-01
    spot_range: tuple[float, float] = (50.0, 200.0)
    vol_range: tuple[float, float] = (0.10, 0.45)
    rate_range: tuple[float, float] = (-0.005, 0.06)
    div_range: tuple[float, float] = (0.0, 0.04)
    curve_kind: Literal["flat", "nss"] = "nss"
    nss_pillars_years: tuple[float, ...] = (1/12, 3/12, 6/12,
                                             1, 2, 3, 5, 7,
                                             10, 15, 20, 30)
    nss_param_ranges: tuple[tuple[float, float], ...] = (...)      # 6 NSS parameters
    correlation_kind: Literal["identity", "random", "block"] = "random"
    min_corr: float = -0.3
    max_corr: float = 0.85
    day_count: str = "act_365"
```

Static configuration consumed by every generator. All fields are
plain Python tuples or scalars so the config is a pure static pytree
leaf — no JAX arrays escape it.

```python
from valax.market import SyntheticMarketConfig, default_config
cfg = default_config(n_assets=5)                # convenience
cfg = SyntheticMarketConfig(n_assets=5,         # explicit
                            spot_range=(80, 150),
                            curve_kind="flat")
```

### `SeedRegistry`

```python
@dataclass
class SeedRegistry:
    master_seed: int
    library_version: str
    def key(self, name: str, version: int = 1) -> jax.Array: ...
    def split(self, name: str, n: int, version: int = 1) -> jax.Array: ...
    def snapshot(self) -> dict: ...
```

Reproducible namespace of PRNG keys derived from a single
`master_seed`. Every generator in `valax.market.synthetic` takes a
`SeedRegistry` instead of a raw `jax.random.PRNGKey` so:

- The same `(master_seed, library_version, name, version)` always
  yields the same key — across processes, machines, architectures.
- Renaming a stream is a breaking change; bumping `version` is the
  only non-breaking way to change a stream's bytes.
- `snapshot()` returns a JSON-serialisable record of every
  `(name, version)` pair consumed, suitable for embedding in golden
  manifests.

Key derivation (purely informational — callers do not need to know):

```
k = fold_in(fold_in(PRNGKey(master_seed),
                    sha256(library_version + "::v" + str(version))[:4]),
            sha256(name)[:4])
```

See [Reproducibility & Arbitrage Tests](../guide/reproducibility_and_arbitrage_tests.md) for the broader contract.

---

## Stage 1 — Snapshots

### `sample_market_data`

```python
sample_market_data(registry: SeedRegistry,
                   cfg: SyntheticMarketConfig) -> MarketData
```

Draw a complete `MarketData`. Spots and vols are strictly positive,
dividends are non-negative, the curve is whatever `cfg.curve_kind`
selects.

Stream names consumed: `synthetic.snapshot.{spots,vols,dividends}` plus
whatever the curve sampler uses.

### `sample_market_with_correlation`

```python
sample_market_with_correlation(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
    *,
    block_sizes: tuple[int, ...] | None = None,
) -> tuple[MarketData, Float[Array, "n_assets n_assets"]]
```

Returns the snapshot plus a correlation matrix compatible with
[`MultiAssetGBMModel`](models.md#multiassetgbmmodel). Pass
`block_sizes` only when `cfg.correlation_kind == "block"`.

### `sample_scalar_market`

```python
sample_scalar_market(
    registry: SeedRegistry,
    cfg: SyntheticMarketConfig,
    *,
    expiry_range: tuple[float, float] = (0.05, 2.0),
    moneyness_range: tuple[float, float] = (0.7, 1.3),
) -> dict[str, Float[Array, ""]]
```

Single-asset, single-option scalar draw — drop-in for the inputs
used in `examples/comparisons/01_european_options.py`. Returns a dict
keyed `{spot, vol, rate, dividend, expiry, strike}`.

---

## Stage 1 — Curves

### `flat_discount_curve`

```python
flat_discount_curve(
    rate: float | Float[Array, ""],
    reference_date: Int[Array, ""],
    horizon_years: float = 50.0,
    day_count: str = "act_365",
) -> DiscountCurve
```

Deterministic helper (no RNG) that builds a two-pillar flat-rate
discount curve. Fills the gap that `valax.curves` has no explicit
flat-curve constructor.

### `sample_flat_curve`

```python
sample_flat_curve(registry, cfg) -> DiscountCurve
```

Uniform draw of `r` from `cfg.rate_range`, then `flat_discount_curve`.

### `sample_nss_curve`

```python
sample_nss_curve(registry, cfg) -> DiscountCurve
```

Nelson-Siegel-Svensson curve sampler.

$$
r(\tau) = \beta_0
        + \beta_1 \frac{1 - e^{-\tau/\tau_1}}{\tau/\tau_1}
        + \beta_2 \left(\frac{1 - e^{-\tau/\tau_1}}{\tau/\tau_1} - e^{-\tau/\tau_1}\right)
        + \beta_3 \left(\frac{1 - e^{-\tau/\tau_2}}{\tau/\tau_2} - e^{-\tau/\tau_2}\right)
$$

Six parameters sampled uniformly from `cfg.nss_param_ranges`; discount
factors built on `cfg.nss_pillars_years` and clipped to `(0, 1]` so a
wildly negative draw cannot break the DF contract.

### `sample_discount_curve`

```python
sample_discount_curve(registry, cfg) -> DiscountCurve
```

Dispatches on `cfg.curve_kind`.

---

## Stage 1 — Correlations

### `sample_correlation`

```python
sample_correlation(
    registry: SeedRegistry,
    n: int,
    *,
    min_corr: float = -0.3,
    max_corr: float = 0.85,
    kind: str = "random",
) -> Float[Array, "n n"]
```

Symmetric, unit-diagonal, PSD matrix sampled from a Wishart-style
distribution and then clipped + eigenvalue-projected back into the
PSD cone. `validate_correlation(C)` is guaranteed to return a
minimum eigenvalue `>= 0` on the output.

### `block_correlation`

```python
block_correlation(
    registry: SeedRegistry,
    block_sizes: tuple[int, ...],
    intra: float = 0.7,
    inter: float = 0.2,
    jitter: float = 0.02,
) -> Float[Array, "n n"]
```

Block-structured correlation: within-block entries near `intra`,
cross-block entries near `inter`, perturbed by `jitter` and projected
to PSD.

### `sample_correlation_from_config`

```python
sample_correlation_from_config(registry, cfg, block_sizes=None) -> Float[Array, "n n"]
```

Dispatches on `cfg.correlation_kind`.

---

## Stage 1 — Model parameter samplers

Each function draws a ground-truth model from documented domain-aware
ranges (positivity, Feller condition, `|rho| < 1`, etc.). Use these
when you want to test the **calibrator**, not the analytic formula.

```python
sample_bs_params(registry, cfg) -> BlackScholesModel
sample_heston_params(registry, cfg, *, enforce_feller=True) -> HestonModel
sample_sabr_params(registry, cfg, *, fixed_beta=0.5) -> SABRModel
sample_hull_white_params(registry, initial_curve) -> HullWhiteModel
sample_multi_asset_gbm_params(registry, cfg, *, block_sizes=None) -> MultiAssetGBMModel
```

Domain contracts (enforced by construction):

| Model | Contract |
|---|---|
| `HestonModel` | `v0, kappa, theta, xi > 0`; `-1 < rho < 0` (equity leverage). With `enforce_feller=True`, `xi < sqrt(2 * kappa * theta)` so the variance process stays positive. |
| `SABRModel` | `alpha, nu > 0`; `0 <= beta <= 1`; `-1 < rho < 1`. |
| `HullWhiteModel` | `mean_reversion, volatility > 0`; the supplied `initial_curve` is *not* re-sampled. |
| `MultiAssetGBMModel` | All vols, dividends in their config ranges; correlation is PSD with unit diagonal. |

---

## Stage 2 — Observation layer

These functions turn a clean truth into the kind of noisy data a desk
actually receives. Set the noise argument to zero for the noiseless
version.

### `synthesize_sabr_smile`

```python
synthesize_sabr_smile(
    registry: SeedRegistry,
    model: SABRModel,
    forward: Float[Array, ""],
    expiry: Float[Array, ""],
    strikes: Float[Array, " n"],
    *,
    vol_bp_noise: float = 5.0,
) -> Float[Array, " n"]
```

Clean smile via `sabr_implied_vol`, plus additive Gaussian noise with
`vol_bp_noise` basis points of standard deviation.

### `synthesize_price_strip`

```python
synthesize_price_strip(
    registry: SeedRegistry,
    pricer: Callable[..., Float[Array, ""]],
    pricer_args_per_strike: Callable[[Float[Array, ""]], tuple],
    strikes: Float[Array, " n"],
    *,
    price_rel_noise: float = 1e-3,
) -> Float[Array, " n"]
```

Multiplicative Gaussian noise on the clean prices. Pricer is abstract
so this works for analytic BS, semi-analytic Heston, or a MC wrapper.

### `synthesize_curve_quotes`

```python
synthesize_curve_quotes(
    registry: SeedRegistry,
    par_rates: Float[Array, " n"],
    *,
    bp_noise: float = 1.0,
) -> Float[Array, " n"]
```

Additive bp-noise on a clean par-rate vector. Compute the clean
par rates from a truth curve via the standard curve helpers, then
pipe through this function.

---

## Stage 4 — Portfolio

### `OptionPortfolioSpec`

```python
@dataclass(frozen=True)
class OptionPortfolioSpec:
    n_per_asset: int = 6
    expiry_range: tuple[float, float] = (0.1, 2.0)
    moneyness_range: tuple[float, float] = (0.8, 1.2)
    call_probability: float = 0.5
```

### `sample_option_portfolio`

```python
sample_option_portfolio(registry, md, spec=OptionPortfolioSpec()
) -> dict[str, tuple[EuropeanOption, Int[Array, " n"]]]
```

Returns `{"calls": (stacked_calls, asset_idx), "puts": (stacked_puts, asset_idx)}`.

The split is necessary because `EuropeanOption.is_call` is a static
field — a stacked pytree carrying both flags is not representable.
`asset_idx` is the asset-index vector that lets the caller gather
`md.spots[idx]` etc. for [`batch_price`](pricing.md#batch_price).

### `SwapPortfolioSpec` / `sample_swap_portfolio`

```python
sample_swap_portfolio(registry, curve, spec=SwapPortfolioSpec()
) -> list[InterestRateSwap]
```

Returned as a list (not a stacked pytree) because every swap has a
different number of fixed coupons; rectangular stacking is impossible
without padding.

---

## Stage 5/6 — Time evolution & scenarios

### `evolve_market`

```python
evolve_market(
    registry: SeedRegistry,
    md0: MarketData,
    dates: Int[Array, " n_dates"],
    correlation: Float[Array, "n_assets n_assets"],
    *,
    n_paths: int = 1,
) -> MarketData
```

Evolve `md0` along the requested `dates` via correlated GBM on spots
(reuses [`generate_correlated_gbm_paths`](pricing.md#generate_correlated_gbm_paths)).

The returned `MarketData` is a *stacked pytree*: `spots` has shape
`(n_dates, n_assets)` when `n_paths == 1`, else
`(n_paths, n_dates, n_assets)`. Vols, dividends, and the curve are
broadcast unchanged across time (first-iteration limitation; see
[roadmap](../roadmap.md) item *stochastic curve evolution*).

### `sample_scenario_set`

```python
sample_scenario_set(
    registry: SeedRegistry,
    n_scenarios: int,
    n_assets: int,
    n_pillars: int,
    *,
    spot_sigma_bps: float = 200.0,
    vol_sigma_bps: float = 100.0,
    rate_sigma_bps: float = 25.0,
    dividend_sigma_bps: float = 25.0,
    multiplicative: bool = True,
) -> ScenarioSet
```

iid Gaussian shocks with configurable per-factor standard deviation
(in basis points). Drop-in for [`valax.risk`](risk.md) engines.

---

## Arbitrage injection (stress tests)

A deliberately-broken-data layer for proving the library detects,
sanitises, or at least fails *loudly* on inputs that violate static-
arbitrage constraints. Every injector takes valid data and returns
the minimally-invalid version plus a diagnosis.

### `ArbDiagnosis`

```python
@dataclass(frozen=True)
class ArbDiagnosis:
    kind: ArbKind                 # "non_psd_correlation", "butterfly", ...
    magnitude: float              # severity in natural units
    location: object | None       # index, strike, name, …
    note: str = ""
```

### Injectors

```python
inject_non_psd_correlation(correlation, eps=0.05)              -> (Float[n,n], ArbDiagnosis)
inject_butterfly_arb(strikes, vols, k_index, bump=-0.05)       -> (Float[n], ArbDiagnosis)
inject_non_convex_smile(strikes, vols, k_index, bump=0.10)     -> (Float[n], ArbDiagnosis)
inject_calendar_arb(total_variances, i, j)                     -> (Float[m], ArbDiagnosis)
inject_pcp_violation(call_prices, put_prices, bp=50.0)         -> ((Float[n], Float[n]), ArbDiagnosis)
inject_negative_density(strikes, call_prices, k_index, bump)   -> (Float[n], ArbDiagnosis)
inject_inconsistent_bootstrap_quotes(quotes, bp_offset, index) -> (Float[n], ArbDiagnosis)
inject_basket_variance_violation(correlation, i, j, new_value) -> (Float[n,n], ArbDiagnosis)
```

| Injector | Breaks | Severity knob |
|---|---|---|
| `inject_non_psd_correlation` | PSD requirement of `MultiAssetGBMModel.correlation` | `eps` |
| `inject_basket_variance_violation` | `|ρ_ij| ≤ 1` | `new_value` |
| `inject_butterfly_arb` | convexity of `C(K)` (positive density) | `bump` |
| `inject_non_convex_smile` | same, via single upward spike | `bump` |
| `inject_calendar_arb` | monotonicity of `w(T) = σ²·T` | swap indices `(i, j)` |
| `inject_pcp_violation` | put-call parity | `bp` |
| `inject_negative_density` | second strike difference ≥ 0 | `bump` |
| `inject_inconsistent_bootstrap_quotes` | bootstrap residual feasibility | `bp_offset` |

The detect-or-regularise test pattern that consumes these is in
`tests/test_market/test_arbitrage_handling.py`. See also the
[reproducibility & arbitrage tests guide](../guide/reproducibility_and_arbitrage_tests.md).

---

## Reserved exception types

The library does not raise these today — they are *reserved* so test
code can already name the error the library *should* raise once the
corresponding detector lands. Each is a `ValueError` subclass.

```python
from valax import (
    ArbitrageError,            # base class
    NonPSDCorrelationError,
    ButterflyArbError,
    CalendarArbError,
    PutCallParityError,
    NonConvexSmileError,
    InconsistentQuotesError,
)
```

The current set of `@pytest.mark.xfail` markers in
`tests/test_market/test_arbitrage_handling.py` is the machine-readable
backlog of missing safety checks; each xfail turns green the day the
corresponding constructor or validator starts raising one of the
exceptions above. See the [arbitrage detection backlog](../roadmap.md#arbitrage-detection--session-backlog)
in the roadmap.

---

## Golden datasets

Versioned reference outputs live under `tests/golden/v{version}/` and
are indexed by `tests/golden/golden_manifest.json`. The helper is
test-only and lives at `tests/golden/_helpers.py`:

```python
from tests.golden._helpers import assert_matches_golden
assert_matches_golden("synthetic.snapshot.spots", md.spots, version=1)
```

- `REGEN_GOLDEN=1` (env var) writes / overwrites the artifact and
  manifest entry; otherwise the helper compares + fails loudly on
  drift.
- `scripts/regen_goldens.py` is the single entry point for batch
  regeneration.

Full schema and lifecycle in
[Reproducibility & Arbitrage Tests](../guide/reproducibility_and_arbitrage_tests.md).

---

## Non-goals

These are intentionally *not* provided by the current synthetic layer:

- **Random vol-surface synthesis attached to `MarketData`.** The vols
  field is scalar-per-asset; full SVI/SABR surface generation
  requires a schema extension and is on the roadmap.
- **Discrete-dividend schedules, FX market handles, inflation /
  credit curve generators.** Mechanical to add but out of scope for
  the first iteration.
- **Active arbitrage detection.** The injectors and exception types
  are in place; the corresponding library-side checkers are tracked
  as `xfail` items in the test suite.

---

## See also

- [User Guide → Synthetic Market Data](../guide/synthetic_market.md) — tutorial walkthrough.
- [User Guide → Reproducibility & Arbitrage Tests](../guide/reproducibility_and_arbitrage_tests.md) — seed contract, golden harness, arbitrage methodology.
- [API → Curves](curves.md) — `DiscountCurve` and bootstrap instruments.
- [API → Models](models.md) — the model classes that the parameter samplers produce.
- [API → Calibration](calibration.md) — the consumers of the observation layer.
- [API → Risk](risk.md) — the consumer of `MarketScenario` / `ScenarioSet`.
- [Roadmap → Arbitrage detection backlog](../roadmap.md#arbitrage-detection--session-backlog).
