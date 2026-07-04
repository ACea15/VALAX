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
| 5 | [`valax.pricing`](pricing.md), [`valax.greeks`](greeks.md) | Vectorised prices and Greeks. |
| 6 | `valax.market.synthetic.paths`, `scenarios` | `MarketData` tapes and `ScenarioSet` batches. |

A user-facing tour of the same workflow lives in
[User Guide → Synthetic Market Data](../guide/synthetic_market.md);
the runnable counterpart is `examples/08_end_to_end_workflow.py`.

## Core market state

Canonical container of the market state at a single valuation date.
All four leaves of `MarketData` are differentiable, so `jax.grad`
through a pricing function that consumes a `MarketData` returns
sensitivities to every spot, vol, dividend, and curve pillar
simultaneously.

::: valax.market.MarketData

::: valax.market.MarketScenario

::: valax.market.ScenarioSet

::: valax.market.zero_scenario

::: valax.market.stack_scenarios

## Synthetic data — configuration & seeds

### Config

Static configuration consumed by every generator. All fields are
plain Python tuples or scalars so the config is a pure static pytree
leaf — no JAX arrays escape it.

::: valax.market.SyntheticMarketConfig

::: valax.market.default_config

### Seed registry

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

See
[Reproducibility & Arbitrage Tests](../guide/reproducibility_and_arbitrage_tests.md)
for the broader contract.

::: valax.market.SeedRegistry

## Stage 1 — Snapshots

::: valax.market.sample_market_data

::: valax.market.sample_market_with_correlation

::: valax.market.sample_scalar_market

## Stage 1 — Curves

Deterministic helper (no RNG) that builds a two-pillar flat-rate
discount curve; the RNG-driven samplers wrap this and NSS.

::: valax.market.flat_discount_curve

::: valax.market.sample_flat_curve

Nelson–Siegel–Svensson curve sampler:

\[
    r(\tau) = \beta_0
        + \beta_1 \frac{1 - e^{-\tau/\tau_1}}{\tau/\tau_1}
        + \beta_2 \left(\frac{1 - e^{-\tau/\tau_1}}{\tau/\tau_1} - e^{-\tau/\tau_1}\right)
        + \beta_3 \left(\frac{1 - e^{-\tau/\tau_2}}{\tau/\tau_2} - e^{-\tau/\tau_2}\right).
\]

Six parameters sampled uniformly from `cfg.nss_param_ranges`; discount
factors are built on `cfg.nss_pillars_years` and clipped to \((0, 1]\)
so a wildly negative draw cannot break the DF contract.

::: valax.market.sample_nss_curve

::: valax.market.sample_discount_curve

## Stage 1 — Correlations

Symmetric, unit-diagonal, PSD matrices sampled from a Wishart-style
distribution and then clipped + eigenvalue-projected back into the
PSD cone. `validate_correlation(C)` is guaranteed to return a minimum
eigenvalue \(\geq 0\) on the output.

::: valax.market.sample_correlation

::: valax.market.block_correlation

::: valax.market.sample_correlation_from_config

## Stage 1 — Model parameter samplers

Each sampler draws a ground-truth model from documented domain-aware
ranges (positivity, Feller condition, \(|\rho| < 1\), etc.). Use these
when you want to test the **calibrator**, not the analytic formula.

Domain contracts (enforced by construction):

| Model | Contract |
|---|---|
| `HestonModel` | \(v_0, \kappa, \theta, \xi > 0\); \(-1 < \rho < 0\) (equity leverage). With `enforce_feller=True`, \(\xi < \sqrt{2 \kappa \theta}\) so the variance process stays positive. |
| `SABRModel` | \(\alpha, \nu > 0\); \(0 \le \beta \le 1\); \(-1 < \rho < 1\). |
| `HullWhiteModel` | `mean_reversion, volatility > 0`; the supplied `initial_curve` is *not* re-sampled. |
| `MultiAssetGBMModel` | All vols, dividends in their config ranges; correlation is PSD with unit diagonal. |

::: valax.market.sample_bs_params

::: valax.market.sample_heston_params

::: valax.market.sample_sabr_params

::: valax.market.sample_hull_white_params

::: valax.market.sample_multi_asset_gbm_params

## Stage 2 — Observation layer

These functions turn a clean truth into the kind of noisy data a desk
actually receives. Set the noise argument to zero for the noiseless
version.

::: valax.market.synthesize_sabr_smile

::: valax.market.synthesize_price_strip

::: valax.market.synthesize_curve_quotes

## Stage 4 — Portfolio

`sample_option_portfolio` returns a dict
`{"calls": (stacked_calls, asset_idx), "puts": (stacked_puts, asset_idx)}`.
The split is necessary because `EuropeanOption.is_call` is a static
field — a stacked pytree carrying both flags is not representable.
`asset_idx` is the asset-index vector that lets the caller gather
`md.spots[idx]` etc. for downstream vectorized pricing.

::: valax.market.OptionPortfolioSpec

::: valax.market.sample_option_portfolio

`sample_swap_portfolio` returns a list (not a stacked pytree) because
every swap has a different number of fixed coupons; rectangular
stacking is impossible without padding.

::: valax.market.SwapPortfolioSpec

::: valax.market.sample_swap_portfolio

## Stage 5/6 — Time evolution & scenarios

`evolve_market` returns a *stacked pytree*: `spots` has shape
`(n_dates, n_assets)` when `n_paths == 1`, else
`(n_paths, n_dates, n_assets)`. Vols, dividends, and the curve are
broadcast unchanged across time (first-iteration limitation; see
[roadmap](../roadmap.md)).

::: valax.market.evolve_market

`sample_scenario_set` produces iid Gaussian shocks with configurable
per-factor standard deviation (in basis points). Drop-in for
[`valax.risk`](risk.md) engines.

::: valax.market.sample_scenario_set

## Arbitrage injection (stress tests)

A deliberately-broken-data layer for proving the library detects,
sanitises, or at least fails *loudly* on inputs that violate
static-arbitrage constraints. Every injector takes valid data and
returns the minimally-invalid version plus a diagnosis.

::: valax.market.ArbDiagnosis

### Injectors

| Injector | Breaks | Severity knob |
|---|---|---|
| `inject_non_psd_correlation` | PSD requirement of `MultiAssetGBMModel.correlation` | `eps` |
| `inject_basket_variance_violation` | \(\|\rho_{ij}\| \le 1\) | `new_value` |
| `inject_butterfly_arb` | convexity of \(C(K)\) (positive density) | `bump` |
| `inject_non_convex_smile` | same, via single upward spike | `bump` |
| `inject_calendar_arb` | monotonicity of \(w(T) = \sigma^2 T\) | swap indices `(i, j)` |
| `inject_pcp_violation` | put-call parity | `bp` |
| `inject_negative_density` | second strike difference \(\geq 0\) | `bump` |
| `inject_inconsistent_bootstrap_quotes` | bootstrap residual feasibility | `bp_offset` |

The detect-or-regularise test pattern that consumes these is in
`tests/test_market/test_arbitrage_handling.py`. See also the
[reproducibility & arbitrage tests guide](../guide/reproducibility_and_arbitrage_tests.md).

::: valax.market.inject_non_psd_correlation

::: valax.market.inject_basket_variance_violation

::: valax.market.inject_butterfly_arb

::: valax.market.inject_non_convex_smile

::: valax.market.inject_calendar_arb

::: valax.market.inject_pcp_violation

::: valax.market.inject_negative_density

::: valax.market.inject_inconsistent_bootstrap_quotes

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
exceptions above.

## Non-goals

These are intentionally *not* provided by the current synthetic layer:

- **Random vol-surface synthesis attached to `MarketData`.** The
  `vols` field is scalar-per-asset; full SVI/SABR surface generation
  requires a schema extension and is on the roadmap.
- **Discrete-dividend schedules, FX market handles, inflation /
  credit curve generators.** Mechanical to add but out of scope for
  the first iteration.
- **Active arbitrage detection.** The injectors and exception types
  are in place; the corresponding library-side checkers are tracked
  as `xfail` items in the test suite.

## See also

- [User Guide → Synthetic Market Data](../guide/synthetic_market.md) — tutorial walkthrough.
- [User Guide → Reproducibility & Arbitrage Tests](../guide/reproducibility_and_arbitrage_tests.md) — seed contract, golden harness, arbitrage methodology.
- [API → Curves](curves.md) — `DiscountCurve` and bootstrap instruments.
- [API → Models](models.md) — the model classes that the parameter samplers produce.
- [API → Calibration](calibration.md) — the consumers of the observation layer.
- [API → Risk](risk.md) — the consumer of `MarketScenario` / `ScenarioSet`.
