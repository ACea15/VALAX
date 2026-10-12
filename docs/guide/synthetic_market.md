# Synthetic market data

VALAX ships a generator subpackage at `valax.market.synthetic` that
produces every kind of input the library consumes — spots,
volatilities, dividends, discount curves, correlations, ground-truth
model parameters, noisy quotes, portfolios, market histories, and
scenario sets — without any external data source.

The intent is **not** to replace real market data, but to make every
component of the library testable end-to-end with deterministic,
reproducible inputs.

## End-to-end workflow

The six stages drive each other in a single pipeline:

```
Stage 1  Ground truth     →  Stage 2  Observations  →  Stage 3  Calibration
                                                            ↓
Stage 6  Risk          ←  Stage 5  Pricing & Greeks  ←  Stage 4  Portfolio
```

| Stage | Module | Produces |
|------|--------|----------|
| 1 | `snapshots.py`, `curves.py`, `correlations.py`, `model_params.py` | `MarketData`, `DiscountCurve`, correlation matrix, ground-truth `HestonModel` / `SABRModel` / `HullWhiteModel` / `MultiAssetGBMModel`. |
| 2 | `observations.py` | Noisy implied-vol smile, price strip, par-rate quotes. |
| 3 | `valax.calibration` | Fitted models. |
| 4 | `portfolio.py` | Stacked `EuropeanOption` / `InterestRateSwap` pytrees. |
| 5 | `valax.portfolio.batch` | Vectorised prices + Greeks. |
| 6 | `scenarios.py`, `paths.py` | `ScenarioSet`, `MarketData` tapes. |

A worked example walking all six stages lives in
`examples/08_end_to_end_workflow.py`.

!!! tip "Prefer a narrative tutorial?"
    The [Rates End-to-End Tutorial](tutorial-rates.md) walks the same six
    stages on rates products — synthetic NSS curve, noisy par-rate quotes,
    bootstrap, validation, IRS + swaption pricing, and autodiff DV01 — with
    inline commentary on the modelling assumptions at each step.

## Synthetic quote universes

The stages above produce *calibrated objects* directly. To exercise the
unified calibration framework end to end, `valax.market` can also produce what
a desk observes — **quotes** — from a sampled true market, together with the
calibration chain that consumes them:

1. sample a true market — an OIS curve (Nelson-Siegel-Svensson), projection
   curves (OIS plus a smooth tenor basis), spot, dividend, a SABR smile term
   structure;
2. quote it with the library's own code — curve quotes are the par values
   that zero each bootstrap instrument's residual on the true curves
   (`valax.curves.par_quote`), smile quotes are the true smile's vols,
   premium quotes are `price` under a true model
   (`synthesize_model_quotes`);
3. add Gaussian noise (rate bp, vol bp, relative premium);
4. return the quotes **with their chain nodes**, built from the same
   specification, plus the truth.

```python
import valax
from valax.calibration import calibrate_market
from valax.market import SeedRegistry, default_config, equity_snapshot, rates_snapshot

registry = SeedRegistry(master_seed=7, library_version=valax.__version__)
cfg = default_config()

rates, world = rates_snapshot(registry, cfg, bp_noise=0.5)         # OIS + 3M
equity, eq_world = equity_snapshot(registry, cfg, vol_bp_noise=10.0)

calibrated = calibrate_market(equity.chain, equity.quotes, equity.fixed, strict=True)
```

The true curves lie exactly in the bootstrap's representation (log-linear
discount factors on the quote maturities), so a **noise-free** snapshot
recalibrates to the truth to solver tolerance — the strongest end-to-end check
of a chain. `RatesWorldSpec(zero_basis=True)` gives the single-curve limit with
multi-curve plumbing. `sample_quote_scenarios` stacks correlated quote
scenarios (rate level/slope, spot, vol level/skew with spot-vol correlation)
for `jax.vmap` over `calibrate_market`; `sample_equity_trades` and
`sample_swap_book` build products on the same world.

Examples 10–13 (`examples/1*_unified_*.py`) walk the full workflow on these
snapshots; see also [From Quotes to Risk](calibration-to-risk.md).

!!! note "Noise model"
    Quotes receive i.i.d. Gaussian noise today. Planned: bid/ask pairs whose
    half-spread becomes the least-squares weight, noise correlated across a
    strip, and stale / missing quotes flagged through `QuoteProvenance` — the
    extension point is documented in `valax/market/synthetic/universe.py`.

## Quick start

```python
import valax
from valax.market.synthetic import (
    SeedRegistry, SyntheticMarketConfig, sample_market_with_correlation,
)

registry = SeedRegistry(master_seed=20260101,
                        library_version=valax.__version__)
cfg = SyntheticMarketConfig(n_assets=3)
md, corr = sample_market_with_correlation(registry, cfg)

print(md.spots, md.vols, md.dividends)
print(md.discount_curve.discount_factors[:5])
print(corr)
```

## Configuration

`SyntheticMarketConfig` (in `valax/market/synthetic/config.py`) is a
single `eqx.Module` that carries every range and shape parameter:

```python
cfg = SyntheticMarketConfig(
    n_assets=5,
    spot_range=(80.0, 150.0),
    vol_range=(0.12, 0.40),
    rate_range=(0.0, 0.05),
    div_range=(0.0, 0.03),
    curve_kind="nss",                # or "flat"
    correlation_kind="random",       # or "identity", "block"
    min_corr=-0.2, max_corr=0.85,
)
```

Defaults are configured for plausible equity-desk scenarios; override
any field for a stress-test regime.

## Reproducibility contract

Every random draw goes through `SeedRegistry.key(name, version)`:

```python
key = registry.key("synthetic.snapshot.spots", version=1)
```

Two registries with identical `(master_seed, library_version)`
produce identical bytes for identical `(name, version)` pairs.
Bumping the version is the **only** way to change a stream's bytes
without renaming it; both are explicit acts.

`registry.snapshot()` returns a JSON-serialisable manifest of every
consumed `(name, version)` pair, which is embedded automatically in
every golden artifact (see
`docs/guide/reproducibility_and_arbitrage_tests.md`).

## Public surface

Everything in `valax.market.synthetic` is re-exported from
`valax.market`, so user code reads naturally:

```python
from valax.market import (
    # config + seeds
    SyntheticMarketConfig, SeedRegistry, default_config,
    # Stage 1
    sample_market_data, sample_market_with_correlation,
    sample_scalar_market,
    sample_flat_curve, sample_nss_curve, flat_discount_curve,
    sample_correlation, block_correlation,
    sample_bs_params, sample_heston_params, sample_sabr_params,
    sample_hull_white_params, sample_multi_asset_gbm_params,
    # Stage 2
    synthesize_sabr_smile, synthesize_price_strip,
    synthesize_curve_quotes,
    # Stage 4
    sample_option_portfolio, sample_swap_portfolio,
    OptionPortfolioSpec, SwapPortfolioSpec,
    # Time evolution + risk
    evolve_market, sample_scenario_set,
    # Arbitrage stress tests
    ArbDiagnosis,
    inject_non_psd_correlation, inject_butterfly_arb,
    inject_non_convex_smile, inject_calendar_arb,
    inject_pcp_violation, inject_negative_density,
    inject_inconsistent_bootstrap_quotes,
    inject_basket_variance_violation,
    # Quote universes (truth → quotes → calibration chain)
    RatesWorldSpec, sample_rates_world, synthesize_curve_quote_set,
    EquityWorldSpec, sample_equity_world, synthesize_smile_quote_set,
    synthesize_model_quotes, rates_snapshot, equity_snapshot,
    sample_quote_scenarios, sample_equity_trades, sample_swap_book,
)
```

## What this module deliberately does *not* do

- It does not perform real calibration to market data.
- It does not attach vol surfaces to the legacy `MarketData` —
  `MarketData.vols` is a scalar per asset. SABR smile term structures are
  synthesised on the `MarketState` path instead
  ([quote universes](#synthetic-quote-universes)).
- It does not synthesise FX, credit, inflation or rates-vol (swaption, cap)
  quotes yet.
- It does not implement detection of injected arbitrages — those
  exception types are *reserved* (see
  `valax.core.diagnostics`), and the corresponding tests are
  `xfail`-tracked.
