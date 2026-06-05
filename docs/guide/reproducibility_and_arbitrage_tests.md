# Reproducibility & arbitrage stress tests

Two infrastructure pieces live alongside the synthetic-market
generator and are essential for trustworthy tests:

1. **Versioned golden datasets** — every reference output is stored on
   disk with a content hash and a manifest version.
2. **Deliberate arbitrage injection** — every kind of static arbitrage
   the library should *eventually* detect has a dedicated injector
   and an `xfail`-tracked test that turns green the day detection
   lands.

## Reproducibility

### `SeedRegistry`

Every random draw in `valax/market/synthetic/` derives its key from a
single `SeedRegistry` constructed with two ingredients:

- `master_seed` — an integer pinned per test session (override via
  `VALAX_MASTER_SEED`).
- `library_version` — the VALAX version string folded into every
  derived key so a library upgrade that intentionally changes a
  numerical contract can also rotate the seed.

```python
from valax.market.synthetic import SeedRegistry
registry = SeedRegistry(master_seed=20260101, library_version="0.1.0")

k1 = registry.key("synthetic.snapshot.spots", version=1)
k2 = registry.key("synthetic.snapshot.spots", version=2)   # ≠ k1
```

Bumping `version` is the **only** way to change a stream's bytes
without renaming it. Renaming a stream is breaking for any golden
artifact that consumed it.

### Golden manifest

Reference outputs live under `tests/golden/v{version}/{name}.npz`
and are indexed by `tests/golden/golden_manifest.json`:

```json
{
  "synthetic.snapshot.spots": {
    "version": 1,
    "sha256": "…",
    "library_version": "0.1.0",
    "jax_version": "0.4.x",
    "shape": [3],
    "dtype": "float64",
    "master_seed": 20260101
  }
}
```

In a test, use the harness:

```python
from tests.golden._helpers import assert_matches_golden
assert_matches_golden("synthetic.snapshot.spots", md.spots, version=1)
```

Three outcomes:

- **No manifest entry** → `AssertionError` ("run `REGEN_GOLDEN=1` to
  create it").
- **sha256 drift without version bump** → `AssertionError` ("drifted;
  bump version= and regenerate").
- **Match** → silent pass.

To intentionally update a golden:

```bash
REGEN_GOLDEN=1 pytest tests/test_market/  # or scripts/regen_goldens.py
```

The manifest update is a real diff in the PR.

## Calibration-residual closed loops

The synthetic layer enables a class of automated assertions that real
market data cannot: **the calibrator's residual must lie at the
observation-noise floor**. Demonstrated end-to-end in
`examples/08_end_to_end_workflow.py` and locked in as automated tests
in `tests/test_market/test_calibration_residual.py`.

### SABR smile residual

```python
sabr_truth   = sample_sabr_params(registry, cfg)
smile_clean  = synthesize_sabr_smile(registry, sabr_truth, F, T,
                                      strikes, vol_bp_noise=0.0)
smile_noisy  = synthesize_sabr_smile(registry, sabr_truth, F, T,
                                      strikes, vol_bp_noise=10.0)
sabr_fit, _  = calibrate_sabr(strikes, smile_noisy, F, T,
                              fixed_beta=sabr_truth.beta)
smile_fit    = jax.vmap(lambda K: sabr_implied_vol(sabr_fit, F, K, T))(strikes)

noise_rms = float(jnp.sqrt(jnp.mean((smile_noisy - smile_clean) ** 2)))
fit_rms   = float(jnp.sqrt(jnp.mean((smile_fit   - smile_noisy) ** 2)))
assert fit_rms <= 1.5 * noise_rms
```

The test is **non-tautological**: it does not compare parameters
(`alpha_fit ≈ alpha_truth`), which would only verify the optimiser.
It compares **smile residuals to the noise floor on the same draw**,
so a passing test means the calibrator is bottlenecked by data
quality, not by convergence. A failure points at either
`calibrate_sabr` convergence, `sabr_implied_vol`, or the noise model.

### Bootstrap non-self roundtrip

Three deliberately disjoint date grids:

1. NSS truth pillars (1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y).
2. **Bootstrap quote tenors** (4M, 18M, 4Y, 8Y) — chosen to *avoid*
   the NSS pillars.
3. **Comparison dates** (8M, 30M, 5Y) — disjoint from both.

```python
truth = sample_nss_curve(registry, cfg)
quotes = [DepositRate(start_date=ref, end_date=ref + d_days,
                      rate=jnp.array(par_rate_from_truth(truth, d_days)),
                      day_count=truth.day_count)
          for d_days in (120, 540, 1460, 2920)]
fitted = bootstrap_sequential(ref, quotes, day_count=truth.day_count)

for d_days in (240, 900, 1825):
    date    = ref + d_days
    r_truth = float(zero_rate(truth,  date))
    r_fit   = float(zero_rate(fitted, date))
    assert abs(r_fit - r_truth) < 25e-4   # 25 bp interpolation tolerance
```

Because no date appears on more than one grid, a passing test
measures real interpolation accuracy on a coarser, mis-aligned grid —
the bootstrap had to *reconstruct* truth, not merely echo it.

A second case (`test_flat_truth_recovers_exactly`) tightens the
tolerance to 1e-8 by using a flat truth curve (where log-DF
interpolation is exact). That variant catches drift in the deposit
DF formula or the day-count wiring.

### Pattern in one sentence

> **Compare the fitted residual to the empirical noise on the same
> draw**, not the fitted parameters to the true parameters.

## Arbitrage stress tests

VALAX has no library-wide arbitrage detector today. Rather than
quietly hope it never matters, the synthetic module ships a deliberate
**injection layer** that breaks valid data in ways that ought to be
detected. The associated tests document the gap.

### Reserved exception types

```python
from valax import (
    ArbitrageError,
    NonPSDCorrelationError,
    ButterflyArbError,
    CalendarArbError,
    PutCallParityError,
    NonConvexSmileError,
    InconsistentQuotesError,
)
```

All subclass `ValueError`. None of them is raised by the library
*yet*. They exist so tests can already name the error the consumer
*should* raise, and so the day detection lands, the test flips from
`xfail` to `pass` with a one-line change to the production code.

### Injectors

| Function | Breaks | Severity knob |
|---|---|---|
| `inject_non_psd_correlation` | symmetric PSD requirement of `MultiAssetGBMModel.correlation` | `eps` |
| `inject_basket_variance_violation` | `|ρ_ij| ≤ 1` | `new_value` |
| `inject_butterfly_arb` | convexity of `C(K)` (positive density) | `bump` |
| `inject_non_convex_smile` | same, via single upward spike | `bump` |
| `inject_calendar_arb` | monotonicity of `w(T) = σ²·T` | swap indices `(i, j)` |
| `inject_pcp_violation` | put-call parity | `bp` |
| `inject_negative_density` | second strike difference ≥ 0 | `bump` |
| `inject_inconsistent_bootstrap_quotes` | bootstrap residual feasibility | `bp_offset` |

Each returns `(perturbed, ArbDiagnosis(kind, magnitude, location, note))`
so a test can report exactly what it tried.

### Detect-or-regularise tests

In `tests/test_market/test_arbitrage_handling.py`, every injected
dataset is fed into each plausible consumer (model constructor,
path generator, bootstrap, surface fitter). Each assertion has one of
three flavours:

```python
# DETECT (preferred, currently green for one case):
min_eig = float(validate_correlation(bad))
assert min_eig < -1e-6

# GRACEFUL FAILURE (currently green via NaN propagation):
L = jnp.linalg.cholesky(bad)
assert bool(jnp.any(jnp.isnan(L)))

# DESIRED FUTURE BEHAVIOUR (currently xfail; turns green when detection lands):
@pytest.mark.xfail(strict=True, reason="Roadmap: MultiAssetGBMModel "
                                       "should raise on non-PSD corr.")
def test_constructor_should_raise():
    with pytest.raises(NonPSDCorrelationError):
        MultiAssetGBMModel(..., correlation=bad)
```

The current `xfail` set is the **public, machine-readable backlog of
missing safety checks**. Removing one entry is a real engineering
deliverable.

### Pytest markers

```toml
[tool.pytest.ini_options]
markers = [
    "arbitrage: ...",
    "golden: ...",
    "detects: ...",
]
```

Run only the arbitrage tests:

```bash
pytest -m arbitrage tests/
```

Run only the currently-passing detection assertions (skip the
xfail backlog):

```bash
pytest -m detects tests/
```
