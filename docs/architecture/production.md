# Productionisation Design

> **Status:** Draft / Request for Comment
> **Scope:** Turning VALAX from a pure pricing library into a production-ready
> valuation tool: market data ingestion, calibration workflows, vol-surface
> construction, snapshot persistence, and mark-to-market pricing.
> **Audience:** Library maintainers and quants integrating VALAX into a
> production stack.

This document describes the architecture for promoting VALAX from a pure
pricing kernel into a deployable valuation engine. It is **not** a
roadmap of features (see [`roadmap.md`](../roadmap.md) for that). It is
the structural design that every production-related feature plugs
into.

---

## 1. Goals and non-goals

### Goals

1. **One canonical market state.** A single, typed, content-addressable
   pytree that fully describes "the market as of time `T`" — sufficient
   to price every instrument the library supports.
2. **Reproducible end-of-day runs.** Given the same raw quotes and the
   same `MarketSpec`, the build produces a bitwise-identical snapshot.
   Given the same snapshot and portfolio, MTM produces bitwise-identical
   P&L.
3. **Clean separation of pure / impure code.** The pure pricing kernel
   stays JIT/grad/vmap-compatible. All I/O, time, mutation, and
   persistence live in a thin imperative shell *around* the kernel.
4. **Composable workflows.** Calibration, MTM, risk, and stress runs
   are independent functions over the same `MarketState` /
   `Portfolio` types.
5. **Auditable provenance.** Every calibrated artefact (curve, surface)
   carries the inputs, method, residuals, and timestamps that produced
   it.

### Non-goals (for this design)

- Real-time tick handling and incremental state updates. Deliberately
  out of scope; addressed under
  [Tier 6.3 Real-Time Risk](../roadmap.md#63-real-time-risk).
- A relational schema for trades. We assume trades arrive at the
  pricing boundary as a `Portfolio` pytree; how they are stored is an
  integration concern.
- Vendor-specific market data parsers. We define adapter interfaces;
  Bloomberg / Refinitiv implementations live in private extensions.
- A general workflow scheduler. We rely on the host environment
  (Airflow, Dagster, cron, k8s `Job`) to invoke our entry points.

---

## 2. Architectural overview

### 2.1 Layered architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  L6  Service / API layer                                        │
│      gRPC + REST handlers, CLI                                  │
├─────────────────────────────────────────────────────────────────┤
│  L5  Workflow orchestration         (imperative)                │
│      build_market_state, mark_to_market, run_risk               │
├─────────────────────────────────────────────────────────────────┤
│  L4  Persistence                    (impure I/O)                │
│      SnapshotStore, AuditStore, PortfolioStore                  │
├─────────────────────────────────────────────────────────────────┤
│  L3  Calibration drivers            (pure JAX, dispatched)      │
│      bootstrap, surface fits, FX/inflation/credit calibrators   │
├─────────────────────────────────────────────────────────────────┤
│  L2  MarketState construction       (pure functional)           │
│      QuoteSet → MarketState   (deterministic build)             │
├─────────────────────────────────────────────────────────────────┤
│  L1  Ingestion / validation         (impure I/O at boundary)    │
│      adapters: csv, parquet, bloomberg → QuoteSet               │
├─────────────────────────────────────────────────────────────────┤
│  L0  Pure pricing kernel            (existing valax)            │
│      instruments, models, pricing, greeks, surfaces, risk       │
└─────────────────────────────────────────────────────────────────┘
```

The hard rule across this stack: **JAX purity is required at L0 and
inside any function called from L0**. Above L2, ordinary imperative
Python is allowed and expected. Layer L2 is the bridge — the boundary
where pure pytrees are first constructed from impure inputs.

### 2.2 The pure-core / imperative-shell principle

JAX transformations (`jit`, `grad`, `vmap`) require pure functions on
pytrees. The library you have today *is* the pure core. To make it
production-deployable we add an imperative shell that:

- **Reads** raw inputs (quotes, fixings, trades) from the outside
  world.
- **Builds** typed pytrees by calling pure constructors and
  calibrators.
- **Persists** pytrees into snapshots and audit tables.
- **Invokes** the kernel (priced via `vmap` / `jit`) on those
  snapshots.
- **Emits** results (prices, greeks, P&L) back into stores or APIs.

The shell is allowed to mutate, log, retry, raise, depend on the wall
clock, and call vendor SDKs. The kernel never does any of those
things.

### 2.3 Data flow for an end-of-day run

```
   raw vendor feeds
         │ (impure)
         ▼
   ┌────────────┐
   │  adapters  │  L1
   └────────────┘
         │  QuoteSet  (typed, validated)
         ▼
   ┌────────────────────────┐
   │  build_market_state    │  L2 + L3
   │  - dependency graph    │
   │  - calibrators         │
   │  - provenance attached │
   └────────────────────────┘
         │  (MarketState, BuildReport)
         ▼
   ┌────────────────┐         ┌────────────────┐
   │ SnapshotStore  │◄────────┤ content hash   │
   └────────────────┘         └────────────────┘
         │  SnapshotRef
         ▼
   ┌──────────────────┐    ┌──────────────────┐
   │  mark_to_market  │    │   run_risk       │   L5
   └──────────────────┘    └──────────────────┘
         │                          │
         │  jax.jit + jax.vmap       │
         ▼                          ▼
   ┌────────────────────────────────────────┐
   │   pricing kernel (L0) — pure JAX       │
   └────────────────────────────────────────┘
         │
         ▼
     P&L, greeks, risk reports → AuditStore
```

---

## 3. Data model

### 3.1 Identifier scheme

Production data needs identity. We adopt a hierarchical
dotted-namespace string identifier with a small, fixed alphabet:

```
<asset_class>.<issuer_or_pair>.<index_or_tenor>[.<qualifier>]
```

Examples:

| Identifier              | Meaning                                       |
|-------------------------|-----------------------------------------------|
| `USD.SOFR.OIS`          | USD SOFR OIS discount curve                   |
| `USD.SOFR.3M`           | USD 3M-SOFR forward projection curve          |
| `EUR.ESTR.OIS`          | EUR ESTR OIS discount curve                   |
| `EURUSD.SPOT`           | EUR/USD spot FX rate                          |
| `EURUSD.FWD`            | EUR/USD forward curve                         |
| `SPX.VOL.SVI`           | SPX equity vol surface, SVI parameterisation  |
| `EURUSD.VOL.SABR`       | EUR/USD FX vol surface, SABR per expiry       |
| `SPX.DIV`               | SPX dividend yield curve                      |
| `USCPI.INF`             | US CPI inflation curve                        |
| `USD.SOFR.FIXINGS`      | USD SOFR realised fixings history             |
| `IBM.SURV`              | IBM survival curve (CDS-implied)              |

Identifiers are treated as **opaque static strings** by the pricing
kernel. There is no parsing inside JIT-traced code; the workflow layer
resolves them up front.

### 3.2 `QuoteSet` — the validated input

```python
# valax/io/quotes.py  (new)

@dataclass(frozen=True)
class Quote:
    """A single market observation."""
    quote_id:  str              # e.g. "USD.SOFR.OIS.2Y"
    quote_type: str             # "deposit_rate", "swap_rate", "iv_point", ...
    value:     float
    asof:      datetime         # exchange / tick timestamp
    source:    str              # "BBG", "ICE", "MANUAL", "MOCK"
    metadata:  Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuoteSet:
    """All quotes required to build a single MarketState."""
    asof:   date                # valuation date
    quotes: Sequence[Quote]

    def by_id(self, quote_id: str) -> Quote: ...
    def filter(self, prefix: str) -> "QuoteSet": ...
    def hash(self) -> str:      # content hash for reproducibility
        ...
```

Notes:

- `QuoteSet` is **not** an `eqx.Module`. It is plain Python data — it
  never enters JIT-traced code. It is the boundary type.
- `value` is plain `float`, not `jnp.array`. Conversion happens during
  L2 build.
- `Quote.asof` is the *tick* timestamp; `QuoteSet.asof` is the
  *valuation date* assigned by the ingestion layer.

### 3.3 `MarketState` — the canonical pytree

Replaces the current narrow `MarketData`. This is the single object
every pricing function above L2 takes as its market argument:

```python
# valax/market/state.py  (new — supersedes valax/market/data.py)

class MarketState(eqx.Module):
    # ── Time anchor ─────────────────────────────────────────────
    asof:               Int[Array, ""]      # ordinal valuation date
    asof_timestamp_ns:  int = eqx.field(static=True)
    snapshot_id:        str = eqx.field(static=True)   # content hash

    # ── Curves ──────────────────────────────────────────────────
    discount_curves:    dict[str, DiscountCurve]   # OIS / risk-free
    forward_curves:     dict[str, DiscountCurve]   # tenor-specific
    survival_curves:    dict[str, SurvivalCurve]   # credit (later)
    inflation_curves:   dict[str, InflationCurve]
    dividend_curves:    dict[str, DividendCurve]

    # ── FX ──────────────────────────────────────────────────────
    fx_spots:           dict[str, Float[Array, ""]]
    fx_forward_curves:  dict[str, FXForwardCurve]   # optional

    # ── Vol surfaces ────────────────────────────────────────────
    vol_surfaces:       dict[str, VolSurface]      # equity / FX / rates

    # ── History / state ─────────────────────────────────────────
    fixings:            dict[str, FixingHistory]
```

Why a `dict[str, ...]` registry rather than flat fields?

- It is JAX-pytree native. `jax.tree_util` traverses dicts; the string
  keys are static metadata and the leaves are arrays.
- It is open-ended: adding a new curve type only requires adding a new
  field group, not breaking existing pricing signatures.
- Lookup by identifier is a Python dict access — happens **outside**
  JIT-traced code at the workflow layer, which then passes only the
  resolved leaf into the kernel call.
- Scenario application via `eqx.tree_at` becomes natural: replace the
  leaf at a given path, leave everything else identical.

The single mandatory invariant: **every leaf curve, surface, and
fixing series shares the same `asof` as the parent `MarketState`**. The
build layer enforces this; pricing functions assume it.

### 3.4 `Portfolio` — the trade container

```python
# valax/portfolio/portfolio.py  (new)

class Portfolio(eqx.Module):
    """A vmappable batch of trades with stable identity."""
    trade_ids:   tuple[str, ...] = eqx.field(static=True)
    instruments: eqx.Module        # batched pytree, leading dim = n_trades
    pricing_keys: tuple[str, ...] = eqx.field(static=True)
        # which MarketState keys each trade depends on (for narrowing)
```

The trade IDs are static (string tuple); the instrument data is a
batched pytree that `vmap` handles. `pricing_keys` lets the workflow
layer pull out only the curves/surfaces a trade actually needs, which
matters for cache-locality and minimising the recomputation surface.

### 3.5 `BuildReport` — calibration provenance

```python
# valax/workflows/report.py

@dataclass(frozen=True)
class CalibrationDiagnostic:
    artifact_id: str
    method:      str
    rmse:        float
    max_error:   float
    n_inputs:    int
    n_iter:      int
    converged:   bool
    elapsed_ms:  float
    residuals:   list[float] | None    # full residual vector if requested

@dataclass(frozen=True)
class BuildReport:
    asof:         date
    quote_hash:   str
    spec_hash:    str
    snapshot_id:  str
    diagnostics:  list[CalibrationDiagnostic]
    errors:       list[BuildError]    # per-artifact failures
    elapsed_ms:   float
```

The report is what gets persisted alongside the snapshot in the audit
store. It is emphatically *not* part of `MarketState` — pricing
functions never see it.

---

## 4. Layer 1 — Ingestion

### 4.1 Adapter interface

```python
# valax/io/adapters/base.py

class QuoteAdapter(Protocol):
    name: str   # "csv", "parquet", "bloomberg", "mock"

    def read(self, asof: date, *, universe: Sequence[str] | None = None) -> QuoteSet: ...
```

Initial concrete implementations:

| Adapter      | Module                            | Status   |
|--------------|-----------------------------------|----------|
| CSV          | `valax/io/adapters/csv.py`        | MVP      |
| Parquet      | `valax/io/adapters/parquet.py`    | MVP      |
| Mock / synth | `valax/io/adapters/mock.py`       | MVP (testing) |
| Bloomberg    | external extension                | Out of scope |
| Refinitiv    | external extension                | Out of scope |

The `mock` adapter is essential for tests and tutorials — it
deterministically generates a `QuoteSet` from a `MarketSpec` so we
can round-trip `Spec → quotes → MarketState → prices` in unit tests
without any external dependencies.

### 4.2 Validation

Validation runs immediately after ingestion and before L2:

- **Completeness** — every quote referenced in the `MarketSpec` is
  present.
- **Staleness** — quotes older than a per-source threshold are flagged.
- **Range checks** — rates within `(-5%, 30%)`, vols within
  `(0%, 500%)`, etc. (configurable).
- **Monotonicity** — for FX forward points across tenors, etc.
- **Cross-checks** — par swap rate vs. zero curve already present, FX
  triangulation, put-call parity on listed options.

Validators return a `ValidationReport`. The build can run in:

- `strict` mode — any error halts the build,
- `lenient` mode — errors are recorded but the build proceeds, missing
  inputs cause the affected artifacts to fail to calibrate.

The choice is per-deployment, not hard-coded.

---

## 5. Layer 2 / 3 — Build and calibration workflow

### 5.1 `MarketSpec` — declarative description

A `MarketSpec` describes *what* to build, not *how*. It is the
deployment artefact that defines a market (e.g. "USD EOD market"):

```yaml
# specs/usd_eod.yaml
asof_calendar: NY
day_count: act_365

discount_curves:
  USD.SOFR.OIS:
    method: sequential_bootstrap
    instruments:
      - {kind: deposit, id: USD.SOFR.ON,   tenor: 1D}
      - {kind: ois,     id: USD.SOFR.OIS.1W, tenor: 1W}
      # ...
      - {kind: ois,     id: USD.SOFR.OIS.30Y, tenor: 30Y}

forward_curves:
  USD.SOFR.3M:
    discount: USD.SOFR.OIS               # ← dependency
    method: dual_curve_bootstrap
    instruments:
      - {kind: future,  id: USD.SOFR.3M.IMM1}
      # ...
      - {kind: swap,    id: USD.SOFR.3M.10Y, tenor: 10Y}

vol_surfaces:
  SPX.VOL.SVI:
    method: svi_per_expiry
    spot:        SPX.SPOT
    discount:    USD.SOFR.OIS
    dividend:    SPX.DIV
    quote_filter: "SPX.OPT.*"
    expiries:    [1M, 3M, 6M, 1Y, 2Y]
```

The spec is parsed once at the start of the build into a typed
`MarketSpec` object, then the build engine walks it.

### 5.2 Build engine — dependency resolution

```python
# valax/workflows/build_state.py

def build_market_state(
    quotes: QuoteSet,
    spec: MarketSpec,
) -> tuple[MarketState, BuildReport]:
    """Deterministic, dependency-ordered build of a MarketState.

    1. Topologically sort artifacts by dependency edges in the spec.
    2. For each artifact, dispatch to the correct calibrator from
       the registry.
    3. Wrap the calibrator call to capture diagnostics.
    4. Aggregate into MarketState; hash inputs to compute snapshot_id.
    """
```

The dispatch table maps `(artifact_kind, method)` to a calibrator
function — exactly the pattern used by the MC dispatcher today
(`mc_price_dispatch`). Contributors register new calibrators with a
decorator:

```python
@register_calibrator(kind="discount_curve", method="sequential_bootstrap")
def _build_seq_bootstrap(quotes, spec_node, deps) -> tuple[DiscountCurve, CalibrationDiagnostic]:
    ...
```

The calibrator function itself is **pure JAX** — same code paths the
library already exposes (`bootstrap_sequential`,
`calibrate_svi_surface`, etc.). The wrapper around it is what computes
diagnostics and packages them into the report.

### 5.3 Determinism

For a given `(QuoteSet, MarketSpec)` pair, `build_market_state` must
return an identical `MarketState`. To achieve this:

- All array dtypes are pinned (`float64` for curves and surfaces — the
  library default).
- Calibrator solvers run with fixed iteration limits and tolerances
  taken from the spec, not from environment defaults.
- Iteration over dicts is sorted by key during construction.
- The `snapshot_id` is the SHA-256 of the canonical serialisation of
  `(QuoteSet.hash(), MarketSpec.hash(), version_pin)`.

This gives us "same inputs → same snapshot ID → same prices" — the
core auditability property.

### 5.4 Failure model

Calibration of a single artifact can fail (insufficient quotes,
solver non-convergence, validation breach). The build engine:

1. Continues past the failure if `mode == "lenient"`,
2. Records a `BuildError` in the report,
3. Marks downstream artifacts that depended on it as
   `skipped (upstream failure)`,
4. Returns a partial `MarketState` (with the failed leaves absent
   from their dicts).

Pricing functions later must defensively check key presence and surface
a clean error, rather than NaN-propagating into P&L.

---

## 6. Layer 4 — Persistence

### 6.1 Snapshot store (binary, pytree-native)

```python
# valax/io/store.py

class SnapshotStore(Protocol):
    def write(self, state: MarketState, report: BuildReport) -> SnapshotRef: ...
    def read(self, ref: SnapshotRef) -> tuple[MarketState, BuildReport]: ...
    def latest(self, asof: date) -> SnapshotRef: ...
    def list(self, asof_from: date, asof_to: date) -> list[SnapshotRef]: ...
```

Backends:

| Backend     | Module                           | Use case                |
|-------------|----------------------------------|-------------------------|
| Local FS    | `valax/io/store/local.py`        | Dev, single-node prod   |
| S3          | `valax/io/store/s3.py`           | Cloud / multi-region    |
| GCS / Azure | follow same pattern              | Future                  |

Serialisation: **`orbax.checkpoint`** for the pytree (handles
`jnp.array`s and arbitrary dict structures natively). The
`BuildReport` is written alongside as JSON.

A snapshot directory layout:

```
snapshots/
  2024-06-14/
    run-USD-EOD-20240614T170000Z-<sha8>/
      state.orbax/             # orbax checkpoint
      report.json              # BuildReport
      manifest.json            # snapshot_id, versions, sizes
```

### 6.2 Audit store (tabular)

A separate, queryable representation for compliance and BI:

| Table                 | Grain                                          |
|-----------------------|------------------------------------------------|
| `curves_audit`        | (asof, snapshot_id, curve_id, pillar_date, df) |
| `surfaces_audit`      | (asof, snapshot_id, surface_id, expiry, strike, iv) |
| `calibration_audit`   | (asof, snapshot_id, artifact_id, method, rmse, max_error, converged) |
| `pricing_audit`       | (asof, snapshot_id, trade_id, price, greeks_json) |
| `mtm_audit`           | (asof, snapshot_id, run_id, trade_id, pv, pv_ccy, pnl_explain_json) |

Backends: Parquet for development; Postgres/BigQuery in production.
The interface is identical (`AuditStore` protocol), the SQL/file
choice is per-deployment.

The audit store is **never read by the pricing kernel.** It is a
write-only sink from the workflow layer's perspective, queried
externally by reporting tools.

### 6.3 Portfolio store

Trades are mutable, audited, often supplied by an upstream booking
system. We expose only an interface:

```python
class PortfolioStore(Protocol):
    def load(self, asof: date, book: str) -> Portfolio: ...
    def save_pnl(self, run_id: str, rows: Sequence[PnLRow]) -> None: ...
```

The MVP ships a `ParquetPortfolioStore` (one parquet per book per
day). A SQL-backed implementation is downstream work.

---

## 7. Layer 5 — Workflow drivers

### 7.1 Mark-to-market

```python
# valax/workflows/mtm.py

def mark_to_market(
    portfolio: Portfolio,
    state:     MarketState,
    config:    MTMConfig,
) -> MTMReport:
    """Price a portfolio against a MarketState.

    Returns per-trade PV, greeks, and risk-bucket attribution.
    Internally vmap-batched across trades, jit-compiled.
    """
```

Internally this is little more than `batch_price` + `batch_greeks` +
`pnl_attribution`, but the *signature* is what production needs:
trade IDs in, P&L rows out, plus a single deterministic
`(snapshot_id, portfolio_version) → results` mapping. Re-running with
the same inputs produces bitwise-identical results.

The CLI entry point:

```bash
valax mtm \
    --asof 2024-06-14 \
    --snapshot run-USD-EOD-20240614T170000Z-3a7b1e \
    --portfolio s3://valax-prod/portfolios/usd_book.parquet \
    --output    s3://valax-prod/mtm/
```

### 7.2 Calibration / build

```bash
valax build-state \
    --asof 2024-06-14 \
    --quotes s3://valax-prod/quotes/usd_eod_20240614.parquet \
    --spec   specs/usd_eod.yaml \
    --store  s3://valax-prod/snapshots/
```

Output: a `SnapshotRef` printed to stdout, full `BuildReport` written
to the audit store.

### 7.3 Risk run

```bash
valax run-risk \
    --asof 2024-06-14 \
    --snapshot run-USD-EOD-20240614T170000Z-3a7b1e \
    --portfolio s3://valax-prod/portfolios/usd_book.parquet \
    --scenarios specs/historical_var.yaml \
    --output    s3://valax-prod/risk/
```

The scenarios spec drives `ScenarioSet` generation (parametric /
historical / stress) — wiring through to the existing
`valax.risk.scenarios` module. The driver applies each scenario via
`apply_scenario` to produce a vmapped `MarketState` batch and prices
the portfolio under each.

### 7.4 Common shape

All three drivers share:

```
load (impure) → build / vmap-price (pure JAX) → write (impure)
```

This is the **only** structure used at L5. New workflows (XVA, FRTB,
P&L explain) follow the same shape.

---

## 8. Layer 6 — Service layer (deferred)

Once the CLI workflows of §7 are stable, we expose them over gRPC and
REST without adding any new business logic:

- `PriceService.MarkToMarket(portfolio_ref, snapshot_ref) → mtm_report`
- `MarketDataService.GetSnapshot(snapshot_ref) → MarketState`
- `MarketDataService.BuildState(quote_ref, spec_ref) → snapshot_ref`
- `RiskService.RunRisk(portfolio_ref, snapshot_ref, scenario_ref) → risk_report`

The handlers are 5-line wrappers around the workflow drivers. We
defer building these until §7 is proven; otherwise we'll be reshaping
proto definitions every week.

---

## 9. Reproducibility, determinism, and audit

The non-negotiable production guarantee:

> Given the same `QuoteSet`, the same `MarketSpec`, the same VALAX
> version pin, and the same JAX backend, we produce a snapshot with
> the same `snapshot_id`, and any pricing run against it produces
> bitwise-identical numerical output.

Mechanisms:

1. **Hashing.** `QuoteSet.hash()`, `MarketSpec.hash()`, version pin →
   `snapshot_id`. Stored in the snapshot manifest.
2. **Pinned dtypes.** `float64` everywhere on CPU; document the GPU
   reproducibility caveats explicitly (cuBLAS non-determinism is real).
3. **Deterministic dict iteration.** All dicts iterated in
   sorted-key order during build and write.
4. **Pinned solvers.** `optimistix` solvers configured with fixed
   `rtol`/`atol`/`max_steps` from the spec, not defaults.
5. **PRNG keys carried in `MarketSpec`** for any Monte Carlo step that
   participates in calibration. No implicit `time.time()` seeds.
6. **Version pinning.** `pyproject.toml` lower bounds + a
   `valax.__version__` written into every snapshot manifest. Refusing
   to load a snapshot built with a different *major* version is
   acceptable; *minor* versions read older snapshots transparently.

For audit: every prices/risk run records `(run_id, snapshot_id,
portfolio_hash, code_version, started_at, finished_at, exit_code)` to
the audit store. A regulator asking "why did this trade book at this
price on this date?" can be answered by re-running with the recorded
IDs.

---

## 10. Module layout

New modules introduced by this design:

```
valax/
├── io/                            ← NEW
│   ├── quotes.py                  # QuoteSet, Quote
│   ├── validation.py              # validators, ValidationReport
│   ├── adapters/
│   │   ├── base.py                # QuoteAdapter protocol
│   │   ├── csv.py
│   │   ├── parquet.py
│   │   └── mock.py
│   └── store/
│       ├── base.py                # SnapshotStore, AuditStore, PortfolioStore protocols
│       ├── local.py               # filesystem snapshot store
│       ├── s3.py                  # S3 snapshot store
│       ├── parquet_audit.py
│       └── parquet_portfolio.py
│
├── curves/
│   ├── discount.py                # existing
│   ├── inflation.py               # existing
│   ├── instruments.py             # existing — extended in MC-Curves-1
│   ├── bootstrap.py               # existing (single-curve)
│   ├── multi_curve.py             # ← deprecated once graph solver lands
│   ├── graph.py                   ← NEW: CurveSpec, CurveGraph
│   ├── bootstrap_proto.py         ← NEW: BootstrapInstrument protocol
│   ├── bootstrap_graph.py         ← NEW: joint Newton solver
│   ├── interpolation.py           ← NEW: log_linear_df, monotone_convex, ...
│   ├── convexity.py               ← NEW: futures convexity adjustment plug-ins
│   ├── fixings.py                 ← NEW: FixingHistory, FixingSeries
│   └── diagnostics.py             ← NEW: CurveBuildDiagnostics, InstrumentFit
│
├── market/
│   ├── data.py                    # ← deprecated, becomes thin alias of state.py
│   ├── state.py                   ← NEW: MarketState
│   └── scenario.py                # existing
│
├── portfolio/
│   ├── batch.py                   # existing
│   └── portfolio.py               ← NEW: Portfolio container
│
├── workflows/                     ← NEW
│   ├── spec.py                    # MarketSpec parser
│   ├── build_state.py             # L2/L3 build engine
│   ├── registry.py                # @register_calibrator decorator
│   ├── mtm.py                     # mark_to_market driver
│   ├── risk_run.py                # run_risk driver
│   └── report.py                  # BuildReport, MTMReport, RiskReport
│
└── cli/                           ← NEW
    ├── __main__.py                # `python -m valax`
    ├── build.py                   # `valax build-state`
    ├── mtm.py                     # `valax mtm`
    └── risk.py                    # `valax run-risk`
```

No existing module's public API changes. `valax/market/data.py` keeps
exporting `MarketData` as a deprecated alias for a narrow subset of
`MarketState` to give downstream code a quiet migration path.

---

## 11. Multi-curve framework

The curve subsystem is the most structurally complex piece of the
build workflow and the one most directly tied to revenue on a rates
desk. This section is a focused deep-dive on what a production-grade
multi-curve framework looks like inside the design from §3–§7. It
defines the data model, the solver topology, the new bootstrap
instruments, and the calibration diagnostics.

### 11.1 Status today

| Capability                                  | Module                                       | State |
|---------------------------------------------|----------------------------------------------|:-----:|
| `DiscountCurve` pytree, log-linear DF interp| `valax/curves/discount.py`                   | ✅    |
| `InflationCurve` (log-CPI interp)           | `valax/curves/inflation.py`                  | ✅    |
| Quote types: `DepositRate`, `FRA`, `SwapRate` | `valax/curves/instruments.py`              | ✅    |
| Sequential single-curve bootstrap           | `bootstrap_sequential`                       | ✅    |
| Simultaneous single-curve bootstrap (Newton in log-DF) | `bootstrap_simultaneous`            | ✅    |
| `MultiCurveSet` container                   | `valax/curves/multi_curve.py`                | ⚠ Two-tier only |
| Dual-curve swap par bootstrap (OIS + tenor) | `bootstrap_multi_curve`                      | ✅ Sequential between curves |
| Differentiable through bootstrap (incl. `ImplicitAdjoint`) | `optimistix` integration      | ✅    |
| Bootstrap diagnostics (RMSE, max error)     | —                                            | ❌    |
| Money-market futures + convexity adjustment | —                                            | ❌    |
| OIS swap as distinct quote type             | —                                            | ❌    |
| Tenor basis swap                            | —                                            | ❌    |
| Cross-currency basis swap (CCBS)            | —                                            | ❌    |
| FX swap / forward as bootstrap input        | —                                            | ❌    |
| Joint global multi-currency solve           | —                                            | ❌    |
| Interpolation variants (monotone-convex, etc.) | —                                         | ❌    |
| Fixing-history injection                    | —                                            | ❌    |
| Curve identifiers (`USD.SOFR.OIS` etc.)     | —                                            | ❌    |
| Persistence / provenance                    | covered by §6 once Phase 3 lands             | →     |

The architectural skeleton is in place: post-2008 OIS-discounting plus
tenor forward curves is the assumed model, the bootstrap is fully
differentiable end-to-end, and the test suite covers single-currency
two-curve bootstraps with autodiff. The gap is in *coverage* of
instruments and *topology* of the solve.

### 11.2 Production gaps in five buckets

1. **Instrument coverage.** Deposits / FRAs / par swaps are not enough.
   Production bootstraps need money-market futures (with convexity
   adjustment), OIS swaps, tenor basis swaps, FX swaps / forwards, and
   cross-currency basis swaps.
2. **Solver topology.** Today's `bootstrap_multi_curve` is *sequential
   between curves* — OIS first, then each tenor independently. That
   breaks the moment a single instrument touches two unknown curves
   (tenor basis swap, CCBS).
3. **Conventions.** Single day count per swap, no business calendars,
   no stubs, no compounding-in-arrears, no fixings. These are the
   roadmap P1.1 / P1.2 prerequisites and they bite the bootstrap
   directly.
4. **Numerics.** One interpolation method (log-linear DF), no
   no-arbitrage constraints, no diagnostics returned to the caller, no
   turn-of-year handling.
5. **Production integration.** No identifiers, no `asof`, no
   provenance, no spec, no persistence. These are addressed by Phases
   1–3 of §12; the present section doesn't repeat them.

Buckets 1–4 are pure library work and are not addressed by the rest
of this document. They form a parallel workstream, planned in §13 as
**MC-Curves-1 / -2 / -3**.

### 11.3 Curve graph data model

The current `MultiCurveSet` is a fixed two-tier shape:
`discount_curve + forward_curves: dict[str, DiscountCurve]`. We
replace it with an open **curve graph** keyed by identifier:

```python
# valax/curves/graph.py  (NEW)

class CurveSpec(eqx.Module):
    """Static description of one curve to be bootstrapped."""
    curve_id:     str = eqx.field(static=True)        # "USD.SOFR.OIS"
    currency:     str = eqx.field(static=True)        # "USD"
    pillar_dates: Int[Array, " n"]
    interp:       str = eqx.field(static=True, default="log_linear_df")
    day_count:    str = eqx.field(static=True, default="act_365")


class CurveGraph(eqx.Module):
    """Result of a multi-curve bootstrap: many curves indexed by id."""
    curves: dict[str, DiscountCurve]    # str → calibrated curve
```

`CurveGraph` is a flat dict — it is intentionally *not* hierarchical.
"OIS-ness" or "tenor-ness" is conveyed by the identifier, not by the
container shape. This matches the `MarketState` design (§3.3): the
`MarketState.discount_curves`, `forward_curves`, etc. dicts are
populated by partitioning a single `CurveGraph` by identifier prefix
in the workflow layer.

#### Bootstrap instrument protocol

Today's `_compute_residuals` does `isinstance` dispatch over a closed
set of three instrument types. We replace it with a protocol so new
instruments don't require edits to a central function:

```python
# valax/curves/bootstrap_proto.py  (NEW)

class BootstrapInstrument(Protocol):
    """A market quote that imposes one residual on the curve graph."""

    # Identifiers of every curve this instrument touches.
    curves_touched: tuple[str, ...]      # static

    def residual(
        self,
        graph: CurveGraph,
        fixings: FixingHistory,
        ref_date: Int[Array, ""],
    ) -> Float[Array, ""]:
        """Return zero when the graph correctly reprices this quote."""
```

Every instrument provides its own residual. The global solver does
not know what kind of instrument it is — only that it owns one
residual equation and lists the curves it touches. New instruments
register themselves; no central dispatch table.

### 11.4 Joint global solver

```python
# valax/curves/bootstrap_graph.py  (NEW)

def bootstrap_curve_graph(
    reference_date: Int[Array, ""],
    curve_specs:    Sequence[CurveSpec],
    instruments:    Sequence[BootstrapInstrument],
    fixings:        FixingHistory | None = None,
    solver:         optx.AbstractRootFinder | None = None,
    initial_guess:  Mapping[str, Float[Array, " n"]] | None = None,
    max_steps:      int = 256,
) -> tuple[CurveGraph, CurveBuildDiagnostics]:
    """Solve for every curve in the graph in one Newton iteration.

    The solver concatenates per-curve log-DFs into a single state
    vector. The residual function reconstructs each curve, then
    asks each instrument for its residual. The system is
    well-determined when len(instruments) == sum(spec.pillars).
    """
```

Why this is the structural change that matters:

- **A tenor basis swap** (3M-vs-6M) constrains both the 3M and the 6M
  forward curve simultaneously. There is no ordering of single-curve
  solves that produces the right answer.
- **A cross-currency basis swap** (EURUSD CCBS) constrains the EUR
  discount curve, the EUR forward curve (for the float leg in EUR),
  and the USD discount curve, all together with the FX spot. This
  requires a joint solve across currencies.
- **An FX forward** ties two short-end curves via covered interest
  parity.

Once the solver is graph-shaped, all of these are just instruments
that return a residual; the solver doesn't grow new code paths per
asset class. The math generalises the existing
`_dual_curve_swap_residuals` in `multi_curve.py` from "two curves"
to "n curves".

### 11.5 New bootstrap instruments

The instrument set required by a real curve build, with the
residual each one imposes:

| Instrument                | Curves touched                                      | Residual |
|---------------------------|-----------------------------------------------------|----------|
| `DepositRate`             | one (forward or OIS)                                | $DF(T_1) (1 + r\tau) - DF(T_0)$ |
| `FRA`                     | one                                                 | $DF(T_1) (1 + r\tau) - DF(T_0)$ |
| `MoneyMarketFuture`       | one (forward) + optional `convexity_adj_fn`         | $F_{\text{quote}} - \text{adj} - \text{forward}(C_{\text{fwd}}, T_0, T_1)$ |
| `OISSwapRate`             | one (OIS) — compounded float leg via DF telescope   | $r \cdot A_{\text{ois}} - (DF_{\text{ois}}(T_0) - DF_{\text{ois}}(T_n))$ |
| `IborSwapRate`            | two (OIS discount + tenor forward)                  | dual-curve par condition |
| `TenorBasisSwap`          | two forward curves (+ OIS for discounting)          | float($C_a$) − float($C_b$) − spread · annuity |
| `FXSwap` / `FXForward`    | two OIS curves + FX spot                            | covered-interest-parity equation |
| `CrossCurrencyBasisSwap`  | two OIS, two forward, FX spot                       | MTM and constant-notional variants |
| `TurnInstrument`          | one                                                 | discrete jump anchor at year-end |

Convexity adjustment for futures is pluggable: `MoneyMarketFuture`
takes a `convexity_adj_fn(curve, T_0, T_1) -> Float[""]`. Initial
shipped variants:

- `constant_convexity_adj(bps)` — desk-supplied,
- `hull_white_convexity_adj(model)` — once the HW short-rate model is
  wired into the curve build (§ roadmap 2.1, already implemented).

### 11.6 Calibration diagnostics

```python
class CurveBuildDiagnostics(eqx.Module):
    """Per-curve and per-instrument repricing diagnostics."""
    rmse_per_curve:        dict[str, float]
    max_error_per_curve:   dict[str, float]
    fitted_vs_quoted:      list[InstrumentFit]   # one per input instrument
    n_iter:                int
    converged:             bool
    jacobian_condition:    float                 # of dR/d(log-DF)
    elapsed_ms:            float
```

`InstrumentFit` records `(instrument_id, curves_touched, quoted,
fitted, residual_bp)` per input quote. This is what the build engine
of §5.2 packages into `BuildReport.diagnostics` for each curve
artifact — same shape, same audit pipeline as the rest of the system.

A trader looking at a build report can see which quote is straining
the fit. A regulator can see exact per-instrument repricing error.

### 11.7 Interpolation variants

`DiscountCurve` is fixed to log-linear DF interpolation today. We
generalise via a string field on `CurveSpec`, dispatched at curve
construction time:

| `interp` value          | Method                                 | Use case |
|-------------------------|----------------------------------------|----------|
| `log_linear_df`         | linear in log-DF (current default)     | baseline; piecewise-flat forwards |
| `linear_zero`           | linear in continuously-compounded zero | smoother but non-monotone |
| `monotone_convex`       | Hagan-West                             | smooth, monotone, no negative forwards |
| `tension_spline`        | tension cubic on log-DF                | ultra-smooth forwards, optional |

The `CurveSpec.interp` string is **static** so the dispatch happens at
trace time. All variants implement the same `__call__(dates) -> dfs`
signature, so downstream pricing functions are oblivious to the
choice.

### 11.8 Fixings and partially-seasoned curves

```python
class FixingHistory(eqx.Module):
    """Realised resets keyed by index id and fixing date."""
    indices: dict[str, FixingSeries]    # str → series

class FixingSeries(eqx.Module):
    fixing_dates: Int[Array, " n"]      # ordinals
    fixings:      Float[Array, " n"]
```

The bootstrap instruments take a `FixingHistory` argument. When
computing a residual for a swap whose first reset has already fixed,
the float-leg residual reads from `FixingHistory` rather than
projecting from the curve. Without this, partially-seasoned curves
mis-bootstrap by exactly the amount of the realised vs. forward fixing
gap on the first coupon.

`FixingHistory` lives in `MarketState.fixings` (§3.3) and is passed
through unchanged from the ingestion layer.

### 11.9 Quote-sensitivity Jacobian export

Today, computing `∂DF / ∂quote` requires a closure passed to
`jax.jacobian` — straightforward, but boilerplate. We expose a
convenience wrapper:

```python
def quote_jacobian(
    graph:       CurveGraph,
    instruments: Sequence[BootstrapInstrument],
    *,
    by:          str = "log_df",       # or "df", or "zero_rate"
) -> Float[Array, "n_outputs n_quotes"]:
    """Differentiate the calibrated graph with respect to input quotes.

    Uses optimistix.ImplicitAdjoint internally so the cost is one
    linear solve, independent of Newton iteration count.
    """
```

This is the matrix a trading desk uses to hedge curve risk with
liquid instruments. Cheap with autodiff, expensive with finite
differences. Exposing it as a one-liner removes friction.

### 11.10 Worked spec — USD + EUR + XCCY

End state — what `MarketSpec` can describe once MC-Curves-1/-2/-3 are
landed:

```yaml
# specs/g7_eod.yaml — illustrative
asof_calendar: NY+LON+TARGET

curves:
  USD.SOFR.OIS:
    currency: USD
    interp:   log_linear_df
    pillars:  [1W, 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y]
  USD.SOFR.3M:
    currency: USD
    interp:   monotone_convex
    pillars:  [3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y]
  EUR.ESTR.OIS:
    currency: EUR
    interp:   log_linear_df
    pillars:  [1W, 1M, 3M, 6M, 1Y, 5Y, 10Y, 30Y]
  EUR.EURIBOR.6M:
    currency: EUR
    interp:   monotone_convex
    pillars:  [6M, 1Y, 2Y, 5Y, 10Y, 30Y]

instruments:
  # USD short end + OIS strip
  - {kind: deposit,    id: USD.SOFR.ON,            curves: [USD.SOFR.OIS]}
  - {kind: ois_swap,   id: USD.SOFR.OIS.10Y,       curves: [USD.SOFR.OIS]}

  # USD 3M tenor: futures + IBOR swaps (both curves in residual)
  - {kind: future,     id: USD.SOFR.IMM.U5,        curves: [USD.SOFR.OIS, USD.SOFR.3M], convexity: hull_white}
  - {kind: ibor_swap,  id: USD.SOFR.3M.10Y,        curves: [USD.SOFR.OIS, USD.SOFR.3M]}

  # EUR strip
  - {kind: ois_swap,   id: EUR.ESTR.OIS.10Y,       curves: [EUR.ESTR.OIS]}
  - {kind: ibor_swap,  id: EUR.EURIBOR.6M.10Y,     curves: [EUR.ESTR.OIS, EUR.EURIBOR.6M]}

  # EUR-USD cross-currency (joint solve across both currencies)
  - {kind: fx_forward, id: EURUSD.FWD.6M,          curves: [USD.SOFR.OIS, EUR.ESTR.OIS], fx_spot: EURUSD.SPOT}
  - {kind: xccy_basis_mtm, id: EUR.XCCY.5Y,        curves: [USD.SOFR.OIS, EUR.ESTR.OIS, EUR.EURIBOR.6M], fx_spot: EURUSD.SPOT}

solver:
  topology:    joint
  rtol:        1e-12
  atol:        1e-12
  max_steps:   256
```

The build dispatches a single call to `bootstrap_curve_graph` with
all four `CurveSpec`s and the full instrument list. One Newton solve
returns all four calibrated curves plus a `CurveBuildDiagnostics`
object covering every input quote. The result is partitioned by
identifier prefix into `MarketState.discount_curves` (`*.OIS`) and
`MarketState.forward_curves` (`*.3M`, `*.6M`).

### 11.11 Out of scope here

The following are real production concerns but live in other parts
of the roadmap and are not tackled in this section:

- **Business calendars and date adjustment** — roadmap P1.1.
  Prerequisite: every quote refers to calendar-adjusted dates.
- **Cashflow engine** (stubs, compounding, amortisation) — roadmap
  P1.2. Prerequisite for accurately representing real swap legs.
- **Constrained curves** (forward positivity, calendar-spread
  no-arbitrage) — roadmap §1.1 backlog.
- **MBS prepayment / OAS curves** — roadmap §3.8.
- **Hazard / survival curves** for credit — roadmap §3.4. Will be
  added to the curve graph as additional curves with their own
  instrument types (CDS spreads), reusing the same solver.

---

## 12. Testing strategy

### 11.1 Unit tests (existing pattern)

Each new module gets pytest tests mirroring the layout under `tests/`:

- `tests/test_io/test_quotes.py` — `QuoteSet` round-trips, hashing.
- `tests/test_io/test_adapters/test_csv.py` — adapter contract tests.
- `tests/test_io/test_store/test_local.py` — write/read fidelity.
- `tests/test_workflows/test_build_state.py` — golden snapshot tests.
- `tests/test_workflows/test_mtm.py` — determinism + correctness vs.
  direct pricing kernel calls.

### 11.2 Round-trip tests

The mock adapter enables a full closed-loop test in CI:

```python
def test_full_roundtrip():
    spec = load_spec("specs/test_usd.yaml")
    quotes = MockAdapter(spec=spec).read(asof=date(2024, 6, 14))

    state, report = build_market_state(quotes, spec)
    assert report.errors == []

    # Snapshot round-trip
    ref = LocalFileSnapshotStore(tmp_path).write(state, report)
    state2, report2 = LocalFileSnapshotStore(tmp_path).read(ref)
    assert tree_allclose(state, state2)

    # Re-build determinism
    state3, _ = build_market_state(quotes, spec)
    assert state.snapshot_id == state3.snapshot_id

    # MTM determinism
    portfolio = mock_portfolio()
    r1 = mark_to_market(portfolio, state,  MTMConfig())
    r2 = mark_to_market(portfolio, state2, MTMConfig())
    assert tree_allclose(r1, r2, rtol=0, atol=0)   # bitwise
```

### 11.3 Property-based tests

Using `hypothesis` (already a dev dependency):

- For any valid `MarketSpec`, the build either succeeds or returns a
  `BuildReport` with at least one error — never silent NaN.
- `apply_scenario(state, zero_scenario(...))` is idempotent and
  identity on prices.
- `SnapshotStore.read(write(state)) == state` for all generated
  states.

### 11.4 Performance gates

A small benchmark suite under `tests/test_workflows/test_perf.py`:

- A canonical 5,000-trade portfolio MTM must finish in < 5 s on CPU
  after warm JIT.
- Build of the test USD market must finish in < 2 s.

These prevent latent regressions in the workflow layer that would
surface only in production.

---

## 13. Phased delivery plan

The work is organised into two parallel workstreams: the **production
scaffolding** (Phases 1–4, sequential) and the **multi-curve
framework** (MC-Curves-1 to -3, sequential within itself but
parallelisable against the scaffolding).

### Workstream A — Production scaffolding

#### Phase 1 — `MarketState` and `QuoteSet`
*Estimated effort: ~2 weeks*

- Introduce `valax/market/state.py` with `MarketState`.
- Introduce `valax/io/quotes.py` with `QuoteSet`, `Quote`.
- Migrate all existing pricing functions and tests to accept
  `MarketState` (via lookup by ID) while keeping `MarketData` as a
  deprecated thin wrapper.
- No persistence yet, no workflow yet.
- **Deliverable:** every existing test passes against `MarketState`.

#### Phase 2 — Calibration workflow
*Estimated effort: ~2 weeks*

- Introduce `valax/workflows/spec.py` and `build_state.py`.
- Wire all existing calibrators (curve bootstraps, surface fits) into
  the `@register_calibrator` registry.
- Mock adapter and validation layer.
- `BuildReport` and round-trip tests with synthetic `MarketSpec`.
- **Deliverable:** `build_market_state(quotes, spec)` produces a
  fully populated `MarketState` from a YAML spec, end to end.

#### Phase 3 — Snapshot persistence
*Estimated effort: ~1.5 weeks*

- `SnapshotStore` interface + `LocalFileSnapshotStore`.
- `AuditStore` interface + `ParquetAuditStore`.
- Content hashing and the `snapshot_id` discipline.
- `valax build-state` CLI command.
- **Deliverable:** an EOD build can be persisted, re-loaded, and
  regenerates byte-identical state.

#### Phase 4 — MTM workflow
*Estimated effort: ~2 weeks*

- `Portfolio` container.
- `mark_to_market` driver with greeks and bucket attribution.
- `valax mtm` CLI command.
- `ParquetPortfolioStore`.
- **Deliverable:** end-to-end EOD example —
  `valax build-state` then `valax mtm` — produces a P&L parquet
  that regulators / risk managers can consume.

After Phase 4 is proven, Phase 5+ (risk run, S3 store, gRPC service)
plug into the same skeleton without any further structural change.

### Workstream B — Multi-curve framework

Independent of Workstream A in scheduling but feeds into Phase 2 once
landed (the registered curve calibrator becomes
`bootstrap_curve_graph`). Without these PRs the production stack can
still build single-currency two-curve setups; with them it covers
real desks.

#### MC-Curves-1 — Bootstrap instrument expansion
*Estimated effort: ~1.5 weeks*

- Define `BootstrapInstrument` protocol (§11.3) and migrate the
  existing three quote types onto it. Existing single-curve bootstrap
  keeps working unchanged.
- New instrument types: `OISSwapRate`, `IborSwapRate` (separates
  fixed/float day counts and exposes fixings hooks),
  `MoneyMarketFuture` (with pluggable convexity adjustment),
  `TenorBasisSwap`, `FXSwap` / `FXForward`,
  `CrossCurrencyBasisSwap` (MTM and constant-notional variants),
  `TurnInstrument`.
- `FixingHistory` / `FixingSeries` data types.
- Tests: each new instrument's residual is zero on a hand-built curve
  graph that satisfies its no-arb relation.
- **Deliverable:** every quote type a real curve build needs is
  representable as a `BootstrapInstrument`.

#### MC-Curves-2 — Joint global solver
*Estimated effort: ~2 weeks*

- `CurveSpec` and `CurveGraph` data types (§11.3).
- `bootstrap_curve_graph` driving one Newton solve over the
  concatenated log-DF state vector (§11.4).
- Quote-sensitivity Jacobian helper `quote_jacobian` (§11.9).
- Replace `bootstrap_multi_curve` internals with a graph call that
  solves a two-curve subgraph; the public function keeps its
  signature for one minor release with a deprecation warning.
- Tests: USD 3M (futures + IBOR swap) reproduces today's dual-curve
  result; cross-currency basis swap closes the EURUSD curve graph
  to micro-bps; gradients flow through `bootstrap_curve_graph` via
  `ImplicitAdjoint`.
- **Deliverable:** `bootstrap_curve_graph(specs, instruments)`
  bootstraps tenor-basis and cross-currency systems jointly.

#### MC-Curves-3 — Diagnostics, interpolation variants, fixings hookup
*Estimated effort: ~1.5 weeks*

- `CurveBuildDiagnostics` pytree with per-instrument fitted-vs-quoted,
  RMSE, max error, Newton steps, Jacobian condition number.
- Pluggable interpolation: `log_linear_df` (default), `linear_zero`,
  `monotone_convex` (Hagan-West), optionally `tension_spline`.
- `FixingHistory` plumbed through to all relevant instrument
  residuals; tests covering partially-seasoned curves.
- Once Phase 2 of Workstream A is in flight: register the graph
  calibrator with `@register_calibrator(kind="curve_graph",
  method="joint_newton")` and surface the diagnostics into
  `BuildReport`.
- **Deliverable:** the worked example in §11.10 runs end-to-end from
  YAML spec to calibrated `MarketState` with full diagnostics
  attached to the build report.

### Sequencing summary

```
Workstream A:   Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4
                              ▲
                              │ (registers graph calibrator)
                              │
Workstream B:   MC-Curves-1 ──► MC-Curves-2 ──► MC-Curves-3
```

MC-Curves-1 and -2 can begin immediately; they touch only
`valax/curves/`. MC-Curves-3 must overlap with Phase 2 because that
is where `BuildReport` and `@register_calibrator` are introduced.
End-to-end coverage of the §11.10 worked example requires both
workstreams complete — nominally ~6 weeks of single-engineer work
each, executable in parallel.

---

## 14. Open questions

The following are flagged for discussion before Phase 1 / MC-Curves-1
starts.

### Production scaffolding

1. **`MarketData` deprecation strategy.** Hard rename in 0.2.0 vs.
   long-lived alias? Recommend: alias for one minor release, hard
   rename in the next.
2. **Identifier alphabet — do we standardise on ISO codes
   (`USD.SOFR.OIS`) or vendor identifiers (`SOFRRATE Index`)?**
   Recommend: ISO-like internal names with vendor-specific aliases
   in the adapter layer.
3. **Where does the CSA / collateral curve live?** Probably under
   `discount_curves` keyed `<ccy>.<csa_label>.OIS`, but worth
   confirming once XVA is on the table.
4. **Float precision for pricing kernels.** `float64` is the
   library default and required for curve work; do we allow
   surface evaluation in `float32` on GPU for speed? Recommend:
   no, keep `float64` everywhere until a benchmark forces the issue.
5. **Spec hash stability across YAML formatters.** Two YAMLs that
   differ only in whitespace must hash identically. We'll compute
   the hash on the *parsed* canonical form, not the source bytes.
6. **CI requirements before Phase 1 ships.** P1.3 in the roadmap
   (CI/CD pipeline) is currently unstarted; we should land at least
   a basic GitHub Actions matrix (lint + test + type-check) before
   merging Phase 1, since it touches every pricing path.

### Multi-curve framework

7. **Instrument abstraction.** Adopt the `BootstrapInstrument`
   protocol from §11.3 with each instrument owning its `residual`
   method, or keep central `isinstance` dispatch and grow the
   `_compute_residuals` switch? Recommend: protocol — adding a new
   instrument should never require editing the solver.
8. **Curve identifier scheme — lock now or evolve.** §11.10 assumes
   `<ccy>.<index>.<tenor>` as the identifier alphabet; this is also
   what `MarketState.discount_curves` etc. partition on. Recommend:
   freeze the scheme before MC-Curves-2 lands so the partitioning
   logic in `build_market_state` doesn't need rework.
9. **Convexity adjustment policy for futures.** Ship MC-Curves-1
   with a `constant_convexity_adj(bps)` plug-in only and defer the
   Hull-White-derived adjustment until the short-rate model is
   wired into the curve build, or do both in one go? Recommend:
   constant first — desk-supplied bps is the default in many
   shops anyway, and HW-derived adjustment can be added without
   touching the instrument's interface.
10. **Single-curve bootstrap retention.** Once
    `bootstrap_curve_graph` exists, do we keep
    `bootstrap_sequential` and `bootstrap_simultaneous` as
    user-facing APIs, or have them become thin wrappers over the
    graph solver? Recommend: keep the simple single-curve API for
    pedagogy and tests; reroute internals to the graph solver in a
    follow-up cleanup once it has shipped and stabilised.
11. **Default interpolation per curve type.** Today: log-linear DF
    for everything. After MC-Curves-3, what should the default per
    curve type be? Recommend: log-linear DF for OIS / discount
    curves (matches the log-DF Newton iteration); monotone-convex
    for forward / IBOR curves (smooth forward extraction is the
    point of those curves).

---

## 15. Summary

VALAX today is a clean pure-functional pricing kernel; this design
keeps that core untouched and adds an imperative shell that turns it
into a deployable valuation tool. Five new module groups (`io`,
`workflows`, `cli`, plus extensions to `market`, `curves`, and
`portfolio`) provide ingestion, calibration orchestration,
persistence, mark-to-market, and a graph-shaped multi-curve solver
covering tenor-basis and cross-currency systems.

Two parallel workstreams take us from "library" to "EOD batch job
that produces a reproducible P&L file":

- **Workstream A — Production scaffolding** (Phases 1–4, ~7.5 weeks
  sequential): `MarketState`, build workflow, snapshot persistence,
  MTM driver.
- **Workstream B — Multi-curve framework** (MC-Curves-1 to -3, ~5
  weeks sequential, parallelisable against A): bootstrap instrument
  expansion, joint global solver, diagnostics + interpolation
  variants + fixings.

The structural property we are buying with this design is
**determinism**: same quotes + same spec + same code → same snapshot
ID → same prices, every time. Everything else — gRPC, XVA, FRTB,
real-time — plugs into this skeleton without reshaping it.
