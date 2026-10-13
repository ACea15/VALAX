# From Quotes to Risk

One workflow covers every market, from flat-parameter Black-Scholes to
multi-curve with a calibrated smile and model:

1. load **quotes** into a `QuoteSet`;
2. describe the calibration as a **chain** of nodes and run `calibrate_market`;
3. optionally **calibrate a model** on the market with `calibrate_model`;
4. **price** with `price(instrument, model, market)`;
5. get **quote risk** by the adjoint sweep, and check it against the
   independent autodiff path.

The maths is on [Calibration Chains](../theory/calibration-chains.md) and the
design on [Calibration, Pricing & Risk Architecture](../architecture/calibration-pricing-risk.md).
Runnable end-to-end versions on synthetic quotes generated from a known true
market are `examples/10_unified_market_tiers.py` to
`examples/13_unified_quote_scenarios.py`, and
`examples/14_unified_rates_per_trade_model.py` for the rates per-trade
workflow below; the quote generator is described in
[Synthetic market data](synthetic_market.md#synthetic-quote-universes).
Every snippet below is JIT-safe: the calibration is traceable, so
`eqx.filter_jit`, `jax.grad` and `jax.vmap` all apply.

## Tier 0 — constant parameters

A flat market is a chain with one explicit level that installs the quoted
spot, dividend, rate and vol. Quote risk is then exactly the textbook
Black-Scholes delta, rho, vega and dividend sensitivity.

```python
import jax.numpy as jnp
from valax.calibration import CalibrationChain, calibrate_market, market_inputs_node
from valax.core.pricing import price
from valax.greeks import quote_risk_adjoint, quote_risk_autodiff, reconcile_quote_risk
from valax.instruments.options import EuropeanOption
from valax.market import QuoteSet
from valax.models.black_scholes import BlackScholesModel

inputs = market_inputs_node(
    spots={"_default_": "SPX.SPOT"},
    dividends={"_default_": "SPX.DIV"},
    flat_rates={"_default_": "USD.RATE"},
    flat_vols={"_default_": "SPX.VOL"},
)
chain = CalibrationChain(nodes=(inputs,))
quotes = QuoteSet.from_dict(
    {"SPX.SPOT": 100.0, "SPX.DIV": 0.01, "USD.RATE": 0.05, "SPX.VOL": 0.2}
)

calibrated = calibrate_market(chain, quotes)
option = EuropeanOption(strike=jnp.asarray(105.0), expiry=jnp.asarray(1.0))
model = BlackScholesModel()          # every field bound from the market

pv = price(option, model, calibrated.market).price
risk = quote_risk_adjoint(option, model, calibrated)
risk.quotes["SPX.SPOT"]              # delta; also USD.RATE (rho), SPX.VOL (vega)
```

Without a chain at all, `MarketState.flat(0.05, spot=100.0, vol=0.2,
dividend=0.01)` builds the same market directly; the model may also carry its
own constants (`BlackScholesModel(vol=0.2, rate=0.05, dividend=0.01)`), in
which case they are model parameters rather than market quotes and their
sensitivities come from `model_gradient`.

## Tier 1 — a bootstrapped curve, a spot and a smile

Replace the flat rate with an OIS bootstrap and the flat vol with SABR smiles
fitted on the **curve-implied forwards**. Nothing else changes.

```python
from valax.calibration import (
    curve_graph_node, equity_forward, instrument_quotes, sabr_smile_node,
)
from valax.curves import CurveSpec, DepositRate, OISSwapRate
from valax.dates.daycounts import ymd_to_ordinal

ref = ymd_to_ordinal(2025, 1, 1)
d = lambda y, m, dd: int(ymd_to_ordinal(y, m, dd))
pillars = jnp.array([d(2026, 1, 1), d(2027, 1, 1), d(2030, 1, 1)], jnp.int32)
spec = CurveSpec(curve_id="_default_", currency="USD", pillar_dates=pillars)
annual = lambda n: jnp.array([d(2026 + i, 1, 1) for i in range(n)], jnp.int32)
ois = [
    DepositRate(start_date=ref, end_date=int(pillars[0]), rate=jnp.asarray(0.045),
                curves_touched=("_default_",)),
    OISSwapRate(start_date=ref, fixed_dates=annual(2), rate=jnp.asarray(0.047),
                curves_touched=("_default_",)),
    OISSwapRate(start_date=ref, fixed_dates=annual(5), rate=jnp.asarray(0.050),
                curves_touched=("_default_",)),
]
curve_keys = ("USD.OIS.1Y", "USD.OIS.2Y", "USD.OIS.5Y")

expiries = jnp.array([0.5, 1.0])
strikes = jnp.array([[90.0, 100.0, 110.0, 120.0], [85.0, 100.0, 115.0, 130.0]])
vol_keys = [[f"SPX.VOL.{t}.{int(k)}" for k in row]
            for t, row in zip(("6M", "1Y"), strikes.tolist())]
vols = [[0.24, 0.21, 0.195, 0.19], [0.25, 0.215, 0.195, 0.19]]

chain = CalibrationChain(nodes=(
    curve_graph_node("usd_ois", ref, [spec], ois, curve_keys),
    market_inputs_node("spx", spots={"_default_": "SPX.SPOT"},
                       dividends={"_default_": "SPX.DIV"}),
    sabr_smile_node("_default_", expiries=expiries, strikes=strikes,
                    quote_keys=vol_keys, forward_fn=equity_forward(),
                    beta_key="SPX.SABR.BETA", name="spx_smile"),
))
quotes = instrument_quotes(ois, curve_keys).merge(QuoteSet.from_dict({
    "SPX.SPOT": 100.0, "SPX.DIV": 0.01,
    **{k: v for kr, vr in zip(vol_keys, vols) for k, v in zip(kr, vr)},
}))
fixed = QuoteSet.from_dict({"SPX.SABR.BETA": 0.5})   # held-fixed input p

calibrated = calibrate_market(chain, quotes, fixed, strict=True)
option = EuropeanOption(strike=jnp.asarray(105.0), expiry=jnp.asarray(0.75))

adjoint = quote_risk_adjoint(option, model, calibrated)
autodiff = quote_risk_autodiff(option, model, chain, quotes, fixed)
assert reconcile_quote_risk(adjoint, autodiff, rtol=1e-6).ok
adjoint.quotes["USD.OIS.1Y"], adjoint.fixed["SPX.SABR.BETA"]
```

The 1Y OIS delta includes the effect of the curve on the **smile's anchor**:
the smile level hands its forward dependence back to the curve level during
the sweep. The beta sensitivity is reported because beta was a fixed input.

A single curve is just one `CurveSpec`. For **multi-curve**, pass several
specs (OIS discount, IBOR projection) and the instruments that couple them
(`IborSwapRate` with `curves_touched=(ois_id, proj_id)`) to the same
`curve_graph_node`; trades select curves with `discount_id` / `forward_id`.
See [Yield Curves & Bootstrapping](curves.md).

## Tier 2 — a calibrated model

A pricing model is calibrated **on** the market, as a separate input. Any model
with a registered pricer can be fitted through `price` with `price_fit_node`;
here Heston is fitted once for the underlying to the 1Y smile quotes, with
mean reversion held fixed as an input.

```python
from valax.calibration import calibrate_model, price_fit_node
from valax.calibration.transforms import HESTON_TRANSFORMS
from valax.models.heston import HestonModel

basket = [EuropeanOption(strike=jnp.asarray(k), expiry=expiries[1])
          for k in strikes[1].tolist()]
heston = price_fit_node(
    "spx_heston",
    template=HestonModel(v0=jnp.asarray(0.045), kappa=jnp.asarray(1.5),
                         theta=jnp.asarray(0.045), xi=jnp.asarray(0.6),
                         rho=jnp.asarray(-0.6)),
    transforms={k: HESTON_TRANSFORMS[k] for k in ("v0", "xi", "rho")},
    basket=basket,
    quote_keys=vol_keys[1],
    quote_form="lognormal_vol",            # quotes are BS vols
    fixed_fields={"kappa": "SPX.HESTON.KAPPA"},
)
fixed = fixed.merge(QuoteSet.from_dict({"SPX.HESTON.KAPPA": 1.5}))
fit = calibrate_model(heston, quotes, calibrated.market, fixed, strict=True)

trade = EuropeanOption(strike=jnp.asarray(105.0), expiry=jnp.asarray(1.0))
risk = quote_risk_adjoint(trade, None, calibrated, model_result=fit)
risk.quotes[vol_keys[1][1]], risk.fixed["SPX.HESTON.KAPPA"]
```

The same factory with an `install_fn` (e.g.
`valax.calibration.install_surface("SPX.VOL")`) calibrates a **market**
object instead — a premium-to-vol quote conversion, or a model-implied
surface — and joins the chain like any other level.

## Rates — per-trade model calibration

A rates exotic calibrates its model **per trade**, to a basket chosen by the
trade, off a market that is calibrated once: curves → swaption cube (on the
curve-implied forwards) → Hull-White fitted to the trade's co-terminal
swaptions → Bermudan priced through `price` → quote risk by one sweep.

- The vol levels are chain nodes on the upstream curves: `swaption_cube_node`
  recomputes each cell's forward and annuity from the curves, so the cube
  carries curve risk; `caplet_strip_node` → `optionlet_sabr_node` does the
  same for caps. Every cell / bucket is its own fit and a failure is named
  (`"5Yx10Y"`); see [Calibration § Solvers](calibration.md#solvers).
- `swaption_basket_node` is a model-scope level: targets are the basket
  swaptions priced off the calibrated cube and curves, model prices go through
  `price` (by default the closed form — Jamshidian for Hull-White,
  Brigo-Mercurio for G2++), and the fitted model is curve-free —
  `price` binds its curve from the market.
- Zero-valued **target shifts**, one fixed input per basket instrument, report
  `∂V/∂T_k`: the trade's vega profile across its basket.

```python
import valax
from valax.calibration import coterminal_swaptions, positive, swaption_basket_node
from valax.instruments.rates import BermudanSwaption
from valax.market import (
    RatesVolSpec, RatesWorldSpec, SeedRegistry, default_config, rates_vol_snapshot,
)
from valax.models.hull_white import HullWhiteModel
from valax.pricing.pde.config import PDEConfig

# 1. A synthetic rates market: curves + a SABR swaption cube, noisy quotes.
vol_spec = RatesVolSpec(expiry_years=(1, 2, 5), tenor_years=(2, 5), cap_years=(1,))
snap, world = rates_vol_snapshot(
    SeedRegistry(master_seed=7, library_version=valax.__version__),
    default_config(), RatesWorldSpec(pillar_years=(1, 2, 3, 5, 7, 10)), vol_spec,
    bp_noise=0.5, vol_bp_noise=0.5,
)
chain = CalibrationChain(nodes=snap.chain.nodes[:2])   # curves -> cube
quotes = snap.quotes.select(chain.quote_keys)

# 2. The trade and its basket: a 2Y-into-5Y Bermudan payer, co-terminals.
ref = int(world.rates.reference_date)
trade = BermudanSwaption(
    exercise_dates=jnp.array([ref + 365 * k for k in (2, 3, 4)], jnp.int32),
    fixed_dates=jnp.array([ref + 365 * k for k in range(3, 8)], jnp.int32),
    strike=jnp.asarray(0.03), notional=jnp.asarray(1.0),
    day_count=world.rates.day_count,
    discount_id=world.discount_id, forward_id=world.forward_id,
)
basket = coterminal_swaptions(trade)
targets = tuple(f"HW.TARGET.{i}" for i in range(len(basket)))
hw = swaption_basket_node(
    "hw", template=HullWhiteModel(mean_reversion=jnp.asarray(0.03),
                                  volatility=jnp.asarray(0.008)),
    basket=basket, cube_id=vol_spec.cube_id,
    transforms={"volatility": positive()},
    fixed_fields={"mean_reversion": "HW.MEAN_REVERSION"},
    target_keys=targets,                     # zero shifts -> vega profile
)
fixed = snap.fixed.merge(QuoteSet.from_dict(
    {"HW.MEAN_REVERSION": 0.03, **{k: 0.0 for k in targets}}))

# 3. Market once, model per trade, Bermudan through price().
calibrated = calibrate_market(chain, quotes, fixed, strict=True)
fit = calibrate_model(hw, quotes, calibrated.market, fixed, strict=True)
pde = PDEConfig(n_spot=100, n_time=60)
pv = price(trade, fit.value, calibrated.market, method="pde", config=pde).price

# 4. Quote risk: the model level's λ_m swept through the cube and the curves.
risk = quote_risk_adjoint(trade, None, calibrated, model_result=fit,
                          method="pde", config=pde)
check = quote_risk_autodiff(trade, None, chain, quotes, fixed, model_node=hw,
                            method="pde", config=pde)
assert reconcile_quote_risk(risk, check, rtol=1e-6, atol=1e-10).ok
profile = [risk.fixed[k] for k in targets]   # ∂V/∂T_k across the basket
risk.fixed["HW.MEAN_REVERSION"]              # held-fixed mean reversion
```

`quote_risk_adjoint(..., model_result=fit)` is `chain_adjoint` with the model
level in `model_levels`: it solves the basket level's transposed system for
`λ_m`, then hands the upstream cotangent to the cube level and the curve level,
so `risk.quotes` holds both the trade's vega to every cube quote and its delta
to every curve quote (including the curve paths through the basket's targets).

**Many trades.** Baskets may be traced, so same-shaped per-trade baskets
calibrate in one compiled program; the curve-free fitted models stack:

```python
import equinox as eqx
import jax

hw_template = HullWhiteModel(mean_reversion=jnp.asarray(0.03),
                             volatility=jnp.asarray(0.008))

def fit_one(basket):
    node = swaption_basket_node(
        "hw", template=hw_template, basket=basket, cube_id=vol_spec.cube_id,
        transforms={"volatility": positive()},
        fixed_fields={"mean_reversion": "HW.MEAN_REVERSION"}, target_keys=targets,
    )
    return node.calibrate(QuoteSet(), calibrated.market, fixed)

trades = [trade, eqx.tree_at(lambda t: t.strike, trade, jnp.asarray(0.035))]
baskets = [coterminal_swaptions(t) for t in trades]       # same shapes
stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *baskets)
batch = eqx.filter_vmap(fit_one)(stacked)
batch.diagnostics.converged                                # one flag per trade

# Per-trade conditioning, inside the vmap; a single jitted fit keeps its
# diagnostics too (the basket travels in LevelInputs.data, not a closure).
kappas = eqx.filter_vmap(lambda b: fit_one(b).condition_number())(stacked)
eqx.filter_jit(fit_one)(baskets[0]).condition_number()
```

**Other models, and which engine fits the basket.** The same node fits G2++
and Cheyette. G2++ prices its basket by its own closed form by default
(Brigo-Mercurio, `price`'s auto route for `Swaption × G2PPModel`) but needs a
basket beyond the co-terminals — add the short-tenor swaption at the first
exercise — to separate its two factors; Cheyette has no closed form and fits
on the frozen-`y` PDE, single-curve only. Passing `method="pde", config=...`
fits the basket *on the Bermudan's own grid* instead: the parameters become
grid-bound, but the grid's Bermudan error largely cancels against its
basket's. The node records the choice as provenance (`exact`; `discretised`
with the grid; `modelling_bias` for Cheyette, which `check_engine` binds to
that grid). [Example 18](../examples.md#per-trade-g2-and-cheyette) compares
the two G2++ fits against a converged reference.

Example 14 runs the whole workflow self-checking, including the identity that
the total cube vega equals the summed basket profile. The maths of the
per-trade level and the target-shift trick is
[Calibration Chains § I.10](../theory/calibration-chains.md#i10-per-trade-model-levels-and-the-vega-profile);
the design decisions are
[D11–D12](../architecture/calibration-pricing-risk.md#3-decisions).

## Book risk in one sweep

The chain adjoint is linear in its seed, so a book's quote risk costs one
sweep: add the trades' market cotangents and call `chain_adjoint` once.

```python
from valax.greeks import chain_adjoint, market_gradient
from valax.greeks.quote_risk import tree_add

book = [option, EuropeanOption(strike=jnp.asarray(95.0), expiry=jnp.asarray(0.5),
                               is_call=False)]
book_bar = None
for t in book:
    book_bar = tree_add(book_bar, market_gradient(t, model, calibrated.market))
book_risk = chain_adjoint(calibrated, book_bar)
```

## Convergence and settings

Every level reports a traceable `converged` flag, residuals and iteration
count; `strict=True` (or `.check()`) raises eagerly. Solver algorithm and
tolerances are a `SolverSettings` passed to the node factory:

```python
from valax.calibration import SolverSettings

loose = SolverSettings(method="levenberg_marquardt", rtol=1e-6, atol=1e-6, max_steps=100)
node = sabr_smile_node("_default_", expiries=expiries, strikes=strikes,
                       quote_keys=vol_keys, forward_fn=equity_forward(),
                       settings=loose)
calibrated.result("spx_smile").diagnostics.converged
calibrated.result("usd_ois").condition_number()
```

Non-convergence invalidates **both** risk paths (the implicit function theorem
needs a solution); the adjoint path additionally reports each level's
stationarity residual in `QuoteRisk.level_residuals`.
