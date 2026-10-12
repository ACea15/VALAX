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
