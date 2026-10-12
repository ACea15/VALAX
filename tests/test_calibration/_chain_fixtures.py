"""Shared calibration-chain fixtures (three market tiers).

* Tier 0 — constant parameters: quoted spot / dividend / rate / vol installed
  directly (``market_inputs_node``).
* Tier 1 — single curve: a three-pillar OIS bootstrap under the sentinel id,
  plus a quoted spot and dividend and a SABR smile on the curve-implied
  forwards.
* Tier 2 — multi curve: OIS discount + 3M projection solved jointly.
"""

import jax.numpy as jnp

from valax.calibration import (
    CalibrationChain,
    curve_graph_node,
    equity_forward,
    instrument_quotes,
    market_inputs_node,
    sabr_smile_node,
)
from valax.curves import CurveSpec, DepositRate, IborSwapRate, OISSwapRate
from valax.dates.daycounts import ymd_to_ordinal
from valax.market import QuoteSet

REF = ymd_to_ordinal(2025, 1, 1)
DEFAULT = "_default_"


def d(y, m, dd):
    return int(ymd_to_ordinal(y, m, dd))


# ── Tier 0 ───────────────────────────────────────────────────────────

FLAT_QUOTES = {"SPX.SPOT": 100.0, "SPX.DIV": 0.01, "USD.RATE": 0.05, "SPX.VOL": 0.2}


def flat_chain():
    node = market_inputs_node(
        spots={DEFAULT: "SPX.SPOT"},
        dividends={DEFAULT: "SPX.DIV"},
        flat_rates={DEFAULT: "USD.RATE"},
        flat_vols={DEFAULT: "SPX.VOL"},
    )
    return CalibrationChain(nodes=(node,)), QuoteSet.from_dict(FLAT_QUOTES)


# ── Tier 1: single curve + spot + SABR smile ─────────────────────────

SC_PILLARS = jnp.array([d(2026, 1, 1), d(2027, 1, 1), d(2030, 1, 1)], jnp.int32)
SC_KEYS = ("USD.OIS.1Y", "USD.OIS.2Y", "USD.OIS.5Y")


def sc_instruments(curve_id=DEFAULT, rates=(0.045, 0.047, 0.050)):
    return [
        DepositRate(
            start_date=REF, end_date=int(SC_PILLARS[0]),
            rate=jnp.asarray(rates[0]), day_count="act_365",
            curves_touched=(curve_id,),
        ),
        OISSwapRate(
            start_date=REF,
            fixed_dates=jnp.array([d(2026, 1, 1), d(2027, 1, 1)], jnp.int32),
            rate=jnp.asarray(rates[1]), day_count="act_365",
            curves_touched=(curve_id,),
        ),
        OISSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(
                [d(2026, 1, 1), d(2027, 1, 1), d(2028, 1, 1),
                 d(2029, 1, 1), d(2030, 1, 1)], jnp.int32,
            ),
            rate=jnp.asarray(rates[2]), day_count="act_365",
            curves_touched=(curve_id,),
        ),
    ]


def sc_curve_node(curve_id=DEFAULT):
    spec = CurveSpec(
        curve_id=curve_id, currency="USD",
        pillar_dates=SC_PILLARS, day_count="act_365",
    )
    return curve_graph_node(
        "usd_ois", REF, [spec], sc_instruments(curve_id), SC_KEYS
    )


EXPIRIES = jnp.array([0.5, 1.0])
STRIKES = jnp.array([[90.0, 100.0, 110.0, 120.0], [85.0, 100.0, 115.0, 130.0]])
VOL_KEYS = tuple(
    tuple(f"SPX.VOL.{t}.{int(k)}" for k in row)
    for t, row in zip(("6M", "1Y"), STRIKES.tolist())
)
VOLS = ((0.24, 0.21, 0.195, 0.19), (0.25, 0.215, 0.195, 0.19))


def equity_chain():
    spot = market_inputs_node(
        "spx_inputs", spots={DEFAULT: "SPX.SPOT"}, dividends={DEFAULT: "SPX.DIV"}
    )
    smile = sabr_smile_node(
        DEFAULT,
        expiries=EXPIRIES,
        strikes=STRIKES,
        quote_keys=VOL_KEYS,
        forward_fn=equity_forward(),
        beta_key="SPX.SABR.BETA",
        name="spx_smile",
    )
    chain = CalibrationChain(nodes=(sc_curve_node(), spot, smile))
    vols = {
        k: v for row, vr in zip(VOL_KEYS, VOLS) for k, v in zip(row, vr)
    }
    quotes = instrument_quotes(sc_instruments(), SC_KEYS).merge(
        QuoteSet.from_dict({"SPX.SPOT": 100.0, "SPX.DIV": 0.01, **vols})
    )
    fixed = QuoteSet.from_dict({"SPX.SABR.BETA": 0.5})
    return chain, quotes, fixed


# ── Tier 2: OIS discount + 3M projection, solved jointly ─────────────

OIS = "USD.SOFR.OIS"
P3M = "USD.TERM.3M"
DC_PILLARS = jnp.array([d(2026, 1, 1), d(2028, 1, 1)], jnp.int32)
ANN_1Y = jnp.array([d(2026, 1, 1)], jnp.int32)
ANN_3Y = jnp.array([d(2026, 1, 1), d(2027, 1, 1), d(2028, 1, 1)], jnp.int32)
FIX_1Y = jnp.array([REF], jnp.int32)
FIX_3Y = jnp.array([REF, d(2026, 1, 1), d(2027, 1, 1)], jnp.int32)
DC_KEYS = ("USD.OIS.1Y", "USD.OIS.3Y", "USD.IRS3M.1Y", "USD.IRS3M.3Y")
DC_RATES = (0.045, 0.048, 0.0475, 0.0505)


def dc_specs():
    return [
        CurveSpec(curve_id=OIS, currency="USD", pillar_dates=DC_PILLARS,
                  day_count="act_365"),
        CurveSpec(curve_id=P3M, currency="USD", pillar_dates=DC_PILLARS,
                  day_count="act_365"),
    ]


def dc_instruments(q=DC_RATES):
    return [
        DepositRate(
            start_date=REF, end_date=int(DC_PILLARS[0]),
            rate=jnp.asarray(q[0]), day_count="act_365", curves_touched=(OIS,),
        ),
        OISSwapRate(
            start_date=REF, fixed_dates=ANN_3Y, rate=jnp.asarray(q[1]),
            day_count="act_365", curves_touched=(OIS,),
        ),
        IborSwapRate(
            start_date=REF, fixed_dates=ANN_1Y, float_dates=ANN_1Y,
            fixing_dates=FIX_1Y, rate=jnp.asarray(q[2]),
            fixed_day_count="act_365", float_day_count="act_365",
            curves_touched=(OIS, P3M), index_id=P3M,
        ),
        IborSwapRate(
            start_date=REF, fixed_dates=ANN_3Y, float_dates=ANN_3Y,
            fixing_dates=FIX_3Y, rate=jnp.asarray(q[3]),
            fixed_day_count="act_365", float_day_count="act_365",
            curves_touched=(OIS, P3M), index_id=P3M,
        ),
    ]


def dual_curve_chain():
    node = curve_graph_node("usd_curves", REF, dc_specs(), dc_instruments(), DC_KEYS)
    return (
        CalibrationChain(nodes=(node,)),
        instrument_quotes(dc_instruments(), DC_KEYS),
    )
