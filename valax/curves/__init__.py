"""Yield curve construction, interpolation, and bootstrapping."""

from valax.curves.discount import DiscountCurve, forward_rate, zero_rate
from valax.curves.instruments import DepositRate, FRA, SwapRate
from valax.curves.bootstrap import bootstrap_sequential, bootstrap_simultaneous
from valax.curves.multi_curve import MultiCurveSet, bootstrap_multi_curve
from valax.curves.inflation import (
    InflationCurve,
    forward_cpi,
    zc_inflation_rate,
    yoy_forward_rate,
    from_zc_rates,
)
