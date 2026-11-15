"""Deprecated module: renamed to :mod:`valax.risk.state_sensitivities`.

VALAX now calls base-point bucketed Greeks *sensitivities* (the FRTB SA /
ISDA SIMM term) and reserves *ladder* for revaluation at shifted levels.
This shim keeps old imports working, with a :class:`DeprecationWarning`,
until the removal scheduled on the roadmap.

==================================  ======================================
Old name                            New name
==================================  ======================================
``KeyedLadder``                     ``KeyedSensitivities``
``keyed_ladder``                    ``keyed_sensitivities``
``KeyedLadder.curve_ladder``        ``KeyedSensitivities.key_rate_dv01``
``ladder=`` keyword                 ``sensitivities=``
==================================  ======================================
"""

import warnings

from valax.risk.state_sensitivities import (  # noqa: F401  (re-exports)
    BASIS_POINT,
    KeyedSensitivities,
    KeyedWaterfall,
    keyed_component_var,
    keyed_parametric_var,
    keyed_sensitivities,
    keyed_waterfall_pnl,
    keyed_waterfall_report,
)

warnings.warn(
    "valax.risk.state_ladders is deprecated; import from "
    "valax.risk.state_sensitivities (KeyedLadder -> KeyedSensitivities, "
    "keyed_ladder -> keyed_sensitivities).",
    DeprecationWarning,
    stacklevel=2,
)

KeyedLadder = KeyedSensitivities
keyed_ladder = keyed_sensitivities
