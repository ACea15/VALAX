"""Deprecated module: renamed to :mod:`valax.risk.sensitivities`.

VALAX now calls base-point bucketed Greeks *sensitivities* (the FRTB SA /
ISDA SIMM term) and reserves *ladder* for revaluation at shifted levels.
This shim keeps old imports working, with a :class:`DeprecationWarning`,
until the removal scheduled on the roadmap.

========================  ====================================
Old name                  New name
========================  ====================================
``SensitivityLadder``     ``FlatSensitivities``
``compute_ladder``        ``compute_flat_sensitivities``
========================  ====================================
"""

import warnings

from valax.risk.sensitivities import (  # noqa: F401  (re-exports)
    FlatSensitivities,
    WaterfallPnL,
    compute_flat_sensitivities,
    waterfall_pnl,
    waterfall_pnl_report,
)

warnings.warn(
    "valax.risk.ladders is deprecated; import from valax.risk.sensitivities "
    "(SensitivityLadder -> FlatSensitivities, compute_ladder -> "
    "compute_flat_sensitivities).",
    DeprecationWarning,
    stacklevel=2,
)

SensitivityLadder = FlatSensitivities
compute_ladder = compute_flat_sensitivities
