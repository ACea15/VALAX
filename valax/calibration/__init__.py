"""Calibration: quotes → calibrated market → calibrated model.

The unified layer (``settings``, ``level``, ``result``, ``chain``, ``nodes``)
expresses every calibration as a level ``R(theta; quotes, upstream, fixed) = 0``
returning a :class:`CalibrationResult`, and composes levels into a
:class:`CalibrationChain`. The per-model calibrators below predate it and are
being migrated onto it.
"""

from valax.calibration.settings import (
    EXACT_SETTINGS,
    FIT_SETTINGS,
    PROBLEMS,
    SolverSettings,
    make_solver,
    run_solver,
)
from valax.calibration.level import (
    ILL_CONDITIONED_THRESHOLD,
    CalibrationLevel,
    LevelAdjoint,
    LevelInputs,
    condition_alert,
    level_adjoint,
    level_condition_number,
)
from valax.calibration.result import (
    CLOSED_FORM_PROVENANCE,
    ENGINE_KINDS,
    LEVEL_KINDS,
    CalibrationDiagnostics,
    CalibrationError,
    CalibrationProvenance,
    CalibrationResult,
    check_engine,
    diagnostics_from_solution,
    engine_provenance,
    explicit_diagnostics,
    fixed_point_diagnostics,
)
from valax.calibration.solve import (
    explicit_level,
    period_label,
    period_labels,
    solve_batched_level,
    solve_level,
    stack_slices,
    triangular_root_level,
)

from valax.calibration.transforms import (
    TransformSpec,
    positive,
    bounded,
    unit_interval,
    correlation,
    free,
    SABR_TRANSFORMS,
    HESTON_TRANSFORMS,
    G2PP_TRANSFORMS,
    CHEYETTE_TRANSFORMS,
    CHEYETTE_2F_TRANSFORMS,
    model_to_unconstrained,
    unconstrained_to_model,
)
from valax.calibration.sabr import (
    SABR_FIXED_BETA_TRANSFORMS,
    calibrate_sabr,
    sabr_batch_level,
    sabr_slice_inputs,
    sabr_slice_residual_fn,
    sabr_vol_fn,
    stacked_sabr_inputs,
)
from valax.calibration.basket import BASKET_SETTINGS, basket_fit_level
from valax.calibration.hull_white import (
    HULL_WHITE_TRANSFORMS,
    JAMSHIDIAN_PROVENANCE,
    calibrate_hull_white,
    hw_swaption_prices,
    swaption_prices_from_vols,
)
from valax.calibration.g2pp import (
    calibrate_g2pp,
    g2pp_swaption_prices,
)
from valax.calibration.cheyette import (
    CHEYETTE_SETTINGS,
    calibrate_cheyette,
    calibrate_cheyette_2f,
    cheyette_pde_prices,
    cheyette_provenance,
)
from valax.calibration.heston import calibrate_heston
from valax.calibration.slv import (
    SLV_SETTINGS,
    calibrate_slv,
    calibrate_slv_leverage,
)

# The chain layer depends on valax.core / valax.market / valax.surfaces, and
# valax.core itself imports this package (via valax.surfaces), so its symbols
# are resolved lazily on first access (PEP 562) instead of at import time.
_LAZY = {
    "valax.calibration.chain": (
        "CalibratedMarket",
        "CalibrationChain",
        "CalibrationNode",
        "calibrate_market",
        "calibrate_model",
        "empty_market",
        "install_curves",
        "install_market_fields",
        "install_surface",
    ),
    "valax.calibration.nodes": (
        "QUOTE_FORMS",
        "curve_graph_node",
        "equity_forward",
        "instrument_quotes",
        "market_inputs_node",
        "price_fit_node",
        "sabr_smile_node",
        "swaption_cube_node",
        "caplet_strip_node",
        "optionlet_sabr_node",
        "coterminal_swaptions",
        "swaption_basket_node",
        "BASKET_MODES",
    ),
}
_LAZY_INDEX = {name: mod for mod, names in _LAZY.items() for name in names}


def __getattr__(name: str):
    """Resolve chain-layer symbols on first access."""
    if name in _LAZY_INDEX:
        import importlib

        value = getattr(importlib.import_module(_LAZY_INDEX[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'valax.calibration' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_INDEX))
