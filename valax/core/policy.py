"""The pricing policy: ``auto`` method and default configs of every pair.

Every ``(instrument, model)`` pair that :func:`valax.core.pricing.price` can
reach has an explicit entry here — its ``auto`` method — so nothing is chosen
by registration order, and every numerical method of every pair has a default
config, so ``price(..., config=None)`` works for any method. Adding a pricer
for a new pair means adding its line; ``tests/test_core/test_pricing_policy.py``
fails for a registered pair without a policy or a default, and pins the map.

For every pair that existed then, the map reproduces what the former global
fallback (analytic → lattice → pde → mc) chose, so no ``auto`` price changed
when the fallback was removed (Phase 4, D15); the pairs added since (the
:class:`~valax.models.implied_vol.ImpliedVolModel` routes) state theirs here.

Imported for its side effects by
:func:`valax.core.pricing._ensure_backends`, after the adapters and the MC /
PDE recipe registries.
"""

from __future__ import annotations

from valax.calibration.cheyette import (
    _DEFAULT_CONFIG_FROZEN_1F as _CHEYETTE_FROZEN_1F,
    _DEFAULT_CONFIG_FROZEN_2F as _CHEYETTE_FROZEN_2F,
)
from valax.core.pricing import set_auto_method, set_default_config
from valax.instruments.bonds import (
    CallableBond,
    FixedRateBond,
    FloatingRateBond,
    PuttableBond,
)
from valax.instruments.inflation import (
    InflationCapFloor,
    YearOnYearInflationSwap,
    ZeroCouponInflationSwap,
)
from valax.instruments.legs import MultiLegSwap
from valax.instruments.options import (
    AmericanOption,
    AsianOption,
    DigitalOption,
    EquityBarrierOption,
    EuropeanOption,
    LookbackOption,
    SpreadOption,
    VarianceSwap,
    WorstOfBasketOption,
)
from valax.instruments.rates import (
    BermudanSwaption,
    CMSSpreadSwap,
    Cap,
    Caplet,
    CrossCurrencySwap,
    InterestRateSwap,
    OISSwap,
    Swaption,
)
from valax.models.black_scholes import BlackScholesModel
from valax.models.cheyette import CheyetteModel, CheyetteModel2F
from valax.models.g2pp import G2PPModel
from valax.models.heston import HestonModel
from valax.models.hull_white import HullWhiteModel
from valax.models.implied_vol import ImpliedVolModel
from valax.models.lmm import LMMModel
from valax.models.local_vol import LocalVolModel
from valax.models.multi_asset import MultiAssetGBMModel
from valax.models.slv import SLVModel
from valax.pricing.lattice.binomial import BinomialConfig
from valax.pricing.lattice.hull_white_tree import TreeConfig
from valax.pricing.mc import MCConfig
from valax.pricing.pde.config import PDEConfig, PDEConfig2D

_NONE = type(None)

#: ``(instrument_cls, model_cls) -> auto method``, grouped by model.
AUTO_METHODS: dict[tuple[type, type], str] = {
    # Model-free: curve discounting / projection.
    (FixedRateBond, _NONE): "analytic",
    (FloatingRateBond, _NONE): "analytic",
    (InterestRateSwap, _NONE): "analytic",
    (OISSwap, _NONE): "analytic",
    (MultiLegSwap, _NONE): "analytic",
    (CrossCurrencySwap, _NONE): "analytic",
    (ZeroCouponInflationSwap, _NONE): "analytic",
    (YearOnYearInflationSwap, _NONE): "analytic",
    (InflationCapFloor, ImpliedVolModel): "analytic",
    # Market implied vol (cube / optionlet surface / flat), Black-76 or
    # Bachelier per the source.
    (Caplet, ImpliedVolModel): "analytic",
    (Cap, ImpliedVolModel): "analytic",
    (Swaption, ImpliedVolModel): "analytic",
    # Black-Scholes.
    (EuropeanOption, BlackScholesModel): "analytic",
    (DigitalOption, BlackScholesModel): "analytic",
    (AmericanOption, BlackScholesModel): "lattice",
    (EquityBarrierOption, BlackScholesModel): "pde",
    (AsianOption, BlackScholesModel): "mc",
    (LookbackOption, BlackScholesModel): "mc",
    (VarianceSwap, BlackScholesModel): "mc",
    # Heston.
    (EuropeanOption, HestonModel): "analytic",
    (EquityBarrierOption, HestonModel): "mc",
    (AsianOption, HestonModel): "mc",
    (LookbackOption, HestonModel): "mc",
    (VarianceSwap, HestonModel): "mc",
    # Local volatility.
    (EuropeanOption, LocalVolModel): "pde",
    (EquityBarrierOption, LocalVolModel): "mc",
    (AsianOption, LocalVolModel): "mc",
    (LookbackOption, LocalVolModel): "mc",
    (VarianceSwap, LocalVolModel): "mc",
    # Stochastic-local volatility.
    (EuropeanOption, SLVModel): "mc",
    (EquityBarrierOption, SLVModel): "mc",
    (AsianOption, SLVModel): "mc",
    (LookbackOption, SLVModel): "mc",
    (VarianceSwap, SLVModel): "mc",
    # Multi-asset GBM.
    (SpreadOption, MultiAssetGBMModel): "mc",
    (WorstOfBasketOption, MultiAssetGBMModel): "mc",
    # Hull-White.
    (Swaption, HullWhiteModel): "analytic",
    (Caplet, HullWhiteModel): "pde",
    (Cap, HullWhiteModel): "pde",
    (BermudanSwaption, HullWhiteModel): "pde",
    (FixedRateBond, HullWhiteModel): "pde",
    (FloatingRateBond, HullWhiteModel): "mc",
    (CallableBond, HullWhiteModel): "pde",
    (PuttableBond, HullWhiteModel): "pde",
    # G2++.
    (Swaption, G2PPModel): "analytic",
    (Caplet, G2PPModel): "pde",
    (Cap, G2PPModel): "pde",
    (BermudanSwaption, G2PPModel): "pde",
    (CMSSpreadSwap, G2PPModel): "pde",
    (FixedRateBond, G2PPModel): "pde",
    (FloatingRateBond, G2PPModel): "mc",
    (CallableBond, G2PPModel): "pde",
    (PuttableBond, G2PPModel): "pde",
    # Cheyette (one factor).
    (Swaption, CheyetteModel): "pde",
    (Caplet, CheyetteModel): "pde",
    (Cap, CheyetteModel): "pde",
    (BermudanSwaption, CheyetteModel): "pde",
    (FixedRateBond, CheyetteModel): "pde",
    (FloatingRateBond, CheyetteModel): "mc",
    (CallableBond, CheyetteModel): "pde",
    (PuttableBond, CheyetteModel): "pde",
    # Cheyette (two factor).
    (Swaption, CheyetteModel2F): "pde",
    (Caplet, CheyetteModel2F): "pde",
    (Cap, CheyetteModel2F): "pde",
    (BermudanSwaption, CheyetteModel2F): "pde",
    (CMSSpreadSwap, CheyetteModel2F): "pde",
    (FixedRateBond, CheyetteModel2F): "pde",
    (FloatingRateBond, CheyetteModel2F): "mc",
    (CallableBond, CheyetteModel2F): "pde",
    (PuttableBond, CheyetteModel2F): "pde",
    # LMM.
    (Swaption, LMMModel): "mc",
    (Caplet, LMMModel): "mc",
    (Cap, LMMModel): "mc",
    (BermudanSwaption, LMMModel): "mc",
}

for (_instrument, _model), _method in AUTO_METHODS.items():
    set_auto_method(_instrument, _model, _method)


# ── Default configs ──────────────────────────────────────────────────
#
# Every numerical method of every pair has a default, so ``price`` works
# with ``config=None`` for any method. The defaults follow the engine and
# the model; an instrument that needs finer settings gets its own entry in
# ``CONFIG_OVERRIDES`` (none yet: add one only with evidence).

#: Monte Carlo: one default for every MC pair.
MC_DEFAULT = MCConfig(n_paths=100_000, n_steps=100)

#: PDE defaults by model. The 1-D and 2-D grids are the config classes'
#: own field defaults; Cheyette uses its calibrator's grids, so a fit and a
#: price that both leave ``config=None`` run on the same (engine-bound)
#: frozen-``y`` grid.
PDE_DEFAULTS: dict[type, object] = {
    BlackScholesModel: PDEConfig(),
    LocalVolModel: PDEConfig(),
    HestonModel: PDEConfig2D(),
    HullWhiteModel: PDEConfig(),
    G2PPModel: PDEConfig2D(),
    CheyetteModel: _CHEYETTE_FROZEN_1F,
    CheyetteModel2F: _CHEYETTE_FROZEN_2F,
}

#: Lattice defaults per pair. The Hull-White tree routes run on the general
#: tree with J = 2 n_steps (``TreeConfig()``); ``auto`` stays the PDE / the
#: closed form for those pairs — the tree is the independent check.
LATTICE_DEFAULTS: dict[tuple[type, type], object] = {
    (EuropeanOption, BlackScholesModel): BinomialConfig(american=False),
    (AmericanOption, BlackScholesModel): BinomialConfig(american=True),
    (CallableBond, HullWhiteModel): TreeConfig(),
    (PuttableBond, HullWhiteModel): TreeConfig(),
    (Swaption, HullWhiteModel): TreeConfig(),
    (BermudanSwaption, HullWhiteModel): TreeConfig(),
}

#: Per-(instrument, model, method) refinements of the defaults above.
CONFIG_OVERRIDES: dict[tuple[type, type, str], object] = {}


def _registered_methods(instrument_cls: type, model_cls: type) -> set[str]:
    """Methods registered for a pair (no back-end loading: called on import)."""
    from valax.core import pricing

    methods = {
        method for (i, m, method) in pricing._REGISTRY
        if i is instrument_cls and m is model_cls
    }
    for method in ("pde", "mc"):
        if (instrument_cls, model_cls) in pricing._legacy_registry(method):
            methods.add(method)
    return methods


def _default_for(instrument_cls: type, model_cls: type, method: str):
    """The default config of one (pair, method), or ``None`` (analytic)."""
    override = CONFIG_OVERRIDES.get((instrument_cls, model_cls, method))
    if override is not None:
        return override
    if method == "mc":
        return MC_DEFAULT
    if method == "pde":
        return PDE_DEFAULTS[model_cls]
    if method == "lattice":
        return LATTICE_DEFAULTS[(instrument_cls, model_cls)]
    return None


for (_instrument, _model) in AUTO_METHODS:
    for _method in _registered_methods(_instrument, _model):
        _config = _default_for(_instrument, _model, _method)
        if _config is not None:
            set_default_config(_instrument, _model, _method, _config)

__all__ = [
    "AUTO_METHODS",
    "CONFIG_OVERRIDES",
    "LATTICE_DEFAULTS",
    "MC_DEFAULT",
    "PDE_DEFAULTS",
]
