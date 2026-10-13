"""Every public calibration entry point has an entry in the assumptions register.

The register (``docs/applications/calibration-register.md``) is the single
place a calibration's methodology, assumptions and limitations are recorded.
Docs rot when code lands without them, so this test discovers calibration
entry points **by naming rule** from the public exports — a new
``calibrate_*``, ``*_node``, ``bootstrap_*``, ``strip_*``, ``from_*`` (curve
constructors from quotes) … is caught the day it is exported — and asserts
each is named (in backticks) on the register page.
"""

from __future__ import annotations

import re
import types
from pathlib import Path

import pytest

import valax.calibration as calibration
import valax.curves as curves
import valax.surfaces as surfaces

REGISTER = (
    Path(__file__).resolve().parents[2]
    / "docs" / "applications" / "calibration-register.md"
)

# Naming rules for calibration entry points, per package.
_RULES = {
    calibration: re.compile(r"^(calibrate_\w+|\w+_node|basket_fit_level|coterminal_swaptions)$"),
    surfaces: re.compile(r"^(calibrate_\w+|strip_\w+|build_\w+|\w+_cube_level|swaption_implied_normal_vol\w*)$"),
    curves: re.compile(r"^(bootstrap_\w+|from_\w+|fitted_curve_quotes|quote_jacobian)$"),
}

# Matches of the rules that are not calibrations (chain drivers, helpers).
_NOT_CALIBRATIONS = {
    "calibrate_market",   # chain driver
    "calibrate_model",    # chain driver
    "from_hazard_rates",  # curve constructor from model parameters, not quotes
}

# Level builders not re-exported at package level, documented by module.
_EXTRA = ("caplet_strip_level", "strip_columns_level")


def _public_names(module) -> set[str]:
    """Public, non-module attributes of a package (lazy exports included)."""
    names = set(getattr(module, "__all__", ()) or ())
    names |= {n for n in dir(module) if not n.startswith("_")}
    names |= set(getattr(module, "_LAZY_INDEX", {}))
    return {n for n in names
            if not isinstance(getattr(module, n, None), types.ModuleType)}


def _entry_points() -> list[str]:
    found: set[str] = set(_EXTRA)
    for module, rule in _RULES.items():
        found |= {n for n in _public_names(module) if rule.match(n)}
    return sorted(found - _NOT_CALIBRATIONS)


def _documented() -> set[str]:
    text = REGISTER.read_text(encoding="utf-8")
    return set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", text))


def test_the_rules_find_the_known_entry_points():
    """Anti-vacuity: the discovery really sees the calibrators."""
    found = set(_entry_points())
    for name in ("bootstrap_curve_graph", "calibrate_sabr", "calibrate_hull_white",
                 "swaption_basket_node", "strip_caplet_vols",
                 "calibrate_swaption_cube_from_curves", "calibrate_slv_leverage",
                 "from_cds_spreads"):
        assert name in found, name


@pytest.mark.parametrize("name", _entry_points())
def test_entry_point_is_in_the_register(name):
    assert name in _documented(), (
        f"`{name}` is a public calibration entry point with no entry in "
        f"{REGISTER.relative_to(REGISTER.parents[2])}. Add (or extend) its "
        "entry: fits / holds fixed, targets & objective, engine & provenance, "
        "settings, market inputs, assumptions, limitations, validation, "
        "improvement path. If it is not a calibration, add it to "
        "_NOT_CALIBRATIONS here with the reason."
    )
