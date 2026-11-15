"""Every public risk-measure entry point has an entry in the risk register.

The register (``docs/applications/risk-register.md``) is the single place a
risk measure's methodology, assumptions and limitations are recorded. This
test discovers the risk-measure entry points **by defining module** from the
public exports of ``valax.risk`` — anything exported from the sensitivity,
P&L-vector, backtesting and bucketing modules, plus the named VaR / ES,
repricing and scenario functions — and asserts each is named (in backticks)
on the register page. It also checks that every gate test the register
names exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import valax.risk as risk

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "docs" / "applications" / "risk-register.md"

#: Modules whose every public export is a risk measure (or its data type).
_MEASURE_MODULES = (
    "valax.risk.sensitivities",
    "valax.risk.state_sensitivities",
    "valax.risk.pnl_vectors",
    "valax.risk.backtesting",
    "valax.risk.bucketing",
    "valax.risk.revaluation_grids",
)

#: Risk measures and scenario machinery exported from mixed modules.
_NAMED = (
    "value_at_risk",
    "expected_shortfall",
    "parametric_var",
    "pnl_attribution",
    "portfolio_pnl",
    "group_scenario_pnl",
    "keyed_parametric_scenarios",
    "keyed_historical_scenarios",
    "parametric_scenarios",
    "historical_scenarios",
    "block_covariance",
    "historical_block_covariance",
    "psd_factor",
    "shock",
    "shift_surface",
    "KeyedFactorLayout",
    "keyed_stress_scenario",
    "keyed_curve_stress",
    "keyed_basis_shift",
)


def _entry_points() -> list[str]:
    found = set(_NAMED)
    for name in dir(risk):
        if name.startswith("_"):
            continue
        obj = getattr(risk, name)
        if getattr(obj, "__module__", None) in _MEASURE_MODULES:
            found.add(name)
    found.add("BASIS_POINT")
    return sorted(found)


def _documented() -> set[str]:
    text = REGISTER.read_text(encoding="utf-8")
    return set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", text))


def test_discovery_sees_the_known_measures():
    """Anti-vacuity: the discovery really sees the risk measures."""
    found = set(_entry_points())
    for name in ("keyed_sensitivities", "KeyedSensitivities", "compute_flat_sensitivities",
                 "risk_theoretical_pnl_vector", "pla_traffic_light", "kupiec_pof",
                 "tenor_bucket_map", "bucket_sensitivities", "value_at_risk"):
        assert name in found, name


@pytest.mark.parametrize("name", _entry_points())
def test_entry_point_is_in_the_register(name):
    assert name in _documented(), (
        f"`{name}` is a public risk-measure entry point with no entry in "
        f"{REGISTER.relative_to(ROOT)}. Add (or extend) its entry: definition "
        "& defaults, assumptions, limitations with evidence, validation, "
        "improvement path."
    )


def test_named_gates_exist():
    text = REGISTER.read_text(encoding="utf-8")
    gates = set(re.findall(r"`(tests/[\w/]+\.py)(?:::\w+)?`", text))
    assert gates, "the register names no gate tests"
    missing = sorted(g for g in gates if not (ROOT / g).is_file())
    assert not missing, f"register names missing gate tests: {missing}"
