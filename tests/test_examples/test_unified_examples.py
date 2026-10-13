"""Run the unified-framework examples end to end.

Each example asserts its own claims (truth recovery, path reconciliation,
FD agreement, second-order P&L remainder), so running it is the test. The
``VALAX_EXAMPLES_FAST`` switch shrinks grids, path counts and scenario counts.
"""

import os
import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
UNIFIED = (
    "10_unified_market_tiers.py",
    "11_unified_multicurve_rates.py",
    "12_unified_equity_smile_and_model.py",
    "13_unified_quote_scenarios.py",
    "14_unified_rates_per_trade_model.py",
    "15_unified_rates_caps_optionlet_chain.py",
    "16_unified_calibration_diagnostics.py",
    "17_unified_slv_fixed_point.py",
    "18_unified_rates_per_trade_g2pp_cheyette.py",
)


@pytest.mark.parametrize("name", UNIFIED)
def test_example_runs(name, monkeypatch, capsys):
    monkeypatch.setenv("VALAX_EXAMPLES_FAST", "1")
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    out = capsys.readouterr().out
    assert "Traceback" not in out
    assert out.strip()
