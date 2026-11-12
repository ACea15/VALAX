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
# Examples taking more than ~15 s even in fast mode are ``slow`` (deselected
# by default; ``pytest --runslow``). Every example script still runs on each
# pull request in the Examples CI workflow, in full mode.
_slow = pytest.mark.slow
UNIFIED = (
    pytest.param("10_unified_market_tiers.py", marks=_slow),
    "11_unified_multicurve_rates.py",
    pytest.param("12_unified_equity_smile_and_model.py", marks=_slow),
    "13_unified_quote_scenarios.py",
    pytest.param("14_unified_rates_per_trade_model.py", marks=_slow),
    pytest.param("15_unified_rates_caps_optionlet_chain.py", marks=_slow),
    pytest.param("16_unified_calibration_diagnostics.py", marks=_slow),
    pytest.param("17_unified_slv_fixed_point.py", marks=_slow),
    pytest.param("18_unified_rates_per_trade_g2pp_cheyette.py", marks=_slow),
    "19_unified_book_workflow.py",
    "20_unified_fx_inflation_xccy.py",
    pytest.param("21_unified_rates_engine_ladder.py", marks=_slow),
    pytest.param("22_unified_multi_asset_exotics.py", marks=_slow),
)


@pytest.mark.parametrize("name", UNIFIED)
def test_example_runs(name, monkeypatch, capsys):
    monkeypatch.setenv("VALAX_EXAMPLES_FAST", "1")
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    out = capsys.readouterr().out
    assert "Traceback" not in out
    assert out.strip()
