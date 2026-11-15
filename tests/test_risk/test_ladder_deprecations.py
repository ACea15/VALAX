"""The ``*Ladder`` → ``*Sensitivities`` rename keeps every old name working.

VALAX renamed its base-point bucketed Greeks from *ladders* to
*sensitivities* (the FRTB SA / ISDA SIMM term).  Every old public name —
symbol, module, method and keyword — must still resolve to the new object
and emit a :class:`DeprecationWarning` until the roadmap's scheduled removal.
"""

import importlib
import sys

import jax.numpy as jnp
import pytest

import valax.risk as risk
from valax.risk._deprecation import RENAMED_SYMBOLS
from valax.risk.scenarios import FactorBlock, KeyedFactorLayout
from valax.risk.state_sensitivities import KeyedSensitivities


@pytest.mark.parametrize("old, new", sorted(RENAMED_SYMBOLS.items()))
def test_package_alias_warns_and_resolves(old, new):
    with pytest.warns(DeprecationWarning, match=f"{old} is deprecated; use .*{new}"):
        obj = getattr(risk, old)
    assert obj is getattr(risk, new)


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        risk.NotARealSymbol  # noqa: B018


@pytest.mark.parametrize(
    "module, old, new",
    [
        ("valax.risk.state_sensitivities", "KeyedLadder", "KeyedSensitivities"),
        ("valax.risk.state_sensitivities", "keyed_ladder", "keyed_sensitivities"),
        ("valax.risk.bucketing", "BucketedLadder", "BucketedSensitivities"),
        ("valax.risk.bucketing", "bucket_sensitivity_ladder", "bucket_sensitivities"),
    ],
)
def test_module_alias_warns_and_resolves(module, old, new):
    mod = importlib.import_module(module)
    with pytest.warns(DeprecationWarning, match=old):
        obj = getattr(mod, old)
    assert obj is getattr(mod, new)


@pytest.mark.parametrize(
    "old_module, new_module, aliases",
    [
        (
            "valax.risk.ladders",
            "valax.risk.sensitivities",
            {
                "SensitivityLadder": "FlatSensitivities",
                "compute_ladder": "compute_flat_sensitivities",
                "waterfall_pnl": "waterfall_pnl",
                "WaterfallPnL": "WaterfallPnL",
            },
        ),
        (
            "valax.risk.state_ladders",
            "valax.risk.state_sensitivities",
            {
                "KeyedLadder": "KeyedSensitivities",
                "keyed_ladder": "keyed_sensitivities",
                "keyed_waterfall_pnl": "keyed_waterfall_pnl",
                "keyed_parametric_var": "keyed_parametric_var",
            },
        ),
    ],
)
def test_old_module_shim_warns_on_import(old_module, new_module, aliases):
    sys.modules.pop(old_module, None)
    with pytest.warns(DeprecationWarning, match=f"{old_module} is deprecated"):
        shim = importlib.import_module(old_module)
    new = importlib.import_module(new_module)
    for old_name, new_name in aliases.items():
        assert getattr(shim, old_name) is getattr(new, new_name)


def _toy_sensitivities() -> KeyedSensitivities:
    layout = KeyedFactorLayout(
        blocks=(FactorBlock(kind="curve", key="OIS", size=2, times=(1.0, 2.0)),),
    )
    return KeyedSensitivities(
        layout=layout, pv=jnp.asarray(1.0), delta=jnp.asarray([-100.0, -200.0]),
    )


def test_curve_ladder_method_forwards_to_key_rate_dv01():
    sens = _toy_sensitivities()
    with pytest.warns(DeprecationWarning, match="curve_ladder is deprecated"):
        old = sens.curve_ladder("OIS")
    assert jnp.array_equal(old, sens.key_rate_dv01("OIS"))


def test_ladder_keyword_forwards_with_warning():
    sens = _toy_sensitivities()
    cov = jnp.eye(2) * 1e-8
    expected = risk.keyed_parametric_var(sens, cov)
    with pytest.warns(DeprecationWarning, match="'ladder' argument"):
        got = risk.keyed_parametric_var(ladder=sens, cov=cov)
    assert jnp.allclose(got, expected)


def test_both_keywords_is_an_error():
    sens = _toy_sensitivities()
    with pytest.raises(TypeError, match="pass only 'sensitivities'"):
        risk.keyed_parametric_var(
            sensitivities=sens, ladder=sens, cov=jnp.eye(2),
        )
