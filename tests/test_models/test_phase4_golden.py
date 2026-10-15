"""Phase 4 regression gate: constant volatility is unchanged by σ(t).

``tests/data/phase4_golden.json`` was captured by
``scripts/capture_phase4_golden.py`` at the commit *before* the short-rate
volatilities became piecewise-constant arrays. A one-bucket model must
reproduce it **bit for bit** — the constant case runs the original closed
forms — except for entries listed in :data:`MIGRATED`, whose model was
deliberately re-parameterised (G2++ / Cheyette-2F ``volatility_ratio``) and
which are held to a stated relative tolerance instead.

The full recomputation takes ~25 s and is marked slow; the fast sibling
recomputes the Hull-White model-level entries.
"""

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "capture_phase4_golden.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("capture_phase4_golden", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden_script = _load_script()
GOLDEN = golden_script.load_golden(golden_script.DEFAULT_OUT)

#: Entries whose model was re-parameterised on purpose, with the relative
#: tolerance they are held to (migration evidence, not bit identity).
#:
#: G2++ now carries ``volatility_ratio`` with ``eta = ratio * sigma`` (the
#: capture script builds it as ``ratio = eta / sigma``), and the Cheyette-2F
#: frozen-y PDE reuses a G2++ twin built the same way. Values are held to
#: 1e-14 relative, gradients (whose reverse-mode accumulation now runs through
#: the ratio) to 1e-12. Measured at migration: every value bit-identical; the
#: only differences are ``g2pp.grad.analytic_payer.volatility_x`` (1.4e-15
#: relative) and ``g2pp.grad.analytic_payer.volatility_y`` (1.5e-16).
_G2PP_VALUES = (
    "g2pp.V",
    "g2pp.bond_price",
    "g2pp.price.analytic.payer",
    "g2pp.price.analytic.receiver",
    "g2pp.price.mc.payer",
    "g2pp.price.mc.payer.stderr",
    "g2pp.price.pde.bermudan",
    "g2pp.price.pde.payer",
    "cheyette2f.price.mc.fixed_bond",
    "cheyette2f.price.mc.fixed_bond.stderr",
    "cheyette2f.price.pde_frozen.payer",
)
_G2PP_GRADS = tuple(
    f"g2pp.grad.analytic_payer.{p}"
    for p in (
        "mean_reversion_x", "mean_reversion_y", "volatility_x",
        "volatility_y", "correlation",
    )
) + tuple(
    f"cheyette2f.grad.pde_frozen_payer.{p}"
    for p in (
        "mean_reversion_1", "mean_reversion_2", "volatility_1",
        "volatility_2", "skew_1", "skew_2", "correlation",
    )
)
MIGRATED: dict[str, float] = {
    **{name: 1e-14 for name in _G2PP_VALUES},
    **{name: 1e-12 for name in _G2PP_GRADS},
}


def _assert_matches(name: str, value: np.ndarray) -> None:
    expected = GOLDEN[name]
    value = np.asarray(value, dtype=np.float64).reshape(expected.shape)
    if name in MIGRATED:
        np.testing.assert_allclose(value, expected, rtol=MIGRATED[name], err_msg=name)
    else:
        assert np.array_equal(value, expected), (
            f"{name} not bit-identical: max |diff| = "
            f"{np.max(np.abs(value - expected)):.3e}"
        )


def test_migrated_entries_exist():
    """Every tolerance-held entry names a real golden value (no silent typo)."""
    assert set(MIGRATED) <= set(GOLDEN)


class TestHullWhiteModelLevelGolden:
    """Fast sibling: the one-bucket Hull-White kernels, bit for bit."""

    @pytest.fixture(scope="class")
    def hw(self):
        from valax.core.binding import bind_curves

        model = golden_script._hw(*golden_script.HW_PARAMS.values())
        return bind_curves(model, golden_script.build_curve())

    def test_volatility_is_one_bucket(self, hw):
        assert hw.volatility.shape == (1,)
        assert hw.volatility_times.shape == (0,)

    def test_bond_price(self, hw):
        from valax.models.hull_white import hw_bond_price

        _assert_matches("hw.bond_price", jnp.stack([
            hw_bond_price(hw, jnp.asarray(r), jnp.asarray(t), jnp.asarray(T))
            for r, t, T in [
                (0.02, 0.0, 1.0), (0.03, 1.0, 5.0), (0.05, 2.5, 7.0), (-0.01, 3.0, 12.0)
            ]
        ]))

    def test_alpha_and_variance(self, hw):
        from valax.models.hull_white import (
            hw_alpha,
            hw_alpha_average,
            hw_short_rate_variance,
        )

        t_grid = jnp.array([0.0, 0.25, 1.0, 2.0, 3.7, 5.0, 10.0, 12.0])
        _assert_matches("hw.alpha", hw_alpha(hw, t_grid))
        _assert_matches("hw.alpha_average", hw_alpha_average(
            hw, jnp.array([0.0, 0.5, 2.0, 5.0]), jnp.array([1.0, 2.0, 4.5, 11.0])
        ))
        _assert_matches("hw.short_rate_variance", jnp.stack(
            [hw_short_rate_variance(hw, t) for t in t_grid]
        ))


@pytest.mark.slow
def test_full_golden_recomputation():
    """Every captured value (all engines, gradients, paths) is unchanged."""
    fresh = golden_script.compute_values()
    assert set(fresh) == set(GOLDEN)
    failures = []
    for name in sorted(GOLDEN):
        try:
            _assert_matches(name, fresh[name])
        except AssertionError as exc:
            failures.append(str(exc).splitlines()[0])
    assert not failures, "\n".join(failures)
