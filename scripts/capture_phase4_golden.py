"""Capture the Phase 4 bit-identity baseline for the short-rate models.

Phase 4 turns the Hull-White, G2++ and Cheyette volatility parameters into
piecewise-constant arrays.  For the constant-volatility special case that
refactor must be a pure no-op, so this script records — from the commit
*before* piecewise volatility lands — a deterministic set of reference values
(model-level kernels, analytic / PDE / lattice / MC prices, autodiff
gradients, simulated path summaries) together with the ``method="auto"`` map
of :func:`valax.core.pricing.price`.  Every float is stored both as ``repr``
and as :meth:`float.hex`, so a later run can be compared **bit for bit**.

Everything is fixed: float64 (``jax_enable_x64``), one hand-picked upward
sloping log-linear discount curve, explicit grids, explicit PRNG keys and an
explicit trinomial-tree half-width.

Re-run / verify::

    python scripts/capture_phase4_golden.py            # (re)write the baseline
    python scripts/capture_phase4_golden.py --check    # recompute and compare

``--check`` exits non-zero and lists every mismatching entry when any value
differs from the stored file in even the last bit.  Use :func:`load_golden`
to read the file from a test.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from valax.core import pricing as core_pricing  # noqa: E402
from valax.core.binding import bind_curves  # noqa: E402
from valax.core.market_state import MarketState  # noqa: E402
from valax.core.pricing import auto_method, available_methods, price  # noqa: E402
from valax.curves.discount import DiscountCurve  # noqa: E402
from valax.dates.daycounts import ymd_to_ordinal  # noqa: E402
from valax.dates.schedule import generate_schedule  # noqa: E402
from valax.instruments.bonds import CallableBond, FixedRateBond, PuttableBond  # noqa: E402
from valax.instruments.rates import BermudanSwaption, Cap, Swaption  # noqa: E402
from valax.models.cheyette import (  # noqa: E402
    CheyetteModel,
    CheyetteModel2F,
    cheyette_bond_price,
    cheyette_y_gaussian,
)
from valax.models.g2pp import G2PPModel, g2pp_bond_price, g2pp_V  # noqa: E402
from valax.models.hull_white import (  # noqa: E402
    HullWhiteModel,
    hw_alpha,
    hw_alpha_average,
    hw_bond_price,
    hw_short_rate_variance,
)
from valax.pricing.analytic.hull_white_swaptions import (  # noqa: E402
    hw_critical_rate,
    hw_swaption_price,
    hw_zcb_option_price,
)
from valax.pricing.lattice.hull_white_tree import (  # noqa: E402
    callable_bond_price,
    hw_tree_j_max,
    puttable_bond_price,
)
from valax.pricing.mc import MCConfig  # noqa: E402
from valax.pricing.mc.hull_white_paths import generate_hull_white_paths  # noqa: E402
from valax.pricing.pde import PDEConfig, PDEConfig2D, Scheme, pde_price_dispatch  # noqa: E402
from valax.pricing.pde.cheyette import cheyette_exact_price  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "tests" / "data" / "phase4_golden.json"

# ── Fixed setup ───────────────────────────────────────────────────────

DAY_COUNT = "act_365"
REF = int(ymd_to_ordinal(2026, 1, 1))

#: Pillar tenors (years) and continuously-compounded zero rates: 2% -> 4%.
CURVE_YEARS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0)
CURVE_ZEROS = (0.020, 0.021, 0.022, 0.025, 0.027, 0.031, 0.034, 0.038, 0.040)

PDE_1D = PDEConfig(n_spot=101, n_time=60, spot_range=6.0, scheme=Scheme.CRANK_NICOLSON)
PDE_2D = PDEConfig2D(n_x=41, n_y=41, n_time=30, x_range=6.0, scheme=Scheme.CRAIG_SNEYD)
PDE_EXACT_Y = PDEConfig2D(n_x=41, n_y=31, n_time=36, x_range=6.0, y_max=3.0)
#: Short-rate MC only. ``n_steps=100``: the values were captured when the
#: short-rate recipes ignored ``MCConfig.n_steps`` and always used their
#: 100-step keyword default (this said 50, silently unused). Since D15 the
#: recipes read the config, so stating the step count actually used keeps the
#: captured values bit-identical.
MC_CONFIG = MCConfig(n_paths=4096, n_steps=100)

TREE_STEPS = 60
#: Concrete tree half-width: ``hw_tree_j_max(0.05, 5Y / 60)`` = 45, pinned so
#: the baseline does not depend on that helper.
TREE_J_MAX = 45

KEY_MC = 20260101
KEY_PATHS = 4

HW_PARAMS = dict(mean_reversion=0.05, volatility=0.01)
G2_PARAMS = dict(
    mean_reversion_x=0.5,
    mean_reversion_y=0.1,
    volatility_x=0.01,
    volatility_y=0.008,
    correlation=-0.7,
)
CH1_PARAMS = dict(mean_reversion=0.05, volatility=0.01, skew=0.5)
CH2_PARAMS = dict(
    mean_reversion_1=0.5,
    mean_reversion_2=0.1,
    volatility_1=0.01,
    volatility_2=0.008,
    skew_1=0.1,
    skew_2=0.05,
    correlation=-0.7,
)


def build_curve() -> DiscountCurve:
    """The single non-flat, log-linear discount curve used everywhere.

    Returns:
        A ``DiscountCurve`` anchored at ``REF`` with ``DF(0) = 1``.
    """
    pillars = jnp.array(
        [REF + int(round(y * 365)) for y in CURVE_YEARS], dtype=jnp.int32
    )
    times = (pillars - REF).astype(jnp.float64) / 365.0
    zeros = jnp.array(CURVE_ZEROS, dtype=jnp.float64)
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-zeros * times),
        reference_date=jnp.int32(REF),
        day_count=DAY_COUNT,
    )


def _ords_years(years) -> jnp.ndarray:
    """Ordinals ``REF + 365 * y`` for each year count ``y``."""
    return jnp.array([REF + 365 * y for y in years], dtype=jnp.int32)


def _jan1(years) -> jnp.ndarray:
    """Calendar 1-January ordinals for each calendar year."""
    return jnp.array([int(ymd_to_ordinal(y, 1, 1)) for y in years], dtype=jnp.int32)


def build_instruments() -> dict[str, Any]:
    """The fixed instrument set.

    Returns:
        Mapping from a short name to the instrument pytree.
    """
    expiry = REF + 2 * 365
    swap_fixed = jnp.array([expiry + 365 * k for k in range(1, 6)], dtype=jnp.int32)
    swaption_kw = dict(
        expiry_date=jnp.int32(expiry),
        fixed_dates=swap_fixed,
        strike=jnp.array(0.03),
        notional=jnp.array(1.0),
        day_count=DAY_COUNT,
    )
    bond_schedule = generate_schedule(2026, 1, 1, 2031, 1, 1, frequency=2).payment_dates
    call_dates = _jan1((2028, 2029, 2030))
    bond_kw = dict(
        payment_dates=bond_schedule,
        settlement_date=jnp.int32(REF),
        coupon_rate=jnp.array(0.04),
        face_value=jnp.array(100.0),
        frequency=2,
    )
    cap_fix = jnp.array(
        [int(ymd_to_ordinal(2027 + i // 2, 1 if i % 2 == 0 else 7, 1)) for i in range(6)],
        dtype=jnp.int32,
    )
    cap_end = jnp.array(
        [int(ymd_to_ordinal(2027 + (i + 1) // 2, 7 if i % 2 == 0 else 1, 1)) for i in range(6)],
        dtype=jnp.int32,
    )
    return {
        "payer": Swaption(is_payer=True, **swaption_kw),
        "receiver": Swaption(is_payer=False, **swaption_kw),
        "bermudan": BermudanSwaption(
            exercise_dates=_ords_years((2, 3, 4, 5)),
            fixed_dates=_ords_years((3, 4, 5, 6, 7)),
            strike=jnp.array(0.03),
            notional=jnp.array(1.0),
            is_payer=True,
            day_count=DAY_COUNT,
        ),
        "fixed_bond": FixedRateBond(**bond_kw),
        "callable": CallableBond(
            call_dates=call_dates, call_prices=jnp.array([1.0, 1.0, 1.0]), **bond_kw
        ),
        "puttable": PuttableBond(
            put_dates=call_dates, put_prices=jnp.array([1.0, 1.0, 1.0]), **bond_kw
        ),
        "cap": Cap(
            fixing_dates=cap_fix,
            start_dates=cap_fix,
            end_dates=cap_end,
            strike=jnp.array(0.03),
            notional=jnp.array(1.0),
            is_cap=True,
            day_count=DAY_COUNT,
        ),
    }


def _hw(a, sigma) -> HullWhiteModel:
    return HullWhiteModel(mean_reversion=jnp.asarray(a), volatility=jnp.asarray(sigma))


def _g2(a, b, sigma, eta, rho) -> G2PPModel:
    # The model is parameterised by the ratio r = eta / sigma; building it from
    # (sigma, eta) keeps the golden entries (and d/d eta) in the old variables.
    sigma = jnp.asarray(sigma)
    return G2PPModel(
        mean_reversion_x=jnp.asarray(a),
        mean_reversion_y=jnp.asarray(b),
        volatility_x=sigma,
        volatility_ratio=jnp.asarray(eta) / sigma,
        correlation=jnp.asarray(rho),
    )


def _ch1(kappa, sigma, skew) -> CheyetteModel:
    return CheyetteModel(
        mean_reversion=jnp.asarray(kappa),
        volatility=jnp.asarray(sigma),
        skew=jnp.asarray(skew),
    )


def _ch2(k1, k2, sigma1, sigma2, skew1, skew2, rho) -> CheyetteModel2F:
    # The model is parameterised by the ratio r = sigma_2 / sigma_1; building
    # it from (sigma_1, sigma_2) keeps the golden entries (and d/d sigma_2) in
    # the old variables.
    sigma1 = jnp.asarray(sigma1)
    return CheyetteModel2F(
        mean_reversion_1=jnp.asarray(k1),
        mean_reversion_2=jnp.asarray(k2),
        volatility_1=sigma1,
        volatility_ratio=jnp.asarray(sigma2) / sigma1,
        skew_1=jnp.asarray(skew1),
        skew_2=jnp.asarray(skew2),
        correlation=jnp.asarray(rho),
    )


def _grads(
    record: Callable[[str, Any], None],
    prefix: str,
    fn: Callable[..., Any],
    params: dict[str, float],
) -> None:
    """Record ``d fn / d p`` for every scalar parameter ``p``, one entry each."""
    args = tuple(jnp.asarray(v, dtype=jnp.float64) for v in params.values())
    grads = jax.grad(fn, argnums=tuple(range(len(args))))(*args)
    for name, g in zip(params, grads):
        record(f"{prefix}.{name}", g)


# ── Value capture ─────────────────────────────────────────────────────


def compute_values() -> dict[str, np.ndarray]:
    """Recompute every golden value from the current code.

    Returns:
        Mapping from entry name to a float64 numpy array.
    """
    values: dict[str, np.ndarray] = {}

    def record(name: str, x: Any) -> None:
        if name in values:
            raise ValueError(f"duplicate golden entry {name!r}")
        values[name] = np.asarray(jax.device_get(x), dtype=np.float64)

    curve = build_curve()
    market = MarketState.single_curve(curve)
    inst = build_instruments()
    key = jax.random.PRNGKey(KEY_MC)
    t_grid = jnp.array([0.0, 0.25, 1.0, 2.0, 3.7, 5.0, 10.0, 12.0])

    # ── Hull-White ───────────────────────────────────────────────────
    hw_free = _hw(*HW_PARAMS.values())
    hw = bind_curves(hw_free, curve)
    record("hw.bond_price", jnp.stack([
        hw_bond_price(hw, jnp.asarray(r), jnp.asarray(t), jnp.asarray(T))
        for r, t, T in [(0.02, 0.0, 1.0), (0.03, 1.0, 5.0), (0.05, 2.5, 7.0), (-0.01, 3.0, 12.0)]
    ]))
    record("hw.alpha", hw_alpha(hw, t_grid))
    record("hw.alpha_average", hw_alpha_average(
        hw, jnp.array([0.0, 0.5, 2.0, 5.0]), jnp.array([1.0, 2.0, 4.5, 11.0])
    ))
    record("hw.short_rate_variance", jnp.stack(
        [hw_short_rate_variance(hw, t) for t in t_grid]
    ))
    zcb_T = jnp.asarray(2.0)
    zcb_S = jnp.array([3.0, 5.0, 7.0])
    zcb_X = jnp.array([0.97, 0.90, 0.82])
    record("hw.zcb_option.call", hw_zcb_option_price(hw, zcb_T, zcb_S, zcb_X, is_call=True))
    record("hw.zcb_option.put", hw_zcb_option_price(hw, zcb_T, zcb_S, zcb_X, is_call=False))
    record("hw.jamshidian.payer", hw_swaption_price(inst["payer"], hw))
    record("hw.jamshidian.receiver", hw_swaption_price(inst["receiver"], hw))
    record("hw.critical_rate", hw_critical_rate(
        hw, jnp.asarray(2.0), jnp.array([3.0, 4.0, 5.0, 6.0, 7.0]),
        jnp.array([0.03, 0.03, 0.03, 0.03, 1.03]),
    ))

    record("hw.price.analytic.payer", price(inst["payer"], hw_free, market, method="analytic").price)
    record("hw.price.analytic.receiver", price(inst["receiver"], hw_free, market, method="analytic").price)
    for name in ("payer", "receiver", "bermudan", "callable", "puttable", "fixed_bond", "cap"):
        record(f"hw.price.pde.{name}", price(inst[name], hw_free, market, method="pde", config=PDE_1D).price)
    for name in ("fixed_bond", "callable", "puttable"):
        res = price(inst[name], hw_free, market, method="mc", config=MC_CONFIG, key=key)
        record(f"hw.price.mc.{name}", res.price)
        record(f"hw.price.mc.{name}.stderr", res.stderr)
    record("hw.tree.callable", callable_bond_price(inst["callable"], hw, n_steps=TREE_STEPS, j_max=TREE_J_MAX))
    record("hw.tree.puttable", puttable_bond_price(inst["puttable"], hw, n_steps=TREE_STEPS, j_max=TREE_J_MAX))

    def hw_bound(a, s):
        return bind_curves(_hw(a, s), curve)

    _grads(record, "hw.grad.jamshidian_payer",
           lambda a, s: hw_swaption_price(inst["payer"], hw_bound(a, s)), HW_PARAMS)
    _grads(record, "hw.grad.pde_bermudan",
           lambda a, s: pde_price_dispatch(inst["bermudan"], hw_bound(a, s), PDE_1D).price, HW_PARAMS)
    _grads(record, "hw.grad.tree_callable",
           lambda a, s: callable_bond_price(inst["callable"], hw_bound(a, s),
                                            n_steps=TREE_STEPS, j_max=TREE_J_MAX), HW_PARAMS)

    paths = generate_hull_white_paths(hw, T=5.0, n_steps=50, n_paths=4096,
                                      key=jax.random.PRNGKey(KEY_PATHS))
    record("hw.paths.mean_short_rate_last", jnp.mean(paths.short_rates[:, -1]))
    record("hw.paths.mean_df_last", jnp.mean(jnp.exp(paths.log_discount_factors[:, -1])))
    record("hw.paths.short_rates_sample", paths.short_rates[jnp.array([0, 1, 2047, 4095])][:, jnp.array([0, 1, 25, 50])])
    record("hw.paths.log_df_sample", paths.log_discount_factors[jnp.array([0, 1, 2047, 4095])][:, jnp.array([1, 25, 50])])

    # ── G2++ ─────────────────────────────────────────────────────────
    g2_free = _g2(*G2_PARAMS.values())
    g2 = bind_curves(g2_free, curve)
    record("g2pp.V", g2pp_V(g2, jnp.array([0.0, 0.0, 1.0, 2.0]), jnp.array([1.0, 5.0, 3.0, 12.0])))
    record("g2pp.bond_price", jnp.stack([
        g2pp_bond_price(g2, jnp.asarray(x), jnp.asarray(y), jnp.asarray(t), jnp.asarray(T))
        for x, y, t, T in [(0.0, 0.0, 0.0, 2.0), (0.01, -0.005, 1.0, 5.0), (-0.02, 0.01, 2.0, 7.0)]
    ]))
    record("g2pp.price.analytic.payer", price(inst["payer"], g2_free, market, method="analytic").price)
    record("g2pp.price.analytic.receiver", price(inst["receiver"], g2_free, market, method="analytic").price)
    record("g2pp.price.pde.payer", price(inst["payer"], g2_free, market, method="pde", config=PDE_2D).price)
    record("g2pp.price.pde.bermudan", price(inst["bermudan"], g2_free, market, method="pde", config=PDE_2D).price)
    res = price(inst["payer"], g2_free, market, method="mc", config=MC_CONFIG, key=key)
    record("g2pp.price.mc.payer", res.price)
    record("g2pp.price.mc.payer.stderr", res.stderr)
    _grads(record, "g2pp.grad.analytic_payer",
           lambda *p: price(inst["payer"], _g2(*p), market, method="analytic").price, G2_PARAMS)

    # ── Cheyette 1F ──────────────────────────────────────────────────
    ch_free = _ch1(*CH1_PARAMS.values())
    ch = bind_curves(ch_free, curve)
    record("cheyette.y_gaussian", cheyette_y_gaussian(ch, t_grid))
    record("cheyette.bond_price", jnp.stack([
        cheyette_bond_price(ch, jnp.asarray(x), jnp.asarray(y), jnp.asarray(t), jnp.asarray(T))
        for x, y, t, T in [(0.0, 0.0, 0.0, 2.0), (0.01, 1e-4, 1.0, 5.0), (-0.015, 3e-4, 2.0, 7.0)]
    ]))
    record("cheyette.price.pde_frozen.payer", price(inst["payer"], ch_free, market, method="pde", config=PDE_1D).price)
    record("cheyette.price.pde_frozen.bermudan", price(inst["bermudan"], ch_free, market, method="pde", config=PDE_1D).price)
    record("cheyette.price.pde_exact.payer", cheyette_exact_price(inst["payer"], ch, PDE_EXACT_Y).price)
    res = price(inst["payer"], ch_free, market, method="mc", config=MC_CONFIG, key=key)
    record("cheyette.price.mc.payer", res.price)
    record("cheyette.price.mc.payer.stderr", res.stderr)
    _grads(record, "cheyette.grad.pde_frozen_payer",
           lambda *p: pde_price_dispatch(inst["payer"], bind_curves(_ch1(*p), curve), PDE_1D).price,
           CH1_PARAMS)

    # ── Cheyette 2F ──────────────────────────────────────────────────
    ch2_free = _ch2(*CH2_PARAMS.values())
    record("cheyette2f.price.pde_frozen.payer", price(inst["payer"], ch2_free, market, method="pde", config=PDE_2D).price)
    res = price(inst["fixed_bond"], ch2_free, market, method="mc", config=MC_CONFIG, key=key)
    record("cheyette2f.price.mc.fixed_bond", res.price)
    record("cheyette2f.price.mc.fixed_bond.stderr", res.stderr)
    _grads(record, "cheyette2f.grad.pde_frozen_payer",
           lambda *p: pde_price_dispatch(inst["payer"], bind_curves(_ch2(*p), curve), PDE_2D).price,
           CH2_PARAMS)

    return values


def _cls_label(cls: type, collisions: set[str]) -> str:
    """Class name, module-qualified when the bare name is ambiguous."""
    if cls is type(None):
        return "NoneType"
    if cls.__qualname__ in collisions:
        return f"{cls.__module__}.{cls.__qualname__}"
    return cls.__qualname__


def compute_auto_methods() -> dict[str, dict[str, Any]]:
    """``auto_method`` / ``available_methods`` for every reachable pair.

    Returns:
        Mapping ``"Instrument|Model"`` to ``{"auto": ..., "available": [...]}``.
    """
    core_pricing._ensure_backends()
    pairs = {(i, m) for (i, m, _method) in core_pricing._REGISTRY}
    for method in ("mc", "pde"):
        pairs.update(core_pricing._legacy_registry(method).keys())

    def collisions(classes) -> set[str]:
        seen: dict[str, set[type]] = {}
        for c in classes:
            seen.setdefault(c.__qualname__, set()).add(c)
        return {n for n, cs in seen.items() if len(cs) > 1}

    inst_coll = collisions(i for i, _ in pairs)
    model_coll = collisions(m for _, m in pairs)
    out: dict[str, dict[str, Any]] = {}
    for i_cls, m_cls in pairs:
        label = f"{_cls_label(i_cls, inst_coll)}|{_cls_label(m_cls, model_coll)}"
        out[label] = {
            "auto": auto_method(i_cls, m_cls),
            "available": list(available_methods(i_cls, m_cls)),
        }
    return dict(sorted(out.items()))


# ── Encoding / decoding ───────────────────────────────────────────────


def encode_values(values: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Encode arrays as shape + ``repr`` + ``float.hex`` lists.

    Args:
        values: Mapping from entry name to array.

    Returns:
        JSON-serialisable mapping.
    """
    out = {}
    for name in sorted(values):
        arr = values[name]
        flat = [float(v) for v in arr.reshape(-1)]
        out[name] = {
            "shape": list(arr.shape),
            "values": [repr(v) for v in flat],
            "hex": [v.hex() for v in flat],
        }
    return out


def load_golden(path: str | Path) -> dict[str, np.ndarray]:
    """Load a golden file, decoding each entry exactly from its hex strings.

    Args:
        path: Path to the JSON file written by this script.

    Returns:
        Mapping from entry name to a float64 numpy array of the stored shape.
    """
    with open(path) as fh:
        data = json.load(fh)
    return {
        name: np.array([float.fromhex(h) for h in entry["hex"]], dtype=np.float64).reshape(
            entry["shape"]
        )
        for name, entry in data["values"].items()
    }


def build_payload() -> dict[str, Any]:
    """Compute everything and assemble the JSON payload.

    Returns:
        The full golden document.
    """
    return {
        "meta": {
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "x64": bool(jax.config.jax_enable_x64),
            "created_by": "scripts/capture_phase4_golden.py",
        },
        "values": encode_values(compute_values()),
        "auto_methods": compute_auto_methods(),
    }


def compare(stored: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    """List every bit-level difference between two payloads.

    Args:
        stored: The payload read from disk.
        fresh: The freshly computed payload.

    Returns:
        Human-readable mismatch descriptions (empty when identical).
    """
    problems = []
    sv, fv = stored["values"], fresh["values"]
    for name in sorted(set(sv) | set(fv)):
        if name not in fv:
            problems.append(f"{name}: missing from recomputation")
        elif name not in sv:
            problems.append(f"{name}: not in stored file")
        elif sv[name]["shape"] != fv[name]["shape"]:
            problems.append(f"{name}: shape {sv[name]['shape']} -> {fv[name]['shape']}")
        elif sv[name]["hex"] != fv[name]["hex"]:
            for k, (a, b) in enumerate(zip(sv[name]["hex"], fv[name]["hex"])):
                if a != b:
                    problems.append(
                        f"{name}[{k}]: {float.fromhex(a)!r} -> {float.fromhex(b)!r}"
                    )
    sa, fa = stored["auto_methods"], fresh["auto_methods"]
    for pair in sorted(set(sa) | set(fa)):
        if sa.get(pair) != fa.get(pair):
            problems.append(f"auto_methods[{pair}]: {sa.get(pair)} -> {fa.get(pair)}")
    return problems


def main() -> None:
    """Command-line entry point: write the baseline or ``--check`` it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="golden JSON path (default: tests/data/phase4_golden.json)")
    parser.add_argument("--check", action="store_true",
                        help="recompute and compare bit-for-bit against --out")
    args = parser.parse_args()

    start = time.perf_counter()
    payload = build_payload()
    elapsed = time.perf_counter() - start
    n_values = len(payload["values"])
    n_pairs = len(payload["auto_methods"])

    if args.check:
        with open(args.out) as fh:
            stored = json.load(fh)
        problems = compare(stored, payload)
        for line in problems:
            print(f"MISMATCH {line}")
        print(f"{len(problems)} mismatches over {n_values} values and "
              f"{n_pairs} auto-method pairs ({elapsed:.1f}s)")
        sys.exit(1 if problems else 0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, sort_keys=True, indent=1)
        fh.write("\n")
    print(f"wrote {n_values} values and {n_pairs} auto-method pairs to "
          f"{args.out} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
