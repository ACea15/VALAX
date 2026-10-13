r"""Quote-space risk through a calibration chain — two independent paths.

A price depends on market quotes ``q`` and held-fixed inputs ``p`` only
through the calibration chain (:mod:`valax.calibration.chain`):

.. math::

    V = \mathrm{price}\bigl(I,\; M(\theta_m),\; \mathcal{M}(\theta_1, \dots, \theta_L)\bigr),
    \qquad R_\ell(\theta_\ell;\, q_\ell,\, \mathcal{M}_{<\ell},\, p_\ell) = 0.

This module computes :math:`(\bar q, \bar p) = (\partial V/\partial q,\,
\partial V/\partial p)` two ways, deliberately sharing as little code as
possible so that each validates the other:

**Adjoint path** (:func:`chain_adjoint`, :func:`quote_risk_adjoint`). One
reverse sweep through the pricer gives the market cotangent (and the model
cotangent, if the model was calibrated). The levels are then visited in
reverse: for each, the cotangent on its calibrated object is pulled back to
:math:`\bar\theta_\ell`, one transposed solve gives :math:`\lambda_\ell`, and
one VJP of the residual splits :math:`-(\partial R_\ell/\partial\,\cdot)^T
\lambda_\ell` into quote, upstream-market and fixed-input adjoints. Upstream
cotangents accumulate into the running market cotangent, which is how a vol
level's forward dependence reaches the curve quotes. The market cotangent is
linear in the trade, so a whole book's cotangents can be summed first and the
chain walked **once**.

**Autodiff path** (:func:`quote_risk_autodiff`). ``jax.grad`` of
``price(calibrate(q, p))`` w.r.t. ``(q, p)``, relying on the implicit adjoint
``optimistix`` attaches to every solve. No level is inspected.

The paths agree to solver tolerance on exact levels. On least-squares levels
the adjoint path uses the full Hessian of the first-order condition; a material
disagreement there means a poorly fitted level. Neither path survives a
non-converged calibration — the implicit function theorem needs a solution —
so check ``CalibratedMarket.converged`` (or ``.check()``) first; the adjoint
path also reports each level's stationarity residual. Compare the two with
:func:`reconcile_quote_risk`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Float
from jax import Array

from valax.calibration.chain import (
    CalibratedMarket,
    CalibrationChain,
    CalibrationNode,
    calibrate_market,
)
from valax.calibration.level import LevelInputs, level_adjoint
from valax.calibration.result import CalibrationResult
from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.market.quotes import QuoteSet


# ── Pytree helpers (None-aware cotangent algebra) ────────────────────


def _is_none(x: Any) -> bool:
    return x is None


def tree_vdot(bar: Any, primal: Any) -> Float[Array, ""]:
    """Inner product of a cotangent with a primal, skipping ``None`` leaves.

    Gradients from ``eqx.filter_grad`` carry ``None`` at non-differentiable
    leaves (integer dates, static data). This contracts every array leaf of
    ``bar`` with the matching leaf of ``primal`` and ignores the rest, which
    makes ``jax.grad`` of ``tree_vdot(bar, f(x))`` a VJP of ``f`` that
    tolerates partial cotangents.

    Args:
        bar: Cotangent pytree (``None`` leaves allowed), or ``None``.
        primal: Primal pytree with ``bar`` as a prefix.

    Returns:
        The scalar inner product.
    """
    if bar is None:
        return jnp.asarray(0.0)
    terms = jax.tree.leaves(
        jax.tree.map(
            lambda b, p: None if b is None else jnp.sum(b * p),
            bar,
            primal,
            is_leaf=_is_none,
        )
    )
    return sum(terms, jnp.asarray(0.0))


def tree_add(a: Any, b: Any) -> Any:
    """Leaf-wise sum of two cotangents, treating ``None`` as zero.

    Args:
        a: Cotangent pytree or ``None``.
        b: Cotangent pytree of the same structure, or ``None``.

    Returns:
        The sum.
    """
    if a is None:
        return b
    if b is None:
        return a

    def add(x: Any, y: Any) -> Any:
        if x is None:
            return y
        if y is None:
            return x
        return x + y

    return jax.tree.map(add, a, b, is_leaf=_is_none)


# ── Result container ─────────────────────────────────────────────────


class QuoteRisk(eqx.Module):
    """Price sensitivities to market quotes and held-fixed inputs.

    Attributes:
        quotes: :math:`\\partial V/\\partial q` keyed like the quotes.
        fixed: :math:`\\partial V/\\partial p` keyed like the fixed inputs.
        level_residuals: ``level name -> max|R|`` at the solution, as seen by
            the adjoint path (empty for the autodiff path). A value far from
            zero means the implicit function theorem did not apply at that
            level.
        method: ``"adjoint"`` or ``"autodiff"``. Static.
    """

    quotes: QuoteSet
    fixed: QuoteSet
    level_residuals: dict = eqx.field(default_factory=dict)
    method: str = eqx.field(static=True, default="adjoint")


# ── Adjoint path ─────────────────────────────────────────────────────


def _retarget(result: CalibrationResult, market: MarketState) -> CalibrationResult:
    """Point a level's upstream input at the final market.

    A level is calibrated against the market *as built so far*; its residual
    only reads slots installed before it, which are identical in the final
    market. Re-pointing makes every upstream cotangent ``MarketState``-shaped
    like the price's, so they can be added.
    """
    if result.inputs.upstream is None:
        return result
    inputs = LevelInputs(
        quotes=result.inputs.quotes,
        upstream=market,
        fixed=result.inputs.fixed,
        data=result.inputs.data,
    )
    return dataclasses.replace(result, inputs=inputs)


def _pull_back(
    result: CalibrationResult,
    target_fn: Callable[[Any], Any],
    bar: Any,
    solver: lx.AbstractLinearSolver | None,
) -> tuple[LevelInputs, Float[Array, ""]]:
    """Pull a cotangent on ``target_fn(value)`` back to the level's inputs.

    Two contributions: the explicit one through ``assemble_fn`` (objects that
    read their inputs directly, e.g. a surface storing its forwards) and, for
    solved levels, the implicit one through ``theta`` by the implicit function
    theorem. The cotangent of ``inputs.data`` (trade data such as a basket's
    strikes) is computed but never accumulated: it is not a risk input.
    """

    def scalar(theta_inputs: tuple) -> Float[Array, ""]:
        theta, inputs = theta_inputs
        return tree_vdot(bar, target_fn(result.assemble_fn(theta, inputs)))

    theta_bar, inputs_bar = eqx.filter_grad(scalar)((result.theta, result.inputs))
    if result.is_explicit:
        return inputs_bar, jnp.asarray(0.0)
    adj = level_adjoint(result.level, theta_bar, solver=solver)
    return tree_add(inputs_bar, adj.inputs_bar), adj.residual_norm


def _accumulate(total: dict, part: Any) -> dict:
    """Add a ``QuoteSet``-shaped cotangent into a key -> value dict."""
    if part is None:
        return total
    for k, v in part.values.items():
        if v is not None:
            total[k] = total[k] + v
    return total


@eqx.filter_jit
def _chain_adjoint_jit(
    calibrated: CalibratedMarket,
    market_bar: MarketState | None,
    model_levels: tuple,
    solver: lx.AbstractLinearSolver | None,
) -> QuoteRisk:
    """Compiled body of :func:`chain_adjoint` (one trace per chain shape)."""
    return _chain_adjoint_impl(calibrated, market_bar, model_levels, solver)


def chain_adjoint(
    calibrated: CalibratedMarket,
    market_bar: MarketState | None,
    *,
    model_levels: Sequence[tuple[CalibrationResult, Any]] = (),
    solver: lx.AbstractLinearSolver | None = None,
) -> QuoteRisk:
    """Push market (and model) cotangents back through the chain to quotes.

    The generic reverse sweep of the calibration chain: model levels first,
    then market levels in reverse order, each by one implicit-function step
    (or plain AD for an explicit level). Linear in the cotangents, so pass
    the **sum** over a book's trades to get book-level quote risk for the
    price of one sweep.

    Args:
        calibrated: The calibrated market from
            :func:`~valax.calibration.chain.calibrate_market`.
        market_bar: ``∂V/∂market``, a ``MarketState``-shaped gradient of the
            price at ``calibrated.market`` (e.g. from
            :func:`~valax.greeks.market_greeks.market_gradient`), or ``None``.
        model_levels: ``(model_result, model_bar)`` pairs for calibrated
            models: each model's result from
            :func:`~valax.calibration.chain.calibrate_model` and the price's
            gradient w.r.t. that model.
        solver: Linear solver for the transposed systems.

    Returns:
        :class:`QuoteRisk` with ``method="adjoint"``.
    """
    return _chain_adjoint_jit(
        calibrated, market_bar, tuple(tuple(pair) for pair in model_levels), solver
    )


def _chain_adjoint_impl(
    calibrated: CalibratedMarket,
    market_bar: MarketState | None,
    model_levels: tuple,
    solver: lx.AbstractLinearSolver | None,
) -> QuoteRisk:
    """Reverse sweep of :func:`chain_adjoint` (traceable)."""
    final = calibrated.market
    q_keys = set(calibrated.quotes.values)
    p_keys = set(calibrated.fixed.values)
    for res, _ in model_levels:
        q_keys |= set(res.inputs.quotes.values)
        p_keys |= set(res.inputs.fixed.values) if res.inputs.fixed is not None else set()
    q_bar = {k: jnp.asarray(0.0) for k in q_keys}
    p_bar = {k: jnp.asarray(0.0) for k in p_keys}
    norms: dict = {}
    running = market_bar

    for res, model_bar in model_levels:
        inputs_bar, norm = _pull_back(res, lambda v: v, model_bar, solver)
        q_bar = _accumulate(q_bar, inputs_bar.quotes)
        p_bar = _accumulate(p_bar, inputs_bar.fixed)
        running = tree_add(running, inputs_bar.upstream)
        norms[res.name] = norm

    for node, res in reversed(list(zip(calibrated.chain.nodes, calibrated.results))):
        res = _retarget(res, final)
        install = node.install_fn

        def target(value: Any, _install=install) -> MarketState:
            return _install(final, value)

        inputs_bar, norm = _pull_back(res, target, running, solver)
        q_bar = _accumulate(q_bar, inputs_bar.quotes)
        p_bar = _accumulate(p_bar, inputs_bar.fixed)
        running = tree_add(running, inputs_bar.upstream)
        norms[res.name] = norm

    return QuoteRisk(
        quotes=QuoteSet(values=q_bar),
        fixed=QuoteSet(values=p_bar),
        level_residuals=norms,
        method="adjoint",
    )


@eqx.filter_jit
def _price_cotangents(
    instrument: Any,
    model: Any,
    market: MarketState,
    method: str,
    config: Any,
    key: Any,
) -> tuple[Any, MarketState]:
    """``(∂V/∂model, ∂V/∂market)`` in one reverse sweep through ``price``."""

    def value(model_market: tuple) -> Float[Array, ""]:
        m, mkt = model_market
        return price(instrument, m, mkt, method=method, config=config, key=key).price

    return eqx.filter_grad(value)((model, market))


def quote_risk_adjoint(
    instrument: Any,
    model: Any,
    calibrated: CalibratedMarket,
    *,
    model_result: CalibrationResult | None = None,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
    solver: lx.AbstractLinearSolver | None = None,
) -> QuoteRisk:
    """Quote risk of one trade by the adjoint path.

    One reverse sweep through ``price`` for the market (and model)
    cotangent, then :func:`chain_adjoint`.

    Args:
        instrument: The trade.
        model: The pricing model (``None`` for curve-discounted pricing).
            Ignored when ``model_result`` is given.
        calibrated: The calibrated market.
        model_result: A calibrated model's result; its value is used as the
            model and its level joins the sweep.
        method: Pricing method.
        config: Pricing config.
        key: PRNG key (Monte Carlo).
        solver: Linear solver for the transposed systems.

    Returns:
        :class:`QuoteRisk` with ``method="adjoint"``.
    """
    if model_result is not None:
        model = model_result.value
    model_bar, market_bar = _price_cotangents(
        instrument, model, calibrated.market, method, config, key
    )
    levels = () if model_result is None else ((model_result, model_bar),)
    return chain_adjoint(
        calibrated, market_bar, model_levels=levels, solver=solver
    )


# ── Autodiff path ────────────────────────────────────────────────────


def quote_risk_autodiff(
    instrument: Any,
    model: Any,
    chain: CalibrationChain,
    quotes: QuoteSet,
    fixed: QuoteSet | None = None,
    *,
    model_node: CalibrationNode | None = None,
    base: MarketState | None = None,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> QuoteRisk:
    """Quote risk of one trade by differentiating through the calibration.

    ``jax.grad`` of quotes → :func:`~valax.calibration.chain.calibrate_market`
    → (optionally) model calibration → ``price``. Each solve is differentiated
    by the implicit adjoint ``optimistix`` attaches to it; no level object is
    used, which is what makes this an independent check of
    :func:`quote_risk_adjoint`.

    Args:
        instrument: The trade.
        model: The pricing model; ignored when ``model_node`` is given.
        chain: The market calibration chain.
        quotes: Market quotes.
        fixed: Held-fixed inputs.
        model_node: Optional model-scope node calibrated on the market.
        base: Optional base market for the chain.
        method: Pricing method.
        config: Pricing config.
        key: PRNG key (Monte Carlo).

    Returns:
        :class:`QuoteRisk` with ``method="autodiff"``.
    """
    fixed = QuoteSet() if fixed is None else fixed
    q_bar, p_bar = _autodiff_grads(
        instrument, model, chain, quotes, fixed, model_node, base,
        method, config, key,
    )
    return QuoteRisk(quotes=q_bar, fixed=p_bar, method="autodiff")


@eqx.filter_jit
def _autodiff_grads(
    instrument: Any,
    model: Any,
    chain: CalibrationChain,
    quotes: QuoteSet,
    fixed: QuoteSet,
    model_node: CalibrationNode | None,
    base: MarketState | None,
    method: str,
    config: Any,
    key: Any,
) -> tuple[QuoteSet, QuoteSet]:
    """Compiled ``jax.grad`` of quotes → calibration → price."""

    def value(q: QuoteSet, p: QuoteSet) -> Float[Array, ""]:
        calibrated = calibrate_market(chain, q, p, base=base)
        m = model
        if model_node is not None:
            m = model_node.calibrate(q, calibrated.market, p).value
        return price(
            instrument, m, calibrated.market, method=method, config=config, key=key
        ).price

    return jax.grad(value, argnums=(0, 1))(quotes, fixed)


# ── Reconciliation ───────────────────────────────────────────────────


class QuoteRiskReconciliation(eqx.Module):
    """Comparison of two :class:`QuoteRisk` results (eager, reporting only).

    Attributes:
        max_abs_diff: Largest absolute difference across quotes and fixed
            inputs.
        mismatched: Identifiers whose difference exceeds
            ``atol + rtol * max(|a|, |b|)``.
        ok: ``True`` when nothing is mismatched.
    """

    max_abs_diff: float = eqx.field(static=True)
    mismatched: tuple[str, ...] = eqx.field(static=True)
    ok: bool = eqx.field(static=True)


def reconcile_quote_risk(
    a: QuoteRisk,
    b: QuoteRisk,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> QuoteRiskReconciliation:
    """Compare two quote-risk results key by key.

    Use it to cross-check the adjoint and autodiff paths: report both and the
    mismatch rather than silently trusting one.

    Args:
        a: First result.
        b: Second result.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        A :class:`QuoteRiskReconciliation`.
    """
    worst = 0.0
    bad: list[str] = []
    for prefix, sa, sb in (("q:", a.quotes, b.quotes), ("p:", a.fixed, b.fixed)):
        for k in sorted(set(sa.values) | set(sb.values)):
            va = float(np.asarray(sa.get(k)))
            vb = float(np.asarray(sb.get(k)))
            diff = abs(va - vb)
            worst = max(worst, diff)
            if diff > atol + rtol * max(abs(va), abs(vb)):
                bad.append(prefix + k)
    return QuoteRiskReconciliation(
        max_abs_diff=worst, mismatched=tuple(bad), ok=not bad
    )


__all__ = [
    "QuoteRisk",
    "QuoteRiskReconciliation",
    "chain_adjoint",
    "quote_risk_adjoint",
    "quote_risk_autodiff",
    "reconcile_quote_risk",
    "tree_add",
    "tree_vdot",
]
