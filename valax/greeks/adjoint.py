"""Implicit-function-theorem adjoints: sensitivities to *market quotes*.

Where :mod:`valax.greeks.market_greeks` differentiates a price w.r.t. the
**calibrated** market (curve discount factors, surface nodes), this module
pushes that gradient one step further back — to the **quotes** the market was
calibrated from. Those are the numbers a desk actually trades, so this is the
layer that turns a node-space gradient into a hedge-space one.

The mechanism (one level)
-------------------------

Every VALAX calibration can be written as a residual driven to zero,

.. math::

    R(\\theta; \\text{inputs}) = 0,

with ``theta`` the solved unknowns. The implicit function theorem gives

.. math::

    \\frac{d\\theta}{dq} = -\\left(\\frac{\\partial R}{\\partial\\theta}\\right)^{-1}
                            \\frac{\\partial R}{\\partial q},

so for a price :math:`V` with node cotangent
:math:`\\bar\\theta = \\partial V/\\partial\\theta` the quote sensitivity is

.. math::

    \\bar q = -\\left(\\frac{\\partial R}{\\partial q}\\right)^{\\!T} \\lambda,
    \\qquad
    \\left(\\frac{\\partial R}{\\partial\\theta}\\right)^{\\!T} \\lambda = \\bar\\theta.

Two properties matter for cost and for correctness:

* **The optimiser is never differentiated.** Only the residual *at the
  converged solution* is. Newton iteration count is irrelevant to the
  sensitivity, and no tape records the solve.
* :math:`d\\theta/dq` **is never formed.** Both
  :math:`(\\partial R/\\partial\\theta)^T\\lambda` and
  :math:`(\\partial R/\\partial q)^T\\lambda` are vector-Jacobian products —
  single reverse sweeps through the residual. Cost is one transposed linear
  solve *regardless of how many quotes there are*, which is what makes this
  cheaper than the dense :func:`valax.curves.bootstrap_graph.quote_jacobian`
  (that one re-bootstraps once per quote).

What *is* materialised, deliberately, is :math:`\\partial R/\\partial\\theta`
itself: a square Jacobian that is factorised **once per market snapshot** and
then reused across every trade, each costing one back-substitution. See
:class:`CurveAdjoint`.

Residual units
--------------

VALAX residuals are *not* uniformly in quote units — a futures residual is a
rate difference (:math:`\\partial R/\\partial q = +I`) while a swap residual is
``rate * annuity - float_pv`` (PV units, so :math:`\\partial R/\\partial q` is
the annuity). This module therefore differentiates
:math:`\\partial R/\\partial q` rather than assuming :math:`-I`, so mixed-unit
systems come out correctly scaled without per-instrument special-casing.

Multi-curve
-----------

:func:`curve_quote_deltas` assembles the node cotangent across **every** curve
in the graph, so a discount-curve quote picks up both its direct effect and the
indirect one (moving an OIS quote moves the discount curve, which moves the
projection nodes required to reprice the IRS quotes). A bump that held
projection *nodes* fixed would miss the second part; the transposed solve
captures it automatically because the coupled system is solved jointly.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Float, Int
from jax import Array

from valax.core.market_state import MarketState
from valax.curves.bootstrap_graph import (
    curve_quote_vector,
    curve_residual_fn,
    log_dfs_from_graph,
)
from valax.curves.bootstrap_proto import BootstrapInstrument
from valax.curves.fixings import FixingHistory
from valax.curves.graph import CurveGraph, CurveSpec
from valax.greeks.market_greeks import market_gradient


# ── Level-generic machinery ──────────────────────────────────────────


class CalibrationLevel(eqx.Module):
    """One calibration written as ``R(theta; inputs) = 0``.

    The uniform shape every VALAX calibration is expressed in so that a
    single adjoint routine serves curve bootstraps, volatility stripping
    and model fits alike. Holding the residual, the solved unknowns and
    the inputs as three explicit members is what makes the partials of
    ``R`` evaluable at the solution.

    Attributes:
        residual_fn: Callable ``(theta, inputs) -> residual`` returning
            the residual vector. Must be the *same* code path the solver
            drove to zero — a re-implementation for the adjoint would
            silently decouple the two.
        theta: The solved unknowns, at the converged solution.
        inputs: Everything ``residual_fn`` treats as fixed: market
            quotes, upstream calibrated objects, and non-solved
            parameters.
    """

    residual_fn: Callable[[Any, Any], Float[Array, " n_residuals"]] = eqx.field(
        static=True
    )
    theta: Any
    inputs: Any


class LevelAdjoint(eqx.Module):
    """Result of one implicit-function-theorem step.

    Attributes:
        lam: The adjoint multiplier solving
            ``(dR/dtheta)^T lam = theta_bar``. One entry per residual.
        inputs_bar: ``-(dR/dinputs)^T lam`` — the sensitivity of the
            price to this level's inputs, shaped like
            :attr:`CalibrationLevel.inputs`.
        residual_norm: ``max|R|`` at ``theta``. A large value means the
            level was not converged tightly enough for the theorem to
            apply, and every sensitivity downstream of it is suspect.
    """

    lam: Float[Array, " n_residuals"]
    inputs_bar: Any
    residual_norm: Float[Array, ""]


def level_adjoint(
    level: CalibrationLevel,
    theta_bar: Any,
    *,
    solver: lx.AbstractLinearSolver | None = None,
) -> LevelAdjoint:
    """Differentiate one calibration level by the implicit function theorem.

    Solves ``(dR/dtheta)^T lam = theta_bar`` and contracts the result
    against ``dR/dinputs``. Both Jacobian contractions are
    vector-Jacobian products: no Jacobian of the inputs is formed, so the
    cost is independent of the number of inputs.

    Args:
        level: The converged calibration level.
        theta_bar: Cotangent of the price w.r.t. ``level.theta``, same
            pytree structure as ``level.theta``.
        solver: Linear solver for the transposed system. Defaults to
            ``lineax.AutoLinearSolver(well_posed=True)``.

    Returns:
        A :class:`LevelAdjoint` carrying ``lam``, the input sensitivities
        and the residual norm at the solution.

    Examples:
        >>> lvl = CalibrationLevel(residual_fn=r, theta=th, inputs=q)
        >>> adj = level_adjoint(lvl, theta_bar)
        >>> adj.inputs_bar.shape == q.shape
        True
    """
    if solver is None:
        solver = lx.AutoLinearSolver(well_posed=True)

    inputs = level.inputs

    def residual_of_theta(theta: Any) -> Float[Array, " n_residuals"]:
        return level.residual_fn(theta, inputs)

    residual_value, vjp_theta = jax.vjp(residual_of_theta, level.theta)

    # (dR/dtheta)^T as a linear operator: lam |-> vjp(lam).  Solving
    # against it yields lam without ever transposing a stored matrix.
    operator = lx.FunctionLinearOperator(
        lambda lam: vjp_theta(lam)[0],
        jax.eval_shape(lambda: residual_value),
    )
    lam = lx.linear_solve(operator, theta_bar, solver).value

    def residual_of_inputs(inp: Any) -> Float[Array, " n_residuals"]:
        return level.residual_fn(level.theta, inp)

    _, vjp_inputs = jax.vjp(residual_of_inputs, inputs)
    (inputs_bar,) = vjp_inputs(lam)

    return LevelAdjoint(
        lam=lam,
        inputs_bar=jax.tree.map(lambda x: -x, inputs_bar),
        residual_norm=jnp.max(jnp.abs(residual_value)),
    )


# ── Layer 1a: curve quote deltas ─────────────────────────────────────


class CurveAdjoint(eqx.Module):
    """Snapshot-level adjoint handle for a calibrated :class:`CurveGraph`.

    Materialises and factorises ``(dR_c/dtheta_c)^T`` **once**, so each
    subsequent trade costs one back-substitution plus one reverse sweep
    through the residual. For a portfolio this is the difference between
    one factorisation and one per trade; it is also the only part of the
    chain whose cost grows with the number of pillars rather than the
    number of trades.

    Build it with :func:`curve_adjoint` rather than directly.

    Attributes:
        log_dfs: The solver state ``theta_c`` — concatenated
            ``log(DF)`` across curves in ``curve_specs`` order, excluding
            each curve's reference-date pillar.
        quotes: The quote vector ``q_c``, in ``instruments`` order.
        residual_jacobian: ``dR_c/dtheta_c`` at the solution.
        jacobian_condition_number: 2-norm condition number of
            ``dR_c/dtheta_c``. Ill-conditioning destabilises quote
            sensitivities even when prices themselves look healthy, so it
            is surfaced here and not only at build time.
        operator: ``(dR_c/dtheta_c)^T`` as a ``lineax`` operator.
        factorisation: Pre-computed LU factorisation state for
            :attr:`operator`. This is the member that makes per-trade
            risk a back-substitution rather than a fresh decomposition.
        curve_specs: The specs defining the state layout.
        instruments: The bootstrap quotes defining the residual rows.
        fixings: Realised-fixings registry used by the residual.
        reference_date: Valuation date ordinal.
    """

    log_dfs: Float[Array, " n_theta"]
    quotes: Float[Array, " n_quotes"]
    residual_jacobian: Float[Array, "n_quotes n_theta"]
    jacobian_condition_number: Float[Array, ""]
    operator: lx.AbstractLinearOperator
    factorisation: Any
    curve_specs: tuple[CurveSpec, ...]
    instruments: tuple[BootstrapInstrument, ...]
    fixings: FixingHistory
    reference_date: Int[Array, ""]

    @property
    def residual_fn(self) -> Callable[
        [Float[Array, " n_theta"], Float[Array, " n_quotes"]],
        Float[Array, " n_quotes"],
    ]:
        """The Layer-1a residual ``R(theta_c, q_c)``, rebuilt on demand.

        Delegates to
        :func:`valax.curves.bootstrap_graph.curve_residual_fn`, the same
        entry point :func:`~valax.curves.bootstrap_graph.bootstrap_curve_graph`
        drives to zero.
        """
        return curve_residual_fn(
            self.reference_date,
            self.curve_specs,
            self.instruments,
            fixings=self.fixings,
        )

    def level(self) -> CalibrationLevel:
        """Expose this snapshot as a generic :class:`CalibrationLevel`."""
        return CalibrationLevel(
            residual_fn=self.residual_fn,
            theta=self.log_dfs,
            inputs=self.quotes,
        )


def curve_adjoint(
    graph: CurveGraph,
    reference_date: Int[Array, ""],
    curve_specs: Sequence[CurveSpec],
    instruments: Sequence[BootstrapInstrument],
    *,
    fixings: FixingHistory | None = None,
) -> CurveAdjoint:
    """Build a reusable adjoint handle from an already-calibrated graph.

    Does **not** re-solve: it recovers ``theta_c`` from ``graph`` and
    evaluates the residual Jacobian there. Build this once per market
    snapshot and pass it to :func:`curve_quote_deltas` for every trade.

    Args:
        graph: The calibrated :class:`CurveGraph`.
        reference_date: Valuation date as an integer ordinal scalar.
        curve_specs: The specs the graph was bootstrapped from; their
            order fixes the state layout.
        instruments: The bootstrap quotes; their order fixes the quote
            (column) layout of every sensitivity returned.
        fixings: Optional realised-fixings registry. Must match the one
            used at build time, or the differentiated residual will not
            be the solved one.

    Returns:
        A :class:`CurveAdjoint` with the residual Jacobian and its
        condition number evaluated at the solution.

    Examples:
        >>> graph, diag = bootstrap_curve_graph(ref, specs, quotes)
        >>> adj = curve_adjoint(graph, ref, specs, quotes)
        >>> bool(adj.jacobian_condition_number < 1e10)
        True
    """
    specs = tuple(curve_specs)
    insts = tuple(instruments)

    log_dfs = log_dfs_from_graph(graph, specs)
    quotes = curve_quote_vector(insts)
    residual = curve_residual_fn(
        reference_date, specs, insts, fixings=fixings
    )

    jacobian = jax.jacrev(residual, argnums=0)(log_dfs, quotes)

    # Factorise once, here -- not once per trade (spec A-7).
    operator = lx.MatrixLinearOperator(jacobian.T)
    factorisation = lx.LU().init(operator, {})

    return CurveAdjoint(
        log_dfs=log_dfs,
        quotes=quotes,
        residual_jacobian=jacobian,
        jacobian_condition_number=jnp.linalg.cond(jacobian),
        operator=operator,
        factorisation=factorisation,
        curve_specs=specs,
        instruments=insts,
        fixings=fixings,
        reference_date=jnp.asarray(reference_date, dtype=jnp.int32),
    )


def curve_log_df_cotangent(
    market_grad: MarketState,
    market: MarketState,
    curve_specs: Sequence[CurveSpec],
) -> Float[Array, " n_theta"]:
    """Convert a ``MarketState`` gradient into the solver's log-DF cotangent.

    The bootstrap solves in ``log(DF)`` space with each curve's
    reference-date pillar held out (``DF == 1`` there by construction, so
    it is not a free unknown), while :func:`~valax.greeks.market_greeks.market_gradient`
    reports ``dV/dDF`` over *all* pillars. The change of variables is
    therefore a drop of the first entry and a multiply by ``DF``:

    .. math::

        \\frac{\\partial V}{\\partial \\theta_j}
            = \\frac{\\partial V}{\\partial DF_{j+1}} \\, DF_{j+1}.

    Curves present in the market but untouched by the trade contribute
    zeros, which is what lets a single transposed solve serve every
    instrument in a portfolio.

    Args:
        market_grad: A ``MarketState``-shaped gradient pytree.
        market: The market the gradient was taken at (supplies ``DF``).
        curve_specs: Specs fixing the concatenation order.

    Returns:
        The cotangent ``theta_bar``, matching
        :attr:`CurveAdjoint.log_dfs` in length and layout.

    Raises:
        KeyError: If a spec names a curve absent from ``market``.
    """
    chunks: list[Float[Array, " n"]] = []
    for spec in curve_specs:
        cid = spec.curve_id
        if cid not in market.curves.curves:
            raise KeyError(
                f"curve_log_df_cotangent: curve {cid!r} is in curve_specs "
                f"but not in the market (available: "
                f"{sorted(market.curves.curves)})."
            )
        n_pillars = int(spec.pillar_dates.shape[0])
        dfs = market.curves.curves[cid].discount_factors
        grad_curve = market_grad.curves.curves.get(cid)
        grad_dfs = None if grad_curve is None else grad_curve.discount_factors
        if grad_dfs is None:
            chunks.append(jnp.zeros((n_pillars,)))
        else:
            chunks.append(grad_dfs[1:] * dfs[1:])
    return (
        jnp.concatenate(chunks) if chunks else jnp.zeros((0,))
    )


def quote_deltas_from_cotangent(
    adjoint: CurveAdjoint,
    theta_bar: Float[Array, " n_theta"],
    *,
    solver: lx.AbstractLinearSolver | None = None,
) -> Float[Array, " n_quotes"]:
    """Push a log-DF cotangent back to quote space.

    The transposed-solve half of the chain, split out so a portfolio can
    accumulate ``theta_bar`` over all trades first and then pay for a
    single solve (the curve build is shared, so its adjoint should be
    too).

    Args:
        adjoint: Snapshot handle from :func:`curve_adjoint`.
        theta_bar: Node cotangent in log-DF space, e.g. from
            :func:`curve_log_df_cotangent`.
        solver: Linear solver for the transposed system. Defaults to
            ``lineax.LU`` reusing ``adjoint.factorisation``, which is
            exact for the square well-posed curve system and costs only
            a back-substitution. Passing a solver explicitly forces a
            fresh decomposition.

    Returns:
        ``q_bar`` — the price sensitivity to each market quote, in
        ``adjoint.instruments`` order and in each quote's own units.
    """
    if solver is None:
        lam = lx.linear_solve(
            adjoint.operator,
            theta_bar,
            lx.LU(),
            state=adjoint.factorisation,
        ).value
    else:
        lam = lx.linear_solve(adjoint.operator, theta_bar, solver).value

    residual = adjoint.residual_fn

    def residual_of_quotes(
        q: Float[Array, " n_quotes"],
    ) -> Float[Array, " n_quotes"]:
        return residual(adjoint.log_dfs, q)

    _, vjp_quotes = jax.vjp(residual_of_quotes, adjoint.quotes)
    (quotes_bar,) = vjp_quotes(lam)
    return -quotes_bar


def curve_quote_deltas(
    instrument: Any,
    model: Any,
    market: MarketState,
    *,
    adjoint: CurveAdjoint,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
    solver: lx.AbstractLinearSolver | None = None,
) -> Float[Array, " n_quotes"]:
    """Sensitivity of a price to every market quote the curves were built from.

    The end-to-end Layer-1a chain: one reverse sweep through the pricer
    for the node cotangent, one transposed linear solve for the adjoint
    multiplier, one reverse sweep through the residual for the quote
    contraction. Equivalently, delta expressed in the hedge instruments
    the desk actually trades rather than in curve pillars.

    This differs from bumping pillar nodes directly: holding the *other
    quotes* fixed (as a market bump does) forces the whole graph to
    re-solve, so a discount-curve quote picks up its indirect effect on
    the projection curves. Bumping nodes while holding other nodes fixed
    reports a different — and for hedging, wrong — number.

    Args:
        instrument: A VALAX instrument pytree.
        model: A VALAX model pytree, or ``None`` for curve-discounted
            pricing.
        market: The calibrated market, whose curves came from
            ``adjoint``'s graph.
        adjoint: Snapshot handle from :func:`curve_adjoint`, carrying the
            factorisable residual Jacobian. Reuse one handle across
            trades.
        method: ``"analytic" | "lattice" | "pde" | "mc" | "auto"``.
        config: Method-specific numerical settings.
        key: PRNG key (required for ``method="mc"``).
        solver: Linear solver for the transposed system.

    Returns:
        ``q_bar`` of shape ``(n_quotes,)``, aligned with
        ``adjoint.instruments``.

    Examples:
        >>> adj = curve_adjoint(graph, ref, specs, quotes)
        >>> q_bar = curve_quote_deltas(swap, None, market, adjoint=adj)
        >>> q_bar.shape == (len(quotes),)
        True
    """
    grad = market_gradient(
        instrument, model, market, method=method, config=config, key=key
    )
    theta_bar = curve_log_df_cotangent(grad, market, adjoint.curve_specs)
    return quote_deltas_from_cotangent(adjoint, theta_bar, solver=solver)
