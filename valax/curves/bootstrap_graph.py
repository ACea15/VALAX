"""Joint multi-curve Newton solver.

This module ships the central deliverable of MC-Curves-2: a single
Newton solve over the concatenated log-DFs of *every* curve in the
graph.  It replaces the sequential dual-curve pipeline
(:func:`valax.curves.multi_curve.bootstrap_multi_curve`) that could
not handle joint constraints such as tenor-basis swaps or
cross-currency basis swaps.

Why a joint solve?

* A **tenor basis swap** (e.g. 3M-vs-6M SOFR) constrains both the 3M
  and the 6M forward curve simultaneously.  There is no ordering of
  single-curve solves that produces the right answer.
* A **cross-currency basis swap** touches two OIS discount curves,
  two forward curves, and FX spot — a four-curve joint constraint.
* An **FX forward** ties two short-end curves via covered interest
  parity.

Once the solver is graph-shaped, all of these are just instruments
that return a residual; the solver is agnostic to instrument type
via the :class:`~valax.curves.bootstrap_proto.BootstrapInstrument`
protocol.

The residual system is well-determined when the total number of
instruments equals the total number of pillar unknowns across all
curves in the graph.  Working in log-DF space guarantees positive
discount factors throughout the Newton iteration and matches the
log-linear interpolation of :class:`DiscountCurve`.

Autodiff notes:

* ``optimistix.root_find`` defaults to ``ImplicitAdjoint`` so
  ``jax.grad`` of any function of the calibrated graph w.r.t. a
  quote rate costs *one linear solve* independent of Newton iteration
  count.  This is the mechanism behind :func:`quote_jacobian`.
* Every instrument's residual is JAX-traceable (no Python-level
  branching on traced values), so the concatenated residual vector
  is fully differentiable.

See ``production.md`` §11.4 for the design spec and
``docs/roadmap.md`` ("Multi-Curve Framework — What Ships, What's Left")
for the status of the wider multi-curve workstream.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Float, Int
from jax import Array

from valax.curves.bootstrap_proto import BootstrapInstrument
from valax.curves.discount import (
    DISCOUNT_CURVE_INTERP,
    DiscountCurve,
    zero_rate,
)
from valax.curves.fixings import FixingHistory, empty_fixing_history
from valax.curves.graph import CurveGraph, CurveSpec
from valax.dates.daycounts import year_fraction


# ── Diagnostics container ────────────────────────────────────────────


class CurveBuildDiagnostics(eqx.Module):
    """Per-curve and per-instrument repricing diagnostics.

    Returned alongside the calibrated :class:`CurveGraph` by
    :func:`bootstrap_curve_graph`.  Production-grade diagnostics
    covering both convergence health (``residuals``,
    ``max_abs_residual``, ``rmse``, ``n_steps``, ``converged``) and
    curve well-posedness (``jacobian_condition_number``), plus a
    per-instrument fit table in the quote's native units
    (``fitted_quotes``, ``quoted_values``) for desk sign-off.

    The relationship
    ``fitted_quotes[i] = quoted_values[i] - residuals[i] / slope[i]``
    with ``slope[i] = d(inst.residual)/d(quote)`` at the calibrated
    graph is exact for every VALAX quote type — each residual is
    linear in its primary quote scalar (rate / spread /
    futures_rate / quoted_forward / far_rate / jump_size).  Hence
    ``fitted_quotes - quoted_values`` is exactly the quote-unit
    fit error, expressed in the same units the desk trades
    (e.g. rate points, basis points).

    Attributes:
        residuals: Final per-instrument residuals (shape
            ``(n_instruments,)``).  Zero at convergence.
        max_abs_residual: Scalar ``max(|residuals|)`` — a quick
            single-number convergence check.
        rmse: Root-mean-square of ``residuals`` — a smoother
            convergence indicator than ``max_abs_residual``.
        n_steps: Number of Newton iterations consumed
            (``jnp.asarray(-1)`` if the solver did not populate stats).
        converged: Boolean flag reported by the underlying
            ``optimistix`` solver.
        fitted_quotes: Per-instrument quote value implied by the
            calibrated graph (shape ``(n_instruments,)``), in each
            instrument's native quote units.  Equals
            ``quoted_values`` up to Newton tolerance at convergence.
        quoted_values: Per-instrument input quote scalars extracted
            via :data:`_QUOTE_FIELD` (shape ``(n_instruments,)``).
        jacobian_condition_number: 2-norm condition number of the
            residual Jacobian ``d(residuals)/d(log_dfs)`` at the
            solution.  A large value flags an ill-posed curve build —
            typically caused by pillars placed too close together or an
            instrument that constrains an already-anchored region.  Pass
            it to :func:`valax.greeks.adjoint.condition_alert` to test it
            against the shared
            :data:`~valax.greeks.adjoint.ILL_CONDITIONED_THRESHOLD`
            rather than against a locally restated number.
    """

    residuals: Float[Array, " n_instruments"]
    max_abs_residual: Float[Array, ""]
    rmse: Float[Array, ""]
    n_steps: Int[Array, ""]
    fitted_quotes: Float[Array, " n_instruments"]
    quoted_values: Float[Array, " n_instruments"]
    jacobian_condition_number: Float[Array, ""]
    converged: bool = eqx.field(static=True)


# ── Internals ────────────────────────────────────────────────────────


def _validate_square(
    curve_specs: Sequence[CurveSpec],
    instruments: Sequence[BootstrapInstrument],
) -> None:
    """Ensure the residual system is square."""
    n_pillars = sum(int(s.pillar_dates.shape[0]) for s in curve_specs)
    n_instruments = len(instruments)
    if n_instruments != n_pillars:
        raise ValueError(
            f"Joint bootstrap requires one instrument per pillar: got "
            f"{n_instruments} instruments for {n_pillars} pillars across "
            f"{len(curve_specs)} curve(s)."
        )


def _validate_curves_touched(
    curve_specs: Sequence[CurveSpec],
    instruments: Sequence[BootstrapInstrument],
) -> None:
    """Ensure every ``inst.curves_touched`` entry names a spec in the graph."""
    known = {s.curve_id for s in curve_specs}
    for i, inst in enumerate(instruments):
        for cid in inst.curves_touched:
            if cid not in known:
                raise ValueError(
                    f"Instrument #{i} ({type(inst).__name__}) touches "
                    f"unknown curve {cid!r}.  Known curves: "
                    f"{sorted(known)}."
                )


def _spec_offsets(curve_specs: Sequence[CurveSpec]) -> tuple[int, ...]:
    """Return cumulative offsets into the flat log-DF state vector.

    ``offsets[i]`` is the starting index of ``curve_specs[i]``'s pillars
    in the flat state; ``offsets[-1]`` is the total length.
    """
    offsets = [0]
    for s in curve_specs:
        offsets.append(offsets[-1] + int(s.pillar_dates.shape[0]))
    return tuple(offsets)


def _build_graph_from_log_dfs(
    log_dfs: Float[Array, " n_total"],
    curve_specs: Sequence[CurveSpec],
    offsets: tuple[int, ...],
    reference_date: Int[Array, ""],
) -> CurveGraph:
    """Reconstruct a :class:`CurveGraph` from the flat log-DF state.

    Prepends ``DF == 1`` at the reference date to each curve so that
    :class:`DiscountCurve` has the reference pillar it needs for
    interpolation semantics.
    """
    ref_int = reference_date.astype(jnp.int32)
    curves: dict[str, DiscountCurve] = {}
    for i, spec in enumerate(curve_specs):
        # This is the one place a declared interpolation mode would have
        # to be forwarded, and ``DiscountCurve`` has no field to forward
        # it to — it hard-codes log-linear-in-DF.  Today that is not a
        # defect, because ``CurveSpec.__check_init__`` admits only the
        # mode the curve implements.  The guard exists so that widening
        # ``_SUPPORTED_INTERP`` without first teaching ``DiscountCurve``
        # to dispatch fails loudly here, instead of silently honouring a
        # mode the caller did not ask for.
        if spec.interp != DISCOUNT_CURVE_INTERP:
            raise NotImplementedError(
                f"CurveSpec(curve_id={spec.curve_id!r}) declares "
                f"interp={spec.interp!r}, but DiscountCurve only "
                f"implements {DISCOUNT_CURVE_INTERP!r}. Give "
                f"DiscountCurve a static `interp` field and dispatch on "
                f"it in its __call__ before admitting this mode in "
                f"valax.curves.graph._SUPPORTED_INTERP."
            )
        chunk = log_dfs[offsets[i] : offsets[i + 1]]
        dfs = jnp.concatenate([jnp.ones(1), jnp.exp(chunk)])
        dates = jnp.concatenate(
            [ref_int[None], spec.pillar_dates.astype(jnp.int32)]
        )
        curves[spec.curve_id] = DiscountCurve(
            pillar_dates=dates,
            discount_factors=dfs,
            reference_date=ref_int,
            day_count=spec.day_count,
        )
    return CurveGraph(curves=curves)


def _graph_residual(
    log_dfs: Float[Array, " n_total"],
    curve_specs: Sequence[CurveSpec],
    offsets: tuple[int, ...],
    instruments: Sequence[BootstrapInstrument],
    fixings: FixingHistory,
    ref_date: Int[Array, ""],
) -> Float[Array, " n_instruments"]:
    """The residual vector ``R(theta_c)`` of the joint curve system.

    The single implementation of the Layer-1a residual.  Both
    :func:`bootstrap_curve_graph` (which drives it to zero) and
    :func:`curve_residual_fn` (which exposes it for adjoint
    differentiation) route through here, so the solved system and the
    differentiated system are the *same code path* by construction.
    """
    graph = _build_graph_from_log_dfs(
        log_dfs, curve_specs, offsets, ref_date
    )
    return jnp.stack(
        [inst.residual(graph, fixings, ref_date) for inst in instruments]
    )


def _default_initial_guess(
    reference_date: Int[Array, ""],
    curve_specs: Sequence[CurveSpec],
) -> Float[Array, " n_total"]:
    """Flat 4% guess in log-DF space, concatenated per spec order."""
    chunks: list[Float[Array, " n"]] = []
    for spec in curve_specs:
        pillar_times = year_fraction(
            reference_date, spec.pillar_dates, spec.day_count
        )
        chunks.append(-0.04 * pillar_times)
    return jnp.concatenate(chunks) if chunks else jnp.zeros((0,))


def _pack_initial_guess(
    initial_guess: Mapping[str, Float[Array, " n"]] | None,
    reference_date: Int[Array, ""],
    curve_specs: Sequence[CurveSpec],
) -> Float[Array, " n_total"]:
    """Convert per-curve initial guesses into the flat state vector.

    Missing curves fall back to the flat-4% default; provided arrays
    must match their spec's pillar count.
    """
    if initial_guess is None:
        return _default_initial_guess(reference_date, curve_specs)

    default = _default_initial_guess(reference_date, curve_specs)
    offsets = _spec_offsets(curve_specs)
    result = default
    for i, spec in enumerate(curve_specs):
        if spec.curve_id in initial_guess:
            provided = jnp.asarray(initial_guess[spec.curve_id])
            expected = int(spec.pillar_dates.shape[0])
            if provided.shape != (expected,):
                raise ValueError(
                    f"initial_guess[{spec.curve_id!r}] has shape "
                    f"{provided.shape}, expected ({expected},)."
                )
            result = result.at[offsets[i] : offsets[i + 1]].set(provided)
    return result


# ── Quote-field dispatch ─────────────────────────────────────────────


# Attribute name carrying the primary quote scalar for each supported
# bootstrap-instrument type.  Used by :func:`quote_jacobian` (for the
# reverse-mode Jacobian rows) and by :func:`bootstrap_curve_graph` (for
# the ``fitted_quotes`` / ``quoted_values`` diagnostic fields).  Any
# type whose class name is not registered here cannot participate in
# either.  The FXSwap entry uses ``far_rate`` because market convention
# holds the near leg at spot and quotes the swap points, which
# effectively bump the far leg.
_QUOTE_FIELD: dict[str, str] = {
    "DepositRate": "rate",
    "FRA": "rate",
    "SwapRate": "rate",
    "OISSwapRate": "rate",
    "IborSwapRate": "rate",
    "MoneyMarketFuture": "futures_rate",
    "TenorBasisSwap": "spread",
    "CrossCurrencyBasisSwap": "spread",
    "FXForward": "quoted_forward",
    "FXSwap": "far_rate",
    "TurnInstrument": "jump_size",
}


def _quote_field_name(inst: BootstrapInstrument) -> str:
    """Return the attribute name carrying ``inst``'s primary quote scalar.

    Raises:
        TypeError: If ``inst``'s class name is not registered in
            :data:`_QUOTE_FIELD`.
    """
    name = type(inst).__name__
    if name not in _QUOTE_FIELD:
        raise TypeError(
            f"quote_jacobian: instrument {name} has no registered "
            f"primary-quote field.  Register it in "
            f"valax.curves.bootstrap_graph._QUOTE_FIELD or extend the "
            f"dispatch to add support."
        )
    return _QUOTE_FIELD[name]


def _extract_quote_rate(inst: BootstrapInstrument) -> Float[Array, ""]:
    """Extract the primary quote scalar from a bootstrap instrument.

    Dispatches on ``type(inst).__name__`` via :data:`_QUOTE_FIELD`, so
    ``.rate`` is returned for classic swap / deposit / FRA quotes,
    ``.spread`` for :class:`TenorBasisSwap` / :class:`CrossCurrencyBasisSwap`,
    ``.futures_rate`` for :class:`MoneyMarketFuture`,
    ``.quoted_forward`` for :class:`FXForward`,
    ``.far_rate`` for :class:`FXSwap`, and
    ``.jump_size`` for :class:`TurnInstrument`.
    """
    field = _quote_field_name(inst)
    return jnp.asarray(getattr(inst, field))


def _replace_quote_rate(
    inst: BootstrapInstrument, new_rate: Float[Array, ""]
) -> BootstrapInstrument:
    """Return a copy of ``inst`` with its primary quote replaced.

    Uses the same dispatch as :func:`_extract_quote_rate`, so the round
    trip ``_replace_quote_rate(inst, _extract_quote_rate(inst))`` returns
    a value-equal copy for every registered instrument type.
    """
    field = _quote_field_name(inst)
    return eqx.tree_at(
        lambda x, f=field: getattr(x, f), inst, new_rate
    )


def _fitted_quote(
    inst: BootstrapInstrument,
    graph: CurveGraph,
    fixings: FixingHistory,
    ref_date: Int[Array, ""],
    residual_value: Float[Array, ""],
) -> Float[Array, ""]:
    """Newton correction of the input quote to zero its residual.

    For every VALAX quote type the residual is linear in the primary
    quote scalar, so the correction
    ``fitted = quoted - residual / slope`` — with
    ``slope = d(residual)/d(quote)`` at the calibrated graph — is
    exact rather than first-order.  Computed via ``jax.grad`` on the
    instrument's own residual method.
    """
    import jax  # local import: diagnostics are one-shot, not JIT-hot

    quoted = _extract_quote_rate(inst)

    def residual_of_quote(q: Float[Array, ""]) -> Float[Array, ""]:
        return _replace_quote_rate(inst, q).residual(
            graph, fixings, ref_date
        )

    slope = jax.grad(residual_of_quote)(quoted)
    return quoted - residual_value / slope


# ── Public API ───────────────────────────────────────────────────────


def bootstrap_curve_graph(
    reference_date: Int[Array, ""],
    curve_specs: Sequence[CurveSpec],
    instruments: Sequence[BootstrapInstrument],
    *,
    fixings: FixingHistory | None = None,
    solver: optx.AbstractRootFinder | None = None,
    initial_guess: Mapping[str, Float[Array, " n"]] | None = None,
    max_steps: int = 256,
) -> tuple[CurveGraph, CurveBuildDiagnostics]:
    """Jointly bootstrap every curve in the graph in one Newton solve.

    The solver concatenates the per-curve log-DFs into a single flat
    state vector.  A residual function reconstructs each
    :class:`DiscountCurve` from its slice, assembles them into a
    :class:`CurveGraph`, and asks every instrument for its residual
    via :meth:`BootstrapInstrument.residual`.  The system is
    well-determined when ``len(instruments) == sum(spec.pillar_dates.shape[0])``.

    Because ``optimistix.root_find`` defaults to
    ``optimistix.ImplicitAdjoint``, ``jax.grad`` of any downstream
    function of the returned graph w.r.t. an input quote rate costs
    one linear solve — independent of Newton iteration count.  This
    is what :func:`quote_jacobian` exploits.

    .. note::

       This function is **not jittable by design**: it validates the
       system shape eagerly, and materialises Python-level diagnostics
       (e.g. the concrete ``converged`` flag) that require leaving the
       trace.  Curve building is a one-shot operation, not a JIT-hot
       path — build the graph once, then JIT the pricing functions
       that consume it.

    Args:
        reference_date: Valuation date as an integer ordinal scalar.
        curve_specs: Sequence of :class:`CurveSpec` describing each
            curve to be solved for.  Order determines the layout of
            the flat state vector; ordering is otherwise irrelevant.
        instruments: Sequence of :class:`BootstrapInstrument` quotes.
            Every instrument's ``curves_touched`` entries must name a
            spec in ``curve_specs``.
        fixings: Optional :class:`FixingHistory` for
            partially-seasoned float legs.  Defaults to an empty
            registry.
        solver: An ``optimistix.AbstractRootFinder``.  Defaults to
            :class:`optimistix.Newton` at tolerance 1e-10.
        initial_guess: Optional per-curve initial log-DF vectors keyed
            by ``curve_id``.  Missing curves fall back to a flat 4%
            guess.
        max_steps: Maximum Newton iterations.

    Returns:
        A pair ``(graph, diagnostics)`` where ``graph`` is the
        calibrated :class:`CurveGraph` and ``diagnostics`` is a
        :class:`CurveBuildDiagnostics` recording final residuals,
        iteration count, and the converged flag.

    Raises:
        ValueError: If the residual system is not square, or if an
            instrument touches a curve not present in ``curve_specs``,
            or if an ``initial_guess`` shape does not match its spec.

    Examples:
        >>> ref = ymd_to_ordinal(2025, 1, 1)
        >>> spec = CurveSpec(
        ...     curve_id="USD.SOFR.OIS", currency="USD",
        ...     pillar_dates=jnp.array([ymd_to_ordinal(2026, 1, 1)]),
        ... )
        >>> dep = DepositRate(
        ...     start_date=ref, end_date=spec.pillar_dates[0],
        ...     rate=jnp.array(0.05),
        ...     curves_touched=("USD.SOFR.OIS",),
        ... )
        >>> graph, diag = bootstrap_curve_graph(ref, [spec], [dep])
        >>> bool(diag.max_abs_residual < 1e-10)
        True

    References:
        * ``docs/architecture/production.md`` §11.4 for the design
          spec.
        * ``docs/roadmap.md`` ("Multi-Curve Framework — What Ships,
          What's Left") for the workstream status.
        * ``docs/theory.md`` §3.7, §3.8 for the no-arbitrage identities
          that each instrument's residual encodes.
    """
    _validate_square(curve_specs, instruments)
    _validate_curves_touched(curve_specs, instruments)

    if fixings is None:
        fixings = empty_fixing_history()

    if solver is None:
        solver = optx.Newton(rtol=1e-10, atol=1e-10)

    ref_i32 = jnp.asarray(reference_date, dtype=jnp.int32)
    offsets = _spec_offsets(curve_specs)

    x0 = _pack_initial_guess(initial_guess, ref_i32, curve_specs)

    # Materialise Python-static context that the residual function needs.
    # ``optimistix`` traces ``residual_fn`` with concrete ``args``, so we
    # pass everything the trace needs; curve_specs and instruments enter
    # as closed-over Python objects (they are static / pytree-valid).
    def residual_fn(
        log_dfs: Float[Array, " n_total"],
        args,
    ) -> Float[Array, " n_instruments"]:
        ref_inner, fixings_inner = args
        return _graph_residual(
            log_dfs,
            curve_specs,
            offsets,
            instruments,
            fixings_inner,
            ref_inner,
        )

    args = (ref_i32, fixings)

    sol = optx.root_find(
        residual_fn,
        solver,
        x0,
        args=args,
        max_steps=max_steps,
        throw=False,  # let the caller inspect diagnostics on non-convergence
    )

    graph = _build_graph_from_log_dfs(
        sol.value, curve_specs, offsets, ref_i32
    )

    final_residuals = residual_fn(sol.value, args)
    max_abs = jnp.max(jnp.abs(final_residuals))
    rmse = jnp.sqrt(jnp.mean(final_residuals ** 2))
    # ``sol.stats`` shape varies by solver; expose iteration count if present.
    n_steps_val = sol.stats.get("num_steps", jnp.asarray(-1))
    converged = bool(sol.result == optx.RESULTS.successful)

    # Per-instrument fitted-vs-quoted diagnostic (in each quote's native
    # units).  Computed in a Python loop rather than a vmap because each
    # instrument type has different pytree structure — vmap would need
    # per-type stacking that isn't worth the complexity for a one-shot
    # post-solve diagnostic.
    quoted_values = jnp.stack(
        [_extract_quote_rate(inst) for inst in instruments]
    )
    fitted_quotes = jnp.stack(
        [
            _fitted_quote(inst, graph, fixings, ref_i32, final_residuals[i])
            for i, inst in enumerate(instruments)
        ]
    )

    # Jacobian condition number of the residual system at the solution.
    # Uses ``jax.jacrev`` rather than ``jax.jacfwd`` to match the
    # implicit-adjoint solve path that ``optimistix`` uses internally.
    import jax  # local import: one-shot post-solve diagnostic
    residual_jac = jax.jacrev(residual_fn)(sol.value, args)
    jacobian_cond = jnp.linalg.cond(residual_jac)

    diagnostics = CurveBuildDiagnostics(
        residuals=final_residuals,
        max_abs_residual=max_abs,
        rmse=rmse,
        n_steps=jnp.asarray(n_steps_val, dtype=jnp.int32),
        fitted_quotes=fitted_quotes,
        quoted_values=quoted_values,
        jacobian_condition_number=jacobian_cond,
        converged=converged,
    )

    return graph, diagnostics


# ── Residual / state accessors for adjoint differentiation ───────────


def curve_quote_vector(
    instruments: Sequence[BootstrapInstrument],
) -> Float[Array, " n_quotes"]:
    """Stack the primary quote scalar of every instrument into ``q_c``.

    The column ordering of every quote-space sensitivity in VALAX --
    :func:`quote_jacobian` and
    :func:`valax.greeks.adjoint.curve_quote_deltas` alike -- is the
    order of ``instruments``.

    Args:
        instruments: Bootstrap quotes, each with a primary-quote field
            registered in :data:`_QUOTE_FIELD`.

    Returns:
        The quote vector, one entry per instrument.

    Raises:
        TypeError: If any instrument type has no registered
            primary-quote field.
    """
    return jnp.stack([_extract_quote_rate(i) for i in instruments])


def log_dfs_from_graph(
    graph: CurveGraph,
    curve_specs: Sequence[CurveSpec],
) -> Float[Array, " n_total"]:
    """Recover the solver state ``theta_c`` from a calibrated graph.

    Inverts the layout of :func:`_build_graph_from_log_dfs`: drops each
    curve's reference-date pillar (where ``DF == 1`` by construction, so
    it is not a free unknown) and concatenates ``log(DF)`` across curves
    in ``curve_specs`` order.

    Args:
        graph: A calibrated :class:`CurveGraph`.
        curve_specs: The specs the graph was built from; their order
            defines the layout of the returned vector.

    Returns:
        The flat log-DF state vector, length
        ``sum(spec.pillar_dates.shape[0])``.
    """
    return jnp.concatenate(
        [
            jnp.log(graph[spec.curve_id].discount_factors[1:])
            for spec in curve_specs
        ]
    )


def curve_residual_fn(
    reference_date: Int[Array, ""],
    curve_specs: Sequence[CurveSpec],
    instruments: Sequence[BootstrapInstrument],
    *,
    fixings: FixingHistory | None = None,
):
    """Expose the Layer-1a residual as ``R(theta_c, q_c)`` for the adjoint.

    Returns the *same* residual :func:`bootstrap_curve_graph` drives to
    zero (both call :func:`_graph_residual`), re-parameterised so the
    quote vector is an explicit differentiable argument.  This is what
    lets :mod:`valax.greeks.adjoint` form the two vector-Jacobian
    products of the implicit function theorem --
    ``(dR/dtheta)^T lambda`` and ``(dR/dq)^T lambda`` -- without ever
    re-implementing the residual.

    Note that VALAX residuals are **not** uniformly in quote units: a
    futures residual is a rate difference (``dR/dq = +I``) while a swap
    residual is ``rate * annuity - float_pv`` (PV units, so
    ``dR/dq = annuity``).  Consumers must therefore differentiate
    ``dR/dq`` rather than assume ``-I``.

    Args:
        reference_date: Valuation date as an integer ordinal scalar.
        curve_specs: The curve specs defining the state layout.
        instruments: Bootstrap quotes; column order of ``q_c``.
        fixings: Optional realised-fixings registry; defaults to empty.

    Returns:
        A callable ``residual(log_dfs, quotes)`` returning the residual
        vector of shape ``(n_instruments,)``.
    """
    offsets = _spec_offsets(curve_specs)
    insts = tuple(instruments)
    fix = empty_fixing_history() if fixings is None else fixings
    ref_i32 = jnp.asarray(reference_date, dtype=jnp.int32)

    def residual(
        log_dfs: Float[Array, " n_total"],
        quotes: Float[Array, " n_quotes"],
    ) -> Float[Array, " n_instruments"]:
        rebuilt = tuple(
            _replace_quote_rate(inst, quotes[k])
            for k, inst in enumerate(insts)
        )
        return _graph_residual(
            log_dfs, curve_specs, offsets, rebuilt, fix, ref_i32
        )

    return residual


def validate_curve_system(
    curve_specs: Sequence[CurveSpec],
    instruments: Sequence[BootstrapInstrument],
) -> None:
    """Eagerly check that a joint curve system is square and closed.

    The eager half of a curve build, split out so a traceable solve (see
    :func:`valax.calibration.nodes.curve_graph_node`) can validate once at
    construction and then run inside ``jit`` / ``grad``.

    Args:
        curve_specs: The curves to solve for.
        instruments: The bootstrap quotes.

    Raises:
        ValueError: If the system is not square, or an instrument touches a
            curve absent from ``curve_specs``.
    """
    _validate_square(curve_specs, instruments)
    _validate_curves_touched(curve_specs, instruments)


def initial_log_dfs(
    reference_date: Int[Array, ""],
    curve_specs: Sequence[CurveSpec],
    initial_guess: Mapping[str, Float[Array, " n"]] | None = None,
) -> Float[Array, " n_total"]:
    """Initial log-DF state for a joint curve solve.

    Args:
        reference_date: Valuation date ordinal.
        curve_specs: The curves to solve for; their order fixes the layout.
        initial_guess: Optional per-curve log-DF vectors keyed by
            ``curve_id``; missing curves fall back to a flat 4% guess.

    Returns:
        The flat initial state vector.

    Raises:
        ValueError: If a provided guess has the wrong length.
    """
    ref_i32 = jnp.asarray(reference_date, dtype=jnp.int32)
    return _pack_initial_guess(initial_guess, ref_i32, curve_specs)


def curve_graph_from_log_dfs(
    log_dfs: Float[Array, " n_total"],
    curve_specs: Sequence[CurveSpec],
    reference_date: Int[Array, ""],
) -> CurveGraph:
    """Rebuild a :class:`CurveGraph` from the flat solver state ``theta_c``.

    The inverse of :func:`log_dfs_from_graph`, and the ``assemble`` map of the
    curve calibration level: it is traceable and differentiable in
    ``log_dfs``.

    Args:
        log_dfs: Concatenated pillar log-DFs in ``curve_specs`` order.
        curve_specs: The curve specs defining the layout.
        reference_date: Valuation date ordinal.

    Returns:
        The calibrated :class:`CurveGraph`.
    """
    return _build_graph_from_log_dfs(
        log_dfs,
        curve_specs,
        _spec_offsets(curve_specs),
        jnp.asarray(reference_date, dtype=jnp.int32),
    )


# ── Quote-sensitivity Jacobian ───────────────────────────────────────


def quote_jacobian(
    reference_date: Int[Array, ""],
    curve_specs: Sequence[CurveSpec],
    instruments: Sequence[BootstrapInstrument],
    *,
    by: str = "log_df",
    fixings: FixingHistory | None = None,
    solver: optx.AbstractRootFinder | None = None,
    max_steps: int = 256,
) -> Float[Array, "n_outputs n_quotes"]:
    """Jacobian of calibrated curve outputs w.r.t. input quote rates.

    Runs :func:`bootstrap_curve_graph` inside a ``jax.jacrev`` closure
    over the primary quote scalar of every instrument.  Uses the
    implicit-adjoint pass through the Newton solve so the cost is
    ``O(n_quotes)`` reverse-mode sweeps rather than ``O(newton_iters)``
    forward AD.

    The output rows are laid out per-curve in the order given by
    ``curve_specs``; within each curve, one row per pillar.  Columns
    are laid out in the order of ``instruments``.

    Args:
        reference_date: Valuation date (integer ordinal scalar).
        curve_specs: Same as :func:`bootstrap_curve_graph`.
        instruments: Same as :func:`bootstrap_curve_graph`.  Each
            instrument type must have a registered primary-quote field
            in :data:`_QUOTE_FIELD` (``rate`` for deposits/FRAs/swaps,
            ``futures_rate``, ``spread``, ``quoted_forward``,
            ``far_rate``, or ``jump_size``); unregistered types raise
            ``TypeError``.
        by: One of ``"log_df"``, ``"df"``, ``"zero_rate"``.  Selects
            which representation of the calibrated pillar values to
            differentiate.
        fixings: Optional realised-fixings registry.
        solver: Root finder; defaults to :class:`optimistix.Newton`.
        max_steps: Maximum Newton iterations.

    Returns:
        A Jacobian of shape ``(n_pillars_total, n_quotes)`` where
        ``n_pillars_total = sum(spec.pillar_dates.shape[0])``.

    Raises:
        ValueError: For unsupported ``by`` values or upstream from
            :func:`bootstrap_curve_graph`.
    """
    if by not in ("log_df", "df", "zero_rate"):
        raise ValueError(
            f"quote_jacobian: by={by!r} not supported. "
            f"Choose one of 'log_df', 'df', 'zero_rate'."
        )

    quotes = jnp.stack([_extract_quote_rate(i) for i in instruments])
    instruments_tuple = tuple(instruments)

    def calibrate_and_extract(
        quote_vec: Float[Array, " n_quotes"],
    ) -> Float[Array, " n_pillars_total"]:
        rebuilt = [
            _replace_quote_rate(inst, quote_vec[k])
            for k, inst in enumerate(instruments_tuple)
        ]
        graph, _ = bootstrap_curve_graph(
            reference_date,
            curve_specs,
            rebuilt,
            fixings=fixings,
            solver=solver,
            max_steps=max_steps,
        )
        out_chunks: list[Float[Array, " n"]] = []
        for spec in curve_specs:
            curve = graph[spec.curve_id]
            # Drop the reference-date pillar (DF == 1 by construction) so
            # the row count matches ``sum(spec.pillar_dates.shape[0])``.
            pillar_dfs = curve.discount_factors[1:]
            if by == "log_df":
                out_chunks.append(jnp.log(pillar_dfs))
            elif by == "df":
                out_chunks.append(pillar_dfs)
            else:  # by == "zero_rate"
                out_chunks.append(
                    jnp.stack(
                        [
                            zero_rate(curve, d)
                            for d in spec.pillar_dates
                        ]
                    )
                )
        return jnp.concatenate(out_chunks)

    import jax  # local import: quote_jacobian is a build-time helper
    return jax.jacrev(calibrate_and_extract)(quotes)
