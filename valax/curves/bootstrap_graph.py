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

from typing import Any, Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
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
    initial_guess: Mapping[str, Float[Array, " n"]] | None = None,
    settings: Any = None,
    name: str = "curve_graph",
) -> Any:
    """Jointly bootstrap every curve in the graph in one Newton solve.

    The solver concatenates the per-curve log-DFs into a single flat
    state vector.  A residual function reconstructs each
    :class:`DiscountCurve` from its slice, assembles them into a
    :class:`CurveGraph`, and asks every instrument for its residual
    via :meth:`BootstrapInstrument.residual`.  The system is
    well-determined when ``len(instruments) == sum(spec.pillar_dates.shape[0])``.

    The build is a calibration level like any other (see
    :func:`valax.calibration.solve.solve_level`): the quotes embedded in the
    instruments are the level's quotes, the residual is
    :func:`curve_residual_fn` — the *same* residual
    :func:`valax.calibration.nodes.curve_graph_node` solves — and the result
    carries the level, so its adjoint is the curve leg of quote-space risk.
    Validation is eager; the solve itself is traceable, so ``jax.grad`` of a
    downstream function of the graph w.r.t. an input quote costs one linear
    solve (``optimistix``'s implicit adjoint) independent of the Newton
    iteration count — which is what :func:`quote_jacobian` exploits.

    Desk diagnostics map onto the result as follows:

    * convergence, residuals, ``rmse``, ``n_steps`` — ``result.diagnostics``
      (``converged`` is a traceable JAX boolean; ``result.check()`` raises);
    * Jacobian condition number — ``result.condition_number()``;
    * quoted values — ``result.inputs.quotes``;
    * fitted quotes in native units — :func:`fitted_curve_quotes`.

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
        initial_guess: Optional per-curve initial log-DF vectors keyed
            by ``curve_id``.  Missing curves fall back to a flat 4%
            guess.
        settings: :class:`~valax.calibration.settings.SolverSettings`;
            defaults to :data:`~valax.calibration.settings.EXACT_SETTINGS`
            (Newton to 1e-10).
        name: Level name used in diagnostics.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the
        calibrated :class:`CurveGraph` is ``result.value``. Its labels name
        each instrument (``"<i>:<Type>"``) so a failed build points at a
        quote.

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
        >>> result = bootstrap_curve_graph(ref, [spec], [dep]).check()
        >>> graph = result.value

    References:
        * ``docs/architecture/production.md`` §11.4 for the design
          spec.
        * ``docs/theory.md`` §3.7, §3.8 for the no-arbitrage identities
          that each instrument's residual encodes.
    """
    # Lazy: valax.calibration imports the pricing stack, which imports
    # valax.curves.
    from valax.calibration.level import LevelInputs
    from valax.calibration.settings import EXACT_SETTINGS
    from valax.calibration.solve import solve_level

    validate_curve_system(curve_specs, instruments)
    specs = tuple(curve_specs)
    insts = tuple(instruments)
    ref_i32 = jnp.asarray(reference_date, dtype=jnp.int32)
    residual = curve_residual_fn(ref_i32, specs, insts, fixings=fixings)

    def residual_fn(theta: Float[Array, " n"], inputs: Any) -> Float[Array, " n"]:
        return residual(theta, inputs.quotes)

    def assemble(theta: Float[Array, " n"], _inputs: Any) -> CurveGraph:
        return curve_graph_from_log_dfs(theta, specs, ref_i32)

    return solve_level(
        name,
        "root",
        residual_fn,
        initial_log_dfs(ref_i32, specs, initial_guess),
        LevelInputs(quotes=curve_quote_vector(insts)),
        assemble,
        EXACT_SETTINGS if settings is None else settings,
        labels=curve_instrument_labels(insts),
    )


def curve_instrument_labels(
    instruments: Sequence[BootstrapInstrument],
) -> tuple[str, ...]:
    """Node labels of a curve build: ``"<index>:<InstrumentType>"``.

    Args:
        instruments: Bootstrap quotes, in residual order.

    Returns:
        One label per instrument.
    """
    return tuple(f"{i}:{type(inst).__name__}" for i, inst in enumerate(instruments))


def fitted_curve_quotes(
    instruments: Sequence[BootstrapInstrument],
    graph: CurveGraph,
    reference_date: Int[Array, ""],
    *,
    fixings: FixingHistory | None = None,
) -> Float[Array, " n_instruments"]:
    """Per-instrument quote implied by a calibrated graph, in native units.

    The desk sign-off table of a curve build: ``fitted - quoted`` is the
    fit error in the units the desk trades (rate points, basis points).
    Each entry is :func:`par_quote` — exact for every VALAX quote type,
    whose residual is linear in its primary quote.

    Args:
        instruments: Bootstrap quotes.
        graph: Calibrated graph (e.g. ``bootstrap_curve_graph(...).value``).
        reference_date: Valuation date ordinal.
        fixings: Optional realised-fixings registry.

    Returns:
        Fitted quote per instrument, in instrument order.
    """
    return jnp.stack([
        par_quote(inst, graph, reference_date, fixings=fixings)
        for inst in instruments
    ])


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


def par_quote(
    instrument: BootstrapInstrument,
    graph: CurveGraph,
    reference_date: Int[Array, ""],
    *,
    fixings: FixingHistory | None = None,
) -> Float[Array, ""]:
    """Fair quote of a bootstrap instrument on a given curve graph.

    Every VALAX quote residual is linear in its primary quote, so one
    Newton step from any starting quote lands exactly on the value that
    zeroes the residual. This is how synthetic markets produce quotes that
    are *exactly* consistent with a ground-truth curve: price the quote
    instrument with its own residual, rather than a re-implementation.

    Args:
        instrument: The bootstrap quote (its embedded quote value is only a
            starting point).
        graph: Curves to price against (must contain every curve the
            instrument touches).
        reference_date: Valuation date ordinal.
        fixings: Optional realised-fixings registry.

    Returns:
        The quote, in the instrument's native units, at which its residual
        vanishes on ``graph``.

    Raises:
        TypeError: If the instrument type has no registered quote field.
    """
    fix = empty_fixing_history() if fixings is None else fixings
    ref = jnp.asarray(reference_date, dtype=jnp.int32)
    residual = instrument.residual(graph, fix, ref)
    return _fitted_quote(instrument, graph, fix, ref, residual)


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
    settings: Any = None,
) -> Float[Array, "n_outputs n_quotes"]:
    """Jacobian of calibrated curve outputs w.r.t. input quote rates.

    Runs :func:`bootstrap_curve_graph` inside a ``jax.jacrev`` closure
    (and checks the build's convergence flag, carried out as auxiliary
    output, before returning)
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
        settings: Solver settings forwarded to :func:`bootstrap_curve_graph`.

    Returns:
        A Jacobian of shape ``(n_pillars_total, n_quotes)`` where
        ``n_pillars_total = sum(spec.pillar_dates.shape[0])``.

    Raises:
        ValueError: For unsupported ``by`` values or upstream from
            :func:`bootstrap_curve_graph`.
        CalibrationError: If the curve build did not converge — the
            implicit-function Jacobian of an unconverged build is
            meaningless, so it is never returned silently.
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
        result = bootstrap_curve_graph(
            reference_date,
            curve_specs,
            rebuilt,
            fixings=fixings,
            settings=settings,
        )
        graph = result.value
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
        return jnp.concatenate(out_chunks), result.converged

    import jax  # local import: quote_jacobian is a build-time helper
    from valax.calibration.result import CalibrationError

    jacobian, converged = jax.jacrev(calibrate_and_extract, has_aux=True)(quotes)
    if not bool(converged):
        raise CalibrationError(
            "quote_jacobian: the curve build did not converge; its "
            "implicit-function Jacobian is meaningless. Inspect "
            "bootstrap_curve_graph(...).diagnostics."
        )
    return jacobian
