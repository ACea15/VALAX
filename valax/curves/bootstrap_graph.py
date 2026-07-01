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
``docs/architecture/mc-curves-2.md`` for the session-handoff notes
that motivated this module.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Float, Int
from jax import Array

from valax.curves.bootstrap_proto import BootstrapInstrument
from valax.curves.discount import DiscountCurve, zero_rate
from valax.curves.fixings import FixingHistory, empty_fixing_history
from valax.curves.graph import CurveGraph, CurveSpec
from valax.dates.daycounts import year_fraction


# ── Diagnostics container ────────────────────────────────────────────


class CurveBuildDiagnostics(eqx.Module):
    """Per-curve and per-instrument repricing diagnostics.

    Returned alongside the calibrated :class:`CurveGraph` by
    :func:`bootstrap_curve_graph`.  A minimal subset of the full spec
    in ``production.md`` §11.6; extended in MC-Curves-3.

    Attributes:
        residuals: Final per-instrument residuals (shape
            ``(n_instruments,)``).  Zero at convergence.
        max_abs_residual: Scalar ``max(|residuals|)`` — a quick
            single-number convergence check.
        n_steps: Number of Newton iterations consumed
            (``jnp.asarray(-1)`` if the solver did not populate stats).
        converged: Boolean flag reported by the underlying
            ``optimistix`` solver.
    """

    residuals: Float[Array, " n_instruments"]
    max_abs_residual: Float[Array, ""]
    n_steps: Int[Array, ""]
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
        * ``docs/architecture/mc-curves-2.md`` for the session-handoff
          notes.
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
        graph = _build_graph_from_log_dfs(
            log_dfs, curve_specs, offsets, ref_inner
        )
        residuals = [
            inst.residual(graph, fixings_inner, ref_inner)
            for inst in instruments
        ]
        return jnp.stack(residuals)

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
    # ``sol.stats`` shape varies by solver; expose iteration count if present.
    n_steps_val = sol.stats.get("num_steps", jnp.asarray(-1))
    converged = bool(sol.result == optx.RESULTS.successful)

    diagnostics = CurveBuildDiagnostics(
        residuals=final_residuals,
        max_abs_residual=max_abs,
        n_steps=jnp.asarray(n_steps_val, dtype=jnp.int32),
        converged=converged,
    )

    return graph, diagnostics


# ── Quote-sensitivity Jacobian ───────────────────────────────────────


def _extract_quote_rate(inst: BootstrapInstrument) -> Float[Array, ""]:
    """Extract the primary quote scalar from a bootstrap instrument.

    Supports the ``.rate`` field found on every classic quote type
    (:class:`DepositRate`, :class:`FRA`, :class:`SwapRate`,
    :class:`OISSwapRate`, :class:`IborSwapRate`).  Extended coverage
    (``.spread``, ``.futures_rate``, ``.quoted_forward``) is a
    MC-Curves-3 refinement.
    """
    if hasattr(inst, "rate"):
        return jnp.asarray(inst.rate)
    raise TypeError(
        f"quote_jacobian: instrument {type(inst).__name__} does not "
        f"expose a .rate field.  Extended-quote support is queued for "
        f"MC-Curves-3."
    )


def _replace_quote_rate(
    inst: BootstrapInstrument, new_rate: Float[Array, ""]
) -> BootstrapInstrument:
    """Return a copy of ``inst`` with its primary quote replaced."""
    return eqx.tree_at(lambda x: x.rate, inst, new_rate)


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
            instrument must expose a ``.rate`` field
            (:func:`_extract_quote_rate`).
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
