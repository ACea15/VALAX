"""Multi-curve framework: separate discount and forward projection curves.

Post-crisis, OIS discounting + tenor-specific forward curves is the standard.
For example, SOFR OIS for discounting and separate 1M/3M SOFR curves for
forward rate projection.

Since MC-Curves-2b this module is a **thin view layer** over the joint
multi-curve Newton solver:
:func:`valax.curves.bootstrap_graph.bootstrap_curve_graph` does the work
and :class:`MultiCurveSet` is a two-tier *view* over the resulting
:class:`~valax.curves.graph.CurveGraph` (see :meth:`MultiCurveSet.from_graph`
and :meth:`MultiCurveSet.to_graph`).  The sequential OIS-then-tenor
pipeline that used to live here is gone; every curve in the set is now
solved in one Newton iteration, which also means ``jax.grad`` flows
through the build via ``optimistix.ImplicitAdjoint``.

.. deprecated:: MC-Curves-2
    :func:`bootstrap_multi_curve` and :class:`MultiCurveSet` are
    superseded by
    :func:`valax.curves.bootstrap_graph.bootstrap_curve_graph` and
    :class:`~valax.curves.graph.CurveGraph`.  The joint solver handles
    every quote type (tenor-basis, cross-currency, futures, …) that the
    two-tier container cannot express.  A ``DeprecationWarning`` is
    emitted on every :func:`bootstrap_multi_curve` call; the function is
    kept for the deprecation cycle so existing user code continues to
    work.
"""

import dataclasses
import warnings

import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from jaxtyping import Int
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph, CurveSpec, _CURVE_ID_RE
from valax.curves.instruments import DepositRate, FRA, IborSwapRate, SwapRate


# ── Synthetic curve identifiers for the legacy two-tier build ────────

#: Currency prefix used for the synthetic curve ids the legacy wrapper
#: feeds to the joint solver.  ``XXX`` is the ISO-4217 code for "no
#: currency", which is honest: :func:`bootstrap_multi_curve` carries no
#: currency information in its signature.  The ids never leak to the
#: caller — :class:`MultiCurveSet` is keyed by the user's own tenor
#: labels.
_LEGACY_PREFIX = "XXX.LEGACY"

#: Discount (OIS) leg of the legacy two-tier build.
_LEGACY_DISCOUNT_ID = f"{_LEGACY_PREFIX}.OIS"


def _legacy_forward_id(tenor: str, position: int) -> str:
    """Synthetic curve id for the forward leg labelled ``tenor``.

    Uses ``XXX.LEGACY.<TENOR>`` when the (upper-cased) tenor label is
    conformant with the frozen curve-id alphabet — the ``"3M"`` /
    ``"6M"`` labels every real caller uses — and falls back to the
    qualifier form ``XXX.LEGACY.OIS.FWD<k>`` for free-form labels so
    that arbitrary legacy keys never fail validation.

    Args:
        tenor: The caller's tenor label, e.g. ``"3M"``.
        position: Index of this forward leg in the input dict, used
            only by the fallback id.

    Returns:
        A curve id accepted by :class:`CurveSpec`.
    """
    candidate = f"{_LEGACY_PREFIX}.{tenor.upper()}"
    if _CURVE_ID_RE.fullmatch(candidate):
        return candidate
    return f"{_LEGACY_DISCOUNT_ID}.FWD{position}"


class MultiCurveSet(eqx.Module):
    """Two-tier view over a :class:`~valax.curves.graph.CurveGraph`.

    A pedagogical container for the single-currency case: one OIS
    discount curve plus a tenor-keyed dict of forward projection
    curves.  It holds exactly the same :class:`DiscountCurve` objects
    that live in the graph — :meth:`from_graph` and :meth:`to_graph`
    convert between the two shapes without copying or re-solving.

    New code should prefer the :class:`CurveGraph` directly: it is
    n-currency, n-tenor, and carries the basis / cross-currency curves
    that this two-tier shape cannot name.

    Attributes:
        discount_curve: OIS discount curve used for present-value discounting.
        forward_curves: Tenor-keyed forward projection curves
            (e.g., ``{"3M": curve_3m, "6M": curve_6m}``).
    """

    discount_curve: DiscountCurve
    forward_curves: dict  # str -> DiscountCurve (keys are static in the pytree)

    @classmethod
    def from_graph(
        cls,
        graph: CurveGraph,
        discount_id: str,
        forward_ids: dict[str, str],
    ) -> "MultiCurveSet":
        """Build the two-tier view from a calibrated curve graph.

        Args:
            graph: Curve graph holding the curves (typically the output
                of :func:`valax.curves.bootstrap_curve_graph`).
            discount_id: Identifier of the discounting curve on the graph.
            forward_ids: Mapping from the tenor label you want in
                ``forward_curves`` to the curve identifier on the graph,
                e.g. ``{"3M": "USD.SOFR.3M"}``.

        Returns:
            A :class:`MultiCurveSet` sharing the graph's curve objects.

        Raises:
            KeyError: If any identifier is absent from the graph.
        """
        return cls(
            discount_curve=graph[discount_id],
            forward_curves={
                tenor: graph[curve_id]
                for tenor, curve_id in forward_ids.items()
            },
        )

    def to_graph(
        self,
        discount_id: str = _LEGACY_DISCOUNT_ID,
        forward_ids: dict[str, str] | None = None,
    ) -> CurveGraph:
        """Expose this set as a :class:`CurveGraph`.

        The inverse of :meth:`from_graph`.  Use it to feed legacy
        containers into the graph-aware pricers
        (:mod:`valax.pricing.analytic.graph_pricing`) and the
        ``CurveGraph`` shock primitives in :mod:`valax.risk.shocks`.

        Args:
            discount_id: Identifier to file the discount curve under.
                Defaults to the synthetic legacy id.
            forward_ids: Optional mapping from tenor label to the
                identifier to file that forward curve under.  Missing
                labels fall back to the synthetic legacy ids.

        Returns:
            A :class:`CurveGraph` holding the same curve objects.
        """
        forward_ids = forward_ids or {}
        curves: dict[str, DiscountCurve] = {discount_id: self.discount_curve}
        for position, (tenor, curve) in enumerate(self.forward_curves.items()):
            curve_id = forward_ids.get(
                tenor, _legacy_forward_id(tenor, position)
            )
            curves[curve_id] = curve
        return CurveGraph(curves=curves)


def bootstrap_multi_curve(
    reference_date: Int[Array, ""],
    discount_instruments: list,
    forward_instruments: dict[str, list],
    day_count: str = "act_365",
    solver: optx.AbstractRootFinder | None = None,
    max_steps: int = 256,
) -> MultiCurveSet:
    """Build a multi-curve set from market quotes.

    Since MC-Curves-2b this is a thin wrapper over the joint solver
    :func:`valax.curves.bootstrap_graph.bootstrap_curve_graph`: the OIS
    curve and every forward curve are solved **simultaneously** in one
    Newton iteration rather than sequentially.  The calibration target
    is unchanged, so the curves are the same to solver tolerance:

    * ``discount_instruments`` price on the discount curve alone
      (single-curve residuals).
    * Deposits and FRAs in ``forward_instruments`` price on their own
      forward curve (their DF relationship is independent of the
      discount curve).
    * Swaps in ``forward_instruments`` use the dual-curve par condition

      .. math::
          \\sum_i \\left(\\frac{DF_{\\text{fwd}}(T_{i-1})}{DF_{\\text{fwd}}(T_i)}
          - 1\\right) DF_{\\text{ois}}(T_i)
          = r \\sum_i \\tau_i\\, DF_{\\text{ois}}(T_i)

      which is exactly :class:`~valax.curves.instruments.IborSwapRate`
      with the floating schedule set equal to the fixed schedule; the
      wrapper performs that translation for you.

    Each instrument contributes exactly one pillar: ``end_date`` for
    deposits and FRAs, ``fixed_dates[-1]`` for swaps.

    Args:
        reference_date: Valuation date (ordinal).
        discount_instruments: Instruments for the OIS discount curve.
        forward_instruments: Dict mapping tenor label to instrument list
            for each forward curve.
        day_count: Day count convention for the output curves.
        solver: Root finder for the joint Newton solve.
        max_steps: Maximum solver iterations.

    Returns:
        A ``MultiCurveSet`` with the discount curve and all forward curves.

    Raises:
        TypeError: If an instrument is not a
            :class:`~valax.curves.instruments.DepositRate`,
            :class:`~valax.curves.instruments.FRA`, or
            :class:`~valax.curves.instruments.SwapRate`.

    .. deprecated:: MC-Curves-2
        Use :func:`valax.curves.bootstrap_graph.bootstrap_curve_graph`
        instead.  The joint solver handles tenor-basis, futures,
        cross-currency, and every other quote type from MC-Curves-1 that
        this two-tier container cannot name, and returns a
        :class:`~valax.curves.bootstrap_graph.CurveBuildDiagnostics`
        report alongside the curves.
    """
    warnings.warn(
        "bootstrap_multi_curve is deprecated (MC-Curves-2): use "
        "valax.curves.bootstrap_curve_graph for multi-curve builds. "
        "The joint solver handles tenor-basis, futures, cross-currency, "
        "and every other quote type this two-tier wrapper cannot.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Deferred import breaks a would-be cycle
    # (bootstrap_graph → bootstrap_proto → graph → discount → daycounts).
    from valax.curves.bootstrap_graph import bootstrap_curve_graph

    ref_i32 = jnp.asarray(reference_date, dtype=jnp.int32)

    specs: list[CurveSpec] = []
    graph_instruments: list = []

    # ── Discount (OIS) leg: single-curve residuals ───────────────────
    disc_pillars, disc_insts = _retarget_single_curve(
        discount_instruments, _LEGACY_DISCOUNT_ID,
    )
    specs.append(
        CurveSpec(
            curve_id=_LEGACY_DISCOUNT_ID,
            currency="XXX",
            pillar_dates=disc_pillars,
            day_count=day_count,
        )
    )
    graph_instruments.extend(disc_insts)

    # ── Forward legs: deposits/FRAs single-curve, swaps dual-curve ───
    forward_ids: dict[str, str] = {}
    for position, (tenor, instruments) in enumerate(
        forward_instruments.items()
    ):
        curve_id = _legacy_forward_id(tenor, position)
        forward_ids[tenor] = curve_id
        fwd_pillars, fwd_insts = _retarget_forward_curve(
            instruments, _LEGACY_DISCOUNT_ID, curve_id,
        )
        specs.append(
            CurveSpec(
                curve_id=curve_id,
                currency="XXX",
                pillar_dates=fwd_pillars,
                day_count=day_count,
            )
        )
        graph_instruments.extend(fwd_insts)

    graph, _ = bootstrap_curve_graph(
        ref_i32,
        specs,
        graph_instruments,
        solver=solver,
        max_steps=max_steps,
    )

    return MultiCurveSet.from_graph(graph, _LEGACY_DISCOUNT_ID, forward_ids)


# ── Legacy-instrument → graph-instrument translation ─────────────────


def _pillar_date(inst) -> Int[Array, ""]:
    """Ordinal date of the single pillar this quote determines.

    Raises:
        TypeError: For quote types the legacy two-tier build never
            supported.
    """
    if isinstance(inst, (DepositRate, FRA)):
        return inst.end_date
    if isinstance(inst, SwapRate):
        return inst.fixed_dates[-1]
    raise TypeError(
        f"bootstrap_multi_curve supports DepositRate, FRA, and SwapRate "
        f"only.  Got {type(inst).__name__}.  Use bootstrap_curve_graph "
        f"for the full MC-Curves-1 quote set."
    )


def _sorted_pillars(instruments: list) -> Int[Array, " n"]:
    """Sorted pillar dates implied by ``instruments`` (one each)."""
    if not instruments:
        return jnp.zeros((0,), dtype=jnp.int32)
    pillars = jnp.stack(
        [jnp.asarray(_pillar_date(i), dtype=jnp.int32) for i in instruments]
    )
    return jnp.sort(pillars)


def _retarget_single_curve(
    instruments: list,
    curve_id: str,
) -> tuple[Int[Array, " n"], list]:
    """Point every quote at ``curve_id`` with single-curve residuals.

    ``_sorted_pillars`` has already rejected unsupported quote types, so
    every instrument here is a
    :class:`DepositRate` / :class:`FRA` / :class:`SwapRate`.
    """
    pillars = _sorted_pillars(instruments)
    retargeted = [_with_curves_touched(inst, (curve_id,)) for inst in instruments]
    return pillars, retargeted


def _with_curves_touched(inst, curves_touched: tuple):
    """Copy ``inst`` with a new static ``curves_touched`` tuple.

    ``curves_touched`` is an ``eqx.field(static=True)``, i.e. part of the
    pytree *structure* rather than a leaf, so ``eqx.tree_at`` cannot
    reach it.  Equinox modules are frozen dataclasses, so
    :func:`dataclasses.replace` is the supported way to rebuild one with
    a different static field.
    """
    return dataclasses.replace(inst, curves_touched=curves_touched)


def _retarget_forward_curve(
    instruments: list,
    discount_id: str,
    forward_id: str,
) -> tuple[Int[Array, " n"], list]:
    """Translate a legacy forward-curve instrument list for the graph.

    Deposits and FRAs keep their single-curve residual on the forward
    curve.  Swaps become :class:`IborSwapRate` quotes discounting on
    ``discount_id`` and projecting on ``forward_id``, with the floating
    schedule set equal to the fixed schedule — the exact dual-curve par
    condition the sequential pipeline solved.
    """
    pillars = _sorted_pillars(instruments)
    translated: list = []
    for inst in instruments:
        if isinstance(inst, (DepositRate, FRA)):
            translated.append(_with_curves_touched(inst, (forward_id,)))
        elif isinstance(inst, SwapRate):
            translated.append(
                _as_dual_curve_swap(inst, discount_id, forward_id)
            )
        else:
            raise TypeError(
                f"bootstrap_multi_curve supports DepositRate, FRA, and "
                f"SwapRate only.  Got {type(inst).__name__}.  Use "
                f"bootstrap_curve_graph for the full MC-Curves-1 quote set."
            )
    return pillars, translated


def _as_dual_curve_swap(
    swap: SwapRate,
    discount_id: str,
    forward_id: str,
) -> IborSwapRate:
    """Reinterpret a legacy :class:`SwapRate` as a dual-curve quote.

    The legacy sequential pipeline read a ``SwapRate`` on a forward leg
    as "fixed leg discounted on OIS, floating leg projected off the
    tenor curve and discounted on OIS", with the floating schedule
    equal to the fixed schedule.  That is precisely an
    :class:`IborSwapRate` whose ``float_dates == fixed_dates`` and whose
    fixing dates are the period starts.  The ``index_id`` is left at the
    ``"IBOR"`` sentinel, which is absent from the empty fixing history
    the wrapper uses, so every coupon is curve-projected.
    """
    fixing_dates = jnp.concatenate(
        [swap.start_date[None], swap.fixed_dates[:-1]]
    )
    return IborSwapRate(
        start_date=swap.start_date,
        fixed_dates=swap.fixed_dates,
        float_dates=swap.fixed_dates,
        fixing_dates=fixing_dates,
        rate=swap.rate,
        fixed_day_count=swap.day_count,
        float_day_count=swap.day_count,
        curves_touched=(discount_id, forward_id),
    )
