r"""Cap → caplet (optionlet) stripping: quotes to a dual-curve vol surface.

Caps and floors are quoted with **one flat volatility per maturity**: a
single number that, applied to every optionlet of the strip, reproduces
the market premium.  Nothing prices with a flat vol — an optionlet
expiring in 6M and one expiring in 9Y do not share a volatility — so the
first step of every rates-vol pipeline is to *strip* the quotes into a
term structure of per-optionlet vols that reprices every quoted cap
exactly.

This is **the canonical dual-curve operation**.  A cap has two curve
dependencies that move independently:

* the forward rate :math:`F_i` of each accrual period is **projected** off
  the tenor (IBOR/term) curve, and
* the payoff at :math:`T_i` is **discounted** off the OIS / collateral
  curve.

So the stripped vols are a function of *both* curves, and the same quote
strips to different optionlet vols under different bases.  Every entry
point here therefore takes ``curve`` (discounting) plus an optional
``forward_curve`` (projection), exactly matching the convention of
:mod:`valax.pricing.analytic.caplets`.

The bootstrap
-------------
Caps on a common schedule are **nested**: the 2Y cap's optionlets are the
1Y cap's optionlets plus the incremental ones.  Writing
:math:`\mathcal{C}_i` for the optionlet index set of the :math:`i`-th
quoted cap, the strip is solved forward in maturity, one scalar unknown
at a time:

.. math::

    \text{Cap}^{\text{mkt}}_i
      = \sum_{j \in \mathcal{C}_{i-1}} \text{Optionlet}_j(\sigma_j)
      + \sum_{j \in \mathcal{C}_i \setminus \mathcal{C}_{i-1}}
        \text{Optionlet}_j(\sigma_i),

where the first sum is already known from the previous steps and
:math:`\sigma_i` — the vol shared by the incremental bucket — is the only
unknown.  Each step is a scalar root-find (``optimistix.Newton``), so the
whole strip is differentiable by implicit differentiation: ``jax.grad`` of
a stripped vol with respect to a quoted cap vol, a discount pillar or a
projection pillar works without unrolling the solver.

Quoting conventions
-------------------
``is_normal`` selects the pricing convention the quotes are given in —
lognormal Black-76 (:func:`valax.pricing.analytic.caplets.cap_price_black76`)
or normal Bachelier
(:func:`~valax.pricing.analytic.caplets.cap_price_bachelier`), which is
the branch to use when forwards or strikes are at or below zero.  No
pricing maths is re-derived here: the stripper calls those pricers, which
already accept a per-optionlet vol vector.

From a strip to a surface
-------------------------
:func:`strip_caplet_vols` produces one *column* — the vol term structure
at a single strike.  :func:`strip_optionlet_surface` strips a whole
quoted matrix (strikes × cap maturities), reads the projected forwards
off the curves with :func:`optionlet_forwards`, and hands the result to
:func:`valax.surfaces.optionlet_surface.build_sabr_caplet_surface` to fit
one SABR smile per optionlet expiry.  That closes the
quotes → strip → surface → exotic chain inside the framework: the
forwards the SABR fit sees are curve-derived rather than caller-supplied.

References:
    Hagan, P. (2003), "Convexity conundrums: pricing CMS swaps, caps and
    floors", Wilmott Magazine (§ on caplet stripping conventions).

    Brigo, D. and Mercurio, F. (2006), *Interest Rate Models — Theory and
    Practice*, 2nd ed., § 1.6 (cap/floor quoting) and § 6.4.

    Bianchetti, M. (2010), "Two curves, one price", Risk Magazine — the
    projection/discounting split this module implements.
"""

import dataclasses
from typing import Any, Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jaxtyping import Float
from jax import Array

from valax.calibration.level import LevelInputs
from valax.calibration.result import CalibrationDiagnostics, CalibrationResult
from valax.calibration.settings import SolverSettings, run_solver
from valax.calibration.solve import period_label, solve_batched_level

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction
from valax.instruments.rates import Cap
from valax.pricing.analytic.caplets import (
    cap_forwards,
    cap_price_bachelier,
    cap_price_black76,
)
from valax.surfaces.optionlet_surface import (
    OptionletVolSurface,
    build_sabr_caplet_surface,
)


#: Interpolation modes accepted by :class:`CapletVolCurve`.
_SUPPORTED_INTERP = ("piecewise_constant", "linear")

#: Default first guess for the scalar root-find, per quoting convention.
_DEFAULT_GUESS = {False: 0.20, True: 0.0080}


# ── Stripped vol term structure ──────────────────────────────────────


class CapletVolCurve(eqx.Module):
    """Stripped optionlet volatility term structure at a single strike.

    One *column* of an optionlet surface: the per-expiry volatilities
    that reprice a quoted cap strip exactly.  The strike axis is added by
    :func:`strip_optionlet_surface`, which stacks several of these into a
    smile-carrying :class:`~valax.surfaces.optionlet_surface.OptionletVolSurface`.

    Attributes:
        expiries: Optionlet expiries (year fractions to fixing), sorted
            ascending — one entry per optionlet of the longest quoted cap.
        vols: Stripped volatility of each optionlet, in the ``is_normal``
            convention.  Constant within each bootstrapped bucket.
        strike: The strike the strip was performed at; carried so a
            column cannot be silently read at the wrong strike.
        forwards: Projected forward rate of each optionlet's accrual
            period, read off the projection curve during the strip.
        bucket_ends: Index (exclusive) of the last optionlet of each
            quoted cap, i.e. the bootstrap bucket boundaries.  Static.
        is_normal: True when ``vols`` are normal (Bachelier) vols, False
            when they are lognormal (Black-76) vols.  Static.
        interp: ``"piecewise_constant"`` (default) returns the vol of the
            bucket an expiry falls in — the convention the bootstrap
            itself assumes, and the only one that reprices the quotes
            exactly.  ``"linear"`` interpolates between expiries instead,
            giving a smoother term structure at the cost of that exact
            repricing.  Static.
    """

    expiries: Float[Array, " n_optionlets"]
    vols: Float[Array, " n_optionlets"]
    strike: Float[Array, ""]
    forwards: Float[Array, " n_optionlets"]
    bucket_ends: tuple = eqx.field(static=True, default=())
    is_normal: bool = eqx.field(static=True, default=False)
    interp: str = eqx.field(static=True, default="piecewise_constant")

    def __check_init__(self) -> None:
        """Validate the interpolation mode.

        Raises:
            ValueError: If ``interp`` is not a supported mode.
        """
        if self.interp not in _SUPPORTED_INTERP:
            raise ValueError(
                f"CapletVolCurve.interp must be one of {_SUPPORTED_INTERP}, "
                f"got {self.interp!r}."
            )

    def __call__(
        self,
        expiry: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        """Optionlet volatility at ``expiry``.

        Args:
            expiry: Optionlet expiry (year fraction); may be an array.

        Returns:
            Volatility in the curve's quoting convention, with flat
            extrapolation beyond the stripped range.
        """
        if self.interp == "linear":
            return jnp.interp(expiry, self.expiries, self.vols)
        # Piecewise constant: an expiry belongs to the first bucket whose
        # right edge is at or beyond it, which is what the bootstrap solved.
        idx = jnp.searchsorted(self.expiries, expiry, side="left")
        idx = jnp.clip(idx, 0, self.vols.shape[0] - 1)
        return self.vols[idx]

    def cap_vols(self) -> Float[Array, " n_optionlets"]:
        """The stripped vols, aliased for symmetry with ``__call__``.

        Returns:
            The per-optionlet volatility vector.
        """
        return self.vols


# ── Dual-curve projection ────────────────────────────────────────────


def optionlet_forwards(
    cap: Cap,
    curve: DiscountCurve,
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, " n"]:
    r"""Projected forward rate of every optionlet in a cap strip.

    The projection half of the dual-curve split, exposed on its own
    because a smile fit needs the forwards as its at-the-money anchor:

    .. math::

        F_i = \frac{1}{\tau_i}
              \left(\frac{P^{\text{fwd}}(T_{i-1})}{P^{\text{fwd}}(T_i)}
              - 1\right),

    read off ``forward_curve`` when one is supplied and off ``curve``
    otherwise (single-curve setup).  Delegates to
    :func:`valax.pricing.analytic.caplets.cap_forwards` rather than
    re-deriving the projection, so a surface built on these forwards is
    consistent by construction with the pricer that will consume it.

    Args:
        cap: Cap/floor whose accrual schedule defines the periods.
        curve: Discount curve; also the projection curve when
            ``forward_curve`` is not given.
        forward_curve: Optional tenor/projection curve.

    Returns:
        Simply-compounded forward rate of each accrual period.
    """
    return cap_forwards(cap, curve, forward_curve)


def optionlet_expiries(
    cap: Cap,
    curve: DiscountCurve,
) -> Float[Array, " n"]:
    """Expiry (year fraction to fixing) of every optionlet in a cap strip.

    Args:
        cap: Cap/floor whose fixing schedule defines the expiries.
        curve: Curve supplying the valuation (reference) date.

    Returns:
        Year fraction from the reference date to each fixing date.
    """
    return year_fraction(
        curve.reference_date, cap.fixing_dates, cap.day_count,
    )


# ── Validation (user boundary, pre-trace) ────────────────────────────


def _concrete(x: Any) -> "np.ndarray | None":
    """NumPy view of ``x``, or ``None`` when it is a JAX tracer.

    The structural checks below compare *schedules and strikes*, which are
    metadata rather than numbers to be differentiated.  Doing that with
    NumPy keeps them genuinely pre-trace: a ``jnp`` comparison inside an
    enclosing ``jit`` returns a tracer even for concrete operands, so
    ``bool(...)`` on it would raise.  When a leaf really is traced (the
    caller jitted over the schedule itself) the check is skipped rather
    than failing — it cannot be answered at trace time.

    Args:
        x: Any array-like leaf.

    Returns:
        A NumPy array, or ``None`` if ``x`` could not be materialised.
    """
    try:
        return np.asarray(x)
    except Exception:  # pragma: no cover - tracer / non-materialisable leaf
        return None


def _validate_strip(
    caps: Sequence[Cap],
    curve: DiscountCurve,
    cap_vols: Float[Array, " n_caps"] | None,
    cap_premiums: Float[Array, " n_caps"] | None,
) -> None:
    """Validate a quoted cap strip before any tracing happens.

    Args:
        caps: Quoted caps, ascending in maturity.
        curve: Discount curve, for its reference date.
        cap_vols: Flat quoted vols, or ``None``.
        cap_premiums: Quoted premiums, or ``None``.

    Raises:
        ValueError: If the strip is empty, the quotes are ambiguous or
            mis-sized, the caps disagree on strike / payoff direction /
            day count, their schedules are not nested, or any optionlet
            has already fixed.
    """
    if len(caps) == 0:
        raise ValueError(
            "strip_caplet_vols needs at least one quoted cap; got an "
            "empty sequence."
        )
    if (cap_vols is None) == (cap_premiums is None):
        raise ValueError(
            "strip_caplet_vols takes exactly one of cap_vols (flat quoted "
            "volatilities) or cap_premiums (quoted prices); got "
            f"{'both' if cap_vols is not None else 'neither'}."
        )
    quotes = cap_vols if cap_vols is not None else cap_premiums
    n_quotes = int(jnp.asarray(quotes).shape[0])
    if n_quotes != len(caps):
        raise ValueError(
            f"strip_caplet_vols got {len(caps)} caps but {n_quotes} "
            f"quotes; there must be exactly one quote per cap."
        )

    first = caps[0]
    for i, cap in enumerate(caps):
        if cap.is_cap != first.is_cap:
            raise ValueError(
                f"strip_caplet_vols: cap {i} has is_cap={cap.is_cap} but "
                f"cap 0 has is_cap={first.is_cap}. A strip must be all "
                f"caps or all floors."
            )
        if cap.day_count != first.day_count:
            raise ValueError(
                f"strip_caplet_vols: cap {i} uses day_count "
                f"{cap.day_count!r} but cap 0 uses {first.day_count!r}."
            )
        strike, ref_strike = _concrete(cap.strike), _concrete(first.strike)
        if strike is not None and ref_strike is not None:
            if not np.array_equal(strike, ref_strike):
                raise ValueError(
                    f"strip_caplet_vols: cap {i} has a different strike "
                    f"from cap 0. One call strips a single strike column; "
                    f"loop over strikes (or use strip_optionlet_surface) "
                    f"for a surface."
                )

    n_prev = 0
    for i, cap in enumerate(caps):
        n_i = int(jnp.asarray(cap.fixing_dates).shape[0])
        if n_i <= n_prev:
            raise ValueError(
                f"strip_caplet_vols: cap {i} has {n_i} optionlets but cap "
                f"{i - 1} already has {n_prev}. Quoted caps must be sorted "
                f"ascending in maturity and each must add at least one "
                f"optionlet (duplicate maturities are not separable)."
            )
        if i > 0:
            prev = caps[i - 1]
            for field in ("fixing_dates", "start_dates", "end_dates"):
                head = _concrete(getattr(cap, field))
                tail = _concrete(getattr(prev, field))
                if head is None or tail is None:
                    continue
                if not np.array_equal(head[:n_prev], tail):
                    raise ValueError(
                        f"strip_caplet_vols: cap {i}'s {field} do not "
                        f"extend cap {i - 1}'s. The bootstrap requires "
                        f"nested schedules — every cap must share the "
                        f"shorter caps' optionlets and append new ones."
                    )
        n_prev = n_i

    # An optionlet that has already fixed carries no volatility to strip,
    # and its Black/Bachelier vega is identically zero — the bootstrap
    # would be solving for an unidentifiable parameter (and the T=0 limit
    # is not even differentiable).  Market convention is to drop the
    # already-fixed first period from the quoted strip.  Compared as
    # ordinals, which is sign-equivalent to the year fraction under every
    # day count and needs no arithmetic.
    fixings = _concrete(caps[-1].fixing_dates)
    reference = _concrete(curve.reference_date)
    if fixings is not None and reference is not None:
        n_bad = int(np.sum(fixings <= reference))
        if n_bad:
            raise ValueError(
                f"strip_caplet_vols: {n_bad} optionlet(s) fix on or before "
                f"the curve's reference date, so they have no volatility "
                f"to strip (zero vega). Drop the already-fixed period(s) "
                f"from the quoted strip — the market convention for a "
                f"spot-starting cap — and value them off the fixing "
                f"history instead."
            )


# ── The strip ────────────────────────────────────────────────────────


#: Default settings of the strip: Newton to 1e-12 on the bucket vols.
STRIP_SETTINGS = SolverSettings(
    method="newton", rtol=1e-12, atol=1e-12, max_steps=100
)


def _strip_plan(caps: Sequence[Cap]) -> tuple[np.ndarray, tuple[int, ...]]:
    """Optionlet → bucket index map and bucket ends of a nested strip."""
    ends = tuple(int(jnp.asarray(c.fixing_dates).shape[0]) for c in caps)
    index = np.concatenate([
        np.full(end - start, i, dtype=np.int32)
        for i, (start, end) in enumerate(zip((0,) + ends[:-1], ends))
    ])
    return index, ends


def cap_strip_labels(caps: Sequence[Cap]) -> tuple[str, ...]:
    """Bucket labels of a strip, e.g. ``("cap 1Y", "cap 2Y")`` (eager).

    Each label is the cap's tenor (first accrual start to last accrual end).
    Falls back to positional labels when the dates are traced.

    Args:
        caps: Quoted caps, ascending in maturity.

    Returns:
        One label per quoted cap.
    """
    labels = []
    for i, cap in enumerate(caps):
        start, end = _concrete(cap.start_dates), _concrete(cap.end_dates)
        if start is None or end is None:
            labels.append(f"cap {i + 1}")
            continue
        with jax.ensure_compile_time_eval():
            t = float(year_fraction(
                jnp.asarray(int(start[0])), jnp.asarray(int(end[-1])),
                cap.day_count,
            ))
        labels.append(f"cap {period_label(t)}")
    return tuple(labels)


def caplet_strip_level(
    name: str,
    caps: Sequence[Cap],
    inputs: LevelInputs,
    curves_fn: Callable[[Any], tuple[DiscountCurve, DiscountCurve | None]],
    *,
    quote_kind: str = "vol",
    is_normal: bool = False,
    interp: str = "piecewise_constant",
    initial_vols: Float[Array, " n_caps"] | None = None,
    labels: Sequence[str] | None = None,
    settings: SolverSettings | None = None,
    quotes_fn: Callable[[LevelInputs], Float[Array, " n_caps"]] | None = None,
) -> CalibrationResult:
    r"""A cap strip as one root level on upstream curves.

    The single implementation of the strip, shared by
    :func:`strip_caplet_vols` and the chain node
    :func:`valax.calibration.nodes.caplet_strip_node`. The unknowns are the
    bucket vols :math:`\sigma_i` (one per quoted cap); the level residual is
    the price error of every quoted cap per unit notional,

    .. math::

        R_i(\sigma) = \frac{\text{Cap}_i(\sigma_{1..i}; P_d, P_f)
        - \text{Cap}^{\text{mkt}}_i(q_i; P_d, P_f)}{N_i},

    which is lower-triangular in :math:`\sigma` (cap ``i`` only sees buckets
    ``1..i``). It is *solved* bucket by bucket — the classical bootstrap,
    one scalar root-find each (:func:`strip_buckets`) — so every bucket
    reports its own convergence and a failed bucket is named without
    poisoning the ones before it; it is *differentiated* as the one
    triangular system, so the IFT adjoint is a single step with the curves
    ``curves_fn(inputs.upstream)`` as differentiable upstream inputs.

    Args:
        name: Level name used in diagnostics.
        caps: Quoted caps, ascending and nested (validated by the caller).
        inputs: ``quotes`` — one flat vol (``quote_kind="vol"``) or premium
            (``"premium"``) per cap; ``upstream`` — whatever ``curves_fn``
            reads; ``fixed`` — optionally ``{"strike": K}`` to re-strike the
            caps.
        curves_fn: ``upstream -> (discount_curve, forward_curve | None)``.
        quote_kind: ``"vol"`` or ``"premium"``.
        is_normal: Bachelier (True) or Black-76 (False) pricing.
        interp: Interpolation recorded on the returned
            :class:`CapletVolCurve`.
        initial_vols: First guess per bucket (defaults to the quoted vols,
            or 20% / 80 bp then the previous bucket for premium quotes).
        labels: Bucket labels (default: positional).
        settings: Per-bucket solver settings; defaults to
            :data:`STRIP_SETTINGS`.
        quotes_fn: ``inputs -> (n_caps,)`` quotes; defaults to
            ``inputs.quotes`` (a chain node reads them from a ``QuoteSet``).

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`, with
        per-bucket diagnostics; the :class:`CapletVolCurve` is
        ``result.value``.

    Raises:
        ValueError: If ``quote_kind`` is unknown.
    """
    if quote_kind not in ("vol", "premium"):
        raise ValueError(
            f"caplet_strip_level: quote_kind must be 'vol' or 'premium', got "
            f"{quote_kind!r}."
        )
    settings = STRIP_SETTINGS if settings is None else settings
    caps = tuple(caps)
    index, ends = _strip_plan(caps)
    labels = tuple(labels) if labels is not None else tuple(
        f"cap {i + 1}" for i in range(len(caps))
    )
    residual = caplet_strip_residual_fn(
        caps, curves_fn, quote_kind=quote_kind, is_normal=is_normal,
        quotes_fn=quotes_fn,
    )
    theta, diagnostics = strip_buckets(
        caps, curves_fn, inputs, quote_kind=quote_kind, is_normal=is_normal,
        initial_vols=initial_vols, settings=settings, quotes_fn=quotes_fn,
    )

    def assemble(sigmas: Float[Array, " n_caps"], inp: LevelInputs) -> CapletVolCurve:
        curve, forward_curve = curves_fn(inp.upstream)
        longest = _struck(caps[-1], inp)
        return CapletVolCurve(
            expiries=optionlet_expiries(longest, curve),
            vols=sigmas[index],
            strike=jnp.asarray(longest.strike),
            forwards=optionlet_forwards(longest, curve, forward_curve),
            bucket_ends=ends,
            is_normal=is_normal,
            interp=interp,
        )

    return CalibrationResult(
        value=assemble(theta, inputs),
        theta=theta,
        inputs=inputs,
        diagnostics=diagnostics,
        residual_fn=residual,
        assemble_fn=assemble,
        settings=settings,
        name=name,
        labels=labels,
        problem="root",
    )


def strip_buckets(
    caps: Sequence[Cap],
    curves_fn: Callable[[Any], tuple[DiscountCurve, DiscountCurve | None]],
    inputs: LevelInputs,
    *,
    quote_kind: str = "vol",
    is_normal: bool = False,
    initial_vols: Float[Array, " n_caps"] | None = None,
    settings: SolverSettings | None = None,
    quotes_fn: Callable[[LevelInputs], Float[Array, " n_caps"]] | None = None,
) -> tuple[Float[Array, " n_caps"], CalibrationDiagnostics]:
    """Solve a cap strip bucket by bucket (the sequential bootstrap).

    One scalar root-find per quoted cap through
    :func:`~valax.calibration.settings.run_solver`, holding the earlier
    buckets fixed; the earlier buckets and the level inputs travel in the
    solver's ``args`` so the autodiff path differentiates through every
    stage. Traceable and ``vmap``-safe.

    Args:
        caps: Quoted caps, ascending and nested.
        curves_fn: ``upstream -> (discount_curve, forward_curve | None)``.
        inputs: The strip's level inputs (see :func:`caplet_strip_level`).
        quote_kind: ``"vol"`` or ``"premium"``.
        is_normal: Bachelier (True) or Black-76 (False) pricing.
        initial_vols: First guess per bucket.
        settings: Per-bucket solver settings; defaults to
            :data:`STRIP_SETTINGS`.
        quotes_fn: ``inputs -> (n_caps,)`` quotes; defaults to
            ``inputs.quotes``.

    Returns:
        ``(bucket_vols, diagnostics)`` with one diagnostics entry per bucket.
    """
    settings = STRIP_SETTINGS if settings is None else settings
    caps = tuple(caps)
    n_caps = len(caps)
    quotes_fn = _default_quotes if quotes_fn is None else quotes_fn
    one_cap = _cap_residual_fn(caps, curves_fn, quote_kind, is_normal, quotes_fn)
    if initial_vols is not None:
        guesses = jax.lax.stop_gradient(
            jnp.asarray(initial_vols, dtype=jnp.float64)
        )
    elif quote_kind == "vol":
        guesses = jax.lax.stop_gradient(
            jnp.asarray(quotes_fn(inputs), dtype=jnp.float64)
        )
    else:
        guesses = None

    known = jnp.zeros((0,), dtype=jnp.float64)
    guess = jnp.asarray(_DEFAULT_GUESS[is_normal], dtype=jnp.float64)
    converged, steps = [], []
    for i in range(n_caps):

        def bucket_residual(sigma, args, _i=i):
            prev, inp = args
            sigmas = jnp.concatenate(
                [prev, jnp.reshape(sigma, (1,)), jnp.zeros((n_caps - _i - 1,))]
            )
            return one_cap(_i, sigmas, inp)

        if guesses is not None:
            guess = guesses[i]
        sol = run_solver("root", bucket_residual, guess, (known, inputs), settings)
        known = jnp.concatenate([known, jnp.reshape(sol.value, (1,))])
        guess = jax.lax.stop_gradient(sol.value)
        converged.append(sol.result == optx.RESULTS.successful)
        steps.append(sol.stats.get("num_steps", jnp.asarray(-1)))

    final = jnp.stack([one_cap(i, known, inputs) for i in range(n_caps)])
    diagnostics = CalibrationDiagnostics(
        residuals=final[:, None],
        max_abs_residual=jnp.abs(final),
        rmse=jnp.abs(final),
        n_steps=jnp.stack(steps).astype(jnp.int32),
        converged=jnp.stack(converged),
    )
    return known, diagnostics


def _struck(cap: Cap, inputs: LevelInputs) -> Cap:
    """Re-strike ``cap`` when the level carries a strike input."""
    fixed = inputs.fixed or {}
    if "strike" not in fixed:
        return cap
    return eqx.tree_at(lambda c: c.strike, cap, fixed["strike"])


def _default_quotes(inputs: LevelInputs) -> Float[Array, " n_caps"]:
    """Quotes stored directly as an array on the level inputs."""
    return inputs.quotes


def _cap_residual_fn(
    caps: tuple,
    curves_fn: Callable,
    quote_kind: str,
    is_normal: bool,
    quotes_fn: Callable[[LevelInputs], Float[Array, " n_caps"]] = _default_quotes,
) -> Callable[[int, Float[Array, " n_caps"], LevelInputs], Float[Array, ""]]:
    """Price error of quoted cap ``i`` per unit notional (``i`` static)."""
    index, _ = _strip_plan(caps)
    pricer = cap_price_bachelier if is_normal else cap_price_black76

    def one(i: int, sigmas: Float[Array, " n_caps"], inputs: LevelInputs):
        curve, forward_curve = curves_fn(inputs.upstream)
        cap = _struck(caps[i], inputs)
        n_i = int(jnp.asarray(cap.fixing_dates).shape[0])
        model = pricer(cap, curve, sigmas[index[:n_i]], forward_curve)
        q = quotes_fn(inputs)[i]
        if quote_kind == "vol":
            target = pricer(cap, curve, q, forward_curve)
        else:
            target = q
        return (model - target) / cap.notional

    return one


def caplet_strip_residual_fn(
    caps: Sequence[Cap],
    curves_fn: Callable[[Any], tuple[DiscountCurve, DiscountCurve | None]],
    *,
    quote_kind: str = "vol",
    is_normal: bool = False,
    quotes_fn: Callable[[LevelInputs], Float[Array, " n_caps"]] | None = None,
) -> Callable[[Float[Array, " n_caps"], LevelInputs], Float[Array, " n_caps"]]:
    """Residual of the cap strip (see :func:`caplet_strip_level`).

    Args:
        caps: Quoted caps, ascending and nested.
        curves_fn: ``upstream -> (discount_curve, forward_curve | None)``.
        quote_kind: ``"vol"`` or ``"premium"``.
        is_normal: Bachelier (True) or Black-76 (False) pricing.
        quotes_fn: ``inputs -> (n_caps,)`` quotes; defaults to
            ``inputs.quotes``.

    Returns:
        ``(bucket_vols, inputs) -> per-cap price error per unit notional``.
    """
    caps = tuple(caps)
    one = _cap_residual_fn(
        caps, curves_fn, quote_kind, is_normal,
        _default_quotes if quotes_fn is None else quotes_fn,
    )

    def residual(sigmas: Float[Array, " n_caps"], inputs: LevelInputs):
        return jnp.stack([one(i, sigmas, inputs) for i in range(len(caps))])

    return residual


def strip_caplet_vols(
    caps: Sequence[Cap],
    curve: DiscountCurve,
    *,
    cap_vols: Float[Array, " n_caps"] | None = None,
    cap_premiums: Float[Array, " n_caps"] | None = None,
    forward_curve: DiscountCurve | None = None,
    is_normal: bool = False,
    interp: str = "piecewise_constant",
    initial_vol: Float[Array, ""] | float | None = None,
    settings: SolverSettings | None = None,
    name: str = "caplet_strip",
) -> CalibrationResult:
    """Strip a quoted cap strip into per-optionlet volatilities.

    Solves the nested bootstrap described in the module docstring as one
    lower-triangular root level (:func:`caplet_strip_level`): the vol shared
    by each maturity's *incremental* optionlets is the unknown of that
    maturity's equation. The result reprices every quoted cap by
    construction, reports convergence like every other level, and names a
    failed bucket (``"cap 5Y"``) in ``result.check()``.

    Dual-curve by default: forwards are projected off ``forward_curve``
    and payoffs discounted off ``curve``, so the same quotes strip to
    different vols under a different tenor basis.  Pass no
    ``forward_curve`` for the single-curve strip.

    Args:
        caps: Quoted caps (or floors) on a common schedule, ascending in
            maturity and nested — cap ``i``'s optionlets must be cap
            ``i-1``'s plus new ones.  All must share strike, payoff
            direction and day count.
        curve: Discount (OIS / collateral) curve.
        cap_vols: Flat quoted volatility per cap, in the ``is_normal``
            convention.  The market's usual quoting form; converted to
            premiums inside the residual.  Mutually exclusive with
            ``cap_premiums``.
        cap_premiums: Quoted premium per cap, for desks that quote price.
            Mutually exclusive with ``cap_vols``.
        forward_curve: Tenor / projection curve for the forward rates.
            Defaults to ``curve``.
        is_normal: If True the quotes are normal (Bachelier) vols and the
            strip runs on
            :func:`~valax.pricing.analytic.caplets.cap_price_bachelier`
            — the branch to use at or below zero rates.  Otherwise
            lognormal Black-76.
        interp: Interpolation mode recorded on the returned curve; see
            :class:`CapletVolCurve`.
        initial_vol: First guess for every bucket.  Defaults to the quoted
            flat vols when ``cap_vols`` is given, else 20% / 80 bp by
            convention.
        settings: Solver settings; defaults to :data:`STRIP_SETTINGS`
            (Newton to 1e-12).
        name: Level name used in diagnostics.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the
        :class:`CapletVolCurve` covering every optionlet of the longest
        quoted cap is ``result.value``.

    Raises:
        ValueError: If the strip is empty, the quotes are ambiguous or
            mis-sized, the caps disagree on strike / direction / day
            count, the schedules are not nested, or an optionlet has
            already fixed (zero vega — drop it from the quoted strip).
    """
    _validate_strip(caps, curve, cap_vols, cap_premiums)
    quote_kind = "vol" if cap_vols is not None else "premium"
    quotes = jnp.asarray(cap_vols if cap_vols is not None else cap_premiums)
    initial = None
    if initial_vol is not None:
        initial = jnp.full((len(caps),), jnp.asarray(initial_vol, dtype=float))
    return caplet_strip_level(
        name, caps,
        LevelInputs(quotes=quotes, upstream=(curve, forward_curve)),
        lambda up: up,
        quote_kind=quote_kind, is_normal=is_normal, interp=interp,
        initial_vols=initial,
        labels=cap_strip_labels(caps),
        settings=settings,
    )


def cap_price_from_strip(
    cap: Cap,
    vol_curve: CapletVolCurve,
    curve: DiscountCurve,
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, ""]:
    """Reprice a cap with a stripped vol term structure.

    The round-trip check the strip guarantees: applied to a quoted cap
    this returns its market premium.  Reads one vol per optionlet from
    ``vol_curve`` and delegates to the same pricer the strip used, so
    nothing about the payoff is re-derived here.

    Args:
        cap: Cap/floor to price.  Its strike should match
            ``vol_curve.strike`` — a strip is a single-strike column.
        vol_curve: Stripped volatilities.
        curve: Discount curve.
        forward_curve: Optional projection curve; must be the one the
            strip was run against for the repricing to be consistent.

    Returns:
        Cap or floor price.
    """
    pricer = cap_price_bachelier if vol_curve.is_normal else cap_price_black76
    vols = vol_curve(optionlet_expiries(cap, curve))
    return pricer(cap, curve, vols, forward_curve)


# ── Strip → surface ──────────────────────────────────────────────────


def strip_optionlet_surface(
    caps: Sequence[Cap],
    strikes: Float[Array, " n_strikes"],
    cap_vol_matrix: Float[Array, "n_strikes n_caps"],
    curve: DiscountCurve,
    *,
    forward_curve: DiscountCurve | None = None,
    is_normal: bool = False,
    shift: Float[Array, ""] | float = 0.0,
    fixed_beta: Float[Array, ""] | float | None = 0.5,
    settings: SolverSettings | None = None,
    strip_settings: SolverSettings | None = None,
    name: str = "optionlet_surface",
) -> CalibrationResult:
    r"""Strip a quoted cap matrix into a SABR optionlet volatility surface.

    The whole quotes → surface leg in one call, as two levels:

    1. **strip** every strike column (:func:`caplet_strip_level`, batched
       over strikes so each column reports its own convergence, labelled
       ``"K=3.00%"``) — every quoted cap is repriced exactly, dual-curve;
    2. **fit** one SABR smile per optionlet expiry on the stripped vols
       (:func:`valax.surfaces.optionlet_surface.build_sabr_caplet_surface`),
       anchored on the projected forwards read off the curves with
       :func:`optionlet_forwards`.

    The returned result is the SABR level, with the strip level as its
    child, so ``result.check()`` covers both and names the failing strike
    column or expiry slice.

    Args:
        caps: Nested cap schedule, ascending in maturity.  The strike
            carried by these instruments is ignored — each column is
            re-struck at the corresponding entry of ``strikes`` — but
            every other field (schedule, notional, direction, day count)
            is used as given.
        strikes: Quoted strike grid.  Needs at least four points per
            expiry for a four-parameter SABR fit (three with
            ``fixed_beta``).
        cap_vol_matrix: Flat quoted cap vols, row ``k`` being the term
            structure at ``strikes[k]``, in the ``is_normal``
            convention.
        curve: Discount (OIS / collateral) curve.
        forward_curve: Tenor / projection curve.  Defaults to ``curve``.
        is_normal: Quote convention of ``cap_vol_matrix``; also selects
            the SABR expansion used for the fit
            (:func:`~valax.pricing.analytic.sabr.sabr_normal_implied_vol`
            when True).
        shift: Displacement for the normal SABR expansion; used only when
            ``is_normal`` is True.  It does **not** enter the strip
            itself, which prices the normal branch with Bachelier.
        fixed_beta: SABR beta, held fixed in every slice; defaults to
            ``0.5``.  Unlike
            :func:`~valax.surfaces.optionlet_surface.build_sabr_caplet_surface`,
            which leaves beta free, this routine fixes it by default
            because beta and alpha share a backbone
            (:math:`\sigma_{ATM} \approx \alpha F^{\beta-1}`) and are not
            jointly identifiable from a single cap smile — the
            Levenberg-Marquardt fit stalls rather than converging, which
            is the arity/identifiability trap documented in the
            [calibration guide](../guide/calibration.md#solvers).  Rates
            desks fix beta by convention (0.5, or 0 for a normal-quoted
            market).  Pass ``None`` to free it anyway.
        settings: Per-slice SABR solver settings; defaults to
            :data:`~valax.calibration.settings.FIT_SETTINGS`.
        strip_settings: Strip solver settings; defaults to
            :data:`STRIP_SETTINGS`.
        name: Level name of the SABR fit (the strip is ``name + ".strip"``).

    Returns:
        The SABR level's
        :class:`~valax.calibration.result.CalibrationResult`; the
        :class:`~valax.surfaces.optionlet_surface.OptionletVolSurface`
        (expiry grid = the optionlet grid of the longest cap) is
        ``result.value`` and the strip is ``result.children[0]``.

    Raises:
        ValueError: If ``cap_vol_matrix`` does not have shape
            ``(len(strikes), len(caps))``, or any per-strike strip is
            invalid (see :func:`strip_caplet_vols`).
    """
    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    cap_vol_matrix = jnp.asarray(cap_vol_matrix, dtype=jnp.float64)
    n_strikes = int(strikes.shape[0])
    expected = (n_strikes, len(caps))
    if cap_vol_matrix.shape != expected:
        raise ValueError(
            f"strip_optionlet_surface: cap_vol_matrix has shape "
            f"{cap_vol_matrix.shape}, expected {expected} = "
            f"(len(strikes), len(caps))."
        )
    for k in range(n_strikes):
        struck = [eqx.tree_at(lambda c: c.strike, cap, strikes[k]) for cap in caps]
        _validate_strip(struck, curve, cap_vol_matrix[k], None)

    strip = strip_columns_level(
        f"{name}.strip", caps,
        LevelInputs(quotes=cap_vol_matrix, upstream=(curve, forward_curve)),
        lambda up: up, strikes=strikes, is_normal=is_normal,
        settings=strip_settings,
    )
    columns = strip.value
    vol_grid = columns.vols                       # (n_strikes, n_optionlets)
    expiries = columns.expiries[-1]
    forwards = columns.forwards[-1]
    n_expiries = int(expiries.shape[0])

    fit = build_sabr_caplet_surface(
        strikes_per_expiry=[strikes for _ in range(n_expiries)],
        market_vols_per_expiry=[vol_grid[:, j] for j in range(n_expiries)],
        forwards=forwards,
        expiries=expiries,
        is_normal=is_normal,
        shift=shift,
        fixed_beta=(
            None if fixed_beta is None else jnp.asarray(fixed_beta)
        ),
        settings=settings,
        name=name,
    )
    return dataclasses.replace(fit, children=(strip,))


def strip_columns_level(
    name: str,
    caps: Sequence[Cap],
    inputs: LevelInputs,
    curves_fn: Callable[[Any], tuple[DiscountCurve, DiscountCurve | None]],
    *,
    strikes: Float[Array, " n_strikes"],
    quotes_fn: Callable[[LevelInputs], Float[Array, "n_strikes n_caps"]] | None = None,
    is_normal: bool = False,
    settings: SolverSettings | None = None,
) -> CalibrationResult:
    """Strip several strike columns of a cap matrix as one batched level.

    Each column is the strip of :func:`caplet_strip_level` re-struck at its
    strike, solved bucket by bucket and independently of the other columns
    (own convergence flags, labelled ``"K=3.00%"``) inside one ``vmap``. The
    curves ``curves_fn(inputs.upstream)`` are shared by every column and stay
    differentiable upstream inputs.

    Args:
        name: Level name used in diagnostics.
        caps: Nested cap schedule (strikes are replaced per column).
        inputs: The level's inputs as stored on the result.
        curves_fn: ``upstream -> (discount_curve, forward_curve | None)``.
        strikes: Column strikes (conventions, held constant).
        quotes_fn: ``inputs -> (n_strikes, n_caps)`` flat cap vols; defaults
            to ``inputs.quotes``.
        is_normal: Bachelier (True) or Black-76 (False) pricing.
        settings: Per-column solver settings; defaults to
            :data:`STRIP_SETTINGS`.

    Returns:
        A batched :class:`~valax.calibration.result.CalibrationResult` whose
        value is a :class:`CapletVolCurve` with a leading strike axis.
    """
    settings = STRIP_SETTINGS if settings is None else settings
    caps = tuple(caps)
    index, ends = _strip_plan(caps)
    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    n = int(strikes.shape[0])
    quotes_fn = _default_quotes if quotes_fn is None else quotes_fn
    identity = lambda up: up  # noqa: E731 — node inputs carry the curves
    residual = caplet_strip_residual_fn(caps, identity, is_normal=is_normal)

    def node_inputs(inp: LevelInputs) -> LevelInputs:
        curves = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (n,) + jnp.shape(x)),
            curves_fn(inp.upstream),
        )
        return LevelInputs(
            quotes=quotes_fn(inp), upstream=curves, fixed={"strike": strikes},
        )

    def assemble(sigmas: Float[Array, "n_strikes n_caps"], inp: LevelInputs):
        curve, forward_curve = curves_fn(inp.upstream)
        longest = caps[-1]
        expiries = optionlet_expiries(longest, curve)
        fwds = optionlet_forwards(longest, curve, forward_curve)
        return CapletVolCurve(
            expiries=jnp.broadcast_to(expiries, (n,) + expiries.shape),
            vols=sigmas[:, index],
            strike=strikes,
            forwards=jnp.broadcast_to(fwds, (n,) + fwds.shape),
            bucket_ends=ends,
            is_normal=is_normal,
        )

    concrete = _concrete(strikes)
    labels = (
        tuple(f"K={100 * float(k):.2f}%" for k in concrete)
        if concrete is not None
        else tuple(f"K{i}" for i in range(n))
    )

    def node_solve(theta0, node_in: LevelInputs):
        return strip_buckets(
            caps, identity, node_in, is_normal=is_normal,
            initial_vols=theta0, settings=settings,
        )

    return solve_batched_level(
        name, "root", residual,
        jax.lax.stop_gradient(jnp.asarray(quotes_fn(inputs), dtype=jnp.float64)),
        inputs, node_inputs, assemble, settings, labels=labels,
        node_solve_fn=node_solve,
    )


__all__ = [
    "STRIP_SETTINGS",
    "CapletVolCurve",
    "cap_price_from_strip",
    "cap_strip_labels",
    "caplet_strip_level",
    "caplet_strip_residual_fn",
    "strip_buckets",
    "optionlet_expiries",
    "optionlet_forwards",
    "strip_caplet_vols",
    "strip_columns_level",
    "strip_optionlet_surface",
]
