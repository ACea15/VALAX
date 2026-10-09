"""Swap legs and multi-leg trade representation (data-only pytrees).

A *leg* is a schedule of accrual periods plus the rule that turns each
period into a cashflow. Legs are the reusable unit the older pytrees in
:mod:`valax.instruments.rates` and :mod:`valax.instruments.bonds` lack:
those classes hard-code a two-leg shape (``fixed_dates`` + ``float_dates``
and a singular ``forward_id``), which cannot express a trade whose legs
project off *different* indices.

Two things are therefore built here as one structure, because the
roadmap requires it:

* **The leg abstraction** — :class:`FixedLeg` and :class:`FloatingLeg`,
  carrying compounding conventions, an amortising notional and the index
  whose realised fixings season the leg.
* **Per-leg index descriptors** — every leg names its *own*
  ``discount_id`` / ``forward_id`` / ``index_id``, and
  :class:`MultiLegSwap` holds ``legs: tuple[...]``. A singular
  ``forward_id`` on a trade expresses only the single-index case; a
  single-currency tenor basis swap needs three curves (OIS discount plus
  two projections) and a CMS-spread swap three. Promoting the descriptor
  to the leg is what lifts that ceiling.

Pricing lives in :mod:`valax.pricing.analytic.legs`; these classes carry
no methods beyond structural metadata, per the project's data-only
instrument rule.

Notional conventions
--------------------
``notional`` is a **scalar reference notional** and ``amortisation`` is
an optional array of **dimensionless outstanding-notional factors**, one
per period, so period :math:`i` pays on
:math:`N \\cdot a_i`. A 100mm swap amortising to 50mm half way through
carries ``notional=1e8`` and factors ``[1, 1, 0.5, 0.5]``, not absolute
notionals.

That is a deliberate choice, not a stylistic one. Every pricer in VALAX
computes a unit-notional leg PV and scales it *afterwards*
(``notional * jnp.sum(...)``). Folding absolute per-period notionals into
the sum instead would change the floating-point result:
:math:`\\sum_i (N x_i)` and :math:`N \\sum_i x_i` differ in the last bit
for most inputs. Factors keep the notional outside the sum, and because
multiplication by exactly ``1.0`` is exact in IEEE-754, a leg with no
amortisation reproduces the flat pricers **bit for bit** by construction
rather than by luck.
"""

from typing import Optional, Union

import equinox as eqx
import numpy as np
from jaxtyping import Float, Int
from jax import Array

from valax.dates.daycounts import ALL_DAY_COUNT_NAMES, DAY_COUNT_FNS


def max_period_days(
    accrual_start_dates: Int[Array, " n"],
    accrual_end_dates: Int[Array, " n"],
) -> int:
    """Longest accrual period in days, as a Python ``int``.

    The bound to pass as :attr:`FloatingLeg.max_days` for an averaged
    leg. Deliberately **eager**: it reads the date values, so it must be
    called before a ``jit`` boundary, and it returns a plain ``int`` so
    the result is hashable and therefore usable as a static field.

    It is a separate function rather than something the leg derives in
    ``__check_init__`` because ``__check_init__`` re-runs on any
    ``__init__`` executed inside a trace, where the dates are tracers and
    reading their values raises. See
    [JAX Patterns § 2.5](../architecture/jax-patterns.md).

    Args:
        accrual_start_dates: Accrual start of each period (ordinals).
        accrual_end_dates: Accrual end of each period (ordinals).

    Returns:
        The maximum of ``end - start`` across periods.

    Raises:
        ValueError: If the dates are traced, or if even the longest
            period is shorter than a day.
    """
    try:
        spans = np.asarray(accrual_end_dates) - np.asarray(
            accrual_start_dates
        )
    except Exception as exc:
        raise ValueError(
            "max_period_days needs concrete dates: it is the eager step "
            "that turns a schedule into the static bound an averaged leg "
            "compiles against. Call it outside the jit boundary and pass "
            "the result as FloatingLeg(max_days=...)."
        ) from exc
    longest = int(spans.max())
    if longest < 1:
        raise ValueError(
            f"The longest accrual period spans {longest} days; it needs "
            f"at least one. Check that accrual_end_dates are strictly "
            f"after accrual_start_dates."
        )
    return longest


#: Compounding conventions available to :class:`FloatingLeg`, which
#: documents each one and the two constraints on ``"averaged"``.
COMPOUNDING_CONVENTIONS = ("simple", "compounded", "averaged")


def _check_day_count(leg: Union["FixedLeg", "FloatingLeg"]) -> None:
    """Validate a leg's day count at construction time.

    Both the name and its usability are checked here rather than left to
    fail inside ``year_fraction`` at pricing time, which would surface a
    configuration mistake as a pricing error.

    Args:
        leg: The leg being constructed.

    Raises:
        ValueError: If the convention is unknown, or needs a calendar the
            leg pricers cannot supply.
    """
    name = leg.day_count
    if name not in ALL_DAY_COUNT_NAMES:
        raise ValueError(
            f"Unknown day count {name!r}; expected one of "
            f"{ALL_DAY_COUNT_NAMES}."
        )
    if name not in DAY_COUNT_FNS:
        # i.e. it is in CALENDAR_DAY_COUNT_FNS -- Bus/252 today.
        raise ValueError(
            f"day_count={name!r} needs a business-day calendar, but a "
            f"leg carries none and the leg pricers measure accrual with "
            f"the two-argument conventions only. Giving it a default "
            f"calendar is precisely the silent-reference-data failure "
            f"valax.dates.conventions exists to prevent, so it is "
            f"refused instead."
        )


def _check_leg_shapes(
    leg: Union["FixedLeg", "FloatingLeg"],
    extra: dict[str, Optional[Array]],
) -> None:
    """Validate that a leg's aligned arrays share one length.

    Structural only — reads ``.shape`` and ``is None``, never a numeric
    value, so it is safe to call from ``__check_init__``, which re-runs
    on any ``__init__`` executed inside a trace, where every array field
    is a tracer and a value check would raise
    ``TracerBoolConversionError``.

    Args:
        leg: The leg being constructed.
        extra: Optional per-period arrays to check when present, keyed by
            field name for the error message.

    Raises:
        ValueError: If any array is not one-dimensional, if the schedule
            arrays disagree in length, or if a supplied optional array
            does not match the schedule length.
    """
    schedule = {
        "accrual_start_dates": leg.accrual_start_dates,
        "accrual_end_dates": leg.accrual_end_dates,
        "payment_dates": leg.payment_dates,
    }
    for name, arr in schedule.items():
        if arr.ndim != 1:
            raise ValueError(
                f"{type(leg).__name__}.{name} must be one-dimensional, "
                f"got shape {arr.shape}. A leg is a flat sequence of "
                f"accrual periods."
            )
    n = leg.accrual_end_dates.shape[0]
    if n == 0:
        raise ValueError(
            f"{type(leg).__name__} has no accrual periods. A leg needs at "
            f"least one."
        )
    for name, arr in schedule.items():
        if arr.shape[0] != n:
            raise ValueError(
                f"{type(leg).__name__} schedule arrays disagree: "
                f"accrual_end_dates has {n} periods but {name} has "
                f"{arr.shape[0]}. Period i accruing over "
                f"[accrual_start[i], accrual_end[i]) and paying on "
                f"payment_dates[i] only makes sense if all three align."
            )
    for name, arr in extra.items():
        if arr is None:
            continue
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(
                f"{type(leg).__name__}.{name} must have one entry per "
                f"accrual period ({n}), got shape {arr.shape}."
            )


class FixedLeg(eqx.Module):
    """A leg paying a fixed rate on an accrual schedule.

    Period :math:`i` accrues over
    :math:`[T_i^s, T_i^e)` and pays on :math:`T_i^p`:

    .. math::
        \\text{PV} = N \\sum_i a_i\\, c\\, \\tau_i\\, P_d(0, T_i^p),

    with :math:`c` the fixed rate, :math:`\\tau_i` the accrual fraction
    in the leg's day count, :math:`a_i` the outstanding-notional factor
    and :math:`P_d` the discount curve named by ``discount_id``.

    Note that :math:`\\tau_i` comes from the *accrual* schedule while the
    discount factor is read at the *payment* date. The two coincide only
    when there is no payment lag.

    Attributes:
        accrual_start_dates: Accrual start of each period (ordinals).
        accrual_end_dates: Accrual end of each period (ordinals).
        payment_dates: Date each period's cashflow settles (ordinals).
            Equals ``accrual_end_dates`` when there is no payment lag.
        fixed_rate: The annual fixed rate, as a decimal.
        notional: Reference notional. See the module docstring for why
            amortisation is expressed as factors against this scalar.
        amortisation: Outstanding-notional factors, one per period.
            ``None`` means a flat leg, which is both the common case and
            the bit-for-bit-preserved fast path.
        pay: ``True`` if the leg is paid away, so its present value
            enters the trade with a minus sign.
        day_count: Day count convention for the accrual fractions.
        discount_id: Identifier of the curve this leg discounts on.
    """

    accrual_start_dates: Int[Array, " n"]
    accrual_end_dates: Int[Array, " n"]
    payment_dates: Int[Array, " n"]
    fixed_rate: Float[Array, ""]
    notional: Float[Array, ""]
    amortisation: Optional[Float[Array, " n"]] = None
    pay: bool = eqx.field(static=True, default=False)
    day_count: str = eqx.field(static=True, default="act_360")
    discount_id: str = eqx.field(static=True, default="_default_")

    def __check_init__(self) -> None:
        """Validate structure only — never a traced value."""
        _check_day_count(self)
        _check_leg_shapes(self, {"amortisation": self.amortisation})

    @property
    def n_periods(self) -> int:
        """Number of accrual periods on the leg."""
        return int(self.accrual_end_dates.shape[0])

    @property
    def curves_touched(self) -> tuple[str, ...]:
        """Curve identifiers this leg reads. A fixed leg only discounts."""
        return (self.discount_id,)


class FloatingLeg(eqx.Module):
    """A leg paying a floating index on an accrual schedule.

    The coupon for period :math:`i` depends on ``compounding``:

    .. math::
        \\text{simple:}\\quad & (r_i + s)\\,\\tau_i \\\\
        \\text{compounded:}\\quad & \\prod_j (1 + r_j \\delta_j) - 1
            + s\\,\\tau_i \\\\
        \\text{averaged:}\\quad & (\\bar{r}_i + s)\\,\\tau_i

    where :math:`r_j` runs over the daily index values inside the period:

    * ``"simple"`` — one fixing per accrual period. The IBOR-style leg.
    * ``"compounded"`` — the post-LIBOR RFR standard (SOFR, €STR,
      SONIA). Evaluated as the discount-factor ratio
      :math:`P_f(0,T_i^s)/P_f(0,T_i^e) - 1`, because the daily product
      telescopes, so the day count cancels out of the index term and
      survives only on the spread.
    * ``"averaged"`` — arithmetic rather than compound, discarding
      interest-on-interest. Used by a few markets and by averaged
      overnight futures.

    **The averaged convention needs a shape at trace time.** The average
    is defined per rate day, so a period must be subdivided into its
    actual days, and that array's length depends on the dates' *values*.
    There are therefore three ways to price one:

    * **eagerly** — always works;
    * **closed over** a traced function — works, because the dates are
      then compile-time constants;
    * **as a traced argument** (and hence under ``vmap``) — needs
      ``max_days``, a static bound on the subdivision. Derive it with
      :func:`max_period_days`.

    Without ``max_days`` the third case raises, naming the way out. Only
    the *bound* is static; the dates stay dynamic, which is what keeps
    one compilation serving every schedule and keeps legs stackable. See
    [JAX Patterns § 2.5](../architecture/jax-patterns.md).

    An averaged leg also requires an actual/N day count, since the
    weighted average collapses to a sum of daily ratios only when a
    period's year fraction equals the sum of its daily fractions.

    The vocabulary matches
    :attr:`valax.instruments.rates.OISSwap.compounding`
    (``"compounded"`` / ``"averaged"``), which is the market's wording.
    :data:`valax.curves.instruments.AVERAGING_CONVENTIONS` says
    ``"arithmetic"`` for the same convention on a money-market future;
    the two names denote the same arithmetic, and
    :func:`valax.pricing.analytic.legs.floating_leg_pv` reuses that
    module's implementation rather than defining it twice.

    **Fixing timing is carried by the dates, not by a flag.** An
    in-advance leg sets ``fixing_dates`` to the accrual starts; a
    compounded or averaged RFR leg observes the index *across* its
    accrual period, so in-arrears is structural and ``fixing_dates`` is
    not used at all — supplying it is rejected rather than silently
    ignored. There is deliberately no ``in_advance``/``in_arrears``
    switch, because one that changed no behaviour would be the same dead
    convention field as the never-read
    :attr:`valax.instruments.rates.OISSwap.compounding`.

    **Not modelled:** a genuine *LIBOR-in-arrears* leg, where the rate
    observed at :math:`T_i` for :math:`[T_i, T_i + \\text{tenor}]` is
    applied to :math:`[T_{i-1}, T_i]`. Its value is not the forward — it
    needs a convexity adjustment — so projecting it off the curve would
    be silently wrong rather than approximate. It is absent, not
    defaulted.

    Seasoning is resolved at pricing time from a
    :class:`valax.curves.fixings.FixingHistory` keyed by ``index_id``:
    a period whose fixing is already recorded uses it, a period that
    straddles the valuation date blends its realised compounding with the
    remaining projection, and everything later is pure projection. The
    leg pytree itself holds no fixing values, so one leg can be revalued
    on successive histories.

    Attributes:
        accrual_start_dates: Accrual start of each period (ordinals).
        accrual_end_dates: Accrual end of each period (ordinals).
        payment_dates: Date each period's cashflow settles (ordinals).
        spread: Constant spread over the index, as a decimal.
        notional: Reference notional. See the module docstring for why
            amortisation is expressed as factors against this scalar.
        fixing_dates: Observation date of each period's rate (ordinals).
            Required for ``compounding="simple"``; rejected for the
            compounded and averaged conventions, which observe the index
            over the accrual period itself.
        amortisation: Outstanding-notional factors, one per period.
            ``None`` means a flat leg.
        pay: ``True`` if the leg is paid away, so its present value
            enters the trade with a minus sign.
        compounding: One of :data:`COMPOUNDING_CONVENTIONS`.
        day_count: Day count convention for the accrual fractions.
        index_id: Identifier of the floating index, used to look up
            realised fixings.
        discount_id: Identifier of the curve this leg discounts on.
        forward_id: Identifier of the projection curve. ``None`` means
            project off the discount curve, i.e. single-curve.
        max_days: Static upper bound on any accrual period's span in
            days, used only by ``compounding="averaged"``. Supplying it
            lets the leg be priced as a *traced* argument — and hence
            under ``vmap`` — because the daily subdivision then has a
            trace-time shape. Derive it with :func:`max_period_days`.
            ``None`` keeps the eager route, which reads the span off the
            dates and so needs them concrete.
    """

    accrual_start_dates: Int[Array, " n"]
    accrual_end_dates: Int[Array, " n"]
    payment_dates: Int[Array, " n"]
    spread: Float[Array, ""]
    notional: Float[Array, ""]
    fixing_dates: Optional[Int[Array, " n"]] = None
    amortisation: Optional[Float[Array, " n"]] = None
    pay: bool = eqx.field(static=True, default=False)
    compounding: str = eqx.field(static=True, default="compounded")
    day_count: str = eqx.field(static=True, default="act_360")
    index_id: str = eqx.field(static=True, default="_default_")
    discount_id: str = eqx.field(static=True, default="_default_")
    forward_id: Optional[str] = eqx.field(static=True, default=None)
    max_days: Optional[int] = eqx.field(static=True, default=None)

    def __check_init__(self) -> None:
        """Validate structure and convention pairing — never a value."""
        if self.compounding not in COMPOUNDING_CONVENTIONS:
            raise ValueError(
                f"Unknown compounding {self.compounding!r}; expected one "
                f"of {COMPOUNDING_CONVENTIONS}."
            )
        _check_day_count(self)
        if self.max_days is not None:
            # Same pairing discipline as ``fixing_dates``: a field no
            # branch would consult is a configuration error, not a
            # harmless extra.  Only the averaged convention subdivides a
            # period, so only it has a shape to bound.
            if self.compounding != "averaged":
                raise ValueError(
                    f"max_days only applies to "
                    f"compounding='averaged', not "
                    f"{self.compounding!r}: it bounds the daily "
                    f"subdivision of an accrual period, and the simple "
                    f"and compounded conventions never subdivide one. "
                    f"Drop max_days."
                )
            if self.max_days < 1:
                raise ValueError(
                    f"max_days must be at least 1 day, got "
                    f"{self.max_days}."
                )
        if self.compounding == "averaged" and self.day_count == "30_360":
            # The averaged forward is a *day-count-weighted* average, and
            # it collapses to a sum of daily discount-factor ratios only
            # because tau equals the sum of the daily accrual fractions.
            # That holds for the actual/N conventions and fails for
            # 30/360, whose year fraction is not proportional to elapsed
            # days -- so the result would be a different quantity, not a
            # coarser approximation of the same one.
            raise ValueError(
                "compounding='averaged' needs an actual/N day count: the "
                "averaged forward is a day-count-weighted average, which "
                "collapses to a sum of daily ratios only when the "
                "period's year fraction equals the sum of its daily "
                "fractions. That is false for '30_360', so the average "
                "would silently be a different quantity. Use 'act_360' "
                "(the market convention for overnight indices), or "
                "compounding='compounded'."
            )
        _check_leg_shapes(
            self,
            {
                "fixing_dates": self.fixing_dates,
                "amortisation": self.amortisation,
            },
        )
        # Pairing guards.  A simple leg has nowhere to read its rate from
        # without an observation date, and a compounded or averaged leg
        # would never consult one -- accepting it would imply an
        # observation shift or lookback that is not modelled here.
        if self.compounding == "simple" and self.fixing_dates is None:
            raise ValueError(
                "compounding='simple' needs fixing_dates: a simple leg "
                "pays one observed rate per period, so each period must "
                "name the date it is observed. Pass the accrual starts "
                "for a standard in-advance leg."
            )
        if self.compounding != "simple" and self.fixing_dates is not None:
            raise ValueError(
                f"compounding={self.compounding!r} does not read "
                f"fixing_dates: the index is observed across the whole "
                f"accrual period, so a single observation date per "
                f"period has no meaning. Drop fixing_dates, or use "
                f"compounding='simple' if the leg really does pay one "
                f"fixing per period."
            )

    @property
    def n_periods(self) -> int:
        """Number of accrual periods on the leg."""
        return int(self.accrual_end_dates.shape[0])

    @property
    def curves_touched(self) -> tuple[str, ...]:
        """Curve identifiers this leg reads.

        ``(discount_id,)`` when the leg is single-curve, otherwise
        ``(discount_id, forward_id)`` — the same convention as
        :mod:`valax.curves.instruments`.
        """
        if self.forward_id is None:
            return (self.discount_id,)
        return (self.discount_id, self.forward_id)


#: A leg of either kind.
Leg = Union[FixedLeg, FloatingLeg]


class MultiLegSwap(eqx.Module):
    """A trade represented as an arbitrary tuple of legs.

    The present value is the sum of the legs' present values, each
    already signed by its own ``pay`` flag. Nothing here constrains how
    many legs there are or what they project off, which is the point: a
    vanilla IRS is one fixed and one floating leg, a tenor basis swap is
    two floating legs with *different* ``forward_id``s over a shared
    discount curve, and a CMS-spread leg structure is three curves.

    **Single currency.** The legs' present values are added directly, so
    every leg must discount in the same currency. A cross-currency swap
    additionally needs an FX rate and a second discount curve to convert
    one leg into the other's currency; that conversion is not performed
    here and no ``currency`` field is carried, rather than carrying one
    that nothing reads.

    Attributes:
        legs: The legs of the trade, in any order.
    """

    legs: tuple[Leg, ...]

    def __check_init__(self) -> None:
        """Validate structure only — never a traced value."""
        if not isinstance(self.legs, tuple):
            raise TypeError(
                f"MultiLegSwap.legs must be a tuple, got "
                f"{type(self.legs).__name__}. A tuple is what makes the "
                f"leg count static, so the trade's pytree structure is "
                f"fixed at trace time."
            )
        if len(self.legs) == 0:
            raise ValueError("MultiLegSwap has no legs.")
        for i, leg in enumerate(self.legs):
            if not isinstance(leg, (FixedLeg, FloatingLeg)):
                raise TypeError(
                    f"MultiLegSwap.legs[{i}] is a "
                    f"{type(leg).__name__}; expected a FixedLeg or a "
                    f"FloatingLeg."
                )

    @property
    def curves_touched(self) -> tuple[str, ...]:
        """Every curve identifier the trade reads, de-duplicated.

        Order is first appearance across ``legs``, so the result is
        deterministic and independent of hashing. This is the multi-index
        generalisation of the ``(discount_id, forward_id)`` pair carried
        by the single-index quote types in
        :mod:`valax.curves.instruments`.
        """
        seen: list[str] = []
        for leg in self.legs:
            for curve_id in leg.curves_touched:
                if curve_id not in seen:
                    seen.append(curve_id)
        return tuple(seen)

    @property
    def indices_touched(self) -> tuple[str, ...]:
        """Identifiers of the floating indices the trade fixes against.

        De-duplicated in order of first appearance; empty for an
        all-fixed trade. This is what a fixing registry would have to
        supply a history for.
        """
        seen: list[str] = []
        for leg in self.legs:
            if isinstance(leg, FloatingLeg) and leg.index_id not in seen:
                seen.append(leg.index_id)
        return tuple(seen)
