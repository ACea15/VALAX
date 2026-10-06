r"""Smile-dynamics conventions: what moves when the curve moves.

A fitted smile is anchored on a forward. When the curve moves the forward moves,
and the convention for what happens to the *volatility* is a modelling decision
that determines the delta of every option priced off the surface. The three
standard conventions differ in exactly one place -- which forward enters the
Hagan evaluation, and at which strike:

================================  ======================================
Convention                        Vol read
================================  ======================================
``"sticky_strike"``               :math:`\sigma(K;\, F_0)`
``"sticky_moneyness"``            :math:`\sigma(K - (F - F_0);\, F_0)`
``"sabr_backbone"``               :math:`\sigma(K;\, F)`
================================  ======================================

with :math:`F_0` the forward the node was *fitted* at and :math:`F` the live
forward the curve implies now.

This module owns the convention *mechanics* only, shared by
:class:`~valax.surfaces.swaption_cube.SwaptionCube` and
:class:`~valax.surfaces.optionlet_surface.OptionletVolSurface` so neither
re-implements it. The maths, the chosen default and the caveats are documented
once in the
[vol-surfaces guide](../guide/vol-surfaces.md#smile-dynamics-what-moves-when-the-curve-moves).

The ``dynamics`` / ``quote_form`` pair is validated rather than left free: the
dynamics is a *consequence* of what the quote holds fixed. An offset-to-ATM
quote moves with the forward and so implies sticky moneyness; an absolute-strike
quote does not and so implies sticky strike. ``"sabr_backbone"`` consults no
stored forward at all and is therefore compatible with either quote form. A
contradictory pair is a configuration error, caught at construction.
"""

from jaxtyping import Float
from jax import Array


#: Supported smile-dynamics conventions.
SMILE_DYNAMICS = ("sticky_strike", "sticky_moneyness", "sabr_backbone")

#: Supported quote forms (how the calibrating quote's strike was expressed).
QUOTE_FORMS = ("absolute_strike", "atm_offset")

#: Which dynamics each quote form is consistent with. ``"sabr_backbone"`` is
#: listed under both because it holds the *parameters* fixed rather than the
#: strike or the offset, so it does not contradict either quoting form.
CONSISTENT_DYNAMICS = {
    "absolute_strike": ("sticky_strike", "sabr_backbone"),
    "atm_offset": ("sticky_moneyness", "sabr_backbone"),
}

#: Dynamics that need the live forward at query time.
_NEEDS_LIVE_FORWARD = ("sticky_moneyness", "sabr_backbone")


def validate_smile_dynamics(owner: str, dynamics: str, quote_form: str) -> None:
    """Validate a ``(dynamics, quote_form)`` pair at the user boundary.

    Args:
        owner: Name of the owning class, used in the error messages.
        dynamics: Candidate smile-dynamics convention.
        quote_form: Candidate quote form.

    Raises:
        ValueError: If either value is unrecognised, or if the pair is
            contradictory (e.g. absolute-strike quotes declared sticky in
            moneyness).
    """
    if dynamics not in SMILE_DYNAMICS:
        raise ValueError(
            f"{owner}.dynamics must be one of {SMILE_DYNAMICS}, "
            f"got {dynamics!r}."
        )
    if quote_form not in QUOTE_FORMS:
        raise ValueError(
            f"{owner}.quote_form must be one of {QUOTE_FORMS}, "
            f"got {quote_form!r}."
        )
    allowed = CONSISTENT_DYNAMICS[quote_form]
    if dynamics not in allowed:
        raise ValueError(
            f"{owner} declares quote_form={quote_form!r} with "
            f"dynamics={dynamics!r}, which contradict each other: a "
            f"{quote_form!r} quote is consistent with {allowed}. The "
            f"dynamics is a consequence of what the quote holds fixed -- an "
            f"offset-to-ATM quote moves with the forward (sticky moneyness), "
            f"an absolute-strike quote does not (sticky strike). See the "
            f"vol-surfaces guide, 'Smile dynamics'."
        )


def default_dynamics_for(quote_form: str) -> str:
    """The dynamics a quote form naturally implies.

    Used by calibration entry points that are told how the quotes were
    expressed but not what the desk's dynamics policy is: an absolute-strike
    quote implies sticky strike, an offset-to-ATM quote implies sticky
    moneyness. ``"sabr_backbone"`` is never chosen implicitly -- it is a model
    assertion with a self-consistency prerequisite and must be asked for.

    Args:
        quote_form: One of :data:`QUOTE_FORMS`.

    Returns:
        The implied smile-dynamics convention.

    Raises:
        ValueError: If ``quote_form`` is unrecognised.
    """
    if quote_form not in QUOTE_FORMS:
        raise ValueError(
            f"quote_form must be one of {QUOTE_FORMS}, got {quote_form!r}."
        )
    return CONSISTENT_DYNAMICS[quote_form][0]


def resolve_smile_query(
    dynamics: str,
    strike: Float[Array, ""],
    fitted_forward: Float[Array, ""],
    forward: Float[Array, ""] | None,
    *,
    owner: str,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Map a query onto the ``(strike, forward)`` pair the expansion is evaluated at.

    The single implementation of the convention table in the module docstring.
    ``dynamics`` is a *static* field, so the branch is resolved at trace time
    and the result is JIT-safe.

    Args:
        dynamics: One of :data:`SMILE_DYNAMICS`.
        strike: Absolute strike being queried.
        fitted_forward: The forward this node was fitted at, :math:`F_0`
            (interpolated to the query coordinate by the caller).
        forward: The live forward implied by the curves now, :math:`F`, or
            ``None`` when the caller supplied none.
        owner: Name of the owning class, used in the error message.

    Returns:
        Tuple ``(effective_strike, anchor_forward)`` to pass to the Hagan
        expansion.

    Raises:
        ValueError: If ``dynamics`` needs the live forward but ``forward`` is
            ``None``, or if ``dynamics`` is unrecognised.
    """
    if dynamics == "sticky_strike":
        # Vol-at-absolute-strike is invariant under a curve move: the live
        # forward is deliberately not consulted.
        return strike, fitted_forward

    if dynamics not in _NEEDS_LIVE_FORWARD:
        raise ValueError(
            f"{owner}.dynamics must be one of {SMILE_DYNAMICS}, "
            f"got {dynamics!r}."
        )
    if forward is None:
        raise ValueError(
            f"{owner} has dynamics={dynamics!r}, which is a function of the "
            f"live forward, but was queried without one. Call it through a "
            f"curve-aware pricer (valax.pricing.analytic.rates_smile) or pass "
            f"forward=... explicitly. Only dynamics='sticky_strike' can be "
            f"evaluated with no forward."
        )

    if dynamics == "sticky_moneyness":
        # The quote is attached to the offset delta = K - F, so reading at
        # absolute K means reading the fitted smile at K_0 = F_0 + (K - F).
        return strike - (forward - fitted_forward), fitted_forward

    # "sabr_backbone": parameters held fixed, smile re-evaluated at the live
    # forward. Consults no stored forward.
    return strike, forward
