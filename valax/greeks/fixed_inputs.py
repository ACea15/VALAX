r"""Sensitivities to the *fixed inputs* ``p``: the quasi-market parameters.

A calibration has two kinds of input. The **quotes** ``q`` are solved against —
they define the residual and come back as hedge-space deltas through
:mod:`valax.greeks.adjoint`. The **fixed inputs** ``p`` are everything else the
answer depends on but the solver holds still: Hull-White's mean reversion, the
SABR exponent and shift, futures convexity adjustments, turn spreads, wing
settings.

They are not market quotes and they are not calibrated, which is exactly why
they tend to go unreported — and why the specification makes reporting them a
MUST (A-8, M-5, Q-5). A number that moves the price but appears in no risk
report is an unhedged, unmonitored position. Mean reversion is the standard
example: on a Bermudan it can move the price by more than a basis point of
rates delta, and it is chosen rather than observed.

Why a lens and not a container
------------------------------

The obvious design is a struct that *holds* ``kappa``, ``beta`` and the rest.
That is the wrong one. Those values already live on the model and surface
pytrees the pricer consumes, so a second copy is a second source of truth, and
the day they disagree the risk report describes a model nobody priced with.
The failure mode A-8 names — a sensitivity that is silently wrong rather than
absent — would simply reappear in a new place.

So a :class:`FixedInput` stores **where** the value lives, not the value. The
adjoint is taken by reading the leaf out of the live pytree, re-injecting it,
and differentiating through the injection, so the number reported is by
construction a sensitivity to the leaf that was actually priced with. There is
nothing to keep in sync.

Relationship to the rest of the chain
-------------------------------------

``p̄`` is the third output of the adjoint pass of specification § 7.3,
alongside ``q̄_c`` (curve-quote delta, :func:`~valax.greeks.adjoint.curve_quote_deltas`)
and ``q̄_v`` (vol-quote vega, :func:`~valax.greeks.vol_adjoint.vol_quote_vegas`).
It is reported *separately* rather than folded into either, because a fixed
input is not tradeable: a desk cannot hedge mean reversion, it can only size
and govern the exposure.

Unlike the quote adjoints, ``p̄`` needs no implicit-function-theorem step. A
fixed input is held constant *through* the calibration, so its total derivative
is the direct one — provided the calibration is re-run when it moves, which is
what :func:`fixed_input_adjoints` differentiates when it is handed a function
that recalibrates. See
:func:`~valax.greeks.fixed_inputs.fixed_input_adjoints` for which of the two
questions you are asking.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


class FixedInput(eqx.Module):
    """One fixed input ``p_i``, named and located rather than stored.

    Attributes:
        name: Reporting label, e.g. ``"hull_white.mean_reversion"``. Use a
            dotted path so a risk report sorts sensibly.
        where: Lens from the pytree to the leaf, e.g.
            ``lambda m: m.mean_reversion``. Must select exactly one
            **whole, scalar** leaf. Two limits follow from
            :func:`equinox.tree_at`, and both are checked with a pointed
            error rather than left to surface as a tree-surgery
            failure:

            * *Whole*: ``lambda c: c.betas[0, 0]`` indexes **into** a
              leaf and cannot be used. Lens the scalar field itself, or
              reshape the parameter so the quantity you want to report
              on is its own leaf.
            * *Scalar*: a fixed input is one reported number. Every
              example in the specification — mean reversion, the SABR
              exponent, a displacement, a convexity adjustment, a turn
              spread — is a scalar. A grid is a set of fixed inputs, not
              one.

            Static, so it does not become a traced value.
        units: Free-text units for the report, e.g. ``"1/year"`` or
            ``"decimal"``. Carried so a reader can tell a per-unit
            sensitivity from a per-basis-point one.
    """

    name: str = eqx.field(static=True)
    where: Callable[[Any], Any] = eqx.field(static=True)
    units: str = eqx.field(static=True, default="")


class FixedInputs(eqx.Module):
    """An ordered, named set of fixed inputs over one pytree.

    The order fixes the layout of every vector this module returns, the
    same way ``instruments`` order fixes the layout of ``q̄_c``.

    Attributes:
        entries: The individual inputs, in report order.
    """

    entries: tuple[FixedInput, ...] = eqx.field(static=True)

    def __check_init__(self) -> None:
        names = [entry.name for entry in self.entries]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"FixedInputs names must be unique; duplicated: "
                f"{sorted(duplicates)}. Names label the reported "
                f"sensitivities, so a duplicate makes the report ambiguous."
            )

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def names(self) -> tuple[str, ...]:
        """Reporting labels, in vector order."""
        return tuple(entry.name for entry in self.entries)

    @property
    def units(self) -> tuple[str, ...]:
        """Units, in vector order."""
        return tuple(entry.units for entry in self.entries)

    def values(self, tree: Any) -> Float[Array, " n_p"]:
        """Read the current values out of ``tree``.

        Args:
            tree: The pytree the lenses point into (a model, a surface, a
                tuple of both — whatever ``where`` expects).

        Returns:
            The fixed-input vector ``p``, in :attr:`entries` order.

        Raises:
            ValueError: If a lens selects a non-scalar leaf. Caught here
                rather than left to :func:`equinox.tree_at`, whose
                failure mode for an indexed lens is an opaque
                ``'_LeafWrapper' object is not subscriptable``.
        """
        read = []
        for entry in self.entries:
            value = jnp.asarray(entry.where(tree))
            if value.shape != ():
                raise ValueError(
                    f"Fixed input {entry.name!r} selects a leaf of shape "
                    f"{value.shape}, but a fixed input must be a scalar — "
                    f"it is one reported number. Lens a scalar field, or "
                    f"split the array into the individual parameters you "
                    f"want reported. Note that indexing into a leaf "
                    f"(`lambda m: m.betas[0, 0]`) is not a valid lens for "
                    f"`equinox.tree_at`; the leaf itself must be scalar."
                )
            read.append(value)
        return jnp.stack(read)

    def inject(
        self, tree: Any, values: Float[Array, " n_p"]
    ) -> Any:
        """Return a copy of ``tree`` with ``p`` replaced by ``values``.

        The step that makes the adjoint honest: differentiating through
        this guarantees the reported sensitivity belongs to the leaf the
        pricer reads, not to a stored duplicate of it.

        Args:
            tree: The pytree to update.
            values: Replacement vector, in :attr:`entries` order.

        Returns:
            A new pytree. ``tree`` is unchanged.
        """
        for index, entry in enumerate(self.entries):
            tree = eqx.tree_at(entry.where, tree, values[index])
        return tree


class FixedInputAdjoint(eqx.Module):
    """Reported sensitivities to the fixed inputs.

    Attributes:
        names: Labels, in vector order.
        units: Units, in vector order.
        values: The fixed inputs ``p`` themselves, carried so a report
            states what the sensitivity was taken at.
        adjoints: ``p̄ = dV/dp``, per unit of each input.
    """

    names: tuple[str, ...] = eqx.field(static=True)
    units: tuple[str, ...] = eqx.field(static=True)
    values: Float[Array, " n_p"]
    adjoints: Float[Array, " n_p"]

    def as_dict(self) -> dict[str, float]:
        """``{name: adjoint}``, for dropping into a risk report."""
        return {
            name: float(value)
            for name, value in zip(self.names, self.adjoints)
        }

    def rows(self) -> tuple[tuple[str, float, float, str], ...]:
        """``(name, value, adjoint, units)`` per input, in order."""
        return tuple(
            (name, float(value), float(adjoint), unit)
            for name, value, adjoint, unit in zip(
                self.names, self.values, self.adjoints, self.units
            )
        )


def fixed_input_adjoints(
    price_fn: Callable[[Any], Float[Array, ""]],
    tree: Any,
    fixed: FixedInputs,
) -> FixedInputAdjoint:
    r"""Differentiate a price with respect to the fixed inputs.

    Reads ``p`` out of ``tree``, re-injects it, and differentiates
    ``price_fn`` through the injection. One reverse sweep, whatever the
    number of inputs.

    **Which question you are asking is decided by ``price_fn``.** This
    matters more than the mechanics:

    * Hand it a plain pricer and you get the **direct** sensitivity,
      :math:`\partial V/\partial p` with the calibrated objects frozen.
      That is the right number when ``p`` genuinely does not feed a
      calibration — a futures convexity adjustment applied downstream,
      say.
    * Hand it a function that **recalibrates and then prices** and you get
      the total derivative, including the model's re-fit response. That is
      the right number for mean reversion or SABR beta, which change what
      the calibration lands on. No implicit-function-theorem step is
      needed either way: ``p`` is held fixed *through* the solve, so
      differentiating the recalibration directly is correct — and it is
      cheap, because the calibration is differentiated through its own
      implicit solution by ``optimistix``, not by taping its iterations.

    Args:
        price_fn: Maps the pytree to a scalar price. Must be the same code
            path used to price for real; a re-implementation here would
            decouple the reported sensitivity from the traded number.
        tree: The pytree holding the fixed inputs (model, surface, or a
            container of several).
        fixed: Which leaves to report, and under what names.

    Returns:
        A :class:`FixedInputAdjoint` carrying ``p``, ``p̄`` and the labels.

    Examples:
        >>> fixed = FixedInputs(entries=(
        ...     FixedInput(
        ...         name="hull_white.mean_reversion",
        ...         where=lambda m: m.mean_reversion,
        ...         units="1/year",
        ...     ),
        ... ))
        >>> report = fixed_input_adjoints(price_bermudan, model, fixed)
        >>> report.as_dict()
        {'hull_white.mean_reversion': ...}
    """
    values = fixed.values(tree)

    def of_values(candidate: Float[Array, " n_p"]) -> Float[Array, ""]:
        return price_fn(fixed.inject(tree, candidate))

    adjoints = jax.grad(of_values)(values)

    return FixedInputAdjoint(
        names=fixed.names,
        units=fixed.units,
        values=values,
        adjoints=adjoints,
    )


def fixed_inputs_from_paths(
    paths: Sequence[tuple[str, Callable[[Any], Any]]],
    *,
    units: Sequence[str] | None = None,
) -> FixedInputs:
    """Build a :class:`FixedInputs` from ``(name, lens)`` pairs.

    A small convenience for the common case of listing a handful of
    leaves inline.

    Args:
        paths: ``(name, where)`` pairs, in report order.
        units: Optional units, one per path.

    Returns:
        The assembled :class:`FixedInputs`.

    Raises:
        ValueError: If ``units`` is given with a different length.
    """
    if units is not None and len(units) != len(paths):
        raise ValueError(
            f"units has length {len(units)} but there are {len(paths)} "
            f"fixed inputs; supply one unit per input or none at all."
        )
    unit_values = tuple(units) if units is not None else ("",) * len(paths)
    return FixedInputs(
        entries=tuple(
            FixedInput(name=name, where=where, units=unit)
            for (name, where), unit in zip(paths, unit_values)
        )
    )
