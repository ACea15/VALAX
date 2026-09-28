"""Keyed shocks on a :class:`MarketState`.

The counterpart of :func:`valax.risk.shocks.apply_scenario` for the
container the unified pricer takes.  Where ``apply_scenario`` maps four
flat positional vectors onto a :class:`~valax.market.data.MarketData`,
:func:`shock` maps an identifier-keyed
:class:`~valax.market.keyed_scenario.KeyedScenario` onto a
:class:`~valax.core.market_state.MarketState`:

* **curves** move by id through :func:`valax.risk.shocks.bump_graph_curve`,
  so an OIS shock and a projection-curve (basis) shock are two different,
  separately expressible scenarios on the same graph;
* **surfaces** move by id through :func:`shift_surface`, a parallel vol
  shift that is uniform across every surface type (see
  :mod:`valax.surfaces.shifted` for why that fork was taken);
* **spots / dividends** move by asset id;
* everything is addressed by **identifier**, never by position, and the
  ``"_default_"`` sentinel keeps a single-curve, single-asset market
  zero-configuration.

The result is a **new** pytree built with ``eqx.tree_at`` — nothing is
mutated — and the whole path is differentiable in both the base market
and the shock magnitudes, so ``jax.grad`` of a repricing through
:func:`shock` gives sensitivities to the scenario itself.

An id that resolves to nothing raises: a shock that silently applies to
no market object is the failure mode this module exists to remove.
"""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.core.market_state import DEFAULT_ID, MarketState
from valax.market.keyed_scenario import KeyedScenario
from valax.market.scenario import MarketScenario
from valax.risk.shocks import bump_graph_curve
from valax.surfaces.constant import ConstantVol
from valax.surfaces.shifted import ShiftedVolSurface


# ── Vol-surface shifts ───────────────────────────────────────────────


def shift_surface(surface: Any, shift: Float[Array, ""]) -> Any:
    """Apply an additive parallel vol shift to any duck-typed surface.

    Uniform across surface types, with two structure-preserving special
    cases so the shift never costs expressiveness elsewhere:

    * a :class:`~valax.surfaces.constant.ConstantVol` is shifted **in
      place** (``vol + shift``).  This is bit-for-bit the legacy
      ``MarketData.vols + scenario.vol_shocks`` arithmetic, and it keeps
      the shocked market projectable back through
      :meth:`valax.core.market_state.MarketState.to_market_data`, which
      accepts only ``ConstantVol``;
    * an existing :class:`~valax.surfaces.shifted.ShiftedVolSurface`
      **accumulates** into its ``shift`` instead of nesting, so applying
      two scenarios in sequence stays one wrapper deep;
    * anything else (grid, SVI, SABR, cube, leverage, ...) is wrapped in
      a :class:`~valax.surfaces.shifted.ShiftedVolSurface`.

    Args:
        surface: The surface to shift.
        shift: Additive parallel shift in the surface's quoting
            convention.

    Returns:
        A new surface object; the input is never mutated.
    """
    shift = jnp.asarray(shift)
    if isinstance(surface, ConstantVol):
        return eqx.tree_at(
            lambda s: s.vol, surface, jnp.asarray(surface.vol) + shift,
        )
    if isinstance(surface, ShiftedVolSurface):
        return eqx.tree_at(
            lambda s: s.shift, surface, jnp.asarray(surface.shift) + shift,
        )
    return ShiftedVolSurface(base=surface, shift=shift)


# ── Identifier resolution ────────────────────────────────────────────


def _resolve_id(table: dict, key: str, what: str) -> str:
    """Resolve a scenario key against a keyed market table.

    Args:
        table: The market's mapping (spots, dividends, curves, ...).
        key: Identifier named by the scenario.
        what: Human-readable name of the table, used in the error.

    Returns:
        The key to shock.

    Raises:
        KeyError: If the identifier addresses nothing.  A shock is never
            silently dropped.
    """
    if key in table:
        return key
    if key == DEFAULT_ID and len(table) == 1:
        return next(iter(table))
    raise KeyError(
        f"KeyedScenario shocks {what} {key!r}, which is not in the "
        f"MarketState (available: {sorted(table)})."
    )


def _resolve_surface_id(surfaces: dict, key: str) -> str:
    """Resolve a vol-shock key, accepting the ``"<ASSET>.VOL"`` convention.

    Args:
        surfaces: The market's surface table.
        key: Surface id, or an asset id whose surface is filed under
            ``"<ASSET>.VOL"`` (the key
            :meth:`valax.core.market_state.MarketState.from_market_data`
            emits).

    Returns:
        The surface key to shift.

    Raises:
        KeyError: If neither form addresses a surface.
    """
    if key not in surfaces and f"{key}.VOL" in surfaces:
        return f"{key}.VOL"
    return _resolve_id(surfaces, key, "surface")


# ── Scenario application ─────────────────────────────────────────────


def shock(market: MarketState, scenario: KeyedScenario) -> MarketState:
    """Apply an identifier-keyed scenario to a :class:`MarketState`.

    Returns a new market state; the input is untouched.  Shocks are
    additive deltas (spot shocks are returns when
    ``scenario.multiplicative`` is set), and an identifier absent from a
    scenario mapping is simply not shocked.

    Curve moves go through :func:`valax.risk.shocks.bump_graph_curve`, so
    the per-pillar zero-rate → discount-factor algebra is shared with the
    legacy path and only the *named* curve moves.  Vol moves go through
    :func:`shift_surface`.  For a single-curve, single-asset market every
    id can be left as the ``"_default_"`` sentinel — or omitted, since
    the sentinel resolves the sole entry of a one-entry table.

    Example:
        >>> from valax.core import MarketState
        >>> from valax.market.keyed_scenario import KeyedScenario
        >>> state = MarketState.single_curve(curve, spot=jnp.array(100.0),
        ...                                  vol=jnp.array(0.2))
        >>> shocked = shock(state, KeyedScenario(
        ...     spot_shocks={"_default_": jnp.array(5.0)},
        ...     vol_shocks={"_default_": jnp.array(0.01)},
        ...     curve_shocks={"_default_": jnp.full(3, 0.0010)},
        ... ))

    Args:
        market: Base calibrated market.
        scenario: Keyed shocks to apply.

    Returns:
        A new shocked :class:`~valax.core.market_state.MarketState`.

    Raises:
        TypeError: If handed a legacy
            :class:`~valax.market.scenario.MarketScenario` (lift it with
            :func:`valax.market.keyed_scenario.keyed_from_market_scenario`).
        KeyError: If any shock identifier addresses no curve, surface,
            spot or dividend on the market.
    """
    if isinstance(scenario, MarketScenario):
        raise TypeError(
            "shock() takes a KeyedScenario, not a legacy MarketScenario: "
            "the flat vectors carry no curve/surface/asset identity. Lift "
            "it with valax.market.keyed_scenario.keyed_from_market_scenario"
            "(scenario, asset_ids=..., curve_id=...), or stay on "
            "valax.risk.apply_scenario with a MarketData."
        )

    shocked = market

    if scenario.spot_shocks:
        spots = dict(market.spots)
        for asset_id, delta in scenario.spot_shocks.items():
            key = _resolve_id(spots, asset_id, "spot for asset")
            level = jnp.asarray(spots[key])
            spots[key] = (
                level * (1.0 + delta) if scenario.multiplicative
                else level + delta
            )
        shocked = eqx.tree_at(lambda m: m.spots, shocked, spots)

    if scenario.dividend_shocks:
        dividends = dict(market.dividends)
        for asset_id, delta in scenario.dividend_shocks.items():
            key = _resolve_id(dividends, asset_id, "dividend for asset")
            dividends[key] = jnp.asarray(dividends[key]) + delta
        shocked = eqx.tree_at(lambda m: m.dividends, shocked, dividends)

    if scenario.vol_shocks:
        surfaces = dict(market.surfaces)
        for surface_id, delta in scenario.vol_shocks.items():
            key = _resolve_surface_id(surfaces, surface_id)
            surfaces[key] = shift_surface(surfaces[key], delta)
        # Node-level replacement: wrapping a surface changes the subtree
        # structure, which ``tree_at`` handles for a whole node.
        shocked = eqx.tree_at(lambda m: m.surfaces, shocked, surfaces)

    if scenario.curve_shocks:
        graph = market.curves
        for curve_id, bumps in scenario.curve_shocks.items():
            key = _resolve_id(graph.curves, curve_id, "curve")
            graph = bump_graph_curve(graph, key, bumps)
        shocked = eqx.tree_at(lambda m: m.curves, shocked, graph)

    return shocked
