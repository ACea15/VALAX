r"""Calibration chains: quotes → calibrated market (→ calibrated model).

The calibration stack is a sequence of **levels**, each solving
``R(theta; quotes, upstream, fixed) = 0`` (or an explicit map with no solve):

.. code-block:: text

    quotes q, fixed inputs p
        │
        ▼  level 1a   curves (OIS, projection, basis)           ┐
        ▼  level 1b   vol objects (SABR / SVI smiles, cubes)    │ market scope:
        ▼  …          spots, dividends, flat inputs             ┘ shared, once per snapshot
    MarketState ─────────────────────────────────────────────────┐
        ▼  level 2    pricing model fitted to a basket           │ model scope:
    model ───────────────────────────────────────────────────────┤ per basket / underlying
        ▼                                                        │
    price(instrument, model, market)  ◄──────────────────────────┘

Two scopes, one contract. A :class:`CalibrationNode` is the same object whether
it calibrates a piece of the market (``install_fn`` puts the value into a
``MarketState`` slot) or a pricing model (``install_fn is None``; the value is
passed as the ``model`` argument of :func:`valax.core.pricing.price`). The same
model class may appear in either scope — the node, not the class, decides.

Every node returns a :class:`~valax.calibration.result.CalibrationResult`
carrying its level, so the chain can be differentiated two ways (see
:mod:`valax.greeks.quote_risk`):

* the **adjoint path** walks the levels backwards with one transposed solve per
  level, reusing what the calibration already computed;
* the **autodiff path** simply ``jax.grad``\ s through :func:`calibrate_market`,
  relying on the implicit adjoint ``optimistix`` attaches to each solve.

Everything here is traceable: ``calibrate_market`` runs under ``jit``, ``grad``
and ``vmap`` (validation happens eagerly when a node is *built*, not when it
runs).

Adding a level means writing one node factory: a residual, an ``assemble``
map from unknowns to the calibrated object, and an ``install`` map into the
market — see :func:`solve_level` and :mod:`valax.calibration.nodes`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence

import equinox as eqx
import jax.numpy as jnp

from valax.calibration.level import LevelInputs
from valax.calibration.result import CalibrationResult
from valax.calibration.solve import (
    explicit_level,
    solve_batched_level,
    solve_level,
)
from valax.core.market_state import MarketState
from valax.curves.graph import CurveGraph
from valax.market.quotes import QuoteSet

# ── Market installation helpers ──────────────────────────────────────


def empty_market() -> MarketState:
    """A ``MarketState`` with nothing in it: the root of a calibration chain.

    Returns:
        An empty :class:`~valax.core.market_state.MarketState`.
    """
    return MarketState(curves=CurveGraph(curves={}))


def install_curves(market: MarketState, graph: CurveGraph) -> MarketState:
    """Add (or replace) every curve of ``graph`` in ``market``.

    Args:
        market: The market to extend.
        graph: Calibrated curves.

    Returns:
        A new :class:`~valax.core.market_state.MarketState`.
    """
    curves = {**market.curves.curves, **graph.curves}
    return dataclasses.replace(market, curves=CurveGraph(curves=curves))


def install_surface(surface_id: str) -> Callable[[MarketState, Any], MarketState]:
    """Installer filing a calibrated surface under ``surface_id``.

    Args:
        surface_id: Surface key, e.g. ``"SPX.VOL"``.

    Returns:
        ``install(market, surface) -> MarketState``.
    """

    def install(market: MarketState, surface: Any) -> MarketState:
        return dataclasses.replace(
            market, surfaces={**market.surfaces, surface_id: surface}
        )

    return install


def install_market_fields(market: MarketState, fields: dict) -> MarketState:
    """Merge keyed entries into the dict-valued slots of a market.

    Args:
        market: The market to extend.
        fields: ``slot name -> {key: value}``, e.g.
            ``{"spots": {"SPX": s}, "curves": {"USD.OIS": curve}}``. The
            ``"curves"`` slot is merged into the curve graph.

    Returns:
        A new :class:`~valax.core.market_state.MarketState`.
    """
    updates: dict = {}
    for slot, entries in fields.items():
        if not entries:
            continue
        if slot == "curves":
            updates["curves"] = CurveGraph(curves={**market.curves.curves, **entries})
        else:
            updates[slot] = {**getattr(market, slot), **entries}
    if "spots" in updates and market.asset_ids:
        new_ids = tuple(k for k in updates["spots"] if k not in market.asset_ids)
        updates["asset_ids"] = market.asset_ids + new_ids
    return dataclasses.replace(market, **updates)


# ── Nodes and chains ─────────────────────────────────────────────────


class CalibrationNode(eqx.Module):
    """One calibration step: which inputs it reads and where its output goes.

    Build with a factory from :mod:`valax.calibration.nodes` rather than
    directly. A node is pure data plus two functions; it holds no quote
    values, so one node serves every snapshot and every scenario.

    Attributes:
        name: Unique level name.
        quote_keys: Identifiers of the quotes ``q`` this level consumes.
        fixed_keys: Identifiers of the held-fixed inputs ``p`` it consumes.
        calibrate_fn: ``LevelInputs -> CalibrationResult``.
        install_fn: ``(market, value) -> market`` for a market-scope node;
            ``None`` for a model-scope node, whose value is the ``model``
            argument of ``price``.
        needs_upstream: Whether the level reads the upstream market. A curve
            bootstrap does not; a smile fit on equity forwards does.
    """

    name: str = eqx.field(static=True)
    quote_keys: tuple[str, ...] = eqx.field(static=True)
    fixed_keys: tuple[str, ...] = eqx.field(static=True)
    calibrate_fn: Callable[[LevelInputs], CalibrationResult] = eqx.field(static=True)
    install_fn: Callable[[MarketState, Any], MarketState] | None = eqx.field(
        static=True, default=None
    )
    needs_upstream: bool = eqx.field(static=True, default=True)

    @property
    def scope(self) -> str:
        """``"market"`` if the node installs into the market, else ``"model"``."""
        return "model" if self.install_fn is None else "market"

    def level_inputs(
        self,
        quotes: QuoteSet,
        market: MarketState | None,
        fixed: QuoteSet | None = None,
    ) -> LevelInputs:
        """Slice the chain-wide inputs down to this node's.

        Args:
            quotes: All quotes.
            market: The upstream market.
            fixed: All held-fixed inputs.

        Returns:
            The node's :class:`~valax.calibration.level.LevelInputs`.

        Raises:
            KeyError: If a required quote or fixed input is missing.
        """
        fixed = QuoteSet() if fixed is None else fixed
        return LevelInputs(
            quotes=quotes.select(self.quote_keys),
            upstream=market if self.needs_upstream else None,
            fixed=fixed.select(self.fixed_keys),
        )

    def calibrate(
        self,
        quotes: QuoteSet,
        market: MarketState | None = None,
        fixed: QuoteSet | None = None,
    ) -> CalibrationResult:
        """Run this node against the given quotes and upstream market.

        Args:
            quotes: Quotes (may contain more keys than the node uses).
            market: Upstream market (required when ``needs_upstream``).
            fixed: Held-fixed inputs (may contain more keys than used).

        Returns:
            The node's :class:`~valax.calibration.result.CalibrationResult`.
        """
        return self.calibrate_fn(self.level_inputs(quotes, market, fixed))


class CalibrationChain(eqx.Module):
    """An ordered sequence of market-scope calibration nodes.

    Order is dependency order: a node may read everything installed by the
    nodes before it (and by the base market). The empty chain is valid — it
    is the constant-parameter market, where risk to "quotes" is just the
    market gradient.

    Attributes:
        nodes: The nodes, in calibration order. Static.
    """

    nodes: tuple[CalibrationNode, ...] = eqx.field(static=True, default=())

    def __check_init__(self) -> None:
        """Reject duplicate names and model-scope nodes."""
        names = [n.name for n in self.nodes]
        if len(set(names)) != len(names):
            raise ValueError(f"CalibrationChain: duplicate node names {names}.")
        for n in self.nodes:
            if n.install_fn is None:
                raise ValueError(
                    f"CalibrationChain: node {n.name!r} is model-scope "
                    f"(no install_fn). Calibrate models separately with "
                    f"calibrate_model()."
                )

    @property
    def quote_keys(self) -> tuple[str, ...]:
        """Every quote identifier the chain consumes, first-use order."""
        return tuple(dict.fromkeys(k for n in self.nodes for k in n.quote_keys))

    @property
    def fixed_keys(self) -> tuple[str, ...]:
        """Every fixed-input identifier the chain consumes."""
        return tuple(dict.fromkeys(k for n in self.nodes for k in n.fixed_keys))


class CalibratedMarket(eqx.Module):
    """A calibrated market and the levels that produced it.

    Attributes:
        market: The calibrated :class:`~valax.core.market_state.MarketState`.
        results: One :class:`~valax.calibration.result.CalibrationResult` per
            chain node, in chain order.
        quotes: The quotes the chain was calibrated from.
        fixed: The held-fixed inputs.
        chain: The chain. Static.
    """

    market: MarketState
    results: tuple[CalibrationResult, ...]
    quotes: QuoteSet
    fixed: QuoteSet
    chain: CalibrationChain = eqx.field(static=True)

    @property
    def converged(self) -> Any:
        """Traceable ``all(converged)`` across every level."""
        flags = [r.converged for r in self.results]
        return jnp.all(jnp.stack(flags)) if flags else jnp.asarray(True)

    def result(self, name: str) -> CalibrationResult:
        """Look up a level's result by node name.

        Args:
            name: Node name.

        Returns:
            The level's result.

        Raises:
            KeyError: If no node has that name.
        """
        for node, res in zip(self.chain.nodes, self.results):
            if node.name == name:
                return res
        raise KeyError(
            f"No level {name!r} (levels: {[n.name for n in self.chain.nodes]})."
        )

    def check(self, **kwargs: Any) -> "CalibratedMarket":
        """Eagerly check every level; see :meth:`CalibrationResult.check`.

        Args:
            **kwargs: Forwarded to each level's ``check``.

        Returns:
            ``self``.

        Raises:
            CalibrationError: On the first failed level.
        """
        for res in self.results:
            res.check(**kwargs)
        return self


def calibrate_market(
    chain: CalibrationChain,
    quotes: QuoteSet,
    fixed: QuoteSet | None = None,
    *,
    base: MarketState | None = None,
    strict: bool = False,
) -> CalibratedMarket:
    """Run a market calibration chain.

    Traceable (with ``strict=False``): wrap in ``jax.jit``, differentiate with
    ``jax.grad`` w.r.t. ``quotes`` / ``fixed`` (the autodiff path), or
    ``jax.vmap`` over a batch of quote scenarios.

    Args:
        chain: The market-scope nodes, in dependency order.
        quotes: Market quotes; must contain every ``chain.quote_keys``.
        fixed: Held-fixed inputs; must contain every ``chain.fixed_keys``.
        base: Market to start from (e.g. one carrying fixings or
            externally-built objects). Defaults to an empty market. Its
            contents are inputs, not calibrated levels: risk to them is not
            pushed back to any quote.
        strict: If ``True``, eagerly :meth:`~CalibratedMarket.check` every
            level after calibrating (not traceable).

    Returns:
        The :class:`CalibratedMarket`.

    Raises:
        KeyError: If a node's quote or fixed input is missing.
        CalibrationError: With ``strict=True``, if a level failed.
    """
    fixed = QuoteSet() if fixed is None else fixed
    market = empty_market() if base is None else base
    results = []
    for node in chain.nodes:
        res = node.calibrate(quotes, market, fixed)
        market = node.install_fn(market, res.value)
        results.append(res)
    out = CalibratedMarket(
        market=market,
        results=tuple(results),
        quotes=quotes,
        fixed=fixed,
        chain=chain,
    )
    return out.check() if strict else out


def calibrate_model(
    node: CalibrationNode,
    quotes: QuoteSet,
    market: MarketState,
    fixed: QuoteSet | None = None,
    *,
    strict: bool = False,
) -> CalibrationResult:
    """Calibrate a pricing model against an already-calibrated market.

    The model is a separate input from the market: markets are shared by
    every trade, while a model is fitted per basket (per trade for a rates
    exotic, per underlying for an equity book). Batch per-trade calibrations
    with ``eqx.filter_vmap`` over same-shaped baskets.

    Args:
        node: A model-scope node (``install_fn is None``), or a market-scope
            node whose value you want without installing it.
        quotes: Quotes; must contain ``node.quote_keys``.
        market: The calibrated market the basket is priced against.
        fixed: Held-fixed inputs; must contain ``node.fixed_keys``.
        strict: If ``True``, eagerly check the result (not traceable).

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the model
        is ``result.value``.

    Raises:
        CalibrationError: With ``strict=True``, if the fit failed.
    """
    res = node.calibrate(quotes, market, fixed)
    return res.check() if strict else res


__all__ = [
    "CalibratedMarket",
    "CalibrationChain",
    "CalibrationNode",
    "calibrate_market",
    "calibrate_model",
    "empty_market",
    "explicit_level",
    "install_curves",
    "install_market_fields",
    "install_surface",
    "solve_batched_level",
    "solve_level",
]
