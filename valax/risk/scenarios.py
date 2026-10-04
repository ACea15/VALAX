"""Scenario generation for VaR and stress testing.

Provides three flavours, in two coordinate systems:

1. **Parametric** — correlated samples from multivariate normal or t.
2. **Historical** — observed risk factor changes sliced into scenarios.
3. **Stress** — deterministic named shocks (parallel, steepener, etc.).

Two layouts, one set of flavours
--------------------------------

The **legacy** generators (:func:`parametric_scenarios`,
:func:`historical_scenarios`, :func:`stress_scenario`,
:func:`steepener` / :func:`flattener` / :func:`butterfly`) emit
:class:`~valax.market.scenario.MarketScenario` /
:class:`~valax.market.scenario.ScenarioSet` over the **flat positional**
factor layout ``[spots, vols, rates, dividends]``.  That layout has one
undivided ``rates`` block, so it cannot say *which* curve of a
multi-curve book moved — a basis move is inexpressible, and a projection
curve cannot carry its own VaR.

The **keyed** generators (:func:`keyed_parametric_scenarios`,
:func:`keyed_historical_scenarios`, :func:`keyed_stress_scenario`,
:func:`keyed_steepener` / :func:`keyed_flattener` /
:func:`keyed_butterfly`, :func:`keyed_basis_shift`) emit
:class:`~valax.market.keyed_scenario.KeyedScenario` over a
:class:`KeyedFactorLayout`: an ordered tuple of :class:`FactorBlock`\\ s,
**one block per named market object**, so the covariance matrix has a
separate block for every curve on the
:class:`~valax.curves.graph.CurveGraph`.  They apply to a
:class:`~valax.core.market_state.MarketState` through
:func:`valax.risk.state_shocks.shock` and price through
:func:`valax.risk.state_var.group_scenario_pnl`.

The layout is the bridge between the three representations a risk number
needs to agree on:

* a **scenario** (which shocks are applied),
* a **covariance matrix** (rows/columns of :math:`\\Sigma`), and
* a **ladder** (:mod:`valax.risk.state_ladders`, whose sensitivities are
  indexed by the same blocks),

and :meth:`KeyedFactorLayout.factor_ids` names every coordinate with the
hierarchical identifier of :doc:`/risk-factors` § 1.2, so the three can
be lined up by *name* rather than by position.

The basis as a coordinate
-------------------------
:func:`as_basis_layout` reparameterises a projection curve's block from
an absolute zero-rate move into a **basis** over a discount curve
(:math:`r^{\\text{proj}} = r^{\\text{disc}} + b`).  That is what turns
``IR.BASIS.<ccy>.<tenorA>.<tenorB>.<pillar>`` from a description into an
addressable factor: in a basis layout the discount curve and the basis
move independently, which is the way the market actually trades them, and
a basis variance in :math:`\\Sigma` produces basis P&L and basis VaR.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.core.market_state import DEFAULT_ID, MarketState
from valax.dates.daycounts import year_fraction
from valax.market.keyed_scenario import KeyedScenario
from valax.market.scenario import MarketScenario, ScenarioSet, stack_scenarios


# ── Parametric (Monte Carlo) scenarios ───────────────────────────────


def parametric_scenarios(
    key: jax.Array,
    cov: Float[Array, "n_factors n_factors"],
    n_scenarios: int,
    n_assets: int,
    n_pillars: int,
    distribution: str = "normal",
    df: float = 5.0,
) -> ScenarioSet:
    """Sample correlated risk factor scenarios from a parametric distribution.

    Column ordering in the covariance matrix must be::

        [spot_0 .. spot_{n-1}, vol_0 .. vol_{n-1},
         rate_0 .. rate_{p-1}, div_0 .. div_{n-1}]

    where ``n = n_assets`` and ``p = n_pillars``.

    Args:
        key: JAX PRNG key.
        cov: Covariance matrix of all risk factors.
        n_scenarios: Number of scenarios to generate.
        n_assets: Number of spot / vol / dividend factors.
        n_pillars: Number of rate curve pillars.
        distribution: ``"normal"`` or ``"t"``.
        df: Degrees of freedom for the t-distribution.

    Returns:
        A ``ScenarioSet`` with ``n_scenarios`` rows.
    """
    n_factors = cov.shape[0]
    L = jnp.linalg.cholesky(cov)

    key1, key2 = jax.random.split(key)
    z = jax.random.normal(key1, shape=(n_scenarios, n_factors))

    if distribution == "t":
        # t = normal / sqrt(chi2 / df)
        # chi2(df) = 2 * Gamma(df/2, 1)
        chi2 = 2.0 * jax.random.gamma(key2, df / 2.0, shape=(n_scenarios, 1))
        z = z * jnp.sqrt(df / chi2)

    samples = z @ L.T  # (n_scenarios, n_factors)

    return _split_factors(samples, n_assets, n_pillars)


# ── Historical simulation scenarios ──────────────────────────────────


def historical_scenarios(
    returns: Float[Array, "n_obs n_factors"],
    n_assets: int,
    n_pillars: int,
) -> ScenarioSet:
    """Build scenarios from historical risk factor changes.

    Args:
        returns: Observed daily changes with columns ordered as
            ``[spots, vols, rates, dividends]``.
        n_assets: Number of spot / vol / dividend factors.
        n_pillars: Number of rate curve pillars.

    Returns:
        A ``ScenarioSet`` with ``n_scenarios = n_obs``.
    """
    return _split_factors(returns, n_assets, n_pillars)


# ── Stress / deterministic scenarios ─────────────────────────────────


def stress_scenario(
    n_assets: int,
    n_pillars: int,
    spot_shock: float = 0.0,
    vol_shock: float = 0.0,
    parallel_rate_shift: float = 0.0,
    rate_shocks: Float[Array, " n_pillars"] | None = None,
    dividend_shock: float = 0.0,
) -> MarketScenario:
    """Construct a single deterministic stress scenario.

    If ``rate_shocks`` is provided it is used directly; otherwise
    a uniform ``parallel_rate_shift`` is applied to all pillars.
    """
    if rate_shocks is None:
        rate_shocks = jnp.full(n_pillars, parallel_rate_shift)
    return MarketScenario(
        spot_shocks=jnp.full(n_assets, spot_shock),
        vol_shocks=jnp.full(n_assets, vol_shock),
        rate_shocks=rate_shocks,
        dividend_shocks=jnp.full(n_assets, dividend_shock),
    )


def steepener(
    n_assets: int,
    n_pillars: int,
    short_bump: float,
    long_bump: float,
) -> MarketScenario:
    """Steepener: linearly interpolated rate shock from short to long end."""
    rate_shocks = curve_shape_bumps(n_pillars, "steepener", short_bump, long_bump)
    return stress_scenario(n_assets, n_pillars, rate_shocks=rate_shocks)


def flattener(
    n_assets: int,
    n_pillars: int,
    short_bump: float,
    long_bump: float,
) -> MarketScenario:
    """Flattener: linearly interpolated rate shock (typically short up, long down)."""
    rate_shocks = curve_shape_bumps(n_pillars, "flattener", short_bump, long_bump)
    return stress_scenario(n_assets, n_pillars, rate_shocks=rate_shocks)


def butterfly(
    n_assets: int,
    n_pillars: int,
    wing_bump: float,
    belly_bump: float,
) -> MarketScenario:
    """Butterfly: quadratic profile — wings at ``wing_bump``, belly at ``belly_bump``.

    The bump at each pillar follows a parabola::

        bump(x) = wing + (belly - wing) * 4 * x * (1 - x)

    where ``x`` runs from 0 to 1 across the pillars.
    """
    rate_shocks = curve_shape_bumps(n_pillars, "butterfly", wing_bump, belly_bump)
    return stress_scenario(n_assets, n_pillars, rate_shocks=rate_shocks)


# ── Curve-shape presets (shared by both layouts) ─────────────────────

#: Curve-shape names accepted by :func:`curve_shape_bumps`.
CURVE_SHAPES = ("parallel", "steepener", "flattener", "butterfly")


def curve_shape_bumps(
    n_pillars: int,
    shape: str = "parallel",
    low: Float[Array, ""] | float = 0.0,
    high: Float[Array, ""] | float = 0.0,
) -> Float[Array, " n_pillars"]:
    """Per-pillar bumps for a named curve-shape stress.

    The one home for the shape algebra, shared by the legacy
    :func:`steepener` / :func:`flattener` / :func:`butterfly` and by their
    keyed counterparts, so the two layouts cannot drift apart.

    Args:
        n_pillars: Number of pillars on the curve being shocked.
        shape:

            - ``"parallel"``: every pillar moves by ``low``.
            - ``"steepener"`` / ``"flattener"``: linear from ``low`` at the
              short end to ``high`` at the long end.  The two names are
              the same algebra — a flattener is a steepener with the sign
              of ``high - low`` reversed — and both are kept because the
              *intent* is what a stress report names.
            - ``"butterfly"``: parabola with wings at ``low`` and belly at
              ``high``, ``bump(x) = low + (high - low) * 4x(1 - x)``.
        low: Short-end bump (wings, for a butterfly), or the parallel
            bump.
        high: Long-end bump (belly, for a butterfly).

    Returns:
        Additive per-pillar zero-rate bumps.

    Raises:
        ValueError: If ``shape`` is not one of :data:`CURVE_SHAPES`.
    """
    if shape == "parallel":
        return jnp.full((n_pillars,), low)
    if shape in ("steepener", "flattener"):
        return jnp.linspace(low, high, n_pillars)
    if shape == "butterfly":
        x = jnp.linspace(0.0, 1.0, n_pillars)
        return low + (high - low) * 4.0 * x * (1.0 - x)
    raise ValueError(
        f"shape must be one of {CURVE_SHAPES}, got {shape!r}."
    )


# ── Internal helpers ─────────────────────────────────────────────────


def _split_factors(
    samples: Float[Array, "n_scenarios n_factors"],
    n_assets: int,
    n_pillars: int,
) -> ScenarioSet:
    """Slice a factor matrix into a ScenarioSet.

    Column ordering: [spots, vols, rates, dividends].
    """
    i = 0
    spot_shocks = samples[:, i : i + n_assets]; i += n_assets
    vol_shocks = samples[:, i : i + n_assets]; i += n_assets
    rate_shocks = samples[:, i : i + n_pillars]; i += n_pillars
    dividend_shocks = samples[:, i : i + n_assets]

    return ScenarioSet(
        spot_shocks=spot_shocks,
        vol_shocks=vol_shocks,
        rate_shocks=rate_shocks,
        dividend_shocks=dividend_shocks,
    )


# ── Keyed factor layout ──────────────────────────────────────────────

#: Factor kinds, in the canonical block order emitted by
#: :meth:`KeyedFactorLayout.from_market_state`.  Spots / vols / dividends
#: first (matching the legacy reading order) and curves in the middle,
#: where the legacy layout had its single ``rates`` block.
FACTOR_KINDS = ("spot", "vol", "curve", "dividend")

#: Category prefix of the hierarchical factor identifier for each kind
#: (see :doc:`/risk-factors` § 1.2).  ``IR.CURVE`` is deliberately generic:
#: whether a given curve is an OIS, a tenor forward or something else is a
#: property of the curve id, which is appended after the prefix.
FACTOR_CATEGORIES = {
    "spot": "EQ.SPOT",
    "vol": "EQ.VOL",
    "curve": "IR.CURVE",
    "dividend": "EQ.DIV",
}

#: Kind → the :class:`~valax.market.keyed_scenario.KeyedScenario` field it
#: addresses.
_SCENARIO_FIELDS = {
    "spot": "spot_shocks",
    "vol": "vol_shocks",
    "curve": "curve_shocks",
    "dividend": "dividend_shocks",
}


def _tenor_label(time: float) -> str:
    """Format a pillar year fraction as a tenor label (``"5Y"``, ``"0.4959Y"``)."""
    return f"{time:.4g}Y"


class FactorBlock(eqx.Module):
    """One contiguous block of the factor vector, owned by one market object.

    Entirely static: a block is *metadata* describing where a market
    object's shocks live in a flat coordinate vector, not data to be
    differentiated.  The shock magnitudes themselves live on a
    :class:`~valax.market.keyed_scenario.KeyedScenario`.

    Attributes:
        kind: One of :data:`FACTOR_KINDS`.
        key: The identifier of the market object — curve id, surface id or
            asset id.
        size: Number of coordinates the block occupies: ``1`` for a scalar
            shock (spot, parallel vol shift, dividend) and ``n_pillars``
            for a curve.
        times: Optional pillar year fractions, one per coordinate.  Used
            for tenor labels in :attr:`ids` and for tenor bucketing; never
            for arithmetic.
        basis_of: When set, this curve block holds a **basis** over the
            curve named here rather than an absolute zero-rate move, so
            the absolute move is ``basis + shock(basis_of)``.  Set by
            :func:`as_basis_layout`; ``None`` for an ordinary block.
        factor_prefix: Optional override for the factor-identifier prefix,
            replacing ``"<CATEGORY>.<key>"``.  Used to give a basis block
            its ``IR.BASIS.<ccy>.<tenorA>.<tenorB>`` name.
    """

    kind: str = eqx.field(static=True)
    key: str = eqx.field(static=True)
    size: int = eqx.field(static=True, default=1)
    times: tuple = eqx.field(static=True, default=())
    basis_of: str | None = eqx.field(static=True, default=None)
    factor_prefix: str | None = eqx.field(static=True, default=None)

    def __check_init__(self) -> None:
        """Validate the block's metadata.

        Raises:
            ValueError: If ``kind`` is unknown, ``size`` is not positive,
                ``times`` has the wrong length, or ``basis_of`` is set on
                a non-curve block.
        """
        if self.kind not in FACTOR_KINDS:
            raise ValueError(
                f"FactorBlock.kind must be one of {FACTOR_KINDS}, "
                f"got {self.kind!r}."
            )
        if self.size < 1:
            raise ValueError(
                f"FactorBlock({self.kind}, {self.key!r}).size must be >= 1, "
                f"got {self.size}."
            )
        if self.times and len(self.times) != self.size:
            raise ValueError(
                f"FactorBlock({self.kind}, {self.key!r}) has {self.size} "
                f"coordinates but {len(self.times)} pillar times."
            )
        if self.basis_of is not None and self.kind != "curve":
            raise ValueError(
                f"FactorBlock.basis_of is only meaningful for a curve "
                f"block; got kind={self.kind!r}."
            )

    @property
    def prefix(self) -> str:
        """Factor-identifier prefix of the block."""
        if self.factor_prefix is not None:
            return self.factor_prefix
        return f"{FACTOR_CATEGORIES[self.kind]}.{self.key}"

    @property
    def ids(self) -> tuple[str, ...]:
        """Hierarchical factor identifier of each coordinate in the block."""
        if self.size == 1 and not self.times:
            return (self.prefix,)
        if self.times:
            return tuple(f"{self.prefix}.{_tenor_label(t)}" for t in self.times)
        return tuple(f"{self.prefix}.{i}" for i in range(self.size))


class KeyedFactorLayout(eqx.Module):
    """Ordered, identifier-keyed factor coordinates for a ``MarketState``.

    The replacement for the flat positional ``[spots, vols, rates,
    dividends]`` layout.  Each :class:`FactorBlock` owns a contiguous slice
    of the factor vector, and **every curve on the graph gets its own
    block** — which is what lets a covariance matrix hold a separate block
    per curve, a scenario move one curve without touching another, and a
    ladder report a projection curve's risk in its own right.

    Attributes:
        blocks: The blocks, in coordinate order.  Static.
        multiplicative: Spot-shock convention carried onto every scenario
            built by :meth:`unflatten` and the generators (``True`` means
            spot shocks are returns).  Static.
    """

    blocks: tuple = eqx.field(static=True)
    multiplicative: bool = eqx.field(static=True, default=False)

    def __check_init__(self) -> None:
        """Validate block uniqueness and basis references.

        Raises:
            ValueError: If two blocks address the same ``(kind, key)``, or
                a basis block references a curve that is absent from the
                layout or has a different pillar count.
        """
        seen = set()
        for block in self.blocks:
            token = (block.kind, block.key)
            if token in seen:
                raise ValueError(
                    f"KeyedFactorLayout has two blocks for "
                    f"{block.kind} {block.key!r}; each market object owns "
                    f"exactly one block."
                )
            seen.add(token)
        for block in self.blocks:
            if block.basis_of is None:
                continue
            base = [
                b for b in self.blocks
                if b.kind == "curve" and b.key == block.basis_of
            ]
            if not base:
                raise ValueError(
                    f"Curve block {block.key!r} is a basis over "
                    f"{block.basis_of!r}, which is not a curve block of "
                    f"this layout (curves: {sorted(self.curve_ids())})."
                )
            if base[0].size != block.size:
                raise ValueError(
                    f"Basis block {block.key!r} has {block.size} pillars "
                    f"but its base curve {block.basis_of!r} has "
                    f"{base[0].size}; a basis is defined pillar by pillar."
                )

    # ── Construction ─────────────────────────────────────────────────

    @classmethod
    def from_market_state(
        cls,
        market: MarketState,
        *,
        curve_ids: tuple | list | None = None,
        surface_ids: tuple | list | None = None,
        asset_ids: tuple | list | None = None,
        include_dividends: bool = True,
        multiplicative: bool = False,
    ) -> "KeyedFactorLayout":
        """Derive a layout from the market itself.

        Reads the identifiers and pillar counts off ``market`` so the
        factor coordinates cannot disagree with the container they shock —
        the failure mode of a hand-written positional layout.  Every curve
        on the graph contributes its own block, sized by that curve's
        pillar count and labelled with its pillar year fractions.

        Args:
            market: The calibrated market.
            curve_ids: Curves to include; defaults to every curve on the
                graph, sorted.
            surface_ids: Vol surfaces to include (one parallel-shift
                coordinate each); defaults to every surface, sorted.
            asset_ids: Assets to include; defaults to
                ``market.asset_ids`` or the sorted spot keys.
            include_dividends: Whether to give each asset a dividend
                coordinate.  Only assets that actually carry a dividend
                entry get one.
            multiplicative: Spot-shock convention for scenarios built from
                this layout.

        Returns:
            A :class:`KeyedFactorLayout`.
        """
        graph = market.curves
        if curve_ids is None:
            curve_ids = sorted(graph.keys())
        if surface_ids is None:
            surface_ids = sorted(market.surfaces)
        if asset_ids is None:
            asset_ids = market.asset_ids or tuple(sorted(market.spots))

        blocks: list[FactorBlock] = []
        for asset_id in asset_ids:
            blocks.append(FactorBlock(kind="spot", key=asset_id, size=1))
        for surface_id in surface_ids:
            blocks.append(FactorBlock(kind="vol", key=surface_id, size=1))
        for curve_id in curve_ids:
            curve = graph[curve_id]
            times = year_fraction(
                curve.reference_date, curve.pillar_dates, curve.day_count,
            )
            blocks.append(FactorBlock(
                kind="curve",
                key=curve_id,
                size=int(curve.pillar_dates.shape[0]),
                times=tuple(float(t) for t in times),
            ))
        if include_dividends:
            for asset_id in asset_ids:
                if asset_id in market.dividends:
                    blocks.append(
                        FactorBlock(kind="dividend", key=asset_id, size=1)
                    )

        return cls(blocks=tuple(blocks), multiplicative=multiplicative)

    # ── Coordinates ──────────────────────────────────────────────────

    @property
    def n_factors(self) -> int:
        """Total number of factor coordinates."""
        return sum(block.size for block in self.blocks)

    def factor_ids(self) -> tuple[str, ...]:
        """Hierarchical identifier of every coordinate, in order.

        The names that line a scenario up with a covariance matrix and a
        ladder.  See :doc:`/risk-factors` § 1.2 for the convention.

        Returns:
            One identifier per coordinate, length :attr:`n_factors`.
        """
        ids: list[str] = []
        for block in self.blocks:
            ids.extend(block.ids)
        return tuple(ids)

    def has(self, kind: str, key: str) -> bool:
        """Whether the layout carries a block for ``(kind, key)``."""
        return any(b.kind == kind and b.key == key for b in self.blocks)

    def block(self, kind: str, key: str) -> FactorBlock:
        """Look up a block by kind and identifier.

        Args:
            kind: One of :data:`FACTOR_KINDS`.
            key: Market-object identifier.

        Returns:
            The :class:`FactorBlock`.

        Raises:
            KeyError: If the layout has no such block.
        """
        for block in self.blocks:
            if block.kind == kind and block.key == key:
                return block
        raise KeyError(
            f"KeyedFactorLayout has no {kind} block {key!r} "
            f"(available: {sorted(b.key for b in self.blocks if b.kind == kind)})."
        )

    def index(self, kind: str, key: str) -> slice:
        """Slice of the factor vector owned by ``(kind, key)``.

        Args:
            kind: One of :data:`FACTOR_KINDS`.
            key: Market-object identifier.

        Returns:
            A ``slice`` into the last axis of a factor vector / matrix.

        Raises:
            KeyError: If the layout has no such block.
        """
        start = 0
        for block in self.blocks:
            if block.kind == kind and block.key == key:
                return slice(start, start + block.size)
            start += block.size
        raise KeyError(
            f"KeyedFactorLayout has no {kind} block {key!r}."
        )

    def curve_ids(self) -> tuple[str, ...]:
        """Identifiers of the curve blocks, in coordinate order."""
        return tuple(b.key for b in self.blocks if b.kind == "curve")

    def keys_of(self, kind: str) -> tuple[str, ...]:
        """Identifiers of every block of one kind, in coordinate order."""
        return tuple(b.key for b in self.blocks if b.kind == kind)

    # ── Scenario ↔ vector ────────────────────────────────────────────

    def flatten(
        self,
        scenario: KeyedScenario,
    ) -> Float[Array, "... n_factors"]:
        """Project a keyed scenario onto the layout's factor vector.

        The inverse of :meth:`unflatten`.  Identifiers the scenario leaves
        untouched contribute zeros; a basis block contributes the
        *difference* between its curve's move and its base curve's, which
        is the coordinate change that makes a basis an independent factor.

        Args:
            scenario: The scenario to project.  May be batched (every leaf
                carrying a leading scenario axis), in which case the
                result carries that axis too.

        Returns:
            Factor vector of shape ``(..., n_factors)``.

        Raises:
            ValueError: If the scenario shocks an identifier the layout
                has no block for — a coordinate change that silently
                dropped a shock would produce a plausible wrong number.
        """
        for kind, field in _SCENARIO_FIELDS.items():
            for key in getattr(scenario, field):
                if not self.has(kind, key):
                    raise ValueError(
                        f"KeyedScenario shocks {kind} {key!r}, which has "
                        f"no block in this layout (blocks: "
                        f"{[(b.kind, b.key) for b in self.blocks]}). "
                        f"Flattening it would silently drop the shock."
                    )

        parts = []
        for block in self.blocks:
            table = getattr(scenario, _SCENARIO_FIELDS[block.kind])
            value = table.get(block.key)
            value = jnp.zeros(()) if value is None else jnp.asarray(value)
            if block.basis_of is not None:
                base = table.get(block.basis_of)
                if base is not None:
                    value = value - jnp.asarray(base)
            # A scalar block's magnitude carries no coordinate axis of its
            # own; give it one so every part ends in (..., size).
            if block.size == 1:
                value = value[..., None]
            elif value.ndim == 0:
                value = jnp.broadcast_to(value, (block.size,))
            parts.append(value)

        # Unshocked blocks contributed unbatched zeros; lift them to the
        # batch shape of the shocked ones so a batched scenario flattens.
        batch = jnp.broadcast_shapes(*[p.shape[:-1] for p in parts])
        parts = [
            jnp.broadcast_to(p, batch + (p.shape[-1],)) for p in parts
        ]
        return jnp.concatenate(parts, axis=-1)

    def unflatten(
        self,
        factors: Float[Array, "... n_factors"],
    ) -> KeyedScenario:
        """Build a keyed scenario from the layout's factor vector.

        The inverse of :meth:`flatten`, and the step every keyed generator
        ends with.  Works batched: pass ``(n_scenarios, n_factors)`` and
        every leaf gains the leading scenario axis, giving a pytree ready
        for ``jax.vmap`` (the keyed analogue of
        :class:`~valax.market.scenario.ScenarioSet`).

        A basis block is *re-added* to its base curve here, so the curve
        shock handed to :func:`valax.risk.state_shocks.shock` is always an
        absolute zero-rate move: ``r_proj = r_disc + b``.

        Args:
            factors: Factor vector of shape ``(..., n_factors)``.

        Returns:
            A :class:`~valax.market.keyed_scenario.KeyedScenario`.

        Raises:
            ValueError: If the trailing axis is not :attr:`n_factors` long.
        """
        factors = jnp.asarray(factors)
        if factors.shape[-1] != self.n_factors:
            raise ValueError(
                f"unflatten expected a trailing axis of {self.n_factors} "
                f"factors, got {factors.shape[-1]}."
            )

        tables: dict[str, dict] = {kind: {} for kind in FACTOR_KINDS}
        raw_curves: dict[str, Any] = {}
        for block in self.blocks:
            sl = self.index(block.kind, block.key)
            value = (
                factors[..., sl.start] if block.size == 1
                else factors[..., sl]
            )
            if block.kind == "curve":
                raw_curves[block.key] = value
            else:
                tables[block.kind][block.key] = value

        # Basis blocks read their base curve's *raw* move, so the order of
        # the blocks in the layout cannot change the answer.
        for block in self.blocks:
            if block.kind != "curve":
                continue
            value = raw_curves[block.key]
            if block.basis_of is not None:
                value = value + raw_curves[block.basis_of]
            tables["curve"][block.key] = value

        return KeyedScenario(
            spot_shocks=tables["spot"],
            vol_shocks=tables["vol"],
            curve_shocks=tables["curve"],
            dividend_shocks=tables["dividend"],
            multiplicative=self.multiplicative,
        )

    def zero(self) -> KeyedScenario:
        """The no-op scenario with every coordinate present and zero.

        Unlike :func:`valax.market.keyed_scenario.zero_keyed_scenario`,
        which is empty, this carries an explicit zero for every block —
        the base point a ladder differentiates at
        (:func:`valax.risk.state_ladders.keyed_ladder`) and the structural
        template :func:`valax.market.keyed_scenario.stack_keyed_scenarios`
        requires.

        Returns:
            A :class:`~valax.market.keyed_scenario.KeyedScenario` of zeros.
        """
        return self.unflatten(jnp.zeros((self.n_factors,)))


# ── The basis as a coordinate ────────────────────────────────────────


def basis_factor_prefix(
    projection_id: str,
    discount_id: str,
) -> str:
    """Factor-identifier prefix for the basis between two curves.

    Builds ``IR.BASIS.<ccy>.<tenorA>.<tenorB>`` from the curve alphabet
    ``<CCY>.<INDEX>.<TENOR>`` frozen by MC-Curves-2, e.g.
    ``("USD.SOFR.3M", "USD.OIS")`` → ``"IR.BASIS.USD.SOFR3M.OIS"`` —
    matching the registry entry in :doc:`/risk-factors` § 3.2.

    Currency is taken from the projection curve's leading field when both
    curves agree on it (a single-currency tenor basis); otherwise the two
    full ids are used, since a cross-currency spread is not one currency's
    factor.

    Args:
        projection_id: The curve carrying the basis (tenor A).
        discount_id: The curve it is quoted against (tenor B).

    Returns:
        The identifier prefix; append ``.<pillar>`` for a coordinate.
    """
    proj = projection_id.split(".")
    disc = discount_id.split(".")
    if len(proj) > 1 and len(disc) > 1 and proj[0] == disc[0]:
        return (
            f"IR.BASIS.{proj[0]}.{''.join(proj[1:])}.{''.join(disc[1:])}"
        )
    return f"IR.BASIS.{projection_id}.{discount_id}"


def as_basis_layout(
    layout: KeyedFactorLayout,
    projection_id: str,
    over: str = DEFAULT_ID,
    *,
    factor_prefix: str | None = None,
) -> KeyedFactorLayout:
    r"""Reparameterise a projection curve's block as a **basis**.

    Turns the absolute-zero-rate coordinates of ``projection_id`` into
    spread coordinates over ``over``:

    .. math::

        r^{\text{proj}}_i = r^{\text{disc}}_i + b_i,

    leaving the discount curve's own block untouched.  The pair
    (discount, basis) spans exactly the same space as (discount,
    projection) — :meth:`KeyedFactorLayout.flatten` and
    :meth:`~KeyedFactorLayout.unflatten` are still exact inverses — but it
    is the *market's* basis, so:

    * a covariance block on the basis coordinates is a **basis variance**,
      and a parametric scenario set drawn from it moves discounting and
      basis independently (in absolute coordinates, a pure discount move
      drags the projection curve with it, which is the empirically right
      behaviour and is impossible to express in the absolute layout
      without a near-unit correlation);
    * the corresponding ladder entries are a **basis DV01**, named
      ``IR.BASIS.<ccy>.<tenorA>.<tenorB>.<pillar>``.

    Args:
        layout: Layout to reparameterise.
        projection_id: Curve whose block becomes a basis.
        over: Curve the basis is quoted over — the discount curve.
            Defaults to the ``"_default_"`` sentinel.
        factor_prefix: Override for the generated ``IR.BASIS...`` prefix.

    Returns:
        A new :class:`KeyedFactorLayout`; the input is unchanged.

    Raises:
        KeyError: If either curve is absent from the layout.
        ValueError: If the two curves are the same, they disagree on
            pillar count, or the projection block is already a basis.
    """
    if projection_id == over:
        raise ValueError(
            f"as_basis_layout: a curve cannot be a basis over itself "
            f"({projection_id!r})."
        )
    projection = layout.block("curve", projection_id)
    base = layout.block("curve", over)
    if projection.basis_of is not None:
        raise ValueError(
            f"Curve block {projection_id!r} is already a basis over "
            f"{projection.basis_of!r}."
        )
    if projection.size != base.size:
        raise ValueError(
            f"as_basis_layout: {projection_id!r} has {projection.size} "
            f"pillars but {over!r} has {base.size}; a basis is defined "
            f"pillar by pillar, so the two curves must share a grid."
        )

    prefix = (
        factor_prefix if factor_prefix is not None
        else basis_factor_prefix(projection_id, over)
    )
    blocks = tuple(
        FactorBlock(
            kind=b.kind, key=b.key, size=b.size, times=b.times,
            basis_of=over, factor_prefix=prefix,
        )
        if (b.kind == "curve" and b.key == projection_id) else b
        for b in layout.blocks
    )
    return KeyedFactorLayout(
        blocks=blocks, multiplicative=layout.multiplicative,
    )


# ── Per-curve covariance assembly ────────────────────────────────────


def block_covariance(
    layout: KeyedFactorLayout,
    blocks: dict,
    cross_correlations: dict | None = None,
    *,
    jitter: float = 0.0,
) -> Float[Array, "n_factors n_factors"]:
    r"""Assemble a covariance matrix from per-block covariances.

    The replacement for hand-laying a single ``(n_factors, n_factors)``
    matrix over ``[spots, vols, rates, dividends]``: each named market
    object supplies **its own block**, so a two-curve book has a distinct
    OIS block and projection (or basis) block and the cross-correlation
    between them is stated explicitly rather than implied by position.

    Diagonal blocks are taken as given.  An off-diagonal block is built
    from a scalar correlation and the two blocks' standard deviations,

    .. math::

        \Sigma_{AB} = \rho_{AB}\, \sigma_A \sigma_B^{\top},
        \qquad \sigma = \sqrt{\operatorname{diag} \Sigma},

    which is the usual "one number per factor pair" convention.  Unstated
    pairs are uncorrelated.

    The result is symmetric but not necessarily positive **definite**:
    deterministic blocks (absent from ``blocks``) make it singular by
    construction, and arbitrary correlations between three or more blocks
    can make it indefinite.  :func:`keyed_parametric_scenarios` factors it
    spectrally rather than by Cholesky, so a singular matrix samples
    correctly (a zero-variance block simply never moves) and a mildly
    indefinite one is projected onto the nearest PSD matrix instead of
    producing NaNs.  ``jitter`` adds a small multiple of the identity if
    you would rather regularise the input.

    Args:
        layout: The coordinate system.
        blocks: ``(kind, key) -> block covariance``.  A block may be given
            as a full ``(size, size)`` matrix, a length-``size`` vector of
            variances (diagonal block), or a scalar variance (a scalar
            block, or a common variance for every pillar of a curve).
            Blocks of the layout that are absent from this mapping are
            deterministic (zero variance).
        cross_correlations: ``((kind_a, key_a), (kind_b, key_b)) -> rho``.
            Order-insensitive; each pair need only be given once.
        jitter: Added to the diagonal.

    Returns:
        A symmetric ``(n_factors, n_factors)`` covariance matrix.

    Raises:
        KeyError: If a key of ``blocks`` or ``cross_correlations`` is not
            a block of ``layout``.
        ValueError: If a block covariance has an incompatible shape.
    """
    n = layout.n_factors
    cov = jnp.zeros((n, n))

    resolved: dict[tuple, Float[Array, "s s"]] = {}
    for token, value in blocks.items():
        kind, key = token
        block = layout.block(kind, key)   # raises KeyError if unknown
        value = jnp.asarray(value, dtype=float)
        if value.ndim == 0:
            value = jnp.eye(block.size) * value
        elif value.ndim == 1:
            if value.shape[0] != block.size:
                raise ValueError(
                    f"block ({kind}, {key!r}) expects {block.size} "
                    f"variances, got {value.shape[0]}."
                )
            value = jnp.diag(value)
        elif value.shape != (block.size, block.size):
            raise ValueError(
                f"block ({kind}, {key!r}) expects shape "
                f"{(block.size, block.size)}, got {value.shape}."
            )
        resolved[token] = value
        sl = layout.index(kind, key)
        cov = cov.at[sl, sl].set(value)

    for pair, rho in (cross_correlations or {}).items():
        token_a, token_b = pair
        layout.block(*token_a)
        layout.block(*token_b)
        sigma_a = jnp.sqrt(jnp.diag(
            resolved.get(token_a, jnp.zeros((layout.block(*token_a).size,) * 2))
        ))
        sigma_b = jnp.sqrt(jnp.diag(
            resolved.get(token_b, jnp.zeros((layout.block(*token_b).size,) * 2))
        ))
        cross = rho * jnp.outer(sigma_a, sigma_b)
        sl_a = layout.index(*token_a)
        sl_b = layout.index(*token_b)
        cov = cov.at[sl_a, sl_b].set(cross)
        cov = cov.at[sl_b, sl_a].set(cross.T)

    cov = 0.5 * (cov + cov.T)
    if jitter:
        cov = cov + jitter * jnp.eye(n)
    return cov


def historical_block_covariance(
    returns: Float[Array, "n_obs n_factors"],
    layout: KeyedFactorLayout | None = None,
) -> Float[Array, "n_factors n_factors"]:
    """Sample covariance of observed factor changes, in layout coordinates.

    The empirical counterpart of :func:`block_covariance`: the per-curve
    block structure comes from the data rather than from assumption.  The
    columns of ``returns`` must be ordered as
    :meth:`KeyedFactorLayout.factor_ids`.

    Args:
        returns: ``(n_obs, n_factors)`` matrix of factor changes.
        layout: Optional layout, used only to check the column count.

    Returns:
        The ``(n_factors, n_factors)`` sample covariance.

    Raises:
        ValueError: If ``returns`` has a different width from ``layout``.
    """
    returns = jnp.asarray(returns)
    if layout is not None and returns.shape[-1] != layout.n_factors:
        raise ValueError(
            f"historical_block_covariance: returns has "
            f"{returns.shape[-1]} columns but the layout has "
            f"{layout.n_factors} factors."
        )
    centred = returns - jnp.mean(returns, axis=0, keepdims=True)
    n_obs = returns.shape[0]
    return centred.T @ centred / jnp.maximum(n_obs - 1, 1)


# ── Keyed generators ─────────────────────────────────────────────────


def psd_factor(
    cov: Float[Array, "n n"],
) -> Float[Array, "n n"]:
    r"""Spectral factor :math:`L` with :math:`L L^{\top} = \Sigma`.

    Used instead of a Cholesky factor because a risk covariance is
    routinely **semi**-definite rather than definite: a deterministic
    factor (a curve held fixed in a stress), a block omitted from
    :func:`block_covariance`, or simply fewer observations than factors
    all make :math:`\Sigma` singular, and ``jnp.linalg.cholesky`` returns
    NaNs for all of them.  The eigendecomposition

    .. math::

        \Sigma = V \Lambda V^{\top}, \qquad
        L = V \sqrt{\max(\Lambda, 0)}

    handles those cases exactly, and clamping the eigenvalues projects a
    mildly indefinite input onto the nearest PSD matrix in the spectral
    sense — a visible, documented approximation instead of a silent NaN.

    Args:
        cov: Symmetric covariance matrix.

    Returns:
        A factor ``L`` such that ``L @ L.T`` reproduces the PSD part of
        ``cov``.
    """
    eigvals, eigvecs = jnp.linalg.eigh(cov)
    return eigvecs * jnp.sqrt(jnp.maximum(eigvals, 0.0))


def keyed_parametric_scenarios(
    key: jax.Array,
    cov: Float[Array, "n_factors n_factors"],
    layout: KeyedFactorLayout,
    n_scenarios: int,
    distribution: str = "normal",
    df: float = 5.0,
) -> KeyedScenario:
    """Sample correlated keyed scenarios from a parametric distribution.

    The keyed counterpart of :func:`parametric_scenarios`.  ``cov`` is
    indexed by :meth:`KeyedFactorLayout.factor_ids` — so a projection
    curve (or a basis, in a layout from :func:`as_basis_layout`) owns its
    own rows and columns, and the P&L it generates through
    :func:`valax.risk.state_var.group_scenario_pnl` is its own.

    Args:
        key: JAX PRNG key.
        cov: Covariance of the factor changes, in layout coordinates;
            build one with :func:`block_covariance`.  Factored with
            :func:`psd_factor`, so a singular covariance (a block with no
            variance, e.g. a stress on the basis alone) is fine.
        layout: The coordinate system.
        n_scenarios: Number of scenarios to draw.
        distribution: ``"normal"`` or ``"t"``.
        df: Degrees of freedom for the t-distribution.

    Returns:
        A :class:`~valax.market.keyed_scenario.KeyedScenario` whose leaves
        carry a leading ``n_scenarios`` axis.

    Raises:
        ValueError: If ``cov`` does not match the layout's factor count.
    """
    n = layout.n_factors
    cov = jnp.asarray(cov)
    if cov.shape != (n, n):
        raise ValueError(
            f"keyed_parametric_scenarios: cov has shape {cov.shape} but "
            f"the layout has {n} factors."
        )

    L = psd_factor(cov)
    key1, key2 = jax.random.split(key)
    z = jax.random.normal(key1, shape=(n_scenarios, n))

    if distribution == "t":
        # t = normal / sqrt(chi2 / df);  chi2(df) = 2 * Gamma(df/2, 1).
        chi2 = 2.0 * jax.random.gamma(key2, df / 2.0, shape=(n_scenarios, 1))
        z = z * jnp.sqrt(df / chi2)

    return layout.unflatten(z @ L.T)


def keyed_historical_scenarios(
    returns: Float[Array, "n_obs n_factors"],
    layout: KeyedFactorLayout,
) -> KeyedScenario:
    """Build keyed scenarios from historical factor changes.

    The keyed counterpart of :func:`historical_scenarios`.  Columns must be
    ordered as :meth:`KeyedFactorLayout.factor_ids`.

    Args:
        returns: ``(n_obs, n_factors)`` observed changes.
        layout: The coordinate system.

    Returns:
        A batched :class:`~valax.market.keyed_scenario.KeyedScenario` with
        ``n_scenarios = n_obs``.

    Raises:
        ValueError: If ``returns`` has the wrong width.
    """
    return layout.unflatten(jnp.asarray(returns))


def keyed_stress_scenario(
    layout: KeyedFactorLayout,
    *,
    spot_shock: Float[Array, ""] | float = 0.0,
    vol_shock: Float[Array, ""] | float = 0.0,
    dividend_shock: Float[Array, ""] | float = 0.0,
    parallel_shift: Float[Array, ""] | float = 0.0,
    curve_shifts: dict | None = None,
    curve_shocks: dict | None = None,
) -> KeyedScenario:
    """Construct a single deterministic keyed stress scenario.

    The keyed counterpart of :func:`stress_scenario`.  Sizes every
    per-curve vector from the layout, so the caller never has to know a
    curve's pillar count — and names the curve it moves, so
    ``parallel_shift`` on a two-curve book is no longer ambiguous.

    Args:
        layout: The coordinate system.
        spot_shock: Applied to every spot block.
        vol_shock: Applied to every vol block (parallel surface shift).
        dividend_shock: Applied to every dividend block.
        parallel_shift: Applied to every pillar of every curve block that
            ``curve_shifts`` / ``curve_shocks`` do not name.
        curve_shifts: ``curve_id -> parallel bump`` for named curves,
            overriding ``parallel_shift``.
        curve_shocks: ``curve_id -> per-pillar bumps`` for full control,
            overriding both of the above.

    Returns:
        A :class:`~valax.market.keyed_scenario.KeyedScenario`.

    Raises:
        KeyError: If a named curve has no block in the layout.
        ValueError: If a per-pillar vector has the wrong length.
    """
    curve_shifts = curve_shifts or {}
    curve_shocks = curve_shocks or {}
    for curve_id in (*curve_shifts, *curve_shocks):
        layout.block("curve", curve_id)   # raises KeyError if unknown

    factors = jnp.zeros((layout.n_factors,))
    for block in layout.blocks:
        sl = layout.index(block.kind, block.key)
        if block.kind == "spot":
            value = jnp.full((1,), spot_shock)
        elif block.kind == "vol":
            value = jnp.full((1,), vol_shock)
        elif block.kind == "dividend":
            value = jnp.full((1,), dividend_shock)
        elif block.key in curve_shocks:
            value = jnp.asarray(curve_shocks[block.key])
            if value.shape != (block.size,):
                raise ValueError(
                    f"keyed_stress_scenario: curve_shocks[{block.key!r}] "
                    f"has shape {value.shape}, expected {(block.size,)}."
                )
        elif block.key in curve_shifts:
            value = jnp.full((block.size,), curve_shifts[block.key])
        else:
            value = jnp.full((block.size,), parallel_shift)
        factors = factors.at[sl].set(value)

    return layout.unflatten(factors)


def keyed_curve_stress(
    layout: KeyedFactorLayout,
    curve_id: str,
    shape: str = "parallel",
    low: Float[Array, ""] | float = 0.0,
    high: Float[Array, ""] | float = 0.0,
) -> KeyedScenario:
    """Shape stress on one named curve, every other factor held fixed.

    The keyed generalisation of :func:`steepener` / :func:`flattener` /
    :func:`butterfly`, which move "the" curve because the flat layout has
    only one.  Shares its shape algebra with them through
    :func:`curve_shape_bumps`.

    In a layout from :func:`as_basis_layout`, naming the basis block
    shapes the **basis** rather than the projection curve's absolute
    level — a steepening basis, holding discounting flat.

    Args:
        layout: The coordinate system.
        curve_id: Curve (or basis) block to shock.
        shape: One of :data:`CURVE_SHAPES`.
        low: Short-end bump (wings, for a butterfly), or the parallel bump.
        high: Long-end bump (belly, for a butterfly).

    Returns:
        A :class:`~valax.market.keyed_scenario.KeyedScenario`.

    Raises:
        KeyError: If ``curve_id`` has no block in the layout.
        ValueError: If ``shape`` is unknown.
    """
    block = layout.block("curve", curve_id)
    bumps = curve_shape_bumps(block.size, shape, low, high)
    return keyed_stress_scenario(layout, curve_shocks={curve_id: bumps})


def keyed_steepener(
    layout: KeyedFactorLayout,
    curve_id: str,
    short_bump: Float[Array, ""] | float,
    long_bump: Float[Array, ""] | float,
) -> KeyedScenario:
    """Steepener on one named curve (see :func:`keyed_curve_stress`)."""
    return keyed_curve_stress(
        layout, curve_id, "steepener", short_bump, long_bump,
    )


def keyed_flattener(
    layout: KeyedFactorLayout,
    curve_id: str,
    short_bump: Float[Array, ""] | float,
    long_bump: Float[Array, ""] | float,
) -> KeyedScenario:
    """Flattener on one named curve (see :func:`keyed_curve_stress`)."""
    return keyed_curve_stress(
        layout, curve_id, "flattener", short_bump, long_bump,
    )


def keyed_butterfly(
    layout: KeyedFactorLayout,
    curve_id: str,
    wing_bump: Float[Array, ""] | float,
    belly_bump: Float[Array, ""] | float,
) -> KeyedScenario:
    """Butterfly on one named curve (see :func:`keyed_curve_stress`)."""
    return keyed_curve_stress(
        layout, curve_id, "butterfly", wing_bump, belly_bump,
    )


def keyed_basis_shift(
    layout: KeyedFactorLayout,
    projection_id: str,
    bump: Float[Array, ""] | float,
    over: str = DEFAULT_ID,
    *,
    shape: str = "parallel",
    high: Float[Array, ""] | float = 0.0,
) -> KeyedScenario:
    r"""Widen the basis between two curves, discounting held fixed.

    The ``IR.BASIS.<ccy>.<tenorA>.<tenorB>`` stress, stated as a **basis**
    rather than as a raw zero-rate bump: the projection curve moves and
    the discount curve does not, so the spread — the thing a basis swap
    quotes — is what changed.

    Works on either layout. In an absolute layout it emits the equivalent
    absolute move on ``projection_id`` (and validates that ``over`` is a
    real curve, so the intent is recorded); in a basis layout from
    :func:`as_basis_layout` it sets the basis coordinates directly, which
    is also what makes the move appear on the basis rung of a
    :class:`~valax.risk.state_ladders.KeyedLadder`.

    Args:
        layout: The coordinate system.
        projection_id: Curve carrying the basis (tenor A).
        bump: Basis widening at the short end, or the parallel widening
            (e.g. ``0.0010`` for +10 bp).
        over: Discount curve the basis is quoted against (tenor B).
            Defaults to the ``"_default_"`` sentinel.
        shape: Basis shape, one of :data:`CURVE_SHAPES`.
        high: Long-end basis widening, for the non-parallel shapes.

    Returns:
        A :class:`~valax.market.keyed_scenario.KeyedScenario` moving only
        the basis.

    Raises:
        KeyError: If either curve has no block in the layout.
        ValueError: If the two ids are the same, or ``shape`` is unknown.
    """
    if projection_id == over:
        raise ValueError(
            f"keyed_basis_shift: a basis needs two distinct curves, got "
            f"{projection_id!r} twice."
        )
    layout.block("curve", over)
    return keyed_curve_stress(layout, projection_id, shape, bump, high)
