r"""Keyed sensitivity ladders on a ``MarketState``, in scenario coordinates.

The counterpart of :mod:`valax.risk.ladders` for the container the unified
pricer takes, and the layer that makes a projection curve's risk a number
rather than an aspiration.

Two things change relative to the legacy ladder
-----------------------------------------------

**1. Coordinates are keyed, not positional.**
:class:`~valax.risk.ladders.SensitivityLadder` has one ``delta_rate``
vector, so a dual-curve book's OIS and projection risk are added together
before anyone sees them.  A :class:`KeyedLadder` is indexed by a
:class:`~valax.risk.scenarios.KeyedFactorLayout`: one block per named
curve, surface and asset, each coordinate carrying the hierarchical
factor identifier of :doc:`/risk-factors` § 1.2.  The same layout indexes
the covariance matrix and the scenario set, so a ladder, a stress and a
VaR number are all expressed in one coordinate system and line up by
*name*.

**2. Sensitivities are taken with respect to the shock, not the leaf.**
The ladder differentiates

.. math::

    x \;\longmapsto\; \mathrm{PV}\bigl(\mathrm{shock}(\text{market},
      \;\mathrm{unflatten}(x))\bigr)

at :math:`x = 0`, where :math:`x` is the layout's factor vector.  Three
consequences, all of them the point:

* **No chain rule to get wrong.** The legacy ladder differentiates w.r.t.
  discount factors and then converts to zero-rate space by hand
  (``∂V/∂r_i = ∂V/∂DF_i · (−t_i · DF_i)``, plus a second-order correction
  for gamma).  Here the zero-rate convention is whatever
  :func:`valax.risk.shocks.bump_graph_curve` means by it, exactly, to all
  orders — because that function is *inside* the differentiated map.
* **Vol works for every surface type.** A vol coordinate is the magnitude
  of the parallel shift applied by
  :func:`valax.risk.state_shocks.shift_surface`, so a grid, SVI, SABR,
  cube or constant surface all yield one well-defined vega. There is no
  need for a scalar ``vols`` field, and no node-level ambiguity.
* **Ladder and scenario cannot disagree.** A predicted P&L is a Taylor
  expansion of the *same* function a full revaluation evaluates, so
  :func:`keyed_waterfall_pnl`'s ``unexplained`` is genuinely the
  higher-order remainder and not a convention mismatch.

Everything goes through the one front door,
:func:`valax.core.pricing.price` (via
:func:`valax.risk.state_var.group_pv`): one ``jax.grad`` for the whole
first-order ladder and one ``jax.hessian`` for the whole second-order
one — against the ``2N`` and ``N²`` repricings a bump-and-reprice system
needs.

The basis rung
--------------
Applying :func:`valax.risk.scenarios.as_basis_layout` before building the
ladder reparameterises a projection curve's coordinates into a **basis**
over the discount curve.  The ladder's corresponding entries are then a
basis DV01 named ``IR.BASIS.<ccy>.<tenorA>.<tenorB>.<pillar>``, the
factor :doc:`/risk-factors` § 3.2 asks for — and
:func:`keyed_component_var` reports its VaR contribution separately from
the discount curve's.

Quantity and sign are not here, for the same reason as in
:mod:`valax.risk.state_var`: ``price`` returns a unit price, so a
position-aware layer does the weighting.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.core.market_state import DEFAULT_ID, MarketState
from valax.market.keyed_scenario import KeyedScenario
from valax.risk.scenarios import KeyedFactorLayout
from valax.risk.state_shocks import shock
from valax.risk.state_var import _warn_shadowed_shocks, group_pv


#: One basis point, the conventional unit of a rate ladder.
BASIS_POINT = 1e-4


# ── Ladder ───────────────────────────────────────────────────────────


class KeyedLadder(eqx.Module):
    """First- and second-order sensitivities in keyed scenario coordinates.

    Attributes:
        layout: The coordinate system; its
            :meth:`~valax.risk.scenarios.KeyedFactorLayout.factor_ids`
            name every entry of :attr:`delta` and every row/column of
            :attr:`gamma`.  Static.
        pv: Present value at the base market — the point the expansion is
            taken around.
        delta: ``∂PV/∂x`` at ``x = 0``, shape ``(n_factors,)``.  For a
            curve block the entries are per unit of zero rate, so a
            one-basis-point ladder is ``delta * BASIS_POINT``.
        gamma: ``∂²PV/∂x∂xᵀ``, shape ``(n_factors, n_factors)``, or
            ``None`` when the ladder was built first-order only.
    """

    layout: KeyedFactorLayout = eqx.field(static=True)
    pv: Float[Array, ""]
    delta: Float[Array, " n_factors"]
    gamma: Float[Array, "n_factors n_factors"] | None = None

    # ── Block accessors ──────────────────────────────────────────────

    def block(self, kind: str, key: str) -> Float[Array, " size"]:
        """First-order sensitivities of one block.

        Args:
            kind: One of :data:`~valax.risk.scenarios.FACTOR_KINDS`.
            key: Market-object identifier.

        Returns:
            The block's slice of :attr:`delta`.

        Raises:
            KeyError: If the layout has no such block.
        """
        return self.delta[self.layout.index(kind, key)]

    def gamma_block(
        self,
        kind_a: str,
        key_a: str,
        kind_b: str | None = None,
        key_b: str | None = None,
    ) -> Float[Array, "size_a size_b"]:
        """Second-order block ``∂²PV/∂x_a ∂x_b``.

        Args:
            kind_a: Kind of the first block.
            key_a: Identifier of the first block.
            kind_b: Kind of the second block; defaults to ``kind_a``.
            key_b: Identifier of the second block; defaults to ``key_a``
                (the diagonal block).

        Returns:
            The requested sub-matrix of :attr:`gamma`.

        Raises:
            ValueError: If the ladder carries no second-order term.
            KeyError: If either block is unknown.
        """
        if self.gamma is None:
            raise ValueError(
                "This KeyedLadder is first-order only; rebuild it with "
                "keyed_ladder(..., second_order=True)."
            )
        kind_b = kind_a if kind_b is None else kind_b
        key_b = key_a if key_b is None else key_b
        return self.gamma[
            self.layout.index(kind_a, key_a), self.layout.index(kind_b, key_b)
        ]

    def curve_ladder(
        self,
        curve_id: str,
        bump: float = BASIS_POINT,
    ) -> Float[Array, " n_pillars"]:
        """Per-pillar key-rate ladder of one named curve.

        Args:
            curve_id: Curve (or basis) block identifier.
            bump: Rate move the ladder is quoted per; one basis point by
                default.

        Returns:
            P&L per ``bump`` of zero rate at each pillar of that curve.

        Raises:
            KeyError: If the layout has no block for ``curve_id``.
        """
        return self.block("curve", curve_id) * bump

    def dv01(self, curve_id: str, bump: float = BASIS_POINT) -> Float[Array, ""]:
        """Parallel DV01 of one named curve.

        The sum of its key-rate ladder: the P&L of a parallel one-basis-point
        move in *that* curve with every other curve held fixed. On a
        dual-curve book the OIS and projection DV01s are separate numbers,
        which is the whole reason this module exists.

        Args:
            curve_id: Curve (or basis) block identifier.
            bump: Rate move to quote per.

        Returns:
            Scalar DV01.
        """
        return jnp.sum(self.curve_ladder(curve_id, bump))

    def basis_dv01(
        self,
        projection_id: str,
        over: str = DEFAULT_ID,
        bump: float = BASIS_POINT,
    ) -> Float[Array, ""]:
        r"""DV01 to the basis between two curves.

        The sensitivity to :math:`b` in
        :math:`r^{\text{proj}} = r^{\text{disc}} + b` with the discount
        curve held fixed — the ``IR.BASIS.<ccy>.<tenorA>.<tenorB>`` risk.

        Numerically this is the projection curve's own parallel DV01,
        because moving the basis with discounting fixed *is* moving the
        projection curve; the distinction is one of coordinates, and it
        becomes material in VaR and P&L attribution, where a basis layout
        (:func:`valax.risk.scenarios.as_basis_layout`) lets discounting
        and basis carry separate variances. ``over`` is validated so the
        pair being quoted is recorded rather than assumed.

        Args:
            projection_id: Curve carrying the basis (tenor A).
            over: Discount curve the basis is quoted against (tenor B).
            bump: Basis move to quote per; one basis point by default.

        Returns:
            Scalar basis DV01.

        Raises:
            KeyError: If either curve is absent from the layout.
            ValueError: If the two identifiers are the same.
        """
        if projection_id == over:
            raise ValueError(
                f"basis_dv01: a basis needs two distinct curves, got "
                f"{projection_id!r} twice."
            )
        self.layout.block("curve", over)
        return self.dv01(projection_id, bump)

    def vega(self, surface_id: str, bump: float = 0.01) -> Float[Array, ""]:
        """Parallel vega of one named vol surface.

        Args:
            surface_id: Surface block identifier.
            bump: Vol move to quote per; one vol point (1%) by default.

        Returns:
            P&L per ``bump`` of parallel shift in that surface.
        """
        return jnp.sum(self.block("vol", surface_id)) * bump

    def spot_delta(self, asset_id: str) -> Float[Array, ""]:
        """Spot delta of one asset (per unit of the shock's convention)."""
        return jnp.sum(self.block("spot", asset_id))

    def dividend_delta(self, asset_id: str, bump: float = BASIS_POINT) -> Float[Array, ""]:
        """Dividend-yield sensitivity of one asset, per ``bump``."""
        return jnp.sum(self.block("dividend", asset_id)) * bump

    # ── Reporting ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """First-order ladder as ``factor_id -> sensitivity``.

        Returns:
            One entry per coordinate, keyed by hierarchical factor id.
        """
        ids = self.layout.factor_ids()
        return {fid: self.delta[i] for i, fid in enumerate(ids)}

    def block_totals(self) -> dict:
        """Summed first-order sensitivity of each block, keyed by prefix.

        The coarsest honest summary of a keyed ladder: one number per named
        market object (per curve, per surface, per asset), rather than one
        number per asset class.

        Returns:
            ``factor-id prefix -> summed sensitivity``.
        """
        return {
            block.prefix: jnp.sum(self.block(block.kind, block.key))
            for block in self.layout.blocks
        }


# ── Construction ─────────────────────────────────────────────────────


def _pv_of_factors(
    instruments: Any,
    model: Any,
    market: MarketState,
    layout: KeyedFactorLayout,
    method: str,
    config: Any,
    key: Any,
):
    """Build the scalar map ``factor vector -> group PV`` that is differentiated.

    Args:
        instruments: Batched instrument pytree.
        model: Model pytree shared by the group, or ``None``.
        market: Base calibrated market.
        layout: The coordinate system.
        method: Pricing method.
        config: Method-specific numerical settings.
        key: PRNG key, shared by the base and every perturbed valuation.

    Returns:
        A pure function of the factor vector.
    """
    def pv(factors: Float[Array, " n_factors"]) -> Float[Array, ""]:
        shocked = shock(market, layout.unflatten(factors))
        return group_pv(
            instruments, model, shocked,
            method=method, config=config, key=key,
        )

    return pv


def keyed_ladder(
    instruments: Any,
    model: Any,
    market: MarketState,
    layout: KeyedFactorLayout | None = None,
    *,
    second_order: bool = False,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> KeyedLadder:
    """Build a keyed sensitivity ladder for one homogeneous group.

    One ``jax.grad`` gives every first-order sensitivity — every pillar of
    every curve, every surface, every spot and dividend — and one
    ``jax.hessian`` gives the whole second-order block matrix.  No
    bump-and-reprice anywhere, and no hand-written chain rule: the ladder
    is the derivative of the *shock-and-reprice* map itself (see the
    module docstring).

    ``key`` is threaded unchanged into every evaluation, so an MC group's
    ladder is a response to the factors rather than to fresh randomness.

    Args:
        instruments: Batched instrument pytree (leading batch dim), as for
            :func:`valax.risk.state_var.group_pv`.
        model: Model pytree shared by the group, or ``None`` for
            curve-discounted instruments.
        market: The calibrated market.
        layout: Factor coordinates; defaults to
            :meth:`~valax.risk.scenarios.KeyedFactorLayout.from_market_state`
            on ``market`` — every curve, surface and asset it holds.
        second_order: Also compute the ``(n_factors, n_factors)`` Hessian.
            Cost grows with the square of the factor count, so it is
            opt-in.
        method: ``"analytic" | "lattice" | "pde" | "mc" | "auto"``.
        config: Method-specific numerical settings.
        key: PRNG key, required for ``method="mc"``.

    Returns:
        A :class:`KeyedLadder`.

    Warns:
        UserWarning: If ``model`` carries its own scalar ``rate`` / ``vol``
            / ``dividend``, which shadows the corresponding market object
            so those ladder entries are exactly zero.  The risk is on the
            model gradient
            (:func:`valax.risk.state_var.group_model_gradient`), not lost.
    """
    if layout is None:
        layout = KeyedFactorLayout.from_market_state(market)

    _warn_shadowed_shocks(model, layout.zero(), "keyed_ladder")

    pv = _pv_of_factors(
        instruments, model, market, layout, method, config, key,
    )
    x0 = jnp.zeros((layout.n_factors,))

    value, delta = jax.value_and_grad(pv)(x0)
    gamma = jax.hessian(pv)(x0) if second_order else None

    return KeyedLadder(layout=layout, pv=value, delta=delta, gamma=gamma)


# ── Waterfall P&L ────────────────────────────────────────────────────


class KeyedWaterfall(eqx.Module):
    """Per-block P&L decomposition of a keyed scenario.

    The keyed analogue of :class:`valax.risk.ladders.WaterfallPnL`: where
    that has a fixed set of rungs (spot / vol / rate / dividend, plus
    second-order terms), this has **one rung per named market object**, so
    a dual-curve book reports an OIS rung and a projection (or basis) rung
    separately.  The algebra is the same Taylor expansion — see
    :doc:`/risk-ladders` — applied in keyed coordinates.

    Attributes:
        delta_pnl: ``factor-id prefix -> δ_bᵀ x_b``, the first-order P&L of
            each block.
        gamma_pnl: ``factor-id prefix -> ½ x_bᵀ H_bb x_b``, each block's
            own second-order P&L.  Empty when the ladder was first-order.
        cross_pnl: The remaining second-order P&L, i.e. every
            block-against-other-block term (vanna, cross-curve convexity,
            spot × rate ...) collected into one number.  Zero when the
            ladder was first-order.
        total_first_order: Sum of :attr:`delta_pnl`.
        total_second_order: Sum of :attr:`gamma_pnl` plus
            :attr:`cross_pnl`.
        predicted: ``total_first_order + total_second_order``.
        actual: Full-repricing P&L, or NaN when not supplied.
        unexplained: ``actual - predicted``, or NaN.
    """

    delta_pnl: dict
    gamma_pnl: dict
    cross_pnl: Float[Array, ""]
    total_first_order: Float[Array, ""]
    total_second_order: Float[Array, ""]
    predicted: Float[Array, ""]
    actual: Float[Array, ""]
    unexplained: Float[Array, ""]


def keyed_waterfall_pnl(
    ladder: KeyedLadder,
    scenario: KeyedScenario,
    actual: Float[Array, ""] | None = None,
) -> KeyedWaterfall:
    r"""Decompose a keyed scenario into per-block P&L rungs.

    Pure arithmetic on the ladder — no repricing:

    .. math::

        \Delta V \approx \delta^{\top} x
          + \tfrac12 x^{\top} H x,
        \qquad x = \mathrm{flatten}(\text{scenario}),

    split by block so every named curve, surface and asset owns a rung.
    Because :math:`x` is the *same* coordinate vector the ladder was
    differentiated in, the split is exact to the order retained and the
    residual is genuinely higher-order.

    Args:
        ladder: Ladder to expand (second-order rungs require
            ``second_order=True``).
        scenario: The scenario to decompose; its shocks must address
            blocks of the ladder's layout.
        actual: Optional full-repricing P&L, e.g. from
            :func:`valax.risk.state_var.group_reprice` minus
            :attr:`KeyedLadder.pv`.  Supplied → ``unexplained`` is
            populated.

    Returns:
        A :class:`KeyedWaterfall`.

    Raises:
        ValueError: If the scenario shocks an identifier the layout has no
            block for.
    """
    layout = ladder.layout
    x = layout.flatten(scenario)

    delta_pnl = {}
    for block in layout.blocks:
        sl = layout.index(block.kind, block.key)
        delta_pnl[block.prefix] = jnp.sum(ladder.delta[sl] * x[sl])
    total_1st = sum(delta_pnl.values(), jnp.zeros(()))

    gamma_pnl: dict = {}
    cross = jnp.zeros(())
    if ladder.gamma is not None:
        total_2nd_all = 0.5 * (x @ ladder.gamma @ x)
        for block in layout.blocks:
            sl = layout.index(block.kind, block.key)
            gamma_pnl[block.prefix] = 0.5 * (
                x[sl] @ ladder.gamma[sl, sl] @ x[sl]
            )
        cross = total_2nd_all - sum(gamma_pnl.values(), jnp.zeros(()))
        total_2nd = total_2nd_all
    else:
        total_2nd = jnp.zeros(())

    predicted = total_1st + total_2nd
    nan = jnp.asarray(jnp.nan)
    actual_value = nan if actual is None else jnp.asarray(actual)

    return KeyedWaterfall(
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        cross_pnl=cross,
        total_first_order=total_1st,
        total_second_order=total_2nd,
        predicted=predicted,
        actual=actual_value,
        unexplained=actual_value - predicted,
    )


def keyed_waterfall_report(
    instruments: Any,
    model: Any,
    market: MarketState,
    scenario: KeyedScenario,
    ladder: KeyedLadder | None = None,
    *,
    layout: KeyedFactorLayout | None = None,
    second_order: bool = True,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> KeyedWaterfall:
    """Keyed waterfall including the full-repricing ``actual``.

    Builds the ladder if one is not supplied, decomposes ``scenario``, and
    reprices the group under it so ``actual`` and ``unexplained`` are
    populated.  For many scenarios against one market, build the ladder
    once with :func:`keyed_ladder` and pass it in — the decomposition is
    then arithmetic only.

    Args:
        instruments: Batched instrument pytree (leading batch dim).
        model: Model pytree shared by the group, or ``None``.
        market: Base calibrated market.
        scenario: Scenario to decompose.
        ladder: Optional precomputed ladder.
        layout: Factor coordinates used when building a ladder; ignored if
            ``ladder`` is given.
        second_order: Whether a ladder built here carries a Hessian.
        method: ``"analytic" | "lattice" | "pde" | "mc" | "auto"``.
        config: Method-specific numerical settings.
        key: PRNG key; reused for the base and shocked valuations so an MC
            group's P&L is a scenario response, not sampling noise.

    Returns:
        A :class:`KeyedWaterfall` with ``actual`` and ``unexplained`` set.
    """
    if ladder is None:
        ladder = keyed_ladder(
            instruments, model, market, layout,
            second_order=second_order,
            method=method, config=config, key=key,
        )

    shocked_pv = group_pv(
        instruments, model, shock(market, scenario),
        method=method, config=config, key=key,
    )
    return keyed_waterfall_pnl(ladder, scenario, actual=shocked_pv - ladder.pv)


# ── Delta-normal VaR in keyed coordinates ────────────────────────────


def keyed_parametric_var(
    ladder: KeyedLadder,
    cov: Float[Array, "n_factors n_factors"],
    confidence: float = 0.99,
    horizon: float = 1.0,
) -> Float[Array, ""]:
    r"""Delta-normal VaR from a keyed ladder.

    .. math::

        \sigma_P = \sqrt{\delta^{\top} \Sigma \delta}, \qquad
        \mathrm{VaR} = z_{\alpha}\, \sigma_P \sqrt{h}.

    The keyed counterpart of :func:`valax.risk.var.parametric_var`, whose
    docstring has to warn that named projection / basis curves "contribute
    no delta and their risk is not in the number".  Here they do and it
    is: ``cov`` is indexed by the ladder's layout, so every curve has its
    own block (build one with
    :func:`valax.risk.scenarios.block_covariance`).

    This is a first-order measure; a convex book wants full-revaluation
    VaR (:func:`valax.risk.state_var.group_scenario_pnl` into
    :func:`valax.risk.var.value_at_risk`), and a quantile must be taken at
    the *aggregate* level, never summed across groups.

    Args:
        ladder: The sensitivities.
        cov: Covariance of factor changes over the horizon, in layout
            coordinates.
        confidence: Confidence level (e.g. 0.99).
        horizon: Horizon scaling in units of the covariance's period
            (square-root-of-time).

    Returns:
        VaR as a positive loss threshold.

    Raises:
        ValueError: If ``cov`` does not match the layout's factor count.
    """
    cov = _checked_cov(ladder, cov)
    variance = ladder.delta @ cov @ ladder.delta
    z_alpha = -jax.scipy.stats.norm.ppf(1.0 - confidence)
    return z_alpha * jnp.sqrt(jnp.maximum(variance, 0.0) * horizon)


def keyed_component_var(
    ladder: KeyedLadder,
    cov: Float[Array, "n_factors n_factors"],
    confidence: float = 0.99,
    horizon: float = 1.0,
) -> dict:
    r"""Per-block VaR contributions (Euler allocation).

    Splits :func:`keyed_parametric_var` across the layout's blocks:

    .. math::

        C_b = \mathrm{VaR}\,
              \frac{x_b^{\top}(\Sigma \delta)_b}{\delta^{\top}\Sigma\delta},
        \qquad \sum_b C_b = \mathrm{VaR},

    which is the standard additive decomposition of a delta-normal VaR
    (Euler / marginal allocation; it adds up exactly because VaR is
    homogeneous of degree one in the positions).

    This is the number the roadmap's multi-curve story asks for: with a
    basis layout (:func:`valax.risk.scenarios.as_basis_layout`) the
    discount curve and the ``IR.BASIS...`` factor appear as separate,
    additive contributions to the same VaR.

    Args:
        ladder: The sensitivities.
        cov: Covariance of factor changes, in layout coordinates.
        confidence: Confidence level.
        horizon: Horizon scaling.

    Returns:
        ``factor-id prefix -> VaR contribution``, summing to the total.

    Raises:
        ValueError: If ``cov`` does not match the layout's factor count.
    """
    cov = _checked_cov(ladder, cov)
    delta = ladder.delta
    variance = delta @ cov @ delta
    total = keyed_parametric_var(ladder, cov, confidence, horizon)
    weights = delta * (cov @ delta) / jnp.maximum(variance, 1e-300)

    layout = ladder.layout
    return {
        block.prefix: total * jnp.sum(weights[layout.index(block.kind, block.key)])
        for block in layout.blocks
    }


def _checked_cov(
    ladder: KeyedLadder,
    cov: Float[Array, "n_factors n_factors"],
) -> Float[Array, "n_factors n_factors"]:
    """Validate a covariance matrix against a ladder's layout.

    Args:
        ladder: The ladder supplying the expected factor count.
        cov: Candidate covariance matrix.

    Returns:
        ``cov`` as a JAX array.

    Raises:
        ValueError: If the shape does not match the layout.
    """
    cov = jnp.asarray(cov)
    n = ladder.layout.n_factors
    if cov.shape != (n, n):
        raise ValueError(
            f"covariance has shape {cov.shape} but the ladder's layout has "
            f"{n} factors ({ladder.layout.factor_ids()[:3]}...)."
        )
    return cov


__all__ = [
    "BASIS_POINT",
    "KeyedLadder",
    "KeyedWaterfall",
    "keyed_component_var",
    "keyed_ladder",
    "keyed_parametric_var",
    "keyed_waterfall_pnl",
    "keyed_waterfall_report",
]
