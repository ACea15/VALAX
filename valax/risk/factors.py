"""PCA factor model for yield-curve risk.

Wraps :func:`valax.risk.bucketing.pca_jacobian` into a rates-aware
workflow:

1. Convert a stack of :class:`~valax.curves.discount.DiscountCurve`
   snapshots into a ``(n_obs, n_pillars)`` matrix of additive zero-rate
   changes via :func:`zero_rate_returns_from_snapshots`.
2. Fit a low-rank factor model with :func:`fit_rates_pca` — returns a
   :class:`RatesFactorModel` packaging the pillar grid, loadings,
   eigenvalues, and explained-variance fraction.
3. Apply PC-score shocks to a base curve via
   :meth:`RatesFactorModel.shock_curve`, or build a full
   :class:`~valax.market.scenario.MarketScenario` via
   :meth:`RatesFactorModel.scenario`, which plugs straight into
   :func:`valax.risk.shocks.apply_scenario` and
   :func:`valax.risk.var.pnl_attribution`.

The first three components on a well-behaved yield curve are
interpretable as level / slope / curvature (Litterman & Scheinkman 1991)
and typically explain ≥99% of pillar variance — see
:doc:`/guide/pca-rates` for the end-to-end workflow and
:doc:`/theory` §7.8 for the underlying math.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction
from valax.market.scenario import MarketScenario
from valax.risk.bucketing import pca_jacobian, pullback_shocks
from valax.risk.shocks import bump_curve_zero_rates


# ── Returns extraction ──────────────────────────────────────────────


def _zero_rates_at(
    curve: DiscountCurve,
    dates: Int[Array, " n"],
) -> Float[Array, " n"]:
    """Continuously-compounded zero rates of ``curve`` at absolute dates.

    Internal helper.  ``r(t) = -log(DF(t)) / tau(ref, t)`` with a small
    floor on ``tau`` to keep the divide well-defined at the reference
    pillar (where the rate is undefined but the DF equals 1).
    """
    dfs = curve(dates)
    taus = year_fraction(curve.reference_date, dates, curve.day_count)
    return -jnp.log(dfs) / jnp.maximum(taus, 1e-12)


def zero_rate_returns_from_snapshots(
    curves: list[DiscountCurve],
    query_dates: Int[Array, " n_pillars"],
) -> Float[Array, "n_obs n_pillars"]:
    """Build a returns matrix from a time series of discount curves.

    For each snapshot, evaluate the continuously-compounded zero rate at
    every ``query_dates`` entry, stack into a ``(n_snapshots, n_pillars)``
    matrix, and first-difference along the time axis.  The result is
    the additive zero-rate change between consecutive snapshots, in the
    same units expected by :func:`valax.risk.bucketing.pca_jacobian` and
    the ``rate_shocks`` field of
    :class:`~valax.market.scenario.MarketScenario`.

    ``query_dates`` is a fixed grid of absolute ordinal dates shared
    across all snapshots — *not* a tenor offset.  This keeps the pillar
    grid fixed even when each ``curve.pillar_dates`` shifts forward;
    the rolling time-to-maturity drift is baked in deliberately, which
    is the standard convention for daily yield-curve PCA.

    Args:
        curves: Time series of :class:`DiscountCurve` snapshots, ordered
            ascending in observation time.  Length must be ≥ 2.
        query_dates: Absolute pillar dates at which to read zero rates,
            shared across all snapshots.

    Returns:
        ``(n_snapshots - 1, n_pillars)`` matrix of zero-rate differences.

    Raises:
        ValueError: If fewer than two snapshots are supplied.
    """
    if len(curves) < 2:
        raise ValueError(
            f"Need at least 2 snapshots to compute returns, got {len(curves)}.",
        )
    rates = jnp.stack(
        [_zero_rates_at(c, query_dates) for c in curves],
        axis=0,
    )  # (n_snapshots, n_pillars)
    return jnp.diff(rates, axis=0)


# ── Sign convention ─────────────────────────────────────────────────


def _canonicalize_sign_positive_level(
    jacobian: Float[Array, "n_pillars n_components"],
) -> Float[Array, "n_pillars n_components"]:
    """Flip column signs so each component has a non-negative mean loading.

    PCA loadings are sign-ambiguous: ``v`` and ``-v`` are both valid
    eigenvectors.  Forcing the column mean non-negative makes PC1 the
    "parallel up" factor (positive loadings at every pillar), PC2 a
    positive slope (long-end loading positive), and so on — the
    standard rates-PCA convention.
    """
    signs = jnp.where(jnp.mean(jacobian, axis=0) >= 0.0, 1.0, -1.0)
    return jacobian * signs


# ── Fitted model ────────────────────────────────────────────────────


class RatesFactorModel(eqx.Module):
    """Fitted PCA model for a yield-curve pillar grid.

    Wraps the output of :func:`fit_rates_pca` in a typed pytree so PC
    shocks can be applied to a :class:`~valax.curves.discount.DiscountCurve`
    or projected into a :class:`~valax.market.scenario.MarketScenario`
    without manually carrying the loading matrix around.

    Attributes:
        pillar_times: Year fractions corresponding to each pillar of
            the loading matrix.  Carried for diagnostics and scenario
            construction; not used by the linear algebra.
        jacobian: Orthonormal loading matrix of shape
            ``(n_pillars, n_components)``; column ``k`` is the ``k``-th
            principal component — the "level / slope / curvature / …"
            data-driven factors.
        eigenvalues: Top ``n_components`` eigenvalues of the sample
            covariance — variances of the PC scores in the same units
            as the input returns.  ``sqrt(eigenvalues[k])`` is the
            one-sigma score on component ``k``.
        fraction_explained: Scalar in ``[0, 1]`` — the fraction of total
            return variance captured by the retained components.
        n_components: Number of retained components (static field).
    """

    pillar_times: Float[Array, " n_pillars"]
    jacobian: Float[Array, "n_pillars n_components"]
    eigenvalues: Float[Array, " n_components"]
    fraction_explained: Float[Array, ""]
    n_components: int = eqx.field(static=True)

    # ── Shock application ───────────────────────────────────────────

    def shock_curve(
        self,
        curve: DiscountCurve,
        pc_scores: Float[Array, " n_components"],
    ) -> DiscountCurve:
        """Apply a PC-score shock to a base curve.

        Equivalent to ``bump_curve_zero_rates(curve, J @ pc_scores)``
        and shares its semantics: every pillar's zero rate is bumped
        by the corresponding entry of the reconstructed shock.  The
        ``curve`` must have the same number of pillars as the model
        (the linear algebra does not check this).

        Args:
            curve: Base discount curve.
            pc_scores: Length-``n_components`` vector of PC scores.
                Each entry is in score units, typically interpreted as
                multiples of ``sqrt(eigenvalues[k])``.

        Returns:
            A new shocked :class:`DiscountCurve`.
        """
        rate_shocks = pullback_shocks(self.jacobian, pc_scores)
        return bump_curve_zero_rates(curve, rate_shocks)

    def scenario(
        self,
        pc_scores: Float[Array, " n_components"],
        n_assets: int = 0,
    ) -> MarketScenario:
        """Build a :class:`MarketScenario` with rate shocks set from PC scores.

        Spot, vol, and dividend shocks are filled with zeros so the
        scenario is purely a rates move.  Feeds straight into
        :func:`valax.risk.shocks.apply_scenario` and
        :func:`valax.risk.var.pnl_attribution`.

        Args:
            pc_scores: PC scores to convert to pillar shocks; length
                must equal ``self.n_components``.
            n_assets: Number of equity-like assets in the target
                :class:`~valax.market.data.MarketData`.  Used to size
                the zero spot / vol / dividend shock vectors.

        Returns:
            A no-op-on-equities :class:`MarketScenario` carrying the
            reconstructed rate bumps.
        """
        rate_shocks = pullback_shocks(self.jacobian, pc_scores)
        return MarketScenario(
            spot_shocks=jnp.zeros(n_assets),
            vol_shocks=jnp.zeros(n_assets),
            rate_shocks=rate_shocks,
            dividend_shocks=jnp.zeros(n_assets),
        )

    # ── Diagnostics ─────────────────────────────────────────────────

    def reconstruct(
        self,
        returns: Float[Array, "n_obs n_pillars"],
    ) -> Float[Array, "n_obs n_pillars"]:
        """Project returns onto the retained components and reconstruct.

        Returns ``returns @ J @ Jᵀ``.  With ``n_components`` equal to
        the pillar count this is the identity (up to numerical noise);
        with fewer components it is the optimal rank-``n_components``
        approximation of ``returns`` in the Frobenius-norm sense.
        Sign flips applied by the sign convention cancel because they
        appear in both ``J`` and ``Jᵀ``.
        """
        J = self.jacobian
        return returns @ J @ J.T

    def r_squared_per_pillar(
        self,
        returns: Float[Array, "n_obs n_pillars"],
    ) -> Float[Array, " n_pillars"]:
        """Per-pillar coefficient of determination of the reconstruction.

        For each pillar, ``1 - SS_res / SS_tot`` on the *centred*
        returns.  The standard quality check for rates PCA: with three
        components on a well-behaved yield curve every entry should be
        ≥0.95, and PC1 + PC2 + PC3 commonly give ≥0.99 at the long end.
        """
        X = returns - jnp.mean(returns, axis=0, keepdims=True)
        X_hat = self.reconstruct(X)
        ss_res = jnp.sum((X - X_hat) ** 2, axis=0)
        ss_tot = jnp.sum(X ** 2, axis=0)
        return 1.0 - ss_res / jnp.maximum(ss_tot, 1e-12)


# ── Fitter ──────────────────────────────────────────────────────────


def fit_rates_pca(
    returns: Float[Array, "n_obs n_pillars"],
    pillar_times: Float[Array, " n_pillars"],
    n_components: int = 3,
    *,
    center: bool = True,
    sign_convention: str = "positive_level",
) -> RatesFactorModel:
    """Fit a top-``n_components`` PCA factor model on rate returns.

    Delegates the SVD to :func:`valax.risk.bucketing.pca_jacobian`,
    then canonicalises column signs per ``sign_convention`` so that the
    components have a stable, interpretable orientation across runs.

    Args:
        returns: ``(n_obs, n_pillars)`` matrix of additive zero-rate
            changes per pillar (e.g. produced by
            :func:`zero_rate_returns_from_snapshots`).
        pillar_times: Year fractions of the pillars; carried on the
            returned model for diagnostics and scenario construction.
            Must have length ``n_pillars``.
        n_components: Number of components to retain (defaults to 3,
            the standard level / slope / curvature triple).
        center: Forwarded to
            :func:`~valax.risk.bucketing.pca_jacobian`.  Subtract the
            per-pillar mean before decomposing.
        sign_convention:

            - ``"positive_level"`` (default): flip signs so each
              component has a non-negative mean loading — PC1 is
              "parallel up", PC2 is a positive long-end slope, etc.
            - ``"raw"``: leave the SVD signs untouched.

    Returns:
        A :class:`RatesFactorModel` ready for use in scenario / P&L
        pipelines.

    Raises:
        ValueError: If ``sign_convention`` is not one of the two
            supported values.
    """
    J, eigvals, frac = pca_jacobian(returns, n_components, center=center)
    if sign_convention == "positive_level":
        J = _canonicalize_sign_positive_level(J)
    elif sign_convention != "raw":
        raise ValueError(
            f"sign_convention must be 'positive_level' or 'raw', "
            f"got {sign_convention!r}",
        )
    return RatesFactorModel(
        pillar_times=jnp.asarray(pillar_times),
        jacobian=J,
        eigenvalues=eigvals,
        fraction_explained=frac,
        n_components=n_components,
    )


__all__ = [
    "RatesFactorModel",
    "fit_rates_pca",
    "zero_rate_returns_from_snapshots",
]
