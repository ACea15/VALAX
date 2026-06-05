"""Risk-factor bucketing: linear aggregation and Jacobian reparameterization.

Two transformations live in this module — same matrix algebra, different
semantics.

**Linear aggregation** ``b = A x`` collapses a granular factor vector
``x`` of length ``n`` into a coarser bucket vector ``b`` of length
``m`` (``m \\le n``).  Used for regulatory bucketing (FRTB SBA, ISDA
SIMM), sector / currency aggregation, and dimension reduction by
coarsening of a curve grid.  The induced transformations are

.. math::

    \\delta_b = A\\,\\delta_x, \\qquad \\Delta x = A^{\\top}\\,\\Delta b,
    \\qquad \\Sigma_b = A\\,\\Sigma_x\\,A^{\\top}.

The shock formula is the canonical dual of the sensitivity formula —
the unique choice that makes ``\\delta_b\\cdot\\Delta b = \\delta_x\\cdot\\Delta x``
hold for every ``\\delta_x``.

**Jacobian reparameterization** ``x = g(b)`` expresses raw factors as a
smooth function of bucket factors.  With ``J = \\partial x / \\partial b``
the chain rule gives

.. math::

    \\delta_b = J^{\\top}\\,\\delta_x, \\qquad \\Delta x = J\\,\\Delta b,
    \\qquad \\Sigma_b = J^{\\top}\\,\\Sigma_x\\,J,

which reduces to the linear case for ``J = A^{\\top}``.  PCA scores
(``J`` = top eigenvectors of ``\\Sigma_x``), classical
level / slope / curvature factors, and SVI / SABR parameter ladders all
fit this template.

See :doc:`../theory.md` §7.8 for the full derivation and the
``docs/risk-factors.md`` registry for which factor categories support
which transformations.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


# ── Core type ────────────────────────────────────────────────────────


class BucketMap(eqx.Module):
    """Linear aggregation matrix ``A`` mapping factors to buckets.

    The matrix ``A`` has shape ``(n_buckets, n_factors)``; row ``i``
    encodes the weights with which the granular factors enter bucket
    ``i``.  Common bucket structures (FRTB tenor vertices, equity
    sectors, currency baskets, …) are produced by the builders later in
    this module.

    The labels are static (Python tuples), so a ``BucketMap`` remains
    fully JIT-compatible while still carrying enough metadata to drive
    human-readable risk reports.

    Attributes:
        matrix: ``A`` matrix of shape ``(n_buckets, n_factors)``.
        bucket_labels: Optional human-readable bucket names, length
            ``n_buckets``.  Stored as a tuple so the pytree stays
            JIT-safe.
        factor_labels: Optional factor names, length ``n_factors``.
    """

    matrix: Float[Array, "n_buckets n_factors"]
    bucket_labels: tuple = eqx.field(static=True, default=())
    factor_labels: tuple = eqx.field(static=True, default=())

    @property
    def n_buckets(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_factors(self) -> int:
        return self.matrix.shape[1]


# ── Linear aggregation ops ──────────────────────────────────────────


def aggregate(
    bucket_map: BucketMap,
    sensitivities: Float[Array, " n_factors"],
) -> Float[Array, " n_buckets"]:
    """Bucket sensitivities ``\\delta_b = A\\,\\delta_x``.

    Sum (or weighted-sum) raw-factor sensitivities into bucket
    sensitivities.  Linear in ``\\delta_x``; preserves the per-bucket
    PnL when paired with :func:`pushforward_scenario` shocks.
    """
    return bucket_map.matrix @ sensitivities


def pushforward_scenario(
    bucket_map: BucketMap,
    bucket_shocks: Float[Array, " n_buckets"],
) -> Float[Array, " n_factors"]:
    """Spread a bucket-level shock to the underlying raw factors.

    ``\\Delta x = A^{\\top}\\,\\Delta b`` — the unique factor shock for
    which ``\\delta_x\\cdot\\Delta x = \\delta_b\\cdot\\Delta b`` holds for
    every ``\\delta_x``.  Concretely, the bucket move is broadcast to
    each of its constituent factors with the same weights used by the
    aggregation.

    The result is in *factor space* and can be fed straight to
    ``MarketScenario.rate_shocks`` (or any other ``ScenarioSet`` leaf)
    after slicing.
    """
    return bucket_map.matrix.T @ bucket_shocks


def aggregate_covariance(
    bucket_map: BucketMap,
    cov: Float[Array, "n_factors n_factors"],
) -> Float[Array, "n_buckets n_buckets"]:
    """Project a factor covariance into bucket space.

    ``\\Sigma_b = A\\,\\Sigma_x\\,A^{\\top}``.  PSD is preserved because
    the projection is a congruence by ``A``.  For a 30 × 30 raw covariance
    and a 10-bucket map this returns a 10 × 10 well-conditioned matrix.
    """
    return bucket_map.matrix @ cov @ bucket_map.matrix.T


def aggregate_matrix(
    bucket_map_rows: BucketMap,
    matrix: Float[Array, "n_rows n_cols"],
    bucket_map_cols: BucketMap,
) -> Float[Array, "n_row_buckets n_col_buckets"]:
    """Bilaterally bucket a rectangular block (e.g. a cross-gamma matrix).

    ``M_b = A_r\\,M\\,A_c^{\\top}``.  Useful for the
    ``cross_spot_rate`` / ``cross_vol_rate`` blocks of a
    :class:`valax.risk.ladders.SensitivityLadder`.
    """
    return bucket_map_rows.matrix @ matrix @ bucket_map_cols.matrix.T


# ── Jacobian reparameterization ops ─────────────────────────────────


def pushforward_sensitivities(
    jacobian: Float[Array, "n_factors n_buckets"],
    sensitivities: Float[Array, " n_factors"],
) -> Float[Array, " n_buckets"]:
    """Push sensitivities through a smooth reparameterization.

    ``\\delta_b = J^{\\top}\\,\\delta_x`` where ``J = \\partial x/\\partial b``.
    Reduces to linear aggregation when ``J = A^{\\top}``.
    """
    return jacobian.T @ sensitivities


def pullback_shocks(
    jacobian: Float[Array, "n_factors n_buckets"],
    bucket_shocks: Float[Array, " n_buckets"],
) -> Float[Array, " n_factors"]:
    """Pull a bucket shock back to factor space through the Jacobian.

    ``\\Delta x = J\\,\\Delta b``.  Exact for linear ``J``; first-order
    accurate for smooth nonlinear reparameterizations.
    """
    return jacobian @ bucket_shocks


def reparameterize_covariance(
    jacobian: Float[Array, "n_factors n_buckets"],
    cov: Float[Array, "n_factors n_factors"],
) -> Float[Array, "n_buckets n_buckets"]:
    """Project a factor covariance into reparameterized space.

    ``\\Sigma_b = J^{\\top}\\,\\Sigma_x\\,J``.  PSD-preserving congruence.
    """
    return jacobian.T @ cov @ jacobian


def jacobian_from_fn(
    b_to_x: Callable[[Float[Array, " n_buckets"]], Float[Array, " n_factors"]],
    b_base: Float[Array, " n_buckets"],
) -> Float[Array, "n_factors n_buckets"]:
    """Compute ``J = \\partial x/\\partial b`` of a smooth reparameterization.

    Thin wrapper over ``jax.jacobian`` for ergonomics in the bucketing
    pipeline.  ``b_to_x`` is any pure JAX function (e.g. an SVI / SABR
    vol-grid reconstruction).
    """
    return jax.jacobian(b_to_x)(b_base)


# ── Common builders: tenor / sector aggregation ─────────────────────


def tenor_bucket_map(
    pillar_times: Float[Array, " n_pillars"],
    bucket_edges: Float[Array, " n_buckets"],
    weight: str = "indicator",
) -> BucketMap:
    """Build a :class:`BucketMap` from a pillar grid to a coarser bucket grid.

    ``bucket_edges`` is interpreted as the *representative tenor* of each
    bucket (e.g. the FRTB IR vertices ``[0.25, 0.5, 1, 2, 3, 5, 10, 15,
    20, 30]``).

    Two weighting conventions are supported:

    - ``weight="indicator"``: each pillar contributes 100% of its
      sensitivity to its nearest bucket vertex.  Result is the sum of
      DV01s within each bucket — the standard FRTB SBA / SIMM rule.
    - ``weight="linear"``: each pillar is split linearly between its two
      adjacent bucket vertices, with weights summing to 1.  Avoids step
      discontinuities when the pillar grid moves between buckets and is
      the conventional approach for *internal* risk reports.

    Args:
        pillar_times: Year fractions of the source pillars.
        bucket_edges: Year fractions of the bucket vertices.  Must be
            sorted ascending.
        weight: ``"indicator"`` (default) or ``"linear"``.

    Returns:
        A :class:`BucketMap` with ``matrix`` of shape
        ``(n_buckets, n_pillars)``.
    """
    pillar_times = jnp.asarray(pillar_times)
    bucket_edges = jnp.asarray(bucket_edges)
    n_pillars = pillar_times.shape[0]
    n_buckets = bucket_edges.shape[0]
    labels = tuple(f"T={float(t):.2g}" for t in bucket_edges)

    if weight == "indicator":
        # Nearest-vertex assignment.
        dist = jnp.abs(pillar_times[None, :] - bucket_edges[:, None])
        assignment = jnp.argmin(dist, axis=0)  # (n_pillars,)
        # One-hot encoding ⇒ matrix of shape (n_buckets, n_pillars)
        matrix = (jnp.arange(n_buckets)[:, None] == assignment[None, :]).astype(
            jnp.float64,
        )
    elif weight == "linear":
        # For each pillar t_j find the bucket interval (e_i, e_{i+1}]
        # containing it, then split the weight linearly between i and
        # i+1.  Pillars below the first vertex or above the last vertex
        # land entirely on the nearest endpoint.
        # Build (n_buckets, n_pillars) matrix.
        idx = jnp.clip(
            jnp.searchsorted(bucket_edges, pillar_times, side="right") - 1,
            0,
            n_buckets - 2,
        )
        e_lo = bucket_edges[idx]
        e_hi = bucket_edges[idx + 1]
        w_hi = jnp.clip((pillar_times - e_lo) / jnp.maximum(e_hi - e_lo, 1e-12),
                       0.0, 1.0)
        w_lo = 1.0 - w_hi
        # Place w_lo at row idx and w_hi at row idx + 1 for each pillar.
        rows = jnp.arange(n_buckets)
        is_lo = rows[:, None] == idx[None, :]
        is_hi = rows[:, None] == (idx + 1)[None, :]
        matrix = (
            is_lo.astype(jnp.float64) * w_lo[None, :]
            + is_hi.astype(jnp.float64) * w_hi[None, :]
        )
    else:
        raise ValueError(
            f"weight must be 'indicator' or 'linear', got {weight!r}",
        )

    return BucketMap(
        matrix=matrix,
        bucket_labels=labels,
        factor_labels=tuple(f"t={float(t):.3g}" for t in pillar_times),
    )


def equal_weight_bucket_map(
    group_membership: tuple[int, ...],
    n_buckets: int,
    bucket_labels: tuple = (),
) -> BucketMap:
    """Each factor goes to exactly one bucket with weight 1.

    Standard for non-tenor aggregations: each equity in a single sector,
    each name in a single rating bucket, each currency in a single
    regional bucket.

    Args:
        group_membership: ``(n_factors,)`` tuple of integers in
            ``[0, n_buckets)``, one per factor.
        n_buckets: Total number of buckets.
        bucket_labels: Optional bucket names.

    Returns:
        A :class:`BucketMap` with ``matrix[i, j] = 1`` iff
        ``group_membership[j] == i``.
    """
    membership = jnp.asarray(group_membership, dtype=jnp.int32)
    n_factors = membership.shape[0]
    matrix = (jnp.arange(n_buckets)[:, None] == membership[None, :]).astype(
        jnp.float64,
    )
    return BucketMap(
        matrix=matrix,
        bucket_labels=bucket_labels or tuple(
            f"bucket_{i}" for i in range(n_buckets)
        ),
        factor_labels=tuple(f"factor_{j}" for j in range(n_factors)),
    )


# ── Common Jacobians ────────────────────────────────────────────────


def level_slope_curvature_jacobian(
    pillar_times: Float[Array, " n_pillars"],
) -> Float[Array, "n_pillars 3"]:
    """Litterman-Scheinkman level / slope / curvature factors.

    Builds a fixed (non-PCA) reparameterization of a yield curve into
    three interpretable buckets:

    - **Level**: constant 1 across pillars.
    - **Slope**: centred linear in ``t``.
    - **Curvature**: centred quadratic with the mean removed.

    The result is the Jacobian ``J = \\partial x / \\partial b`` where
    ``b = (\\text{level}, \\text{slope}, \\text{curvature})``.  The
    columns are *not* orthonormalised (use Gram-Schmidt if a strictly
    orthogonal basis is required).
    """
    t = jnp.asarray(pillar_times)
    n = t.shape[0]
    level = jnp.ones((n,))
    slope = t - jnp.mean(t)
    curvature = t ** 2 - jnp.mean(t ** 2)
    return jnp.stack([level, slope, curvature], axis=1)


def pca_jacobian(
    returns: Float[Array, "n_obs n_factors"],
    n_components: int,
    center: bool = True,
) -> tuple[
    Float[Array, "n_factors n_components"],
    Float[Array, " n_components"],
    Float[Array, ""],
]:
    """Top-``n_components`` PCA Jacobian for risk-factor reduction.

    Returns the eigenvectors (PC loadings) as the Jacobian ``J``, with
    columns ordered by decreasing eigenvalue.  These are the natural
    "level / slope / curvature / …" data-driven factors.

    Args:
        returns: ``(n_obs, n_factors)`` matrix of daily factor changes.
        n_components: Number of principal components to keep.
        center: If True (default) subtract the per-column mean before
            decomposing.

    Returns:
        ``(jacobian, eigenvalues, fraction_explained)``:

        - ``jacobian``: ``(n_factors, n_components)`` orthonormal columns.
          Push raw sensitivities through with
          :func:`pushforward_sensitivities`.
        - ``eigenvalues``: top ``n_components`` eigenvalues of the
          sample covariance.
        - ``fraction_explained``: scalar in [0, 1] — the fraction of
          *total* variance captured by the retained components.
    """
    X = jnp.asarray(returns)
    if center:
        X = X - jnp.mean(X, axis=0, keepdims=True)
    # SVD of the centred data: X = U S V^T ⇒ cov = V (S^2 / (n-1)) V^T
    # and the eigenvectors of cov are the columns of V.
    n_obs = X.shape[0]
    _, s, vt = jnp.linalg.svd(X, full_matrices=False)
    eigvals = s ** 2 / jnp.maximum(n_obs - 1, 1)
    total_var = jnp.sum(eigvals)
    jacobian = vt[:n_components, :].T  # (n_factors, n_components)
    top_eigvals = eigvals[:n_components]
    fraction_explained = jnp.sum(top_eigvals) / jnp.maximum(total_var, 1e-12)
    return jacobian, top_eigvals, fraction_explained


# ── Ladder convenience ─────────────────────────────────────────────


class BucketedLadder(eqx.Module):
    """Sensitivity ladder expressed in bucket coordinates.

    Mirrors :class:`valax.risk.ladders.SensitivityLadder` but each
    component is reduced to its respective bucket dimension.  Cross
    blocks are bilaterally bucketed: ``cross_spot_rate`` becomes
    ``(n_spot_buckets, n_rate_buckets)``.

    Optional labels are preserved as static tuples on each ladder
    component for human-readable reporting.

    Attributes mirror :class:`SensitivityLadder` with the trailing
    dimensions replaced by their bucket counts; ``*_labels`` fields
    carry the corresponding bucket names.
    """

    # First order
    delta_spot: Float[Array, " n_spot_buckets"]
    delta_vol: Float[Array, " n_vol_buckets"]
    delta_rate: Float[Array, " n_rate_buckets"]
    delta_div: Float[Array, " n_div_buckets"]

    # Second order — diagonals
    gamma_spot: Float[Array, " n_spot_buckets"]
    gamma_rate: Float[Array, " n_rate_buckets"]
    volga: Float[Array, " n_vol_buckets"]
    vanna: Float[Array, " n_spot_buckets"]

    # Cross blocks (bilaterally bucketed)
    cross_spot_rate: Float[Array, "n_spot_buckets n_rate_buckets"]
    cross_vol_rate: Float[Array, "n_vol_buckets n_rate_buckets"]

    # Labels (static)
    spot_bucket_labels: tuple = eqx.field(static=True, default=())
    vol_bucket_labels: tuple = eqx.field(static=True, default=())
    rate_bucket_labels: tuple = eqx.field(static=True, default=())
    div_bucket_labels: tuple = eqx.field(static=True, default=())


def _identity_bucket_map(n: int) -> BucketMap:
    """Bucket map that leaves a vector unchanged (n buckets = n factors)."""
    return BucketMap(matrix=jnp.eye(n), bucket_labels=(), factor_labels=())


def bucket_sensitivity_ladder(
    ladder,
    *,
    rate_bucket: BucketMap | None = None,
    spot_bucket: BucketMap | None = None,
    vol_bucket: BucketMap | None = None,
    div_bucket: BucketMap | None = None,
) -> BucketedLadder:
    """Aggregate every component of a :class:`SensitivityLadder` into buckets.

    Each ``*_bucket`` argument is optional; when omitted the
    corresponding factor axis is left at full granularity (identity
    bucket map).  Diagonal second-order rungs are bucketed with the
    same map as the corresponding first-order rung (gamma_spot via
    ``spot_bucket``, volga via ``vol_bucket``, gamma_rate via
    ``rate_bucket``, vanna via ``spot_bucket``).  Cross blocks are
    bucketed bilaterally.

    The ``vanna`` rung in :class:`SensitivityLadder` is stored as a
    diagonal indexed by asset, so it is bucketed by ``spot_bucket``
    only — its vol axis is implicit (same asset index as the spot).
    For full vol-axis bucketing of vanna, build the explicit
    cross-vol-spot Hessian block instead.

    Args:
        ladder: Source :class:`SensitivityLadder`.
        rate_bucket, spot_bucket, vol_bucket, div_bucket: Optional
            per-category :class:`BucketMap` objects.

    Returns:
        A :class:`BucketedLadder` in bucket coordinates.
    """
    rb = rate_bucket or _identity_bucket_map(ladder.delta_rate.shape[0])
    sb = spot_bucket or _identity_bucket_map(ladder.delta_spot.shape[0])
    vb = vol_bucket or _identity_bucket_map(ladder.delta_vol.shape[0])
    db = div_bucket or _identity_bucket_map(ladder.delta_div.shape[0])

    return BucketedLadder(
        delta_spot=aggregate(sb, ladder.delta_spot),
        delta_vol=aggregate(vb, ladder.delta_vol),
        delta_rate=aggregate(rb, ladder.delta_rate),
        delta_div=aggregate(db, ladder.delta_div),
        gamma_spot=aggregate(sb, ladder.gamma_spot),
        gamma_rate=aggregate(rb, ladder.gamma_rate),
        volga=aggregate(vb, ladder.volga),
        vanna=aggregate(sb, ladder.vanna),
        cross_spot_rate=aggregate_matrix(sb, ladder.cross_spot_rate, rb),
        cross_vol_rate=aggregate_matrix(vb, ladder.cross_vol_rate, rb),
        spot_bucket_labels=sb.bucket_labels,
        vol_bucket_labels=vb.bucket_labels,
        rate_bucket_labels=rb.bucket_labels,
        div_bucket_labels=db.bucket_labels,
    )
