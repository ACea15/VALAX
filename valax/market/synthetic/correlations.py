"""Random correlation-matrix samplers.

All functions return a matrix that is symmetric, unit-diagonal, and
positive semi-definite by construction.  The PSD guarantee comes from
projecting any clipped result onto the PSD cone via eigenvalue
flooring, then renormalising the diagonal back to one.

If you need a *deliberately broken* matrix for arbitrage tests, see
:func:`valax.market.synthetic.arbitrage.inject_non_psd_correlation`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.market.synthetic.seeds import SeedRegistry


def _project_to_correlation(
    m: Float[Array, "n n"],
    floor: float = 1e-10,
) -> Float[Array, "n n"]:
    """Project a symmetric near-correlation matrix onto the PSD cone
    with unit diagonal.

    Steps:
      1. Symmetrise.
      2. Floor eigenvalues at ``floor`` and reconstruct.
      3. Rescale by the inverse square root of the diagonal so the
         result has unit diagonal.
    """
    m = 0.5 * (m + m.T)
    eigvals, eigvecs = jnp.linalg.eigh(m)
    eigvals = jnp.maximum(eigvals, floor)
    m = (eigvecs * eigvals) @ eigvecs.T
    d_inv_sqrt = 1.0 / jnp.sqrt(jnp.clip(jnp.diag(m), min=floor))
    m = m * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    # Force exact unit diagonal post-rescale.
    return m.at[jnp.diag_indices_from(m)].set(1.0)


def sample_correlation(
    registry: SeedRegistry,
    n: int,
    *,
    min_corr: float = -0.3,
    max_corr: float = 0.85,
    kind: str = "random",
    stream_name: str = "synthetic.correlation.random",
) -> Float[Array, "n n"]:
    """Draw a random correlation matrix.

    Args:
        registry: Seed registry.
        n: Matrix dimension.
        min_corr: Lower clip applied to off-diagonal entries before
            PSD projection.
        max_corr: Upper clip applied to off-diagonal entries before
            PSD projection.
        kind: ``"random"`` (default), ``"identity"``, or pass to
            :func:`block_correlation` via the wrapper
            :func:`sample_correlation_from_config`.
        stream_name: Override for the registry stream name.

    Returns:
        ``(n, n)`` matrix that is symmetric, unit-diagonal, and PSD.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if kind == "identity":
        return jnp.eye(n, dtype=jnp.float64)
    if kind != "random":
        raise ValueError(
            f"Unknown kind {kind!r}; use sample_correlation_from_config "
            "for 'block'."
        )

    key = registry.key(stream_name)
    # Draw a Gaussian matrix, form Z Z^T / sqrt(diag), giving a valid
    # correlation matrix sampled from the Wishart family.  Then clip
    # the off-diagonals and reproject for the desired range.
    z = jax.random.normal(key, shape=(n, n + 2), dtype=jnp.float64)
    w = z @ z.T
    d_inv_sqrt = 1.0 / jnp.sqrt(jnp.diag(w))
    c = w * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    # Clip off-diagonals to the requested band, keeping the diagonal at 1.
    off_diag = jnp.clip(c, min_corr, max_corr)
    c_clipped = off_diag.at[jnp.diag_indices(n)].set(1.0)
    return _project_to_correlation(c_clipped)


def block_correlation(
    registry: SeedRegistry,
    block_sizes: tuple[int, ...],
    intra: float = 0.7,
    inter: float = 0.2,
    jitter: float = 0.02,
    stream_name: str = "synthetic.correlation.block",
) -> Float[Array, "n n"]:
    """Build a block-structured correlation matrix.

    Within-block entries are sampled around ``intra``; cross-block
    entries around ``inter``.  ``jitter`` controls the per-entry
    Gaussian noise added before PSD projection.

    Args:
        registry: Seed registry.
        block_sizes: Tuple of block sizes; ``sum(block_sizes)`` is the
            matrix dimension.
        intra: Mean correlation within a block.
        inter: Mean correlation across blocks.
        jitter: Standard deviation of per-entry noise.
        stream_name: Override for the registry stream name.

    Returns:
        ``(n, n)`` correlation matrix.
    """
    n = sum(block_sizes)
    if n <= 0:
        raise ValueError("block_sizes must sum to a positive integer")

    # Build block-membership vector: [0,0,1,1,1,2,...].
    membership = jnp.concatenate(
        [jnp.full((sz,), i, dtype=jnp.int32) for i, sz in enumerate(block_sizes)]
    )
    same_block = membership[:, None] == membership[None, :]
    base = jnp.where(same_block, intra, inter)

    key = registry.key(stream_name)
    noise = jitter * jax.random.normal(key, shape=(n, n), dtype=jnp.float64)
    noise = 0.5 * (noise + noise.T)
    m = base + noise
    m = m.at[jnp.diag_indices(n)].set(1.0)
    return _project_to_correlation(m)


def sample_correlation_from_config(
    registry: SeedRegistry,
    cfg,  # SyntheticMarketConfig (avoid circular import)
    block_sizes: tuple[int, ...] | None = None,
) -> Float[Array, "n n"]:
    """Dispatch on ``cfg.correlation_kind``.

    For ``"block"``, ``block_sizes`` must be supplied (and must sum to
    ``cfg.n_assets``).
    """
    kind = cfg.correlation_kind
    if kind == "identity":
        return jnp.eye(cfg.n_assets, dtype=jnp.float64)
    if kind == "random":
        return sample_correlation(
            registry,
            cfg.n_assets,
            min_corr=cfg.min_corr,
            max_corr=cfg.max_corr,
            kind="random",
        )
    if kind == "block":
        if block_sizes is None:
            raise ValueError(
                "correlation_kind='block' requires explicit block_sizes."
            )
        if sum(block_sizes) != cfg.n_assets:
            raise ValueError(
                f"block_sizes sum {sum(block_sizes)} != n_assets {cfg.n_assets}"
            )
        return block_correlation(registry, block_sizes)
    raise ValueError(f"Unknown correlation_kind: {kind!r}")


__all__ = [
    "sample_correlation",
    "block_correlation",
    "sample_correlation_from_config",
]
