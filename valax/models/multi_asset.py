r"""Multi-asset geometric Brownian motion model.

Extends the single-asset :class:`~valax.models.black_scholes.BlackScholesModel`
to :math:`n` correlated underlyings under a single risk-neutral measure:

.. math::

    dS_i(t) = (r - q_i)\,S_i(t)\,dt + \sigma_i\,S_i(t)\,dW_i(t)

.. math::

    \langle dW_i, dW_j\rangle = \rho_{ij}\,dt

The correlation matrix :math:`\\rho` must be symmetric with unit diagonal
and positive semi-definite. The path generator in
:func:`valax.pricing.mc.generate_correlated_gbm_paths` uses its Cholesky
factor to produce correlated Brownian increments.

This model unlocks multi-asset MC for:

- :class:`~valax.instruments.options.SpreadOption` (validation of
  Margrabe / Kirk closed forms; pricing for path-dependent spread
  exotics)
- :class:`~valax.instruments.options.WorstOfBasketOption`
- Correlated autocallables and other basket structured products
  (on the roadmap)

References:
    Glasserman (2004), *Monte Carlo Methods in Financial Engineering*,
    ch. 3.2.3 (Correlated Brownian motions).
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


class MultiAssetGBMModel(eqx.Module):
    """N-asset geometric Brownian motion under a single risk-neutral measure.

    Attributes:
        vols: Per-asset volatilities :math:`\\sigma_i`, shape ``(n_assets,)``.
        rate: Risk-free rate :math:`r` (scalar) used for both drift and
            discounting.
        dividends: Per-asset continuous dividend yields :math:`q_i`,
            shape ``(n_assets,)``. Set to zeros for non-dividend-paying
            assets.
        correlation: Instantaneous correlation matrix of the Brownian
            motions, shape ``(n_assets, n_assets)``. Must be symmetric
            with unit diagonal and positive semi-definite.

    Notes:
        All fields are JAX arrays and participate in autodiff. Taking
        ``jax.grad`` of a price w.r.t. ``correlation`` gives the
        correlation Greeks directly; w.r.t. ``vols`` gives per-asset
        vegas; w.r.t. ``dividends`` and ``rate`` gives the dividend
        and rate sensitivities.
    """

    vols: Float[Array, " n_assets"]
    rate: Float[Array, ""]
    dividends: Float[Array, " n_assets"]
    correlation: Float[Array, "n_assets n_assets"]

    @property
    def n_assets(self) -> int:
        """Number of underlyings (static at trace time)."""
        return self.vols.shape[0]


def validate_correlation(
    correlation: Float[Array, "n n"],
    tol: float = 1e-6,
) -> Float[Array, ""]:
    """Return the minimum eigenvalue of ``correlation`` for PSD checking.

    Does *not* raise inside JIT-traced code — returns a scalar you can
    inspect concretely or assert on outside JAX. At construction time
    (e.g. in user scripts), compare the returned value against ``-tol``
    and raise if it's below.

    The function also checks symmetry and unit diagonal, setting the
    returned value to ``-inf`` if either fails (an obviously invalid
    eigenvalue).

    Args:
        correlation: Candidate correlation matrix, shape ``(n, n)``.
        tol: Tolerance for PSD check (numerical slack).

    Returns:
        Minimum eigenvalue of the correlation matrix. Must be
        ``>= -tol`` for the matrix to be a valid correlation matrix.

    Example:
        >>> import jax.numpy as jnp
        >>> C = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        >>> float(validate_correlation(C)) >= 0.0
        True
    """
    n = correlation.shape[0]

    # Symmetry check: max absolute asymmetry.
    sym_err = jnp.max(jnp.abs(correlation - correlation.T))

    # Unit diagonal check.
    diag_err = jnp.max(jnp.abs(jnp.diag(correlation) - 1.0))

    # Minimum eigenvalue.
    min_eig = jnp.min(jnp.linalg.eigvalsh(correlation))

    # If structure checks fail, return -inf so the PSD assertion fails.
    structure_ok = (sym_err < tol) & (diag_err < tol)
    return jnp.where(structure_ok, min_eig, -jnp.inf)
