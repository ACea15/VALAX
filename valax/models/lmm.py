"""LIBOR Market Model (BGM model).

Models N simply-compounded forward rates F_i(t) = F(t; T_i, T_{i+1})
under the spot LIBOR measure. Simulated in log-space for guaranteed
positivity via log-Euler discretization.

SDE under the spot measure:

    dF_i = mu_i dt + sigma_i(t) F_i sum_j L_ij dW_j

    mu_i = sigma_i F_i * sum_{j=eta(t)}^{i} [
        tau_j sigma_j rho_ij F_j / (1 + tau_j F_j)
    ]

where L is the Cholesky (or PCA) factor of the correlation matrix,
and eta(t) is the index of the next alive forward at time t.

References:
    Brace, Gatarek, Musiela (1997), "The market model of interest rate dynamics".
    Rebonato (2002), "Modern Pricing of Interest-Rate Derivatives".
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction


# ── Volatility structures ────────────────────────────────────────────

class PiecewiseConstantVol(eqx.Module):
    """Piecewise-constant volatility: sigma_i(t) = vols[i, k] for t in [T_{k-1}, T_k).

    The vol matrix is lower-triangular by convention: vols[i, k] is only
    meaningful for k <= i (forward i is dead after T_i).

    Attributes:
        vols: Volatility matrix, shape (N, N). vols[i, k] is the
              instantaneous vol of forward i during period k.
    """

    vols: Float[Array, "n_forwards n_periods"]

    def __call__(
        self,
        t: Float[Array, ""],
        tenor_times_fwd: Float[Array, " n_forwards"],
    ) -> Float[Array, " n_forwards"]:
        """Return sigma_i(t) for all forwards.

        Args:
            t: Current time (year fraction from reference date).
            tenor_times_fwd: Year fractions from reference to each T_i.

        Returns:
            Instantaneous volatility for each forward, shape (N,).
        """
        k = jnp.searchsorted(tenor_times_fwd, t, side="right")
        k = jnp.clip(k, 0, self.vols.shape[1] - 1)
        return self.vols[:, k]


class RebonatoVol(eqx.Module):
    """Rebonato abcd parametric volatility.

        sigma_i(t) = (a + b * (T_i - t)) * exp(-c * (T_i - t)) + d

    Parsimonious 4-parameter fit of the entire vol surface.

    Attributes:
        a, b, c, d: Rebonato parameters (scalars).
    """

    a: Float[Array, ""]
    b: Float[Array, ""]
    c: Float[Array, ""]
    d: Float[Array, ""]

    def __call__(
        self,
        t: Float[Array, ""],
        tenor_times_fwd: Float[Array, " n_forwards"],
    ) -> Float[Array, " n_forwards"]:
        tau = jnp.maximum(tenor_times_fwd - t, 0.0)
        return (self.a + self.b * tau) * jnp.exp(-self.c * tau) + self.d


# ── Correlation structures ───────────────────────────────────────────

class ExponentialCorrelation(eqx.Module):
    """Exponential decay correlation: rho_ij = exp(-beta * |T_i - T_j|).

    Attributes:
        beta: Decay parameter (scalar, > 0).
    """

    beta: Float[Array, ""]

    def matrix(
        self, tenor_times_fwd: Float[Array, " n_forwards"]
    ) -> Float[Array, "n_forwards n_forwards"]:
        diff = jnp.abs(tenor_times_fwd[:, None] - tenor_times_fwd[None, :])
        return jnp.exp(-self.beta * diff)


class TwoParameterCorrelation(eqx.Module):
    """Two-parameter correlation:
        rho_ij = rho_inf + (1 - rho_inf) * exp(-beta * |T_i - T_j|)

    Attributes:
        rho_inf: Long-range correlation floor (scalar, in [0, 1]).
        beta: Decay parameter (scalar, > 0).
    """

    rho_inf: Float[Array, ""]
    beta: Float[Array, ""]

    def matrix(
        self, tenor_times_fwd: Float[Array, " n_forwards"]
    ) -> Float[Array, "n_forwards n_forwards"]:
        diff = jnp.abs(tenor_times_fwd[:, None] - tenor_times_fwd[None, :])
        return self.rho_inf + (1.0 - self.rho_inf) * jnp.exp(-self.beta * diff)


# ── Factor loading ───────────────────────────────────────────────────

def compute_loading_matrix(
    corr_matrix: Float[Array, "N N"],
    n_factors: int | None = None,
) -> Float[Array, "N k"]:
    """Compute factor loading matrix from a correlation matrix.

    If n_factors is None, returns the Cholesky factor (N x N).
    If n_factors < N, returns the top-k PCA loading matrix (N x k),
    scaled so that L @ L.T approximates the correlation matrix.

    Args:
        corr_matrix: Positive-definite correlation matrix, shape (N, N).
        n_factors: Number of PCA factors. None = full rank Cholesky.

    Returns:
        Loading matrix L, shape (N, k).
    """
    if n_factors is None:
        return jnp.linalg.cholesky(corr_matrix)
    # PCA: eigendecomposition, keep top-k
    eigenvalues, eigenvectors = jnp.linalg.eigh(corr_matrix)
    # eigh returns ascending order; take last n_factors
    top_vals = eigenvalues[-n_factors:]
    top_vecs = eigenvectors[:, -n_factors:]
    return top_vecs * jnp.sqrt(jnp.maximum(top_vals, 0.0))[None, :]


# ── LMM Model ────────────────────────────────────────────────────────

class LMMModel(eqx.Module):
    """LIBOR Market Model (BGM) parameters.

    Stores the initial forward rates, tenor structure, volatility and
    correlation parameterizations, and precomputed factor loading matrix.

    Attributes:
        initial_forwards: Initial simply-compounded forward rates F_i(0), shape (N,).
        tenor_dates: Ordinal dates [T_0, T_1, ..., T_N], shape (N+1,).
                     Forward i covers [T_i, T_{i+1}].
        vol_structure: Volatility parameterization (callable eqx.Module).
        corr_structure: Correlation parameterization (eqx.Module with .matrix()).
        loading_matrix: Precomputed factor loading L, shape (N, k).
        accrual_fractions: Year fractions tau_i = yf(T_i, T_{i+1}), shape (N,).
        tenor_times: Year fractions from reference date to each T_i, shape (N+1,).
        n_factors: Number of Brownian factors (k <= N). None means full rank.
        measure: Pricing measure: "spot" (default).
        reference_date: Valuation date (ordinal).
        day_count: Day count convention.
    """

    initial_forwards: Float[Array, " n_forwards"]
    tenor_dates: Int[Array, " n_tenors"]
    vol_structure: eqx.Module
    corr_structure: eqx.Module
    loading_matrix: Float[Array, "n_forwards k"]
    accrual_fractions: Float[Array, " n_forwards"]
    tenor_times: Float[Array, " n_tenors"]
    initial_df: Float[Array, ""]  # DF(0, T_0) from curve to first tenor
    n_factors: int | None = eqx.field(static=True, default=None)
    measure: str = eqx.field(static=True, default="spot")
    reference_date: Int[Array, ""] = None
    day_count: str = eqx.field(static=True, default="act_360")


# ── Drift and Diffusion for diffrax ──────────────────────────────────

class LMMDrift(eqx.Module):
    """Drift of log-forward rates under the spot LIBOR measure.

    For log-forward x_i = log(F_i), the drift is:

        drift_i = [mu_i / F_i - 0.5 sigma_i^2 sum_j L_ij^2] * alive_i

    where mu_i is the spot-measure no-arbitrage drift:

        mu_i = sigma_i F_i * sum_{j=eta(t)}^{i} [
            tau_j sigma_j (L L^T)_{ij} F_j / (1 + tau_j F_j)
        ]

    Attributes:
        accrual_fractions: tau_i for each forward, shape (N,).
        loading_matrix: Factor loadings L, shape (N, k).
        tenor_times_fwd: Year fracs from ref to forward start dates T_i, shape (N,).
        vol_structure: Volatility parameterization.
    """

    accrual_fractions: Float[Array, " n_forwards"]
    loading_matrix: Float[Array, "n_forwards k"]
    tenor_times_fwd: Float[Array, " n_forwards"]
    vol_structure: eqx.Module

    def __call__(self, t, y, args):
        """Compute drift for all N log-forwards.

        Args:
            t: Current time (year fraction).
            y: Log-forward rates, shape (N,).
            args: Unused (diffrax signature).

        Returns:
            Drift vector, shape (N,).
        """
        N = y.shape[0]
        F = jnp.exp(y)
        tau = self.accrual_fractions
        sigma = self.vol_structure(t, self.tenor_times_fwd)
        L = self.loading_matrix

        # Alive mask: forward i is alive if t <= T_i
        alive = (t <= self.tenor_times_fwd).astype(y.dtype)

        # g_j = tau_j * sigma_j * F_j / (1 + tau_j * F_j)
        g = tau * sigma * F / (1.0 + tau * F)

        # eta(t): index of first alive forward
        eta = jnp.searchsorted(self.tenor_times_fwd, t, side="left")

        # Build mask[i, j] = (j >= eta) & (j <= i) — lower-triangular from eta
        j_idx = jnp.arange(N)
        i_idx = jnp.arange(N)
        mask = ((j_idx[None, :] >= eta) & (j_idx[None, :] <= i_idx[:, None]))
        mask = mask.astype(y.dtype)

        # Correlation via loading matrix: rho = L @ L^T
        corr = L @ L.T

        # drift_sum[i] = sum_j mask[i,j] * corr[i,j] * g[j]
        drift_sum = jnp.sum(mask * corr * g[None, :], axis=1)

        # mu_i = sigma_i * F_i * drift_sum_i
        mu = sigma * F * drift_sum

        # Log-forward drift: mu_i/F_i - 0.5 * sigma_i^2 * sum_j L_ij^2
        vol_sq = jnp.sum(L ** 2, axis=1)
        log_drift = mu / F - 0.5 * sigma ** 2 * vol_sq

        return log_drift * alive


class LMMDiffusion(eqx.Module):
    """Diffusion of log-forward rates.

    Returns an (N, k) matrix: sigma_i(t) * alive_i * L_i (row i of loading).
    When multiplied by the (k,) Brownian increment, gives the (N,) shock.

    Attributes:
        loading_matrix: Factor loadings L, shape (N, k).
        tenor_times_fwd: Year fracs from ref to T_i, shape (N,).
        vol_structure: Volatility parameterization.
    """

    loading_matrix: Float[Array, "n_forwards k"]
    tenor_times_fwd: Float[Array, " n_forwards"]
    vol_structure: eqx.Module

    def __call__(self, t, y, args):
        """Compute diffusion matrix.

        Args:
            t: Current time.
            y: Log-forward rates, shape (N,).
            args: Unused.

        Returns:
            Diffusion matrix, shape (N, k).
        """
        sigma = self.vol_structure(t, self.tenor_times_fwd)
        alive = (t <= self.tenor_times_fwd).astype(y.dtype)
        return (sigma * alive)[:, None] * self.loading_matrix


# ── Model construction ───────────────────────────────────────────────

def build_lmm_model(
    curve: DiscountCurve,
    tenor_dates: Int[Array, " n_tenors"],
    vol_structure: eqx.Module,
    corr_structure: eqx.Module,
    n_factors: int | None = None,
    measure: str = "spot",
) -> LMMModel:
    """Construct an LMMModel from a discount curve and tenor schedule.

    Extracts initial forward rates and accrual fractions from the curve,
    builds the correlation matrix and factor loading matrix.

    Args:
        curve: Discount curve for computing initial forward rates.
        tenor_dates: Ordinal dates [T_0, ..., T_N], shape (N+1,).
        vol_structure: Volatility parameterization (PiecewiseConstantVol or RebonatoVol).
        corr_structure: Correlation parameterization with a .matrix() method.
        n_factors: Number of PCA factors. None = full rank (Cholesky).
        measure: Pricing measure ("spot").

    Returns:
        Fully initialized LMMModel.
    """
    # Accrual fractions: tau_i = yf(T_i, T_{i+1})
    taus = year_fraction(tenor_dates[:-1], tenor_dates[1:], curve.day_count)

    # Initial forward rates from the curve: F_i(0) = (DF(T_i)/DF(T_{i+1}) - 1) / tau_i
    df_starts = curve(tenor_dates[:-1])
    df_ends = curve(tenor_dates[1:])
    initial_forwards = (df_starts / df_ends - 1.0) / taus

    # Tenor times in year fractions from reference date
    tenor_times = year_fraction(curve.reference_date, tenor_dates, curve.day_count)

    # DF from reference date to first tenor date
    initial_df = curve(tenor_dates[0])

    # Build correlation matrix and loading matrix
    tenor_times_fwd = tenor_times[:-1]
    corr_matrix = corr_structure.matrix(tenor_times_fwd)
    loading_matrix = compute_loading_matrix(corr_matrix, n_factors)

    return LMMModel(
        initial_forwards=initial_forwards,
        tenor_dates=tenor_dates,
        vol_structure=vol_structure,
        corr_structure=corr_structure,
        loading_matrix=loading_matrix,
        accrual_fractions=taus,
        tenor_times=tenor_times,
        initial_df=initial_df,
        n_factors=n_factors,
        measure=measure,
        reference_date=curve.reference_date,
        day_count=curve.day_count,
    )
