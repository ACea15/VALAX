"""SABR Monte Carlo path generation via diffrax."""

import jax
import jax.numpy as jnp
import diffrax
from jaxtyping import Float
from jax import Array

from valax.models.sabr import SABRModel, SABRDrift, SABRDiffusion


def generate_sabr_paths(
    model: SABRModel,
    forward: Float[Array, ""],
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.Array,
) -> tuple[Float[Array, "n_paths n_steps+1"], Float[Array, "n_paths n_steps+1"]]:
    """Generate SABR paths (forward + vol) using diffrax Euler-Maruyama.

    Args:
        model: SABR model parameters.
        forward: Initial forward price.
        T: Time horizon.
        n_steps: Number of time steps.
        n_paths: Number of Monte Carlo paths.
        key: JAX PRNG key.

    Returns:
        (forward_paths, vol_paths) each of shape (n_paths, n_steps+1).
    """
    drift = SABRDrift(beta=model.beta)
    diffusion = SABRDiffusion(beta=model.beta, nu=model.nu, rho=model.rho)
    dt = T / n_steps
    ts = jnp.linspace(0.0, T, n_steps + 1)

    y0 = jnp.array([forward, model.alpha])

    def single_path(subkey):
        bm = diffrax.VirtualBrownianTree(
            t0=0.0, t1=T, tol=dt / 2, shape=(2,), key=subkey
        )
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift),
            diffrax.ControlTerm(diffusion, bm),
        )
        sol = diffrax.diffeqsolve(
            terms,
            diffrax.Euler(),
            t0=0.0,
            t1=T,
            dt0=dt,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        # sol.ys shape: (n_steps+1, 2)
        return sol.ys[:, 0], sol.ys[:, 1]

    keys = jax.random.split(key, n_paths)
    return jax.vmap(single_path)(keys)
