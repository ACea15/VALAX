"""SDE path generation via diffrax."""

import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.models.black_scholes import BlackScholesModel, GBMDrift, GBMDiffusion
from valax.models.heston import HestonModel, HestonDrift, HestonDiffusion


def generate_gbm_paths(
    model: BlackScholesModel,
    spot: Float[Array, ""],
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.Array,
) -> Float[Array, "n_paths n_steps+1"]:
    """Generate GBM paths using diffrax Euler-Maruyama.

    Returns array of shape (n_paths, n_steps+1) including initial spot.
    """
    mu = model.rate - model.dividend
    drift = GBMDrift(mu=mu)
    diffusion = GBMDiffusion(sigma=model.vol)
    dt = T / n_steps
    ts = jnp.linspace(0.0, T, n_steps + 1)

    def single_path(subkey):
        bm = diffrax.VirtualBrownianTree(
            t0=0.0, t1=T, tol=dt / 2, shape=(), key=subkey
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
            y0=spot,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys  # shape (n_steps+1,)

    keys = jax.random.split(key, n_paths)
    return jax.vmap(single_path)(keys)


def generate_heston_paths(
    model: HestonModel,
    spot: Float[Array, ""],
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.Array,
) -> tuple[Float[Array, "n_paths n_steps+1"], Float[Array, "n_paths n_steps+1"]]:
    """Generate Heston paths (spot + variance) using diffrax.

    Works in log-spot space for numerical stability.

    Returns:
        (spot_paths, var_paths) each of shape (n_paths, n_steps+1).
    """
    drift = HestonDrift(
        rate=model.rate, dividend=model.dividend,
        kappa=model.kappa, theta=model.theta,
    )
    diffusion = HestonDiffusion(xi=model.xi, rho=model.rho)
    dt = T / n_steps
    ts = jnp.linspace(0.0, T, n_steps + 1)

    y0 = jnp.array([jnp.log(spot), model.v0])

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
        log_s = sol.ys[:, 0]
        v = sol.ys[:, 1]
        return jnp.exp(log_s), v

    keys = jax.random.split(key, n_paths)
    return jax.vmap(single_path)(keys)
