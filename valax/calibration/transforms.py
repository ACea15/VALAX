"""Parameter reparametrization for unconstrained optimization.

Maps constrained model parameters (e.g., alpha > 0, -1 < rho < 1)
to unconstrained R^n space and back, so JAX optimizers can operate
without explicit constraints.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array


class TransformSpec(NamedTuple):
    """A pair of inverse transforms for a single parameter."""

    to_unconstrained: Callable[[Float[Array, ""]], Float[Array, ""]]
    from_unconstrained: Callable[[Float[Array, ""]], Float[Array, ""]]


# ── Concrete transforms ─────────────────────────────────────────────

def positive() -> TransformSpec:
    """x > 0 via softplus / inverse-softplus."""
    def _to_unconstrained(x):
        # Numerically stable inverse softplus: log(exp(x) - 1)
        return jnp.where(x > 20.0, x, jnp.log(jnp.expm1(x)))

    return TransformSpec(
        to_unconstrained=_to_unconstrained,
        from_unconstrained=jax.nn.softplus,
    )


def bounded(lo: float, hi: float) -> TransformSpec:
    """lo < x < hi via scaled sigmoid / logit."""
    def _to_unconstrained(x):
        # Logit of normalized value
        t = (x - lo) / (hi - lo)
        t = jnp.clip(t, 1e-7, 1.0 - 1e-7)
        return jnp.log(t / (1.0 - t))

    def _from_unconstrained(y):
        return lo + (hi - lo) * jax.nn.sigmoid(y)

    return TransformSpec(
        to_unconstrained=_to_unconstrained,
        from_unconstrained=_from_unconstrained,
    )


def unit_interval() -> TransformSpec:
    """0 <= x <= 1 via sigmoid / logit."""
    return bounded(0.0, 1.0)


def correlation() -> TransformSpec:
    """-1 < x < 1 via tanh / arctanh."""
    def _to_unconstrained(x):
        x = jnp.clip(x, -1.0 + 1e-7, 1.0 - 1e-7)
        return jnp.arctanh(x)

    return TransformSpec(
        to_unconstrained=_to_unconstrained,
        from_unconstrained=jnp.tanh,
    )


# ── Model-level transform specs ─────────────────────────────────────

SABR_TRANSFORMS: dict[str, TransformSpec] = {
    "alpha": positive(),
    "beta": unit_interval(),
    "rho": correlation(),
    "nu": positive(),
}

HESTON_TRANSFORMS: dict[str, TransformSpec] = {
    "v0": positive(),
    "kappa": positive(),
    "theta": positive(),
    "xi": positive(),
    "rho": correlation(),
}


# ── Model ↔ unconstrained conversion ────────────────────────────────

def model_to_unconstrained(
    model: eqx.Module,
    transforms: dict[str, TransformSpec],
) -> dict[str, Float[Array, ""]]:
    """Convert model parameters to an unconstrained dict.

    Only fields present in `transforms` are converted; other fields
    are assumed fixed and will be taken from the template model during
    reconstruction.
    """
    raw = {}
    for name, spec in transforms.items():
        raw[name] = spec.to_unconstrained(getattr(model, name))
    return raw


def unconstrained_to_model(
    raw: dict[str, Float[Array, ""]],
    transforms: dict[str, TransformSpec],
    template: eqx.Module,
) -> eqx.Module:
    """Reconstruct a model from unconstrained params + a template.

    Fields in `raw` are inverse-transformed; all other fields are
    copied from `template` (including static fields like `is_call`).
    """
    model = template
    for name, spec in transforms.items():
        value = spec.from_unconstrained(raw[name])
        model = eqx.tree_at(lambda m, n=name: getattr(m, n), model, value)
    return model
