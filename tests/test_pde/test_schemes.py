"""Tests for the 1-D time-stepping scheme (``theta_for_scheme``, ``solve_backward_1d``).

Guards two things:

- the 1-D/2-D split: the 2-D ADI schemes (Douglas, Craig-Sneyd, HV) must be
  rejected by the 1-D theta-scheme solver rather than silently degrading to
  Crank-Nicolson;
- the **time direction** in which the Dirichlet boundary data is sampled (see
  ``TestBoundaryTimeDirection``).
"""

import jax.numpy as jnp
import pytest

from valax.pricing.pde.boundary import Boundary1D
from valax.pricing.pde.config import Scheme
from valax.pricing.pde.grids import uniform_linear_grid
from valax.pricing.pde.operators import build_operator_1d
from valax.pricing.pde.schemes import solve_backward_1d, theta_for_scheme


class TestSchemeIsAdi:
    @pytest.mark.parametrize(
        "scheme", [Scheme.DOUGLAS, Scheme.CRAIG_SNEYD, Scheme.HV]
    )
    def test_adi_schemes_flagged(self, scheme):
        assert scheme.is_adi() is True

    @pytest.mark.parametrize(
        "scheme", [Scheme.IMPLICIT, Scheme.CRANK_NICOLSON]
    )
    def test_one_dimensional_schemes_not_flagged(self, scheme):
        assert scheme.is_adi() is False


class TestThetaForScheme:
    def test_implicit_is_one(self):
        assert theta_for_scheme(Scheme.IMPLICIT) == 1.0

    def test_crank_nicolson_is_half(self):
        assert theta_for_scheme(Scheme.CRANK_NICOLSON) == 0.5

    @pytest.mark.parametrize(
        "scheme", [Scheme.DOUGLAS, Scheme.CRAIG_SNEYD, Scheme.HV]
    )
    def test_adi_schemes_rejected(self, scheme):
        # A 2-D ADI scheme has no 1-D theta interpretation: it must raise at
        # the user boundary rather than silently degrade to Crank-Nicolson.
        with pytest.raises(ValueError, match="ADI"):
            theta_for_scheme(scheme)


# ── Boundary sampling time direction ─────────────────────────────────
#
# Regression guard. The stepper marches backward from expiry via ``lax.scan``,
# so the level entering step ``m`` carries time-remaining ``m*dt`` (``m = 0``
# is the terminal payoff, tau = 0) and the level being solved carries
# ``(m+1)*dt``. A previous version passed ``(n_time - m)*dt`` and
# ``(n_time - m - 1)*dt`` — exactly time-*reversed* — so every time-dependent
# Dirichlet boundary was evaluated with the wrong discount factor. The error
# was masked at the default 4-sigma grid width but grew by three orders of
# magnitude on narrower grids.
#
# The probe below is the sharpest possible: a *spatially constant* solution.
# Take terminal data ``V(x, 0) = C`` and pure discounting, whose solution
# ``V(x, tau) = C exp(-r tau)`` has no spatial variation at all. The discrete
# operator annihilates a constant field exactly (``lower + diag + upper = -r``
# on every row), so if the ghost values injected at the two edge rows are taken
# at the same time level as the interior, the field stays constant in ``x`` to
# machine precision. Sampling them at the mirrored tau makes the edge rows
# inconsistent with the interior and spatial structure appears immediately.
#
# To make the identity exact rather than O(dt^2), the boundary is fed the
# *discrete* theta-scheme decay (the same recursion the interior obeys) rather
# than the continuous ``exp(-r tau)``: feeding the exact exponential leaves a
# genuine ~1e-5 time-discretisation mismatch between the analytic edge and the
# discretely-stepped interior, which would blunt the test.

class TestBoundaryTimeDirection:
    RATE = 0.05
    CONST = 3.0
    EXPIRY = 2.0
    N_TIME = 40

    def _step_factors(self, *, theta, rannacher_steps):
        """Per-step theta-scheme multipliers for ``dV/dtau = -r V``."""
        dt = self.EXPIRY / self.N_TIME
        factors = []
        for m in range(self.N_TIME):
            th = 1.0 if m < rannacher_steps else theta
            factors.append(
                (1.0 - (1.0 - th) * dt * self.RATE) / (1.0 + th * dt * self.RATE)
            )
        return jnp.asarray(factors)

    def _discrete_levels(self, *, theta, rannacher_steps):
        """Value at each time level: ``levels[m]`` holds time-remaining ``m*dt``."""
        factors = self._step_factors(theta=theta, rannacher_steps=rannacher_steps)
        return self.CONST * jnp.concatenate(
            [jnp.ones(1), jnp.cumprod(factors)]
        )

    def _solve(self, *, theta, rannacher_steps=0):
        dt = self.EXPIRY / self.N_TIME
        levels = self._discrete_levels(theta=theta, rannacher_steps=rannacher_steps)
        level_taus = jnp.arange(self.N_TIME + 1) * dt

        # Queried only at exact multiples of dt, where it returns the nodal
        # value; the interpolation never actually interpolates.
        def bc(tau):
            return jnp.interp(tau, level_taus, levels)

        grid = uniform_linear_grid(jnp.array(-1.0), jnp.array(1.0), n=16)
        operator = build_operator_1d(
            grid,
            drift=jnp.array(0.3),
            diffusion=jnp.array(0.04),
            discount=jnp.array(self.RATE),
        )
        terminal = jnp.full(grid.n, self.CONST)
        values = solve_backward_1d(
            operator,
            Boundary1D(bc, bc),
            terminal,
            expiry=jnp.array(self.EXPIRY),
            n_time=self.N_TIME,
            theta=theta,
            rannacher_steps=rannacher_steps,
        )
        return values, float(levels[-1])

    @pytest.mark.parametrize("theta,rannacher", [(0.5, 0), (1.0, 0), (0.5, 2)])
    def test_constant_solution_stays_constant(self, theta, rannacher):
        values, _ = self._solve(theta=theta, rannacher_steps=rannacher)
        spread = float(jnp.max(values) - jnp.min(values))
        assert spread < 1e-12, (
            f"boundary/interior time levels disagree: spatial spread {spread:.3e}"
        )

    @pytest.mark.parametrize("theta,rannacher", [(0.5, 0), (1.0, 0), (0.5, 2)])
    def test_matches_theta_scheme_ode(self, theta, rannacher):
        values, expected = self._solve(theta=theta, rannacher_steps=rannacher)
        assert float(jnp.max(jnp.abs(values - expected))) < 1e-12

    def test_solution_decays_over_the_full_horizon(self):
        """Guards orientation: the returned level is ``t = 0``, i.e. tau = T."""
        values, _ = self._solve(theta=1.0)
        assert float(values[0]) == pytest.approx(
            self.CONST * float(jnp.exp(-self.RATE * self.EXPIRY)), rel=1e-3
        )


# ── Discrete-event seam ──────────────────────────────────────────────
#
# ``event_fn`` is how contractual events that are *not* continuous — coupon
# payments, Bermudan/callable exercise — enter the backward sweep. The whole
# correctness of the Hull-White callable-bond recipes rests on the meaning of
# the level index handed to the hook, so it is pinned here directly rather than
# only implicitly through a price comparison.

class TestEventSeam:
    RATE = 0.04
    EXPIRY = 1.0
    N_TIME = 20

    def _solve(self, *, terminal_value, event_fn, theta=1.0):
        grid = uniform_linear_grid(jnp.array(-1.0), jnp.array(1.0), n=8)
        # Pure discounting: no drift, no diffusion, so every node evolves
        # independently and the answer is exactly the discount recursion.
        operator = build_operator_1d(
            grid,
            drift=jnp.array(0.0),
            diffusion=jnp.array(0.0),
            discount=jnp.array(self.RATE),
        )
        boundary = Boundary1D(
            lambda tau: jnp.zeros_like(tau), lambda tau: jnp.zeros_like(tau)
        )
        return solve_backward_1d(
            operator,
            boundary,
            jnp.full(grid.n, terminal_value),
            expiry=jnp.array(self.EXPIRY),
            n_time=self.N_TIME,
            theta=theta,
            event_fn=event_fn,
        )

    def _discount_factor(self, n_steps, theta=1.0):
        dt = self.EXPIRY / self.N_TIME
        return ((1.0 - (1.0 - theta) * dt * self.RATE)
                / (1.0 + theta * dt * self.RATE)) ** n_steps

    def test_none_is_a_no_op(self):
        """Omitting the hook must leave results bit-identical."""
        without = self._solve(terminal_value=1.0, event_fn=None)
        identity = self._solve(terminal_value=1.0, event_fn=lambda level, v: v)
        assert jnp.array_equal(without, identity)

    @pytest.mark.parametrize("pay_level", [0, 1, 7, 19])
    def test_level_index_is_the_forward_time_level(self, pay_level):
        """A unit cashflow injected at level ``k`` discounts over ``k`` steps.

        This is the contract the coupon/exercise recipes rely on: ``level``
        counts forward time, ``0`` at ``t = 0`` and ``n_time`` at expiry.
        """
        def event_fn(level, values):
            return values + jnp.where(level == pay_level, 1.0, 0.0)

        values = self._solve(terminal_value=0.0, event_fn=event_fn)
        expected = self._discount_factor(pay_level)
        assert float(values[0]) == pytest.approx(expected, rel=1e-12)

    def test_hook_runs_once_per_step_and_never_at_the_terminal_level(self):
        """Levels ``n_time-1 ... 0`` are visited; ``n_time`` is the caller's
        terminal condition and must not be re-triggered."""
        def event_fn(level, values):
            return values + jnp.where(level == self.N_TIME, 100.0, 0.0)

        values = self._solve(terminal_value=0.0, event_fn=event_fn)
        assert float(jnp.max(jnp.abs(values))) == 0.0

    def test_counts_every_intermediate_level_exactly_once(self):
        counter = self._solve(
            terminal_value=0.0, event_fn=lambda level, v: v + 1.0
        )
        # No discounting of the count itself would be wrong — each unit added
        # at level k is discounted over k steps — so compare against the sum.
        expected = sum(self._discount_factor(k) for k in range(self.N_TIME))
        assert float(counter[0]) == pytest.approx(expected, rel=1e-12)

    def test_projection_hook_enforces_an_obstacle(self):
        """The Bermudan/callable use-case: a cap applied at one level."""
        cap = 0.5

        def event_fn(level, values):
            return jnp.where(level == 10, jnp.minimum(values, cap), values)

        values = self._solve(terminal_value=1.0, event_fn=event_fn)
        # Value at level 10 is min(1 * df(10), cap) = cap, then discounted 10.
        expected = min(self._discount_factor(self.N_TIME - 10), cap) * \
            self._discount_factor(10)
        assert float(values[0]) == pytest.approx(expected, rel=1e-12)
