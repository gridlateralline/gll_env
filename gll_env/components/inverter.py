# Copyright 2026 ewz - Zurich Municipal Electric Utility.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import field
from functools import cached_property

import chex
import jax.numpy as jnp
import jax.random as jr

from gll_env.algorithms.radial_projection import RadialProjection
from gll_env.components.battery import (
    BatteryDynamics,
    BatteryObservation,
    BatteryState,
)
from gll_env.components.day_time import DaytimeDynamics, DaytimeState
from gll_env.components.solar import SolarDynamics, SolarObservation, SolarState
from gll_env.types import ActionConstraints
from gll_env.utils import safe_normalize


@chex.dataclass(frozen=True)
class InverterState:
    # Power flow direction convention:
    # positive = generation (out of inverter),
    # negative = consumption (into inverter)
    s_inv_realized_kvah: chex.Array  # (num_inv,) complex64 -- past interval, realized
    s_inv_request_constraint: ActionConstraints

    battery_state: BatteryState
    solar_state: SolarState
    time_state: DaytimeState


@chex.dataclass(frozen=True)
class InverterObservation:
    p_inv_realized: chex.Array  # (num_inv,) float32 in [-1, 1]
    q_inv_realized: chex.Array  # (num_inv,) float32 in [-1, 1]
    p_inv_min: chex.Array  # (num_inv,) float32 in [-1, 0]
    p_inv_max: chex.Array  # (num_inv,) float32 in [0, 1]
    battery_observation: BatteryObservation
    solar_observation: SolarObservation


@chex.dataclass(frozen=True)
class InverterDynamics:
    s_inv_max_kva: chex.Array  # (num_inv,) float32 -- inverter apparent-power rating (nameplate)

    battery_dynamics: BatteryDynamics
    solar_dynamics: SolarDynamics
    time: DaytimeDynamics = field(default_factory=DaytimeDynamics)
    projection: RadialProjection = field(default_factory=RadialProjection)

    @cached_property
    def num_inv(self) -> int:
        return jnp.atleast_1d(self.s_inv_max_kva).shape[0]

    @cached_property
    def s_inv_max_kvah(self) -> chex.Array:
        return self.s_inv_max_kva * self.time.step_duration_h

    def __post_init__(self) -> None:
        chex.assert_shape(self.s_inv_max_kva, (self.num_inv,))
        chex.assert_type(self.s_inv_max_kva, jnp.float32)

        chex.assert_equal(self.num_inv, self.battery_dynamics.num_bat)
        chex.assert_equal(self.num_inv, self.solar_dynamics.num_sol)

        chex.assert_equal(self.time.n_steps_per_day, self.battery_dynamics.time.n_steps_per_day)
        chex.assert_equal(self.time.n_steps_per_day, self.solar_dynamics.time.n_steps_per_day)
        object.__setattr__(self, "s_inv_max_kva", jnp.maximum(self.s_inv_max_kva, 0.0))

    def _inv_bounds_kwh(
        self,
        sol_min_kwh: chex.Array,
        sol_max_kwh: chex.Array,
        bat_min_kwh: chex.Array,
        bat_max_kwh: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
        p_inv_min_kwh = jnp.maximum(sol_min_kwh + bat_min_kwh, jnp.negative(self.s_inv_max_kvah))
        p_inv_max_kwh = jnp.minimum(sol_max_kwh + bat_max_kwh, self.s_inv_max_kvah)
        return p_inv_min_kwh, p_inv_max_kwh

    def _new_request_constraint(
        self, p_inv_min_kwh: chex.Array, p_inv_max_kwh: chex.Array
    ) -> ActionConstraints:
        """The feasible s_inv_request range for the coming interval:

            halfspace 0: [1, 0] @ [p, q] <= p_inv_max_kwh   (p <= max)
            halfspace 1: [-1, 0] @ [p, q] <= -p_inv_min_kwh (p >= min)
            ball 0: |[p, q]| <= s_inv_max_kvah

        Unlike Battery/Solar's pure box, this constraint genuinely needs the
        ball: it's what actually couples p and q together (the halfspaces
        alone say nothing about q). Same halfspace ordering convention as
        Battery/Solar (0=upper, 1=lower) -- request_bounds relies on it.
        """
        zeros = jnp.zeros((self.num_inv,), dtype=jnp.float32)
        ones = jnp.ones((self.num_inv,), dtype=jnp.float32)
        halfspace_a = jnp.stack(
            [
                jnp.stack([ones, zeros], axis=-1),
                jnp.stack([jnp.negative(ones), zeros], axis=-1),
            ],
            axis=1,
        )  # (num_inv, 2, 2)
        halfspace_b = jnp.stack(
            [p_inv_max_kwh, jnp.negative(p_inv_min_kwh)], axis=1
        )  # (num_inv, 2)

        ball_center = jnp.zeros((self.num_inv, 1, 2), dtype=jnp.float32)
        ball_radius = jnp.asarray(self.s_inv_max_kvah)[:, None]  # (num_inv, 1)

        return ActionConstraints(
            halfspace_a=halfspace_a,
            halfspace_b=halfspace_b,
            ball_center=ball_center,
            ball_radius=ball_radius,
        )

    def request_bounds(self, constraint: ActionConstraints) -> tuple[chex.Array, chex.Array]:
        """Unpack the real-power (p) bound from a request constraint --
        just the halfspace half of it; the ball (apparent-power circle)
        isn't representable as a simple (min, max) interval, so this
        returns the same thing Battery/Solar's own request_bounds returns:
        the box's edges, matching the halfspace ordering fixed by
        _new_request_constraint.
        """
        p_inv_max_kwh = jnp.asarray(constraint.halfspace_b)[:, 0]
        p_inv_min_kwh = jnp.negative(jnp.asarray(constraint.halfspace_b)[:, 1])
        return p_inv_min_kwh, p_inv_max_kwh

    def observation(self, state: InverterState) -> InverterObservation:
        p_inv_min_kwh, p_inv_max_kwh = self.request_bounds(state.s_inv_request_constraint)
        return InverterObservation(
            p_inv_realized=safe_normalize(jnp.real(state.s_inv_realized_kvah), self.s_inv_max_kvah),
            q_inv_realized=safe_normalize(jnp.imag(state.s_inv_realized_kvah), self.s_inv_max_kvah),
            p_inv_min=safe_normalize(p_inv_min_kwh, self.s_inv_max_kvah),
            p_inv_max=safe_normalize(p_inv_max_kwh, self.s_inv_max_kvah),
            solar_observation=self.solar_dynamics.observation(state.solar_state),
            battery_observation=self.battery_dynamics.observation(state.battery_state),
        )

    def reset(self, key: chex.PRNGKey, time_state: DaytimeState | None = None) -> InverterState:
        if time_state is None:
            key, time_key = jr.split(key)
            time_state = self.time.reset(time_key)

        battery_key, solar_key, p_key, q_key = jr.split(key, 4)
        time_state_prev = self.time.previous(time_state)
        battery_state_prev = self.battery_dynamics.reset(battery_key)
        solar_state_prev = self.solar_dynamics.reset(solar_key, time_state_prev)

        sol_min_kwh_prev, sol_max_kwh_prev = self.solar_dynamics.request_bounds(
            solar_state_prev.sol_request_constraint
        )
        bat_min_kwh_prev, bat_max_kwh_prev = self.battery_dynamics.request_bounds(
            battery_state_prev.bat_request_constraint
        )
        p_inv_min_kwh_prev, p_inv_max_kwh_prev = self._inv_bounds_kwh(
            sol_min_kwh=sol_min_kwh_prev,
            sol_max_kwh=sol_max_kwh_prev,
            bat_min_kwh=bat_min_kwh_prev,
            bat_max_kwh=bat_max_kwh_prev,
        )
        request_constraint_prev = self._new_request_constraint(
            p_inv_min_kwh_prev, p_inv_max_kwh_prev
        )

        state_prev = InverterState(
            s_inv_realized_kvah=jnp.zeros((self.num_inv,), dtype=jnp.complex64),  # unused
            s_inv_request_constraint=request_constraint_prev,
            battery_state=battery_state_prev,
            solar_state=solar_state_prev,
            time_state=time_state_prev,
        )

        # Sample p within the TRUE feasible range -- box ∩ circle -- then q
        # within the exact circle slice at that p. Both projections inside
        # step() become no-ops, so nothing piles up at a boundary the way it
        # would if p were sampled over the box alone, or q over a fixed wide
        # range. p_inv_min/max_kwh_prev are already intersected with the
        # inverter's own rating (see _inv_bounds_kwh), so they can be used
        # directly here.
        p_inv_request_kwh_prev = jr.uniform(
            key=p_key,
            shape=p_inv_min_kwh_prev.shape,
            minval=p_inv_min_kwh_prev,
            maxval=p_inv_max_kwh_prev,
        )
        q_inv_max_kvarh_prev = jnp.sqrt(
            jnp.maximum(self.s_inv_max_kvah**2 - p_inv_request_kwh_prev**2, 0.0)
        )
        q_inv_request_kvarh_prev = jr.uniform(
            key=q_key,
            shape=q_inv_max_kvarh_prev.shape,
            minval=jnp.negative(q_inv_max_kvarh_prev),
            maxval=q_inv_max_kvarh_prev,
        )
        s_inv_request_prev = jnp.stack([p_inv_request_kwh_prev, q_inv_request_kvarh_prev], axis=-1)

        return self.step(state_prev, s_inv_request_prev, next_time_state=time_state)

    def step(
        self,
        state: InverterState,
        s_inv_request: chex.Array,
        next_time_state: DaytimeState | None = None,
    ) -> InverterState:
        """Advance one interval given a physical request.

        Parameters
        ----------
        s_inv_request : chex.Array
            Shape (num_inv, 2) float32: [p_inv_kwh, q_inv_kvarh] bundled
            into one array to match the action_dim=2 used throughout
            ActionConstraints -- see s_inv_realized_kvah/s_inv_request_constraint
            for the naming rationale.

        Feasibility is enforced via a real projection.solve() call against
        state.s_inv_request_constraint, not a cheap approximation.
        InverterDynamics is a standalone, reusable component (called
        directly in tests/notebooks without EnvironmentModel involved at
        all), so it shouldn't just assume an upstream caller already solved
        feasibility -- it should guarantee it itself. This costs nothing
        extra in the common case: projection.solve() short-circuits as a
        no-op whenever the request already satisfies every constraint,
        which -- given EnvironmentModel's own projection already solves
        against this exact same constraint (in normalized form) before
        scaling and dispatching down here -- is true almost always. The
        convergence result itself isn't surfaced as InverterState fields
        (unlike Prosumer/Environment's valid): Inverter's own constraint is
        provably always origin-feasible (the halfspaces always straddle 0
        by construction -- see _inv_bounds_kwh -- and the ball is centered
        at the origin), so non-convergence here could only ever indicate a
        malformed constraint, not a genuine external assumption the way
        Prosumer's grid-ball sizing is.

        s_inv_request is a request for the inverter's own OUTFLOWING power
        -- it says nothing about how that flow should be split internally
        between solar and battery. This is where that split happens, and it
        maximizes solar's own contribution first (never curtailing solar
        just because the external request doesn't strictly need it):
        battery only ever covers the residual, whether that means
        discharging to cover a shortfall (solar can't reach p_inv alone) or
        charging to absorb a surplus (solar exceeds p_inv, e.g. p_inv == 0
        with the sun out). Solar is only curtailed once battery is ALSO
        saturated and has nowhere left to put the excess -- a true last
        resort, not the default. Mirrors how a real MPPT-plus-battery
        inverter behaves, and only requires solving a tiny linear program
        in closed form (see the derivation below), not an iterative search.
        """
        chex.assert_shape(s_inv_request, (self.num_inv, 2))
        chex.assert_type(s_inv_request, jnp.float32)

        s_inv_request, _ = self.projection.solve(s_inv_request, state.s_inv_request_constraint)
        p_inv_kwh = jnp.asarray(s_inv_request)[:, 0]
        q_inv_kvarh = jnp.asarray(s_inv_request)[:, 1]
        next_s_inv_kvah = p_inv_kwh + 1j * q_inv_kvarh

        # Internal dispatch: split the now-feasible p_inv_kwh between solar
        # and battery by maximizing solar's contribution, i.e. solve
        #     maximize sol_realized
        #     s.t.     sol_realized + bat_realized == p_inv_kwh
        #              sol_realized in [sol_min_kwh, sol_max_kwh]
        #              bat_realized in [bat_min_kwh, bat_max_kwh]
        # Substituting bat_realized = p_inv_kwh - sol_realized into battery's
        # own bound gives sol_realized >= p_inv_kwh - bat_max_kwh and
        # sol_realized <= p_inv_kwh - bat_min_kwh; combined with solar's own
        # [sol_min_kwh, sol_max_kwh], the maximizing choice is the upper end
        # of that intersection: clip(p_inv_kwh - bat_min_kwh, sol_min_kwh,
        # sol_max_kwh). Both clips are provably no-ops given a feasible
        # p_inv_kwh (sol_min_kwh + bat_min_kwh <= p_inv_kwh <= sol_max_kwh +
        # bat_max_kwh is exactly what _inv_bounds_kwh/_new_request_constraint
        # already guarantee), so sol_realized + bat_realized == p_inv_kwh
        # holds exactly and s_inv_kvah above already reflects it -- no need
        # to recompute s_inv_kvah from the leaves afterward.
        sol_min_kwh, sol_max_kwh = self.solar_dynamics.request_bounds(
            state.solar_state.sol_request_constraint
        )
        bat_min_kwh, _ = self.battery_dynamics.request_bounds(
            state.battery_state.bat_request_constraint
        )
        sol_request_kwh = jnp.clip(p_inv_kwh - bat_min_kwh, sol_min_kwh, sol_max_kwh)

        next_time_state = (
            self.time.step(state.time_state) if next_time_state is None else next_time_state
        )
        next_solar_state = self.solar_dynamics.step(
            state.solar_state,
            sol_request_kwh=sol_request_kwh,
            next_time_state=next_time_state,
        )
        next_battery_state = self.battery_dynamics.step(
            state.battery_state,
            bat_request_kwh=p_inv_kwh
            - next_solar_state.sol_realized_kwh,  # residual: shortfall (discharge) surplus (charge)
        )

        sol_min_kwh, sol_max_kwh = self.solar_dynamics.request_bounds(
            next_solar_state.sol_request_constraint
        )
        bat_min_kwh, bat_max_kwh = self.battery_dynamics.request_bounds(
            next_battery_state.bat_request_constraint
        )
        p_inv_min_kwh, p_inv_max_kwh = self._inv_bounds_kwh(
            sol_min_kwh=sol_min_kwh,
            sol_max_kwh=sol_max_kwh,
            bat_min_kwh=bat_min_kwh,
            bat_max_kwh=bat_max_kwh,
        )
        next_request_constraint = self._new_request_constraint(p_inv_min_kwh, p_inv_max_kwh)

        return InverterState(
            s_inv_realized_kvah=next_s_inv_kvah,
            s_inv_request_constraint=next_request_constraint,
            battery_state=next_battery_state,
            solar_state=next_solar_state,
            time_state=next_time_state,
        )
