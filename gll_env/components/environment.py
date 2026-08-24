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

"""Top-level environment model — fixed concrete class.

:class:`EnvironmentModel` couples :class:`~grid.GridDynamics` and
:class:`~prosumer.ProsumerDynamics`, owns the intra-day clock, and exposes
``reset`` / ``step`` consumed by :class:`~generator.ProsumerGridGenerator`.

:class:`EnvironmentState` is the complete JAX-traceable episode state:
it holds both sub-states, the day-time bookkeeping, and the PRNG key
needed by Jumanji's :class:`~jumanji.wrappers.AutoResetWrapper`.

The agent-facing action space is normalized [-1, 1]^2 per agent (matching
Jumanji convention), but ALL feasibility enforcement happens inside
ProsumerDynamics.step() itself, in physical units, against its own
s_inv_request_constraint (which is itself built from
InverterDynamics.step()'s own constraint, augmented with the grid-connection
ball -- see prosumer.py). EnvironmentModel does no projection of its own: it
only scales the normalized action into a physical s_inv_request (a pure
multiply) and normalizes Prosumer's already-built constraint back down for
reporting on EnvironmentState.action_constraints (also a pure scale, not a
rebuild). There is deliberately no projection call here at all -- this
class doesn't run one.
"""

from dataclasses import field
from functools import cached_property

import chex
import jax.numpy as jnp
import jax.random as jr
from jumanji.env import StateProtocol

from gll_env.components.day_time import DaytimeDynamics, DaytimeObservation, DaytimeState
from gll_env.components.grid import GridDynamics, GridObservation, GridState
from gll_env.components.prosumer import (
    ProsumerDynamics,
    ProsumerObservation,
    ProsumerState,
)
from gll_env.types import ActionConstraints
from gll_env.utils import safe_normalize


@chex.dataclass(frozen=True)
class EnvironmentState(StateProtocol):
    prosumer_state: ProsumerState
    grid_state: GridState
    time_state: DaytimeState

    action_constraints: ActionConstraints  # normalized [-1, 1]^2, for the COMING interval

    step_count: chex.Numeric  # () int32
    valid: chex.Array  # () bool
    key: chex.PRNGKey  # (2,)


@chex.dataclass(frozen=True)
class EnvironmentObservation:
    """Observation of the environment and its nested component state.

    Attributes:
        prosumer_observation: Nested prosumer observation, normalized when this
            observation is normalized.
        grid_observation: Nested grid observation, normalized when this
            observation is normalized.
        time_observation: Nested daytime observation, normalized when this
            observation is normalized.
        step_count: Number of completed simulation steps; unchanged by
            normalization.
        is_normalized: Defaults to ``False``.
    """

    prosumer_observation: ProsumerObservation
    grid_observation: GridObservation
    time_observation: DaytimeObservation
    step_count: chex.Numeric  # () int32
    is_normalized: bool = field(default=False)

    def normalize(self, environment_model: "EnvironmentModel") -> "EnvironmentObservation":
        return EnvironmentObservation(
            prosumer_observation=self.prosumer_observation.normalize(environment_model.prosumer),
            grid_observation=self.grid_observation.normalize(environment_model.grid),
            time_observation=self.time_observation.normalize(environment_model.time),
            step_count=self.step_count,
            is_normalized=True,
        )


@chex.dataclass(frozen=True)
class EnvironmentModel:
    prosumer: ProsumerDynamics
    grid: GridDynamics
    time: DaytimeDynamics = field(default_factory=DaytimeDynamics)

    def __post_init__(self) -> None:
        chex.assert_equal(self.grid.num_pq, self.prosumer.num_pq)
        chex.assert_equal(self.time.n_steps_per_day, self.prosumer.time.n_steps_per_day)
        chex.assert_equal(self.time.n_steps_per_day, self.grid.time.n_steps_per_day)

    @cached_property
    def num_agents(self) -> int:
        return self.prosumer.num_inv

    def _normalized_action_constraints(
        self, s_inv_request_constraint: ActionConstraints
    ) -> ActionConstraints:
        """Normalize Prosumer's own physical s_inv_request_constraint into
        [-1, 1]^2 action space, using the same isotropic scale
        (s_inv_max_kvah) InverterDynamics itself is built around -- must
        match :meth:`_action_to_request`'s own scale exactly, or the
        reported action_constraints and the physical request produced from
        an action inside it would disagree.

        Exposed on EnvironmentState purely for observability (agent-facing
        action-space bounds); NOT re-enforced here. Actual feasibility is
        ProsumerDynamics.step()'s own job. Scaling by a positive scalar
        preserves the origin-membership invariant (ActionConstraints'
        geometric requirement), and preserves shape (elementwise division),
        so there's nothing to re-validate here -- Prosumer already did.
        """
        s_inv_max_kvah = jnp.asarray(self.prosumer.inverter_dynamics.s_inv_max_kvah)  # (num_inv,)
        s_col = jnp.expand_dims(s_inv_max_kvah, axis=1)  # (num_inv, 1)
        return ActionConstraints(
            halfspace_a=s_inv_request_constraint.halfspace_a,  # normals unchanged
            halfspace_b=safe_normalize(s_inv_request_constraint.halfspace_b, s_col),
            ball_center=safe_normalize(
                s_inv_request_constraint.ball_center,
                jnp.expand_dims(s_inv_max_kvah, axis=(1, 2)),
            ),
            ball_radius=safe_normalize(s_inv_request_constraint.ball_radius, s_col),
        )

    def _action_to_request(self, action: chex.Array) -> chex.Array:
        """Scale a normalized (num_agents, 2) action in [-1, 1] into the
        physical s_inv_request ProsumerDynamics.step() expects, using the
        same s_inv_max_kvah scalar as :meth:`_normalized_action_constraints`
        -- see that method's docstring for why they must match. A pure
        scale, not a projection: whatever feasibility clipping is needed
        happens inside ProsumerDynamics.step() itself.
        """
        s_inv_max_kvah = self.prosumer.inverter_dynamics.s_inv_max_kvah
        return action * jnp.asarray(s_inv_max_kvah)[:, None]

    def observation(self, state: EnvironmentState) -> EnvironmentObservation:
        return EnvironmentObservation(
            prosumer_observation=self.prosumer.observation(state.prosumer_state),
            grid_observation=self.grid.observation(state.grid_state),
            time_observation=self.time.observation(state.time_state),
            step_count=state.step_count,
        )

    def reset(self, key: chex.PRNGKey, time_state: DaytimeState | None = None) -> EnvironmentState:
        """Sample a random initial state.

        1. Draw a random intra-day step so episodes start at different times
           (unless a ``time_state`` is supplied).
        2. Reset the prosumer sub-components at that time (Prosumer's own
           reset() already fabricates a plausible previous interval and
           dispatches a physically-sampled fictional request internally,
           running its own projection.solve() -- no action-space machinery
           needed here at all).
        3. Run one power-flow step so grid and prosumer are consistent.
        4. Report the normalized action_constraints for the coming interval,
           for whatever action the agent chooses at the first real step().
        """
        if time_state is None:
            key, time_key = jr.split(key)
            time_state = self.time.reset(time_key)

        key, prosumer_key = jr.split(key)
        prosumer_state = self.prosumer.reset(key=prosumer_key, time_state=time_state)
        grid_state = self.grid.reset()
        grid_state = self.grid.step(
            state=grid_state,
            p_pq_request_kwh=prosumer_state.s_pq_realized_kvah.real,
            q_pq_request_kvarh=prosumer_state.s_pq_realized_kvah.imag,
        )
        action_constraints = self._normalized_action_constraints(
            prosumer_state.s_inv_request_constraint
        )

        step_count = jnp.asarray(0, dtype=jnp.int32)
        valid = jnp.logical_and(grid_state.valid, prosumer_state.valid)
        return EnvironmentState(
            prosumer_state=prosumer_state,
            grid_state=grid_state,
            time_state=time_state,
            action_constraints=action_constraints,
            step_count=step_count,
            valid=valid,
            key=key,
        )

    def step(self, state: EnvironmentState, action: chex.Array) -> EnvironmentState:
        """Advance the environment by one simulation interval.

        Parameters
        ----------
        action : chex.Array
            Shape (num_agents, 2) float32: the normalized [a_p, a_q] request
            per agent in [-1, 1]. Scaled to a physical s_inv_request (see
            _action_to_request) and dispatched directly into
            ProsumerDynamics.step(), which enforces feasibility itself.
        """
        chex.assert_shape(action, (self.num_agents, 2))
        chex.assert_type(action, jnp.float32)

        s_inv_request = self._action_to_request(action)

        next_time_state = self.time.step(state.time_state)
        next_prosumer_state = self.prosumer.step(
            state.prosumer_state,
            s_inv_request,
            next_time_state=next_time_state,
        )
        next_grid_state = self.grid.step(
            state=state.grid_state,
            p_pq_request_kwh=next_prosumer_state.s_pq_realized_kvah.real,
            q_pq_request_kvarh=next_prosumer_state.s_pq_realized_kvah.imag,
        )
        next_action_constraints = self._normalized_action_constraints(
            next_prosumer_state.s_inv_request_constraint
        )
        next_step_count = jnp.asarray(state.step_count + 1, dtype=jnp.int32)

        # prosumer_state.valid already carries the prosumer's own history
        # (its own projection_converged AND previous prosumer validity), but
        # grid validity is per-step, so fold in state.valid to make env-level
        # validity sticky.
        next_valid = jnp.logical_and(
            state.valid,
            jnp.logical_and(next_grid_state.valid, next_prosumer_state.valid),
        )
        next_key = jr.fold_in(state.key, state.step_count + 1)
        return EnvironmentState(
            prosumer_state=next_prosumer_state,
            grid_state=next_grid_state,
            time_state=next_time_state,
            action_constraints=next_action_constraints,
            step_count=next_step_count,
            valid=next_valid,
            key=next_key,
        )
