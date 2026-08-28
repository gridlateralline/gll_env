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
``reset`` / ``step`` consumed by :class:`~gll_env.generator.DynamicsGenerator`.

:class:`EnvironmentState` is the complete JAX-traceable episode state:
it holds both sub-states, the day-time bookkeeping, and the PRNG key
needed by Jumanji's :class:`~jumanji.wrappers.AutoResetWrapper`.

The agent-facing action space is normalized [-1, 1]^action_dim per agent
(matching Jumanji convention), and ALL physical feasibility enforcement
happens inside ProsumerDynamics.step(), in physical units, against its own
s_inv_request_constraint (which is itself built from
InverterDynamics.step()'s own constraint, augmented with the grid-connection
ball -- see prosumer.py).

What the agent may choose within that is the `grid_code`'s call, not this
class's. This class supplies the measurement and the physical constraint;
the code reduces them to the axes the agent still controls and names any
setpoint it imposed on the rest (`_next_action_constraints`), and rebuilds the full
(p, q) request afterwards (`_to_request`). `NoGridCode`, the default,
reduces to nothing and lifts to nothing, which is the two-degree-of-freedom
action space this environment started with.

No projection runs here in either case; that remains ProsumerDynamics.step()'s
job. Under a code that reduces the action space it simply finds nothing to do,
because the bounds reported to the agent were built as a subset of Prosumer's
own feasible set rather than as an independent guess at it.
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
from gll_env.grid_codes.base import GridCode, NoGridCode
from gll_env.types import ActionConstraints


@chex.dataclass(frozen=True)
class EnvironmentState(StateProtocol):
    time_state: DaytimeState
    grid_state: GridState
    prosumer_state: ProsumerState

    # Normalized to [-1, 1]^action_dim, for the COMING interval.
    action_constraints: ActionConstraints

    # (num_inv,) float32 kvarh -- the reactive setpoint the grid code imposes
    # for the COMING interval, decided at the end of the step that produced
    # this state from the voltage measured then. Zero when no grid code
    # applies, in which case the agent's own action supplies q instead.
    #
    # Stored rather than recomputed on the next step() because
    # action_constraints was built by carving out the active-power range that
    # is feasible ALONGSIDE this exact value; recomputing it there would make
    # their agreement an invariant to maintain instead of a fact.
    q_setpoint_kvarh: chex.Array

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

    time_observation: DaytimeObservation
    grid_observation: GridObservation
    prosumer_observation: ProsumerObservation
    step_count: chex.Numeric  # () int32
    is_normalized: bool = field(default=False)

    def normalize(self, environment_model: "EnvironmentDynamics") -> "EnvironmentObservation":
        return EnvironmentObservation(
            time_observation=self.time_observation.normalize(environment_model.time),
            grid_observation=self.grid_observation.normalize(environment_model.grid),
            prosumer_observation=self.prosumer_observation.normalize(environment_model.prosumer),
            step_count=self.step_count,
            is_normalized=True,
        )


@chex.dataclass(frozen=True)
class EnvironmentDynamics:
    prosumer: ProsumerDynamics
    grid: GridDynamics
    time: DaytimeDynamics = field(default_factory=DaytimeDynamics)
    grid_code: GridCode = field(default_factory=NoGridCode)

    def __post_init__(self) -> None:
        chex.assert_equal(self.grid.num_pq, self.prosumer.num_pq)
        chex.assert_equal(self.time.n_steps_per_day, self.prosumer.time.n_steps_per_day)
        chex.assert_equal(self.time.n_steps_per_day, self.grid.time.n_steps_per_day)

    @cached_property
    def num_agents(self) -> int:
        return self.prosumer.num_inv

    @cached_property
    def action_dim(self) -> int:
        """Degrees of freedom per agent, decided by the grid code in force."""
        return self.grid_code.action_dim

    @cached_property
    def action_scale(self) -> chex.Array:
        """The factor relating the agent's [-1, 1] action space to physical
        units, shape (num_agents,).

        A single named source for both halves of that bijection --
        :meth:`_to_action_space` divides by it,
        :meth:`_to_request` multiplies by it. They have to agree, or
        the constraints reported to the agent and the request produced from an
        action inside them would describe different sets; reading one property
        is what makes agreeing the default rather than something to maintain.
        """
        return jnp.asarray(self.prosumer.inverter_dynamics.s_inv_max_kvah)

    @cached_property
    def agent_bus_id(self) -> chex.Array:
        """Global bus index of each inverter agent, shape (num_inv,).

        Inverters are indexed against PQ connection points (`inverter_id`),
        while grid quantities are indexed against buses, so reading a voltage
        for an agent needs both hops. Lives here rather than in the observer
        because the Q(U) law needs the same gather.
        """
        return jnp.take(self.grid.pq_id, self.prosumer.inverter_id)

    def agent_voltage_pu(self, bus_voltage_pu: chex.Array) -> chex.Array:
        """Voltage magnitude at each agent's own connection point, shape
        (num_agents,), gathered from a full (num_bus,) bus voltage array.

        The measurement a grid code reads. Named rather than inlined because
        it is the input to :meth:`GridCode.reduce`, and anything checking that
        reduction should feed it the same gather the environment does instead
        of rebuilding it.
        """
        return jnp.abs(jnp.asarray(bus_voltage_pu))[self.agent_bus_id]

    def _to_action_space(self, s_inv_request_constraint: ActionConstraints) -> ActionConstraints:
        """Map a physical constraint into [-1, 1]^action_dim action space.

        The inverse of :meth:`_to_request`, and deliberately named as one:
        the transform is :meth:`ActionConstraints.normalized_by` and the
        factor is :attr:`action_scale`, shared between them so the two
        directions cannot drift apart.

        Takes whichever constraint describes the agent's actual freedom --
        Prosumer's own 2-D s_inv_request_constraint with no grid code, or the
        1-D active-power box _grid_code_bounds carved out of it under Q(U).
        `scale` is dimension-agnostic, so neither case needs special casing.

        Exposed on EnvironmentState as the agent-facing action-space bounds;
        NOT re-enforced here. Feasibility is ProsumerDynamics.step()'s own
        job in both cases -- though under Q(U) it finds nothing to do, the
        bounds having already been built as a feasible subset.
        """
        return s_inv_request_constraint.normalized_by(self.action_scale)

    def _next_action_constraints(
        self,
        prosumer_state: ProsumerState,
        grid_state: GridState,
    ) -> tuple[ActionConstraints, chex.Array]:
        """The agent-facing action constraints for the COMING interval, and
        the reactive setpoint they were carved out alongside -- the two
        fields EnvironmentState stores as a pair, returned as one so they
        cannot be derived from different voltages.

        This class supplies the measurement and the physical constraint; the
        grid code decides what the agent may do with them, and how many
        degrees of freedom that leaves. Whether any law applies at all is the
        code's business, not this method's -- NoGridCode passes the
        constraint straight through.

        The voltage read here is the one the power flow just produced, i.e.
        the most recent measurement available. The setpoint derived from it
        is applied during the NEXT interval, so the control lags the
        measurement by exactly one step -- which is what a real Q(U)
        controller does, reacting to measured terminal voltage with a
        5-second time constant (NE7 4.3.2(2)) that is fully settled inside a
        15-minute interval. It also avoids an algebraic loop: the voltage of
        the coming interval is not knowable without first choosing the
        reactive power that helps determine it.
        """
        constraint, q_setpoint_kvarh = self.grid_code.reduce(
            prosumer_state.s_inv_request_constraint,
            voltage_pu=self.agent_voltage_pu(grid_state.bus_voltage_pu),
            step_duration_h=self.time.step_duration_h,
        )
        return self._to_action_space(constraint), q_setpoint_kvarh

    def _to_request(self, action: chex.Array, q_setpoint_kvarh: chex.Array) -> chex.Array:
        """Scale a normalized (num_agents, action_dim) action in [-1, 1] into
        the physical (num_agents, 2) s_inv_request ProsumerDynamics.step()
        expects, by the same :attr:`action_scale`
        :meth:`_to_action_space` divides by.

        The scale is this class's business; rebuilding the full (p, q) pair
        from whatever axes the agent still controls is the grid code's, and
        is the exact inverse of the reduction it performed in
        :meth:`_next_action_constraints`.

        Still no projection either way: any clipping needed happens inside
        ProsumerDynamics.step(), and under a code that reduces the action
        space there is nothing left to clip, the reported bounds having been
        built as a subset of its feasible set.
        """
        return self.grid_code.lift(action * self.action_scale[:, None], q_setpoint_kvarh)

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
        action_constraints, q_setpoint_kvarh = self._next_action_constraints(
            prosumer_state, grid_state
        )

        step_count = jnp.asarray(0, dtype=jnp.int32)
        valid = jnp.logical_and(grid_state.valid, prosumer_state.valid)
        return EnvironmentState(
            time_state=time_state,
            grid_state=grid_state,
            prosumer_state=prosumer_state,
            action_constraints=action_constraints,
            q_setpoint_kvarh=q_setpoint_kvarh,
            step_count=step_count,
            valid=valid,
            key=key,
        )

    def step(self, state: EnvironmentState, action: chex.Array) -> EnvironmentState:
        """Advance the environment by one simulation interval.

        Parameters
        ----------
        action : chex.Array
            Shape (num_agents, action_dim) float32, normalized to [-1, 1]:
            the [a_p, a_q] request per agent with no grid code, or [a_p]
            alone under Q(U), where the reactive half comes from
            state.q_setpoint_kvarh instead. Scaled to a physical
            s_inv_request (see _to_request) and dispatched directly
            into ProsumerDynamics.step(), which enforces feasibility itself.
        """
        chex.assert_shape(action, (self.num_agents, self.action_dim))
        chex.assert_type(action, jnp.float32)

        s_inv_request = self._to_request(action, state.q_setpoint_kvarh)

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
        next_action_constraints, next_q_setpoint_kvarh = self._next_action_constraints(
            next_prosumer_state, next_grid_state
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
            time_state=next_time_state,
            grid_state=next_grid_state,
            prosumer_state=next_prosumer_state,
            action_constraints=next_action_constraints,
            q_setpoint_kvarh=next_q_setpoint_kvarh,
            step_count=next_step_count,
            valid=next_valid,
            key=next_key,
        )
