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

What this class does on top of that depends on `grid_code`:

* `GridCode()` (the default, no law): action_dim is 2, and this class runs
  no solver of its own. It scales the normalized [a_p, a_q] into a physical
  s_inv_request (a pure multiply) and normalizes Prosumer's already-built
  constraint back down for reporting on EnvironmentState.action_constraints
  (also a pure scale, not a rebuild). Unchanged from before grid codes
  existed.

* `GridCode(q_of_u=...)`: action_dim is 1. Reactive power is no longer the
  agent's to choose -- it is fixed by the Q(U) curve at the voltage measured
  last interval -- so this class must decide the reactive setpoint, work out
  what active-power range remains feasible alongside it, and lift the
  agent's one-dimensional action back into the (p, q) pair Prosumer expects.
  That is what _grid_code_bounds does, in closed form -- see its docstring
  for the reasoning, and ActionConstraints.restrict for the geometry it
  leans on.

Neither case runs a projection here; that remains ProsumerDynamics.step()'s
job. Under Q(U) it simply finds nothing to do, because the bounds reported
to the agent were built as a subset of Prosumer's own feasible set rather
than as an independent guess at it.
"""

from dataclasses import field
from functools import cached_property

import chex
import jax.numpy as jnp
import jax.random as jr
from jumanji.env import StateProtocol

from gll_env.components.day_time import DaytimeDynamics, DaytimeObservation, DaytimeState
from gll_env.components.grid import GridDynamics, GridObservation, GridState
from gll_env.components.grid_code import GridCode
from gll_env.components.prosumer import (
    ProsumerDynamics,
    ProsumerObservation,
    ProsumerState,
)
from gll_env.types import ActionConstraints

# Axis positions inside the (num_agents, 2) s_inv_request the prosumer tree
# passes around: active power first, reactive second.
_P_AXIS = 0
_Q_AXIS = 1


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
    grid_code: GridCode = field(default_factory=GridCode)

    def __post_init__(self) -> None:
        chex.assert_equal(self.grid.num_pq, self.prosumer.num_pq)
        chex.assert_equal(self.time.n_steps_per_day, self.prosumer.time.n_steps_per_day)
        chex.assert_equal(self.time.n_steps_per_day, self.grid.time.n_steps_per_day)
        if self.grid_code.q_of_u is not None:
            chex.assert_equal(self.num_agents, self.grid_code.q_of_u.num_inv)

    @cached_property
    def num_agents(self) -> int:
        return self.prosumer.num_inv

    @cached_property
    def action_dim(self) -> int:
        """Degrees of freedom per agent: 1 under Q(U), 2 with no grid code."""
        return self.grid_code.action_dim

    @cached_property
    def agent_bus_id(self) -> chex.Array:
        """Global bus index of each inverter agent, shape (num_inv,).

        Inverters are indexed against PQ connection points (`inverter_id`),
        while grid quantities are indexed against buses, so reading a voltage
        for an agent needs both hops. Lives here rather than in the observer
        because the Q(U) law needs the same gather.
        """
        return jnp.take(self.grid.pq_id, self.prosumer.inverter_id)

    def _normalized_action_constraints(
        self, s_inv_request_constraint: ActionConstraints
    ) -> ActionConstraints:
        """Normalize a physical constraint into [-1, 1]^action_dim action space.

        All this method decides is the scale; the transform itself is
        :meth:`ActionConstraints.scale`. The choice matters more than the
        arithmetic: it must be the same s_inv_max_kvah
        :meth:`_action_to_request` multiplies back by, or the reported
        constraints and the physical request produced from an action inside
        them would describe different sets.

        Takes whichever constraint describes the agent's actual freedom --
        Prosumer's own 2-D s_inv_request_constraint with no grid code, or the
        1-D active-power box _grid_code_bounds carved out of it under Q(U).
        `scale` is dimension-agnostic, so neither case needs special casing.

        Exposed on EnvironmentState as the agent-facing action-space bounds;
        NOT re-enforced here. Feasibility is ProsumerDynamics.step()'s own
        job in both cases -- though under Q(U) it finds nothing to do, the
        bounds having already been built as a feasible subset.
        """
        return s_inv_request_constraint.scale(
            jnp.asarray(self.prosumer.inverter_dynamics.s_inv_max_kvah)
        )

    def _grid_code_bounds(
        self,
        s_inv_request_constraint: ActionConstraints,
        bus_voltage_pu: chex.Array,
    ) -> tuple[ActionConstraints, chex.Array]:
        """Reduce the 2-D (p, q) constraint to the 1-D active-power box that
        remains once Q(U) has claimed the reactive axis.

        Returns the physical (num_agents, 1) constraint and the reactive
        setpoint in kvarh that it was carved out alongside.

        Both steps are exact restrictions of the constraint set to a line --
        see :meth:`ActionConstraints.restrict` for why that is closed-form
        rather than a search, and for what `origin_feasible` promises.

        Step 1 picks the setpoint. The Q(U) curve reads voltage alone (see
        grid_code.py), so it can ask for reactive power the connection cannot
        carry: it is written against the plant's nameplate rating, which says
        nothing about the load sharing its meter. Restricting to p = 0 gives
        the reactive values that leave zero active power feasible, and
        clamping the curve's target into that range derates it to what the
        connection can actually absorb.

        That range always contains q = 0, because (0, 0) is feasible --
        ProsumerDynamics' own proven invariant -- so the clamp is always
        well-defined and can only pull the setpoint toward the curve's own
        zero, never past it into the opposite sign.

        Step 2 restricts to q = q* and reads off the active-power range,
        which is what step 1 earns the right to do with `origin_feasible`:
        (0, q*) was just established feasible, so p = 0 is guaranteed inside
        the result. The interval is therefore never empty, with no sizing
        condition to satisfy, and every point in it is feasible in 2-D --
        the endpoints by construction, the interior by convexity. That is
        what leaves Prosumer's own projection with nothing to do.

        Note this is the SLICE at q*, not the shadow of the 2-D set onto the
        p axis. The slice is the smaller of the two and the correct one: it
        answers "what active power is available given this reactive
        setpoint", which is exactly the question the agent faces.
        """
        q_of_u = self.grid_code.q_of_u
        assert q_of_u is not None  # only reached under a Q(U) grid code

        voltage_pu = jnp.abs(jnp.asarray(bus_voltage_pu))[self.agent_bus_id]
        q_target_kvarh = q_of_u.q_setpoint_kvarh(voltage_pu, self.time.step_duration_h)

        zeros = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        q_min_kvarh, q_max_kvarh = s_inv_request_constraint.restrict(_P_AXIS, zeros).bounds()
        # The range provably straddles zero; clamping its ends against 0.0
        # keeps that true through float error.
        q_setpoint_kvarh = jnp.clip(
            q_target_kvarh, jnp.minimum(q_min_kvarh, 0.0), jnp.maximum(q_max_kvarh, 0.0)
        )

        p_min_kwh, p_max_kwh = s_inv_request_constraint.restrict(
            _Q_AXIS, q_setpoint_kvarh, origin_feasible=True
        ).bounds()
        return ActionConstraints.from_bounds(p_min_kwh, p_max_kwh), q_setpoint_kvarh

    def _coming_interval(
        self,
        prosumer_state: ProsumerState,
        grid_state: GridState,
    ) -> tuple[ActionConstraints, chex.Array]:
        """The agent-facing action constraints and reactive setpoint for the
        coming interval: (normalized constraints, q_setpoint_kvarh).

        The voltage read here is the one the power flow just produced, i.e.
        the most recent measurement available. The setpoint derived from it
        is applied during the NEXT interval, so the control lags the
        measurement by exactly one step -- which is what a real Q(U)
        controller does, reacting to measured terminal voltage with a
        5-second time constant (NE7 §4.3.2(2)) that is fully settled inside a
        15-minute interval. It also avoids an algebraic loop: the voltage of
        the coming interval is not knowable without first choosing the
        reactive power that helps determine it.
        """
        if self.grid_code.q_of_u is None:
            return (
                self._normalized_action_constraints(prosumer_state.s_inv_request_constraint),
                jnp.zeros((self.num_agents,), dtype=jnp.float32),
            )
        constraint, q_setpoint_kvarh = self._grid_code_bounds(
            prosumer_state.s_inv_request_constraint, grid_state.bus_voltage_pu
        )
        return self._normalized_action_constraints(constraint), q_setpoint_kvarh

    def _action_to_request(self, action: chex.Array, q_setpoint_kvarh: chex.Array) -> chex.Array:
        """Scale a normalized (num_agents, action_dim) action in [-1, 1] into
        the physical (num_agents, 2) s_inv_request ProsumerDynamics.step()
        expects, using the same s_inv_max_kvah scalar as
        :meth:`_normalized_action_constraints` -- see that method's docstring
        for why they must match.

        With no grid code this is the pure scale it has always been. Under
        Q(U) the action carries active power alone, so the scaled value is
        paired with the stored reactive setpoint to rebuild the (p, q) pair.
        Still no projection either way: any clipping needed happens inside
        ProsumerDynamics.step(), and under Q(U) there is nothing left to clip
        because the reported bounds were already a subset of its feasible set.
        """
        s_inv_max_kvah = jnp.asarray(self.prosumer.inverter_dynamics.s_inv_max_kvah)
        scaled = action * s_inv_max_kvah[:, None]
        if self.grid_code.q_of_u is None:
            return scaled
        return jnp.stack([scaled[:, 0], q_setpoint_kvarh], axis=-1)

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
        action_constraints, q_setpoint_kvarh = self._coming_interval(prosumer_state, grid_state)

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
            s_inv_request (see _action_to_request) and dispatched directly
            into ProsumerDynamics.step(), which enforces feasibility itself.
        """
        chex.assert_shape(action, (self.num_agents, self.action_dim))
        chex.assert_type(action, jnp.float32)

        s_inv_request = self._action_to_request(action, state.q_setpoint_kvarh)

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
        next_action_constraints, next_q_setpoint_kvarh = self._coming_interval(
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
