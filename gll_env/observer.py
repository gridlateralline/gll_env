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

import abc
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from functools import cached_property
from typing import Any, Dict, Union

import chex
import jax.numpy as jnp
import jax.random as jr
from jumanji import specs

from gll_env.components.environment import (
    EnvironmentDynamics,
    EnvironmentObservation,
    EnvironmentState,
)
from gll_env.types import MarlObservation

AnyObservation = Union[EnvironmentObservation, MarlObservation]


class Observer(abc.ABC):
    """Abstract observer — maps :class:`EnvironmentState` → typed observation.

    Captures ``env_model`` at construction time so the call signature is
    simply ``observer(state)``, matching Jumanji's ``VectorObserver`` pattern.
    The :attr:`observation_spec` is derived lazily from a dummy reset call
    so shapes are always consistent with the actual ``__call__`` output.
    """

    _env_model: EnvironmentDynamics

    def __init__(self, env_model: EnvironmentDynamics) -> None:
        self._env_model = env_model

    @classmethod
    def spec_from_example(cls, example: Any, name: str | None = None) -> specs.Spec:
        if is_dataclass(example):
            fields_spec: Dict[str, specs.Spec] = {}
            for field in dataclass_fields(example):
                fields_spec[field.name] = cls.spec_from_example(
                    getattr(example, field.name),
                    name=field.name,
                )
            return specs.Spec(type(example), name or type(example).__name__, **fields_spec)

        if isinstance(example, tuple) and hasattr(example, "_fields"):
            fields_spec = {
                field_name: cls.spec_from_example(getattr(example, field_name), name=field_name)
                for field_name in example._fields
            }
            return specs.Spec(type(example), name or type(example).__name__, **fields_spec)

        array_value = jnp.asarray(example)
        return specs.Array(array_value.shape, array_value.dtype, name or "value")

    @abc.abstractmethod
    def state_to_observation(self, state: EnvironmentState) -> AnyObservation:
        """Compute observation from state."""

    @cached_property
    def observation_spec(self) -> specs.Spec:
        """Derive spec by running the observer on a dummy reset state."""
        example_state = self._env_model.reset(jr.PRNGKey(0))
        return self.spec_from_example(self.state_to_observation(example_state), name="observation")


# ---------------------------------------------------------------------------
# Concrete observers
# ---------------------------------------------------------------------------


class RawObserver(Observer):
    """Returns :class:`EnvironmentObservation` unchanged.

    Use this for debugging, visualisation, or custom (non-Mava) training loops
    that need direct access to grid voltages and per-component quantities.
    """

    def state_to_observation(self, state: EnvironmentState) -> EnvironmentObservation:
        return self._env_model.observation(state)


class MarlObserver(Observer):
    """Returns :class:`ObservationMarl` for multi-agent RL.

    The observation is a concatenation of all relevant per-agent and global
    features, with shape (num_agents, NUM_AGENT_FEATURES + NUM_GLOBAL_FEATURES).
    """

    _normalize: bool

    def __init__(self, env_model: EnvironmentDynamics, normalize: bool = False) -> None:
        super().__init__(env_model)
        self._normalize = normalize

    @staticmethod
    def _agent_bus_id(env_model: EnvironmentDynamics) -> chex.Array:
        """Global bus index of each inverter agent, shape (num_inv,)."""
        return jnp.take(env_model.grid.pq_id, env_model.prosumer.inverter_id)

    def _agents_view(self, state: EnvironmentState) -> chex.Array:
        env_obs = self._env_model.observation(state)

        if self._normalize:
            env_obs = env_obs.normalize(self._env_model)

        time_obs = env_obs.time_observation
        grid_obs = env_obs.grid_observation
        prosumer_obs = env_obs.prosumer_observation
        load_obs = prosumer_obs.load_observation
        inverter_obs = prosumer_obs.inverter_observation
        battery_obs = inverter_obs.battery_observation
        solar_obs = inverter_obs.solar_observation

        agent_in_pq_id = self._env_model.prosumer.inverter_id
        agent_in_bus_id = self._agent_bus_id(self._env_model)
        num_agents = self._env_model.num_agents

        parts = [
            jnp.broadcast_to(time_obs.time_sin, (num_agents,)),
            jnp.broadcast_to(time_obs.time_cos, (num_agents,)),
            jnp.take(grid_obs.bus_voltage_deviation, agent_in_bus_id),
            jnp.take(grid_obs.bus_active_power_injection, agent_in_bus_id),
            jnp.take(grid_obs.bus_reactive_power_injection, agent_in_bus_id),
            jnp.take(prosumer_obs.p_pq_realized, agent_in_pq_id),
            jnp.take(prosumer_obs.q_pq_realized, agent_in_pq_id),
            load_obs.p_load_realized,
            load_obs.q_load_realized,
            load_obs.p_load_forecast,
            load_obs.q_load_forecast,
            inverter_obs.p_inv_realized,
            inverter_obs.q_inv_realized,
            inverter_obs.p_inv_min,
            inverter_obs.p_inv_max,
            battery_obs.bat_realized,
            battery_obs.bat_request_min,
            battery_obs.bat_request_max,
            battery_obs.bat_free,
            battery_obs.bat_full,
            solar_obs.sol_realized,
            solar_obs.sol_request_max,
        ]

        return jnp.stack(parts, axis=-1)

    def _global_state(self, state: EnvironmentState) -> chex.Array:
        env_obs = self._env_model.observation(state)

        if self._normalize:
            env_obs = env_obs.normalize(self._env_model)

        time_obs = env_obs.time_observation
        grid_obs = env_obs.grid_observation
        prosumer_obs = env_obs.prosumer_observation
        load_obs = prosumer_obs.load_observation
        inverter_obs = prosumer_obs.inverter_observation
        battery_obs = inverter_obs.battery_observation
        solar_obs = inverter_obs.solar_observation

        num_agents = self._env_model.num_agents

        parts = [
            time_obs.time_sin,
            time_obs.time_cos,
            grid_obs.bus_voltage_deviation,
            grid_obs.bus_voltage_angle,
            grid_obs.bus_active_power_injection,
            grid_obs.bus_reactive_power_injection,
            prosumer_obs.p_pq_realized,
            prosumer_obs.q_pq_realized,
            load_obs.p_load_realized,
            load_obs.q_load_realized,
            load_obs.p_load_forecast,
            load_obs.q_load_forecast,
            inverter_obs.p_inv_realized,
            inverter_obs.q_inv_realized,
            inverter_obs.p_inv_min,
            inverter_obs.p_inv_max,
            battery_obs.bat_realized,
            battery_obs.bat_request_min,
            battery_obs.bat_request_max,
            battery_obs.bat_free,
            battery_obs.bat_full,
            solar_obs.sol_realized,
            solar_obs.sol_request_max,
        ]

        row = jnp.concatenate([jnp.ravel(part) for part in parts])
        return jnp.tile(row, (num_agents, 1))

    def state_to_observation(self, state: EnvironmentState) -> MarlObservation:
        num_agents = self._env_model.num_agents
        return MarlObservation(
            agents_view=self._agents_view(state),
            action_mask=jnp.ones((num_agents, 2), dtype=bool),
            global_state=self._global_state(state),
            action_constraints=state.action_constraints,
            step_count=jnp.repeat(state.step_count, num_agents),
        )
