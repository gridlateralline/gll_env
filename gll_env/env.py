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

from functools import cached_property
from typing import Optional

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition

from gll_env.components.environment import EnvironmentDynamics, EnvironmentState
from gll_env.generator import DynamicsGenerator
from gll_env.observer import (
    AnyObservation,
    MarlObserver,
    Observer,
)
from gll_env.rewards import BaseReward, RewardFn


class ProsumerGrid(
    Environment[
        EnvironmentState,
        specs.BoundedArray,
        AnyObservation,
    ],
):
    """Jumanji environment for a prosumer-based AC distribution grid.

    Follows the Jumanji composability pattern:

    * **generator** — encodes scenario parameters and produces the initial
      state; swap to change network topology or prosumer sizing.
    * **observer** — maps state → observation; swap to change what agents see
      (raw physics, flat MARL vector, global state for CTDE).
    * **reward_fn** — maps (state, new_state) → per-agent reward; swap to
      change the reward signal without touching dynamics.

    Args:
        generator: Scenario generator (:class:`~generator.DynamicsGenerator`).
            Build one with :func:`~factories.environment_model` or construct
            :class:`~components.environment.EnvironmentDynamics` directly. Defaults to
            :meth:`~generator.DynamicsGenerator.default` — a small, self-contained
            scenario — so ``ProsumerGrid()`` and ``jumanji.make("ProsumerGrid-v0")``
            work with no configuration, matching every other Jumanji environment.
        time_limit: Maximum episode length in steps. ``None`` means unlimited.
        observer: Observer instance. Defaults to :class:`~observer.MarlObserver`.
            The observer produces ``global_state`` natively; Mava's wrapper
            optionally strips it if not needed (JIT eliminates dead code).
        reward_fn: Reward function. Defaults to :class:`~reward.BaseReward` (placeholder).

    Mava integration
    ----------------
    ``JumanjiMarlWrapper`` works without any subclassing — just use
    the default :class:`~observer.MarlObserver`::

        generator = ConfigGenerator(
            n_steps_per_day=96,
            grid=grid_cfg,
            prosumer=prosumer_cfg,
        )
        env = ProsumerGrid(
            generator=generator,
            observer=MarlObserver(generator.env_dynamics),
            time_limit=96,
        )
        wrapped = JumanjiMarlWrapper(env)

    This works because :class:`~observer.MarlObserver` already produces the exact
    shapes Mava's wrapper reads:

    * ``agents_view``  — ``(num_agents, NUM_AGENT_FEATURES)`` float32
    * ``action_mask``  — ``(num_agents, action_dim)`` bool
    * ``step_count``   — scalar int32 (Mava broadcasts internally)

    And :class:`ProsumerGrid` already exposes the two attributes Mava's wrapper
    reads at construction time: ``env.num_agents`` and ``env.time_limit``.
    """

    def __init__(
        self,
        generator: Optional[DynamicsGenerator] = None,
        time_limit: Optional[int] = None,
        observer: Optional[Observer] = None,
        reward_fn: Optional[RewardFn] = None,
    ):
        self._generator = generator or DynamicsGenerator.default()
        self._env_model = self._generator.env_dynamics
        self._time_limit = time_limit
        self._observer: Observer = observer or MarlObserver(self._env_model)
        self._reward_fn: RewardFn = reward_fn or BaseReward()
        super().__init__()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_agents={self.num_agents}, time_limit={self.time_limit})"
        )

    # ------------------------------------------------------------------
    # Public properties (Mava reads num_agents and time_limit directly)
    # ------------------------------------------------------------------

    @cached_property
    def num_agents(self) -> int:
        return self._env_model.num_agents

    @cached_property
    def time_limit(self) -> Optional[int]:
        return self._time_limit

    @cached_property
    def action_dim(self) -> int:
        """Degrees of freedom per agent, set by the model's grid code: 2
        (active and reactive power) with no grid code, 1 (active power only)
        under a Q(U) characteristic, where reactive power is prescribed by
        the curve rather than chosen. Read off the model for the same reason
        `num_agents` is -- the scenario decides, not this wrapper.
        """
        return self._env_model.action_dim

    @cached_property
    def environment(self) -> EnvironmentDynamics:
        """Expose the wrapped environment model."""
        return self._env_model

    # ------------------------------------------------------------------
    # Jumanji Environment interface
    # ------------------------------------------------------------------

    def reset(self, key: chex.PRNGKey) -> tuple[EnvironmentState, TimeStep[AnyObservation]]:
        state = self._generator(key)
        observation = self._observer.state_to_observation(state)
        shape = (self.num_agents,)
        extras = {"nr_steps": state.grid_state.nr_steps}
        timestep = restart(
            observation=observation,
            extras=extras,
            shape=shape,
            dtype=jnp.float32,
        )
        return state, timestep

    def step(
        self, state: EnvironmentState, action: chex.Array
    ) -> tuple[EnvironmentState, TimeStep[AnyObservation]]:
        chex.assert_shape(action, (self.num_agents, self.action_dim))
        action = jnp.asarray(action, dtype=jnp.float32)
        new_state = self._env_model.step(state=state, action=action)

        observation = self._observer.state_to_observation(new_state)
        reward = self._reward_fn(state, new_state)
        shape = (self.num_agents,)
        extras = {"nr_steps": new_state.grid_state.nr_steps}

        if self._time_limit is not None:
            in_time_limit = new_state.step_count < self._time_limit
        else:
            in_time_limit = jnp.bool(True)

        timestep = jax.lax.cond(
            jnp.logical_and(new_state.valid, in_time_limit),
            lambda: transition(
                reward=reward,
                observation=observation,
                extras=extras,
                shape=shape,
                dtype=jnp.float32,
            ),
            lambda: termination(
                reward=reward,
                observation=observation,
                extras=extras,
                shape=shape,
                dtype=jnp.float32,
            ),
        )
        return new_state, timestep

    # ------------------------------------------------------------------
    # Specs
    # ------------------------------------------------------------------

    @cached_property
    def action_spec(self) -> specs.BoundedArray:
        """Normalised action per agent in [-1, 1]: [P_request, Q_request],
        or [P_request] alone when a Q(U) grid code sets the reactive half.
        """
        return specs.BoundedArray(
            shape=(self.num_agents, self.action_dim),
            dtype=jnp.float32,
            minimum=-1.0,
            maximum=+1.0,
            name="action",
        )

    @cached_property
    def observation_spec(self) -> specs.Spec[AnyObservation]:
        """Delegates to the observer — consistent with whatever ``__call__`` returns."""
        return self._observer.observation_spec

    @cached_property
    def reward_spec(self) -> specs.Array:
        """Per-agent reward, shape (num_agents,) — overrides Jumanji's scalar default."""
        return specs.Array(shape=(self.num_agents,), dtype=jnp.float32, name="reward")

    @cached_property
    def discount_spec(self) -> specs.BoundedArray:
        """Per-agent discount, shape (num_agents,) — overrides Jumanji's scalar default."""
        return specs.BoundedArray(
            shape=(self.num_agents,),
            dtype=jnp.float32,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        )

    def close(self) -> None:
        """Perform any necessary cleanup."""
        return None
