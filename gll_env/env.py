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
from typing import Callable, Optional, Union

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition, truncation

from gll_env.components.environment import EnvironmentDynamics, EnvironmentState
from gll_env.generator import DynamicsGenerator
from gll_env.observer import (
    AnyObservation,
    MarlObserver,
    Observer,
)
from gll_env.rewards import RewardDynamics, RewardFn, StatelessReward
from gll_env.types import AlignedTransition


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
        reward_fn: Optional[Union[RewardDynamics, RewardFn]] = None,
    ):
        self._generator = generator or DynamicsGenerator.default()
        env_model = self._generator.env_dynamics
        if reward_fn is not None:
            # The reward is a component of the model (it owns state the model
            # must reset and carry), but injecting it here is the Jumanji
            # idiom and what Mava's configs do. Rebuild rather than choose:
            # a legacy stateless RewardFn is adapted on the way in.
            reward = (
                reward_fn if isinstance(reward_fn, RewardDynamics) else StatelessReward(reward_fn)
            )
            env_model = env_model.replace(reward=reward)
            self._generator = DynamicsGenerator(env_model)
        self._env_model = env_model
        self._time_limit = time_limit
        self._observer: Observer = observer or MarlObserver(self._env_model)
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

    def get_aligned_timestep(
        self,
        state: EnvironmentState,
        action: chex.Array,
        reward: chex.Array,
        new_state: EnvironmentState,
    ) -> AlignedTransition:
        """Assemble the temporally coherent transition for ``extras``.

        Every field describes the interval running from `state` to
        `new_state`: the action taken in `state`, and the settlement that
        action produced. Learners and scorers read this; they must NOT pair
        ``timestep.reward`` with ``timestep.observation``, which follow
        different clocks the moment a lookahead reward exists.

        Today the environment is causal, so all of this is degenerate --
        ``next_observation`` equals ``timestep.observation``, ``action`` is the
        action just passed, ``valid`` is always true. Emitting it anyway is
        the point: a consumer written against this contract now keeps working
        unchanged when the re-timing wrapper arrives, where none of those
        equalities hold. See ``docs/lookahead_rewards.md``.

        Costs one extra observer evaluation per step, for `state`. That buys
        a SELF-CONTAINED transition -- nothing has to be carried across scan
        iterations, which is also what makes auto-reset stop being a special
        case, since there is no previous observation to be clobbered.
        """
        return AlignedTransition(
            observation=self._observer.state_to_observation(state),
            action=action,
            reward=reward,
            next_observation=self._observer.state_to_observation(new_state),
            valid=jnp.bool_(True),
        )

    def reset(self, key: chex.PRNGKey) -> tuple[EnvironmentState, TimeStep[AnyObservation]]:
        state = self._generator(key)
        observation = self._observer.state_to_observation(state)
        shape = (self.num_agents,)
        zero_action = jnp.zeros((self.num_agents, self.action_dim), dtype=jnp.float32)
        extras = {
            "nr_steps": state.grid_state.nr_steps,
            "reward": self._env_model.reward.observation(state.reward_state),
            # No interval has been settled yet, so the transition is a
            # placeholder with the right structure and valid=False. Emitting
            # it keeps the extras pytree identical across reset and step,
            # which is what lets a scan carry it.
            "transition": AlignedTransition(
                observation=observation,
                action=zero_action,
                reward=jnp.zeros(shape, dtype=jnp.float32),
                next_observation=observation,
                valid=jnp.bool_(False),
            ),
        }
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
        """Advance one interval.

        The two clocks, and which to read:

        * ``timestep.observation`` is the state AFTER the action -- what the
          policy conditions on next. **Act on this.**
        * ``extras["transition"]`` is the settled interval, self-contained.
          **Learn and score from this.**

        They coincide while the environment is causal. They stop coinciding
        the moment a lookahead reward is introduced, and the rule above is
        chosen so that a consumer following the plain Jumanji contract --
        "act on ``timestep.observation``" -- stays correct either way, and
        only the learning path needs the extras.
        """
        chex.assert_shape(action, (self.num_agents, self.action_dim))
        action = jnp.asarray(action, dtype=jnp.float32)
        new_state, reward = self._env_model.step(state=state, action=action)

        observation = self._observer.state_to_observation(new_state)
        shape = (self.num_agents,)
        extras = {
            "nr_steps": new_state.grid_state.nr_steps,
            # Whatever the reward publishes, forwarded whole. Carries the
            # per-connection-point detail the (num_agents,) reward cannot:
            # connection points with no inverter have no agent to be
            # attributed to and are otherwise invisible downstream.
            "reward": self._env_model.reward.observation(new_state.reward_state),
            "transition": self.get_aligned_timestep(state, action, reward, new_state),
        }

        if self._time_limit is not None:
            in_time_limit = new_state.step_count < self._time_limit
        else:
            in_time_limit = jnp.bool(True)

        def _timestep(builder: Callable[..., TimeStep]) -> TimeStep:
            return builder(
                reward=reward,
                observation=observation,
                extras=extras,
                shape=shape,
                dtype=jnp.float32,
            )

        # Three outcomes, not two. An invalid state is a real termination and
        # bootstrapping past it is meaningless, so discount 0. Running out of
        # time is NOT -- the episode was cut, not ended, and `termination`
        # there trains every state near the horizon toward a value of zero.
        # `truncation` keeps discount 1 so the value function bootstraps.
        timestep = jax.lax.cond(
            new_state.valid,
            lambda: jax.lax.cond(
                in_time_limit,
                lambda: _timestep(transition),
                lambda: _timestep(truncation),
            ),
            lambda: _timestep(termination),
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
