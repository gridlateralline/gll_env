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

"""The reward-component contract: state lives in the episode, settlement is
aligned with the interval it describes, and lookahead is refused rather than
silently mis-settled.
"""

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from gll_env.components.environment import EnvironmentDynamics, EnvironmentState
from gll_env.factories import default_environment_model
from gll_env.rewards.base import (
    CausalReward,
    EmptyRewardObservation,
    RewardDynamics,
    RewardState,
    StatelessReward,
    default_reward,
)
from gll_env.types import RewardObservation


@chex.dataclass(frozen=True)
class _CounterState(RewardState):
    """Running count of settled intervals, and the last settlement."""

    settled: chex.Array  # () int32
    last: chex.Array  # (num_agents,) float32


@chex.dataclass(frozen=True)
class _CounterObservation(RewardObservation):
    settled: chex.Array  # () int32

    def normalize(self, reward_dynamics: "RewardDynamics") -> "_CounterObservation":
        del reward_dynamics
        return self


class _CounterReward(CausalReward):
    """Pays each agent the number of intervals settled so far.

    Deliberately history-dependent: its value is not a function of the
    transition alone, which is the whole reason RewardState has to live inside
    EnvironmentState.
    """

    def __init__(self, num_agents: int) -> None:
        self._num_agents = num_agents

    def reset(self, key: chex.PRNGKey) -> _CounterState:
        del key
        return _CounterState(
            settled=jnp.int32(0),
            last=jnp.zeros((self._num_agents,), dtype=jnp.float32),
        )

    def settle(
        self,
        reward_state: RewardState,
        state: EnvironmentState,
        new_state: EnvironmentState,
        dynamics: EnvironmentDynamics,
    ) -> tuple[_CounterState, chex.Array]:
        del state, new_state, dynamics
        settled = reward_state.settled + 1
        reward = jnp.full((self._num_agents,), settled, dtype=jnp.float32)
        return _CounterState(settled=settled, last=reward), reward

    def observation(self, reward_state: RewardState) -> _CounterObservation:
        return _CounterObservation(settled=reward_state.settled)


class _LookaheadReward(_CounterReward):
    lookahead = 3


def _zero_action(model: EnvironmentDynamics) -> chex.Array:
    return jnp.zeros((model.num_agents, model.action_dim), dtype=jnp.float32)


def test_reward_memory_advances_with_the_episode() -> None:
    """A history-dependent reward accumulates across steps, and the state it
    accumulates in rides inside EnvironmentState -- so the trajectory is a
    function of reset(key) alone."""
    base = default_environment_model()
    model = base.replace(reward=_CounterReward(base.num_agents))

    state = model.reset(jr.PRNGKey(0))
    assert int(state.reward_state.settled) == 0

    rewards = []
    for _ in range(3):
        state, reward = model.step(state, _zero_action(model))
        rewards.append(float(reward[0]))

    assert rewards == [1.0, 2.0, 3.0]
    assert int(state.reward_state.settled) == 3


def test_reward_state_resets_with_the_episode() -> None:
    """Two resets from the same key give the same reward memory, and a reset
    never inherits the previous episode's. An auto-reset wrapper delivering
    the last episode's bills to a fresh household would otherwise be invisible."""
    base = default_environment_model()
    model = base.replace(reward=_CounterReward(base.num_agents))

    state = model.reset(jr.PRNGKey(0))
    for _ in range(4):
        state, _ = model.step(state, _zero_action(model))
    assert int(state.reward_state.settled) == 4

    fresh = model.reset(jr.PRNGKey(0))
    assert int(fresh.reward_state.settled) == 0
    chex.assert_trees_all_close(fresh, model.reset(jr.PRNGKey(0)))


def test_observation_sees_the_settlement_from_its_own_interval() -> None:
    """step() must settle BEFORE the observation is derived. Off by one here
    and anything a reward publishes is one interval stale, silently."""
    base = default_environment_model()
    model = base.replace(reward=_CounterReward(base.num_agents))

    state = model.reset(jr.PRNGKey(0))
    for expected in (1, 2, 3):
        state, _ = model.step(state, _zero_action(model))
        assert int(model.observation(state).reward_observation.settled) == expected


def test_lookahead_reward_is_refused_rather_than_mis_settled() -> None:
    """The environment settles each interval as it ends and cannot see the
    future. Declaring lookahead must fail loudly, naming the way forward."""
    base = default_environment_model()
    with pytest.raises(ValueError, match="lookahead"):
        base.replace(reward=_LookaheadReward(base.num_agents))


def test_causal_reward_rejects_a_trajectory_it_cannot_settle() -> None:
    """CausalReward's contract is exactly two states. A longer window means a
    caller believes it is getting lookahead it is not getting."""
    base = default_environment_model()
    reward = _CounterReward(base.num_agents)
    state = base.reset(jr.PRNGKey(0))

    with pytest.raises(ValueError, match="causal"):
        reward(reward.reset(jr.PRNGKey(0)), (state, state, state), base)


def test_stateless_reward_publishes_nothing() -> None:
    """Wrapping a legacy RewardFn must not widen the observation -- that is
    what keeps existing training configurations working untouched."""
    model = default_environment_model()
    assert isinstance(model.reward, StatelessReward)

    state = model.reset(jr.PRNGKey(0))
    observation = model.reward.observation(state.reward_state)
    assert isinstance(observation, EmptyRewardObservation)
    assert jax.tree_util.tree_leaves(observation) == []
    assert default_reward().lookahead == 0


def test_reward_survives_jit_and_scan() -> None:
    """The reward state is part of the state pytree, so a scanned rollout
    carries it with no special handling."""
    base = default_environment_model()
    model = base.replace(reward=_CounterReward(base.num_agents))

    def body(state: EnvironmentState, _: None) -> tuple[EnvironmentState, chex.Array]:
        state, reward = model.step(state, _zero_action(model))
        return state, reward

    state = model.reset(jr.PRNGKey(0))
    final, rewards = jax.jit(lambda s: jax.lax.scan(body, s, None, length=5))(state)

    chex.assert_trees_all_close(rewards[:, 0], jnp.arange(1.0, 6.0))
    assert int(final.reward_state.settled) == 5
