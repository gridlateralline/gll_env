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

"""Reward-component ABC, its causal specialisation, and the default reward.

A reward is a *rule*, not physics. It sits in the same category as
:mod:`gll_env.grid_codes` and :mod:`gll_env.observer` -- imposed on the agent
from outside the dynamics -- which is why it lives here rather than in
``components/``, and why swapping one must never change the physical
trajectory. What it *does* borrow from ``components/`` is the split between
dynamics and state: a reward owns a :class:`~gll_env.types.RewardState`
carried inside :class:`~gll_env.components.environment.EnvironmentState`, and
exposes a :class:`~gll_env.types.RewardObservation` like any other component.

Three levels, each adding exactly one thing:

* :class:`RewardDynamics` -- the general contract. Takes a *trajectory* of
  ``lookahead + 2`` states and returns the settlement for the interval that
  trajectory brackets. General enough that introducing a future-dependent
  tariff later needs no change here.
* :class:`CausalReward` -- ``lookahead = 0``, unpacks the trajectory to the
  familiar ``(state, new_state)`` pair. **This is what concrete tariffs
  subclass.**
* :class:`StatelessReward` -- adapts a legacy :class:`RewardFn`, which carries
  no state at all.

Why the trajectory signature, when everything shipped is causal
---------------------------------------------------------------
A settlement that depends on *realized* future intervals -- a monthly peak
charge attributed per interval, a forward-looking rolling window -- cannot be
emitted at the interval it describes, because the future depends on actions
not yet taken. The resolution is to delay *emission*, not to guess the future:
buffer the transitions and emit interval ``t``'s settlement once ``t + K`` has
happened. That is a re-timing of the episode and belongs in a wrapper (see
``docs/lookahead_rewards.md``), which is additive.

The one thing that would NOT be additive is this signature, so it is taken
now. :attr:`RewardDynamics.lookahead` is a class attribute, so the environment
dispatches on it in Python rather than in a trace: at ``lookahead == 0``
nothing is ever stacked or buffered and the general form costs nothing.

Most tariffs that seem to need lookahead do not. A peak charge is causal with
state -- emit ``rate * max(0, P_t - running_peak)`` each interval and the
episode sum is exactly ``rate * max_t(P_t)``. Any total that is a function of
an online-updatable statistic works the same way. And retrospective
*attribution* is the analyst's job, not the environment's: a scorer holds the
whole trajectory and may look backwards freely. Only the agent's reward
signal has to be causal.
"""

import abc
from typing import TYPE_CHECKING, ClassVar

import chex
import jax.numpy as jnp

from gll_env.types import RewardObservation, RewardState

if TYPE_CHECKING:  # pragma: no cover -- import cycle; see gll_env.types.RewardState
    from gll_env.components.environment import EnvironmentDynamics, EnvironmentState

# A trajectory of consecutive states, oldest first, length ``lookahead + 2``.
# A tuple rather than a stacked pytree: `lookahead` is small, indexing reads
# naturally, and the causal case is then literally ``(state, new_state)`` with
# no allocation.
Trajectory = tuple["EnvironmentState", ...]


@chex.dataclass(frozen=True)
class EmptyRewardState(RewardState):
    """State for a reward that carries nothing between intervals."""


@chex.dataclass(frozen=True)
class EmptyRewardObservation(RewardObservation):
    """Observation for a reward that publishes nothing.

    Contributing no fields is what keeps an existing policy's observation
    width unchanged when a stateless reward is used: only a reward that
    actually publishes a quantity changes the observation shape.
    """


class RewardDynamics(abc.ABC):
    """Abstract reward component.

    Attributes:
        lookahead: How many intervals of *realized future* the settlement for
            an interval needs before it can be computed. ``0`` -- the default
            and the only value the environment supports directly -- means the
            settlement for interval ``t`` depends on nothing after ``t``.
            A reward declaring more needs the re-timing wrapper described in
            ``docs/lookahead_rewards.md``; :class:`EnvironmentDynamics` raises
            rather than silently computing something wrong.

            Declared as a class attribute deliberately, so the environment can
            branch on it at construction time instead of under a trace.
    """

    lookahead: ClassVar[int] = 0

    @abc.abstractmethod
    def reset(self, key: chex.PRNGKey) -> RewardState:
        """Fresh state for a new episode.

        Called from :meth:`EnvironmentDynamics.reset`, so the reward's memory
        is reset with the episode by construction -- which is what stops one
        episode's history leaking into the next under an auto-reset wrapper.
        """

    @abc.abstractmethod
    def __call__(
        self,
        reward_state: RewardState,
        trajectory: Trajectory,
        dynamics: "EnvironmentDynamics",
    ) -> tuple[RewardState, chex.Array]:
        """Settle one interval.

        Args:
            reward_state: The reward's own state at the start of the interval.
            trajectory: ``lookahead + 2`` consecutive states, oldest first.
                ``trajectory[0]`` and ``trajectory[1]`` bracket the interval
                being settled; anything after is its realized future.
            dynamics: The environment model, for ratings, index maps and unit
                conversions. Passed rather than stored -- the same convention
                as ``GridObservation.normalize(grid_dynamics)`` -- so a reward
                held as a field of ``EnvironmentDynamics`` creates no reference
                cycle. Most rewards ``del`` it.

        Returns:
            The updated reward state, and the settlement for the interval
            ``trajectory[:2]`` brackets, shape ``(num_agents,)`` float32.

            The returned reward is ALWAYS the settlement for that interval.
            Any delay a mechanism wants is delay in *disclosure* -- what the
            observation reveals -- never a shift of this array, so no consumer
            ever has to re-index.
        """

    @abc.abstractmethod
    def observation(self, reward_state: RewardState) -> RewardObservation:
        """What this reward makes visible.

        Forwarded whole into ``extras["reward"]``, and available to observers.
        Use it for anything the ``(num_agents,)`` reward array cannot carry --
        most usefully a per-connection-point settlement, since connection
        points without an inverter have no agent to be attributed to and are
        otherwise invisible.
        """


class CausalReward(RewardDynamics):
    """A reward whose settlement depends on nothing after the interval.

    The base class for concrete tariffs. Implement :meth:`settle`; the
    trajectory is unpacked for you into the ``(state, new_state)`` pair every
    other component in this tree works with.
    """

    lookahead: ClassVar[int] = 0

    def __call__(
        self,
        reward_state: RewardState,
        trajectory: Trajectory,
        dynamics: "EnvironmentDynamics",
    ) -> tuple[RewardState, chex.Array]:
        if len(trajectory) != 2:
            raise ValueError(
                f"{type(self).__name__} is causal (lookahead=0) and expects exactly two "
                f"states, got {len(trajectory)}."
            )
        return self.settle(reward_state, trajectory[0], trajectory[1], dynamics)

    @abc.abstractmethod
    def settle(
        self,
        reward_state: RewardState,
        state: "EnvironmentState",
        new_state: "EnvironmentState",
        dynamics: "EnvironmentDynamics",
    ) -> tuple[RewardState, chex.Array]:
        """Settle the interval running from `state` to `new_state`.

        Returns the updated reward state and a ``(num_agents,)`` float32
        settlement.
        """


class RewardFn(abc.ABC):
    """Legacy stateless reward: ``(state, new_state) -> (num_agents,)``.

    Predates the reward becoming a component. Kept because it is the simplest
    thing that can express a reward, and because wrapping one in
    :class:`StatelessReward` is a one-liner. New rewards should subclass
    :class:`CausalReward` instead, which can carry state.
    """

    @abc.abstractmethod
    def __call__(
        self,
        state: "EnvironmentState",
        new_state: "EnvironmentState",
    ) -> chex.Array:
        """Compute per-agent reward, shape ``(num_agents,)`` float32."""


class StatelessReward(CausalReward):
    """Adapts a :class:`RewardFn` into the component interface.

    Carries no state and publishes nothing, so an environment using one has
    exactly the observation it had before the reward became a component --
    which is what keeps existing training configurations working untouched.
    """

    def __init__(self, reward_fn: RewardFn) -> None:
        self._reward_fn = reward_fn

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._reward_fn!r})"

    def reset(self, key: chex.PRNGKey) -> RewardState:
        del key
        return EmptyRewardState()

    def settle(
        self,
        reward_state: RewardState,
        state: "EnvironmentState",
        new_state: "EnvironmentState",
        dynamics: "EnvironmentDynamics",
    ) -> tuple[RewardState, chex.Array]:
        del dynamics
        reward = jnp.asarray(self._reward_fn(state, new_state), dtype=jnp.float32)
        return reward_state, reward

    def observation(self, reward_state: RewardState) -> RewardObservation:
        del reward_state
        return EmptyRewardObservation()


class BaseReward(RewardFn):
    """Placeholder reward: each agent's own realized active power injection.

    Reward shaping is a research concern separate from environment dynamics,
    so the default deliberately says nothing about what good behaviour is.
    """

    def __call__(
        self,
        state: "EnvironmentState",
        new_state: "EnvironmentState",
    ) -> chex.Array:
        del state
        return new_state.prosumer_state.inverter_state.s_inv_realized_kvah.real


def default_reward() -> RewardDynamics:
    """The reward :class:`EnvironmentDynamics` falls back to: :class:`BaseReward`."""
    return StatelessReward(BaseReward())
