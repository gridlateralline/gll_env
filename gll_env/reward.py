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

"""Reward functions for the ProsumerGrid environment.

Follows the Jumanji pattern (see ``connector/reward.py``):
an injectable :class:`RewardFn` ABC so the reward signal can be swapped
without subclassing :class:`ProsumerGrid`.

The default :class:`BaseReward` is a placeholder — reward shaping is a
research concern separate from environment dynamics.
"""

import abc

import chex

from gll_env.components.environment import EnvironmentState


class RewardFn(abc.ABC):
    """Abstract reward function.

    Receives the state *before* and *after* the agent action so implementations
    can reward both instantaneous outcomes (e.g. voltage deviation) and
    trajectory-level quantities (e.g. energy arbitrage profit).

    Returns per-agent rewards of shape ``(num_agents,)`` (one per inverter)
    and dtype ``float32``.
    """

    @abc.abstractmethod
    def __call__(
        self,
        state: EnvironmentState,
        new_state: EnvironmentState,
    ) -> chex.Array:
        """Compute per-agent reward.

        Args:
            state: Environment state at the *start* of the interval (before the action).
            new_state: Environment state at the *end* of the interval (after the action).

        Returns:
            Shape ``(num_agents,)`` float32.
        """


class BaseReward(RewardFn):
    """Placeholder reward: each agent's own realized active power injection."""

    def __call__(
        self,
        state: EnvironmentState,
        new_state: EnvironmentState,
    ) -> chex.Array:
        return new_state.prosumer_state.inverter_state.s_inv_realized_kvah.real
