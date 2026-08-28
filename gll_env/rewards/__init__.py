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

A reward is a *rule*, not physics -- the same category as
:mod:`gll_env.grid_codes`, which is why neither lives in ``components/``.
What it does borrow from ``components/`` is the dynamics/state split: see
:class:`~gll_env.rewards.base.RewardDynamics`.

Concrete tariffs subclass :class:`~gll_env.rewards.base.CausalReward`, live in
their own module here, and are selected from config through
:func:`gll_env.factories.reward_fn`.
"""

from gll_env.rewards.base import (
    BaseReward,
    CausalReward,
    EmptyRewardObservation,
    EmptyRewardState,
    RewardDynamics,
    RewardFn,
    StatelessReward,
    default_reward,
)
from gll_env.rewards.leg import (
    LegRewardObservation,
    LegRewardState,
    LegSettlementReward,
    Payments,
)

__all__ = [
    "BaseReward",
    "CausalReward",
    "EmptyRewardObservation",
    "EmptyRewardState",
    "LegRewardObservation",
    "LegRewardState",
    "LegSettlementReward",
    "Payments",
    "RewardDynamics",
    "RewardFn",
    "StatelessReward",
    "default_reward",
]
