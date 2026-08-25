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

:class:`~gll_env.rewards.base.RewardFn` is the injectable ABC; every concrete
reward lives in its own module here and is selected from config through
:func:`gll_env.factories.reward_fn`.
"""

from gll_env.rewards.base import BaseReward, RewardFn
from gll_env.rewards.leg import LegSettlementReward, Payments

__all__ = [
    "BaseReward",
    "LegSettlementReward",
    "Payments",
    "RewardFn",
]
