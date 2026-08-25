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
"""Jumanji environment for prosumer-based AC distribution grids.

The default environment is ready to use::

    import jax
    import gll_env

    env = gll_env.ProsumerGrid(time_limit=96)
    state, timestep = env.reset(jax.random.PRNGKey(0))

For configuration-driven construction, pass a :class:`ConfigGenerator`::

    generator = gll_env.ConfigGenerator(
        n_steps_per_day=96,
        grid=grid_cfg,
        prosumer=prosumer_cfg,
    )
    env = gll_env.ProsumerGrid(
        generator=generator,
        observer=gll_env.MarlObserver(generator.env_dynamics),
        time_limit=96,
    )

Importing this package registers ``ProsumerGrid-v0`` with Jumanji, so it can
also be created with ``jumanji.make("ProsumerGrid-v0")`` after ``import gll_env``.
"""

import jumanji

from gll_env.env import EnvironmentState, ProsumerGrid
from gll_env.generator import ConfigGenerator, DynamicsGenerator
from gll_env.observer import MarlObserver, RawObserver
from gll_env.reward import BaseReward, RewardFn

jumanji.register(
    id="ProsumerGrid-v0",
    entry_point=f"{ProsumerGrid.__module__}:{ProsumerGrid.__name__}",
)


__all__ = [
    "BaseReward",
    "ConfigGenerator",
    "DynamicsGenerator",
    "EnvironmentState",
    "MarlObserver",
    "ProsumerGrid",
    "RawObserver",
    "RewardFn",
]
