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

"""Scenario generators for the ProsumerGrid environment.

Mirrors Jumanji's generator pattern: a plain
callable class that takes a PRNG key and returns a fresh initial
:class:`~gll_env.components.environment.EnvironmentState`.

Scenario variation is achieved by composing different leaf components
(:class:`~gll_env.components.battery.BatteryDynamics`,
:class:`~gll_env.components.solar.SolarDynamics`,
:class:`~gll_env.components.load.LoadDynamics`) inside
:class:`~gll_env.components.environment.EnvironmentDynamics`.
"""

import chex
from omegaconf import DictConfig, OmegaConf

from gll_env.components.environment import EnvironmentDynamics, EnvironmentState


class DynamicsGenerator:
    """Wraps an :class:`EnvironmentDynamics` and produces fresh initial states.

    Construct directly when a model is already available, via :meth:`default`
    for zero-config standalone use, or via :class:`ConfigGenerator` for the
    typical Hydra/Mava training path.

    Examples:
        Zero-config standalone use::

            generator = DynamicsGenerator.default()
            env = ProsumerGrid(generator=generator, time_limit=96)

        Configuration-driven construction::

            generator = ConfigGenerator(
                n_steps_per_day=96,
                grid=grid_cfg,
                prosumer=prosumer_cfg,
            )
            env = ProsumerGrid(generator=generator, time_limit=96)
    """

    def __init__(self, env_dynamics: EnvironmentDynamics) -> None:
        self.env_dynamics = env_dynamics

    @classmethod
    def default(cls) -> "DynamicsGenerator":
        """A small, self-contained scenario for standalone use.

        Built from the bundled ``cigre_lv_consumer`` asset with modest,
        hardcoded prosumer/battery/solar/load sizing — no OmegaConf config
        authoring needed. This is what :class:`~components.environment.ProsumerGrid` falls back
        to when constructed with no ``generator`` (including via
        ``jumanji.make("ProsumerGrid-v0")``), matching every other Jumanji
        environment's zero-argument ergonomics. Mava never uses this path —
        it always builds and passes its own generator (see
        :class:`ConfigGenerator`).
        """
        # Keep the default factory out of module import time.
        from gll_env.factories import default_environment_model

        return cls(default_environment_model())

    @property
    def num_agents(self) -> int:
        """Number of prosumer agents (= number of inverters)."""
        return self.env_dynamics.num_agents

    def __call__(self, key: chex.PRNGKey) -> EnvironmentState:
        """Sample a randomised initial :class:`EnvironmentState`."""
        return self.env_dynamics.reset(key)


class ConfigGenerator(DynamicsGenerator):
    """Hydra-compatible generator constructor used by Mava.

    Mava builds Jumanji generators as `Generator(**task_config)`. This class
    mirrors that constructor shape and delegates model construction to the
    factories.
    """

    def __init__(self, n_steps_per_day: int, grid: DictConfig, prosumer: DictConfig) -> None:
        # Keep the factory import lazy for callers that construct a generator
        # directly or use the default scenario.
        from gll_env.factories import environment_model

        task_config = OmegaConf.create(
            {
                "n_steps_per_day": n_steps_per_day,
                "grid": grid,
                "prosumer": prosumer,
            }
        )
        env_model = environment_model(task_config)
        super().__init__(env_model)
