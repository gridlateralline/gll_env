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

"""Tests for the config-driven factories.

These cover the wiring between config keys and dataclass fields -- the seam
where a field rename on one side leaves the other side referring to a name
that no longer exists. Nothing else in the suite constructs components
through the factories, so an unreachable-in-tests keyword can sit here
looking correct indefinitely.
"""

import jax.numpy as jnp
from omegaconf import OmegaConf

from gll_env.factories import daytime_dynamics, grid_dynamics


def test_grid_accepts_the_documented_optional_deviation_key() -> None:
    """``v_bus_deviation_pu`` is documented as an optional grid config key,
    so supplying it must build a grid -- and must actually reach the
    ``voltage_deviation_ref_pu`` field it configures, not just fail to raise.
    """
    config = OmegaConf.create({"grid_model": "cigre_lv_consumer", "v_bus_deviation_pu": 0.07})

    grid = grid_dynamics(config, time=daytime_dynamics(96))

    assert jnp.allclose(grid.voltage_deviation_ref_pu, 0.07)


def test_grid_falls_back_to_the_default_deviation_reference() -> None:
    config = OmegaConf.create({"grid_model": "cigre_lv_consumer"})

    grid = grid_dynamics(config, time=daytime_dynamics(96))

    assert float(grid.voltage_deviation_ref_pu) > 0.0
