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

from dataclasses import field
from functools import cached_property

import chex
import jax.numpy as jnp
import jax.random as jr


@chex.dataclass(frozen=True)
class DaytimeState:
    day_progress: chex.Numeric  # float32 in [0, 1]
    day_step: chex.Numeric  # int32


@chex.dataclass(frozen=True)
class DaytimeDynamics:
    n_steps_per_day: chex.Numeric = field(
        default_factory=lambda: jnp.int32(24 / 0.25)
    )  # 15-minute intervals

    def __post_init__(self) -> None:
        chex.assert_shape(self.n_steps_per_day, ())
        chex.assert_type(self.n_steps_per_day, jnp.int32)

    @cached_property
    def step_duration_h(self) -> chex.Numeric:
        return 24.0 / self.n_steps_per_day

    def _day_step_to_progress(self, day_step: chex.Numeric) -> chex.Numeric:
        # Convert day_step to a fraction of the day in [0, 1].
        # The +0.5 offset represents the midpoint of the comming interval.
        return jnp.float32(day_step + 0.5) / self.n_steps_per_day

    def reset(self, key: chex.PRNGKey) -> DaytimeState:
        day_step = jr.randint(
            key=key,
            shape=(),
            minval=0,
            maxval=self.n_steps_per_day,
            dtype=jnp.int32,
        )
        day_progress = self._day_step_to_progress(day_step)
        return DaytimeState(day_progress=day_progress, day_step=day_step)

    def previous(self, state: DaytimeState) -> DaytimeState:
        day_step_prev = jnp.mod(state.day_step - 1, self.n_steps_per_day)
        day_progress_prev = self._day_step_to_progress(day_step_prev)
        return DaytimeState(day_progress=day_progress_prev, day_step=day_step_prev)

    def step(self, state: DaytimeState) -> DaytimeState:
        day_step = jnp.mod(state.day_step + 1, self.n_steps_per_day)
        day_progress = self._day_step_to_progress(day_step)
        return DaytimeState(day_progress=day_progress, day_step=day_step)
