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

import jax.numpy as jnp
import jax.random as jr

from gll_env.components.day_time import DaytimeDynamics, DaytimeState


def test_step_duration_and_progress() -> None:
    dynamics = DaytimeDynamics(n_steps_per_day=jnp.int32(4))
    state = DaytimeState(day_step=jnp.int32(1), day_progress=jnp.float32(0.375))

    assert dynamics.step_duration_h == 6.0
    assert jnp.allclose(dynamics.step(state).day_progress, 0.625)


def test_reset_returns_a_valid_random_step() -> None:
    dynamics = DaytimeDynamics(n_steps_per_day=jnp.int32(4))

    state = dynamics.reset(jr.PRNGKey(0))

    assert 0 <= int(state.day_step) < 4
    assert jnp.allclose(state.day_progress, (state.day_step + 0.5) / 4)


def test_step_wraps_to_the_start_of_the_day() -> None:
    dynamics = DaytimeDynamics(n_steps_per_day=jnp.int32(4))
    state = DaytimeState(day_step=jnp.int32(3), day_progress=jnp.float32(0.875))

    next_state = dynamics.step(state)

    assert int(next_state.day_step) == 0
    assert jnp.allclose(next_state.day_progress, 0.125)


def test_previous_wraps_to_the_end_of_the_day() -> None:
    dynamics = DaytimeDynamics(n_steps_per_day=jnp.int32(4))
    state = DaytimeState(day_step=jnp.int32(0), day_progress=jnp.float32(0.125))

    previous_state = dynamics.previous(state)

    assert int(previous_state.day_step) == 3
    assert jnp.allclose(previous_state.day_progress, 0.875)
