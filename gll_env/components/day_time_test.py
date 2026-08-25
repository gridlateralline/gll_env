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
    dynamics = DaytimeDynamics(n_steps_per_day=jnp.int32(12))
    state = DaytimeState(
        day_step=jnp.int32(1), interval_start=jnp.float32(1 / 12), interval_end=jnp.float32(2 / 12)
    )

    assert dynamics.step_duration_h == 2.0
    next_state = dynamics.step(state)
    assert jnp.allclose(next_state.interval_start, 2 / 12)
    assert jnp.allclose(next_state.interval_end, 3 / 12)


def test_reset_returns_a_valid_random_step() -> None:
    dynamics = DaytimeDynamics(n_steps_per_day=jnp.int32(12))

    state = dynamics.reset(jr.PRNGKey(0))

    assert 0 <= int(state.day_step) < 12
    assert jnp.allclose(state.interval_start, state.day_step / 12)
    assert jnp.allclose(state.interval_end, (state.day_step + 1) / 12)


def test_step_wraps_to_the_start_of_the_day() -> None:
    dynamics = DaytimeDynamics(n_steps_per_day=jnp.int32(12))
    state = DaytimeState(
        day_step=jnp.int32(11), interval_start=jnp.float32(11 / 12), interval_end=jnp.float32(1.0)
    )

    next_state = dynamics.step(state)

    assert int(next_state.day_step) == 0
    assert jnp.allclose(next_state.interval_start, 0.0)
    assert jnp.allclose(next_state.interval_end, 1 / 12)


def test_previous_wraps_to_the_end_of_the_day() -> None:
    dynamics = DaytimeDynamics(n_steps_per_day=jnp.int32(12))
    state = DaytimeState(
        day_step=jnp.int32(0), interval_start=jnp.float32(0.0), interval_end=jnp.float32(1 / 12)
    )

    previous_state = dynamics.previous(state)

    assert int(previous_state.day_step) == 11
    assert jnp.allclose(previous_state.interval_start, 11 / 12)
    assert jnp.allclose(previous_state.interval_end, 1.0)


def test_observation_uses_hours_since_midnight() -> None:
    dynamics = DaytimeDynamics(n_steps_per_day=jnp.int32(12))
    state = DaytimeState(
        day_step=jnp.int32(1), interval_start=jnp.float32(1 / 12), interval_end=jnp.float32(2 / 12)
    )

    observation = dynamics.observation(state)
    normalized = observation.normalize(dynamics)

    assert not observation.is_normalized
    assert normalized.is_normalized
    assert int(observation.day_step) == 1
    assert jnp.allclose(observation.interval_start, 2.0)
    assert jnp.allclose(observation.interval_end, 4.0)
    assert jnp.allclose(normalized.interval_start, 1 / 12)
    assert jnp.allclose(normalized.interval_end, 2 / 12)
