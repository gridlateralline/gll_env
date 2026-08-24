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
from gll_env.components.solar import SolarDynamics, SolarState


def build_solar() -> SolarDynamics:
    return SolarDynamics(
        peak_power_kw=jnp.array([2.0, 4.0], dtype=jnp.float32),
        clearness_reversion=jnp.float32(0.2),
        clearness_mean=jnp.float32(0.6),
        clearness_std=jnp.float32(0.1),
        time=DaytimeDynamics(n_steps_per_day=jnp.int32(4)),
    )


def test_cosine_profile_and_power_to_energy_conversion() -> None:
    solar = build_solar()
    clearness = jnp.array([0.5, 1.0], dtype=jnp.float32)

    assert jnp.allclose(solar._available_fraction(jnp.float32(0.0), clearness), 0.0)
    assert jnp.allclose(solar._available_fraction(jnp.float32(0.5), clearness), clearness)
    assert jnp.allclose(
        solar._available_energy(jnp.float32(0.5), clearness),
        jnp.array([6.0, 24.0], dtype=jnp.float32),
    )


def test_step_clips_generation_and_advances_clearness() -> None:
    solar = build_solar()
    time_state = DaytimeState(day_progress=jnp.float32(0.5), day_step=jnp.int32(2))
    clearness = jnp.array([0.5, 1.0], dtype=jnp.float32)
    maximum = solar._available_energy(time_state.day_progress, clearness)
    state = SolarState(
        sol_realized_kwh=jnp.zeros((2,), dtype=jnp.float32),
        sol_request_constraint=solar._new_request_constraint(maximum),
        time_state=time_state,
        clearness=clearness,
        key=jr.PRNGKey(0),
    )

    next_state = solar.step(state, jnp.array([-1.0, 100.0], dtype=jnp.float32))

    assert jnp.allclose(next_state.sol_realized_kwh, jnp.array([0.0, 24.0]))
    assert jnp.all(next_state.clearness >= 0.0)
    assert jnp.all(next_state.clearness <= 1.0)
    assert int(next_state.time_state.day_step) == 3


def test_reset_returns_finite_bounded_state() -> None:
    solar = build_solar()

    state = solar.reset(
        jr.PRNGKey(1),
        time_state=DaytimeState(day_progress=jnp.float32(0.5), day_step=jnp.int32(2)),
    )
    observation = solar.observation(state)

    assert jnp.all(jnp.isfinite(state.sol_realized_kwh))
    assert jnp.all(state.sol_realized_kwh >= 0.0)
    assert jnp.all(state.sol_realized_kwh <= solar.s_sol_max_kwh)
    assert jnp.all(observation.sol_realized >= 0.0)
    assert jnp.all(observation.sol_realized <= 1.0)
    assert jnp.all(observation.sol_available >= 0.0)
    assert jnp.all(observation.sol_available <= 1.0)


def test_zero_peak_and_negative_configuration_values_are_safe() -> None:
    solar = SolarDynamics(
        peak_power_kw=jnp.array([-2.0], dtype=jnp.float32),
        clearness_reversion=jnp.float32(-1.0),
        clearness_mean=jnp.float32(2.0),
        clearness_std=jnp.float32(-1.0),
    )

    state = solar.reset(jr.PRNGKey(2))
    observation = solar.observation(state)

    assert jnp.allclose(solar.peak_power_kw, 0.0)
    assert solar.clearness_reversion > 0.0
    assert jnp.allclose(solar.clearness_mean, 1.0)
    assert jnp.allclose(solar.clearness_std, 0.0)
    assert jnp.all(jnp.isfinite(state.sol_realized_kwh))
    assert jnp.all(jnp.isfinite(observation.sol_realized))
    assert jnp.allclose(state.sol_realized_kwh, 0.0)
    assert jnp.allclose(observation.sol_realized, 0.0)
