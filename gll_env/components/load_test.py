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
from gll_env.components.load import LoadDynamics, LoadState


def build_load() -> LoadDynamics:
    return LoadDynamics(
        daily_consumption_kwh=jnp.array([96.0, 48.0], dtype=jnp.float32),
        s_load_max_kva=jnp.array([20.0, 20.0], dtype=jnp.float32),
        load_factor_reversion=jnp.float32(0.2),
        load_factor_std=jnp.float32(0.1),
        power_factor=jnp.float32(0.8),
        time=DaytimeDynamics(n_steps_per_day=jnp.int32(96)),
    )


def test_profile_energy_and_reactive_energy_use_correct_units() -> None:
    load = build_load()
    progress = jnp.float32(0.5)

    active_kwh = load._new_p_load_kwh(progress, jnp.array([1.0, 1.0], dtype=jnp.float32))
    apparent_kvah = load._new_s_load_kvah(progress, jnp.array([1.0, 1.0], dtype=jnp.float32))

    assert jnp.allclose(apparent_kvah.real, active_kwh * load.time.step_duration_h / 0.25)
    assert jnp.all(apparent_kvah.imag > 0.0)
    assert jnp.allclose(apparent_kvah.imag / apparent_kvah.real, load._reactive_to_active_ratio)


def test_step_realizes_previous_forecast_and_clamps_load_factor() -> None:
    load = build_load()
    time_state = DaytimeState(
        interval_start=jnp.float32(0.5),
        interval_end=jnp.float32(49.0 / 96.0),
        day_step=jnp.int32(48),
    )
    forecast = load._new_s_load_kvah(time_state.interval_midpoint, jnp.array([1.0, 1.0]))
    state = LoadState(
        s_load_realized_kvah=jnp.zeros((2,), dtype=jnp.complex64),
        s_load_kvah=forecast,
        load_factor=jnp.array([10.0, -10.0], dtype=jnp.float32),
        time_state=time_state,
        key=jr.PRNGKey(0),
    )

    next_state = load.step(state)

    assert jnp.allclose(next_state.s_load_realized_kvah, forecast)
    assert jnp.all(next_state.load_factor >= 0.0)
    assert jnp.all(next_state.load_factor <= load._load_factor_max)
    assert next_state.time_state.day_step == 49


def test_reset_returns_a_consistent_finite_state() -> None:
    load = build_load()

    state = load.reset(
        jr.PRNGKey(1),
        time_state=DaytimeState(
            interval_start=jnp.float32(0.25),
            interval_end=jnp.float32(25.0 / 96.0),
            day_step=jnp.int32(24),
        ),
    )
    observation = load.observation(state)

    assert jnp.all(jnp.isfinite(state.s_load_kvah))
    assert jnp.all(state.s_load_kvah.real >= 0.0)
    assert jnp.all(state.s_load_kvah.imag >= 0.0)
    assert jnp.all(jnp.isfinite(observation.p_load_forecast))
    assert jnp.all(observation.p_load_forecast >= 0.0)
    assert jnp.all(observation.p_load_forecast <= 1.0)


def test_zero_load_has_finite_zero_observation() -> None:
    load = LoadDynamics(
        daily_consumption_kwh=jnp.array([0.0], dtype=jnp.float32),
        s_load_max_kva=jnp.array([0.0], dtype=jnp.float32),
        time=DaytimeDynamics(n_steps_per_day=jnp.int32(96)),
    )

    state = load.reset(jr.PRNGKey(2))
    observation = load.observation(state)

    assert jnp.all(jnp.isfinite(state.s_load_kvah))
    assert jnp.all(jnp.isfinite(observation.p_load_forecast))
    assert jnp.all(jnp.isfinite(observation.q_load_forecast))
    assert jnp.all(jnp.isfinite(observation.load_factor))
    assert jnp.allclose(state.s_load_kvah, 0.0)
    assert jnp.allclose(observation.p_load_forecast, 0.0)
    assert jnp.allclose(observation.q_load_forecast, 0.0)
    assert jnp.allclose(observation.load_factor, 0.0)


def test_negative_configuration_values_are_clamped() -> None:
    load = LoadDynamics(
        daily_consumption_kwh=jnp.array([-1.0], dtype=jnp.float32),
        s_load_max_kva=jnp.array([-1.0], dtype=jnp.float32),
        load_factor_reversion=jnp.float32(-1.0),
        load_factor_std=jnp.float32(-1.0),
        power_factor=jnp.float32(-1.0),
    )

    assert jnp.allclose(load.daily_consumption_kwh, 0.0)
    assert jnp.allclose(load.s_load_max_kva, 0.0)
    assert load.load_factor_reversion > 0.0
    assert jnp.allclose(load.load_factor_std, 0.0)
    assert load.power_factor > 0.0
