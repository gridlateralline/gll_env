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

from gll_env.components.battery import BatteryDynamics, BatteryState
from gll_env.components.day_time import DaytimeDynamics


def build_battery() -> BatteryDynamics:
    return BatteryDynamics(
        capacity_kwh=jnp.array([10.0, 8.0], dtype=jnp.float32),
        peak_charge_kw=jnp.array([1.0, 2.0], dtype=jnp.float32),
        peak_discharge_kw=jnp.array([2.0, 1.0], dtype=jnp.float32),
        time=DaytimeDynamics(n_steps_per_day=jnp.int32(4)),
    )


def build_state(
    battery: BatteryDynamics,
    full_kwh: jnp.ndarray,
    realized_kwh: jnp.ndarray | None = None,
) -> BatteryState:
    free_kwh = battery.capacity_kwh - full_kwh
    if realized_kwh is None:
        realized_kwh = jnp.zeros_like(full_kwh)
    return BatteryState(
        bat_realized_kwh=realized_kwh,
        bat_request_constraint=battery._new_request_constraint(e_free=free_kwh, e_full=full_kwh),
        bat_free_kwh=free_kwh,
        bat_full_kwh=full_kwh,
    )


def test_request_bounds_are_limited_by_energy_and_power() -> None:
    battery = build_battery()
    state = build_state(battery, jnp.array([3.0, 7.0], dtype=jnp.float32))

    request_min, request_max = battery.request_bounds(state.bat_request_constraint)

    assert jnp.allclose(request_min, jnp.array([-6.0, -1.0]))
    assert jnp.allclose(request_max, jnp.array([3.0, 6.0]))


def test_step_clips_requests_and_updates_storage() -> None:
    battery = build_battery()
    state = build_state(battery, jnp.array([3.0, 7.0], dtype=jnp.float32))

    next_state = battery.step(state, jnp.array([100.0, -100.0], dtype=jnp.float32))

    expected_realized = jnp.array([3.0, -1.0], dtype=jnp.float32)
    expected_full = jnp.array([0.0, 8.0], dtype=jnp.float32)
    assert jnp.allclose(next_state.bat_realized_kwh, expected_realized)
    assert jnp.allclose(next_state.bat_full_kwh, expected_full)
    assert jnp.allclose(next_state.bat_free_kwh, battery.capacity_kwh - expected_full)


def test_step_recomputes_next_request_bounds() -> None:
    battery = build_battery()
    state = build_state(battery, jnp.array([3.0, 7.0], dtype=jnp.float32))

    next_state = battery.step(state, jnp.array([2.0, 0.5], dtype=jnp.float32))
    request_min, request_max = battery.request_bounds(next_state.bat_request_constraint)

    assert jnp.allclose(next_state.bat_full_kwh, jnp.array([1.0, 6.5]))
    assert jnp.allclose(request_min, jnp.array([-6.0, -1.5]))
    assert jnp.allclose(request_max, jnp.array([1.0, 6.0]))


def test_reset_and_observation_stay_within_expected_ranges() -> None:
    battery = build_battery()
    state = battery.reset(jr.PRNGKey(0))
    observation = battery.observation(state)

    assert jnp.all(state.bat_full_kwh >= 0.0)
    assert jnp.all(state.bat_full_kwh <= battery.capacity_kwh)
    assert jnp.allclose(state.bat_free_kwh, jnp.subtract(battery.capacity_kwh, state.bat_full_kwh))
    assert jnp.all(observation.bat_free >= 0.0)
    assert jnp.all(observation.bat_free <= 1.0)
    assert jnp.all(observation.bat_full >= 0.0)
    assert jnp.all(observation.bat_full <= 1.0)


def test_zero_capacity_and_power_ratings_have_finite_observations() -> None:
    battery = BatteryDynamics(
        capacity_kwh=jnp.array([0.0], dtype=jnp.float32),
        peak_charge_kw=jnp.array([0.0], dtype=jnp.float32),
        peak_discharge_kw=jnp.array([0.0], dtype=jnp.float32),
        time=DaytimeDynamics(n_steps_per_day=jnp.int32(4)),
    )

    observation = battery.observation(battery.reset(jr.PRNGKey(1)))

    assert jnp.all(jnp.isfinite(observation.bat_realized))
    assert jnp.all(jnp.isfinite(observation.bat_max_charge))
    assert jnp.all(jnp.isfinite(observation.bat_max_discharge))
    assert jnp.all(jnp.isfinite(observation.bat_free))
    assert jnp.all(jnp.isfinite(observation.bat_full))
    assert jnp.allclose(observation.bat_realized, 0.0)
    assert jnp.allclose(observation.bat_free, 0.0)
    assert jnp.allclose(observation.bat_full, 0.0)


def test_negative_physical_ratings_are_clamped_to_zero() -> None:

    battery = BatteryDynamics(
        capacity_kwh=jnp.array([-10.0], dtype=jnp.float32),
        peak_charge_kw=jnp.array([-1.0], dtype=jnp.float32),
        peak_discharge_kw=jnp.array([-1.0], dtype=jnp.float32),
        time=DaytimeDynamics(n_steps_per_day=jnp.int32(4)),
    )

    assert jnp.allclose(battery.capacity_kwh, 0.0)
    assert jnp.allclose(battery.peak_charge_kw, 0.0)
    assert jnp.allclose(battery.peak_discharge_kw, 0.0)
