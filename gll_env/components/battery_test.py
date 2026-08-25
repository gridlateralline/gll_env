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
        charge_rating_kw=jnp.array([1.0, 2.0], dtype=jnp.float32),
        discharge_rating_kw=jnp.array([2.0, 1.0], dtype=jnp.float32),
        time=DaytimeDynamics(n_steps_per_day=jnp.int32(12)),
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

    assert jnp.allclose(request_min, jnp.array([-2.0, -1.0]))
    assert jnp.allclose(request_max, jnp.array([3.0, 2.0]))


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
    assert jnp.allclose(request_min, jnp.array([-2.0, -1.5]))
    assert jnp.allclose(request_max, jnp.array([1.0, 2.0]))


def test_reset_and_observation_stay_within_expected_ranges() -> None:
    battery = build_battery()
    state = battery.reset(jr.PRNGKey(0))
    observation = battery.observation(state).normalize(battery)

    assert jnp.all(state.bat_full_kwh >= 0.0)
    assert jnp.all(state.bat_full_kwh <= battery.capacity_kwh)
    assert jnp.allclose(state.bat_free_kwh, jnp.subtract(battery.capacity_kwh, state.bat_full_kwh))
    raw_observation = battery.observation(state)
    request_min, request_max = battery.request_bounds(state.bat_request_constraint)
    assert jnp.allclose(raw_observation.bat_request_min, request_min)
    assert jnp.allclose(raw_observation.bat_request_max, request_max)
    assert not raw_observation.is_normalized
    assert jnp.all(observation.bat_free >= 0.0)
    assert jnp.all(observation.bat_free <= 1.0)
    assert jnp.all(observation.bat_full >= 0.0)
    assert jnp.all(observation.bat_full <= 1.0)


def test_zero_capacity_and_power_ratings_have_finite_observations() -> None:
    battery = BatteryDynamics(
        capacity_kwh=jnp.array([0.0], dtype=jnp.float32),
        charge_rating_kw=jnp.array([0.0], dtype=jnp.float32),
        discharge_rating_kw=jnp.array([0.0], dtype=jnp.float32),
        time=DaytimeDynamics(n_steps_per_day=jnp.int32(12)),
    )

    observation = battery.observation(battery.reset(jr.PRNGKey(1)))

    assert jnp.all(jnp.isfinite(observation.bat_realized))
    assert jnp.all(jnp.isfinite(observation.bat_request_min))
    assert jnp.all(jnp.isfinite(observation.bat_request_max))
    assert jnp.all(jnp.isfinite(observation.bat_free))
    assert jnp.all(jnp.isfinite(observation.bat_full))
    assert jnp.allclose(observation.bat_realized, 0.0)
    assert jnp.allclose(observation.bat_free, 0.0)
    assert jnp.allclose(observation.bat_full, 0.0)


def test_negative_physical_ratings_are_clamped_to_zero() -> None:

    battery = BatteryDynamics(
        capacity_kwh=jnp.array([-10.0], dtype=jnp.float32),
        charge_rating_kw=jnp.array([-1.0], dtype=jnp.float32),
        discharge_rating_kw=jnp.array([-1.0], dtype=jnp.float32),
        time=DaytimeDynamics(n_steps_per_day=jnp.int32(12)),
    )

    assert jnp.allclose(battery.capacity_kwh, 0.0)
    assert jnp.allclose(battery.charge_rating_kw, 0.0)
    assert jnp.allclose(battery.discharge_rating_kw, 0.0)


def test_state_of_charge_bookkeeping_is_conservative() -> None:
    """Energy bookkeeping across a step, from first principles.

    Three things must hold simultaneously, for any request including wildly
    out-of-range ones:

    * ``full + free == capacity`` -- free is defined as headroom, so the two
      partition a fixed capacity and cannot drift apart.
    * ``next_full - full == -realized`` -- the sign convention is positive =
      discharge, so stored energy falls by exactly what flowed out. This is
      what ties the kWh flow to the kWh stock; a rate/energy confusion in
      either one breaks it.
    * ``0 <= full <= capacity`` -- the battery never charges past full or
      discharges past empty, which is enforced by the request bounds rather
      than by a clamp on the stock itself.
    """
    battery = build_battery()
    state = battery.reset(jr.PRNGKey(4))

    for request in (0.5, -0.5, 2.0, -2.0, 0.0, 100.0, -100.0):
        previous_full_kwh = state.bat_full_kwh
        state = battery.step(state, jnp.full((battery.num_bat,), request, dtype=jnp.float32))

        assert jnp.allclose(state.bat_full_kwh + state.bat_free_kwh, battery.capacity_kwh)
        assert jnp.allclose(state.bat_full_kwh - previous_full_kwh, -state.bat_realized_kwh)
        assert jnp.all(state.bat_full_kwh >= -1e-5)
        assert jnp.all(state.bat_full_kwh <= battery.capacity_kwh + 1e-5)


def test_reset_samples_a_realized_flow_its_own_history_could_have_produced() -> None:
    """reset() fabricates a PAST-interval flow, so it is bounded by the state
    that flow led INTO, not the one it leads out of.

    A realized flow ``r`` (positive = discharge) implies the previous stock
    was ``full + r``, which must itself have been within ``[0, capacity]`` --
    giving ``r in [-full, free]``, the mirror image of the forward-looking
    request bounds. Getting this backwards would seed episodes with histories
    that no reachable state could have produced.
    """
    battery = build_battery()

    for seed in range(16):
        state = battery.reset(jr.PRNGKey(seed))
        implied_previous_full_kwh = state.bat_full_kwh + state.bat_realized_kwh

        assert jnp.all(implied_previous_full_kwh >= -1e-5)
        assert jnp.all(implied_previous_full_kwh <= battery.capacity_kwh + 1e-5)
        assert jnp.all(state.bat_realized_kwh >= -battery.peak_charge_per_step_kwh - 1e-5)
        assert jnp.all(state.bat_realized_kwh <= battery.peak_discharge_per_step_kwh + 1e-5)
