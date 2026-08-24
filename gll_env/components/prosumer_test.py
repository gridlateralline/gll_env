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

from gll_env.components.battery import BatteryDynamics
from gll_env.components.day_time import DaytimeDynamics
from gll_env.components.inverter import InverterDynamics
from gll_env.components.load import LoadDynamics
from gll_env.components.prosumer import ProsumerDynamics
from gll_env.components.solar import SolarDynamics


def build_prosumer(s_pq_max_kva: jnp.ndarray | None = None) -> ProsumerDynamics:
    time = DaytimeDynamics(n_steps_per_day=jnp.int32(4))
    battery = BatteryDynamics(
        capacity_kwh=jnp.array([10.0], dtype=jnp.float32),
        peak_charge_kw=jnp.array([1.0], dtype=jnp.float32),
        peak_discharge_kw=jnp.array([2.0], dtype=jnp.float32),
        time=time,
    )
    solar = SolarDynamics(peak_power_kw=jnp.array([2.0], dtype=jnp.float32), time=time)
    inverter = InverterDynamics(
        s_inv_max_kva=jnp.array([5.0], dtype=jnp.float32),
        battery_dynamics=battery,
        solar_dynamics=solar,
        time=time,
    )
    load = LoadDynamics(
        daily_consumption_kwh=jnp.array([96.0], dtype=jnp.float32),
        s_load_max_kva=jnp.array([20.0], dtype=jnp.float32),
        time=time,
    )
    if s_pq_max_kva is None:
        s_pq_max_kva = jnp.array([25.0], dtype=jnp.float32)
    return ProsumerDynamics(
        s_pq_max_kva=s_pq_max_kva,
        inverter_id=jnp.array([0], dtype=jnp.int32),
        inverter_dynamics=inverter,
        load_dynamics=load,
        time=time,
    )


def test_undersized_grid_connection_is_clamped_up_to_load_rating() -> None:
    prosumer = build_prosumer(s_pq_max_kva=jnp.array([1.0], dtype=jnp.float32))

    assert jnp.allclose(prosumer.s_pq_max_kva, prosumer.load_dynamics.s_load_max_kva)


def test_negative_grid_connection_is_clamped_to_load_rating() -> None:
    prosumer = build_prosumer(s_pq_max_kva=jnp.array([-5.0], dtype=jnp.float32))

    assert jnp.allclose(prosumer.s_pq_max_kva, prosumer.load_dynamics.s_load_max_kva)
    assert jnp.all(prosumer.s_pq_max_kva >= 0.0)


def test_zero_rating_prosumer_has_finite_observation() -> None:
    time = DaytimeDynamics(n_steps_per_day=jnp.int32(4))
    zero_battery = BatteryDynamics(
        capacity_kwh=jnp.array([0.0], dtype=jnp.float32),
        peak_charge_kw=jnp.array([0.0], dtype=jnp.float32),
        peak_discharge_kw=jnp.array([0.0], dtype=jnp.float32),
        time=time,
    )
    zero_solar = SolarDynamics(peak_power_kw=jnp.array([0.0], dtype=jnp.float32), time=time)
    zero_inverter = InverterDynamics(
        s_inv_max_kva=jnp.array([0.0], dtype=jnp.float32),
        battery_dynamics=zero_battery,
        solar_dynamics=zero_solar,
        time=time,
    )
    zero_load = LoadDynamics(
        daily_consumption_kwh=jnp.array([0.0], dtype=jnp.float32),
        s_load_max_kva=jnp.array([0.0], dtype=jnp.float32),
        time=time,
    )
    prosumer = ProsumerDynamics(
        s_pq_max_kva=jnp.array([0.0], dtype=jnp.float32),
        inverter_id=jnp.array([0], dtype=jnp.int32),
        inverter_dynamics=zero_inverter,
        load_dynamics=zero_load,
        time=time,
    )

    state = prosumer.reset(jr.PRNGKey(0))
    observation = prosumer.observation(state)

    assert jnp.all(jnp.isfinite(observation.p_pq_realized))
    assert jnp.all(jnp.isfinite(observation.q_pq_realized))
    assert jnp.allclose(observation.p_pq_realized, 0.0)
    assert jnp.allclose(observation.q_pq_realized, 0.0)


def test_step_reports_net_grid_injection_as_generation_minus_load() -> None:
    prosumer = build_prosumer()
    state = prosumer.reset(jr.PRNGKey(1))

    next_state = prosumer.step(state, jnp.array([[3.0, 0.0]], dtype=jnp.float32))

    expected = jnp.subtract(
        next_state.inverter_state.s_inv_realized_kvah, next_state.load_state.s_load_realized_kvah
    )
    assert jnp.allclose(next_state.s_pq_realized_kvah, expected)


def test_infeasible_request_is_projected_into_the_augmented_constraint() -> None:
    prosumer = build_prosumer()
    state = prosumer.reset(jr.PRNGKey(2))

    next_state = prosumer.step(state, jnp.array([[100.0, -100.0]], dtype=jnp.float32))
    realized_request = next_state.inverter_state.s_inv_realized_kvah
    realized_action = jnp.stack([realized_request.real, realized_request.imag], axis=-1)

    assert bool(state.s_inv_request_constraint.is_feasible(realized_action, tol=1e-3))
    assert bool(next_state.valid)
