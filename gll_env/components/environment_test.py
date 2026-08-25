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

from gll_env.algorithms.newton_raphson import NewtonRaphson
from gll_env.components.battery import BatteryDynamics
from gll_env.components.day_time import DaytimeDynamics, DaytimeState
from gll_env.components.environment import EnvironmentDynamics
from gll_env.components.grid import GridDynamics
from gll_env.components.inverter import InverterDynamics
from gll_env.components.load import LoadDynamics
from gll_env.components.prosumer import ProsumerDynamics
from gll_env.components.solar import SolarDynamics


def build_environment() -> EnvironmentDynamics:
    time = DaytimeDynamics(n_steps_per_day=jnp.int32(12))
    battery = BatteryDynamics(
        capacity_kwh=jnp.array([10.0], dtype=jnp.float32),
        charge_rating_kw=jnp.array([1.0], dtype=jnp.float32),
        discharge_rating_kw=jnp.array([2.0], dtype=jnp.float32),
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
    prosumer = ProsumerDynamics(
        s_pq_max_kva=jnp.array([25.0], dtype=jnp.float32),
        inverter_id=jnp.array([0], dtype=jnp.int32),
        inverter_dynamics=inverter,
        load_dynamics=load,
        time=time,
    )
    grid = GridDynamics(
        slack_id=jnp.array([0], dtype=jnp.int32),
        pq_id=jnp.array([1], dtype=jnp.int32),
        pv_id=jnp.array([], dtype=jnp.int32),
        base_s_mva=jnp.array(1.0, dtype=jnp.float32),
        base_v_kv=jnp.array([11.0, 11.0], dtype=jnp.float32),
        admittance=jnp.array(
            [[1.0 + 2.0j, -0.5 - 0.25j], [-0.5 - 0.25j, 0.75 + 1.25j]],
            dtype=jnp.complex64,
        ),
        position=jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        nr=NewtonRaphson(),
        time=time,
    )
    return EnvironmentDynamics(prosumer=prosumer, grid=grid, time=time)


def test_normalized_constraints_are_equivalent_to_physical_constraints() -> None:
    environment = build_environment()
    time_state = DaytimeState(
        interval_start=jnp.float32(0.5), interval_end=jnp.float32(0.75), day_step=jnp.int32(2)
    )
    state = environment.reset(jr.PRNGKey(0), time_state=time_state)

    normalized_action = jnp.array([[0.25, -0.5]], dtype=jnp.float32)
    physical_request = environment._action_to_request(normalized_action)
    normalized_constraints = state.action_constraints
    physical_constraints = state.prosumer_state.s_inv_request_constraint
    scale = jnp.asarray(environment.prosumer.inverter_dynamics.s_inv_max_kvah)

    assert bool(normalized_constraints.is_feasible(normalized_action, tol=1e-5)) == bool(
        physical_constraints.is_feasible(physical_request, tol=1e-5)
    )
    assert jnp.allclose(
        normalized_action * scale[:, None],
        physical_request,
    )


def test_infeasible_normalized_requests_are_projected_and_state_remains_consistent() -> None:
    environment = build_environment()
    state = environment.reset(
        jr.PRNGKey(1),
        time_state=DaytimeState(
            interval_start=jnp.float32(0.5), interval_end=jnp.float32(0.75), day_step=jnp.int32(2)
        ),
    )
    old_key = state.key
    next_state = environment.step(state, jnp.array([[100.0, -100.0]], dtype=jnp.float32))

    realized_request = next_state.prosumer_state.inverter_state.s_inv_realized_kvah
    scale = jnp.asarray(environment.prosumer.inverter_dynamics.s_inv_max_kvah)
    realized_action = jnp.stack(
        [realized_request.real / scale, realized_request.imag / scale], axis=-1
    )
    assert bool(state.action_constraints.is_feasible(realized_action, tol=1e-3))
    assert bool(next_state.valid)
    assert int(next_state.step_count) == 1
    assert int(next_state.time_state.day_step) == 3
    assert int(next_state.prosumer_state.time_state.day_step) == 3
    assert int(next_state.prosumer_state.inverter_state.time_state.day_step) == 3
    assert int(next_state.prosumer_state.inverter_state.solar_state.time_state.day_step) == 3
    assert int(next_state.prosumer_state.load_state.time_state.day_step) == 3
    assert not bool(jnp.array_equal(next_state.key, old_key))


def test_clock_wraps_at_day_boundary() -> None:
    environment = build_environment()
    state = environment.reset(
        jr.PRNGKey(2),
        time_state=DaytimeState(
            interval_start=jnp.float32(11 / 12),
            interval_end=jnp.float32(1.0),
            day_step=jnp.int32(11),
        ),
    )

    next_state = environment.step(state, jnp.zeros((1, 2), dtype=jnp.float32))

    assert int(next_state.time_state.day_step) == 0
    assert int(next_state.prosumer_state.time_state.day_step) == 0
    assert int(next_state.prosumer_state.inverter_state.time_state.day_step) == 0
    assert int(next_state.prosumer_state.load_state.time_state.day_step) == 0


def test_every_physical_invariant_survives_a_full_day_of_out_of_range_actions() -> None:
    """End-to-end feasibility and conservation through the real action path.

    The component tests each check one layer against the constraint object it
    was handed. This drives the whole stack the way a training loop does --
    normalized action in, ``_action_to_request`` scaling, Prosumer's
    projection, Inverter's internal dispatch, the leaves, then power flow --
    and asserts the PHYSICAL invariants on the realized state, which is the
    only place a layer disagreeing with its neighbour would show up.

    Actions are drawn well outside the ``[-1, 1]`` box the agent is supposed
    to emit, in both directions, so the projection is genuinely load-bearing
    on most steps rather than a formality.
    """
    environment = build_environment()
    inverter = environment.prosumer.inverter_dynamics
    battery = inverter.battery_dynamics
    state = environment.reset(jr.PRNGKey(21))
    key = jr.PRNGKey(22)
    constraint_ever_bound = False

    for _ in range(96):  # a full simulated day
        key, subkey = jr.split(key)
        action = jr.normal(subkey, (environment.num_agents, 2), dtype=jnp.float32) * 4.0
        constraint_before = state.prosumer_state.s_inv_request_constraint

        state = environment.step(state, action)
        prosumer_state = state.prosumer_state
        inverter_state = prosumer_state.inverter_state
        battery_state = inverter_state.battery_state
        solar_state = inverter_state.solar_state

        # Feasibility: the realized flow satisfies the constraint it was
        # projected against, as a whole (halfspaces and balls together).
        realized_action = jnp.stack(
            [inverter_state.s_inv_realized_kvah.real, inverter_state.s_inv_realized_kvah.imag],
            axis=-1,
        )
        assert bool(constraint_before.is_feasible(realized_action, tol=1e-3))
        assert jnp.all(
            jnp.abs(inverter_state.s_inv_realized_kvah) <= inverter.s_inv_max_kvah + 1e-3
        )

        # Conservation: the inverter's output came from its own two sources.
        leaves_kwh = solar_state.sol_realized_kwh + battery_state.bat_realized_kwh
        assert jnp.allclose(
            inverter_state.s_inv_realized_kvah.real, leaves_kwh, rtol=1e-5, atol=1e-5
        )

        # Storage and generation stay physical.
        assert jnp.all(battery_state.bat_full_kwh >= -1e-5)
        assert jnp.all(battery_state.bat_full_kwh <= battery.capacity_kwh + 1e-5)
        assert jnp.allclose(
            battery_state.bat_full_kwh + battery_state.bat_free_kwh, battery.capacity_kwh
        )
        assert jnp.all(solar_state.sol_realized_kwh >= 0.0)

        # Net grid flow is exactly generation minus consumption.
        assert jnp.allclose(
            prosumer_state.s_pq_realized_kvah,
            inverter_state.s_inv_realized_kvah - prosumer_state.load_state.s_load_realized_kvah,
        )

        # The power flow converged, so the episode stays live.
        assert bool(state.valid)

        utilization = jnp.abs(inverter_state.s_inv_realized_kvah) / inverter.s_inv_max_kvah
        constraint_ever_bound = constraint_ever_bound or bool(jnp.any(utilization > 0.999))

    assert constraint_ever_bound, (
        "no constraint ever bound over the whole rollout, so the projection "
        "was never actually exercised"
    )
