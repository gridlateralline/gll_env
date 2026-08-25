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
    time = DaytimeDynamics(n_steps_per_day=jnp.int32(12))
    zero_battery = BatteryDynamics(
        capacity_kwh=jnp.array([0.0], dtype=jnp.float32),
        charge_rating_kw=jnp.array([0.0], dtype=jnp.float32),
        discharge_rating_kw=jnp.array([0.0], dtype=jnp.float32),
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


def test_observation_reports_net_grid_flow_in_energy_units() -> None:
    # step_duration_h == 2.0 here (12 steps/day), so any stray scaling by the
    # step duration shows up as a factor of 2, not as a no-op.
    prosumer = build_prosumer()
    state = prosumer.reset(jr.PRNGKey(3))

    s_pq_max_kvah = prosumer.s_pq_max_kvah
    s_pq_realized_kvah = (0.5 * s_pq_max_kvah - 0.25j * s_pq_max_kvah).astype(jnp.complex64)
    state = state.replace(s_pq_realized_kvah=s_pq_realized_kvah)

    observation = prosumer.observation(state)

    # Raw observation carries the stored energy through untouched, exactly as
    # the sibling Inverter/Load observations do.
    assert jnp.allclose(observation.p_pq_realized, s_pq_realized_kvah.real)
    assert jnp.allclose(observation.q_pq_realized, s_pq_realized_kvah.imag)

    # ... which is what makes normalizing by s_pq_max_kvah (an energy) land on
    # the intended fraction, inside [-1, 1].
    normalized = observation.normalize(prosumer)
    assert jnp.allclose(normalized.p_pq_realized, 0.5)
    assert jnp.allclose(normalized.q_pq_realized, -0.25)


def build_constrained_prosumer() -> ProsumerDynamics:
    """A prosumer whose GRID CONNECTION is the tightest constraint.

    ``build_prosumer``'s sizing cannot exercise the grid ball at all: its
    connection is clamped up to the load's own 20 kVA nameplate (see
    ``__post_init__``), landing far above anything the 5 kVA inverter and
    that load can produce between them, so the inverter's own ball always
    binds first. Shrinking the load rating alongside the connection is what
    lets Ball 2 actually become active.
    """
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
        daily_consumption_kwh=jnp.array([12.0], dtype=jnp.float32),
        s_load_max_kva=jnp.array([1.0], dtype=jnp.float32),
        time=time,
    )
    return ProsumerDynamics(
        s_pq_max_kva=jnp.array([1.0], dtype=jnp.float32),
        inverter_id=jnp.array([0], dtype=jnp.int32),
        inverter_dynamics=inverter,
        load_dynamics=load,
        time=time,
    )


def test_net_grid_flow_never_exceeds_the_connection_rating() -> None:
    """``|s_pq| <= s_pq_max_kvah`` for every step, under requests far outside
    any feasible region.

    This is the physical point of the augmented constraint's second ball: the
    grid connection is a real conductor with a real rating, and the net flow
    ``s_inv - s_load`` is what actually passes through it. Checked on the
    realized flow rather than on the constraint object, so it covers the
    whole path -- projection, dispatch, and the scatter onto the load axis --
    rather than just the geometry.

    Uses the deliberately connection-limited fixture and ASSERTS the ball
    actually binds. A feasibility test whose constraint never activates says
    nothing about that constraint, and with ordinary sizing this one does not
    activate: the inverter's own ball binds first at every reachable state.
    """
    prosumer = build_constrained_prosumer()
    state = prosumer.reset(jr.PRNGKey(11))
    key = jr.PRNGKey(12)
    ever_binding = False

    for _ in range(60):
        key, subkey = jr.split(key)
        state = prosumer.step(state, jr.normal(subkey, shape=(prosumer.num_inv, 2)) * 30.0)

        utilization = jnp.abs(state.s_pq_realized_kvah) / prosumer.s_pq_max_kvah
        assert jnp.all(utilization <= 1.0 + 1e-4)
        # The origin-feasibility invariant the constraint's own docstring
        # rests on: if it failed, the grid ball would exclude the origin and
        # the projection would have no feasible fallback.
        assert jnp.all(jnp.abs(state.load_state.s_load_kvah) <= prosumer.s_pq_max_kvah + 1e-4)
        assert bool(state.valid)
        ever_binding = ever_binding or bool(jnp.any(utilization > 0.999))

    assert ever_binding, (
        "the grid-connection ball never bound, so this test proved nothing "
        "about it; tighten the connection until it does"
    )
