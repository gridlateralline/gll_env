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
from gll_env.components.solar import SolarDynamics


def build_inverter() -> InverterDynamics:
    time = DaytimeDynamics(n_steps_per_day=jnp.int32(12))
    battery = BatteryDynamics(
        capacity_kwh=jnp.array([10.0, 8.0], dtype=jnp.float32),
        charge_rating_kw=jnp.array([1.0, 2.0], dtype=jnp.float32),
        discharge_rating_kw=jnp.array([2.0, 1.0], dtype=jnp.float32),
        time=time,
    )
    solar = SolarDynamics(
        peak_power_kw=jnp.array([2.0, 4.0], dtype=jnp.float32),
        clearness_reversion=jnp.float32(0.2),
        clearness_mean=jnp.float32(0.6),
        clearness_std=jnp.float32(0.1),
        time=time,
    )
    return InverterDynamics(
        s_inv_max_kva=jnp.array([5.0, 5.0], dtype=jnp.float32),
        battery_dynamics=battery,
        solar_dynamics=solar,
        time=time,
    )


def test_step_is_internally_consistent_for_arbitrary_requests() -> None:
    """Drive step() with many out-of-range requests and check Inverter's contract."""
    inverter = build_inverter()
    state = inverter.reset(jr.PRNGKey(0))
    key = jr.PRNGKey(1)

    for _ in range(20):
        key, subkey = jr.split(key)
        # Deliberately far outside any feasible region to exercise clipping.
        request = jr.normal(subkey, shape=(inverter.num_inv, 2)) * 20.0

        constraint_before = state.s_inv_request_constraint
        next_state = inverter.step(state, request)

        s_inv = jnp.stack(
            [next_state.s_inv_realized_kvah.real, next_state.s_inv_realized_kvah.imag],
            axis=-1,
        )
        assert bool(constraint_before.is_feasible(s_inv, tol=1e-3))

        request_min, request_max = inverter.request_bounds(next_state.s_inv_request_constraint)
        assert jnp.all(request_min <= request_max + 1e-3)
        assert jnp.all(jnp.asarray(next_state.s_inv_request_constraint.ball_radius) >= 0.0)

        state = next_state


def test_dispatch_maximizes_solar_before_using_battery() -> None:
    """Direct closed-form check of the solar/battery split derivation."""
    inverter = build_inverter()
    time_state = inverter.time.reset(jr.PRNGKey(2))
    state = inverter.reset(jr.PRNGKey(3), time_state=time_state)

    sol_min, sol_max = inverter.solar_dynamics.request_bounds(
        state.solar_state.sol_request_constraint
    )
    bat_min, _ = inverter.battery_dynamics.request_bounds(
        state.battery_state.bat_request_constraint
    )

    # A request comfortably inside the box: solar should cover as much as
    # possible before battery contributes anything.
    p_inv_request = jnp.minimum(sol_max, jnp.full_like(sol_max, 0.1)) + bat_min
    request = jnp.stack([p_inv_request, jnp.zeros_like(p_inv_request)], axis=-1)

    next_state = inverter.step(state, request)

    expected_sol = jnp.clip(p_inv_request - bat_min, sol_min, sol_max)
    expected_bat = p_inv_request - expected_sol
    assert jnp.allclose(next_state.solar_state.sol_realized_kwh, expected_sol, atol=1e-3)
    assert jnp.allclose(next_state.battery_state.bat_realized_kwh, expected_bat, atol=1e-3)


def test_zero_rating_has_finite_observation() -> None:
    time = DaytimeDynamics(n_steps_per_day=jnp.int32(12))
    inverter = InverterDynamics(
        s_inv_max_kva=jnp.array([0.0], dtype=jnp.float32),
        battery_dynamics=BatteryDynamics(
            capacity_kwh=jnp.array([0.0], dtype=jnp.float32),
            charge_rating_kw=jnp.array([0.0], dtype=jnp.float32),
            discharge_rating_kw=jnp.array([0.0], dtype=jnp.float32),
            time=time,
        ),
        solar_dynamics=SolarDynamics(
            peak_power_kw=jnp.array([0.0], dtype=jnp.float32),
            time=time,
        ),
        time=time,
    )

    state = inverter.reset(jr.PRNGKey(4))
    observation = inverter.observation(state)

    assert jnp.all(jnp.isfinite(observation.p_inv_realized))
    assert jnp.all(jnp.isfinite(observation.q_inv_realized))
    assert jnp.all(jnp.isfinite(observation.p_inv_min))
    assert jnp.all(jnp.isfinite(observation.p_inv_max))
    assert jnp.allclose(observation.p_inv_realized, 0.0)
    assert jnp.allclose(observation.p_inv_min, 0.0)
    assert jnp.allclose(observation.p_inv_max, 0.0)


def test_negative_rating_is_clamped_to_zero() -> None:
    time = DaytimeDynamics(n_steps_per_day=jnp.int32(12))
    inverter = InverterDynamics(
        s_inv_max_kva=jnp.array([-5.0], dtype=jnp.float32),
        battery_dynamics=BatteryDynamics(
            capacity_kwh=jnp.array([1.0], dtype=jnp.float32),
            charge_rating_kw=jnp.array([1.0], dtype=jnp.float32),
            discharge_rating_kw=jnp.array([1.0], dtype=jnp.float32),
            time=time,
        ),
        solar_dynamics=SolarDynamics(peak_power_kw=jnp.array([1.0], dtype=jnp.float32), time=time),
        time=time,
    )

    assert jnp.allclose(inverter.s_inv_max_kva, 0.0)


def test_dispatch_conserves_energy_across_the_internal_split() -> None:
    """First law at the inverter node: whatever the inverter puts out over
    the interval came from exactly two places, so
    ``p_inv_realized == sol_realized + bat_realized`` must hold EXACTLY --
    not to a tolerance.

    step()'s derivation relies on both clips in the solar/battery split being
    provable no-ops for a feasible ``p_inv_kwh``, which is what lets it keep
    ``s_inv_realized_kvah`` from the projected request instead of
    recomputing it from the leaves. If that reasoning ever breaks, the two
    diverge silently: ``s_inv_realized_kvah`` would keep reporting a flow the
    leaves never actually supplied. Driven with far-out-of-range requests in
    both directions so the bounds are genuinely active.

    Exact in exact arithmetic, but asserted to a float32-epsilon-scaled
    tolerance rather than bit-equality: the split computes
    ``sol = clip(p_inv - bat_min, ...)`` and then ``bat = p_inv - sol``, and
    those two subtractions round independently, so the residual lands at
    about 1 ULP of the operands. A genuine dispatch error -- a clip that is
    no longer a no-op, or a rate/energy mixup in one leaf -- is O(kWh), many
    orders of magnitude above that, so the tolerance costs no sensitivity.
    """
    inverter = build_inverter()
    state = inverter.reset(jr.PRNGKey(4))
    key = jr.PRNGKey(5)

    for _ in range(20):
        key, subkey = jr.split(key)
        request = jr.normal(subkey, shape=(inverter.num_inv, 2)) * 20.0
        state = inverter.step(state, request)

        leaves_kwh = state.solar_state.sol_realized_kwh + state.battery_state.bat_realized_kwh
        p_inv_kwh = state.s_inv_realized_kvah.real
        tolerance = 8.0 * jnp.finfo(jnp.float32).eps * jnp.maximum(jnp.abs(p_inv_kwh), 1.0)
        assert jnp.all(jnp.abs(p_inv_kwh - leaves_kwh) <= tolerance), (
            "p_inv_realized must equal sol_realized + bat_realized; "
            f"got {p_inv_kwh} vs {leaves_kwh}"
        )
        # Solar is generation-only and never exceeds what was available.
        assert jnp.all(state.solar_state.sol_realized_kwh >= 0.0)


def test_apparent_energy_never_exceeds_the_inverter_nameplate() -> None:
    """|S| <= s_inv_max_kvah, i.e. the nameplate apparent-power rating
    converted to this interval's energy budget -- the ball constraint, read
    back off the realized flow rather than off the constraint object.
    """
    inverter = build_inverter()
    state = inverter.reset(jr.PRNGKey(6))
    key = jr.PRNGKey(7)

    for _ in range(20):
        key, subkey = jr.split(key)
        state = inverter.step(state, jr.normal(subkey, shape=(inverter.num_inv, 2)) * 20.0)
        assert jnp.all(jnp.abs(state.s_inv_realized_kvah) <= inverter.s_inv_max_kvah + 1e-3)
