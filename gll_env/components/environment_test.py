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

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from omegaconf import OmegaConf

from gll_env.algorithms.newton_raphson import NewtonRaphson
from gll_env.components.battery import BatteryDynamics
from gll_env.components.day_time import DaytimeDynamics, DaytimeState
from gll_env.components.environment import EnvironmentDynamics
from gll_env.components.grid import GridDynamics
from gll_env.components.inverter import InverterDynamics
from gll_env.components.load import LoadDynamics
from gll_env.components.prosumer import ProsumerDynamics
from gll_env.components.solar import SolarDynamics
from gll_env.factories import environment_model
from gll_env.types import ActionConstraints


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
    physical_request = environment._to_request(normalized_action, state.q_setpoint_kvarh)
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
    next_state, _ = environment.step(state, jnp.array([[100.0, -100.0]], dtype=jnp.float32))

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

    next_state, _ = environment.step(state, jnp.zeros((1, 2), dtype=jnp.float32))

    assert int(next_state.time_state.day_step) == 0
    assert int(next_state.prosumer_state.time_state.day_step) == 0
    assert int(next_state.prosumer_state.inverter_state.time_state.day_step) == 0
    assert int(next_state.prosumer_state.load_state.time_state.day_step) == 0


def test_every_physical_invariant_survives_a_full_day_of_out_of_range_actions() -> None:
    """End-to-end feasibility and conservation through the real action path.

    The component tests each check one layer against the constraint object it
    was handed. This drives the whole stack the way a training loop does --
    normalized action in, ``_to_request`` scaling, Prosumer's
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

        state, _ = environment.step(state, action)
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


# ---------------------------------------------------------------------------
# Q(U) grid code -- action_dim 1
# ---------------------------------------------------------------------------


def build_swiss_lv_environment(
    s_pq_max_kva: float = 20.0,
    s_inv_max_kva: float = 15.0,
    q_max_kvar: float | None = None,
) -> EnvironmentDynamics:
    """The default scenario with the Swiss LV code binding the reactive axis.

    Parameters let a test push the connection into regimes the default sizing
    never reaches -- an inverter far larger than the connection it hangs off,
    or a Q_max the connection cannot possibly carry.
    """
    config: dict = {
        "n_steps_per_day": 96,
        "grid": {"grid_model": "cigre_lv_consumer"},
        "grid_code": {"name": "swiss_lv"},
        "prosumer": {
            "s_pq_max_kVA": s_pq_max_kva,
            "load": {"daily_consumption_kWh": 15.0, "s_load_max_kVA": 15.0},
            "inverter": {
                "s_inv_max_kVA": s_inv_max_kva,
                "battery": {
                    "capacity_kWh": 10.0,
                    "peak_charge_kW": 5.0,
                    "peak_discharge_kW": 5.0,
                },
                "solar": {"peak_power_kW": 8.0},
            },
        },
    }
    if q_max_kvar is not None:
        config["grid_code"]["q_max_kvar"] = q_max_kvar
    return environment_model(OmegaConf.create(config))


def _reduce(
    environment: EnvironmentDynamics,
    constraint: ActionConstraints,
    bus_voltage_pu: chex.Array,
) -> tuple[ActionConstraints, chex.Array]:
    """Run the grid code's reduction with the same bus gather the environment
    uses, so tests exercise the real seam rather than a hand-built voltage.
    """
    return environment.grid_code.reduce(
        constraint,
        voltage_pu=environment.agent_voltage_pu(bus_voltage_pu),
        step_duration_h=environment.time.step_duration_h,
    )


def test_q_of_u_reduces_the_action_space_to_active_power() -> None:
    environment = build_swiss_lv_environment()
    state = environment.reset(jr.PRNGKey(0))

    assert environment.action_dim == 1
    chex.assert_shape(state.action_constraints.halfspace_a, (environment.num_agents, 2, 1))
    chex.assert_shape(state.q_setpoint_kvarh, (environment.num_agents,))
    # A pure box, like Battery's and Solar's own 1-D constraints -- the ball
    # coupling p to q is gone, having been resolved into the bounds.
    chex.assert_shape(state.action_constraints.ball_center, (environment.num_agents, 0, 1))


def test_no_grid_code_leaves_the_two_dimensional_action_space_untouched() -> None:
    environment = build_environment()
    state = environment.reset(jr.PRNGKey(0))

    assert environment.action_dim == 2
    chex.assert_shape(state.action_constraints.halfspace_a, (environment.num_agents, 2, 2))
    assert jnp.all(state.q_setpoint_kvarh == 0.0)


def test_reported_bounds_are_always_feasible_in_the_full_two_dimensional_set() -> None:
    """The property the whole bounds derivation exists to guarantee.

    Every reported bound, lifted back to (p, q*), must satisfy the physical
    2-D constraint -- which is what makes ProsumerDynamics' own projection a
    no-op rather than a correction. Checked at both ends of the reported
    interval and at the origin, across a full day.
    """
    environment = build_swiss_lv_environment()
    state = environment.reset(jr.PRNGKey(0))

    for step in range(environment.time.n_steps_per_day // 4):
        constraint = state.action_constraints
        p_max = jnp.asarray(constraint.halfspace_b)[:, 0]
        p_min = jnp.negative(jnp.asarray(constraint.halfspace_b)[:, 1])
        scale = jnp.asarray(environment.prosumer.inverter_dynamics.s_inv_max_kvah)

        for normalized_p in (p_min, p_max, jnp.zeros_like(p_max)):
            request = jnp.stack([normalized_p * scale, state.q_setpoint_kvarh], axis=-1)
            assert bool(
                state.prosumer_state.s_inv_request_constraint.is_feasible(request, tol=1e-5)
            ), f"reported bound infeasible at step {step}"

        action = jnp.full((environment.num_agents, 1), 0.8, dtype=jnp.float32)
        state, _ = environment.step(state, action)


def test_reported_interval_always_contains_zero_active_power() -> None:
    """The origin invariant `ActionConstraints` requires, in the 1-D action
    space: p = 0 must stay feasible so the radial map remains well-defined.
    """
    environment = build_swiss_lv_environment()
    state = environment.reset(jr.PRNGKey(3))

    for _ in range(24):
        p_max = jnp.asarray(state.action_constraints.halfspace_b)[:, 0]
        p_min = jnp.negative(jnp.asarray(state.action_constraints.halfspace_b)[:, 1])
        assert jnp.all(p_min <= 0.0)
        assert jnp.all(p_max >= 0.0)
        state, _ = environment.step(
            state, jnp.full((environment.num_agents, 1), -0.9, dtype=jnp.float32)
        )


def test_stays_feasible_when_the_curve_demands_more_than_the_connection_can_carry() -> None:
    """A Q_max far beyond the connection's reactive headroom must derate, not
    produce an empty active-power range.

    This is the case a static sizing rule would have to exclude by
    configuration; here it is absorbed by construction. The inverter is also
    four times the connection rating, so the grid-connection ball binds hard.
    """
    environment = build_swiss_lv_environment(s_pq_max_kva=15.1, s_inv_max_kva=60.0, q_max_kvar=60.0)
    state = environment.reset(jr.PRNGKey(0))

    for _ in range(24):
        assert bool(state.valid)
        p_max = jnp.asarray(state.action_constraints.halfspace_b)[:, 0]
        p_min = jnp.negative(jnp.asarray(state.action_constraints.halfspace_b)[:, 1])
        assert jnp.all(p_min <= 0.0) and jnp.all(p_max >= 0.0)
        state, _ = environment.step(
            state, jnp.full((environment.num_agents, 1), 1.0, dtype=jnp.float32)
        )
    assert bool(state.valid)


def test_realized_reactive_power_matches_the_setpoint_the_agent_was_shown() -> None:
    """The agent's action must not move q off the curve.

    Radial projection would have pulled p and q back together, preserving the
    power factor -- exactly wrong when the power factor is prescribed. The
    reported bounds being a feasible subset is what prevents that.
    """
    environment = build_swiss_lv_environment()
    state = environment.reset(jr.PRNGKey(1))

    for _ in range(12):
        expected_q = state.q_setpoint_kvarh
        state, _ = environment.step(
            state, jnp.full((environment.num_agents, 1), 1.0, dtype=jnp.float32)
        )
        realized_q = state.prosumer_state.inverter_state.s_inv_realized_kvah.imag
        assert jnp.allclose(realized_q, expected_q, atol=1e-4)


def test_setpoint_follows_the_curve_from_the_previous_interval_voltage() -> None:
    """q* is the curve evaluated at the voltage the power flow just produced --
    one interval of lag, which is what avoids an algebraic loop.
    """
    environment = build_swiss_lv_environment()
    state = environment.reset(jr.PRNGKey(2))
    q_of_u = environment.grid_code.q_of_u

    voltage_pu = jnp.abs(state.grid_state.bus_voltage_pu)[environment.agent_bus_id]
    target = q_of_u.q_setpoint_kvarh(voltage_pu, environment.time.step_duration_h)
    # Equal where the connection can carry the curve's demand; never larger,
    # and never of the opposite sign, where it derates.
    assert jnp.all(jnp.abs(state.q_setpoint_kvarh) <= jnp.abs(target) + 1e-5)
    assert jnp.all(state.q_setpoint_kvarh * target >= -1e-9)


def test_a_full_day_of_out_of_range_actions_keeps_every_invariant() -> None:
    environment = build_swiss_lv_environment()
    state = environment.reset(jr.PRNGKey(7))
    scale = jnp.asarray(environment.prosumer.inverter_dynamics.s_inv_max_kvah)

    for step in range(int(environment.time.n_steps_per_day)):
        action = jnp.full((environment.num_agents, 1), 50.0 * (-1.0) ** step, dtype=jnp.float32)
        state, _ = environment.step(state, action)
        assert bool(state.valid)
        realized = state.prosumer_state.inverter_state.s_inv_realized_kvah
        assert jnp.all(jnp.abs(realized) <= scale + 1e-4)
        assert jnp.all(jnp.isfinite(state.q_setpoint_kvarh))


def test_setpoint_sign_supports_voltage_through_the_real_bus_gather() -> None:
    """Under-voltage must draw injection and over-voltage absorption, checked
    through the whole chain rather than on the curve in isolation.

    A sign error anywhere between the bus gather, the curve, and the
    inverter's own generator-arrow convention would drive voltage AWAY from
    nominal while every magnitude assertion still passed. Voltage is imposed
    directly here rather than reached by simulation, because the bundled
    feeder is stiff enough that ordinary operation rarely leaves the
    0.97-1.03 deadband -- so a passive test would silently check nothing.
    """
    environment = build_swiss_lv_environment()
    state = environment.reset(jr.PRNGKey(0))
    constraint = state.prosumer_state.s_inv_request_constraint

    for voltage_pu, expected_sign in ((0.90, 1.0), (1.10, -1.0)):
        bus_voltage_pu = jnp.full_like(state.grid_state.bus_voltage_pu, voltage_pu)
        _, q_setpoint_kvarh = _reduce(environment, constraint, bus_voltage_pu)
        assert jnp.all(q_setpoint_kvarh * expected_sign > 0.0), (
            f"at {voltage_pu} pu the setpoint must have sign {expected_sign:+.0f}; "
            "the opposite sign would amplify the deviation instead of opposing it"
        )


def test_deadband_leaves_the_plant_exchanging_no_reactive_power() -> None:
    environment = build_swiss_lv_environment()
    state = environment.reset(jr.PRNGKey(0))
    constraint = state.prosumer_state.s_inv_request_constraint

    for voltage_pu in (0.97, 1.00, 1.03):
        bus_voltage_pu = jnp.full_like(state.grid_state.bus_voltage_pu, voltage_pu)
        _, q_setpoint_kvarh = _reduce(environment, constraint, bus_voltage_pu)
        assert jnp.allclose(q_setpoint_kvarh, 0.0, atol=1e-5)


def test_bounds_widen_when_the_curve_stops_demanding_reactive_power() -> None:
    """Reactive power spends apparent-power headroom, so full droop must leave
    strictly less room for active power than the deadband does.
    """
    environment = build_swiss_lv_environment()
    state = environment.reset(jr.PRNGKey(0))
    constraint = state.prosumer_state.s_inv_request_constraint

    def active_width(voltage_pu: float) -> chex.Array:
        bus_voltage_pu = jnp.full_like(state.grid_state.bus_voltage_pu, voltage_pu)
        bounds, _ = _reduce(environment, constraint, bus_voltage_pu)
        return jnp.asarray(bounds.halfspace_b).sum(axis=1)

    assert jnp.all(active_width(0.90) <= active_width(1.00) + 1e-5)


def _bisect_extent(
    constraints: ActionConstraints,
    anchor: chex.Array,
    direction: chex.Array,
    iterations: int = 30,
) -> chex.Array:
    """Independent reference for the closed form in `SwissLvGridCode.reduce`.

    Walks the ray ``anchor + t * direction`` and keeps the last ``t`` whose
    point the feasibility predicate accepted. Slow and obviously correct: it
    knows nothing about halfspaces or balls, only whether a point is in the
    set, so it shares no algebra with the implementation it checks. The
    boundary bug that motivated the ``c_p^2`` floor came precisely from two
    expressions of the same geometry disagreeing, which a test written in
    that same algebra could not have caught.

    Requires a feasible anchor, and under-claims by at most 2^-iterations.
    """

    def body(_: int, bracket: tuple[chex.Array, chex.Array]) -> tuple[chex.Array, chex.Array]:
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        feasible = constraints.feasible_mask(anchor + mid[:, None] * direction, tol=0.0)
        return jnp.where(feasible, mid, lo), jnp.where(feasible, hi, mid)

    num_agents = anchor.shape[0]
    lo, _ = jax.lax.fori_loop(
        0,
        iterations,
        body,
        (jnp.zeros((num_agents,)), jnp.ones((num_agents,))),
    )
    return lo[:, None] * direction


def test_closed_form_bounds_agree_with_an_independent_search() -> None:
    """Differential test: the closed form against brute-force bisection.

    Swept across the whole Q(U) curve including full droop in both
    directions, so the reactive setpoint ranges over everything the standard
    curve can produce rather than just the deadband the default feeder sits
    in.
    """
    environment = build_swiss_lv_environment()
    inverter = environment.prosumer.inverter_dynamics
    scale = jnp.asarray(inverter.s_inv_max_kvah)

    for seed in range(4):
        state = environment.reset(jr.PRNGKey(seed))
        constraint = state.prosumer_state.s_inv_request_constraint
        for voltage_pu in (0.90, 0.95, 1.00, 1.05, 1.10):
            bus_voltage_pu = jnp.full_like(state.grid_state.bus_voltage_pu, voltage_pu)
            bounds, q_setpoint_kvarh = _reduce(environment, constraint, bus_voltage_pu)
            p_max = jnp.asarray(bounds.halfspace_b)[:, 0]
            p_min = jnp.negative(jnp.asarray(bounds.halfspace_b)[:, 1])

            anchor = jnp.stack([jnp.zeros_like(q_setpoint_kvarh), q_setpoint_kvarh], axis=-1)
            zeros = jnp.zeros_like(p_max)
            reference_max = _bisect_extent(
                constraint, anchor, jnp.stack([jnp.full_like(p_max, 4.0) * scale, zeros], -1)
            )[:, 0]
            reference_min = _bisect_extent(
                constraint, anchor, jnp.stack([jnp.full_like(p_min, -4.0) * scale, zeros], -1)
            )[:, 0]

            assert jnp.allclose(p_max, reference_max, atol=1e-4), (
                f"upper bound disagrees at {voltage_pu} pu, seed {seed}"
            )
            assert jnp.allclose(p_min, reference_min, atol=1e-4), (
                f"lower bound disagrees at {voltage_pu} pu, seed {seed}"
            )


def test_reactive_setpoint_is_the_largest_feasible_point_toward_the_target() -> None:
    """Derating must stop at the connection's limit, not short of it.

    Checked against the predicate rather than against a formula: nudging the
    setpoint further toward the curve's target must leave the feasible set
    whenever the clamp actually bit.
    """
    environment = build_swiss_lv_environment(s_pq_max_kva=15.1, s_inv_max_kva=60.0, q_max_kvar=60.0)
    state = environment.reset(jr.PRNGKey(0))
    constraint = state.prosumer_state.s_inv_request_constraint

    for voltage_pu in (0.90, 1.10):
        bus_voltage_pu = jnp.full_like(state.grid_state.bus_voltage_pu, voltage_pu)
        _, q_setpoint_kvarh = _reduce(environment, constraint, bus_voltage_pu)
        origin = jnp.zeros_like(q_setpoint_kvarh)

        at_setpoint = jnp.stack([origin, q_setpoint_kvarh], axis=-1)
        assert bool(constraint.is_feasible(at_setpoint, tol=1e-5))

        beyond = jnp.stack([origin, q_setpoint_kvarh * 1.05], axis=-1)
        assert not bool(constraint.is_feasible(beyond, tol=0.0))
