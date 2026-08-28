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
import jax.numpy as jnp
import pytest

from gll_env.grid_codes.base import NoGridCode
from gll_env.grid_codes.swiss_lv import (
    QofUCharacteristic,
    SwissLvGridCode,
    limiting_power_factor,
    rated_q_max_kvar,
)
from gll_env.types import ActionConstraints

STEP_DURATION_H = 0.25


def characteristic(q_max_kvar: float = 4.0) -> QofUCharacteristic:
    return QofUCharacteristic(q_max_kvar=jnp.asarray([q_max_kvar], dtype=jnp.float32))


def setpoint(curve: QofUCharacteristic, voltage_pu: float) -> float:
    return float(
        curve.q_setpoint_kvarh(jnp.asarray([voltage_pu], dtype=jnp.float32), STEP_DURATION_H)[0]
    )


@pytest.mark.parametrize(
    ("s_inv_max_kva", "expected_power_factor"),
    [(0.5, 1.0), (0.8, 1.0), (3.7, 0.95), (15.0, 0.90), (100.0, 0.90)],
)
def test_power_factor_follows_table_3_rating_bands(
    s_inv_max_kva: float, expected_power_factor: float
) -> None:
    """Tabelle 3 widens the required cos phi band as the plant grows."""
    actual = limiting_power_factor(jnp.asarray([s_inv_max_kva], dtype=jnp.float32))
    assert float(actual[0]) == pytest.approx(expected_power_factor)


def test_plants_below_800_va_are_asked_for_no_reactive_power() -> None:
    """No reactive requirement below 800 VA, so a flat-zero curve."""
    q_max_kvar = rated_q_max_kvar(jnp.asarray([0.5], dtype=jnp.float32))
    assert float(q_max_kvar[0]) == pytest.approx(0.0)


def test_rated_q_max_is_the_reactive_leg_of_the_limiting_power_factor() -> None:
    """Q_max = sin(arccos(cos phi)) * S_Emax, referenced to NAMEPLATE rating."""
    q_max_kvar = rated_q_max_kvar(jnp.asarray([15.0], dtype=jnp.float32))
    assert float(q_max_kvar[0]) == pytest.approx(15.0 * (1.0 - 0.90**2) ** 0.5, rel=1e-5)


@pytest.mark.parametrize(
    ("voltage_pu", "expected_ratio"),
    [
        (0.93, +1.0),  # full over-excited support at the lower knee
        (0.95, +0.5),  # linear through the lower ramp
        (0.97, 0.0),  # deadband opens
        (1.00, 0.0),  # nominal
        (1.03, 0.0),  # deadband closes
        (1.05, -0.5),  # linear through the upper ramp
        (1.07, -1.0),  # full under-excited absorption at the upper knee
    ],
)
def test_curve_matches_figure_5_breakpoints(voltage_pu: float, expected_ratio: float) -> None:
    curve = characteristic(q_max_kvar=4.0)
    expected = expected_ratio * 4.0 * STEP_DURATION_H
    # abs=1e-5, not tighter: jnp.interp runs in float32, so the ramp midpoints
    # land ~1.5e-6 off exact. That is the dtype's resolution, not slack.
    assert setpoint(curve, voltage_pu) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize(("voltage_pu", "expected_ratio"), [(0.80, +1.0), (1.20, -1.0)])
def test_curve_saturates_beyond_the_outer_knees(voltage_pu: float, expected_ratio: float) -> None:
    """Past 0.93/1.07 the plant holds full droop rather than extrapolating."""
    curve = characteristic(q_max_kvar=4.0)
    expected = expected_ratio * 4.0 * STEP_DURATION_H
    # abs=1e-5, not tighter: jnp.interp runs in float32, so the ramp midpoints
    # land ~1.5e-6 off exact. That is the dtype's resolution, not slack.
    assert setpoint(curve, voltage_pu) == pytest.approx(expected, abs=1e-5)


def test_low_voltage_supplies_and_high_voltage_absorbs_reactive_power() -> None:
    """The sign convention that makes the curve voltage-supporting rather than
    voltage-amplifying: Erzeugerzaehlpfeilsystem, positive out of the inverter.
    Under-voltage must inject (uebererregt, +Q, raising voltage); over-voltage
    must absorb (untererregt, -Q, lowering it). A flipped sign would drive
    voltage away from nominal and still pass every magnitude check above.
    """
    curve = characteristic()
    assert setpoint(curve, 0.90) > 0.0
    assert setpoint(curve, 1.10) < 0.0


def test_setpoint_is_energy_so_it_scales_with_interval_length() -> None:
    curve = characteristic(q_max_kvar=4.0)
    voltage = jnp.asarray([0.93], dtype=jnp.float32)
    quarter_hour = float(curve.q_setpoint_kvarh(voltage, 0.25)[0])
    full_hour = float(curve.q_setpoint_kvarh(voltage, 1.0)[0])
    assert full_hour == pytest.approx(4.0 * quarter_hour, rel=1e-6)


def test_negative_q_max_is_clamped_rather_than_inverting_the_control() -> None:
    """A negative q_max would mirror the curve into voltage-amplifying."""
    curve = QofUCharacteristic(q_max_kvar=jnp.asarray([-4.0], dtype=jnp.float32))
    assert float(jnp.asarray(curve.q_max_kvar)[0]) == pytest.approx(0.0)
    assert setpoint(curve, 0.90) == pytest.approx(0.0)


def two_dimensional_constraint() -> ActionConstraints:
    """p in [-3, 5], |s| <= 4 -- a plausible inverter constraint, origin inside."""
    return ActionConstraints(
        halfspace_a=jnp.asarray([[[1.0, 0.0], [-1.0, 0.0]]]),
        halfspace_b=jnp.asarray([[5.0, 3.0]]),
        ball_center=jnp.zeros((1, 1, 2)),
        ball_radius=jnp.asarray([[4.0]]),
    )


def test_no_grid_code_leaves_both_degrees_of_freedom_to_the_agent() -> None:
    code = NoGridCode()
    constraints = two_dimensional_constraint()
    reduced, setpoint = code.reduce(constraints, jnp.asarray([1.0]), STEP_DURATION_H)

    assert code.action_dim == 2
    assert reduced is constraints  # passed straight through, not rebuilt
    assert jnp.allclose(setpoint, 0.0)


def test_no_grid_code_lift_is_the_identity() -> None:
    request = jnp.asarray([[1.5, -0.5]])
    lifted = NoGridCode().lift(request, jnp.asarray([99.0]))
    assert jnp.allclose(lifted, request)  # the setpoint is ignored; the agent set q


def test_swiss_lv_takes_the_reactive_degree_of_freedom_away() -> None:
    assert SwissLvGridCode(q_of_u=characteristic()).action_dim == 1


def test_swiss_lv_reduce_returns_a_one_dimensional_box_containing_zero() -> None:
    code = SwissLvGridCode(q_of_u=characteristic())
    reduced, setpoint = code.reduce(
        two_dimensional_constraint(), jnp.asarray([0.90]), STEP_DURATION_H
    )
    chex.assert_shape(reduced.halfspace_a, (1, 2, 1))
    chex.assert_shape(reduced.ball_center, (1, 0, 1))
    assert bool(reduced.feasible_mask(jnp.zeros((1, 1)), tol=0.0)[0])
    assert float(setpoint[0]) > 0.0  # under-voltage -> supplying reactive power


def test_swiss_lv_lift_pairs_the_action_with_the_setpoint() -> None:
    """lift must invert reduce: the pairing is the ABC's contract."""
    code = SwissLvGridCode(q_of_u=characteristic())
    constraints = two_dimensional_constraint()
    reduced, setpoint = code.reduce(constraints, jnp.asarray([0.90]), STEP_DURATION_H)

    p_max = jnp.asarray(reduced.halfspace_b)[:, 0]
    lifted = code.lift(p_max[:, None], setpoint)

    assert jnp.allclose(lifted[:, 1], setpoint)
    # An action feasible under the reduced constraint lifts to a request
    # feasible under the constraint reduce was handed.
    assert bool(constraints.is_feasible(lifted, tol=1e-5))
