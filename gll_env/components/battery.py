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

from dataclasses import field
from functools import cached_property

import chex
import jax.numpy as jnp
import jax.random as jr

from gll_env.components.day_time import DaytimeDynamics
from gll_env.types import ActionConstraints
from gll_env.utils import safe_normalize


@chex.dataclass(frozen=True)
class BatteryState:
    # Power flow direction convention:
    # positive = discharge (out of battery)
    # negative = charge (into battery)
    bat_realized_kwh: chex.Array  # (num_bat,) float32 -- past interval
    bat_request_constraint: ActionConstraints

    bat_free_kwh: chex.Array  # (num_bat,) float32
    bat_full_kwh: chex.Array  # (num_bat,) float32


@chex.dataclass(frozen=True)
class BatteryObservation:
    """Observation of past battery flow and upcoming request bounds.

    Attributes:
        bat_realized: Past-interval battery energy flow. Unnormalized values
            are in kWh, with charging negative and discharging positive;
            normalized values are in ``[-1, 1]``.
        bat_request_min: Coming-interval minimum battery request. Unnormalized
            values are in kWh and are non-positive; normalized values are in
            ``[-1, 0]``.
        bat_request_max: Coming-interval maximum battery request. Unnormalized
            values are in kWh and are non-negative; normalized values are in
            ``[0, 1]``.
        bat_free: Available battery energy. Unnormalized values are in kWh;
            normalized values are in ``[0, 1]``.
        bat_full: Stored battery energy. Unnormalized values are in kWh;
            normalized values are in ``[0, 1]``.
        is_normalized: Defaults to ``False``.
    """

    bat_realized: chex.Array  # (num_bat,) float32 -- past interval
    bat_request_min: chex.Array  # (num_bat,) float32 -- coming interval
    bat_request_max: chex.Array  # (num_bat,) float32 -- coming interval
    bat_free: chex.Array  # (num_bat,) float32 -- instantaneous
    bat_full: chex.Array  # (num_bat,) float32 -- instantaneous
    is_normalized: bool = field(default=False)

    def normalize(self, battery_dynamics: "BatteryDynamics") -> "BatteryObservation":
        peak_per_step_kwh = jnp.maximum(
            battery_dynamics.peak_charge_per_step_kwh, battery_dynamics.peak_discharge_per_step_kwh
        )
        return BatteryObservation(
            bat_realized=safe_normalize(self.bat_realized, peak_per_step_kwh),
            bat_request_min=safe_normalize(
                self.bat_request_min, battery_dynamics.peak_charge_per_step_kwh
            ),
            bat_request_max=safe_normalize(
                self.bat_request_max, battery_dynamics.peak_discharge_per_step_kwh
            ),
            bat_free=safe_normalize(self.bat_free, battery_dynamics.capacity_kwh),
            bat_full=safe_normalize(self.bat_full, battery_dynamics.capacity_kwh),
            is_normalized=True,
        )


@chex.dataclass(frozen=True)
class BatteryDynamics:
    """Linear battery with explicitly defined per-unit power and energy capacities."""

    capacity_kwh: chex.Array  # (num_bat,) float32, non-negative
    charge_rating_kw: chex.Array  # (num_bat,) float32, non-negative
    discharge_rating_kw: chex.Array  # (num_bat,) float32, non-negative
    time: DaytimeDynamics = field(default_factory=DaytimeDynamics)

    @cached_property
    def num_bat(self) -> int:
        return jnp.atleast_1d(self.capacity_kwh).shape[0]

    def __post_init__(self) -> None:
        chex.assert_shape(self.capacity_kwh, (self.num_bat,))
        chex.assert_shape(self.charge_rating_kw, (self.num_bat,))
        chex.assert_shape(self.discharge_rating_kw, (self.num_bat,))
        chex.assert_type(self.capacity_kwh, jnp.float32)
        chex.assert_type(self.charge_rating_kw, jnp.float32)
        chex.assert_type(self.discharge_rating_kw, jnp.float32)
        object.__setattr__(self, "capacity_kwh", jnp.maximum(self.capacity_kwh, 0.0))
        object.__setattr__(self, "charge_rating_kw", jnp.maximum(self.charge_rating_kw, 0.0))
        object.__setattr__(self, "discharge_rating_kw", jnp.maximum(self.discharge_rating_kw, 0.0))

    @property
    def peak_charge_per_step_kwh(self) -> chex.Array:
        return self.charge_rating_kw * self.time.step_duration_h

    @property
    def peak_discharge_per_step_kwh(self) -> chex.Array:
        return self.discharge_rating_kw * self.time.step_duration_h

    def _bat_request_min(self, e_free: chex.Array) -> chex.Array:
        """The most negative (charging) request allowed by the battery's
        charge rating and remaining headroom (e_free).
        """
        return jnp.negative(jnp.minimum(self.peak_charge_per_step_kwh, e_free))

    def _bat_request_max(self, e_full: chex.Array) -> chex.Array:
        """The most positive (discharging) request allowed by the battery's
        discharge rating and currently stored energy (e_full).
        """
        return jnp.minimum(self.peak_discharge_per_step_kwh, e_full)

    def _new_request_constraint(self, e_free: chex.Array, e_full: chex.Array) -> ActionConstraints:
        """The feasible bat_request_kWh range for the coming interval, given
        the currently free/full storage, expressed as a 1D-box
        ActionConstraints:

            halfspace 0: [+1] @ request <= max   (request <= max)
            halfspace 1: [-1] @ request <= -min  (request >= min)
            0 balls.

        Charging (negative direction) is limited by whichever is smaller:
        the charge rating over one interval, or the remaining headroom
        (e_free). Discharging (positive direction) is limited by whichever
        is smaller: the discharge rating, or what's actually stored (e_full).

        This ordering (halfspace 0 = upper bound, halfspace 1 = lower bound)
        is a fixed convention -- request_bounds relies on it to unpack the
        constraint back into (min, max) elsewhere. No `bat_` prefix on the
        locals here -- we're already inside BatteryDynamics, the whole file
        is battery-scoped.
        """
        request_max_kwh = self._bat_request_max(e_full)
        request_min_kwh = self._bat_request_min(e_free)

        ones = jnp.ones((self.num_bat, 1), dtype=jnp.float32)
        halfspace_a = jnp.stack([ones, jnp.negative(ones)], axis=1)  # (num_bat, 2, 1)
        halfspace_b = jnp.stack(
            [request_max_kwh, jnp.negative(request_min_kwh)], axis=1
        )  # (num_bat, 2)

        return ActionConstraints(
            halfspace_a=halfspace_a,
            halfspace_b=halfspace_b,
            ball_center=jnp.zeros((self.num_bat, 0, 1), dtype=jnp.float32),
            ball_radius=jnp.zeros((self.num_bat, 0), dtype=jnp.float32),
        )

    def request_bounds(self, constraint: ActionConstraints) -> tuple[chex.Array, chex.Array]:
        """Return the signed ``(min, max)`` bounds encoded by a battery box.

        The constraint stores the upper bound in halfspace 0 and the lower
        bound as its negation in halfspace 1. This method is public because
        inverter dynamics compose these bounds with solar and inverter limits.
        """
        request_max_kwh = jnp.asarray(constraint.halfspace_b)[:, 0]
        request_min_kwh = jnp.negative(jnp.asarray(constraint.halfspace_b)[:, 1])
        return request_min_kwh, request_max_kwh

    def observation(self, state: BatteryState) -> BatteryObservation:
        request_min_kwh, request_max_kwh = self.request_bounds(state.bat_request_constraint)
        return BatteryObservation(
            bat_realized=state.bat_realized_kwh,
            bat_request_min=request_min_kwh,
            bat_request_max=request_max_kwh,
            bat_free=state.bat_free_kwh,
            bat_full=state.bat_full_kwh,
        )

    def reset(self, key: chex.PRNGKey) -> BatteryState:
        full_key, p_key = jr.split(key)

        full_kwh = jr.uniform(
            key=full_key,
            shape=self.capacity_kwh.shape,
            minval=0.0,
            maxval=self.capacity_kwh,
        )
        free_kwh = jnp.subtract(self.capacity_kwh, full_kwh)
        request_constraint = self._new_request_constraint(e_free=free_kwh, e_full=full_kwh)

        # Fictional past flow: NOT request_constraint (that's the
        # forward-looking bound for the COMING interval) -- see
        # _new_request_constraint's docstring for why the backward-looking
        # version swaps the roles of full/free.
        realized_kwh = jr.uniform(
            key=p_key,
            shape=self.capacity_kwh.shape,
            minval=jnp.negative(jnp.minimum(self.peak_charge_per_step_kwh, full_kwh)),
            maxval=jnp.minimum(self.peak_discharge_per_step_kwh, free_kwh),
        )

        return BatteryState(
            bat_realized_kwh=realized_kwh,
            bat_request_constraint=request_constraint,
            bat_full_kwh=full_kwh,
            bat_free_kwh=free_kwh,
        )

    def step(self, state: BatteryState, bat_request_kwh: chex.Array) -> BatteryState:
        request_min_kwh, request_max_kwh = self.request_bounds(state.bat_request_constraint)
        request_kwh = jnp.clip(bat_request_kwh, request_min_kwh, request_max_kwh)
        next_realized_kwh = request_kwh

        next_full_kwh = state.bat_full_kwh - request_kwh
        next_free_kwh = self.capacity_kwh - next_full_kwh
        next_request_constraint = self._new_request_constraint(
            e_free=next_free_kwh, e_full=next_full_kwh
        )

        return BatteryState(
            bat_realized_kwh=next_realized_kwh,
            bat_request_constraint=next_request_constraint,
            bat_full_kwh=next_full_kwh,
            bat_free_kwh=next_free_kwh,
        )
