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

from gll_env.components.day_time import DaytimeDynamics, DaytimeState
from gll_env.types import ActionConstraints
from gll_env.utils import safe_normalize


@chex.dataclass(frozen=True)
class SolarState:
    # Power flow direction convention:
    # positive = generation (out of solar)
    # negative = consumption = 0 (into solar)
    sol_realized_kwh: chex.Array  # (num_sol,) float32 -- past interval
    sol_request_constraint: ActionConstraints

    time_state: DaytimeState
    clearness: chex.Array  # (num_sol,) float32
    key: chex.PRNGKey  # (2,)


@chex.dataclass(frozen=True)
class SolarObservation:
    """Observation of past realized and upcoming available solar energy.

    Attributes:
        sol_realized: Past-interval solar generation. Unnormalized values are
            in ``[0, s_sol_max_kwh]``; normalized values are in ``[0, 1]``.
        sol_available: Coming-interval available solar generation with the
            same range as ``sol_realized``.
        sol_clearness: Coming-interval clearness index in ``[0, 1]``.
        is_normalized: Defaults to ``False``.
    """

    sol_realized: chex.Array  # (num_sol,) float32 -- past interval
    sol_available: chex.Array  # (num_sol,) float32 -- coming interval
    sol_clearness: chex.Array  # (num_sol,) float32 -- coming interval
    is_normalized: bool = field(default=False)

    def normalize(self, solar_dynamics: "SolarDynamics") -> "SolarObservation":
        return SolarObservation(
            sol_realized=safe_normalize(self.sol_realized, solar_dynamics.s_sol_max_kwh),
            sol_available=safe_normalize(self.sol_available, solar_dynamics.s_sol_max_kwh),
            sol_clearness=self.sol_clearness,
            is_normalized=True,
        )


@chex.dataclass(frozen=True)
class SolarDynamics:
    """Ornstein-Uhlenbeck clearness process with a cosine day profile.

    The clearness index follows a mean-reverting stochastic process
    (Ornstein-Uhlenbeck) modulated by a cosine envelope representing
    the sun's position across the day.
    """

    peak_power_kw: chex.Array  # (num_sol,) float32

    clearness_reversion: chex.Array = field(default_factory=lambda: jnp.float32(0.05))
    clearness_mean: chex.Array = field(default_factory=lambda: jnp.float32(0.6))
    clearness_std: chex.Array = field(default_factory=lambda: jnp.float32(0.1))
    time: DaytimeDynamics = field(default_factory=DaytimeDynamics)

    @cached_property
    def num_sol(self) -> int:
        return jnp.atleast_1d(self.peak_power_kw).shape[0]

    @cached_property
    def s_sol_max_kwh(self) -> chex.Array:
        return self.peak_power_kw * self.time.step_duration_h

    def __post_init__(self) -> None:
        chex.assert_shape(self.peak_power_kw, (self.num_sol,))
        chex.assert_shape(self.clearness_reversion, ())
        chex.assert_shape(self.clearness_mean, ())
        chex.assert_shape(self.clearness_std, ())
        chex.assert_type(self.peak_power_kw, jnp.float32)
        chex.assert_type(self.clearness_reversion, jnp.float32)
        chex.assert_type(self.clearness_mean, jnp.float32)
        chex.assert_type(self.clearness_std, jnp.float32)
        object.__setattr__(self, "peak_power_kw", jnp.maximum(self.peak_power_kw, 0.0))
        object.__setattr__(
            self, "clearness_reversion", jnp.clip(self.clearness_reversion, 1e-6, 1.0)
        )
        object.__setattr__(self, "clearness_mean", jnp.clip(self.clearness_mean, 0.0, 1.0))
        object.__setattr__(self, "clearness_std", jnp.maximum(self.clearness_std, 0.0))

    def _reset_clearness(self, key: chex.PRNGKey) -> chex.Array:
        """Sample a clearness index from the stationary distribution.

        The AR(1) update is clearness_{t+1} = phi*clearness_t + (1-phi)*mean
        + std*noise with phi = 1 - clearness_reversion (read off _new_clearness
        below). The stationary variance of that process is std**2/(1-phi**2).
        """
        phi = 1.0 - self.clearness_reversion
        stationary_std = self.clearness_std / jnp.sqrt(1.0 - phi**2)
        clearness = self.clearness_mean + stationary_std * jr.normal(key=key, shape=(self.num_sol,))
        return jnp.clip(clearness, 0.0, 1.0)

    def _new_clearness(self, clearness: chex.Array, key: chex.PRNGKey) -> chex.Array:
        clearness = jnp.atleast_1d(clearness)
        iid_noise = jr.normal(key, shape=clearness.shape)
        clearness = (
            clearness
            + self.clearness_reversion * (self.clearness_mean - clearness)
            + self.clearness_std * iid_noise
        )
        return jnp.clip(clearness, 0.0, 1.0)

    def _available_fraction(self, day_progress: chex.Numeric, clearness: chex.Array) -> chex.Array:
        # Cosine envelope peaking at noon (day_angle = π)
        day_angle = 2 * jnp.pi * day_progress
        fraction = jnp.maximum(0.0, jnp.cos(day_angle - jnp.pi))
        fraction = jnp.clip(fraction * clearness, 0.0, 1.0)
        return fraction

    def _available_power(self, day_progress: chex.Numeric, clearness: chex.Array) -> chex.Array:
        fraction = self._available_fraction(day_progress, clearness)
        return fraction * jnp.asarray(self.peak_power_kw)

    def _available_energy(self, day_progress: chex.Numeric, clearness: chex.Array) -> chex.Array:
        # Primary operation uses 15-minute steps, so midpoint power is a
        # constant-over-an-interval approximation.
        available_power = self._available_power(day_progress, clearness)
        return available_power * self.time.step_duration_h

    def _new_request_constraint(self, request_max_kwh: chex.Array) -> ActionConstraints:
        """The feasible sol_request_kWh range for the coming interval:
        [0, request_max_kWh]. Solar can only generate, never consume, so the
        lower bound is pinned to 0 -- a degenerate case of the same box
        structure BatteryDynamics uses, not a special one:

            halfspace 0: [+1] @ request <= max
            halfspace 1: [-1] @ request <= -min = 0
            0 balls

        This ordering (halfspace 0 = upper bound, halfspace 1 = lower bound)
        is a fixed convention -- request_bounds relies on it. No `sol_`
        prefix on locals here -- we're already inside SolarDynamics.
        """
        ones = jnp.ones((self.num_sol, 1), dtype=jnp.float32)
        request_min_kwh = jnp.zeros((self.num_sol,), dtype=jnp.float32)  # solar cannot consume
        halfspace_a = jnp.stack([ones, jnp.negative(ones)], axis=1)  # (num_sol, 2, 1)
        halfspace_b = jnp.stack(
            [jnp.maximum(request_max_kwh, 0.0), request_min_kwh], axis=1
        )  # (num_sol, 2)

        return ActionConstraints(
            halfspace_a=halfspace_a,
            halfspace_b=halfspace_b,
            ball_center=jnp.zeros((self.num_sol, 0, 1), dtype=jnp.float32),
            ball_radius=jnp.zeros((self.num_sol, 0), dtype=jnp.float32),
        )

    def request_bounds(self, constraint: ActionConstraints) -> tuple[chex.Array, chex.Array]:
        """Unpack (min, max) from a box-only ActionConstraints (2 opposing
        halfspaces on a 1D action, 0 balls), matching the ordering fixed by
        _new_request_constraint.

        Public (no leading underscore): other components read solar's bound
        directly, e.g. InverterDynamics currently reads
        solar_state.sol_min_kWh/sol_max_kWh to build
        p_inv_min_kWh/p_inv_max_kWh, and would call
        self.solar_dynamics.request_bounds(solar_state.sol_request_constraint)
        for the same purpose once that call site is updated to match.
        """
        request_max_kwh = jnp.asarray(constraint.halfspace_b)[:, 0]
        request_min_kwh = jnp.negative(constraint.halfspace_b)[:, 1]
        return request_min_kwh, request_max_kwh

    def observation(self, state: SolarState) -> SolarObservation:
        return SolarObservation(
            sol_realized=state.sol_realized_kwh,
            sol_available=self._available_energy(
                state.time_state.interval_midpoint, state.clearness
            ),
            sol_clearness=state.clearness,
        )

    def reset(self, key: chex.PRNGKey, time_state: DaytimeState | None = None) -> SolarState:
        if time_state is None:
            key, time_key = jr.split(key)
            time_state = self.time.reset(time_key)

        key, clear_key = jr.split(key)
        time_state_prev = self.time.previous(time_state)
        clearness_prev = self._reset_clearness(key=clear_key)
        max_kwh_prev = self._available_energy(time_state_prev.interval_midpoint, clearness_prev)
        request_constraint_prev = self._new_request_constraint(max_kwh_prev)
        state_prev = SolarState(
            sol_realized_kwh=jnp.zeros((self.num_sol,), jnp.float32),  # unused, overwritten in step
            sol_request_constraint=request_constraint_prev,
            time_state=time_state_prev,
            clearness=clearness_prev,
            key=key,
        )
        return self.step(state_prev, sol_request_kwh=max_kwh_prev, next_time_state=time_state)

    def step(
        self,
        state: SolarState,
        sol_request_kwh: chex.Array,
        next_time_state: DaytimeState | None = None,
    ) -> SolarState:
        request_min_kwh, request_max_kwh = self.request_bounds(state.sol_request_constraint)
        next_realized_kwh = jnp.clip(sol_request_kwh, request_min_kwh, request_max_kwh)

        next_key, subkey = jr.split(state.key)
        next_time_state = (
            self.time.step(state.time_state) if next_time_state is None else next_time_state
        )
        next_clearness = self._new_clearness(state.clearness, subkey)
        max_kwh = self._available_energy(next_time_state.interval_midpoint, next_clearness)
        next_request_constraint = self._new_request_constraint(max_kwh)

        return SolarState(
            sol_realized_kwh=next_realized_kwh,
            sol_request_constraint=next_request_constraint,
            time_state=next_time_state,
            clearness=next_clearness,
            key=next_key,
        )
