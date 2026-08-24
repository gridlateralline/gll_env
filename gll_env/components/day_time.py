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


@chex.dataclass(frozen=True)
class DaytimeState:
    interval_start: chex.Numeric  # float32 in [0, 1)
    interval_end: chex.Numeric  # float32 in (0, 1]
    day_step: chex.Numeric  # int32

    @property
    def interval_midpoint(self) -> chex.Numeric:
        return (
            jnp.asarray(self.interval_start, dtype=jnp.float32)
            + jnp.asarray(self.interval_end, dtype=jnp.float32)
        ) / 2.0


@chex.dataclass(frozen=True)
class DaytimeObservation:
    """Observation of the time of day.

    Attributes:
        day_step: Index of the coming interval.
        interval_start: Start time of the coming interval since midnight.
            Unnormalized values are in ``[0, 24h)``; normalized values are in
            ``[0, 1)``.
        interval_end: End time of the coming interval since midnight,
            represented as 24h at the end of the day. Unnormalized values are
            in ``(0, 24h]``; normalized values are in ``(0, 1]``.
        is_normalized: Defaults to ``False``.
    """

    day_step: chex.Numeric  # () int32
    interval_start: chex.Numeric  # () float32 -- coming interval
    interval_end: chex.Numeric  # () float32 -- coming interval
    is_normalized: bool = field(default=False)

    def normalize(self, daytime_dynamics: "DaytimeDynamics") -> "DaytimeObservation":
        return DaytimeObservation(
            day_step=self.day_step,
            interval_start=self.interval_start / 24.0,
            interval_end=self.interval_end / 24.0,
            is_normalized=True,
        )


@chex.dataclass(frozen=True)
class DaytimeDynamics:
    n_steps_per_day: chex.Numeric = field(
        default_factory=lambda: jnp.int32(24 / 0.25)
    )  # 15-minute intervals

    def __post_init__(self) -> None:
        chex.assert_shape(self.n_steps_per_day, ())
        chex.assert_type(self.n_steps_per_day, jnp.int32)
        object.__setattr__(self, "n_steps_per_day", jnp.maximum(self.n_steps_per_day, 12))

    @cached_property
    def step_duration_h(self) -> chex.Numeric:
        return 24.0 / self.n_steps_per_day

    def _day_step_to_interval(self, day_step: chex.Numeric) -> tuple[chex.Numeric, chex.Numeric]:
        start = jnp.float32(day_step) / self.n_steps_per_day
        end = jnp.float32(day_step + 1) / self.n_steps_per_day
        return start, end

    def observation(self, state: DaytimeState) -> DaytimeObservation:
        return DaytimeObservation(
            day_step=state.day_step,
            interval_start=state.interval_start * 24.0,
            interval_end=state.interval_end * 24.0,
        )

    def reset(self, key: chex.PRNGKey) -> DaytimeState:
        day_step = jr.randint(
            key=key,
            shape=(),
            minval=0,
            maxval=self.n_steps_per_day,
            dtype=jnp.int32,
        )
        interval_start, interval_end = self._day_step_to_interval(day_step)
        return DaytimeState(
            interval_start=interval_start, interval_end=interval_end, day_step=day_step
        )

    def previous(self, state: DaytimeState) -> DaytimeState:
        day_step_prev = jnp.mod(state.day_step - 1, self.n_steps_per_day)
        interval_start, interval_end = self._day_step_to_interval(day_step_prev)
        return DaytimeState(
            interval_start=interval_start, interval_end=interval_end, day_step=day_step_prev
        )

    def step(self, state: DaytimeState) -> DaytimeState:
        day_step = jnp.mod(state.day_step + 1, self.n_steps_per_day)
        interval_start, interval_end = self._day_step_to_interval(day_step)
        return DaytimeState(
            interval_start=interval_start, interval_end=interval_end, day_step=day_step
        )
