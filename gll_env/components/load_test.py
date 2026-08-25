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
import pytest

from gll_env.components.day_time import DaytimeDynamics, DaytimeState
from gll_env.components.load import LoadDynamics, LoadState


def build_load() -> LoadDynamics:
    return LoadDynamics(
        daily_consumption_kwh=jnp.array([96.0, 48.0], dtype=jnp.float32),
        s_load_max_kva=jnp.array([20.0, 20.0], dtype=jnp.float32),
        load_factor_reversion=jnp.float32(0.2),
        load_factor_std=jnp.float32(0.1),
        power_factor=jnp.float32(0.8),
        time=DaytimeDynamics(n_steps_per_day=jnp.int32(96)),
    )


def test_profile_energy_and_reactive_energy_use_correct_units() -> None:
    load = build_load()
    progress = jnp.float32(0.5)

    active_kwh = load._new_p_load_kwh(progress, jnp.array([1.0, 1.0], dtype=jnp.float32))
    apparent_kvah = load._new_s_load_kvah(progress, jnp.array([1.0, 1.0], dtype=jnp.float32))

    assert jnp.allclose(apparent_kvah.real, active_kwh * load.time.step_duration_h / 0.25)
    assert jnp.all(apparent_kvah.imag > 0.0)
    assert jnp.allclose(apparent_kvah.imag / apparent_kvah.real, load._reactive_to_active_ratio)


def test_step_realizes_previous_forecast_and_clamps_load_factor() -> None:
    load = build_load()
    time_state = DaytimeState(
        interval_start=jnp.float32(0.5),
        interval_end=jnp.float32(49.0 / 96.0),
        day_step=jnp.int32(48),
    )
    forecast = load._new_s_load_kvah(time_state.interval_midpoint, jnp.array([1.0, 1.0]))
    state = LoadState(
        s_load_realized_kvah=jnp.zeros((2,), dtype=jnp.complex64),
        s_load_kvah=forecast,
        load_factor=jnp.array([10.0, -10.0], dtype=jnp.float32),
        time_state=time_state,
        key=jr.PRNGKey(0),
    )

    next_state = load.step(state)

    assert jnp.allclose(next_state.s_load_realized_kvah, forecast)
    assert jnp.all(next_state.load_factor >= 0.0)
    assert jnp.all(next_state.load_factor <= load._load_factor_max)
    assert next_state.time_state.day_step == 49


def test_reset_returns_a_consistent_finite_state() -> None:
    load = build_load()

    state = load.reset(
        jr.PRNGKey(1),
        time_state=DaytimeState(
            interval_start=jnp.float32(0.25),
            interval_end=jnp.float32(25.0 / 96.0),
            day_step=jnp.int32(24),
        ),
    )
    observation = load.observation(state)

    assert jnp.all(jnp.isfinite(state.s_load_kvah))
    assert jnp.all(state.s_load_kvah.real >= 0.0)
    assert jnp.all(state.s_load_kvah.imag >= 0.0)
    assert jnp.all(jnp.isfinite(observation.p_load_forecast))
    assert jnp.all(observation.p_load_forecast >= 0.0)
    assert jnp.all(observation.p_load_forecast <= 1.0)


def test_zero_load_has_finite_zero_observation() -> None:
    load = LoadDynamics(
        daily_consumption_kwh=jnp.array([0.0], dtype=jnp.float32),
        s_load_max_kva=jnp.array([0.0], dtype=jnp.float32),
        time=DaytimeDynamics(n_steps_per_day=jnp.int32(96)),
    )

    state = load.reset(jr.PRNGKey(2))
    observation = load.observation(state)

    assert jnp.all(jnp.isfinite(state.s_load_kvah))
    assert jnp.all(jnp.isfinite(observation.p_load_forecast))
    assert jnp.all(jnp.isfinite(observation.q_load_forecast))
    assert jnp.all(jnp.isfinite(state.load_factor))
    assert jnp.allclose(state.s_load_kvah, 0.0)
    assert jnp.allclose(observation.p_load_forecast, 0.0)
    assert jnp.allclose(observation.q_load_forecast, 0.0)
    # A zero-rated load has _load_factor_max == 0, so the OU process is
    # pinned rather than merely producing zero energy against a live factor.
    assert jnp.allclose(state.load_factor, 0.0)


def test_negative_configuration_values_are_clamped() -> None:
    load = LoadDynamics(
        daily_consumption_kwh=jnp.array([-1.0], dtype=jnp.float32),
        s_load_max_kva=jnp.array([-1.0], dtype=jnp.float32),
        load_factor_reversion=jnp.float32(-1.0),
        load_factor_std=jnp.float32(-1.0),
        power_factor=jnp.float32(-1.0),
    )

    assert jnp.allclose(load.daily_consumption_kwh, 0.0)
    assert jnp.allclose(load.s_load_max_kva, 0.0)
    assert load.load_factor_reversion > 0.0
    assert jnp.allclose(load.load_factor_std, 0.0)
    assert load.power_factor > 0.0


@pytest.mark.parametrize("n_steps_per_day", [12, 24, 48, 96, 192])
def test_daily_energy_matches_the_configured_consumption(n_steps_per_day: int) -> None:
    """Summing the per-interval active energy over one whole day at
    ``load_factor == 1`` returns ``daily_consumption_kwh``, whatever
    resolution the simulation runs at.

    This is the end-to-end units check for the H0 path, and the one that
    actually exercises the profile rescale: the table is read at its own
    fixed 96-slot resolution and rescaled to the simulation's step duration,
    so the two factors must cancel over a full day. Drop the rescale and the
    total is off by ``step_duration_h / 0.25``; apply it twice and it is off
    by the square.

    Away from 96 steps/day the profile is midpoint-sampled rather than
    integrated, so the sum carries a small discretization error -- bounded at
    1% here, which is far tighter than any factor a units slip would
    introduce (the smallest one in play is 2x).
    """
    load = LoadDynamics(
        daily_consumption_kwh=jnp.array([96.0], dtype=jnp.float32),
        s_load_max_kva=jnp.array([20.0], dtype=jnp.float32),
        time=DaytimeDynamics(n_steps_per_day=jnp.int32(n_steps_per_day)),
    )
    unit_load_factor = jnp.array([1.0], dtype=jnp.float32)

    midpoints = (jnp.arange(n_steps_per_day, dtype=jnp.float32) + 0.5) / n_steps_per_day
    daily_kwh = sum(
        float(load._new_s_load_kvah(midpoint, unit_load_factor).real[0]) for midpoint in midpoints
    )

    assert daily_kwh == pytest.approx(96.0, rel=0.01)


def test_apparent_energy_respects_the_nameplate_rating_under_the_ou_process() -> None:
    """``|S| <= s_load_max_kvah`` at every step, for every time of day.

    LoadDynamics claims this as an enforced bound (not a typical value) via
    the ``_load_factor_max`` clip, and both ProsumerDynamics' grid-ball
    origin-feasibility argument and LoadObservation's advertised ``[0, 1]``
    normalized range rest on it. The clip is anchored at the profile's peak
    slot, so this walks a full day at a deliberately noisy load factor to
    confirm the single threshold really does cover every other slot too.
    """
    load = LoadDynamics(
        daily_consumption_kwh=jnp.array([96.0, 40.0], dtype=jnp.float32),
        s_load_max_kva=jnp.array([20.0, 5.0], dtype=jnp.float32),
        load_factor_reversion=jnp.float32(0.05),
        load_factor_std=jnp.float32(0.5),  # deliberately wild, to drive into the clip
        time=DaytimeDynamics(n_steps_per_day=jnp.int32(96)),
    )
    state = load.reset(jr.PRNGKey(7))

    for _ in range(96):
        state = load.step(state)
        assert jnp.all(jnp.abs(state.s_load_kvah) <= load.s_load_max_kvah + 1e-4)
        assert jnp.all(state.s_load_kvah.real >= 0.0)  # a load consumes, never injects

    normalized = load.observation(state).normalize(load)
    assert jnp.all(normalized.p_load_forecast >= 0.0)
    assert jnp.all(normalized.p_load_forecast <= 1.0)
