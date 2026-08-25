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
from gll_env.utils import safe_normalize


@chex.dataclass(frozen=True)
class LoadState:
    # Power flow direction convention:
    # positive = consumption (into load)
    # negative = generation = 0 (out of load)
    s_load_realized_kvah: chex.Array  # (num_load,) complex64 -- past interval, realized
    s_load_kvah: chex.Array  # (num_load,) complex64 -- coming interval, fixed

    load_factor: chex.Array  # (num_load,) float32
    time_state: DaytimeState
    key: chex.PRNGKey  # (2,)


@chex.dataclass(frozen=True)
class LoadObservation:
    """Observation of past and forecast load energy.

    Attributes:
        p_load_realized: Past-interval active load energy. Unnormalized values
            are in kWh; normalized values are in ``[0, 1]``.
        q_load_realized: Past-interval reactive load energy. Unnormalized
            values are in kvarh; normalized values are in ``[0, 1]``.
        p_load_forecast: Coming-interval active load energy, with the same
            raw and normalized ranges as ``p_load_realized``.
        q_load_forecast: Coming-interval reactive load energy, with the same
            raw and normalized ranges as ``q_load_realized``.
        is_normalized: Whether the physical energy fields are normalized.
            Defaults to ``False``.
    """

    p_load_realized: chex.Array  # (num_load,) float32 -- past interval
    q_load_realized: chex.Array  # (num_load,) float32 -- past interval
    p_load_forecast: chex.Array  # (num_load,) float32 -- coming interval
    q_load_forecast: chex.Array  # (num_load,) float32 -- coming interval
    is_normalized: bool = field(default=False)

    def normalize(self, load_dynamics: "LoadDynamics") -> "LoadObservation":
        s_load_max_kvah = load_dynamics.s_load_max_kvah
        return LoadObservation(
            p_load_realized=safe_normalize(self.p_load_realized, s_load_max_kvah),
            q_load_realized=safe_normalize(self.q_load_realized, s_load_max_kvah),
            p_load_forecast=safe_normalize(self.p_load_forecast, s_load_max_kvah),
            q_load_forecast=safe_normalize(self.q_load_forecast, s_load_max_kvah),
            is_normalized=True,
        )


@chex.dataclass(frozen=True)
class LoadDynamics:
    """Ornstein-Uhlenbeck load-factor process modulating an H0 standard load profile.

    A dimensionless `load_factor` follows a mean-reverting stochastic
    process (Ornstein-Uhlenbeck) around 1.0. Each H0 profile entry is read
    as the energy consumed during its own fixed 15-minute slot; scaled by
    load_factor and rescaled so that at load_factor == 1.0 the total energy
    over one day matches `daily_consumption_kWh` -- independent of the
    simulation's own `n_steps_per_day`, since the H0 profile lookup remaps
    the shared day_progress into the table's own fixed 96-slot resolution,
    then the result is rescaled from that fixed slot duration to whatever
    the simulation's own step duration is (see _new_s_load_kVAh). Reactive
    energy is derived deterministically from active energy at a fixed
    `power_factor`.

    Keeping the noise on the dimensionless load_factor rather than directly
    on the power itself means noise_scale has the same relative effect at
    every time of day, instead of implicitly tracking the profile shape.
    """

    daily_consumption_kwh: chex.Array  # (num_load,) float32 -- total energy consumed per day
    s_load_max_kva: chex.Array  # (num_load,) float32 -- nameplate max apparent-power draw

    load_factor_reversion: chex.Array = field(default_factory=lambda: jnp.float32(0.05))
    load_factor_std: chex.Array = field(default_factory=lambda: jnp.float32(0.1))
    power_factor: chex.Array = field(default_factory=lambda: jnp.float32(0.95))
    time: DaytimeDynamics = field(default_factory=DaytimeDynamics)

    _H0_PROFILE = jnp.array(
        [
            24.849,
            23.605,
            22.732,
            21.924,
            21.275,
            20.630,
            20.100,
            19.651,
            19.425,
            19.212,
            19.024,
            18.892,
            18.902,
            18.804,
            18.792,
            18.726,
            19.013,
            19.205,
            19.539,
            20.054,
            21.112,
            21.535,
            22.381,
            22.901,
            24.469,
            25.242,
            25.837,
            26.271,
            26.798,
            27.122,
            27.125,
            26.694,
            26.371,
            26.412,
            26.272,
            26.272,
            26.568,
            26.709,
            26.649,
            26.668,
            26.710,
            26.871,
            27.321,
            27.847,
            28.688,
            29.553,
            30.346,
            30.955,
            31.414,
            31.470,
            31.245,
            30.953,
            30.675,
            30.528,
            30.245,
            29.855,
            29.621,
            29.387,
            29.070,
            29.221,
            29.453,
            29.481,
            29.601,
            30.035,
            30.777,
            31.376,
            32.060,
            32.937,
            34.282,
            35.568,
            36.860,
            38.250,
            39.435,
            40.511,
            41.439,
            42.122,
            42.278,
            42.402,
            42.516,
            42.570,
            42.483,
            42.451,
            42.476,
            42.577,
            42.430,
            41.502,
            40.547,
            39.723,
            38.550,
            36.435,
            34.594,
            32.689,
            30.886,
            29.182,
            27.818,
            26.488,
        ],
        dtype=jnp.float32,
    )

    @cached_property
    def num_load(self) -> int:
        return jnp.atleast_1d(self.daily_consumption_kwh).shape[0]

    @cached_property
    def _h0_steps_per_day(self) -> int:
        """Resolution of the H0 profile table itself, inferred from its length."""
        return self._H0_PROFILE.shape[0]

    @cached_property
    def _h0_hours_per_step(self) -> chex.Array:
        """Duration of one H0 profile sample, inferred so the table spans 24 hours."""
        return jnp.float32(24.0 / self._h0_steps_per_day)

    @cached_property
    def _profile_total(self) -> chex.Array:
        """Total (relative) energy represented by the raw H0 profile shape
        over one day: each entry already represents the energy consumed
        during its own fixed 15-minute slot (the native BDEW H0 convention),
        so this is just their sum -- no extra hours-per-step factor needed
        here. (Reading the table as power-per-slot instead, and folding
        _h0_hours_per_step into this sum, would cancel out identically once
        _new_s_load_kVAh's final rescale is applied -- the two conventions
        are mathematically interchangeable. Energy-per-slot is the more
        direct reading, so that's what the code follows.)
        """
        return jnp.sum(self._H0_PROFILE)

    @cached_property
    def _reactive_to_active_ratio(self) -> chex.Array:
        """tan(phi), where phi = arccos(power_factor): Q/P at the configured power factor."""
        return jnp.tan(jnp.arccos(self.power_factor))

    @cached_property
    def _load_factor_stationary_std(self) -> chex.Array:
        """Stationary std of the load_factor AR(1) process: load_factor_std is
        the *one-step* noise std, not the equilibrium spread. Same derivation
        as SolarDynamics._reset_clearness's stationary_std -- used to seed
        reset() with a sample that actually looks like a typical point on the
        settled trajectory, instead of under-dispersed around the mean.
        """
        phi = 1.0 - self.load_factor_reversion
        return self.load_factor_std / jnp.sqrt(1.0 - phi**2)

    @cached_property
    def _p_load_reference_kwh(self) -> chex.Array:
        """Nominal (load_factor==1) per-simulation-interval active energy at
        the H0 profile's own peak slot. Private: purely an intermediate used
        to derive _load_factor_max below, not a bound in its own right (that
        role now belongs to s_load_max_kVAh -- see its docstring for why
        this distinction matters).
        """
        scale_factor = self.daily_consumption_kwh / self._profile_total
        rescale = self.time.step_duration_h / self._h0_hours_per_step
        return jnp.max(self._H0_PROFILE) * scale_factor * rescale

    @cached_property
    def _s_load_reference_kvah(self) -> chex.Array:
        """Nominal (load_factor==1) apparent energy at the profile's peak
        slot: |S| = P / power_factor. Private -- see _p_load_reference_kWh.
        """
        return self._p_load_reference_kwh / self.power_factor

    @cached_property
    def s_load_max_kvah(self) -> chex.Array:
        """The load's nameplate apparent-power rating, converted to a
        per-interval energy bound -- same pattern as
        InverterDynamics.s_inv_max_kvah. Unlike the old version of this
        property (now private, _s_load_reference_kVAh), this IS an enforced
        bound: load_factor is clipped (see _load_factor_max) so that worst
        case |S| -- at the profile's peak slot -- never exceeds this.
        That's what lets LoadObservation report genuine [0, 1] ranges
        instead of "typically near [0, 1]", and what lets
        ProsumerDynamics/EnvironmentModel clamp s_pq_max_kva up to cover
        s_load_max_kva at construction time, rather than only ever checking
        the grid-ball origin-feasibility invariant at runtime.
        """
        return self.s_load_max_kva * self.time.step_duration_h

    @cached_property
    def _load_factor_max(self) -> chex.Array:
        """The load_factor value at which worst-case |S| (profile's peak
        slot) exactly equals s_load_max_kVAh. Since every quantity in the
        p_load/q_load derivation is linear in load_factor, this ONE
        conservative threshold -- anchored at the peak slot -- keeps |S|
        within s_load_max_kVAh at every OTHER time of day too (all other
        slots are comfortably under the peak already). Clipped to at both
        _new_load_factor (the OU step) and _reset_load_factor (the
        fabricated previous interval), so load_factor is now bounded on
        both sides: >= 0 (can't inject) and <= _load_factor_max (can't
        exceed the load's own nameplate rating).
        """
        reference = self._s_load_reference_kvah
        return jnp.where(
            reference > 0.0,
            self.s_load_max_kvah / jnp.maximum(reference, jnp.finfo(reference.dtype).tiny),
            jnp.where(self.s_load_max_kvah > 0.0, 1.0, 0.0),
        )

    def __post_init__(self) -> None:
        chex.assert_shape(self.daily_consumption_kwh, (self.num_load,))
        chex.assert_shape(self.s_load_max_kva, (self.num_load,))
        chex.assert_shape(self.load_factor_reversion, ())
        chex.assert_shape(self.load_factor_std, ())
        chex.assert_shape(self.power_factor, ())
        chex.assert_type(self.daily_consumption_kwh, jnp.float32)
        chex.assert_type(self.s_load_max_kva, jnp.float32)
        chex.assert_type(self.load_factor_reversion, jnp.float32)
        chex.assert_type(self.load_factor_std, jnp.float32)
        chex.assert_type(self.power_factor, jnp.float32)
        object.__setattr__(
            self, "daily_consumption_kwh", jnp.maximum(self.daily_consumption_kwh, 0.0)
        )
        object.__setattr__(self, "s_load_max_kva", jnp.maximum(self.s_load_max_kva, 0.0))
        object.__setattr__(
            self, "load_factor_reversion", jnp.clip(self.load_factor_reversion, 1e-6, 1.0)
        )
        object.__setattr__(self, "load_factor_std", jnp.maximum(self.load_factor_std, 0.0))
        object.__setattr__(self, "power_factor", jnp.clip(self.power_factor, 1e-6, 1.0))

    def observation(self, state: LoadState) -> LoadObservation:
        return LoadObservation(
            p_load_realized=state.s_load_realized_kvah.real,
            q_load_realized=state.s_load_realized_kvah.imag,
            p_load_forecast=state.s_load_kvah.real,
            q_load_forecast=state.s_load_kvah.imag,
        )

    def _new_load_factor(self, load_factor: chex.Array, key: chex.PRNGKey) -> chex.Array:
        load_factor = jnp.atleast_1d(load_factor)
        iid_noise = jr.normal(key, shape=load_factor.shape)
        load_factor = (
            load_factor
            + self.load_factor_reversion * (1.0 - load_factor)
            + self.load_factor_std * iid_noise
        )
        return jnp.clip(
            load_factor, 0.0, self._load_factor_max
        )  # 0: can't inject; max: nameplate rating

    def _reset_load_factor(self, key: chex.PRNGKey) -> chex.Array:
        """Sample the fabricated previous-interval load_factor from the
        process's stationary distribution, not the one-step noise std --
        see _load_factor_stationary_std.
        """
        load_factor = 1.0 + self._load_factor_stationary_std * jr.normal(
            key=key, shape=(self.num_load,)
        )
        return jnp.clip(load_factor, 0.0, self._load_factor_max)

    def _profile_index(self, day_progress: chex.Numeric) -> chex.Array:
        """Map a fraction-of-day in [0, 1) to its 15-minute H0 profile slot."""
        day_progress = jnp.asarray(day_progress, dtype=jnp.float32)
        return jnp.clip(
            jnp.floor(
                day_progress * self._h0_steps_per_day + jnp.finfo(day_progress.dtype).eps
            ).astype(jnp.int32),
            0,
            self._h0_steps_per_day - 1,
        )

    def _new_p_load_kwh(self, day_progress: chex.Numeric, load_factor: chex.Array) -> chex.Array:
        """Active energy for one fixed H0 slot (0.25h), scaled by load_factor.
        Still needs rescaling to the simulation's own step duration -- see
        _new_s_load_kVAh.
        """
        idx = self._profile_index(day_progress)
        profile_value = self._H0_PROFILE[idx]

        # scale_factor makes the raw profile shape sum to daily_consumption_kWh
        # over one day at load_factor == 1, independent of the simulation's
        # own n_steps_per_day.
        scale_factor = self.daily_consumption_kwh / self._profile_total
        return profile_value * scale_factor * load_factor

    def _new_q_load_kvarh(self, p_load_kwh: chex.Array) -> chex.Array:
        return p_load_kwh * self._reactive_to_active_ratio

    def _new_s_load_kvah(self, day_progress: chex.Numeric, load_factor: chex.Array) -> chex.Array:
        # Primary operation uses 15-minute steps, so midpoint sampling of H0
        # is intentionally retained as a constant-over-an-interval estimate.
        p_load_kwh = self._new_p_load_kwh(day_progress, load_factor)  # per fixed H0 slot (0.25h)
        q_load_kvarh = self._new_q_load_kvarh(p_load_kwh)
        s_load_kvah = p_load_kwh + 1j * q_load_kvarh  # still per H0 slot, not per sim step
        # Rescale from the H0 table's own fixed slot duration to the
        # simulation's own step duration. Without this, s_load is off by a
        # factor of step_duration_h / _h0_hours_per_step wherever it's later
        # combined with the rest of the (per-sim-interval) energy quantities
        # (e.g. ProsumerDynamics._s_pq_realized_kvah).
        return (s_load_kvah * (self.time.step_duration_h / self._h0_hours_per_step)).astype(
            jnp.complex64
        )

    def reset(self, key: chex.PRNGKey, time_state: DaytimeState | None = None) -> LoadState:
        """Initialize by fabricating a plausible previous interval (day_step -
        1, wrapped) with no-history load_factor sampling from the process's
        stationary distribution, then stepping the real OU dynamics forward
        once via `step`. Mirrors SolarDynamics.reset() exactly.
        """
        if time_state is None:
            key, time_key = jr.split(key)
            time_state = self.time.reset(time_key)

        key, subkey = jr.split(key)
        time_state_prev = self.time.previous(time_state)
        load_factor_prev = self._reset_load_factor(subkey)
        s_load_kvah_prev = self._new_s_load_kvah(
            time_state_prev.interval_midpoint, load_factor_prev
        )

        fictional_prev_state = LoadState(
            s_load_realized_kvah=jnp.zeros_like(
                s_load_kvah_prev
            ),  # unused, no interval before this one
            s_load_kvah=s_load_kvah_prev,
            load_factor=load_factor_prev,
            time_state=time_state_prev,
            key=key,
        )
        return self.step(fictional_prev_state, next_time_state=time_state)

    def step(self, state: LoadState, next_time_state: DaytimeState | None = None) -> LoadState:
        """Advance the load one step via an OU update of load_factor."""
        next_key, subkey = jr.split(state.key)
        next_time_state = (
            self.time.step(state.time_state) if next_time_state is None else next_time_state
        )
        next_load_factor = self._new_load_factor(state.load_factor, subkey)
        next_s_load_kvah = self._new_s_load_kvah(
            next_time_state.interval_midpoint, next_load_factor
        )
        # The interval that just ended consumed exactly what the incoming
        # state's s_load_kVAh said it would -- exogenous, no request/clip
        # step, same relationship as solar's sol_realized_kWh has to the
        # previous state's sol_max_kWh.
        next_s_load_realized_kvah = state.s_load_kvah
        return LoadState(
            s_load_realized_kvah=next_s_load_realized_kvah,
            s_load_kvah=next_s_load_kvah,
            load_factor=next_load_factor,
            time_state=next_time_state,
            key=next_key,
        )
