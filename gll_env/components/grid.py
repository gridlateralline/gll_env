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

"""AC grid component — fixed concrete class.

:class:`GridDynamics` runs Newton-Raphson AC power-flow on a static network topology.
Network parameters (admittance matrix, bus indices, base quantities) are loaded from
a safetensors asset; the solver is :class:`~algorithms.newton_raphson.NewtonRaphson`.

There is no abstract base class here — the power-flow calculation is fixed.  Scenario
variation comes from swapping the *prosumer leaf components*, not the grid solver.
"""

from dataclasses import field
from functools import cached_property

import chex
import jax.numpy as jnp

from gll_env.algorithms.newton_raphson import NewtonRaphson
from gll_env.components.day_time import DaytimeDynamics


@chex.dataclass(frozen=True)
class GridState:
    """State of the AC grid at the end of a simulation interval.

    :var v_bus_pu: Complex voltage at each bus in pu, shape (num_bus,).
    :var s_inj_bus_pu: Complex power injection at each bus in pu, shape (num_bus,).
    :var nr_steps: Number of Newton-Raphson iterations taken this step.
    :var valid: True iff the power-flow solver converged.
    """

    v_bus_pu: chex.Array  # (num_bus,) complex64
    s_inj_bus_pu: chex.Array  # (num_bus,) complex64
    nr_steps: chex.Array
    valid: chex.Array


@chex.dataclass(frozen=True)
class GridObservation:
    # Kept approximately in [-1, 1] for consistent RL-agent input scaling
    v_bus_mag: chex.Array  # (num_bus,) float32 -- (|v_bus_pu| - 1.0) / v_bus_deviation_pu.
    v_bus_angle: chex.Array  # (num_bus,) float32 in (-1, 1] -- v_bus_angle_rad / pi.
    p_inj_bus: chex.Array  # (num_bus,) float32 -- real power injection, already in normalized pu.
    q_inj_bus: chex.Array  # (num_bus,) float32 -- reactive power injection, pu


@chex.dataclass(frozen=True)
class GridDynamics:
    """Newton-Raphson AC power-flow grid.

    Parameters are loaded from a safetensors asset file (e.g. ``case9``).
    Complex arrays (admittance) are stored as split real/imag pairs in the
    asset file and rehydrated on load.

    Parameters
    ----------
    slack_id: shape (1,) int32
    pq_id: shape (num_pq,) int32
    pv_id: shape (num_pv,) int32
    base_s_mva: scalar float32 — system MVA base
    base_v_kv: shape (num_bus,) float32 — per-bus voltage base in kV
    admittance: shape (num_bus, num_bus) complex64 — nodal admittance matrix
    position: shape (num_bus, 2) float32 — bus (x, y) coordinates
    nr: Newton-Raphson solver instance
    time: DaytimeDynamics — owns the simulation's step duration, used for the
        internal energy<->per-unit-power conversion (see kwh_to_pu).
    v_bus_deviation_pu: scalar float32 — reference deviation from nominal
        (1.0 pu) used to scale GridObservation.v_bus_mag into roughly
        [-1, 1]. NOT an enforced bound -- purely an observation-normalization
        reference, default 0.1 (a common +-10% grid-code operating band).
    """

    slack_id: chex.Array  # (1,) int32
    pq_id: chex.Array  # (num_pq,) int32
    pv_id: chex.Array  # (num_pv,) int32
    base_s_mva: chex.Array  # () float32
    base_v_kv: chex.Array  # (num_bus,) float32
    admittance: chex.Array  # (num_bus, num_bus) complex64
    position: chex.Array  # (num_bus, 2) float32
    nr: NewtonRaphson
    time: DaytimeDynamics = field(default_factory=DaytimeDynamics)
    v_bus_deviation_pu: chex.Array = field(default_factory=lambda: jnp.float32(0.1))

    @cached_property
    def num_pq(self) -> int:
        return jnp.atleast_1d(self.pq_id).shape[0]

    @cached_property
    def num_pv(self) -> int:
        return jnp.atleast_1d(self.pv_id).shape[0]

    @cached_property
    def num_bus(self) -> int:
        return 1 + self.num_pq + self.num_pv

    def pu_to_mw(self, s_pu: chex.Array) -> chex.Array:
        """Convert power from per-unit to MW."""
        return s_pu * self.base_s_mva

    def mw_to_pu(self, s_mw: chex.Array) -> chex.Array:
        """Convert power from MW to per-unit."""
        return s_mw / self.base_s_mva

    def pu_to_kw(self, s_pu: chex.Array) -> chex.Array:
        """Convert power from per-unit to kW."""
        return self.pu_to_mw(s_pu) * 1000.0

    def kw_to_pu(self, s_kw: chex.Array) -> chex.Array:
        """Convert power from kW to per-unit."""
        return self.mw_to_pu(s_kw / 1000.0)

    def pu_to_mwh(self, e_pu: chex.Array) -> chex.Array:
        """Convert a per-unit POWER value into the MWh it represents over one
        simulation interval (pu -> power -> energy). NOT the inverse of a
        "per-unit energy" concept -- there isn't one. `e_pu` here is read as
        a power, exactly like pu_to_mw's input; the "e" naming and the `* h`
        just reflect what the caller is about to do with the result.
        """
        return self.pu_to_mw(e_pu) * self.time.step_duration_h

    def mwh_to_pu(self, e_mwh: chex.Array) -> chex.Array:
        """Convert a per-interval energy (MWh) into per-unit POWER (not a
        "per-unit energy" -- pu is always power-normalized here) by first
        converting to the constant MW level that would have produced that
        energy over one simulation interval, then normalizing by base_s_mva.
        """
        return self.mw_to_pu(e_mwh / self.time.step_duration_h)

    def pu_to_kwh(self, e_pu: chex.Array) -> chex.Array:
        """Convert a per-unit POWER value into the kWh it represents over one
        simulation interval. See pu_to_mwh.
        """
        return self.pu_to_mwh(e_pu) * 1000.0

    def kwh_to_pu(self, e_kwh: chex.Array) -> chex.Array:
        """Convert a per-interval energy (kWh) into per-unit POWER -- the
        quantity the power-flow solver actually operates on. See mwh_to_pu
        for why the result is power-domain pu, not a separate energy-pu.

        This conversion is entirely internal to Grid: step() accepts
        physical energy (p_pq_request_kWh/q_pq_request_kvarh) directly and
        calls this itself. No other module needs to call kwh_to_pu/pu_to_kwh
        -- keeping the pu representation fully behind Grid's own API surface
        is the point (matches every other component's step(), which takes
        physical kWh/kvarh, never a bespoke internal representation).
        """
        return self.mwh_to_pu(e_kwh / 1000.0)

    def pu_to_kv(self, v_pu: chex.Array) -> chex.Array:
        """Convert voltage from per-unit to kV."""
        chex.assert_shape(v_pu, (self.num_bus,))
        return v_pu * self.base_v_kv

    def kv_to_pu(self, v_kv: chex.Array) -> chex.Array:
        """Convert voltage from kV to per-unit."""
        chex.assert_shape(v_kv, (self.num_bus,))
        return v_kv / self.base_v_kv

    def __post_init__(self) -> None:
        chex.assert_shape(self.slack_id, (1,))
        chex.assert_shape(self.pq_id, (self.num_pq,))
        chex.assert_shape(self.pv_id, (self.num_pv,))
        chex.assert_shape(self.base_s_mva, ())
        chex.assert_shape(self.base_v_kv, (self.num_bus,))
        chex.assert_shape(self.admittance, (self.num_bus, self.num_bus))
        chex.assert_shape(self.position, (self.num_bus, 2))
        chex.assert_shape(self.v_bus_deviation_pu, ())
        chex.assert_type(self.slack_id, jnp.int32)
        chex.assert_type(self.pq_id, jnp.int32)
        chex.assert_type(self.pv_id, jnp.int32)
        chex.assert_type(self.base_s_mva, jnp.float32)
        chex.assert_type(self.base_v_kv, jnp.float32)
        chex.assert_type(self.admittance, jnp.complex64)
        chex.assert_type(self.position, jnp.float32)
        chex.assert_type(self.v_bus_deviation_pu, jnp.float32)
        # base_s_mva/base_v_kv/v_bus_deviation_pu are divisors throughout the
        # pu<->physical conversions and the observation normalization below;
        # unlike a component rating, a non-positive base isn't a meaningful
        # "disabled" state, just invalid data, so clamp to a tiny positive
        # floor (not zero) to avoid NaN/Inf instead of a hard, exploding
        # result.
        tiny = jnp.finfo(jnp.float32).tiny
        object.__setattr__(self, "base_s_mva", jnp.maximum(self.base_s_mva, tiny))
        object.__setattr__(self, "base_v_kv", jnp.maximum(self.base_v_kv, tiny))
        object.__setattr__(self, "v_bus_deviation_pu", jnp.maximum(self.v_bus_deviation_pu, tiny))

    def to_asset_dict(self) -> dict:
        """Return a dict of arrays suitable for passing to :func:`save_arrays`.

        Deliberately excludes `time`/`v_bus_deviation_pu`: those are runtime
        simulation/observation settings, not network-topology parameters, so
        they don't belong in the electrical-network asset file.
        """
        return {
            "slack_id": self.slack_id,
            "pq_id": self.pq_id,
            "pv_id": self.pv_id,
            "base_s_mva": self.base_s_mva,
            "base_v_kv": self.base_v_kv,
            "admittance": self.admittance,
            "position": self.position,
        }

    def observation(self, state: GridState) -> GridObservation:
        return GridObservation(
            v_bus_mag=(jnp.abs(state.v_bus_pu) - 1.0) / self.v_bus_deviation_pu,
            v_bus_angle=jnp.angle(state.v_bus_pu) / jnp.pi,
            p_inj_bus=state.s_inj_bus_pu.real,
            q_inj_bus=state.s_inj_bus_pu.imag,
        )

    def reset(self) -> GridState:
        v_bus_pu = jnp.ones((self.num_bus,), dtype=jnp.complex64)
        s_inj_bus_pu = v_bus_pu * jnp.conj(self.admittance @ v_bus_pu)
        return GridState(
            v_bus_pu=v_bus_pu,
            s_inj_bus_pu=s_inj_bus_pu,
            nr_steps=jnp.asarray(0, dtype=jnp.int32),
            valid=jnp.bool_(True),
        )

    def _update_pq(self, state: GridState, s_pq_pu: chex.Array) -> GridState:
        s_inj_bus_pu = jnp.array(state.s_inj_bus_pu).at[self.pq_id].set(s_pq_pu)
        return GridState(
            v_bus_pu=state.v_bus_pu,
            s_inj_bus_pu=s_inj_bus_pu,
            nr_steps=jnp.zeros_like(state.nr_steps),
            valid=jnp.bool_(False),
        )

    def _clean(self, state: GridState) -> GridState:
        v_bus_pu_out, s_inj_bus_pu_out, (nr_steps, nr_converged) = self.nr.solve(
            v_bus_in=state.v_bus_pu,
            s_inj_bus_in=state.s_inj_bus_pu,
            pq_id=self.pq_id,
            pv_id=self.pv_id,
            admittance=self.admittance,
        )
        return GridState(
            v_bus_pu=v_bus_pu_out,
            s_inj_bus_pu=s_inj_bus_pu_out,
            nr_steps=nr_steps,
            valid=nr_converged,
        )

    def step(
        self,
        state: GridState,
        p_pq_request_kwh: chex.Array,
        q_pq_request_kvarh: chex.Array,
    ) -> GridState:
        """Advance the grid one interval given a physical net-injection
        request per PQ bus (per-interval energy: kWh for p, kvarh for q,
        matching every other component's step() in this tree). The
        conversion to the per-unit power the Newton-Raphson solver needs is
        entirely internal -- see kwh_to_pu.
        """
        chex.assert_shape(p_pq_request_kwh, (self.num_pq,))
        chex.assert_shape(q_pq_request_kvarh, (self.num_pq,))
        chex.assert_type(p_pq_request_kwh, jnp.float32)
        chex.assert_type(q_pq_request_kvarh, jnp.float32)

        s_pq_kvah = (p_pq_request_kwh + 1j * q_pq_request_kvarh).astype(jnp.complex64)
        s_pq_pu = self.kwh_to_pu(s_pq_kvah)

        return self._clean(self._update_pq(state, s_pq_pu))
