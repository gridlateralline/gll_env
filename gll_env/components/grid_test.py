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

from gll_env.algorithms.newton_raphson import NewtonRaphson
from gll_env.components.day_time import DaytimeDynamics
from gll_env.components.grid import GridDynamics


class FakeNewtonRaphson(NewtonRaphson):
    """`NewtonRaphson.solve()` stub, subclassed so static type checkers
    accept it wherever a `NewtonRaphson` is expected (e.g. `GridDynamics.nr`).

    `chex.dataclass`'s default `mappable_dataclass=True` rebuilds the class
    via `type(...)` after `dataclasses.dataclass` runs, so the generated
    frozen `__setattr__` closes over that pre-rebuild class object rather
    than the one actually in this subclass's MRO -- plain `self.x = value`
    hits it and raises `TypeError: super(type, obj): obj must be an
    instance or subtype of type`. `object.__setattr__` bypasses that
    generated `__setattr__` entirely, sidestepping the problem.
    """

    v_bus_out: chex.Array
    s_inj_bus_out: chex.Array
    nr_steps: chex.Array
    converged: chex.Array
    calls: list[dict[str, chex.Array]]

    def __init__(
        self,
        v_bus_out: chex.Array,
        s_inj_bus_out: chex.Array,
        nr_steps: chex.Array,
        converged: chex.Array,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "v_bus_out", v_bus_out)
        object.__setattr__(self, "s_inj_bus_out", s_inj_bus_out)
        object.__setattr__(self, "nr_steps", nr_steps)
        object.__setattr__(self, "converged", converged)
        object.__setattr__(self, "calls", [])

    def solve(
        self,
        v_bus_in: chex.Array,
        s_inj_bus_in: chex.Array,
        pq_id: chex.Array,
        pv_id: chex.Array,
        admittance: chex.Array,
    ) -> tuple[chex.Array, chex.Array, tuple[chex.Array, chex.Array]]:
        self.calls.append(
            {
                "v_bus_in": v_bus_in,
                "s_inj_bus_in": s_inj_bus_in,
                "pq_id": pq_id,
                "pv_id": pv_id,
                "admittance": admittance,
            }
        )
        return self.v_bus_out, self.s_inj_bus_out, (self.nr_steps, self.converged)


def build_grid_model(nr: FakeNewtonRaphson) -> GridDynamics:
    return GridDynamics(
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
        nr=nr,
        time=DaytimeDynamics(n_steps_per_day=jnp.int32(96)),
    )


class TestGridReset:
    def test_reset_initializes_clean_state(self) -> None:
        model = build_grid_model(
            FakeNewtonRaphson(
                v_bus_out=jnp.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=jnp.complex64),
                s_inj_bus_out=jnp.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex64),
                nr_steps=jnp.asarray(0, dtype=jnp.int32),
                converged=jnp.bool_(True),
            )
        )

        state = model.reset()
        expected_s_inj = jnp.ones((model.num_bus,), dtype=jnp.complex64) * jnp.conj(
            model.admittance @ jnp.ones((model.num_bus,), dtype=jnp.complex64)
        )

        assert state.bus_voltage_pu.shape == (model.num_bus,)
        assert jnp.allclose(state.bus_voltage_pu, jnp.ones((model.num_bus,), dtype=jnp.complex64))
        assert jnp.allclose(state.bus_power_injection_pu, expected_s_inj)
        assert int(state.nr_steps) == 0
        assert bool(state.valid)


class TestGridStep:
    def test_step_updates_pq_and_passes_solver_results(self) -> None:
        fake_nr = FakeNewtonRaphson(
            v_bus_out=jnp.array([1.0 + 0.0j, 0.98 - 0.02j], dtype=jnp.complex64),
            s_inj_bus_out=jnp.array([0.2 + 0.1j, 0.3 - 0.2j], dtype=jnp.complex64),
            nr_steps=jnp.asarray(7, dtype=jnp.int32),
            converged=jnp.bool_(False),
        )
        model = build_grid_model(fake_nr)
        state = model.reset()
        p_pq_request_kwh = jnp.array([12.5], dtype=jnp.float32)
        q_pq_request_kvarh = jnp.array([-5.0], dtype=jnp.float32)
        expected_s_pq_pu = model.kwh_to_pu(
            (p_pq_request_kwh + 1j * q_pq_request_kvarh).astype(jnp.complex64)
        )

        next_state = model.step(
            state,
            p_pq_request_kwh=p_pq_request_kwh,
            q_pq_request_kvarh=q_pq_request_kvarh,
        )

        assert len(fake_nr.calls) == 1
        call = fake_nr.calls[0]
        s_inj_bus_in = jnp.asarray(call["s_inj_bus_in"])  # narrow chex.Array for indexing
        s_inj_bus_pu = jnp.asarray(state.bus_power_injection_pu)  # narrow chex.Array for indexing
        assert jnp.allclose(call["v_bus_in"], state.bus_voltage_pu)
        assert jnp.allclose(s_inj_bus_in[model.slack_id], s_inj_bus_pu[model.slack_id])
        assert jnp.allclose(s_inj_bus_in[model.pq_id], expected_s_pq_pu)
        assert jnp.allclose(call["pq_id"], model.pq_id)
        assert jnp.allclose(call["pv_id"], model.pv_id)
        assert jnp.allclose(call["admittance"], model.admittance)

        assert jnp.allclose(next_state.bus_voltage_pu, fake_nr.v_bus_out)
        assert jnp.allclose(next_state.bus_power_injection_pu, fake_nr.s_inj_bus_out)
        assert int(next_state.nr_steps) == 7
        assert not bool(next_state.valid)
