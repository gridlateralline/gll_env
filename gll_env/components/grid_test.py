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


def build_grid_model(nr: FakeNewtonRaphson, n_steps_per_day: int = 96) -> GridDynamics:
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
        time=DaytimeDynamics(n_steps_per_day=jnp.int32(n_steps_per_day)),
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


class TestGridUnits:
    """Grid is the one place in the tree where per-interval ENERGY has to
    become POWER.

    Every component upstream speaks kWh/kvarh per interval, but an AC
    power-flow solution is a statement about instantaneous power: the
    admittance matrix relates voltages to power injections, with no notion of
    an interval at all. ``kwh_to_pu`` is the conversion, and it is the only
    correct place in the tree to divide by ``step_duration_h`` -- which is
    exactly why the factor must appear here and nowhere else.
    """

    @staticmethod
    def _model(n_steps_per_day: int) -> GridDynamics:
        return build_grid_model(
            FakeNewtonRaphson(
                v_bus_out=jnp.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=jnp.complex64),
                s_inj_bus_out=jnp.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex64),
                nr_steps=jnp.asarray(3, dtype=jnp.int32),
                converged=jnp.asarray(True),
            ),
            n_steps_per_day=n_steps_per_day,
        )

    def test_kwh_to_pu_converts_energy_to_the_power_that_delivers_it(self) -> None:
        """100 kWh drawn over a 2h interval is 50 kW, not 100 kW."""
        model = self._model(n_steps_per_day=12)  # step_duration_h == 2.0

        power_pu = model.kwh_to_pu(jnp.float32(100.0))

        assert float(model.pu_to_kw(power_pu)) == pytest.approx(50.0)
        assert float(power_pu) == pytest.approx(100.0 / 2.0 / 1000.0 / float(model.base_s_mva))

    @pytest.mark.parametrize("n_steps_per_day", [12, 96])
    def test_energy_pu_round_trip_is_the_identity(self, n_steps_per_day: int) -> None:
        model = self._model(n_steps_per_day)
        energy_kwh = jnp.array([100.0, -37.5], dtype=jnp.float32)

        assert jnp.allclose(model.pu_to_kwh(model.kwh_to_pu(energy_kwh)), energy_kwh, rtol=1e-5)
        assert jnp.allclose(model.pu_to_kw(model.kw_to_pu(energy_kwh)), energy_kwh, rtol=1e-5)

    def test_the_solver_sees_the_same_power_at_any_step_duration(self) -> None:
        """The physical scenario -- a constant 50 kW / 20 kvar draw -- is one
        thing; how finely the day is discretized is another. The injection
        handed to Newton-Raphson must depend only on the former.

        Read off the FakeNewtonRaphson's recorded call rather than off a
        conversion helper, so this covers step()'s own wiring and not just the
        arithmetic in isolation.
        """
        injections = []
        for n_steps_per_day in (12, 96):
            model = self._model(n_steps_per_day)
            step_duration_h = float(model.time.step_duration_h)
            state = model.reset()

            model.step(
                state=state,
                p_pq_request_kwh=jnp.array([50.0 * step_duration_h], dtype=jnp.float32),
                q_pq_request_kvarh=jnp.array([20.0 * step_duration_h], dtype=jnp.float32),
            )
            recorded = model.nr.calls[-1]["s_inj_bus_in"]
            injections.append(recorded[model.pq_id])

        assert jnp.allclose(injections[0], injections[1], rtol=1e-5), (
            "per-unit injection must depend on the physical power, not on "
            f"n_steps_per_day; got {injections[0]} vs {injections[1]}"
        )
        # And it is the right value: 50 kW / (base_s_mva * 1000).
        model = self._model(96)
        assert jnp.allclose(injections[0].real, 50.0 / (float(model.base_s_mva) * 1000.0))


class TestGridConfiguration:
    def test_non_positive_voltage_deviation_reference_is_clamped(self) -> None:
        """The clamp has to land on the field ``normalize()`` actually reads.

        ``voltage_deviation_ref_pu`` is a divisor in the observation
        normalization, so ``__post_init__`` floors it at a tiny positive
        value. Asserted on the public field rather than on whatever
        ``__post_init__`` happens to write, because an earlier version wrote
        the clamped value to a differently-named attribute and left the real
        field untouched -- silently inert, since ``safe_normalize`` maps the
        unclamped non-positive scale to zero instead of raising.
        """
        model = build_grid_model(
            FakeNewtonRaphson(
                v_bus_out=jnp.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=jnp.complex64),
                s_inj_bus_out=jnp.array([0.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex64),
                nr_steps=jnp.asarray(0, dtype=jnp.int32),
                converged=jnp.asarray(True),
            ),
        )
        object.__setattr__(model, "voltage_deviation_ref_pu", jnp.float32(-0.5))
        model.__post_init__()

        assert float(model.voltage_deviation_ref_pu) > 0.0
        assert not hasattr(model, "v_bus_deviation_ref")
