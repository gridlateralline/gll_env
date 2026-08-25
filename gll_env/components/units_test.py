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

"""Cross-component dimensional audit.

Every physical observation field in this tree is a PER-INTERVAL ENERGY (kWh
/ kvarh / kVAh), and every scale it is normalized by is likewise a
per-interval energy -- a nameplate POWER rating times ``step_duration_h``
(``s_inv_max_kvah``, ``s_load_max_kvah``, ``s_pq_max_kvah``,
``s_sol_max_kwh``, ``peak_charge_per_step_kwh``). Two consequences follow,
and both are pinned here for every component in one sweep:

1. ``observation()`` passes the stored state energy through UNTOUCHED. A
   component that re-applies ``step_duration_h`` on the way out produces
   kVAh*h, which is not a physical quantity at all -- and because that
   product is dimensionally invisible in a single-step-duration test, it can
   sit undetected in code whose every individual line looks plausible.

2. Because numerator and denominator both carry the same factor of
   ``step_duration_h``, the NORMALIZED observation of a fixed physical
   scenario is INVARIANT to the simulation's step duration. This is the
   property that fails loudly on a double-scaling bug: the observation comes
   out scaled by ``step_duration_h`` (or its reciprocal), so running the same
   scenario at two different ``n_steps_per_day`` and comparing is enough to
   expose it, without needing to know the correct value in advance.

Stored battery charge is deliberately excluded from the scenario's scaling:
it is an energy in its own right, not a per-interval flow, so it must NOT
scale with the step duration -- and ``capacity_kwh`` (also not per-interval)
normalizes it to the same invariant fraction.

These live in their own module rather than in each component's tests because
the invariant is a property of the tree as a whole: the bug they guard
against is a component disagreeing with its siblings about what a stored
energy means, which no single component's own tests can see.
"""

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from gll_env.components.battery import BatteryDynamics
from gll_env.components.day_time import DaytimeDynamics
from gll_env.components.inverter import InverterDynamics
from gll_env.components.load import LoadDynamics
from gll_env.components.prosumer import ProsumerDynamics
from gll_env.components.solar import SolarDynamics

# Two step durations that differ by 8x, so a single stray factor of
# step_duration_h can never coincidentally agree between them.
STEP_COUNTS = (12, 96)

# The fixed physical scenario, stated in POWER (kW / kvar) -- the units that
# are genuinely independent of how the day is discretized. Per-interval
# energies are formed as power * step_duration_h at each resolution.
P_BAT_KW = 0.8
P_SOL_KW = 1.5
P_INV_KW, Q_INV_KVAR = 3.0, -1.0
P_LOAD_KW, Q_LOAD_KVAR = 4.0, 1.0
P_PQ_KW, Q_PQ_KVAR = -2.0, 0.5

# Stored charge: an energy in its own right, NOT per-interval -- see module docstring.
BAT_CAPACITY_KWH = 10.0
BAT_FULL_KWH = 6.0


def _f32(*values: float) -> jnp.ndarray:
    return jnp.array(list(values), dtype=jnp.float32)


def build_stack(n_steps_per_day: int) -> dict:
    """One single-prosumer stack, identical in every PHYSICAL respect (all
    ratings are powers, all capacities are energies) apart from how finely
    the day is discretized.
    """
    time = DaytimeDynamics(n_steps_per_day=jnp.int32(n_steps_per_day))
    battery = BatteryDynamics(
        capacity_kwh=_f32(BAT_CAPACITY_KWH),
        charge_rating_kw=_f32(1.0),
        discharge_rating_kw=_f32(2.0),
        time=time,
    )
    solar = SolarDynamics(peak_power_kw=_f32(2.0), time=time)
    inverter = InverterDynamics(
        s_inv_max_kva=_f32(5.0),
        battery_dynamics=battery,
        solar_dynamics=solar,
        time=time,
    )
    load = LoadDynamics(
        daily_consumption_kwh=_f32(96.0),
        s_load_max_kva=_f32(20.0),
        time=time,
    )
    prosumer = ProsumerDynamics(
        s_pq_max_kva=_f32(25.0),
        inverter_id=jnp.array([0], dtype=jnp.int32),
        inverter_dynamics=inverter,
        load_dynamics=load,
        time=time,
    )
    return {
        "time": time,
        "battery": battery,
        "solar": solar,
        "inverter": inverter,
        "load": load,
        "prosumer": prosumer,
    }


def observe_scenario(n_steps_per_day: int) -> dict[str, tuple[float, float]]:
    """Impose the fixed physical scenario at this resolution and report
    ``{field: (raw, normalized)}`` for every energy-valued observation field
    in the tree.

    States are built by ``reset`` then overwritten, so the scenario is exact
    and deterministic rather than whatever the stochastic processes happened
    to produce -- the point here is the units, not the dynamics.
    """
    stack = build_stack(n_steps_per_day)
    h = float(stack["time"].step_duration_h)
    out: dict[str, tuple[float, float]] = {}

    def record(
        name: str,
        raw_field: chex.Array,
        normalized_field: chex.Array,
        expected_raw: float,
    ) -> None:
        raw = float(jnp.ravel(raw_field)[0])
        assert raw == pytest.approx(expected_raw, rel=1e-5, abs=1e-6), (
            f"{name}: observation() altered the stored energy "
            f"({raw} != {expected_raw}); it must pass through unscaled"
        )
        out[name] = (raw, float(jnp.ravel(normalized_field)[0]))

    battery = stack["battery"]
    battery_state = battery.reset(jr.PRNGKey(0)).replace(
        bat_realized_kwh=_f32(P_BAT_KW * h),
        bat_full_kwh=_f32(BAT_FULL_KWH),
        bat_free_kwh=_f32(BAT_CAPACITY_KWH - BAT_FULL_KWH),
    )
    battery_obs = battery.observation(battery_state)
    battery_norm = battery_obs.normalize(battery)
    record(
        "battery.bat_realized", battery_obs.bat_realized, battery_norm.bat_realized, P_BAT_KW * h
    )
    record("battery.bat_full", battery_obs.bat_full, battery_norm.bat_full, BAT_FULL_KWH)
    record(
        "battery.bat_free",
        battery_obs.bat_free,
        battery_norm.bat_free,
        BAT_CAPACITY_KWH - BAT_FULL_KWH,
    )

    solar = stack["solar"]
    solar_state = solar.reset(jr.PRNGKey(0)).replace(sol_realized_kwh=_f32(P_SOL_KW * h))
    solar_obs = solar.observation(solar_state)
    record(
        "solar.sol_realized",
        solar_obs.sol_realized,
        solar_obs.normalize(solar).sol_realized,
        P_SOL_KW * h,
    )

    inverter = stack["inverter"]
    inverter_state = inverter.reset(jr.PRNGKey(0)).replace(
        s_inv_realized_kvah=(_f32(P_INV_KW * h) + 1j * _f32(Q_INV_KVAR * h)).astype(jnp.complex64)
    )
    inverter_obs = inverter.observation(inverter_state)
    inverter_norm = inverter_obs.normalize(inverter)
    record(
        "inverter.p_inv_realized",
        inverter_obs.p_inv_realized,
        inverter_norm.p_inv_realized,
        P_INV_KW * h,
    )
    record(
        "inverter.q_inv_realized",
        inverter_obs.q_inv_realized,
        inverter_norm.q_inv_realized,
        Q_INV_KVAR * h,
    )

    load = stack["load"]
    load_state = load.reset(jr.PRNGKey(0)).replace(
        s_load_realized_kvah=(_f32(P_LOAD_KW * h) + 1j * _f32(Q_LOAD_KVAR * h)).astype(
            jnp.complex64
        )
    )
    load_obs = load.observation(load_state)
    load_norm = load_obs.normalize(load)
    record(
        "load.p_load_realized", load_obs.p_load_realized, load_norm.p_load_realized, P_LOAD_KW * h
    )
    record(
        "load.q_load_realized", load_obs.q_load_realized, load_norm.q_load_realized, Q_LOAD_KVAR * h
    )

    prosumer = stack["prosumer"]
    prosumer_state = prosumer.reset(jr.PRNGKey(0)).replace(
        s_pq_realized_kvah=(_f32(P_PQ_KW * h) + 1j * _f32(Q_PQ_KVAR * h)).astype(jnp.complex64)
    )
    prosumer_obs = prosumer.observation(prosumer_state)
    prosumer_norm = prosumer_obs.normalize(prosumer)
    record(
        "prosumer.p_pq_realized",
        prosumer_obs.p_pq_realized,
        prosumer_norm.p_pq_realized,
        P_PQ_KW * h,
    )
    record(
        "prosumer.q_pq_realized",
        prosumer_obs.q_pq_realized,
        prosumer_norm.q_pq_realized,
        Q_PQ_KVAR * h,
    )

    return out


@pytest.mark.parametrize("n_steps_per_day", STEP_COUNTS)
def test_raw_observations_pass_stored_energy_through_unscaled(n_steps_per_day: int) -> None:
    """observation() reports the stored per-interval energy as-is.

    The assertion itself lives inside observe_scenario's ``record`` helper so
    that the invariance test below cannot silently pass on values that were
    already wrong in the raw domain.
    """
    observed = observe_scenario(n_steps_per_day)

    assert observed, "no observation fields were probed"


def test_normalized_observations_are_invariant_to_step_duration() -> None:
    """The headline dimensional invariant -- see the module docstring.

    A component that scales an already-per-interval energy by
    ``step_duration_h`` on the way out shows up here as that field alone
    differing between the two resolutions, by exactly the ratio of their step
    durations.
    """
    coarse, fine = (observe_scenario(n) for n in STEP_COUNTS)

    assert coarse.keys() == fine.keys()
    mismatched = {
        name: (coarse[name][1], fine[name][1])
        for name in coarse
        if coarse[name][1] != pytest.approx(fine[name][1], rel=1e-5, abs=1e-6)
    }
    assert not mismatched, (
        "normalized observations must not depend on the simulation's step "
        f"duration; these fields do: {mismatched}"
    )


@pytest.mark.parametrize("n_steps_per_day", STEP_COUNTS)
def test_per_interval_ratings_are_nameplate_power_times_step_duration(
    n_steps_per_day: int,
) -> None:
    """Every normalization scale is a nameplate POWER converted to a
    per-interval energy by exactly one factor of ``step_duration_h`` -- the
    other half of what makes the invariance above hold.
    """
    stack = build_stack(n_steps_per_day)
    h = stack["time"].step_duration_h

    assert h == pytest.approx(24.0 / n_steps_per_day)
    assert jnp.allclose(stack["solar"].s_sol_max_kwh, stack["solar"].peak_power_kw * h)
    assert jnp.allclose(stack["inverter"].s_inv_max_kvah, stack["inverter"].s_inv_max_kva * h)
    assert jnp.allclose(stack["load"].s_load_max_kvah, stack["load"].s_load_max_kva * h)
    assert jnp.allclose(stack["prosumer"].s_pq_max_kvah, stack["prosumer"].s_pq_max_kva * h)
    assert jnp.allclose(
        stack["battery"].peak_charge_per_step_kwh, stack["battery"].charge_rating_kw * h
    )
    assert jnp.allclose(
        stack["battery"].peak_discharge_per_step_kwh, stack["battery"].discharge_rating_kw * h
    )
    # Storage capacity is NOT per-interval: it must not pick up the factor.
    assert jnp.allclose(stack["battery"].capacity_kwh, BAT_CAPACITY_KWH)
