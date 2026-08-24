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

"""Config-to-component factories using OmegaConf values.

Everything in ``components/``, ``algorithms/``, and
``grid_asset_generator.py`` is pure Python/JAX with no config dependency;
construct objects directly in tests.

Every physical parameter here is consumed in the same physical units
(kW/kWh/kVA) used by the components. :class:`gll_env.components.grid.GridDynamics`
is the only component with an internal per-unit representation, and it never
leaks past its own ``step``/``pu_to_*``/``*_to_pu`` boundary, so no config-side
unit conversion is needed in this module.

Entry point::

    from gll_env.factories import environment_model
    env_model = environment_model(cfg)

Component tree::

    EnvironmentModel
    ├── GridDynamics (AC power-flow)
    │   └── NewtonRaphson (inner solver)
    └── ProsumerDynamics (orchestrator)
        ├── LoadDynamics
        ├── RadialProjection (shared with InverterDynamics below; solves
        │   against the augmented constraint in ProsumerDynamics)
        └── InverterDynamics (orchestrator)
            ├── BatteryDynamics
            ├── SolarDynamics
            └── RadialProjection (same instance as Prosumer's own; solves
                against its smaller, un-augmented constraint)

Config layout::

    n_steps_per_day: 96
    tolerance: 1.0e-6                # optional shared RadialProjection tolerance
    grid:
        grid_model: cigre_lv_consumer   # asset name
        newton_raphson: {max_iterations: 10, tolerance: 1.0e-4}   # optional
        v_bus_deviation_pu: 0.1                                   # optional
    prosumer:
        s_pq_max_kVA: 15.0              # scalar or per-pq list
        inverter_id: [0, 1, 2]          # optional, defaults to one inverter per pq bus
        load:
            daily_consumption_kWh: 20.0  # scalar or per-pq list
            s_load_max_kVA: 15.0         # scalar or per-pq list
        inverter:
            s_inv_max_kVA: 10.0          # scalar or per-inverter list
            battery:
                capacity_kWh: 10.0
                peak_charge_kW: 5.0
                peak_discharge_kW: 5.0
            solar:
                peak_power_kW: 8.0
"""

from pathlib import Path
from typing import Any

import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

from gll_env.algorithms.newton_raphson import NewtonRaphson
from gll_env.algorithms.radial_projection import RadialProjection
from gll_env.asset_serialization import load_asset_arrays
from gll_env.components.battery import BatteryDynamics
from gll_env.components.day_time import DaytimeDynamics
from gll_env.components.environment import EnvironmentModel
from gll_env.components.grid import GridDynamics
from gll_env.components.inverter import InverterDynamics
from gll_env.components.load import LoadDynamics
from gll_env.components.prosumer import ProsumerDynamics
from gll_env.components.solar import SolarDynamics

GRID_ASSETS_DIR = Path(__file__).with_name("grid_assets")


def _broadcast(value: Any, shape: tuple[int, ...]) -> jnp.ndarray:
    """Broadcast a scalar or per-element config value to `shape`, float32."""
    return jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float32), shape)


def newton_raphson(config: DictConfig) -> NewtonRaphson:
    """Build a :class:`NewtonRaphson` solver.

    Config: ``max_iterations``, ``tolerance`` (both optional).
    """
    return NewtonRaphson(
        max_iterations=int(config.get("max_iterations", 10)),
        tolerance=float(config.get("tolerance", 1e-4)),
    )


def radial_projection(tolerance: float) -> RadialProjection:
    """Build the :class:`RadialProjection` shared by Prosumer and Inverter."""
    return RadialProjection(tolerance=tolerance)


def daytime_dynamics(n_steps_per_day: int) -> DaytimeDynamics:
    """Build the intra-day clock shared by every component below."""
    return DaytimeDynamics(n_steps_per_day=jnp.asarray(n_steps_per_day, dtype=jnp.int32))


def grid_dynamics(config: DictConfig, time: DaytimeDynamics) -> GridDynamics:
    """Load grid topology from a safetensors asset.

    Config: ``grid_model`` (asset name), ``newton_raphson`` (sub-config,
    optional), ``v_bus_deviation_pu`` (optional).
    """
    params: dict[str, Any] = dict(load_asset_arrays(config.grid_model, asset_dir=GRID_ASSETS_DIR))
    if "v_bus_deviation_pu" in config:
        params["v_bus_deviation_pu"] = jnp.asarray(config.v_bus_deviation_pu, dtype=jnp.float32)
    return GridDynamics(
        **params,
        nr=newton_raphson(config.get("newton_raphson", {})),
        time=time,
    )


def battery_dynamics(config: DictConfig, num_inv: int, time: DaytimeDynamics) -> BatteryDynamics:
    """Build the battery leaf.

    Config: ``capacity_kWh``, ``peak_charge_kW``, ``peak_discharge_kW``
    (each a scalar or a per-inverter list).
    """
    shape = (num_inv,)
    return BatteryDynamics(
        capacity_kwh=_broadcast(config.capacity_kWh, shape),
        peak_charge_kw=_broadcast(config.peak_charge_kW, shape),
        peak_discharge_kw=_broadcast(config.peak_discharge_kW, shape),
        time=time,
    )


def solar_dynamics(config: DictConfig, num_inv: int, time: DaytimeDynamics) -> SolarDynamics:
    """Build the solar leaf.

    Config: ``peak_power_kW`` (scalar or per-inverter list), plus optional
    ``clearness_reversion``/``clearness_mean``/``clearness_std``.
    """
    kwargs: dict[str, Any] = {
        key: jnp.asarray(config[key], dtype=jnp.float32)
        for key in ("clearness_reversion", "clearness_mean", "clearness_std")
        if key in config
    }
    return SolarDynamics(
        peak_power_kw=_broadcast(config.peak_power_kW, (num_inv,)),
        time=time,
        **kwargs,
    )


def load_dynamics(config: DictConfig, num_pq: int, time: DaytimeDynamics) -> LoadDynamics:
    """Build the load leaf.

    Config: ``daily_consumption_kWh``, ``s_load_max_kVA`` (each a scalar or
    a per-pq list), plus optional ``load_factor_reversion``/
    ``load_factor_std``/``power_factor``.
    """
    kwargs: dict[str, Any] = {
        key: jnp.asarray(config[key], dtype=jnp.float32)
        for key in ("load_factor_reversion", "load_factor_std", "power_factor")
        if key in config
    }
    shape = (num_pq,)
    return LoadDynamics(
        daily_consumption_kwh=_broadcast(config.daily_consumption_kWh, shape),
        s_load_max_kva=_broadcast(config.s_load_max_kVA, shape),
        time=time,
        **kwargs,
    )


def inverter_dynamics(
    config: DictConfig,
    num_inv: int,
    time: DaytimeDynamics,
    projection: RadialProjection | None,
) -> InverterDynamics:
    """Build the inverter orchestrator.

    Config: ``s_inv_max_kVA`` (scalar or per-inverter list), ``battery``,
    ``solar`` (sub-configs). ``projection`` is the shared
    :class:`RadialProjection` built once in :func:`environment_model` from
    the top-level ``tolerance`` -- ``None`` (no top-level ``tolerance``
    given) is passed straight through so ``InverterDynamics`` falls back to
    its own field default instead.
    """
    kwargs = {} if projection is None else {"projection": projection}
    return InverterDynamics(
        s_inv_max_kva=_broadcast(config.s_inv_max_kVA, (num_inv,)),
        battery_dynamics=battery_dynamics(config.battery, num_inv, time),
        solar_dynamics=solar_dynamics(config.solar, num_inv, time),
        time=time,
        **kwargs,
    )


def prosumer_dynamics(
    config: DictConfig,
    num_pq: int,
    time: DaytimeDynamics,
    projection: RadialProjection | None = None,
) -> ProsumerDynamics:
    """Build the prosumer orchestrator.

    Config: ``s_pq_max_kVA`` (scalar or per-pq list), ``inverter_id``
    (optional list mapping each inverter to its pq bus -- defaults to one
    inverter per pq bus, i.e. ``arange(num_pq)``), ``load``, ``inverter``
    (sub-configs). ``projection`` is shared with the ``InverterDynamics``
    built here -- see :func:`inverter_dynamics`.
    """
    inverter_id = (
        jnp.asarray(config.inverter_id, dtype=jnp.int32)
        if "inverter_id" in config
        else jnp.arange(num_pq, dtype=jnp.int32)
    )
    num_inv = inverter_id.shape[0]

    kwargs = {} if projection is None else {"projection": projection}
    return ProsumerDynamics(
        s_pq_max_kva=_broadcast(config.s_pq_max_kVA, (num_pq,)),
        inverter_id=inverter_id,
        inverter_dynamics=inverter_dynamics(config.inverter, num_inv, time, projection),
        load_dynamics=load_dynamics(config.load, num_pq, time),
        time=time,
        **kwargs,
    )


def environment_model(config: DictConfig) -> EnvironmentModel:
    """Build a complete :class:`EnvironmentModel` from config.

    Config keys: ``n_steps_per_day``, ``grid``, ``prosumer``, and an
    optional top-level ``tolerance`` -- see the module docstring for the
    expected layout. A single :class:`DaytimeDynamics` is built once from
    ``n_steps_per_day`` and shared by every sub-component, matching the
    invariant every ``__post_init__`` in this tree already asserts (all
    share one clock). Likewise, if ``tolerance`` is given, a single
    :class:`RadialProjection` is built once and shared by ``ProsumerDynamics``
    and its ``InverterDynamics``; if omitted, both fall back independently
    to their own field default (``1e-4``, matching ``NewtonRaphson``'s own
    default) rather than being forced to agree on one.
    """
    time = daytime_dynamics(config.n_steps_per_day)
    grid = grid_dynamics(config.grid, time)
    projection = radial_projection(float(config.tolerance)) if "tolerance" in config else None
    prosumer = prosumer_dynamics(
        config.prosumer, num_pq=grid.num_pq, time=time, projection=projection
    )
    return EnvironmentModel(prosumer=prosumer, grid=grid, time=time)


# A small, self-contained scenario for standalone use (jumanji.make(...),
# notebooks, sandboxes) -- the bundled cigre_lv_consumer asset (a residential
# LV feeder, see asset_generator.py) with one inverter per pq bus and modest
# sizing. s_pq_max_kVA=20 comfortably covers s_load_max_kVA=15, satisfying
# ProsumerDynamics.__post_init__'s static origin-feasibility invariant.
_DEFAULT_CONFIG: dict[str, Any] = {
    "n_steps_per_day": 96,
    "grid": {
        "grid_model": "cigre_lv_consumer",
    },
    "prosumer": {
        "s_pq_max_kVA": 20.0,
        "load": {
            "daily_consumption_kWh": 15.0,
            "s_load_max_kVA": 15.0,
        },
        "inverter": {
            "s_inv_max_kVA": 15.0,
            "battery": {
                "capacity_kWh": 10.0,
                "peak_charge_kW": 5.0,
                "peak_discharge_kW": 5.0,
            },
            "solar": {
                "peak_power_kW": 8.0,
            },
        },
    },
}


def default_environment_model() -> EnvironmentModel:
    """Build the default scenario described by :data:`_DEFAULT_CONFIG`.

    No config authoring needed -- this is what
    :func:`default_environment_model` builds for standalone use when no
    generator is supplied.
    """
    return environment_model(OmegaConf.create(_DEFAULT_CONFIG))
