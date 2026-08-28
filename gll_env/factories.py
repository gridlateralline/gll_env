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
``assets.components_grid.generator`` is pure Python/JAX with no config dependency;
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
        ├── RadialFeasibility (shared with InverterDynamics below; solves
        │   against the augmented constraint in ProsumerDynamics)
        └── InverterDynamics (orchestrator)
            ├── BatteryDynamics
            ├── SolarDynamics
            └── RadialFeasibility (same instance as Prosumer's own; solves
                against its smaller, un-augmented constraint)

Config layout::

    n_steps_per_day: 96
    tolerance: 1.0e-6                # optional shared RadialFeasibility tolerance
    grid:
        grid_model: cigre_lv_consumer   # asset name
        newton_raphson: {max_iterations: 10, tolerance: 1.0e-4}   # optional
        v_bus_deviation_pu: 0.1                                   # optional
    grid_code:                       # optional; absent means no law, action_dim 2
        q_of_u: true                 # NE7 4.3.2 Q(U) curve -> action_dim 1
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
from gll_env.algorithms.radial_feasibility import RadialFeasibility
from gll_env.assets.serialization import load_asset_arrays
from gll_env.components.battery import BatteryDynamics
from gll_env.components.day_time import DaytimeDynamics
from gll_env.components.environment import EnvironmentDynamics
from gll_env.components.grid import GridDynamics
from gll_env.components.grid_code import GridCode, QofUCharacteristic, rated_q_max_kvar
from gll_env.components.inverter import InverterDynamics
from gll_env.components.load import LoadDynamics
from gll_env.components.prosumer import ProsumerDynamics
from gll_env.components.solar import SolarDynamics
from gll_env.rewards.base import BaseReward, RewardFn
from gll_env.rewards.leg import LegSettlementReward, Payments

GRID_ASSETS_DIR = Path(__file__).with_name("assets").joinpath("components_grid")
LEG_ASSETS_DIR = Path(__file__).with_name("assets").joinpath("rewards_leg")


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


def radial_feasibility(tolerance: float) -> RadialFeasibility:
    """Build the :class:`RadialFeasibility` shared by Prosumer and Inverter."""
    return RadialFeasibility(tolerance=tolerance)


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
        # Config key kept for backward compatibility; the field it feeds was
        # renamed to voltage_deviation_ref_pu.
        params["voltage_deviation_ref_pu"] = jnp.asarray(
            config.v_bus_deviation_pu, dtype=jnp.float32
        )
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
        charge_rating_kw=_broadcast(config.peak_charge_kW, shape),
        discharge_rating_kw=_broadcast(config.peak_discharge_kW, shape),
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
    projection: RadialFeasibility | None,
) -> InverterDynamics:
    """Build the inverter orchestrator.

    Config: ``s_inv_max_kVA`` (scalar or per-inverter list), ``battery``,
    ``solar`` (sub-configs). ``projection`` is the shared
    :class:`RadialFeasibility` built once in :func:`environment_model` from
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
    projection: RadialFeasibility | None = None,
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


def grid_code(config: DictConfig, prosumer: ProsumerDynamics) -> GridCode:
    """Build the :class:`GridCode` binding the agent's action space.

    Config (all optional; an absent ``grid_code`` block means no law applies
    and the agent keeps both degrees of freedom)::

        grid_code:
            q_of_u: true            # NE7 4.3.2 standard curve -> action_dim 1
            q_max_kvar: 6.5         # optional override of Tabelle 3's rating-based Q_max
            voltage_pu: [0.93, 0.97, 1.03, 1.07]   # optional VNB-specific curve
            q_ratio:   [1.0, 0.0, 0.0, -1.0]       # (NE7 4.3(2) allows per-plant settings)

    ``q_max_kvar`` defaults to :func:`rated_q_max_kvar` applied to each
    inverter's own ``s_inv_max_kVA``, which is what Tabelle 3 prescribes --
    so the standard case needs ``q_of_u: true`` and nothing else.
    """
    if not config.get("q_of_u", False):
        return GridCode()

    num_inv = prosumer.num_inv
    s_inv_max_kva = jnp.asarray(prosumer.inverter_dynamics.s_inv_max_kva, dtype=jnp.float32)
    q_max_kvar = (
        _broadcast(config.q_max_kvar, (num_inv,))
        if "q_max_kvar" in config
        else rated_q_max_kvar(s_inv_max_kva)
    )
    kwargs: dict[str, Any] = {}
    if "voltage_pu" in config:
        kwargs["voltage_pu"] = jnp.asarray(config.voltage_pu, dtype=jnp.float32)
    if "q_ratio" in config:
        kwargs["ratio"] = jnp.asarray(config.q_ratio, dtype=jnp.float32)
    return GridCode(q_of_u=QofUCharacteristic(q_max_kvar=q_max_kvar, **kwargs))


def payments(config: DictConfig) -> Payments:
    """Load LEG tariff rates from a safetensors asset.

    Config: ``payments`` (asset name, e.g. ``solarquartier`` or ``fair_leg``
    -- see ``assets.rewards_leg.generator``).
    """
    return Payments(**load_asset_arrays(config.payments, asset_dir=LEG_ASSETS_DIR))


def leg_settlement_reward(config: DictConfig, env_model: EnvironmentDynamics) -> RewardFn:
    """Build a :class:`LegSettlementReward`.

    Config: ``payments`` (asset name) -- see :func:`payments`.
    """
    return LegSettlementReward(payments=payments(config), prosumer=env_model.prosumer)


def base_reward(config: DictConfig, env_model: EnvironmentDynamics) -> RewardFn:
    """Build the placeholder :class:`BaseReward`, which takes no parameters."""
    del config, env_model
    return BaseReward()


# Reward name -> builder. Add an entry here to make a new reward selectable
# from config; the builder signature is uniform so callers never special-case.
REWARD_BUILDERS = {
    "base": base_reward,
    "leg_settlement": leg_settlement_reward,
}


def reward_fn(config: DictConfig, env_model: EnvironmentDynamics) -> RewardFn:
    """Build the reward named by config, against an already-built environment model.

    :class:`~gll_env.env.ProsumerGrid` takes the reward as a constructor
    argument rather than reading it off the generator, so a scenario and a
    reward can be recombined freely::

        generator = ConfigGenerator(n_steps_per_day=96, grid=grid_cfg, prosumer=prosumer_cfg)
        env = ProsumerGrid(
            generator=generator,
            reward_fn=reward_fn(reward_cfg, generator.env_dynamics),
            time_limit=96,
        )

    Config: ``name`` (a key of :data:`REWARD_BUILDERS`), plus whatever that
    reward needs -- ``leg_settlement`` additionally takes ``payments``::

        reward:
            name: leg_settlement
            payments: solarquartier
    """
    name = str(config.get("name", "base"))
    if name not in REWARD_BUILDERS:
        raise ValueError(f"Unknown reward {name!r}; expected one of {sorted(REWARD_BUILDERS)}.")
    return REWARD_BUILDERS[name](config, env_model)


def environment_model(config: DictConfig) -> EnvironmentDynamics:
    """Build a complete :class:`EnvironmentModel` from config.

    Config keys: ``n_steps_per_day``, ``grid``, ``prosumer``, and an
    optional top-level ``tolerance`` -- see the module docstring for the
    expected layout. A single :class:`DaytimeDynamics` is built once from
    ``n_steps_per_day`` and shared by every sub-component, matching the
    invariant every ``__post_init__`` in this tree already asserts (all
    share one clock). Likewise, if ``tolerance`` is given, a single
    :class:`RadialFeasibility` is built once and shared by ``ProsumerDynamics``
    and its ``InverterDynamics``; if omitted, both fall back independently
    to their own field default (``1e-4``, matching ``NewtonRaphson``'s own
    default) rather than being forced to agree on one.
    """
    time = daytime_dynamics(config.n_steps_per_day)
    grid = grid_dynamics(config.grid, time)
    projection = radial_feasibility(float(config.tolerance)) if "tolerance" in config else None
    prosumer = prosumer_dynamics(
        config.prosumer, num_pq=grid.num_pq, time=time, projection=projection
    )
    return EnvironmentDynamics(
        prosumer=prosumer,
        grid=grid,
        time=time,
        grid_code=grid_code(config.get("grid_code", {}), prosumer),
    )


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


def default_environment_model() -> EnvironmentDynamics:
    """Build the default scenario described by :data:`_DEFAULT_CONFIG`.

    No config authoring needed -- this is what
    :func:`default_environment_model` builds for standalone use when no
    generator is supplied.
    """
    return environment_model(OmegaConf.create(_DEFAULT_CONFIG))
