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

"""Generate bundled grid assets from pandapower networks.

This module handles converting pandapower networks to gll_env grid assets.
It's decoupled from core gll_env to avoid making pandapower a hard dependency.

Functions:
    - pandapower_network_to_grid(): Convert a pandapower network to GridDynamics.
    - generate_and_save_case9_grid_asset(): Generate and save IEEE Case 9.
    - generate_and_save_cigre_lv_consumer_grid_asset(): Generate and save CIGRE LV.

Pandapower is optional - only needed to generate new assets.
Pre-generated assets are stored in gll_env/grid_assets for regular use.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np

from gll_env.algorithms.newton_raphson import NewtonRaphson
from gll_env.asset_serialization import save_asset_arrays
from gll_env.components.grid import GridDynamics

try:
    import pandapower as pp  # type: ignore[import-untyped]
    import pandapower.networks as pn  # type: ignore[import-untyped]
    import pandapower.toolbox as ppt  # type: ignore[import-untyped]

    HAS_PANDAPOWER = True
except ImportError:
    pp = None  # type: ignore[import-untyped]
    pn = None  # type: ignore[import-untyped]
    ppt = None  # type: ignore[import-untyped]

    HAS_PANDAPOWER = False

# Package path to the bundled safetensors grid assets.
GRID_ASSETS_DIR = Path(__file__).with_name("grid_assets")


def _validate_pandapower_internals(net: Any) -> None:
    """Ensure the internal pandapower/MATPOWER structures match our assumptions."""
    current_version = getattr(pp, "__version__", "unknown")
    error_msg = (
        f"Pandapower internal structure mismatch. This converter assumes specific "
        f"MATPOWER internal keys which failed for v{current_version}."
    )

    try:
        internal = net._ppc["internal"]
        _ = internal["Ybus"]
        _ = internal["ref"]
        _ = internal["pv"]
        _ = internal["pq"]

        # Base KV index is assumed to be 9 in the bus table
        _ = internal["bus"][:, 9]

        # Verify geodata lookups if any spatial format is present
        has_bgd = (
            hasattr(net, "bus_geodata")
            and net.bus_geodata is not None
            and not net.bus_geodata.empty
        )
        has_geo = hasattr(net, "bus") and "geo" in net.bus.columns
        if has_bgd or has_geo:
            _ = net._pd2ppc_lookups["bus"]

    except (KeyError, AttributeError, IndexError, TypeError) as e:
        raise RuntimeError(f"{error_msg} Caused by: {e!r}") from e


def _extract_admittance_and_topology(
    net: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Isolates the extraction of Ybus and bus indices from pandapower internals."""
    internal = net._ppc["internal"]

    admittance = jnp.array(internal["Ybus"].todense(), dtype=jnp.complex64)
    slack_id = jnp.array(internal["ref"], dtype=jnp.int32)
    pv_id = jnp.array(internal["pv"], dtype=jnp.int32)
    pq_id = jnp.array(internal["pq"], dtype=jnp.int32)

    return admittance, slack_id, pv_id, pq_id


def _extract_base_values(net: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Isolates the extraction of base values."""
    # BASE_KV index in ppc bus table is 9
    base_v_kv_np = net._ppc["internal"]["bus"][:, 9]

    base_s_mva = jnp.array(net.sn_mva, dtype=jnp.float32)
    base_v_kv = jnp.array(base_v_kv_np, dtype=jnp.float32)

    return base_s_mva, base_v_kv


def _extract_positions(net: Any, n_internal_buses: int) -> jnp.ndarray:
    """Safely extracts geodata from modern or legacy formats, falling back to zeros."""
    coords_by_bus: Dict[int, List[float]] = {}

    if hasattr(net, "bus") and "geo" in net.bus.columns:
        # Use itertuples for faster iteration over pandas DataFrame
        for row in net.bus[["geo"]].itertuples():
            geo = row.geo
            if isinstance(geo, str):
                try:
                    geo = json.loads(geo)
                except (ValueError, TypeError):
                    continue
            if isinstance(geo, dict) and "coordinates" in geo:
                coords = geo["coordinates"]
                if isinstance(coords, list) and len(coords) >= 2:
                    coords_by_bus[int(row.Index)] = [float(coords[0]), float(coords[1])]

    # Map the parsed positions to the MATPOWER internal bus ordering
    if coords_by_bus and hasattr(net, "_pd2ppc_lookups") and "bus" in net._pd2ppc_lookups:
        bus_lookup = net._pd2ppc_lookups["bus"]

        coords_by_ppc = np.zeros((n_internal_buses, 2), dtype=np.float32)
        counts_by_ppc = np.zeros((n_internal_buses,), dtype=np.float32)

        for bus_idx, coords in coords_by_bus.items():
            if bus_idx in bus_lookup:
                ppc_idx = int(bus_lookup[bus_idx])
                if 0 <= ppc_idx < n_internal_buses:
                    coords_by_ppc[ppc_idx] += np.array(coords, dtype=np.float32)
                    counts_by_ppc[ppc_idx] += 1.0

        valid = counts_by_ppc > 0
        coords_by_ppc[valid] = coords_by_ppc[valid] / counts_by_ppc[valid][:, None]
        return jnp.array(coords_by_ppc, dtype=jnp.float32)

    # Fallback if no geodata is resolvable
    return jnp.zeros((n_internal_buses, 2), dtype=jnp.float32)


def _strip_pq_elements(net: Any) -> None:
    """Remove all PQ injections to create a bare structural asset.

    Preserves ext_grid (slack) and gen (PV) elements to maintain voltage
    regulation physics, while stripping everything that the JAX environment
    will dynamically overwrite at each step.
    """
    net.load = net.load.iloc[0:0]
    net.sgen = net.sgen.iloc[0:0]
    net.storage = net.storage.iloc[0:0]
    net.shunt = net.shunt.iloc[0:0]
    net.ward = net.ward.iloc[0:0]
    net.xward = net.xward.iloc[0:0]


def pandapower_network_to_grid(net: Any) -> "GridDynamics":
    """Convert a pandapower network to GridModel.

    This function runs power flow and extracts the grid topology
    and state from pandapower's internal structures.

    Args:
        net: A pandapower network (with bus, line, gen, load definitions).

    Returns:
        Grid parameters with all arrays instantiated from pandapower.

    Raises:
        ImportError: If pandapower is not installed.
        RuntimeError: If pandapower internal structures mismatch expectations.
    """
    if not HAS_PANDAPOWER:
        raise ImportError(
            "pandapower is required for asset generation. Install with: pip install pandapower"
        )

    # 1. Run power flow to populate net._ppc internal structures
    pp.runpp(net, algorithm="nr", numba=False)

    # 2. Validate internal assumptions fail-fast
    _validate_pandapower_internals(net)

    # 3. Extract core matrices and values using adapters
    admittance, slack_id, pv_id, pq_id = _extract_admittance_and_topology(net)
    base_s_mva, base_v_kv = _extract_base_values(net)
    position = _extract_positions(net, n_internal_buses=base_v_kv.shape[0])

    # 4. Instantiate model -- simulation time resolution (`time`) is not a
    # network-topology parameter, so it's left at GridDynamics' own default
    # here; to_asset_dict() (used to persist this model) excludes it too.
    return GridDynamics(
        slack_id=slack_id,
        pq_id=pq_id,
        pv_id=pv_id,
        base_s_mva=base_s_mva,
        base_v_kv=base_v_kv,
        admittance=admittance,
        position=position,
        nr=NewtonRaphson(),  # Use default NR parameters
    )


def generate_and_save_case9_grid_asset() -> None:
    """Generate IEEE Case 9 grid and save as safetensors assets."""
    if not HAS_PANDAPOWER:
        raise ImportError(
            "pandapower is required for asset generation. Install with: pip install pandapower"
        )

    print("Generating IEEE Case 9 network...")
    net = pn.case9()

    print("Stripping PQ elements for structural-only asset...")
    _strip_pq_elements(net)

    print("Converting pandapower to grid parameters...")
    grid = pandapower_network_to_grid(net)

    save_asset_arrays("case9", asset_dir=GRID_ASSETS_DIR, **grid.to_asset_dict())

    print(
        f"✓ Generated Case 9 grid: {grid.num_bus} buses "
        f"({grid.num_pq} PQ + {grid.num_pv} PV + 1 slack)"
    )


def _build_cigre_lv_consumer_grid() -> Any:
    """Build structural-only 'amputated' CIGRE LV consumer topology.

    Truncates the network to only keep the residential feeder (Bus R0 -> Bus R18).
    All loads, generators, and passive shunts (ward/xward) are removed so the
    saved asset represents the bare admittance topology.
    """
    if not HAS_PANDAPOWER:
        raise ImportError(
            "pandapower is required for asset generation. Install with: pip install pandapower"
        )

    net = pn.create_cigre_network_lv()

    # 1) Identify new root and whitelist
    r0_idx = net.bus[net.bus["name"] == "Bus R0"].index[0]
    r_buses = net.bus[net.bus["name"].str.startswith("Bus R", na=False)].index.tolist()

    # 2) Amputate upstream elements
    buses_to_drop = [b for b in net.bus.index if b not in r_buses]
    ppt.drop_buses(net, buses_to_drop, drop_elements=True)

    # 3) Drop extra transformers (keep only the R0 -> R1 Trafo)
    trafos_to_kill = net.trafo[net.trafo["hv_bus"] != r0_idx].index.tolist()
    if trafos_to_kill:
        ppt.drop_trafos(net, trafos_to_kill)

    # 4) Plant the new source at Bus R0
    pp.create_ext_grid(net, bus=r0_idx, vm_pu=1.0, name="Feeder_Root_Source")

    # 5) Clear switches to lock topology
    net.switch = net.switch.iloc[0:0]

    # 6) Structural-only asset: remove all non-topology injections and shunts.
    _strip_pq_elements(net)

    return net


def generate_and_save_cigre_lv_consumer_grid_asset() -> None:
    """Generate reduced CIGRE LV consumer topology and save as safetensors."""
    if not HAS_PANDAPOWER:
        raise ImportError(
            "pandapower is required for asset generation. Install with: pip install pandapower"
        )

    print("Generating amputated CIGRE LV consumer topology...")
    net = _build_cigre_lv_consumer_grid()

    print("Converting pandapower to grid parameters...")
    grid = pandapower_network_to_grid(net)

    save_asset_arrays("cigre_lv_consumer", asset_dir=GRID_ASSETS_DIR, **grid.to_asset_dict())

    print(
        f"✓ Generated CIGRE LV consumer grid: {grid.num_bus} buses "
        f"({grid.num_pq} PQ + {grid.num_pv} PV + 1 slack)"
    )


if __name__ == "__main__":
    generate_and_save_case9_grid_asset()
    generate_and_save_cigre_lv_consumer_grid_asset()
