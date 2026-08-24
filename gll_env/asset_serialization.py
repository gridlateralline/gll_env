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

"""Safetensors serialization utilities for gll_env assets.

This module provides utilities for:
1. Saving/loading JAX arrays to/from safetensors (for grid topology)
2. Loading configuration for environment instantiation

Grid asset generation is in grid_asset_generator.py to keep pandapower optional.
"""

from importlib import resources
from os import PathLike
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import safetensors.numpy as stn


def _resolve_asset_file(name: str, asset_dir: str | PathLike[str]) -> Any:
    """Resolve an asset from a package resource or a filesystem directory."""
    filename = name + ".safetensors"
    if isinstance(asset_dir, PathLike):
        return Path(asset_dir).joinpath(filename)
    try:
        return resources.files(asset_dir).joinpath(filename)
    except (ModuleNotFoundError, TypeError):
        return Path(asset_dir).joinpath(filename)


def save_asset_arrays(
    name: str,
    asset_dir: str | PathLike[str],
    **items: Any,
) -> None:
    """Save multiple arrays to a single safetensors file.

    Parameters
    ----------
    asset_name :  str
        Name of the asset (e.g., "case9") to create
    asset_dir : str
    Name of the asset directory (e.g., "gll_env.grid_assets") to save to
    **items
        Named items to save, e.g., admittance=Y, slack_id=slack

    Examples
    --------
    >>> save_asset_arrays(
    ...     name="case9",
    ...     asset_dir="grid_physical.assets",
    ...     admittance=Ybus,
    ...     slack_id=slack_idx,
    ...     pq_id=pq_indices,
    ... )
    """
    # Convert items to JAX arrays, splitting complex arrays into real/imag parts.
    arrays_jnp = {}
    for key, val in items.items():
        array = jnp.asarray(val)
        if jnp.iscomplexobj(array):
            arrays_jnp[f"{key}__real"] = jnp.real(array)
            arrays_jnp[f"{key}__imag"] = jnp.imag(array)
        else:
            arrays_jnp[key] = array

    # Convert JAX arrays to numpy arrays for safetensors compatibility
    arrays_np = {k: np.asarray(v) for k, v in arrays_jnp.items()}

    # Compute output path using importlib.resources
    output_file = _resolve_asset_file(name, asset_dir)

    # Save arrays to safetensors file
    with resources.as_file(output_file) as output_path:
        stn.save_file(arrays_np, output_path)
    print(f"Saved arrays to {output_file}")


def load_asset_arrays(
    name: str,
    asset_dir: str | PathLike[str],
) -> dict[str, jnp.ndarray]:
    """Load arrays from safetensors file.

    Parameters
    ----------
    name : str
        Name of the asset (e.g., "case9") to load.
    asset_dir : str
    Name of the asset directory (e.g., "gll_env.grid_assets") to load from

    Returns
    -------
    Dict[str, jnp.ndarray]
        Loaded items as JAX arrays.

    Examples
    --------
    >>> arrays = load_asset_arrays("case9", asset_dir="gll_env.grid_assets")
    >>> Ybus = arrays["admittance"]
    >>> slack = arrays["slack_id"]
    """
    # Compute output path using importlib.resources
    input_file = _resolve_asset_file(name, asset_dir)

    # Check if the file exists before loading
    if not input_file.is_file():
        raise FileNotFoundError(f"Asset {input_file}.safetensors not found in {asset_dir}")

    # Load arrays using safetensors
    with resources.as_file(input_file) as input_path:
        arrays_np = stn.load_file(input_path)

    # Convert items to JAX arrays and rehydrate complex arrays if present.
    arrays_jax = {key: jnp.asarray(val) for key, val in arrays_np.items()}
    rehydrated: dict[str, jnp.ndarray] = {}
    consumed: set[str] = set()

    for key, val in arrays_jax.items():
        if key.endswith("__real"):
            base = key[: -len("__real")]
            imag_key = f"{base}__imag"
            if imag_key in arrays_jax:
                rehydrated[base] = val + 1j * arrays_jax[imag_key]
                consumed.update({key, imag_key})
        elif key.endswith("__imag"):
            continue

    for key, val in arrays_jax.items():
        if key not in consumed and key not in rehydrated:
            rehydrated[key] = val

    return rehydrated
