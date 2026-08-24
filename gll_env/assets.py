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

"""Configuration utilities for hg_env.

This module provides utilities for:
1. Saving/loading JAX arrays to/from safetensors (for grid topology)
2. Loading configuration for environment instantiation

Asset generation is in assets_generator.py to keep pandapower optional.
"""

from importlib import resources
from typing import Any

import jax.numpy as jnp
import numpy as np
import safetensors.numpy as stn


def save_arrays(
    name: str,
    asset_dir: str,
    **items: Any,
) -> None:
    """Save multiple arrays to a single safetensors file.

    Parameters
    ----------
    asset_name :  str
        Name of the asset (e.g., "case9") to create
    asset_dir : str
        Name of the asset directory (e.g., "grid_physical.assets") to save to
    **items
        Named items to save, e.g., admittance=Y, slack_id=slack

    Examples
    --------
    >>> save_arrays(
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
    filename = name + ".safetensors"
    output_file = resources.files(asset_dir).joinpath(filename)

    # Save arrays to safetensors file
    with resources.as_file(output_file) as output_path:
        stn.save_file(arrays_np, output_path)
    print(f"Saved arrays to {output_file}")


def load_arrays(
    name: str,
    asset_dir: str,
) -> dict[str, jnp.ndarray]:
    """Load arrays from safetensors file.

    Parameters
    ----------
    name : str
        Name of the asset (e.g., "case9") to load.
    asset_dir : str
        Name of the asset directory (e.g., "grid_physical.assets") to load from

    Returns
    -------
    Dict[str, jnp.ndarray]
        Loaded items as JAX arrays.

    Examples
    --------
    >>> arrays = load_arrays("case9", asset_dir="grid_physical.assets")
    >>> Ybus = arrays["admittance"]
    >>> slack = arrays["slack_id"]
    """
    # Compute output path using importlib.resources
    filename = name + ".safetensors"
    input_file = resources.files(asset_dir).joinpath(filename)

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
