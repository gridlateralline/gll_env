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

from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any, Dict

import jax.numpy as jnp
from jumanji import specs


def safe_normalize(value: Any, scale: Any) -> jnp.ndarray:
    """Normalize by a non-negative scale, returning zero for zero scales."""
    value_array = jnp.asarray(value)
    scale_array = jnp.asarray(scale)
    return jnp.where(scale_array > 0.0, value_array / scale_array, jnp.zeros_like(value_array))


def _is_namedtuple_instance(value: Any) -> bool:
    return isinstance(value, tuple) and hasattr(value, "_fields")


def _array_spec(value: Any, name: str) -> specs.Array:
    array_value = jnp.asarray(value)
    return specs.Array(array_value.shape, array_value.dtype, name)


def spec_from_example(example: Any, name: str | None = None) -> specs.Spec:
    if is_dataclass(example):
        fields_spec: Dict[str, specs.Spec] = {}
        for field in dataclass_fields(example):
            fields_spec[field.name] = spec_from_example(
                getattr(example, field.name),
                name=field.name,
            )
        return specs.Spec(type(example), name or type(example).__name__, **fields_spec)

    if _is_namedtuple_instance(example):
        fields_spec = {
            field_name: spec_from_example(getattr(example, field_name), name=field_name)
            for field_name in example._fields
        }
        return specs.Spec(type(example), name or type(example).__name__, **fields_spec)

    return _array_spec(example, name or "value")
