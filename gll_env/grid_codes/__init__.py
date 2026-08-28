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

"""Grid codes for the ProsumerGrid environment.

:class:`~gll_env.grid_codes.base.GridCode` is the injectable ABC; each concrete
code models one jurisdiction's ruleset in its own module here, and is selected
from config through :func:`gll_env.factories.grid_code`.

A grid code decides what the agent may choose, the way an observer decides what
it sees and a reward decides what it earns -- rules around the physics rather
than physics, which is why this is not part of ``components/``.
"""

from gll_env.grid_codes.base import GridCode, NoGridCode
from gll_env.grid_codes.ne7 import (
    Ne7GridCode,
    QofUCharacteristic,
    limiting_power_factor,
    rated_q_max_kvar,
)

__all__ = [
    "GridCode",
    "Ne7GridCode",
    "NoGridCode",
    "QofUCharacteristic",
    "limiting_power_factor",
    "rated_q_max_kvar",
]
