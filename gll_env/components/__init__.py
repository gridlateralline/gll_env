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

"""Component tree for the ProsumerGrid environment.

Fixed concrete classes — one physics model per role, no swapping::

    GridDynamics      Newton-Raphson AC power-flow on a static network
    ProsumerDynamics  Couples inverter + load via DR feasibility projection
    InverterDynamics  Couples battery + solar via DR feasibility projection
    EnvironmentDynamics Couples grid + prosumer, owns the day-time clock
    BatteryDynamics   Linear battery
    SolarDynamics     OU clearness process + cosine day envelope
    LoadDynamics      OU load-factor process modulating an H0 profile
    DaytimeDynamics   Intra-day clock shared by every component above

Every module here is one physics model as a State / Observation / Dynamics
triple. Rules imposed on the agent from OUTSIDE the physics are not physics
and do not live here -- see :mod:`gll_env.grid_codes` for what the agent may
choose, :mod:`gll_env.observer` for what it sees, :mod:`gll_env.rewards` for
what it earns.
"""

from gll_env.components.battery import BatteryDynamics, BatteryState
from gll_env.components.day_time import DaytimeDynamics, DaytimeState
from gll_env.components.environment import (
    EnvironmentDynamics,
    EnvironmentObservation,
    EnvironmentState,
)
from gll_env.components.grid import GridDynamics, GridState
from gll_env.components.inverter import InverterDynamics, InverterState
from gll_env.components.load import LoadDynamics, LoadState
from gll_env.components.prosumer import ProsumerDynamics, ProsumerState
from gll_env.components.solar import SolarDynamics, SolarState
