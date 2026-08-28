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

"""Grid-code ABC and the default no-law grid code.

A grid code is what the connection rules take out of the agent's hands. It
sits in the same category as :mod:`gll_env.observer` and
:mod:`gll_env.rewards` -- rules imposed on the agent from outside the physics,
answering "what may the agent choose" where those answer "what does the agent
see" and "what does the agent get paid". None of the three is physics, which
is why none of them lives in ``components/``.

Concrete codes live in sibling modules of this package, one per jurisdiction's
ruleset (see :mod:`gll_env.grid_codes.swiss_lv`), and are selected from config
through :func:`gll_env.factories.grid_code`.

A code owns three things:

* :attr:`action_dim` -- how many degrees of freedom the agent keeps.
* :meth:`reduce` -- given the physical 2-D (p, q) constraint and the voltage
  measured last interval, the constraint over the axes the agent still
  controls, plus whatever setpoint the code imposed on the rest.
* :meth:`lift` -- the inverse: rebuild the full (p, q) request from an action
  in the reduced space and that setpoint.

`reduce` and `lift` are a pair, and the pairing is the contract: an action
feasible under `reduce`'s constraint must lift to a request feasible under the
constraint `reduce` was given. That is what lets the layers below keep working
in two dimensions, unmodified, with their own projections finding nothing to do.

Why a code, not a menu of curves
--------------------------------
A grid code is a jurisdiction's ruleset, not an a-la-carte set of
characteristics. NE7's Q(U) and P(U) share one document, one set of rating
bands and one sign convention; VDE-AR-N 4105 would bring its own. Modelling
each document as a class keeps a curve attached to the text that defines it,
and makes adding a second jurisdiction a new class rather than more optional
fields on a generic one.
"""

import abc

import chex
import jax.numpy as jnp

from gll_env.types import ActionConstraints


class GridCode(abc.ABC):
    """Abstract grid code: the rules binding the agent's action space."""

    @property
    @abc.abstractmethod
    def action_dim(self) -> int:
        """Degrees of freedom left to the agent, per agent."""

    @abc.abstractmethod
    def reduce(
        self,
        s_inv_request_constraint: ActionConstraints,
        voltage_pu: chex.Array,
        step_duration_h: chex.Numeric,
    ) -> tuple[ActionConstraints, chex.Array]:
        """Constraints over the agent's remaining freedom, and the setpoint
        imposed on the rest.

        Args:
            s_inv_request_constraint: The physical 2-D (p, q) constraint for
                the coming interval, in kWh/kvarh.
            voltage_pu: Shape (num_agents,). Voltage magnitude at each
                agent's own connection point, measured over the interval that
                just ended; ``EnvironmentDynamics._next_action_constraints`` explains
                why it is the previous interval's rather than this one's.
            step_duration_h: Interval length, for the power-to-energy
                conversion every component in this tree works in.

        Returns:
            A ``(num_agents, action_dim)`` constraint in physical units, and
            a ``(num_agents,)`` reactive setpoint in kvarh (zero where the
            code imposes none).
        """

    @abc.abstractmethod
    def lift(self, request: chex.Array, q_setpoint_kvarh: chex.Array) -> chex.Array:
        """Rebuild a full ``(num_agents, 2)`` [p, q] request.

        Args:
            request: Shape (num_agents, action_dim), already scaled to
                physical units.
            q_setpoint_kvarh: The setpoint :meth:`reduce` returned alongside
                the constraint this request was chosen within.
        """


class NoGridCode(GridCode):
    """No law: the agent sets active and reactive power independently.

    The environment's original behaviour, and the default. Not today's legal
    regime -- it is the counterfactual worth measuring one against.
    """

    @property
    def action_dim(self) -> int:
        return 2

    def reduce(
        self,
        s_inv_request_constraint: ActionConstraints,
        voltage_pu: chex.Array,
        step_duration_h: chex.Numeric,
    ) -> tuple[ActionConstraints, chex.Array]:
        del step_duration_h  # no voltage-dependent law to evaluate
        return s_inv_request_constraint, jnp.zeros_like(jnp.asarray(voltage_pu))

    def lift(self, request: chex.Array, q_setpoint_kvarh: chex.Array) -> chex.Array:
        del q_setpoint_kvarh  # the agent supplies q itself
        return request
