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

from typing import TYPE_CHECKING, Optional

import chex
import jax.numpy as jnp

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass(frozen=True)
class ActionConstraints:
    """Fixed-shape continuous action constraints for a batch of observations.

    Geometric assumption:
    - Every halfspace and every ball must contain the origin!
    - The feasible action set is the intersection of all such constraints.
    - This is what makes a radial map from the origin well-defined.

    A @ action <= b defines a halfspace, and ||action - center||^2 <= radius^2 defines a ball.
    """

    halfspace_a: chex.Array  # (num_agents, num_halfspaces, action_dim)
    halfspace_b: chex.Array  # (num_agents, num_halfspaces)
    ball_center: chex.Array  # (num_agents, num_balls, action_dim)
    ball_radius: chex.Array  # (num_agents, num_balls)

    def is_feasible(self, action: chex.Array, tol: float = 1e-6) -> chex.Array:
        """Check whether `action` satisfies every halfspace and ball constraint.

        Parameters
        ----------
        action : chex.Array
            Action vector(s) to check, shape (num_agents, action_dim).
        tol : float
            Slack allowed when checking each constraint, to absorb floating
            point error from upstream projections.

        Returns
        -------
        chex.Array
            Scalar boolean, `True` iff every agent's action satisfies every
            halfspace and ball constraint. An agent with zero halfspaces
            and/or zero balls is trivially feasible with respect to that
            (empty) family of constraints.
        """

        halfspace_ok = jnp.all(
            jnp.sum(self.halfspace_a * jnp.array(action)[:, None, :], axis=-1)
            <= self.halfspace_b + tol
        )
        ball_ok = jnp.all(
            jnp.linalg.norm(jnp.array(action)[:, None, :] - self.ball_center, axis=-1)
            <= self.ball_radius + tol
        )
        return jnp.logical_and(halfspace_ok, ball_ok)


@dataclass(frozen=True)
class ObservationMarl:
    """The observation that the agent sees.

    agents_view: the agent's view of the environment.
    action_mask: boolean array specifying, for each agent, which action is legal.
    action_constraints: optional action constraints for each agent.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agents_view: chex.Array  # (num_agents, num_obs_features)
    action_mask: chex.Array  # (num_agents, num_actions)
    action_constraints: Optional[ActionConstraints] = None
    step_count: Optional[chex.Array] = None  # (num_agents, )


@dataclass(frozen=True)
class ObservationMarlGlobalState:
    """The observation seen by agents in centralised systems.

    Extends `ObservationMarl` by adding a `global_state` attribute for centralised training.
    global_state: The global state of the environment, often a concatenation of agents' views.
    """

    agents_view: chex.Array  # (num_agents, num_obs_features)
    action_mask: chex.Array  # (num_agents, num_actions)
    global_state: chex.Array  # (num_agents, num_state_features)
    action_constraints: Optional[ActionConstraints] = None
    step_count: Optional[chex.Array] = None  # (num_agents, )
