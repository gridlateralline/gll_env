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

from gll_env.utils import safe_normalize

# A halfspace whose normal has no component along the axis being solved for
# does not bound that axis at all. Dividing by that zero would give inf/nan
# and poison the min/max reductions, so a 1.0 denominator is substituted;
# every such entry is masked out by the sign guards at the use site anyway.
_ZERO_NORMAL = 1e-12

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

        return jnp.all(self.feasible_mask(action, tol=tol))

    def feasible_mask(self, action: chex.Array, tol: float = 1e-6) -> chex.Array:
        """Per-agent version of :meth:`is_feasible`.

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
            Boolean array of shape (num_agents,), `True` where that agent's
            action satisfies every halfspace and ball constraint.

        `is_feasible` reduces this to a single scalar over the whole batch,
        which is the right answer for "did the projection succeed" but not
        for anything that needs to act on one agent at a time -- a per-agent
        search or clamp, or a test asserting that exactly one agent is
        saturated.
        """

        halfspace_ok = jnp.all(
            jnp.sum(self.halfspace_a * jnp.array(action)[:, None, :], axis=-1)
            <= self.halfspace_b + tol,
            axis=-1,
        )
        ball_ok = jnp.all(
            jnp.linalg.norm(jnp.array(action)[:, None, :] - self.ball_center, axis=-1)
            <= self.ball_radius + tol,
            axis=-1,
        )
        return jnp.logical_and(halfspace_ok, ball_ok)

    def scale(self, factor: chex.Array) -> "ActionConstraints":
        """Divide the action space by a positive per-agent `factor`.

        Parameters
        ----------
        factor : chex.Array
            Shape (num_agents,). A zero factor collapses that agent's set to
            the origin -- see :func:`~gll_env.utils.safe_normalize` -- which
            is the right reading for a component with a zero rating.

        The map is the homothety ``x -> x / factor``, so halfspace normals
        pass through untouched (``a . (x/f) <= b/f`` describes the same set)
        and only the offsets, centres and radii move. Being ISOTROPIC -- one
        scalar for every axis -- is what keeps balls as balls; a per-axis
        factor would turn them into ellipses, which this representation
        cannot express. It also preserves the origin-membership invariant,
        since scaling by a positive scalar fixes the origin.
        """
        factor = jnp.asarray(factor)
        return ActionConstraints(
            halfspace_a=self.halfspace_a,
            halfspace_b=safe_normalize(self.halfspace_b, jnp.expand_dims(factor, axis=1)),
            ball_center=safe_normalize(self.ball_center, jnp.expand_dims(factor, axis=(1, 2))),
            ball_radius=safe_normalize(self.ball_radius, jnp.expand_dims(factor, axis=1)),
        )

    def restrict(
        self,
        axis: int,
        value: chex.Array,
        origin_feasible: bool = False,
    ) -> "ActionConstraints":
        """Restrict to the affine slice ``action[axis] == value``, returning
        constraints over the remaining axes.

        Parameters
        ----------
        axis : int
            The action-space axis to pin. Static -- it changes the result's
            shape.
        value : chex.Array
            Shape (num_agents,). The value to pin that axis to; need not be
            zero, and need not be the same for every agent.
        origin_feasible : bool
            Assert that the caller has ALREADY established that the point
            with `value` on `axis` and zeros elsewhere is feasible. See below.

        Both families are closed under this operation, so it is exact rather
        than an approximation of one:

            halfspace  a . x <= b       ->  a_kept . x_kept <= b - a_axis * value
            ball  ||x - c|| <= r        ->  ||x_kept - c_kept|| <= sqrt(r^2 - (value - c_axis)^2)

        A ball whose slice misses it entirely yields a negative radicand,
        clamped to zero. That is a genuinely empty (or single-point) slice,
        and this method does not treat it as an error: origin-membership of
        the RESULT is the caller's obligation, exactly as it already is for
        any hand-built `ActionConstraints`.

        `origin_feasible` and why it exists
        -----------------------------------
        Restricting at an admissible `value` guarantees, algebraically, that
        each resulting radius is at least ``||c_kept||`` -- which is exactly
        what keeps the reduced origin inside. In float32 it does not: a
        `value` sitting on the boundary makes the radicand
        ``||c_kept||^2 +- 1e-7``, and the minus case excludes the origin and
        collapses the set, from two algebraically equivalent expressions
        disagreeing at the boundary.

        Setting `origin_feasible` floors each radicand at ``||c_kept||^2``.
        Under the stated precondition that is a no-op in exact arithmetic,
        and it makes the invariant hold SYNTACTICALLY rather than depending
        on how the rounding falls. Do NOT set it without the precondition:
        it enlarges each ball, so on an inadmissible `value` it would admit
        infeasible points instead of reporting an empty slice.

        Halfspaces need no equivalent guard -- under the same precondition
        ``b - a_axis * value >= 0``, so their reduced offsets already carry
        the sign that keeps the origin inside.
        """
        halfspace_a = jnp.asarray(self.halfspace_a)
        ball_center = jnp.asarray(self.ball_center)
        action_dim = halfspace_a.shape[-1]
        kept = [dim for dim in range(action_dim) if dim != axis]
        value = jnp.asarray(value)[:, None]  # (num_agents, 1)

        center_kept = ball_center[:, :, kept]
        gap = value - ball_center[:, :, axis]
        radicand = jnp.square(jnp.asarray(self.ball_radius)) - jnp.square(gap)
        if origin_feasible:
            radicand = jnp.maximum(radicand, jnp.sum(jnp.square(center_kept), axis=-1))

        return ActionConstraints(
            halfspace_a=halfspace_a[:, :, kept],
            halfspace_b=jnp.asarray(self.halfspace_b) - halfspace_a[:, :, axis] * value,
            ball_center=center_kept,
            ball_radius=jnp.sqrt(jnp.maximum(radicand, 0.0)),
        )

    def bounds(self) -> tuple[chex.Array, chex.Array]:
        """Collapse a ONE-dimensional constraint into ``(min, max)`` per agent.

        Returns
        -------
        tuple[chex.Array, chex.Array]
            Two (num_agents,) arrays. An agent bounded on neither side gets
            -inf / +inf, which only happens for a constraint that does not
            bound its single axis at all.

        Only valid at ``action_dim == 1``, and the restriction is essential
        rather than defensive. In one dimension every halfspace is a
        half-line and every ball is an interval, so the feasible set IS the
        intersection of per-constraint intervals and taking max-of-lowers /
        min-of-uppers is exact.

        The same arithmetic on a higher-dimensional set -- "the range of
        ``x[axis]`` over the feasible set" -- computes something different
        and gets it wrong. That is the SHADOW of the set, and it does not
        decompose per constraint: for two unit balls centred at (0, 0) and
        (0, 0.9), each shadow is [-1, 1] and so is their intersection, while
        the true shadow of the lens is [-0.893, 0.893]. Per-constraint
        reduction returns a SUPERSET there -- bounds an agent could act
        inside and be infeasible. Reduce to one dimension with
        :meth:`restrict` first; do not generalize this method to take an axis.
        """
        halfspace_a = jnp.asarray(self.halfspace_a)
        chex.assert_axis_dimension(halfspace_a, 2, 1)

        normal = halfspace_a[:, :, 0]  # (num_agents, num_halfspaces)
        is_zero = jnp.abs(normal) < _ZERO_NORMAL
        edge = jnp.asarray(self.halfspace_b) / jnp.where(is_zero, 1.0, normal)
        center = jnp.asarray(self.ball_center)[:, :, 0]  # (num_agents, num_balls)
        radius = jnp.asarray(self.ball_radius)

        maximum = jnp.minimum(
            jnp.min(center + radius, axis=1, initial=jnp.inf),
            jnp.min(jnp.where(normal > 0.0, edge, jnp.inf), axis=1, initial=jnp.inf),
        )
        minimum = jnp.maximum(
            jnp.max(center - radius, axis=1, initial=-jnp.inf),
            jnp.max(jnp.where(normal < 0.0, edge, -jnp.inf), axis=1, initial=-jnp.inf),
        )
        return minimum, maximum

    @classmethod
    def from_bounds(cls, minimum: chex.Array, maximum: chex.Array) -> "ActionConstraints":
        """Build a one-dimensional box from per-agent ``(min, max)`` arrays.

        The inverse of :meth:`bounds`, in the same encoding -- halfspace 0
        upper, halfspace 1 lower, no balls -- that Battery and Solar already
        use for their own 1-D actions, so ``request_bounds`` reads it
        unchanged.
        """
        num_agents = jnp.asarray(maximum).shape[0]
        ones = jnp.ones((num_agents, 1), dtype=jnp.float32)
        return cls(
            halfspace_a=jnp.stack([ones, jnp.negative(ones)], axis=1),  # (num_agents, 2, 1)
            halfspace_b=jnp.stack([maximum, jnp.negative(minimum)], axis=1),
            ball_center=jnp.zeros((num_agents, 0, 1), dtype=jnp.float32),
            ball_radius=jnp.zeros((num_agents, 0), dtype=jnp.float32),
        )


@dataclass(frozen=True)
class MarlObservation:
    """The observation seen by agents in centralised systems."""

    agents_view: chex.Array  # (num_agents, num_obs_features)
    action_mask: chex.Array  # (num_agents, num_actions)
    global_state: chex.Array  # (num_agents, num_state_features)
    action_constraints: Optional[ActionConstraints] = None  # (num_agents, ...)
    step_count: Optional[chex.Array] = None  # (num_agents, )
