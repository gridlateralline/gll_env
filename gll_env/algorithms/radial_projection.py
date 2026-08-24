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

"""
Convex projection utilities.

This module maps real-valued action vectors onto a feasible region
defined as the intersection of convex sets described by an
`ActionConstraints` instance: a batch of halfspaces (`a @ x <= b`) and
balls (`||x - center|| <= radius`).

Every halfspace and ball is assumed to contain the origin (see
`ActionConstraints`), which guarantees the intersection is always
nonempty (the origin itself is always feasible) *provided that
invariant actually holds*, and -- this is the fact this module actually
exploits -- makes the feasible set star-shaped from the origin. That
means the projection of any point `x` onto the set can be computed
exactly in closed form: walk the ray `t * x` for `t` from 0 outward, and
stop at the first `t` where the ray touches the boundary of any
halfspace or ball. Each halfspace/ball contributes one such `t`
independently (a 1D root-find: linear for a halfspace, the positive
root of a quadratic for a ball -- the standard ray-sphere intersection
formula), and the binding constraint is whichever gives the smallest
`t`. `t` is clipped to `[0, 1]` so an already-feasible `x` is returned
unchanged rather than extrapolated outward.

This is exact (no iteration, no relaxation parameter, no convergence
criterion to satisfy) and correct by construction. An earlier version of
this module ran a naive multi-set generalization of Douglas-Rachford/
POCS-style reflect-and-average splitting instead; that only guarantees
converging to *some* point in the intersection (a feasibility problem),
not the nearest point to `x` (a projection problem) -- for a request far
outside a tight box (the common case here: a policy asking for more than
the currently available battery/solar headroom), an over-relaxed
reflection step could overshoot straight through the feasible region and
"converge" (zero further movement) the moment it happened to land
anywhere inside, a different point every time the box shifted slightly.

`RadialProjection.solve` short-circuits in two cases before computing
anything: if the origin is infeasible (malformed constraints; see
above), or if the input `x` is already feasible, in which case it is
returned unchanged. Both checks are evaluated as ordinary JAX values via
`jax.lax.switch`, so it remains jit- and vmap-compatible.

Implementations are vectorized, numerically stable, and JAX-compatible.
"""

from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp

from gll_env.types import ActionConstraints

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

# Numerical safety and convergence parameters
_SAFE_NORM_MIN = 1e-12  # Avoid division by zero in norm computation
_SAFE_DENOM_MIN = 1e-12  # Avoid division by zero in halfspace normal norm


def project_halfspace(x: chex.Array, a: chex.Array, b: chex.Array) -> chex.Array:
    """Project real vector(s) onto a halfspace `a @ x <= b`.

    Points already satisfying the constraint are unchanged; points that
    violate it are moved along `a` by the minimal Euclidean amount needed
    to satisfy it with equality.

    Parameters
    ----------
    x : chex.Array
        Input vector(s), shape (..., action_dim).
    a : chex.Array
        Halfspace normal(s), shape (..., action_dim), broadcastable with `x`.
    b : chex.Array
        Halfspace offset(s), shape (...), broadcastable with `x`'s leading dims.

    Returns
    -------
    chex.Array
        Projected vector(s), same shape as `x`.
    """

    violation = jnp.sum(a * x, axis=-1) - b
    a_sq_norm = jnp.maximum(jnp.sum(jnp.square(a), axis=-1), _SAFE_DENOM_MIN)
    # Only pull back when the constraint is actually violated.
    scale = jnp.maximum(0.0, violation) / a_sq_norm
    return x - scale[..., None] * a


def project_ball(x: chex.Array, center: chex.Array, radius: chex.Array) -> chex.Array:
    """Project real vector(s) onto a Euclidean ball interior.

    Projects point(s) x onto the ball of radius `radius` centered at
    `center`. Points inside the ball remain unchanged; points outside
    are scaled toward the center.

    Parameters
    ----------
    x : chex.Array
        Input vector(s), shape (..., action_dim).
    center : chex.Array
        Ball center(s), shape (..., action_dim), broadcastable with `x`.
    radius : chex.Array
        Ball radius/radii, shape (...), broadcastable with `x`'s leading dims.

    Returns
    -------
    chex.Array
        Projected vector(s) within the ball, same shape as `x`.
    """

    x_shifted: chex.Array = jnp.subtract(x, center)
    # Compute norm safely to avoid numerical issues
    safe_norm = jnp.maximum(jnp.linalg.norm(x_shifted, axis=-1), _SAFE_NORM_MIN)
    scale = jnp.minimum(1.0, radius / safe_norm)
    return center + x_shifted * scale[..., None]


def _project_all_constraints(x: chex.Array, constraints: ActionConstraints) -> chex.Array:
    """Sequentially project `x` onto every halfspace, then every ball.

    `x` has shape (num_agents, action_dim). Constraints hold
    (num_agents, num_halfspaces / num_balls, action_dim) worth of
    per-agent sets. Used both as a cheap fallback and as the final
    floating-point cleanup pass after `RadialProjection`'s closed-form scale.
    """

    num_halfspaces = constraints.halfspace_a.shape[-2]
    num_balls = constraints.ball_center.shape[-2]

    def halfspace_body(i: int, y: chex.Array) -> chex.Array:
        a_i: chex.Array = jnp.asarray(constraints.halfspace_a)[:, i, :]
        b_i: chex.Array = jnp.asarray(constraints.halfspace_b)[:, i]
        return project_halfspace(y, a_i, b_i)

    def ball_body(i: int, y: chex.Array) -> chex.Array:
        center_i: chex.Array = jnp.asarray(constraints.ball_center)[:, i, :]
        radius_i: chex.Array = jnp.asarray(constraints.ball_radius)[:, i]
        return project_ball(y, center_i, radius_i)

    # `num_halfspaces`/`num_balls` are static (Python ints from `.shape`), so
    # this branch is resolved at trace time. It must be skipped explicitly
    # when zero: `fori_loop` still traces its body once to build the loop's
    # jaxpr even for a loop that will run zero times, and indexing a
    # zero-length axis (e.g. `array_of_shape_(N,0,D)[:, 0, :]`) is invalid
    # even under abstract/trace-time evaluation.
    y = x
    if num_halfspaces > 0:
        y = jax.lax.fori_loop(0, num_halfspaces, halfspace_body, y)
    if num_balls > 0:
        y = jax.lax.fori_loop(0, num_balls, ball_body, y)
    return y


@dataclass(frozen=True)
class RadialProjection:
    tolerance: float = 1e-4

    def solve(
        self,
        x: chex.Array,
        constraints: ActionConstraints,
    ) -> tuple[chex.Array, chex.Array]:
        """Project real-valued action vectors onto the feasible action set.

        The feasible set for each agent is the intersection of the
        halfspaces (`a @ action <= b`) and balls
        (`||action - center|| <= radius`) described by `constraints`.
        Because every set is assumed to contain the origin, the
        intersection is always nonempty and star-shaped from the origin,
        so the projection is computed exactly via ray-boundary
        intersection -- see the module docstring. The implementation is
        vectorized over the leading `num_agents` axis and JAX-compatible.

        Parameters
        ----------
        x : chex.Array
            Input action vectors to be projected, shape (num_agents, action_dim).
        constraints : ActionConstraints
            Per-agent halfspace and ball constraints defining the feasible set.

        Returns
        -------
        tuple[chex.Array, chex.Array]
            Projected action vectors lying in the intersection of the
            constraint sets (shape matches `x`), and a boolean `converged`
            flag -- `False` only if the origin itself is infeasible (a
            malformed `constraints`, since the projection is otherwise exact
            by construction), `True` otherwise.
        """
        chex.assert_rank(x, 2)  # (num_agents, action_dim)
        chex.assert_rank(constraints.halfspace_a, 3)  # (num_agents, num_halfspaces, action_dim)
        chex.assert_rank(constraints.ball_center, 3)  # (num_agents, num_balls, action_dim)
        chex.assert_equal_shape_prefix([x, constraints.halfspace_a, constraints.ball_center], 1)

        def origin_infeasible_branch(_: None) -> tuple[chex.Array, chex.Array]:
            # The whole algorithm relies on every constraint containing the
            # origin (see `ActionConstraints`). If that invariant is violated
            # -- e.g. malformed constraints from upstream -- the feasible set
            # may be empty or the ray from the origin is no longer
            # meaningful. Rather than run an algorithm whose correctness
            # assumption doesn't hold, bail out immediately and report
            # non-convergence. `x` is returned unchanged since there is no
            # principled point to project onto in this regime.
            return x, jnp.asarray(False)

        def already_feasible_branch(_: None) -> tuple[chex.Array, chex.Array]:
            # `x` already satisfies every constraint, so no scaling is needed at all.
            return x, jnp.asarray(True)

        def solve_branch(_: None) -> tuple[chex.Array, chex.Array]:
            """Scale `x` down along the ray from the origin until it first
            touches the feasible set's boundary: `t* = min` over every
            halfspace/ball of the largest `t` for which `t * x` still
            satisfies that one constraint, clipped to `[0, 1]`.

            Per halfspace `a @ y <= b`: substituting `y = t * x` gives
            `t <= b / (a @ x)` when `a @ x > 0` (otherwise this halfspace
            never binds along this ray, since `a @ x <= 0` makes
            `a @ (t * x) <= 0 <= b` for every `t >= 0`, `b >= 0` following
            from the halfspace containing the origin).

            Per ball `||y - center|| <= radius`: substituting `y = t * x`
            gives a quadratic `A t^2 - 2 B t + C <= 0` with `A = x . x`,
            `B = x . center`, `C = center . center - radius^2` (`C <= 0`,
            the ball containing the origin). This is exactly the
            ray-sphere intersection from computer graphics: `t=0` is
            always a root of the boundary equation's feasible side (since
            `C <= 0`), so the *other*, non-negative root
            `t = (B + sqrt(B^2 - A*C)) / A` is where the ray exits.
            """
            num_halfspaces = constraints.halfspace_a.shape[-2]
            num_balls = constraints.ball_center.shape[-2]
            num_agents = x.shape[0]

            # Never scale up past t=1 -- an already-feasible x is handled
            # by already_feasible_branch, but individual constraints can
            # still report t >= 1 here (e.g. a slack halfspace), and this
            # keeps the overall min well-defined regardless.
            t_candidates: list[chex.Array] = [jnp.ones((num_agents,))]

            if num_halfspaces > 0:
                a = jnp.asarray(constraints.halfspace_a)  # (num_agents, num_halfspaces, action_dim)
                b = jnp.asarray(constraints.halfspace_b)  # (num_agents, num_halfspaces)
                a_dot_x = jnp.sum(
                    a * jnp.expand_dims(x, axis=1), axis=-1
                )  # (num_agents, num_halfspaces)
                t_halfspace = jnp.where(
                    a_dot_x > _SAFE_DENOM_MIN,
                    b / jnp.maximum(a_dot_x, _SAFE_DENOM_MIN),
                    jnp.inf,
                )
                t_candidates.append(jnp.min(t_halfspace, axis=-1))

            if num_balls > 0:
                center = jnp.asarray(constraints.ball_center)  # (num_agents, num_balls, action_dim)
                radius = jnp.asarray(constraints.ball_radius)  # (num_agents, num_balls)
                a_coef = jnp.sum(x * x, axis=-1, keepdims=True)  # (num_agents, 1)
                b_coef = jnp.sum(
                    jnp.expand_dims(x, axis=1) * center, axis=-1
                )  # (num_agents, num_balls)
                c_coef = jnp.sum(jnp.square(center), axis=-1) - jnp.square(radius)
                discriminant = jnp.maximum(jnp.square(b_coef) - a_coef * c_coef, 0.0)
                t_ball = jnp.where(
                    a_coef > _SAFE_DENOM_MIN,
                    (b_coef + jnp.sqrt(discriminant)) / jnp.maximum(a_coef, _SAFE_DENOM_MIN),
                    jnp.inf,
                )
                t_candidates.append(jnp.min(t_ball, axis=-1))

            t: chex.Array = jnp.clip(jnp.min(jnp.stack(t_candidates, axis=-1), axis=-1), 0.0, 1.0)
            projected: chex.Array = jnp.expand_dims(t, axis=-1) * x
            # Clean up any floating-point creep from the sqrt/division above --
            # one cheap pass, every constraint is already satisfied to within
            # float precision at this point.
            projected = _project_all_constraints(projected, constraints)
            return projected, jnp.asarray(True)

        origin = jnp.zeros_like(x)
        origin_feasible = constraints.is_feasible(origin, tol=self.tolerance)
        x_feasible = constraints.is_feasible(x, tol=self.tolerance)

        # Branch priority, selected via a runtime (traced) integer index so
        # this stays jit- and vmap-compatible -- `jax.lax.switch` dispatches
        # on a traced value directly, no Python-level branching on it:
        #   0 -> origin is not feasible: bail out, not converged.
        #   1 -> x is already feasible: bail out, converged, no-op.
        #   2 -> run the closed-form ray-intersection projection.
        branch_index = jnp.where(
            jnp.logical_not(origin_feasible),
            0,
            jnp.where(x_feasible, 1, 2),
        )

        return jax.lax.switch(
            branch_index,
            (origin_infeasible_branch, already_feasible_branch, solve_branch),
            operand=None,
        )
