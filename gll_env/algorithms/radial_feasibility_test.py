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

import jax.numpy as jnp
import jax.random as jrandom

from gll_env.algorithms.radial_feasibility import (
    RadialFeasibility,
    project_ball,
    project_halfspace,
)
from gll_env.types import ActionConstraints

_TOLERANCE = 1e-6


def generate_feasible_problem(
    key: jnp.ndarray,
    num_agents: int,
    action_dim: int = 2,
    num_halfspaces: int = 3,
    num_balls: int = 2,
) -> tuple[jnp.ndarray, ActionConstraints]:
    """Build a random batch of origin-containing halfspace/ball constraints.

    Halfspace offsets `b` and ball radii are drawn strictly positive so
    that `a @ 0 <= b` and `||0 - center|| <= radius` both hold, satisfying
    the `ActionConstraints` invariant that every set contains the origin.
    """

    keys = jrandom.split(key, 5)

    x = jrandom.normal(keys[0], (num_agents, action_dim)) * 3.0

    halfspace_a = jrandom.normal(keys[1], (num_agents, num_halfspaces, action_dim))
    halfspace_b = jrandom.uniform(keys[2], (num_agents, num_halfspaces), minval=0.5, maxval=2.0)

    ball_radius = jrandom.uniform(keys[3], (num_agents, num_balls), minval=0.5, maxval=2.0)
    # Sample ball centers strictly inside their own radius so the origin is contained:
    # ||center|| < radius.
    raw_center = jrandom.normal(keys[4], (num_agents, num_balls, action_dim))
    raw_center_norm = jnp.linalg.norm(raw_center, axis=-1, keepdims=True)
    safe_norm = jnp.maximum(raw_center_norm, 1e-8)
    max_center_norm = 0.9 * ball_radius[..., None]  # stay strictly inside
    ball_center = raw_center / safe_norm * max_center_norm

    constraints = ActionConstraints(
        halfspace_a=halfspace_a,
        halfspace_b=halfspace_b,
        ball_center=ball_center,
        ball_radius=ball_radius,
    )
    return x, constraints


def test_action_constraints_is_feasible_true_and_false() -> None:
    """Direct unit test of ActionConstraints.is_feasible on known points."""
    constraints = ActionConstraints(
        halfspace_a=jnp.array([[[1.0, 0.0]]]),
        halfspace_b=jnp.array([[1.0]]),
        ball_center=jnp.array([[[0.0, 0.0]]]),
        ball_radius=jnp.array([[2.0]]),
    )

    # Inside the halfspace (x <= 1) and inside the ball (norm <= 2).
    assert bool(constraints.is_feasible(jnp.array([[0.5, 0.5]])))

    # Violates the halfspace (x = 5 > 1).
    assert not bool(constraints.is_feasible(jnp.array([[5.0, 0.0]])))

    # Violates the ball (norm = 3 > 2), even though the halfspace is fine.
    assert not bool(constraints.is_feasible(jnp.array([[0.0, 3.0]])))


def test_action_constraints_is_feasible_vacuous_with_no_constraints() -> None:
    """An agent with zero halfspaces and zero balls is trivially feasible everywhere."""
    constraints = ActionConstraints(
        halfspace_a=jnp.zeros((1, 0, 2)),
        halfspace_b=jnp.zeros((1, 0)),
        ball_center=jnp.zeros((1, 0, 2)),
        ball_radius=jnp.zeros((1, 0)),
    )

    # No constraints to violate, regardless of how far out the point is.
    assert bool(constraints.is_feasible(jnp.array([[1000.0, -1000.0]])))


def test_feasibility_map_radial_projection() -> None:
    key = jrandom.PRNGKey(0)
    batch_size = 1000
    rp = RadialFeasibility(tolerance=_TOLERANCE)

    key, subkey = jrandom.split(key)
    x, constraints = generate_feasible_problem(subkey, batch_size)

    x_proj, _ = rp.solve(x, constraints)

    assert constraints.is_feasible(x_proj, tol=rp.tolerance)


def test_project_halfspace() -> None:
    """Test projection onto a halfspace a @ x <= b."""
    # Point already satisfying the constraint is unchanged.
    x_inside = jnp.array([0.0, 0.0])
    a = jnp.array([1.0, 0.0])
    b = jnp.array(1.0)

    x_proj = project_halfspace(x_inside, a, b)
    assert jnp.allclose(x_proj, x_inside)

    # Point violating the constraint is pulled back to the boundary.
    x_outside = jnp.array([3.0, 0.0])
    x_proj = project_halfspace(x_outside, a, b)
    assert jnp.allclose(x_proj, jnp.array([1.0, 0.0]), atol=1e-6)
    assert jnp.allclose(jnp.dot(a, x_proj), b, atol=1e-6)

    # Test batch.
    x_batch = jnp.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    a_batch = jnp.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    b_batch = jnp.array([1.0, 1.0, 1.0])

    x_proj_batch = project_halfspace(x_batch, a_batch, b_batch)
    violation = jnp.sum(a_batch * x_proj_batch, axis=-1) - b_batch
    assert jnp.all(violation <= 1e-6)


def test_project_ball() -> None:
    """Test projection onto a Euclidean ball interior."""
    # Point inside the ball is unchanged.
    x_inside = jnp.array([1.0, 1.0])
    center = jnp.array([2.0, 0.0])
    radius = jnp.float32(2.0)

    x_proj = project_ball(x_inside, center, radius)
    assert jnp.allclose(x_proj, x_inside)

    # Point outside the ball is scaled toward the center.
    x_outside = jnp.array([5.0, 0.0])
    x_proj = project_ball(x_outside, center, radius)
    assert jnp.allclose(jnp.linalg.norm(x_proj - center), radius, atol=1e-6)
    direction = (x_outside - center) / jnp.linalg.norm(x_outside - center)
    assert jnp.allclose(x_proj, center + radius * direction, atol=1e-6)

    # Point at the center stays at the center.
    x_proj = project_ball(center, center, radius)
    assert jnp.allclose(x_proj, center)

    # Test batch.
    x_batch = jnp.array([[1.0, 1.0], [5.0, 0.0], [2.0, 2.0]])
    center_batch = jnp.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    radius_batch = jnp.array([2.0, 2.0, 2.0])

    x_proj_batch = project_ball(x_batch, center_batch, radius_batch)
    dist = jnp.linalg.norm(x_proj_batch - center_batch, axis=-1)
    assert jnp.all(dist <= radius_batch + 1e-6)


def test_radial_projection_default_tolerance() -> None:
    """Field default matches NewtonRaphson's own tolerance default (1e-4),
    so a config that omits both falls back to consistent numbers."""
    assert RadialFeasibility().tolerance == 1e-4


def test_radial_projection_converges_and_returns_feasible_point() -> None:
    """Test that RadialFeasibility.solve lands in the feasible region and reports converged."""
    key = jrandom.PRNGKey(42)

    batch_size = 10
    key, subkey = jrandom.split(key)
    x, constraints = generate_feasible_problem(subkey, batch_size)

    rp = RadialFeasibility(tolerance=_TOLERANCE)
    x_proj, converged = rp.solve(x, constraints)

    assert bool(converged)
    assert constraints.is_feasible(x_proj, tol=_TOLERANCE)


def test_radial_projection_single_agent() -> None:
    """Test RadialFeasibility.solve with a single agent."""
    x = jnp.array([[2.0, 3.0]])
    constraints = ActionConstraints(
        halfspace_a=jnp.array([[[1.0, 0.0], [0.0, 1.0]]]),
        halfspace_b=jnp.array([[1.5, 1.5]]),
        ball_center=jnp.array([[[0.5, 0.5]]]),
        ball_radius=jnp.array([[1.0]]),
    )

    rp = RadialFeasibility(tolerance=_TOLERANCE)
    x_proj, _ = rp.solve(x, constraints)

    assert constraints.is_feasible(x_proj, tol=_TOLERANCE)


def test_radial_projection_already_feasible_is_a_noop() -> None:
    """Test that feasible points are preserved exactly, with converged=True."""
    key = jrandom.PRNGKey(456)

    batch_size = 5
    key, subkey = jrandom.split(key)
    x_infeasible, constraints = generate_feasible_problem(subkey, batch_size)

    rp = RadialFeasibility(tolerance=_TOLERANCE)

    # Project once to get a feasible point.
    x_feasible, _ = rp.solve(x_infeasible, constraints)

    # Project again -- already feasible, so this should be an exact no-op.
    x_proj2, converged = rp.solve(x_feasible, constraints)

    assert bool(converged)
    assert jnp.allclose(x_feasible, x_proj2, atol=_TOLERANCE * 10)
    assert constraints.is_feasible(x_proj2, tol=_TOLERANCE)


def test_radial_projection_origin_is_a_noop() -> None:
    """The origin is guaranteed feasible for any valid ActionConstraints, so
    projecting it must be an exact no-op."""
    key = jrandom.PRNGKey(99)
    batch_size = 50
    key, subkey = jrandom.split(key)
    _, constraints = generate_feasible_problem(subkey, batch_size)

    origin = jnp.zeros((batch_size, 2))
    rp = RadialFeasibility(tolerance=_TOLERANCE)
    x_proj, converged = rp.solve(origin, constraints)

    assert bool(converged)
    assert jnp.allclose(x_proj, origin, atol=_TOLERANCE * 10)


def test_radial_projection_origin_always_feasible() -> None:
    """The origin must always satisfy every constraint by construction."""
    key = jrandom.PRNGKey(7)
    batch_size = 200
    key, subkey = jrandom.split(key)
    _, constraints = generate_feasible_problem(subkey, batch_size)

    origin = jnp.zeros((batch_size, 2))
    assert constraints.is_feasible(origin, tol=1e-6)


def test_radial_projection_no_halfspaces() -> None:
    """RadialFeasibility.solve should work with zero halfspace constraints (balls only)."""
    x = jnp.array([[3.0, 3.0]])
    constraints = ActionConstraints(
        halfspace_a=jnp.zeros((1, 0, 2)),
        halfspace_b=jnp.zeros((1, 0)),
        ball_center=jnp.array([[[0.0, 0.0]]]),
        ball_radius=jnp.array([[1.0]]),
    )

    rp = RadialFeasibility(tolerance=_TOLERANCE)
    x_proj, _ = rp.solve(x, constraints)

    assert constraints.is_feasible(x_proj, tol=_TOLERANCE)


def test_radial_projection_solves_to_the_true_boundary_not_an_arbitrary_interior_point() -> None:
    """Regression test for a real correctness bug in the old reflect-and-relax
    iteration this module used to run: for a request far outside a tight box
    (the common case here -- a policy asking for more than the currently
    available battery/solar headroom), it could "converge" the moment an
    over-relaxed step happened to land anywhere inside the feasible set,
    landing on an essentially arbitrary interior point instead of the true
    nearest boundary point. This pins the closed-form ray-intersection
    replacement to the correct answer: for a pure box (halfspaces only, ball
    non-binding), projecting a point far along the positive x-axis must land
    exactly on p_max.
    """
    rp = RadialFeasibility(tolerance=_TOLERANCE)
    x = jnp.array([[3.75, 0.0]])

    for p_min, p_max in [(-1.25, 1.25), (-0.157, 1.25), (-0.5, 0.5), (-1.25, 0.05)]:
        constraints = ActionConstraints(
            halfspace_a=jnp.array([[[1.0, 0.0], [-1.0, 0.0]]]),
            halfspace_b=jnp.array([[p_max, -p_min]]),
            ball_center=jnp.zeros((1, 1, 2)),
            ball_radius=jnp.array([[3.75]]),
        )
        y, converged = rp.solve(x, constraints)
        assert bool(converged)
        assert jnp.allclose(jnp.asarray(y)[0], jnp.array([p_max, 0.0]), atol=1e-4)


def test_radial_projection_no_balls() -> None:
    """RadialFeasibility.solve should work with zero ball constraints (halfspaces only)."""
    x = jnp.array([[3.0, 3.0]])
    constraints = ActionConstraints(
        halfspace_a=jnp.array([[[1.0, 0.0], [0.0, 1.0]]]),
        halfspace_b=jnp.array([[1.0, 1.0]]),
        ball_center=jnp.zeros((1, 0, 2)),
        ball_radius=jnp.zeros((1, 0)),
    )

    rp = RadialFeasibility(tolerance=_TOLERANCE)
    x_proj, _ = rp.solve(x, constraints)

    assert constraints.is_feasible(x_proj, tol=_TOLERANCE)


def test_solve_returns_the_radial_point_not_the_euclidean_nearest_point() -> None:
    """Pins which map this is, because the two are easy to confuse and the
    difference is visible to a policy.

    Against ``x_0 <= 1``, the point ``(10, 10)`` retracts to ``(1, 1)``: the
    whole vector is scaled, so the (already feasible) second coordinate is
    pulled back too. The Euclidean projection would be ``(1, 10)``, which is
    strictly closer. Radial is the intended behaviour -- it preserves the
    requested P/Q ratio -- so this asserts the scaling explicitly rather than
    leaving it as an accident someone might later "fix".
    """
    constraints = ActionConstraints(
        halfspace_a=jnp.array([[[1.0, 0.0]]], dtype=jnp.float32),
        halfspace_b=jnp.array([[1.0]], dtype=jnp.float32),
        ball_center=jnp.zeros((1, 0, 2), dtype=jnp.float32),
        ball_radius=jnp.zeros((1, 0), dtype=jnp.float32),
    )
    x = jnp.array([[10.0, 10.0]], dtype=jnp.float32)

    projected, converged = RadialFeasibility().solve(x, constraints)

    assert bool(converged)
    assert jnp.allclose(projected, jnp.array([[1.0, 1.0]]))
    # Direction preserved: the returned point is a non-negative multiple of x.
    assert jnp.allclose(projected[0, 0] * x[0, 1], projected[0, 1] * x[0, 0])
    # And it is NOT the nearest feasible point, by construction.
    assert not jnp.allclose(projected, jnp.array([[1.0, 10.0]]))


def test_origin_infeasible_constraints_fail_open_and_report_non_convergence() -> None:
    """Documents the bail-out contract, including its sharp edge.

    When the origin is outside some constraint the star-shaped assumption is
    void, so ``solve`` refuses to run and reports ``converged=False``. It
    returns ``x`` UNCHANGED -- it does not clamp, and it does not fall back
    to the origin (which is itself infeasible in this regime). Callers must
    therefore treat ``converged=False`` as "this action was never
    constrained", not merely as a diagnostic.

    In this tree that is safe in depth rather than by luck: Prosumer feeds
    the unprojected request into Inverter, whose own constraint is provably
    origin-feasible and so still bounds ``|s_inv|``; only the grid-connection
    ball goes unenforced, and ``valid=False`` propagates up to terminate the
    episode on that same step.
    """
    constraints = ActionConstraints(
        halfspace_a=jnp.array([[[1.0, 0.0]]], dtype=jnp.float32),
        halfspace_b=jnp.array([[-5.0]], dtype=jnp.float32),  # excludes the origin
        ball_center=jnp.zeros((1, 0, 2), dtype=jnp.float32),
        ball_radius=jnp.zeros((1, 0), dtype=jnp.float32),
    )
    x = jnp.array([[1000.0, 1000.0]], dtype=jnp.float32)

    projected, converged = RadialFeasibility().solve(x, constraints)

    assert not bool(converged)
    assert jnp.array_equal(projected, x)
    assert not bool(constraints.is_feasible(projected, tol=1e-3))


def test_one_malformed_agent_suppresses_projection_for_the_whole_batch() -> None:
    """``is_feasible`` reduces over the agent axis, so the origin-feasibility
    check is all-or-nothing across the batch: a single agent with malformed
    constraints sends EVERY agent down the bail-out branch, including
    well-formed ones whose own actions would otherwise have been projected.

    Pinned as current behaviour rather than asserted as desirable. It is
    unreachable through the components in this tree (every constraint built
    here provably contains the origin), but it is a live hazard for a custom
    leaf component, and the blast radius is the whole batch rather than the
    one agent at fault.
    """
    constraints = ActionConstraints(
        halfspace_a=jnp.array([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=jnp.float32),
        halfspace_b=jnp.array([[1.0], [-5.0]], dtype=jnp.float32),  # agent 1 excludes the origin
        ball_center=jnp.zeros((2, 0, 2), dtype=jnp.float32),
        ball_radius=jnp.zeros((2, 0), dtype=jnp.float32),
    )
    x = jnp.array([[10.0, 10.0], [10.0, 10.0]], dtype=jnp.float32)

    projected, converged = RadialFeasibility().solve(x, constraints)

    assert not bool(converged)
    # Agent 0 is well-formed and violates its own halfspace on the way out.
    assert float(projected[0, 0]) > 1.0
