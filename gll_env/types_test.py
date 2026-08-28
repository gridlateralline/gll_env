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

import chex
import jax
import jax.numpy as jnp
import pytest

from gll_env.types import ActionConstraints


def two_dimensional() -> ActionConstraints:
    """p in [-3, 5], |s| <= 4, |s - (1, 0.5)| <= 3.5. Origin inside all three."""
    return ActionConstraints(
        halfspace_a=jnp.asarray([[[1.0, 0.0], [-1.0, 0.0]]]),
        halfspace_b=jnp.asarray([[5.0, 3.0]]),
        ball_center=jnp.asarray([[[0.0, 0.0], [1.0, 0.5]]]),
        ball_radius=jnp.asarray([[4.0, 3.5]]),
    )


def one_dimensional() -> ActionConstraints:
    """p <= 5, p >= -3, |p| <= 4, |p - 2| <= 3  ->  [-1, 4]."""
    return ActionConstraints(
        halfspace_a=jnp.asarray([[[1.0], [-1.0]]]),
        halfspace_b=jnp.asarray([[5.0, 3.0]]),
        ball_center=jnp.asarray([[[0.0], [2.0]]]),
        ball_radius=jnp.asarray([[4.0, 3.0]]),
    )


# ---------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------


def test_bounds_intersects_every_constraint_exactly() -> None:
    minimum, maximum = one_dimensional().bounds()
    assert float(minimum[0]) == pytest.approx(-1.0)
    assert float(maximum[0]) == pytest.approx(4.0)


def test_bounds_ignores_halfspaces_that_do_not_bound_the_axis() -> None:
    """A zero normal bounds nothing; dividing by it must not poison the result."""
    constraints = ActionConstraints(
        halfspace_a=jnp.asarray([[[0.0], [1.0]]]),
        halfspace_b=jnp.asarray([[7.0, 2.0]]),
        ball_center=jnp.zeros((1, 0, 1)),
        ball_radius=jnp.zeros((1, 0)),
    )
    minimum, maximum = constraints.bounds()
    assert float(maximum[0]) == pytest.approx(2.0)
    assert float(minimum[0]) == -jnp.inf  # genuinely unbounded below


def test_bounds_round_trips_through_from_bounds() -> None:
    minimum, maximum = one_dimensional().bounds()
    again = ActionConstraints.from_bounds(minimum, maximum).bounds()
    assert jnp.allclose(again[0], minimum) and jnp.allclose(again[1], maximum)


def test_bounds_rejects_higher_dimensional_constraints() -> None:
    """The guard is load-bearing, not defensive.

    The same per-constraint arithmetic on a 2-D set computes the SHADOW, and
    the shadow does not decompose per constraint -- it would silently return
    a superset. See `test_shadow_does_not_decompose_per_constraint`.
    """
    with pytest.raises(AssertionError):
        two_dimensional().bounds()


def test_shadow_does_not_decompose_per_constraint() -> None:
    """Why bounds() must never grow an `axis` argument.

    Two unit balls, both containing the origin, centred at (0, 0) and
    (0, 0.9). Each one's shadow on the p axis is [-1, 1], so a per-constraint
    reduction reports [-1, 1] -- but the true shadow of their intersection is
    only [-0.893, 0.893]. Reducing per constraint overstates the range by
    12%, handing back bounds an agent could act inside and be infeasible.
    """
    constraints = ActionConstraints(
        halfspace_a=jnp.zeros((1, 0, 2)),
        halfspace_b=jnp.zeros((1, 0)),
        ball_center=jnp.asarray([[[0.0, 0.0], [0.0, 0.9]]]),
        ball_radius=jnp.asarray([[1.0, 1.0]]),
    )
    grid = jnp.linspace(-1.5, 1.5, 601)
    p, q = jnp.meshgrid(grid, grid, indexing="ij")
    points = jnp.stack([jnp.ravel(p), jnp.ravel(q)], axis=-1)
    inside = jax.vmap(lambda x: constraints.feasible_mask(x[None, :])[0])(points)
    true_shadow_max = float(jnp.max(jnp.where(inside, points[:, 0], -jnp.inf)))

    per_constraint_max = 1.0  # min over balls of (c_p + r)
    assert true_shadow_max == pytest.approx(0.893, abs=0.01)
    assert per_constraint_max > true_shadow_max + 0.1


# ---------------------------------------------------------------------------
# restrict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [-2.0, -0.5, 0.0, 0.5, 2.0])
def test_restrict_is_exact_against_direct_membership(value: float) -> None:
    """Membership in the slice must agree with membership in the original.

    This is the property that makes restrict-then-solve equivalent to solving
    the coupled problem, so it is checked against the predicate rather than
    against a reimplementation of the same algebra.
    """
    constraints = two_dimensional()
    sliced = constraints.restrict(axis=1, value=jnp.asarray([value]))
    for p in jnp.linspace(-6.0, 6.0, 61):
        in_slice = bool(sliced.feasible_mask(jnp.asarray([[p]]))[0])
        in_full = bool(constraints.feasible_mask(jnp.asarray([[p, value]]))[0])
        assert in_slice == in_full, f"disagreement at p={p}, q={value}"


def test_restrict_drops_the_pinned_axis() -> None:
    sliced = two_dimensional().restrict(axis=1, value=jnp.asarray([0.5]))
    chex.assert_shape(sliced.halfspace_a, (1, 2, 1))
    chex.assert_shape(sliced.ball_center, (1, 2, 1))


def test_restrict_at_a_nonzero_value_offsets_halfspaces() -> None:
    """`b - a_axis * value` -- a slice need not pass through the origin."""
    constraints = ActionConstraints(
        halfspace_a=jnp.asarray([[[1.0, 2.0]]]),
        halfspace_b=jnp.asarray([[6.0]]),
        ball_center=jnp.zeros((1, 0, 2)),
        ball_radius=jnp.zeros((1, 0)),
    )
    sliced = constraints.restrict(axis=1, value=jnp.asarray([2.0]))
    assert float(jnp.asarray(sliced.halfspace_b)[0, 0]) == pytest.approx(2.0)


def test_restrict_reports_an_empty_slice_as_a_zero_radius() -> None:
    """A slice that misses a ball is not an error -- origin-membership of the
    result is the caller's obligation, as it is for any hand-built set.
    """
    constraints = ActionConstraints(
        halfspace_a=jnp.zeros((1, 0, 2)),
        halfspace_b=jnp.zeros((1, 0)),
        ball_center=jnp.zeros((1, 1, 2)),
        ball_radius=jnp.asarray([[1.0]]),
    )
    sliced = constraints.restrict(axis=1, value=jnp.asarray([5.0]))
    assert float(jnp.asarray(sliced.ball_radius)[0, 0]) == pytest.approx(0.0)


def test_origin_feasible_keeps_zero_inside_at_the_exact_boundary() -> None:
    """The float guard, exercised where it actually bites.

    Restricting at exactly the value that puts the reduced origin on the
    ball's surface leaves a radicand of ||c_kept||^2 in exact arithmetic. Two
    square roots later float32 can land either side of it; the floor makes
    the outcome independent of which.
    """
    center = jnp.asarray([[[0.6, 0.8]]])
    constraints = ActionConstraints(
        halfspace_a=jnp.zeros((1, 0, 2)),
        halfspace_b=jnp.zeros((1, 0)),
        ball_center=center,
        ball_radius=jnp.asarray([[1.0]]),
    )
    # |c| == r exactly, so the origin sits on the surface: the tightest case.
    boundary = jnp.asarray([[0.8]])
    guarded = constraints.restrict(axis=1, value=boundary[:, 0], origin_feasible=True)
    assert bool(guarded.feasible_mask(jnp.zeros((1, 1)), tol=0.0)[0])
    assert float(jnp.asarray(guarded.ball_radius)[0, 0]) >= 0.6


def test_origin_feasible_is_a_no_op_well_inside_the_set() -> None:
    """It must not enlarge anything when the precondition is comfortably met."""
    constraints = two_dimensional()
    plain = constraints.restrict(axis=1, value=jnp.asarray([0.5]))
    guarded = constraints.restrict(axis=1, value=jnp.asarray([0.5]), origin_feasible=True)
    assert jnp.allclose(jnp.asarray(plain.ball_radius), jnp.asarray(guarded.ball_radius))


# ---------------------------------------------------------------------------
# scale, and how it composes with restrict
# ---------------------------------------------------------------------------


def test_scale_maps_membership_by_the_same_factor() -> None:
    constraints = two_dimensional()
    factor = jnp.asarray([2.0])
    scaled = constraints.scale(factor)
    for point in ([1.0, 0.5], [3.9, 0.0], [-2.5, 1.0], [5.0, 5.0]):
        original = jnp.asarray([point])
        assert bool(constraints.feasible_mask(original)[0]) == bool(
            scaled.feasible_mask(original / factor[:, None])[0]
        )


def test_scale_by_zero_collapses_to_the_origin() -> None:
    """A zero rating is a component that can do nothing, not a division error."""
    scaled = two_dimensional().scale(jnp.asarray([0.0]))
    assert bool(scaled.feasible_mask(jnp.zeros((1, 2)))[0])
    assert jnp.all(jnp.asarray(scaled.ball_radius) == 0.0)


def test_scale_commutes_with_restrict_when_the_value_is_scaled_too() -> None:
    """scale(restrict(C, axis, v)) == restrict(scale(C), axis, v / factor).

    Normalization is an isotropic homothety, so it commutes with slicing
    provided the slice value moves with it. This is what makes the order in
    EnvironmentDynamics a matter of which units are readable rather than a
    correctness constraint -- and it holds ONLY while the scale is isotropic.
    """
    constraints = two_dimensional()
    factor = jnp.asarray([2.5])
    value = jnp.asarray([1.5])

    scale_then_restrict = constraints.scale(factor).restrict(axis=1, value=value / factor)
    restrict_then_scale = constraints.restrict(axis=1, value=value).scale(factor)

    for name in ("halfspace_a", "halfspace_b", "ball_center", "ball_radius"):
        assert jnp.allclose(
            jnp.asarray(getattr(scale_then_restrict, name)),
            jnp.asarray(getattr(restrict_then_scale, name)),
            atol=1e-6,
        ), f"{name} differs"


def test_primitives_are_jit_compatible() -> None:
    constraints = two_dimensional()
    fn = jax.jit(lambda v: constraints.restrict(axis=1, value=v).bounds())
    minimum, maximum = fn(jnp.asarray([0.5]))
    assert float(minimum[0]) < 0.0 < float(maximum[0])


def test_restrict_then_bounds_matches_a_brute_force_scan() -> None:
    """Composition check against an independent search over the 2-D set."""
    constraints = two_dimensional()
    grid = jnp.linspace(-8.0, 8.0, 3201)
    for value in (-2.0, -0.5, 0.0, 1.0, 2.5):
        minimum, maximum = constraints.restrict(axis=1, value=jnp.asarray([value])).bounds()
        points = jnp.stack([grid, jnp.full_like(grid, value)], axis=-1)
        inside = jax.vmap(lambda x: constraints.feasible_mask(x[None, :])[0])(points)
        assert jnp.any(inside), f"no feasible p at q={value}"
        scan_max = float(jnp.max(jnp.where(inside, grid, -jnp.inf)))
        scan_min = float(jnp.min(jnp.where(inside, grid, jnp.inf)))
        assert float(maximum[0]) == pytest.approx(scan_max, abs=0.01)
        assert float(minimum[0]) == pytest.approx(scan_min, abs=0.01)


def test_feasible_mask_is_per_agent_where_is_feasible_reduces() -> None:
    constraints = ActionConstraints(
        halfspace_a=jnp.zeros((2, 0, 1)),
        halfspace_b=jnp.zeros((2, 0)),
        ball_center=jnp.zeros((2, 1, 1)),
        ball_radius=jnp.asarray([[1.0], [3.0]]),
    )
    action = jnp.asarray([[2.0], [2.0]])
    assert jnp.array_equal(constraints.feasible_mask(action), jnp.asarray([False, True]))
    assert not bool(constraints.is_feasible(action))
