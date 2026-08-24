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

from dataclasses import field
from functools import cached_property

import chex
import jax.numpy as jnp
import jax.random as jr

from gll_env.algorithms.radial_projection import RadialProjection
from gll_env.components.day_time import DaytimeDynamics, DaytimeState
from gll_env.components.inverter import (
    InverterDynamics,
    InverterObservation,
    InverterState,
)
from gll_env.components.load import LoadDynamics, LoadObservation, LoadState
from gll_env.types import ActionConstraints
from gll_env.utils import safe_normalize


@chex.dataclass(frozen=True)
class ProsumerState:
    s_pq_realized_kvah: chex.Array  # (num_pq,) complex64 -- past interval, realized
    s_inv_request_constraint: ActionConstraints

    inverter_state: InverterState
    load_state: LoadState
    time_state: DaytimeState

    valid: chex.Array  # (,) bool


@chex.dataclass(frozen=True)
class ProsumerObservation:
    p_pq_realized: chex.Array  # (num_pq,) float32 in [-1, 1] -- guaranteed when valid
    q_pq_realized: chex.Array  # (num_pq,) float32 in [-1, 1] -- guaranteed when valid

    inverter_observation: InverterObservation
    load_observation: LoadObservation


@chex.dataclass(frozen=True)
class ProsumerDynamics:
    s_pq_max_kva: chex.Array  # (num_pq,) float32 -- grid-connection nameplate limit
    inverter_id: chex.Array  # (num_inv,) int32

    inverter_dynamics: InverterDynamics
    load_dynamics: LoadDynamics
    time: DaytimeDynamics = field(default_factory=DaytimeDynamics)
    projection: RadialProjection = field(default_factory=RadialProjection)

    @cached_property
    def num_pq(self) -> int:
        return jnp.atleast_1d(self.s_pq_max_kva).shape[0]

    @cached_property
    def num_inv(self) -> int:
        return self.inverter_dynamics.num_inv

    @cached_property
    def s_pq_max_kvah(self) -> chex.Array:
        return self.s_pq_max_kva * self.time.step_duration_h

    def __post_init__(self) -> None:
        chex.assert_shape(self.s_pq_max_kva, (self.num_pq,))
        chex.assert_type(self.s_pq_max_kva, jnp.float32)
        chex.assert_shape(self.inverter_id, (self.num_inv,))
        chex.assert_type(self.inverter_id, jnp.int32)

        chex.assert_equal(self.num_pq, self.load_dynamics.num_load)
        chex.assert_equal(self.num_inv, self.inverter_dynamics.num_inv)

        chex.assert_equal(self.time.n_steps_per_day, self.inverter_dynamics.time.n_steps_per_day)
        chex.assert_equal(self.time.n_steps_per_day, self.load_dynamics.time.n_steps_per_day)

        # Ball 2 (grid-connection limit) must contain the origin, i.e. the
        # grid alone must be able to carry each load's own NAMEPLATE rating,
        # not just its typical demand (see _augmented_request_constraint).
        # Clamp up rather than raise: an undersized connection relative to
        # its own load is a data problem, not something that should crash a
        # simulation step. Since load_factor is itself clipped
        # (LoadDynamics._load_factor_max), s_load_max_kva is a real enforced
        # bound, not a reference -- so this makes |s_load| <= s_pq_max_kvah
        # hold unconditionally, at every timestep, for every load. Same axis
        # for both (num_pq == num_load, already asserted above), so no
        # inverter_id gather needed here.
        object.__setattr__(
            self,
            "s_pq_max_kva",
            jnp.maximum(self.s_pq_max_kva, self.load_dynamics.s_load_max_kva),
        )

    def _validate_constraint_shape(self, constraints: ActionConstraints) -> None:
        chex.assert_rank(constraints.halfspace_a, 3)  # (num_inv, num_halfspaces, action_dim)
        chex.assert_rank(constraints.halfspace_b, 2)  # (num_inv, num_halfspaces)
        chex.assert_rank(constraints.ball_center, 3)  # (num_inv, num_balls, action_dim)
        chex.assert_rank(constraints.ball_radius, 2)  # (num_inv, num_balls)
        chex.assert_equal_shape_prefix(
            [
                constraints.halfspace_a,
                constraints.halfspace_b,
                constraints.ball_center,
                constraints.ball_radius,
            ],
            1,
        )

    def _augmented_request_constraint(
        self,
        inverter_state: InverterState,
        load_state: LoadState,
    ) -> ActionConstraints:
        """Extend inverter_state.s_inv_request_constraint (2 halfspaces
        bounding p, 1 ball bounding |s_inv| <= s_inv_max_kvah) with a second
        ball: |s_inv - s_load| <= s_pq_max_kvah, the grid-connection limit
        applied to the net power actually exchanged with the grid.
        ActionConstraints already supports multiple balls per agent (the
        middle axis of ball_center/ball_radius), so this is concatenation,
        not reconstruction -- avoids duplicating the p-box/Ball 1 logic that
        InverterDynamics._new_request_constraint already owns.

        Power flow convention: s_inv is positive = generation (out of
        inverter); s_load is positive = consumption (into load). So
        s_inv - s_load is the net power that must flow through the grid
        connection -- positive means net export, negative means net import.

        Ball 2 contains the origin iff |s_load| <= s_pq_max_kvah, i.e. the
        grid alone can carry the load's full demand even with no inverter
        help. This is now a PROVEN invariant, not an assumption:
        __post_init__ clamps s_pq_max_kva up to
        load_dynamics.s_load_max_kva component-wise, and s_load_max_kvah is
        itself an enforced bound on |s_load| (load_factor is clipped -- see
        LoadDynamics._load_factor_max), so |s_load_kvah| <= s_load_max_kvah
        <= s_pq_max_kvah holds unconditionally, at every timestep, for every
        load. Combined with the halfspaces and Ball 1 already being proven
        feasible by construction (see InverterDynamics._inv_bounds_kwh),
        this constraint's origin-feasibility invariant is now proven
        end-to-end -- the origin_infeasible_branch of RadialProjection.solve()
        should be unreachable for it.
        """
        inverter_id = jnp.asarray(self.inverter_id, dtype=jnp.int32)
        s_load_kvah = load_state.s_load_kvah.take(inverter_id)
        s_pq_max_kvah = self.s_pq_max_kvah.take(inverter_id)

        grid_ball_center = jnp.stack([s_load_kvah.real, s_load_kvah.imag], axis=-1)[
            :, None, :
        ]  # (num_inv, 1, 2)
        grid_ball_radius = s_pq_max_kvah[:, None]  # (num_inv, 1)

        base = inverter_state.s_inv_request_constraint
        constraint = ActionConstraints(
            halfspace_a=base.halfspace_a,
            halfspace_b=base.halfspace_b,
            ball_center=jnp.concatenate([base.ball_center, grid_ball_center], axis=1),
            ball_radius=jnp.concatenate([base.ball_radius, grid_ball_radius], axis=1),
        )
        self._validate_constraint_shape(constraint)
        return constraint

    def _s_pq_realized_kvah(
        self,
        inverter_state: InverterState,
        load_state: LoadState,
    ) -> chex.Array:
        """Realized net grid flow for the interval that just ended:
        s_pq = s_inv - s_load (see _augmented_request_constraint's docstring
        for the sign convention). s_inv_realized_kvah lives on the
        num_inv-sized inverter axis; scatter it onto the num_pq-sized load
        axis via inverter_id, zero-filling every load with no attached
        inverter -- exact, since such a load's s_pq_realized really is just
        -s_load_realized.
        """
        inverter_id = jnp.asarray(self.inverter_id, dtype=jnp.int32)
        s_inv_extended_kvah = (
            jnp.zeros((self.num_pq,), dtype=jnp.complex64)
            .at[inverter_id]
            .set(inverter_state.s_inv_realized_kvah)
        )
        return s_inv_extended_kvah - load_state.s_load_realized_kvah

    def observation(self, state: ProsumerState) -> ProsumerObservation:
        s_pq_max_kvah = self.s_pq_max_kvah
        return ProsumerObservation(
            p_pq_realized=safe_normalize(state.s_pq_realized_kvah.real, s_pq_max_kvah),
            q_pq_realized=safe_normalize(state.s_pq_realized_kvah.imag, s_pq_max_kvah),
            inverter_observation=self.inverter_dynamics.observation(state.inverter_state),
            load_observation=self.load_dynamics.observation(state.load_state),
        )

    def reset(self, key: chex.PRNGKey, time_state: DaytimeState | None = None) -> ProsumerState:
        if time_state is None:
            key, time_key = jr.split(key)
            time_state = self.time.reset(time_key)

        # Fabricate a plausible previous-interval state, then run the real
        # step() dispatch to reach day_step -- same backstep-then-step
        # pattern as Inverter/Load/SolarDynamics.reset(). This keeps valid's
        # meaning identical to every other X_state.valid in this tree: "did
        # the transition INTO this state succeed", never a direct claim
        # about the constraints currently stored (those are only ever
        # tested on the next step()).
        inverter_key, load_key, p_key, q_key = jr.split(key, 4)
        time_state_prev = self.time.previous(time_state)
        inverter_state_prev = self.inverter_dynamics.reset(inverter_key, time_state_prev)
        load_state_prev = self.load_dynamics.reset(load_key, time_state_prev)
        request_constraint_prev = self._augmented_request_constraint(
            inverter_state_prev, load_state_prev
        )

        fictional_prev_state = ProsumerState(
            s_pq_realized_kvah=jnp.zeros((self.num_pq,), dtype=jnp.complex64),  # unused by step()
            s_inv_request_constraint=request_constraint_prev,
            inverter_state=inverter_state_prev,
            load_state=load_state_prev,
            time_state=time_state_prev,
            valid=jnp.bool_(True),  # no prior interval to inherit invalidity from
        )

        # No real request exists for this fabricated interval; sample within
        # Inverter's own box ∩ Ball 1 (ignoring Ball 2, the grid ball, for
        # simplicity here) -- step()'s own projection.solve() call pulls back
        # to full feasibility regardless, same as it would for any real
        # out-of-bounds request.
        p_inv_min_kwh_prev, p_inv_max_kwh_prev = self.inverter_dynamics.request_bounds(
            inverter_state_prev.s_inv_request_constraint
        )
        p_inv_request_kwh_prev = jr.uniform(
            key=p_key,
            shape=p_inv_min_kwh_prev.shape,
            minval=p_inv_min_kwh_prev,
            maxval=p_inv_max_kwh_prev,
        )
        s_inv_max_kvah = self.inverter_dynamics.s_inv_max_kvah
        q_inv_max_kvarh_prev = jnp.sqrt(
            jnp.maximum(s_inv_max_kvah**2 - p_inv_request_kwh_prev**2, 0.0)
        )
        q_inv_request_kvarh_prev = jr.uniform(
            key=q_key,
            shape=q_inv_max_kvarh_prev.shape,
            minval=jnp.negative(q_inv_max_kvarh_prev),
            maxval=q_inv_max_kvarh_prev,
        )
        s_inv_request_prev = jnp.stack([p_inv_request_kwh_prev, q_inv_request_kvarh_prev], axis=-1)

        return self.step(fictional_prev_state, s_inv_request_prev, next_time_state=time_state)

    def step(
        self,
        state: ProsumerState,
        s_inv_request: chex.Array,
        next_time_state: DaytimeState | None = None,
    ) -> ProsumerState:
        """Advance one interval given a physical per-inverter request.

        Parameters
        ----------
        s_inv_request : chex.Array
            Shape (num_inv, 2) float32: [p_inv_kwh, q_inv_kvarh].

        Feasibility w.r.t. the FULL augmented constraint (p-box + Ball 1 +
        Ball 2) is enforced here via this class's own projection.solve()
        call, before dispatching down to InverterDynamics.step() -- which
        then runs its OWN projection.solve() again, against its smaller,
        un-augmented constraint (p-box + Ball 1 only). That inner call is
        redundant in the common case (whatever already satisfies the
        augmented constraint necessarily satisfies its own subset too, so
        it's a no-op), but not wasted: it's the same "short-circuits when
        already feasible" reasoning applied consistently at every layer, and
        it's what makes InverterDynamics safe to call standalone, without a
        Prosumer in front of it at all.
        """
        chex.assert_shape(s_inv_request, (self.num_inv, 2))

        s_inv_request, projection_converged = self.projection.solve(
            s_inv_request, state.s_inv_request_constraint
        )

        next_time_state = (
            self.time.step(state.time_state) if next_time_state is None else next_time_state
        )
        next_inverter_state = self.inverter_dynamics.step(
            state.inverter_state,
            s_inv_request,
            next_time_state=next_time_state,
        )
        next_load_state = self.load_dynamics.step(state.load_state, next_time_state=next_time_state)
        next_s_pq_realized_kvah = self._s_pq_realized_kvah(next_inverter_state, next_load_state)
        next_request_constraint = self._augmented_request_constraint(
            next_inverter_state, next_load_state
        )

        # InverterState carries no `valid` of its own -- its constraint is
        # provably always origin-feasible by construction, so its internal
        # projection.solve() call is a redundant safety net, not a genuine
        # external assumption (see InverterDynamics.step's docstring).
        next_valid = jnp.logical_and(state.valid, projection_converged)

        return ProsumerState(
            s_pq_realized_kvah=next_s_pq_realized_kvah,
            s_inv_request_constraint=next_request_constraint,
            inverter_state=next_inverter_state,
            load_state=next_load_state,
            time_state=next_time_state,
            valid=next_valid,
        )
