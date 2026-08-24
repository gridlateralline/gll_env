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

from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from dataclasses import dataclass

else:
    from chex import dataclass


@dataclass(frozen=True)
class NewtonRaphson:
    max_iterations: int = 10
    tolerance: float = 1e-4

    def solve(
        self,
        v_bus_in: chex.Array,
        s_inj_bus_in: chex.Array,
        pq_id: chex.Array,
        pv_id: chex.Array,
        admittance: chex.Array,
    ) -> tuple[chex.Array, chex.Array, tuple[chex.Array, chex.Array]]:

        # Reduced Newton system index sets and variable mapping
        #
        # Overview table (which quantities are fixed/free and how they map):
        #
        # | Bus  | v_m     | v_a     | P        | Q         | In `x`?         | In `F(x)`?       |
        # |------|---------|---------|----------|-----------|-----------------|------------------|
        # | Slack| fixed   | fixed   | determ.  | determ.   | no              | no               |
        # | PV   | fixed   | free    | specified| computed  | angle (v_a) yes | P (yes)          |
        # | PQ   | free    | free    | specified| specified | angle & mag yes | P and Q (yes)    |
        #
        # State vector `x` (free voltages) layout:
        #   x = [ theta(pv_and_pq) ; V_mag(pq) ]
        #
        # Mismatch vector `F(x)` (missmatch of fixed power variables) layout:
        #   F(x) = [ P_mis(pv_and_pq) ; Q_mis(pq) ]
        #
        # Slack bus influences the numerical values in `F(x)` through the
        # network (it appears in `admittance @ v_bus`) but is not part of
        # the unknowns in `x` (its voltage remains fixed from `v_bus_in`).

        pvpq_id = jnp.concatenate([pv_id, pq_id])
        num_pvpq = pvpq_id.shape[0]

        # Initial split of voltages into angle (`va_init`) and magnitude
        # (`vm_init`). The reduced state vector `x` concatenates the unknown
        # angles for `pvpq_id` followed by the unknown magnitudes for `pq_id`:
        #    x = [angle(pvpq_id); magnitude(pq_id)].
        # Slack bus entries remain at their initial values (not part of `x`).
        va_init = jnp.angle(v_bus_in)
        vm_init = jnp.abs(v_bus_in)
        x_init = jnp.concatenate([va_init[pvpq_id], vm_init[pq_id]])

        def reconstruct_voltage(x: jnp.ndarray) -> jnp.ndarray:
            """Reconstruct the full complex bus voltage `v_bus` from the
            reduced state vector `x`.

            Mapping details:
            - `x[:num_pvpq]` are angles for buses in `pvpq_id` (PV then PQ).
            - `x[num_pvpq:]` are magnitudes for buses in `pq_id`.

            The function updates `va_init` and `vm_init` at those indices
            and leaves all other buses (including slack) at their initial
            values from `v_bus_in`.
            """
            x = jnp.asarray(x)
            va = va_init.at[pvpq_id].set(x[:num_pvpq])
            vm = vm_init.at[pq_id].set(x[num_pvpq:])
            return vm * jnp.exp(1j * va)

        def mismatch(x: jnp.ndarray) -> jnp.ndarray:
            """Compute the reduced mismatch vector F(x) used by Newton.

            Steps:
            1. Reconstruct full voltages `v_bus` from reduced state `x`.
            2. Compute injected complex power from voltages:
                 S_calc = V * conj(Y * V)
            3. Form power mismatch: S_mis = S_calc - S_spec (s_inj_bus_in).

            Return ordering (real-valued vector):
              [ Re(S_mis[pvpq_id])  -- P residuals for PV and PQ angles
              ; Im(S_mis[pq_id])   -- Q residuals for PQ buses only ]

            This ordering ensures that rows/columns of the Jacobian map to
            the reduced unknowns in `x`.
            """
            v_bus = reconstruct_voltage(x)
            s_calc = v_bus * jnp.conj(admittance @ v_bus)
            s_mis = s_calc - s_inj_bus_in
            return jnp.concatenate([jnp.real(s_mis[pvpq_id]), jnp.imag(s_mis[pq_id])])

        def mismatch_norm(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.linalg.norm(mismatch(x), ord=jnp.inf)

        @dataclass(frozen=True)
        class NRState:
            i: jnp.ndarray
            x: jnp.ndarray
            res: jnp.ndarray

        def converged(state: NRState) -> jnp.ndarray:
            return jnp.linalg.norm(state.res, ord=jnp.inf) <= self.tolerance

        def cond_fun(state: NRState) -> jnp.ndarray:
            bool_converged = converged(state)
            bool_iter_left = state.i < self.max_iterations

            return jnp.logical_and(jnp.logical_not(bool_converged), bool_iter_left)

        def body_fun(state: NRState) -> NRState:
            jacobian = jax.jacfwd(mismatch)(state.x)
            dx = jnp.linalg.solve(jacobian, -state.res)
            x_next = state.x + dx
            res_next = mismatch(x_next)
            return NRState(
                i=state.i + 1,
                x=x_next,
                res=res_next,
            )

        init_state = NRState(
            i=jnp.int32(0),
            x=x_init,
            res=mismatch(x_init),
        )

        # Execute the Newton-Raphson while-loop solver
        final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

        v_bus_out = reconstruct_voltage(final_state.x)
        s_inj_bus_out = v_bus_out * jnp.conj(admittance @ v_bus_out)

        return v_bus_out, s_inj_bus_out, (final_state.i, converged(final_state))
