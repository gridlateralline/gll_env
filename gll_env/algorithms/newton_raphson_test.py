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

from importlib import import_module
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from gll_env.algorithms.newton_raphson import NewtonRaphson


def build_test_pd(net: Any) -> dict[str, jnp.ndarray]:
    """Build a Newton-Raphson test input dict from a pandapower network.

    Returns only the `nr_input` mapping used by the solver. Tests no longer
    compare to pandapower outputs so the reference solution is not returned.
    """
    pp = import_module("pandapower")
    pp.runpp(net, algorithm="nr", numba=False)
    assert net._ppc is not None

    nr_input: dict[str, jnp.ndarray] = {}

    slack_id = jnp.asarray(net._ppc["internal"]["ref"], dtype=jnp.int32)
    pv_id = jnp.asarray(net._ppc["internal"]["pv"], dtype=jnp.int32)
    pq_id = jnp.asarray(net._ppc["internal"]["pq"], dtype=jnp.int32)
    nr_input["slack_id"] = slack_id
    nr_input["pv_id"] = pv_id
    nr_input["pq_id"] = pq_id

    admittance = jnp.asarray(net._ppc["internal"]["Ybus"].todense(), dtype=jnp.complex64)
    nr_input["admittance"] = admittance

    v_bus_out = jnp.asarray(net._ppc["internal"]["V"], dtype=jnp.complex64).flatten()
    s_inj_bus_out = v_bus_out * jnp.conj(admittance @ v_bus_out)

    v_bus_in = jnp.ones_like(v_bus_out, dtype=jnp.complex64)
    s_inj_bus_in = jnp.zeros_like(s_inj_bus_out, dtype=jnp.complex64)

    # Set slack and PV starting values in the provided initial guess
    v_bus_in = v_bus_in.at[slack_id].set(v_bus_out[slack_id])

    for bus_index in pv_id:
        v_bus_in = v_bus_in.at[bus_index].set(jnp.abs(v_bus_out[bus_index]) * jnp.exp(1j * 0.0))
        s_inj_bus_in = s_inj_bus_in.at[bus_index].set(jnp.real(s_inj_bus_out[bus_index]) + 0j)

    for bus_index in pq_id:
        s_inj_bus_in = s_inj_bus_in.at[bus_index].set(s_inj_bus_out[bus_index])

    nr_input["v_bus_in"] = v_bus_in
    nr_input["s_inj_bus_in"] = s_inj_bus_in

    return nr_input


def build_fixed_point_pd(net: Any) -> dict[str, jnp.ndarray]:
    """Build an exact fixed-point Newton-Raphson input from a solved network."""
    pp = import_module("pandapower")
    pp.runpp(net, algorithm="nr", numba=False)
    assert net._ppc is not None

    nr_input: dict[str, jnp.ndarray] = {}

    nr_input["pv_id"] = jnp.asarray(net._ppc["internal"]["pv"], dtype=jnp.int32)
    nr_input["pq_id"] = jnp.asarray(net._ppc["internal"]["pq"], dtype=jnp.int32)
    nr_input["admittance"] = jnp.asarray(
        net._ppc["internal"]["Ybus"].todense(), dtype=jnp.complex64
    )

    v_bus = jnp.asarray(net._ppc["internal"]["V"], dtype=jnp.complex64).flatten()
    nr_input["v_bus_in"] = v_bus
    nr_input["s_inj_bus_in"] = v_bus * jnp.conj(nr_input["admittance"] @ v_bus)

    return nr_input


def test_trivial_solution() -> None:
    """Test that the solver converges immediately on an exact case9 fixed point."""
    pn = import_module("pandapower.networks")
    pp = import_module("pandapower")
    net = pp.pandapowerNet(pn.case9())

    nr_input = build_fixed_point_pd(net)

    nr = NewtonRaphson()
    v_bus_out, s_inj_bus_out, (num_iterations, converged) = nr.solve(
        **nr_input,
    )

    assert num_iterations.shape == ()
    assert int(num_iterations) == 0
    assert bool(converged)
    assert jnp.allclose(v_bus_out, nr_input["v_bus_in"], atol=nr.tolerance)
    assert jnp.allclose(s_inj_bus_out, nr_input["s_inj_bus_in"], atol=nr.tolerance)


@pytest.mark.parametrize("load_scale", [0.5, 1.0, 1.5])
def test_compare_pandapower_case9(load_scale: float) -> None:
    """Compare the solver against pandapower on IEEE case 9."""

    pn = import_module("pandapower.networks")
    pp = import_module("pandapower")
    net = pp.pandapowerNet(pn.case9())
    nr = NewtonRaphson()

    np.random.seed(42)
    net.load["p_mw"] *= load_scale * np.random.uniform(0.8, 1.2, size=len(net.load))
    net.load["q_mvar"] *= load_scale * np.random.uniform(0.8, 1.2, size=len(net.load))

    nr_input = build_test_pd(net)
    # The solver signature does not accept `slack_id` (slack is fixed via v_bus_in).
    # Extract it from the prepared input for later assertions and remove
    # it from the dict before calling the solver.
    slack_id = nr_input.pop("slack_id")

    v_bus_out, _, (num_iterations, converged) = nr.solve(**nr_input)
    v_bus_out = jnp.asarray(v_bus_out)

    # Verify solver reports convergence
    assert num_iterations.shape == ()
    assert int(num_iterations) <= nr.max_iterations
    assert bool(converged)

    # Extract indices and check fixed/free variable constraints and residuals
    pv_id = nr_input["pv_id"]
    pq_id = nr_input["pq_id"]
    pvpq_id = jnp.concatenate([pv_id, pq_id])

    v_bus_in = nr_input["v_bus_in"]
    admittance = nr_input["admittance"]
    s_spec = nr_input["s_inj_bus_in"]

    # Slack: voltage must remain fixed
    assert jnp.allclose(v_bus_out[slack_id], v_bus_in[slack_id], atol=nr.tolerance)

    # PV: magnitude fixed (angle free)
    assert jnp.allclose(jnp.abs(v_bus_out[pv_id]), jnp.abs(v_bus_in[pv_id]), atol=nr.tolerance)

    # Power-flow residuals in reduced form: [P(pv+pq); Q(pq)] infinity-norm
    v = v_bus_out
    s_calc = v * jnp.conj(admittance @ v)
    s_mis = s_calc - s_spec
    residual = jnp.concatenate([jnp.real(s_mis[pvpq_id]), jnp.imag(s_mis[pq_id])])
    assert jnp.allclose(residual, jnp.zeros_like(residual), atol=nr.tolerance)


@pytest.mark.filterwarnings("ignore:tap_dependency_table is missing:DeprecationWarning")
@pytest.mark.parametrize("grid_name", ["case30", "case118"])
@pytest.mark.parametrize(
    "perturb_type, scale",
    [("load", 0.8), ("load", 1.2), ("gen", 0.9), ("gen", 1.1)],
)
def test_compare_pandapower_large_grids(grid_name: str, perturb_type: str, scale: float) -> None:
    """Compare the solver against pandapower on larger IEEE cases."""

    pn = import_module("pandapower.networks")
    net = getattr(pn, grid_name)()
    nr = NewtonRaphson()

    np.random.seed(42)
    if perturb_type == "load":
        net.load["p_mw"] *= scale * np.random.uniform(0.8, 1.2, size=len(net.load))
        net.load["q_mvar"] *= scale * np.random.uniform(0.8, 1.2, size=len(net.load))
    elif perturb_type == "gen":
        net.gen["p_mw"] *= scale * np.random.uniform(0.8, 1.2, size=len(net.gen))
        if len(net.sgen) > 0:
            net.sgen["p_mw"] *= scale * np.random.uniform(0.8, 1.2, size=len(net.sgen))

    nr_input = build_test_pd(net)
    # See comment above: pop `slack_id` for solver call, keep for checks.
    slack_id = nr_input.pop("slack_id")

    v_bus_out, _, (num_iterations, converged) = nr.solve(**nr_input)
    v_bus_out = jnp.asarray(v_bus_out)

    # Verify solver reports convergence
    assert num_iterations.shape == ()
    assert int(num_iterations) <= nr.max_iterations
    assert bool(converged)

    # Extract indices and check fixed/free variable constraints and residuals
    pv_id = nr_input["pv_id"]
    pq_id = nr_input["pq_id"]
    pvpq_id = jnp.concatenate([pv_id, pq_id])

    v_bus_in = nr_input["v_bus_in"]
    admittance = nr_input["admittance"]
    s_spec = nr_input["s_inj_bus_in"]

    # Slack: voltage must remain fixed
    assert jnp.allclose(v_bus_out[slack_id], v_bus_in[slack_id], atol=nr.tolerance)

    # PV: magnitude fixed (angle free)
    assert jnp.allclose(jnp.abs(v_bus_out[pv_id]), jnp.abs(v_bus_in[pv_id]), atol=nr.tolerance)

    # Power-flow residuals in reduced form: [P(pv+pq); Q(pq)] infinity-norm
    v = v_bus_out
    s_calc = v * jnp.conj(admittance @ v)
    s_mis = s_calc - s_spec
    residual = jnp.concatenate([jnp.real(s_mis[pvpq_id]), jnp.imag(s_mis[pq_id])])
    assert jnp.allclose(residual, jnp.zeros_like(residual), atol=nr.tolerance)
