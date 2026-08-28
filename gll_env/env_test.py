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

"""The two clocks a timestep carries, and the promise each one makes.

``timestep.observation`` is what to act on; ``extras["transition"]`` is what to
learn and score from. They coincide while the environment is causal, and these
tests pin the coincidence so it is a checked fact rather than an accident that
quietly stops holding.
"""

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jumanji.types import StepType
from omegaconf import OmegaConf

from gll_env.env import ProsumerGrid
from gll_env.factories import _DEFAULT_CONFIG, environment_model
from gll_env.generator import DynamicsGenerator


def _leg_env(time_limit: int = 96, grid_code: dict | None = None) -> ProsumerGrid:
    config = OmegaConf.create(
        {
            **_DEFAULT_CONFIG,
            "grid_code": grid_code or {},
            "reward": {"name": "leg_settlement", "payments": "fair_leg"},
        }
    )
    model = environment_model(config)
    return ProsumerGrid(generator=DynamicsGenerator(model), time_limit=time_limit)


def _action(env: ProsumerGrid, value: float) -> chex.Array:
    return jnp.full(env.action_spec.shape, value, dtype=jnp.float32)


def test_the_aligned_transition_describes_the_interval_it_settles() -> None:
    """Every field of extras["transition"] belongs to the same interval: the
    action passed, the settlement it produced, and the states bracketing it."""
    env = _leg_env()
    state, first = env.reset(jr.PRNGKey(0))
    action = _action(env, 0.4)

    _, timestep = env.step(state, action)
    aligned = timestep.extras["transition"]

    chex.assert_trees_all_close(aligned.action, action)
    chex.assert_trees_all_close(aligned.reward, timestep.reward)
    # The state the action was chosen in is the one reset reported...
    chex.assert_trees_all_close(aligned.observation, first.observation)
    # ...and while the reward is causal, the settled interval ends where the
    # acting observation now begins.
    chex.assert_trees_all_close(aligned.next_observation, timestep.observation)
    assert bool(aligned.valid)


def test_reset_and_step_agree_on_the_extras_structure() -> None:
    """A scanned rollout carries extras, so the pytree reset emits and the one
    step emits must have identical structure -- including the placeholder
    transition, which is why reset emits one at all."""
    env = _leg_env()
    state, first = env.reset(jr.PRNGKey(0))
    _, timestep = env.step(state, _action(env, 0.0))

    assert jax.tree_util.tree_structure(first.extras) == jax.tree_util.tree_structure(
        timestep.extras
    )
    # Nothing has been settled at reset, so the placeholder must say so.
    assert not bool(first.extras["transition"].valid)


def test_the_reward_publishes_every_connection_point_not_just_the_agents() -> None:
    """The (num_agents,) reward cannot carry a connection point with no
    inverter. extras["reward"] is where the rest of the population lives."""
    env = _leg_env()
    model = env.environment
    state, _ = env.reset(jr.PRNGKey(0))
    _, timestep = env.step(state, _action(env, 0.2))

    settlement = timestep.extras["reward"].settlement_chf
    chex.assert_shape(settlement, (model.prosumer.num_pq,))
    chex.assert_trees_all_close(settlement[model.prosumer.inverter_id], timestep.reward, atol=1e-6)


def test_running_out_of_time_truncates_rather_than_terminates() -> None:
    """The episode was cut, not ended. `termination` here would set discount 0
    and train every state near the horizon toward a value of zero."""
    env = _leg_env(time_limit=2)
    state, _ = env.reset(jr.PRNGKey(0))

    state, timestep = env.step(state, _action(env, 0.0))
    assert timestep.step_type == StepType.MID
    chex.assert_trees_all_close(timestep.discount, jnp.ones_like(timestep.discount))

    state, timestep = env.step(state, _action(env, 0.0))
    assert bool(state.valid), "scenario should still be physically valid at the horizon"
    assert timestep.step_type == StepType.LAST
    chex.assert_trees_all_close(timestep.discount, jnp.ones_like(timestep.discount))


def test_grid_and_prosumer_observations_agree_on_the_same_flow() -> None:
    """bus_active_power_injection at a connection point IS that point's net
    metered energy. Reporting one in kW and the other in kWh put two features
    of the same flow, a factor of four apart, side by side in the agent view."""
    model = environment_model(OmegaConf.create(_DEFAULT_CONFIG))
    state = model.reset(jr.PRNGKey(3))
    state, _ = model.step(
        state, jnp.full((model.num_agents, model.action_dim), 0.3, dtype=jnp.float32)
    )

    observation = model.observation(state)
    at_bus = observation.grid_observation.bus_active_power_injection[model.grid.pq_id]
    at_meter = observation.prosumer_observation.p_pq_realized

    # Newton-Raphson's own tolerance is the only thing between them.
    chex.assert_trees_all_close(at_bus, at_meter, atol=1e-2)


def test_the_actor_view_reports_the_bounds_an_action_is_checked_against() -> None:
    """Under a grid code the feasible active-power interval is strictly
    narrower than the inverter's own range, and it is the narrower one an
    action is judged by."""
    env = _leg_env(grid_code={"name": "swiss_lv"})
    model = env.environment
    state, timestep = env.reset(jr.PRNGKey(0))

    minimum, maximum = state.action_constraints.bounds()
    scale = jnp.asarray(model.action_scale)
    view = timestep.observation.agents_view

    chex.assert_trees_all_close(view[:, -2], minimum * scale, atol=1e-5)
    chex.assert_trees_all_close(view[:, -1], maximum * scale, atol=1e-5)

    inverter = model.prosumer.inverter_dynamics
    del inverter  # bound comparison below reads the observation, not the model
    assert jnp.all(view[:, -1] <= jnp.asarray(model.action_scale) + 1e-5)
