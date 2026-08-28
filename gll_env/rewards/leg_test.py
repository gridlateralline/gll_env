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
import jax.numpy as jnp
import jax.random as jr
import pytest

from gll_env.components.day_time import DaytimeState
from gll_env.env import ProsumerGrid
from gll_env.factories import default_environment_model, reward_fn
from gll_env.generator import DynamicsGenerator
from gll_env.rewards.leg import LegSettlementReward, Payments

N_STEPS = 96
STEP_NT = 0  # 00:00, off-peak
STEP_HT = 40  # 10:00, peak


def _payments(
    leg_injection: float = 0.15,
    vnb_injection: float = 0.10,
    leg_consumption: float = -0.20,
    vnb_consumption: float = -0.25,
    n_steps_per_day: int = N_STEPS,
) -> Payments:
    """Flat rates, so a test's expected value never depends on `day_step`."""
    full = lambda rate: jnp.full((n_steps_per_day,), rate, dtype=jnp.float32)
    return Payments(
        payment_leg_injection=full(leg_injection),
        payment_vnb_injection=full(vnb_injection),
        payment_leg_consumption=full(leg_consumption),
        payment_vnb_consumption=full(vnb_consumption),
    )


def _settle(e_pq_kwh: list[float], payments: Payments, day_step: int = STEP_NT) -> chex.Array:
    """Settle a net-energy vector directly, bypassing the environment."""
    return LegSettlementReward._settlement_chf(
        e_pq_kwh=jnp.asarray(e_pq_kwh, dtype=jnp.float32),
        day_step=jnp.int32(day_step),
        payments=payments,
    )


def test_fully_matched_community_settles_entirely_at_leg_rates() -> None:
    payments = _payments()

    # 5 kWh injected, 5 kWh consumed -> both sides matched in full.
    settlement = _settle([5.0, -5.0], payments)

    assert jnp.allclose(settlement, jnp.asarray([5.0 * 0.15, 5.0 * -0.20]))


def test_unmatched_surplus_falls_back_to_vnb_rates() -> None:
    payments = _payments()

    # 10 kWh injected against 4 kWh of demand -> 40% of injection is matched,
    # while the consumer's whole 4 kWh is.
    settlement = _settle([10.0, -4.0], payments)

    expected_injection = 4.0 * 0.15 + 6.0 * 0.10
    assert jnp.allclose(settlement, jnp.asarray([expected_injection, 4.0 * -0.20]))


def test_match_ratio_is_shared_pro_rata_across_injectors() -> None:
    payments = _payments()

    # 12 kWh injected (9 + 3) against 6 kWh of demand: every injector is
    # matched at the same 50%, none preferentially.
    settlement = _settle([9.0, 3.0, -6.0], payments)

    expected = jnp.asarray(
        [
            9.0 * 0.5 * 0.15 + 9.0 * 0.5 * 0.10,
            3.0 * 0.5 * 0.15 + 3.0 * 0.5 * 0.10,
            6.0 * -0.20,
        ]
    )
    assert jnp.allclose(settlement, expected)


def test_community_with_no_demand_settles_entirely_at_vnb_rates() -> None:
    payments = _payments()

    settlement = _settle([5.0, 2.0], payments)

    assert jnp.allclose(settlement, jnp.asarray([5.0 * 0.10, 2.0 * 0.10]))


def test_idle_community_settles_to_zero() -> None:
    payments = _payments()

    settlement = _settle([0.0, 0.0], payments)

    assert jnp.allclose(settlement, 0.0)


def test_rates_are_looked_up_per_interval() -> None:
    """A peak/off-peak curve must be indexed by `day_step`, not averaged."""
    injection_rates = jnp.zeros((N_STEPS,), dtype=jnp.float32).at[STEP_HT].set(0.5)
    payments = Payments(
        payment_leg_injection=injection_rates,
        payment_vnb_injection=injection_rates,
        payment_leg_consumption=jnp.zeros((N_STEPS,), dtype=jnp.float32),
        payment_vnb_consumption=jnp.zeros((N_STEPS,), dtype=jnp.float32),
    )

    assert jnp.allclose(_settle([2.0, -2.0], payments, day_step=STEP_HT)[0], 1.0)
    assert jnp.allclose(_settle([2.0, -2.0], payments, day_step=STEP_NT)[0], 0.0)


def test_payments_rejects_mismatched_rate_shapes() -> None:
    with pytest.raises(AssertionError):
        Payments(
            payment_leg_injection=jnp.zeros((N_STEPS,), dtype=jnp.float32),
            payment_vnb_injection=jnp.zeros((N_STEPS - 1,), dtype=jnp.float32),
            payment_leg_consumption=jnp.zeros((N_STEPS,), dtype=jnp.float32),
            payment_vnb_consumption=jnp.zeros((N_STEPS,), dtype=jnp.float32),
        )


def test_reward_rejects_payments_on_a_different_clock() -> None:
    env_model = default_environment_model()  # 96 steps per day

    with pytest.raises(AssertionError):
        LegSettlementReward(
            payments=_payments(n_steps_per_day=24),
            prosumer=env_model.prosumer,
        )


def test_reward_is_billed_on_net_metered_flow_at_the_elapsed_interval() -> None:
    """End-to-end: the reward must match a hand settlement of the state it saw."""
    env_model = default_environment_model()
    payments = _payments()
    reward = LegSettlementReward(payments=payments, prosumer=env_model.prosumer)
    env = ProsumerGrid(
        generator=DynamicsGenerator(env_model),
        reward_fn=reward,
        time_limit=N_STEPS,
    )

    state, _ = env.reset(jr.PRNGKey(0))
    action = jnp.zeros((env.num_agents, 2), dtype=jnp.float32)
    new_state, timestep = env.step(state, action)

    e_pq_kwh = jnp.real(new_state.prosumer_state.s_pq_realized_kvah)
    expected = _settle(e_pq_kwh.tolist(), payments, int(new_state.time_state.day_step))

    chex.assert_shape(timestep.reward, (env.num_agents,))
    chex.assert_type(timestep.reward, jnp.float32)
    assert jnp.allclose(timestep.reward, expected, atol=1e-6)


def test_settlement_is_zero_sum_against_a_self_dealing_community() -> None:
    """With LEG rates that net to zero, a fully matched interval costs nothing."""
    payments = _payments(leg_injection=0.20, leg_consumption=-0.20)

    settlement = _settle([3.0, 1.0, -4.0], payments)

    assert jnp.allclose(jnp.sum(settlement), 0.0, atol=1e-6)


def test_reward_is_jittable() -> None:
    env_model = default_environment_model()
    reward = LegSettlementReward(payments=_payments(), prosumer=env_model.prosumer)
    env = ProsumerGrid(generator=DynamicsGenerator(env_model), reward_fn=reward, time_limit=N_STEPS)

    import jax

    state, _ = env.reset(jr.PRNGKey(0))
    action = jnp.zeros((env.num_agents, 2), dtype=jnp.float32)

    _, timestep = jax.jit(env.step)(state, action)

    assert jnp.all(jnp.isfinite(timestep.reward))


def test_day_step_wraps_with_the_clock() -> None:
    """The last interval of the day must index the last rate, not overflow."""
    env_model = default_environment_model()
    payments = _payments()
    reward = LegSettlementReward(payments=payments, prosumer=env_model.prosumer)

    last = DaytimeState(
        day_step=jnp.int32(N_STEPS - 1),
        interval_start=jnp.float32((N_STEPS - 1) / N_STEPS),
        interval_end=jnp.float32(1.0),
    )
    state = env_model.reset(jr.PRNGKey(0), time_state=last)
    new_state, _ = env_model.step(state, jnp.zeros((env_model.num_agents, 2), dtype=jnp.float32))

    assert int(new_state.time_state.day_step) == 0
    _, settled = reward(reward.reset(jr.PRNGKey(0)), (state, new_state), env_model)
    assert jnp.all(jnp.isfinite(settled))


def test_factory_builds_the_bundled_tariffs() -> None:
    from omegaconf import OmegaConf

    env_model = default_environment_model()

    for name in ("solarquartier", "fair_leg"):
        config = OmegaConf.create({"name": "leg_settlement", "payments": name})
        built = reward_fn(config, env_model)
        assert isinstance(built, LegSettlementReward)


def test_factory_rejects_an_unknown_reward_name() -> None:
    from omegaconf import OmegaConf

    with pytest.raises(ValueError, match="Unknown reward"):
        reward_fn(OmegaConf.create({"name": "nope"}), default_environment_model())


def test_fair_leg_rewards_trading_inside_the_community() -> None:
    """fair_leg must make LEG strictly better than VNB on both sides.

    It is built by waiving part of the grid usage fee and splitting the
    saving 50/50, so this holds by construction at every interval -- if it
    ever stops holding, the construction has drifted.
    """
    from omegaconf import OmegaConf

    from gll_env.factories import payments as payments_factory

    for name in ("fair_leg",):
        rates = payments_factory(OmegaConf.create({"payments": name}))
        assert jnp.all(rates.payment_leg_injection > rates.payment_vnb_injection), name
        assert jnp.all(rates.payment_leg_consumption > rates.payment_vnb_consumption), name


def test_solarquartier_is_only_worth_joining_at_peak_hours() -> None:
    """Pin the published tariff's real shape, which is not uniformly favourable.

    ewz quotes a flat 13 Rp. LEG energy rate against ewz.natur's 4.9 Rp.
    off-peak, so the 40% grid usage rebate does not cover the gap: buying
    inside the community is cheaper at peak but *dearer* off-peak. Agents are
    therefore expected to learn a time-of-use policy under this tariff, unlike
    the constructed ones -- so it is asserted, not treated as a defect.
    """
    from omegaconf import OmegaConf

    from gll_env.factories import payments as payments_factory

    rates = payments_factory(OmegaConf.create({"payments": "solarquartier"}))

    assert jnp.all(rates.payment_leg_injection > rates.payment_vnb_injection)
    assert rates.payment_leg_consumption[STEP_HT] > rates.payment_vnb_consumption[STEP_HT]
    assert rates.payment_leg_consumption[STEP_NT] < rates.payment_vnb_consumption[STEP_NT]
