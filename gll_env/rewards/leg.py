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

"""LEG (Lokale Elektrizitätsgemeinschaft) settlement reward.

A LEG is a local electricity community whose members trade with each other
at preferential *LEG* rates before falling back to the grid operator (*VNB*)
rates for whatever cannot be matched inside the community.

Settlement per interval:

1. Split each member's net metered energy into injection and consumption.
2. The community matches ``min(total_injection, total_consumption)``; the
   matched fraction is billed at LEG rates, the remainder at VNB rates.
3. Every member is billed at the *same* matched fraction, pro-rata on their
   own injection (resp. consumption) — no member gets preferential matching.

The reward is the resulting cash flow in CHF for the interval: positive for
a net earner, negative for a net payer. Rates come from a :class:`Payments`
asset (see :mod:`gll_env.assets.rewards_leg.generator`).
"""

from functools import cached_property
from typing import TYPE_CHECKING

import chex
import jax.numpy as jnp

from gll_env.components.prosumer import ProsumerDynamics
from gll_env.rewards.base import CausalReward
from gll_env.types import RewardObservation, RewardState

if TYPE_CHECKING:  # pragma: no cover -- rewards/ is imported by components/
    from gll_env.components.environment import EnvironmentDynamics, EnvironmentState


@chex.dataclass(frozen=True)
class LegRewardState(RewardState):
    """The settlement this reward last computed, kept for publication.

    LEG settlement itself is memoryless -- the community match is recomputed
    from scratch each interval. What this state carries is the ``(num_pq,)``
    result, which the ``(num_agents,)`` reward array cannot: connection points
    without an inverter take part in the community match, so their load is
    real supply and demand for the pool, but they have no agent to be
    attributed to. Without this they are invisible to anything downstream,
    including any fairness analysis -- which is exactly the population whose
    fairness is worth analysing.
    """

    settlement_chf: chex.Array  # (num_pq,) float32, signed


@chex.dataclass(frozen=True)
class LegRewardObservation(RewardObservation):
    """Per-connection-point settlement for the interval that just ended."""

    settlement_chf: chex.Array  # (num_pq,) float32, signed
    is_normalized: bool = False

    def normalize(self, reward_dynamics: "LegSettlementReward") -> "LegRewardObservation":
        """Scale by the largest rate on the books, so the result sits in
        roughly [-1, 1] for a connection point transacting its full rating.

        A settlement is a price times an energy, and the energy half is
        already normalized by ``s_pq_max_kvah`` elsewhere in the observation;
        dividing by the peak rate is the matching half.
        """
        return LegRewardObservation(
            settlement_chf=self.settlement_chf / reward_dynamics.payment_scale_chf_per_kwh,
            is_normalized=True,
        )


@chex.dataclass(frozen=True)
class Payments:
    """LEG settlement rates for one day, per interval, per kWh.

    Consumption rates are negative so that every rate can be multiplied by a
    *positive* energy quantity and summed — a member's reward is then simply
    the sum of its injection and consumption settlements.

    All four arrays are indexed by ``day_step``, so their length fixes the
    billing resolution (96 for 15-minute intervals, 24 for hourly) and must
    match the environment's ``n_steps_per_day``.
    """

    payment_leg_injection: chex.Array  # (n_steps_per_day,) float32 [CHF/kWh], positive
    payment_vnb_injection: chex.Array  # (n_steps_per_day,) float32 [CHF/kWh], positive
    payment_leg_consumption: chex.Array  # (n_steps_per_day,) float32 [CHF/kWh], negative
    payment_vnb_consumption: chex.Array  # (n_steps_per_day,) float32 [CHF/kWh], negative

    @cached_property
    def n_steps_per_day(self) -> int:
        """Number of billing intervals per day (e.g. 96 for 15-minute steps)."""
        return jnp.atleast_1d(self.payment_leg_injection).shape[0]

    def __post_init__(self) -> None:
        expected_shape = (self.n_steps_per_day,)
        chex.assert_shape(self.payment_leg_injection, expected_shape)
        chex.assert_shape(self.payment_vnb_injection, expected_shape)
        chex.assert_shape(self.payment_leg_consumption, expected_shape)
        chex.assert_shape(self.payment_vnb_consumption, expected_shape)

        chex.assert_type(self.payment_leg_injection, jnp.float32)
        chex.assert_type(self.payment_vnb_injection, jnp.float32)
        chex.assert_type(self.payment_leg_consumption, jnp.float32)
        chex.assert_type(self.payment_vnb_consumption, jnp.float32)


class LegSettlementReward(CausalReward):
    """Per-interval LEG settlement in CHF, one reward per agent.

    Billed on ``s_pq_realized_kvah.real`` — the net active energy metered at
    each grid connection point over the interval that just ended, i.e.
    inverter output minus household load, positive when injecting. This is
    the quantity a real LEG meter settles; the inverter's own output is not,
    because it ignores the load sitting behind the same meter.

    Settlement is computed on the ``num_pq`` connection axis and then gathered
    onto the ``num_inv`` agent axis via ``inverter_id``. Connection points
    with no inverter still take part in the community match — their load is
    real supply-demand for the pool — but their settlement is not attributed
    to any agent, since no agent controls them.

    Args:
        payments: LEG/VNB rate curves; ``n_steps_per_day`` must match the
            environment's clock.
        prosumer: The environment's :class:`ProsumerDynamics`, read for
            ``inverter_id`` (the agent → connection-point map) and for the
            clock used to validate ``payments``.
    """

    def __init__(self, payments: Payments, prosumer: ProsumerDynamics) -> None:
        chex.assert_equal(
            int(prosumer.time.n_steps_per_day),
            payments.n_steps_per_day,
        )

        inverter_id = jnp.asarray(prosumer.inverter_id, dtype=jnp.int32)
        # One inverter per connection point: ProsumerDynamics._s_pq_realized_kvah
        # scatters with `.at[inverter_id].set(...)`, which would silently drop
        # all but the last inverter on a shared bus. Settling a shared bus is
        # therefore not merely unattributable here — it is not modelled at all.
        if jnp.unique(inverter_id).shape[0] != inverter_id.shape[0]:
            raise ValueError(
                "LegSettlementReward requires at most one inverter per PQ bus; "
                f"inverter_id={inverter_id.tolist()} maps several agents onto the "
                "same connection point."
            )

        self._payments = payments
        self._inverter_id = inverter_id
        self._num_pq = prosumer.num_pq

    @cached_property
    def payment_scale_chf_per_kwh(self) -> chex.Numeric:
        """Largest rate on the books, the divisor :class:`LegRewardObservation`
        normalizes by. Read off the rates rather than hardcoded so a tariff
        with different bands normalizes against its own peak.
        """
        return jnp.maximum(
            jnp.max(jnp.abs(jnp.asarray(self._payments.payment_leg_injection))),
            jnp.maximum(
                jnp.max(jnp.abs(jnp.asarray(self._payments.payment_vnb_injection))),
                jnp.maximum(
                    jnp.max(jnp.abs(jnp.asarray(self._payments.payment_leg_consumption))),
                    jnp.max(jnp.abs(jnp.asarray(self._payments.payment_vnb_consumption))),
                ),
            ),
        )

    @staticmethod
    def _split_injection_consumption(e_pq_kwh: chex.Array) -> tuple[chex.Array, chex.Array]:
        """Split signed net energy into non-negative injection and consumption."""
        e_injected_kwh = jnp.maximum(e_pq_kwh, 0.0)
        e_consumed_kwh = jnp.maximum(jnp.negative(e_pq_kwh), 0.0)
        return e_injected_kwh, e_consumed_kwh

    @staticmethod
    def _match_ratios(
        e_injected_kwh: chex.Array,
        e_consumed_kwh: chex.Array,
    ) -> tuple[chex.Numeric, chex.Numeric]:
        """Fraction of injection / consumption matched inside the community.

        The community matches ``min(total_injection, total_consumption)``, so
        the long side is only partially matched and the short side fully. With
        no supply or no demand the corresponding ratio is zero, which also
        keeps the division safe.
        """
        e_injected_total_kwh = jnp.sum(e_injected_kwh)
        e_consumed_total_kwh = jnp.sum(e_consumed_kwh)
        e_matched_total_kwh = jnp.minimum(e_injected_total_kwh, e_consumed_total_kwh)

        ratio_injection = jnp.where(
            e_injected_total_kwh > 0.0, e_matched_total_kwh / e_injected_total_kwh, 0.0
        )
        ratio_consumption = jnp.where(
            e_consumed_total_kwh > 0.0, e_matched_total_kwh / e_consumed_total_kwh, 0.0
        )
        return ratio_injection, ratio_consumption

    @staticmethod
    def _settle(
        e_kwh: chex.Array,
        match_ratio: chex.Numeric,
        payment_leg: chex.Numeric,
        payment_vnb: chex.Numeric,
    ) -> chex.Array:
        """Bill non-negative energy at the LEG/VNB blend given by `match_ratio`."""
        e_leg_kwh = match_ratio * e_kwh
        e_vnb_kwh = (1.0 - match_ratio) * e_kwh
        return e_leg_kwh * payment_leg + e_vnb_kwh * payment_vnb

    @classmethod
    def _settlement_chf(
        cls,
        e_pq_kwh: chex.Array,
        day_step: chex.Numeric,
        payments: Payments,
    ) -> chex.Array:
        """Settle every connection point for one interval. Shape ``(num_pq,)``."""
        e_injected_kwh, e_consumed_kwh = cls._split_injection_consumption(e_pq_kwh)
        ratio_injection, ratio_consumption = cls._match_ratios(e_injected_kwh, e_consumed_kwh)

        settlement_injection_chf = cls._settle(
            e_kwh=e_injected_kwh,
            match_ratio=ratio_injection,
            payment_leg=jnp.asarray(payments.payment_leg_injection)[day_step],
            payment_vnb=jnp.asarray(payments.payment_vnb_injection)[day_step],
        )
        settlement_consumption_chf = cls._settle(
            e_kwh=e_consumed_kwh,
            match_ratio=ratio_consumption,
            payment_leg=jnp.asarray(payments.payment_leg_consumption)[day_step],
            payment_vnb=jnp.asarray(payments.payment_vnb_consumption)[day_step],
        )
        return settlement_injection_chf + settlement_consumption_chf

    def reset(self, key: chex.PRNGKey) -> LegRewardState:
        del key
        return LegRewardState(
            settlement_chf=jnp.zeros((self._num_pq,), dtype=jnp.float32),
        )

    def settle(
        self,
        reward_state: RewardState,
        state: "EnvironmentState",
        new_state: "EnvironmentState",
        dynamics: "EnvironmentDynamics",
    ) -> tuple[LegRewardState, chex.Array]:
        del state, reward_state, dynamics  # settlement depends only on the interval that ended
        # `s_pq_realized_kvah` and `time_state` are written together by
        # ProsumerDynamics.step, so `day_step` indexes exactly the interval
        # this energy was metered over.
        e_pq_kwh = jnp.real(new_state.prosumer_state.s_pq_realized_kvah)
        settlement_chf = self._settlement_chf(
            e_pq_kwh=e_pq_kwh,
            day_step=new_state.time_state.day_step,
            payments=self._payments,
        ).astype(jnp.float32)
        reward = settlement_chf[self._inverter_id]
        return LegRewardState(settlement_chf=settlement_chf), reward

    def observation(self, reward_state: RewardState) -> LegRewardObservation:
        return LegRewardObservation(
            settlement_chf=jnp.asarray(reward_state.settlement_chf, dtype=jnp.float32),
        )
