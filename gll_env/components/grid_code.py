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

"""Grid-code control laws — what the connection rules take out of the agent's hands.

Today's tariffs (the LEG settlement among them) price only active energy and
prescribe reactive power by characteristic curve. :class:`QofUCharacteristic`
implements the Q(U) curve of VSE/AES NA/EEA-NE7 §4.3.2, Abbildung 5
("Standardeinstellung Q(U)-Kennlinie in Niederspannung"), which reduces the
agent to a single degree of freedom: active power.

:class:`GridCode` selects which laws apply. Its default — every law ``None`` —
is no law at all, i.e. the two-degree-of-freedom action space this environment
has always had, where the agent sets P and Q independently. That is not
today's legal regime; it is the counterfactual worth measuring against it.

Sign convention
---------------
NE7's figure is drawn in the *Erzeugerzählpfeilsystem* (generator reference
arrows), which is this repo's convention already: positive is out of the
inverter. So the curve transcribes with no sign flip. ``übererregt``
(over-excited, +Q, supplying reactive power) at low voltage raises it;
``untererregt`` (under-excited, -Q, absorbing) at high voltage lowers it.

Not implemented: P(U)
---------------------
NE7 §4.4 mandates a P(U) curve as standard alongside Q(U) — active power
derated linearly from 100% at 1.10 pu to 0 at 1.12 pu. It is deliberately
absent, and :class:`GridCode` is a holder with room for it rather than a bare
Q(U) reference so that adding it stays additive.

The reason is the simulation's 15-minute step, not the law. P(U) responds to
voltage with a 5-second time constant (§4.4(5)), so a real plant settles
*within* one interval, at the fixed point of the curve and the network.
Applying the cap from the previous interval's voltage instead — the only
option at this resolution without an inner power-flow loop — has enormous
loop gain (100% to 0% across 0.02 pu) and produces a period-2 limit cycle:
measured on the bundled ``cigre_lv_consumer`` feeder, active power alternating
between 4.465 and 13.000 kWh, voltage swinging 1.043-1.113 pu and sitting
above the 1.10 knee half the time, delivering 19% less energy than the true
fixed point (10.820 kWh at a steady 1.097 pu). That is worse than not
modelling P(U): it would read as compliance in config while behaving like
neither a compliant plant nor an unregulated one.

Q(U) has no such problem — its loop gain is ≈0.3, so the delayed application
is a contraction that converges on the same fixed point in a few intervals.
See ``docs/model_assumptions.md``.
"""

from dataclasses import field
from functools import cached_property

import chex
import jax.numpy as jnp

# NE7 §4.3.2, Abbildung 5. Voltage breakpoints in pu, and the reactive setpoint
# at each as a signed fraction of q_max_kvar. Between breakpoints the curve is
# linear; outside them it saturates (which `jnp.interp` does by default).
# [0.97, 1.03] is the deadband, where the plant exchanges no reactive power.
Q_OF_U_VOLTAGE_PU = (0.93, 0.97, 1.03, 1.07)
Q_OF_U_RATIO = (1.0, 0.0, 0.0, -1.0)

# NE7 §4.3 Tabelle 3, Typ 2 Stromrichter-EEA (converter-coupled plant, which is
# what a PV-plus-battery inverter is): the permitted power-factor range widens
# with plant rating. Q_max is the reactive power at the limiting cos φ, i.e.
# sin(arccos(cos φ)) * S_Emax. Below 800 VA no reactive requirement applies.
POWER_FACTOR_SMALL = 0.95  # 800 VA < S_Emax <= 3.7 kVA
POWER_FACTOR_LARGE = 0.90  # S_Emax > 3.7 kVA
RATING_THRESHOLD_KVA = 3.7
RATING_MINIMUM_KVA = 0.8


def limiting_power_factor(s_inv_max_kva: chex.Array) -> chex.Array:
    """NE7 Tabelle 3's cos φ for each inverter, selected by its own rating.

    Returns 1.0 below 800 VA — no reactive requirement, hence no reactive
    capability demanded, hence ``q_max_kvar = 0`` and a flat zero curve.
    """
    s_inv_max_kva = jnp.asarray(s_inv_max_kva, dtype=jnp.float32)
    return jnp.where(
        s_inv_max_kva <= RATING_MINIMUM_KVA,
        1.0,
        jnp.where(s_inv_max_kva <= RATING_THRESHOLD_KVA, POWER_FACTOR_SMALL, POWER_FACTOR_LARGE),
    )


def rated_q_max_kvar(s_inv_max_kva: chex.Array) -> chex.Array:
    """Q_max per NE7 Tabelle 3: ``sin(arccos(cos φ)) * S_Emax``, per inverter.

    Referenced to the plant's *nameplate* apparent power, not its instantaneous
    active power — Tabelle 3 is written against ``∑S_Emax`` and Abbildung 3 is
    captioned "bei Pmax". So the setpoint depends on voltage alone, with no
    coupling to whatever active power the agent happens to request.
    """
    s_inv_max_kva = jnp.asarray(s_inv_max_kva, dtype=jnp.float32)
    power_factor = limiting_power_factor(s_inv_max_kva)
    return s_inv_max_kva * jnp.sqrt(jnp.maximum(1.0 - jnp.square(power_factor), 0.0))


@chex.dataclass(frozen=True)
class QofUCharacteristic:
    """Q(U) droop curve — NE7 §4.3.2, Abbildung 5.

    Maps the voltage measured at each inverter's connection point to a
    reactive-energy setpoint for the coming interval. Piecewise linear through
    :data:`Q_OF_U_VOLTAGE_PU` / :data:`Q_OF_U_RATIO`, saturating at ±q_max
    outside them.

    Parameters
    ----------
    q_max_kvar : chex.Array
        Shape (num_inv,). Reactive power at full droop. Build it from
        :func:`rated_q_max_kvar` to follow Tabelle 3.
    voltage_pu, ratio : chex.Array
        The breakpoints, exposed so a VNB-specific curve can replace the
        standard one (§4.3(2) lets the operator set these per plant). Both
        shape (num_breakpoints,); ``voltage_pu`` must be ascending, as
        ``jnp.interp`` requires.

    The setpoint is *not* clamped to what the connection can carry here. That
    is deliberate: this class knows the plant's rating but not its load or its
    grid-connection limit, so it cannot say what is achievable. Derating
    against the live constraint set happens in
    :meth:`EnvironmentDynamics._grid_code_bounds`, which has all three.
    """

    q_max_kvar: chex.Array  # (num_inv,) float32
    voltage_pu: chex.Array = field(
        default_factory=lambda: jnp.asarray(Q_OF_U_VOLTAGE_PU, dtype=jnp.float32)
    )
    ratio: chex.Array = field(default_factory=lambda: jnp.asarray(Q_OF_U_RATIO, dtype=jnp.float32))

    @cached_property
    def num_inv(self) -> int:
        return jnp.atleast_1d(self.q_max_kvar).shape[0]

    def __post_init__(self) -> None:
        chex.assert_shape(self.q_max_kvar, (self.num_inv,))
        chex.assert_type(self.q_max_kvar, jnp.float32)
        chex.assert_rank(self.voltage_pu, 1)
        chex.assert_equal_shape([self.voltage_pu, self.ratio])
        chex.assert_type(self.voltage_pu, jnp.float32)
        chex.assert_type(self.ratio, jnp.float32)
        # A negative q_max would mirror the curve and drive voltage away from
        # nominal -- a data error, not a configuration, so clamp rather than
        # let it invert the control's sign. Same clamp-don't-raise reasoning
        # as InverterDynamics.s_inv_max_kva.
        object.__setattr__(self, "q_max_kvar", jnp.maximum(self.q_max_kvar, 0.0))

    def q_setpoint_kvarh(self, voltage_pu: chex.Array, step_duration_h: chex.Numeric) -> chex.Array:
        """Reactive energy target for the coming interval, shape (num_inv,).

        `voltage_pu` is the magnitude measured at each inverter's own bus,
        shape (num_inv,). Energy, not power, to match every other component's
        step() in this tree.
        """
        chex.assert_shape(voltage_pu, (self.num_inv,))
        ratio = jnp.interp(
            jnp.asarray(voltage_pu, dtype=jnp.float32),
            jnp.asarray(self.voltage_pu),
            jnp.asarray(self.ratio),
        )
        return ratio * jnp.asarray(self.q_max_kvar) * step_duration_h


@chex.dataclass(frozen=True)
class GridCode:
    """Which connection rules bind the agent.

    Every field defaulting to ``None`` means no rule binds, which is the
    two-degree-of-freedom action space this environment started with. Populate
    ``q_of_u`` to hand reactive power to the grid code and leave the agent with
    active power alone.

    A holder rather than a bare ``QofUCharacteristic | None`` because NE7
    prescribes several independent laws (§4.3 reactive, §4.4 active); only Q(U)
    is modelled here, and P(U) should slot in beside it without reshaping
    anything. See this module's docstring for why P(U) is absent.
    """

    q_of_u: QofUCharacteristic | None = None

    @cached_property
    def action_dim(self) -> int:
        """Degrees of freedom left to the agent: 1 under Q(U), otherwise 2."""
        return 2 if self.q_of_u is None else 1
