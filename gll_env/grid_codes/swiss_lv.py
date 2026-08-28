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

"""Swiss low-voltage connection rules for generation plants.

Source: VSE/AES **NA/EEA-NE7** (CH 2025) -- *Netzanschluss fuer
Energieerzeugungsanlagen auf Netzebene 7*, the industry-association template
for connecting generation to the Swiss low-voltage grid. "NE7" in that
designator is *Netzebene 7*, the grid level, not the name of a rule; the
distribution system operator (VNB) adapts the template per plant, which is why
the curve breakpoints here are parameters rather than constants.

Implements 4.3.2, Abbildung 5 ("Standardeinstellung Q(U)-Kennlinie in
Niederspannung"): the plant exchanges reactive power as a function of the
voltage at its own connection point, which leaves the agent one degree of
freedom, active power. Today's tariffs, the LEG settlement among them, price
only active energy and prescribe reactive power exactly this way.

Sign convention
---------------
The figure is drawn in the *Erzeugerzaehlpfeilsystem* (generator reference
arrows), which is this repo's convention already: positive is out of the
inverter. So the curve transcribes with no sign flip. ``uebererregt``
(over-excited, +Q, supplying reactive power) at low voltage raises it;
``untererregt`` (under-excited, -Q, absorbing) at high voltage lowers it.

Not implemented: P(U)
---------------------
Section 4.4 mandates a P(U) curve as standard alongside Q(U) -- active power
derated linearly from 100% at 1.10 pu to 0 at 1.12 pu. It is deliberately
absent, and belongs here beside Q(U) when it lands, as another provision of
this same document.

The reason is the simulation's 15-minute step, not the rule. P(U) responds to
voltage with a 5-second time constant (4.4(5)), so a real plant settles
*within* one interval, at the fixed point of the curve and the network.
Applying the cap from the previous interval's voltage instead -- the only
option at this resolution without an inner power-flow loop -- has enormous
loop gain (100% to 0% across 0.02 pu) and produces a period-2 limit cycle:
measured on the bundled ``cigre_lv_consumer`` feeder, active power alternating
between 4.465 and 13.000 kWh, voltage swinging 1.043-1.113 pu and sitting
above the 1.10 knee half the time, delivering 19% less energy than the true
fixed point (10.820 kWh at a steady 1.097 pu). That is worse than not
modelling P(U): it would read as compliance in config while behaving like
neither a compliant plant nor an unregulated one.

Q(U) has no such problem -- its loop gain is ~0.3, so the delayed application
is a contraction that converges on the same fixed point. See
``docs/model_assumptions.md``.
"""

from dataclasses import field
from functools import cached_property

import chex
import jax.numpy as jnp

from gll_env.components.inverter import P_AXIS, Q_AXIS
from gll_env.grid_codes.base import GridCode
from gll_env.types import ActionConstraints

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
    """Tabelle 3's cos φ for each inverter, selected by its own rating.

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
    """Q_max per Tabelle 3: ``sin(arccos(cos φ)) * S_Emax``, per inverter.

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
    :meth:`SwissLvGridCode.reduce`, which has all three.
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


class SwissLvGridCode(GridCode):
    """The Swiss LV rules as they bind the agent: Q(U) claims the reactive axis.

    Args:
        q_of_u: The Q(U) characteristic, per 4.3.2. Build it with
            :func:`rated_q_max_kvar` to follow Tabelle 3.
    """

    def __init__(self, q_of_u: QofUCharacteristic) -> None:
        self._q_of_u = q_of_u

    @property
    def q_of_u(self) -> QofUCharacteristic:
        return self._q_of_u

    @property
    def action_dim(self) -> int:
        return 1  # active power only; Q(U) fixes the rest

    def reduce(
        self,
        s_inv_request_constraint: ActionConstraints,
        voltage_pu: chex.Array,
        step_duration_h: chex.Numeric,
    ) -> tuple[ActionConstraints, chex.Array]:
        """Pick the reactive setpoint and read off the active power left over.

        Both steps are exact restrictions of the constraint set to a line --
        see :meth:`~gll_env.types.ActionConstraints.restrict` for why that is
        closed-form rather than a search, and for what `origin_feasible`
        promises.

        Step 1 picks the setpoint. The curve reads voltage alone, so it can
        ask for reactive power the connection cannot carry: it is written
        against the plant's nameplate rating (Tabelle 3), which says nothing
        about the load sharing its meter. Restricting to p = 0 gives the
        reactive values that leave zero active power feasible, and clamping
        the curve's target into that range derates it to what the connection
        can actually absorb -- which is what a real inverter does, Tabelle 3's
        Q_max being a capability ceiling rather than a promise the connection
        can take it.

        That range always contains q = 0, because (0, 0) is feasible --
        ProsumerDynamics' own proven invariant -- so the clamp is always
        well-defined and can only pull the setpoint toward the curve's own
        zero, never past it into the opposite sign.

        Step 2 restricts to q = q* and reads off the active-power range,
        which is what step 1 earns the right to do with `origin_feasible`:
        (0, q*) was just established feasible, so p = 0 is guaranteed inside
        the result. The interval is therefore never empty, with no sizing
        condition to satisfy, and every point in it is feasible in 2-D -- the
        endpoints by construction, the interior by convexity. That is what
        leaves Prosumer's own projection with nothing to do.

        Note this is the SLICE at q*, not the shadow of the 2-D set onto the
        p axis. The slice is the smaller of the two and the correct one: it
        answers "what active power is available given this reactive
        setpoint", which is exactly the question the agent faces.
        """
        q_target_kvarh = self._q_of_u.q_setpoint_kvarh(voltage_pu, step_duration_h)

        zeros = jnp.zeros_like(q_target_kvarh)
        q_min_kvarh, q_max_kvarh = s_inv_request_constraint.restrict(P_AXIS, zeros).bounds()
        # The range provably straddles zero; clamping its ends against 0.0
        # keeps that true through float error.
        q_setpoint_kvarh = jnp.clip(
            q_target_kvarh, jnp.minimum(q_min_kvarh, 0.0), jnp.maximum(q_max_kvarh, 0.0)
        )

        p_min_kwh, p_max_kwh = s_inv_request_constraint.restrict(
            Q_AXIS, q_setpoint_kvarh, origin_feasible=True
        ).bounds()
        return ActionConstraints.from_bounds(p_min_kwh, p_max_kwh), q_setpoint_kvarh

    def lift(self, request: chex.Array, q_setpoint_kvarh: chex.Array) -> chex.Array:
        """Pair the agent's active power with the setpoint Q(U) imposed."""
        return jnp.stack([jnp.asarray(request)[:, P_AXIS], q_setpoint_kvarh], axis=-1)
