# Model Assumptions

- Component inputs and outputs are energy over one simulation interval: kWh,
  kvarh, and kVAh. Ratings in kW, kvar, and kVA are converted using the
  interval duration.
- Power is treated as constant over each interval. Solar and load profiles use
  midpoint sampling; the primary intended resolution is 15 minutes.
- Battery charging is negative and discharging is positive. Solar generation
  and load consumption are positive. Net grid flow is inverter output minus
  load consumption.
- Battery, inverter, and grid-connection limits are idealized static bounds.
  Battery efficiency, losses, self-discharge, and inverter losses are not
  modeled.
- Load reactive energy comes from a fixed lagging power factor. Solar reactive
  power is not modeled independently.
- Solar clearness and load factor are bounded mean-reverting stochastic
  processes. The solar daylight envelope is a simplified cosine profile.
- `RadialFeasibility` is a radial feasibility map, not a Euclidean nearest
  point projection. It is deliberately used as an accelerated constraint
  enforcement method.
- The optional Q(U) grid code (VSE/AES NA/EEA-NE7 §4.3.2) reduces the action
  space to active power alone. Reactive power follows the standard curve
  evaluated at the voltage measured in the *previous* interval. The real
  control has a 5-second time constant, so it settles well inside a 15-minute
  step; the one-interval lag avoids an algebraic loop between reactive power
  and the voltage it helps set, and the delayed loop is a contraction that
  converges on the same fixed point.
- `Q_max` follows NE7 Tabelle 3, referenced to the inverter's nameplate
  apparent power, not to instantaneous active power.
- The Q(U) setpoint is derated when the grid connection cannot carry it, so
  the curve is followed only as far as apparent-power headroom allows. Real
  inverters derate reactive output the same way; Tabelle 3's `Q_max` is a
  capability ceiling, not a guarantee the connection can absorb it.
- Active-power bounds under Q(U) are exact and closed-form: both constraint
  families are closed under restriction to a line. The one guard is that the
  ball radius at the chosen reactive setpoint is floored at the distance
  needed to keep zero active power feasible -- a no-op in exact arithmetic,
  which makes the non-empty-interval property hold syntactically rather than
  depending on how two equivalent square roots round.
- NE7 §4.4's P(U) curve is not modelled. At a 15-minute step its delayed
  application limit-cycles rather than settling; see `components/grid_code.py`.
