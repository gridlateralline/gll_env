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
