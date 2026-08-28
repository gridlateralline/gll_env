# Lookahead rewards

Everything shipped today is **causal**: the settlement for interval `t` depends
on nothing after `t`, `RewardDynamics.lookahead == 0`, and
`EnvironmentDynamics` raises for anything else. This document is the design for
the case that is not yet built, written down while the reasoning was fresh so
the next implementer starts from a checked design rather than a blank page.

Read `gll_env/rewards/base.py` first; the signature described there is already
the general one, and that was the only part that could not be added later
without breaking callers.

## When you actually need this

Less often than it looks. A settlement whose *total* is a function of an
online-updatable statistic is causal, and that covers most real tariffs:

```python
# Monthly peak / demand charge, entirely causal.
increment    = rate * jnp.maximum(0.0, p_t - reward_state.running_peak)
running_peak = jnp.maximum(reward_state.running_peak, p_t)
```

Emit `increment` every interval and the episode sum is exactly
`rate * max_t(P_t)`. Max, sum, mean, violation counts and approximate quantiles
all work this way. Capacity tariffs, congestion-hour counters and any nodal
price computed from the power flow that just solved are causal too.

Two things genuinely are not:

1. A **forward-looking window** — "the price at `t` is the mean congestion over
   `[t, t+4]`".
2. **Exact per-interval attribution** of an end-of-episode quantity — "your
   pro-rata share of the peak, attributed to the interval that set it".

For (2), check whether you need it in the *environment* at all. A scorer holds
the whole recorded trajectory and may attribute backwards however it likes.
Only the agent's reward signal has to be causal. Do not build this to satisfy
an analysis requirement.

## Why it needs a wrapper, not a bigger reward

A settlement for `t` that depends on realized states `t+1 … t+K` cannot be
emitted at `t`, because those states depend on actions not yet taken. That is
causality, not an API limitation.

The resolution is to delay **emission**, not to guess the future: buffer the
transitions and emit interval `t`'s settlement once `t+K` has happened. The
future in the buffer is then *real*, not assumed — which is strictly better
than pre-simulating a counterfactual continuation, and it costs responsiveness
rather than accuracy.

Delaying emission re-times the episode, and re-timing is what a Jumanji wrapper
is for. `RewardDynamics` does not change.

## Buffer sizes

With lookahead `K`, at wrapper step `m`:

| Buffer  | Length  | Contents after step `m`        |
| ------- | ------- | ------------------------------ |
| State   | `K + 2` | `[e_{m-1}, e_m, …, e_{m+K}]`   |
| Action  | `K + 1` | `[a_m, …, a_{m+K}]`            |
| Prefill | `K`     | inner steps; state slot 0 padded with `e_0` |

Notation: `e_i` is the state after interval `i` (`e_0` from reset), `a_i` is the
action applied during interval `i`, producing `e_i` from `e_{i-1}`.

The state buffer is `K + 2` and not `K + 1` because the reward brackets an
interval with two states — `RewardDynamics` is handed `trajectory[0]` and
`trajectory[1]` as the pair, so the predecessor counts.

Read positions after the roll:

- reward window — the **whole** state buffer, passed as the `trajectory` tuple
- head (act on this) — state slot **−1**, `e_{m+K}`
- aligned observation — state slot **0**, `e_{m-1}`
- aligned next observation — state slot **1**, `e_m`
- aligned action — action slot **0**, `a_m`

Worked trace, `K = 2` (state buffer 4, action buffer 3, prefill 2). After reset:
`[e₀, e₀, e₁, e₂]`, head `e₂`. Wrapper step 1, caller passes `a₃` (chosen from
head `e₂`) → `[e₀, e₁, e₂, e₃]`. Emits reward `g₁` computed over `e₀…e₃`, with
aligned action `a₁` — a warm-up action, as expected for the first `K` steps.

## The consistency argument

This is the part that makes the scheme legitimate rather than merely tidy, so
do not change the indices without re-deriving it.

At wrapper step `m` the caller sees head `e_{m+K-1}` and returns `a_{m+K}` —
observe, then act, in the ordinary MDP order. So `a_i` is *always* chosen
conditioned on `e_{i-1}`. The stored transition is `(e_{m-1}, a_m, g_m, e_m)`.

**The policy input in the buffer is exactly what the policy saw when it acted**,
surfaced `K` steps later rather than substituted. Without `aligned_action` in
extras, a learner would pair the action it just passed with a reward from `K`
intervals earlier and train on a lie, silently.

## The extras contract

Already emitted today, degenerate at `K = 0`, so consumers can be written
against it now — see `ProsumerGrid.get_aligned_timestep`.

```
timestep.observation            # head, e_{m+K}.  ACT ON THIS.
timestep.reward                 # g_m
extras["transition"]            # AlignedTransition — LEARN AND SCORE FROM THIS
  .observation                  #   e_{m-1}
  .action                       #   a_m
  .reward                       #   g_m  (same value as timestep.reward)
  .next_observation             #   e_m
  .valid                        #   False for the first K warm-up steps
extras["reward"]                # whatever the reward publishes
```

`timestep.observation` carries the **head** rather than the aligned
observation, deliberately. Both conventions fail silently for a consumer that
ignores extras, but asymmetrically: with the head there, such a consumer *acts
correctly* and only mis-assembles training tuples. With the aligned observation
there, it also acts on a `K`-step-stale state and clips against stale
`action_constraints`. The acting path has far more consumers than the learning
path, so the correct default belongs on it.

The consequence to document loudly: when `K > 0`, `timestep.observation` and
`timestep.reward` describe **different intervals**. Anything pairing them for
analysis — price against the voltage that caused it — must read
`extras["transition"]` instead.

## Four things that will bite

**Use `valid`, never `discount`, to mark warm-up.** Setting `discount = 0`
makes the Bellman target `y = r`, which tells the learner the episode *ended*
there; every episode's opening states get systematically under-valued and the
error propagates backwards. The correct treatment is also algorithm-dependent —
off-policy learners can keep warm-up transitions as free exploration data,
on-policy must drop them, scoring must drop them — and a boolean expresses all
three where a corrupted target expresses none.

**Do not zero `timestep.reward`.** It is correct data: summed over an episode
it is exactly the true total settlement, which is what every rollout utility
and return logger does with it. Zeroing saves nothing (the settlement is
computed regardless, for `extras`) and makes all of them silently report zero.

**Set inner `time_limit = emitted_limit + K`, and terminate on the emitted
index.** Terminating on the head drops the last `K` intervals of every episode.

**Warm-up actions should be random, sampled within the feasible bounds.**
Uniform in `[-1, 1]` is projected onto the feasible set and piles up on the
boundary — high entropy in the requested action, low entropy in the realized
one. At `action_dim == 1` use `action_constraints.bounds()` directly. If `K`
ever grows large, hold each action for several steps: SoC is the integral of
the action, so i.i.d. noise makes it a random walk that barely leaves its
starting value.

Also: buffer `valid` alongside the states, so an invalid head terminates the
emitted stream when the emitted index reaches it rather than `K` intervals
early.

## What stays put

`RewardState` lives inside `EnvironmentState` whatever happens here. That is
what makes a history-dependent reward Markov and reproducible from
`reset(key)`, and it is why an auto-reset wrapper cannot leak one episode's
reward memory into the next.

The wrapper's own buffers are a separate concern: **the wrapper is what the
agent remembers, `RewardState` is what the market remembers.** Keep them
apart, and keep market logic in `rewards/` rather than growing a second
implementation inside the wrapper — a windowed reward should be a declared
variant of `RewardDynamics` that the wrapper calls, not a `_get_reward` method
that reimplements settlement.
