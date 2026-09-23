# Instrument: which group action does an actuator induce?

Status as of 28 August 2026. Validated in both directions before being pointed
at anything unfamiliar.

## The restriction that was found, and removed

The criterion inherited from the arm work tests whether an action acts as a
**rotation in a plane**. That is one one-parameter group among many. A bounded
hinge, a position, or any monotone coordinate is a segment rather than a
circle, and the rotation test cannot read one.

This is not academic. Actuated joints in standard continuous-control benchmarks
are predominantly bounded hinges, so the restriction decides whether the
criterion can be pointed at a larger system at all. It is why generalising
moved ahead of the scale experiment rather than after it.

## The detector

A **translation** response is state-independent: every state moves by the same
vector, so the signed mean response is large relative to its spread. A
**rotation** response depends on where the state sits on the circle, so the
signed mean cancels and the spread carries everything. One subtraction
separates the two before any fit is attempted.

Both families are then fitted on the same targets, at the same scale, and the
lower residual wins. Discovery uses actions and the frozen model's own
predictions; joint angles enter afterwards, for scoring only.

## Validation, eight seeds each, GRU, dark condition

| environment | plane | verdict | rotation residual | translation residual | score |
|---|---:|---|---:|---:|---:|
| parity arm, joints turn freely | 1 | rotation 8/8 | **0.084** | 0.446 | 0.371 |
| parity arm, joints turn freely | 2 | rotation 8/8 | **0.080** | 0.444 | 0.369 |
| bounded joints, stops at 1.15 rad | 1 | translation 8/8 | 0.326 | **0.125** | 0.740 |
| bounded joints, stops at 1.15 rad | 2 | translation 8/8 | 0.360 | **0.073** | 0.851 |

The two environments share a renderer, image size, colour scheme, sensor noise,
action space, architecture and training budget. Only the joint dynamics differ:
one wraps, the other saturates. The verdict follows the dynamics, unanimously,
in both directions.

## Two things not to misread

**The cheap score is a heuristic, not the verdict.** On synthetic responses the
two families separate by more than a hundredfold. On real ones the separation is
about twofold, 0.37 against 0.74 to 0.85, because real responses carry a
state-independent component. The residual comparison decides; the score is a
free hint that costs one subtraction.

**`arm_family` does not deflate, and `arm_torus` does.** They answer different
questions and will disagree on the same checkpoint. `arm_family` asks what group
*this* actuator induces, so on the parity arm both actuators legitimately find
the same circle and both score about 0.08. `arm_torus` asks whether there are
*two independent* coordinates, which requires forcing the second plane
orthogonal to the first, and on the same checkpoint reports the second plane as
weak. Neither is wrong. Quote the one whose question you are asking.

## What this unlocks

The criterion can now be pointed at a system whose degrees of freedom are
bounded, which is most of them. That was the blocker for the scale experiment.
It also adds a result rather than only a capability: which family an actuator
induces is an observation about what the model built, not a nuisance parameter.
