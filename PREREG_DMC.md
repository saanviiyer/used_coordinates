# Pre-registration: control-suite reacher

Written 30 August 2026, before any world model was trained on this buffer.
The facts below are properties of the simulator and the replay buffer, read off
before the prediction was made.

## What the domain contains

`reacher-easy` has two actuated hinges, and they differ in the way the
criterion cares about:

| joint | limited | range | observed span |
|---|---|---|---|
| shoulder | **no** | unlimited | 13.63 rad, 2.17 full turns |
| wrist | **yes** | +-2.793 rad | 5.88 rad, 0.94 of a turn |

So one standard benchmark domain, and one trained model, contains both a free
rotation and a bounded hinge. Nothing had to be constructed to arrange this.

## Prediction

Discovery uses actions and the model's own predictions. Joint angles enter
afterwards, for scoring, and the shoulder's angle is wrapped to (-pi, pi]
because the simulator reports it unwrapped.

1. **Shoulder: rotation.** A joint that turns through more than two full
   revolutions cannot be carried as a shift along a direction without the
   representation running off. Predicted clearly, and a translation verdict here
   would be evidence the family detector is biased.
2. **Wrist: genuinely uncertain, and that is why it is worth reporting.**
   Its limit is +-160 degrees, so it covers 89% of a full turn. That is a hinge
   by construction and very nearly a circle by coverage. If the criterion calls
   it a translation, the family tracks the *constraint*; if it calls it a
   rotation, the family tracks the *coverage*. Either answer is informative and
   neither is a failure. What would be a failure is the two joints being
   indistinguishable.
3. **Both joints should beat their matched-variance null**, since both are
   actuated and both visibly move the arm.

## What would invalidate the measurement

- The world model failing to predict the arm at all, in which case there is no
  latent worth probing. Checked first by reconstruction error against a
  frame-persistence baseline, before any family is fitted.
- Every fit landing at the gain grid's edge, which would mean the grid does not
  bracket torque-driven dynamics and must be rewidened before anything is read.
- The two joints returning identical verdicts *and* identical residuals, which
  would suggest the estimator is reading one coordinate twice, as the arm's
  sequential estimator once did.

## Scope this does and does not extend

It extends the claim from environments written for the criterion to a standard
benchmark rendered by a physics engine, with torque actions rather than
velocity commands. It does not make the result large-scale, it does not involve
a real robot, and the model is a 3.75M-parameter RSSM trained on 20,480 frames
of random-policy interaction. Those limits go in the paper.
