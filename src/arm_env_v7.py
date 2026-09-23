"""Bounded decoupled mechanism: two joints that cannot go all the way round.

The decoupled mechanism gives each actuator its own visible pointer, and each
pointer turns freely, so the coordinate the model can build is a circle.  Most
actuated joints in standard continuous-control benchmarks are not like that:
they are hinges with stops, and a hinge with stops is a segment rather than a
circle.  A criterion that only tests for rotation cannot read one.

This environment is the decoupled mechanism with joint limits.  The action
still drives each joint, and saturates against the stop instead of wrapping.
It exists to check the generalised criterion in the direction the circular
environments cannot: a bounded joint should be carried as a shift along a
direction, not a turn in a plane, and the family detector should say so.

Renderer, image size, colours, sensor noise and action space are unchanged from
`arm_env_v5`.  Only the joint dynamics differ.
"""
from __future__ import annotations

import numpy as np

from arm_env import DT_ANGLE
from arm_env_v5 import render
from visual_env import IMAGE_SHAPE

LIMIT = 1.15          # radians; well short of a full turn, well inside the frame

__all__ = ["render", "rollout", "LIMIT"]


def rollout(batch, T, seed=0, palette=0, lighting=1.0, noise=0.01):
    rng = np.random.default_rng(seed)
    q = np.empty((batch, T, 2), dtype=np.float32)
    action = np.empty((batch, T, 2), dtype=np.float32)
    q[:, 0] = rng.uniform(-LIMIT, LIMIT, (batch, 2))
    velocity = rng.uniform(-0.8, 0.8, (batch, 2)).astype(np.float32)
    for t in range(T):
        if t > 0:
            velocity = 0.88 * velocity + 0.25 * rng.normal(
                0, 0.7, (batch, 2)).astype(np.float32)
            reset = rng.random((batch, 2)) < 0.045
            velocity = np.where(reset, rng.uniform(-1, 1, (batch, 2)),
                                velocity)
            velocity = np.clip(velocity, -1, 1)
        action[:, t] = velocity
        if t + 1 < T:
            # The stop is the whole point: the joint saturates, it never wraps.
            q[:, t + 1] = np.clip(q[:, t] + DT_ANGLE * velocity, -LIMIT, LIMIT)
    frames = np.empty((batch, T, *IMAGE_SHAPE), dtype=np.float32)
    render_rng = np.random.default_rng(seed + 73_001)
    for t in range(T):
        frames[:, t] = render(q[:, t], palette, lighting, noise, render_rng)
    return frames, action, q
