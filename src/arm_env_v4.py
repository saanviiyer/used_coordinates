"""Parity arm: the image depends on the joint angles only through their sum.

A serial chain cannot be made distal-dominant.  The proximal joint moves every
link beyond it while the distal joint moves only its own, so the image depends
on q1 and on q1+q2 and the proximal actuator's influence is bounded below by
the distal one's.  Sweeping the link lengths shows the ratio falling to exactly
1.00 and no further.

Parity is reached by shrinking the proximal link to nothing.  The arm becomes a
single visible link whose orientation is q1+q2, and the two actuators become
visually interchangeable.  This is the sharpest available test of whether the
discovered coordinate follows what the image shows or the actuator index: if it
follows the image, the circle should be the absolute distal angle q1+q2 and it
should be reachable by probing either actuator.
"""
from __future__ import annotations

import numpy as np

from arm_env import DT_ANGLE, _segment_distance, wrap
from visual_env import IMAGE_H, IMAGE_W, IMAGE_SHAPE

# Total reach is kept inside the vertical field of view so the tip does not
# leave frame, which would remove information rather than redistribute it.
LINK_LENGTHS = (0.05, 0.86)


def render(q, palette=0, lighting=1.0, noise=0.01, rng=None):
    q = np.asarray(q, dtype=np.float32)
    b = len(q)
    x = np.linspace(-1.75, 1.75, IMAGE_W, dtype=np.float32)
    y = np.linspace(0.95, -0.95, IMAGE_H, dtype=np.float32)
    px, py = np.meshgrid(x, y)
    checker = ((np.arange(IMAGE_H)[:, None] // 2 +
                np.arange(IMAGE_W)[None, :] // 3) % 2).astype(np.float32)
    image = np.empty((b, IMAGE_H, IMAGE_W, 3), dtype=np.float32)
    image[:] = np.asarray([0.09, 0.11, 0.14], dtype=np.float32)
    image += checker[None, :, :, None] * 0.025
    q1, q12 = q[:, 0], q[:, 0] + q[:, 1]
    x0 = np.zeros(b, np.float32); y0 = np.zeros(b, np.float32)
    x1 = LINK_LENGTHS[0] * np.cos(q1); y1 = LINK_LENGTHS[0] * np.sin(q1)
    x2 = x1 + LINK_LENGTHS[1] * np.cos(q12)
    y2 = y1 + LINK_LENGTHS[1] * np.sin(q12)
    d1 = _segment_distance(px, py, x0, y0, x1, y1)
    d2 = _segment_distance(px, py, x1, y1, x2, y2)
    palettes = (((0.95, 0.28, 0.16), (0.12, 0.48, 1.00)),
                ((0.72, 0.24, 0.92), (0.10, 0.82, 0.72)))
    c1, c2 = [np.asarray(c, np.float32)
              for c in palettes[palette % len(palettes)]]
    image = np.where((d1 < 0.070)[..., None], c1, image)
    image = np.where((d2 < 0.100)[..., None], c2, image)
    for cx, cy, color, radius in (
            (x0, y0, np.asarray((0.90, 0.90, 0.90), np.float32), 0.085),
            (x2, y2, np.asarray((0.30, 0.95, 0.42), np.float32), 0.100)):
        dist = np.sqrt((px[None] - cx[:, None, None]) ** 2 +
                       (py[None] - cy[:, None, None]) ** 2)
        image = np.where((dist < radius)[..., None], color, image)
    image *= float(lighting)
    if rng is not None and noise > 0:
        image += rng.normal(0, noise, image.shape).astype(np.float32)
    return np.clip(image, 0, 1).transpose(0, 3, 1, 2).astype(np.float32)


def rollout(batch, T, seed=0, palette=0, lighting=1.0, noise=0.01):
    rng = np.random.default_rng(seed)
    q = np.empty((batch, T, 2), dtype=np.float32)
    action = np.empty((batch, T, 2), dtype=np.float32)
    q[:, 0] = rng.uniform(-np.pi, np.pi, (batch, 2))
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
            q[:, t + 1] = wrap(q[:, t] + DT_ANGLE * velocity)
    frames = np.empty((batch, T, *IMAGE_SHAPE), dtype=np.float32)
    render_rng = np.random.default_rng(seed + 73_001)
    for t in range(T):
        frames[:, t] = render(q[:, t], palette, lighting, noise, render_rng)
    return frames, action, q
