"""Decoupled mechanism: two degrees of freedom that are visually independent.

The parity arm removes the proximal joint's visual advantage but cannot give
the two actuators separate visual signatures, because a serial chain routes
both through the same link.  This environment does: two pointers on separate
pivots, pointer j driven by actuator j alone, so the image factorises as one
function of q1 plus one function of q2 and a genuine two-torus is available to
be found.

It is the control that makes a negative result interpretable.  If no second
circle appears even here, the limit is the world model or the objective, not
the arm's kinematics.  Action space, dynamics, image size and sensor noise are
unchanged, so the only difference from the serial arm is the visual coupling.
"""
from __future__ import annotations

import numpy as np

from arm_env import DT_ANGLE, _segment_distance, wrap
from visual_env import IMAGE_H, IMAGE_W, IMAGE_SHAPE

PIVOTS = ((-0.80, 0.0), (0.80, 0.0))
LENGTH = 0.62


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
    palettes = (((0.95, 0.28, 0.16), (0.12, 0.48, 1.00)),
                ((0.72, 0.24, 0.92), (0.10, 0.82, 0.72)))
    colours = [np.asarray(c, np.float32)
               for c in palettes[palette % len(palettes)]]
    tips = [np.asarray((0.98, 0.78, 0.18), np.float32),
            np.asarray((0.30, 0.95, 0.42), np.float32)]
    for joint in (0, 1):
        cx, cy = PIVOTS[joint]
        ang = q[:, joint]
        x0 = np.full(b, cx, np.float32); y0 = np.full(b, cy, np.float32)
        x1 = x0 + LENGTH * np.cos(ang); y1 = y0 + LENGTH * np.sin(ang)
        d = _segment_distance(px, py, x0, y0, x1, y1)
        image = np.where((d < 0.085)[..., None], colours[joint], image)
        for ex, ey, colour, radius in (
                (x0, y0, np.asarray((0.90, 0.90, 0.90), np.float32), 0.075),
                (x1, y1, tips[joint], 0.090)):
            dist = np.sqrt((px[None] - ex[:, None, None]) ** 2 +
                           (py[None] - ey[:, None, None]) ** 2)
            image = np.where((dist < radius)[..., None], colour, image)
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
