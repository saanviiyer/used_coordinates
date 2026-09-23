"""Vectorized two-joint visual robot-arm environment.

The model observes only a low-resolution RGB image and normalized joint
velocity commands.  Joint angles are retained by the simulator for held-out
evaluation and never supplied to training or coordinate discovery.
"""
from __future__ import annotations

import numpy as np

from visual_env import IMAGE_H, IMAGE_W, IMAGE_SHAPE


DT_ANGLE = 0.16
LINK_LENGTHS = (0.82, 0.64)


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _segment_distance(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax[:, None, None], py - ay[:, None, None]
    denom = vx * vx + vy * vy + 1e-12
    t = np.clip((wx * vx[:, None, None] + wy * vy[:, None, None]) /
                denom[:, None, None], 0.0, 1.0)
    dx = px - (ax[:, None, None] + t * vx[:, None, None])
    dy = py - (ay[:, None, None] + t * vy[:, None, None])
    return np.sqrt(dx * dx + dy * dy)


def render(q, palette=0, lighting=1.0, noise=0.01, rng=None):
    """Render joint angles ``(B,2)`` as overhead RGB images."""
    q = np.asarray(q, dtype=np.float32)
    b = len(q)
    x = np.linspace(-1.75, 1.75, IMAGE_W, dtype=np.float32)
    y = np.linspace(0.95, -0.95, IMAGE_H, dtype=np.float32)
    px, py = np.meshgrid(x, y)
    # Textured but pose-independent background.
    checker = ((np.arange(IMAGE_H)[:, None] // 2 +
                np.arange(IMAGE_W)[None, :] // 3) % 2).astype(np.float32)
    image = np.empty((b, IMAGE_H, IMAGE_W, 3), dtype=np.float32)
    image[:] = np.asarray([0.09, 0.11, 0.14], dtype=np.float32)
    image += checker[None, :, :, None] * 0.025

    q1, q12 = q[:, 0], q[:, 0] + q[:, 1]
    x0 = np.zeros(b, dtype=np.float32)
    y0 = np.zeros(b, dtype=np.float32)
    x1 = LINK_LENGTHS[0] * np.cos(q1)
    y1 = LINK_LENGTHS[0] * np.sin(q1)
    x2 = x1 + LINK_LENGTHS[1] * np.cos(q12)
    y2 = y1 + LINK_LENGTHS[1] * np.sin(q12)
    d1 = _segment_distance(px, py, x0, y0, x1, y1)
    d2 = _segment_distance(px, py, x1, y1, x2, y2)
    palettes = (
        ((0.92, 0.30, 0.20), (0.20, 0.62, 0.96)),
        ((0.68, 0.28, 0.88), (0.22, 0.82, 0.57)),
    )
    c1, c2 = [np.asarray(c, dtype=np.float32)
              for c in palettes[palette % len(palettes)]]
    image = np.where((d1 < 0.075)[..., None], c1, image)
    image = np.where((d2 < 0.065)[..., None], c2, image)
    # Joints and end effector give local landmarks without exposing angles.
    for cx, cy, color, radius in (
            (x0, y0, np.asarray((0.90, 0.90, 0.90), np.float32), 0.10),
            (x1, y1, np.asarray((0.98, 0.78, 0.18), np.float32), 0.095),
            (x2, y2, np.asarray((0.35, 0.95, 0.45), np.float32), 0.085)):
        dist = np.sqrt((px[None] - cx[:, None, None]) ** 2 +
                       (py[None] - cy[:, None, None]) ** 2)
        image = np.where((dist < radius)[..., None], color, image)
    image *= float(lighting)
    if rng is not None and noise > 0:
        image += rng.normal(0, noise, image.shape).astype(np.float32)
    return np.clip(image, 0, 1).transpose(0, 3, 1, 2).astype(np.float32)


def rollout(batch, T, seed=0, palette=0, lighting=1.0, noise=0.01):
    """Return frames, normalized actions, and evaluation-only joint angles."""
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
