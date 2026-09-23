"""Self-contained low-resolution egocentric RGB observations.

The image is a cylindrical first-person panorama. Each column is a physical
ray; projected wall height depends on depth, while texture, floor, ceiling, and
lighting create a spatial image that must pass through a convolutional encoder.
"""
from __future__ import annotations

import numpy as np

import envs


IMAGE_H, IMAGE_W = 16, 32
IMAGE_SHAPE = (3, IMAGE_H, IMAGE_W)

PALETTES = np.asarray([
    [[.82,.18,.16], [.14,.48,.84], [.18,.68,.31], [.86,.62,.12]],
    [[.55,.22,.72], [.10,.67,.65], [.88,.38,.18], [.44,.65,.16]],
    [[.72,.28,.35], [.20,.58,.72], [.55,.38,.76], [.82,.70,.20]],
], dtype=np.float32)


def render(pos, head, domain="rectangle", palette=0, lighting=1.0,
           noise=0.01, rng=None):
    """Render `(B,3,H,W)` panoramas from pose without exposing pose to models."""
    pos = np.asarray(pos); head = np.asarray(head)
    depth, wall = envs._raycast(pos, head, n_rays=IMAGE_W, domain=domain)
    B = len(pos)
    image = np.zeros((B, IMAGE_H, IMAGE_W, 3), dtype=np.float32)
    # Sky and floor gradients give the image a vertical spatial structure.
    y = np.linspace(0, 1, IMAGE_H, dtype=np.float32)
    sky = np.stack([.18 + .10*y, .23 + .12*y, .30 + .15*y], 1)
    floor = np.stack([.24 - .10*y, .22 - .09*y, .20 - .08*y], 1)
    image[:] = sky[None, :, None, :]
    image[:, IMAGE_H//2:] = floor[None, IMAGE_H//2:, None, :]
    half_height = np.clip((IMAGE_H * .34) / (depth + .18), 1, IMAGE_H/2)
    colors = PALETTES[palette % len(PALETTES)][wall]
    rows = np.arange(IMAGE_H)[None, :, None]
    lo = IMAGE_H/2 - half_height[:, None, :]
    hi = IMAGE_H/2 + half_height[:, None, :]
    mask = (rows >= lo) & (rows <= hi)
    # Image-space texture is deliberately nuisance variation: geometry and
    # landmark identity stay fixed while palette/lighting can be held out.
    stripes = (.82 + .18 * ((np.arange(IMAGE_W) // 3) % 2))[None, :, None]
    wall_rgb = colors * stripes
    image = np.where(mask[..., None], wall_rgb[:, None, :, :], image)
    shade = np.clip(1.15 / (1 + .18 * depth), .65, 1.05)
    image = np.where(mask[..., None], image * shade[:, None, :, None], image)
    image *= float(lighting)
    if rng is not None and noise > 0:
        image += rng.normal(0, noise, image.shape).astype(np.float32)
    image = np.clip(image, 0, 1)
    return image.transpose(0, 3, 1, 2).astype(np.float32)


def rollout(batch, T, seed=0, domain="rectangle", palette=0,
            lighting=1.0, noise=0.01):
    """Generate trajectories with RGB frames and normalized actions."""
    _, action, pos, head = envs.rollout(batch, T, seed=seed, domain=domain,
                                        noise=0)
    rng = np.random.default_rng(seed + 43_001)
    frames = np.empty((batch, T, *IMAGE_SHAPE), dtype=np.float32)
    for t in range(T):
        frames[:, t] = render(pos[:, t], head[:, t], domain, palette,
                              lighting, noise, rng)
    return frames, action, pos, head
