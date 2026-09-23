"""2D navigation with egocentric range-and-colour observations.

Ground-truth pose is generated so that it can be used for *analysis only*.
It is never an input to any model and never appears in any training loss.
"""
import numpy as np

# Arena is deliberately non-square and its four walls carry distinct colours,
# so heading is identifiable from a single observation (no residual symmetry).
W, H = 2.2, 1.4
N_RAYS = 12
N_WALL = 4
OBS_DIM = N_RAYS * (1 + N_WALL)

V_MAX, W_MAX = 0.08, 0.35


def _raycast(pos, head, n_rays=N_RAYS, arena=None, domain="rectangle"):
    """Distances and wall identities for a fan of rays, vectorised over batch.

    pos: (B, 2), head: (B,). Returns depth (B, R) and wall index (B, R).
    """
    B = pos.shape[0]
    width, height = (W, H) if arena is None else arena
    offs = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
    ang = head[:, None] + offs[None, :]              # (B, R)
    ux, uy = np.cos(ang), np.sin(ang)
    x = pos[:, 0:1]
    y = pos[:, 1:2]

    eps = 1e-9
    if domain == "ellipse":
        # Analytic ray--ellipse intersection. Boundary colours are assigned by
        # hit-point quadrant, retaining four landmarks while replacing the
        # rectangle's axis-aligned distance structure with a curved boundary.
        cx, cy = width / 2, height / 2
        a, b = width / 2, height / 2
        px, py = (x - cx) / a, (y - cy) / b
        dx, dy = ux / a, uy / b
        qa = dx * dx + dy * dy
        qb = 2 * (px * dx + py * dy)
        qc = px * px + py * py - 1
        disc = np.maximum(qb * qb - 4 * qa * qc, 0)
        depth = (-qb + np.sqrt(disc)) / (2 * qa + eps)
        hx, hy = x + depth * ux, y + depth * uy
        boundary_angle = np.arctan2((hy - cy) / b, (hx - cx) / a)
        wall = np.floor((boundary_angle + np.pi) / (np.pi / 2)).astype(int) % N_WALL
        return np.clip(depth, 0.0, 10.0), wall.astype(np.int64)
    if domain != "rectangle":
        raise ValueError(f"unknown domain: {domain}")
    # Parametric hit distance to each of the four bounding lines.
    tx = np.where(ux > 0, (width - x) / (ux + eps), (0.0 - x) / (ux - eps))
    ty = np.where(uy > 0, (height - y) / (uy + eps), (0.0 - y) / (uy - eps))
    tx = np.where(np.abs(ux) < 1e-8, np.inf, tx)
    ty = np.where(np.abs(uy) < 1e-8, np.inf, ty)

    hit_x = tx < ty
    depth = np.where(hit_x, tx, ty)
    # Wall id: 0 = x-max, 1 = x-min, 2 = y-max, 3 = y-min.
    wall = np.where(hit_x, np.where(ux > 0, 0, 1), np.where(uy > 0, 2, 3))
    return np.clip(depth, 0.0, 10.0), wall.astype(np.int64)


def observe(pos, head, rng=None, noise=0.02, arena=None, domain="rectangle"):
    depth, wall = _raycast(pos, head, arena=arena, domain=domain)
    inv = 1.0 / (1.0 + depth)                        # bounded, ~contrast coded
    onehot = np.eye(N_WALL, dtype=np.float32)[wall]  # (B, R, 4)
    obs = np.concatenate([inv[..., None], onehot], axis=-1)  # (B, R, 5)
    obs = obs.reshape(pos.shape[0], -1).astype(np.float32)
    if rng is not None and noise > 0:
        obs = obs + rng.normal(0, noise, obs.shape).astype(np.float32)
    return obs


def advance(pos, head, action, arena=None, domain="rectangle", margin=0.12):
    """Apply normalized forward/turn commands for closed-loop evaluation.

    Unlike exploratory ``rollout``, this function never changes the requested
    turn. Positions that would cross a boundary are projected to the navigable
    boundary, and the returned collision flag makes that event measurable.
    """
    width, height = (W, H) if arena is None else arena
    pos = np.asarray(pos, dtype=float)
    head = np.asarray(head, dtype=float)
    action = np.asarray(action, dtype=float)
    v = np.clip(action[..., 0], 0, 1) * V_MAX
    om = np.clip(action[..., 1], -1, 1) * W_MAX
    new_head = np.arctan2(np.sin(head + om), np.cos(head + om))
    nx = pos[..., 0] + v * np.cos(new_head)
    ny = pos[..., 1] + v * np.sin(new_head)
    if domain == "rectangle":
        collision = ((nx < margin) | (nx > width - margin)
                     | (ny < margin) | (ny > height - margin))
        nx = np.clip(nx, margin, width - margin)
        ny = np.clip(ny, margin, height - margin)
    elif domain == "ellipse":
        aa, bb = width / 2 - margin, height / 2 - margin
        q = np.sqrt(((nx - width / 2) / aa) ** 2
                    + ((ny - height / 2) / bb) ** 2)
        collision = q > 1
        safe_q = np.maximum(q, 1.0)
        nx = np.where(collision, width / 2 + (nx - width / 2) / safe_q * 0.999, nx)
        ny = np.where(collision, height / 2 + (ny - height / 2) / safe_q * 0.999, ny)
    else:
        raise ValueError(f"unknown domain: {domain}")
    return np.stack([nx, ny], axis=-1), new_head, collision


def rollout(batch, T, seed=0, noise=0.02, arena=None, domain="rectangle"):
    """Smooth exploratory trajectories with wall avoidance.

    Returns obs (B,T,OBS_DIM), act (B,T,2), pos (B,T,2), head (B,T).
    act[t] is the command applied to move from state t to state t+1.
    """
    rng = np.random.default_rng(seed)
    width, height = (W, H) if arena is None else arena
    if width <= 0.4 or height <= 0.4:
        raise ValueError("each arena dimension must exceed 0.4")
    if domain == "ellipse":
        angle = rng.uniform(-np.pi, np.pi, batch)
        radius = np.sqrt(rng.uniform(0, 0.70 ** 2, batch))
        pos = np.stack([width / 2 + radius * (width / 2) * np.cos(angle),
                        height / 2 + radius * (height / 2) * np.sin(angle)], axis=1)
    elif domain == "rectangle":
        pos = np.stack([rng.uniform(0.2, width - 0.2, batch),
                        rng.uniform(0.2, height - 0.2, batch)], axis=1)
    else:
        raise ValueError(f"unknown domain: {domain}")
    head = rng.uniform(-np.pi, np.pi, batch)
    om = np.zeros(batch)
    v = rng.uniform(0.3, 1.0, batch) * V_MAX

    O = np.zeros((batch, T, OBS_DIM), dtype=np.float32)
    A = np.zeros((batch, T, 2), dtype=np.float32)
    P = np.zeros((batch, T, 2), dtype=np.float32)
    Hd = np.zeros((batch, T), dtype=np.float32)

    for t in range(T):
        O[:, t] = observe(pos, head, rng, noise, arena=arena, domain=domain)
        P[:, t] = pos
        Hd[:, t] = head

        # AR(1) angular velocity keeps the heading signal temporally smooth.
        om = 0.8 * om + 0.2 * rng.normal(0, W_MAX, batch)
        v = np.clip(0.9 * v + 0.1 * rng.uniform(0.2, 1.0, batch) * V_MAX,
                    0.0, V_MAX)

        nx = pos[:, 0] + v * np.cos(head + om)
        ny = pos[:, 1] + v * np.sin(head + om)
        margin = 0.12
        if domain == "ellipse":
            aa, bb = width / 2 - margin, height / 2 - margin
            bad = (((nx - width / 2) / aa) ** 2
                   + ((ny - height / 2) / bb) ** 2) > 1
        else:
            bad = ((nx < margin) | (nx > width - margin)
                   | (ny < margin) | (ny > height - margin))
        # On a wall approach, steer hard instead of clipping, so the action
        # sequence stays a faithful description of the realised motion.
        om = np.where(bad, np.sign(rng.normal(size=batch)) * W_MAX, om)
        om = np.clip(om, -W_MAX, W_MAX)

        head = np.arctan2(np.sin(head + om), np.cos(head + om))
        nx = pos[:, 0] + v * np.cos(head)
        ny = pos[:, 1] + v * np.sin(head)
        if domain == "ellipse":
            aa, bb = width / 2 - margin, height / 2 - margin
            q = np.sqrt(((nx - width / 2) / aa) ** 2
                        + ((ny - height / 2) / bb) ** 2)
            outside = q > 1
            nx = np.where(outside, width / 2 + (nx - width / 2) / q * 0.999, nx)
            ny = np.where(outside, height / 2 + (ny - height / 2) / q * 0.999, ny)
        else:
            nx = np.clip(nx, margin, width - margin)
            ny = np.clip(ny, margin, height - margin)
        A[:, t, 0] = v / V_MAX
        A[:, t, 1] = om / W_MAX
        pos = np.stack([nx, ny], axis=1)

    return O, A, P, Hd
