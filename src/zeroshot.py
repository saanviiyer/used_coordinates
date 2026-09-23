"""Finding the heading coordinate without pose labels.

Every plane used elsewhere in this repository is fitted with help from the
simulator: `ring_plane_geom` restricts the state cloud to one small spatial
bin before the PCA, and that restriction needs position. The measurement is
therefore honest about heading, but it is not label free, and a reviewer can
still ask whether the coordinate exists outside the analyst's choice of bin.

This module removes the labels. The model is asked a counterfactual question
about its own predictions, in the style of counterfactual world modelling:
perturb, compare, aggregate. Turning by omega and rotating the state inside
the right plane must have the same consequence for what the model expects to
see next, because both take the agent from heading theta to heading
theta + omega at the same position. So the heading plane can be defined as the
plane whose rotation best imitates a turn:

    Q*, g* = argmin  E_h E_d || f(h, turn + d) - f(rot(h, Q, g*d), turn) ||^2

where f rolls the model's own dynamics forward through the blackout and reads
out its own predicted observations. The comparison is differential: the agent
keeps the turn and the forward speed it actually commanded, and only the extra
increment d is moved from the action channel into the state.
Nothing in that objective knows the agent's pose. The simulator supplies the
trajectories and nothing else; pose enters only afterwards, to score the
coordinate that has already been found.

Three quantities come out of it:

    residual   the matched conjugacy error, divided by the error of doing
               nothing (rotating by zero). Below 1 means rotation imitates a
               turn better than inaction does; near 1 means no plane in the
               state space acts like a turn.
    gain       the discovered radians of ring phase per radian of physical
               turn. A model with a faithful heading coordinate reports ~1.
    kappa      circular concentration of the discovered phase against true
               heading, scored on held-out states after the search.

The controls are the same random planes used for the lesion analysis, refitted
with their own gain, so the comparison is against the best a matched arbitrary
subspace can do rather than against zero.
"""
import numpy as np
import torch

import envs, train


def _roll(model, h, rows, delta=0.0, mu=None, Q=None, phi=None):
    """Advance the model through `rows`, optionally after a state rotation and
    optionally with an extra turn added to the first action.

    Both branches see the same blacked-out inputs, so the only difference
    between them is where the extra heading came from: the action channel or
    the state.
    """
    if Q is not None:
        h = _rotate(h, mu, Q, phi)
    preds = []
    for k in range(rows.shape[1]):
        x = rows[:, k]
        if k == 0 and delta != 0.0:
            x = x.clone()
            x[:, -1] = torch.clamp(x[:, -1] + delta, -1.0, 1.0)
        h = model.step(x, h)
        preds.append(model.readout(h))
    return torch.stack(preds)


@torch.no_grad()
def collect_states(model, device, cond="dark", B=256, T=150, t_hit=70,
                   deep=6, horizon=4, seed=4321):
    """States mid-blackout, the blacked-out inputs that follow them, and the
    pose at that moment, which is kept aside for scoring only."""
    rng = np.random.default_rng(seed)
    x, O, m, P, Hd = train.make_batch(B, T, cond, seed, rng)
    x = x.to(device)
    x[:, t_hit - deep:t_hit + horizon + 1, :-3] = 0.0
    x[:, t_hit - deep:t_hit + horizon + 1, -3] = 1.0
    h = torch.zeros(B, model.hidden, device=device)
    for t in range(t_hit):
        h = model.step(x[:, t], h)
    rows = x[:, t_hit:t_hit + horizon]
    return h, rows, P.numpy()[:, t_hit], Hd.numpy()[:, t_hit]


@torch.no_grad()
def turn_predictions(model, h, rows, deltas):
    """What the model expects to see after each extra turn increment."""
    return torch.stack([_roll(model, h, rows, delta=d) for d in deltas])


def _rotate(h, mu, Q, phi):
    c = (h - mu) @ Q
    ca, sa = torch.cos(phi), torch.sin(phi)
    c2 = torch.stack([ca * c[:, 0] - sa * c[:, 1],
                      sa * c[:, 0] + ca * c[:, 1]], 1)
    return h + (c2 - c) @ Q.T


def conjugacy_loss(model, h, rows, mu, Q, gain, deltas, targets):
    """Mean squared disagreement between turning and rotating."""
    pred = torch.stack([_roll(model, h, rows, mu=mu, Q=Q,
                              phi=gain * d * envs.W_MAX) for d in deltas])
    return ((pred - targets) ** 2).mean()


def _search_once(model, h, rows, deltas, targets, mu, A0, gain0, steps, lr):
    A = torch.nn.Parameter(A0.clone())
    gain = torch.nn.Parameter(torch.tensor(float(gain0), device=h.device))
    opt = torch.optim.Adam([A, gain], lr=lr)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        Q, _ = torch.linalg.qr(A)
        loss = conjugacy_loss(model, h, rows, mu, Q[:, :2], gain, deltas, targets)
        loss.backward()
        opt.step()
        hist.append(float(loss))
    with torch.no_grad():
        Q, _ = torch.linalg.qr(A)
        Q = Q[:, :2]
        final = conjugacy_loss(model, h, rows, mu, Q, gain, deltas, targets)
        null = conjugacy_loss(model, h, rows, mu, Q, torch.zeros(()),
                              deltas, targets)
    return {"Q": Q.detach(), "gain": float(gain.detach()),
            "loss": float(final), "loss_still": float(null),
            "history": [round(v, 8) for v in hist[::20]]}


def _pca_init(h, k=2):
    """Top principal directions of the state cloud. Uses no labels."""
    hc = h - h.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(hc, full_matrices=False)
    return V[:k].T.contiguous()


def discover_plane(model, h, rows, deltas=(-1.0, -0.6, -0.3, 0.3, 0.6, 1.0),
                   steps=400, lr=0.05, seed=0, gain0=1.0, restarts=3):
    """Search for the plane whose rotation imitates an extra turn.

    The model's weights are frozen; the only free parameters are the plane and
    the gain. Orthonormality is imposed by a QR at every step, so the search
    runs over the Grassmannian rather than over arbitrary matrices.

    The objective has local minima: a single random start disagrees with itself
    across model seeds while the recovered coordinate does not, so the search is
    restarted. The first start is the state cloud's leading plane, which needs
    no labels and is a far better conditioned starting point than noise; the
    rest are random. The restart with the lowest matched loss wins, and the
    spread across restarts is reported rather than hidden.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    d = h.shape[1]
    mu = h.mean(0, keepdim=True)
    targets = turn_predictions(model, h, rows, deltas)
    g = torch.Generator(device="cpu").manual_seed(seed)
    inits = [_pca_init(h)]
    for _ in range(max(0, restarts - 1)):
        inits.append(torch.randn(d, 2, generator=g).to(h.device) / d ** 0.5)

    runs = [_search_once(model, h, rows, deltas, targets, mu, A0, gain0,
                         steps, lr) for A0 in inits]
    best = min(runs, key=lambda r: r["loss"])
    out = {"Q": best["Q"].cpu().numpy(), "mu": mu.detach().cpu().numpy(),
           "gain": best["gain"], "loss": best["loss"],
           "loss_still": best["loss_still"],
           "residual": best["loss"] / (best["loss_still"] + 1e-12),
           "history": best["history"],
           "restart_residuals": [round(r["loss"] / (r["loss_still"] + 1e-12), 4)
                                 for r in runs],
           "restart_gains": [round(r["gain"], 4) for r in runs],
           "won_by_pca_init": bool(best is runs[0])}
    return out


def fit_gain_only(model, h, rows, Q, deltas, steps=150, lr=0.05, gain0=1.0):
    """Best gain for a plane that was not chosen by the search."""
    for p in model.parameters():
        p.requires_grad_(False)
    mu = h.mean(0, keepdim=True)
    targets = turn_predictions(model, h, rows, deltas)
    Qt = torch.as_tensor(Q, dtype=torch.float32, device=h.device)
    gain = torch.nn.Parameter(torch.tensor(float(gain0), device=h.device))
    opt = torch.optim.Adam([gain], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = conjugacy_loss(model, h, rows, mu, Qt, gain, deltas, targets)
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = conjugacy_loss(model, h, rows, mu, Qt, gain, deltas, targets)
        null = conjugacy_loss(model, h, rows, mu, Qt, torch.zeros(()),
                              deltas, targets)
    return {"gain": float(gain.detach()), "loss": float(final),
            "residual": float(final) / (float(null) + 1e-12)}


def score_phase(h, mu, Q, head):
    """Score a discovered coordinate against pose. Evaluation only."""
    c = (h - mu) @ Q
    a = np.arctan2(c[:, 1], c[:, 0])
    best = None
    for w in (1, -1, 2, -2):
        r = np.arctan2(np.sin(a - w * head), np.cos(a - w * head))
        conc = float(np.abs(np.mean(np.exp(1j * r))))
        if best is None or conc > best[1]:
            best = (w, conc)
    return {"winding": best[0], "kappa": round(best[1], 4)}


@torch.no_grad()
def response_plane(model, h, rows, deltas=(-0.6, 0.6), n_comp=2):
    """The plane spanned by the state's response to an action increment.

    Cheaper and better conditioned than the search above, and closer to the
    aggregate step of counterfactual world modelling: perturb the action,
    subtract, and take the principal directions of the difference field. If an
    increment moves the state along a ring, the difference vectors are tangent
    to that ring, and as the phase varies over the batch the tangents sweep out
    the plane the ring lives in. No labels, no optimizer, one pass.
    """
    base = _roll_state(model, h, rows, delta=0.0)
    D = torch.cat([_roll_state(model, h, rows, delta=d) - base for d in deltas])
    D = D - D.mean(0, keepdim=True)
    _, S, V = torch.linalg.svd(D, full_matrices=False)
    var = (S ** 2) / float((S ** 2).sum() + 1e-12)
    return (V[:n_comp].T.contiguous().cpu().numpy(),
            [round(float(v), 4) for v in var[:4]])


def _roll_state(model, h, rows, delta=0.0):
    """The state after rolling through `rows`, with an optional extra turn."""
    for k in range(rows.shape[1]):
        x = rows[:, k]
        if k == 0 and delta != 0.0:
            x = x.clone()
            x[:, -1] = torch.clamp(x[:, -1] + delta, -1.0, 1.0)
        h = model.step(x, h)
    return h


def gain_on_plane(model, h, rows, Q, deltas=(-1.0, -0.6, -0.3, 0.3, 0.6, 1.0),
                  grid=None):
    """Best rotation gain for a fixed plane, by grid search.

    A scalar swept on a grid cannot land in a local minimum, which is the whole
    reason to fix the plane first.
    """
    grid = np.linspace(-2.0, 2.0, 81) if grid is None else np.asarray(grid)
    mu = h.mean(0, keepdim=True)
    Qt = torch.as_tensor(Q, dtype=torch.float32, device=h.device)
    with torch.no_grad():
        targets = turn_predictions(model, h, rows, deltas)
        losses = [float(conjugacy_loss(model, h, rows, mu, Qt,
                                       torch.tensor(float(g)), deltas, targets))
                  for g in grid]
        still = float(conjugacy_loss(model, h, rows, mu, Qt, torch.zeros(()),
                                     deltas, targets))
    i = int(np.argmin(losses))
    return {"gain": float(grid[i]), "loss": losses[i], "loss_still": still,
            "residual": losses[i] / (still + 1e-12),
            "curve": [round(v, 8) for v in losses[::8]]}
