"""Causal tests on the recurrent state.

relaxation() distinguishes a continuous attractor from a point cloud that
merely happens to be ring shaped. A ring attractor must relax anisotropically:
displacements off the manifold decay, while displacement *along* it (a phase
shift) is marginally stable and persists. A contracting map with no attractor
returns to the unperturbed trajectory in every direction; an unstable one
diverges in every direction.

lesion() removes a subspace from the state at every timestep and re-runs the
dynamics, against two controls matched on the variance removed and on the
number of dimensions removed.
"""
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

import train


def fit_heading_readout(h, head, alpha=1.0):
    y = np.stack([np.cos(head), np.sin(head)], 1)
    return Ridge(alpha).fit(h, y)


def decode_angle(rd, h):
    p = rd.predict(h)
    return np.arctan2(p[:, 1], p[:, 0])


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


@torch.no_grad()
def relaxation(model, device, rd, ref_cloud, cond="dark", B=256, T=150,
               t_hit=70, horizon=40, eps_scale=1.0, seed=808, deep=4):
    """Perturb the state mid-blackout, then advance perturbed and unperturbed
    copies through identical inputs and compare them."""
    rng = np.random.default_rng(seed)
    x, O, m, P, Hd = train.make_batch(B, T, cond, seed, rng)
    x = x.to(device)
    # Force a long blackout spanning the perturbation and the whole horizon.
    x[:, t_hit - deep:t_hit + horizon, :-3] = 0.0
    x[:, t_hit - deep:t_hit + horizon, -3] = 1.0

    h = torch.zeros(B, model.hidden, device=device)
    for t in range(t_hit):
        h = model.step(x[:, t], h)
    sd = h.std(0, keepdim=True)
    eta = torch.randn_like(h)
    eta = eta / eta.norm(dim=1, keepdim=True) * sd.norm() * eps_scale
    hp = h + eta

    nn_ref = NearestNeighbors(n_neighbors=5).fit(ref_cloud)
    d_state, d_phase, d_mani, d_mani_ref = [], [], [], []
    for k in range(horizon):
        a0 = decode_angle(rd, h.cpu().numpy())
        a1 = decode_angle(rd, hp.cpu().numpy())
        d_state.append(float((hp - h).norm(dim=1).mean()))
        d_phase.append(float(np.abs(_wrap(a1 - a0)).mean()))
        d_mani.append(float(nn_ref.kneighbors(hp.cpu().numpy())[0][:, -1].mean()))
        d_mani_ref.append(float(nn_ref.kneighbors(h.cpu().numpy())[0][:, -1].mean()))
        h = model.step(x[:, t_hit + k], h)
        hp = model.step(x[:, t_hit + k], hp)
    return {"d_state": d_state, "d_phase": d_phase,
            "d_manifold": d_mani, "d_manifold_unperturbed": d_mani_ref}


def ring_plane_geom(h, pos, center, radius=0.25, n_comp=2):
    """Ring plane defined from state geometry, not from any readout.

    States are restricted to one small spatial bin before the PCA, so the plane
    describes variation with heading and cannot be a loop traced out by
    position. Defining it this way keeps the lesion independent of the decoder
    that later measures it.
    """
    sel = np.linalg.norm(pos - np.asarray(center), axis=1) < radius
    hb = h[sel]
    pca = PCA(n_comp).fit(hb - hb.mean(0))
    return pca.components_.T, int(sel.sum())


def principal_angles(A, B):
    """Principal angles (degrees) between two subspaces, ascending."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(sv, -1, 1)))


def matched_random_planes(h, R, rng, n_planes=20, tol=0.12):
    """Random 2D planes inside the orthogonal complement of R, drawn from the
    span of the top-K complement PCs with K chosen so that the variance they
    remove matches the ring plane's. Gives a null distribution matched on both
    dimension and variance removed."""
    hc = h - h.mean(0)
    var_ring = float((hc @ R).var(0).sum())
    Pc = np.eye(h.shape[1]) - R @ R.T
    pca = PCA(min(60, (hc @ Pc).shape[1])).fit(hc @ Pc)
    V = pca.components_.T

    def sample(K, n):
        out = []
        for _ in range(n):
            G = rng.normal(size=(K, 2))
            Q, _ = np.linalg.qr(V[:, :K] @ G)
            out.append(Q[:, :2])
        return out

    best_K, best_gap = 2, np.inf
    for K in range(2, V.shape[1] + 1):
        v = np.mean([float((hc @ q).var(0).sum()) for q in sample(K, 6)])
        gap = abs(v - var_ring) / max(var_ring, 1e-9)
        if gap < best_gap:
            best_gap, best_K = gap, K
        if gap < tol:
            break
    planes = sample(best_K, n_planes)
    got = float(np.mean([float((hc @ q).var(0).sum()) for q in planes]))
    return planes, var_ring, got, best_K


@torch.no_grad()
def lesion_eval(model, device, sub, cond="dark", B=256, T=140, seed=909,
                rd=None, deep=4, center=None):
    """Re-run the dynamics with `sub` projected out of the state each step."""
    rng = np.random.default_rng(seed)
    x, O, m, P, Hd = train.make_batch(B, T, cond, seed, rng)
    x, O = x.to(device), O.to(device)
    if sub is None:
        Pr = torch.eye(model.hidden, device=device)
        mu = torch.zeros(1, model.hidden, device=device)
    else:
        S = torch.tensor(sub, dtype=torch.float32, device=device)
        Pr = torch.eye(model.hidden, device=device) - S @ S.T
        mu = torch.tensor(center, dtype=torch.float32, device=device)[None] \
            if center is not None else torch.zeros(1, model.hidden, device=device)

    h = torch.zeros(B, model.hidden, device=device)
    H, preds = [], []
    for t in range(T):
        h = model.step(x[:, t], h)
        # Clamp the removed subspace to its dataset-average coordinate, so the
        # lesion deletes information without shifting the state's overall norm.
        h = (h - mu) @ Pr + mu
        H.append(h); preds.append(model.readout(h))
    H = torch.stack(H, 1); preds = torch.stack(preds, 1)

    se = ((preds[:, :-1] - O[:, 1:]) ** 2).mean(-1).cpu().numpy()
    mm = m.numpy()[:, :-1]
    run = np.zeros_like(mm)
    for t in range(1, mm.shape[1]):
        run[:, t] = (run[:, t - 1] + 1) * mm[:, t]
    deep_sel = run >= deep
    out = {"lit_mse": float(se[mm == 0].mean()),
           "dark_mse": float(se[mm == 1].mean()),
           "deep_dark_mse": float(se[deep_sel].mean())}
    # Heading is measured with a decoder REFIT on the lesioned states, under
    # cross-validation. Reusing the pre-lesion decoder would be circular: any
    # subspace removal that intersects its weights fails by construction.
    Hn = H.cpu().numpy()[:, :-1][deep_sel]
    hd = Hd.numpy()[:, :-1][deep_sel]
    out["deep_head_err_deg"] = cv_circ_decode(Hn, hd)
    out["n_deep"] = int(deep_sel.sum())
    return out


def cv_circ_decode(h, head, folds=4, alpha=1.0, max_n=20000, seed=0):
    """Cross-validated circular decode; median absolute error in degrees."""
    if len(h) > max_n:
        idx = np.random.default_rng(seed).choice(len(h), max_n, replace=False)
        h, head = h[idx], head[idx]
    y = np.stack([np.cos(head), np.sin(head)], 1)
    idx = np.arange(len(h)); errs = []
    for f in range(folds):
        te = idx % folds == f; tr = ~te
        if tr.sum() < 20 or te.sum() < 5:
            continue
        r = Ridge(alpha).fit(h[tr], y[tr])
        pr = r.predict(h[te]); pa = np.arctan2(pr[:, 1], pr[:, 0])
        errs.append(np.abs(_wrap(pa - head[te])))
    return float(np.degrees(np.median(np.concatenate(errs))))
