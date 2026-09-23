"""Train action-conditioned world models under three input regimes.

lit     : observation always available (no integration pressure)
dark    : observation withheld in contiguous blackouts (integration required)
shuffle : same blackout burden, but actions are temporally shuffled, so the
          action stream carries no usable information about self-motion.
          This is the matched null for `dark`.
"""
import argparse, json, os, time, warnings
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
import envs, models

MEAN_RUN = 12.0


def blackout_mask(B, T, occupancy, rng, mean_run=MEAN_RUN):
    """Markov visible/blacked-out process. mask=1 means the observation is
    withheld at that step. Step 0 is always visible."""
    if occupancy <= 0:
        return np.zeros((B, T), dtype=np.float32)
    p_hide = (1.0 / mean_run) * occupancy / max(1e-6, 1 - occupancy)
    p_show = 1.0 / mean_run
    m = np.zeros((B, T), dtype=np.float32)
    state = np.zeros(B, dtype=bool)
    for t in range(1, T):
        u = rng.random(B)
        state = np.where(state, u > p_show, u < p_hide)
        m[:, t] = state
    return m


def make_batch(B, T, cond, seed, rng, arena=None, noise=0.02,
               domain="rectangle"):
    """cond is lit / dark / shuffle, or durN for the duration sweep."""
    O, A, P, Hd = envs.rollout(B, T, seed=seed, noise=noise, arena=arena,
                               domain=domain)
    # durN sweeps blackout length at fixed occupancy; occN sweeps occupancy
    # (as a percentage) at fixed length.
    if cond == "lit":
        occ = 0.0
    elif cond.startswith("occ"):
        occ = int(cond[3:]) / 100.0
    else:
        occ = 0.5
    run = int(cond[3:]) if cond.startswith("dur") else MEAN_RUN
    m = blackout_mask(B, T, occ, rng, mean_run=run)
    act = A.copy()
    if cond == "shuffle":
        idx = np.argsort(rng.random((B, T)), axis=1)
        act = np.take_along_axis(act, idx[..., None], axis=1)
    x = np.concatenate([O * (1 - m[..., None]), m[..., None], act], axis=-1)
    return (torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(O), torch.from_numpy(m),
            torch.from_numpy(P), torch.from_numpy(Hd))


def evaluate(model, cond, device, n=8, B=128, T=120, seed0=900000,
             domain="rectangle"):
    model.eval()
    rng = np.random.default_rng(12345)
    lit_se = lit_n = dark_se = dark_n = 0.0
    with torch.no_grad():
        for i in range(n):
            x, O, m, _, _ = make_batch(B, T, cond, seed0 + i, rng,
                                       domain=domain)
            x, O, m = x.to(device), O.to(device), m.to(device)
            pred, _ = model(x)
            se = ((pred[:, :-1] - O[:, 1:]) ** 2).mean(-1)
            mm = m[:, :-1]
            lit_se += float((se * (1 - mm)).sum()); lit_n += float((1 - mm).sum())
            dark_se += float((se * mm).sum()); dark_n += float(mm.sum())
    model.train()
    return {"lit_mse": lit_se / max(lit_n, 1),
            "dark_mse": dark_se / max(dark_n, 1) if dark_n > 0 else None}


def train(kind, cond, seed, steps, B, T, device, outdir, lr=2e-3,
          domain="rectangle"):
    torch.manual_seed(seed); np.random.seed(seed)
    model = models.build(kind, seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed + 777)
    t0 = time.time(); hist = []
    for s in range(steps):
        x, O, m, _, _ = make_batch(B, T, cond, seed * 100003 + s, rng,
                                   domain=domain)
        x, O = x.to(device), O.to(device)
        pred, _ = model(x)
        loss = ((pred[:, :-1] - O[:, 1:]) ** 2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (s + 1) % max(1, steps // 10) == 0:
            hist.append({"step": s + 1, "train_mse": float(loss)})
            print(f"  [{kind}/{cond}/s{seed}] {s+1}/{steps} "
                  f"mse={float(loss):.5f} ({time.time()-t0:.0f}s)", flush=True)
    # Evaluate every model under BOTH regimes, so lit-trained models are also
    # measured on the blackout task they were never trained for.
    res = {"kind": kind, "cond": cond, "seed": seed,
           "params": models.n_params(model), "hidden": model.hidden,
           "steps": steps, "domain": domain, "train_hist": hist,
           "eval_lit": evaluate(model, "lit", device, domain=domain),
           "eval_dark": evaluate(model, "dark", device, domain=domain),
           "wall_s": round(time.time() - t0, 1)}
    tag = f"{kind}_{cond}_s{seed}"
    torch.save(model.state_dict(), os.path.join(outdir, tag + ".pt"))
    json.dump(res, open(os.path.join(outdir, tag + ".json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="gru,rnn,tf")
    ap.add_argument("--conds", default="lit,dark,shuffle")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="runs/main")
    ap.add_argument("--domain", choices=("rectangle", "ellipse"),
                    default="rectangle")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for kind in a.kinds.split(","):
        for cond in a.conds.split(","):
            for seed in [int(s) for s in a.seeds.split(",")]:
                tag = f"{kind}_{cond}_s{seed}"
                if os.path.exists(os.path.join(a.outdir, tag + ".json")):
                    print("skip", tag); continue
                r = train(kind, cond, seed, a.steps, a.batch, a.T,
                          a.device, a.outdir, domain=a.domain)
                print(tag, "lit", round(r["eval_lit"]["lit_mse"], 5),
                      "dark", round(r["eval_dark"]["dark_mse"], 5), flush=True)
