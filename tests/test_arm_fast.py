"""The cached rollout must agree with the reference estimator."""
import numpy as np
import torch

import analyze_arm_zeroshot as ref
import arm_env
import arm_fast
from analyze_visual_zeroshot import FlatVisualAdapter
from arm_models import build_arm_model


def _setup(kind="gru", batch=8):
    torch.manual_seed(0)
    base = build_arm_model(kind)
    base.eval()
    h, rows, _ = ref.collect_states(base, 5, batch=batch, T=24, t_hit=12,
                                    variant="v3")
    return base, h, rows


def test_cached_rollout_matches_reference():
    base, h, rows = _setup()
    model = FlatVisualAdapter(base)
    cache = arm_fast.CachedRollout(base, rows)
    for joint in (0, 1):
        for delta in (0.0, 0.3):
            want = ref.roll(model, h, rows, joint, delta=delta)
            got = cache.roll(h, joint, delta)
            assert torch.allclose(want.reshape(got.shape), got, atol=1e-5)


def test_conjugacy_losses_match_reference():
    base, h, rows = _setup()
    model = FlatVisualAdapter(base)
    cache = arm_fast.CachedRollout(base, rows)
    mu = h.mean(0, keepdim=True)
    plane = np.linalg.qr(np.random.default_rng(1).normal(
        size=(base.hidden, 2)))[0]
    gains = np.array([0.0, 0.5, -1.25])
    got = arm_fast.conjugacy_losses(cache, h, mu, plane, 0, ref.DELTAS,
                                    gains, arm_env.DT_ANGLE)
    plane_t = torch.as_tensor(plane, dtype=h.dtype)
    targets = torch.stack([ref.roll(model, h, rows, 0, delta=d)
                           for d in ref.DELTAS])
    for i, g in enumerate(gains):
        preds = torch.stack([
            ref.roll(model, h, rows, 0, mu=mu, plane=plane_t,
                     phi=torch.tensor(float(g * d * arm_env.DT_ANGLE)))
            for d in ref.DELTAS])
        want = float(((preds - targets) ** 2).mean())
        assert abs(want - got[i]) < 1e-6 * max(1.0, want), (g, want, got[i])


def test_modular_model_path():
    base, h, rows = _setup("modular")
    model = FlatVisualAdapter(base)
    cache = arm_fast.CachedRollout(base, rows)
    want = ref.roll(model, h, rows, 1, delta=0.2)
    got = cache.roll(h, 1, 0.2)
    assert torch.allclose(want.reshape(got.shape), got, atol=1e-5)
