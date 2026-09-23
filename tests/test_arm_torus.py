"""Invariants for the torus estimator."""
import numpy as np
import torch

import analyze_arm_zeroshot as ref
import arm_fast
import arm_torus
from arm_models import build_arm_model


def _cache(kind="gru", batch=72):
    torch.manual_seed(0)
    base = build_arm_model(kind)
    base.eval()
    h, rows, _ = ref.collect_states(base, 5, batch=batch, T=24, t_hit=12,
                                    variant="v3")
    return base, h, rows, arm_fast.CachedRollout(base, rows)


def test_deflation_makes_the_second_plane_orthogonal():
    base, h, rows, cache = _cache()
    mu = h.mean(0, keepdim=True)
    rng = np.random.default_rng(0)
    planes, _ = arm_torus.discover(cache, h, mu, (0, 1), "deflate", 1, rng)
    overlap = planes[0].T @ planes[1]
    assert np.max(np.abs(overlap)) < 1e-8


def test_sequential_mode_does_not_enforce_orthogonality():
    base, h, rows, cache = _cache()
    mu = h.mean(0, keepdim=True)
    rng = np.random.default_rng(0)
    planes, _ = arm_torus.discover(cache, h, mu, (0, 1), "sequential", 1, rng)
    assert planes[0].shape == (base.hidden, 2)


def test_planes_are_orthonormal():
    base, h, rows, cache = _cache()
    mu = h.mean(0, keepdim=True)
    rng = np.random.default_rng(1)
    planes, _ = arm_torus.discover(cache, h, mu, (0, 1), "deflate", 1, rng)
    for plane in planes.values():
        assert np.allclose(plane.T @ plane, np.eye(2), atol=1e-8)


def test_gain_grid_brackets_the_reported_gain():
    """Any fit sitting at the coarse grid edge must be flagged, not reported."""
    base, h, rows, cache = _cache()
    mu = h.mean(0, keepdim=True)
    plane = np.linalg.qr(np.random.default_rng(2).normal(
        size=(base.hidden, 2)))[0]
    got = arm_torus.conjugacy(cache, h, mu, plane, 0)
    assert abs(got["gain"]) <= arm_torus.COARSE.max() + 1e-6
    assert got["gain_at_grid_edge"] in (True, False)


def test_uniformity_detects_a_uniform_circle():
    """A phase that is exactly the angle must show a local slope near one."""
    rng = np.random.default_rng(3)
    n = 3000
    angle = rng.uniform(-np.pi, np.pi, n)
    h = torch.zeros(n, 8)
    h[:, 0] = torch.from_numpy(np.cos(angle)).float()
    h[:, 1] = torch.from_numpy(np.sin(angle)).float()
    mu = torch.zeros(1, 8)
    plane = np.zeros((8, 2)); plane[0, 0] = 1.0; plane[1, 1] = 1.0
    got = arm_torus.uniformity(h, mu, plane, angle)
    assert abs(got["median_local_slope"] - 1.0) < 0.15
    assert got["iqr_local_slope"] < 0.2
