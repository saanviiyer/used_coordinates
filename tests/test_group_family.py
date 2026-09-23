"""The criterion must distinguish which one-parameter group an action induces.

A rotation and a translation are both one-parameter groups acting on the latent
state.  The estimator inherited from the arm work tests only for rotation, which
cannot read a bounded joint, a position, or any monotone coordinate.  These
tests pin the generalisation.
"""
import numpy as np
import torch

import arm_fast


def test_translate_states_scalar_and_per_state():
    h = torch.zeros(5, 4)
    v = np.array([1.0, 0.0, 0.0, 0.0])
    got = arm_fast.translate_states(h, v, 0.7)
    assert torch.allclose(got[:, 0], torch.full((5,), 0.7))
    assert torch.allclose(got[:, 1:], torch.zeros(5, 3))
    per = arm_fast.translate_states(h, v, torch.arange(5, dtype=torch.float32))
    assert torch.allclose(per[:, 0], torch.arange(5, dtype=torch.float32))


def test_translation_is_identity_at_zero_shift():
    rng = np.random.default_rng(0)
    h = torch.from_numpy(rng.normal(size=(16, 6)).astype(np.float32))
    v = rng.normal(size=6); v /= np.linalg.norm(v)
    assert torch.allclose(arm_fast.translate_states(h, v, 0.0), h)


def test_family_detector_calls_a_translation_a_translation():
    """A state-independent response is a translation."""
    rng = np.random.default_rng(1)
    n, d = 256, 8
    v = rng.normal(size=d); v /= np.linalg.norm(v)
    step = torch.from_numpy((0.3 * v).astype(np.float32)).expand(n, d)
    noise = torch.from_numpy(rng.normal(0, 0.002, (n, d)).astype(np.float32))
    got = arm_fast.response_family(step + noise, -step + noise)
    assert got["translation_score"] > 20, got
    assert abs(abs(float(np.dot(got["direction"], v))) - 1.0) < 0.02


def test_family_detector_calls_a_rotation_a_rotation():
    """A phase-dependent response cancels in the mean, so it is not a shift."""
    rng = np.random.default_rng(2)
    n = 512
    theta = rng.uniform(-np.pi, np.pi, n)
    eps = 0.15
    # State on a circle; the response to +eps is the tangent at that phase.
    plus = np.stack([np.cos(theta + eps) - np.cos(theta),
                     np.sin(theta + eps) - np.sin(theta)], 1)
    minus = np.stack([np.cos(theta - eps) - np.cos(theta),
                      np.sin(theta - eps) - np.sin(theta)], 1)
    pad = np.zeros((n, 6))
    got = arm_fast.response_family(
        torch.from_numpy(np.hstack([plus, pad]).astype(np.float32)),
        torch.from_numpy(np.hstack([minus, pad]).astype(np.float32)))
    assert got["translation_score"] < 0.2, got


def test_the_two_families_are_separated_by_orders_of_magnitude():
    rng = np.random.default_rng(3)
    n = 512
    v = np.zeros(8); v[0] = 1.0
    shift = torch.from_numpy(np.tile(0.2 * v, (n, 1)).astype(np.float32))
    translation = arm_fast.response_family(shift, -shift)["translation_score"]
    theta = rng.uniform(-np.pi, np.pi, n)
    plus = np.zeros((n, 8)); minus = np.zeros((n, 8))
    plus[:, 0] = np.cos(theta + 0.2) - np.cos(theta)
    plus[:, 1] = np.sin(theta + 0.2) - np.sin(theta)
    minus[:, 0] = np.cos(theta - 0.2) - np.cos(theta)
    minus[:, 1] = np.sin(theta - 0.2) - np.sin(theta)
    rotation = arm_fast.response_family(
        torch.from_numpy(plus.astype(np.float32)),
        torch.from_numpy(minus.astype(np.float32)))["translation_score"]
    assert translation > 100 * rotation, (translation, rotation)
