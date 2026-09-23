"""Structural properties the two new arm environments must have.

These are what the follow-up experiment rests on, so they are asserted rather
than assumed.
"""
import numpy as np

import arm_env
import arm_env_v2
import arm_env_v4
import arm_env_v5


def _sensitivity(env, delta=0.12, n=192, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-np.pi, np.pi, (n, 2))
    base = env.render(q, noise=0.0)
    out = []
    for joint in (0, 1):
        step = np.zeros_like(q)
        step[:, joint] = delta
        out.append(float(np.abs(env.render(arm_env.wrap(q + step), noise=0.0)
                                - base).mean()))
    return out


def test_parity_arm_is_balanced_and_serial_arm_is_not():
    a, b = _sensitivity(arm_env_v4)
    assert 0.9 < a / b < 1.15, (a, b)
    c, d = _sensitivity(arm_env_v2)
    assert c / d > 1.8, (c, d)


def test_parity_arm_depends_on_the_sum():
    """Two configurations with the same q1+q2 must look nearly identical."""
    rng = np.random.default_rng(1)
    n = 128
    total = rng.uniform(-np.pi, np.pi, n)
    split = rng.uniform(-np.pi, np.pi, n)
    qa = np.stack([split, arm_env.wrap(total - split)], 1)
    qb = np.stack([arm_env.wrap(split + 1.3),
                   arm_env.wrap(total - split - 1.3)], 1)
    same_sum = np.abs(arm_env_v4.render(qa, noise=0.0)
                      - arm_env_v4.render(qb, noise=0.0)).mean()
    shifted = np.stack([qa[:, 0], arm_env.wrap(qa[:, 1] + 1.3)], 1)
    different_sum = np.abs(arm_env_v4.render(qa, noise=0.0)
                           - arm_env_v4.render(shifted, noise=0.0)).mean()
    assert same_sum < 0.25 * different_sum, (same_sum, different_sum)


def test_decoupled_mechanism_factorises():
    """Moving one actuator must leave the other pointer's pixels untouched."""
    rng = np.random.default_rng(2)
    n = 64
    q = rng.uniform(-np.pi, np.pi, (n, 2))
    base = arm_env_v5.render(q, noise=0.0)
    half = base.shape[-1] // 2
    for joint, untouched in ((0, slice(half, None)), (1, slice(0, half))):
        step = np.zeros_like(q)
        step[:, joint] = 1.1
        moved = arm_env_v5.render(arm_env.wrap(q + step), noise=0.0)
        assert np.abs(moved[..., untouched] - base[..., untouched]).max() == 0.0
        assert np.abs(moved - base).mean() > 1e-4


def test_decoupled_sum_carries_nothing():
    """The two angles are independent, so their sum has no visual signature."""
    rng = np.random.default_rng(3)
    n = 256
    q = rng.uniform(-np.pi, np.pi, (n, 2))
    frames = arm_env_v5.render(q, noise=0.0).reshape(n, -1)
    frames = frames - frames.mean(0, keepdims=True)

    def concentration(angle):
        y = np.stack([np.cos(angle), np.sin(angle)], 1)
        w = np.linalg.solve(frames[:128].T @ frames[:128]
                            + np.eye(frames.shape[1]), frames[:128].T @ y[:128])
        pred = frames[128:] @ w
        got = np.arctan2(pred[:, 1], pred[:, 0])
        r = np.arctan2(np.sin(got - angle[128:]), np.cos(got - angle[128:]))
        return float(np.abs(np.mean(np.exp(1j * r))))

    assert concentration(q[:, 0]) > 0.9
    assert concentration(arm_env.wrap(q[:, 0] + q[:, 1])) < 0.4


def test_rollouts_share_the_dynamics():
    """Only the renderer differs; the joint trajectories must be identical."""
    _, a4, q4 = arm_env_v4.rollout(6, 12, seed=7)
    _, a5, q5 = arm_env_v5.rollout(6, 12, seed=7)
    _, a2, q2 = arm_env_v2.rollout(6, 12, seed=7)
    assert np.allclose(q4, q5) and np.allclose(q4, q2)
    assert np.allclose(a4, a5) and np.allclose(a4, a2)


def test_bounded_environment_never_wraps():
    """The bounded mechanism's joints must saturate, not wrap."""
    import arm_env_v7
    _, _, q = arm_env_v7.rollout(48, 60, seed=3)
    assert np.abs(q).max() <= arm_env_v7.LIMIT + 1e-6
    # A wrapping joint would show large single-step jumps; a stop shows none.
    step = np.abs(np.diff(q, axis=1))
    assert step.max() < 2 * arm_env_v7.LIMIT
    assert (np.abs(q) > arm_env_v7.LIMIT - 1e-4).mean() > 0.02


def test_bounded_environment_shares_the_renderer():
    import arm_env_v5
    import arm_env_v7
    rng = np.random.default_rng(0)
    q = rng.uniform(-1.0, 1.0, (8, 2))
    assert np.allclose(arm_env_v5.render(q, noise=0.0),
                       arm_env_v7.render(q, noise=0.0))
