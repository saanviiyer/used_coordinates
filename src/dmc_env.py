"""DeepMind Control Suite rollouts in the same shape as the arm environments.

The criterion has so far only been pointed at environments written for it. A
workshop on robot learning wants a robot, so this wraps the standard control
suite and returns exactly what `train_arm_visual.make_batch` returns: rendered
frames, the actions that produced them, and the ground-truth joint angles held
back for scoring.

Two properties of the suite matter for the measurement.

Its actuated joints are **bounded hinges**, not free rotations. A criterion
that only tests for a planar rotation cannot read one, which is why the
translation family had to exist before this file was worth writing.

Actions are torques, not position commands. Perturbing an action moves a joint
over the following steps rather than instantaneously, so the probe window has
to be long enough for the effect to appear in the model's predictions.
"""
from __future__ import annotations

import numpy as np

RENDER_H = RENDER_W = 64
CAMERA = 0


def make(domain, task, seed=0):
    from dm_control import suite
    return suite.load(domain, task, task_kwargs={"random": seed})


def joint_angles(env):
    """Hinge angles in radians. Evaluation only; never seen by discovery."""
    return np.asarray(env.physics.data.qpos, dtype=np.float32).copy()


def rollout(batch, T, domain="reacher", task="easy", seed=0, action_scale=1.0,
            height=RENDER_H, width=RENDER_W):
    """Random-policy rollouts. Returns frames, actions, joint angles.

    A random policy is deliberate: the world model is trained on what the body
    does under undirected torques, so nothing about the measurement depends on
    a task-specific policy having been learned first.
    """
    frames = np.empty((batch, T, 3, height, width), dtype=np.float32)
    spec = None
    actions, angles = None, None
    for b in range(batch):
        env = make(domain, task, seed=seed * 1_000 + b)
        spec = env.action_spec()
        if actions is None:
            actions = np.empty((batch, T, spec.shape[0]), dtype=np.float32)
        rng = np.random.default_rng(seed * 7919 + b)
        env.reset()
        if angles is None:
            angles = np.empty((batch, T, len(joint_angles(env))),
                              dtype=np.float32)
        velocity = rng.uniform(-1, 1, spec.shape[0])
        for t in range(T):
            # Smoothed random torques: white noise barely moves a body with
            # inertia, so the joints would never leave their start pose.
            velocity = 0.85 * velocity + 0.35 * rng.normal(0, 0.8,
                                                           spec.shape[0])
            action = np.clip(velocity * action_scale, spec.minimum,
                             spec.maximum)
            actions[b, t] = action
            angles[b, t] = joint_angles(env)
            px = env.physics.render(height, width, camera_id=CAMERA)
            frames[b, t] = px.transpose(2, 0, 1).astype(np.float32) / 255.0
            env.step(action)
    return frames, actions, angles


def action_dim(domain="reacher", task="easy"):
    return int(make(domain, task).action_spec().shape[0])


def n_joints(domain="reacher", task="easy"):
    env = make(domain, task)
    env.reset()
    return int(len(joint_angles(env)))
