"""Cached, batched rollout for arm-model conjugacy fits.

Two facts make the reference estimator far more expensive than it needs to be.
The discovery window is fully blacked out, so every frame the encoder sees in
that window is a constant image whose embedding can be computed once.  And the
gain search evaluates the same rollout at many rotation angles, which is one
wide batch rather than many narrow ones.

Nothing here changes what is computed.  ``tests/test_arm_fast.py`` checks
agreement with the reference path in ``analyze_arm_zeroshot``.
"""
from __future__ import annotations

import numpy as np
import torch

from visual_env import IMAGE_H, IMAGE_W

N_PIXEL = 3 * IMAGE_H * IMAGE_W


class CachedRollout:
    """Frame embeddings and action stream for one fixed probe window."""

    def __init__(self, base, rows):
        self.base = base
        self.kind = getattr(base, "kind", "visual_gru")
        B, T = rows.shape[0], rows.shape[1]
        self.B, self.T = B, T
        frames = rows[..., :N_PIXEL].reshape(B * T, 3, IMAGE_H, IMAGE_W)
        with torch.no_grad():
            self.z = base.frame_encoder(frames).reshape(B, T, -1)
        self.mask = rows[..., N_PIXEL]
        self.action = rows[..., N_PIXEL + 1:N_PIXEL + 3]

    def effective_delta(self, joint, delta):
        """Actual change in the actuator once the [-1, 1] clamp is applied."""
        a = self.action[:, 0, joint]
        return (torch.clamp(a + delta, -1.0, 1.0) - a)

    def _inputs(self, t, rep, joint=0, delta=0.0):
        z, mask, action = self.z[:, t], self.mask[:, t], self.action[:, t]
        if delta != 0.0:
            action = action.clone()
            action[:, joint] = torch.clamp(action[:, joint] + delta, -1.0, 1.0)
        if rep > 1:
            z, mask, action = z.repeat(rep, 1), mask.repeat(rep), action.repeat(rep, 1)
        return z, mask, action

    def _advance(self, z, mask, action, h):
        base = self.base
        if self.kind == "rssm":
            # In the probe window the observation is withheld, so the model
            # rolls on its prior.  ``advance`` blends by the mask, and the
            # prior mean is taken rather than sampled so the estimator's
            # target is deterministic given a checkpoint.
            state, _ = base.advance(h, action, mask,
                                    embed=z * (1.0 - mask)[:, None],
                                    sample=False)
            return state
        if self.kind == "arm_modular":
            out = []
            for j, core in enumerate(base.cores):
                x = torch.cat([z, mask[:, None], action[:, j:j + 1]], 1)[:, None]
                lo = j * base.module_hidden
                _, hn = core(x, h[:, lo:lo + base.module_hidden][None].contiguous())
                out.append(hn[0])
            return torch.cat(out, dim=-1)
        x = torch.cat([z, mask[:, None], action], 1)[:, None]
        _, hn = base.core(x, h[None].contiguous())
        return hn[0]

    @torch.no_grad()
    def roll(self, h, joint=0, delta=0.0, rep=1, want_pred=True):
        """Roll the frozen model.  ``delta`` perturbs one actuator at t=0."""
        preds = []
        for t in range(self.T):
            z, mask, action = self._inputs(t, rep, joint,
                                           delta if t == 0 else 0.0)
            h = self._advance(z, mask, action, h)
            if want_pred:
                preds.append(self.base.readout(h).reshape(h.shape[0], -1))
        return torch.stack(preds) if want_pred else h


def rotate_states(h, mu, plane, phi):
    """Rotate the ``plane`` coordinates of ``h`` by ``phi``.

    ``phi`` may be a scalar or one angle per state, which is what the
    clamp-corrected fit needs: the actuator saturates at +-1, so a nominal
    probe of 0.3 moves an action already at 0.9 by only 0.1, and every state
    must be rotated by the angle its own effective perturbation implies.
    """
    c = (h - mu) @ plane
    phi = torch.as_tensor(phi, dtype=h.dtype)
    if phi.ndim == 1:
        phi = phi[:, None]
    ca, sa = torch.cos(phi), torch.sin(phi)
    ca = ca.reshape(-1) if ca.numel() > 1 else ca.reshape(())
    sa = sa.reshape(-1) if sa.numel() > 1 else sa.reshape(())
    c2 = torch.stack([ca * c[:, 0] - sa * c[:, 1],
                      sa * c[:, 0] + ca * c[:, 1]], 1)
    return h + (c2 - c) @ plane.T


def translate_states(h, direction, shift):
    """Shift ``h`` along ``direction``.

    The translation counterpart of ``rotate_states``.  A bounded joint, a
    position, or any monotone coordinate is carried by a direction the action
    slides the state along, not by a plane it turns the state within.  ``shift``
    may be a scalar or one value per state.
    """
    direction = torch.as_tensor(direction, dtype=h.dtype).reshape(1, -1)
    shift = torch.as_tensor(shift, dtype=h.dtype)
    if shift.ndim == 0:
        shift = shift.reshape(1, 1)
    else:
        shift = shift.reshape(-1, 1)
    return h + shift * direction


def response_family(diff_plus, diff_minus):
    """Is this action a rotation or a translation of the latent state?

    A translation moves every state by the same vector, so the signed mean
    response is large relative to its spread.  A rotation moves each state
    according to where it sits on the circle, so the signed mean cancels and
    the spread carries everything.  The ratio separates the two before any
    fit is attempted, and costs one subtraction.
    """
    signed = (diff_plus - diff_minus) / 2.0
    mean = signed.mean(0)
    spread = (signed - mean).pow(2).sum(1).mean().sqrt()
    return {"mean_norm": float(mean.norm()),
            "residual_spread": float(spread),
            "translation_score": float(mean.norm() / (spread + 1e-12)),
            "direction": (mean / (mean.norm() + 1e-12)).tolist()}


@torch.no_grad()
def conjugacy_losses(cache, h, mu, plane, joint, deltas, gains, dt, chunk=6,
                     clamp_corrected=True):
    """Mean squared prediction error at each candidate gain.

    The target for perturbation ``d`` is the frozen model's own rollout with
    actuator ``joint`` perturbed by ``d``.  The candidate is the unperturbed
    rollout started from a state rotated by ``gain * d * dt`` in ``plane``.
    Include 0.0 in ``gains`` to obtain the no-rotation reference.

    With ``clamp_corrected`` the rotation angle uses each state's *effective*
    perturbation rather than the nominal one.  Without it, saturated actuators
    make the fitted gain read high, which is the failure the parent project
    traced to an action clamp on 22 August.
    """
    plane_t = torch.as_tensor(np.asarray(plane), dtype=h.dtype)
    mu = mu.to(h.dtype)
    targets = torch.stack([cache.roll(h, joint, float(d)) for d in deltas], 1)
    T, D, B, P = targets.shape
    flat = targets.reshape(T, D * B, P)
    gains = np.asarray(gains, dtype=np.float64)
    out = np.empty(len(gains), dtype=np.float64)
    for start in range(0, len(gains), chunk):
        block = gains[start:start + chunk]
        states = []
        for g in block:
            for d in deltas:
                if clamp_corrected:
                    phi = float(g * dt) * cache.effective_delta(joint, float(d))
                else:
                    phi = torch.tensor(float(g * d * dt), dtype=h.dtype)
                states.append(rotate_states(h, mu, plane_t, phi))
        H = torch.cat(states, 0)
        rep = len(block) * D
        total = torch.zeros(len(block), dtype=torch.float64)
        for t in range(cache.T):
            z, mask, action = cache._inputs(t, rep)
            H = cache._advance(z, mask, action, H)
            pred = cache.base.readout(H).reshape(rep * B, -1)
            err = (pred - flat[t].repeat(len(block), 1)) ** 2
            total += err.reshape(len(block), D * B * P).mean(1).double()
        out[start:start + len(block)] = (total / cache.T).numpy()
    return out


@torch.no_grad()
def translation_losses(cache, h, direction, joint, deltas, gains, dt, chunk=6,
                       clamp_corrected=True):
    """Mean squared prediction error at each candidate translation gain.

    Identical in structure to ``conjugacy_losses`` but the candidate transform
    slides the state along ``direction`` instead of turning it in a plane, so
    the two are directly comparable on the same targets and the same scale.
    """
    targets = torch.stack([cache.roll(h, joint, float(d)) for d in deltas], 1)
    T, D, B, P = targets.shape
    flat = targets.reshape(T, D * B, P)
    gains = np.asarray(gains, dtype=np.float64)
    out = np.empty(len(gains), dtype=np.float64)
    for start in range(0, len(gains), chunk):
        block = gains[start:start + chunk]
        states = []
        for g in block:
            for d in deltas:
                if clamp_corrected:
                    shift = float(g * dt) * cache.effective_delta(joint,
                                                                  float(d))
                else:
                    shift = torch.tensor(float(g * d * dt), dtype=h.dtype)
                states.append(translate_states(h, direction, shift))
        H = torch.cat(states, 0)
        rep = len(block) * D
        total = torch.zeros(len(block), dtype=torch.float64)
        for t in range(cache.T):
            z, mask, action = cache._inputs(t, rep)
            H = cache._advance(z, mask, action, H)
            pred = cache.base.readout(H).reshape(rep * B, -1)
            err = (pred - flat[t].repeat(len(block), 1)) ** 2
            total += err.reshape(len(block), D * B * P).mean(1).double()
        out[start:start + len(block)] = (total / cache.T).numpy()
    return out
