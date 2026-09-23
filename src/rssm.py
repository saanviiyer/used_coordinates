"""A small recurrent state-space model, the architecture the criterion has to
survive if it is to be called a world-model result.

Everything measured so far used a plain convolutional GRU: a deterministic
recurrent state and nothing else.  A reviewer is entitled to ask whether the
criterion reads a *stochastic latent-variable* world model, which is what the
term usually means.  The difference is not cosmetic.  An RSSM carries a
deterministic path and a stochastic one, trains against a KL between a prior
that sees only the action and a posterior that also sees the observation, and
therefore has somewhere to put information that the deterministic path does not
carry.

Two properties make it a natural fit here rather than an awkward port.

The blackout condition already *is* the prior path.  When no observation
arrives the model has no posterior to form and must roll on the prior, which is
exactly the regime the criterion probes.  Nothing has to be bolted on.

The state is still a vector, so a plane in it still means something.  The
criterion's transform acts on the concatenation of the deterministic and
stochastic parts, and the rollout during a blackout is driven by actions alone.

Rollouts used for measurement take the prior mean rather than sampling, so the
comparison is deterministic given a checkpoint.  Sampling would make the
estimator's target a random variable and the residual meaningless.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from visual_env import IMAGE_H, IMAGE_W

ACTION_DIM = 2


class RSSM(nn.Module):
    kind = "rssm"

    def __init__(self, deter=192, stoch=32, embed=128, hidden=192,
                 min_std=0.1, image_h=IMAGE_H, image_w=IMAGE_W,
                 action_dim=ACTION_DIM):
        super().__init__()
        self.deter, self.stoch, self.min_std = deter, stoch, min_std
        self.image_h, self.image_w, self.action_dim = (image_h, image_w,
                                                       action_dim)
        # ``hidden`` is the width of the state the criterion sees.
        self.hidden = deter + stoch
        self.n_pixel = 3 * image_h * image_w

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * (image_h // 8) * (image_w // 8), embed),
            nn.ReLU())
        self.cell = nn.GRUCell(stoch + action_dim + 1, deter)
        self.prior_net = nn.Sequential(
            nn.Linear(deter, hidden), nn.ReLU(), nn.Linear(hidden, 2 * stoch))
        self.post_net = nn.Sequential(
            nn.Linear(deter + embed, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * stoch))
        self.decoder = nn.Sequential(
            nn.Linear(deter + stoch, 256), nn.ReLU(),
            nn.Linear(256, 3 * image_h * image_w), nn.Sigmoid())

    # -- distributions -----------------------------------------------------
    def _split(self, raw):
        mean, std = raw.chunk(2, dim=-1)
        return mean, F.softplus(std) + self.min_std

    def prior(self, deter):
        return self._split(self.prior_net(deter))

    def posterior(self, deter, embed):
        return self._split(self.post_net(torch.cat([deter, embed], -1)))

    # -- state plumbing ----------------------------------------------------
    def split_state(self, state):
        return state[:, :self.deter], state[:, self.deter:]

    def join_state(self, deter, stoch):
        return torch.cat([deter, stoch], -1)

    def initial(self, batch, ref):
        return ref.new_zeros(batch, self.hidden)

    def advance(self, state, action, mask, embed=None, sample=False):
        """One transition.  ``mask`` of 1 means the observation is withheld,
        so the prior is used; that is the blackout regime."""
        deter, stoch = self.split_state(state)
        x = torch.cat([stoch, action, mask[:, None]], -1)
        deter = self.cell(x, deter)
        mean, std = self.prior(deter)
        if embed is not None:
            pm, ps = self.posterior(deter, embed)
            visible = (1.0 - mask)[:, None]
            mean = visible * pm + (1 - visible) * mean
            std = visible * ps + (1 - visible) * std
        stoch = mean + std * torch.randn_like(std) if sample else mean
        return self.join_state(deter, stoch), (mean, std)

    # -- interfaces the criterion needs ------------------------------------
    def step(self, frame, mask, action, state):
        """Advance from a raw frame.  Masked frames contribute nothing."""
        embed = self.frame_encoder(frame)
        embed = embed * (1.0 - mask)[:, None]
        state, _ = self.advance(state, action, mask, embed)
        return state

    def readout(self, state):
        return self.decoder(state).reshape(-1, 3, self.image_h,
                                          self.image_w)

    # -- training ----------------------------------------------------------
    def observe(self, frames, mask, action, sample=True):
        B, T = frames.shape[:2]
        embed = self.frame_encoder(
            frames.reshape(B * T, 3, self.image_h,
                           self.image_w)).reshape(B, T, -1)
        embed = embed * (1.0 - mask)[..., None]
        state = self.initial(B, embed)
        states, kls = [], []
        for t in range(T):
            deter, stoch = self.split_state(state)
            x = torch.cat([stoch, action[:, t], mask[:, t, None]], -1)
            deter = self.cell(x, deter)
            prior_mean, prior_std = self.prior(deter)
            post_mean, post_std = self.posterior(deter, embed[:, t])
            visible = (1.0 - mask[:, t])[:, None]
            mean = visible * post_mean + (1 - visible) * prior_mean
            std = visible * post_std + (1 - visible) * prior_std
            stoch = mean + std * torch.randn_like(std) if sample else mean
            state = self.join_state(deter, stoch)
            states.append(state)
            kl = (torch.log(prior_std / post_std)
                  + (post_std ** 2 + (post_mean - prior_mean) ** 2)
                  / (2 * prior_std ** 2) - 0.5).sum(-1)
            kls.append(kl * visible[:, 0])
        states = torch.stack(states, 1)
        pred = self.decoder(states).reshape(B, T, 3, self.image_h,
                                            self.image_w)
        return pred, states, torch.stack(kls, 1)

    def forward(self, frames, mask, action, h0=None, return_h=False):
        pred, states, _ = self.observe(frames, mask, action)
        return (pred, states) if return_h else pred


def build_rssm(**kwargs):
    return RSSM(**kwargs)
