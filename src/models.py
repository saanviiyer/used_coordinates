"""Action-conditioned predictive world models.

Every model receives a masked observation, a blackout flag, and the action.
None receives pose. All are trained on next-observation prediction alone.
Hidden sizes are chosen so parameter counts agree to within 2%.
"""
import torch
import torch.nn as nn

from envs import OBS_DIM

IN_DIM = OBS_DIM + 1 + 2
HIDDEN = {"gru": 128, "rnn": 190, "tf": 76}


class RecurrentWM(nn.Module):
    core_cls = nn.GRU

    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.enc = nn.Linear(IN_DIM, hidden)
        self.core = self.core_cls(hidden, hidden, batch_first=True)
        self.dec = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, OBS_DIM))

    def forward(self, x, h0=None, return_h=False):
        z = torch.relu(self.enc(x))
        h0 = z.new_zeros(1, x.shape[0], self.hidden) if h0 is None \
            else h0.unsqueeze(0).contiguous()
        Hs, _ = self.core(z, h0)
        return (self.dec(Hs), Hs) if return_h else (self.dec(Hs), None)

    def step(self, x, h):
        """One timestep of the same dynamics, for perturbation analysis."""
        z = torch.relu(self.enc(x)).unsqueeze(1)
        _, hn = self.core(z, h.unsqueeze(0).contiguous())
        return hn.squeeze(0)

    def readout(self, h):
        return self.dec(h)


class GRUWM(RecurrentWM):
    kind, core_cls = "gru", nn.GRU

    def __init__(self, hidden=HIDDEN["gru"]):
        super().__init__(hidden)


class RNNWM(RecurrentWM):
    kind, core_cls = "rnn", nn.RNN

    def __init__(self, hidden=HIDDEN["rnn"]):
        super().__init__(hidden)


class TFWM(nn.Module):
    """Causal transformer. Its residual stream has no recurrent dynamics, so
    the relaxation analysis does not apply; it is included for topology and
    decoding comparisons only."""
    kind = "tf"

    def __init__(self, hidden=HIDDEN["tf"], layers=2, heads=4, maxlen=320):
        super().__init__()
        self.hidden = hidden
        self.enc = nn.Linear(IN_DIM, hidden)
        self.pos = nn.Parameter(torch.randn(1, maxlen, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2,
                                           batch_first=True, dropout=0.0,
                                           norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers)
        self.dec = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, OBS_DIM))

    def forward(self, x, h0=None, return_h=False):
        T = x.shape[1]
        z = torch.relu(self.enc(x)) + self.pos[:, :T]
        m = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        Hs = self.body(z, mask=m)
        return (self.dec(Hs), Hs) if return_h else (self.dec(Hs), None)

    def readout(self, h):
        return self.dec(h)


def build(kind, seed=0):
    torch.manual_seed(seed)
    return {"gru": GRUWM, "rnn": RNNWM, "tf": TFWM}[kind]()


def n_params(m):
    return sum(p.numel() for p in m.parameters())
