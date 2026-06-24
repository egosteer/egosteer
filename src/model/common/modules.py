import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, min_period: float = 0.004, max_period: float = 4.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.half_dim = dim // 2
        self.min_period = min_period
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (B, ) or (B, 1)
        if t.ndim == 2:
            t = t.squeeze(-1)
        fraction = torch.linspace(0.0, 1.0, self.half_dim, device=t.device, dtype=torch.float64)
        # period = min * (max/min)^fraction
        period = self.min_period * (self.max_period / self.min_period) ** fraction
        scaling_factor = 1.0 / period * 2 * math.pi
        emb = t[:, None] * scaling_factor[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        emb = emb.to(t.dtype)
        return emb


class TimeEncoder(nn.Module):
    """Matching pi0.5 appendix"""

    def __init__(self, time_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(time_dim, time_dim)
        self.linear_2 = nn.Linear(time_dim, time_dim)
        self.nonlinearity = nn.SiLU()

    def forward(
        self,
        time: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # [Batch_Size, Time_Dim]
        emb = self.nonlinearity(self.linear_1(time))
        emb = self.nonlinearity(self.linear_2(emb))
        return emb


class AdaLNZero(nn.Module):
    def __init__(self, dim: int, dim_cond: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.modulation = nn.Linear(dim_cond, dim * 3)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def _norm(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.FloatTensor, cond: torch.FloatTensor) -> torch.FloatTensor:
        output = self._norm(x)
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        scale, shift, gate = self.modulation(cond).chunk(3, dim=-1)
        return output * (1.0 + scale) + shift, gate


class TimeEmbedding(nn.Module):
    """Wrapper for SinusoidalPosEmb + TimeEncoder to support Hydra _target_ instantiation."""

    def __init__(self, time_hidden_size: int, min_period: float = 0.004, max_period: float = 4.0):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalPosEmb(time_hidden_size, min_period=min_period, max_period=max_period),
            TimeEncoder(time_hidden_size),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x)
