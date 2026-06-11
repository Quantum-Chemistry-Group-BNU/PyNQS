from __future__ import annotations

import torch

from abc import ABC, abstractmethod
from typing import List, Callable, Tuple, Union, Any
from torch import nn, Tensor

from pynqs.utils.public_function import spin_flip_sign, SpinProjection, torch_lexless


class SU2(nn.Module):
    def __init__(
        self,
        original: nn.Module,
        sorb: int,
    ) -> None:
        super(SU2, self).__init__()

        self.original = original
        self.sorb = sorb
        self.device = original.device
        self.dtype = original.dtype

    def forward(self, x0: Tensor) -> Tensor:
        x = x0.view((-1, x0.shape[-1]))
        y = torch.empty_like(x, device=x.device)
        y[:, 0::2] = x[:, 1::2]
        y[:, 1::2] = x[:, 0::2]
        ans = self.original(x) + spin_flip_sign(x, self.sorb) * SpinProjection.eta * self.original(y)
        if x0.dim() == 1:
            ans = ans[0]

        return ans

    @property
    def normalization(self):
        return self.original.normalization

    @normalization.setter
    def normalization(self, value):
        self.original.normalization = value

    @torch.no_grad
    def update_normalization(self, L2):
        if hasattr(self.original, "update_normalization"):
            self.original.update_normalization(L2)


class SU2BatchedForward(SU2):
    """SU2 projection using one batched call to the wrapped wavefunction."""

    def forward(self, x0: Tensor) -> Tensor:
        x = x0.view((-1, x0.shape[-1]))
        y = torch.empty_like(x, device=x.device)
        y[:, 0::2] = x[:, 1::2]
        y[:, 1::2] = x[:, 0::2]

        psi = self.original(torch.cat((x, y), dim=0)).reshape(2, -1)
        psi_x = psi[0]
        psi_y = psi[1]
        ans = psi_x + spin_flip_sign(x, self.sorb) * SpinProjection.eta * psi_y
        if x0.dim() == 1:
            ans = ans[0]

        return ans


class SU2_v3(nn.Module):
    def __init__(
        self,
        original: nn.Module,
        sorb: int,
    ) -> None:
        super(SU2_v3, self).__init__()

        self.original = original
        self.sorb = sorb

        # self.exp = torch.arange(sorb, dtype=torch.float64, device=original.device)
        # self.exp = torch.exp(self.exp)

    def forward(self, x0: Tensor) -> Tensor:
        x = x0.view((-1, x0.shape[-1]))
        y = torch.empty_like(x, device=x.device)
        y[:, 0::2] = x[:, 1::2]
        y[:, 1::2] = x[:, 0::2]

        # hash_x = (x * self.exp).sum(dim=-1)
        # hash_y = (y * self.exp).sum(dim=-1)

        sign_y = spin_flip_sign(x, self.sorb) * SpinProjection.eta
        sign_x = torch.ones_like(sign_y)

        # z = torch.where(hash_x.unsqueeze(1)>hash_y.unsqueeze(1), x, y)
        # sign_z = torch.where(hash_x>hash_y, sign_x, sign_y)

        cond = torch_lexless(x, y)
        z = torch.where(cond.unsqueeze(1), x, y)
        sign_z = torch.where(cond, sign_x, sign_y)

        ans = sign_z * self.original(z)
        if x0.dim() == 1:
            ans = ans[0]

        return ans

    @torch.no_grad
    def update_normalization(self, L2):
        if hasattr(self.original, "update_normalization"):
            self.original.update_normalization(L2)
