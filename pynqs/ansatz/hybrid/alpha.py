from __future__ import annotations

import torch

from abc import ABC, abstractmethod
from typing import List, Callable, Tuple, Union, Any
from torch import nn, Tensor

from pynqs.ansatz.ansatz_base import AnsatzARBase


class AlphaPsi(nn.Module):
    """
    Psi = psi(x)**(1-alpha) * psi(x)**alpha
    """

    def __init__(
        self,
        ansatz_origin: nn.Module,
        alpha: float = 0.0,
    ) -> None:
        super(AlphaPsi, self).__init__()

        self.sample = psi_pow(ansatz_origin, 1 - alpha, True)
        self.extra = psi_pow(ansatz_origin, alpha, False)
        self.use_multi_psi = True

    def forward(self, x: Tensor) -> Tensor:
        return self.sample(x) * self.extra(x)


class psi_pow(nn.Module):
    def __init__(
        self,
        psi: nn.Module,
        alpha: float = 0.0,
        keep_phase: bool = True,
    ) -> None:
        super(psi_pow, self).__init__()

        self.psi = psi
        self.alpha = alpha
        self.keep_phase = keep_phase

    def forward(self, x):
        if self.keep_phase:
            psi = self.psi(x)
            return psi / psi.abs() ** (1 - self.alpha)
        else:
            return self.psi(x).abs() ** self.alpha
