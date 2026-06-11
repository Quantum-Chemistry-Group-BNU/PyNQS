from __future__ import annotations
import warnings
import torch

from abc import ABC, abstractmethod
from typing import List, Callable, Tuple, Union, Any

# from typing_extensions import Any
from torch import nn, Tensor

from pynqs.ansatz.ansatz_base import AnsatzARBase


class MultiPsi(AnsatzARBase):
    """
    Psi = psi(x) * phi(x)
    """

    def __init__(
        self,
        ansatz_sample: nn.Module,
        ansatz_extra: Union[nn.Module, Callable[[Tensor], Any]],
        debug: bool = False,
    ) -> None:
        super(MultiPsi, self).__init__()

        self.sample = ansatz_sample
        self.extra = ansatz_extra
        if debug:
            self.extra.forward = self.call

        if not hasattr(self.sample, "ar_sampling"):
            warnings.warn(f"{self.sample} does not have 'ar_sampling'", UserWarning)

        self.use_multi_psi = True

    def ar_sampling(
        self,
        n_sample: int,
        min_batch: int = -1,
        min_tree_height: int = 8,
        use_dfs_sample: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return self.sample.ar_sampling(n_sample, min_batch, min_tree_height, use_dfs_sample)

    def forward(self, x: Tensor) -> Tensor:
        return self.sample(x) * self.extra(x)

    @torch.no_grad
    def call(self, x) -> Tensor:
        device = self.sample.device
        return torch.ones(x.size(0), device=device, dtype=torch.double) * 1.0
