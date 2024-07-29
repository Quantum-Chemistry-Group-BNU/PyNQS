from __future__ import annotations

import torch

from abc import ABC, abstractmethod
from typing import List, Callable, Tuple

from torch import nn, Tensor


class AnsatzARBase(nn.Module):
    """
    Base class for autoregressive neural networks.
    and implement ar_sampling
    """

    def __init__(self) -> None:
        super(AnsatzARBase, self).__init__()

    @abstractmethod
    def ar_sampling(
        self,
        n_sample: int,
        min_batch: int = -1,
        min_tree_height: int = 8,
        use_dfs_sample: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        ar sample

        Returns:
        --------
            sample_unique: the unique of sample, s.t 0: unoccupied 1: occupied
            sample_counts: the counts of unique sample, s.t. sum(sample_counts) = n_sample
            wf_value: the wavefunction of unique sample
        """
        raise NotImplementedError