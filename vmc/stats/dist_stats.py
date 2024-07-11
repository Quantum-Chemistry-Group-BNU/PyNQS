"""
torch.distributed stats 'mean', 'var'
"""
from __future__ import annotations

import torch

from torch import Tensor
from typing import Tuple

from utils.distributed import (
    all_gather_tensor,
    all_reduce_tensor,
    gather_tensor,
    scatter_tensor,
    get_rank,
    get_world_size,
    synchronize,
    broadcast_tensor,
)


def dist_mean(
    x: Tensor,
    prob: Tensor = None,
    world_size: int = 1,
) -> Tensor:
    """
    Compute tensors means using 'torch.distributed'
    """
    assert x.dim() == 1
    assert prob.dim() == 1
    if prob is None:
        x_mean = x.mean()
    else:
        if torch.is_complex(x):
            real = torch.dot(x.real, prob)
            imag = torch.dot(x.imag, prob)
            x_mean = torch.complex(real, imag)
        else:
            x_mean = torch.dot(x, prob)
    all_reduce_tensor(x_mean, world_size=world_size)

    return x_mean


def dist_var(
    x: Tensor,
    prob: Tensor = None,
    world_size: int = 1,
) -> Tuple[Tensor, Tensor]:
    """
    Compute tensors variance using 'torch.distributed'
    """
    x_mean = dist_mean(x, prob, world_size)
    corr = x_mean - x

    x_var = ((corr * corr.conj() * prob).sum())
    all_reduce_tensor(x_var, world_size=world_size)

    return x_mean, x_var


def dist_stats(
    x: Tensor,
    prob: Tensor = None,
    counts: int = None,
    world_size: int = 1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    return 'mean', 'var', 'sd' , 'se'
    """
    x_mean, x_var = dist_var(x, prob, world_size)
    x_sd = torch.sqrt(x_var)

    if counts is None:
        # counts = float("inf")
        N = torch.tensor(x.size(0), dtype=torch.int64, device=x.device)
        N_all = all_gather_tensor(N, device=x.device, world_size=world_size)
        synchronize()
        counts = torch.cat(N_all).item()
    x_se = x_sd / counts**0.5

    return x_mean, x_var, x_sd, x_se
