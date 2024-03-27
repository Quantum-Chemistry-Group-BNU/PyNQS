from __future__ import annotations

from typing import Callable

import torch
import numpy as np

from torch import Tensor
from optree.integration.torch import tree_ravel

from utils.distributed import (
    all_to_all_tensor,
    all_reduce_tensor,
    get_rank,
    get_world_size,
    scatter_tensor,
    gather_tensor,
)

# https://github.com/netket/netket/blob/master/netket/experimental/driver/vmc_srt.py multi-GPU
# https://github.com/HannahLange/Fermionic-RNNs/blob/main/src/stoch_reconfig.py single-GPU
def SRt(
    O_L: tuple[Tensor, Tensor | None],
    eloc: Tensor,
    eloc_mean: Tensor,
    prob: Tensor,
    diag_shift: float,
    solver_fn: Callable[[Tensor, Tensor], Tensor],
    mode: str = "complex",
) -> Tensor:
    """
    min-SR: see: https://arxiv.org/abs/2310.05715
    O_L: (n_sample, n_params) (Single-Rank) Real, Imag
    eloc: (n_sample) (Single-Rank)
    eloc_mean: (1) (All-Rank)
    prob: (n-sample) (Single-Rank)
    diag_shift: I_{2M} * lambda
    solver_fn: torch.linalg.solve or other Linear-operator
    """

    world_size = get_world_size()
    rank = get_rank()
    device = eloc.device
    dtype = eloc.dtype
    n_params = O_L.shape[-1]
    n_sample = O_L.shape[0]

    corr = (eloc - eloc_mean).conj()
    dv = -2 * (corr * prob.sqrt()).conj()

    Ore = O_L[0]  # (n-sample, n-param)
    Ore_mean = torch.matmul(Ore, prob) * world_size
    all_reduce_tensor(Ore_mean, world_size=world_size, in_place=True)
    Yre = (Ore - Oim_mean) * prob.sqrt()  # (n-sample, n-param)

    if mode == "complex":
        Oim = O_L[1]
        Oim_mean = torch.matmul(Oim, prob) * world_size
        all_reduce_tensor(Oim_mean, world_size=world_size, in_place=True)
        Yim = (Oim - Oim_mean) * prob.sqrt()
        X = torch.cat([Yre, Yim], dim=0)  # (2 * n-sample, n-param)
        dv = torch.cat([dv.real, -dv.imag], dim=-1)  # (2 * n-sample)
    elif mode == "real":
        X = Yre
        dv = dv.real
    else:
        raise NotImplementedError()

    # (n-sample-rank, n-param) -> (n-param-rank, n-sample-all)
    X = torch.cat(all_to_all_tensor(X, world_size=world_size))

    # all-reduce-sum X * X.T -> (n-sample-all, n-sample-all)
    matrix = torch.matmul(X, X.T) * world_size
    all_reduce_tensor(matrix, world_size=world_size, in_place=True)

    dv_all = gather_tensor(dv, device, world_size)  # (n-sample-all)
    if rank == 0:
        matrix_side = matrix.shape[-1]  # n-sample-all / n-sample-all * 2
        matrix = matrix + diag_shift * torch.eye(
            matrix_side, dtype=dtype, device=device
        )  # add lambda * I
        # solve (X* X.T)^{-1} dv
        vector = solver_fn(matrix, torch.cat(dv_all, dim=-1))  # (n-sample-all)
    else:
        vector: Tensor = None
    vector = scatter_tensor(vector, device, dtype, world_size)  # (n-sample-rank)
    updates = X.T @ vector  #  (n-param, n-sample-rank) * (n-sample-rank)

    assert updates.size(0) == n_params

    # single-rank
    return -updates
