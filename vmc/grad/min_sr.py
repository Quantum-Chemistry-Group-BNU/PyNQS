from typing import Callable

import torch
import numpy as np

from torch import Tensor
from optree.integration.torch import tree_ravel

from utils.distributed import all_to_all_tensor, all_reduce_tensor, get_rank, get_world_size
from utils.public_function import split_batch_idx


def SRt(
    O_L: tuple[Tensor, Tensor],
    eloc: Tensor,
    eloc_mean: Tensor,
    prob: Tensor,
    diag_shift: float,
    solver_fn: Callable[[Tensor], Tensor],
):
    """
    min-SR: see: https://arxiv.org/abs/2310.05715

    O_L: (n_sample, n_params) (Single-Rank) Real, Imag
    eloc: (n_sample) (Single-Rank)
    eloc_mean: (1) (All-Rank)
    prob: (n-sample) (Single-Rank)
    
    """

    world_size = get_world_size()
    rank = get_rank()
    n_params = O_L.shape[-1]
    n_sample = O_L.shape[0]

    corr = (eloc - eloc_mean).conj()
    epsilon = -2 * (corr * prob.sqrt()).conj()

    Ore = O_L[0] # (n-sample, n-param)
    Ore_mean = torch.matmul(Ore, prob) * world_size
    all_reduce_tensor(Ore_mean, world_size=world_size, in_place=True)
    Oim = O_L[1]
    Oim_mean = torch.matmul(Oim, prob) * world_size
    all_reduce_tensor(Oim_mean, world_size=world_size, in_place=True)
    
    Yre = (Ore - Oim_mean) * prob.sqrt()
    Yim = (Oim - Oim_mean) * prob.sqrt()
    X = torch.complex(Yre, Yim)  # (n-sample-rank, n-param)
    
    size = (n_params - 1) // world_size + 1
    if rank == world_size:
        size = n_params - world_size * size

    # (n-param-rank, n-sample-all) # TODO: check
    X = torch.cat(all_to_all_tensor(X.flatten(), world_size=world_size)).reshape(size, -1)

    # all-reduce-sum X * X.T
    matrix = torch.matmul(X, X.T) * world_size
    all_reduce_tensor(matrix, world_size=world_size, in_place=True)
    
    if rank == 0:
        matrix_side = matrix.shape[-1] # Ns
        matrix = matrix + diag_shift * torch.eye(matrix_side)  # add lambda * I
        vector = solver_fn(matrix, epsilon)
        vector = vector.reshape()