import time
import torch
import torch.distributed as dist
import numpy as np

from typing import Tuple, Callable, Union
from torch import Tensor
from loguru import logger

from .eloc import local_energy


def total_energy(
    x: Tensor,
    nbatch: int,
    h1e: Tensor,
    h2e: Tensor,
    ansatz: Callable,
    ecore: float,
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    state_prob: Tensor = None,
    state_counts: Tensor = None,
    exact: bool = False,
    dtype=torch.double,
) -> Tuple[Union[complex, float], Tensor, dict]:
    dim: int = x.shape[0]
    device = x.device
    eloc_lst = torch.zeros(dim, device=device).to(dtype)
    psi_lst = torch.zeros_like(eloc_lst)
    idx_lst = torch.arange(dim).to(device)
    time_lst = []
    statistics = {}

    # calculate the total energy using splits
    t0 = time.time_ns()
    # ons_dataset = Data.TensorDataset(x, idx_lst)
    # loader = Data.DataLoader(dataset=ons_dataset, batch_size=nbatch,
    #                           shuffle=False, drop_last=False)

    # for step, (ons, idx) in enumerate(loader):
    # for ons, idx in loader: # why is slower than using split?
    for ons, idx in zip(x.split(nbatch), idx_lst.split(nbatch)):
        eloc_lst[idx], psi_lst[idx], x_time = local_energy(
            ons, h1e, h2e, ansatz, sorb, nele, noa, nob, dtype=dtype
        )
        time_lst.append(x_time)

    # check local energy
    if torch.any(torch.isnan(eloc_lst)):
        raise ValueError(f"The Local energy exists nan")

    if exact:
        state_prob = (psi_lst * psi_lst.conj()) / psi_lst.norm() ** 2
    else:
        if state_prob is None:
            state_prob = torch.ones(dim, dtype=dtype, device=device) / dim

    state_prob = state_prob.to(dtype)
    eloc_mean = torch.einsum("i, i ->", eloc_lst, state_prob)
    e_total = eloc_mean + ecore
    if not exact:
        if state_counts is None:
            state_counts = torch.ones(dim, dtype=dtype, device=device)
        n_sample = state_counts.sum()
        variance = torch.sum((eloc_lst - eloc_mean) ** 2 * state_counts) / (n_sample - 1)
        sd = torch.sqrt(variance)
        se = sd / torch.sqrt(n_sample)
        statistics["mean"] = e_total.item()
        statistics["var"] = variance.item()
        statistics["SD"] = sd.item()
        statistics["SE"] = se.item()

    t1 = time.time_ns()
    time_lst = np.stack(time_lst, axis=0)
    delta0 = time_lst[:, 0].sum()
    delta1 = time_lst[:, 1].sum()
    delta2 = time_lst[:, 2].sum()
    logger.debug(
        f"Total energy cost time: {(t1-t0)/1.0E06:.3E} ms, "
        + f"Detail time: {delta0:.3E} ms {delta1:.3E} ms {delta2:.3E} ms"
    )
    del psi_lst, idx_lst
    return e_total.item(), eloc_lst, statistics
