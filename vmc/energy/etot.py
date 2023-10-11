import time
import torch
import numpy as np

from typing import Tuple, Callable, Union
from torch import Tensor
from loguru import logger

from .eloc import local_energy
from utils.distributed import gather_tensor, get_world_size, synchronize, get_rank, scatter_tensor


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
) -> Tuple[Union[complex, float], Tensor, Tensor, dict]:
    r"""
    
    Return
    ------
        e_total: Total energy
        eloc_lst: local energy
        state_prob: if exact: the state-prob would be calculated again, 
                    else zeros-tensor.
        statistics: ... 
    """
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
        y = torch.zeros(0, ons.shape[1], dtype=torch.uint8, device=ons.device)
        time_lst.append(x_time)

    # check local energy
    if torch.any(torch.isnan(eloc_lst)):
        raise ValueError(f"The Local energy exists nan")

    if exact:
        t_exact0 = time.time_ns()
        world_size = get_world_size()
        rank = get_rank()
        # gather psi_lst from all rank
        psi_lst_all = gather_tensor(psi_lst, device, world_size, master_rank=0)
        eloc_lst_all = gather_tensor(eloc_lst, device, world_size, master_rank=0)
        synchronize()
        t_exact1 = time.time_ns()
        if rank == 0:
            psi_lst_all = torch.cat(psi_lst_all)
            eloc_lst_all = torch.cat(eloc_lst_all)
            state_prob_all = (psi_lst_all * psi_lst_all.conj()).real / psi_lst_all.norm() ** 2
            state_prob_all =state_prob_all.to(dtype)
            eloc_mean = torch.einsum("i, i ->", eloc_lst_all, state_prob_all)
        else:
            state_prob_all = None
        # Scatter state_prob to very rank
        t_exact2 = time.time_ns()
        state_prob = scatter_tensor(state_prob_all, device, dtype, world_size, master_rank=0)
        state_prob *= world_size
        synchronize()
        t_exact3 = time.time_ns()

        # logger
        if rank == 0:
            delta_all = (t_exact3 - t_exact0) / 1.0e09
            delta_gather = (t_exact1 - t_exact0) / 1.0e09
            delta_scatter = (t_exact3 - t_exact2) / 1.0e09
            delta_cal = (t_exact2 - t_exact1) / 1.0e09
            s = f"Exact-prob: {delta_all:.3E} s, Calculate: {delta_cal:.3E} s, "
            s += f"Gather: {delta_gather:.3E} s, Scatter: {delta_scatter:.3E} s"
            logger.info(s, master=True)

        # assure length is true.
        assert state_prob.shape[0] == dim
        del psi_lst_all, state_prob_all
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
    logger.info(
        f"Total energy cost time: {(t1-t0)/1.0E06:.3E} ms, "
        + f"Detail time: {delta0:.3E} ms {delta1:.3E} ms {delta2:.3E} ms"
    )

    del psi_lst, idx_lst
    if x.is_cuda:
        torch.cuda.empty_cache()

    if exact:
        return e_total.item(), eloc_lst, state_prob, statistics
    else:
        placeholders = torch.zeros(1, device=device, dtype=dtype)
        return e_total.item(), placeholders, eloc_lst, statistics
