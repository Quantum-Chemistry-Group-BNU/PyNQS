from __future__ import annotations

import time
import torch
import numpy as np

from typing import Tuple, Callable, Union, List
from torch import Tensor
from loguru import logger

from .eloc import local_energy
from utils.distributed import gather_tensor, get_world_size, synchronize, get_rank, scatter_tensor
from utils.public_function import WavefunctionLUT, MemoryTrack


def total_energy(
    x: Tensor,
    nbatch: int,
    h1e: Tensor,
    h2e: Tensor,
    ansatz: Callable,
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    state_prob: Tensor = None,
    exact: bool = False,
    WF_LUT: WavefunctionLUT = None,
    use_unique: bool = True,
    dtype=torch.double,
    use_spin_raising: bool = False,
    h1e_spin: Tensor = None,
    h2e_spin: Tensor = None,
    reduce_psi: bool = False,
    eps: float = 1.0e-12,
    use_sample_space: bool = False,
    alpha: float = 2.0,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""

    Calculate total-energy, <S-S+>, local-energy and state-prob

    Return
    ------
        eloc: local energy (Single-Rank)
        sloc : spin-raising<S-S+> (Single-Rank)
        state_prob: if exact: the state-prob would be calculated again,
                    else zeros-tensor. (Single-Rank)
    """
    t0 = time.time_ns()
    dim: int = x.shape[0]
    device = x.device
    eloc = torch.zeros(dim, device=device).to(dtype)
    psi = torch.zeros_like(eloc)
    sloc = torch.zeros_like(eloc)

    time_lst = []
    idx_lst = torch.empty(int(np.ceil(dim / nbatch)), dtype=torch.int64).fill_(nbatch)
    idx_lst[-1] = dim - (idx_lst.size(0) - 1) * nbatch
    idx_lst: List[int] = idx_lst.cumsum(dim=0).tolist()
    rank = get_rank()

    # Calculate local energy in batches, better method?
    if rank == 0:
        s = f"nbatch: {nbatch}, dim: {dim}, split: {len(idx_lst)}"
        logger.info(s, master=True)

    with MemoryTrack(device) as track:
        begin = 0
        for i in range(len(idx_lst)):
            end = idx_lst[i]
            ons = x[begin:end]
            _eloc, _sloc, _psi, x_time = local_energy(
                ons,
                h1e,
                h2e,
                ansatz,
                sorb,
                nele,
                noa,
                nob,
                dtype=dtype,
                WF_LUT=WF_LUT,
                use_spin_raising=use_spin_raising,
                h1e_spin=h1e_spin,
                h2e_spin=h2e_spin,
                use_unique=use_unique,
                reduce_psi=reduce_psi,
                eps=eps,
                use_sample_space=use_sample_space,
                index=(begin, end),
                alpha=alpha,
            )
            eloc[begin:end] = _eloc
            psi[begin:end] = _psi
            sloc[begin:end] = _sloc

            time_lst.append(x_time)
            begin = end
        # track.manually_clean_cache((eloc, psi))

    # check local energy
    if torch.any(torch.isnan(eloc)):
        raise ValueError(f"The Local energy exists nan")

    if exact:
        t_exact0 = time.time_ns()
        world_size = get_world_size()
        # gather psi_lst from all rank
        psi_all = gather_tensor(psi, device, world_size, master_rank=0)
        # eloc_all = gather_tensor(eloc, device, world_size, master_rank=0)
        synchronize()
        t_exact1 = time.time_ns()
        if rank == 0:
            psi_all = torch.cat(psi_all)
            # eloc_all = torch.cat(eloc_all)
            state_prob_all = (psi_all * psi_all.conj()).real / psi_all.norm() ** 2
            state_prob_all = state_prob_all.to(dtype)
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
        del psi_all, state_prob_all
    else:
        if state_prob is None:
            state_prob = torch.ones(dim, dtype=dtype, device=device) / dim

    # eloc_mean = torch.einsum("i, i ->", eloc, state_prob)
    # e_total = eloc_mean + ecore

    # if use_spin_raising:
    #     spin_mean = torch.dot(loc_spin, state_prob).real.item()
    # else:
    #     spin_mean = 0.0

    t1 = time.time_ns()
    time_lst = np.stack(time_lst, axis=0)
    delta0 = time_lst[:, 0].sum()
    delta1 = time_lst[:, 1].sum()
    delta2 = time_lst[:, 2].sum()
    logger.info(
        f"Total energy cost time: {(t1-t0)/1.0E06:.3E} ms, "
        + f"Detail time: {delta0:.3E} ms {delta1:.3E} ms {delta2:.3E} ms"
    )

    del psi, idx_lst
    if x.is_cuda:
        torch.cuda.empty_cache()

    if exact:
        return eloc, sloc, state_prob.real
    else:
        placeholders = torch.zeros(1, device=device, dtype=dtype)
        return eloc, sloc, placeholders
