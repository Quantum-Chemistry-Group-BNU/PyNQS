from __future__ import annotations

import time
import torch
import numpy as np

from typing import Tuple, Callable, Union, List
from torch import Tensor
from loguru import logger

from .eloc import local_energy
from utils.distributed import (
    gather_tensor,
    get_world_size,
    synchronize,
    get_rank,
    scatter_tensor,
    all_reduce_tensor,
)
from utils.public_function import WavefunctionLUT, MemoryTrack, split_batch_idx, ansatz_batch
from libs.C_extension import onv_to_tensor


def total_energy(
    x: Tensor,
    nbatch: int,
    fp_batch: int,
    h1e: Tensor,
    h2e: Tensor,
    ansatz: Callable[..., Tensor],
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
    eps_sample: int = 0,
    use_sample_space: bool = False,
    alpha: float = 2.0,
    use_multi_psi: bool = False,
    extra_norm: Tensor = None,
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
    rank = get_rank()
    world_size = get_world_size()
    eloc = torch.zeros(dim, device=device).to(dtype)
    psi = torch.zeros_like(eloc)
    sloc = torch.zeros_like(eloc)

    # Calculate local energy in batches, better method?
    assert fp_batch > 0 or fp_batch == -1
    assert nbatch > 0 or nbatch == -1
    if nbatch == -1:
        nbatch = dim

    idx_lst = split_batch_idx(dim, min_batch=nbatch)

    def _ansatz_batch(x: Tensor, func: Callable[[Tensor], Tensor]) -> Tensor:
        return ansatz_batch(batch=fp_batch, device=device, dtype=dtype, func=func, x=x)

    if rank == 0:
        s = f"eloc: nbatch: {nbatch}, dim: {dim}, split: {len(idx_lst)}"
        s += f", Forward batch: {fp_batch}"
        logger.info(s, master=True)

    time_lst = []
    with MemoryTrack(device) as track:
        begin = 0
        for i in range(len(idx_lst)):
            end = idx_lst[i]
            ons = x[begin:end]
            # reduce-psi S-S+ is error
            _eloc, _sloc, _psi, x_time = local_energy(
                ons,
                h1e,
                h2e,
                ansatz,
                _ansatz_batch,
                sorb,
                nele,
                noa,
                nob,
                dtype=dtype,
                WF_LUT=WF_LUT,
                use_spin_raising=False if reduce_psi else use_spin_raising,
                h1e_spin=h1e_spin,
                h2e_spin=h2e_spin,
                use_unique=use_unique,
                reduce_psi=reduce_psi,
                eps=eps,
                eps_sample=eps_sample,
                use_sample_space=use_sample_space,
                index=(begin, end),
                alpha=alpha,
                use_multi_psi=use_multi_psi,
                extra_norm=extra_norm,
            )
            if reduce_psi and use_spin_raising:
                # recalculate S-S+ in Sample-space
                _sloc, _, _, _x_time = local_energy(
                    ons,
                    h1e_spin,
                    h2e_spin,
                    ansatz,
                    ansatz_batch,
                    sorb,
                    nele,
                    noa,
                    nob,
                    dtype=dtype,
                    WF_LUT=WF_LUT,
                    use_spin_raising=False,
                    use_sample_space=True,
                    index=(begin, end),
                    alpha=alpha,
                )
                x_time = list(x_time)
                for i in range(3):
                    x_time[i] += _x_time[i]
            eloc[begin:end] = _eloc
            psi[begin:end] = _psi
            sloc[begin:end] = _sloc

            time_lst.append(x_time)
            begin = end
        # track.manually_clean_cache((eloc, psi))

    # check local energy
    if torch.any(torch.isnan(eloc)):
        raise ValueError(f"The Local energy exists nan")

    if False:
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
