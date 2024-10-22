from __future__ import annotations

import time
import torch
import numpy as np

from typing import Tuple, Callable, Union, List, Optional
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
    state_prob: Optional[Tensor] = None,
    exact: bool = False,
    WF_LUT: Optional[WavefunctionLUT] = None,
    use_unique: bool = True,
    dtype=torch.double,
    use_spin_raising: bool = False,
    h1e_spin: Optional[Tensor] = None,
    h2e_spin: Optional[Tensor] = None,
    reduce_psi: bool = False,
    eps: float = 1.0e-12,
    eps_sample: int = 0,
    use_sample_space: bool = False,
    alpha: float = 2.0,
    use_multi_psi: bool = False,
    use_spin_flip: bool = False,
    extra_norm: Optional[Tensor] = None,
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
        return ansatz_batch(func, x, fp_batch, sorb, device, dtype)

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
                use_spin_flip=use_spin_flip,
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

    placeholders = torch.zeros(1, device=device, dtype=dtype)
    return eloc, sloc, placeholders
