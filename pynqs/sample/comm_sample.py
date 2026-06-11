"""
Distributed Communication about the sample
"""

import time
import torch
from collections.abc import Callable

from loguru import logger
from torch import Tensor

from pynqs.ansatz.hybrid.excited import nes_to_nqs, nqs_to_nes
from pynqs.libs.C_extension import merge_rank_sample, packbits, unpackbits
from pynqs.utils.lut import WavefunctionLUT
from pynqs.distributed import (
    broadcast_tensor,
    gather_tensor,
    get_rank,
    get_world_size,
    scatter_tensor,
    processes_synchronize,
    all_reduce_tensor,
)
from pynqs.utils.public_function import split_length_idx, torch_unique_index
from pynqs import samples_topk_config


def x2string(x0):
    x = x0.to(torch.int8)
    ans = ""
    for i in range(len(x) // 2):
        j, k = x[2 * i], x[2 * i + 1]
        if j == 1 and k == 0:
            ans += "a"
        elif j == 0 and k == 1:
            ans += "b"
        elif j == 1 and k == 1:
            ans += "2"
        elif j == 0 and k == 0:
            ans += "0"
    return ans[::-1]


def gather_scatter_sample(
    sorb: int,
    unique: torch.Tensor,
    counts: Tensor,
    wf_value: Tensor,
    use_LUT: bool,
    use_same_tree: bool,
    device: str,
    dtype: torch.dtype,
    compress: bool = True,
    alpha: float = 2.0,
    gamma: float | None = None,
    reweight_func: Callable[[Tensor], Tensor] = None,
    NES_K: int = 1,
) -> tuple[Tensor, Tensor, Tensor, WavefunctionLUT]:
    """
    1. Gather sample-unique/counts from every rank, compress uint64 to uint8
    2. Merge all unique/counts in master-rank(rank0)
    3. Scatter unique/counts/prob to every rank

    Meanwhile, All-Gather sample-unique and wf-value in order to make wf-lookup-table

    Returns
    -------
        unique_rank: Tensor (uint8, onv)
        counts_rank: Tensor (the counts of the rank sample)
        prob_rank: Tensor, prob_rank = prob * world_size
        WF_LUT: wavefunction LookUP-Table about all-sample-unique and wf-value
    """

    rank = get_rank()
    world_size = get_world_size()
    need_wf_values = use_LUT or abs(alpha - 2.0) > 1.0e-12
    t0 = time.time_ns()
    # Gather unique, counts, wf_value
    if compress:
        unique = packbits(unique.byte(), sorb)  # compress 0/1 states -> packed ONV
    else:
        assert unique.dtype == torch.uint8
    unique_all = gather_tensor(unique, device)
    count_all = gather_tensor(counts, device)
    wf_value_all: Tensor | None = None
    if need_wf_values:
        wf_value_all = gather_tensor(wf_value, device)
        if rank == 0:
            wf_value_all = torch.cat(wf_value_all)
    processes_synchronize()

    # check unique and counts
    if rank == 0:
        unique_length = [i.size(0) for i in unique_all]
        count_length = [i.sum().item() for i in count_all]
        s = f"All-rank samples {sum(unique_length)//NES_K}, counts: {sum(count_length)}"
        logger.info(s, master=True)

    t1 = time.time_ns()
    if rank == 0:
        split_idx = torch.tensor([0] + [i.shape[0] // NES_K for i in unique_all])
        unique_all = torch.cat(unique_all)
        count_all = torch.cat(count_all)
        if not use_same_tree:
            # every-rank sample part is the different, so use 'torch.unique'
            split_idx = split_idx.long().to(device).cumsum_(dim=0)
            unique_all = nqs_to_nes(unique_all, NES_K)  # (K*ns,sorb) -> (ns,K,sorb)
            merge_unique, merge_inv, merge_idx = torch_unique_index(unique_all, dim=0)[:3]
            length = merge_unique.shape[0]  # ns
            merge_unique = nes_to_nqs(merge_unique, NES_K)  # (ns,K,sorb) -> (K*ns,sorb)
            if need_wf_values:
                wf_value_unique = wf_value_all[merge_idx]

            # merge prob
            merge_counts = merge_rank_sample(merge_inv, count_all, split_idx, length)
            merge_prob = merge_counts / merge_counts.sum()
            # _, counts_test = torch.unique(unique_all, dim=0, return_counts=True)
            # assert(torch.allclose(counts_test, merge_counts))
        else:
            # every-rank sample is unique
            # merge_counts: Tensor = None
            merge_counts = count_all
            merge_prob = count_all / count_all.sum()
            merge_unique = unique_all
            if need_wf_values:
                wf_value_unique = wf_value_all
    else:
        merge_counts: Tensor = None
        merge_unique: Tensor = None
        merge_prob: Tensor = None
        wf_value_unique: Tensor = None

    t2 = time.time_ns()

    debug = samples_topk_config.debug
    if rank == 0 and debug:
        if not use_LUT:
            raise ValueError(f"use LUT when inputting top-k samples")
        prob_value_unique = wf_value_unique.abs() ** 2
        top_k = samples_topk_config.topk
        k = k = min(len(prob_value_unique), top_k)
        n_all = torch.sum(count_all)
        _, topk_indices = torch.topk(prob_value_unique, k=k, dim=0)
        logger.info(f"Top-{k} configuration:", master=True)
        logger.info(f"\tocc.{' '*((sorb//2)-4)}\t wf       \tcount\tfreq", master=True)
        for i in range(k):
            index = topk_indices[i]
            x = unpackbits(merge_unique[index], sorb)
            temp_string = x2string(x.flatten())
            temp_wf = wf_value_unique[index]
            temp_count = merge_counts[index]
            freq = (temp_count / n_all).item()
            logger.info(f"{i}\t{temp_string}\t{temp_wf: .3e}\t{temp_count}\t{freq:.3E}", master=True)

    # Scatter unique, counts
    if merge_unique is not None:
        # (ns*K,bra_len) -> (ns,K*bra_len)
        merge_unique = merge_unique.reshape(-1, NES_K * merge_unique.shape[-1])
    unique_rank = scatter_tensor(merge_unique, device, torch.uint8)
    unique_rank = unique_rank.reshape(NES_K * unique_rank.shape[0], -1)
    counts_rank = scatter_tensor(merge_counts, device, torch.int64)
    prob_rank = scatter_tensor(merge_prob, device, dtype.to_real())

    t3 = time.time_ns()
    if need_wf_values:
        # XXX: unique_rank split merge_unique when broadcast to all-rank,
        # this maybe efficiency than scatter->broadcast
        merge_unique = broadcast_tensor(merge_unique, device, torch.uint8)
        wf_value_unique = broadcast_tensor(wf_value_unique, device, dtype)
    processes_synchronize()
    t4 = time.time_ns()

    # Testing prob
    use_subspace = False
    if use_subspace:
        if not use_LUT:
            raise NotImplementedError
        else:
            prob1 = wf_value_unique.abs() ** 2 / wf_value_unique.norm() ** 2
            dim = merge_unique.size(0)
            idx_lst = [0] + split_length_idx(dim, world_size)
            begin_rank = idx_lst[rank]
            end_rank = idx_lst[rank + 1]
            prob_rank = prob1[begin_rank:end_rank]

    if rank == 0:
        delta1 = (t1 - t0) / 1.0e09
        delta2 = (t3 - t2) / 1.0e09
        delta3 = (t2 - t1) / 1.0e09
        delta4 = (t4 - t3) / 1.0e09
        s = f"Sample-Comm, Gather: {delta1:.3E} s, Scatter: {delta2:.3E} s, merge: {delta3:.3E} s\n"
        s += f"All-Rank unique sample: {merge_unique.size(0)}, Broadcast LUT: {delta4:.3E} s"
        logger.info(s, master=True)

    if use_LUT:
        dim = merge_unique.size(0)
        bra_key = merge_unique
        # bra_key = packbits(merge_unique, sorb)
        WF_LUT = WavefunctionLUT(
            bra_key,
            wf_value_unique,
            sorb,
            device,
        )
        idx_lst = [0] + split_length_idx(dim, world_size)
        begin_rank = idx_lst[rank]
        end_rank = idx_lst[rank + 1]
    else:
        WF_LUT: WavefunctionLUT = None

    if need_wf_values:
        dim = merge_unique.size(0)
        idx_lst = [0] + split_length_idx(dim, world_size)
        begin_rank = idx_lst[rank]
        end_rank = idx_lst[rank + 1]
        wf_unique_rank = wf_value_unique[begin_rank:end_rank]
    else:
        wf_unique_rank = None

    if alpha - 2.0 != 0.0:
        prob_rank = change_sample_prob(
            sorb,
            unique_rank,
            prob_rank,
            wf_lut=WF_LUT,
            wf_unique=wf_unique_rank,
            alpha=alpha,
            reweight_func=reweight_func,
            gamma=gamma,
            NES_K=NES_K,
        )

    del merge_counts, merge_prob, unique_all, count_all, wf_value_all

    # placeholders = torch.ones([], device=device, dtype=torch.int64)
    return (unique_rank, counts_rank, prob_rank, WF_LUT)


def change_sample_prob(
    sorb: int,
    sample_unique: Tensor,
    sample_prob: Tensor,
    wf_lut: WavefunctionLUT | None = None,
    wf_unique: Tensor | None = None,
    alpha: float = 2.0,
    reweight_func: Callable[[Tensor], Tensor] = None,
    gamma: float | None = None,
    NES_K: int = 1,
) -> Tensor:
    if wf_unique is None:
        if wf_lut is None:
            raise NotImplementedError(f"Need wf values or LUT when alpha != 2.0 in MCMC")
        _, __, wf_unique = wf_lut.lookup(sample_unique)

    if reweight_func is not None:
        x_unique = unpackbits(sample_unique, sorb)
        if NES_K > 1:
            x_unique = nqs_to_nes(x_unique, NES_K)
        reweight_unique = reweight_func(x_unique, wf_unique)
        sample_prob /= reweight_unique
        if get_rank() == 0:
            logger.info(f"Reweight sample frequency with {reweight_func}", master=True)

    if gamma is None:
        gamma = 2.0 - alpha
    if gamma != 0.0:
        sample_prob *= torch.abs(wf_unique) ** (gamma)
        if get_rank() == 0:
            logger.info(f"Reweight sample frequency with gamma={gamma}", master=True)

    prob_sum = torch.sum(sample_prob)
    all_reduce_tensor(prob_sum)
    sample_prob /= prob_sum
    return sample_prob
