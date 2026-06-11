from __future__ import annotations

import time
from typing import Callable
from scipy import special
import torch
import torch.distributed as dist

from dataclasses import dataclass
from loguru import logger
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import Module

from pynqs import samples_topk_config
from pynqs.libs.C_extension import unpackbits, packbits
from pynqs.utils.lut import WavefunctionLUT
from pynqs.utils.det_helper.determinant_lut import DetLUT
from pynqs.distributed import (
    all_gather_tensor,
    all_reduce_tensor,
    get_rank,
    scatter_tensor,
    processes_synchronize,
    get_world_size,
)
from pynqs.utils.public_function import (
    ansatz_batch,
    setup_seed,
    split_length_idx,
    torch_sort_onv,
)
from pynqs.ansatz import Excitedwavefunctions
from pynqs.ansatz.hybrid.alpha import AlphaPsi
from pynqs.ansatz.hybrid.multi import MultiPsi
from pynqs.sample.base import BaseSampler, ExactParams
from .comm_sample import x2string


def _log_exact_topk(ci_space: Tensor, psi_all: Tensor, sorb: int, alpha: float, norm_all: Tensor):
    top_k = samples_topk_config.topk
    prob_value_all = psi_all.abs() ** 2
    k = min(len(prob_value_all), top_k)
    if k == 0:
        return

    sample_prob_all = (prob_value_all.real ** (alpha / 2)) / norm_all
    _, topk_indices = torch.topk(prob_value_all, k=k, dim=0)

    logger.info(f"Top-{k} configuration:", master=True)
    logger.info(f"\tocc.{' '*((sorb//2)-4)}\t wf       \tprob", master=True)
    for i in range(k):
        index = topk_indices[i]
        x = unpackbits(ci_space[index], sorb)
        temp_string = x2string(x.flatten())
        temp_wf = psi_all[index]
        temp_prob = sample_prob_all[index]
        logger.info(f"{i}\t{temp_string}\t{temp_wf: .3e}\t{temp_prob:.3E}", master=True)


def build_K_ci_space(tensor_ci_space: torch.Tensor, K: int) -> torch.Tensor:
    """
    Args:
        tensor_ci_space: (dim, sorb), dim: dim of FCI space
        K: K copies
    Return:
        Tensor(dim**K, sorb*K)
    """
    if K == 1:
        return tensor_ci_space
    dim, sorb = tensor_ci_space.shape
    complex_ci_space = torch.empty(
        (dim**K, sorb * K), dtype=tensor_ci_space.dtype, device=tensor_ci_space.device
    )

    indices = torch.cartesian_prod(*([torch.arange(dim, device=tensor_ci_space.device)] * K))
    for t in range(K):
        complex_ci_space[:, t * sorb : (t + 1) * sorb] = tensor_ci_space[indices[:, t]]
    complex_ci_space = complex_ci_space.view(-1, K, sorb)
    return complex_ci_space.view(-1, sorb)


@torch.no_grad()
def construct_space_lut(
    model: Callable[[Tensor], Tensor],
    ci_space: Tensor,
    sorb: int,
    fp_batch: int,
    dtype: torch.dtype,
    sort_space: bool,
    alpha: float,
    return_rank: bool = True,
) -> tuple[WavefunctionLUT, Tensor]:
    """
    CI_space LUT
    Args:
    ----
        model: wf ansatz
        ci_space: onv like
                tensor([[ 15,   0,   0,   0,   0,   0,   0,   0],
                        [ 39,   0,   0,   0,   0,   0,   0,   0],
                        ...  ...,
                        [240,   0,   0,   0,   0,   0,   0,   0]], dtype=torch.uint8)
        sorb: sorb
        fp_batch: batch to calculate wf value
    Return:
    ------
        WF_LUT: CI-space LUT
        state_prob: Single-rank if return_rank else all-rank
    """

    rank = get_rank()
    world_size = get_world_size()
    device = ci_space.device

    dim = ci_space.size(0)
    idx_rank_lst = [0] + split_length_idx(dim, length=world_size)
    begin = idx_rank_lst[rank]
    end = idx_rank_lst[rank + 1]
    space_rank = ci_space[begin:end]
    if fp_batch == -1 or fp_batch > space_rank.size(0):
        fp_batch = space_rank.size(0)
    if rank == 0:
        s = f"Begin construct ci-space LUT, ci-space: {ci_space.shape[0]}\n"
        s += f"Rank-dim: {space_rank.size(0)}, Split-batch: {fp_batch}"
        logger.info(s, master=True)

    t0 = time.time_ns()
    psi_rank = ansatz_batch(model, space_rank, fp_batch, sorb, device, dtype)

    psi_all = all_gather_tensor(psi_rank, device)
    processes_synchronize()
    if world_size == 1:
        psi_all = psi_all[0]  # avoid memory copy
    else:
        psi_all = torch.cat(psi_all)

    WF_LUT = WavefunctionLUT(
        ci_space,
        psi_all,
        sorb,
        device,
        sort=sort_space,
    )

    norm_all = (psi_rank.norm() ** alpha).reshape(-1)
    all_reduce_tensor(norm_all)

    if rank == 0 and samples_topk_config.debug:
        _log_exact_topk(ci_space, psi_all, sorb, alpha, norm_all)

    if return_rank:
        state_prob = ((psi_rank * psi_rank.conj()).real) ** (alpha / 2) / norm_all
    else:
        state_prob = ((psi_all * psi_all.conj()).real) ** (alpha / 2) / norm_all

    t1 = time.time_ns()
    if rank == 0:
        logger.info(f"End construct, cost time: {(t1-t0)/1.0e9:.3E} s", master=True)

    del psi_all, psi_rank
    return WF_LUT, state_prob


class ExactSampler(BaseSampler):
    def __init__(
        self,
        model: DDP | Module,
        exact_params: ExactParams,
        fci_size: int,
        sorb: int,
        nele: int,
        noa: int,
        nob: int,
        use_LUT: bool,
        device: torch.device | str,
        NES_K: int = 1,
    ):
        if isinstance(model, (DDP)):
            self.use_multi_psi = isinstance(model.module, MultiPsi) or isinstance(model.module, AlphaPsi)
            model = model.module.sample if self.use_multi_psi else model.module
        else:
            ...

        self.ci_space = exact_params.ci_space.to(device)
        self.fp_batch = exact_params.fp_batch
        self.alpha = exact_params.alpha
        if self.ci_space.size(0) != fci_size:
            raise ValueError(f"Dim of FCI space is {fci_size} != {self.ci_space.size(0)}")

        # Det-LUT, remove part det in CI-NQS
        det_lut = exact_params.det_lut
        self.remove_det = False
        self.det_lut: DetLUT = None
        if det_lut is not None:
            self.remove_det = True
            self.det_lut = det_lut

        # sort FCI space
        self.sort_fci_space = False
        if not self.sort_fci_space:
            idx = torch_sort_onv(self.ci_space)
            self.ci_space = self.ci_space[idx]
            self.sort_fci_space = True

        super().__init__(
            model,
            exact_params,
            fci_size,
            sorb,
            nele,
            noa,
            nob,
            use_LUT,
            device,
            False,
            NES_K,
        )

    @torch.no_grad
    def ansatz_batch(
        self,
        x: Tensor,
        ansatz: Callable[[Tensor], Tensor],
        batch: int = -1,
    ) -> Tensor:
        return ansatz_batch(
            ansatz,
            x,
            batch,
            self.sorb,
            self.device,
            self.dtype,
        )

    @torch.no_grad
    def run(self, epoch: int, seed: int) -> tuple[Tensor, Tensor, Tensor, WavefunctionLUT]:
        if self.NES_K == 1:
            return self.ground_state_exact(epoch, seed)
        else:
            return self.excited_state_exact(epoch, seed)

    @torch.no_grad
    def excited_state_exact(self, epoch: int, seed: int):
        """
        Return
        ------
            ci_space_rank, sample_counts, sample_prob, WF_LUT
        """
        assert isinstance(self.model, Excitedwavefunctions)
        # Construct ci_space**K, calculate wf, construct WF_LUT
        if self.ci_space.shape[0] == self.fci_size:
            tensor_ci_space = unpackbits(self.ci_space, self.sorb)
            K_ci_space = build_K_ci_space(tensor_ci_space, self.NES_K)  # (K*ns,sorb)
            onv_K_ci_space = packbits(K_ci_space.to(torch.uint8), self.sorb)  # [0, 1] -> packed ONV
            self.ci_space_single = self.ci_space.clone()
            self.ci_space = onv_K_ci_space.clone()
            del onv_K_ci_space
        else:
            assert self.ci_space.size(0) == self.fci_size**self.NES_K * self.NES_K

        tensor_ci_space = unpackbits(self.ci_space, self.sorb).view(-1, self.NES_K * self.sorb)
        onv_K_ci_space = packbits(tensor_ci_space.to(torch.uint8), self.NES_K * self.sorb)

        WF_LUT, sample_prob = construct_space_lut(
            self.model,
            onv_K_ci_space,
            self.NES_K * self.sorb,
            self.fp_batch,
            self.dtype,
            sort_space=False,
            return_rank=True,
            alpha=self.alpha,
        )

        # To onv
        ci_space_rank = scatter_tensor(self.ci_space, self.device, torch.uint8)
        sample_counts = torch.tensor([float("inf")], device=self.device)
        return ci_space_rank, sample_counts, sample_prob, WF_LUT

    @torch.no_grad
    def ground_state_exact(self, epoch: int, seed: int) -> tuple[Tensor, Tensor, Tensor, WavefunctionLUT]:
        # WF_LUT, sample_prob = self.construct_FCI_lut(self.model, self.fp_batch)
        assert self.ci_space.size(0) == self.fci_size
        WF_LUT, sample_prob = construct_space_lut(
            self.model,
            self.ci_space,
            self.sorb,
            self.fp_batch,
            self.dtype,
            sort_space=False,
            return_rank=True,
            alpha=self.alpha,
        )

        if self.remove_det:
            raise NotImplementedError
            # TODO:(25-10-12 zbwu)
            # avoid wf is 0.00, if not found, set to -1
            array_idx = self.det_lut.lookup(self.ci_space, is_onv=True)[0]
            # idx = ~array_idx.gt(-1)
            # ci_space = self.ci_space[idx]
            # sample_prob = sample_prob[idx]
            mask = array_idx.gt(-1)
            ci_space = self.ci_space[~mask]
            sample_prob = sample_prob[~mask]
        else:
            ci_space = self.ci_space
        ci_space_rank = scatter_tensor(ci_space, self.device, torch.uint8)

        sample_counts = torch.tensor([float("inf")], device=self.device)
        return ci_space_rank, sample_counts, sample_prob, WF_LUT
