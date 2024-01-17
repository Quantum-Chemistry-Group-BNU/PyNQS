from __future__ import annotations

import time
import os
import torch
import torch.distributed as dist
import numpy as np

from dataclasses import dataclass
from typing import Union, Tuple, List
from loguru import logger
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from numpy import ndarray

from vmc.optim import BaseVMCOptimizer
from utils.public_function import (
    split_length_idx,
    split_batch_idx,
    MemoryTrack,
)
from utils.determinant_lut import DetLUT
from utils.distributed import all_reduce_tensor, synchronize, all_gather_tensor
from libs.C_extension import get_hij_torch, onv_to_tensor, tensor_to_onv, get_comb_tensor
from ci import CIWavefunction


@dataclass
class NqsCi(BaseVMCOptimizer):
    """
    CI-NQS WaveFunction
    """

    ci_det: Tensor = None
    """CI determinant, uint8"""

    ci_num: int = None
    """Number of CI determinant, m"""

    det_lut: DetLUT = None
    """LookUp-Table CI-det"""

    Ham_matrix: Tensor = None
    """the Hamiltonian matrix, (m +1, m + 1)"""

    total_coeff: Tensor = None
    """total coeff, (m + 1)"""

    nqs_coeff: Tensor = None
    """the NQS coeff, (1)"""

    ci_coeff: Tensor = None
    """CI-det coeff, (m)"""

    def __init__(
        self,
        CI: CIWavefunction,
        cNqs_pow_min: float = 1.0e-4,
        start_iter: int = -1,
        **vmc_opt_kwargs: dict,
    ) -> None:
        # Remove pre-train info
        vmc_opt_kwargs.pop("pre_CI", None)
        vmc_opt_kwargs.pop("pre_train_info", None)
        vmc_opt_kwargs.pop("noise_lambda", None)
        super(NqsCi, self).__init__(**vmc_opt_kwargs)

        self.ci_det = CI.space
        self.ci_num = CI.space.size(0)
        self.factory_kwargs = {"device": self.device, "dtype": self.dtype}
        self.det_lut = self.sampler.det_lut
        dim = self.ci_num + 1
        self.Ham_matrix = torch.zeros((dim, dim), **self.factory_kwargs)
        self.make_ci_hij()

        # init total-coeff
        self.total_coeff = torch.ones(dim, dtype=self.dtype).to(self.device)
        # self.total_coeff[: self.ci_num] = CI.coeff.to(self.device)
        # normalization
        self.total_coeff /= self.total_coeff.norm()
        self.nqs_coeff = self.total_coeff[-1]
        self.ci_coeff = self.total_coeff[: self.ci_num]
        self.coeff_lst = []

        # split different rank
        assert self.ci_num > self.world_size
        idx_lst = [0] + split_length_idx(dim=self.ci_num, length=self.world_size)
        begin = idx_lst[self.rank]
        end = idx_lst[self.rank + 1]
        self.rank_ci_det = self.ci_det[begin:end]
        self.rank_ci_num = self.rank_ci_det.size(0)
        self.rank_ci_coeff = self.ci_coeff[begin:end]
        self.rank_idx_lst = idx_lst

        # min cNqs^2
        assert cNqs_pow_min > 0.0 and cNqs_pow_min <= 1.0
        self.cNqs_pow_min = cNqs_pow_min
        # change cNqs^2 in grad
        if start_iter < 0:
            self.start_iter = self.max_iter
        else:
            self.start_iter = start_iter
        if self.rank == 0:
            s = f"det-num: {self.ci_num}, "
            s += f"Matrix shape: ({dim}, {dim}), "
            s += f"min cNqs^2: {cNqs_pow_min:.4E}, "
            s += f"start_iter: {self.start_iter}"
            logger.info(s, master=True)

        # hij, x1, inverse_index, onv_not_idx
        self.n_sd = self.sampler.n_SinglesDoubles
        self.ci_nqs_info = self.init_ci_nqs()
        # hij: (rank_ci_num, rank_ci_num * n_sd)
        numel = self.ci_nqs_info[0].numel()
        # psi: (ci_num * n_sd), <ci|H|nqs>: (ci_num)
        numel1 = self.ci_nqs_info[1].size(0)  # (unique, sorb)
        self.ci_nqs_value = torch.zeros(numel1 + numel + self.rank_ci_num, **self.factory_kwargs)
        # synchronize()
        # logger.info(f"ci-nqs-value: {self.ci_nqs_value.shape}")
        # logger.info(f"rank-ci-num: {self.rank_ci_num}")
        # logger.info(f"numel: {numel}, numel1: {numel1}")

    def make_ci_hij(self) -> None:
        """
        construe ci hij
        """
        h1e = self.sampler.h1e
        h2e = self.sampler.h2e
        hij = get_hij_torch(self.ci_det, self.ci_det, h1e, h2e, self.sorb, self.nele)
        self.Ham_matrix[: self.ci_num, : self.ci_num] = hij

    @torch.no_grad()
    def init_ci_nqs(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        init <phi_i|H|phi_nqs>
        """
        comb_x, _ = get_comb_tensor(
            self.rank_ci_det, self.sorb, self.nele, self.noa, self.nob, flag_bit=False
        )
        nbatch, n_sd, bra_len = comb_x.shape
        assert nbatch == self.rank_ci_num
        assert self.n_sd + 1 == n_sd

        hij = get_hij_torch(
            self.rank_ci_det, comb_x, self.h1e, self.h2e, self.sorb, self.nele
        )  # (rank_ci_num, n_sd)

        # Binary Search x1 in CI-Det/Nqs
        comb_x = comb_x.reshape(-1, bra_len)  # (n_sd * rank_ci_num, bra_len)
        array_idx = self.det_lut.lookup(comb_x, is_onv=True)[0]
        mask = array_idx.gt(-1)  # if not found, set to -1
        baseline = torch.arange(comb_x.size(0), device=self.device, dtype=torch.int64)
        det_not_idx = baseline[~mask]
        comb_x = comb_x[~mask]

        # remove repeated comb_x
        unique_comb, inverse_index = torch.unique(comb_x, dim=0, return_inverse=True)
        x1 = onv_to_tensor(unique_comb, self.sorb)  # x1: [n_unique, sorb], +1/-1
        onv_x1 = unique_comb

        info = (hij, x1, onv_x1, inverse_index, det_not_idx)
        return info

    @torch.no_grad()
    def make_ci_nqs(self) -> None:
        """
        \sum_j <phi_i|H|phi_j><phi_j|phi_nqs>
        """
        # onv_x1: uint8, x1 not in Det
        hij, x1, onv_x1, inverse_index, det_not_idx = self.ci_nqs_info
        offset0 = x1.size(0)

        # psi_x1: (unique-rank)
        psi_x1 = self.ci_nqs_value[:offset0]
        if self.exact:
            with torch.no_grad():
                psi_x1 = self.model(x1)  # +1/-1
        else:
            WF_LUT = self.sampler.WF_LUT
            if WF_LUT is None:
                raise ValueError(f"Use LUT to speed up <ci|H|phi_nqs>")
            lut_idx, lut_not_idx, lut_value = WF_LUT.lookup(onv_x1)
            psi_x1[lut_idx] = lut_value
            # XXX:(zbwu-01-17) use-sample-space, like eloc-energy
            psi_x1[lut_not_idx] = self.model(x1[lut_not_idx])

        offset1 = self.rank_ci_num * (self.n_sd + 1) + offset0
        psi = self.ci_nqs_value[offset0:offset1]  # (rank-ci-num, n_sd)
        rank_value = self.ci_nqs_value[offset1:]  # (rank-ci-num)
        psi[det_not_idx] = psi_x1[inverse_index]  # unique inverse

        # All-Gather value
        # XXX:(zbwu-01-16) allocate all_value memory in initial
        # value = torch.einsum("ij, ij ->i", hij, psi.reshape(self.rank_ci_num, -1))
        rank_value = (psi.reshape(self.rank_ci_num, -1) * hij).sum(-1)
        all_value = all_gather_tensor(rank_value, self.device, self.world_size)
        all_value = torch.cat(all_value)

        self.Ham_matrix[: self.ci_num, -1] = all_value
        self.Ham_matrix[-1, : self.ci_num] = all_value.conj()

        self.ci_nqs_value.zero_()

    @torch.no_grad()
    def make_nqs_nqs(self, phi_nqs: Tensor, eloc_mean: Tensor) -> None:
        """
        <phi_NQS|H|phi_NQS> = sum(prob * eloc) * <phi_nqs|phi_nqs>
        """
        if self.exact:
            # Single-Rank
            value = torch.dot(phi_nqs.conj(), phi_nqs).real * self.world_size
        else:
            # Single-Rank
            value = torch.dot(phi_nqs.conj(), phi_nqs).real * self.world_size
        all_reduce_tensor(value, world_size=self.world_size)

        self.Ham_matrix[-1, -1] = eloc_mean * value

    def solve_eigh(self) -> float:
        # TODO:(zbwu-23-12-27) davsion or scipy.sparse.linalg.eigsh
        """
        HC = εC;
        H is Hermitian matrix
        return
        ------
            ε0(ground energy)
        """
        result = torch.linalg.eigh(self.Ham_matrix)
        E0 = result[0][0]
        coeff = result[1][:, 0]

        # update ci-coeff and nqs-coeff
        self.total_coeff = coeff.to(self.device)
        self.nqs_coeff = self.total_coeff[-1]
        self.ci_coeff = self.total_coeff[: self.ci_num]
        begin = self.rank_idx_lst[self.rank]
        end = self.rank_idx_lst[self.rank + 1]
        self.rank_ci_coeff = self.ci_coeff[begin:end]
        return E0.item()

    @torch.no_grad()
    def calculate_new_term(
        self,
        state: Tensor,
        phi_nqs: Tensor,
        cNqs: Tensor,
    ) -> Tensor:
        """
        calculate (<n|H|phi_i> * c_i) / (<n|phi_nqs> * c_N)
        """
        # Single-Rank
        bra = tensor_to_onv(((state + 1) / 2).to(torch.uint8), self.sorb)
        ket = self.ci_det  # ci-det, not rank-ci-det
        hij = get_hij_torch(bra, ket, self.h1e, self.h2e, self.sorb, self.nele)

        ci = self.ci_coeff
        # Single-Rank
        x = torch.einsum("ij, j, i ->i", hij.to(self.dtype), ci, 1 / (phi_nqs * cNqs))
        return x

    def new_nqs_grad(
        self,
        nqs: DDP,
        state: Tensor,
        state_prob: Tensor,
        eloc: Tensor,
        new_term: Tensor,
        cNqs: Tensor,
        E0: float,
        epoch: int,
        MAX_AD_DIM: int = -1,
    ) -> None:
        """
        calculate NQS grad
        """
        device = state.device
        dim = state.size(0)
        loss_sum = torch.zeros(1, device=device, dtype=torch.double)

        if MAX_AD_DIM == -1:
            MAX_AD_DIM = dim
        idx_lst = [0] + split_batch_idx(dim=dim, min_batch=MAX_AD_DIM)

        # TODO:(zbwu-24-01-24) how to scale c0
        c0 = cNqs.norm().item() ** 2
        if epoch < self.start_iter:
            c0 = max(c0, self.cNqs_pow_min)

        def batch_loss_backward(begin: int, end: int) -> None:
            nonlocal loss_sum
            log_psi_batch = nqs(state[begin:end].requires_grad_()).log()
            if torch.any(torch.isnan(log_psi_batch)):
                raise ValueError(f"negative numbers in the log-psi")

            state_prob_batch = state_prob[begin:end]
            eloc_batch = eloc[begin:end]
            new_term_batch = new_term[begin:end]
            corr = eloc_batch + new_term_batch - E0
            loss = 2 * (c0 * (torch.sum(state_prob_batch * log_psi_batch.conj() * corr))).real
            loss.backward()
            loss_sum += loss.detach()

            del log_psi_batch, loss

        with MemoryTrack(device) as track:
            # disable gradient synchronizations in the rank
            with nqs.no_sync():
                for i in range(len(idx_lst) - 2):
                    begin = idx_lst[i]
                    end = idx_lst[i + 1]
                    batch_loss_backward(begin, end)
                    track.manually_clean_cache()

            begin = idx_lst[-2]
            end = idx_lst[-1]
            # synchronization gradient in the rank
            batch_loss_backward(begin, end)

        # logger.info(f"loss: {loss_sum.item():.5E}")
        synchronize()
        reduce_loss = all_reduce_tensor(loss_sum, world_size=self.world_size, in_place=False)
        synchronize()
        if self.rank == 0:
            logger.info(f"Reduce-loss: {reduce_loss[0].item():.10E}", master=True)

    def pre_train(self, prefix: str = None) -> None:
        raise NotImplementedError(f"CI-NQS does not support pre-train")

    def run(self) -> None:
        begin_vmc = time.time_ns()
        if self.rank == 0:
            logger.info(f"Begin CI-NQS iteration: {time.ctime()}", master=True)
        for epoch in range(self.max_iter):
            t0 = time.time_ns()
            initial_state = self.onstate[0].clone().detach()
            state, state_prob, eloc, e_total, stats, eloc_mean = self.sampler.run(
                initial_state, epoch=epoch
            )
            sample_state = onv_to_tensor(state, self.sorb)  # -1:unoccupied, 1: occupied

            # construct Hmat
            t1 = time.time_ns()
            if self.exact:
                with torch.no_grad():
                    phi_nqs = self.model(sample_state)
            else:
                WF_LUT = self.sampler.WF_LUT
                not_idx, phi_nqs = WF_LUT.lookup(state)[1:]
                assert not_idx.size(0) == 0

            self.make_ci_nqs()  # <phi_i|H|phi_nqs>
            self.make_nqs_nqs(phi_nqs, eloc_mean)  # <phi_nqs|H|phi_nqs>
            E0 = self.solve_eigh()  # solve HC = εC, C0({ci},c_N)

            delta_Hmat = (time.time_ns() - t1) / 1.0e09

            # logging coeff
            if self.rank == 0:
                self.coeff_lst.append(self.total_coeff.to("cpu").numpy())
                c1 = self.ci_coeff.norm().item()
                c2 = self.nqs_coeff.norm().item()
                logger.info(f"E0: {E0}, Coeff: {c1:.6E} {c2:6E}", master=True)

            t2 = time.time_ns()
            new_term = self.calculate_new_term(sample_state, phi_nqs, self.nqs_coeff)
            delta_new_term = (time.time_ns() - t2) / 1.00e09

            # backward
            t3 = time.time_ns()
            self.new_nqs_grad(
                nqs=self.model,
                state=sample_state,
                state_prob=state_prob,
                eloc=eloc,
                new_term=new_term,
                cNqs=self.nqs_coeff,
                E0=E0,
                epoch=epoch,
                MAX_AD_DIM=self.MAX_AD_DIM,
            )

            # if self.rank == 0:
            #     for param in self.model.parameters():
            #         print(param.grad.reshape(-1))
            #         break

            delta_grad = (time.time_ns() - t3) / 1.00e09

            # save the energy grad
            self.save_grad_energy(E0 + self.ecore)

            # update param
            t4 = time.time_ns()
            self.update_param(epoch=epoch)
            delta_param = (time.time_ns() - t4) / 1.00e09

            # save the checkpoint, different-version maybe error
            self.save_checkpoint(epoch=epoch)

            delta = (time.time_ns() - t0) / 1.00e09

            if self.rank == 0:
                s = f"Construct Hmat: {delta_Hmat:.3E} s, new-term: {delta_new_term:.3E} s"
                logger.info(s, master=True)
            cost = torch.tensor([delta_grad, delta_param, delta], device=self.device)
            self.logger_iteration_info(epoch=epoch, cost=cost)

        # End CI-NQS iteration
        vmc_time = (time.time_ns() - begin_vmc) / 1.0e09
        if self.rank == 0:
            path = os.path.split(self.prefix)[0]
            coeff = np.asarray(self.coeff_lst)
            np.save(f"{path}/{self.ci_num}-coeff.npy", coeff)
            s = f"End CI-NQS iteration: {time.ctime()}\n"
            s += f"total cost time: {vmc_time:.3E} s, {vmc_time/60:.3E} min {vmc_time/3600:.3E} h"
            logger.info(s, master=True)
