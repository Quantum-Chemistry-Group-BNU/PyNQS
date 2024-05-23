from __future__ import annotations

import time
import os
import warnings
import torch
import numpy as np
import scipy

from dataclasses import dataclass
from loguru import logger
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from vmc.optim import BaseVMCOptimizer
from utils.public_function import (
    split_length_idx,
    split_batch_idx,
    MemoryTrack,
)
from utils.det_helper import DetLUT
from utils.distributed import all_reduce_tensor, synchronize, all_gather_tensor
from libs.C_extension import get_hij_torch, onv_to_tensor, tensor_to_onv, get_comb_tensor
from ci import CIWavefunction

USE_SCIPY = False
USE_CPU = False


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

    Ham_matrix_spin: Tensor = None
    """the <S-S+> matrix, (m + 1, m +1)"""

    total_coeff: Tensor = None
    """total coeff, (m + 1)"""

    nqs_coeff: Tensor = None
    """the NQS coeff, (1)"""

    ci_coeff: Tensor = None
    """CI-det coeff, (m)"""

    grad_strategy: int = None
    """
    update grad strategy.
    0: ||cN||^2 * scale * ...
    1: cN * scale * ...
    2: only opt NQS, fail
    """

    def __init__(
        self,
        CI: CIWavefunction,
        cNqs_pow_min: float = 1.0e-4,
        start_iter: int = -1,
        use_sample_space: bool = False,
        MAX_FP_DIM: int = -1,
        grad_strategy: int = 0,
        use_KNN_E0: bool = False,
        K_step: int = 1000,
        **vmc_opt_kwargs: dict,
    ) -> None:
        # Remove pre-train info
        vmc_opt_kwargs.pop("pre_CI", None)
        vmc_opt_kwargs.pop("pre_train_info", None)
        vmc_opt_kwargs.pop("noise_lambda", None)
        vmc_opt_kwargs.pop("clean_opt_state", None)
        super(NqsCi, self).__init__(**vmc_opt_kwargs)

        # check ansatz remove det
        model = self.model_raw.module
        if not (hasattr(model, "remove_det") and model.remove_det):
            raise TypeError(f"CI-NQS must remove CI-Det")
        if not (hasattr(model, "det_lut") and model.det_lut is not None):
            raise TypeError(f"CI-NQS must remove CI-Det")

        if grad_strategy not in (0, 1, 2):
            raise NotImplementedError(f"grad-strategy muse be in 0,1,2")
        self.grad_strategy = grad_strategy
        if self.grad_strategy == 2:
            warnings.warn(f"This method is fail", RuntimeWarning)

        self.ci_det = CI.space
        self.ci_num = CI.space.size(0)
        self.factory_kwargs = {"device": self.device, "dtype": self.dtype}
        self.det_lut = self.sampler.det_lut
        dim = self.ci_num + 1
        self.Ham_matrix = torch.zeros((dim, dim), **self.factory_kwargs)
        if self.use_spin_raising:
            self.Ham_matrix_spin = torch.zeros_like(self.Ham_matrix)

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
        assert self.ci_num >= self.world_size
        idx_lst = [0] + split_length_idx(dim=self.ci_num, length=self.world_size)
        begin = idx_lst[self.rank]
        end = idx_lst[self.rank + 1]
        self.rank_ci_det = self.ci_det[begin:end]
        self.rank_ci_num = self.rank_ci_det.size(0)
        self.rank_ci_coeff = self.ci_coeff[begin:end]
        self.rank_idx_lst = idx_lst

        # TODO:(zbwu-01-31)How to better adjust cNqs^2, e.g. N2-ccpvdz
        # min cNqs^2
        assert cNqs_pow_min > 0.0 and cNqs_pow_min <= 1.0
        self.cNqs_pow_min = cNqs_pow_min
        # change cNqs^2 in grad
        if start_iter < 0:
            self.start_iter = self.max_iter
        else:
            self.start_iter = start_iter

        # max forward dim in <phi_ci|H|phi_nqs>
        self.MAX_FP_DIM = MAX_FP_DIM

        # <phi_ci|H|phi_nqs> in sample-space
        self.use_sample_space = use_sample_space
        # hij, x1, inverse_index, onv_not_idx
        self.n_sd = self.sampler.n_SinglesDoubles
        self.ci_nqs_info = self.init_ci_nqs()
        # hij: (2/1, rank_ci_num, rank_ci_num * n_sd)
        numel = self.ci_nqs_info[0][0].numel()
        # psi: (ci_num * n_sd), <ci|H|nqs>: (ci_num)
        numel1 = self.ci_nqs_info[1].size(0)  # (unique, sorb)
        self.ci_nqs_value = torch.zeros(
            numel1 + numel + self.rank_ci_num + self.ci_num, **self.factory_kwargs
        )
        # synchronize()
        # logger.info(f"ci-nqs-value: {self.ci_nqs_value.shape}")
        # logger.info(f"rank-ci-num: {self.rank_ci_num}")
        # logger.info(f"numel: {numel}, numel1: {numel1}")

        # E0_mean: K-step E0 mean
        self.use_KNN_E0 = use_KNN_E0
        self.k_step = K_step
        self.E0_mean: float = 0.0
        self.E0_lst = np.zeros(self.max_iter, dtype=np.double)
        if self.rank == 0:
            # double or int64
            pre_memory = sum(map(torch.numel, self.ci_nqs_info)) * 8 / 2**30
            # uint8, comb_x
            pre_memory -= self.ci_nqs_info[2].numel() * 7 / 2**30
            if self.dtype == torch.complex128:
                pre_memory += self.ci_nqs_value.numel() * 8 * 2 / 2**30
            else:
                pre_memory += self.ci_nqs_value.numel() * 8 / 2**30
            s = "CI-NQS:(\n"
            s += f"    det-num: {self.ci_num}\n"
            s += f"    Matrix shape: ({dim}, {dim})\n"
            s += f"    min cNqs^2: {cNqs_pow_min:.4E}\n"
            s += f"    start_iter: {self.start_iter}\n"
            s += f"    use-sample-space: {self.use_sample_space}\n"
            s += f"    pre-allocated memory: {pre_memory:.5f}GiB\n"
            s += f"    Grad-strategy: {self.grad_strategy}\n"
            s += f"    MAX_FP_DIM: {self.MAX_FP_DIM}\n"
            s += f"    USE KNN E0-mean: {self.use_KNN_E0}\n"
            s += f"    K-step: {self.k_step}"
            s += "\n)"
            logger.info(s, master=True)

    def make_ci_hij(self) -> None:
        """
        construe ci hij
        """
        h1e = self.sampler.h1e
        h2e = self.sampler.h2e
        hij = get_hij_torch(self.ci_det, self.ci_det, h1e, h2e, self.sorb, self.nele)
        self.Ham_matrix[: self.ci_num, : self.ci_num] = hij

        if self.use_spin_raising:
            h1e_spin = self.sampler.h1e_spin
            h2e_spin = self.sampler.h2e_spin
            hij_spin = get_hij_torch(
                self.ci_det, self.ci_det, h1e_spin, h2e_spin, self.sorb, self.nele
            )
            self.Ham_matrix_spin[: self.ci_num, : self.ci_num] = hij_spin

    @torch.no_grad()
    def init_ci_nqs(self) -> list[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        init <phi_i|H|phi_nqs>
        """
        comb_x, _ = get_comb_tensor(
            self.rank_ci_det, self.sorb, self.nele, self.noa, self.nob, flag_bit=False
        )
        nbatch, n_sd, bra_len = comb_x.shape
        assert nbatch == self.rank_ci_num
        assert self.n_sd + 1 == n_sd

        if self.use_spin_raising:
            hij_all = torch.empty(2, self.rank_ci_num, n_sd, device=self.device, dtype=torch.double)
        else:
            hij_all = torch.empty(1, self.rank_ci_num, n_sd, device=self.device, dtype=torch.double)

        hij_all[0] = get_hij_torch(
            self.rank_ci_det, comb_x, self.h1e, self.h2e, self.sorb, self.nele
        )  # (rank_ci_num, n_sd)

        if self.use_spin_raising:
            hij_all[1] = get_hij_torch(
                self.rank_ci_det, comb_x, self.h1e_spin, self.h2e_spin, self.sorb, self.nele
            )

        # Binary Search x1 in CI-Det/Nqs
        comb_x = comb_x.reshape(-1, bra_len)  # (n_sd * rank_ci_num, bra_len)
        array_idx = self.det_lut.lookup(comb_x, is_onv=True)[0]
        mask = array_idx.gt(-1)  # if not found, set to -1
        baseline = torch.arange(comb_x.size(0), device=self.device, dtype=torch.int64)
        det_not_idx = baseline[~mask]
        comb_x = comb_x[~mask]

        # remove repeated comb_x
        unique_comb, inverse_index = torch.unique(comb_x, dim=0, return_inverse=True)
        onv_x1 = unique_comb
        # save the memory: unique (nSD * nCI, sorb)
        if not self.use_sample_space or self.exact:
            x1 = onv_to_tensor(unique_comb, self.sorb)  # x1: [n_unique, sorb], +1/-1
        else:
            # x1: [n_unique, 0], placeholders
            x1 = torch.zeros(unique_comb.size(0), 0, device=self.device, dtype=torch.double)

        info = (hij_all, x1, onv_x1, inverse_index, det_not_idx)
        return list(info)

    @torch.no_grad()
    def _batch_forward(self, x: Tensor, MAX_DIM: int = -1) -> Tensor:
        """
        forward model(x) with batch used in exact <phi_ci|H|phi_nqs>
        """
        dim = x.size(0)
        if self.rank == 0:
            logger.info(f"CI-NQS forward dim: {dim}", master=True)
        if dim == 0:
            placeholder = torch.empty(0, dtype=self.dtype, device=self.device)
            return placeholder

        if MAX_DIM == -1:
            min_batch = dim
        else:
            assert MAX_DIM > 0
            min_batch = MAX_DIM
        idx_lst = [0] + split_batch_idx(dim=dim, min_batch=min_batch)

        result = torch.empty(dim, dtype=self.dtype, device=self.device)
        for i in range(len(idx_lst) - 1):
            begin, end = idx_lst[i], idx_lst[i + 1]
            x_batch = x[begin:end]
            result[begin:end] = self.model(x_batch)

        return result

    @torch.no_grad()
    def make_ci_nqs(self) -> None:
        """
        \sum_j <phi_i|H|phi_j><phi_j|phi_nqs>
        """
        # onv_x1: uint8, x1 not in Det
        hij_all, x1, onv_x1, inverse_index, det_not_idx = self.ci_nqs_info
        offset0 = x1.size(0)

        # psi_x1: (unique-rank)
        psi_x1 = self.ci_nqs_value[:offset0]
        if self.exact:
            with torch.no_grad():
                # psi_x1 = self.model(x1)  # +1/-1
                psi_x1 = self._batch_forward(x1, self.MAX_FP_DIM)
        else:
            WF_LUT = self.sampler.WF_LUT
            if WF_LUT is None:
                raise ValueError(f"Use LUT to speed up <ci|H|phi_nqs>")
            lut_idx, lut_not_idx, lut_value = WF_LUT.lookup(onv_x1)
            psi_x1[lut_idx] = lut_value.to(self.dtype)
            # use-sample-space, like eloc-energy
            if self.use_sample_space:
                ...
            else:
                # psi_x1[lut_not_idx] = self.model(x1[lut_not_idx])
                psi_x1[lut_not_idx] = self._batch_forward(x1[lut_not_idx], self.MAX_FP_DIM)

        offset1 = self.rank_ci_num * (self.n_sd + 1) + offset0
        psi = self.ci_nqs_value[offset0:offset1]  # (rank-ci-num, n_sd)
        offset2 = self.rank_ci_num + offset1
        rank_value = self.ci_nqs_value[offset1:offset2]  # (rank-ci-num)
        psi[det_not_idx] = psi_x1[inverse_index]  # unique inverse

        # All-Gather-value
        all_value = self.ci_nqs_value[offset2:]  # (ci-num)
        assert all_value.size(0) == self.ci_num

        # value = torch.einsum("ij, ij ->i", hij, psi.reshape(self.rank_ci_num, -1))
        rank_value = (psi.reshape(self.rank_ci_num, -1) * hij_all[0]).sum(-1)
        torch.cat(all_gather_tensor(rank_value, self.device, self.world_size), out=all_value)
        self.Ham_matrix[: self.ci_num, -1] = all_value
        self.Ham_matrix[-1, : self.ci_num] = all_value.conj()

        if self.use_spin_raising:
            rank_value_spin = (psi.reshape(self.rank_ci_num, -1) * hij_all[1]).sum(-1)
            torch.cat(
                all_gather_tensor(rank_value_spin, self.device, self.world_size), out=all_value
            )
            self.Ham_matrix_spin[: self.ci_num, -1] = all_value
            self.Ham_matrix_spin[-1, : self.ci_num] = all_value.conj()

        self.ci_nqs_value.zero_()

    @torch.no_grad()
    def make_nqs_nqs(self, phi_nqs: Tensor, eloc_mean: Tensor, sloc_mean: Tensor) -> None:
        """
        <phi_NQS|H|phi_NQS> = sum(prob * eloc)
        """
        # if self.exact:
        #     # Single-Rank, value = 1.0 if use AR-ansatz
        #     value = torch.dot(phi_nqs.conj(), phi_nqs).real * self.world_size
        # else:
        #     # Single-Rank
        #     value = torch.dot(phi_nqs.conj(), phi_nqs).real * self.world_size
        # all_reduce_tensor(value, world_size=self.world_size)

        self.Ham_matrix[-1, -1] = eloc_mean.real

        if self.use_spin_raising:
            self.Ham_matrix_spin[-1, -1] = sloc_mean.real

    def solve_eigh(self) -> tuple[float, float]:
        # TODO:(zbwu-23-12-27) davsion or scipy.sparse.linalg.eigsh
        """
        HC = εC;
        H is Hermitian matrix
        return
        ------
            E0(ground energy)
            e_spin(<S-S+> )
        """

        c1 = self.spin_raising_coeff
        if self.use_spin_raising and not self.only_output_spin_raising:
            Ham_matrix = self.Ham_matrix + c1 * self.Ham_matrix_spin
        else:
            Ham_matrix = self.Ham_matrix

        if USE_SCIPY:
            # use scipy.linalg.eigh in CPU
            result = scipy.linalg.eigh(Ham_matrix.cpu().numpy())
            E_all = result[0][0]
            coeff = torch.from_numpy(result[1][:, 0])
        else:
            # use torch.linalg.eigh in CPU or CUDA
            if USE_CPU:
                result = torch.linalg.eigh(Ham_matrix.cpu())
            else:
                result = torch.linalg.eigh(Ham_matrix)
            E_all = result[0][0]
            coeff = result[1][:, 0]

        # assert (self.Ham_matrix - self.Ham_matrix.T.conj()).norm() < 1e-6

        # update ci-coeff and nqs-coeff
        self.total_coeff = coeff.to(self.device)
        self.nqs_coeff = self.total_coeff[-1]
        self.ci_coeff = self.total_coeff[: self.ci_num]
        begin = self.rank_idx_lst[self.rank]
        end = self.rank_idx_lst[self.rank + 1]
        self.rank_ci_coeff = self.ci_coeff[begin:end]

        if self.use_spin_raising:
            e_spin = torch.einsum(
                "i, ij, j ->", self.total_coeff.conj(), c1 * self.Ham_matrix_spin, self.total_coeff
            ).real.item()
        else:
            e_spin = 0.00

        if self.use_spin_raising and not self.only_output_spin_raising:
            E0 = E_all.item() - e_spin
        else:
            E0 = E_all.item()

        # E0 = torch.einsum("i, ij, j ->", self.total_coeff.conj(), self.Ham_matrix, self.total_coeff)
        # logger.info(f"delta: {E0.real.item() - (E_all.item() - e_spin):.10f}")
        return E0, e_spin

    @torch.no_grad()
    def calculate_new_term(
        self,
        state: Tensor,
        phi_nqs: Tensor,
        cNqs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        calculate new-term (<n|H|phi_i> * c_i) / <n|phi_nqs>
        """
        if self.grad_strategy == 2:
            # not calculate new-term, so return zeros
            x = torch.zeros(state.size(0), dtype=self.dtype, device=self.device)
            return x

        # Single-Rank
        bra = tensor_to_onv(((state + 1) / 2).to(torch.uint8), self.sorb)
        ket = self.ci_det  # ci-det, not rank-ci-det

        # TODO: check
        hij = get_hij_torch(bra, ket, self.h1e, self.h2e, self.sorb, self.nele)

        if self.use_spin_raising:
            hij_spin = get_hij_torch(bra, ket, self.h1e_spin, self.h2e_spin, self.sorb, self.nele)

        ci = self.ci_coeff
        # Single-Rank
        # x = torch.einsum("ij, j, i ->i", hij.to(self.dtype), ci, 1 / phi_nqs)
        if self.dtype == torch.complex128:
            value = torch.empty(bra.size(0) * 2, device=self.device, dtype=torch.double)
            value[0::2] = torch.matmul(hij, ci.real)  # Real-part
            value[1::2] = torch.matmul(hij, ci.imag)  # imag-part
            x = torch.view_as_complex(value.view(-1, 2)).div(phi_nqs)
            if self.use_spin_raising:
                value[0::2] = torch.matmul(hij_spin, ci.real)
                value[1::2] = torch.matmul(hij_spin, ci.imag)
                x1 = torch.view_as_complex(value.view(-1, 2)).div(phi_nqs)
            else:
                x1 = torch.zeros_like(x)

        elif self.dtype == torch.double:
            x = torch.matmul(hij, ci).div(phi_nqs)
            if self.use_spin_raising:
                x1 = torch.matmul(hij_spin, ci).div(phi_nqs)
            else:
                x1 = torch.zeros_like(x)
        else:
            raise NotImplementedError(f"Single/Complex-Single dose not been supported")
        return x, x1

    def new_nqs_grad(
        self,
        nqs: DDP,
        state: Tensor,
        state_prob: Tensor,
        loc: Tensor,
        loc_mean: Tensor,
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

        if self.use_KNN_E0:
            self.E0_lst[epoch] = E0
            if epoch < self.k_step:
                E0 = E0
            else:
                start = epoch - self.k_step
                end = epoch
                E0 = self.E0_lst[start:end].mean()
            if self.rank == 0:
                logger.info(f"KNN-E0: {E0:.12f}", master=True)

        # scale cNqs
        scale = 1.0
        if self.grad_strategy == 0:
            c0 = cNqs.norm().item() ** 2
            if epoch < self.start_iter:
                # c0 = max(c0, self.cNqs_pow_min)
                scale = max(c0, self.cNqs_pow_min) / c0
        elif self.grad_strategy == 1:
            c0 = cNqs.conj().clone()
            if epoch < self.start_iter:
                # c0 /= c0.norm()/max(c0.norm(), self.cNqs_pow_min**0.5)
                scale = (max(c0.norm(), self.cNqs_pow_min**0.5) / c0.norm()).item()
        elif self.grad_strategy == 2:
            # only optimize NQS
            c0 = 1.0
        if self.rank == 0:
            logger.info(f"scale: {scale:.4f}", master=True)

        def batch_loss_backward(begin: int, end: int) -> None:
            nonlocal loss_sum
            log_psi_batch = nqs(state[begin:end].requires_grad_()).log()
            if torch.any(torch.isnan(log_psi_batch)):
                raise ValueError(f"negative numbers in the log-psi")

            state_prob_batch = state_prob[begin:end]
            eloc_batch = loc[begin:end]
            new_term_batch = new_term[begin:end]
            if self.grad_strategy == 0:
                corr = eloc_batch + new_term_batch / cNqs - E0
            elif self.grad_strategy == 1:
                corr = eloc_batch * cNqs + new_term_batch - E0 * cNqs
            elif self.grad_strategy == 2:
                corr = eloc_batch - loc_mean
            loss = (
                2 * (c0 * scale * (torch.sum(state_prob_batch * log_psi_batch.conj() * corr))).real
            )
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
            state, state_prob, (eloc, sloc), (eloc_mean, sloc_mean) = self.sampler.run(
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
                if WF_LUT is None:
                    raise ValueError(f"Use LUT to speed up <phi_nqs|H|phi_nqs>")
                not_idx, phi_nqs = WF_LUT.lookup(state)[1:]
                assert not_idx.size(0) == 0

            # change spin-raising-coeff
            if self.spin_raising_scheduler is not None:
                c0 = self.initial_spin_spin_coeff
                self.spin_raising_coeff = self.spin_raising_scheduler(epoch) * c0
            self.make_ci_nqs()  # <phi_i|H|phi_nqs>
            self.make_nqs_nqs(phi_nqs, eloc_mean, sloc_mean)  # <phi_nqs|H|phi_nqs>
            E0, e_spin = self.solve_eigh()  # solve HC = εC, C0({ci},c_N)

            delta_Hmat = (time.time_ns() - t1) / 1.0e09

            # logging coeff
            if self.rank == 0:
                self.coeff_lst.append(self.total_coeff.to("cpu").numpy())
                c1 = self.ci_coeff.norm().item()
                c2 = self.nqs_coeff.norm().item()
                s = f"Hybrid energy: {E0:.9f}, spin-raising: {e_spin/self.spin_raising_coeff:.5E}, "
                s += f"Coeff: {c1:.6E} {c2:6E}"
                logger.info(s, master=True)

            t2 = time.time_ns()

            if self.only_sample:
                delta = (time.time_ns() - t0) / 1.00e06
                if self.rank == 0:
                    s = f"{epoch}-th only Sampling finished, cost time {delta:.3f} ms\n"
                    s += "=" * 100
                    logger.info(s, master=True)
                continue

            # TODO: check
            new_term, new_term_spin = self.calculate_new_term(sample_state, phi_nqs, self.nqs_coeff)
            delta_new_term = (time.time_ns() - t2) / 1.00e09

            # backward
            t3 = time.time_ns()
            sloc = sloc * self.spin_raising_coeff
            sloc_mean = sloc_mean * self.spin_raising_coeff
            new_term_spin = new_term_spin * self.spin_raising_coeff
            if self.only_output_spin_raising:
                sloc = torch.zeros_like(eloc)
                sloc_mean = torch.zeros_like(eloc_mean)
                e_spin = 0.0
                new_term_spin = torch.zeros_like(new_term)

            self.new_nqs_grad(
                nqs=self.model,
                state=sample_state,
                state_prob=state_prob,
                loc=eloc + sloc,
                loc_mean=eloc_mean + sloc_mean,
                new_term=new_term + new_term_spin,
                cNqs=self.nqs_coeff,
                E0=E0 + e_spin,
                epoch=epoch,
                MAX_AD_DIM=self.MAX_AD_DIM,
            )

            # if self.rank == 0:
            #     for param in self.model.parameters():
            #         print(param.grad.reshape(-1))
            #         break

            delta_grad = (time.time_ns() - t3) / 1.00e09

            # save the energy grad and clip-grad
            self.clip_grad(epoch=epoch)
            self.save_grad_energy(E0 + self.ecore + e_spin)

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
            if self.sampler.use_LUT:
                self.sampler.WF_LUT.clean_memory()
        # End CI-NQS iteration
        vmc_time = (time.time_ns() - begin_vmc) / 1.0e09
        if self.rank == 0:
            path = os.path.split(self.prefix)[0]
            coeff = np.asarray(self.coeff_lst)
            np.save(f"{path}/{self.ci_num}-coeff.npy", coeff)
            s = f"End CI-NQS iteration: {time.ctime()}\n"
            s += f"total cost time: {vmc_time:.3E} s, {vmc_time/60:.3E} min {vmc_time/3600:.3E} h"
            logger.info(s, master=True)

    def operator_expected(
        self,
        h1e: Tensor,
        h2e: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor]:
        """
        calculate <O> using different h1e, h2e, e.g. S_S+, H.

        Returns:
            state, prob, eloc, eloc_mean, E0, ci_coeff, nqs_coeff
        """
        if self.rank == 0:
            logger.info(f"{'*' * 30}Begin calculating <O>{'*' * 30}", master=True)

        if len(self.e_lst) == 0:  # not CI-NQS iteration
            # calculate the before ci/cNqs coeff
            if self.rank == 0:
                logger.info(f"{'=' * 20}Calculate before ci/cNqs coeff{'=' * 20}", master=True)
            initial_state = self.onstate[0].clone().detach()
            state, state_prob, (eloc, sloc), (eloc_mean, sloc_mean) = self.sampler.run(
                initial_state, epoch=self.max_iter
            )
            sample_state = onv_to_tensor(state, self.sorb)  # -1:unoccupied, 1: occupied
            if self.exact:
                with torch.no_grad():
                    phi_nqs = self.model(sample_state)
            else:
                WF_LUT = self.sampler.WF_LUT
                if WF_LUT is None:
                    raise ValueError(f"Use LUT to speed up <phi_nqs|H|phi_nqs>")
                not_idx, phi_nqs = WF_LUT.lookup(state)[1:]
                assert not_idx.size(0) == 0

            self.make_ci_nqs()  # <phi_i|H|phi_nqs>
            self.make_nqs_nqs(phi_nqs, eloc_mean, sloc_mean)  # <phi_nqs|H|phi_nqs>
            E0, e_spin = self.solve_eigh()  # solve HC = εC, C0({ci},c_N)
            if self.rank == 0:
                logger.info(f"{'=' * 25}Finish calculation{'=' * 25}", master=True)

        h1e_old = self.sampler.h1e
        assert h1e.shape == h1e_old.shape
        self.sampler.h1e = h1e.to(self.device)

        h2e_old = self.sampler.h2e
        assert h2e.shape == h2e_old.shape
        self.sampler.h2e = h2e.to(self.device)

        # not add <S-S+>
        h1e_spin_old = self.sampler.h1e_spin
        h2e_spin_old = self.sampler.h2e_spin
        use_spin_raising = self.sampler.use_spin_raising
        self.sampler.h1e_spin = None
        self.sampler.h2e_spin = None
        self.sampler.use_spin_raising = False

        # Run sampling
        initial_state = self.onstate[0].clone().detach()
        state, state_prob, (eloc, sloc), (eloc_mean, sloc_mean) = self.sampler.run(
            initial_state, self.max_iter
        )
        sample_state = onv_to_tensor(state, self.sorb)  # -1:unoccupied, 1: occupied
        # state, prob, eloc, eloc_mean = self._operator_expected(h1e, h2e)

        if self.exact:
            with torch.no_grad():
                phi_nqs = self.model(sample_state)  # +1/-1
        else:
            WF_LUT = self.sampler.WF_LUT
            if WF_LUT is None:
                raise ValueError(f"Use LUT to speed up <phi_nqs|H|phi_nqs>")
            # onv = tensor_to_onv((state + 1) / 2).to(torch.uint8), self.sorb
            not_idx, phi_nqs = WF_LUT.lookup(state)[1:]
            assert not_idx.size(0) == 0  # state must be in WaveFunction-LUT

        # New matrix-element using new h1e/h2e
        h1e = self.sampler.h1e
        h2e = self.sampler.h2e
        ci_hij_old = self.Ham_matrix[: self.ci_num, : self.ci_num]
        ci_hij_new = get_hij_torch(self.ci_det, self.ci_det, h1e, h2e, self.sorb, self.nele)

        ci_nqs_hij_old = self.ci_nqs_info[0]
        comb_x, _ = get_comb_tensor(
            self.rank_ci_det, self.sorb, self.nele, self.noa, self.nob, flag_bit=False
        )
        ci_nqs_hij_new = get_hij_torch(self.rank_ci_det, comb_x, h1e, h2e, self.sorb, self.nele)
        self.ci_nqs_info[0] = ci_nqs_hij_new

        self.Ham_matrix[: self.ci_num, : self.ci_num] = ci_hij_new  # <phi_i|H|phi_i>
        self.make_ci_nqs()  # <phi_i|H|phi_nqs>
        self.make_nqs_nqs(phi_nqs, eloc_mean, sloc_mean)  # <phi_nqs|H|phi_nqs>

        # Using before coeff:
        e0 = torch.einsum("i, ij, j", self.total_coeff.conj(), self.Ham_matrix, self.total_coeff)

        if self.rank == 0:
            c1 = self.ci_coeff.norm().item()
            c2 = self.nqs_coeff.norm().item()
            s = f"<O>: {e0.item().real:.10f}, c1: {c1:.5E}, c2: {c2:.5E}"
            logger.info(s, master=True)

        # revise
        self.Ham_matrix[: self.ci_num, : self.ci_num] = ci_hij_old
        self.ci_nqs_info[0] = ci_nqs_hij_old
        self.sampler.h1e = h1e_old
        self.sampler.h2e = h2e_old
        self.sampler.h1e_spin = h1e_spin_old
        self.sampler.h2e_spin = h2e_spin_old
        self.sampler.use_spin_raising = use_spin_raising

        if self.rank == 0:
            logger.info(f"{'*'* 30}End <O>{'*' * 30}", master=True)

        del comb_x, ci_hij_new, ci_nqs_hij_new

        return sample_state, state_prob, eloc, eloc_mean, e0
