from __future__ import annotations

import time
import os
import torch
import torch.distributed as dist
import numpy as np

from typing import Union, Tuple, List
from loguru import logger
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from numpy import ndarray

from vmc.optim import VMCOptimizer
from utils.public_function import check_para, find_common_state
from libs.C_extension import get_hij_torch, onv_to_tensor, tensor_to_onv, get_comb_tensor
from ci import CIWavefunction


class NqsCi(VMCOptimizer):
    def __init__(
        self,
        CI: CIWavefunction,
        cNqs_pow_min: float = 1.0e-4,
        start_iter: int = -1,
        **vmc_opt_kwargs: dict,
    ) -> None:
        super(NqsCi, self).__init__(**vmc_opt_kwargs)

        self.ci_det = CI.space
        self.ci_num = CI.space.size(0)
        self.factory_kwargs = {"device": self.device, "dtype": self.dtype}

        self.det_lut = self.sampler.det_lut
        dim = self.ci_num + 1
        self.Ham_matrix = torch.zeros((dim, dim), **self.factory_kwargs)
        self.make_ci_hij()

        # init total-coeff
        self.total_coeff = torch.rand(dim, dtype=self.dtype).to(self.device)
        # self.total_coeff[: self.ci_num] = CI.coeff.to(self.device)
        # normalization
        self.total_coeff /= self.total_coeff.norm()
        self.nqs_coeff = self.total_coeff[-1]
        self.ci_coeff = self.total_coeff[: self.ci_num]

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
        # hij: (ci_num, ci_num * n_sd)
        numel = self.ci_nqs_info[0].numel()
        # psi: (ci_num * n_sd), <ci|H|nqs>: (ci_num)
        numel1 = self.ci_nqs_info[1].size(0)  # (unique, sorb)
        self.ci_nqs_value = torch.zeros(numel1 + numel + self.ci_num, **self.factory_kwargs)

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
        use_unique = True
        comb_x, _ = get_comb_tensor(
            self.ci_det, self.sorb, self.nele, self.noa, self.nob, flag_bit=False
        )
        nbatch, n_sd, bra_len = comb_x.shape
        assert nbatch == self.ci_num
        assert self.n_sd + 1 == n_sd

        hij = get_hij_torch(
            self.ci_det, comb_x, self.h1e, self.h2e, self.sorb, self.nele
        )  # (ci_num, n_sd)

        # Binary Search x1 in CI-Det/Nqs
        comb_x = comb_x.reshape(-1, bra_len)  # (n_sd * ci_num, bra_len)
        array_idx = self.det_lut.lookup(comb_x, is_onv=True)[0]
        mask = array_idx.gt(-1)  # if not found, set to -1
        baseline = torch.arange(comb_x.size(0), device=self.device, dtype=torch.int64)
        det_not_idx = baseline[~mask]
        comb_x = comb_x[~mask]

        # remove repeated comb_x
        unique_comb, inverse_index = torch.unique(comb_x, dim=0, return_inverse=True)
        x1 = onv_to_tensor(unique_comb, self.sorb)  # x1: [n_unique, sorb], +1/-1
        onv_x1 = unique_comb

        info = (hij.to(self.dtype), x1, onv_x1, inverse_index, det_not_idx)
        return info

    @torch.no_grad()
    def make_ci_nqs(self) -> None:
        """
        \sum_j <phi_i|H|phi_j><phi_j|phi_nqs>
        """
        # onv_x1: uint8, x1 not in Det
        hij, x1, onv_x1, inverse_index, det_not_idx = self.ci_nqs_info
        offset0 = x1.size(0)

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
            psi_x1[lut_not_idx] = self.model(x1[lut_not_idx])

        offset1 = self.ci_num * (self.n_sd + 1) + offset0
        psi = self.ci_nqs_value[offset0:offset1]  # (ci_num, n_sd)
        value = self.ci_nqs_value[offset1:]  # (ci_num)
        psi[det_not_idx] = psi_x1[inverse_index]  # unique inverse

        value = torch.einsum("ij, ij ->i", hij.to(self.dtype), psi.reshape(self.ci_num, -1))
        self.Ham_matrix[: self.ci_num, -1] = value
        self.Ham_matrix[-1, : self.ci_num] = value.conj()

        # clean: self.ci_nqs_value
        self.ci_nqs_value.zero_()

    @torch.no_grad()
    def make_nqs_nqs(self, phi_nqs: Tensor, eloc_mean: Tensor) -> None:
        """
        <phi_NQS|H|phi_NQS> = sum(prob * eloc) * <phi_nqs|phi_nqs>
        """
        # breakpoint()
        x = eloc_mean * torch.dot(phi_nqs.conj(), phi_nqs)

        self.Ham_matrix[-1, -1] = x

    def solve_eigh(self) -> Tuple[float, Tensor]:
        # TODO:(zbwu-23-12-27) davsion or scipy.sparse.linalg.eigsh
        """
        HC = εC;
        H is Hermitian matrix
        return
        ------
            ε0(ground energy),
            C0({ci},c_N)
        """
        result = torch.linalg.eigh(self.Ham_matrix)
        E0 = result[0][0]
        coeff = result[1][:, 0]
        self.total_coeff = coeff.to(self.device)
        self.nqs_coeff = self.total_coeff[-1]
        self.ci_coeff = self.total_coeff[: self.ci_num]
        return E0.item(), coeff

    @torch.no_grad()
    def calculate_new_term(
        self,
        state: Tensor,
        phi_nqs: Tensor,
        cNqs: Tensor,
        state_prob: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        calculate (<n|H|phi_i> * c_i) / (<n|phi_nqs> * c_N)
        """
        bra = tensor_to_onv(((state + 1) / 2).to(torch.uint8), self.sorb)
        ket = self.ci_det
        hij = get_hij_torch(bra, ket, self.h1e, self.h2e, self.sorb, self.nele)

        ci = self.ci_coeff
        x = torch.einsum("ij, j, i ->i", hij.to(self.dtype), ci, 1 / (phi_nqs * cNqs))
        x_mean = torch.dot(x, state_prob.to(self.dtype))
        return x, x_mean

    def new_nqs_grad(
        self,
        NQS: DDP,
        state: Tensor,
        state_prob: Tensor,
        eloc: Tensor,
        eloc_mean: Tensor,
        new_term: Tensor,
        new_term_mean: Tensor,
        cNqs: Tensor,
        E0: float,
        epoch: int,
    ) -> Tensor:
        """
        calculate NQS grad
        """
        log_psi = NQS(state.requires_grad_()).log()
        if torch.any(torch.isnan(log_psi)):
            raise ValueError(f"negative numbers in the log-psi")

        # eloc_mean + new_term_mean == E0 ????
        corr = eloc + new_term - E0

        # TODO:(zbwu-24-01-24) how to scale c0
        c0 = cNqs.norm().item() ** 2
        if epoch < self.start_iter:
            c0 = max(c0, self.cNqs_pow_min)
        loss = 2 * (c0 * (torch.sum(state_prob * log_psi.conj() * corr))).real
        loss.backward()
        return loss.detach()

    def run_progress(self) -> None:
        begin_vmc = time.time_ns()
        logger.info(f"Begin VMC iteration: {time.ctime()}", master=True)
        self.grad_e_lst = [[], []]
        self.e_lst = []
        self.coeff_lst = []
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
            E0 = self.solve_eigh()[0]  # solve HC = εC
            delta_Hmat = (time.time_ns() - t1) / 1.0e09

            if self.rank == 0:
                self.e_lst.append(E0 + self.ecore)
                self.coeff_lst.append(self.total_coeff.to("cpu").numpy())
                c1 = self.ci_coeff.norm().item()
                c2 = self.nqs_coeff.norm().item()
                logger.info(f"E0: {E0}, Coeff: {c1:.6E} {c2:6E}", master=True)

            t2 = time.time_ns()
            new_term, new_term_mean = self.calculate_new_term(
                sample_state, phi_nqs, self.nqs_coeff, state_prob
            )
            delta_new_term = (time.time_ns() - t2) / 1.0e09

            # backward
            t3 = time.time_ns()
            loss = self.new_nqs_grad(
                self.model,
                sample_state,
                state_prob,
                eloc,
                eloc_mean,
                new_term,
                new_term_mean,
                self.nqs_coeff,
                E0,
                epoch,
            )
            delta_grad = (time.time_ns() - t3) / 1.0e09
            if self.rank == 0:
                logger.info(f"loss: {loss.item():.5E}", master=True)

            x = []
            for param in self.model.parameters():
                if param.grad is not None:
                    x.append(param.grad.reshape(-1).detach().to("cpu").numpy())
            x = np.concatenate(x)
            l2_grad = np.linalg.norm(x)
            max_grad = np.abs(x).max()
            self.grad_e_lst[0].append(l2_grad)
            self.grad_e_lst[1].append(max_grad)

            # update param
            t4 = time.time_ns()
            if epoch < self.max_iter - 1:
                self.opt.step()
                self.opt.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            delta_param = (time.time_ns() - t4) / 1.00e09

            if self.rank == 0:
                if epoch % self.nprt == 0 or epoch == self.max_iter - 1:
                    checkpoint_file = f"{self.prefix}-checkpoint.pth"
                    logger.info(f"Save model/opt state: -> {checkpoint_file}", master=True)
                    lr = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": self.model_raw.state_dict(),
                            "optimizer": self.opt.state_dict(),
                            "scheduler": lr,
                        },
                        checkpoint_file,
                    )

            delta = (time.time_ns() - t0) / 1.0e09
            if self.rank == 0:
                s = f"Calculating grad: {delta_grad:.3E} s, update params: {delta_param:.3E} s\n"
                s += f"Construct Hmat: {delta_Hmat:.3E} s, Calculating new-term: {delta_new_term:.3E} s\n"
                s += f"Total energy {(E0 + self.ecore):.9f} a.u., cost time {delta:.3E} s\n"
                s += f"L2-Gradient: {l2_grad:.5E}, Max-Gradient: {max_grad:.5E} \n"
                s += f"{epoch} iteration end {time.ctime()}\n"
                s += "=" * 100
                logger.info(s, master=True)

        # End iteration
        vmc_time = (time.time_ns() - begin_vmc) / 1.0e09
        if self.rank == 0:
            path = os.path.split(self.prefix)[0]
            coeff = np.asarray(self.coeff_lst)
            np.save(f"{path}/{self.ci_num}-coeff.npy", coeff)
            s = f"End VMC iteration: {time.ctime()}\n"
            s += f"total cost time: {vmc_time:.3E} s, {vmc_time/60:.3E} min {vmc_time/3600:.3E} h"
            logger.info(s, master=True)
