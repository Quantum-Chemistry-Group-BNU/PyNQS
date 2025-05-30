from __future__ import annotations

import os
import time
import torch
import numpy as np
import torch.distributed as dist

from collections.abc import Callable

from functools import partial
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor

from libs.C_extension import get_comb_hij_fused, tensor_to_onv
from utils.distributed import (
    get_rank,
    get_world_size,
    all_reduce_tensor,
    gather_tensor,
    all_gather_tensor,
    scatter_tensor,
)
from utils.public_function import (
    diff_rank_seed,
    setup_seed,
    WavefunctionLUT,
    ElectronInfo,
    ansatz_batch,
    split_batch_idx,
)
from utils.tools import sys_info
from utils.config import dtype_config
from utils.enums import ElocMethod
from utils.stats import operator_statistics
from utils.ci import CIWavefunction

from vmc.sample import ElocParams, Sampler
from vmc.energy.flip import Func

class CIAnsatz:

    def __init__(
        self,
        coeff: Tensor,
        onv: Tensor,
        sorb: int,
        nele: int,
        device: str = None,
    ) -> None:

        self.sorb = sorb
        self.coeff = coeff
        self.nele = nele
        self.device = device
        self.onv = onv
        self.WF_LUT = WavefunctionLUT(onv, coeff, sorb, device)

    def __call__(self, x: Tensor) -> Tensor:
        x = ((x + 1) / 2).byte()
        x = tensor_to_onv(x, self.sorb)
        result = torch.zeros(x.size(0), device=self.device, dtype=self.coeff.dtype)
        idx, not_idx, value = self.WF_LUT.lookup(x)
        result[idx] = value
        # result[not_idx] = (torch.rand(not_idx.size(0)) - 0.5) * 1e-8
        return result


class GFMC:
    def __init__(
        self,
        trial_wf: DDP,
        Lambda: float,
        ele_info: ElectronInfo,
        eloc_param: ElocParams,
        max_iter: int,
        n_sample: int,
        branch_interval: int,
        vmc_sample_params: dict,
        p_step: int = 50,
        CI_wf: CIWavefunction = None,
        prefix: str = "GFMC",
    ) -> None:

        self.rank = get_rank()
        self.world_size = get_world_size()
        # assert self.world_size == 1, f"dose not support {self.world_size} > 1"

        seed = 2023
        self.seed = diff_rank_seed(seed, rank=self.rank)
        logger.info(f"GFMC sample-seed: {self.seed}")

        self.ele_info = ele_info
        self.read_electron_info(self.ele_info)
        self.nqs = trial_wf
        self.Lambda = Lambda

        self.device = dtype_config.device
        if dtype_config.use_complex:
            dtype = dtype_config.complex_dtype
        else:
            dtype = dtype_config.default_dtype
        self.device = dtype_config.device
        self.dtype = dtype

        self.eloc_method = eloc_param["method"]
        # assert self.eloc_method == ElocMethod.SIMPLE
        self.use_unique = eloc_param["use_unique"]
        self.use_LUT = eloc_param["use_LUT"]
        self.eloc_param = eloc_param

        self.max_iter = max_iter
        self.n_sample = n_sample
        self.branch_interval = branch_interval
        self.p_step = p_step
        self.prefix = prefix

        self.vmc_sampler = Sampler(
            self.nqs,
            ele_info,
            dtype=self.dtype,
            use_spin_raising=False,
            spin_raising_coeff=1.0,
            only_sample=False,
            **vmc_sample_params,
        )

        self.ci_ansatz: CIAnsatz = None
        if CI_wf is not None:
            ci_ansatz = CIAnsatz(CI_wf.coeff, CI_wf.space, self.sorb, self.nele, self.device)
            self.ci_ansatz = ci_ansatz

    def read_electron_info(self, ele_info: ElectronInfo) -> None:
        self.sorb = ele_info.sorb
        self.nele = ele_info.nele
        self.no = ele_info.nele
        self.nv = ele_info.nv
        self.nob = ele_info.nob
        self.noa = ele_info.noa
        self.nva = ele_info.nva
        self.nvb = ele_info.nvb
        self.h1e: Tensor = ele_info.h1e
        self.h2e: Tensor = ele_info.h2e
        self.ecore = ele_info.ecore
        self.n_SinglesDoubles = ele_info.n_SinglesDoubles
        self.ci_space = ele_info.ci_space

    @torch.no_grad
    def _ansatz_batch(
        self,
        x: Tensor,
        func: Callable[..., Tensor],
    ) -> Tensor:
        fp_batch = self.eloc_param["fp_batch"]
        return ansatz_batch(func, x, fp_batch, self.sorb, self.device, self.dtype)

    def _calculate_green_kernel(
        self,
        x: Tensor,
        Lambda: float,
        h1e: Tensor,
        h2e: Tensor,
        ansatz: Callable[..., Tensor],
        ansatz_batch: Callable[[Callable], Tensor],
        sorb: int,
        nele: int,
        noa: int,
        nob: int,
        dtype: torch.dtype = torch.double,
        WF_LUT: WavefunctionLUT | None = None,
        use_unique: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:

        device = h1e.device
        dim: int = x.dim()
        assert dim == 2
        batch: int = x.shape[0]
        t0 = time.time_ns()
        ansatz = partial(ansatz_batch, func=ansatz)
        comb_x, comb_hij = get_comb_hij_fused(x, h1e, h2e, sorb, nele, noa, nob)

        t1 = time.time_ns()
        bra_len = comb_x.shape[2]
        t2 = time.time_ns()

        # hij = |hij|exp(i gamma)
        gamma = torch.where(comb_hij >= 0, 0, torch.pi)  # [nbatch, nSD]

        t3 = time.time_ns()
        psi_x1 = Func(ansatz, comb_x.reshape(-1, bra_len), WF_LUT, use_unique).reshape(batch, -1)
        # assert torch.allclose(psi_x1.imag, torch.zeros_like(psi_x1.imag))
        psi_x1 = psi_x1.real
        # breakpoint()
        # psi(x) = |psi(x)|exp(i phi(x)), alpha(x, x') = phi(x') - phi(x)
        phase = torch.angle(psi_x1)  # [nbatch, nSD]
        alpha = phase - phase[..., 0].unsqueeze(-1)

        # effective Hamiltonian
        mask = torch.cos(alpha + gamma) < 0.0
        # fixed-node approximation
        # sign = torch.sign(psi_x1[..., 0].unsqueeze(-1)) * torch.sign(psi_x1) * torch.sign(comb_hij)
        # assert (sign[(~torch.eq(mask, sign < 0.0))]).abs().sum().item() == 0.0
        # x' != x
        hij_eff = torch.zeros(comb_hij.shape, dtype=psi_x1.dtype, device=device)
        hij_eff.real = torch.where(mask, comb_hij, 0.0)
        # spin-flip potential
        V_sf = comb_hij[..., 0].reshape(-1)  # [nbatch]
        V_sf = V_sf + torch.sum(
            torch.where(~mask, comb_hij, 0.0) * (psi_x1 / psi_x1[..., 0].unsqueeze(-1)), -1
        )
        hij_eff[..., 0] = V_sf

        # fix-node G(x'<-x) = psi*(x')<x'|Lambda-H|x>/psi*(x)
        dirac = torch.zeros_like(hij_eff)
        dirac[..., 0] = 1
        K = dirac * Lambda - hij_eff
        green_kernel = psi_x1.conj() * K.conj() / psi_x1[..., 0].unsqueeze(-1).conj()
        eloc = ((psi_x1 / psi_x1[..., 0].unsqueeze(-1)) * hij_eff).sum(-1)  # [nbatch]
        # breakpoint()
        try:
            assert torch.all(green_kernel[..., 0] >= 0)
        except AssertionError:
            value = green_kernel[..., 0].topk(10, largest=False)[0]
            logger.error(f"Green kernel is negative, min-value: {value}")
            exit(-1)

        return eloc, green_kernel, comb_x

    def calculate_green_kernel(self, x: Tensor, Lambda: float) -> tuple[Tensor, Tensor, Tensor]:
        unique_x, inverse = torch.unique(x, dim=0, return_inverse=True)
        # logger.info(f"unique_x: {unique_x.shape}")
        # logger.info(f"x: {x.shape}")
        result = self._calculate_green_kernel(
            unique_x,
            Lambda,
            self.h1e,
            self.h2e,
            self.nqs,
            self._ansatz_batch,
            self.sorb,
            self.nele,
            self.noa,
            self.nob,
            self.dtype,
            self.WF_LUT,
            self.use_unique,
        )
        return tuple(torch.index_select(x, 0, inverse) for x in result)

    def sample_update(
        self,
        x: Tensor,
        weight: Tensor,
        comb_x: Tensor,
        green_kernel: Tensor,
        rand_num: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor]:

        beta = green_kernel.sum(-1, keepdim=True)  # [nbatch, 1]
        cum_prob = green_kernel.cumsum(-1) / beta
        # rand_num = torch.rand_like(beta)  # [nbatch, 1]
        if rand_num is None:
            rand_num = torch.rand_like(beta) 
        index = torch.searchsorted(cum_prob, rand_num, right=False).reshape(-1)

        # update sample weight
        x_new = comb_x[torch.arange(beta.size(0)), index]
        weight_new = weight * beta.squeeze()

        return x_new, weight_new, beta

    def branching(
        self,
        x: Tensor,
        weight: Tensor,
    ) -> tuple[Tensor, Tensor]:

        # batch = x.size(0)
        # xi = torch.rand(batch, device=self.device)
        # rand_prob = (torch.arange(batch, device=self.device) + xi) / batch
        # cum_prob = (weight / weight.sum(0)).cumsum(0)
        # index = torch.searchsorted(cum_prob, rand_prob, right=False).reshape(-1)

        # x_new = x[index]

        # TODO: gather rand_prod/cum_prod to master-rank
        # distributed
        x_all = gather_tensor(x, self.device, self.world_size, 0)
        w_all = gather_tensor(weight, self.device, self.world_size, 0)
        if self.rank == 0:
            x_all = torch.cat(x_all) if self.world_size > 1 else x_all[0]
            w_all = torch.cat(w_all) if self.world_size > 1 else w_all[0]
            batch = x_all.size(0)
            xi = torch.rand(batch, device=self.device)
            rand_prob = (torch.arange(batch, device=self.device) + xi) / batch
            cum_prob = (w_all / w_all.sum(0)).cumsum(0)
            index = torch.searchsorted(cum_prob, rand_prob, right=False).reshape(-1)
            logger.info(index)
            x_new = x_all[index]
        else:
            x_new: Tensor = None
        x_new = scatter_tensor(x_new, self.device, torch.uint8, self.world_size)
        # x_new = x_rank
        return x_new

    def batch_green_kernel(
        self,
        sample: Tensor,
        weight: Tensor,
        Lambda: float,
        idx_lst: list[int],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, tuple[float, float]]:
        dim = sample.size(0)
        beta = torch.empty(dim, device=sample.device, dtype=torch.float64)
        eloc = torch.empty_like(beta)
        weight_new = torch.empty_like(weight)
        sample_new = torch.empty_like(sample)
        # avoid numerical different
        rand_num = torch.rand(dim, 1, device=sample.device, dtype=torch.float64)

        time_green = 0.0
        time_sample = 0.0
        start = 0
        for end in idx_lst:
            t0 = time.time_ns()
            _sample = sample[start:end]
            _weight = weight[start:end]
            _eloc, _green_kernel, _comb_x = self.calculate_green_kernel(_sample, Lambda)

            t1 = time.time_ns()
            _sample_new, _weight_new, _beta = self.sample_update(
                _sample,
                _weight,
                _comb_x,
                _green_kernel,
                rand_num[start:end],
            )
            # _weight_new /= _weight_new.max()
            beta[start:end] = _beta.reshape(-1)
            sample_new[start:end] = _sample_new
            weight_new[start:end] = _weight_new
            eloc[start:end] = _eloc
            t2 = time.time_ns()
            time_green += (t1 - t0) / 1.0e09
            time_sample += (t2 - t1) / 1.0e09
            start = end
        return sample_new, weight_new, beta, eloc, (time_green, time_sample)

    @torch.no_grad()
    def run(self) -> None:

        if self.rank == 0:
            logger.info(f"Begin GFMC iteration: {time.ctime()}", master=True)

        self.WF_LUT = None
        # if self.ci_ansatz is None:
        if True:
            initial_state = self.ci_space[0].clone().detach()
            state, state_prob, (eloc, sloc), (eloc_mean, sloc_mean) = self.vmc_sampler.run(
                initial_state,
                epoch=0,
            )
            repeat_nums = (state_prob / self.world_size * self.vmc_sampler.n_sample).long()
            # logger.info(repeat_nums.sum())
            nums_all = gather_tensor(repeat_nums, self.device, self.world_size)
            state_all = gather_tensor(state, self.device, self.world_size)
            if self.rank == 0:
                nums_all = torch.cat(nums_all)
                state_all = torch.cat(state_all)
                assert nums_all.sum() == self.vmc_sampler.n_sample
                # logger.info(repeat_nums.sum())
                sample_all = torch.repeat_interleave(state_all, nums_all, dim=0)
            else:
                sample_all: Tensor = None

            sample_rank = scatter_tensor(sample_all, self.device, torch.uint8, self.world_size, 0)
            self.sample = sample_rank
            self.weight = torch.ones(self.sample.size(0), device=self.device)
            e_vmc = (eloc_mean + sloc_mean).real.item() + self.ecore
            # self.WF_LUT = self.vmc_sampler.WF_LUT
            from utils.public_function import WavefunctionLUT

            WF_LUT = WavefunctionLUT(
                self.vmc_sampler.WF_LUT.bra_key,
                self.vmc_sampler.WF_LUT.wf_value.real,
                self.sorb,
                self.device,
                sort=True,
            )
            self.WF_LUT = WF_LUT
        else:
            coeff = self.ci_ansatz.coeff
            space = self.ci_ansatz.onv
            prob = torch.abs(coeff)**2
            _counts = torch.multinomial(prob, self.vmc_sampler.n_sample, replacement=True)
            from utils.ci import energy_CI
            e = energy_CI(coeff, space, self.h1e, self.h2e, self.ecore, self.sorb, self.nele, 10000)
            self.sample = space[_counts]
            self.weight = torch.ones(self.sample.size(0), device=self.device, dtype=torch.double)
            self.nqs = self.ci_ansatz
            e_vmc = e
            self.WF_LUT = None

        if self.rank == 0:
            s = f"VMC initial energy: {e_vmc:.9f}\n"
            s += f"{'='*40} Begin GFMC{'='*40}"
            logger.info(s, master=True)
        nbatch = self.sample.size(0)
        e_lst = []
        e_lst_1 = []
        p_step = self.p_step
        # p_step = 50
        cumprod_beta = torch.ones(nbatch, p_step, device=self.device)
        # TODO: 增加热浴时间
        prob = torch.ones(nbatch, device=self.device) / self.sample.size(0)
        cost_time = []
        idx_lst = split_batch_idx(nbatch, min_batch=nbatch)
        for epoch in range(self.max_iter):
            # t0 = time.time_ns()

            # eloc, green_kernel, comb_x = self.calculate_green_kernel(self.sample, self.Lambda)

            # t1 = time.time_ns()

            # # if epoch >= 100:
            # #     breakpoint()
            # sample_new, weight_new, beta = self.sample_update(
            #     self.sample,
            #     self.weight,
            #     comb_x,
            #     green_kernel,
            # )

            # weight_new /= weight_new.max()
            # self.sample, self.weight = sample_new, weight_new
            # t2 = time.time_ns()

            sample_new, weight_new, beta, eloc, cost_time = self.batch_green_kernel(
                self.sample,
                self.weight,
                self.Lambda,
                idx_lst,
            )
            max_w = weight_new.max() * self.world_size
            max_b = beta.max() * self.world_size
            all_reduce_tensor([max_w, max_b], dist.ReduceOp.MAX, self.world_size, True)
            weight_new /= max_w
            self.sample, self.weight = sample_new, weight_new
            cumprod_beta[..., epoch % p_step] = beta / max_b
            beta_prod = cumprod_beta.prod(-1)

            # mixed estimate energy
            # eloc_mean = (self.weight * eloc).sum() / (self.weight.sum())
            # e_total = eloc_mean.real.item() + self.ecore

            # beta_prod = cumprod_beta.prod(-1)
            # eloc_mean = (beta_prod * eloc).sum() / beta_prod.sum()
            # cumprod_e = eloc_mean.item() + self.ecore

            # distributed
            sum_w = self.weight.sum()
            sum_b = beta_prod.sum()
            all_reduce_tensor([sum_w, sum_b], dist.ReduceOp.SUM, self.world_size, True)
            stats_eloc = operator_statistics(eloc, self.weight/sum_w, self.sample.size(0), "E")
            stats_eloc_p = operator_statistics(eloc, beta_prod/sum_b, self.sample.size(0), "E-prod")
            e_total = stats_eloc["mean"].item() + self.ecore
            cumprod_e = stats_eloc_p["mean"].item() + self.ecore
    
            e_lst.append(e_total)
            e_lst_1.append(cumprod_e)
            if self.rank == 0:
                s = f"{epoch} iteration weight_e/cumprod_e {e_total:.9f} {cumprod_e:.9f}"
                logger.info(s, master=True)

            stats_weight = operator_statistics(self.weight, prob, self.sample.size(0), "ω")
            stats_beta = operator_statistics(beta_prod, prob, self.sample.size(0), "β-prod")
            if self.rank == 0:
                s = str(stats_weight) + "\n"
                s += str(stats_beta)
                logger.info(s, master=True)

            if epoch % 50 == 0:
                logger.info(f"weight: min {self.weight.min()}")
            # 能量计算在branching 前后？？
            # branching
            if epoch > 0 and (epoch % self.branch_interval == 0):
                # eloc, green_kernel, _ = self.calculate_green_kernel(self.sample, self.Lambda)
                # eloc_mean = (self.weight * eloc).sum() / (self.weight.sum())
                # e_total = eloc_mean.real.item() + self.ecore
                # beta = green_kernel.sum(-1)
                # cumprod_beta[..., epoch % p_step] = beta.square() / beta.max(0, keepdim=True)[0]
                # eloc_mean = (cumprod_beta.prod(-1) * eloc).sum() / cumprod_beta.prod(-1).sum()
                # cumprod_e = eloc_mean.item() + self.ecore

                # if self.rank == 0:
                #     s = f"{epoch//self.branch_interval} branching total energy {e_total:.9f} {cumprod_e:.9f}"
                #     logger.info(s, master=True)

                # TODO: Buonaura and Sorella, Physical Review B 57, 11446–11456 (1998).
                # 前后两次计算能量 确认branching 没问题
                # self.sample, self.weight = self.branching(self.sample, self.weight)
                # logger.info(f"sample-old: {sample_old}")
                sample_new = self.branching(self.sample, self.weight)
                self.weight = torch.ones_like(self.weight)

                # eloc, green_kernel, _ = self.calculate_green_kernel(self.sample, self.Lambda)
                # eloc_mean = (self.weight * eloc).sum() / (self.weight.sum())
                # e_total = eloc_mean.real.item() + self.ecore
                # beta = green_kernel.sum(-1)
                # cumprod_beta[..., epoch % p_step] = beta.square() / beta.max(0, keepdim=True)[0]
                # eloc_mean = (cumprod_beta.prod(-1) * eloc).sum() / cumprod_beta.prod(-1).sum()
                # cumprod_e = eloc_mean.item() + self.ecore

                cumprod_beta.fill_(1)

                if self.rank == 0:
                    logger.info(
                        f"{epoch//self.branch_interval} branching total energy {e_total:.9f} {cumprod_e:.9f}",
                        master=True,
                    )
            delta_green, delta_sample = cost_time
            s = f"Calculate Green's Function {delta_green:.3E} s, "
            s += f"Update Sampling {delta_sample:.3E} s\n"
            s += f"{epoch} iteration end {time.ctime()}\n"
            s += "=" * 100
            if self.rank == 0:
                logger.info(s, master=True)
            # del green_kernel, comb_x

        if self.rank == 0:
            e_mean0 = np.mean(np.array(e_lst)[-50:])
            e_mean1 = np.mean(np.array(e_lst_1)[-50:])
            s = f"Last 50-th energy: {e_mean0:.10f} {e_mean1:.10f} vmc-e: {e_vmc:.10f}"
            logger.info(s, master=True)
