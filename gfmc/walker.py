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
    destroy_all_rank,
    broadcast_tensor,
    synchronize,
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
        branch_interval: int,
        vmc_sample_params: dict,
        p_step: int = 50,
        CI_wf: CIWavefunction = None,
        prefix: str = "GFMC",
        green_batch: int = 30000,
        use_vmc_sample: bool = True,
        n_sample: int = 10000,
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
        self.real_dtype = dtype_config.default_dtype
        self.complex_dtype = dtype_config.complex_dtype
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
        self.green_batch = int(green_batch)
        self.use_vmc_sample = use_vmc_sample
        assert self.green_batch == -1 or green_batch > 0

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
    ) -> tuple[Tensor, Tensor, Tensor, bool]:

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

        # fixed-node G(x'<-x) = psi*(x')<x'|Lambda-H|x>/psi*(x)
        dirac = torch.zeros_like(hij_eff)
        dirac[..., 0] = Lambda
        K = dirac - hij_eff
        green_kernel = psi_x1.conj() * K.conj() / psi_x1[..., 0].unsqueeze(-1).conj()
        eloc = ((psi_x1 / psi_x1[..., 0].unsqueeze(-1)) * hij_eff).sum(-1)  # [nbatch]

        stop_flag = False
        try:
            assert torch.all(green_kernel[..., 0] >= 0)
        except AssertionError:
            value, index = green_kernel[..., 0].topk(5, largest=False)
            s = f"Green kernel is negative, min-value: {value}\n"
            s += f"x: {x[index]}\n"
            s += f"psi: {psi_x1[index, 0]}\n"
            s += f"eloc: {eloc[index]}\n"
            s += f"idx: {index}"
            logger.error(s)
            # exit(-1)
            stop_flag = True

        return eloc, green_kernel, comb_x, stop_flag

    def calculate_green_kernel(self, x: Tensor, Lambda: float) -> tuple[Tensor, Tensor, Tensor, bool]:
        unique_x, inverse = torch.unique(x, dim=0, return_inverse=True)
        # logger.info(f"unique_x: {unique_x.shape}")
        # logger.info(f"x: {x.shape}")
        *result, stop_flag = self._calculate_green_kernel(
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
        return *tuple(torch.index_select(x, 0, inverse) for x in result[:3]), stop_flag

    def sample_update(
        self,
        x: Tensor,
        weight: Tensor,
        comb_x: Tensor,
        green_kernel: Tensor,
        rand_num: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor, float]:

        beta = green_kernel.sum(-1, keepdim=True)  # [nbatch, 1]
        cum_prob = green_kernel.cumsum(-1) / beta
        # rand_num = torch.rand_like(beta)  # [nbatch, 1]
        if rand_num is None:
            rand_num = torch.rand_like(beta)
        index = torch.searchsorted(cum_prob, rand_num, right=False).reshape(-1)
        # update sample weight
        x_new = comb_x[torch.arange(beta.size(0)), index]
        weight_new = weight * beta.squeeze()
        accept_rate = index.nonzero().size(0) / index.size(0)
        return x_new, weight_new, beta, accept_rate

    def batch_green_kernel(
        self,
        sample: Tensor,
        weight: Tensor,
        Lambda: float,
        idx_lst: list[int],
        not_sampling: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, tuple[float, float]]:
        dim = sample.size(0)
        beta = torch.empty(dim, device=sample.device, dtype=self.real_dtype)
        # only support fixed-node approximation
        eloc = torch.empty(dim, device=sample.device, dtype=self.real_dtype)
        weight_new = torch.empty_like(weight)
        sample_new = torch.empty_like(sample)
        if not_sampling:
            ...
        else:
            # avoid numerical different
            rand_num = torch.rand(dim, 1, device=sample.device, dtype=self.real_dtype)

        time_green = 0.0
        time_sample = 0.0
        start = 0
        rates = 0.0
        for end in idx_lst:
            t0 = time.time_ns()
            _sample = sample[start:end]
            _weight = weight[start:end]
            _eloc, _green_kernel, _comb_x, stop_flag = self.calculate_green_kernel(_sample, Lambda)
            eloc[start:end] = _eloc

            # green's function < 0
            destroy_all_rank(stop_flag, sample.device)

            t1 = time.time_ns()
            if not_sampling:
                ...
            else:
                _sample_new, _weight_new, _beta, _rate = self.sample_update(
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
                rates += _rate
            t2 = time.time_ns()
            time_green += (t1 - t0) / 1.0e09
            time_sample += (t2 - t1) / 1.0e09
            start = end
        if not not_sampling:
            logger.info(f"Accept rates: {rates * 100 /len(idx_lst):.3f}%")
        return sample_new, weight_new, beta, eloc, (time_green, time_sample)

    def branching(
        self,
        x: Tensor,
        weight: Tensor,
    ) -> Tensor:

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
            x_new = x_all[index]
        else:
            x_new: Tensor = None
        x_new = scatter_tensor(x_new, self.device, torch.uint8, self.world_size)
        return x_new

    @torch.no_grad()
    def run(self) -> None:

        if self.rank == 0:
            logger.info(f"Begin GFMC iteration: {time.ctime()}", master=True)

        start_gfmc = time.time_ns()
        self.WF_LUT = None
        if True:
            initial_state = self.ci_space[0].clone().detach()
            state, state_prob, (eloc, sloc), (eloc_mean, sloc_mean) = self.vmc_sampler.run(
                initial_state,
                epoch=0,
            )
            if self.use_vmc_sample:
                # repeat_nums = (state_prob / self.world_size * self.vmc_sampler.n_sample).long()
                # nums_all = gather_tensor(repeat_nums, self.device, self.world_size)
                state_all = gather_tensor(state, self.device, self.world_size)
                if self.rank == 0:
                    nums_all = self.vmc_sampler.all_sample_counts
                    state_all = torch.cat(state_all) if self.world_size > 1 else state_all[0]
                    assert nums_all.sum() == self.vmc_sampler.n_sample
                    # nums_all = torch.cat(nums_all) if self.world_size > 1 else nums_all[0]
                    # assert torch.allclose(self.vmc_sampler.all_sample_counts, nums_all)
                    sample_all = torch.repeat_interleave(state_all, nums_all, dim=0)
                else:
                    sample_all: Tensor = None
                e_vmc = (eloc_mean + sloc_mean).real.item() + self.ecore
            else:
                state_prob /= self.world_size
                prob_all = gather_tensor(state_prob, self.device, self.world_size)
                state_all = gather_tensor(state, self.device, self.world_size)
                eloc_all = gather_tensor(eloc, self.device, self.world_size)
                sloc_all = gather_tensor(sloc, self.device, self.world_size)
                if self.rank == 0:
                    prob_all = torch.cat(prob_all) if self.world_size > 1 else prob_all[0]
                    state_all = torch.cat(state_all) if self.world_size > 1 else state_all[0]
                    eloc_all = torch.cat(eloc_all) if self.world_size > 1 else eloc_all[0]
                    sloc_all = torch.cat(sloc_all) if self.world_size > 1 else sloc_all[0]
                    _counts = torch.multinomial(prob_all, self.n_sample, replacement=True)
                    sample_all = state_all[_counts]
                    eloc = eloc_all[_counts]
                    sloc = sloc_all[_counts]
                    e_vmc = (eloc + sloc).mean().real
                    # _, _count = _counts.unique(sorted=True, return_counts=True)
                    # logger.info(f"unique-counts: {_count.shape}")
                    # logger.info(f"prob sum: {prob_all.sum()}")
                else:
                    sample_all: Tensor = None
                    e_vmc = None

                e_vmc = broadcast_tensor(e_vmc, self.device, self.real_dtype) + self.ecore

            sample_rank = scatter_tensor(sample_all, self.device, torch.uint8, self.world_size, 0)
            self.sample = sample_rank
            self.weight = torch.ones(self.sample.size(0), device=self.device)
            self.WF_LUT = self.vmc_sampler.WF_LUT
            # WF_LUT = WavefunctionLUT(
            #     self.vmc_sampler.WF_LUT.bra_key,
            #     self.vmc_sampler.WF_LUT.wf_value.real,
            #     self.sorb,
            #     self.device,
            #     sort=True,
            # )
            # self.WF_LUT = WF_LUT
        else:
            coeff = self.ci_ansatz.coeff
            space = self.ci_ansatz.onv
            prob = torch.abs(coeff) ** 2
            _counts = torch.multinomial(prob, self.vmc_sampler.n_sample, replacement=True)
            from utils.ci import energy_CI

            e = energy_CI(coeff, space, self.h1e, self.h2e, self.ecore, self.sorb, self.nele, 10000)
            self.sample = space[_counts]
            self.weight = torch.ones(self.sample.size(0), device=self.device, dtype=torch.double)
            self.nqs = self.ci_ansatz
            e_vmc = e
            self.WF_LUT = None

        if self.rank == 0:
            n_sample = self.vmc_sampler.n_sample if self.use_vmc_sample else self.n_sample
            s = f"VMC initial energy: {e_vmc:.9f}\n"
            s += f"{'='*40} Begin GFMC{'='*40}\n"
            s += f"GFMC sample-nums: {n_sample}"
            logger.info(s, master=True)
        nbatch = self.sample.size(0)
        e_lst = []
        e_lst_1 = []
        p_step = self.p_step
        # p_step = 50
        cumprod_beta = torch.ones(nbatch, p_step, device=self.device)
        prob = torch.ones(nbatch, device=self.device) / self.sample.size(0)
        cost_time = []
        min_batch = nbatch if self.green_batch == -1 else self.green_batch
        idx_lst = split_batch_idx(nbatch, min_batch=min_batch)
        for epoch in range(self.max_iter):
            setup_seed(self.seed + epoch)
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

            # mixed estimate energy
            sum_w = self.weight.sum()
            sum_b = beta_prod.sum()
            all_reduce_tensor([sum_w, sum_b], dist.ReduceOp.SUM, self.world_size, True)
            stats_eloc = operator_statistics(eloc, self.weight / sum_w, self.sample.size(0), "E")
            stats_eloc_p = operator_statistics(eloc, beta_prod / sum_b, self.sample.size(0), "E-prod")
            e_total = stats_eloc["mean"].real.item() + self.ecore
            cumprod_e = stats_eloc_p["mean"].real.item() + self.ecore

            e_lst.append(e_total)
            e_lst_1.append(cumprod_e)
            if self.rank == 0:
                s = f"{epoch} iteration weight_e/cumprod_e {e_total:.9f} {cumprod_e:.9f}\n"
                s += str(stats_eloc) + "\n"
                s += str(stats_eloc_p)
                logger.info(s, master=True)

            stats_weight = operator_statistics(self.weight, prob, self.sample.size(0), "ω")
            stats_beta = operator_statistics(beta_prod, prob, self.sample.size(0), "β-prod")
            if self.rank == 0:
                s = str(stats_weight) + "\n"
                s += str(stats_beta)
                logger.info(s, master=True)

            if epoch % 50 == 0:
                logger.info(f"weight: min {self.weight.min():.4E}")
            # 能量计算在branching 前后？？
            # branching
            if epoch > 0 and (epoch % self.branch_interval == 0):
                # Buonaura and Sorella, Physical Review B 57, 11446–11456 (1998).
                start_branch = time.time_ns()
                weight_new = torch.ones_like(self.weight)
                sum_w = torch.tensor(self.weight.size(0) * 1.0, device=self.device)
                all_reduce_tensor([sum_w], dist.ReduceOp.SUM, self.world_size, True)
                sample_new: Tensor = None
                n_reconfigure = 0
                max_reconfigure = 10
                while True:
                    sample_new = self.branching(self.sample, self.weight)

                    eloc_new = self.batch_green_kernel(
                        sample_new,
                        weight_new,
                        self.Lambda,
                        idx_lst,
                        not_sampling=True,
                    )[-2]
                    stats_eloc = operator_statistics(eloc_new, weight_new / sum_w, sample_new.size(0), "E")
                    e_total_new = stats_eloc["mean"].real.item() + self.ecore
                    diff = abs(e_total_new - e_total) * 1000
                    n_reconfigure += 1
                    if diff <= 0.1:
                        if self.rank == 0:
                            s = f"Finished reconfigure, Delta-E: {diff:.5f} mHa < 0.1 mHa"
                            logger.info(s, master=True)
                        break
                    else:
                        if self.rank == 0:
                            s = f"Continued reconfigure, Delta-E: {diff:.5f} mHa > 0.1 mHa"
                            logger.info(s, master=True)
                        self.seed += 1
                        setup_seed(self.seed)
                    if n_reconfigure > max_reconfigure:
                        if self.rank == 0:
                            logger.info(f"Out of max reconfigure times {max_reconfigure}", master=True)
                        break

                self.sample = sample_new
                self.weight = weight_new
                cumprod_beta.fill_(1)
                end_branch = time.time_ns()
                if self.rank == 0:
                    delta = (end_branch - start_branch) / 1e09
                    logger.info(f"Completed Branching {delta:.3E} s", master=True)

            delta_green, delta_sample = cost_time
            if self.rank == 0:
                s = f"Calculate Green's Function {delta_green:.3E} s, "
                s += f"Update Sampling {delta_sample:.3E} s\n"
                s += f"{epoch} iteration end {time.ctime()}\n"
                s += "=" * 100
                logger.info(s, master=True)

        total_time = (time.time_ns() - start_gfmc) / 1.0e09
        synchronize()
        if self.rank == 0:
            e_mean0 = np.mean(np.array(e_lst)[-50:])
            e_mean1 = np.mean(np.array(e_lst_1)[-50:])
            s = f"End GFMC iteration: {time.ctime()} "
            s += f"total cost time: {total_time:.3E} s, "
            s += f"{total_time/60:.3E} min {total_time/3600:.3E} h\n"
            s += f"Last 50-th energy: {e_mean0:.10f} {e_mean1:.10f} vmc-e: {e_vmc:.10f}\n"
            s += f"Delta-E: {(e_vmc - e_mean0)* 1000:.3f}/{(e_vmc - e_mean1)* 1000:.3f} mHa"
            logger.info(s, master=True)
