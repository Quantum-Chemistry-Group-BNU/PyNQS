from __future__ import annotations

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
    scatter_tensor,
    destroy_all_rank,
    broadcast_tensor,
    synchronize,
    all_gather_tensor,
)
from utils.public_function import (
    diff_rank_seed,
    setup_seed,
    WavefunctionLUT,
    ElectronInfo,
    ansatz_batch,
    split_batch_idx,
)
from utils.config import dtype_config
from utils.stats import operator_statistics
from utils.ci import CIWavefunction
from utils.tools import dump_input

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

        if self.rank == 0:
            logger.info(dump_input())
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
    ) -> tuple[Tensor, Tensor, Tensor, bool, int]:

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

        # hij = |hij|exp(i gamma), 0/pi
        gamma = torch.where(comb_hij >= 0, 0, torch.pi)  # [nbatch, nSD]

        t3 = time.time_ns()
        psi_x1 = Func(ansatz, comb_x.reshape(-1, bra_len), WF_LUT, use_unique).reshape(batch, -1)
        # assert torch.allclose(psi_x1.imag, torch.zeros_like(psi_x1.imag))
        psi_x1 = psi_x1.real
        # breakpoint()
        # psi(x) = |psi(x)|exp(i phi(x)), alpha(x, x') = phi(x') - phi(x)
        phase = torch.angle(psi_x1)  # [nbatch, nSD]
        alpha = phase - phase[..., 0].unsqueeze(-1)

        # effective Hamiltonian, fixed-node approximation
        mask = torch.cos(alpha + gamma) < 0.0
        mask[..., 0] = True
        # sign = torch.sign(psi_x1[..., 0].unsqueeze(-1)) * torch.sign(psi_x1) * torch.sign(comb_hij)
        # assert (sign[(~torch.eq(mask, sign < 0.0))]).abs().sum().item() == 0.0
        # x' != x
        hij_eff = torch.zeros(comb_hij.shape, dtype=psi_x1.dtype, device=device)
        hij_eff.real = torch.where(mask, comb_hij, 0.0)
        # spin-flip potential
        # V_sf = comb_hij[..., 0].reshape(-1).clone()  # [nbatch]
        V_sf = torch.sum(torch.where(~mask, comb_hij, 0.0) * (psi_x1 / psi_x1[..., 0].unsqueeze(-1)), -1)
        hij_eff[..., 0] += V_sf

        eloc = ((psi_x1 / psi_x1[..., 0].unsqueeze(-1)) * hij_eff).sum(-1)  # [nbatch]
        # eloc_1 = ((psi_x1 / psi_x1[..., 0].unsqueeze(-1)) * comb_hij).sum(-1)
        # logger.info(f"Diff: {(eloc - eloc_1).norm().item():.7f}")

        # fixed-node G(x'<-x) = psi*(x')<x'|Lambda-H|x>/psi*(x)
        dirac = torch.zeros_like(hij_eff)
        dirac[..., 0] = Lambda
        K = dirac - hij_eff
        green_kernel = psi_x1.conj() * K.conj() / psi_x1[..., 0].unsqueeze(-1).conj()

        mask = green_kernel[..., 0] < 0
        green_kernel[..., 0][mask] = 0.0
        assert torch.all(green_kernel[..., 1:] >= 0)
        stop_flag = False
        return eloc, green_kernel, comb_x, stop_flag, mask

    def calculate_green_kernel(self, x: Tensor, Lambda: float) -> tuple[Tensor, Tensor, Tensor, bool, int]:
        unique_x, inverse = torch.unique(x, dim=0, return_inverse=True)
        # logger.info(f"unique_x: {unique_x.shape}")
        # logger.info(f"x: {x.shape}")
        *result, stop_flag, dig_mask = self._calculate_green_kernel(
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
        dig_mask = torch.index_select(dig_mask, 0, inverse)
        nums_lt = dig_mask.sum()
        return *(torch.index_select(x, 0, inverse) for x in result[:3]), stop_flag, nums_lt

    def sample_update(
        self,
        x: Tensor,
        weight: Tensor,
        comb_x: Tensor,
        green_kernel: Tensor,
        rand_num: Tensor = None,
    ) -> tuple[Tensor, Tensor, Tensor, int]:

        beta = green_kernel.sum(-1, keepdim=True)  # [nbatch, 1]
        cum_prob = green_kernel.cumsum(-1) / beta
        # rand_num = torch.rand_like(beta)  # [nbatch, 1]
        if rand_num is None:
            rand_num = torch.rand_like(beta)
        index = torch.searchsorted(cum_prob, rand_num, right=False).reshape(-1)
        # update sample weight
        x_new = comb_x[torch.arange(beta.size(0)), index]
        weight_new = weight * beta.squeeze()
        accept_nums = index.nonzero().size(0)
        return x_new, weight_new, beta, accept_nums

    def batch_green_kernel(
        self,
        sample: Tensor,
        weight: Tensor,
        Lambda: float,
        idx_lst: list[int],
        sampling: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, tuple[float, float]]:
        dim = sample.size(0)
        beta = torch.empty(dim, device=sample.device, dtype=self.real_dtype)
        # only support fixed-node approximation
        eloc = torch.empty(dim, device=sample.device, dtype=self.real_dtype)
        weight_new = torch.empty_like(weight)
        sample_new = torch.empty_like(sample)
        if sampling:
            # avoid numerical different
            rand_num = torch.rand(dim, 1, device=sample.device, dtype=self.real_dtype)

        time_green = 0.0
        time_sample = 0.0
        start = 0
        accept_nums = 0
        dig_lt0 = 0
        for end in idx_lst:
            t0 = time.time_ns()
            _sample = sample[start:end]
            _weight = weight[start:end]
            _eloc, _kernel, _comb_x, stop_flag, _nums_lt = self.calculate_green_kernel(_sample, Lambda)
            eloc[start:end] = _eloc

            # green's function < 0
            # destroy_all_rank(stop_flag, sample.device)

            t1 = time.time_ns()
            if sampling:
                _sample_new, _weight_new, _beta, _nums = self.sample_update(
                    _sample,
                    _weight,
                    _comb_x,
                    _kernel,
                    rand_num[start:end],
                )
                # _weight_new /= _weight_new.max()
                beta[start:end] = _beta.reshape(-1)
                sample_new[start:end] = _sample_new
                weight_new[start:end] = _weight_new
                accept_nums += _nums
            t2 = time.time_ns()
            time_green += (t1 - t0) / 1.0e09
            time_sample += (t2 - t1) / 1.0e09
            start = end
            dig_lt0 += _nums_lt
        if sampling:
            N = sample.size(0)
            s = f"Accept nums: {N} -> {accept_nums}, {accept_nums/N * 100:.3f} % "
            s += f"Diagonal less 0: {dig_lt0}"
            logger.info(s)
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

        # simple distributed
        # x_all = gather_tensor(x, self.device, self.world_size, 0)
        # w_all = gather_tensor(weight, self.device, self.world_size, 0)
        # if self.rank == 0:
        #     x_all = torch.cat(x_all) if self.world_size > 1 else x_all[0]
        #     w_all = torch.cat(w_all) if self.world_size > 1 else w_all[0]
        #     batch = x_all.size(0)
        #     xi = torch.rand(batch, device=self.device)
        #     rand_prob = (torch.arange(batch, device=self.device) + xi) / batch
        #     cum_prob = (w_all / w_all.sum(0)).cumsum(0)
        #     index = torch.searchsorted(cum_prob, rand_prob, right=False).reshape(-1)
        #     x_new = x_all[index]
        # else:
        #     x_new: Tensor = None
        # x_new = scatter_tensor(x_new, self.device, torch.uint8, self.world_size)

        w_sum = all_gather_tensor(weight.sum(0, keepdim=True), self.device, self.world_size)
        w_sum_all = torch.cat(w_sum).cumsum(0)
        x_size = torch.tensor([x.size(0)], dtype=torch.int64, device=self.device)
        x_size_all = all_gather_tensor(x_size, self.device, self.world_size)
        x_size_all = torch.cat(x_size_all).cumsum(0)

        # rank: rand-prod
        offset = x_size_all[self.rank] - x_size_all[0]
        batch = x.size(0)
        xi = torch.rand(batch, device=self.device)  
        # logger.info(f"seed: {self.seed}") # check rank random-seed
        rand_prob = (torch.arange(offset, batch + offset, device=self.device) + xi) / x_size_all[-1]

        # rank cum-prod
        pre_sum = 0 if self.rank == 0 else w_sum_all[self.rank - 1]
        offset = pre_sum / w_sum_all[-1]
        cum_prob = (weight / w_sum_all[-1]).cumsum(0) + offset
        cum_prob.clamp_(max=1.0)

        # comm batch * (8 * bra_len + 8 + 8) bytes
        x_all = gather_tensor(x, self.device, self.world_size, 0)
        cum_prob_all = gather_tensor(cum_prob, self.device, self.world_size, 0)
        rand_prob_all = gather_tensor(rand_prob, self.device, self.world_size, 0)
        # w_all = gather_tensor(weight, self.device, self.world_size, 0)

        if self.rank == 0:
            x_all = torch.cat(x_all) if self.world_size > 1 else x_all[0]
            cum_prob_all = torch.cat(cum_prob_all) if self.world_size > 1 else cum_prob_all[0]
            rand_prob_all = torch.cat(rand_prob_all) if self.world_size > 1 else rand_prob_all[0]
            index = torch.searchsorted(cum_prob_all, rand_prob_all, right=False).reshape(-1)
            x_new = x_all[index]
            # w_all = torch.cat(w_all) if self.world_size > 1 else w_all[0]
            # cum_prob0 = (w_all / w_all.sum(0)).cumsum(0)
            # assert torch.allclose(cum_prob0, cum_prob_all)
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

        # save energy/max-weight, max-beta
        nbatch = self.sample.size(0)
        e_lst = []
        e_lst_1 = []
        p_step = self.p_step
        # p_step = 50
        # max_w = torch.zeros(self.max_iter, device=self.device)
        # max_b = torch.zeros(self.max_iter, device=self.device)
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
                sampling=True,
            )

            delta_CE = self.Lambda - e_vmc
            if self.rank == 0:
                logger.info(f"Λ-E: {delta_CE:.5f}", master=True)
            if delta_CE < 0:
                destroy_all_rank(True, self.device)

            # mixed estimate energy
            beta_prod = cumprod_beta.prod(-1)
            _sum_w = self.weight.sum()
            _sum_b = beta_prod.sum()
            all_reduce_tensor([_sum_w, _sum_b], dist.ReduceOp.SUM, self.world_size, True)
            stats_eloc = operator_statistics(eloc, self.weight / _sum_w, self.sample.size(0), "E")
            stats_eloc_p = operator_statistics(eloc, beta_prod / _sum_b, self.sample.size(0), "E-prod")
            e_total = stats_eloc["mean"].real.item() + self.ecore
            cumprod_e = stats_eloc_p["mean"].real.item() + self.ecore
            e_se = stats_eloc["se"].real.item()
            e_lst.append(e_total)
            e_lst_1.append(cumprod_e)

            # update sample
            _max_w = weight_new.max() * self.world_size
            _max_b = beta.max() * self.world_size
            all_reduce_tensor([_max_w, _max_b], dist.ReduceOp.MAX, self.world_size, True)
            # cumprod_beta[..., epoch % p_step] = beta / _max_b
            cumprod_beta[..., epoch % p_step] = beta / delta_CE
            weight_new /= delta_CE
            self.sample, self.weight = sample_new, weight_new

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
            # branching
            if epoch > 0 and (epoch % self.branch_interval == 0):
                # Buonaura and Sorella, Physical Review B 57, 11446–11456 (1998).
                start_branch = time.time_ns()
                weight_new = torch.ones_like(self.weight)
                _sum_w = torch.tensor(self.weight.size(0) * 1.0, device=self.device)
                all_reduce_tensor([_sum_w], dist.ReduceOp.SUM, self.world_size, True)
                sample_new: Tensor = None
                n_reconfigure = 0
                max_reconfigure = 1
                while True:
                    sample_new = self.branching(self.sample, self.weight)

                    eloc_new = self.batch_green_kernel(
                        sample_new,
                        weight_new,
                        self.Lambda,
                        idx_lst,
                        sampling=True,
                    )[-2]
                    stats_eloc = operator_statistics(eloc_new, weight_new / _sum_w, sample_new.size(0), "E")
                    e_total_new = stats_eloc["mean"].real.item() + self.ecore
                    diff = abs(e_total_new - e_total) * 1000
                    n_reconfigure += 1
                    threshold = e_se * 3.0 * 1000
                    if diff <= threshold:
                        if self.rank == 0:
                            s = f"Finished reconfigure, Delta-E: {diff:.5f} mHa < {threshold:.5f} mHa"
                            logger.info(s, master=True)
                        break
                    else:
                        if self.rank == 0:
                            s = f"Continued reconfigure, Delta-E: {diff:.5f} mHa > {threshold:.5f} mHa"
                            logger.info(s, master=True)
                        self.seed += 1
                        setup_seed(self.seed)
                    if n_reconfigure >= max_reconfigure:
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
