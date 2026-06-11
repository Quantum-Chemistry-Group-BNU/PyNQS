from __future__ import annotations

import os
import time
import torch
import numpy as np
from functools import partial
from contextlib import nullcontext

from collections.abc import Callable
from typing import Literal
from scipy import special
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import Optimizer
from torch import nn, Tensor
from torch.optim.lr_scheduler import LRScheduler

from pynqs.libs.C_extension import unpackbits
from pynqs.utils.enums import SampleMethod
from pynqs.utils.hamiltonian import ElectronInfo
from pynqs.distributed.comm import (
    all_gather_tensor,
    get_rank,
    get_world_size,
    processes_synchronize,
    all_reduce_tensor,
    broadcast_tensor,
    scatter_tensor,
    AllReduceFunc,
)
from pynqs.utils.public_function import ansatz_batch, diff_rank_seed, split_batch_idx, torch_unique_index
from pynqs.stats.mc_stats import operator_statistics
from pynqs.config import dtype_config
from pynqs.utils.tools import VERSION, dump_input, sys_info
from pynqs.sample import ARParams, BaseSampler, MCMCParams, ExactParams, CUSTOMParams
from pynqs.sample import SAMPLER_MAPPING, sampler_string

Params = ARParams | MCMCParams | CUSTOMParams | ExactParams

from pynqs.utils.public_function import spin_flip_onv


@torch.no_grad
def compute_fidelity(
    phi: Tensor,
    psi_on_phi: Tensor,
    psi: Tensor,
    phi_on_psi: Tensor,
    prob_phi: Tensor,
    prob_psi: Tensor,
    Ns_phi: int,
    Ns_psi: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    ratio_phi = psi_on_phi / phi
    ratio_psi = phi_on_psi / psi

    stats_ratio_phi = operator_statistics(ratio_phi, prob_phi, Ns_phi, "ψ(n)/ϕ(n)")
    stats_ratio_psi = operator_statistics(ratio_psi, prob_psi, Ns_psi, "ϕ(n)/ψ(n)")
    fidelity = (stats_ratio_phi["mean"] * stats_ratio_psi["mean"]).real
    # if fidelity >= 1:
    #    logger.info(f"real fidelity : {fidelity:.4E} and scaling to 1")
    #    scaling = fidelity
    #    ratio_phi = (psi_on_phi / phi) / scaling
    # stats_ratio_phi = operator_statistics(ratio_phi, prob_phi, Ns_phi, "ψ(n)/ϕ(n)")

    if get_rank() == 0:
        logger.info(str(stats_ratio_phi), master=True)
        logger.info(str(stats_ratio_psi), master=True)
    fidelity = (stats_ratio_phi["mean"] * stats_ratio_psi["mean"]).real
    # breakpoint()

    return fidelity, ratio_phi, ratio_psi, stats_ratio_phi["mean"], stats_ratio_psi["mean"]


def compute_grad_fidelity(
    model: DDP,
    fidelity: Tensor,
    state_phi: Tensor,
    state_psi: Tensor,
    ratio_phi: Tensor,
    ratio_phi_mean: Tensor,
    phi_prob: Tensor,
    ratio_psi: Tensor,
    ratio_psi_mean: Tensor,
    psi_prob: Tensor,
    dtype: torch.dtype,
    scaling_factor: float = 1.0,
):
    # Notice: DDP all reduce
    psi_prob = psi_prob * get_world_size()
    phi_prob = phi_prob * get_world_size()
    # logger.info(f"phi_prob: {phi_prob.sum()}")
    # logger.info(f"psi_prob: {psi_prob.sum()}")
    if True:
        state_phi = state_phi.requires_grad_()
        state_psi = state_psi.requires_grad_()
        call_abs = lambda x: torch.abs(x) if not dtype.is_complex else x
        log_phi = call_abs(model(state_phi) / scaling_factor).to(dtype).log()
        log_phi_on_psi = call_abs(model(state_psi) / scaling_factor).to(dtype).log()

        # loss1 = (log_phi.conj() * ratio_phi * phi_prob).sum() * ratio_psi_mean
        # loss2 = (log_phi_on_psi * ratio_psi * psi_prob).sum() * ratio_phi_mean
        # loss3 = fidelity * (log_phi.conj() * phi_prob).sum()
        # loss4 = fidelity * (log_phi * phi_prob).sum()
        # loss = -(loss1 + loss2 - loss3 - loss4).real
        # loss13 = (log_phi.conj() * (ratio_phi - ratio_phi_mean) * phi_prob).sum() * ratio_psi_mean
        # loss2 = (log_phi_on_psi * ratio_psi * psi_prob).sum() * ratio_phi_mean
        # loss4 = fidelity * (log_phi * phi_prob).sum()
        # loss = -(loss13 + loss2 - loss4).real
        # fidelity = ratio_psi_mean * ratio_phi_mean
        # -ln Fidelity
        loss13 = (log_phi.conj() * (ratio_phi - ratio_phi_mean) * phi_prob).sum() / ratio_phi_mean
        loss2 = (log_phi_on_psi * ratio_psi * psi_prob).sum() / ratio_psi_mean
        loss4 = (log_phi * phi_prob).sum()
        loss = -(loss13 + loss2 - loss4).real
    else:
        state_phi = state_phi.requires_grad_()
        phi = self.model(state_phi).to(dtype)
        loss = -((phi * psi).sum() ** 2) / ((phi * phi).sum() * (psi * psi).sum())
        # breakpoint()
    logger.info(f"loss : {loss.item():.3f}")
    loss.backward()


@torch.no_grad
def compute_alpha(
    phi: Tensor,
    psi_on_phi: Tensor,
    psi: Tensor,
    phi_on_psi: Tensor,
    prob_phi: Tensor,
    prob_psi: Tensor,
) -> float:
    # w(n) = P_\psi(n) + P_{\phi_\theta(n)}
    # \sum_n w(n) \phi^2(n)
    term_phi = (prob_phi * phi * phi.conj()).sum()
    term_psi = (prob_psi * phi_on_psi * phi_on_psi.conj()).sum()
    term = term_phi + term_psi
    all_reduce_tensor(term)

    # \sum_n w(n)\phi_\theta(n) \psi(n)
    term1_psi = (prob_psi * psi * phi_on_psi).sum()
    term1_phi = (prob_phi * psi_on_phi * phi).sum()
    term1 = term1_phi + term1_psi
    all_reduce_tensor(term1)

    alpha = term1 / term
    return alpha.item()


@torch.no_grad()
def compute_ovlp_weight(
    phi: Tensor,
    psi_on_phi: Tensor,
    psi: Tensor,
    phi_on_psi: Tensor,
    prob_phi: Tensor,
    prob_psi: Tensor,
    Ns_phi: int,
    Ns_psi: int,
) -> tuple[Tensor, float, Tensor, Tensor, Tensor, Tensor]:
    # L = \sum_n [P_\psi(n) + P_{\phi_\theta(n)}] \left(\alpha\phi_\theta(n) - \psi(n)\right)^2
    alpha = compute_alpha(
        phi,
        psi_on_phi,
        psi,
        phi_on_psi,
        prob_phi,
        prob_psi,
    )

    cross_phi = (alpha * phi - psi_on_phi).abs().pow(2)
    cross_psi = (alpha * phi_on_psi - psi).abs().pow(2)
    stats_cross_phi = operator_statistics(cross_phi, prob_phi, Ns_phi, "αϕ(n)-ψ(n)")
    stats_cross_psi = operator_statistics(cross_psi, prob_psi, Ns_psi, "αϕ(m)-ψ(m)")

    if get_rank() == 0:
        logger.info(str(stats_cross_phi), master=True)
        logger.info(str(stats_cross_psi), master=True)
    ovlp = (stats_cross_phi["mean"] + stats_cross_psi["mean"]).real
    logger.info(f"L: {ovlp:.4E}, alpha: {alpha:4E}")

    return ovlp, alpha, cross_phi, cross_psi, stats_cross_phi["mean"], stats_cross_psi["mean"]


def compute_grad_weight(
    model: DDP | Callable[[Tensor], Tensor],
    psi_model: Callable[[Tensor], Tensor],
    prob_phi: Tensor,
    prob_psi: Tensor,
    state_phi: Tensor,
    state_psi: Tensor,
    cross_phi: Tensor,
    cross_phi_mean: Tensor,
    alpha: float,
    dtype: torch.dtype,
) -> None:
    # Notice: DDP all reduce
    prob_psi = prob_psi * get_world_size()
    prob_phi = prob_phi * get_world_size()
    if True:
        with torch.no_grad():
            # \left(\alpha\phi_\theta(n) - \psi(n)\right)
            res_phi = alpha * model(state_phi) - psi_model(state_phi)
            res_psi = alpha * model(state_psi) - psi_model(state_psi)

        state_phi = state_phi.requires_grad_()
        state_psi = state_psi.requires_grad_()
        call_abs = lambda x: torch.abs(x) if not dtype.is_complex else x
        # log_phi = call_abs(model(state_phi)).to(dtype).log()
        # log_phi_on_psi = call_abs(model(state_psi)).to(dtype).log()

        phi = model(state_phi)
        phi_on_psi = model(state_psi)

        # w(n) = P_\psi(n) + P_{\phi_\theta(n)}
        # \sum_n w(n) 2\left(\alpha\phi_\theta(n) - \psi(n)\right)
        loss1_phi = 2 * alpha * (res_phi * prob_phi * phi).sum()
        loss1_psi = 2 * alpha * (res_psi * prob_psi * phi_on_psi).sum()
        loss1 = loss1_phi + loss1_psi

        log_phi = call_abs(phi).to(dtype).log()
        loss2_ln = (log_phi * prob_phi).sum()
        # res_phi_pow = res_phi.conj() * res_phi
        # res_mean = (res_phi_pow * prob_phi).sum()  # All-reduce
        # loss2_cov = (log_phi * res_phi_pow * prob_phi).sum()
        # loss2 = 2 * (loss2_cov - res_mean * loss2_ln)
        loss2_cov = (log_phi * cross_phi * prob_phi).sum()
        loss2 = 2 * (loss2_cov - cross_phi_mean * loss2_ln)
        # breakpoint()
        loss = (loss1 + loss2).real
        loss.backward()
    else:
        assert get_world_size() == 1
        state_phi = state_phi.requires_grad_()
        state_psi = state_psi.requires_grad_()
        with torch.no_grad():
            psi_on_phi = psi_model(state_phi)
            psi = psi_model(state_psi)
        phi = model(state_phi)
        phi_on_psi = model(state_psi)
        prob_phi = phi * phi.conj() / phi.norm() ** 2
        loss1 = (prob_phi * (alpha * phi - psi_on_phi) ** 2).sum()
        loss2 = (prob_psi * (alpha * phi_on_psi - psi) ** 2).sum()
        loss = (loss1 + loss2).real
        loss.backward()
        logger.info(f"brute backward, loss: {loss:.4E}")


class PreTrain:
    METHOD = ("Fidelity", "Weight", "Overlap", "LDS")

    def __init__(
        self,
        max_iter: int,
        optimizer: Optimizer,
        phi_model: DDP | Callable[[Tensor], Tensor],
        phi_sample_method: SampleMethod | sampler_string,
        phi_params: Params,
        psi_model: Callable[..., Tensor],
        psi_sample_method: SampleMethod | sampler_string,
        psi_params: Params,
        device: str,
        seed: int,
        ele_info: ElectronInfo,
        method: str = "fidelity",
        interval: int = 50,
        prefix: str = None,
        clip_grad_scheduler: Callable[[int], float] | None = None,
        lr_scheduler: list[LRScheduler] = None,
    ) -> None:
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.model = phi_model
        use_LUT = True
        self.dtype = dtype_config.default_dtype
        self.device = device
        self.read_electron_info(ele_info)

        method = method.capitalize()
        assert method in self.METHOD
        self.method: Literal["Fidelity", "Weight", "Overlap", "LDS"] = method
        self.max_iter: int = max_iter
        # self.seed = diff_rank_seed(seed, self.rank)
        self.seed = seed
        self.interval = interval
        self.prefix = prefix

        # optimizer
        self.opt = optimizer
        self.clip_grad_scheduler = clip_grad_scheduler
        self.lr_scheduler: list[LRScheduler] = []
        if lr_scheduler is not None:
            if not isinstance(lr_scheduler, list):
                lr_scheduler = [lr_scheduler]
            for p in lr_scheduler:
                if not isinstance(p, LRScheduler):
                    raise TypeError(f"{type(p).__name__ } is not a LRScheduler")
                self.lr_scheduler.append(p)
        else:
            self.lr_scheduler = None

        # model to be pretrained
        self.phi_model = phi_model
        state_phi = SAMPLER_MAPPING[phi_sample_method]
        self.phi_SamplerState: BaseSampler = state_phi(
            phi_model,
            phi_params,
            self.fci_size,
            self.sorb,
            self.nele,
            self.noa,
            self.nob,
            use_LUT,
            device,
        )
        self.seed_phi = self.set_model_seed(seed, phi_params)

        # MPS
        self._psi_model = psi_model
        self.psi_model: Callable[[Tensor], Tensor] = partial(self._no_grad_model, psi_model)
        state_psi = SAMPLER_MAPPING[psi_sample_method]
        self.psi_SamplerState: BaseSampler = state_psi(
            psi_model,
            psi_params,
            self.fci_size,
            self.sorb,
            self.nele,
            self.noa,
            self.nob,
            use_LUT,
            device,
        )
        self.seed_psi = self.set_model_seed(seed, psi_params) + 1
        logger.info(f"phi/psi model random seed: {self.seed_phi}/{self.seed_psi}")

        if self.rank == 0:
            logger.info(dump_input())

    @torch.no_grad
    def _no_grad_model(self, model, *args, **kwargs):
        return model(*args, **kwargs)

    def read_electron_info(self, ele_info: ElectronInfo):
        if self.rank == 0:
            logger.info(f"Read electronic structure information From {ele_info.__name__}", master=True)
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
        n1 = special.comb(self.noa + self.nva, self.noa, exact=True)
        n2 = special.comb(self.nob + self.nvb, self.nvb, exact=True)
        self.fci_size: int = n1 * n2

    @torch.no_grad
    def ansatz_batch(
        self,
        x: Tensor,
        ansatz: Callable[[Tensor], Tensor],
        fp_batch: int = -1,
        dtype: torch.dtype = None,
    ) -> Tensor:
        if dtype is None:
            dtype = self.dtype
        return ansatz_batch(ansatz, x, fp_batch, self.sorb, self.device, dtype)

    def set_model_seed(self, seed: int, sample_params: Params) -> int:
        if isinstance(sample_params, ExactParams):
            seed_model = seed
        elif isinstance(sample_params, MCMCParams):
            seed_model = diff_rank_seed(seed, self.rank)
        elif isinstance(sample_params, ARParams):
            if sample_params.use_same_tree == True:
                seed_model = seed
            else:
                seed_model = diff_rank_seed(seed, self.rank)
        else:
            raise NotImplementedError

        return seed_model

    def train(self):
        start_pre_train = time.time_ns()
        if self.rank == 0:
            logger.info(f"Begin pre-train using fidelity {time.ctime()}\n", master=True)

        self.CostFunc_lst = []
        self.loss_lst = []
        self.grad_e_lst = [[], []]
        for epoch in range(self.max_iter):
            t0 = time.time_ns()
            # NQS
            phi_sample_unique, phi_sample_counts, phi_prob, _ = self.phi_SamplerState.run(
                epoch,
                self.seed_phi,
            )
            # MPS
            psi_sample_unique, psi_sample_counts, psi_prob, _ = self.psi_SamplerState.run(
                epoch,
                self.seed_psi,
            )

            with torch.no_grad():
                # TODO: check it in weight
                state_phi = unpackbits(phi_sample_unique, self.sorb)
                phi = self.phi_model(state_phi)
                psi_on_phi = self.psi_model(state_phi)

                state_psi = unpackbits(psi_sample_unique, self.sorb)
                psi = self.psi_model(state_psi)
                phi_on_psi = self.phi_model(state_psi)

                if False:
                    # re-scaling phi/psi
                    phi_row = phi.clone()
                    psi_row = psi.clone()
                    _phi_norm = (phi.norm() ** 2 + phi_on_psi.norm() ** 2) / 2
                    _psi_norm = (psi.norm() ** 2 + psi_on_phi.norm() ** 2) / 2
                    all_reduce_tensor([_psi_norm, _phi_norm])
                    _phi_norm = _phi_norm.sqrt()
                    _psi_norm = _psi_norm.sqrt()

                    _phi_norm = _psi_norm = 1.0
                    phi /= _phi_norm
                    phi_on_psi /= _phi_norm
                    psi /= _psi_norm
                    psi_on_phi /= _psi_norm

                    if self.rank == 0:
                        logger.info(f"scaling-factor phi/psi: {_phi_norm:.3E} {_psi_norm:.3E}", master=True)
                else:
                    phi_row = phi
                    psi_row = psi
                    _phi_norm = 1.0

            def gather_sample(n_sample_rank: Tensor) -> int:
                counts_all = all_gather_tensor(n_sample_rank, self.device)
                counts_all = torch.cat(counts_all) if self.world_size > 1 else counts_all[0]
                n_sample = counts_all.sum().item()
                return n_sample

            Ns_phi = gather_sample(phi_sample_counts)
            Ns_psi = gather_sample(psi_sample_counts)

            if self.method == "Fidelity":
                fidelity, ratio_phi, ratio_psi, ratio_phi_mean, ratio_psi_mean = compute_fidelity(
                    phi,
                    psi_on_phi,
                    psi,
                    phi_on_psi,
                    phi_prob,
                    psi_prob,
                    Ns_phi,
                    Ns_psi,
                )
                self.CostFunc_lst.append(fidelity.item())
                # check fidelity
                # with torch.no_grad():
                #     fidelity1 = ((phi * psi).sum()**2) / ((phi * phi).sum() * (psi * psi).sum())

                compute_grad_fidelity(
                    self.model,
                    fidelity,
                    state_phi,
                    state_psi,
                    ratio_phi,
                    ratio_phi_mean,
                    phi_prob,
                    ratio_psi,
                    ratio_psi_mean,
                    psi_prob,
                    self.dtype,
                    scaling_factor=_phi_norm,
                )
                cost_func = fidelity

            elif self.method == "Weight":
                ovlp, alpha, cross_phi, cross_psi, cross_phi_mean, cross_psi_mean = compute_ovlp_weight(
                    phi,
                    psi_on_phi,
                    psi,
                    phi_on_psi,
                    phi_prob,
                    psi_prob,
                    Ns_phi,
                    Ns_psi,
                )
                if epoch == self.max_iter - 1:
                    ...
                    # breakpoint()
                compute_grad_weight(
                    self.model,
                    self.psi_model,
                    phi_prob,
                    psi_prob,
                    state_phi,
                    state_psi,
                    cross_phi,
                    cross_phi_mean,
                    alpha,
                    self.dtype,
                )
                if self.rank == 0:
                    logger.info(f"alpha: {alpha:.3E}")
                cost_func = ovlp
                self.CostFunc_lst.append(ovlp.item())

            logger.info(f"cost-func: {cost_func:.5E}")
            # save the energy grad and clip-grad
            self._clip_grad_L2(epoch=epoch)

            # gradient
            x1 = []
            x2 = []
            for param in self.model.parameters():
                if param.grad is not None:
                    x1.append(param.grad.detach().norm().reshape(-1))
                    x2.append(param.grad.detach().abs().max().reshape(-1))
            x1 = torch.cat(x1)
            x2 = torch.cat(x2)
            l2_grad = x1.norm().item()
            max_grad = x2.max().item()
            self.grad_e_lst[0].append(l2_grad)
            self.grad_e_lst[1].append(max_grad)

            # update parameters
            self.update_param(epoch=epoch)

            self.update_backflow_norm(self.model, phi_row, phi_prob, "ϕ(n)")
            self.update_backflow_norm(self._psi_model, psi_row, psi_prob, "ψ(n)")

            self.save_checkpoint(epoch)
            if self.rank == 0:
                delta = (time.time_ns() - t0) / 1.0e09
                lrs = [p["lr"] for p in self.opt.param_groups]
                s = f"{self.method}: {cost_func:.4E} cost time {delta:.3E} s\n"
                s += f"L2-Gradient: {l2_grad:.5E}, Max-Gradient: {max_grad:.5E}\n"
                s += f"Learning Rate: {' '.join(['{:.5E}'.format(lr) for lr in lrs])}\n"
                s += f"{epoch} iteration end {time.ctime()}\n"
                s += "=" * 100
                logger.info(f"{s}", master=True)

        processes_synchronize()
        if self.rank == 0:
            delta = (time.time_ns() - start_pre_train) / 1.0e09
            length = min(50, len(self.CostFunc_lst))
            s = f"Last {length}-th {self.method}: {np.mean(self.CostFunc_lst[-length:]):.4E}\n"
            s += f"End {self.method} pre-train: {time.ctime()}, "
            s += f"cost time: {delta:.3E}s, {delta/60:.3E} min {delta/3600:.3E}h"
            logger.info(s, master=True)

    def train_no_sampling(
        self,
        method: Literal["Overlap", "LDS"] = "Overlap",
        batch_size: int = 1024,
        forward_batch_size: int = -1,
        n_macro: int = 1,
        ab_flip=False,
    ):
        # the lr_scheduler is n_macro *  max_iter
        start_pre_train = time.time_ns()

        self.method = method
        if self.rank == 0:
            logger.info(f"Begin pre-train using {self.method} without sampling {time.ctime()}\n", master=True)

        self.CostFunc_lst = []
        self.grad_e_lst = [[], []]
        if forward_batch_size != -1 and forward_batch_size < 1:
            raise ValueError("forward_batch_size must be -1 or a positive integer")

        def overlap_loss(phi_pred: Tensor, psi_target: Tensor, scaling: float = 1.0):
            phi_pred = phi_pred / scaling
            norm_pred = torch.sum(phi_pred * phi_pred.conj())
            norm_target = torch.sum(psi_target * psi_target.conj())
            cross1 = torch.sum(phi_pred * psi_target.conj())
            # reduce
            norm_pred = AllReduceFunc.apply(norm_pred)
            norm_target = AllReduceFunc.apply(norm_target)
            cross1 = AllReduceFunc.apply(cross1)
            # Fidelity
            overlap = cross1 * cross1.conj() / (norm_pred * norm_target)
            loss = -torch.log(overlap + 1e-12)
            return loss, overlap

        def LDS_loss(phi_M: Tensor, psi_M):
            # phi_M, psi_M: (nbatch, ndet, nqubits, nele)
            n_batch = phi_M.shape[0]
            diff = phi_M - psi_M
            LDS = torch.sqrt(torch.sum((diff * diff.conj()).real, dim=(-1, -2)))  # -> (nbatch, ndet)
            LDS = torch.sum(LDS, dim=(0, 1)) / n_batch
            loss = LDS + 1e-12
            return loss, LDS

        def overlap_loss_batch(
            samples: Tensor,
            wf: Tensor,
            scaling: float = 1.0,
        ) -> tuple[None, Tensor, Tensor]:
            # Overlap is a ratio of global sums. First get those sums, then replay
            # each micro-batch with an equivalent local surrogate gradient.
            idx_lst = [0] + split_batch_idx(samples.size(0), forward_batch_size)
            with torch.no_grad():
                norm_pred: Tensor = None
                norm_target: Tensor = None
                cross1: Tensor = None
                phi_pred_lst: list[Tensor] = []

                for start, end in zip(idx_lst[:-1], idx_lst[1:]):
                    _samples = samples[start:end]
                    _wf = wf[start:end]
                    _phi_model = self.model(_samples)
                    phi_pred_lst.append(_phi_model)
                    _phi = _phi_model / scaling
                    _norm_pred = torch.sum(_phi * _phi.conj())
                    _norm_target = torch.sum(_wf * _wf.conj())
                    _cross1 = torch.sum(_phi * _wf.conj())
                    norm_pred = _norm_pred if norm_pred is None else norm_pred + _norm_pred
                    norm_target = _norm_target if norm_target is None else norm_target + _norm_target
                    cross1 = _cross1 if cross1 is None else cross1 + _cross1

                norm_pred = AllReduceFunc.apply(norm_pred)
                norm_target = AllReduceFunc.apply(norm_target)
                cross1 = AllReduceFunc.apply(cross1)
                overlap = cross1 * cross1.conj() / (norm_pred * norm_target)
                # Keep the 1e-12 regularization in the backward formula.
                loss_scale = (overlap / (overlap + 1e-12)).real
                phi_pred = torch.cat(phi_pred_lst).detach()

            backward_scale = loss_scale * self.world_size
            for batch_idx, (start, end) in enumerate(zip(idx_lst[:-1], idx_lst[1:])):
                is_last_batch = batch_idx == len(idx_lst) - 2
                backward_context = nullcontext() if is_last_batch else self.model.no_sync()
                with backward_context:
                    _samples = samples[start:end]
                    _wf = wf[start:end]
                    _phi = self.model(_samples) / scaling
                    _norm_pred = torch.sum(_phi * _phi.conj())
                    _cross1 = torch.sum(_phi * _wf.conj())
                    loss = backward_scale * ((_norm_pred / norm_pred).real - 2 * (_cross1 / cross1).real)
                    loss.backward()

            return None, overlap, phi_pred

        self.train_samples: Tensor = None
        self.train_wf: Tensor = None
        self.train_dtype = None
        self.train_samples_flipped = None
        self.train_wf_flipped = None

        @torch.no_grad()
        def generate_train_data(k: int = 0) -> tuple[Tensor, Tensor]:
            # Generate samples using psi=MPS
            psi_sample_unique, psi_sample_counts, psi_prob, psi_lut = self.psi_SamplerState.run(
                k,
                self.seed_psi,
            )
            new_samples = psi_lut.bra_key
            new_wf = psi_lut.wf_value
            self.train_dtype = new_wf.dtype
            # new_samples = unpackbits(psi_sample_unique, self.sorb)
            # new_wf = self.psi_model(new_samples)
            # new_wf = self.ansatz_batch(psi_sample_unique, self.psi_model, 100000)
            if self.train_samples is None:
                self.train_samples = new_samples
                self.train_wf = new_wf
                if ab_flip:
                    self.train_samples_flipped = spin_flip_onv(self.train_samples, self.sorb)
                    x_flipped = unpackbits(self.train_samples_flipped, self.sorb)
                    self.train_wf_flipped = self.psi_model(x_flipped)
                if self.rank == 0:
                    s = f"Initialize train data with {new_samples.size(0)} unique-samples"
                    logger.info(s, master=True)
            else:
                # Update sample pool
                M_old = self.train_samples.size(0)
                if self.rank == 0:
                    _samples = torch.cat([new_samples, self.train_samples])
                    _wf = torch.cat([new_wf, self.train_wf])

                    sample_unique, _, unique_idx, _counts = torch_unique_index(_samples, dim=0)
                    # self.train_samples = sample_unique
                    # self.train_wf = _wf[unique_idx]
                    train_wf = _wf[unique_idx]

                    if ab_flip:
                        new_samples_flipped = spin_flip_onv(new_samples, self.sorb)
                        x_flipped = unpackbits(new_samples_flipped, self.sorb)
                        new_wf_flipped = self.psi_model(x_flipped)
                        _samples_flipped = torch.cat([new_samples_flipped, self.train_samples_flipped])
                        _wf_flipped = torch.cat([new_wf_flipped, self.train_wf_flipped])
                        sample_unique_flipped = _samples_flipped[unique_idx]
                        train_wf_flipped = _wf_flipped[unique_idx]
                else:
                    sample_unique: Tensor = None
                    train_wf: Tensor = None
                    sample_unique_flipped = None
                    train_wf_flipped = None

                self.train_wf = broadcast_tensor(train_wf, self.device, self.train_dtype)
                self.train_samples = broadcast_tensor(sample_unique, self.device, torch.uint8)
                if ab_flip:
                    self.train_wf_flipped = broadcast_tensor(train_wf_flipped, self.device, self.train_dtype)
                    self.train_samples_flipped = broadcast_tensor(
                        sample_unique_flipped, self.device, torch.uint8
                    )

                if self.rank == 0:
                    M_new = self.train_samples.size(0)
                    logger.info(f"Update unique-sample: {M_old} -> {M_new}", master=True)

        last_overlap = []
        for k in range(n_macro):
            if self.rank == 0:
                logger.info(f"Begin {k}-th macro train", master=True)

            generate_train_data(k)
            M = self.train_wf.size(0)

            if self.rank == 0:
                s = f"train data unique-sample: {M}, batch-size: {batch_size}"
                if self.method == "Overlap":
                    s += f", forward-batch-size: {forward_batch_size}"
                logger.info(s, master=True)

            for epoch in range(self.max_iter):
                t0 = time.time_ns()
                if self.rank == 0:
                    # randomly choose batch_size samples from M samples
                    idx = torch.randperm(M, device=self.device)[:batch_size]
                else:
                    idx: Tensor = None
                # scatter idx other rank
                idx = scatter_tensor(idx, self.device, torch.int64)
                if ab_flip:
                    samples = torch.cat([self.train_samples[idx], self.train_samples_flipped[idx]])
                    samples = unpackbits(samples, self.sorb)
                    wf = torch.cat([self.train_wf[idx], self.train_wf_flipped[idx]])
                else:
                    samples = unpackbits(self.train_samples[idx], self.sorb)
                    wf = self.train_wf[idx]  # psi=MPS

                # loss function
                if self.method == "Overlap":
                    if forward_batch_size < samples.size(0) and forward_batch_size != -1:
                        loss, overlap, phi_pred = overlap_loss_batch(samples, wf.detach())
                    else:
                        phi_pred = self.model(samples)  # phi=NQS
                        loss, overlap = overlap_loss(phi_pred, wf.detach())
                        loss.real.backward()
                    rep = "Overlap"
                elif self.method == "LDS":
                    phi_pred = self.model(samples)  # phi=NQS
                    phi_M = self.model(samples, True)
                    psi_M = self.psi_model(samples, True)
                    loss, overlap = LDS_loss(phi_M, psi_M)
                    rep = "orbital-coeff-loss"
                    loss.real.backward()

                cost_func = overlap.real.detach().cpu().item()
                self.CostFunc_lst.append(cost_func)
                logger.info(f"cost-func: {cost_func:.5E}")

                # save the energy grad and clip-grad
                self._clip_grad_L2(epoch + k * self.max_iter)
                x1 = []
                x2 = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        x1.append(param.grad.detach().norm().reshape(-1))
                        x2.append(param.grad.detach().abs().max().reshape(-1))
                x1 = torch.cat(x1)
                x2 = torch.cat(x2)
                l2_grad = x1.norm().item()
                max_grad = x2.max().item()
                self.grad_e_lst[0].append(l2_grad)
                self.grad_e_lst[1].append(max_grad)

                self.update_backflow_norm(self.model, phi_pred.detach(), None, "ϕ(n)")
                # update parameters
                self.update_param(epoch=epoch)

                self.save_checkpoint(epoch)
                if self.rank == 0:
                    delta = (time.time_ns() - t0) / 1.0e09
                    lrs = [p["lr"] for p in self.opt.param_groups]
                    s = f"{rep} without sampling: {cost_func:.4E} cost time {delta:.3E} s\n"
                    s += f"L2-Gradient: {l2_grad:.5E}, Max-Gradient: {max_grad:.5E}\n"
                    s += f"Learning Rate: {' '.join(['{:.5E}'.format(lr) for lr in lrs])}\n"
                    s += f"{epoch} iteration end {time.ctime()}\n"
                    s += "=" * 100
                    logger.info(f"{s}", master=True)

            processes_synchronize()
            if self.rank == 0:
                delta = (time.time_ns() - start_pre_train) / 1.0e09
                length = min(50, len(self.CostFunc_lst))
                mean = np.mean(self.CostFunc_lst[-length:])
                last_overlap.append(mean)
                s = f"Last {length}-th {self.method}: {mean:.4E}\n"
                s += f"End {self.method} pre-train: {time.ctime()}, "
                s += f"cost time: {delta:.3E}s, {delta/60:.3E} min {delta/3600:.3E}h"
                logger.info(s, master=True)

        # End N-macro:
        if self.rank == 0:
            s = "=" * 100 + "\n"
            s += f"End {n_macro} iteration pre-train, {n_macro} overlap: "
            s += " ".join(["{:.4E}".format(x) for x in last_overlap]) + "\n"
            s += f"End pre-train using overlap without sampling {time.ctime()}"
            logger.info(s, master=True)

    def update_backflow_norm(
        self,
        model: Callable[[Tensor], Tensor],
        wf: Tensor,
        prob: Tensor,
        names: str = "psi",
    ) -> None:
        model = model.module if hasattr(model, "module") else model
        if not hasattr(model, "update_normalization"):
            return None
        # all-reduce
        _l2_norm = wf.norm() ** 2
        all_reduce_tensor(_l2_norm)
        _l2_norm = _l2_norm.sqrt_()
        model.update_normalization(_l2_norm)
        if get_rank() == 0:
            logger.info(f"L2({names}): {_l2_norm.item():.4E}", master=True)

    def save_checkpoint(self, epoch: int) -> None:
        """
        save the model/opt/lr_scheduler to '.pth' file for resuming calculations
        """
        if self.rank == 0 and epoch > 0:
            if epoch % self.interval == 0 or epoch == self.max_iter - 1:
                checkpoint_file = f"{self.prefix}-{self.method}-checkpoint.pth"
                dir_path = os.path.dirname(checkpoint_file)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Save model/opt state: -> {checkpoint_file}", master=True)
                if self.lr_scheduler is None:
                    lr_scheduler = None
                else:
                    lr_scheduler = [p.state_dict() for p in self.lr_scheduler]

                torch.save(
                    {
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "optimizer": self.opt.state_dict(),
                        "scheduler": lr_scheduler,
                        "l2_grad": self.grad_e_lst[0],
                        "max_grad": self.grad_e_lst[1],
                        self.method: self.CostFunc_lst,
                        "version": VERSION,
                        "timestamp": time.ctime(),
                        "sys_info": sys_info(),
                    },
                    checkpoint_file,
                )

    def _clip_grad_L2(self, epoch: int) -> None:
        """
        clip model grad use L2-norm
        """
        if self.clip_grad_scheduler is not None:
            g0 = self.clip_grad_scheduler(epoch)
            x = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=g0, foreach=True)
            if self.rank == 0:
                logger.info(f"Clip-grad, g: {x:.4E}, L2-g0: {g0:4E}", master=True)

    def update_param(self, epoch: int) -> None:
        """
        update model param, and adjust learning rate
        """
        assert epoch <= self.max_iter - 1
        # if epoch <= self.max_iter - 1:
        before_param = [param.data.flatten().detach().clone() for param in self.model.parameters()]
        grads = [
            (
                param.grad.flatten().detach()
                if param.grad is not None
                else torch.zeros_like(param.data).flatten()
            )
            for param in self.model.parameters()
        ]
        dtheta = torch.cat(grads)
        dtheta_l2 = torch.linalg.vector_norm(dtheta).item()
        self.opt.step()
        after_param = [param.data.flatten().detach().clone() for param in self.model.parameters()]
        dtheta_opt = torch.cat([p0 - p1 for p0, p1 in zip(before_param, after_param)])
        p_diff = torch.linalg.vector_norm(dtheta_opt).item()
        den = torch.linalg.vector_norm(dtheta) * torch.linalg.vector_norm(dtheta_opt)
        angle = (
            float("nan")
            if den.item() == 0
            else torch.rad2deg(torch.acos((torch.vdot(dtheta, dtheta_opt).real / den).clamp(-1, 1))).item()
        )
        theta = torch.cat(before_param)
        theta_den = torch.linalg.vector_norm(theta) * torch.linalg.vector_norm(dtheta_opt)
        theta_angle = (
            float("nan")
            if theta_den.item() == 0
            else torch.rad2deg(
                torch.acos((torch.vdot(theta, dtheta_opt).real / theta_den).clamp(-1, 1))
            ).item()
        )
        theta_l2 = torch.linalg.vector_norm(theta).item()
        rel = float("nan") if theta_l2 == 0 else p_diff / theta_l2
        if self.rank == 0:
            s = f"|△𝜃| {dtheta_l2:.4e} -> {p_diff:.4e} (diff: {p_diff - dtheta_l2:.4e}, angle: {angle:.4e}°)"
            logger.info(s, master=True)
            logger.info(f"angle(𝜃, △𝜃): {theta_angle:.4e}°, |△𝜃|/|𝜃|: {rel:.4e}", master=True)
        self.opt.zero_grad()
        if self.lr_scheduler is not None:
            for i, p in enumerate(self.lr_scheduler):
                self.lr_scheduler[i].step()

    def plot_figure(self) -> None:
        import matplotlib.pyplot as plt
        import os

        if self.rank == 0:
            cost_func = self.method
            prefix = self.prefix if self.prefix is not None else f"{cost_func}-pre-train"
            fig = plt.figure()
            # plot ovlp and loss
            fidelity = np.array(self.CostFunc_lst)
            x = np.arange(self.max_iter)
            ax = fig.add_subplot(2, 1, 1)
            line1 = ax.plot(x, fidelity, color="cadetblue", label=cost_func)
            ax.set_ylabel(cost_func)
            ax.set_yscale("log")
            # ax1 = ax.twinx()
            # line2 = ax1.plot(x, np.abs(self.loss_lst), color="tomato", label="loss")
            # ax1.set_ylabel("Loss")
            # lines = line1 + line2
            lines = line1
            labels = [name.get_label() for name in lines]
            ax.legend(lines, labels, loc="best")

            # plot the L2-norm and max-abs of the gradients
            param_L2 = np.asarray(self.grad_e_lst[0])
            param_max = np.asarray(self.grad_e_lst[1])
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(np.arange(len(param_L2)), param_L2, label=r"$||g||$")
            ax2.plot(np.arange(len(param_max)), param_max, label=r"$||g||_{\infty}$")
            ax2.set_xlabel("Iteration Time")
            ax2.set_yscale("log")
            ax2.set_ylabel("Gradients")
            plt.title(os.path.split(prefix)[1])  # remove path
            plt.legend(loc="best")

            fig.subplots_adjust(wspace=0, hspace=0.5)
            suffix = f"-{cost_func}-pre-train"
            path = prefix + suffix
            fig.savefig(path + ".png", format="png", dpi=1000, bbox_inches="tight")
            plt.close()
            np.savez(path, fidelity, param_L2, param_max)
            # logger.info(f"Save {path}.png")
            logger.info(f"Save figure -> {path}.png", master=True)
