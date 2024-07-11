from __future__ import annotations

import __main__
import time
import random
import os
import collections
import sys
import warnings
import torch
import torch.distributed as dist
import numpy as np

from typing import Callable
from dataclasses import dataclass
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer, required
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger

from vmc.grad import energy_grad, sr_grad
from vmc.optim._base import BaseVMCOptimizer
from ci import CITrain, CIWavefunction
from utils.distributed import (
    all_reduce_tensor,
    get_rank,
    get_world_size,
    synchronize,
)
from utils import ElectronInfo, Dtype
from libs.C_extension import onv_to_tensor

# from memory_profiler import profile
# from line_profiler import LineProfiler

TORCH_VERSION: str = torch.__version__
if TORCH_VERSION >= "2.0.0":
    from torch.optim.lr_scheduler import LRScheduler
else:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


@dataclass
class VMCOptimizer(BaseVMCOptimizer):
    """
    General VMC optimization and pre-train process
    """

    def __init__(
        self,
        nqs: DDP,
        sampler_param: dict,
        electron_info: ElectronInfo,
        opt_type: Optimizer = torch.optim.Adam,
        opt_params: dict = {"lr": 0.005, "weight_decay": 0.001},
        lr_scheduler: LRScheduler = None,
        lr_sch_params: dict = None,
        max_iter: int = 2000,
        dtype: Dtype = None,
        HF_init: int = 0,
        external_model: any = None,
        check_point: str = None,
        read_model_only: bool = False,
        only_sample: bool = False,
        pre_CI: CIWavefunction = None,
        pre_train_info: dict = None,
        clean_opt_state: bool = False,
        noise_lambda: float = 0.05,
        method_grad: str = "AD",
        sr: bool = False,
        method_jacobian: str = "vector",
        interval: int = 100,
        prefix: str = "VMC",
        MAX_AD_DIM: int = -1,
        kfac: KFACPreconditioner = None,  # type: ignore
        use_clip_grad: bool = False,
        max_grad_norm: float = 1.0,
        max_grad_value: float = 1.0,
        start_clip_grad: int = None,
        clip_grad_method: str = "l2",
        clip_grad_scheduler: Callable[[int], float] = None,
        use_spin_raising: bool = False,
        spin_raising_coeff: float = 1.0,
        only_output_spin_raising: bool = False,
        spin_raising_scheduler: Callable[[int], float] = None,
    ) -> None:
        super().__init__(
            nqs=nqs,
            sampler_param=sampler_param,
            electron_info=electron_info,
            opt_type=opt_type,
            opt_params=opt_params,
            lr_scheduler=lr_scheduler,
            lr_sch_params=lr_sch_params,
            max_iter=max_iter,
            dtype=dtype,
            HF_init=HF_init,
            external_model=external_model,
            check_point=check_point,
            read_model_only=read_model_only,
            only_sample=only_sample,
            method_grad=method_grad,
            sr=sr,
            method_jacobian=method_jacobian,
            interval=interval,
            prefix=prefix,
            MAX_AD_DIM=MAX_AD_DIM,
            kfac=kfac,
            use_clip_grad=use_clip_grad,
            clip_grad_method=clip_grad_method,
            max_grad_norm=max_grad_norm,
            max_grad_value=max_grad_value,
            clip_grad_scheduler=clip_grad_scheduler,
            start_clip_grad=start_clip_grad,
            use_spin_raising=use_spin_raising,
            spin_raising_coeff=spin_raising_coeff,
            only_output_spin_raising=only_output_spin_raising,
            spin_raising_scheduler=spin_raising_scheduler,
        )

        # pre-train CI wavefunction
        self.pre_CI = pre_CI
        self.pre_train_info = pre_train_info
        self.noise_lambda = noise_lambda
        self.clean_opt_state = clean_opt_state

        # avoid ansatz remove CI-Det
        model = self.model_raw.module
        if hasattr(model, "remove_det") and model.remove_det:
            raise TypeError(f"NQS does not support removing CI-Det")
        if hasattr(model, "det_lut") and model.det_lut is not None:
            raise TypeError(f"NQS does not support removing CI-Det")

    # @profile(precision=4, stream=open('opt_memory_profiler.log','w+'))
    def run(self) -> None:
        begin_vmc = time.time_ns()
        if self.rank == 0:
            logger.info(f"Begin VMC iteration: {time.ctime()}", master=True)
        for epoch in range(self.max_iter):
            t0 = time.time_ns()
            if epoch < self.HF_init:
                initial_state = self.onstate[random.randrange(self.dim)].clone().detach()
            else:
                initial_state = self.onstate[0].clone().detach()

            state, state_prob, (eloc, sloc), (eloc_mean, sloc_mean) = self.sampler.run(
                initial_state, epoch=epoch
            )
            if self.only_sample:
                delta = (time.time_ns() - t0) / 1.00e06
                if self.rank == 0:
                    s = f"{epoch}-th only Sampling finished, cost time {delta:.3f} ms\n"
                    s += "=" * 100
                    logger.info(s, master=True)
                continue

            sample_state = onv_to_tensor(state, self.sorb)  # -1:unoccupied, 1: occupied

            if self.spin_raising_scheduler is not None:
                c0 = self.initial_spin_spin_coeff
                self.spin_raising_coeff = self.spin_raising_scheduler(epoch) * c0
            # calculate model grad
            t1 = time.time_ns()
            if self.sr:
                raise NotImplementedError(f"SR-distributed will be implement in future")
                psi = sr_grad(
                    self.model,
                    sample_state,
                    state_prob,
                    eloc,
                    self.exact,
                    self.dtype,
                    self.method_grad,
                    self.method_jacobian,
                )
            else:
                sloc = sloc * self.spin_raising_coeff
                sloc_mean = sloc_mean * self.spin_raising_coeff
                if self.only_output_spin_raising:
                    sloc = torch.zeros_like(eloc)
                    sloc_mean = torch.zeros_like(eloc_mean)
                psi = energy_grad(
                    self.model,
                    sample_state,
                    state_prob,
                    eloc + sloc,
                    eloc_mean + sloc_mean,
                    self.MAX_AD_DIM,
                    self.dtype,
                    self.method_grad,
                )
            delta_grad = (time.time_ns() - t1) / 1.00e09

            # save the energy grad and clip-grad
            self.clip_grad(epoch=epoch)
            e_total = (eloc_mean + sloc_mean).real.item() + self.ecore
            self.save_grad_energy(e_total)

            t2 = time.time_ns()
            self.update_param(epoch=epoch)
            delta_update = (time.time_ns() - t2) / 1.00e09

            # save the checkpoint, different-version maybe error
            self.save_checkpoint(epoch=epoch)

            delta = (time.time_ns() - t0) / 1.00e09

            # All-Reduce max-time
            cost = torch.tensor([delta_grad, delta_update, delta], device=self.device)
            all_reduce_tensor(cost, op=dist.ReduceOp.MAX)
            synchronize()
            self.logger_iteration_info(epoch=epoch, cost=cost)

            if self.sampler.use_LUT:
                self.sampler.WF_LUT.clean_memory()
            del sample_state, eloc, state, psi, cost

        # end vmc iterations
        total_time = (time.time_ns() - begin_vmc) / 1.0e09
        synchronize()
        if self.rank == 0:
            s = f"End VMC iteration: {time.ctime()}"
            s += f"total cost time: {total_time:.3E} s, "
            s += f"{total_time/60:.3E} min {total_time/3600:.3E} h"
            logger.info(s, master=True)

    def operator_expected(
        self,
        h1e: Tensor,
        h2e: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        calculate <O> using different h1e, h2e, e.g. S_S+, H.

        Returns:
            state, prob, eloc, eloc-mean
        """
        if self.rank == 0:
            logger.info(f"{'*' * 30}Begin calculating <O>{'*' * 30}", master=True)

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

        # Sampling
        initial_state = self.onstate[0].clone().detach()
        epoch = self.max_iter
        state, state_prob, (eloc, sloc), (eloc_mean, sloc_mean) = self.sampler.run(
            initial_state, epoch
        )
        sample_state = onv_to_tensor(state, self.sorb)  # -1:unoccupied, 1: occupied

        if self.rank == 0:
            logger.info(f"<O>: {eloc_mean.real.item():.10f}")

        # revise
        self.sampler.h1e = h1e_old
        self.sampler.h2e = h2e_old
        self.sampler.h1e_spin = h1e_spin_old
        self.sampler.h2e_spin = h2e_spin_old
        self.sampler.use_spin_raising = use_spin_raising

        if self.rank == 0:
            logger.info(f"{'*'* 30}End <O>{'*' * 30}", master=True)

        return sample_state, state_prob, eloc, eloc_mean

    def noise_tune(self, noise_lambda: float = None) -> None:
        """
        NoisyTune
        ref: https://aclanthology.org/2022.acl-short.76.pdf
        """
        if noise_lambda is None:
            noise_lambda = self.noise_lambda

        # avoid tensor.numel() == 1
        def _std(tensor: Tensor):
            if tensor.numel() > 1:
                return torch.std(tensor)
            else:
                return torch.zeros_like(tensor)

        if noise_lambda > 0.0:
            for name, para in self.model.named_parameters():
                dtype = para.dtype
                device = para.device
                self.model.state_dict()[name][:] += (
                    (torch.rand(para.size(), device=device, dtype=dtype) - 0.5)
                    * noise_lambda
                    * _std(para)
                )

    def pre_train(self, prefix: str = None) -> None:
        if prefix is None:
            prefix = self.prefix
        t = CITrain(
            self.model,
            self.opt,
            self.pre_CI,
            self.pre_train_info,
            self.sorb,
            self.dtype,
            self.lr_scheduler,
            self.exact,
        )

        # clip-grad using VMC-opt params
        t.max_grad_norm = self.max_grad_norm
        t.use_clip_grad = self.use_clip_grad
        t.start_clip_grad = self.start_clip_grad
        if self.rank == 0:
            logger.info(f"pre-train:\n{t}", master=True)
        t.train(prefix=prefix, electron_info=self.sampler.ele_info, sampler=self.sampler)

        if self.clean_opt_state:
            self.opt.state = collections.defaultdict(dict)
            if self.rank == 0:
                s = "Clean opt-state after pre-train"
                s += "*" * 100
                logger.info(s, master=True)
        # Add noise
        self.noise_tune(self.noise_lambda)
        del t
