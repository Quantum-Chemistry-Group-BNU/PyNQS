from __future__ import annotations

import __main__
import time
import os
import collections
import sys
import warnings
import torch
import torch.distributed as dist
import numpy as np
import math
import types

from copy import deepcopy
from typing import Callable, Literal, Union, List, Optional
from dataclasses import dataclass
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger

from pynqs.ansatz.hybrid.excited import nqs_to_nes, nes_to_nqs
from pynqs.optim.grad.march import March
from pynqs.utils.hamiltonian import ElectronInfo
from pynqs.ci import CITrain
from pynqs.utils.ci import CIWavefunction
from pynqs.distributed import (
    all_reduce_tensor,
    get_rank,
    get_world_size,
    processes_synchronize,
    gather_tensor,
)
from pynqs.libs.C_extension import unpackbits
from .base import BaseVMCOptimizer
from .grad.kfac import KFACPreconditioner
from .grad import format_layer_groups, AutoSR_grad, LM_grad, RGN_grad, try_step_update

from torch.optim.lr_scheduler import LRScheduler


@dataclass
class SRConfig:
    layer_sr: bool = False
    layer_groups: list[list[int]] | None = None
    sr_method: Literal["auto", "sr", "minsr"] = "auto"
    damping_lambda: float | Callable[[int], float] = 1.0e-4

    def __post_init__(self) -> None:
        sr_method = self.sr_method.lower()
        if sr_method not in ("auto", "sr", "minsr"):
            raise ValueError(f"sr_method must be one of ('auto', 'sr', 'minsr')")
        self.layer_sr = bool(self.layer_sr)
        self.layer_groups = (
            [list(group) for group in self.layer_groups] if self.layer_groups is not None else None
        )
        self.sr_method = sr_method

    def get_damping_lambda(self, step: int) -> float:
        damping_lambda = self.damping_lambda(step) if callable(self.damping_lambda) else self.damping_lambda
        damping_lambda = float(damping_lambda)
        if not math.isfinite(damping_lambda) or damping_lambda <= 0:
            raise ValueError(f"damping_lambda must be a positive finite value, got {damping_lambda}")
        return damping_lambda


LayerSRConfig = SRConfig


@dataclass
class LMConfig:
    delta: float | Callable[[int], float] = 0.1

    def get_delta(self, step: int) -> float:
        if callable(self.delta):
            delta = float(self.delta(step))
        else:
            initial_delta = float(self.delta)
            if not math.isfinite(initial_delta) or initial_delta < 0:
                raise ValueError(f"lm delta must be a non-negative finite value, got {initial_delta}")
            delta = max(initial_delta * 0.9**step, 1.0e-6)
        if not math.isfinite(delta) or delta < 0:
            raise ValueError(f"lm delta must be a non-negative finite value, got {delta}")
        return delta


@dataclass
class RGNConfig:
    epsilon: float | Callable[[int], float] = 1.0
    delta: float | Callable[[int], float] = 0.0
    damping_lambda: float | Callable[[int], float] = 1.0e-3

    def get_epsilon(self, step: int) -> float:
        epsilon = float(self.epsilon(step) if callable(self.epsilon) else self.epsilon)
        if not (math.isinf(epsilon) and epsilon > 0) and (not math.isfinite(epsilon) or epsilon <= 0):
            raise ValueError(f"rgn epsilon must be positive finite or +inf, got {epsilon}")
        return epsilon

    def get_delta(self, step: int) -> float:
        delta = float(self.delta(step) if callable(self.delta) else self.delta)
        if not math.isfinite(delta) or delta < 0:
            raise ValueError(f"rgn delta must be a non-negative finite value, got {delta}")
        return delta

    def get_damping_lambda(self, step: int) -> float:
        damping_lambda = float(
            self.damping_lambda(step) if callable(self.damping_lambda) else self.damping_lambda
        )
        if not math.isfinite(damping_lambda) or damping_lambda <= 0:
            raise ValueError(f"rgn damping_lambda must be a positive finite value, got {damping_lambda}")
        return damping_lambda


class VMCOptimizer(BaseVMCOptimizer):
    """
    General VMC optimization and pre-train process
    """

    def __init__(
        self,
        nqs: DDP,
        sampler_param: dict,
        electron_info: ElectronInfo,
        opt: Optimizer,
        lr_scheduler: Union[List[LRScheduler], LRScheduler] = None,
        max_iter: int = 2000,
        check_point: str = None,
        read_model_only: bool = False,
        only_sample: bool = False,
        pre_CI: CIWavefunction = None,
        pre_train_info: dict = None,
        clean_opt_state: bool = False,
        noise_lambda: float = 0.05,
        sr: bool = False,
        sr_config: SRConfig | None = None,
        use_lm: bool = False,
        lm_config: LMConfig | None = None,
        use_rgn: bool = False,
        rgn_config: RGNConfig | None = None,
        interval: int = 100,
        prefix: str = "VMC",
        MAX_AD_DIM: int = -1,
        kfac: KFACPreconditioner | None = None,
        use_clip_grad: bool = False,
        max_grad_norm: float = 1.0,
        max_grad_value: float = 1.0,
        start_clip_grad: int = 0,
        clip_grad_method: str = "l2",
        clip_grad_scheduler: Optional[Callable[[int], float]] = None,
        use_3sigma: bool = False,
        k_step_clip: int = 100,
        use_spin_raising: bool = False,
        spin_raising_coeff: float = 1.0,
        only_output_spin_raising: bool = False,
        spin_raising_scheduler: Optional[Callable[[int], float]] = None,
        clip_eloc: bool = False,
        store_O_on_cpu: bool = False,
        intermediate: int = None,
        callback: Callable = None,
        extra_penalty: Callable = None,
        NES_w: Tensor = None,
    ) -> None:
        super().__init__(
            nqs=nqs,
            sampler_param=sampler_param,
            electron_info=electron_info,
            opt=opt,
            lr_scheduler=lr_scheduler,
            max_iter=max_iter,
            check_point=check_point,
            read_model_only=read_model_only,
            only_sample=only_sample,
            sr=sr,
            use_lm=use_lm,
            use_rgn=use_rgn,
            interval=interval,
            prefix=prefix,
            MAX_AD_DIM=MAX_AD_DIM,
            kfac=kfac,
            use_clip_grad=use_clip_grad,
            clip_grad_method=clip_grad_method,
            use_3sigma=use_3sigma,
            k_step_clip=k_step_clip,
            max_grad_norm=max_grad_norm,
            max_grad_value=max_grad_value,
            clip_grad_scheduler=clip_grad_scheduler,
            start_clip_grad=start_clip_grad,
            use_spin_raising=use_spin_raising,
            spin_raising_coeff=spin_raising_coeff,
            only_output_spin_raising=only_output_spin_raising,
            spin_raising_scheduler=spin_raising_scheduler,
            clip_eloc=clip_eloc,
            NES_w=NES_w,
        )

        # pre-train CI wavefunction
        self.pre_CI = pre_CI
        self.pre_train_info = pre_train_info
        self.noise_lambda = noise_lambda
        self.clean_opt_state = clean_opt_state

        self.store_O_on_cpu = False
        self.intermediate = intermediate

        # layer-SR/SR ref: 2510.08430
        if self.use_sr:
            if sr_config is None:
                self.sr_config = SRConfig()
            else:
                self.sr_config = deepcopy(sr_config)
        if self.use_lm:
            if lm_config is None:
                self.lm_config = LMConfig()
            else:
                self.lm_config = deepcopy(lm_config)
        if self.use_rgn:
            if rgn_config is None:
                self.rgn_config = RGNConfig()
            else:
                self.rgn_config = deepcopy(rgn_config)

        if self.rank == 0 and self.use_sr:
            model_for_groups = self.model.module if isinstance(self.model, DDP) else self.model
            logger.info(format_layer_groups(model_for_groups), master=True)
        if self.rank == 0 and self.use_sr:
            logger.info(
                f"SR config: sr={self.use_sr}, sr_config={self.sr_config}",
                master=True,
            )
        if self.rank == 0 and self.use_lm:
            logger.info(
                f"LM config: use_lm={self.use_lm}, lm_config={self.lm_config}",
                master=True,
            )
        if self.rank == 0 and self.use_rgn:
            logger.info(
                f"RGN config: use_rgn={self.use_rgn}, rgn_config={self.rgn_config}",
                master=True,
            )

        # ref: PHYS. REV. X 15, 011047 (2025)
        self.use_FreeEnergy = False
        self.temp_func = lambda step: 0.5 * math.exp(-0.02 * step)

        # avoid ansatz remove CI-Det
        model = self.model.module
        if hasattr(model, "remove_det") and model.remove_det:
            raise TypeError(f"NQS does not support removing CI-Det")
        if hasattr(model, "det_lut") and model.det_lut is not None:
            raise TypeError(f"NQS does not support removing CI-Det")

        if callback is not None:
            self.callback = types.MethodType(callback, self)
        if extra_penalty is not None:
            self.extra_penalty = types.MethodType(extra_penalty, self)

    # @profile(precision=4, stream=open('opt_memory_profiler.log','w+'))
    def run(self) -> None:
        begin_vmc = time.time_ns()
        if self.rank == 0:
            logger.info(f"Begin VMC iteration: {time.ctime()}", master=True)
        for epoch in range(self.max_iter):
            t0 = time.time_ns()

            # change random seed, continue train from checkpoint
            if self.lr_scheduler is not None:
                _epoch = self.lr_scheduler[0].last_epoch
            else:
                # read checkpoint
                if len(self.grad_e_lst[0]) > epoch:
                    _epoch = len(self.grad_e_lst[0])
                else:
                    _epoch = epoch

            was_training = self.model.training
            kfac_context = self.kfac.disabled() if hasattr(self.kfac, "disabled") else None
            try:
                if kfac_context is not None:
                    kfac_context.__enter__()
                self.model.eval()
                state, state_prob, (eloc, sloc), (eloc_mean, sloc_mean) = self.sampler.run(
                    epoch=_epoch if not self.only_sample else epoch,
                )
            finally:
                self.model.train(was_training)
                if kfac_context is not None:
                    kfac_context.__exit__(None, None, None)

            sample_state = unpackbits(state, self.sorb)
            if self.iNES:
                sample_state = nqs_to_nes(sample_state, self.NES_K)  # (ns,K,sorb) -> (ns*K,sorb)

            if hasattr(self, "extra_penalty"):
                local_vars = locals()
                local_vars.pop("self", None)
                oloc, oloc_mean = self.extra_penalty(**local_vars)
                eloc = eloc + oloc
                eloc_mean = eloc_mean + oloc_mean

            if self.only_sample:
                delta = (time.time_ns() - t0) / 1.00e09
                if self.rank == 0:
                    s = f"{epoch}-th only Sampling finished, cost time {delta:.3f} s\n"
                    s += "=" * 100
                    logger.info(s, master=True)
                continue

            if self.spin_raising_scheduler is not None:
                c0 = self.initial_spin_spin_coeff
                self.spin_raising_coeff = self.spin_raising_scheduler(_epoch) * c0
            # calculate model grad
            t1 = time.time_ns()
            sloc = sloc * self.spin_raising_coeff
            sloc_mean = sloc_mean * self.spin_raising_coeff
            if self.only_output_spin_raising:
                sloc = torch.zeros_like(eloc)
                sloc_mean = torch.zeros_like(eloc_mean)

            if isinstance(self.opt, March):
                self.opt.solve(
                    self.model,
                    sample_state,
                    eloc + sloc,
                    eloc_mean + sloc_mean,
                    state_prob,
                    self.MAX_AD_DIM,
                    self.store_O_on_cpu,
                )
                delta_grad = (time.time_ns() - t1) / 1.00e09
            else:
                if self.use_sr:
                    if self.sampler.use_multi_psi:
                        mode_name = "Layer-SR" if self.sr_config.layer_sr else "SR"
                        raise NotImplementedError(f"{mode_name} with multi-psi will be implemented in future")

                    if hasattr(self.sampler.params, "alpha"):
                        temp_alpha = self.sampler.params.alpha
                    else:
                        temp_alpha = 2.0

                    damping_lambda = self.sr_config.get_damping_lambda(_epoch)
                    if self.rank == 0:
                        logger.info(
                            f"SR damping_lambda at epoch {_epoch}: "
                            f"{damping_lambda:.8e} "
                            f"(method={self.sr_config.sr_method}, layer_sr={self.sr_config.layer_sr})",
                            master=True,
                        )
                    AutoSR_grad(
                        model=self.model.module,
                        eloc=eloc + sloc,
                        eloc_mean=eloc_mean + sloc_mean,
                        state_prob=state_prob,
                        bw_batch=self.MAX_AD_DIM,
                        sample_state=sample_state,
                        dtype=self.default_dtype,
                        device=self.device,
                        damping_lambda=damping_lambda,
                        alpha=temp_alpha,
                        all_sample_counts=self.sampler.all_sample_counts,
                        layerwise=self.sr_config.layer_sr,
                        method=self.sr_config.sr_method,
                        layer_groups=self.sr_config.layer_groups,
                    )

                elif self.use_lm:
                    lm_delta = self.lm_config.get_delta(_epoch)
                    if self.rank == 0:
                        logger.info(f"LM delta at epoch {_epoch}: {lm_delta:.8e}", master=True)
                    dtheta0 = LM_grad(
                        epoch=self.sampler.epoch,
                        x=sample_state,
                        h1e=self.h1e,  # +self.h1e_spin,
                        h2e=self.h2e,  # +self.h2e_spin,
                        sorb=self.sorb,
                        nele=self.nele,
                        noa=self.noa,
                        nob=self.nob,
                        model=self.model.module,
                        Eloc=eloc,  # +sloc,
                        Eloc_mean=eloc_mean,  # +sloc_mean,
                        state_prob=state_prob,
                        bw_batch=self.MAX_AD_DIM,
                        dtype=torch.float64,
                        device=self.device,
                        delta=lm_delta,
                    )
                    # LM update choice begining of J. Chem. Phys. 152, 024111 (2020)
                    if True:
                        try_step_update(
                            model=self.model.module,
                            sampler=self.sampler,
                            opt=self.opt,
                            dtheta0=dtheta0,
                            device=self.device,
                            dtype=self.default_dtype,
                        )
                elif self.use_rgn:
                    rgn_epsilon = self.rgn_config.get_epsilon(_epoch)
                    rgn_delta = self.rgn_config.get_delta(_epoch)
                    rgn_damping_lambda = self.rgn_config.get_damping_lambda(_epoch)
                    RGN_grad(
                        epoch=self.sampler.epoch,
                        x=sample_state,
                        h1e=self.h1e,
                        h2e=self.h2e,
                        sorb=self.sorb,
                        nele=self.nele,
                        noa=self.noa,
                        nob=self.nob,
                        model=self.model.module,
                        Eloc=eloc,
                        Eloc_mean=eloc_mean,
                        state_prob=state_prob,
                        bw_batch=self.MAX_AD_DIM,
                        dtype=torch.float64,
                        device=self.device,
                        epsilon=rgn_epsilon,
                        delta=rgn_delta,
                        damping_lambda=rgn_damping_lambda,
                    )
                else:
                    # sloc = sloc * self.spin_raising_coeff
                    # sloc_mean = sloc_mean * self.spin_raising_coeff
                    # if self.only_output_spin_raising:
                    #     sloc = torch.zeros_like(eloc)
                    #     sloc_mean = torch.zeros_like(eloc_mean)

                    from .grad.energy_grad import grad

                    if not self.sampler.use_multi_psi and not self.sampler.use_spin_flip:
                        extra_psi_pow = 1.0
                    else:
                        extra_psi_pow = self.sampler.extra_psi_pow
                    grad(
                        self.model,
                        sample_state,
                        state_prob,
                        eloc + sloc,
                        eloc_mean + sloc_mean,
                        extra_psi_pow,
                        self.default_dtype,
                        self.MAX_AD_DIM,
                    )
                delta_grad = (time.time_ns() - t1) / 1.00e09

            if self.use_FreeEnergy:
                from .grad.energy_grad import entropy_grad

                t = self.temp_func(epoch)
                entropy_grad(self.model, sample_state, state_prob, t)

            # save the energy grad and clip-grad
            self.clip_grad(epoch=_epoch)
            e_total = (eloc_mean + sloc_mean).real.item() + self.ecore
            self.save_grad_energy(e_total)

            t2 = time.time_ns()
            self.update_param(epoch=epoch)

            with torch.no_grad():
                wf_unique = self.model.module(sample_state)
                self.update_backflow_norm(self.model, wf_unique, state_prob)

            delta_update = (time.time_ns() - t2) / 1.00e09

            # save the checkpoint, different-version maybe error
            self.save_checkpoint(epoch=epoch)
            self.save_intermediate(epoch=epoch)

            if hasattr(self, "callback"):
                local_vars = locals()
                local_vars.pop("self", None)
                self.callback(**local_vars)

            delta = (time.time_ns() - t0) / 1.00e09

            # All-Reduce max-time
            cost = torch.tensor([delta_grad, delta_update, delta], device=self.device)
            all_reduce_tensor(cost, op=dist.ReduceOp.MAX)
            processes_synchronize()
            self.logger_iteration_info(epoch=epoch, cost=cost)
            if self.sampler.use_LUT:
                if self.sampler.WF_LUT is not None:
                    self.sampler.WF_LUT.clean_memory()
            del sample_state, eloc, state, cost

        # end vmc iterations
        total_time = (time.time_ns() - begin_vmc) / 1.0e09
        processes_synchronize()
        if self.rank == 0:
            s = f"End VMC iteration: {time.ctime()}"
            s += f"total cost time: {total_time:.3E} s, "
            s += f"{total_time/60:.3E} min {total_time/3600:.3E} h"
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
        # TODO: check it (zbwu-2025-11-06)
        _l2_norm = (wf.abs() ** 2).sum()
        all_reduce_tensor(_l2_norm)
        model.update_normalization(torch.sqrt(_l2_norm))
        if get_rank() == 0:
            logger.info(f"L2({names}): {_l2_norm.item():.4E}", master=True)

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
        self.sampler.seed += 1  # change random seed
        epoch = self.max_iter
        state, state_prob, (eloc, sloc), (eloc_mean, sloc_mean) = self.sampler.run(epoch)
        sample_state = unpackbits(state, self.sorb)

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
                    (torch.rand(para.size(), device=device, dtype=dtype) - 0.5) * noise_lambda * _std(para)
                )

    def pre_train(self, prefix: str = None) -> None:
        if prefix is None:
            prefix = self.prefix
        if self.lr_scheduler is not None:
            if len(self.lr_scheduler) > 1:
                raise NotImplementedError
            lr_scheduler = self.lr_scheduler[0]
        else:
            lr_scheduler = None
        t = CITrain(
            self.model,
            self.opt,
            self.pre_CI,
            self.pre_train_info,
            self.sorb,
            self.default_dtype,
            lr_scheduler,
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

    def save_intermediate(self, epoch: int) -> None:
        """
        save the model/opt/lr_scheduler to '.pth' file for resuming calculations
        """

        last_all = None
        if hasattr(self.sampler.SamplerState, "last"):
            last_rank = self.sampler.SamplerState.last
            if last_rank is not None:
                last_all = gather_tensor(last_rank, last_rank.device)
                if self.rank == 0:
                    last_all = torch.cat(last_all)

        if self.rank == 0:
            if self.intermediate is not None and epoch > 0 and epoch % self.intermediate == 0:
                checkpoint_file = f"{self.prefix}-intermediate-{epoch}.pth"
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
                        "energy": self.e_lst,
                        "sampler_last": last_all,
                    },
                    checkpoint_file,
                )
