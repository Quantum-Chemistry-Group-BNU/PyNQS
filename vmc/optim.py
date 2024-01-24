from __future__ import annotations
import __main__
from __future__ import annotations
import __main__
import time
import random
import platform
import os
import subprocess
import subprocess
import sys
import warnings
import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List
from memory_profiler import profile
from line_profiler import LineProfiler
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer, required
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from vmc.sample import Sampler
from vmc.grad import energy_grad, sr_grad
from ci import CITrain, CIWavefunction
from utils.distributed import (
    all_reduce_tensor,
    get_rank,
    get_world_size,
    synchronize,
)
from utils import ElectronInfo, Dtype
from libs.C_extension import onv_to_tensor

TORCH_VERSION: str = torch.__version__
if TORCH_VERSION >= "2.0.0":
    from torch.optim.lr_scheduler import LRScheduler
else:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

try:
    from kfac.preconditioner import KFACPreconditioner
except ImportError:
    warnings.warn(
        "KFAC not been found, see: https://github.com/gpauloski/kfac-pytorch", UserWarning
    )
    KFACPreconditioner = None


@dataclass
class BaseVMCOptimizer(ABC):
    r"""
    Base class for VMC optimization, including the definition of
    the ansatz/model, optimizer, sampling parameters, electronic
    structure information, and other related information.

    you need implement 'run' and 'pre_train' method
    """
    sys_name = platform.node()
    # torch.compile is lower in GTX 1650, but faster in A100
    # using_compile: bool = True if sys_name != "myarch" and TORCH_VERSION >= '2.0.0' else False
    using_compile = False
    """use torch.compile to speed up forward and backward"""

    model: DDP = None
    """wavefunction ansatz """

    rank: int = None
    """the i-th process in the DDP"""

    world_size: int = None
    """the number of process in the DDP"""

    device: torch.device = torch.device("cpu")
    """CPU or CUDA device"""

    dtype: torch.dtype = torch.double
    """wavefunction dtype, torch.double or torch.complex128"""

    MAX_AD_DIM: int = None
    """Maximum dim when using loss.backward()"""

    max_iter: int = None
    """Maximum iterations times"""

    exact: bool = None
    """whether exact optimization ansatz"""

    only_sample: bool = None
    """Only sampling, used to test sampling memory cost or calculate energy"""

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
        only_sample: bool = False,
        method_grad: str = "AD",
        sr: bool = False,
        method_jacobian: str = "vector",
        interval: int = 100,
        prefix: str = "VMC",
        MAX_AD_DIM: int = -1,
        kfac: KFACPreconditioner = None,
    ) -> None:
        if dtype is None:
            dtype = Dtype()
        self.dtype = dtype.dtype
        self.device = dtype.device
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.external_model = external_model
        self.dump_input()

        # whether read nqs/h1e-h2e from external file
        if self.external_model is not None:
            self.read_model(self.external_model)
            electron_info.h1e = self.h1e
            electron_info.h2e = self.h2e
        else:
            self.model_raw = nqs

        # Read parameters from an external model or model
        self.opt: Optimizer = opt_type(self.model_raw.parameters(), **opt_params)
        if lr_sch_params is not None and lr_scheduler is not None:
            self.lr_scheduler: LRScheduler = lr_scheduler(self.opt, **lr_sch_params)
        else:
            self.lr_scheduler: LRScheduler = None

        # read checkpoint file:
        if check_point is not None:
            self.read_checkpoint(check_point)
        # TODO: https://pytorch.org/docs/stable/notes/ddp.html torch.compile(ddp_model)
        self.model = torch.compile(self.model_raw) if self.using_compile else self.model_raw

        self.HF_init = int(HF_init)
        self.sr: bool = sr
        self.method_jacobian: str = method_jacobian if self.sr else None
        self.max_iter = max_iter
        self.method_grad = method_grad
        self.MAX_AD_DIM = MAX_AD_DIM

        # Sample
        self.sampler_param = sampler_param
        self.exact = self.sampler_param.get("debug_exact", False)
        self.sampler = Sampler(self.model, electron_info, dtype=self.dtype, **self.sampler_param)
        self.record_sample = self.sampler_param.get("record_sample", False)
        self.only_sample = only_sample

        # electronic structure information
        self.read_electron_info(self.sampler.ele_info)
        self.dim = self.onstate.shape[0]

        # record optim
        self.grad_e_lst = [[], []]  # grad_L2, grad_max
        self.e_lst: List[float] = []

        if self.rank == 0:
            params_num = sum(map(torch.numel, self.model.parameters()))
            s = f"NQS model:\n{self.model}\n"
            s += f"The number param of NQS model: {params_num}\n"
            s += f"Optimizer:\n{self.opt}\n"
            s += f"Sampler:\n{self.sampler}\n"
            s += f"Grad method: {self.method_grad}\n"
            s += f"Jacobian method: {self.method_jacobian}"
            logger.info(s, master=True)

        # save model
        if int(interval) != 1:
            self.nprt = int(self.max_iter / interval)
        else:
            self.nprt = 1
        if self.rank == 0:
            logger.info(f"Save model interval: {self.nprt}", master=True)
        self.prefix = prefix

        self.kfac = kfac
        self.use_kfac = True if self.kfac is not None else False
        if self.rank == 0:
            logger.info(f"Use K-FAC: {self.use_kfac}")

    def read_electron_info(self, info: ElectronInfo) -> None:
        if self.rank == 0:
            logger.info(info)
        self.sorb = info.sorb
        self.nele = info.nele
        self.no = info.nele
        self.nv = info.nv
        self.nob = info.nob
        self.noa = info.noa
        # if read external model, h1e and h2e come from external-model
        if self.external_model is None:
            self.h1e: Tensor = info.h1e
            self.h2e: Tensor = info.h2e
        self.ecore = info.ecore
        self.onstate = info.ci_space

    def read_model(self, external_model: str) -> None:
        """
        Read Nqs state_dict, h1e, he2e from external model.
        """
        warnings.warn("This functions will be removed in future", DeprecationWarning)
        if self.rank == 0:
            s = f"Read nqs model/h1e-h2e from '.pth' file {external_model}"
            logger.info(s, master=True)
        x = torch.load(external_model, map_location=self.device)
        self.model_raw.load_state_dict(x["model"])
        # notice h1e, he2 may be different even if the coordinate and basis are the same.
        self.h1e: Tensor = x["h1e"]
        self.h2e: Tensor = x["h2e"]

    def read_checkpoint(self, checkpoint: str) -> None:
        if self.rank == 0:
            s = f"Read model/optimizer/scheduler from {checkpoint}"
            logger.info(s, master=True)
        x = torch.load(checkpoint, map_location=self.device)
        self.model_raw.load_state_dict(x["model"])
        self.opt.load_state_dict(x["optimizer"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(x["scheduler"])

    def dump_input(self) -> None:
        """
        print main file to resume
        """
        if self.rank == 0:
            s = f"{'=' * 50} Begin PyNQS {'=' * 50}\n"
            s += "System:\n"
            s = f"{'=' * 50} Begin PyNQS {'=' * 50}\n"
            s += "System:\n"
            s += f"System {str(platform.uname())}\n"
            s += f"Python {sys.version}\n"
            s += f"numpy {np.__version__} torch {torch.__version__}\n"
            s += f"Date: {time.ctime()}\n"
            s += f"Device: {self.device}\n"
            s += f"PyNQS Version: {self.get_version()}"
            s += f"PyNQS Version: {self.get_version()}"
            logger.info(s, master=True)
            if hasattr(__main__, "__file__"):
                filename = os.path.abspath(__main__.__file__)
                s = f"Input file: {filename}\n"
                logger.info(s, master=True)
                os.system(f"cat {filename}")
            logger.info("=" * 100, master=True)

    @staticmethod
    def get_version() -> str:
        """
        print Git version
        """
        command = "git rev-parse HEAD"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if len(result.stderr) != 0:
            warnings.warn(f"PyNQS git-version not been found", UserWarning)
            version = "0000000000000000"
        else:
            version = result.stdout
        return version

    def save_grad_energy(self, e_total: float) -> None:
        r"""
        Save L2-grad, max-grad and energy to list in each iteration, for plotting.
        """
        if self.rank == 0:
            x: List[np.ndarray] = []
            for param in self.model.parameters():
                if param.grad is not None:
                    x.append(param.grad.reshape(-1).detach().to("cpu").numpy())
            x = np.concatenate(x)
            l2_grad = np.linalg.norm(x)
            max_grad = np.abs(x).max()
            self.e_lst.append(e_total)
            self.grad_e_lst[0].append(l2_grad)
            self.grad_e_lst[1].append(max_grad)

    def update_param(self, epoch: int) -> None:
        """
        update model param, and adjust learning rate
        """
        # TODO: synchronize
        if epoch < self.max_iter - 1:
            if self.kfac is not None:
                self.kfac.step()
            self.opt.step()
            self.opt.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def save_checkpoint(self, epoch: int) -> None:
        """
        save the model/opt/lr_scheduler to '.pth' file for resuming calculations
        """
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

    def logger_iteration_info(self, epoch: int, cost: Tensor) -> None:
        """
        print iteration_info in last of each iteration,
        include, energy, L2-grad, Max-grad, grad-cost, update-param-cost and total-cost

        epoch(int): the epoch-th iteration.
        cost(Tensor): grad-cost, update-param-cost and total-cost
        """
        if self.rank == 0:
            e_total = self.e_lst[-1]
            l2_grad = self.grad_e_lst[0][-1]
            max_grad = self.grad_e_lst[1][-1]
            s = f"Calculating grad: {cost[0].item():.3E} s, update param: {cost[1].item():.3E} s\n"
            s += f"Total energy {e_total:.9f} a.u., cost time {cost[2].item():.3E} s\n"
            s += f"L2-Gradient: {l2_grad:.5E}, Max-Gradient: {max_grad:.5E} \n"
            s += f"{epoch} iteration end {time.ctime()}\n"
            s += "=" * 100
            logger.info(s, master=True)

    @abstractmethod
    def run(self) -> None:
        """
        Run Vmc or CI-NQS progress
        """

    @abstractmethod
    def pre_train(self, prefix: str = None) -> None:
        """
        pre train
        """

    def summary(
        self,
        e_ref: float = None,
        e_lst: List[float] = None,
        prefix: str = None,
    ) -> None:
        """
        plot energy/grad figure and save model
        """
        if prefix is None:
            prefix = self.prefix
        if self.rank == 0:
            self._save_model(prefix)
            self._plot_figure(e_ref, e_lst, prefix)

    def _save_model(
        self,
        prefix: str = "VMC",
        nqs: bool = True,
        sample: bool = True,
    ) -> None:
        sample_file, model_file = [prefix + i for i in (".csv", ".pth")]
        if not self.exact and sample and self.record_sample:
            self.sampler.frame_sample.to_csv(sample_file)

        if nqs:
            torch.save(
                {
                    # DDP modules
                    "model": self.model_raw.state_dict(),
                    "optimizer": self.opt.state_dict(),
                    # lambda function could not been Serialized
                    # "lr_scheduler": self.lr_scheduler,
                    "HF_init": self.HF_init,
                    "sr": self.sr,
                    "sampler_param": self.sampler_param,
                    "h1e": self.h1e,
                    "h2e": self.h2e,
                },
                model_file,
            )

    def _plot_figure(
        self,
        e_ref: float = None,
        e_lst: List[float] = None,
        prefix: str = "VMC",
    ) -> None:
        if self.rank != 0:
            return None
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        e = np.array(self.e_lst)
        idx = 0
        idx_e = np.arange(len(e))
        ax.plot(idx_e[idx:], e[idx:])
        ax.set_xlabel("Iteration Time")
        ax.set_ylabel("Energy")
        if e_ref is not None:
            ax.axhline(e_ref, color="coral", ls="--")
            if e_lst is not None:
                for i in range(len(e_lst)):
                    ax.axhline(e_lst[i], color=plt.get_cmap("Accent")(i), ls="--")
            # plot partial enlarged view
            axins = inset_axes(
                ax,
                width="50%",
                height="45%",
                loc=1,
                bbox_to_anchor=(0.2, 0.1, 0.8, 0.8),
                bbox_transform=ax.transAxes,
            )
            axins.plot(e[idx:])
            axins.axhline(e_ref, color="coral", ls="--")
            if e_lst is not None:
                for i in range(len(e_lst)):
                    axins.axhline(e_lst[i], color=plt.get_cmap("Accent")(i), ls="--")
            zone_left = len(e) - len(e) // 10 - 1
            zone_right = len(e) - 1
            x_ratio = 0
            y_ratio = 1
            xlim0 = idx_e[zone_left] - (idx_e[zone_right] - idx_e[zone_left]) * x_ratio
            xlim1 = idx_e[zone_right] + (idx_e[zone_right] - idx_e[zone_left]) * x_ratio
            y = e[zone_left:zone_right]
            ylim0 = e_ref - (np.min(y) - e_ref) * y_ratio
            ylim1 = np.max(y) + (np.min(y) - e_ref) * y_ratio
            axins.set_xlim(xlim0, xlim1)
            axins.set_ylim(ylim0, ylim1)
            last = -1 * min(len(e), 100)
            logger.info(f"Last {abs(last)}th energy: {np.average(e[last:]):.9f}", master=True)
            logger.info(
                f"Reference energy: {e_ref:.9f}, error: {abs((np.average(e[last:])-e_ref)) * 1000:.6f} mHa",
                master=True,
            )

        param_L2 = np.asarray(self.grad_e_lst[0])
        param_max = np.asarray(self.grad_e_lst[1])
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(np.arange(len(param_L2))[idx:], param_L2[idx:], label=r"$||g||$")
        ax.plot(np.arange(len(param_max))[idx:], param_max[idx:], label=r"$||g||_{\infty}$")
        ax.set_xlabel("Iteration Time")
        ax.set_yscale("log")
        ax.set_ylabel("Gradients")
        plt.title(os.path.split(prefix)[1])  # remove path
        plt.legend(loc="best")

        plt.subplots_adjust(wspace=0, hspace=0.5)
        # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        plt.savefig(prefix + ".png", format="png", dpi=1000)
        plt.close()

        # save energy, ||g||, max_|g|
        np.savez(prefix, energy=e, grad_L2=param_L2, grad_max=param_max)
        logger.info(f"Save figure -> {prefix}.png", master=True)


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
        only_sample: bool = False,
        pre_CI: CIWavefunction = None,
        pre_train_info: dict = None,
        noise_lambda: float = 0.05,
        method_grad: str = "AD",
        sr: bool = False,
        method_jacobian: str = "vector",
        interval: int = 100,
        prefix: str = "VMC",
        MAX_AD_DIM: int = -1,
        kfac: KFACPreconditioner = None,
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
            only_sample=only_sample,
            method_grad=method_grad,
            sr=sr,
            method_jacobian=method_jacobian,
            interval=interval,
            prefix=prefix,
            MAX_AD_DIM=MAX_AD_DIM,
            kfac=kfac,
        )

        # pre-train CI wavefunction
        self.pre_CI = pre_CI
        self.pre_train_info = pre_train_info
        self.noise_lambda = noise_lambda

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

            state, state_prob, eloc, e_total, stats, eloc_mean = self.sampler.run(
                initial_state, epoch=epoch
            )
            if self.rank == 0:
                s = f"eloc-mean: {eloc_mean.item().real:.5f}{eloc_mean.item().imag:+.5f}j"
                logger.info(s, master=True)

            if self.only_sample:
                delta = (time.time_ns() - t0) / 1.00e06
                if self.rank == 0:
                    s = f"{epoch}-th only Sampling finished, cost time {delta:.3f} ms\n"
                    s += "=" * 100
                    logger.info(s, master=True)
                continue

            sample_state = onv_to_tensor(state, self.sorb)  # -1:unoccupied, 1: occupied

            # calculate model grad
            t1 = time.time_ns()
            if self.sr:
                if self.world_size > 1:
                    raise NotImplementedError(
                        f"Stochastic Reconfiguration does not support distributed"
                    )
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
                psi = energy_grad(
                    self.model,
                    sample_state,
                    state_prob,
                    eloc,
                    eloc_mean,
                    self.MAX_AD_DIM,
                    self.dtype,
                    self.method_grad,
                )
            delta_grad = (time.time_ns() - t1) / 1.00e09

            # save the energy grad
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

            del sample_state, eloc, state, psi, cost

        # end vmc iterations
        total_time = (time.time_ns() - begin_vmc) / 1.0e09
        synchronize()
        if self.rank == 0:
            s = f"End VMC iteration: {time.ctime()}"
            s += f"total cost time: {total_time:.3E} s, "
            s += f"{total_time/60:.3E} min {total_time/3600:.3E} h"
            logger.info(s, master=True)

    def noise_tune(self, noise_lambda: float = None) -> None:
        """
        NoisyTune
        ref: https://aclanthology.org/2022.acl-short.76.pdf
        """
        if noise_lambda is None:
            noise_lambda = self.noise_lambda
        if noise_lambda > 0.0:
            for name, para in self.model.named_parameters():
                dtype = para.dtype
                device = para.device
                self.model.state_dict()[name][:] += (
                    (torch.rand(para.size(), device=device, dtype=dtype) - 0.5)
                    * noise_lambda
                    * torch.std(para)
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
        if self.rank == 0:
            logger.info(f"pre-train:\n{t}", master=True)
        t.train(prefix=prefix, electron_info=self.sampler.ele_info, sampler=self.sampler)

        # Add noise
        self.noise_tune(self.noise_lambda)
        del t


class GD(Optimizer):
    """Naive Gradient Descent"""

    def __init__(self, params, lr=required, weight_decay: float = 0.00) -> None:
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate : {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(GD, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            d_p_list = []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
            _gd_update(
                params_with_grad, d_p_list, lr=group["lr"], weight_decay=group["weight_decay"]
            )


def _gd_update(params: List[Tensor], grads: List[Tensor], lr: float, weight_decay: float):
    for i, param in enumerate(params):
        dp = grads[i]
        if weight_decay != 0:
            dp = dp.add(param, alpha=weight_decay)
        param.data.add_(dp, alpha=-lr)
