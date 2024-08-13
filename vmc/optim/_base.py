from __future__ import annotations

import __main__
import time
import platform
import os
import subprocess
import sys
import warnings
import torch
import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Callable, Tuple
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer, required
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger

from vmc.sample import Sampler
from utils.distributed import (
    get_rank,
    get_world_size,
)
from utils import ElectronInfo, Dtype

TORCH_VERSION: str = torch.__version__
if TORCH_VERSION >= "2.0.0":
    from torch.optim.lr_scheduler import LRScheduler
else:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

try:
    from kfac.preconditioner import KFACPreconditioner
except ImportError:
    warnings.warn("KFAC not been found, see: https://github.com/gpauloski/kfac-pytorch", UserWarning)
    KFACPreconditioner = None


@dataclass
class BaseVMCOptimizer(ABC):
    r"""
    Base class for VMC optimization, including the definition of
    the ansatz/model, optimizer, sampling parameters, electronic
    structure information, and other related information.

    you need implement 'run', 'pre_train' and 'operator_expected' method
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
        opt: Optimizer,
        lr_scheduler: LRScheduler = None,
        lr_sch_params: dict = None,
        max_iter: int = 2000,
        dtype: Dtype = None,
        HF_init: int = 0,
        external_model: any = None,
        check_point: str = None,
        read_model_only: bool = False,
        only_sample: bool = False,
        method_grad: str = "AD",
        sr: bool = False,
        method_jacobian: str = "vector",
        interval: int = 100,
        prefix: str = "VMC",
        MAX_AD_DIM: int = -1,
        kfac: KFACPreconditioner = None,  # type: ignore
        use_clip_grad: bool = False,
        clip_grad_method: str = "L2",
        use_3sigma: bool = False,
        k_step_clip: int = 100,
        max_grad_norm: float = 1.0,
        max_grad_value: float = 1.0,
        start_clip_grad: int = 0,
        clip_grad_scheduler: Callable[[int], float] = None,
        use_spin_raising: bool = False,
        spin_raising_coeff: float = 1.0,
        only_output_spin_raising: bool = False,
        spin_raising_scheduler: Callable[[int], float] = None,
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
        self.opt = opt
        if lr_sch_params is not None and lr_scheduler is not None:
            self.lr_scheduler: LRScheduler = lr_scheduler(self.opt, **lr_sch_params)
        else:
            self.lr_scheduler: LRScheduler = None

        # record optim, grad_L2, grad_max
        self.grad_e_lst: Tuple[List[float], List[float]] = ([], [])
        self.e_lst: List[float] = []
        # read checkpoint file:
        if check_point is not None:
            self.read_checkpoint(check_point, read_model_only)
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
        # spin_raising_coeff: float = 1.0
        # use_spin_raising = True
        self.sampler = Sampler(
            self.model,
            electron_info,
            dtype=self.dtype,
            use_spin_raising=use_spin_raising,
            spin_raising_coeff=spin_raising_coeff,
            only_sample=only_sample,
            **self.sampler_param,
        )
        self.only_sample = only_sample

        # add coeff <S-S+> in Hamiltonian
        self.use_spin_raising = use_spin_raising
        self.spin_raising_coeff = spin_raising_coeff
        # only output <S-S+>, not add in eloc
        self.only_output_spin_raising = only_output_spin_raising
        self.h1e_spin = self.sampler.h1e_spin
        self.h2e_spin = self.sampler.h2e_spin
        self.spin_raising_scheduler = spin_raising_scheduler
        self.initial_spin_spin_coeff = spin_raising_coeff

        # electronic structure information
        self.read_electron_info(self.sampler.ele_info)
        self.dim = self.onstate.shape[0]

        # clip grad
        self.use_clip_grad: bool = use_clip_grad
        if use_clip_grad:
            if start_clip_grad is None or start_clip_grad >= max_iter:
                raise ValueError(f"start-clip-grad:{start_clip_grad} must be in (0, {max_iter})")
            clip_grad_method = clip_grad_method.capitalize()
            if clip_grad_method not in ("L2", "Value"):
                raise ValueError(f"clip_grad_method: {clip_grad_method} excepted in ('L2', 'Value')")
            self.clip_grad_method = clip_grad_method
        self.start_clip_grad = start_clip_grad

        if self.clip_grad_method == "L2":
            self.initial_g0 = max_grad_norm
            self.max_grad_norm = max_grad_norm
        elif self.clip_grad_method == "Value":
            self.initial_g0 = max_grad_value
        self.clip_grad_scheduler = clip_grad_scheduler
        # grad upper bond 3σ
        self.use_3sigma = use_3sigma
        self.k_step_clip = k_step_clip

        if self.rank == 0:
            params_num = sum(map(torch.numel, self.model.parameters()))
            s = f"NQS model:\n{self.model}\n"
            s += f"The number param of NQS model: {params_num}\n"
            s += f"Optimizer:\n{self.opt}\n"
            if self.use_clip_grad:
                s += f"Clip-grad method: {self.clip_grad_method}, "
                if self.use_3sigma and self.clip_grad_method == "L2":
                    s += f"Use 3σ clip grad in {self.k_step_clip}-step, "
                s += f"g0: {self.initial_g0} "
                s += f"after {self.start_clip_grad}-th iteration\n"
            if self.use_spin_raising:
                s += f"penalty S-S+ coeff: {self.spin_raising_coeff:.5f}, "
                s += f"only output: {self.only_output_spin_raising}, "
                s += f"Notice: print 'S-S+' not 'c1 * S-S+'\n"
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
            logger.info(str(info), master=True)
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
        warnings.warn("This functions will be removed in future, use checkpoint", DeprecationWarning)
        if self.rank == 0:
            s = f"Read nqs model/h1e-h2e from '.pth' file {external_model}"
            logger.info(s, master=True)
        x = torch.load(external_model, map_location=self.device)
        self.model_raw.load_state_dict(x["model"])
        # notice h1e, he2 may be different even if the coordinate and basis are the same.
        self.h1e: Tensor = x["h1e"]
        self.h2e: Tensor = x["h2e"]

    def read_checkpoint(self, checkpoint: str, read_model_only: bool = False) -> None:
        if self.rank == 0:
            if not read_model_only:
                s = f"Read model/optimizer/scheduler from {checkpoint}"
            else:
                s = f"Read model from {checkpoint}"
            logger.info(s, master=True)
        x = torch.load(checkpoint, map_location="cpu")
        self.model_raw.load_state_dict(x["model"])
        if not read_model_only:
            self.opt.load_state_dict(x["optimizer"])
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(x["scheduler"])
        if "l2_grad" in x.keys():
            self.grad_e_lst[0].extend(x["l2_grad"])
        if "max_grad" in x.keys():
            self.grad_e_lst[1].extend(x["max_grad"])
        if "energy" in x.keys():
            self.e_lst.extend(x["energy"])

    def dump_input(self) -> None:
        """
        print main file to resume
        """
        if self.rank == 0:
            s = f"{'=' * 50} Begin PyNQS {'=' * 50}\n"
            s += "System:\n"
            s += f"System {str(platform.uname())}\n"
            s += f"Python {sys.version}\n"
            s += f"numpy {np.__version__} torch {torch.__version__}\n"
            s += f"Date: {time.ctime()}\n"
            s += f"Device: {self.device}\n"
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
        if self.sampler.use_multi_psi and self.rank == 0:
            idx = 0
            for param, key in zip(self.model.parameters(), self.model.state_dict().keys()):
                if param.grad is not None and "sample" in key:
                    # module.sample.params_M.all_sites and module.extra.params_weights
                    idx += 1
            l2_grad1 = x1[:idx].norm().item()
            l2_grad2 = x1[idx:].norm().item()
            max_grad1 = x2[:idx].max().item()
            if idx == len(x1):
                max_grad2 = 0
            else:
                max_grad2 = x2[idx:].max().item()
            s = f"Sample/Extra ansatz L2-grad: {l2_grad1:.5E} {l2_grad2:.5E}\n"
            s += f"Sample/Extra ansatz Max-grad: {max_grad1:.5E} {max_grad2:.5E}"
            logger.info(s, master=True)

        self.e_lst.append(e_total)
        self.grad_e_lst[0].append(l2_grad)
        self.grad_e_lst[1].append(max_grad)
        del x1, x2

    def clip_grad(self, epoch: int) -> None:
        if self.clip_grad_method == "L2":
            self._clip_grad_L2(epoch)
        elif self.clip_grad_method == "Value":
            self._clip_grad_value(epoch)
        else:
            raise NotImplementedError

    def _clip_grad_L2(self, epoch: int) -> None:
        """
        clip model grad use 2-norm
        """
        epoch0 = epoch
        if self.lr_scheduler is not None:
            epoch = self.lr_scheduler.last_epoch
        # read checkpoint
        if len(self.grad_e_lst[0]) > epoch:
            epoch0 = epoch
        # change max clip-grad
        self.use_3sigma = False
        self.k_step_clip = 100
        upper: float = None
        if self.clip_grad_scheduler is not None:
            g0 = self.clip_grad_scheduler(epoch) * self.initial_g0
        else:
            g0 = self.initial_g0
        # 3sigma
        if self.use_3sigma and epoch0 > self.k_step_clip:
            k_th = self.k_step_clip
            grad = np.asarray(self.grad_e_lst[0][-k_th:])
            std, mean = np.std(grad), np.mean(grad)
            upper = mean + 3 * std
            g0 = min(upper, g0)

        if self.use_clip_grad and epoch > self.start_clip_grad:
            x = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=g0, foreach=True)
            if self.rank == 0:
                if upper is not None and x > upper:
                    logger.info(f"3sigma-upper: {upper:.4E}", master=True)
                logger.info(f"Clip-grad, g: {x:.4E}, L2-g0: {g0:4E}", master=True)

    def _clip_grad_value(self, epoch: int) -> None:
        """
        clip model grad use max
        """
        if self.lr_scheduler is not None:
            epoch = self.lr_scheduler.last_epoch
        # change max clip-grad
        if self.clip_grad_scheduler is not None:
            g0 = self.clip_grad_scheduler(epoch) * self.initial_g0
        else:
            g0 = self.initial_g0
        if self.use_clip_grad and epoch > self.start_clip_grad:
            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=g0, foreach=True)
            if self.rank == 0:
                logger.info(f"Clip-grad, max-g0: {g0:4E}", master=True)

    def update_param(self, epoch: int) -> None:
        """
        update model param, and adjust learning rate
        """
        if epoch < self.max_iter - 1:
            if self.kfac is not None:
                self.kfac.step()
            # x = list(self.model.parameters())[0].detach().clone()
            self.opt.step()
            # x1 = list(self.model.parameters())[0].detach().clone()
            # logger.info(f"diff: {(x - x1).norm()}")
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
                        "l2_grad": self.grad_e_lst[0],
                        "max_grad": self.grad_e_lst[1],
                        "energy": self.e_lst,
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
            if self.lr_scheduler is not None:
                lrs = self.lr_scheduler.get_last_lr()
            else:
                lrs = [p['lr'] for p in self.opt.param_groups]
            s += f"Learning Rate: {' '.join(['{:.5E}'.format(lr) for lr in lrs])}\n"
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

    @abstractmethod
    def operator_expected(self, h1e: Tensor, h2e: Tensor):
        """
        calculate <O> using different h1e, h2e, e.g. S_S+, H.
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
            # old version and use checkpoint-file
            # self._save_model(prefix)
            self._plot_figure(e_ref, e_lst, prefix)

    def _save_model(
        self,
        prefix: str = "VMC",
    ) -> None:
        model_file = prefix + ".pth"
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
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

        # save energy, ||g||, max_|g|, remove see checkpoint
        # np.savez(prefix, energy=e, grad_L2=param_L2, grad_max=param_max)
        logger.info(f"Save figure -> {prefix}.png", master=True)


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
            _gd_update(params_with_grad, d_p_list, lr=group["lr"], weight_decay=group["weight_decay"])


def _gd_update(params: List[Tensor], grads: List[Tensor], lr: float, weight_decay: float):
    for i, param in enumerate(params):
        dp = grads[i]
        if weight_decay != 0:
            dp = dp.add(param, alpha=weight_decay)
        param.data.add_(dp, alpha=-lr)
