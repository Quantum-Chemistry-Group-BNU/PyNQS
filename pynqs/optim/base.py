from __future__ import annotations

import __main__
import time
import platform
import os
import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import List, Callable, Literal, Tuple, Union, Optional
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer, required
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger

from pynqs.distributed import get_rank, get_world_size, scatter_tensor, gather_tensor
from pynqs.utils.hamiltonian import ElectronInfo
from pynqs.config import dtype_config
from pynqs.utils.tools import VERSION, dump_input, sys_info
from pynqs.sample import Sampler, SampleParams
from pynqs.ansatz import Excitedwavefunctions

from torch.optim.lr_scheduler import LRScheduler

from .grad.kfac import KFACPreconditioner


class BaseVMCOptimizer(ABC):
    r"""
    Base class for VMC optimization, including the definition of
    the ansatz/model, optimizer, sampling parameters, electronic
    structure information, and other related information.

    you need implement 'run', 'pre_train' and 'operator_expected' method
    """

    def __init__(
        self,
        nqs: DDP,
        sampler_param: SampleParams,
        electron_info: ElectronInfo,
        opt: Optimizer,
        lr_scheduler: Union[List[LRScheduler], LRScheduler] = None,
        max_iter: int = 2000,
        check_point: Optional[str] = None,
        read_model_only: bool = False,
        only_sample: bool = False,
        sr: bool = False,
        use_lm: bool = False,
        use_rgn: bool = False,
        interval: int = 100,
        prefix: str = "VMC",
        MAX_AD_DIM: int = -1,
        kfac: Optional[KFACPreconditioner] = None,
        use_clip_grad: bool = False,
        clip_grad_method: str = "L2",
        use_3sigma: bool = False,
        k_step_clip: int = 100,
        max_grad_norm: float = 1.0,
        max_grad_value: float = 1.0,
        start_clip_grad: int = 0,
        clip_grad_scheduler: Optional[Callable[[int], float]] = None,
        use_spin_raising: bool = False,
        spin_raising_coeff: float = 1.0,
        only_output_spin_raising: bool = False,
        spin_raising_scheduler: Optional[Callable[[int], float]] = None,
        clip_eloc: bool = False,
        NES_w: Tensor = None,
    ) -> None:
        self.default_dtype = dtype_config.default_dtype
        self.device = dtype_config.device
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.model = nqs
        self.iNES = False
        self.NES_K = 1
        if isinstance(nqs.module, Excitedwavefunctions):
            self.iNES = True
            self.NES_K = self.NES_K = nqs.module.K
        if NES_w is None:
            self.NES_w = torch.tensor([1] * self.NES_K, dtype=self.default_dtype, device=self.device)
        else:
            self.NES_w = NES_w.to(self.default_dtype).to(self.device)

        # Read parameters from an external model or model
        self.opt = opt
        self.lr_scheduler: List[LRScheduler] = []
        if lr_scheduler is not None:
            if not isinstance(lr_scheduler, list):
                lr_scheduler = [lr_scheduler]
            for p in lr_scheduler:
                if not isinstance(p, LRScheduler):
                    raise TypeError(f"{type(p).__name__ } is not a LRScheduler")
                self.lr_scheduler.append(p)
        else:
            self.lr_scheduler = None

        # record optim, grad_L2, grad_max
        self.grad_e_lst: Tuple[List[float], List[float]] = ([], [])
        self.e_lst: List[float] = []
        # read checkpoint file:
        if check_point is not None:
            self.read_checkpoint(check_point, read_model_only)

        self.use_sr: bool = bool(sr)
        self.use_lm: bool = bool(use_lm)
        self.use_rgn: bool = bool(use_rgn)
        self.max_iter = max_iter
        self.MAX_AD_DIM = MAX_AD_DIM

        # Sample
        self.sampler_param = sampler_param
        # spin_raising_coeff: float = 1.0
        # use_spin_raising = True
        self.sampler = Sampler(
            self.model,
            electron_info,
            self.sampler_param,
            use_spin_raising=use_spin_raising,
            spin_raising_coeff=spin_raising_coeff,
            only_sample=only_sample,
            clip_eloc=clip_eloc,
            NES_K=self.NES_K,
            NES_w=self.NES_w,
        )

        self.exact = self.sampler.exact
        self.only_sample = only_sample

        # if hasattr(self.sampler.SamplerState, "last") and hasattr(self, "sampler_last"):
        #     if self.rank == 0:
        #         sampler_last = self.sampler_last.to(self.device)
        #     else:
        #         sampler_last: Tensor = None
        #     sampler_last = scatter_tensor(sampler_last, self.device, torch.uint8)
        #     self.sampler.SamplerState.last = sampler_last
        #     if get_rank() == 0:
        #         logger.info(
        #             f"Load MCMC starting state ({sampler_last.shape}) from checkpoint",
        #             master=True,
        #         )

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
        self.clip_grad_method: Literal["L2", "Value", None] = None
        if use_clip_grad:
            if start_clip_grad is None or start_clip_grad >= max_iter:
                raise ValueError(f"start-clip-grad:{start_clip_grad} must be in (0, {max_iter})")
            clip_grad_method = clip_grad_method.capitalize()
            if clip_grad_method not in ("L2", "Value", None):
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
            logger.info(dump_input())
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
            logger.info(s, master=True)

        # save model
        if int(interval) > 0:
            self.interval = int(interval)
        else:
            self.interval = 1
        if self.rank == 0:
            logger.info(f"Save model interval: {self.interval}", master=True)
        self.prefix = prefix

        self.kfac = kfac
        self.use_kfac = True if self.kfac is not None else False
        if self.rank == 0:
            logger.info(f"Use K-FAC: {self.use_kfac}")

        # Direct inversion in the iterative subspace
        self.use_diis = False
        if self.use_diis:
            from .diis import Diis

            nums = sum(map(torch.numel, self.model.parameters()))
            self.diss = Diis(n_params=nums, max_diss=5, beta=0.01)
        else:
            self.diss: Diis = None

        # stochastic Anderson Mixing for Nonconvex Stochastic Optimization
        self.use_sam = False
        if self.use_sam:
            from .other.padasam import pAdaSAM

            self.padasam = pAdaSAM(self.opt)

    def read_electron_info(self, info: ElectronInfo) -> None:
        if self.rank == 0:
            logger.info(str(info), master=True)
        self.sorb = info.sorb
        self.nele = info.nele
        self.no = info.nele
        self.nv = info.nv
        self.nob = info.nob
        self.noa = info.noa
        self.h1e: Tensor = info.h1e
        self.h2e: Tensor = info.h2e
        self.ecore = info.ecore * self.NES_K
        self.onstate = info.ci_space

    def read_checkpoint(self, checkpoint: str, read_model_only: bool = False) -> None:
        if self.rank == 0:
            if not read_model_only:
                s = f"Read model/optimizer/scheduler from {checkpoint}"
            else:
                s = f"Read model from {checkpoint}"
            logger.info(s, master=True)
        x = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(x["model"])
        if not read_model_only:
            self.opt.load_state_dict(x["optimizer"])
            if self.lr_scheduler is not None:
                for i, p in enumerate(self.lr_scheduler):
                    self.lr_scheduler[i].load_state_dict(x["scheduler"][i])
        if "l2_grad" in x.keys():
            self.grad_e_lst[0].extend(x["l2_grad"])
        if "max_grad" in x.keys():
            self.grad_e_lst[1].extend(x["max_grad"])
        if "energy" in x.keys():
            self.e_lst.extend(x["energy"])
        if "sampler_last" in x.keys():
            self.sampler_last = x["sampler_last"]

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
        elif self.clip_grad_method == None:
            ...
        else:
            raise NotImplementedError

    def _clip_grad_L2(self, epoch: int) -> None:
        """
        clip model grad use 2-norm
        """
        # change max clip-grad
        self.use_3sigma = False
        self.k_step_clip = 100
        upper: float = None
        if self.clip_grad_scheduler is not None:
            g0 = self.clip_grad_scheduler(epoch) * self.initial_g0
        else:
            g0 = self.initial_g0
        # 3sigma
        if self.use_3sigma and epoch > self.k_step_clip:
            k_th = self.k_step_clip
            grad = np.asarray(self.grad_e_lst[0][-k_th:])
            std, mean = np.std(grad), np.mean(grad)
            upper = mean + 3 * std
            g0 = min(upper, g0)

        if self.use_clip_grad and epoch >= self.start_clip_grad:
            x = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=g0, foreach=True)
            if self.rank == 0:
                if upper is not None and x > upper:
                    logger.info(f"3sigma-upper: {upper:.4E}", master=True)
                logger.info(f"Clip-grad, g: {x:.4E}, L2-g0: {g0:4E}", master=True)

    def _clip_grad_value(self, epoch: int) -> None:
        """
        clip model grad use max
        """
        # change max clip-grad
        if self.clip_grad_scheduler is not None:
            g0 = self.clip_grad_scheduler(epoch) * self.initial_g0
        else:
            g0 = self.initial_g0
        if self.use_clip_grad and epoch > self.start_clip_grad:
            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=g0, foreach=True)
            if self.rank == 0:
                logger.info(f"Clip-grad, max-g0: {g0:4E}", master=True)

    def _dtheta(self) -> Tensor:
        grads = [
            p.grad.detach().view(-1) if p.grad is not None else torch.zeros_like(p.data).view(-1)
            for p in self.model.parameters()
        ]
        if len(grads) == 0:
            return torch.zeros((), device=self.device, dtype=self.default_dtype)
        return torch.cat(grads)

    def _angle(self, x: Tensor, y: Tensor) -> float:
        nx = torch.linalg.vector_norm(x)
        ny = torch.linalg.vector_norm(y)
        if nx.item() == 0 or ny.item() == 0:
            return float("nan")
        cos = (torch.vdot(x, y).real / (nx * ny)).clamp(-1, 1)
        return torch.rad2deg(torch.acos(cos)).item()

    def update_param(self, epoch: int) -> None:
        """
        update model param, and adjust learning rate
        """
        if epoch < self.max_iter - 1:
            if self.kfac is not None:
                self.kfac.step()
            dtheta = self._dtheta()
            dtheta_l2 = torch.linalg.vector_norm(dtheta).item()
            theta = torch.cat([p.data.view(-1) for p in self.model.parameters()])
            if self.use_sam:
                self.padasam.step()
            else:
                self.opt.step()
            theta1 = torch.cat([p.data.view(-1) for p in self.model.parameters()])
            if self.rank == 0:
                dtheta_opt = theta - theta1
                dtheta_after = torch.linalg.vector_norm(dtheta_opt).item()
                angle = self._angle(dtheta, dtheta_opt)
                theta_angle = self._angle(theta, dtheta_opt)
                theta_l2 = torch.linalg.vector_norm(theta).item()
                rel = float("nan") if theta_l2 == 0 else dtheta_after / theta_l2
                logger.info(
                    f"|△𝜃| {dtheta_l2:.4e} -> {dtheta_after:.4e} (diff: {dtheta_after - dtheta_l2:.4e}, angle: {angle:.4e}°)",
                    master=True,
                )
                logger.info(f"angle(𝜃, △𝜃): {theta_angle:.4e}°, |△𝜃|/|𝜃|: {rel:.4e}", master=True)
            if self.use_diis:
                lr = self.opt.param_groups[0]["lr"]
                g = torch.cat([p.grad.view(-1) for p in self.model.parameters()])
                # diss update # delta p^{i+1} = -g^i
                # flag, update = self.diss.kernel(theta1, theta1 -theta)
                flag, update = self.diss.kernel(theta1, -lr * g)
                if flag:
                    start = 0
                    for p in self.model.parameters():
                        params = update[start : start + p.numel()].view(p.size())
                        start += p.numel()
                        p.data.copy_(params)
            self.opt.zero_grad()
            if self.lr_scheduler is not None:
                for i, p in enumerate(self.lr_scheduler):
                    self.lr_scheduler[i].step()
        elif self.kfac is not None and hasattr(self.kfac, "flush_pending_factors"):
            self.kfac.flush_pending_factors()

    def save_checkpoint(self, epoch: int) -> None:
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

        if self.rank == 0 and epoch > 0:
            if epoch % self.interval == 0 or epoch == self.max_iter - 1:
                # space = None
                # if self.sampler.use_pool_sampling and self.rank == 0:
                #     if hasattr(self.sampler.SamplerState_pool, "target_space"):
                #         space = self.sampler.SamplerState_pool.target_space
                checkpoint_file = f"{self.prefix}-checkpoint.pth"
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
                        "energy": self.e_lst,
                        "sampler_last": last_all,
                        "version": VERSION,
                        "sys_info": sys_info(),
                        "timestamp": time.ctime(),
                        # "target_space": space,
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
            lrs = [p["lr"] for p in self.opt.param_groups]
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
        if self.rank == 0 and not self.only_sample:
            # old version and use checkpoint-file
            # self._save_model(prefix)
            self._plot_figure(e_ref, e_lst, prefix)

    def _plot_figure(
        self,
        e_ref: float = None,
        e_lst: List[float] = None,
        prefix: str = "VMC",
    ) -> None:
        if self.rank != 0 and self.only_sample:
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
