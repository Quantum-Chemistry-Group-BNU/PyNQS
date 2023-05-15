import time
import random
import platform
import torch
import numpy as np

from memory_profiler import profile
from line_profiler import LineProfiler
from functools import partial
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer, required

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from typing import List, Tuple

from vmc.sample import MCMCSampler
from vmc.energy import total_energy
from vmc.grad import energy_grad, sr_grad
from ci import CITrain, CIWavefunction

from utils import ElectronInfo, Dtype
from libs.C_extension import onv_to_tensor

print = partial(print, flush=True)

class VMCOptimizer():

    sys_name = platform.node()
    TORCH_VERSION: str = torch.__version__
    # torch.compile is lower in GTX 1650, but faster in A100
    using_compile: bool = True if sys_name != "myarch" and TORCH_VERSION >= '2.0.0' else False

    def __init__(
        self,
        nqs: nn.Module,
        sampler_param: dict,
        electron_info: ElectronInfo,
        opt_type: Optimizer = torch.optim.Adam,
        opt_params: dict = {"lr": 0.005, "weight_decay": 0.001},
        lr_scheduler=None,
        lr_sch_params: dict = None,
        max_iter: int = 2000,
        verbose: bool = False,
        dtype: Dtype = None,
        HF_init: int = None,
        external_model: any = None,
        only_sample: bool = False,
        pre_CI: CIWavefunction = None,
        pre_train_info: dict = None,
        method_grad: str = "AD",
        sr: bool = False,
        method_jacobian: str = "vector",
    ) -> None:
        if dtype is None:
            dtype = Dtype()
        self.dtype = dtype.dtype
        self.device = dtype.device
        self.external_model = external_model

        # whether read nqs/h1e-h2e from external file
        if self.external_model is not None:
            self.read_model(self.external_model)
        else:
            self.model_raw = nqs
        self.model = torch.compile(self.model_raw) if self.using_compile else self.model_raw

        # Read parameters from an external model or model
        self.opt: Optimizer = opt_type(self.model.parameters(), **opt_params)
        if lr_sch_params is not None and lr_sch_params is None:
            self.lr_scheduler = lr_scheduler(self.opt, **lr_sch_params)
        else:
            self.lr_scheduler = None

        self.HF_init = HF_init
        self.sr: bool = sr
        self.method_jacobian: str = method_jacobian if self.sr else None
        self.max_iter = max_iter
        self.method_grad = method_grad
        self.verbose = verbose

        # Sample
        self.sampler_param = sampler_param
        self.exact = self.sampler_param.get("debug_exact", False)
        self.sampler = MCMCSampler(self.model, electron_info, dtype=self.dtype, **self.sampler_param)
        self.n_sample = 0
        self.record_sample = self.sampler_param.get("record_sample", False)
        self.only_sample = only_sample

        # electronic structure information
        self.read_electron_info(self.sampler.ele_info)
        self.dim = self.onstate.shape[0]

        # record optim
        self.n_para = len(list(self.model.parameters()))
        self.grad_e_lst: List[Tensor] = [[] for _ in range(self.n_para)]
        self.grad_param_lst: List[Tensor] = [[] for _ in range(self.n_para)]
        self.e_lst: List[float] = []
        self.stats_lst: List[dict] = []
        self.time_sample: List[float] = []
        self.time_iter: List[float] = []
        print(f"NQS model:\n{self.model}")
        print(f"Optimizer:\n{self.opt}")
        print(f"Sampler:\n{self.sampler}")
        print(f"Grad method: {self.method_grad}")
        if self.sr:
            print(f"Jacobian method: {self.method_jacobian}")

        # pre-train CI wavefunction
        self.pre_CI = pre_CI
        self.pre_train_info = pre_train_info

    def read_electron_info(self, ele_info: ElectronInfo):
        print(ele_info)
        self.sorb = ele_info.sorb
        self.nele = ele_info.nele
        self.no = ele_info.nele
        self.nv = ele_info.nv
        self.nob = ele_info.nob
        self.noa = ele_info.noa
        # if read external model, h1e and h2e come from external-model
        if self.external_model is None:
            self.h1e: Tensor = ele_info.h1e
            self.h2e: Tensor = ele_info.h2e
        self.ecore = ele_info.ecore
        self.onstate = ele_info.ci_space

    def read_model(self, external_model):
        print(f"Read nqs model/h1e-h2e from '.pth' file {external_model}")
        state = torch.load(external_model, map_location=self.device)
        self.model_raw = state["model"]
        # notice h1e, he2 may be different even if the coordinate and basis are the same.
        self.h1e = state["h1e"]
        self.h2e = state["h2e"]

    # @profile(precision=4, stream=open('opt_memory_profiler.log','w+'))
    def run(self):
        for p in range(self.max_iter):
            t0 = time.time_ns()
            if self.HF_init is None or p < self.HF_init:
                initial_state = self.onstate[random.randrange(self.dim)].clone().detach()
            else:
                initial_state = self.onstate[0].clone().detach()

            # print(f"initial_state : {initial_state}")
            # lp = LineProfiler()
            # lp_wrapper = lp(self.sampler.run)
            # lp_wrapper(initial_state)
            # lp.print_stats()
            # exit()

            state, state_prob, eloc, e_total, stats = self.sampler.run(initial_state)
            # breakpoint()

            self.n_sample = len(state)
            self.e_lst.append(e_total)
            self.stats_lst.append(stats)

            if self.only_sample:
                delta = (time.time_ns() - t0) / 1.00E06
                print(f"{p} only Sampling finished, cost time {delta:.3f} ms")
                continue

            sample_state = onv_to_tensor(state, self.sorb) # -1:unoccupied, 1: occupied

            delta = (time.time_ns() - t0) / 1.00E06
            self.time_sample.append(delta)

            # calculate model grad
            if self.sr:
                psi = sr_grad(self.model, sample_state, state_prob, eloc, self.exact, self.dtype,
                              self.method_grad, self.method_jacobian)
            else:
                psi = energy_grad(self.model, sample_state, state_prob, eloc, self.exact, self.dtype,
                                  self.method_grad)

            # for param in self.model.parameters():
            #     print(param.grad)

            # save the energy grad
            for i, param in enumerate(self.model.parameters()):
                self.grad_e_lst[i].append(param.grad.reshape(-1).detach().to("cpu").numpy())

            if p < self.max_iter - 1:
                self.opt.step()
                self.opt.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            delta = (time.time_ns() - t0) / 1.00E09
            print(f"{p} iteration total energy is {e_total:.9f} a.u., cost time {delta:.3E} s")
            self.time_iter.append(delta)

            del sample_state, eloc, state, psi

    def pre_train(self, prefix: str = None):
        t = CITrain(self.model, self.opt, self.pre_CI, self.pre_train_info, self.sorb, self.dtype,
                    self.lr_scheduler, self.exact)
        print(t)
        t.train(prefix=prefix, electron_info=self.sampler.ele_info, sampler=self.sampler)
        del t

    def summary(self, e_ref: float = None, prefix: str = "VMC"):
        self.save(prefix)
        self.plot_figure(e_ref, prefix)

    def save(self, prefix: str = "VMC", nqs: bool = True, sample: bool = True):
        sample_file, model_file = [prefix + i for i in (".csv", ".pth")]
        if not self.exact and sample and self.record_sample:
            self.sampler.frame_sample.to_csv(sample_file)

        if nqs:
            torch.save(
                {
                    "model": self.model_raw,
                    "optimizer": self.opt,
                    "lr_scheduler": self.lr_scheduler,
                    "HF_init": self.HF_init,
                    "sr": self.sr,
                    "sampler_param": self.sampler_param,
                    "h1e": self.h1e,
                    "h2e": self.h2e
                }, model_file)

    def plot_figure(self, e_ref: float = None, prefix: str = "VMC"):
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        e = np.array(self.e_lst)
        idx = 0
        idx_e = np.arange(len(e))
        ax.plot(idx_e[idx:], e[idx:])
        ax.set_xlabel("Iteration Time")
        ax.set_ylabel("Energy")
        if e_ref is not None:
            ax.axhline(e_ref, color='coral', ls='--')
            axins = inset_axes(ax,
                               width="50%",
                               height="45%",
                               loc=1,
                               bbox_to_anchor=(0.2, 0.1, 0.8, 0.8),
                               bbox_transform=ax.transAxes)
            axins.plot(e[idx:])
            axins.axhline(e_ref, color='coral', ls='--')
            zone_left = len(e) - len(e) // 10
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
            print(f"Last energy: {e[-1]:.9f}")
            print(f"Reference energy: {e_ref:.9f}, error: {abs((e[-1]-e_ref)/e_ref) * 100:.6f} %")

        # plot the L2-norm and max-abs of the gradients
        param_L2: List[np.ndarray] = []
        param_max: List[np.ndarray] = []
        for i in range(self.n_para):
            x = np.linalg.norm(np.array(self.grad_e_lst[i]), axis=1)  # ||g||
            param_L2.append(x)
            x1 = np.abs(np.array(self.grad_e_lst[i])).max(axis=1)  # max
            param_max.append(x1)
        param_L2 = np.stack(param_L2, axis=1).sum(axis=1)
        param_max = np.stack(param_max, axis=1).max(axis=1)

        ax = fig.add_subplot(2, 1, 2)
        ax.plot(np.arange(len(param_L2))[idx:], param_L2[idx:], label="||g||")
        ax.plot(np.arange(len(param_max))[idx:], param_max[idx:], label="max|g|")
        ax.set_xlabel("Iteration Time")
        ax.set_yscale("log")
        ax.set_ylabel("Gradients")
        plt.legend(loc="best")

        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.savefig(prefix + ".png", format="png", dpi=1000, bbox_inches='tight')
        plt.close()


class GD(Optimizer):
    """ Naive Gradient Descent"""
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
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
            _gd_update(params_with_grad, d_p_list, lr=group['lr'], weight_decay=group["weight_decay"])


def _gd_update(params: List[Tensor], grads: List[Tensor], lr: float, weight_decay: float):
    for i, param in enumerate(params):
        dp = grads[i]
        if weight_decay != 0:
            dp = dp.add(param, alpha=weight_decay)
        param.data.add_(dp, alpha=-lr)
