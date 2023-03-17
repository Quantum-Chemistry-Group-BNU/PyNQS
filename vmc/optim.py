import time
import random
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

from .sample import MCMCSampler
from .eloc import total_energy, energy_grad
from .PublicFunction import read_integral
from libs.hij_tensor import uint8_to_bit

__all__ = ["SR", "_calculate_sr", "sr_grad"]
print = partial(print, flush=True)

class VMCOptimizer():

    h1e: Tensor = None
    h2e: Tensor = None
    onstate: Tensor = None
    ecore: float = None
    sorb: int = None

    def __init__(self, nqs: nn.Module, opt_type: Optimizer,
                sampler_param: dict,
                nele: int = None,
                lr_scheduler = None,
                integral_file: str = None,
                electron_info: dict = None,
                max_iter: int = 2000,
                num_diff: bool = False,
                verbose: bool = False,
                analytic_derivate: bool = True,
                device = None,
                sr: bool = False,
                HF_init: int = None,
                external_model: any = None,
                only_sample: bool = False) -> None:
        # TODO: model in the difference device
        print(device)
        self.device = device
        if external_model is not None:
            self.read_model(external_model)
        else:
            self.model = nqs 
            self.opt = opt_type
            self.lr_scheduler = lr_scheduler
            self.HF_init = HF_init
            self.sr = sr
        if integral_file is not None and nele is not None:
            self.integral_file = integral_file
            self.nele = nele
            self.read_integral_info()
        else:
            s = 'h1e, h2e, onstate, ecore, sorb, nele'
            print(f"Read the {s} from dict 'electron_info' ")
            info = ("h1e", "h2e", "onstate", "ecore", "sorb", "nele")
            result = tuple(electron_info[i] for i in info)
            self.h1e, self.h2e, self.onstate, self.ecore, self.sorb, self.nele = result
        
        self.max_iter = max_iter
        self.num_diff = num_diff
        self.analytic_derivate = analytic_derivate
        self.verbose = verbose
        self.dim = self.onstate.shape[0]
        self.sampler_param = sampler_param
        self.exact = self.sampler_param["debug_exact"]
        self.n_sample = self.dim if self.exact else self.sampler_param["n_sample"]
        self.n_sample_unique: Tensor = None
        self.record_sample = self.sampler_param["record_sample"]
        self.sampler = MCMCSampler(self.model, self.h1e, self.h2e,
                                 self.sorb, self.nele, self.ecore,
                                 full_space=self.onstate, **self.sampler_param)
        self.noa = self.sampler.noa
        self.nob = self.sampler.nob
        self.nv = self.sampler.nv
        self.no = self.sampler.no
        self.only_sample = only_sample
        self.n_para = len(list(self.model.parameters()))
        self.grad_e_lst: List[Tensor] = [[] for _ in range(self.n_para)]
        self.grad_param_lst: List[Tensor] = [[] for _ in range(self.n_para)]
        self.e_lst: List[float] = []
        self.time_sample: List[float] = []
        self.time_iter: List[float] = []
        print(f"NQS model:\n{self.model}")
        print(f"Optimizer:\n{self.opt}")
        print(f"Sampler:\n{self.sampler}")

    def read_integral_info(self):
        result = read_integral(self.integral_file, self.nele, self.device)
        self.h1e, self.h2e, self.onstate, self.ecore, self.sorb = result

    def read_model(self, external_model):
        print(f"Read nqs model/optimizer from '.pth' file {external_model}")
        state = torch.load(external_model, map_location=self.device)
        self.model = state["model"]
        self.opt = state["optimizer"]
        self.lr_scheduler = state["lr_scheduler"]
        self.HF_init = state["HF_init"]
        self.sr = state["sr"]

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
            state, state_idx, eloc, e_total, = self.sampler.run(initial_state)
            
            self.n_sample = len(state)
            self.e_lst.append(e_total)

            if self.only_sample:
                delta = (time.time_ns() - t0)/1.00E06
                print(f"{p} only Sampling finished, cost time {delta:.3f} ms")
                continue

            sample_state = uint8_to_bit(state, self.sorb)
            
            delta = (time.time_ns() - t0)/1.00E06
            self.time_sample.append(delta)

            grad_update_lst = self._update_grad_model(sample_state, eloc, p, state_idx)
            
            # modify parameters grad (energy grad) manually
            for i, param in enumerate(self.model.parameters()):
                param.grad.data = grad_update_lst[i].detach().clone().reshape(param.shape)

            self.opt.step()
            self.opt.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            delta = (time.time_ns() - t0)/1.00E09
            print(f"{p} iteration total energy is {e_total:.9f} a.u., cost time {delta:.3f} s")
            self.time_iter.append(delta)

            del grad_update_lst, sample_state, eloc, state

    def _numerical_differentiation(self, state, eps: float = 1.0E-07) ->List[Tensor]:
        # TODO: state is uint8 not double
        """
        Calculate energy grad using numerical_differentiation
        """
        grad_e_num_lst: List[Tensor] = []
        for i, param in enumerate(self.model.parameters()):
            shape = param.shape
            N = shape.numel()
            tmp = []
            for j in range(N):
                zero = torch.zeros_like(param).reshape(-1)
                zero[j].add_(eps, alpha=1.0)
                delta = zero.reshape(shape)
                with torch.no_grad():
                    param.data.add_(delta, alpha=2.0)
                    e1 = self._total_energy(state.detach()) # f(x+2h)
                    param.data.add_(delta, alpha=-1.0)
                    e2 = self._total_energy(state.detach()) # f(x+h)
                    param.data.add_(delta, alpha=-1.0)
                    e3 = self._total_energy(state.detach()) # f(x)
                diff = (-1 * e1 + 4 * e2 - 3 * e3)/(2 * eps)
                tmp.append(diff)
            grad_e_num_lst.append(torch.tensor(tmp, dtype=torch.double))
        
        if self.verbose:
            print(f"Numerical: eps: {eps:.4e}")
            for i in grad_e_num_lst:
                print(i)
        
        return grad_e_num_lst
    
    def _total_energy(self, state) -> float:
            e = total_energy(state, self.n_sample, self.h1e, self.h2e,
                        self.model, self.ecore, self.sorb, self.nele, self.noa, self.nob,
                        exact=self.exact)[0]
            return e

    def _analytic_derivate_lnPsi(self, state) -> Tuple[Tensor, Tensor]:
        grad_sample, psi_lst = self.model(state.requires_grad_(), dlnPsi=True)
        # tuple, length: n_para, shape: (n_sample, param.shape)
        # nqs model grad is None, so the Optimizer base maybe be error, and set the gradient
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param)
        
        return grad_sample, psi_lst

    def _auto_diff_lnPsi(self, state) -> Tuple[List[Tensor], Tensor]:
        psi_lst: Tensor = torch.zeros(self.n_sample, device=state.device, dtype=state.dtype)
        dln_grad_lst: List[Tensor] = []
        for i in range(self.n_sample):
            psi = self.model(state[i].requires_grad_())
            # psi_1 = self.model(state[i].requires_grad_()).log()
            psi.backward()
            lst = []
            psi_lst[i] = psi.detach().clone()
            for param in self.model.parameters():
                if param.grad is not None:
                    lst.append(param.grad.detach().clone()/psi.detach().clone())
            dln_grad_lst.append(lst)
            self.model.zero_grad()
            del psi, lst

        # combine all sample grad => tuple, length: n_para (N_sample, n_para)
        grad_comb_lst: List[Tensor] = []
        for i in range(self.n_para):
            comb = []
            for j in range(self.n_sample):
                comb.append(dln_grad_lst[j][i].reshape(1, -1))
            grad_comb_lst.append(torch.cat(comb))  # (n_sample, n_para)
            del comb 
        
        del dln_grad_lst, 
        return (grad_comb_lst, psi_lst)


    def _update_grad_model(self, state, eloc, p: int, state_idx) -> List[Tensor]:

        if self.num_diff:
            F_p_lst = self._numerical_differentiation(state)
        else:
            if self.analytic_derivate:
                grad_lnPsi, psi = self._analytic_derivate_lnPsi(state)
            else:
                # lp = LineProfiler()
                # lp_wrapper = lp(self.sampler.run)
                # lp_wrapper(initial_state)
                # lp.print_stats()
                grad_lnPsi, psi = self._auto_diff_lnPsi(state)

            F_p_lst = energy_grad(eloc.reshape(-1), grad_lnPsi, self.n_sample,
                                  state_idx=state_idx, psi=psi, exact=self.exact)

            if self.verbose:
                print("Energy grad\nAnalytic:")
                for i, F_p in enumerate(F_p_lst):
                    print(F_p)

        # save the energy grad
        for i, F_p in enumerate(F_p_lst):
            self.grad_e_lst[i].append(F_p.reshape(-1).detach().to("cpu").numpy())

        if self.sr:
            sr_grad_lst = sr_grad(list(self.model.parameters()), grad_lnPsi, F_p_lst, p+1, self.n_sample, opt_gd=False, comb=True)

        return sr_grad_lst if self.sr else F_p_lst

    def summary(self, e_ref: float = None, prefix: str = "VMC"):
        self.save(prefix)
        self.plot_figure(e_ref, prefix)

    def save(self, prefix: str = "VMC", nqs: bool =True, sample: bool = True):
        sample_file, model_file = [prefix + i for i in (".csv", ".pth")]
        if not self.exact and sample and self.record_sample:
            self.sampler.frame_sample.to_csv(sample_file)

        if nqs:
            torch.save({"model": self.model,
                        "optimizer": self.opt,
                        "lr_scheduler": self.lr_scheduler,
                        "HF_init": self.HF_init,
                        "sr": self.sr,
                        "sampler_param": self.sampler_param}, model_file)

    def plot_figure(self, e_ref: float = None, prefix: str = "VMC"):
        fig = plt.figure()
        ax = fig.add_subplot(self.n_para+1, 1, 1)
        e = np.array(self.e_lst)
        idx = 0
        idx_e =np.arange(len(e))
        ax.plot(idx_e[idx:], e[idx:])
        if e_ref is not None:
            ax.axhline(e_ref,color='coral',ls='--')
            axins = inset_axes(ax, width="50%", height="45%", loc=1, 
                               bbox_to_anchor=(0.2, 0.1, 0.8, 0.8), 
                               bbox_transform=ax.transAxes)
            axins.plot(e[idx:])
            axins.axhline(e_ref, color='coral',ls='--')
            zone_left = len(e) - len(e)//10
            zone_right = len(e) - 1
            x_ratio = 0
            y_ratio = 1
            xlim0 = idx_e[zone_left]-(idx_e[zone_right] -idx_e[zone_left])*x_ratio
            xlim1 = idx_e[zone_right]+(idx_e[zone_right]-idx_e[zone_left])*x_ratio
            y = e[zone_left: zone_right]
            ylim0 = e_ref - (np.min(y) - e_ref)*y_ratio
            ylim1 = np.max(y) + (np.min(y) - e_ref)*y_ratio
            axins.set_xlim(xlim0, xlim1)
            axins.set_ylim(ylim0, ylim1)
            print(f"last energy: {e[-1]:.9f}")
            print(f"reference energy: {e_ref:.9f}, error: {abs((e[-1]-e_ref)/e_ref) * 100:.6f} %")

        # grad L2-norm/max
        for i in range(self.n_para):
            ax = fig.add_subplot(self.n_para+1, 1, i+2)
            x = np.linalg.norm(np.array(self.grad_e_lst[i]), axis=1) # ||g||
            x1 = np.abs(np.array(self.grad_e_lst[i])).max(axis=1) # max
            ax.plot(np.arange(len(x))[idx:], x[idx:], label="||g||")
            ax.plot(np.arange(len(x1))[idx:], x1[idx:], label="max|g|")
            ax.set_yscale("log")
            plt.legend(loc="best")

        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.savefig(prefix + ".png", format="png", dpi=1000, bbox_inches='tight')
        plt.close()

class SR(Optimizer):
    """Stochastic Reconfiguration in quantum many-body problem
    
        theta^{k+1} = theta^k - alpha * S^{-1} * F \\
        S_{ij}(k) = <O_i^* O_j> - <O_i^*><O_j>  \\
        F_i{k} = <E_{loc}O_i^*> - <E_{loc}><O_i^*> \
    """
    def __init__(self, params, lr=required, N_state: int=required, 
                opt_gd: bool= False, comb: bool = True,
                weight_decay: float =0,
                diag_shift: float =0.02) -> None:
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate : {lr}")
        if N_state <= 0:
            raise ValueError("The number of sample must be great 0")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, N_state=N_state, opt_gd=opt_gd,
                        comb=comb, weight_decay=weight_decay, diag_shift=diag_shift)
        self.Fp_lst: List[Tensor] = []
        super(SR, self).__init__(params, defaults)
    
    def step(self, grad_save: List[Tensor], F_p_lst: Tensor,
             k: int, closure=None):

        for group in self.param_groups:
            params_with_grad = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
            _sr_update(params_with_grad,
                grad_save,
                F_p_lst,
                k,
                opt_gd=group['opt_gd'],
                lr=group['lr'],
                N_state=group['N_state'],
                comb=group['comb'],
                weight_decay=group["weight_decay"],
                diag_shift=group["diag_shift"])


def _sr_update(params: List[Tensor],
        dlnPsi_lst: List[Tensor],
        F_p_lst: List[Tensor], 
        p: int,
        opt_gd: bool,
        lr: float,
        N_state: int,
        comb: bool,
        weight_decay: float,
        diag_shift: float):

    sr_grad_lst = sr_grad(params, dlnPsi_lst, F_p_lst, p, N_state, opt_gd, comb, weight_decay, diag_shift)

    for i, param in enumerate(params):
        dp = sr_grad_lst[i]
        if weight_decay != 0:
            dp = dp.add(param, alpha=weight_decay)
        param.data.add_(dp, alpha=-lr)

def sr_grad(params: List[Tensor],
            dlnPsi_lst: List[Tensor],
            F_p_lst: List[Tensor],
            p: int,
            N_state: int,
            opt_gd: bool = False,
            comb: bool = False,
            diag_shift: float = 0.02) -> List[Tensor]:

    sr_grad_lst: List[Tensor] = []
    if comb:
    # combine all networks parameter 
    # maybe be more precise for the Stochastic-Reconfiguration algorithm
        comb_F_p_lst = []
        comb_dlnPsi_lst = []
        for param, dlnPsi, F_p in zip(params, dlnPsi_lst, F_p_lst):
            comb_F_p_lst.append(F_p.reshape(-1)) # [N_para]
            comb_dlnPsi_lst.append(dlnPsi.reshape(N_state, -1))
        comb_F_p = torch.cat(comb_F_p_lst)# [N_para_all]
        comb_dlnPsi = torch.cat(comb_dlnPsi_lst, 1) # [N_state, N_para_all]
        dp = _calculate_sr(comb_dlnPsi, comb_F_p, N_state, p, opt_gd=opt_gd, diag_shift=diag_shift)
        
        begin_idx = end_idx = 0
        for i, param in enumerate(params):
            end_idx += param.shape.numel() # torch.Size.numel()
            dpi = dp[begin_idx:end_idx]
            begin_idx = end_idx
            sr_grad_lst.append(dpi.reshape(param.shape))
    else:
        for i, param in enumerate(params):
            dlnPsi = dlnPsi_lst[i].reshape(N_state, -1) # (N_state, N_para) two dim 
            dp = _calculate_sr(dlnPsi, F_p_lst[i], N_state, p, opt_gd=opt_gd, diag_shift=diag_shift)
            sr_grad_lst.append(dp.reshape(param.shape))

    return sr_grad_lst


def _calculate_sr(grad_total: Tensor, F_p: Tensor,
                  N_state: int, p: int, opt_gd: bool = False, diag_shift: float = 0.02) -> Tensor:
    """
    see: time-dependent variational principle(TDVP)
        Natural Gradient descent in steepest descent method on
    a Riemannian manifold.
    """
    if opt_gd:
        return F_p

    if grad_total.shape[0] != N_state:
        raise ValueError(f"The shape of grad_total {grad_total.shape} maybe error")

    avg_grad = torch.sum(grad_total, axis=0, keepdim=True)/N_state
    avg_grad_mat = torch.conj(avg_grad.reshape(-1, 1))
    avg_grad_mat = avg_grad_mat * avg_grad.reshape(1, -1)
    moment2 = torch.einsum("ki, kj->ij", grad_total.conj(), grad_total)/N_state
    S_kk = torch.subtract(moment2, avg_grad_mat)

    S_kk2 = torch.eye(S_kk.shape[0], dtype=S_kk.dtype, device=S_kk.device) * diag_shift
    #  _lambda_regular(p) * torch.diag(S_kk)
    S_reg = S_kk + S_kk2
    update = torch.matmul(torch.linalg.inv(S_reg), F_p).reshape(-1)
    return update

def _lambda_regular(p, l0=100, b=0.9, l_min=1e-4):
    """
    Lambda regularization parameter for S_kk matrix,
    see Science, Vol. 355, No. 6325 supplementary materials
    """
    return max(l0 * (b**p) , l_min)




