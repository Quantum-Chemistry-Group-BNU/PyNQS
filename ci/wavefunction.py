import torch
import numpy as np
import time

from functools import partial
from numpy import ndarray
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from typing import List, Tuple
import matplotlib.pyplot as plt

import libs.hij_tensor as pt
from vmc.sample import MCMCSampler
from utils import check_para, ElectronInfo,find_common_state


class CIWavefunction:
    """
    CI Wavefunction class
    """
    ci: Tensor
    onstate: Tensor
    device: str
    def __init__(self, coeff: Tensor, onstate: Tensor, device: str = None) -> None:
        self.device = device
        self._check_type(coeff, onstate)

    def _check_type(self, coeff, onstate):
        assert(isinstance(coeff, (ndarray, Tensor)))
        assert(isinstance(onstate, Tensor))
        check_para(onstate)
        if isinstance(coeff, ndarray):
            self.ci = torch.from_numpy(coeff).to(self.device)
        else:
            self.ci = coeff.to(self.device)
        self.ci.div_(torch.norm(self.ci))
        self.onstate = onstate.to(self.device)

    def energy(self, e: ElectronInfo) -> float:
        h1e = e.h1e.to(self.device) 
        h2e = e.h2e.to(self.device)
        sorb = e.sorb
        ecore = e.ecore
        nele = e.nele
        return energy_CI(self.ci, self.onstate, h1e, h2e, ecore, sorb, nele)
    
    def __repr__(self) -> str:
        return f"{type(self).__name__}, CI shape: {self.ci.shape[0]}"

@torch.no_grad()
def energy_CI(coeff: Tensor, onstate: Tensor, h1e: Tensor, h2e: Tensor, ecore: float,
              sorb: int, nele: int) -> float:
    """
    e = <psi|H|psi>/<psi|psi>
      <psi|H|psi> = \sum_{ij}c_i<i|H|j>c_j*
    """
    if abs(coeff.norm().to("cpu").item() - 1.00) >= 1.0E-06:
        raise ValueError(f"Normalization CI coefficient")

    hij = pt.get_hij_torch(onstate, onstate, h1e, h2e, sorb, nele)
    e = torch.einsum("i, ij, j", coeff.reshape(-1), hij,
                     torch.conj(coeff.reshape(-1))) + ecore

    return e.real.item()

class CITrain:
    """
    pre train CI coeff
    """
    pre_coeff: Tensor
    onstate: Tensor
    pre_max_iter: int
    nprt: int
    def __init__(self, model: nn.Module,
                 opt: Optimizer,
                 pre_CI: CIWavefunction,
                 pre_train_info: dict,
                 sorb: int,
                 lr_scheduler = None) -> None:
        r"""
        Pre train CI wavefunction.
        Args:
            model: the nqs model
            opt: Optimizer
            pre_CI: the pre_trained CI wavefunction 
            pre_train_info: a dict include "pre_max_inter" and "interval"
            sorb: the spin orbital
            lr_scheduler: the schedule of learning rate
        """
        self.model = model
        self.opt = opt
        self.sorb = sorb
        self.pre_ci = pre_CI.ci.reshape(-1)
        self.onstate = pre_CI.onstate
        self.pre_max_iter = pre_train_info["pre_max_iter"]
        self.nprt = int(self.pre_max_iter/pre_train_info["interval"])
        self.lr_scheduler = lr_scheduler

    def train(self, prefix: str = None, electron_info: ElectronInfo = None):
        self.ovlp_list: List[float] = []
        self.loss_list: List[float] = []
        begin = time.time_ns()
        state = pt.uint8_to_bit(self.onstate, self.sorb)
        eCI_0 = eCI_1 = eCI_ref = 0.00
        flag_energy: bool = True if electron_info is not None else False
        if flag_energy:
            energy_CI_1 = partial(energy_CI, onstate = self.onstate, 
                                h1e = electron_info.h1e, h2e = electron_info.h2e,
                                ecore = electron_info.ecore, sorb = electron_info.sorb,
                                nele = electron_info.nele)

        for epoch in range(self.pre_max_iter + 1):
            t0 = time.time_ns()
            psi = self.model(state)
            model_CI = psi/torch.norm(psi).reshape(-1)

            # calculate energy from CI coefficient.
            if epoch == 0:
                if flag_energy:
                    eCI_0 = energy_CI_1(model_CI)
                    eCI_ref = energy_CI_1(self.pre_ci)
            elif epoch == self.pre_max_iter:
                if flag_energy:
                    eCI_1 = energy_CI_1(model_CI)
            ovlp = torch.einsum("i, i", model_CI, self.pre_ci)
            loss = 1 - ovlp**2
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.ovlp_list.append(ovlp.detach().to("cpu").item())
            self.loss_list.append(loss.detach().to("cpu").item())
            if (epoch% self.nprt) == 0:
                delta = (time.time_ns() - t0)/1.E06
                print(f"The {epoch:<5d} training, loss = {loss.item():.4E}, ovlp = {ovlp.item():.4E}, delta = {delta:.3f} ms")

        print(f"Pre-train finished, cost time: {(time.time_ns() - begin)/1.E09:.3f}s")
        if flag_energy:
            print(f"Energy ref, before/after pre_training: {eCI_ref:.8f} {eCI_0:.8f}, {eCI_1:.8f}")

        self.plot_figure(prefix)

    def plot_figure(self, prefix: str = None):
        prefix = prefix if prefix is not None else "CI"
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        x = np.arange(self.pre_max_iter+1)
        ax1.plot(x, np.array(self.loss_list), color='cadetblue', label="Loss")
        # ax1.set_yscale("log")
        ax1.legend(loc="best")
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(x, np.abs(self.ovlp_list), color='tomato', label="Ovlp")
        ax2.legend(loc="best")
        fig.subplots_adjust(wspace=0, hspace=0.5)
        fig.savefig(prefix + "-pre_train.png", format="png", dpi=1000, bbox_inches='tight')
    
    def ci_onstate_loss(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        state is CISD
        """
        psi = self.model(state.requires_grad_())
        model_CI = psi/torch.norm(psi).reshape(-1)
        ovlp = torch.einsum("i, i", model_CI, self.pre_ci)
        loss = 1 - ovlp.norm()**2
        return tuple(loss, ovlp)

    # TODO: how to normalization psi after sampling, date: 23-04-13, No Need
    def sample_loss(self, sampler: MCMCSampler, initial_state: Tensor = None):
        """
        Loss Function = <psi|CI><CI|psi>/(<CI|CI><psi|psi>), psi comes from sampling
        """
        with torch.no_grad():
            sample_unique, sample_counts, state_prob, psi_unique = sampler.MCMC()
            x, idx_ci, idx_sample = find_common_state(self.onstate, sample_unique)
            ovlp_state = pt.uint8_to_bit(x)

        psi = self.model(ovlp_state.requires_grad_())
        t = psi * state_prob[idx_sample]
        ovlp1 = torch.einsum("i, i, i", state_prob[idx_sample], psi, self.pre_ci[idx_ci])
        ovlp = torch.div(ovlp1.norm()**2, t.norm()**2)
        loss = 1 - ovlp

        return tuple(loss, ovlp)

    def __repr__(self) -> str:
        return ( 
            f"{type(self).__name__}" + "(\n"
            f"    Pre train model: {self.model}\n" +
            f"    Pre train time: {self.pre_max_iter}\n" +
            f"    the number of CI coeff: {self.pre_ci.shape[0]}\n" + ")"
        )

