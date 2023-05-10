import torch
import numpy as np
import time
import random

from functools import partial
from typing import List, Tuple
from pyscf import fci
from numpy import ndarray
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
import matplotlib.pyplot as plt

from libs.C_extension import onv_to_tensor, get_hij_torch
from vmc.sample import MCMCSampler
from utils import check_para, ElectronInfo, find_common_state, state_to_string

print = partial(print, flush=True)


class CIWavefunction:
    """
    CI Wavefunction class
    """
    coeff: Tensor
    space: Tensor
    device: str

    def __init__(self, coeff: Tensor, onstate: Tensor, device: str = None) -> None:
        self.device = device
        self._check_type(coeff, onstate)

    def _check_type(self, coeff, onstate):
        assert (isinstance(coeff, (ndarray, Tensor)))
        assert (isinstance(onstate, Tensor))
        check_para(onstate)
        if isinstance(coeff, ndarray):
            self.coeff = torch.from_numpy(coeff).to(self.device)
        else:
            self.coeff = coeff.to(self.device)
        self.coeff.div_(torch.norm(self.coeff))
        self.space = onstate.to(self.device)

    def energy(self, e: ElectronInfo) -> float:
        h1e = e.h1e.to(self.device)
        h2e = e.h2e.to(self.device)
        sorb = e.sorb
        ecore = e.ecore
        nele = e.nele
        return energy_CI(self.coeff, self.space, h1e, h2e, ecore, sorb, nele)

    def __repr__(self) -> str:
        return f"{type(self).__name__}, CI shape: {self.coeff.shape[0]}"


@torch.no_grad()
def energy_CI(coeff: Tensor, onstate: Tensor, h1e: Tensor, h2e: Tensor, ecore: float, sorb: int,
              nele: int) -> float:
    """
    e = <psi|H|psi>/<psi|psi>
      <psi|H|psi> = \sum_{ij}c_i<i|H|j>c_j*
    """
    if abs(coeff.norm().to("cpu").item() - 1.00) >= 1.0E-06:
        raise ValueError(f"Normalization CI coefficient")

    hij = get_hij_torch(onstate, onstate, h1e, h2e, sorb, nele)
    e = torch.einsum("i, ij, j", coeff.reshape(-1), hij, torch.conj(coeff.reshape(-1))) + ecore

    return e.real.item()


class CITrain:
    """
    pre train using given CI coeff.
    """
    pre_coeff: Tensor
    onstate: Tensor
    pre_max_iter: int
    nprt: int
    LOSS_TYPE = ('sample', 'onstate')

    def __init__(self,
                 model: nn.Module,
                 opt: Optimizer,
                 pre_CI: CIWavefunction,
                 pre_train_info: dict,
                 sorb: int,
                 dtype=torch.double,
                 lr_scheduler=None,
                 exact: bool = False) -> None:
        r"""
        Pre train CI wavefunction.
        Args:
            model: the nqs model
            opt: Optimizer
            pre_CI: the pre_trained CI wavefunction
            pre_train_info: a dict include "pre_max_inter", "interval" and "loss_type"
              pre_max_inter: the max time of pre-train, default: 200
              interval: print time, if == -1; print each step 'loss ' and 'ovlp', default: -1
              loss_type: the type of loss functions, there are two loss functions: ('sample', 'onstate'), default: 'onstate'
                sample': this is similar to VMC progress, 'see function 'qgt_loss'. notice: loss is meaningless, and focus on ovlp
                 'onstate': only fits CI-space coefficient and usually is imprecise, but could get relatively well initial parameters.
            sorb: the number of spin orbital
            dtype: Default torch.double
            lr_scheduler: the schedule of the learning rate
            exact: exact sampling
        """
        self.model = model
        self.opt = opt
        self.sorb = sorb
        self.pre_ci = pre_CI.coeff.reshape(-1)
        self.onstate = pre_CI.space
        self.state = onv_to_tensor(self.onstate, self.sorb)
        self.pre_max_iter = pre_train_info.get("pre_max_iter", 200)
        interval = pre_train_info.get("interval", -1)
        if int(interval) != -1:
            self.nprt = int(self.pre_max_iter / interval)
        else:
            self.nprt = 1
        self.lr_scheduler = lr_scheduler
        self.exact = exact
        self.loss_type: str = pre_train_info.get("loss_type", "onstate").lower()
        self.dtype = dtype
        if not self.loss_type in self.LOSS_TYPE:
            raise ValueError(f"Loss function type{self.loss_type} not in {self.LOSS_TYPE}, ")

    def train(self, prefix: str = None, electron_info: ElectronInfo = None, sampler: MCMCSampler = None):
        """
        the train process
        
        Args:
            prefix: the prefix of the loss and ovlp figure, default: 'CI'
            electron_info: theElectronInfo class, if exist, CI-energy will be calculated, default: "None"
            sampler: the Sampler class, if 'loss function is 'sample', this is necessary. default: "None"
             if exist, 'electron_info' will be read from the 'sampler'.
        """
        if False:
            dim = self.onstate.shape[0]
            print(f"State   ci^2")
            for i in range(dim):
                print(f"{state_to_string(self.onstate[i], self.sorb)}  {self.pre_ci[i]**2:.8f}")
        self.ovlp_list: List[float] = []
        self.loss_list: List[float] = []
        begin = time.time_ns()
        state = onv_to_tensor(self.onstate, self.sorb)
        eCI_0 = eCI_1 = eCI_ref = 0.00

        if sampler is not None:
            electron_info = sampler.ele_info
        flag_energy: bool = True if electron_info is not None else False
        if flag_energy:
            get_energy = partial(energy_CI,
                                 h1e=electron_info.h1e,
                                 h2e=electron_info.h2e,
                                 ecore=electron_info.ecore,
                                 sorb=electron_info.sorb,
                                 nele=electron_info.nele)

        # TODO: how to calculate the energy from CI or VMC
        for epoch in range(self.pre_max_iter + 1):
            t0 = time.time_ns()
            if self.loss_type == "onstate":
                loss, ovlp, model_CI = self.onstate_loss(state)
                onstate = self.onstate
            elif self.loss_type == "sample":
                loss, ovlp, model_CI, ovlp_onstate = self.qgt_loss(sampler)
                onstate = self.onstate
                # onstate = ovlp_onstate

            if epoch < self.pre_max_iter:
                loss.backward()
                self.opt.step()
                # self.opt.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            self.ovlp_list.append(ovlp.detach().to("cpu").item())
            self.loss_list.append((1 - ovlp.norm()**2).detach().to("cpu").item())

            # for para in self.model.parameters():
            #     print(para.grad.data)
            # breakpoint()
            self.opt.zero_grad()
            # calculate energy from CI coefficient.
            if epoch == 0:
                if flag_energy:
                    eCI_0 = get_energy(model_CI, onstate)
                    eCI_ref = get_energy(self.pre_ci, onstate)
            elif epoch == self.pre_max_iter:
                if flag_energy:
                    eCI_1 = get_energy(model_CI, onstate)
            if (epoch % self.nprt) == 0:
                delta = (time.time_ns() - t0) / 1.E06
                print(
                    f"The {epoch:<5d} training, loss = {(1-ovlp.norm()**2).item():.4E}, ovlp = {ovlp.item():.4E}, delta = {delta:.3f} ms"
                )
        if True:
            full_space = onv_to_tensor(electron_info.ci_space, self.sorb)
            psi = self.model(full_space)
            psi /= psi.norm()
            x, idx_ci, idx_sample = find_common_state(self.onstate, electron_info.ci_space.clone())
            # <psi|psi_CI> = <psi|n><n|psi_CI>
            ovlp_end = torch.dot(psi[idx_sample], self.pre_ci[idx_ci])
            print(f"<psi|psi_CI>: {ovlp_end.detach().item():.8f}")

        print(f"Pre-train finished, cost time: {(time.time_ns() - begin)/1.E09:.3f}s")
        if flag_energy:
            print(f"Energy ref, before/after pre_training: {eCI_ref:.8f} {eCI_0:.8f}, {eCI_1:.8f}")

        self.plot_figure(prefix)

    def onstate_loss(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        state is CISD
        """
        psi = self.model(state.requires_grad_())
        model_CI = psi / torch.norm(psi).reshape(-1)
        # breakpoint()
        ovlp = torch.einsum("i, i", model_CI, self.pre_ci)
        loss = 1 - ovlp.norm()**2
        return (loss, ovlp.detach(), model_CI.detach())

    def qgt_loss(self, sampler: MCMCSampler):
        """
        Loss Function: <psi|CI><CI|psi>/(<CI|CI><psi|psi>), psi comes from sampling
         E[<n|CI>/<n|psi> * <CI|psi>] Similar to vmc 
        """
        with torch.no_grad():
            # notice onstate[0] is HF state
            dim = self.onstate.shape[0]
            self.mc = sampler
            initial_state = self.onstate[random.randrange(dim)].clone().detach()
            sampler.prepare_sample(initial_state)
            if self.exact:
                sample_unique = self.mc.ele_info.ci_space.clone()
                psi_unique = self.model(onv_to_tensor(sample_unique, self.sorb)).to(self.dtype)
                state_prob = (psi_unique * psi_unique.conj() / psi_unique.norm()**2)
            else:
                sample_unique, sample_counts, state_prob, psi_unique = sampler.MCMC()
            x, idx_ci, idx_sample = find_common_state(self.onstate, sample_unique)
            state_prob = state_prob.to(self.dtype)
            ovlp, oloc = self.get_local_ovlp(state_prob, psi_unique, idx_sample, idx_ci, sample_unique)
            state_sample = onv_to_tensor(sample_unique[idx_sample], self.sorb)

        if len(idx_ci) == 0:
            raise ValueError(f"There is no common states between in CISD and sample-states")

        psi = self.model(state_sample.requires_grad_()).to(self.dtype)
        log_psi = psi.log()
        if torch.any(torch.isnan(log_psi)):
            raise ValueError(f"There are negative numbers in the log-psi, please use complex128")

        # grad = 2R(<O* eloc> - <O*><eloc>)
        loss1 = torch.einsum("i, i, i ->", oloc, log_psi.conj(), state_prob[idx_sample])
        loss2 = ovlp * torch.einsum("i, i->", log_psi.conj(), state_prob[idx_sample])
        loss = 2 * (loss1 - loss2).real
        # loss1 = torch.einsum("i, i, i, i ->", log_psi.conj(), state_prob, log_psi, state_prob)
        # loss2 = torch.einsum("i, i ->", log_psi.conj(), state_prob) * torch.einsum("i, i->", log_psi, state_prob)
        # loss = loss1 - loss2

        return (loss, ovlp.real, state_prob[idx_sample], sample_unique)

    def get_local_ovlp(self, prob, psi_sample: Tensor, idx_sample: Tensor, idx_ci: Tensor, sample_unique):
        """
        local_ovlp = <n|psi_ci><psi_ci|psi>/<n|psi>
        """
        model_ci = self.model(self.state)
        psi0 = torch.dot(self.pre_ci.conj(), model_ci)
        e_lst = -1 * self.pre_ci[idx_ci] * psi0 / psi_sample[idx_sample]
        e = torch.dot(e_lst, prob[idx_sample])

        # Testing optimization
        if False:
            # print(sample_unique)
            full_space = onv_to_tensor(self.mc.ele_info.ci_space, self.sorb)
            psi = self.model(full_space)
            psi /= psi.norm()
            sorb = self.sorb
            nele = self.sorb // 2
            occslstA = fci.cistring._gen_occslst(range(sorb // 2), nele // 2)
            occslstB = fci.cistring._gen_occslst(range(sorb // 2), nele // 2)
            dim = len(occslstA)
            print(f"State:  sample^2     ")
            for i, occsa in enumerate(occslstA):
                for j, occsb in enumerate(occslstB):
                    print(f"{state_to_string(full_space[dim*i+j], sorb)} {psi[dim*i+j]**2:.8f} ")
            print(e.real.item())
        return e, e_lst.to(self.dtype)

    def plot_figure(self, prefix: str = None):
        prefix = prefix if prefix is not None else "CI"
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        x = np.arange(self.pre_max_iter + 1)
        ax1.plot(x, np.array(self.loss_list), color='cadetblue', label="Loss")
        # ax1.set_yscale("log")
        ax1.legend(loc="best")
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(x, np.abs(self.ovlp_list), color='tomato', label="Ovlp")
        ax2.legend(loc="best")
        fig.subplots_adjust(wspace=0, hspace=0.5)
        fig.savefig(prefix + "-pre_train.png", format="png", dpi=1000, bbox_inches='tight')

    def __repr__(self) -> str:
        return (f"{type(self).__name__}" + "(\n"
                f"    Pre train model: {self.model}\n" + f"    Pre train time: {self.pre_max_iter}\n" +
                f"    the number of CI coeff: {self.pre_ci.shape[0]}\n"
                f"    Loss function type: {self.loss_type}\n" + ")")
