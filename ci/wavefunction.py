import os
import torch
import numpy as np
import time
import random
import warnings
import matplotlib.pyplot as plt

from functools import partial
from typing import List, Tuple, Union
from pyscf import fci
from numpy import ndarray
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger

from libs.C_extension import onv_to_tensor, get_hij_torch
from vmc.sample import Sampler
from utils import check_para, ElectronInfo, find_common_state, state_to_string
from utils.distributed import (
    all_reduce_tensor,
    all_gather_tensor,
    gather_tensor,
    get_world_size,
    get_rank,
    scatter_tensor,
    synchronize,
)

print = partial(print, flush=True)


class CIWavefunction:
    """
    CI Wavefunction class
    """

    coeff: Tensor
    space: Tensor
    device: str

    def __init__(self, coeff: Union[Tensor, ndarray], onstate: Tensor, device: str = None) -> None:
        self.device = device
        self._check_type(coeff, onstate)

    def _check_type(self, coeff, onstate):
        assert isinstance(coeff, (ndarray, Tensor))
        assert isinstance(onstate, Tensor)
        check_para(onstate)
        if isinstance(coeff, ndarray):
            self.coeff = torch.from_numpy(coeff).clone().to(self.device)
        else:
            self.coeff = coeff.clone().to(self.device)
        # self.coeff.div_(torch.norm(self.coeff))
        self.coeff = self.coeff / self.coeff.norm()
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
def energy_CI(
    coeff: Tensor, onstate: Tensor, h1e: Tensor, h2e: Tensor, ecore: float, sorb: int, nele: int
) -> float:
    """
    e = <psi|H|psi>/<psi|psi>
      <psi|H|psi> = \sum_{ij}c_i<i|H|j>c_j*
    """
    # if abs(coeff.norm().to("cpu").item() - 1.00) >= 1.0e-06:
    #     raise ValueError(f"Normalization CI coefficient")

    # TODO:how block calculate energy, matrix block
    hij = get_hij_torch(onstate, onstate, h1e, h2e, sorb, nele).type_as(coeff)
    e = (
        torch.einsum("i, ij, j", coeff.flatten(), hij, coeff.flatten().conj())
        / (torch.norm(coeff) ** 2)
        + ecore
    )

    return e.real.item()


class CITrain:
    """
    pre train using given CI coeff.
    """

    pre_coeff: Tensor
    onstate: Tensor
    pre_max_iter: int
    nprt: int
    LOSS_TYPE = ("sample", "onstate")

    def __init__(
        self,
        model: DDP,
        opt: Optimizer,
        pre_CI: CIWavefunction,
        pre_train_info: dict,
        sorb: int,
        dtype=torch.double,
        lr_scheduler=None,
        exact: bool = False,
    ) -> None:
        r"""
        Pre train CI wavefunction.
        Args:
            model: the nqs model
            opt: Optimizer
            pre_CI: the pre_trained CI wavefunction
            pre_train_info: a dict include "pre_max_inter", "interval" and "loss_type"
            pre_max_inter: the max time of pre-train, default: 200
            interval: print time, if == -1; print each step 'loss ' and 'ovlp', default: -1
            loss_type: the type of loss functions, there are two loss functions: ('sample', 'onstate'),
                default: 'onstate'
                'sample': this is similar to VMC progress, 'see function 'QGT_loss'.
                    notice: loss is meaningless, and focus on ovlp
                'onstate': only fits CI-space coefficient and usually is imprecise,
                    but could get relatively well initial parameters.
            sorb: the number of spin orbital
            dtype: Default torch.double
            lr_scheduler: the schedule of the learning rate
            exact: exact sampling
        """
        # distributed
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.device = pre_CI.device

        self.model = model
        self.opt = opt
        self.dtype = dtype
        self.sorb = sorb

        # pre-train wavefunction coeff
        self.pre_ci_coeff = pre_CI.coeff.reshape(-1).to(self.dtype)
        self.eCI_ref = 0.0

        # pre ci-space uint8
        self.onstate = pre_CI.space
        # pre ci-space double
        self.ci_state = onv_to_tensor(self.onstate, self.sorb)

        # Log output frequency
        self.pre_max_iter = pre_train_info.get("pre_max_iter", 200)
        interval = pre_train_info.get("interval", -1)
        if int(interval) != -1:
            self.nprt = int(self.pre_max_iter / interval)
        else:
            self.nprt = 1

        self.lr_scheduler = lr_scheduler
        self.exact = exact

        # loss function type
        self.loss_type: str = pre_train_info.get("loss_type", "onstate").lower()
        if not self.loss_type in self.LOSS_TYPE:
            raise ValueError(f"Loss function type{self.loss_type} not in {self.LOSS_TYPE}, ")

        # record optim
        self.n_para = len(list(self.model.parameters()))
        self.grad_lst: List[Tensor] = [[] for _ in range(self.n_para)]

    def train(
        self, prefix: str = None, electron_info: ElectronInfo = None, sampler: Sampler = None
    ) -> None:
        """
        the train process

        Args:
            prefix: the prefix of the loss and ovlp figure, default: 'CI'
            electron_info: theElectronInfo class, if exist, CI-energy will be calculated, default: "None"
            sampler: the Sampler class, if 'loss function is 'sample', this is necessary. default: "None"
             if exist, 'electron_info' will be read from the 'sampler'.
        """
        self.ovlp_list: List[float] = []
        self.loss_list: List[float] = []

        eCI_begin = eCI_end = 0.00

        if sampler is not None:
            electron_info = sampler.ele_info
        flag_energy: bool = True if electron_info is not None else False
        if flag_energy:
            get_energy = partial(
                energy_CI,
                h1e=electron_info.h1e,
                h2e=electron_info.h2e,
                ecore=electron_info.ecore,
                sorb=electron_info.sorb,
                nele=electron_info.nele,
            )
            self.eCI_ref = get_energy(self.pre_ci_coeff, self.onstate)
        if self.rank == 0:
            logger.info(f"Begin pre-train: {time.ctime()}", master=True)
            if self.loss_type == "sample":
                s = f"NOTICE:{'*'*20}"
                s += f"Loss is meaningless, and focus on ovlp{'*'*20}"
                logger.info(s, master=True)
            elif self.loss_type == "onstate" and self.world_size > 1:
                warnings.warn(
                    "Loss-type(onstate) dose not support distributed pre-train", FutureWarning
                )

        begin = time.time_ns()
        self.opt.zero_grad()
        for epoch in range(self.pre_max_iter + 1):
            t0 = time.time_ns()
            if self.loss_type == "onstate":
                loss, ovlp, model_coeff = self.sqaure_loss()
                onstate = self.onstate
            elif self.loss_type == "sample":
                loss, ovlp, sample_unique, sample_counts, state_prob = self.QGT_loss(sampler, epoch)

            if epoch <= self.pre_max_iter:
                loss.backward()
                self.opt.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            # save the ovlp grad
            if self.rank == 0:
                grad_sum = 0.0
                param_sum = 0.0
                for i, param in enumerate(self.model.parameters()):
                    if param.grad is not None:
                        grad_sum += np.abs(param.grad.reshape(-1).detach().to("cpu").numpy()).sum()
                        param_sum += param.data.reshape(-1).detach().to("cpu").abs().numpy().sum()
                        self.grad_lst[i].append(param.grad.reshape(-1).detach().to("cpu").numpy())
                    else:
                        self.grad_lst[i].append(np.zeros(param.numel()))
            self.opt.zero_grad()

            # if self.rank == 0:
            #     print(f"grad-abs-sum: {grad_sum:.20f}")
            #     print(f"param-abs-sum: {param_sum:.20f}")
            # save ovlp and loss-functions
            # logger.info(f"ovlp: {ovlp.norm().item():.5f}, loss: {loss.item():.5f}")

            reduce_loss = all_reduce_tensor(loss, world_size=self.world_size, in_place=False)[0]
            reduce_ovlp = all_reduce_tensor(ovlp, world_size=self.world_size, in_place=False)[0]
            synchronize()
            # logger.info(f"epoch: {epoch}, ovlp: {reduce_ovlp.norm().detach().item():.10f}")
            if self.rank == 0:
                self.ovlp_list.append(reduce_ovlp.norm().detach().to("cpu").item())
                self.loss_list.append(reduce_loss.detach().to("cpu").item())

            # calculate energy from CI coefficient or local-energy
            # notice, the shape of model_CI maybe not equal of self.pre_ci if using "sample"
            if (epoch == 0) or (epoch == self.pre_max_iter):
                if flag_energy:
                    if self.loss_type == "onstate":
                        e_total = get_energy(model_coeff, onstate)
                    else:
                        e_rank = self.sampler.calculate_energy(
                            sample_unique, state_prob, sample_counts
                        )[0]
                        e_rank = torch.tensor(e_rank, dtype=self.dtype, device=self.device)
                        all_reduce_tensor(e_rank, world_size=self.world_size)
                        synchronize()
                        e_total = e_rank.item().real
                    if epoch == 0:
                        eCI_begin = e_total
                    else:
                        eCI_end = e_total

            # print logging
            if (epoch % self.nprt) == 0 or epoch == self.pre_max_iter:
                logger.info(f"loss: {loss.norm().item():.4E}, ovlp: {ovlp.norm().item():.4E}")
                if self.rank == 0:
                    delta = (time.time_ns() - t0) / 1.0e06
                    s = f"The {epoch:<5d} training, loss = {reduce_loss.norm().item():.4E}, "
                    s += f"ovlp = {reduce_ovlp.norm().item():.4E}, delta = {delta:.3E} ms"
                    logger.info(s, master=True)

                    checkpoint_file = f"{prefix}-pre-train-checkpoint.pth"
                    lr = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    # self.model or self.model.module
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": self.model.state_dict(),
                            "optimizer": self.opt.state_dict(),
                            "scheduler": lr,
                        },
                        checkpoint_file,
                    )

        # END-pre-train
        if self.rank == 0:
            delta = (time.time_ns() - begin) / 1.0e09
            s = f"Pre-train finished, cost time: {delta:.3E}s, in {time.ctime()}"
            s += f"Max ovlp: {np.max(self.ovlp_list):.4E}\n"
            if flag_energy:
                s += (
                    f"Energy ref, begin and end : {self.eCI_ref:.8f} {eCI_begin:.8f}, {eCI_end:.8f}"
                )
            logger.info(s, master=True)
            logger.info("*" * 100, master=True)
            self.plot_figure(prefix)

    def sqaure_loss(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        using least sqaure method to fits CI-space coefficient
        ovlp = <psi_ci|psi>
        loss = 1 - ovlp**2
        """
        psi = self.model(self.ci_state.requires_grad_())
        model_CI = psi / torch.norm(psi).flatten().to(self.dtype)
        ovlp = torch.einsum("i, i", model_CI, self.pre_ci_coeff)
        loss = 1 - ovlp.norm() ** 2
        return (loss, ovlp.detach(), model_CI.detach())

    def QGT_loss(
        self,
        sampler: Sampler,
        epoch: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Quantum geometric tensor(QGT):
            d(psi, phi) = arccos\sqrt((<psi|phi><phi|psi>)/(<psi|psi><phi|phi>))
        define: ovlp:
            ovlp(<psi|CI>) = <psi|CI><CI|psi>/(<CI|CI><psi|psi>)
        psi comes from sampling, CI comes from pre-train wavefunction, <CI|CI> = 1
        define local-ovlp:
            oloc = <n|psi_CI><psi_CI|psi>/<n|psi>
        grad:
            2R[<O*. eloc> - <eloc><O*>], O* =dlnpsi

        Returns
        -------
            loss: loss functions
            ovlp: ovlp<psi|CI>
            model_CI: all-rank ci-coeff if self.rank == 0 else None
            all_ovlp_state: all-rank ovlp-state if self.rak == 0 else None
        """
        with torch.no_grad():
            # notice onstate[0] is HF state
            dim = self.onstate.shape[0]
            self.sampler = sampler
            initial_state = self.onstate[random.randrange(dim)].clone().detach()
            if self.exact:
                # fci-space isn't pre-ci space
                fci_space = self.sampler.ele_info.ci_space
                sample_unique = scatter_tensor(fci_space, self.device, torch.uint8, self.world_size)
                sample_counts = torch.ones(
                    sample_unique.size(0), dtype=torch.int64, device=self.device
                )
                synchronize()
                # sample_unique = self.sampler.ele_info.ci_space.clone()
            else:
                sample_unique, sample_counts, state_prob = self.sampler.sampling(
                    initial_state, epoch=epoch
                )

        # XXX: This is little redundance, psi_unique have been calculated when sampling
        psi = self.model(onv_to_tensor(sample_unique, self.sorb).requires_grad_()).to(self.dtype)
        psi_detach = psi.clone().detach()
        if self.exact:
            # Gather all psi from very rank
            all_psi = gather_tensor(psi_detach, self.device, self.world_size)
            synchronize()
            if self.rank == 0:
                all_psi = torch.cat(all_psi)
                all_prob = (all_psi * all_psi.conj()).real / all_psi.norm() ** 2
                all_prob = all_prob.to(self.dtype)
            else:
                all_prob = None
            # Scatter prob to very rank
            state_prob = scatter_tensor(all_prob, self.device, self.dtype, self.world_size)
            state_prob.mul_(self.world_size)

        # NOTICE: idx_ci and idx_sample maybe empty tensor
        # find ovlp-onv, idx_ci, idx_sample between pre-ci-onv and sampling-onv
        idx_ci, idx_sample = find_common_state(self.onstate, sample_unique)[1:]
        state_prob = state_prob.to(self.dtype)

        # calculate local ovlp
        ovlp, oloc = self.get_local_ovlp(state_prob, psi_detach, idx_sample, idx_ci)
        ovlp_mean = all_reduce_tensor(ovlp, world_size=self.world_size, in_place=False)[0]

        log_psi = psi.log()
        if torch.any(torch.isnan(log_psi)):
            raise ValueError(f"There are negative numbers in the log-psi, please use complex128")

        # NOTICE: idx_sample maybe is empty
        loss1 = torch.sum(oloc * log_psi.conj() * state_prob)
        loss2 = ovlp_mean * torch.sum(log_psi.conj() * state_prob)
        loss = 2 * (loss1 - loss2).real
        # print(f"loss: {loss.item():.15f}")
        # logger.info(f"ovlp: {ovlp}, loss: {loss}")

        del idx_ci, idx_sample, oloc
        if loss.is_cuda:
            torch.cuda.empty_cache()

        return (loss, ovlp, sample_unique, sample_counts, state_prob)

    @torch.no_grad()
    def get_local_ovlp(
        self, prob, psi_sample: Tensor, idx_sample: Tensor, idx_ci: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        local_ovlp = <n|psi_ci><psi_ci|psi>/<n|psi>
        """
        nbatch = psi_sample.size(0)
        oloc = torch.zeros(nbatch, dtype=self.dtype, device=self.device)
        model_psi = self.model(self.ci_state).to(self.dtype)  # <n|psi_ci>
        psi0 = torch.dot(self.pre_ci_coeff.conj(), model_psi)  # <psi_ci|psi>
        # print(f"<psi_ci|psi>: {psi0.norm().item():.20f}")
        # <n|psi_ci><psi_ci|psi>/<n|psi>
        oloc_nonzero = -1 * (self.pre_ci_coeff[idx_ci] * psi0) / psi_sample[idx_sample]
        # print(f"oloc: {oloc_nonzero.norm():.20f}")
        # ovlp part is non-zero, other part is zeros
        oloc[idx_sample] = oloc_nonzero
        ovlp = torch.dot(oloc, prob)
        del oloc_nonzero
        return ovlp, oloc

    def plot_figure(self, prefix: str = None):
        prefix = prefix if prefix is not None else "CI"
        fig = plt.figure()
        x = np.arange(self.pre_max_iter + 1)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(x, np.array(self.loss_list), color="cadetblue", label="Loss")
        ax1.plot(x, np.abs(self.ovlp_list), color="tomato", label="Ovlp")
        ax1.legend(loc="best")
        ax1.set_xlabel("Iteration Time")
        ax1.set_ylabel("Loss/Ovlp")
        # ax2 = fig.add_subplot(3, 1, 2)
        # ax2.plot(x, np.abs(self.ovlp_list), color='tomato', label="Ovlp")
        # ax2.legend(loc="best")
        # ax2.set_xlabel("Iteration Time")
        # ax2.set_ylabel("Ovlp")

        # plot the L2-norm and max-abs of the gradients
        param_L2: List[np.ndarray] = []
        param_max: List[np.ndarray] = []
        for i in range(self.n_para):
            x = np.linalg.norm(np.array(self.grad_lst[i]), axis=1)  # ||g||
            param_L2.append(x)
            x1 = np.abs(np.array(self.grad_lst[i])).max(axis=1)  # max
            param_max.append(x1)
        param_L2 = np.stack(param_L2, axis=1).sum(axis=1)
        param_max = np.stack(param_max, axis=1).max(axis=1)

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(np.arange(len(param_L2)), param_L2, label="||g||")
        ax2.plot(np.arange(len(param_max)), param_max, label="max|g|")
        ax2.set_xlabel("Iteration Time")
        ax2.set_yscale("log")
        ax2.set_ylabel("Gradients")
        plt.title(os.path.split(prefix)[1])  # remove path
        plt.legend(loc="best")

        fig.subplots_adjust(wspace=0, hspace=0.5)
        fig.savefig(prefix + "-pre-train.png", format="png", dpi=1000, bbox_inches="tight")
        np.savez(
            prefix + "-pre-train", np.asarray(self.loss_list), np.asarray(self.ovlp_list), param_L2
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(\n"
            + f"    Pre-train model: {self.model}\n"
            + f"    Pre-train time: {self.pre_max_iter}\n"
            + f"    the number of CI coeff: {self.pre_ci_coeff.shape[0]}\n"
            + f"    Loss function type: {self.loss_type}\n"
            + ")"
        )
