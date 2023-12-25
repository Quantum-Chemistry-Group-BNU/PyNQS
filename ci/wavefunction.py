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
from utils.public_function import (
    check_para,
    ElectronInfo,
    find_common_state,
    state_to_string,
    split_length_idx,
    split_batch_idx,
)
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
    norm: float

    def __init__(
        self,
        coeff: Union[Tensor, ndarray],
        onstate: Tensor,
        norm_coeff: bool = False,
        device: str = None,
    ) -> None:
        self.device = device
        assert isinstance(coeff, (ndarray, Tensor))
        check_para(onstate)  # onstate is torch.uint8

        if isinstance(coeff, ndarray):
            # convert to torch.Tensor
            self.coeff = torch.from_numpy(coeff).clone().to(self.device)
        else:
            # clone avoid shallow copy
            self.coeff = coeff.clone().to(self.device)
        if norm_coeff:
            # normalization coeff
            self.coeff = self.coeff / self.coeff.norm()
        self.norm = self.coeff.norm().item()
        self.space = onstate.to(self.device)
        # check dim
        assert self.space.shape[0] == self.coeff.shape[0]

    def energy(self, e: ElectronInfo) -> float:
        h1e = e.h1e.to(self.device)
        h2e = e.h2e.to(self.device)
        sorb = e.sorb
        ecore = e.ecore
        nele = e.nele
        return energy_CI(self.coeff, self.space, h1e, h2e, ecore, sorb, nele)

    def __repr__(self) -> str:
        s = f" CI shape: {self.coeff.shape[0]}, norm: {self.norm():.6f}"
        return f"{type(self).__name__, {s}}"


@torch.no_grad()
def energy_CI(
    coeff: Tensor,
    onstate: Tensor,
    h1e: Tensor,
    h2e: Tensor,
    ecore: float,
    sorb: int,
    nele: int,
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
    LOSS_TYPE = ("sample", "onstate", "lsm-phase", "lsm")

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
        AD_MAX_DIM: int = -1,
        FR_MAX_DIM: int = -1,
    ) -> None:
        r"""
        Pre train CI wavefunction.

        Parameters
        ----------
            model: the nqs model
            opt: Optimizer
            pre_CI: the pre_trained CI wavefunction
            pre_train_info: a dict include "pre_max_inter", "interval" and "loss_type"
            pre_max_inter: the max time of pre-train, default: 200
            interval: print time, if == -1; print each step 'loss ' and 'ovlp', default: -1
            loss_type: the type of loss functions, there are four loss functions:
                ('sample', 'onstate', "lsm-phase", "lsm"),
                default: 'onstate'
                'sample': this is similar to VMC progress, 'see function 'QGT_loss'.
                    notice: loss is meaningless, and focus on ovlp
                'onstate': only fits CI-space coefficient and usually is imprecise,
                    but could get relatively well initial parameters.
                'lsm': Least Squares Method fits coefficient and used in auto-regressive ansatz (RNN, Transformer)
                'lsm-phase', this is similar to 'lsm' and with Global-Phase exp(i * phi)
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
        # FIXME:(zbwu-23-12-22): check multi-Rank
        self.ci_state = onv_to_tensor(self.onstate, self.sorb)
        self.ci_norm = pre_CI.norm

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
            raise TypeError(f"Loss function type{self.loss_type} not in {self.LOSS_TYPE}")

        # global-phase exp(i * phi)
        self.use_global_phase = False
        if self.loss_type in ("lsm-phase"):
            self.use_global_phase = True

        # only support AR-ansatz, RNN, transformer, AR-RBM
        if self.loss_type in ("lsm", "lsm-phase"):
            if not hasattr(self.model.module, "ar_sampling"):
                raise TypeError(f"loss function {self.loss_type} only support AR-ansatz")

        # split-different rank
        dim = pre_CI.coeff.size(0)
        idx_lst = [0] + split_length_idx(dim, length=self.world_size)
        begin = idx_lst[self.rank]
        end = idx_lst[self.rank + 1]
        self.rank_ci_state = self.ci_state[begin:end]
        self.rank_pre_ci_coeff = self.pre_ci_coeff[begin:end]

        # record optim
        self.n_para = len(list(self.model.parameters()))
        self.grad_e_lst = [[], []]  # grad_L2, grad_max

        # max auto-grad/backward dim
        self.AD_MAX_DIM = AD_MAX_DIM
        # max forward dim
        self.FR_MAX_DIM = FR_MAX_DIM

    def train(
        self,
        prefix: str = None,
        electron_info: ElectronInfo = None,
        sampler: Sampler = None,
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
            # FIXME:(zbwu-23-12-22) OOM if Fe2S2
            get_energy = partial(
                energy_CI,
                h1e=electron_info.h1e,
                h2e=electron_info.h2e,
                ecore=electron_info.ecore,
                sorb=electron_info.sorb,
                nele=electron_info.nele,
            )
            dim = self.pre_ci_coeff.size(0)
            if dim > 10000:
                # e = \sum_{ij}c_i<i|H|j>c_j*
                warnings.warn(
                    f"CI-coeff dim: {dim} to large, does not calculate e-ref avoiding OOM"
                )
                self.eCI_ref = 0.00
            else:
                self.eCI_ref = get_energy(self.pre_ci_coeff, self.onstate)

        # logging
        if self.rank == 0:
            logger.info(f"Begin pre-train: {time.ctime()}", master=True)
            if self.loss_type == "sample":
                s = f"NOTICE:{'*'*20}"
                s += f"Loss is meaningless, and focus on ovlp{'*'*20}"
                logger.info(s, master=True)
            elif self.loss_type == "onstate" and self.world_size > 1:
                raise SyntaxError("Loss-type(onstate) dose not support distributed pre-train")

        # begin pre-train
        begin = time.time_ns()
        self.opt.zero_grad()
        self.sampler = sampler
        for epoch in range(self.pre_max_iter + 1):
            t0 = time.time_ns()
            if self.loss_type == "onstate":
                loss, ovlp, model_coeff = self.sqaure_loss()
            elif self.loss_type == "sample":
                loss, ovlp, sample_unique, sample_counts, state_prob = self.QGT_loss(epoch)
            elif self.loss_type in "lsm":
                loss, ovlp = self.test_phase_loss(use_global_phase=False)
            elif self.loss_type == "lsm-phase":
                loss, ovlp = self.test_phase_loss(use_global_phase=True)

            if epoch <= self.pre_max_iter:
                self.opt.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            # if self.rank == 0:
            #     for param in self.model.parameters():
            #         print(param.grad.reshape(-1))
            #         break

            # record L2-grad and Max-grad
            if self.rank == 0:
                x: List[np.ndarray] = []
                for i, param in enumerate(self.model.parameters()):
                    if param.grad is not None:
                        x.append(param.grad.reshape(-1).detach().to("cpu").numpy())
                x = np.concatenate(x)
                l2_grad = np.linalg.norm(x)
                max_grad = np.abs(x).max()
                self.grad_e_lst[0].append(l2_grad)
                self.grad_e_lst[1].append(max_grad)
            self.opt.zero_grad()

            # if self.rank == 0:
            #     print(f"grad-abs-sum: {grad_sum:.20f}")
            #     print(f"param-abs-sum: {param_sum:.20f}")
            # save ovlp and loss-functions
            # logger.info(f"ovlp: {ovlp.norm().item():.5f}, loss: {loss.item():.5f}")

            reduce_loss = all_reduce_tensor(loss, world_size=self.world_size, in_place=False)[0]
            reduce_ovlp = all_reduce_tensor(ovlp, world_size=self.world_size, in_place=False)[0]
            synchronize()
            if self.rank == 0:
                self.ovlp_list.append(reduce_ovlp.norm().detach().to("cpu").item())
                self.loss_list.append(reduce_loss.detach().to("cpu").item())

            # calculate energy from CI coefficient or local-energy
            # notice, the shape of model_CI maybe not equal of self.pre_ci if using "sample"
            if (epoch == 0) or (epoch == self.pre_max_iter):
                if flag_energy:
                    if self.loss_type == "onstate":
                        # avoid OOM
                        # e_total = get_energy(model_coeff, onstate)
                        e_total = self.sampler.run(initial_state=self.onstate[0], epoch=0)[3]
                    elif self.loss_type == "sample":
                        # single-Rank
                        e_rank = self.sampler.calculate_energy(
                            sample_unique, state_prob, sample_counts
                        )[0]
                        e_rank = torch.tensor(e_rank, dtype=self.dtype, device=self.device)
                        all_reduce_tensor(e_rank, world_size=self.world_size)
                        synchronize()
                        e_total = e_rank.item().real
                    elif self.loss_type in ("lsm-phase", "lsm"):
                        # calculate total-energy use VMC
                        e_total = self.sampler.run(initial_state=self.onstate[0], epoch=0)[3]
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
        # e_ci = get_energy(self.model(self.ci_state), self.onstate)
        # p = self.sampler.run(initial_state=self.onstate[0], epoch=0)
        # fci_space = self.sampler.ci_space
        # e_ci1 = get_energy(self.model((onv_to_tensor(fci_space, self.sorb))), fci_space)
        # print(e_ci, p[3], e_ci1)
        # breakpoint()
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

        states_sample = onv_to_tensor(sample_unique, self.sorb)  # uint8 -> +1/-1
        if self.exact:
            # Gather all psi from very rank
            psi = self.model(states_sample).to(self.dtype)
            psi_sample = psi.clone().detach()
            all_psi = gather_tensor(psi_sample, self.device, self.world_size)
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
        if not self.exact:
            if self.sampler.use_LUT:
                # use Wavefunction LUT
                not_idx, psi_sample = self.sampler.WF_LUT.lookup(sample_unique)[1:]
                # sample_unique from sampling, so sample-unique must been found in WF_LUT.
                assert not_idx.size(0) == 0
            else:
                warnings.warn(f"use 'WF_LUT = True' to reduce calculation")
                psi_sample = self.model(states_sample).to(self.dtype)

        ovlp, oloc = self.get_local_ovlp(state_prob, psi_sample, idx_sample, idx_ci)
        ovlp_mean = all_reduce_tensor(ovlp, world_size=self.world_size, in_place=False)[0]

        # self.AD_MAX_DIM = 20
        loss = self.sample_ovlp_grad(states_sample, oloc, ovlp_mean, state_prob, self.AD_MAX_DIM)

        del idx_ci, idx_sample, oloc
        if loss.is_cuda:
            torch.cuda.empty_cache()

        return (loss, ovlp, sample_unique, sample_counts, state_prob)

    def sample_ovlp_grad(
        self,
        states: Tensor,
        oloc: Tensor,
        ovlp_mean: Tensor,
        state_prob: Tensor,
        AD_MAX_DIM: int = -1,
    ) -> Tensor:
        """
        this is similar to vmc/grad/energy_grad/_ad_grad
        """
        dim = oloc.size(0)
        if AD_MAX_DIM == -1:
            min_batch = dim
        else:
            min_batch = AD_MAX_DIM
        idx_lst = split_batch_idx(dim=dim, min_batch=min_batch)
        # logger.info((oloc.dim(), min_batch))
        # logger.info(f"idx_lst: {idx_lst}")

        loss_sum = torch.zeros(1, device=oloc.device, dtype=torch.double)

        def batch_loss_backward(begin: int, end: int) -> None:
            nonlocal loss_sum
            log_psi_batch = self.model(states[begin:end].requires_grad_()).to(self.dtype).log()
            state_prob_batch = state_prob[begin:end]
            oloc_batch = oloc[begin:end]

            if torch.any(torch.isnan(log_psi_batch)):
                raise ValueError(
                    f"There are negative numbers in the log-psi, please use complex128"
                )

            loss1 = torch.sum(oloc_batch * log_psi_batch * state_prob_batch)
            loss2 = ovlp_mean * torch.sum(log_psi_batch.conj() * state_prob_batch)
            loss = 2 * (loss1 - loss2).real
            loss.backward()
            loss_sum += loss.detach()

            # clean memory cache
            if loss.is_cuda:
                torch.cuda.empty_cache()

        # disable gradient synchronizations in the rank
        begin = 0
        with self.model.no_sync():
            for i in range(len(idx_lst) - 1):
                end = idx_lst[i]
                batch_loss_backward(begin, end)
                begin = end

        end = idx_lst[-1]
        # synchronization gradient in the rank
        batch_loss_backward(begin, end)

        return loss_sum

    @torch.no_grad()
    def get_local_ovlp(
        self, prob, psi_sample: Tensor, idx_sample: Tensor, idx_ci: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""
        calculate local ovlp = <n|psi_ci><psi_ci|psi>/<n|psi>

        Parameters
        ----------
            prob(Tensor): the probability of sample (Single-Rank)
            psi_sample(Tensor): the WF of sample (Single-Rank)
            idx_sample: the index of the common state in the sample(Single-Rank)
            idx_ci: the index of the common state in the pre-CI state(Single-Rank)

        Returns
        -------
            ovlp(Tensor): ovlp(Single-Rank)
            oloc(Tensor): local ovlp(Single-Rank)
        """
        # Single-rank sample
        oloc = torch.zeros(psi_sample.size(0), dtype=self.dtype, device=self.device)
        # <psi_ci|psi>
        psi0 = self._get_rank_psi0(self.FR_MAX_DIM)
        # <n|psi_ci><psi_ci|psi>/<n|psi>
        oloc_nonzero = -1 * (self.pre_ci_coeff[idx_ci] * psi0) / psi_sample[idx_sample]
        # ovlp part is non-zero, other part is zeros
        oloc[idx_sample] = oloc_nonzero
        # divide <psi_ci|psi_ci>
        ovlp = torch.dot(oloc, prob) / self.ci_norm
        del oloc_nonzero
        return ovlp, oloc

    @torch.no_grad()
    def _get_rank_psi0(self, FR_MAX_DIM: int = -1) -> Tensor:
        """
        calculate <psi_ci|psi> using distributed

        Parameters
        ----------
            FR_MAX_DIM(int): default: -1

        Returns
        -------
            psi0_all(Tensor), <psi_ci|psi>(all-rank)
        """
        dim = self.rank_ci_state.size(0)
        if FR_MAX_DIM == -1:
            min_batch = dim
        else:
            min_batch = self.FR_MAX_DIM
        idx_lst = [0] + split_batch_idx(dim=dim, min_batch=min_batch)

        # <n|psi_ci> single-rank
        psi_rank = torch.empty(dim, dtype=self.dtype, device=self.device)
        for i in range(len(idx_lst) - 1):
            begin, end = idx_lst[i], idx_lst[i + 1]
            ci_state_batch = self.rank_ci_state[begin:end]
            psi_rank[begin:end] = self.model(ci_state_batch).to(self.dtype)

        # <psi_ci|psi> single-rank
        psi0 = torch.dot(self.rank_pre_ci_coeff.conj(), psi_rank) * self.world_size

        # all-rank
        all_reduce_tensor(psi0, world_size=self.world_size)
        synchronize()

        return psi0

    def test_phase_loss(self, use_global_phase: bool = True) -> Tuple[Tensor, Tensor]:
        """
        Loss = \sum_i (psi_i - psi_i^{ci})**2
        or \sum_i (psi_i * exp(i * phi) - psi_i^{ci})**2

        Returns
        -------
            loss(Tensor):
            ovlp(Tensor):
        """
        # self.AD_MAX_DIM = 10
        loss, ovlp = self.test_ovlp_grad(self.AD_MAX_DIM, use_global_phase=use_global_phase)
        # if self.rank == 0 and use_global_phase:
        #     logger.info(f"global-phase: {self.model.module.global_phase()}")

        return loss, ovlp.detach()

    def test_ovlp_grad(
        self,
        AD_MAX_DIM: int = -1,
        use_global_phase: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Loss = \sum_i (psi_i - psi_i^{ci})**2
        or = \sum_i (psi_i * exp(i * phi) - psi_i^{ci})**2
        """
        dim = self.rank_ci_state.size(0)
        device = self.rank_ci_state.device
        if AD_MAX_DIM == -1:
            min_batch = dim
        else:
            min_batch = AD_MAX_DIM
        idx_lst = split_batch_idx(dim=dim, min_batch=min_batch)

        loss_sum = torch.zeros(1, device=device, dtype=torch.double)
        ovlp_sum = torch.zeros(1, device=device, dtype=self.dtype)

        def batch_loss_backward(begin: int, end: int) -> None:
            nonlocal loss_sum, ovlp_sum
            ci_state_batch = self.rank_ci_state[begin:end].requires_grad_()
            ci_coeff_batch = self.rank_pre_ci_coeff[begin:end]
            psi_batch = self.model(ci_state_batch, use_global_phase=use_global_phase)
            # * self.world_size: DDP All-Reduce
            loss = torch.sum((psi_batch - ci_coeff_batch).norm().pow(2)) * self.world_size
            loss.backward()

            with torch.no_grad():
                # * self.world_size: DDP All-Reduce
                ovlp = torch.dot(psi_batch, ci_coeff_batch) * self.world_size
                ovlp_sum += ovlp.detach()

            loss_sum += loss.detach()
            # clean memory cache
            if loss.is_cuda:
                torch.cuda.empty_cache()

        # disable gradient synchronizations in the rank
        begin = 0
        with self.model.no_sync():
            for i in range(len(idx_lst) - 1):
                end = idx_lst[i]
                batch_loss_backward(begin, end)
                begin = end

        end = idx_lst[-1]
        # synchronization gradient in the rank
        batch_loss_backward(begin, end)

        return loss_sum, ovlp_sum

    def plot_figure(self, prefix: str = None) -> None:
        prefix = prefix if prefix is not None else "CI"
        fig = plt.figure()

        # plot ovlp and loss
        x = np.arange(self.pre_max_iter + 1)
        ax = fig.add_subplot(2, 1, 1)
        line1 = ax.plot(x, np.array(self.loss_list), color="cadetblue", label="Loss")
        ax.set_ylabel("Loss")
        ax1 = ax.twinx()
        line2 = ax1.plot(x, np.abs(self.ovlp_list), color="tomato", label="Ovlp")
        ax1.set_ylabel("ovlp")
        lines = line1 + line2
        labels = [name.get_label() for name in lines]
        ax.legend(lines, labels, loc="best")

        # plot the L2-norm and max-abs of the gradients
        param_L2 = np.asarray(self.grad_e_lst[0])
        param_max = np.asarray(self.grad_e_lst[1])
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(np.arange(len(param_L2)), param_L2, label=r"$||g||$")
        ax2.plot(np.arange(len(param_max)), param_max, label=r"$||g||_{\infty}$")
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
            # + f"    Pre-train model: {self.model}\n" # avoid repeat print
            + f"    Pre-train time: {self.pre_max_iter}\n"
            + f"    the number of CI coeff: {self.pre_ci_coeff.shape[0]}\n"
            + f"    CI-coeff norm: {self.ci_norm:.6f}\n"
            + f"    Loss function type: {self.loss_type}\n"
            + f"    AD_MAX_DIM: {self.AD_MAX_DIM}\n"
            + f"    FR_MAX_DIM: {self.FR_MAX_DIM}\n"
            + ")"
        )
