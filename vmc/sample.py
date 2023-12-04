import time
import os
import random
import torch
import torch.distributed as dist
import tempfile
import numpy as np
import pandas as pd

from functools import partial
from typing import Callable, Tuple, List, Union
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
from pandas import DataFrame
from scipy import special

from memory_profiler import profile
from line_profiler import LineProfiler

from vmc.energy import total_energy
from libs.C_extension import (
    onv_to_tensor,
    spin_flip_rand,
    MCMC_sample,
    tensor_to_onv,
    merge_rank_sample,
)
from utils import state_to_string, ElectronInfo, check_para, get_nbatch, diff_rank_seed
from utils.distributed import (
    gather_tensor,
    scatter_tensor,
    get_rank,
    get_world_size,
    synchronize,
    broadcast_tensor,
)
from utils.public_function import torch_unique_index, WavefunctionLUT

print = partial(print, flush=True)


class Sampler:
    """
    Generates samples of configurations from a neural quantum state(NQS)
    using Markov chain Monte Carlo(MCMC) or Auto regressive(AR) algorithm
    """

    METHOD_SAMPLE = ("MCMC", "AR")
    n_accept: int
    str_full: List[str]
    frame_sample: DataFrame

    def __init__(
        self,
        nqs: DDP,
        ele_info: ElectronInfo,
        n_sample: int = 100,
        start_iter: int = 100,
        start_n_sample: int = None,
        therm_step: int = 2000,
        debug_exact: bool = False,
        seed: int = 100,
        record_sample: bool = True,
        max_memory: float = 4,
        alpha: float = 0.25,
        dtype=torch.double,
        method_sample="MCMC",
        max_n_sample: int = None,
        max_unique_sample: int = None,
        use_LUT: bool = False,
        use_unique: bool = True,
        reduce_psi: bool = False,
        eps: float = 1e-12,
        only_AD: bool = False,
        use_sample_space: bool = False,
    ) -> None:
        if n_sample < 50:
            raise ValueError(f"The number of sample{n_sample} should great 50")

        # setup random seed
        self.rank = get_rank()
        self.world_size = get_world_size()
        # self.seed = 22022
        # setup_seed(self.seed)
        if not debug_exact:
            # if sampling, very rank have the different random seed
            self.seed = diff_rank_seed(seed, rank=self.rank)
        else:
            # exact optimization does not require sampling
            self.seed = seed
        logger.info((self.seed, self.rank))

        self.ele_info = ele_info
        self.read_electron_info(self.ele_info)
        self.nqs = nqs
        self.debug_exact = debug_exact
        self.therm_step = therm_step

        if method_sample not in self.METHOD_SAMPLE:
            raise TypeError(
                f"Sample method is invalid: {method_sample}, and expected {self.METHOD_SAMPLE}"
            )
        self.method_sample = method_sample

        # device and cuda
        self.is_cuda = True if self.h1e.is_cuda else False
        self.device = self.h1e.device
        self.dtype = dtype

        # save sampler
        n1 = special.comb(self.noa + self.nva, self.noa, exact=True)
        n2 = special.comb(self.nob + self.nvb, self.nvb, exact=True)
        self.fci_size = n1 * n2
        self.record_sample = record_sample
        if self.record_sample:
            self.str_full = state_to_string(self.ci_space, self.sorb)
            self.frame_sample = pd.DataFrame({"full_space": self.str_full})
            if self.ci_space.size(0) != self.fci_size:
                raise ValueError(
                    f"The dim of full space is {self.ci_space.size(0)} != {self.fci_size}"
                )
        self.time_sample = 0

        # memory control and nbatch
        self.max_memory = max_memory
        self.alpha = alpha

        # unique sample, apply to AR sample
        self.max_unique_sample = (
            min(max_unique_sample, self.fci_size)
            if max_unique_sample is not None
            else self.fci_size
        )
        self.max_n_sample = max_n_sample if max_n_sample is not None else n_sample
        self.min_n_sample = n_sample
        self.last_max_n_sample = self.max_n_sample

        # In the beginning of the iteration, unique sample is quite a lot.
        # so, set the small n_sample.
        if start_n_sample is None or start_n_sample >= n_sample:
            start_n_sample = n_sample
        self.start_iter = start_iter
        self.last_n_sample = n_sample
        self.start_n_sample = start_n_sample
        self.n_sample = start_n_sample

        # Use WaveFunction LooKup-Table to speed up local-energy calculations
        self.use_LUT = use_LUT
        self.WF_LUT: WavefunctionLUT = None

        # Use 'torch.unique' to speed up local-energy calculations
        self.use_unique = use_unique

        # ignore x' when <x|H|x'>/psi(x) < eps
        # only apply to when psi(x)^2 is normalization in FCI-space
        self.reduce_psi = reduce_psi
        self.eps = eps

        # only sampling not calculations local-energy, applies to test AD memory
        self.only_AD = only_AD

        # only use x' in n_unique sample not SD, dose not support exact-opt
        # psi(x') can be looked from WaveFunction look-up table
        self.use_sample_space = use_sample_space if not debug_exact else False

    def read_electron_info(self, ele_info: ElectronInfo):
        if self.rank == 0:
            logger.info(
                f"Read electronic structure information From {ele_info.__name__}", master=True
            )
        self.sorb = ele_info.sorb
        self.nele = ele_info.nele
        self.no = ele_info.nele
        self.nv = ele_info.nv
        self.nob = ele_info.nob
        self.noa = ele_info.noa
        self.nva = ele_info.nva
        self.nvb = ele_info.nvb
        self.h1e: Tensor = ele_info.h1e
        self.h2e: Tensor = ele_info.h2e
        self.ecore = ele_info.ecore
        self.n_SinglesDoubles = ele_info.n_SinglesDoubles
        self.ci_space = ele_info.ci_space

    # @profile(precision=4, stream=open('MCMC_memory_profiler.log','w+'))
    def run(
        self,
        initial_state: Tensor,
        epoch: int,
        n_sweep: int = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Union[complex, float], dict]:
        t0 = time.time_ns()
        check_para(initial_state)
        if self.debug_exact:
            # if self.world_size >= 2:
            #     raise NotImplementedError(f"Exact optimization distributed is not implemented")
            ci_space_rank = scatter_tensor(self.ci_space, self.device, torch.uint8, self.world_size)
            synchronize()
            e_total, eloc, sample_prob, stats_dict = self.calculate_energy(ci_space_rank)

            # placeholders only
            dim = ci_space_rank.shape[0]
            return ci_space_rank.detach(), sample_prob, eloc, e_total, stats_dict

        # AR or MCMC sampling
        sample_unique, sample_counts, sample_prob = self.sampling(
            initial_state, epoch=epoch, n_sweep=n_sweep
        )
        if self.method_sample == "MCMC":
            logger.debug(f"rank: {self.rank} Acceptance ratio = {self.n_accept/self.n_sample:.3E}")

        e_total, eloc, _, stats_dict = self.calculate_energy(
            sample_unique,
            state_prob=sample_prob,
            state_counts=sample_counts,
            WF_LUT=self.WF_LUT,
        )
        # print local energy statistics information
        self._statistics(stats_dict)

        if self.record_sample and self.rank == 0:
            # TODO: if given sorb(not full space), this is error.
            counts = sample_counts.to("cpu").numpy()
            sample_str = state_to_string(sample_unique, self.sorb)
            full_dict = dict.fromkeys(self.str_full, 0)
            for s, i in zip(sample_str, counts):
                full_dict[s] += i
            new_df = pd.DataFrame({self.time_sample: full_dict.values()})
            self.frame_sample = pd.concat([self.frame_sample, new_df], axis=1)
            del full_dict

        self.time_sample += 1
        delta = time.time_ns() - t0
        logger.debug(f"Completed Sampling and calculating eloc {delta/1.0E09:.3E} s")

        if self.is_cuda:
            torch.cuda.empty_cache()
        return sample_unique.detach(), sample_prob, eloc, e_total, stats_dict

    def sampling(
        self,
        initial_state: Tensor,
        epoch: int,
        n_sweep: int = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        prob is not real prob, prob = prob * world-size
        """
        self.change_n_sample(epoch=epoch)
        if self.method_sample == "MCMC":
            sample_unique, sample_counts, sample_prob, WF_LUT = self.MCMC(initial_state, n_sweep)
        elif self.method_sample == "AR":
            (sample_unique, sample_counts, sample_prob, WF_LUT) = self.auto_regressive()
        else:
            raise NotImplementedError(f"Other sampling has not been implemented")

        self.WF_LUT = WF_LUT
        return sample_unique, sample_counts, sample_prob

    def MCMC(
        self, initial_state: Tensor, n_sweep: int = None
    ) -> Tuple[Tensor, Tensor, Tensor, WavefunctionLUT]:
        if self.world_size > 1:
            raise NotImplementedError(f"MCMC distributed has not been implemented")
        # TODO: distributed not been implemented
        # prepare sample and this only apply for MCMC sampling
        self.state_sample: Tensor = torch.empty_like(initial_state).repeat(self.n_sample, 1)
        self.current_state: Tensor = initial_state.clone()
        self.next_state: Tensor = initial_state.clone()
        self.n_accept = 0
        self.psi_sample: Tensor = torch.empty(self.n_sample, dtype=self.dtype, device=self.device)
        if (n_sweep is None) or (n_sweep <= self.therm_step + self.n_sample):
            self.n_sweep = self.therm_step + self.n_sample

        # convert to CPU and 'spin_flip_rand' is not implemented in "GPU"
        if self.is_cuda:
            self.state_sample = self.state_sample.to("cpu")
            self.current_state = self.current_state.to("cpu")
            self.next_state = self.next_state.to("cpu")
            self.nqs = self.nqs.to("cpu")
            self.psi_sample = self.psi_sample.to("cpu")

        # Implement MCMC-Sample in CPP functions
        if False:
            example_inputs = onv_to_tensor(
                self.current_state, self.sorb
            )  # -1:unoccupied, 1: occupied
            serialized_model = torch.jit.trace(self.nqs, example_inputs)
            model_file = tempfile.mkstemp()[1]
            serialized_model.save(model_file)
            # print(f"Serialized model time: {(time.time_ns() - t0)/1.E06:.3f} ms")
            with torch.no_grad():
                self.n_accept = MCMC_sample(
                    model_file,
                    self.current_state,
                    self.state_sample,
                    self.psi_sample,
                    self.sorb,
                    self.nele,
                    self.noa,
                    self.nob,
                    self.seed,
                    self.n_sweep,
                    self.therm_step,
                )
            os.remove(model_file)
            # print(f"CPP model time: {(time.time_ns() - t0)/1.E06:.3f} ms")
        else:
            with torch.no_grad():
                psi_current = self.nqs(onv_to_tensor(self.current_state, self.sorb))
                prob_current = psi_current.norm() ** 2
            for i in range(self.n_sweep):
                psi, self.next_state = spin_flip_rand(
                    self.current_state, self.sorb, self.nele, self.noa, self.nob, self.seed
                )
                with torch.no_grad():
                    psi_next = self.nqs(psi)
                    prob_next = psi_next.norm() ** 2
                prob_accept = min(1.00, (prob_next / prob_current).item())
                p = random.random()
                # print(f"{p:.3E}")
                # if self.verbose and i >= self.therm_step:
                #     print(f"prob_next: {prob_next.item()}, prob_current: {prob_current.item()}")
                #     print(f"random p {p:.3f}, prob_accept {prob_accept:.3f}")
                #     print(f"current state: {self.current_state}")
                if p <= prob_accept:
                    self.current_state = self.next_state.clone()
                    psi_current = psi_next.clone()
                    prob_current = prob_next.clone()
                    if i >= self.therm_step:
                        self.n_accept += 1

                if i >= self.therm_step:
                    self.state_sample[i - self.therm_step] = self.current_state.clone()
                    self.psi_sample[i - self.therm_step] = psi_current.clone()

        if self.is_cuda:
            self.state_sample = self.state_sample.to(self.device)
            self.nqs = self.nqs.to(self.device)

        # remove duplicate state
        # torch.unique could not return unique indices in old tensor,
        # see: the utils/public_function.py 'unique_idx' function
        sample_unique, sample_counts = torch.unique(self.state_sample, dim=0, return_counts=True)
        sample_prob = sample_counts / sample_counts.sum()

        # This is only placeholders
        WF_LUT: WavefunctionLUT = None
        return (sample_unique, sample_counts, sample_prob, WF_LUT)

    def auto_regressive(self) -> Tuple[Tensor, Tensor, Tensor, WavefunctionLUT]:
        """
        Auto regressive sampling
        """
        t0 = time.time_ns()
        while True:
            #  0/1
            sample_unique, sample_counts, wf_value = self.nqs.module.ar_sampling(self.n_sample)

            if sample_unique.size(0) >= self.max_unique_sample:
                # reach lower limit of samples or decreased samples times
                self.n_sample = int(max(self.min_n_sample, self.n_sample // 10))
                break
            else:
                # reach upper limits of samples
                if self.n_sample >= self.max_n_sample:
                    break
                else:
                    # continue AR sampling, increase samples
                    self.n_sample = int(min(self.max_n_sample, self.n_sample * 10))
                    continue
        delta = (time.time_ns() - t0) / 1.0e09

        s = f"Completed {self.method_sample} Sampling: {delta:.3E} s, "
        s += f"unique sample: {sample_counts.sum().item():.3E} -> {sample_counts.size(0)}"
        logger.info(s)

        if True:
            # Sample-comm, gather->merge->scatter
            return self.gather_scatter_sample(sample_unique, sample_counts, wf_value)
        sample_prob = sample_counts / sample_counts.sum()
        # convert to onv
        sample_unique = tensor_to_onv(sample_unique.to(torch.uint8), self.sorb)

        return sample_unique, sample_counts, sample_prob

    def gather_scatter_sample(
        self,
        unique: Tensor,
        counts: Tensor,
        wf_value: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, WavefunctionLUT]:
        """
        1. Gather sample-unique/counts from very rank
        2. Merge all unique/counts in master-rank(rank0)
        3. Scatter unique/counts/prob to very rank

        Meanwhile, All-Gather sample-unique and wf-value in order to make wf-lookup-table

        Notice Scatter prob = prob * world-size

        Returns
        -------
            unique_rank: Tensor
            counts_rank: Tensor
            prob_rank: Tensor, prob_rank = prob * world_size
            WF_LUT: wavefunction LookUP-Table about all-sample-unique and wf-value
        """
        t0 = time.time_ns()
        # Gather unique, counts, wf_value
        unique_all = gather_tensor(unique, self.device, self.world_size, master_rank=0)
        count_all = gather_tensor(counts, self.device, self.world_size, master_rank=0)
        if self.use_LUT:
            wf_value_all = gather_tensor(wf_value, self.device, self.world_size, master_rank=0)
        else:
            wf_value_all = None
        # sample_all = gather_tensor(sample, self.device, self.word_size, master_rank=0)
        synchronize()
        t1 = time.time_ns()
        if self.rank == 0:
            split_idx = [0] + [i.shape[0] for i in unique_all]
            split_idx = torch.tensor(split_idx, dtype=torch.int64).cumsum(dim=0)
            unique_all = torch.cat(unique_all)
            merge_unique, merge_inv, merge_idx = torch_unique_index(unique_all)[:3]
            if self.use_LUT:
                wf_value_all = torch.cat(wf_value_all)
                wf_value_unique = wf_value_all[merge_idx]
            # merge_unique, merge_
            count_all = torch.cat(count_all)
            # nbatch = unique_all.shape[0]
            # merge_counts = torch.zeros(merge_unique.shape[0], dtype=torch.int64, device=self.device)

            # merge prob
            length = merge_unique.shape[0]
            merge_counts = merge_rank_sample(merge_inv, count_all, split_idx, length)
            # for i in range(nbatch):
            #     merge_counts[merge_idx[i]] += count_all[i]
            merge_prob = merge_counts / merge_counts.sum()
            # _, counts_test = torch.unique(torch.cat(sample_all), dim=0, return_counts=True)
            # assert(torch.allclose(counts_test, merge_counts))
        else:
            merge_counts: Tensor = None
            merge_unique: Tensor = None
            merge_prob: Tensor = None
            wf_value_unique: Tensor = None

        t2 = time.time_ns()
        # Scatter unique, counts
        unique_rank = scatter_tensor(merge_unique, self.device, torch.int64, self.world_size)
        counts_rank = scatter_tensor(merge_counts, self.device, torch.int64, self.world_size)
        prob_rank = scatter_tensor(merge_prob, self.device, torch.double, self.world_size)

        t3 = time.time_ns()
        if self.use_LUT:
            # XXX: unique_rank split merge_unique when broadcast to all-rank,
            # this maybe efficiency than scatter->broadcast
            merge_unique = broadcast_tensor(merge_unique, self.device, torch.int64, master_rank=0)
            wf_value_unique = broadcast_tensor(
                wf_value_unique, self.device, self.dtype, master_rank=0
            )
        synchronize()
        t4 = time.time_ns()

        if self.rank == 0:
            delta1 = (t1 - t0) / 1.0e09
            delta2 = (t3 - t2) / 1.0e09
            delta3 = (t2 - t1) / 1.0e09
            delta4 = (t4 - t3) / 1.0e09
            s = f"Sample-Comm, Gather: {delta1:.3E} s, Scatter: {delta2:.3E} s, merge: {delta3:.3E} s\n"
            s += f"All-Rank unique sample: {merge_unique.size(0)}, Broadcast LUT: {delta4:.3E} s"
            logger.info(s, master=True)
        # convert to onv
        unique_rank = tensor_to_onv(unique_rank.to(torch.uint8), self.sorb)

        if self.use_LUT:
            WF_LUT = WavefunctionLUT(
                tensor_to_onv(merge_unique.to(torch.uint8), self.sorb),
                wf_value_unique,
                self.sorb,
                self.device,
            )
        else:
            WF_LUT: WavefunctionLUT = None

        del merge_counts, merge_prob, unique_all, count_all, wf_value_all

        return (unique_rank, counts_rank, prob_rank * self.world_size, WF_LUT)

    # TODO: how to calculate batch_size;
    # calculate the max nbatch for given Max Memory
    def calculate_energy(
        self,
        sample: Tensor,
        state_prob: Tensor = None,
        state_counts: Tensor = None,
        WF_LUT: WavefunctionLUT = None,
    ) -> Tuple[Union[complex, float], Tensor, Tensor, dict]:
        r"""
        Returns:
        -------
            e_total(complex|float): total energy(a.u.)
            eloc(Tensor): local energy
            placeholders(Tensor): state prob if exact optimization, else zeros tensor
            stats_dict(dict): statistical information about sampling.
        """
        # this is applied when pre-train
        WF_LUT = self.WF_LUT if WF_LUT is None else WF_LUT
        nbatch = get_nbatch(
            self.sorb,
            len(sample),
            self.n_SinglesDoubles,
            self.max_memory,
            self.alpha,
            device=self.device,
            use_sample=self.use_sample_space,
        )
        if not self.only_AD:
            e_total, eloc, placeholders, stats_dict = total_energy(
                sample,
                nbatch,
                h1e=self.h1e,
                h2e=self.h2e,
                ansatz=self.nqs,
                ecore=self.ecore,
                sorb=self.sorb,
                nele=self.nele,
                noa=self.noa,
                nob=self.nob,
                state_prob=state_prob,
                state_counts=state_counts,
                exact=self.debug_exact,
                WF_LUT=WF_LUT,
                use_unique=self.use_unique,
                dtype=self.dtype,
                reduce_psi=self.reduce_psi,
                eps=self.eps,
                use_sample_space=self.use_sample_space,
            )
        else:
            e_total = -2.33233
            eloc = torch.rand(sample.size(0), device=self.device, dtype=self.dtype)
            stats_dict = {}
            placeholders = torch.zeros(1, device=self.device, dtype=self.dtype)
        return e_total, eloc, placeholders, stats_dict

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}:"
            + " (\n"
            + f"    Sample method: {self.method_sample}\n"
            + f"    the number of sample: {self.last_n_sample:.3E}\n"
            + f"    first {self.start_iter}-th sample: {self.start_n_sample:.3E}\n"
            + f"    Using LUT: {self.use_LUT}\n"
            + f"    local energy unique: {self.use_unique}\n"
            + f"    Reduce psi: {self.reduce_psi}\n"
            + f"    eps: {self.eps:.3E}\n"
            + f"    only use sample-space: {self.use_sample_space}\n"
            + f"    Therm step: {self.therm_step}\n"
            + f"    Exact sampling: {self.debug_exact}\n"
            + f"    Given CI: {self.ci_space.size(0):.3E}\n"
            + f"    FCI space: {self.fci_size:.3E}\n"
            + f"    Record the sample: {self.record_sample}\n"
            + f"    Singles + Doubles: {self.n_SinglesDoubles}\n"
            + f"    Max unique sample: {self.max_unique_sample:3E}\n"
            + f"    Max sample: {self.max_n_sample:3E}\n"
            + f"    Random seed: {self.seed}\n"
            + f"    alpha: {self.alpha}\n"
            + f"    max_memory: {self.max_memory}\n"
            + ")"
        )

    def _statistics(self, data: dict):
        if self.only_AD:
            s = f"**This Auto-grad memory testing**"
        else:
            s = f"E_total = {data['mean'].real:.10f} ± {data['SE'].real:.3E} [σ² = {data['var'].real:.3E}]"
        logger.debug(s)

    def change_n_sample(self, epoch: int) -> None:
        """
        change the number of sample in the beginning of iteration.
        """
        if epoch <= self.start_iter:
            self.n_sample = self.start_n_sample
            self.max_n_sample = self.n_sample
            self.min_n_sample = self.n_sample
        elif epoch == self.start_iter + 1:
            self.n_sample = self.last_n_sample
            self.max_n_sample = self.last_max_n_sample
            self.min_n_sample = self.n_sample
