from __future__ import annotations

import time
import os
import random
import torch
import torch.distributed as dist
import tempfile
import warnings
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
from vmc.stats import operator_statistics

from libs.C_extension import (
    onv_to_tensor,
    spin_flip_rand,
    MCMC_sample,
    tensor_to_onv,
    merge_rank_sample,
)
from utils import state_to_string, ElectronInfo, check_para, get_nbatch, diff_rank_seed
from utils.distributed import (
    all_gather_tensor,
    all_reduce_tensor,
    gather_tensor,
    scatter_tensor,
    get_rank,
    get_world_size,
    synchronize,
    broadcast_tensor,
)
from utils.public_function import (
    torch_unique_index,
    WavefunctionLUT,
    split_length_idx,
    split_batch_idx,
)
from utils.det_helper import DetLUT
from utils.pyscf_helper.operator import spin_raising

print = partial(print, flush=True)


class Sampler:
    """
    Generates samples of configurations from a neural quantum state(NQS)
    using Markov chain Monte Carlo(MCMC) or Auto regressive(AR) algorithm
    """

    METHOD_SAMPLE = ("MCMC", "AR", "RESTRICTED")
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
        use_same_tree: bool = False,
        max_n_sample: int = None,
        max_unique_sample: int = None,
        use_LUT: bool = False,
        use_unique: bool = True,
        reduce_psi: bool = False,
        eps: float = 1e-12,
        only_AD: bool = False,
        use_sample_space: bool = False,
        min_batch: int = 10000,
        min_tree_height: int = None,
        det_lut: DetLUT = None,
        use_dfs_sample: bool = False,
        use_spin_raising: bool = False,
        spin_raising_coeff: float = 1.0,
        given_state: Tensor = None,
    ) -> None:
        if n_sample < 50:
            raise ValueError(f"The number of sample{n_sample} should great 50")

        # setup random seed
        self.rank = get_rank()
        self.world_size = get_world_size()
        # self.seed = 22022
        # setup_seed(self.seed)
        # use_same_tree = True
        self.use_same_tree = use_same_tree
        if not debug_exact and not use_same_tree:
            # if sampling, very rank have the different random seed
            self.seed = diff_rank_seed(seed, rank=self.rank)
        else:
            # exact optimization does not require sampling
            # the different rank sampling using the the same QuadTree or BinaryTree
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
        if self.debug_exact and self.ci_space.size(0) != self.fci_size:
            raise ValueError(f"Dim of FCI space is {self.fci_size} != {self.ci_space.size(0)}")
        self.record_sample = record_sample
        if self.record_sample:
            self.str_full = state_to_string(self.ci_space, self.sorb)
            self.frame_sample = pd.DataFrame({"full_space": self.str_full})
        self.time_sample = 0

        # memory control and nbatch
        self.max_memory = max_memory
        self.alpha = alpha

        # unique sample, apply to AR sample, about all-rank, not single-rank
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

        # nbatch-rank AR-sampling, only implemented in Transformer-ansatz
        flag1 = hasattr(self.nqs.module, "min_batch")
        flag2 = hasattr(self.nqs.module, "min_tree_height")
        self.sampling_batch_rank: bool = flag1 and flag2
        self.sample_min_sample_batch = min_batch
        if min_tree_height is not None:
            assert (
                self.use_same_tree
            ), f"use-same-tree({self.use_same_tree}) muse be is True, if use min-tree-height"
        self.sample_min_tree_height = min_tree_height
        # DFS Sample, default BFS
        self.use_dfs_sample = use_dfs_sample
        if self.use_dfs_sample and not self.sampling_batch_rank:
            raise TypeError(f"DFS only be supported in Multi-Rank-Sampling")

        # Det-LUT, remove part det in CI-NQS
        self.remove_det = False
        self.det_lut: DetLUT = None
        if det_lut is not None:
            self.remove_det = True
            self.det_lut = det_lut

        # <S-S+>
        self.spin_raising_param = spin_raising_coeff
        self.use_spin_raising = use_spin_raising
        self.h1e_spin: Tensor = None
        self.h2e_spin: Tensor = None
        if self.spin_raising_param < 1e-5:
            # self.use_spin_raising = False
            warnings.warn(f"<S-S+> Penalty: {self.spin_raising_param:.5E} too little")
        if self.use_spin_raising:
            x = spin_raising(self.sorb, c1=1.0)
            self.h1e_spin = x[0].to(self.device)
            self.h2e_spin = x[1].to(self.device)

        # Testing, not-sampling.
        self.given_state: Tensor = None
        if self.method_sample == "RESTRICTED":
            if not isinstance(given_state, Tensor):
                raise TypeError(f"Given-state muse be Tensor")
            if given_state.shape[1] != self.sorb:
                raise TypeError(f"Given-state: {tuple(given_state)} must be (nbatch, sorb)")
            self._init_restricted(given_state=given_state)

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
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """
        run sampling using 'MCMC' or 'AR' algorithm and calculate local energy

        Parameters
        ----------
            initial_state(Tensor): the initial-state, only used in MCMC
            epoch(int): the number of VMC iterations, used in changing N-sample
            n_sweep(int): the total cycle, only used in MCMC

        Returns
        -------
            sample_unique(Tensor): the unique of sample (Single-Rank)
            sample_prob(Tensor): the probability of sample (Single-Rank)
            (eloc, sloc): local energy, local-spin(S-S+) (Single-Rank)
            (eloc_mean, sloc_mean): the average of eloc/sloc (All-Rank)
        """
        t0 = time.time_ns()
        check_para(initial_state)
        if self.debug_exact:
            func = self.run_exact
        else:
            func = self.run_sampling

        result = func(initial_state, epoch, n_sweep)
        delta = time.time_ns() - t0
        if self.rank == 0:
            s = f"Completed Sampling and calculating eloc {delta/1.0E09:.3E} s"
            logger.info(s, master=True)

        if self.is_cuda:
            torch.cuda.empty_cache()
        return result

    def run_exact(
        self,
        initial_state: Tensor,
        epoch: int,
        n_sweep: int = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        if self.remove_det:
            # avoid wf is 0.00, if not found, set to -1
            array_idx = self.det_lut.lookup(self.ci_space, is_onv=True)[0]
            ci_space = self.ci_space[~array_idx.gt(-1)]
        else:
            ci_space = self.ci_space
        ci_space_rank = scatter_tensor(ci_space, self.device, torch.uint8, self.world_size)
        synchronize()
        eloc, sloc, sample_prob = self.calculate_energy(ci_space_rank)
        # All-Reduce mean local energy

        stats_eloc = operator_statistics(eloc, sample_prob, float("inf"), "E")
        eloc_mean = stats_eloc["mean"]
        # e_total = eloc_mean + self.ecore
        if self.rank == 0:
            logger.info(str(stats_eloc), master=True)

        if self.use_spin_raising:
            stats_sloc = operator_statistics(sloc, sample_prob, float("inf"), "S-S+")
            sloc_mean = stats_sloc["mean"]
            if self.rank == 0:
                logger.info(str(stats_sloc), master=True)
        else:
            sloc_mean = torch.tensor(0.0, device=self.device)

        return ci_space_rank.detach(), sample_prob, (eloc, sloc), (eloc_mean, sloc_mean)

    def run_sampling(
        self,
        initial_state: Tensor,
        epoch: int,
        n_sweep: int = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        # AR or MCMC sampling
        sample_unique, _, sample_prob = self.sampling(initial_state, epoch=epoch, n_sweep=n_sweep)
        if self.method_sample == "MCMC":
            logger.debug(f"rank: {self.rank} Acceptance ratio = {self.n_accept/self.n_sample:.3E}")

        # Single-Rank
        eloc, sloc, _ = self.calculate_energy(
            sample_unique,
            state_prob=sample_prob,
            WF_LUT=self.WF_LUT,
        )

        # All-Reduce mean local energy
        stats_eloc = operator_statistics(eloc, sample_prob, self.n_sample, "E")
        eloc_mean = stats_eloc["mean"]
        if self.rank == 0:
            # print(stats_eloc)
            logger.info(str(stats_eloc), master=True)

        if self.use_spin_raising:
            stats_sloc = operator_statistics(sloc, sample_prob, self.n_sample, "S-S+")
            sloc_mean = stats_sloc["mean"]
            if self.rank == 0:
                logger.info(str(stats_sloc), master=True)
        else:
            sloc_mean = torch.zeros_like(eloc_mean)

        # Record sampling , only in MCMC.
        # self.time_sample += 1
        # if self.record_sample and self.rank == 0:
        #     # If given space(not full space), this is error.
        #     counts = sample_counts.to("cpu").numpy()
        #     sample_str = state_to_string(sample_unique, self.sorb)
        #     full_dict = dict.fromkeys(self.str_full, 0)
        #     for s, i in zip(sample_str, counts):
        #         full_dict[s] += i
        #     new_df = pd.DataFrame({self.time_sample: full_dict.values()})
        #     self.frame_sample = pd.concat([self.frame_sample, new_df], axis=1)
        #     del full_dict

        return sample_unique.detach(), sample_prob, (eloc, sloc), (eloc_mean, sloc_mean)

    def sampling(
        self,
        initial_state: Tensor,
        epoch: int,
        n_sweep: int = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        prob is not real prob, prob = prob * world-size
        Notice: samples counts is placeholders/empty-Tensor

        Returns
        -------
            sample-unique: Tensor(Single-Rank)
            sample_counts: only is placeholders/empty-Tensor
            sample-prob: Tensor(Single-Rank)
        """
        self.change_n_sample(epoch=epoch)
        if self.method_sample == "MCMC":
            sample_unique, sample_counts, sample_prob, WF_LUT = self.MCMC(initial_state, n_sweep)
        elif self.method_sample == "AR":
            (sample_unique, sample_counts, sample_prob, WF_LUT) = self.auto_regressive()
        elif self.method_sample == "RESTRICTED":
            (sample_unique, sample_counts, sample_prob, WF_LUT) = self.restricted_sample()
        else:
            raise NotImplementedError(f"Other sampling has not been implemented")

        self.WF_LUT = WF_LUT
        # Notice: samples counts is placeholders/empty tensor
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
            if not self.sampling_batch_rank:
                sample_unique, sample_counts, wf_value = self.nqs.module.ar_sampling(self.n_sample)
            else:
                sample_unique, sample_counts, wf_value = self.nqs.module.ar_sampling(
                    self.n_sample,
                    self.sample_min_sample_batch,
                    self.sample_min_tree_height,
                    self.use_dfs_sample,
                )
            dim = sample_unique.size(0)
            rank_counts = torch.tensor([dim], device=self.device, dtype=torch.int64)
            all_counts = all_gather_tensor(rank_counts, self.device, self.world_size)
            if self.sample_min_tree_height is not None:
                # the unique-sample of different rank is different
                # so choose the sum
                counts = torch.cat(all_counts).sum().item()
            else:
                # The unique-sample parts of different rank are the same
                # So choose the average, and this is unreasonable
                # duplicates should be removed
                counts = torch.cat(all_counts).double().mean().item()

            synchronize()
            if int(counts) >= self.max_unique_sample:
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
        if self.rank == 0:
            logger.info(f"{self.method_sample} Sampling {delta:.3E} s", master=True)

        # Sample-comm, gather->merge->scatter
        return self.gather_scatter_sample(sample_unique, sample_counts, wf_value)

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
            placeholders: Tensor (remove counts-rank)
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

        # FIXME:zbwu-24-04-17 remove counts-ranks
        # Scatter unique, counts
        unique_rank = scatter_tensor(merge_unique, self.device, torch.int64, self.world_size)
        # counts_rank = scatter_tensor(merge_counts, self.device, torch.int64, self.world_size)
        prob_rank = scatter_tensor(merge_prob, self.device, torch.double, self.world_size)

        t3 = time.time_ns()
        if self.use_LUT:
            # XXX: unique_rank split merge_unique when broadcast to all-rank,
            # this maybe efficiency than scatter->broadcast
            merge_unique = broadcast_tensor(merge_unique, self.device, torch.int64, master_rank=0)
            # breakpoint()
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

        placeholders = torch.ones([], device=self.device, dtype=torch.int64)
        return (unique_rank, placeholders, prob_rank * self.world_size, WF_LUT)

    def _init_restricted(self, given_state: Tensor) -> None:

        # avoid prob/wf is zeros.
        if self.det_lut is not None:
            x = tensor_to_onv(given_state.to(torch.uint8), self.sorb)
            array_idx = self.det_lut.lookup(x, is_onv=True)[0]
            state = given_state[~array_idx.gt(-1)]
            self.given_state = state
            if self.rank == 0:
                s = "Remove partial state avoid prob/wf is zeros, "
                s += f"Given-state: {given_state.size(0)} -> {state.size(0)}"
                logger.info(s, master=True)
            del x
        else:
            self.given_state = given_state

        # split-rank
        dim = self.given_state.size(0)
        idx_lst = [0] + split_length_idx(dim, self.world_size)
        # logger.info(idx_lst)
        begin_rank = idx_lst[self.rank]
        end_rank = idx_lst[self.rank + 1]
        unique_rank = self.given_state[begin_rank:end_rank]

        onv_all_rank = tensor_to_onv(self.given_state.to(torch.uint8), self.sorb)
        onv_rank = onv_all_rank[begin_rank:end_rank]

        self.restricted_info = (unique_rank, onv_rank, onv_all_rank)

    def restricted_sample(self) -> tuple[Tensor, Tensor, Tensor, WavefunctionLUT]:
        """
        Given-state replace AR/MCMC sampling, only is testing.

        Returns
        -------
            unique_rank: Tensor
            counts_rank: placeholders
            prob_rank: Tensor, prob_rank = prob * world_size
            WF_LUT: wavefunction LookUP-Table about all-sample-unique and wf-value
        """
        unique_rank = self.restricted_info[0]

        # split-batch in single-rank
        dim = unique_rank.size(0)
        idx_lst = [0] + split_batch_idx(dim, min_batch=self.sample_min_sample_batch)
        wf_value = torch.empty(dim, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            for i in range(len(idx_lst) - 1):
                begin = idx_lst[i]
                end = idx_lst[i + 1]
                wf_value[begin:end] = self.nqs(unique_rank[begin:end])

        wf_norm = wf_value.norm() ** 2 * self.world_size
        all_reduce_tensor(wf_norm, world_size=self.world_size)
        prob_rank = wf_value.abs().pow(2) / wf_norm

        if self.use_LUT:
            wf_value_all = all_gather_tensor(wf_value, self.device, self.world_size)
            wf_value_all = torch.cat(wf_value_all)
            WF_LUT = WavefunctionLUT(
                # tensor_to_onv(self.given_state.to(torch.uint8), self.sorb),
                self.restricted_info[2],
                wf_value_all,
                self.sorb,
                self.device,
            )
        else:
            WF_LUT: WavefunctionLUT = None

        # convert to onv
        # unique_rank = tensor_to_onv(unique_rank.to(torch.uint8), self.sorb)
        unique_rank = self.restricted_info[1]
        placeholders = torch.empty(0, device=self.device, dtype=self.dtype)
        return (unique_rank, placeholders, prob_rank * self.world_size, WF_LUT)

    # TODO: how to calculate batch_size;
    # calculate the max nbatch for given Max Memory
    def calculate_energy(
        self,
        sample: Tensor,
        state_prob: Tensor = None,
        WF_LUT: WavefunctionLUT = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""
        Returns:
        -------
            eloc(Tensor): local energy(Single-Rank)
            sloc(Tensor): local spin-raising (Single-Rank)
            placeholders(Tensor): state prob if exact optimization, else zeros tensor
        """
        # this is applied when pre-train
        WF_LUT = self.WF_LUT if WF_LUT is None else WF_LUT
        if self.WF_LUT is not None:
            n_sample = self.WF_LUT.bra_key.size(0)
        else:
            n_sample = sample.size(0)
        nbatch = get_nbatch(
            self.sorb,
            n_sample,
            self.n_SinglesDoubles,
            self.max_memory,
            self.alpha,
            device=self.device,
            use_sample=self.use_sample_space,
            dtype=self.dtype,
        )
        if not self.only_AD:
            use_spin_raising = self.use_spin_raising
            if self.h1e_spin is None and self.h2e_spin is None:
                use_spin_raising = False
            eloc, sloc, placeholders = total_energy(
                sample,
                nbatch,
                h1e=self.h1e,
                h2e=self.h2e,
                ansatz=self.nqs,
                sorb=self.sorb,
                nele=self.nele,
                noa=self.noa,
                nob=self.nob,
                state_prob=state_prob,
                use_spin_raising=use_spin_raising,
                h1e_spin=self.h1e_spin,
                h2e_spin=self.h2e_spin,
                exact=self.debug_exact,
                WF_LUT=WF_LUT,
                use_unique=self.use_unique,
                dtype=self.dtype,
                reduce_psi=self.reduce_psi,
                eps=self.eps,
                use_sample_space=self.use_sample_space,
                alpha=self.alpha,
            )
        else:
            # e_total = -2.33233
            # spin_mean = 0.000
            eloc = torch.zeros(sample.size(0), device=self.device, dtype=self.dtype)
            sloc = torch.zeros_like(eloc)
            placeholders = torch.ones(
                sample.size(0), device=self.device, dtype=torch.double
            ) / sample.size(0)
        return eloc, sloc, placeholders

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
            + f"    use-same-tree: {self.use_same_tree}\n"
            + f"    min-tree-height: {self.sample_min_tree_height}\n"
            + f"    min-nbatch: {self.sample_min_sample_batch}\n"
            + f"    use-dfs-sample: {self.use_dfs_sample}\n"
            + f"    Random seed: {self.seed}\n"
            + f"    alpha: {self.alpha}\n"
            + f"    max_memory: {self.max_memory}\n"
            + ")"
        )

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
