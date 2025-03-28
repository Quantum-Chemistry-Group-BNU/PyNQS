from __future__ import annotations

import time
import os
import random
import tempfile
import warnings
import torch
import torch.distributed as dist

from typing import Callable, Tuple, List, Union, Optional
from typing_extensions import TypedDict, NotRequired
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
from scipy import special

from vmc.energy import total_energy
from vmc.stats import operator_statistics
from libs.C_extension import (
    onv_to_tensor,
    spin_flip_rand,
    MCMC_sample,
    tensor_to_onv,
    merge_rank_sample,
)
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
    split_length_idx,
    split_batch_idx,
    setup_seed,
    diff_rank_seed,
    get_nbatch,
    check_para,
    ansatz_batch,
    spin_flip_onv,
    spin_flip_sign,
    torch_sort_onv,
    torch_unique_index,
    WavefunctionLUT,
    ElectronInfo,
    SpinProjection,
)
from utils.det_helper import DetLUT
from utils.pyscf_helper.operator import spin_raising
from utils.enums import ElocMethod
from utils.tensor_typing import Float, Int, UInt8
from vmc.ansatz import MultiPsi


class ElocParams(TypedDict):
    """
    Eloc-params, see: docs/tutorials/sample/eloc-param
    """

    method: ElocMethod
    use_unique: bool
    use_LUT: bool
    eps: NotRequired[float]
    eps_sample: NotRequired[int]
    alpha: NotRequired[float]
    max_memory: NotRequired[float]
    batch: int
    fp_batch: int


class Sampler:
    """
    Generates samples of configurations from a neural quantum state(NQS)
    using Markov chain Monte Carlo(MCMC) or Auto regressive(AR) algorithm
    """

    METHOD_SAMPLE = ("MCMC", "AR", "RESTRICTED")
    n_accept: int

    def __init__(
        self,
        nqs: DDP,
        ele_info: ElectronInfo,
        eloc_param: ElocParams,
        n_sample: int = 100,
        start_iter: int = 100,
        start_n_sample: Optional[int] = None,
        therm_step: int = 2000,
        debug_exact: bool = False,
        seed: int = 100,
        dtype=torch.double,
        method_sample="AR",
        use_same_tree: bool = False,
        max_n_sample: Optional[int] = None,
        max_unique_sample: Optional[int] = None,
        only_AD: bool = False,
        only_sample: bool = False,
        min_batch: int = 10000,
        min_tree_height: Optional[int] = None,
        det_lut: Optional[DetLUT] = None,
        use_dfs_sample: bool = False,
        use_spin_raising: bool = False,
        spin_raising_coeff: float = 1.0,
        given_state: Optional[Tensor] = None,
        use_spin_flip=False,
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
            # if sampling, every rank have the different random seed
            self.seed = diff_rank_seed(seed, rank=self.rank)
        else:
            # exact optimization does not require sampling
            # the different rank sampling using the the same QuadTree or BinaryTree
            self.seed = seed
        # logger.info(f"{(self.seed, self.rank)})
        logger.info(f"sample-seed: {self.seed}")

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
        # self.record_sample = record_sample
        # if self.record_sample:
        #     self.str_full = state_to_string(self.ci_space, self.sorb)
        #     self.frame_sample = pd.DataFrame({"full_space": self.str_full})
        self.time_sample = 0

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

        # control eloc
        self.eloc_param = eloc_param
        # memory control and nbatch
        # Use 'torch.unique' to speed up local-energy calculations
        self.use_unique = eloc_param["use_unique"]
        eloc_method = eloc_param["method"]

        # ignore x' when <x|H|x'>/psi(x) < eps
        # only apply to when psi(x)^2 is normalization in FCI-space
        self.reduce_psi = False
        self.eps: float = 0.0
        self.eps_sample: int = 0

        # only use x' in n_unique sample not SD, dose not support exact-opt
        # psi(x') can be looked from WaveFunction look-up table
        self.use_sample_space = False

        self.alpha: float = 0.0
        self.max_memory: float = 0.0
        if eloc_method == ElocMethod.SAMPLE_SPACE:
            if self.debug_exact:
                raise TypeError(f"Exact support in FCI-space")
            self.use_sample_space = True
        elif eloc_method == ElocMethod.REDUCE:
            self.reduce_psi = True
            self.eps = eloc_param["eps"]
            if "eps-sample" in eloc_param:
                warnings.warn("'eps-sample' is deprecated, use 'esp_sample'", UserWarning)
                self.eps_sample = eloc_param["eps-sample"]
            else:
                self.eps_sample = eloc_param["eps_sample"]
            assert self.eps >= 0 and self.eps_sample >= 0
        elif eloc_method == ElocMethod.SIMPLE:
            if self.rank == 0:
                logger.info(f"Exact calculate local energy", master=True)
        else:
            raise NotImplementedError

        if eloc_method == ElocMethod.SIMPLE or eloc_method == ElocMethod.REDUCE:
            if self.rank == 0:
                logger.info(f"Using batch/fd_batch control eloc/model batch", master=True)
        elif eloc_method == ElocMethod.SAMPLE_SPACE:
            if self.rank == 0:
                logger.info(f"Use 'max_memory' and 'alpha' control eloc batch", master=True)
            self.max_memory = eloc_param["max_memory"]
            self.alpha = eloc_param["alpha"]

        # only sampling not calculations local-energy, applies to test AD memory
        self.only_AD = only_AD

        # Use WaveFunction LooKup-Table to speed up local-energy calculations
        self.use_LUT: bool = eloc_param.get("use_LUT", True)
        self.WF_LUT: Optional[WavefunctionLUT] = None
        # sort fci space
        self.sort_fci_space: bool = False

        # only sampling not backward
        self.only_sample = only_sample

        # nbatch-rank AR-sampling, only implemented in Transformer/MPS-RNN/Graph-MPS-RNN
        if hasattr(self.nqs.module, "use_multi_psi"):
            ansatz = self.nqs.module.sample
        else:
            ansatz = self.nqs.module
        flag1 = hasattr(ansatz, "min_batch")
        flag2 = hasattr(ansatz, "min_tree_height")
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
        self.det_lut: Optional[DetLUT] = None
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

        self.use_multi_psi = False
        self.extra_norm: Tensor = None
        self.extra_psi_pow: Tensor = None
        if isinstance(self.nqs.module, MultiPsi):
            self.use_multi_psi = True

        self.use_spin_flip = use_spin_flip

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
        # using WF_LUT
        self.WF_LUT, sample_prob = self.construct_FCI_lut()
        if self.remove_det:
            # avoid wf is 0.00, if not found, set to -1
            array_idx = self.det_lut.lookup(self.ci_space, is_onv=True)[0]
            idx = ~array_idx.gt(-1)
            ci_space = self.ci_space[idx]
            sample_prob = sample_prob[idx]
        else:
            ci_space = self.ci_space
        ci_space_rank = scatter_tensor(ci_space, self.device, torch.uint8, self.world_size)
        eloc, sloc, _ = self.calculate_energy(
            ci_space_rank,
            state_prob=sample_prob,
            WF_LUT=self.WF_LUT,
        )
        synchronize()
        # All-Reduce mean local energy
        # Testing
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
        self.epoch = epoch
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
        # change random seed in every iteration
        seed = self.seed + self.epoch
        setup_seed(seed)
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
        1. Gather sample-unique/counts from every rank, compress uint64 to uint8
        2. Merge all unique/counts in master-rank(rank0)
        3. Scatter unique/counts/prob to every rank

        Meanwhile, All-Gather sample-unique and wf-value in order to make wf-lookup-table

        Notice Scatter prob = prob * world-size

        Returns
        -------
            unique_rank: Tensor (uint8, onv)
            placeholders: Tensor (remove counts-rank)
            prob_rank: Tensor, prob_rank = prob * world_size
            WF_LUT: wavefunction LookUP-Table about all-sample-unique and wf-value
        """
        t0 = time.time_ns()
        # Gather unique, counts, wf_value
        unique = tensor_to_onv(unique.byte(), self.sorb)  # compress uint64 -> uint8
        unique_all = gather_tensor(unique, self.device, self.world_size, master_rank=0)
        count_all = gather_tensor(counts, self.device, self.world_size, master_rank=0)
        if self.use_LUT:
            wf_value_all = gather_tensor(wf_value, self.device, self.world_size, master_rank=0)
        else:
            wf_value_all = None
        synchronize()

        # check unique and counts
        if self.rank == 0:
            unique_length = [i.size(0) for i in unique_all]
            count_length = [i.sum().item() for i in count_all]
            s = f"Single-rank: unique: {sum(unique_length)}, counts: {sum(count_length)}"
            logger.info(s, master=True)

        t1 = time.time_ns()
        if self.rank == 0:
            split_idx = torch.tensor([0] + [i.shape[0] for i in unique_all])
            unique_all = torch.cat(unique_all)
            count_all = torch.cat(count_all)
            if not self.use_same_tree:
                # every-rank sample part is the same, so use 'torch.unique'
                split_idx = split_idx.long().to(self.device).cumsum_(dim=0)
                merge_unique, merge_inv, merge_idx = torch_unique_index(unique_all)[:3]
                if self.use_LUT:
                    wf_value_all = torch.cat(wf_value_all)
                    wf_value_unique = wf_value_all[merge_idx]
                # merge_unique, merge_
                # nbatch = unique_all.shape[0]
                # merge_counts = torch.zeros(merge_unique.shape[0], dtype=torch.int64, device=self.device)

                # merge prob
                length = merge_unique.shape[0]
                merge_counts = merge_rank_sample(merge_inv, count_all, split_idx, length)
                # for i in range(nbatch):
                #     merge_counts[merge_idx[i]] += count_all[i]
                merge_prob = merge_counts / merge_counts.sum()
                # _, counts_test = torch.unique(unique_all, dim=0, return_counts=True)
                # assert(torch.allclose(counts_test, merge_counts))
            else:
                # every-rank sample is unique
                merge_counts: Tensor = None
                merge_prob = count_all / count_all.sum()
                merge_unique = unique_all
                if self.use_LUT:
                    wf_value_unique = torch.cat(wf_value_all)
        else:
            merge_counts: Tensor = None
            merge_unique: Tensor = None
            merge_prob: Tensor = None
            wf_value_unique: Tensor = None

        t2 = time.time_ns()

        # FIXME:zbwu-24-04-17 remove counts-ranks
        # Scatter unique, counts
        unique_rank = scatter_tensor(merge_unique, self.device, torch.uint8, self.world_size)
        # counts_rank = scatter_tensor(merge_counts, self.device, torch.int64, self.world_size)
        prob_rank = scatter_tensor(merge_prob, self.device, torch.double, self.world_size)

        t3 = time.time_ns()
        if self.use_LUT:
            # XXX: unique_rank split merge_unique when broadcast to all-rank,
            # this maybe efficiency than scatter->broadcast
            merge_unique = broadcast_tensor(merge_unique, self.device, torch.uint8, master_rank=0)
            wf_value_unique = broadcast_tensor(
                wf_value_unique, self.device, self.dtype, master_rank=0
            )
        synchronize()
        t4 = time.time_ns()

        # Testing prob
        use_subspace = False
        if use_subspace:
            if not self.use_LUT:
                raise NotImplementedError
            else:
                prob1 = wf_value_unique.abs() ** 2 / wf_value_unique.norm() ** 2
                dim = merge_unique.size(0)
                idx_lst = [0] + split_length_idx(dim, self.world_size)
                begin_rank = idx_lst[self.rank]
                end_rank = idx_lst[self.rank + 1]
                prob_rank = prob1[begin_rank:end_rank]

        if self.rank == 0:
            delta1 = (t1 - t0) / 1.0e09
            delta2 = (t3 - t2) / 1.0e09
            delta3 = (t2 - t1) / 1.0e09
            delta4 = (t4 - t3) / 1.0e09
            s = f"Sample-Comm, Gather: {delta1:.3E} s, Scatter: {delta2:.3E} s, merge: {delta3:.3E} s\n"
            s += f"All-Rank unique sample: {merge_unique.size(0)}, Broadcast LUT: {delta4:.3E} s"
            logger.info(s, master=True)

        if self.use_LUT:
            dim = merge_unique.size(0)
            bra_key = merge_unique
            # bra_key = tensor_to_onv(merge_unique, self.sorb)
            WF_LUT = WavefunctionLUT(
                bra_key,
                wf_value_unique,
                self.sorb,
                self.device,
            )
            idx_lst = [0] + split_length_idx(dim, self.world_size)
            begin_rank = idx_lst[self.rank]
            end_rank = idx_lst[self.rank + 1]
            # unique_rank1 = tensor_to_onv(unique_rank.to(torch.uint8), self.sorb)
            # unique_rank = bra_key[begin_rank:end_rank]
            # assert torch.allclose(bra_key[begin_rank:end_rank], unique_rank)
            # assert torch.allclose(unique_rank1, unique_rank)
        else:
            WF_LUT: WavefunctionLUT = None
            # convert to onv
            # unique_rank = tensor_to_onv(unique_rank, self.sorb)

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

        if not "batch" in self.eloc_param.keys():
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
            fp_batch = -1
        else:
            nbatch = self.eloc_param["batch"]
            fp_batch = self.eloc_param["fp_batch"]

        if not self.only_AD:
            use_spin_raising = self.use_spin_raising
            if self.h1e_spin is None and self.h2e_spin is None:
                use_spin_raising = False

            if self.use_spin_flip:
                self.eta = SpinProjection.eta

            if self.use_multi_psi or self.use_spin_flip:
                if self.use_multi_psi:
                    func = self.gather_extra_psi
                else:
                    func = self.gather_flip
                extra_norm, extra_psi_pow = func(sample, state_prob)
                self.extra_norm = extra_norm
                self.extra_psi_pow = extra_psi_pow

            eloc, sloc, placeholders = total_energy(
                sample,
                nbatch,
                fp_batch,
                h1e=self.h1e,
                h2e=self.h2e,
                ansatz=self.nqs,
                sorb=self.sorb,
                nele=self.nele,
                noa=self.noa,
                nob=self.nob,
                use_spin_raising=use_spin_raising,
                h1e_spin=self.h1e_spin,
                h2e_spin=self.h2e_spin,
                WF_LUT=WF_LUT,
                use_unique=self.use_unique,
                dtype=self.dtype,
                reduce_psi=self.reduce_psi,
                eps=self.eps,
                eps_sample=self.eps_sample,
                use_sample_space=self.use_sample_space,
                alpha=self.alpha,
                use_multi_psi=self.use_multi_psi,
                use_spin_flip=self.use_spin_flip,
                extra_norm=self.extra_norm,
            )
        else:
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
            + f"    Therm step: {self.therm_step}\n"
            + f"    Exact sampling: {self.debug_exact}\n"
            + f"    Given CI: {self.ci_space.size(0):.3E}\n"
            + f"    Max unique sample: {self.max_unique_sample:3E}\n"
            + f"    Max sample: {self.max_n_sample:3E}\n"
            + f"    use-same-tree: {self.use_same_tree}\n"
            + f"    min-tree-height: {self.sample_min_tree_height}\n"
            + f"    min-nbatch: {self.sample_min_sample_batch}\n"
            + f"    use-dfs-sample: {self.use_dfs_sample}\n"
            + f"    Random seed: {self.seed}\n"
            + ")\n"
            + f"{self.__print_eloc_param()}"
        )

    def __print_eloc_param(self) -> str:
        if not "batch" in self.eloc_param.keys():
            auto_batch = True
            nbatch = fp_batch = -1
        else:
            nbatch = self.eloc_param["batch"]
            fp_batch = self.eloc_param["fp_batch"]
            auto_batch = False
        return (
            f"Eloc-param: (\n"
            + f"    Eloc-method: {self.eloc_param['method']}:\n"
            + f"    Using LUT: {self.use_LUT}\n"
            + f"    Singles + Doubles: {self.n_SinglesDoubles}\n"
            + f"    FCI space: {self.fci_size:.3E}\n"
            + f"    auto-split-batch: {auto_batch}\n"
            + f"    eps: {self.eps:.3E}, eps-sample: {self.eps_sample}\n"
            + f"    alpha: {self.alpha}, max_memory: {self.max_memory}\n"
            + f"    eloc-batch: {nbatch}, Forward-batch: {fp_batch}\n"
            + ")"
        )

    def change_n_sample(self, epoch: int) -> None:
        """
        change the number of sample in the beginning of iteration.
        """
        self.epoch = epoch
        if epoch <= self.start_iter:
            self.n_sample = self.start_n_sample
            self.max_n_sample = self.n_sample
            self.min_n_sample = self.n_sample
        elif epoch == self.start_iter + 1:
            self.n_sample = self.last_n_sample
            self.max_n_sample = self.last_max_n_sample
            self.min_n_sample = self.n_sample

    def construct_FCI_lut(self) -> Tuple[WavefunctionLUT, Tensor]:
        """
        FCI-space LUT
        """
        # using simple-eloc
        self.use_sample_space = False
        # self.reduce_psi = False
        fp_batch: int = self.eloc_param["fp_batch"]
        assert self.ci_space.size(0) == self.fci_size

        # avoid sort
        if not self.sort_fci_space:
            idx = torch_sort_onv(self.ci_space)
            self.ci_space = self.ci_space[idx]
            self.sort_fci_space = True

        # split rank
        dim = self.ci_space.size(0)
        idx_rank_lst = [0] + split_length_idx(dim, length=self.world_size)
        begin = idx_rank_lst[self.rank]
        end = idx_rank_lst[self.rank + 1]
        ci_space_rank = self.ci_space[begin:end]
        if fp_batch == -1 or fp_batch > ci_space_rank.size(0):
            fp_batch = ci_space_rank.size(0)
        if self.rank == 0:
            s = f"Begin construct FCI-space LUT, FCI-space: {self.fci_size}\n"
            s += f"All-dim: {ci_space_rank.size(0)}, Split-batch: {fp_batch}"
            logger.info(s, master=True)

        t0 = time.time_ns()
        with torch.no_grad():
            if self.use_multi_psi:
                model = self.nqs.module.sample
            else:
                model = self.nqs
            psi_rank = self.ansatz_batch(ci_space_rank, model, fp_batch)
        t1 = time.time_ns()

        psi_all = all_gather_tensor(psi_rank, self.device, self.world_size)
        synchronize()
        if self.world_size == 1:
            psi_all = psi_all[0]  # avoid memory copy
        else:
            psi_all = torch.cat(psi_all)
        WF_LUT = WavefunctionLUT(
            self.ci_space,
            psi_all,
            self.sorb,
            self.device,
            sort = False,
        )
        # calculate prob
        norm_all = (psi_rank.norm()**2).reshape(-1)
        all_reduce_tensor(norm_all, op=dist.ReduceOp.SUM)
        state_prob = (psi_rank * psi_rank.conj()).real / norm_all
        if self.rank == 0:
            logger.info(f"End construct, cost time: {(t1-t0)/1.0e9:.3E} s", master=True)
        return WF_LUT, state_prob

    @torch.no_grad
    def ansatz_batch(
        self,
        x: Tensor,
        ansatz: Callable[[Tensor], Tensor],
        fp_batch: int = -1,
    ) -> Tensor:
        return ansatz_batch(ansatz, x, fp_batch, self.sorb, self.device, self.dtype)

    def gather_extra_psi(
        self,
        x: UInt8[Tensor, "batch bra_len"],
        prob: Float[Tensor, "batch"],
    ) -> tuple[Float[Tensor, "1"], Float[Tensor, "batch"]]:
        """
        return:
            ||f(n)|| / norm**2, norm()
        """
        fp_batch: int = self.eloc_param["fp_batch"]
        f = self.ansatz_batch(x, self.nqs.module.extra, fp_batch)
        _f = f.conj() * f

        # spin flip symmetry
        if self.use_spin_flip:
            x_flip = spin_flip_onv(x, self.sorb)
            if self.use_sample_space:
                from libs.C_extension import wavefunction_lut
                # Maybe construct f lut
                _, mask = wavefunction_lut(self.WF_LUT.bra_key, x_flip, self.sorb)
                f_flip = torch.zeros_like(f)
                f_flip[mask] = self.ansatz_batch(x_flip[mask], self.nqs.module.extra, fp_batch)
            else:
                f_flip = self.ansatz_batch(x_flip, self.nqs.module.extra, fp_batch)
            eta_n = spin_flip_sign(x, self.sorb)
            psi = self.ansatz_batch(x, self.nqs.module.sample, fp_batch)
            if self.use_sample_space:
                psi_flip = torch.zeros_like(psi)
                idx, _, value = self.WF_LUT.lookup(x_flip)
                psi_flip[idx] = value
            else:
                psi_flip = self.ansatz_batch(x_flip, self.nqs.module.sample, fp_batch)
            f_psi = _f + self.eta * eta_n * f.conj() * f_flip * psi_flip / psi

        # stats
        if self.debug_exact:
            n_sample = float("inf")
        else:
            n_sample = self.n_sample
        f_stats = operator_statistics(_f, prob, n_sample, "f(n)²")
        if self.rank == 0:
            logger.info(str(f_stats), master=True)

        if self.use_spin_flip:
            f_psi_stats = operator_statistics(f_psi, prob, n_sample, "F(n)²")
            extra_norm = f_psi_stats["mean"].sqrt()
            extra_psi_pow = f_psi / extra_norm**2
            if self.rank == 0:
                logger.info(str(f_psi_stats), master=True)
        else:
            extra_norm = f_stats["mean"].sqrt()
            extra_psi_pow = _f / extra_norm**2

        return extra_norm, extra_psi_pow

    def gather_flip(
        self,
        x: UInt8[Tensor, "batch bra_len"],
        prob: Float[Tensor, "batch"],
    ) -> tuple[Float[Tensor, "1"], Float[Tensor, "batch"]]:
        """
        return:
            || 1 + η * psi(n-flip)/psi(n)||
        """

        fp_batch: int = self.eloc_param["fp_batch"]
        psi = self.ansatz_batch(x, self.nqs.module, fp_batch)

        # flip-spin
        eta_n = spin_flip_sign(x, self.sorb)
        x_flip = spin_flip_onv(x, self.sorb)
        if self.use_sample_space:
            psi_flip = torch.zeros_like(psi)
            idx, _, value = self.WF_LUT.lookup(x_flip)
            psi_flip[idx] = value
        else:
            psi_flip = self.ansatz_batch(x_flip, self.nqs.module, fp_batch)

        _psi = 1 + self.eta * eta_n * psi_flip / psi
        # stats
        if self.debug_exact:
            n_sample = float("inf")
        else:
            n_sample = self.n_sample

        stats_norm = operator_statistics(_psi, prob, n_sample, "F(n)²")
        extra_norm = stats_norm["mean"].sqrt()
        extra_psi_pow = _psi / extra_norm**2

        if self.rank == 0:
            logger.info(str(stats_norm), master=True)

        # \sqrt(B), C
        return extra_norm, extra_psi_pow