from __future__ import annotations

from functools import partial
import time
import os
import warnings
import torch
import torch.distributed as dist

from dataclasses import dataclass
from typing import Literal, Tuple, Optional
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
from scipy import special

from pynqs.energy.etot import gather_extra_psi, gather_flip
from pynqs.utils.hamiltonian import ElectronInfo
from pynqs.utils.lut import WavefunctionLUT
from pynqs.energy import calculate_energy
from pynqs.distributed import (
    all_gather_tensor,
    get_rank,
    get_world_size,
)
from pynqs.utils.public_function import (
    setup_seed,
    diff_rank_seed,
)
from pynqs.utils.det_helper import DetLUT
from pynqs.utils.pyscf_helper.operator import spin_raising
from pynqs.utils.enums import ElocMethod, SampleMethod
from pynqs.config import dtype_config
from pynqs.ansatz import MultiPsi, AlphaPsi
from pynqs.sample.base import BaseSampler, PoolParams, RESTRICTEDParams, MCMCParams, ARParams, ExactParams
from pynqs.sample.exact import ExactSampler
from pynqs.sample.metropolis import MCMCSampler, PTMCSampler
from pynqs.sample.hybrid import HybridSampler
from pynqs.sample.autoregressive import ARSampler
from pynqs.sample.restricted import RestrictedSampler
from pynqs.sample.pool import PoolSampler, PoolSampler_v1, select_SD_space
from pynqs.energy import calculate_energy, ElocParams


@dataclass
class SampleParams:
    method_sample: SampleMethod
    seed: int
    eloc_params: ElocParams
    params: ARParams | MCMCParams | RESTRICTEDParams | ExactParams
    debug_exact: bool = None  # This is has been deprecated
    use_spin_flip: bool = False
    """"use spin flip, see doc"""
    det_lut: Optional[DetLUT] = None
    use_pool_sampling: bool = False
    pool_params: PoolParams = None
    only_AD: bool = False
    """test backward"""


sampler_string = Literal["MCMC", "PTMC", "HybridMC", "AR", "RESTRICTED"]
SAMPLER_MAPPING: dict[SampleMethod | sampler_string, BaseSampler] = {
    SampleMethod.EXACT: ExactSampler,
    SampleMethod.AR: ARSampler,
    SampleMethod.MCMC: MCMCSampler,
    SampleMethod.RESTRICTED: RestrictedSampler,
    SampleMethod.PTMC: PTMCSampler,
    SampleMethod.HYBRIDMC: MCMCSampler,
    "Exact": ExactSampler,
    "AR": ARSampler,
    "MCMC": MCMCSampler,
    "RESTRICTED": RestrictedSampler,
    "PTMC": PTMCSampler,
    "HybridMC": MCMCSampler,
}


class Sampler:
    """
    Generates samples of configurations from a neural quantum state(NQS)
    using Markov chain Monte Carlo(MCMC) or Auto regressive(AR) algorithm
    """

    def __init__(
        self,
        nqs: DDP,
        ele_info: ElectronInfo,
        sample_params: SampleParams,
        use_spin_raising: bool = False,
        spin_raising_coeff: float = 1.0,
        only_sample: bool = False,
        clip_eloc: bool = False,
        NES_K: int = 1,
        NES_w: Tensor = None,
    ) -> None:
        self.rank = get_rank()
        self.world_size = get_world_size()

        self.ele_info = ele_info
        self.read_electron_info(self.ele_info)
        self.nqs = nqs
        self.NES_w = NES_w
        # self.therm_step = therm_step

        method_sample = sample_params.method_sample
        eloc_params = sample_params.eloc_params
        params = sample_params.params
        seed = sample_params.seed
        only_AD = sample_params.only_AD
        use_spin_flip = sample_params.use_spin_flip
        det_lut = sample_params.det_lut

        if sample_params.debug_exact is not None:
            if self.rank == 0:
                logger.warning(f"SampleParams.debug' will be removed in last version", master=True)

        # if method_sample not in self.METHOD_SAMPLE:
        #     raise TypeError(f"Sample method is invalid: {method_sample}, and expected {self.METHOD_SAMPLE}")
        # self.method_sample = method_sample
        self.method_sample = method_sample

        # device and cuda
        self.is_cuda = True if self.h1e.is_cuda else False
        self.device = self.h1e.device
        self.dtype = dtype_config.default_dtype

        # check fci_dtype
        n1 = special.comb(self.noa + self.nva, self.noa, exact=True)
        n2 = special.comb(self.nob + self.nvb, self.nvb, exact=True)
        self.fci_size: int = n1 * n2

        if self.method_sample == SampleMethod.HYBRIDMC or self.method_sample == "HybridMC":
            logger.info(
                f"WARNING: HybridSampler is now redirected to MCMCSampler, where using MPS as starting is implemented"
            )

        state = SAMPLER_MAPPING[self.method_sample]
        self.exact = False
        if issubclass(state, ExactSampler):
            self.exact = True

        # control eloc
        self.eloc_param = eloc_params
        eloc_method = eloc_params.method
        # only use x' in n_unique sample not SD, dose not support exact-opt
        # psi(x') can be looked from WaveFunction look-up table
        self.use_sample_space = False
        self.reduce_psi = False
        self.use_LUT = self.eloc_param.use_LUT
        if eloc_method == ElocMethod.SAMPLE_SPACE:
            self.use_sample_space = True
        elif eloc_method == ElocMethod.REDUCE:
            self.reduce_psi = True
            assert eloc_params.eps >= 0 and eloc_params.eps_sample >= 0
        elif eloc_method == ElocMethod.SIMPLE:
            if self.rank == 0:
                logger.info(f"Exact calculate local energy", master=True)
        else:
            raise NotImplementedError

        if eloc_method != ElocMethod.SIMPLE and self.exact:
            raise ValueError(f"Exact optimization only support 'eloc_method = ElocMethod.SIMPLE'")

        self.params = params
        self.SamplerState = state(
            self.nqs,
            self.params,
            self.fci_size,
            self.sorb,
            self.nele,
            self.noa,
            self.nob,
            self.use_LUT,
            self.device,
            NES_K=NES_K,
        )
        if self.exact:
            use_same_tree = True
        else:
            use_same_tree = self.SamplerState.use_same_tree

        # setup random seed
        if not self.exact and not use_same_tree:
            # if sampling, every rank have the different random seed
            self.seed = diff_rank_seed(seed, rank=self.rank)
        else:
            # exact optimization does not require sampling
            # the different rank sampling using the the same QuadTree or BinaryTree
            self.seed = seed
            setup_seed(seed)
        logger.info(f"sample-seed: {self.seed}")

        self.start_epoch: int = None
        # GFMC only in rank-0
        self.all_sample_counts: Tensor = None

        # only sampling not calculations local-energy, applies to test AD memory
        self.only_AD = only_AD

        self.WF_LUT: Optional[WavefunctionLUT] = None

        # only sampling not backward
        self.only_sample = only_sample

        # Det-LUT, remove part det in CI-NQS
        # self.remove_det = False
        # self.det_lut: Optional[DetLUT] = None
        # if det_lut is not None:
        #     self.remove_det = True
        #     self.det_lut = det_lut

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

        self.use_multi_psi = False
        self.extra_norm: Tensor = None
        self.extra_psi_pow: Tensor = None
        if isinstance(self.nqs.module, MultiPsi) or isinstance(self.nqs.module, AlphaPsi):
            self.use_multi_psi = True

        self.use_spin_flip = use_spin_flip
        self.use_LUT = self.eloc_param.use_LUT

        self.clip_eloc = clip_eloc
        if self.clip_eloc and self.only_sample:
            warnings.warn(
                f"Local energy clipping is used only in training, not in evaluation.", UserWarning, 2
            )

        self.use_pool_sampling = sample_params.use_pool_sampling
        if self.use_pool_sampling:
            pool_params = sample_params.pool_params
            if pool_params.use_its:
                pool_state = PoolSampler_v1
            else:
                pool_state = PoolSampler

            if self.only_sample:
                raise NotImplementedError(f"'Only-sample=False' does not support 'use_pool_sampling=True'")
            self.SamplerState_pool = pool_state(
                self.nqs,
                pool_params,
                self.fci_size,
                self.sorb,
                self.nele,
                self.noa,
                self.nob,
                use_LUT=True,
                device=self.device,
                NES_K=NES_K,
            )
            self.pool_params = pool_params
        else:
            self.SamplerState_pool: PoolSampler_v1 | PoolSampler = None
            self.pool_params: PoolParams = None

    def read_electron_info(self, ele_info: ElectronInfo):
        if self.rank == 0:
            logger.info(f"Read electronic structure information From {ele_info.__name__}", master=True)
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
        epoch: int,
        n_sweep: int = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """
        run sampling using 'MCMC' or 'AR' algorithm and calculate local energy

        Parameters
        ----------
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

        result = self.run_sampling(epoch, n_sweep)
        delta = time.time_ns() - t0
        if self.rank == 0:
            s = f"Completed Sampling and calculating eloc {delta/1.0E09:.3E} s"
            logger.info(s, master=True)

        if self.is_cuda:
            torch.cuda.empty_cache()
        return result

    def gather_sample(self, sample_unique: Tensor, sample_counts: Tensor):
        if self.method_sample in SAMPLER_MAPPING.keys() and not self.exact:
            counts_all = all_gather_tensor(sample_counts, self.device)
            counts_all = torch.cat(counts_all) if self.world_size > 1 else counts_all[0]
            n_sample = counts_all.sum().item()
            self.all_sample_counts = counts_all
        else:
            n_sample = float("inf")
            self.all_sample_counts = sample_unique.shape[0]
        self.n_sample = n_sample

        return n_sample

    def run_sampling(
        self,
        epoch: int,
        n_sweep: int = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        sample_unique, sample_counts, sample_prob, wf_lut, n_sample = self.sampling(
            epoch=epoch,
            n_sweep=n_sweep,
        )

        self.WF_LUT = wf_lut
        return self.calculate_eloc(
            sample_unique,
            sample_prob,
            n_sample,
            wf_lut,
        )

    @torch.no_grad()
    def sampling(
        self,
        epoch: int,
        n_sweep: int = None,
    ) -> Tuple[Tensor, Tensor, Tensor, WavefunctionLUT | None, int | float]:
        """
        Returns
        -------
            sample-unique: Tensor(Single-Rank)
            sample_counts: sample counts(Single-Rank)
            sample-prob: Tensor(Single-Rank)
            wf_lut: WaveFunction-LUT | None
            n_sample: the all-samples
        """
        if self.start_epoch is None:
            self.start_epoch = epoch
            if self.rank == 0:
                logger.info(f"start-epoch: {self.start_epoch}", master=True)
        self.epoch = epoch

        if not self.use_pool_sampling:
            sample_unique, sample_counts, sample_prob, wf_lut = self.SamplerState.run(epoch, self.seed)
        else:
            if not self.pool_params.use_its:
                raise ValueError(f"use 'use_its = True' in PoolParams")
                mcmc_interval = self.pool_params.mcmc_interval
                pool_interval = self.pool_params.pool_interval
                use_SD = self.pool_params.use_SD
                eps = self.pool_params.eps
                cycle_len = mcmc_interval + pool_interval
                pos_in_cycle = (epoch - self.start_epoch) % cycle_len
                sampling_type = "MCMC" if pos_in_cycle < mcmc_interval else "Pool"

                # -------- Pool start (only once) --------
                if sampling_type == "Pool" and pos_in_cycle == mcmc_interval:
                    self.SamplerState_pool.construct_space(),

                # -------- Run sampler --------
                if sampling_type == "MCMC":
                    sample_unique, sample_counts, sample_prob, wf_lut = self.SamplerState.run(
                        epoch, self.seed
                    )

                    if use_SD:
                        space = select_SD_space(
                            sample_unique,
                            self.h1e,
                            self.h2e,
                            self.sorb,
                            self.nele,
                            self.noa,
                            self.nob,
                            eps=eps,
                        )
                    else:
                        space = sample_unique

                    self.SamplerState_pool.update_pool(space)
                else:
                    sample_unique, sample_counts, sample_prob, wf_lut = self.SamplerState_pool.run(
                        epoch, self.seed
                    )

                # -------- Pool end (only once) --------
                if sampling_type == "MCMC" and (pos_in_cycle == 0 and epoch > 0):
                    self.SamplerState_pool.delete_space()
            else:
                its_interval = self.pool_params.its_interval
                persistent_mcmc = self.pool_params.persistent_mcmc
                reset_mcmc_on_its = self.pool_params.reset_mcmc_on_its

                if reset_mcmc_on_its and not persistent_mcmc:
                    raise ValueError("reset_mcmc_on_its=True requires persistent_mcmc=True")

                # update target-space
                is_its_step = (epoch - self.start_epoch) % its_interval == 0 and epoch > self.start_epoch

                # MCMC propagation
                if persistent_mcmc:
                    #  each walker performs Ne proposed hopping moves per optimization step
                    mcmc_sample, mcmc_counts, mcmc_prob, mcmc_wf_lut = self.SamplerState.run(
                        epoch,
                        self.seed,
                    )
                else:
                    # MCMC when updating target-space(U)
                    if is_its_step:
                        mcmc_sample, mcmc_counts, mcmc_prob, mcmc_wf_lut = self.SamplerState.run(
                            epoch,
                            self.seed,
                        )
                    else:
                        mcmc_sample = mcmc_counts = mcmc_prob = mcmc_wf_lut = None

                # reset mcmc initial state
                if reset_mcmc_on_its and is_its_step:
                    if self.rank == 0:
                        s = f"Notice reset '{self.SamplerState.__class__.__name__}.last = None' "
                        logger.info(s, master=True)
                    if isinstance(self.SamplerState, (HybridSampler, MCMCSampler)):
                        self.SamplerState.last = None
                    else:
                        logger.error(f"SamplerState: {self.SamplerState.__class__.__name__}")
                        raise NotImplementedError("Reset initial only supports MCMC Samplers")

                # ITS update target-space(U)
                if is_its_step:
                    _n_sample = self.gather_sample(mcmc_sample, mcmc_counts)

                    use_sample_space = self.eloc_param.method == ElocMethod.SAMPLE_SPACE
                    if use_sample_space:
                        self.eloc_param.method = ElocMethod.REDUCE
                        if self.rank == 0:
                            logger.info(
                                f"Use REDUCE to calculate eloc in its-interval, "
                                f"eps: {self.eloc_param.eps} eps_sample: {self.eloc_param.eps_sample}",
                                master=True,
                            )

                    _ = self.calculate_eloc(
                        mcmc_sample,
                        mcmc_prob,
                        _n_sample,
                        mcmc_wf_lut,
                        "MCMC-",
                    )

                    if use_sample_space:
                        self.eloc_param.method = ElocMethod.SAMPLE_SPACE

                    self.SamplerState_pool.its_update(
                        mcmc_sample,
                        self.h1e,
                        self.h2e,
                        self.sorb,
                        self.nele,
                        self.noa,
                        self.nob,
                    )

                # gumbel-topk
                if epoch >= self.start_epoch + its_interval:
                    sample_unique, sample_counts, sample_prob, wf_lut = self.SamplerState_pool.run(
                        epoch,
                        self.seed,
                    )
                else:
                    # early-stage fallback
                    if mcmc_sample is None:
                        # no MCMC available: fall back to direct sampler
                        sample_unique, sample_counts, sample_prob, wf_lut = self.SamplerState.run(
                            epoch,
                            self.seed,
                        )
                    else:
                        sample_unique, sample_counts, sample_prob, wf_lut = (
                            mcmc_sample,
                            mcmc_counts,
                            mcmc_prob,
                            mcmc_wf_lut,
                        )

        # Gather all-samples
        n_sample = self.gather_sample(sample_unique, sample_counts)
        return sample_unique, sample_counts, sample_prob, wf_lut, n_sample

    def calculate_eloc(
        self,
        sample: Tensor,
        sample_prob: Tensor,
        n_sample: int,
        wf_lut: WavefunctionLUT,
        operator_prefix: str = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        if self.use_multi_psi or self.use_spin_flip:
            if self.use_multi_psi:
                func = gather_extra_psi
            else:
                func = gather_flip
            extra_norm, extra_psi_pow = func(
                self.nqs.module,
                sample,
                self.sorb,
                sample_prob,
                self.eloc_param.fp_batch,
                self.use_spin_flip,
                self.use_sample_space,
                wf_lut,
                n_sample,
                self.exact,
            )
            self.extra_norm = extra_norm
            self.extra_psi_pow = extra_psi_pow

        eloc, sloc, eloc_mean, sloc_mean = calculate_energy(
            self.nqs.module,
            sample,
            sample_prob,
            n_sample,
            self.h1e,
            self.h2e,
            self.sorb,
            self.nele,
            self.noa,
            self.nob,
            self.eloc_param,
            self.use_spin_raising,
            self.h1e_spin,
            self.h2e_spin,
            wf_lut,
            self.exact,
            self.clip_eloc,
            self.only_sample,
            self.only_AD,
            self.use_spin_flip,
            self.use_multi_psi,
            self.extra_norm,
            operator_prefix,
            self.NES_w,
        )
        return sample.detach(), sample_prob, (eloc, sloc), (eloc_mean, sloc_mean)

    def __repr__(self) -> str:
        s = f"    Sample-params: {self.SamplerState.params}"
        if self.use_pool_sampling:
            s += f"    Pool-params: {str(self.pool_params)}"
        return (
            f"{type(self).__name__}:"
            + " (\n"
            + s
            + f"    Given CI: {self.ci_space.size(0):.3E}\n"
            + f"    Random seed: {self.seed}\n"
            + f"    {self.eloc_param}"
            + f"    Singles + Doubles: {self.n_SinglesDoubles}\n"
            + f"    FCI space: {self.fci_size:.3E}\n"
            + ")\n"
        )
