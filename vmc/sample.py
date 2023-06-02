import time
import os
import random
import torch
import tempfile
import numpy as np
import pandas as pd

from functools import partial
from typing import Callable, Tuple, List
from torch import Tensor, nn
from pandas import DataFrame
from scipy import special

from memory_profiler import profile
from line_profiler import LineProfiler

from vmc.energy import total_energy
from libs.C_extension import onv_to_tensor, spin_flip_rand, MCMC_sample
from libs.C_extension import tensor_to_onv
from utils import state_to_string, ElectronInfo, check_para, get_nbatch

print = partial(print, flush=True)

# @profile(precision=4, stream=open('MCMC_memory_profiler.log','w+'))


class Sampler():
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
        nqs: nn.Module,
        ele_info: ElectronInfo,
        n_sample: int = 100,
        therm_step: int = 2000,
        verbose: bool = False,
        debug_exact: bool = False,
        seed: int = 100,
        record_sample: bool = True,
        max_memory: float = 4,
        alpha: float = 0.25,
        dtype=torch.double,
        method_sample="MCMC",
    ) -> None:
        if n_sample < 50:
            raise ValueError(f"The number of sample{n_sample} should great 50")

        self.ele_info = ele_info
        self.read_electron_info(self.ele_info)
        self.nqs = nqs
        self.debug_exact = debug_exact
        self.seed = seed
        self.verbose = verbose
        self.therm_step = therm_step
        self.n_sample = n_sample

        if method_sample not in self.METHOD_SAMPLE:
            raise TypeError(f"Sample method is invalid: {method_sample}, and expected {self.METHOD_SAMPLE}")
        self.method_sample = method_sample

        # device and cuda
        self.is_cuda = True if self.h1e.is_cuda else False
        self.device = self.h1e.device
        self.dtype = dtype

        # save sampler
        self.record_sample = record_sample
        if self.record_sample:
            self.str_full = state_to_string(self.full_space, self.sorb)
            self.frame_sample = pd.DataFrame({"full_space": self.str_full})
            n1 = special.comb(self.noa + self.nva, self.noa, exact=True)
            n2 = special.comb(self.nob + self.nvb, self.nvb, exact=True)
            if self.full_space.size(0) != (n1 * n2):
                raise ValueError(f"The dim of full space is {self.full_space.size(0)} != {n1 * n2}")
        self.time_sample = 0

        # memory control and nbatch
        self.max_memory = max_memory
        self.alpha = alpha

    def read_electron_info(self, ele_info: ElectronInfo):
        print(f"Read electronic structure information From {ele_info.__name__}")
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
        self.full_space = ele_info.ci_space

    # @profile(precision=4, stream=open('MCMC_memory_profiler.log','w+'))
    def run(self, initial_state: Tensor, n_sweep: int = None) -> Tuple[Tensor, Tensor, Tensor, float, dict]:
        t0 = time.time_ns()
        check_para(initial_state)
        if self.debug_exact:
            dim = len(self.full_space)
            e_total, eloc, stats_dict = self.calculate_energy(self.full_space)

            # placeholders only
            sample_prob = torch.empty(dim, dtype=torch.float64, device=self.device)
            return self.full_space.detach(), sample_prob, eloc, e_total, stats_dict

        sample_unique, sample_counts, sample_prob = self.sampling(initial_state, n_sweep)
        delta = time.time_ns() - t0
        print(f"Completed {self.method_sample} Sampling {delta/1.0E09:.3E} s")
        if self.method_sample == "MCMC":
            print(f"acceptance ratio = {self.n_accept/self.n_sample:.3E}")

        e_total, eloc, stats_dict = self.calculate_energy(sample_unique,
                                                          state_prob=sample_prob,
                                                          state_counts=sample_counts)

        # print local energy statistics information
        self._statistics(stats_dict, sample_counts)

        if self.record_sample:
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
        print(f"Completed Sampling and calculating eloc {delta/1.0E09:.3E} s")

        if self.is_cuda:
            torch.cuda.empty_cache()

        return sample_unique.detach(), sample_prob, eloc, e_total, stats_dict

    def sampling(self, initial_state: Tensor, n_sweep: int = None) -> Tuple[Tensor, Tensor, Tensor]:
        if self.method_sample == "MCMC":
            sample_unique, sample_counts, sample_prob = self.MCMC(initial_state, n_sweep)
        elif self.method_sample == "AR":
            sample_unique, sample_counts, sample_prob = self.auto_regressive()

        return sample_unique, sample_counts, sample_prob

    def MCMC(self, initial_state: Tensor, n_sweep: int = None) -> Tuple[Tensor, Tensor, Tensor]:
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
        if True:
            example_inputs = onv_to_tensor(self.current_state, self.sorb)  # -1:unoccupied, 1: occupied
            serialized_model = torch.jit.trace(self.nqs, example_inputs)
            model_file = tempfile.mkstemp()[1]
            serialized_model.save(model_file)
            # print(f"Serialized model time: {(time.time_ns() - t0)/1.E06:.3f} ms")
            with torch.no_grad():
                self.n_accept = MCMC_sample(model_file, self.current_state, self.state_sample,
                                            self.psi_sample, self.sorb, self.nele, self.noa, self.nob,
                                            self.seed, self.n_sweep, self.therm_step)
            os.remove(model_file)
            # print(f"CPP model time: {(time.time_ns() - t0)/1.E06:.3f} ms")
        else:
            with torch.no_grad():
                psi_current = self.nqs(uint8_to_bit(self.current_state, self.sorb))
                prob_current = psi_current.norm()**2
            for i in range(self.n_sweep):
                psi, self.next_state = spin_flip_rand(self.current_state, self.sorb, self.nele, self.noa,
                                                      self.nob, self.seed)
                s1 = state_to_string(self.current_state, self.sorb)
                s2 = state_to_string(self.next_state, self.sorb)
                # print(s1[0], s2[0], state_to_string(psi)[0])
                with torch.no_grad():
                    psi_next = self.nqs(psi)
                    prob_next = psi_next.norm()**2
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
        # torch.unique could not return unique indices in old tensor
        sample_unique, sample_counts = torch.unique(self.state_sample, dim=0, return_counts=True)
        sample_prob = sample_counts / sample_counts.sum()

        # Numpy version:
        # sample_unique, psi_idx, sample_counts = np.unique(
        #   self.state_sample.numpy(), axis=0, return_counts=True, return_index=True)
        # psi_unique = self.psi_sample[torch.from_numpy(psi_idx)]
        # sample_unique = torch.from_numpy(sample_unique)
        # sample_counts = torch.from_numpy(sample_counts)

        return sample_unique, sample_counts, sample_prob

    def auto_regressive(self) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Auto regressive sampling
        """
        sample = self.nqs.ar_sampling(self.n_sample)  # (n_sample, sorb) 0/1

        # remove duplicate state
        sample_unique, sample_counts = torch.unique(sample, dim=0, return_counts=True)
        sample_prob = sample_counts / sample_counts.sum()

        # convert to onv
        sample_unique = tensor_to_onv(sample_unique.to(torch.uint8), self.sorb)

        del sample
        return sample_unique, sample_counts, sample_prob

    # TODO: how to calculate batch_size;
    # calculate the max nbatch for given Max Memory
    def calculate_energy(self, sample: Tensor, state_prob: Tensor = None, state_counts: Tensor = None):
        nbatch = get_nbatch(self.sorb, len(sample), self.n_SinglesDoubles, self.max_memory, self.alpha)
        e_total, eloc, stats_dict = total_energy(sample,
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
                                                 verbose=self.verbose,
                                                 exact=self.debug_exact,
                                                 dtype=self.dtype)
        return e_total, eloc, stats_dict

    def __repr__(self) -> str:
        return (f"{type(self).__name__}:" + " (\n"
                f"    the number of sample: {self.n_sample}\n" + 
                f"    therm step: {self.therm_step}\n" +
                f"    exact sampling: {self.debug_exact}\n"
                f"    the given full space shape: {self.full_space.shape}\n" +
                f"    Record the sample: {self.record_sample}\n" +
                f"    Singles + Doubles: {self.n_SinglesDoubles}\n" + 
                f"    Random seed: {self.seed}\n" + ")")

    def _statistics(self, data: dict, sample_counts: Tensor):
        s = f"E_total = {data['mean'].real:.10f} ± {data['SE'].real:.3E} [σ² = {data['var'].real:.3E}] "
        s += f"sampling: {len(sample_counts)}/{sample_counts.sum().item()}"
        print(s)
