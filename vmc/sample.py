import time
import random
import torch
import numpy as np
import pandas as pd
from functools import partial
from typing import Callable, Tuple, List
from torch import Tensor
from pandas import DataFrame

from memory_profiler import profile
from line_profiler import LineProfiler

from vmc.PublicFunction import check_para, get_Num_SinglesDoubles, get_nbatch
from vmc.eloc import total_energy
from libs.hij_tensor import uint8_to_bit, spin_flip_rand

print = partial(print, flush=True)

# @profile(precision=4, stream=open('MCMC_memory_profiler.log','w+'))
class MCMCSampler():
    """
    Generates samples of configurations from a neural quantum state(NQS)
    using Markov chain Monte Carlo(MCMC) algorithm
    """
    n_accept: int
    str_full: List[str]
    frame_sample: DataFrame
    def __init__(self, nqs: Callable,
                 h1e: Tensor, h2e: Tensor,
                 sorb: int, nele: int, ecore: float, n_sample: int = 100, 
                 therm_step: int = 2000, verbose: bool = False, 
                 debug_exact: bool = False, full_space: Tensor = None,
                 seed: int = 100, record_sample: bool = True,
                 max_memory: float = 4, alpha: float = 0.25,
                 ) -> None:
        if debug_exact:
            assert full_space is not None, "full space must be given"
        if n_sample < 50:
            raise ValueError(f"The number of sample{n_sample} should great 50")
        self.debug_exact = debug_exact
        self.nqs = nqs
        self.sorb = sorb
        self.nele = nele
        self.no = nele
        self.nv = sorb - nele
        self.nob = nele//2
        self.noa = nele - self.nob 
        self.n_SinglesDoubles = get_Num_SinglesDoubles(self.sorb, self.noa, self.nob)
        self.ecore = ecore
        self.seed = seed
        self.verbose: bool = verbose
        self.h1e: Tensor = h1e
        self.h2e: Tensor = h2e
        self.therm_step = therm_step
        self.n_sample = n_sample
        self.full_space= full_space
        self.is_cuda = True if h1e.is_cuda else False
        self.device = h1e.device

        # save sampler
        self.record_sample = record_sample
        if self.record_sample:
            self.str_full = self.state_str(self.full_space)
            self.frame_sample = pd.DataFrame({"full_space": self.str_full})
        self.time_sample = 0

        # memory control and nbatch
        self.max_memory = max_memory
        self.alpha = alpha
    
    # @profile(precision=4, stream=open('MCMC_memory_profiler.log','w+'))
    def run(self, initial_state: Tensor, n_sweep: int = None) -> Tuple[Tensor, Tensor, float]:
        check_para(initial_state)
        if self.debug_exact:
            dim = len(self.full_space)
            nbatch = get_nbatch(self.sorb, dim, self.n_SinglesDoubles, self.max_memory, self.alpha)
            e_total, eloc = total_energy(self.full_space, nbatch, self.h1e, self.h2e, 
                                        self.nqs, self.ecore, self.sorb, 
                                        self.nele, self.noa, self.nob,
                                        verbose = self.verbose,
                                        exact = self.debug_exact)
            sample_idx = torch.ones(dim, dtype=torch.float64, device=self.device)/dim
            return self.full_space.detach(), sample_idx, eloc, e_total
        else:
            self.state_sample: Tensor = torch.zeros_like(initial_state).repeat(self.n_sample, 1)
            self.current_state: Tensor = initial_state
            self.next_state: Tensor = initial_state
            self.n_accept = 0

        if (n_sweep is None) or (n_sweep <= self.therm_step + self.n_sample):
            n_sweep = self.therm_step + self.n_sample

        # convert to CPU and 'spin_flip_rand' is not implemented in "GPU"
        if self.is_cuda:
            self.state_sample = self.state_sample.to("cpu")
            self.current_state = self.current_state.to("cpu")
            self.next_state = self.next_state.to("cpu")
            self.nqs = self.nqs.to("cpu")
    
        print('Starting MCMC Sampling')
        t0 = time.time_ns()
        prob_current = self.nqs(uint8_to_bit(self.current_state, self.sorb).reshape(1, -1))**2
        spin_time = torch.zeros(n_sweep)
        for i in range(n_sweep):
            t1 = time.time_ns()
            # psi, self.next_state = spin_flip_rand(self.next_state, self.sorb, self.nele, self.noa, self.nob, self.seed)
            psi, self.next_state = spin_flip_rand(self.next_state, self.sorb, self.nele, self.seed)
            spin_time[i] = (time.time_ns() - t1)/1.0E06
            prob_next = self.nqs(psi.reshape(1, -1))**2
            prob_accept = min(1.00, (prob_next/prob_current).item())
            p = random.random()
            # if self.verbose and i >= self.therm_step:
            #     print(f"prob_next: {prob_next.item()}, prob_current: {prob_current.item()}")
            #     print(f"random p {p:.3f}, prob_accept {prob_accept:.3f}")
            #     print(f"current state: {self.current_state}")
            if p <= prob_accept:
                self.current_state = self.next_state.clone()
                prob_current = prob_next.clone()
                if i >= self.therm_step:
                    self.n_accept += 1
                    
            if i >= self.therm_step:
                self.state_sample[i-self.therm_step] = self.current_state.clone()

        if self.is_cuda:
            self.state_sample = self.state_sample.to(self.device)
            self.nqs = self.nqs.to(self.device)

        # remove duplicate state
        sample_unique, sample_idx = torch.unique(self.state_sample, dim=0, return_counts=True)
        
        # calculate the max nbatch for given Max Memory
        nbatch = get_nbatch(self.sorb, len(sample_unique), self.n_SinglesDoubles, self.max_memory, self.alpha)
        e_total, eloc = total_energy(sample_unique, nbatch, self.h1e, self.h2e, self.nqs,
                                     self.ecore, self.sorb, self.nele, self.noa, self.nob,
                                     state_idx=sample_idx,
                                     verbose=self.verbose,
                                     exact=self.debug_exact)

        if self.verbose:
            print(f"sampling: {len(sample_idx)}/{sample_idx.sum().item()}")
            print(f"spin flip average time: {spin_time.mean():.3f} ms, total time {spin_time.sum():.3f} ms")
            # print(f"sample state:\n{(uint8_to_bit(self.state_sample.detach(), self.sorb)+1)//2}")
            # print(f"sample state:\n{self.state_sample}")
            print(f"total energy: {e_total:.10f}")

        if self.record_sample:
            idx = sample_idx.to("cpu").numpy()
            sample_str = self.state_str(sample_unique)
            full_dict = dict.fromkeys(self.str_full, 0)
            for s, i in zip(sample_str, idx):
                full_dict[s] += i
            new_df = pd.DataFrame({self.time_sample: full_dict.values()})
            self.frame_sample = pd.concat([self.frame_sample, new_df], axis=1)
            del full_dict
        
        self.time_sample += 1
        delta = time.time_ns() - t0
        print(f'Completed Monte Carlo Sampling {delta/1.0E09:.3f} s, acceptance ratio = {self.n_accept/self.n_sample:.3f}')

        # return self.state_sample.detach(), eloc1, e_total1,
        return sample_unique.detach(), sample_idx, eloc, e_total

    # right -> left (0011-> HF state H2)
    def state_str(self, state) -> List :
        tmp = []
        full_bit = ((uint8_to_bit(state, self.sorb)+1)//2).to(torch.uint8).tolist()
        for lst in full_bit:
            tmp.append("".join(list(map(str, lst))[::-1]))
        return tmp
    
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}:" + " (\n"
            f"    the number of sample: {self.n_sample}\n" + 
            f"    therm step: {self.therm_step}\n" +
            f"    exact sampling: {self.debug_exact}\n"
            f"    the given full space shape: {self.full_space.shape}\n" + 
            f"    Record the sample: {self.record_sample}\n" + 
            f"    Singles + Doubles: {self.n_SinglesDoubles}\n" + ")"
        )
