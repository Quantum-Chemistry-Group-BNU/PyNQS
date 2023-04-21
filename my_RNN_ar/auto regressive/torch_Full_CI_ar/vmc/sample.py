import time
import os
import random
import torch
import tempfile
import copy
import numpy as np
import pandas as pd
from functools import partial
from typing import Callable, Tuple, List
from torch import Tensor, nn
from pandas import DataFrame

from memory_profiler import profile
from line_profiler import LineProfiler

from vmc.eloc import total_energy
from libs.hij_tensor import uint8_to_bit, spin_flip_rand, MCMC_sample
from utils import state_to_string, ElectronInfo, check_para, get_nbatch, given_onstate

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
    def __init__(self, nqs: nn.Module, ele_info: ElectronInfo,
                 n_sample: int = 100, 
                 therm_step: int = 2000, verbose: bool = False, 
                 debug_exact: bool = False,
                 seed: int = 100, record_sample: bool = True,
                 max_memory: float = 4, alpha: float = 0.25,
                 dtype = torch.double,
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

        # device and cuda
        self.is_cuda = True if self.h1e.is_cuda else False
        self.device = self.h1e.device
        self.dtype = dtype

        # save sampler
        self.record_sample = record_sample
        if self.record_sample:
            self.str_full = state_to_string(self.full_space, self.sorb)
            self.frame_sample = pd.DataFrame({"full_space": self.str_full})
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
        self.h1e: Tensor = ele_info.h1e
        self.h2e: Tensor = ele_info.h2e
        self.ecore = ele_info.ecore
        self.n_SinglesDoubles = ele_info.n_SinglesDoubles
        self.full_space= ele_info.onstate

    # @profile(precision=4, stream=open('MCMC_memory_profiler.log','w+'))
    def run(self, initial_state: Tensor, n_sweep: int = None) -> Tuple[Tensor, Tensor, Tensor, float, dict]:
        check_para(initial_state)
        if self.debug_exact:
            dim = len(self.full_space)
            e_total, eloc, stats_dict = self.calculate_energy(self.full_space)
            sample_counts = torch.ones(dim, dtype=torch.float64, device=self.device)
            return self.full_space.detach(), sample_counts, eloc, e_total, stats_dict

        t0 = time.time_ns()
        sample_unique, sample_counts, state_prob, psi_unique = self.auto_regressive()
        e_total, eloc, stats_dict = self.calculate_energy(sample_unique, state_prob=state_prob)

        # print local energy statistics information
        self._statistics(stats_dict)

        if self.verbose:
            print(f"sampling: {len(sample_counts)}/{sample_counts.sum().item()}")
            # print(f"spin flip average time: {spin_time.mean():.3f} ms, total time {spin_time.sum():.3f} ms")
            print(f"total energy: {e_total:.10f}")

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
        # print(f'Completed Monte Carlo Sampling {delta/1.0E09:.3f} s, acceptance ratio = {self.n_accept/self.n_sample:.3f}')
        return sample_unique.detach(), sample_counts, eloc, e_total, stats_dict

    def MCMC(self, n_sweep: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        print('Starting MCMC Sampling')
        if True:
            example_inputs = uint8_to_bit(self.current_state, self.sorb)
            serialized_model = torch.jit.trace(self.nqs, example_inputs)
            model_file = tempfile.mkstemp()[1]
            serialized_model.save(model_file)
            # print(f"Serialized model time: {(time.time_ns() - t0)/1.E06:.3f} ms")
            with torch.no_grad():
                self.n_accept = MCMC_sample(model_file, self.current_state, self.state_sample, 
                                            self.psi_sample, self.sorb, self.nele, self.noa, 
                                            self.nob, self.seed, n_sweep, self.therm_step)
            os.remove(model_file)
            # print(f"CPP model time: {(time.time_ns() - t0)/1.E06:.3f} ms")
        else:
            with torch.no_grad():
                psi_current = self.nqs(uint8_to_bit(self.current_state, self.sorb))
                prob_current = psi_current.norm()**2
            for i in range(n_sweep):
                psi, self.next_state = spin_flip_rand(self.current_state, self.sorb, self.nele, self.noa, self.nob, self.seed)
                with torch.no_grad():
                    psi_next = self.nqs(psi)
                    prob_next = psi_next.norm()**2
                prob_accept = min(1.00, (prob_next/prob_current).item())
                p = random.random()
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
                    self.state_sample[i-self.therm_step] = self.current_state.clone()
                    self.psi_sample[i - self.therm_step] = psi_current.clone()

        if self.is_cuda:
            self.state_sample = self.state_sample.to(self.device)
            self.nqs = self.nqs.to(self.device)

        # remove duplicate state
        if self.is_cuda:
            # torch.unique could not return unique indices in old tensor
            sample_unique, sample_counts = torch.unique(self.state_sample, dim=0, return_counts=True)
            psi_unique = self.nqs(uint8_to_bit(sample_unique, self.sorb))
        else:
            # CPU using Numpy
            sample_unique, psi_idx, sample_counts = np.unique(self.state_sample.numpy(), axis=0, return_counts=True, return_index=True)
            psi_unique = self.psi_sample[torch.from_numpy(psi_idx)]
            sample_unique = torch.from_numpy(sample_unique)
            sample_counts =torch.from_numpy(sample_counts)

        state_prob = sample_counts/sample_counts.sum()
        return sample_unique, sample_counts, state_prob, psi_unique

    def auto_regressive(self):
        print("==========start autoregressive sampling==========")
        samples = torch.zeros(self.n_sample, 8, dtype=torch.uint8)
        ar = ARSampler(self.sorb, self.n_sample, self.device)
        samples_0b = ar.sample().type(torch.uint8)
        for i in range(self.sorb):
            samples[:,0] += 2**i * samples_0b[:, -1-i]

        # remove the unreasonable state
        samples = samples.tolist()
        samples_ = copy.deepcopy(samples)
        for i in samples:
            if i not in given_onstate(self.sorb, self.sorb, self.noa, self.nob).tolist():
                samples_.remove(i)
        self.state_sample = torch.tensor(samples_, dtype=torch.uint8)

        if self.is_cuda:
            sample_unique, sample_counts = torch.unique(self.state_sample, dim=0, return_counts=True)
            psi_unique = self.nqs(uint8_to_bit(sample_unique, self.sorb))
        else:
            self.psi_sample: Tensor = torch.zeros(self.n_sample, dtype=self.dtype, device=self.device)
            sample_unique, psi_idx, sample_counts = np.unique(self.state_sample.numpy(), axis=0, return_counts=True,
                                                              return_index=True)
            psi_unique = self.psi_sample[torch.from_numpy(psi_idx)]
            sample_unique = torch.from_numpy(sample_unique)
            sample_counts = torch.from_numpy(sample_counts)

        state_prob = sample_counts / sample_counts.sum()
        return sample_unique, sample_counts, state_prob, psi_unique

    def prepare_sample(self, initial_state: Tensor):
        self.state_sample: Tensor = torch.zeros_like(initial_state).repeat(self.n_sample, 1)
        self.current_state: Tensor = initial_state.clone()
        self.next_state: Tensor = initial_state.clone()
        self.n_accept = 0
        self.psi_sample: Tensor = torch.zeros(self.n_sample, dtype=self.dtype, device=self.device)

        if (n_sweep is None) or (n_sweep <= self.therm_step + self.n_sample):
            n_sweep = self.therm_step + self.n_sample

        # convert to CPU and 'spin_flip_rand' is not implemented in "GPU"
        if self.is_cuda:
            self.state_sample = self.state_sample.to("cpu") 
            self.current_state = self.current_state.to("cpu")
            self.next_state = self.next_state.to("cpu")
            self.nqs = self.nqs.to("cpu")
            self.psi_sample = self.psi_sample.to("cpu")

    # TODO: how to calculate batch_size;
    # calculate the max nbatch for given Max Memory
    def calculate_energy(self, sample: Tensor, state_prob: Tensor = None):
        nbatch = get_nbatch(self.sorb, len(sample), self.n_SinglesDoubles, self.max_memory, self.alpha)
        e_total, eloc, stats_dict = total_energy(sample, nbatch, self.h1e, self.h2e, self.nqs,
                                                 self.ecore, self.sorb, self.nele, self.noa, self.nob,
                                                 state_prob=state_prob,
                                                 verbose=self.verbose,
                                                 exact=self.debug_exact,
                                                 dtype=self.dtype)
        return e_total, eloc, stats_dict

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}:" + " (\n"
            f"    the number of sample: {self.n_sample}\n" + 
            f"    therm step: {self.therm_step}\n" +
            f"    exact sampling: {self.debug_exact}\n"
            f"    the given full space shape: {self.full_space.shape}\n" + 
            f"    Record the sample: {self.record_sample}\n" + 
            f"    Singles + Doubles: {self.n_SinglesDoubles}\n" +
            f"    Random seed: {self.seed}\n" + ")"
        )
    
    def _statistics(self, data: dict):
        s = f"E_total = {data['mean'].real:.10f} ± {data['SE'].real:.3E} [σ² = {data['var'].real:.3E}]"
        print(s)

class VMCEnergy:
    def __init__(self, nqs: nn.Module) -> None:
        self._model = nqs

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, nqs: nn.Module):
            self._model = nqs

    def energy(self,e: ElectronInfo, sampler_param: dict, 
               initial_state: Tensor = None) -> float:
        sampler = MCMCSampler(self.model, e, **sampler_param)
        if initial_state is None:
            initial_state = sampler.ele_info.onstate[0]
        return sampler.run(initial_state)[3]


class ARSampler(torch.nn.Module):
    def __init__(self, sorb: int, n_sample: int = 100, device: str = None):
        super(ARSampler, self).__init__()
        self.device = device
        self.factory_kwargs = {'device': self.device, 'dtype':torch.double}
        self.sorb = sorb
        self.n_sample = n_sample
        self.num_hiddens = 50
        self.num_layers = 1

        # 定义神经网络
        self.GRU = nn.GRU(input_size=2, hidden_size=50, num_layers=1, **self.factory_kwargs)
        self.fc = nn.Linear(50, 2, **self.factory_kwargs)

    def sqsoftmax(self, x):
        return torch.sqrt(torch.softmax(x, dim=1))

    def softsign(self, x):
        return torch.pi*(nn.functional.softsign(x))

    def heavyside(self, x):
        sign = torch.sign(torch.sign(x) + 0.1)
        return 0.5 * (sign + 1.0)

    def sample(self):
        samples = torch.zeros(self.n_sample, self.sorb+1, 2)  # 初始化样本集
        hidden = torch.zeros(self.num_layers,self.num_hiddens, **self.factory_kwargs)

        for i in range(self.sorb):
            output, hidden = self.GRU(samples[:, i, :], hidden)
            output = self.fc(output)
            output = self.sqsoftmax(output)

            # 这里保证合理激发
            if i >= self.sorb/2:
                num_up = torch.sum(torch.argmax(samples[:, 1:i+1, :], dim=2), dim=1).float()
                baseline = (self.sorb//2-1) * torch.ones(self.n_sample, dtype=torch.float32)
                num_down = i*torch.ones(self.n_sample, dtype=torch.float32) - num_up
                activations_up = self.heavyside(baseline-num_up)
                activations_down = self.heavyside(baseline - num_down)
                output = output * torch.stack([activations_down, activations_up], dim=1).float()
                output = output / output.norm(dim=1, p=2, keepdim=True).clamp(min=1e-30)

            samples[:, i + 1, :] = nn.functional.one_hot(torch.multinomial(output, 1), num_classes=2).squeeze(1)

        return torch.argmax(samples[:, 1:, :], dim=2)

