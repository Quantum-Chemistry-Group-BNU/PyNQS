import time
import random
import torch
import numpy as np 
import pandas as pd
from typing import Callable, Tuple, List
from torch import Tensor 
from vmc.PublicFunction import check_para
from vmc.eloc import local_energy
from libs.hij_tensor import get_olst_vlst, uint8_to_bit, spin_flip_rand

class MCMCSampler():
    """
    Generates samples of configurations from a neural quantum state(NQS)
    using Markov chain Monte Carlo(MCMC) algorithm
    """
    n_accept: int 
    def __init__(self, nqs: Callable,
                 h1e: Tensor, h2e: Tensor,
                 sorb: int, nele: int, n_sample: int = 100, 
                 therm_step: int = 2000, verbose: bool = False, 
                 debug_exact: bool = False, full_space: Tensor = None,
                 seed: int = 100
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
        self.seed = seed
        self.eloc: Tensor = None
        self.verbose: bool = verbose
        self.h1e: Tensor = h1e
        self.h2e: Tensor = h2e
        self.therm_step = therm_step
        self.n_sample = n_sample
        self.full_space= full_space
        # TODO: initial_state is unit8 or [-1, 1] bit ???
        # save sampler
        self.frame_sample = pd.DataFrame({"full_space": self.state_str(self.full_space)})
        self.time_sample = 0

    def run(self, initial_state: Tensor, n_sweep: int = None) -> Tuple[Tensor,Tensor]:
        check_para(initial_state)
        if self.debug_exact:
            self.eloc = local_energy(self.full_space, self.h1e, self.h2e, 
                                    self.nqs, self.sorb, 
                                    self.nele, self.verbose)[0]
            return self.full_space.detach(), self.eloc
        else:
            self.state_sample: Tensor = torch.zeros_like(initial_state).repeat(self.n_sample, 1)
            self.current_state: Tensor = initial_state
            self.next_state: Tensor = initial_state
            self.n_accept = 0

        if (n_sweep is None) or (n_sweep <= self.therm_step + self.n_sample):
            n_sweep = self.therm_step + self.n_sample

        print('Starting MCMC Sampling')
        t0 = time.time_ns()
        prob_current = self.nqs(uint8_to_bit(self.current_state, self.sorb))**2
        spin_time = torch.zeros(n_sweep)
        for i in range(n_sweep):
            t1 = time.time_ns()
            psi, self.next_state = spin_flip_rand(self.next_state, self.sorb, self.nele, self.seed)
            spin_time[i] = (time.time_ns() - t1)/1.0E06
            prob_next = self.nqs(psi)**2
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
        
        delta = time.time_ns() - t0
        print(f'Completed Monte Carlo Sampling {delta/1.0E09:.3f} s, acceptance ratio = {self.n_accept/self.n_sample:.3f}')

        # calculate local energy
        self.eloc = local_energy(self.state_sample, self.h1e, self.h2e, 
                                 self.nqs, self.sorb, 
                                 self.nele, self.verbose)[0]

        if self.verbose:
            print(f"spin flip average time: {spin_time.mean():.3f} ms, total time {spin_time.sum():.3f} ms")
            # print(f"sample state:\n{(uint8_to_bit(self.state_sample.detach(), self.sorb)+1)//2}")
            print(f"sample state:\n{self.state_sample}")
            print(f"eloc energy:\n {self.eloc.detach()}")

        unique_sample, idx= torch.unique(self.state_sample, dim=0, return_counts=True)
        idx = idx.to("cpu").numpy()
        sample_str = self.state_str(unique_sample)
        full_dict = dict.fromkeys(self.frame_sample["full_space"], 0)
        for s, i in zip(sample_str, idx):
            full_dict[s] += i
        new_df = pd.DataFrame({self.time_sample: full_dict.values()})
        self.frame_sample = pd.concat([self.frame_sample, new_df], axis=1)
        # self.frame_sample[self.time_sample] = full_dict.values()
        self.time_sample += 1
        del full_dict

        return self.state_sample.detach(), self.eloc.detach()

    def state_str(self, state) -> List :
        tmp = []
        full_bit = ((uint8_to_bit(state, self.sorb)+1)//2).to(torch.uint8).tolist()
        for lst in full_bit:
            tmp.append("".join(list(map(str, lst))[::-1]))
        return tmp


    def spin_flip_rand(self, x: Tensor):
        # olst, vlst = self.get_olst(x, self.sorb, self.nele)
        olst, vlst = get_olst_vlst(x, self.sorb, self.nele)

        while True:
            ia = random.randrange(self.no * self.nv-1)
            ido = olst[ia%self.no] 
            idv = vlst[ia//self.no]
            if ( ( ido & 1) == ( idv & 1)):
                # TODO: errors Ms is not equal
                self._spin_flip(x, ido)
                self._spin_flip(x, idv)
                break
            else:
                continue

    @staticmethod    
    def _spin_flip(x: Tensor, n: int):
        x[n//8] ^= (1<<(n%8)) # unit8

    @staticmethod
    def get_olst(x: Tensor, sorb, nele) ->Tuple[Tensor, Tensor]:
        olst = []
        vlst = []
        flag = True
        idx = 0
        bra = x.detach().clone()
        for i in range(len(x)):
            if flag:
                for _ in range(8):
                    if (bra[i] & 1 == 1):
                        olst.append(idx)
                    else:
                        vlst.append(idx)
                    bra[i] >>= 1
                    idx +=1
                    if idx >= sorb:
                        flag = False
                        break
            else:
                break
        
        if len(olst) != nele:
            raise ValueError(f"the number of electron {nele} not equal olst {len(olst)}")

        return (olst, vlst)