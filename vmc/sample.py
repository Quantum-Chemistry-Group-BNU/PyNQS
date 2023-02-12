import time
import random
import torch
from typing import Callable, Tuple
from torch import Tensor 
from vmc.PublicFunction import check_para, unit8_to_bit
from vmc.eloc import local_energy

class MCMCSampler():
    """
    Generates samples of configurations from a neural quantum state(NQS)
    using Markov chain Monte Carlo(MCMC) algorithm
    """
    def __init__(self, nqs: Callable, initial_state: Tensor,
                 h1e: Tensor, h2e: Tensor,
                 n_sample: int, sorb: int, nele: int,
                 therm_step: int = 2000, verbose: bool = False, 
                 debug_exact: bool = False, full_space: Tensor = None
                 ) -> None:
        check_para(initial_state)
        if debug_exact:
            assert full_space is not None, "full space must be given"
        if n_sample < 50:
            raise ValueError(f"The number of sample{n_sample} should great 50")

        self.debug = debug_exact
        self.nqs = nqs
        self.sorb = sorb
        self.nele = nele
        self.no = nele
        self.nv = sorb - nele
        self.eloc: Tensor = None
        self.verbose: bool = verbose
        self.h1e: Tensor = h1e
        self.h2e: Tensor = h2e
        self.therm_step = therm_step
        self.n_sample = n_sample
        self.n_accept = 0

        # TODO: initial_state is unit8 or [-1, 1] bit ???
        self.current_state: Tensor = initial_state
        self.next_state: Tensor = initial_state
        if self.debug:
            self.state_sample: Tensor = full_space
        else:
            self.state_sample: Tensor = initial_state.repeat(self.n_sample, 1)


    def run(self, n_sweep: int = None) -> Tuple[Tensor,Tensor]:
        if self.debug:
            self.eloc = local_energy(self.state_sample, self.h1e, self.h2e, 
                                    self.nqs, self.sorb, 
                                    self.nele, self.verbose)[0]
            return self.state_sample.detach(), self.eloc 
        
        if (n_sweep is None) or (n_sweep <= self.therm_step + self.n_sample):
            n_sweep = self.therm_step+ self.n_sample

        print('Starting MCMC Sampling')
        t0 = time.time_ns()
        prob_current = self.nqs(unit8_to_bit(self.current_state, self.sorb))**2
        spin_time = torch.zeros(n_sweep)
        for i in range(n_sweep):
            t1 = time.time_ns()
            # TODO: Time-consuming
            self.spin_flip_rand(self.next_state)
            psi = unit8_to_bit(self.next_state, self.sorb)
            spin_time[i] = (time.time_ns() - t1)/1.0E06
            prob_next = self.nqs(psi)**2
            prob_accept = min(1.00, (prob_next/prob_current).item())
            p = random.random()
            # print(f"random p {p:.3f}, prob_accept {prob_accept:.3f}")
            if p <= prob_accept:
                self.current_state = self.next_state.clone()
                prob_current = prob_next.clone()
                if i >= self.therm_step:
                    self.n_accept += 1
                    
            if i >= self.therm_step:
                self.state_sample[i-self.therm_step] = self.current_state.clone()
        
        delta = time.time_ns() - t0
        print(f'Completed Monte Carlo Sampling {delta/1.0E09:.3f} s, acceptance ratio = {self.n_accept/self.n_sample:.3f}')
 
        if self.verbose:
            print(f"spin flip average time: {spin_time.mean():.3f} ms, total time {spin_time.sum():.3f} ms")

        # calculate local energy
        self.eloc = local_energy(self.state_sample, self.h1e, self.h2e, 
                                 self.nqs, self.sorb, 
                                 self.nele, self.verbose)[0]

        return self.state_sample.detach(), self.eloc.detach()

    def spin_flip_rand(self, x: Tensor):
        olst, vlst = self.get_olst(x, self.sorb, self.nele)

        while True:
            ia = random.randrange(self.no * self.nv)
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