import time
import torch 
import numpy as np 
import torch.utils.data as Data
from functools import partial
from typing import Callable, Tuple, List
from torch import Tensor

from memory_profiler import profile
from line_profiler import LineProfiler

import libs.hij_tensor as pt
from utils import check_para

__all__ = ["local_energy","total_energy", "energy_grad"]
print = partial(print, flush=True)

# @profile(precision=4, stream=open('eloc_memory_profiler.log','w+'))
def local_energy(x: Tensor, h1e: Tensor, h2e: Tensor, 
                 ansatz: Callable,
                 sorb: int, nele: int,
                 noa: int, nob: int,
                 verbose: bool = False,
                 dtype = torch.double) ->Tuple[Tensor, Tensor]:
    """
    Calculate the local energy for given state.
    E_loc(x) = \sum_x' psi(x')/psi(x) * <x|H|x'> 
    1. the all Singles and Doubles excitation about given state using cpu:
        x: (1, sorb)/(batch, sorb) -> comb_x: (batch, ncomb, sorb)/(ncomb, sorb)
    2. matrix <x|H|x'> (1, ncomb)/(batch, ncomb)
    3. psi(x), psi(comb_x)[ncomb] using NAQS. 
    4. calculate the local energy
    """
    check_para(x)

    dim: int = x.dim()
    batch: int = x.shape[0]
    t0 = time.time_ns()

    # x1: [batch, comb, sorb], comb_x: [batch, comb, bra_len]
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True, profile_memory=True) as prof:
    comb_x, x1 = pt.get_comb_tensor(x, sorb, nele, noa, nob, True)

    # calculate matrix <x|H|x'>
    t1 = time.time_ns()
    comb_hij = pt.get_hij_torch(x, comb_x, h1e, h2e, sorb, nele) # shape (1, comb)/(batch, comb)
    
    t2 = time.time_ns()
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True, profile_memory=True) as prof:
    with torch.no_grad():
        psi_x1 = ansatz(x1.reshape(-1, sorb)).reshape(batch, -1) # [batch, comb]

    if x1.is_cuda:
        torch.cuda.synchronize()
    t3 = time.time_ns()

    if dim == 2 and batch == 1:
        eloc = torch.sum(comb_hij * psi_x1 / psi_x1[..., 0]) # scalar
    elif dim == 2 and batch > 1:
        eloc = torch.sum(torch.div(psi_x1.T, psi_x1[..., 0]).T * comb_hij, -1) # (batch)

    if verbose:
        print(
            f"comb_x/uint8_to_bit time: {(t1-t0)/1.0E06:.3f} ms, <i|H|j> time: {(t2-t1)/1.0E06:.3f} ms," +
            f"nqs time: {(t3-t2)/1.0E06:.3f} ms")
    del x1, comb_hij, comb_x
    # print(eloc, psi_x1[..., 0])

    return eloc.to(dtype), psi_x1[..., 0].to(dtype)

# @profile(precision=4, stream=open('total_memory_profiler.log','w+'))
def total_energy(x: Tensor, nbatch: int, h1e: Tensor, h2e: Tensor, ansatz: Callable,
                ecore: float,
                sorb: int, nele: int,
                noa: int, nob: int,
                state_counts: Tensor= None,
                verbose: bool = False,
                exact: bool = False,
                dtype = torch.double) -> Tuple[float, Tensor, dict]:

    dim: int = x.shape[0]
    device = x.device
    eloc_lst = torch.zeros(dim, device=device).to(dtype)
    psi_lst = torch.zeros_like(eloc_lst)
    idx_lst = torch.arange(dim).to(device)
    statistics = {}

    # calculate the total energy using splits
    t0 = time.time_ns()
    # ons_dataset = Data.TensorDataset(x, idx_lst)
    # loader = Data.DataLoader(dataset=ons_dataset, batch_size=nbatch, 
    #                           shuffle=False, drop_last=False)

    # for step, (ons, idx) in enumerate(loader):
    # for ons, idx in loader: # why is slower than using split?
    for ons, idx in zip(x.split(nbatch), idx_lst.split(nbatch)):
        eloc_lst[idx], psi_lst[idx] = local_energy(ons, h1e, h2e, ansatz, sorb, nele, noa, nob, verbose=verbose, dtype=dtype)

    if exact:
        if torch.any(torch.isnan(eloc_lst)):
            print(eloc_lst)
            print(psi_lst)
            raise ValueError(f"local energy is nan in error")
        # e_total = (eloc_lst * (psi_lst.pow(2)/(psi_lst.pow(2).sum()))).sum() + ecore
        e_total = (eloc_lst * (psi_lst * psi_lst.conj()/psi_lst.norm()**2)).sum() + ecore
    else:
        if state_counts is None:
            # [1, 1, 1, ...]
            state_counts = torch.ones(dim, dtype=dtype).to(device)
        state_prob = (state_counts/state_counts.sum()).to(dtype)
        eloc_mean = torch.einsum("i, i ->", eloc_lst, state_prob)
        e_total = eloc_mean + ecore

        variance = torch.sum((eloc_lst - eloc_mean)**2 * state_counts)
        n_sample = state_counts.sum()
        sd = torch.sqrt(variance/n_sample)
        se = sd/torch.sqrt(n_sample)
        statistics["mean"] = e_total.real.item()
        statistics["var"] = variance.real.item()
        statistics["SD"] = sd.item()
        statistics["SE"] = se.item()


    t1 = time.time_ns()

    if verbose:
        print(f"total energy cost time: {(t1-t0)/1.0E06:.3f} ms")

    del psi_lst, idx_lst
    print(f"total energy: {e_total.real.item():.8f}")
    return e_total.real.item(), eloc_lst, statistics

def energy_grad(eloc: Tensor, dlnPsi_lst: List[Tensor], 
                N_state: int, state_idx: Tensor = None,
                psi: Tensor = None,
                exact: bool = False,
                dtype = torch.double) -> List[Tensor]:
    """
    calculate the energy gradients in sampling and exact:
        sampling:
            F_p = 2*Real(<E_loc * O*> - <E_loc> * <O*>)
        exact:
            F_p = 2*Real(P(n) * (O*_n * E_loc(n) - O*_n * <E_loc> 
             <E_loc> = \sum_n[ P(n)* E_loc(n)]
      return
         List, length: n_para, element: [N_para],one dim
    """
    lst = []
    if exact:
        state_prob = psi * psi.conj() / psi.norm()**2
    else:
        if state_idx is None:
            state_prob = torch.ones(N_state, dtype=dtype, device=eloc.device)/N_state
        else:
            state_prob = state_idx/state_idx.sum()
    state_prob = state_prob.to(dtype)

    for para in dlnPsi_lst:
        dlnPsi = para.reshape(N_state, -1).to(dtype) # (N_state, N_para), two dim
        F_p = torch.einsum("i, ij, i ->j", eloc, dlnPsi.conj(), state_prob)
        F_p -= torch.einsum("i, i ->", eloc, state_prob) * torch.einsum("ij, i -> j", dlnPsi.conj(), state_prob)
        lst.append(2 * F_p.real)
    return lst

def mc_exact_errors(full_state: Tensor, 
                    mc_state: Tensor, 
                    nbatch: int,
                    h1e: Tensor, h2e: Tensor,
                    ansatz: Callable, 
                    ecore: float, 
                    nele:int, sorb: int,
                    noa: int, nob: int):
    e_mc = total_energy(mc_state, nbatch, h1e, h2e, ansatz, ecore, sorb, nele, noa, nob, exact=False)[0]
    e_exact = total_energy(full_state, nbatch, h1e, h2e, ansatz, ecore, sorb, nele, noa, nob, exact=True)[0]

    e_delta = abs(e_mc - e_exact)

    return e_delta
