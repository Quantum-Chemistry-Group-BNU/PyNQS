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
from vmc.PublicFunction import check_para

__all__ = ["local_energy","total_energy", "energy_grad"]
print = partial(print, flush=True)

# @profile(precision=4, stream=open('eloc_memory_profiler.log','w+'))
def local_energy(x: Tensor, h1e: Tensor, h2e: Tensor, 
                 ansatz: Callable,
                 sorb: int, nele: int,
                 noa: int, nob: int,
                 verbose: bool = False) ->Tuple[Tensor, Tensor]:
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

    # x1: [batch, comb, sorb]
    comb_x, x1 = pt.get_comb_tensor(x, sorb, nele, noa, nob, True)

    # calculate matrix <x|H|x'>
    t1 = time.time_ns()
    comb_hij = pt.get_hij_torch(x, comb_x, h1e, h2e, sorb, nele) # shape (1, n)/(batch, n)
    
    t2 = time.time_ns()
    with torch.no_grad():
        psi_x1 = ansatz(x1)

    if x1.is_cuda:
        torch.cuda.synchronize()
    t3 = time.time_ns()

    if dim == 2 and batch == 1:
        eloc = torch.sum(comb_hij * psi_x1 / psi_x1[..., 0]) # scalar
    elif dim == 2 and batch > 1:
        eloc = torch.sum(torch.div(psi_x1.T, psi_x1[..., 0]).T * comb_hij, -1) # (batch)

    # for i in range(dim):
    #     a = np.allclose(
    #         x[i].detach().to("cpu").numpy(),
    #         comb_x[i][0].detach().to("cpu").numpy(),
    #     )
    #     assert(a)

    # device = x.device
    # comb_x_0 = pt.get_comb_tensor_0(x.to("cpu"), sorb, nele, True).to(device)
    # comb_hij_0 = pt.get_hij_torch(x, comb_x_0, h1e, h2e, sorb, nele) 
    # x_bit = pt.uint8_to_bit(comb_x_0, sorb)
    # psi_x1_0 = ansatz(x_bit)
    # eloc_0 = torch.sum(torch.div(psi_x1_0.T, psi_x1_0[..., 0]).T * comb_hij_0, -1) # (batch)

    # for i in range(dim):
    #     a = np.allclose(
    #         x[i].detach().to("cpu").numpy(),
    #         comb_x_0[i][0].detach().to("cpu").numpy(),
    #     )
    #     assert(a)


    # a = np.allclose(
    #     eloc.detach().to("cpu").numpy(),
    #     eloc_0.detach().to("cpu").numpy())
    
    # assert(a)

    if verbose:
        print(
            f"comb_x/uint8_to_bit time: {(t1-t0)/1.0E06:.3f} ms, <i|H|j> time: {(t2-t1)/1.0E06:.3f} ms," +
            f"nqs time: {(t3-t2)/1.0E06:.3f} ms")
    del x1, comb_hij, comb_x
    return eloc, psi_x1[..., 0]

# @profile(precision=4, stream=open('total_memory_profiler.log','w+'))
def total_energy(x: Tensor, nbatch: int, h1e: Tensor, h2e: Tensor, ansatz: Callable,
                ecore: float,
                sorb: int, nele: int,
                noa: int, nob: int,
                state_idx: Tensor= None,
                verbose: bool = False,
                exact: bool = False) -> Tuple[float, Tensor]:

    dim: int = x.shape[0]
    device = x.device
    eloc_lst = torch.zeros(dim, dtype=torch.float64).to(device)
    psi_lst = torch.zeros(dim, dtype=torch.float64).to(device)
    idx_lst = torch.arange(dim).to(device)
   
    # calculate the total energy using splits
    t0 = time.time_ns()
    ons_dataset = Data.TensorDataset(x, idx_lst)
    loader = Data.DataLoader(dataset=ons_dataset, batch_size=nbatch, 
                              shuffle=False, drop_last=False)
    
    # for step, (ons, idx) in enumerate(loader):
    # TODO: memory consuming
    # print(x.shape, nbatch)
    for ons, idx in loader:
    # for ons, idx in zip(x.split(nbatch), idx_lst.split(nbatch)):
        # lp = LineProfiler()
        # lp_wrapper = lp(local_energy)
        # lp_wrapper(ons, h1e, h2e, ansatz, sorb, nele, verbose=verbose)
        # lp.print_stats()
        #  exit()
        eloc_lst[idx], psi_lst[idx] = local_energy(ons, h1e, h2e, ansatz, sorb, nele, noa, nob, verbose=verbose)
    delta = time.time_ns() - t0

    if exact:
        if torch.any(torch.isnan(eloc_lst)):
            print(f"total energy is nan in error")
            print(eloc_lst)
            print(psi_lst)
            exit()
        e_total = (eloc_lst * (psi_lst.pow(2)/(psi_lst.pow(2).sum()))).sum() + ecore
    else:
        if state_idx is not None:
            state_prob = state_idx/state_idx.sum()
            e_total = torch.einsum("i, i ->", eloc_lst, state_prob) + ecore
        else:
            e_total = eloc_lst.mean() + ecore

    if verbose:
        print(f"total energy cost time: {delta/1.0E06:.3f} ms")

    # print(e_total.item())
    # print(f"eloc:")
    # print(eloc_lst)
    del psi_lst, idx_lst
    return e_total.item(), eloc_lst

def energy_grad(eloc: Tensor, dlnPsi_lst: List[Tensor], 
                N_state: int, state_idx: Tensor = None,
                psi: Tensor = None,
                exact: bool = False) -> List[Tensor]:
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
    psi_norm = psi.pow(2)/(psi.pow(2).sum())
    for para in dlnPsi_lst:
        dlnPsi = para.reshape(N_state, -1) # (N_state, N_para), two dim
        # dlnPsi_all = para_all.reshape(state_idx.sum(), -1)
        if not exact:
            if state_idx is None:
                # if state_idx.sum().item() != N_state:
                #     raise Exception(f"The number of state {N_state} is not equal of state_idx {state_idx.sum().item()} ")
                state_prob = torch.ones(N_state, dtype=torch.double, device=eloc.device)/N_state
            else:
                state_prob = state_idx/state_idx.sum()
            F_p = torch.einsum("i, ij, i ->j", eloc, dlnPsi.conj(), state_prob)
            F_p -= torch.einsum("i, i ->", eloc, state_prob) * torch.einsum("ij, i -> j", dlnPsi.conj(), state_prob)
            # a = torch.einsum("i, ij ->j", eloc_all, dlnPsi_all.conj())/(state_idx.sum())
            # a -= eloc_all.mean() * dlnPsi_all.conj().mean(axis=0)
            # print(np.allclose(a.numpy(), F_p.numpy()))
        else:
            F_p = torch.einsum("i, ij, i ->j", eloc, dlnPsi.conj(), psi_norm)
            F_p -= torch.einsum("i, ij ->j", psi_norm, dlnPsi.conj()) * ((psi_norm * eloc).sum())
        lst.append(2 * F_p.real)
    del psi_norm
    return lst
