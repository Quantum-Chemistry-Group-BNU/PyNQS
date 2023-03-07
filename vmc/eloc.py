import time
import torch 
import torch.utils.data as Data
from typing import Callable, Tuple, List
from torch import Tensor

from memory_profiler import profile
from line_profiler import LineProfiler

import libs.hij_tensor as pt
from vmc.PublicFunction import check_para

__all__ = ["local_energy","total_energy", "energy_grad"]

# @profile(precision=4, stream=open('eloc_memory_profiler.log','w+'))
def local_energy(x: Tensor, h1e: Tensor, h2e: Tensor, 
                 ansatz: Callable,
                 sorb: int, nele: int,
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
    comb_x = pt.get_comb_tensor(x, sorb, nele, True)

    # calculate matrix <x|H|x'>
    delta0 = (time.time_ns()-t0)/1.0E06
    comb_hij = pt.get_hij_torch(x, comb_x, h1e, h2e, sorb, nele) # shape (1, n)/(batch, n)
    
    t1 = time.time_ns()
    # TODO: time consuming
    x1 = pt.uint8_to_bit(comb_x, sorb)
    delta1 = (time.time_ns()-t1)/1.0E06

    t2 = time.time_ns()
    psi_x1 = ansatz(x1)
    # print(f"psi_x1: \n {psi_x1[..., 0]}")
    if x1.is_cuda:
        torch.cuda.synchronize()
    delta2 = (time.time_ns()-t2)/1.0E06

    if dim == 2 and batch == 1:
        eloc = torch.sum(comb_hij * psi_x1 / psi_x1[..., 0]) # scalar
    elif dim == 2 and batch > 1:
        eloc = torch.sum(torch.div(psi_x1.T, psi_x1[..., 0]).T * comb_hij, -1) # (batch)
    if verbose:
        print(f"comb_x time: {delta0:.3f} ms, unit8_to_bit time: {delta1:.3f} ms, nqs time: {delta2:.3f} ms")
    
    del x1, comb_hij, comb_x
    return eloc, psi_x1[..., 0]

# @profile(precision=4, stream=open('total_memory_profiler.log','w+'))
def total_energy(x: Tensor, nbatch: int, h1e: Tensor, h2e: Tensor, ansatz: Callable,
                ecore: float,
                sorb: int, nele: int,
                verbose: bool = False, 
                exact: bool = False) -> Tuple[float, Tensor]:
    
    dim: int = x.shape[0]
    device = x.device
    eloc_lst = torch.zeros(dim, dtype=torch.float64).to(device)
    psi_lst = torch.zeros(dim, dtype=torch.float64).to(device)
    idx_lst = torch.arange(dim).to(device)
   
    # calculate the total energy using splits
    t0 = time.time_ns()
    # batch_size = dim// nbatch
    # ons_dataset = Data.TensorDataset(x, idx_lst)
    # loader = Data.DataLoader(dataset=ons_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    # for step, (ons, idx) in enumerate(loader):
    # TODO: memory consuming
    for ons, idx in zip(x.split(nbatch), idx_lst.split(nbatch)):
        # lp = LineProfiler()
        # lp_wrapper = lp(local_energy)
        # lp_wrapper(ons, h1e, h2e, ansatz, sorb, nele, verbose=verbose)
        # lp.print_stats()
        #  exit()
        eloc_lst[idx], psi_lst[idx] = local_energy(ons, h1e, h2e, ansatz, sorb, nele, verbose=verbose)
    delta = time.time_ns() - t0 

    if exact:
        if torch.any(torch.isnan(eloc_lst)):
            print(f"total energy is nan in error")
            print(eloc_lst)
            print(psi_lst)
            exit()
        e_total = (eloc_lst * (psi_lst.pow(2)/(psi_lst.pow(2).sum()))).sum() + ecore
    else:
        e_total = eloc_lst.mean() + ecore
    
    if verbose:
        print(f"total energy cost time: {delta/1.0E06:.3f} ms")
    
    del psi_lst, idx_lst
    return e_total.item(), eloc_lst

def energy_grad(eloc: Tensor, dlnPsi_lst: List[Tensor], 
                N_state: int, psi: Tensor = None,
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
        if not exact: 
            F_p = torch.einsum("i, ij ->j", eloc, dlnPsi.conj())/N_state
            F_p -= eloc.mean() * dlnPsi.conj().mean(axis=0)
        else:
            F_p = torch.einsum("i, ij, i ->j", eloc, dlnPsi.conj(), psi_norm)
            F_p -= torch.einsum("i, ij ->j", psi_norm, dlnPsi.conj()) * ((psi_norm * eloc).sum())
        lst.append(2 * F_p.real)
    
    del psi_norm
    return lst
