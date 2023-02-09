import time
import torch 
from typing import Callable, Tuple
from torch import Tensor

import libs.hij_tensor as pt
from vmc.PublicFunction import check_para

__all__ = ["local_energy","total_energy" ]

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
    # TODO: "get_comb_tensor" in cuda 
    # TODO: python version x->comb_x

    device = x.device
    dim: int = x.dim()
    batch: int = x.shape[0]
    t0 = time.time_ns()
    comb_x = pt.get_comb_tensor(x.to("cpu"), sorb, nele, True).to(device)
    # calculate matrix <x|H|x'>
    delta0 = (time.time_ns()-t0)/1.0E06
    comb_hij = pt.get_hij_torch(x, comb_x, h1e, h2e, sorb, nele) # shape (1, n)/(batch, n)
    
    t1 =  time.time_ns()
    # TODO: time consuming
    x =  pt.unit8_to_bit(comb_x, sorb)
    delta1 = (time.time_ns()-t1)/1.0E06

    t2 = time.time_ns()
    psi_x1 = ansatz(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    delta2 = (time.time_ns()-t2)/1.0E06
    # print(rbm.phase(unit8_to_bit(comb_x, sorb))[1])
    # print(rbm.amplitude(unit8_to_bit(comb_x, sorb))[1])
    if dim == 2 and batch == 1:
        eloc = torch.sum(comb_hij * psi_x1 / psi_x1[..., 0]) # scalar
    elif dim == 2 and batch > 1:
        eloc = torch.sum(torch.div(psi_x1.T, psi_x1[..., 0]).T * comb_hij, -1) # (batch)

    if verbose:
        print(f"comb_x time: {delta0:.3f} ms, unit8_to_bit time: {delta1:.3f} ms, nqs time: {delta2:.3f} ms")
        
    return eloc, psi_x1[..., 0]

def total_energy(x: Tensor, nbatch: int, h1e: Tensor, h2e: Tensor, ansatz,
                ecore: float,
                sorb: int, nele: int, device: str="cuda", 
                sample_weights:Tensor = None,
                verbose: bool = False) -> Tuple[Tensor, Tensor]:
    
    dim: int = x.shape[0]
    eloc_lst = torch.zeros(dim, dtype=torch.float64).to(device)
    psi_lst = torch.zeros(dim, dtype=torch.float64).to(device)
    idx_lst = torch.arange(dim).to(device)

    # calculate the total energy using splits
    t0 = time.time_ns()
    for ons, idx in zip(x.split(nbatch), idx_lst.split(nbatch)):
        eloc_lst[idx], psi_lst[idx] =  local_energy(ons, h1e, h2e, ansatz, sorb, nele)
    delta = time.time_ns() - t0 
    if verbose:
        print(f"local energy is {eloc_lst}, cost time: {delta/1.0E06:.3f} ms")

    log_psi = psi_lst.log()
    if sample_weights is None:
        sample_weights = torch.ones(dim, dtype=torch.float64).to(device) 
        sample_weights /= sample_weights.sum()

    # gradients 2 * Real(Ep[(Eloc -Ep[Eloc]) * grad_thetaln(psi*)])
    eloc_corr = eloc_lst - (sample_weights * eloc_lst).sum(axis=0).detach()
    exp_op = 2 * (sample_weights * torch.matmul(eloc_corr, log_psi)).sum(axis=0) 

    # total energy 
    e_total = (eloc_lst * (psi_lst.pow(2)/(psi_lst.pow(2).sum()))).sum() + ecore

    return e_total, exp_op