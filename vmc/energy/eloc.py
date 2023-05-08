import time
import torch 
import torch.utils.data as Data
from functools import partial
from typing import Callable, Tuple, List
from torch import Tensor

from memory_profiler import profile
from line_profiler import LineProfiler

from libs.C_extension import get_hij_torch, get_comb_tensor
from utils import check_para

print = partial(print, flush=True)

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
    comb_x, x1 = get_comb_tensor(x, sorb, nele, noa, nob, True)

    # calculate matrix <x|H|x'>
    t1 = time.time_ns()
    comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele) # shape (1, comb)/(batch, comb)
    
    t2 = time.time_ns()
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=True, profile_memory=True) as prof:
    with torch.no_grad():
        psi_x1 = ansatz(2*(x1.reshape(-1, sorb))-1.0).reshape(batch, -1) # [batch, comb]

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