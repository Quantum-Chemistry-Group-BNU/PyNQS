import time
import torch
import torch.utils.data as Data
from functools import partial
from typing import Callable, Tuple, List, Union
from torch import Tensor, nn

from memory_profiler import profile
from line_profiler import LineProfiler

from libs.C_extension import get_hij_torch, get_comb_tensor
from utils import check_para

print = partial(print, flush=True)


def local_energy(x: Tensor,
                 h1e: Tensor,
                 h2e: Tensor,
                 ansatz: Union[nn.Module, Callable],
                 sorb: int,
                 nele: int,
                 noa: int,
                 nob: int,
                 verbose: bool = False,
                 dtype=torch.double) -> Tuple[Tensor, Tensor, Tuple[float, float, float]]:
    """
    Calculate the local energy for given state.
    E_loc(x) = \sum_x' psi(x')/psi(x) * <x|H|x'> 
    1. the all Singles and Doubles excitation about given state:
        x: (1, sorb)/(batch, sorb) -> comb_x: (batch, ncomb, sorb)/(ncomb, sorb)
    2. Compute matrix element <x|H|x'> (1, ncomb)/(batch, ncomb)
    3. psi(x), psi(comb_x)[ncomb] using NAQS. 
    4. calculate the local energy

    Return:
        eloc[Tensor]: local energy(nbatch)
        psi[Tensor]: psi(x1) 1D(nbatch)
    """
    check_para(x)

    dim: int = x.dim()
    batch: int = x.shape[0]
    t0 = time.time_ns()

    # x1: [batch, comb, sorb], comb_x: [batch, comb, bra_len]
    comb_x, x1 = get_comb_tensor(x, sorb, nele, noa, nob, True)

    # calculate matrix <x|H|x'>
    t1 = time.time_ns()
    comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele)  # shape (1, comb)/(batch, comb)

    t2 = time.time_ns()
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True, profile_memory=True) as prof:
    with torch.no_grad():
        psi_x1 = ansatz(x1.reshape(-1, sorb)).reshape(batch, -1)  # [batch, comb]
    # print(torch.cuda.mem_get_info())
    # print(prof.table())
    # exit()

    if x1.is_cuda:
        torch.cuda.synchronize(h1e.device)
    t3 = time.time_ns()

    if dim == 2 and batch == 1:
        eloc = torch.sum(comb_hij * psi_x1 / psi_x1[..., 0])  # scalar
    elif dim == 2 and batch > 1:
        eloc = torch.sum(torch.div(psi_x1.T, psi_x1[..., 0]).T * comb_hij, -1)  # (batch)

    delta0 = (t1 - t0) / 1.0E06
    delta1 = (t2 - t1) / 1.0E06
    delta2 = (t3 - t2) / 1.0E06
    if verbose:
        print(f"comb_x/uint8_to_bit time: {delta0:.3E} ms, <i|H|j> time: {delta1:.3E} ms, " +
              f"nqs time: {delta2:.3E} ms")
    del x1, comb_hij, comb_x

    return eloc.to(dtype), psi_x1[..., 0].to(dtype), (delta0, delta1, delta2)