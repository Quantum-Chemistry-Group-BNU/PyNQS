import time
import torch
import torch.utils.data as Data
import torch.distributed as dist
from functools import partial
from typing import Callable, Tuple, List, Union
from loguru import logger
from torch import Tensor, nn

from memory_profiler import profile
from line_profiler import LineProfiler

from libs.C_extension import get_hij_torch, get_comb_tensor, onv_to_tensor
from utils import check_para
from utils.public_function import WavefunctionLUT

print = partial(print, flush=True)


# TODO: how to save unique x eloc energy
def local_energy(
    x: Tensor,
    h1e: Tensor,
    h2e: Tensor,
    ansatz: Union[nn.Module, Callable],
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    dtype=torch.double,
    WF_LUT: WavefunctionLUT = None,
    use_unique: bool = True,
) -> Tuple[Tensor, Tensor, Tuple[float, float, float]]:
    """
    Calculate the local energy for given state.
    E_loc(x) = \sum_x' psi(x')/psi(x) * <x|H|x'>
    1. the all Singles and Doubles excitation about given state:
        x: (1, sorb)/(batch, sorb) -> comb_x: (batch, ncomb, sorb)/(ncomb, sorb)
    2. Compute matrix element <x|H|x'> (1, ncomb)/(batch, ncomb)
    3. psi(x), psi(comb_x)[ncomb] using NAQS,
       meanwhile use WaveFunction LookUp-Table coming from sampling.
    4. calculate the local energy

    Return:
        eloc[Tensor]: local energy(nbatch)
        psi[Tensor]: psi(x1) 1D(nbatch)
        times:[List[Float]]:
            t1: Singles-Doubles excitation and uint8 -> double
            t2: matrix element <x|H|x'>
            t3: psi(x)
    """
    check_para(x)

    dim: int = x.dim()
    assert dim == 2
    use_LUT: bool = True if WF_LUT is not None else False
    batch: int = x.shape[0]
    t0 = time.time_ns()
    device = h1e.device

    if use_unique:
        # x1: [n_unique, sorb], comb_x: [batch, comb, bra_len]
        comb_x, _ = get_comb_tensor(x, sorb, nele, noa, nob, False)
        bra_len: int = comb_x.shape[2]
    else:
        # x1: [batch * comb, sorb], comb_x: [batch, comb, bra_len]
        comb_x, x1 = get_comb_tensor(x, sorb, nele, noa, nob, True)
        x1 = x1.reshape(-1, sorb)
        bra_len = comb_x.shape[2]

    # calculate matrix <x|H|x'>
    t1 = time.time_ns()
    comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele)  # shape (1, comb)/(batch, comb)

    t2 = time.time_ns()
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True, profile_memory=True) as prof:
    # TODO: torch.unique comb_x is faster, but convert -> -1/1 or 0/1 maybe is not order
    # so, fully testing.
    # FIXME: What time remove duplicate onstate, memory consuming,
    # and has been implemented in wavefunction ansatz,
    # if testing, use keyword: 'use_unique = False/True'.
    with torch.no_grad():
        if comb_x.numel() != 0:
            if use_LUT:
                batch_before_lut = batch * comb_x.size(1)  # batch * comb
                lut_idx, lut_not_idx, lut_value = WF_LUT.lookup(comb_x.reshape(-1, bra_len))
            if use_unique:
                if use_LUT:
                    comb_x = comb_x.reshape(-1, bra_len)[lut_not_idx]
                else:
                    comb_x = comb_x.reshape(-1, bra_len)
                unique_comb, inverse = torch.unique(
                    comb_x, dim=0, return_inverse=True
                )
                x1 = onv_to_tensor(unique_comb, sorb)  # x1: [n_unique, sorb]
                psi0 = torch.index_select(ansatz(x1), 0, inverse) # [n_unique]
            else:
                if use_LUT:
                    x1 = x1[lut_not_idx]
                psi0 = ansatz(x1)  # [batch * comb]

            if use_LUT:
                psi = torch.empty(batch_before_lut, device=device, dtype=psi0.dtype)
                psi[lut_idx] = lut_value.to(psi0.dtype)
                psi[lut_not_idx] = psi0
                psi_x1 = psi.reshape(batch, -1)
            else:
                psi_x1 = psi0.reshape(batch, -1)
        else:
            comb = comb_hij.size(1)
            psi_x1 = torch.zeros(batch, comb, device=device, dtype=dtype)

    if x.is_cuda:
        torch.cuda.synchronize(device)
    t3 = time.time_ns()

    if batch == 1:
        eloc = torch.sum(comb_hij * psi_x1 / psi_x1[..., 0])  # scalar
    else:
        eloc = torch.sum(torch.div(psi_x1.T, psi_x1[..., 0]).T * comb_hij, -1)  # (batch)

    delta0 = (t1 - t0) / 1.0e06
    delta1 = (t2 - t1) / 1.0e06
    delta2 = (t3 - t2) / 1.0e06
    logger.debug(
        f"comb_x/uint8_to_bit time: {delta0:.3E} ms, <i|H|j> time: {delta1:.3E} ms, "
        + f"nqs time: {delta2:.3E} ms"
    )
    del comb_hij, comb_x  # index, unique_x1, unique

    if x.is_cuda:
        torch.cuda.empty_cache()
    return eloc.to(dtype), psi_x1[..., 0].to(dtype), (delta0, delta1, delta2)
