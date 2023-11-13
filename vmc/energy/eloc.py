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
from utils.public_function import WavefunctionLUT, get_Num_SinglesDoubles

print = partial(print, flush=True)


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
    reduce_psi: bool = False,
    eps: float = 1e-12,
    use_sample_space: bool = False,
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

    WF_LUT(WavefunctionLUT): WaveFunction lookup-table to seep-up calculations, default: None
    use_unique(bool): remove duplicate state and this may be time-consuming. default: True
    reduce_psi(bool): ignore x' when <x|H|x'>/psi(x) < eps, default: False
    eps(float): default: 1e-12
    use_sample_space(bool): use unique sample as x' not SD. default: False

    Notice:
    'reduce_psi' only applies when psi(x)^2 is normalization in FCI-space

    Return:
        eloc[Tensor]: local energy(nbatch)
        psi[Tensor]: psi(x1) 1D(nbatch)
        times:[List[Float]]:
            t1: Singles-Doubles excitation and uint8 -> double
            t2: matrix element <x|H|x'>
            t3: psi(x)
    """
    with torch.no_grad():
        if use_sample_space:
            assert WF_LUT is not None, "WF_ULT must be used if use_sample"
            func = _only_sample_space
        else:
            if reduce_psi and eps > 0.0:
                func = _reduce_psi
            else:
                func = _simple
        return func(x, h1e, h2e, ansatz, sorb, nele, noa, nob, dtype, WF_LUT, use_unique, eps)


# TODO: how to save unique x eloc energy
def _simple(
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
    eps: float = 1.0e-12,
) -> Tuple[Tensor, Tensor, Tuple[float, float, float]]:
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
    if comb_x.numel() != 0:
        if use_LUT:
            batch_before_lut = batch * comb_x.size(1)  # batch * comb
            lut_idx, lut_not_idx, lut_value = WF_LUT.lookup(comb_x.reshape(-1, bra_len))
        if use_unique:
            if use_LUT:
                comb_x = comb_x.reshape(-1, bra_len)[lut_not_idx]
            else:
                comb_x = comb_x.reshape(-1, bra_len)
            unique_comb, inverse = torch.unique(comb_x, dim=0, return_inverse=True)
            x1 = onv_to_tensor(unique_comb, sorb)  # x1: [n_unique, sorb]
            psi0 = torch.index_select(ansatz(x1), 0, inverse)  # [n_unique]
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

    # if x.is_cuda:
    #     torch.cuda.empty_cache()
    return eloc.to(dtype), psi_x1[..., 0].to(dtype), (delta0, delta1, delta2)


def _reduce_psi(
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
    eps: float = 1.0e-12,
) -> Tuple[Tensor, Tensor, Tuple[float, float, float]]:
    """
    E_loc(x) = \sum_x' psi(x')/psi(x) * <x|H|x'>
    ignore x' when <x|H|x'>/psi(x) < 1e-12
    """
    check_para(x)
    dim: int = x.dim()
    assert dim == 2
    use_LUT: bool = True if WF_LUT is not None else False
    t0 = time.time_ns()
    device = h1e.device

    # comb_x: (batch, comb, bra_len), x1: (batch, comb, sorb)
    if use_unique:
        comb_x = get_comb_tensor(x, sorb, nele, noa, nob, False)[0]
        x0 = onv_to_tensor(x, sorb).reshape(1, -1)
    else:
        comb_x, x1 = get_comb_tensor(x, sorb, nele, noa, nob, True)
        x0 = x1[:, 0, :].reshape(1, -1)
    batch, n_comb, bra_len = tuple(comb_x.size())

    # calculate matrix <x|H|x'>
    t1 = time.time_ns()
    comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele)

    t2 = time.time_ns()
    if use_LUT:
        not_idx, psi_x = WF_LUT.lookup(x)[1:]
        # WF_LUT coming from sampling x must been found in WF_LUT.
        assert not_idx.size(0) == 0
        psi_x = psi_x.unsqueeze(1)  # (batch, 1)
    else:
        psi_x = ansatz(x0).unsqueeze(1)  # (batch, 1)

    # ignore x' when <x|H|x'>/psi(x) < eps, other part is zeros
    # auto-broadcast psi_x (batch * comb)
    comb_psi = torch.div(comb_hij, psi_x).flatten()
    gt_eps_idx = torch.where(comb_psi.abs() > eps)[0]
    # logger.debug(f"reduce rate: {comb_psi.size(0)} -> {gt_eps_idx.size(0)}")
    psi_x1 = torch.zeros(batch * n_comb, dtype=dtype, device=device)

    if comb_x.numel() != 0:
        if use_LUT:
            lut_idx, lut_not_idx, lut_value = WF_LUT.lookup(comb_x.reshape(-1, bra_len)[gt_eps_idx])
            # the index of x1 great than eps and not in LUT
            raw_idx = torch.arange(n_comb * batch, device=device)
            gt_not_lut_idx = raw_idx[gt_eps_idx][lut_not_idx]
            # the index of x1 great than eps and in LUT
            gt_in_lut_idx = raw_idx[gt_eps_idx][lut_idx]
        if use_unique:
            if use_LUT:
                comb_x = comb_x.reshape(-1, bra_len)[gt_not_lut_idx]
            else:
                comb_x = comb_x.reshape(-1, bra_len)[gt_eps_idx]
            unique_comb, inverse = torch.unique(comb_x, dim=0, return_inverse=True)
            x1 = onv_to_tensor(unique_comb, sorb)
            psi_gt_eps = torch.index_select(ansatz(x1), 0, inverse)
        else:
            if use_LUT:
                # x1 great than eps and not in LUT
                x1 = x1[gt_not_lut_idx]
            else:
                # x1 great than eps
                x1 = x1[gt_eps_idx]
            psi_gt_eps = ansatz(x1)

        if use_LUT:
            psi_x1[gt_not_lut_idx] = psi_gt_eps.to(dtype)
            psi_x1[gt_in_lut_idx] = lut_value.to(dtype)
        else:
            psi_x1[gt_eps_idx] = psi_gt_eps.to(dtype)
        psi_x1 = psi_x1.reshape(batch, -1)
    else:
        psi_x1 = torch.zeros(batch, n_comb, device=device, dtype=dtype)

    if batch == 1:
        eloc = torch.sum(comb_hij * psi_x1 / psi_x1[..., 0])  # scalar
    else:
        eloc = torch.sum(torch.div(psi_x1.T, psi_x1[..., 0]).T * comb_hij, -1)  # (batch)

    t3 = time.time_ns()
    delta0 = (t1 - t0) / 1.0e06
    delta1 = (t2 - t1) / 1.0e06
    delta2 = (t3 - t2) / 1.0e06
    logger.debug(
        f"comb_x/uint8_to_bit time: {delta0:.3E} ms, <i|H|j> time: {delta1:.3E} ms, "
        + f"nqs time: {delta2:.3E} ms"
    )
    del comb_hij, comb_x, psi_gt_eps, gt_eps_idx, psi_x  # index, unique_x1, unique

    if use_LUT:
        del raw_idx, gt_not_lut_idx, gt_in_lut_idx, lut_idx, lut_not_idx, lut_value

    if use_unique:
        del unique_comb, inverse

    # if x.is_cuda:
    #     torch.cuda.empty_cache()
    return eloc.to(dtype), psi_x1[..., 0].to(dtype), (delta0, delta1, delta2)


def _only_sample_space(
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
    eps: float = 1.0e-12,
) -> tuple[Tensor, Tensor, tuple[float, float, float]]:
    check_para(x)

    device = x.device
    dim: int = x.dim()
    assert dim == 2
    t0 = time.time_ns()

    batch = x.size(0)
    n_comb_sd = get_Num_SinglesDoubles(sorb, noa, nob) + 1
    n_sample = WF_LUT.bra_key.size(0)

    # XXX: reduce memory usage
    # memory usage: batch * n_comb_sd * (sorb - 1/64 + 1) / 8 / 2**20 MiB
    # maybe n_comb_sd * batch <= n_sample maybe be better
    sd_le_sample: bool = n_comb_sd * 0.5 * batch**0.5 <= n_sample

    if sd_le_sample:
        # (batch, n_comb_sd, bra_len)
        comb_x = get_comb_tensor(x, sorb, nele, noa, nob, False)[0]
    else:
        # (n_sample, bra_len)
        comb_x = WF_LUT.bra_key

    t1 = time.time_ns()
    # (batch, n_comb_sd) or (batch, n_sample)
    comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele)

    t2 = time.time_ns()
    if sd_le_sample:
        bra_len = comb_x.size(2)
        psi_x1 = torch.zeros(batch * n_comb_sd, device=device, dtype=dtype)
        lut_idx, lut_not_idx, lut_value = WF_LUT.lookup(comb_x.reshape(-1, bra_len))
        psi_x1[lut_idx] = lut_value
        psi_x1 = psi_x1.reshape(batch, n_comb_sd)

        # <x|H|x'>psi(x')/psi(x)
        psi_x = psi_x1[..., 0].view(-1)
        eloc = torch.sum(torch.div(psi_x1.T, psi_x).T * comb_hij, -1)  # (batch)
    else:
        sample_value = WF_LUT.wf_value.to(dtype)
        not_idx, psi_x = WF_LUT.lookup(x)[1:]
        psi_x = psi_x.to(dtype)  # (batch)
        # WF_LUT coming from sampling x must been found in WF_LUT.
        assert not_idx.size(0) == 0

        # <x|H|x'> * psi(x') / psi(x)
        # XXX: how to opt the path of einsum, reduce memory use
        # comb_hij is real, sample_value and psi_x is real or complex
        eloc = torch.sum(
            comb_hij * torch.div(sample_value.expand(batch, n_sample).T, psi_x).T, dim=-1
        )
        # eloc = torch.einsum("ij, j, i ->i", comb_hij.to(dtype), sample_value, 1 / psi_x)

    t3 = time.time_ns()
    delta0 = (t1 - t0) / 1.0e06
    delta1 = (t2 - t1) / 1.0e06
    delta2 = (t3 - t2) / 1.0e06
    logger.debug(
        f"comb_x/uint8_to_bit time: {delta0:.3E} ms, <i|H|j> time: {delta1:.3E} ms, "
        + f"nqs time: {delta2:.3E} ms"
    )

    del comb_hij
    return eloc.to(dtype), psi_x, (delta0, delta1, delta2)
