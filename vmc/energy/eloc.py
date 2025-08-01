from __future__ import annotations

import time
import torch

from functools import partial
from typing import Callable, Tuple, List, Union, Optional
from loguru import logger
from torch import Tensor, nn

from libs.C_extension import get_hij_torch, get_comb_tensor, onv_to_tensor
from utils.public_function import WavefunctionLUT, get_Num_SinglesDoubles, check_para

from .flip import Func, _reduce_psi_flip, _simple_flip, _only_sample_space_flip
# from torch.profiler import profile, record_function, ProfilerActivity

FUSED_HIJ = True
try:
    from libs.C_extension import get_comb_hij_fused
except ImportError:
    FUSED_HIJ = False

def local_energy(
    x: Tensor,
    h1e: Tensor,
    h2e: Tensor,
    ansatz: nn.Module | Callable[[Tensor], Tensor],
    ansatz_batch: Callable[..., Tensor],
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    dtype=torch.double,
    use_spin_raising: bool = False,
    h1e_spin: Optional[Tensor] = None,
    h2e_spin: Optional[Tensor] = None,
    WF_LUT: Optional[WavefunctionLUT] = None,
    use_unique: bool = True,
    reduce_psi: bool = False,
    eps: float = 1e-12,
    eps_sample: int = 0,
    use_sample_space: bool = False,
    index: Optional[Tuple[int, int]] = None,
    alpha: float = 2,
    use_multi_psi: bool = False,
    use_spin_flip: bool = False,
    extra_norm: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, tuple[float, float, float]]:
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
            # if use_multi_psi:
            #     raise NotImplementedError
            if use_spin_flip:
                func = _only_sample_space_flip
            else:
                func = _only_sample_space
            func = partial(
                func,
                index=index,
                alpha=alpha,
            )
        else:
            if reduce_psi:
                assert eps >= 0.0 and eps_sample >= 0
                if use_spin_flip:
                    func = _reduce_psi_flip
                else:
                    func = _reduce_psi
                func = partial(
                    func,
                    eps=eps,
                    eps_sample=eps_sample,
                )
            else:
                if use_spin_flip:
                    func = _simple_flip
                else:
                    func = _simple
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #              record_shapes=True, profile_memory=True) as prof:
        #     value = func(x, h1e, h2e, ansatz, sorb, nele, noa, nob, dtype, WF_LUT, use_unique, eps)
        # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=20))
        # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
        # return value

        return func(
            x,
            h1e,
            h2e,
            ansatz,
            ansatz_batch,
            sorb,
            nele,
            noa,
            nob,
            dtype,
            use_spin_raising,
            h1e_spin,
            h2e_spin,
            WF_LUT,
            use_unique,
            use_multi_psi,
            extra_norm,
        )

def _simple(
    x: Tensor,
    h1e: Tensor,
    h2e: Tensor,
    ansatz: Callable[..., Tensor],
    ansatz_batch: Callable[[Callable], Tensor],
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    dtype=torch.double,
    use_spin_raising: bool = False,
    h1e_spin: Optional[Tensor] = None,
    h2e_spin: Optional[Tensor] = None,
    WF_LUT: Optional[WavefunctionLUT] = None,
    use_unique: bool = True,
    use_multi_psi: bool = False,
    extra_norm: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, tuple[float, float, float]]:
    check_para(x)

    dim: int = x.dim()
    assert dim == 2
    batch: int = x.shape[0]
    t0 = time.time_ns()

    if use_multi_psi:
        ansatz_extra = partial(ansatz_batch, func=ansatz.module.extra)
        ansatz = partial(ansatz_batch, func=ansatz.module.sample)
    else:
        ansatz = partial(ansatz_batch, func=ansatz)

    # x1: [batch * comb, sorb], comb_x: [batch, comb, bra_len]
    comb_x, _ = get_comb_tensor(x, sorb, nele, noa, nob, False)
    bra_len = comb_x.shape[2]

    t1 = time.time_ns()
    # calculate matrix <x|H|x'>, [batch, comb]
    comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele)
    if use_spin_raising:
        hij_spin = get_hij_torch(x, comb_x, h1e_spin, h2e_spin, sorb, nele)

    t2 = time.time_ns()

    psi_x1 = Func(ansatz, comb_x.reshape(-1, bra_len), WF_LUT, use_unique).reshape(batch, -1)
    if use_multi_psi:
        f = Func(ansatz_extra, comb_x.reshape(-1, bra_len), None, use_unique).reshape(batch, -1)
        f_psi = psi_x1 * f * f[..., 0].reshape(-1, 1).conj() / extra_norm**2  # [nbatch, nSD]
    else:
        f_psi = psi_x1

    comb_hij = comb_hij.to(dtype.to_real())
    eloc = ((f_psi.T / psi_x1[..., 0]).T * comb_hij).sum(-1)
    if use_spin_raising:
        hij_spin = hij_spin.to(dtype.to_real())
        sloc = ((f_psi.T / psi_x1[..., 0]).T * hij_spin).sum(-1)
    else:
        sloc = torch.zeros_like(eloc)

    t3 = time.time_ns()

    delta0 = (t1 - t0) / 1.0e06
    delta1 = (t2 - t1) / 1.0e06
    delta2 = (t3 - t2) / 1.0e06
    logger.debug(
        f"comb_x/uint8_to_bit time: {delta0:.3E} ms, <i|H|j> time: {delta1:.3E} ms, " + f"nqs time: {delta2:.3E} ms"
    )

    return eloc.to(dtype), sloc.to(dtype), psi_x1[..., 0].to(dtype), (delta0, delta1, delta2)


def _reduce_psi(
    x: Tensor,
    h1e: Tensor,
    h2e: Tensor,
    ansatz: Callable[..., Tensor],
    ansatz_batch: Callable[[Callable], Tensor],
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    dtype=torch.double,
    use_spin_raising: bool = False,
    h1e_spin: Optional[Tensor] = None,
    h2e_spin: Optional[Tensor] = None,
    WF_LUT: Optional[WavefunctionLUT] = None,
    use_unique: bool = True,
    use_multi_psi: bool = False,
    extra_norm: Optional[Tensor] = None,
    eps: float = 1.0e-12,
    eps_sample: int = 0,
) -> tuple[Tensor, Tensor, Tensor, tuple[float, float, float]]:
    """
    E_loc(x) = \sum_x' psi(x')/psi(x) * <x|H|x'>
    ignore x' when <x|H|x'>/psi(x) < 1e-12
    """
    check_para(x)
    dim: int = x.dim()
    assert dim == 2
    t0 = time.time_ns()
    device = h1e.device

    if use_multi_psi:
        ansatz_extra = partial(ansatz_batch, func=ansatz.module.extra)
        ansatz = partial(ansatz_batch, func=ansatz.module.sample)
    else:
        ansatz = partial(ansatz_batch, func=ansatz)

    # comb_x: (batch, comb, bra_len)
    comb_x = get_comb_tensor(x, sorb, nele, noa, nob, False)[0]
    # x0 = onv_to_tensor(x, sorb).reshape(1, -1)
    batch, n_comb, bra_len = tuple(comb_x.size())

    # calculate matrix <x|H|x'>
    t1 = time.time_ns()

    if use_spin_raising:
        hij_spin = get_hij_torch(x, comb_x, h1e_spin, h2e_spin, sorb, nele)
    comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele)

    t2 = time.time_ns()

    # n_sample = 1000
    stochastic = True if eps_sample > 0 else False
    semi_stochastic = True if eps > 0.0 else False
    if stochastic:
        if semi_stochastic:
            hij_abs = comb_hij.abs()
            _mask = hij_abs >= eps
            _index = torch.where(_mask.flatten())[0]
            hij = torch.where(_mask, 0, hij_abs)
        else:
            hij = comb_hij.abs()
        # sampling from p(m) , p(m) \propto |Hnm|
        # 1/N \sum_m' H[n,m'] psi[m'] / p[m']
        _prob = hij / hij.sum(1, keepdim=True)
        # (batch, n_Sample)
        _counts = torch.multinomial(_prob, eps_sample, replacement=True)
        # add index
        _counts += torch.arange(batch, device=device).reshape(-1, 1) * n_comb
        # unique counts
        _index1, _count = _counts.unique(sorted=True, return_counts=True)
        # H[n, m]/p[m'] N_m/N_sample
        _prob = _prob.flatten()
        comb_hij.view(-1)[_index1] = (_count / eps_sample) * comb_hij.flatten()[_index1] / _prob[_index1]
        # if use_spin_raising:
        #     hij_spin.view(-1)[_index1] = (_count / N_SAMPLE) * hij_spin.flatten()[_index1] / _prob[_index1]
        gt_eps_idx = _index1
        if semi_stochastic:
            gt_eps_idx = torch.cat([_index, _index1])
        del hij, _prob, _count, _counts
    else:
        # ignore x' when |<x|H|x'>| < eps
        gt_eps_idx = torch.where(comb_hij.reshape(-1).abs() >= eps)[0]

    rate = gt_eps_idx.size(0) / comb_hij.reshape(-1).size(0) * 100
    s = f"N-sample: {eps_sample}, STOCHASTIC: {stochastic}, SEMI_STOCHASTIC: {semi_stochastic}, "
    s += f"reduce rate: {comb_hij.reshape(-1).size(0)} -> {gt_eps_idx.size(0)}, {rate:.2f} %"
    logger.debug(s)

    x = comb_x.reshape(-1, bra_len)[gt_eps_idx]
    psi_x1 = torch.zeros(batch * n_comb, dtype=dtype, device=device)
    psi_x1[gt_eps_idx] = Func(ansatz, x, WF_LUT, use_unique)
    psi_x1 = psi_x1.reshape(batch, n_comb)
    if use_multi_psi:
        f = torch.zeros(batch * n_comb, dtype=dtype, device=device)
        f[gt_eps_idx] = Func(ansatz_extra, x, None, use_unique).to(dtype)
        f = f.reshape(batch, n_comb)
        f_psi = psi_x1 * f * f[..., 0].reshape(-1, 1).conj() / extra_norm**2  # [nbatch, nSD]
    else:
        f_psi = psi_x1

    comb_hij = comb_hij.to(dtype.to_real())
    eloc = ((f_psi.T / psi_x1[..., 0]).T * comb_hij).sum(-1)
    if use_spin_raising:
        hij_spin = hij_spin.to(dtype.to_real())
        sloc = ((f_psi.T / psi_x1[..., 0]).T * hij_spin).sum(-1)
    else:
        sloc = torch.zeros_like(eloc)

    t3 = time.time_ns()
    delta0 = (t1 - t0) / 1.0e06
    delta1 = (t2 - t1) / 1.0e06
    delta2 = (t3 - t2) / 1.0e06
    logger.debug(
        f"comb_x/uint8_to_bit time: {delta0:.3E} ms, <i|H|j> time: {delta1:.3E} ms, " + f"nqs time: {delta2:.3E} ms"
    )
    del comb_hij, comb_x, gt_eps_idx

    return eloc.to(dtype), sloc.to(dtype), psi_x1[..., 0].to(dtype), (delta0, delta1, delta2)


def _only_sample_space(
    x: Tensor,
    h1e: Tensor,
    h2e: Tensor,
    ansatz: nn.Module | Callable[[Tensor], Tensor],
    ansatz_batch: Callable[[Callable], Tensor],
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    dtype=torch.double,
    use_spin_raising: bool = False,
    h1e_spin: Optional[Tensor] = None,
    h2e_spin: Optional[Tensor] = None,
    WF_LUT: Optional[WavefunctionLUT] = None,
    use_unique: bool = True,
    # eps: float = 1.0e-12,
    use_multi_psi: bool = False,
    extra_norm: Optional[Tensor] = None,
    index: tuple[int, int] = None,
    alpha: float = 2,
) -> tuple[Tensor, Tensor, Tensor, tuple[float, float, float]]:
    check_para(x)

    device = x.device
    dim: int = x.dim()
    assert dim == 2
    t0 = time.time_ns()

    batch = x.size(0)
    nSD = get_Num_SinglesDoubles(sorb, noa, nob) + 1
    n_sample = WF_LUT.bra_key.size(0)

    # XXX: reduce memory usage
    # memory usage: batch * n_comb_sd * (sorb - 1/64 + 1) / 8 / 2**20 MiB
    # maybe n_comb_sd * batch <= n_sample maybe be better
    is_complex: bool = dtype.is_complex
    _len = (sorb - 1) // 64 + 1
    alpha = max(alpha, 1)
    sd_le_sample = nSD * (2 + is_complex + _len) * alpha <= n_sample
    # sd_le_sample = False

    t0 = time.time_ns()
    if FUSED_HIJ:
        comb_x, comb_hij = get_comb_hij_fused(x, h1e, h2e, sorb, nele, noa, nob)
    else:
        comb_x = get_comb_tensor(x, sorb, nele, noa, nob, False)[0]

    t1 = time.time_ns()
    if not FUSED_HIJ:
        comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele)
    if use_spin_raising:
        hij_spin = get_hij_torch(x, comb_x, h1e_spin, h2e_spin, sorb, nele)
    t2 = time.time_ns()

    bra_len = comb_x.size(2)
    x1 = comb_x.reshape(-1, bra_len)
    psi_x1 = torch.zeros(batch, nSD, device=device, dtype=WF_LUT.dtype)
    idx, _, value = WF_LUT.lookup(x1)
    psi_x1.view(-1)[idx] = value

    if use_multi_psi:
        ansatz_extra = partial(ansatz_batch, func=ansatz.module.extra)
        f = torch.zeros_like(psi_x1)
        f.view(-1)[idx] = Func(ansatz_extra, x1[idx], None, True)
        f_psi = psi_x1 * f * f[..., 0].reshape(-1, 1).conj() / extra_norm**2  # [nbatch, nSD]
    else:
        # <x|H|x'>psi(x')/psi(x)
        f_psi = psi_x1

    comb_hij = comb_hij.to(dtype.to_real())
    eloc = ((f_psi.T / psi_x1[..., 0]).T * comb_hij).sum(-1)
    if use_spin_raising:
        hij_spin = hij_spin.to(dtype.to_real())
        sloc = ((f_psi.T / psi_x1[..., 0]).T * hij_spin).sum(-1)
    else:
        sloc = torch.zeros_like(eloc)
    # eloc = ((f_psi.T / psi_x1[..., 0]).T * comb_hij).sum(-1)
    # if use_spin_raising:
    #     sloc = ((f_psi.T / psi_x1[..., 0]).T * hij_spin).sum(-1)
    # else:
        
    # if use_multi_psi:
    #     raise NotImplementedError(f"Not implement in multi-psi")
    #     ansatz_extra = partial(ansatz_batch, func=ansatz.module.extra)

    # if sd_le_sample:
    #     # (batch, n_comb_sd, bra_len)
    #     comb_x = get_comb_tensor(x, sorb, nele, noa, nob, False)[0]
    # else:
    #     # (n_sample, bra_len)
    #     comb_x = WF_LUT.bra_key

    # t1 = time.time_ns()
    # # (batch, n_comb_sd) or (batch, n_sample)

    # if use_spin_raising:
    #     hij_spin = get_hij_torch(x, comb_x, h1e_spin, h2e_spin, sorb, nele)
    # comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele)

    # t2 = time.time_ns()
    # if sd_le_sample:
    #     bra_len = comb_x.size(2)
    #     psi_x1 = torch.zeros(batch * n_comb_sd, device=device, dtype=WF_LUT.dtype)
    #     lut_idx, lut_not_idx, lut_value = WF_LUT.lookup(comb_x.reshape(-1, bra_len))
    #     psi_x1[lut_idx] = lut_value
    #     psi_x1 = psi_x1.reshape(batch, n_comb_sd)

    #     # <x|H|x'>psi(x')/psi(x)
    #     # T1 = time.time_ns()
    #     # psi_x = psi_x1[..., 0].view(-1)
    #     # eloc1 = torch.sum(torch.div(psi_x1.T, psi_x).T * comb_hij, -1)  # (batch)
    #     # torch.cuda.synchronize()
    #     # T2 = time.time_ns()

    #     psi_x = psi_x1[..., 0].view(-1).clone()

    #     if use_multi_psi:
    #         x1 = onv_to_tensor(comb_x.reshape(-1, bra_len), sorb)
    #         _psi = ansatz_extra(x1)
    #         # f(n).conj() Hnm * f(m) / extra_norm**2
    #         _psi = _psi.reshape(batch, -1)
    #         value = _psi * (_psi[:, 0].reshape(-1, 1).conj() / extra_norm**2)
    #         comb_hij = comb_hij * value
    #         if use_spin_raising:
    #             hij_spin = hij_spin * value

    #     if use_spin_raising:
    #         sloc = psi_x1.mul(hij_spin).sum(-1).divide(psi_x)
    #     eloc = psi_x1.mul_(comb_hij).sum(-1).divide_(psi_x)  # (nbatch)

    #     # torch.cuda.synchronize()
    #     # T3 = time.time_ns()
    #     # print(f"{(T2-T1)/1.0e6:.5f} ms, {(T3-T2)/1.0e6:.5f} ms")
    #     # print(torch.allclose(eloc, eloc1))

    #     # breakpoint()
    # else:
    #     sample_value = WF_LUT.wf_value
    #     psi_x = WF_LUT.index_value(*index)
    #     # not_idx, psi_x1 = WF_LUT.lookup(x)[1:]
    #     # assert torch.allclose(psi_x1, psi_x1)
    #     # WF_LUT coming from sampling x must been found in WF_LUT.
    #     # assert not_idx.size(0) == 0

    #     if WF_LUT.dtype == torch.complex128:
    #         value = torch.empty(batch * 2, device=device, dtype=torch.double)
    #         value[0::2] = torch.matmul(comb_hij, sample_value.real)  # Real-part
    #         value[1::2] = torch.matmul(comb_hij, sample_value.imag)  # Imag-part
    #         eloc = torch.view_as_complex(value.view(-1, 2)).div(psi_x)

    #         if use_spin_raising:
    #             value_spin = torch.empty(batch * 2, device=device, dtype=torch.double)
    #             value_spin[0::2] = torch.matmul(hij_spin, sample_value.real)  # Real-part
    #             value_spin[1::2] = torch.matmul(hij_spin, sample_value.imag)  # Imag-part
    #             sloc = torch.view_as_complex(value_spin.view(-1, 2)).div(psi_x)

    #     elif WF_LUT.dtype == torch.double:
    #         eloc = torch.matmul(comb_hij, sample_value).div(psi_x)
    #         # eloc = torch.einsum("ij, j, i ->i", comb_hij, sample_value, 1 / psi_x)

    #         if use_spin_raising:
    #             sloc = torch.matmul(hij_spin, sample_value).div(psi_x)
    #     else:
    #         raise NotImplementedError(f"Single/Complex-Single does not been supported")

    t3 = time.time_ns()
    delta0 = (t1 - t0) / 1.0e06
    delta1 = (t2 - t1) / 1.0e06
    delta2 = (t3 - t2) / 1.0e06
    logger.debug(
        f"comb_x/uint8_to_bit time: {delta0:.3E} ms, <i|H|j> time: {delta1:.3E} ms, " + f"nqs time: {delta2:.3E} ms"
    )

    del comb_hij

    if use_spin_raising:
        del hij_spin

    if not use_spin_raising:
        sloc = torch.zeros_like(eloc)

    return eloc.to(dtype), sloc.to(dtype), psi_x1[..., 0].to(dtype), (delta0, delta1, delta2)
