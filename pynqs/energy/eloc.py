from __future__ import annotations

import time
import torch

from functools import partial
from typing import Tuple, List, Union, Optional
from collections.abc import Callable
from loguru import logger
from torch import Tensor

from pynqs.libs.C_extension import get_hij_torch, get_comb_tensor, unpackbits, packbits
from pynqs.utils.compile import _unwrap_compiled_module
from pynqs.utils.lut import WavefunctionLUT
from pynqs.utils.public_function import get_Num_SinglesDoubles, check_para
from pynqs.config import cuda_synchronize
from pynqs.ansatz import Excitedwavefunctions
from pynqs.utils.public_function import ansatz_batch as ab

from .flip import Func, _reduce_psi_flip, _simple_flip, _only_sample_space_flip

# from torch.profiler import profile, record_function, ProfilerActivity

FUSED_HIJ = True
try:
    from pynqs.libs.C_extension import get_comb_hij_fused
except ImportError:
    FUSED_HIJ = False


def local_energy(
    x: Tensor,
    h1e: Tensor,
    h2e: Tensor,
    ansatz: Callable[[Tensor], Tensor],
    ansatz_batch: Callable[..., Tensor],
    sorb: int,
    nele: int,
    noa: int,
    nob: int,
    dtype=torch.double,
    fp_batch=-1,
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
    r"""
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
        if isinstance(_unwrap_compiled_module(ansatz), Excitedwavefunctions):  # Excited NQS
            return excited_eloc(
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
                fp_batch,
                func,
            )
        else:  # Ground NQS
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
    cuda_synchronize()
    t0 = time.time_ns()

    if use_multi_psi:
        ansatz_extra = partial(ansatz_batch, func=ansatz.extra)
        ansatz = partial(ansatz_batch, func=ansatz.sample)
    else:
        ansatz = partial(ansatz_batch, func=ansatz)

    if FUSED_HIJ:
        comb_x, comb_hij = get_comb_hij_fused(x, h1e, h2e, sorb, nele, noa, nob)
    else:
        # x1: [batch * comb, sorb], comb_x: [batch, comb, bra_len]
        comb_x, _ = get_comb_tensor(x, sorb, nele, noa, nob, False)
    bra_len = comb_x.shape[2]

    cuda_synchronize()
    t1 = time.time_ns()
    if not FUSED_HIJ:
        # calculate matrix <x|H|x'>, [batch, comb]
        comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele)
    if use_spin_raising:
        hij_spin = get_hij_torch(x, comb_x, h1e_spin, h2e_spin, sorb, nele)

    cuda_synchronize()
    t2 = time.time_ns()

    psi_x1 = Func(ansatz, comb_x.reshape(-1, bra_len), WF_LUT, use_unique).reshape(batch, -1)
    if use_multi_psi:
        f = Func(ansatz_extra, comb_x.reshape(-1, bra_len), None, use_unique).reshape(batch, -1)
        f_psi = psi_x1 * f * f[..., 0].reshape(-1, 1).conj() / extra_norm**2  # [nbatch, nSD]
    else:
        f_psi = psi_x1

    eloc = ((f_psi.T / psi_x1[..., 0]).T * comb_hij).sum(-1)
    if use_spin_raising:
        hij_spin = hij_spin.to(dtype.to_real())
        sloc = ((f_psi.T / psi_x1[..., 0]).T * hij_spin).sum(-1)
    else:
        sloc = torch.zeros_like(eloc)

    cuda_synchronize()
    t3 = time.time_ns()

    delta0 = (t1 - t0) / 1.0e06
    delta1 = (t2 - t1) / 1.0e06
    delta2 = (t3 - t2) / 1.0e06
    logger.debug(
        f"comb_x/uint8_to_bit time: {delta0:.3E} ms, <i|H|j> time: {delta1:.3E} ms, "
        + f"nqs time: {delta2:.3E} ms"
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
    r"""
    E_loc(x) = \sum_x' psi(x')/psi(x) * <x|H|x'>
    ignore x' when <x|H|x'>/psi(x) < 1e-12
    """
    check_para(x)
    dim: int = x.dim()
    assert dim == 2

    cuda_synchronize()
    t0 = time.time_ns()
    device = h1e.device

    if use_multi_psi:
        ansatz_extra = partial(ansatz_batch, func=ansatz.extra)
        ansatz = partial(ansatz_batch, func=ansatz.sample)
    else:
        ansatz = partial(ansatz_batch, func=ansatz)

    if FUSED_HIJ:
        # comb_x: (batch, comb, bra_len)
        comb_x, comb_hij = get_comb_hij_fused(x, h1e, h2e, sorb, nele, noa, nob)
    else:
        comb_x = get_comb_tensor(x, sorb, nele, noa, nob, False)[0]
    batch, n_comb, bra_len = tuple(comb_x.size())

    cuda_synchronize()
    # calculate matrix <x|H|x'>
    t1 = time.time_ns()
    if not FUSED_HIJ:
        comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele)
    if use_spin_raising:
        hij_spin = get_hij_torch(x, comb_x, h1e_spin, h2e_spin, sorb, nele)

    cuda_synchronize()
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
        comb_hij[..., 0] += eps  # ensure x is always selected when using S-S+
        gt_eps_idx = torch.where(comb_hij.reshape(-1).abs() >= eps)[0]
        comb_hij[..., 0] -= eps

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

    cuda_synchronize()
    t3 = time.time_ns()
    delta0 = (t1 - t0) / 1.0e06
    delta1 = (t2 - t1) / 1.0e06
    delta2 = (t3 - t2) / 1.0e06
    logger.debug(
        f"comb_x/uint8_to_bit time: {delta0:.3E} ms, <i|H|j> time: {delta1:.3E} ms, "
        + f"nqs time: {delta2:.3E} ms"
    )
    del comb_hij, comb_x, gt_eps_idx

    return eloc.to(dtype), sloc.to(dtype), psi_x1[..., 0].to(dtype), (delta0, delta1, delta2)


def _only_sample_space(
    x: Tensor,
    h1e: Tensor,
    h2e: Tensor,
    ansatz: Callable[[Tensor], Tensor],
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
    cuda_synchronize()
    t0 = time.time_ns()
    if FUSED_HIJ:
        comb_x, comb_hij = get_comb_hij_fused(x, h1e, h2e, sorb, nele, noa, nob)
    else:
        comb_x = get_comb_tensor(x, sorb, nele, noa, nob, False)[0]

    cuda_synchronize()
    t1 = time.time_ns()
    if not FUSED_HIJ:
        comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele)
    if use_spin_raising:
        hij_spin = get_hij_torch(x, comb_x, h1e_spin, h2e_spin, sorb, nele)

    cuda_synchronize()
    t2 = time.time_ns()

    bra_len = comb_x.size(2)
    x1 = comb_x.reshape(-1, bra_len)
    psi_x1 = torch.zeros(batch, nSD, device=device, dtype=WF_LUT.dtype)
    idx, _, value = WF_LUT.lookup(x1)
    psi_x1.view(-1)[idx] = value

    if use_multi_psi:
        ansatz_extra = partial(ansatz_batch, func=ansatz.extra)
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
    #     ansatz_extra = partial(ansatz_batch, func=ansatz.extra)

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
    #         x1 = unpackbits(comb_x.reshape(-1, bra_len), sorb)
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

    cuda_synchronize()
    t3 = time.time_ns()
    delta0 = (t1 - t0) / 1.0e06
    delta1 = (t2 - t1) / 1.0e06
    delta2 = (t3 - t2) / 1.0e06
    logger.debug(
        f"comb_x/uint8_to_bit time: {delta0:.3E} ms, <i|H|j> time: {delta1:.3E} ms, "
        + f"nqs time: {delta2:.3E} ms"
    )

    del comb_hij

    if use_spin_raising:
        del hij_spin

    if not use_spin_raising:
        sloc = torch.zeros_like(eloc)

    return eloc.to(dtype), sloc.to(dtype), psi_x1[..., 0].to(dtype), (delta0, delta1, delta2)


# Excited NQS
def excited_eloc(
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
    fp_batch: int = None,
    func: Callable = None,
) -> tuple[Tensor, Tensor, Tensor, tuple[float, float, float]]:
    check_para(x)

    nbatch: int = x.shape[0]

    cuda_synchronize()
    t0 = time.time_ns()

    def cal_Eloc(ansatz_, ansatz_batch_, x_: Tensor):
        return func(
            x_,
            h1e,
            h2e,
            ansatz_,
            ansatz_batch_,
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

    def cal_HPsi(single_ansatz, x: Tensor, fp_batch: int):
        """
        Returns
        -------
            Matrices
            Psi = ⟨S1|𝜓1⟩ ⟨S1|𝜓2⟩ ... ⟨S1|𝜓K⟩
                  ⟨S2|𝜓1⟩ ⟨S2|𝜓2⟩ ... ⟨S2|𝜓K⟩
                  ...
                  ⟨SK|𝜓1⟩ ⟨SK|𝜓2⟩ ... ⟨SK|𝜓K⟩

            HPsi = ⟨S1|H|𝜓1⟩ ⟨S1|H|𝜓2⟩ ... ⟨S1|H|𝜓K⟩
                   ⟨S2|H|𝜓1⟩ ⟨S2|H|𝜓2⟩ ... ⟨S2|H|𝜓K⟩
                   ...
                   ⟨SK|H|𝜓1⟩ ⟨SK|H|𝜓2⟩ ... ⟨SK|H|𝜓K⟩

            And the matrix (S-S+)Psi is computed in the same way (replace H by S-S+).
        """
        K = len(single_ansatz)
        x = unpackbits(x, sorb).view(-1, K, sorb)
        Scols, Hcols, cols = [], [], []
        for j in range(0, K, 1):
            ansatz_ = single_ansatz[j]
            device = ansatz_.device
            dtype = ansatz_.dtype

            def _ansatz_batch(x: Tensor, func: Callable[[Tensor], Tensor]) -> Tensor:
                return ab(func, x, fp_batch, sorb, device, dtype)

            Scols_, Hcols_, cols_ = [], [], []
            for k in range(0, K, 1):
                eloc_single, sloc_single, psi_single = cal_Eloc(
                    ansatz_, _ansatz_batch, packbits(x[:, k, :].to(torch.uint8), sorb)
                )[:3]
                cols_.append(psi_single)
                Hcols_.append(eloc_single * psi_single)
                Scols_.append(sloc_single * psi_single)
            cols.append(torch.stack(cols_, dim=1))
            Hcols.append(torch.stack(Hcols_, dim=1))
            Scols.append(torch.stack(Scols_, dim=1))
        Psi = torch.stack(cols, dim=2)  # (nbatch, K, K)
        HPsi = torch.stack(Hcols, dim=2)  # (nbatch, K, K)
        SPsi = torch.stack(Scols, dim=2)  # (nbatch, K, K)
        return Psi, HPsi, SPsi

    ansatz_single = ansatz.single_ansatz
    # K = len(ansatz_single)
    # Psi = cal_Psi(ansatz_single, x, fp_batch) # (nbatch, K, K)
    Psi, HPsi, SPsi = cal_HPsi(ansatz_single, x, fp_batch)  # (nbatch, K, K)
    # wf = torch.det(Psi)
    sign, vals = torch.linalg.slogdet(Psi)
    wf = sign * torch.exp(vals)
    cuda_synchronize()
    t1 = time.time_ns()

    # X(n) = Psi^{-1}HPsi
    Psi_inv = torch.linalg.inv(Psi)
    eloc = torch.einsum("nij,njk->nik", Psi_inv, HPsi)
    sloc = torch.einsum("nij,njk->nik", Psi_inv, SPsi)

    # or Solve Psi X(n) = H Psi
    # eloc = torch.linalg.solve(Psi, HPsi).to(ansatz.device)
    # sloc = torch.linalg.solve(Psi, SPsi).to(ansatz.device)

    return eloc.to(dtype), sloc.to(dtype), wf.to(dtype), (0.0, 0.0, (t1 - t0) / 1.0e06)
