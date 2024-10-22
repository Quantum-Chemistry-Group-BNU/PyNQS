from __future__ import annotations

import time
import torch

from functools import partial
from typing import Callable, Tuple, List, Union, Optional
from loguru import logger
from torch import Tensor, nn

from libs.C_extension import get_hij_torch, get_comb_tensor, onv_to_tensor, tensor_to_onv
from utils.public_function import (
    WavefunctionLUT,
    get_Num_SinglesDoubles,
    check_para,
    spin_flip_onv,
    spin_flip_sign,
)


def Func(
    func: Callable[..., Tensor],
    x: Tensor,
    WF_LUT: Optional[WavefunctionLUT] = None,
    use_unique: bool = False,
) -> Tensor:
    check_para(x)
    use_LUT = True if WF_LUT is not None else False
    batch = x.size(0)

    if use_LUT:
        lut_idx, lut_not_idx, lut_value = WF_LUT.lookup(x)

    if use_unique:
        if use_LUT:
            _x = x[lut_not_idx]
        else:
            _x = x
        unique_x, inverse = torch.unique(_x, dim=0, return_inverse=True)
        psi0 = torch.index_select(func(unique_x), 0, inverse)
    else:
        if use_LUT:
            x = x[lut_not_idx]
        psi0 = func(x)

    if use_LUT:
        psi = torch.empty(batch, dtype=psi0.dtype, device=psi0.device)
        psi[lut_idx] = lut_value.to(psi0.dtype)
        psi[lut_not_idx] = psi0
    else:
        psi = psi0

    return psi


def _simple_flip(
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
    eps: float = 1.0e-12,
    use_multi_psi: bool = False,
    # use_spin_flip: bool = False,
    extra_norm: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, tuple[float, float, float]]:
    check_para(x)

    batch = x.size(0)
    S = 0
    η: int = (-1) ** (nele // 2 - S)

    if use_multi_psi:
        ansatz_extra = partial(ansatz_batch, func=ansatz.module.extra)
        ansatz = partial(ansatz_batch, func=ansatz.module.sample)
    else:
        ansatz = partial(ansatz_batch, func=ansatz)

    t0 = time.time_ns()

    # x1: [batch * comb, sorb], comb_x: [batch, comb, bra_len]
    comb_x, _ = get_comb_tensor(x, sorb, nele, noa, nob, False)
    bra_len = comb_x.shape[2]

    t1 = time.time_ns()
    # calculate matrix <x|H|x'>, [batch, comb]
    comb_hij = get_hij_torch(x, comb_x, h1e, h2e, sorb, nele)
    if use_spin_raising:
        hij_spin = get_hij_torch(x, comb_x, h1e_spin, h2e_spin, sorb, nele)

    t2 = time.time_ns()

    if use_multi_psi:
        η_m = spin_flip_sign(comb_x.reshape(-1, bra_len), sorb).reshape(batch, -1)
        x1_flip = spin_flip_onv(comb_x.reshape(-1, bra_len), sorb)  # [comb * batch, bra_len]
        f_flip = Func(ansatz_extra, x1_flip, None, use_unique).reshape(batch, -1)
        f = Func(ansatz_extra, comb_x.reshape(-1, bra_len), None, use_unique).reshape(batch, -1)
        psi_x1 = Func(ansatz, comb_x.reshape(-1, bra_len), WF_LUT, use_unique).reshape(batch, -1)
        psi_x1_flip = Func(ansatz, x1_flip, WF_LUT, use_unique).reshape(batch, -1)

        f_psi = (f * psi_x1 + η * η_m * f_flip * psi_x1_flip) * f[..., 0].reshape(-1, 1).conj() / extra_norm**2  # [batch, nSD]
    else:
        # [batch, comb]
        psi_x1 = Func(ansatz, comb_x.reshape(-1, bra_len), WF_LUT, use_unique).reshape(batch, -1)
        # spin-flip
        η_m = spin_flip_sign(comb_x.reshape(-1, bra_len), sorb).reshape(batch, -1)
        x1_flip = spin_flip_onv(comb_x.reshape(-1, bra_len), sorb)
        
        psi_x1_flip = Func(ansatz, x1_flip, WF_LUT, use_unique).reshape(batch, -1)

        f_psi = (psi_x1 + η * η_m * psi_x1_flip) / extra_norm**2  # [batch, nSD]

    psi_x0 = psi_x1[..., 0]  # [batch] \phi(n)
    eloc = torch.einsum("ij, ij, i -> i", comb_hij.to(dtype), f_psi, 1 / psi_x0)
    if use_spin_raising:
        sloc = torch.einsum("ij, ij, i -> i", hij_spin.to(dtype), f_psi, 1 / psi_x0)
    else:
        sloc = torch.zeros_like(eloc)

    t3 = time.time_ns()

    delta0 = (t1 - t0) / 1.0e06
    delta1 = (t2 - t1) / 1.0e06
    delta2 = (t3 - t2) / 1.0e06

    return eloc.to(dtype), sloc.to(dtype), psi_x1[..., 0].to(dtype), (delta0, delta1, delta2)


def _reduce_psi_flip(
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
    eps: float = 1.0e-12,
    n_sample: int = 0,
    use_multi_psi: bool = False,
    extra_norm: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, tuple[float, float, float]]:
    """
    E_loc(x) = \sum_x' psi(x')/psi(x) * <x|H|x'>
    ignore x' when <x|H|x'>/psi(x) < 1e-12
    """
    # XXX: This is error
    hij_spin: Tensor = None

    check_para(x)
    dim: int = x.dim()
    assert dim == 2
    use_LUT: bool = True if WF_LUT is not None else False
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
    stochastic = True if n_sample > 0 else False
    semi_stochastic = True if eps > 0.0 else False
    # breakpoint()
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
        _counts = torch.multinomial(_prob, n_sample, replacement=True)
        # add index
        _counts += torch.arange(batch, device=device).reshape(-1, 1) * n_comb
        # unique counts
        _index1, _count = _counts.unique(sorted=True, return_counts=True)
        # H[n, m]/p[m'] N_m/N_sample
        _prob = _prob.flatten()
        comb_hij.view(-1)[_index1] = (_count / n_sample) * comb_hij.flatten()[_index1] / _prob[_index1]
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
    s = f"N-sample: {n_sample}, STOCHASTIC: {stochastic}, SEMI_STOCHASTIC: {semi_stochastic}, "
    s += f"reduce rate: {comb_hij.reshape(-1).size(0)} -> {gt_eps_idx.size(0)}, {rate:.2f} %"
    logger.debug(s)
    
    psi_x1 = torch.zeros(batch * n_comb, dtype=dtype, device=device)
    psi_x1_flip = torch.zeros_like(psi_x1)
    gt_eps_idx = torch.arange(psi_x1.size(0), device=device)
    
    S = 0
    η: int = (-1) ** (nele // 2 - S)
    x = comb_x.reshape(-1, bra_len)[gt_eps_idx]
    breakpoint()
    if use_multi_psi:
        _η_m = spin_flip_sign(x, sorb)
        x1_flip = spin_flip_onv(x, sorb)
        _f = Func(ansatz_extra, x, None, use_unique)
        _f_flip = Func(ansatz_extra, x1_flip, None, use_unique)
        _psi_x1 = Func(ansatz, x, WF_LUT, use_unique)
        _psi_x1_flip = Func(ansatz, x1_flip, WF_LUT, use_unique)

        # index
        f = torch.zeros_like(psi_x1)
        f_flip = torch.zeros_like(f)
        f[gt_eps_idx] = _f.to(dtype)
        f_flip[gt_eps_idx] = _f_flip.to(dtype)
        f = f.reshape(batch, n_comb)
        f_flip = f_flip.reshape(batch, n_comb)

        psi_x1[gt_eps_idx] = _psi_x1
        psi_x1_flip[gt_eps_idx] = _psi_x1_flip
        psi_x1 = psi_x1.reshape(batch, n_comb)
        psi_x1_flip =psi_x1_flip.reshape(batch, n_comb)
        η_m = torch.zeros(batch * n_comb, dtype=_η_m.dtype, device=device)
        η_m[gt_eps_idx] = _η_m
        η_m = η_m.reshape(batch, n_comb)

        f_psi = (f * psi_x1 + η * η_m * f_flip * psi_x1_flip) * f[..., 0].reshape(-1, 1).conj() / extra_norm**2  # [batch, nSD]
    else:
        # [batch, comb]
        # spin-flip
        # breakpoint()
        _η_m = spin_flip_sign(x, sorb)
        x1_flip = spin_flip_onv(x, sorb)
        
        _psi_x1 = Func(ansatz, x, WF_LUT, use_unique)
        _psi_x1_flip = Func(ansatz, x1_flip, WF_LUT, use_unique)
        psi_x1[gt_eps_idx] = _psi_x1
        psi_x1_flip[gt_eps_idx] = _psi_x1_flip
        psi_x1 = psi_x1.reshape(batch, n_comb)
        psi_x1_flip =psi_x1_flip.reshape(batch, n_comb)

        η_m = torch.zeros(batch * n_comb, dtype=_η_m.dtype, device=device)
        η_m[gt_eps_idx] = _η_m
        η_m = η_m.reshape(batch, n_comb)

        f_psi = (psi_x1 + η * η_m * psi_x1_flip) / extra_norm**2  # [batch, nSD]

    comb_hij = comb_hij * f_psi
    if use_spin_raising:
        hij_spin = hij_spin * f_psi

    # if comb_x.numel() != 0:
    #     if use_LUT:
    #         lut_idx, lut_not_idx, lut_value = WF_LUT.lookup(comb_x.reshape(-1, bra_len)[gt_eps_idx])
    #         # the index of x1 great than eps and not in LUT
    #         raw_idx = torch.arange(n_comb * batch, device=device)
    #         gt_not_lut_idx = raw_idx[gt_eps_idx][lut_not_idx]
    #         # the index of x1 great than eps and in LUT
    #         gt_in_lut_idx = raw_idx[gt_eps_idx][lut_idx]
    #     if use_unique:
    #         if use_LUT:
    #             _comb_x = comb_x.reshape(-1, bra_len)[gt_not_lut_idx]
    #         else:
    #             _comb_x = comb_x.reshape(-1, bra_len)[gt_eps_idx]
    #         unique_comb, inverse = torch.unique(_comb_x, dim=0, return_inverse=True)
    #         x1 = onv_to_tensor(unique_comb, sorb)
    #         psi_gt_eps = torch.index_select(ansatz(x1), 0, inverse)
    #     else:
    #         if use_LUT:
    #             # x1 great than eps and not in LUT
    #             _comb = comb_x.reshape(-1, bra_len)[gt_not_lut_idx]
    #         else:
    #             # x1 great than eps
    #             _comb = comb_x.reshape(-1, bra_len)[gt_eps_idx]
    #         x1 = onv_to_tensor(_comb, sorb)
    #         psi_gt_eps = ansatz(x1)
    #     if use_LUT:
    #         psi_x1[gt_not_lut_idx] = psi_gt_eps.to(dtype)
    #         psi_x1[gt_in_lut_idx] = lut_value.to(dtype)
    #     else:
    #         psi_x1[gt_eps_idx] = psi_gt_eps.to(dtype)
    #     psi_x1 = psi_x1.reshape(batch, -1)
    # else:
    #     psi_x1 = torch.zeros(batch, n_comb, device=device, dtype=dtype)

    # if use_multi_psi:
    #     _comb = comb_x.reshape(-1, bra_len)[gt_eps_idx]
    #     _psi = ansatz_extra(onv_to_tensor(_comb, sorb))
    #     # f*(n) Hnm f(m) / extra_norm**2
    #     psi_extra[gt_eps_idx] = _psi.to(dtype)
    #     psi_extra = psi_extra.reshape(batch, -1)
    #     # (batch, n_SD)
    #     value = psi_extra * (psi_extra[:, 0].reshape(-1, 1).conj() / extra_norm**2)
    #     comb_hij = comb_hij * value
    #     # comb_hij *= psi_extra.real
    #     if use_spin_raising:
    #         hij_spin = hij_spin * value

    # psi_x0 = psi_x1[..., 0]  # [batch] \phi(n)
    # eloc = torch.einsum("ij, ij, i -> i", comb_hij.to(dtype), f_psi, 1 / psi_x0)
    # if use_spin_raising:
    #     sloc = torch.einsum("ij, ij, i -> i", hij_spin.to(dtype), f_psi, 1 / psi_x0)
    # else:
    #     sloc = torch.zeros_like(eloc)


    if batch == 1:
        if use_spin_raising:
            sloc = torch.sum(torch.div(psi_x1.T, psi_x1[..., 0]).T * hij_spin, -1)
        eloc = torch.sum(comb_hij * psi_x1 / psi_x1[..., 0])  # scalar
    else:
        if use_spin_raising:
            sloc = torch.sum(torch.div(psi_x1.T, psi_x1[..., 0]).T * hij_spin, -1)  # (batch)
        eloc = torch.sum(torch.div(psi_x1.T, psi_x1[..., 0]).T * comb_hij, -1)  # (batch)

    t3 = time.time_ns()
    delta0 = (t1 - t0) / 1.0e06
    delta1 = (t2 - t1) / 1.0e06
    delta2 = (t3 - t2) / 1.0e06
    logger.debug(
        f"comb_x/uint8_to_bit time: {delta0:.3E} ms, <i|H|j> time: {delta1:.3E} ms, " + f"nqs time: {delta2:.3E} ms"
    )
    del comb_hij, comb_x, gt_eps_idx  # index, unique_x1, unique

    # if x.is_cuda:
    #     torch.cuda.empty_cache()
    if not use_spin_raising:
        sloc = torch.zeros_like(eloc)

    return eloc.to(dtype), sloc.to(dtype), psi_x1[..., 0].to(dtype), (delta0, delta1, delta2)